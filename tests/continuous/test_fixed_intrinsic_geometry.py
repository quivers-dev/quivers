"""Fixed scalar constructors share tensor event geometry and safe reductions."""

import math

import pytest
import torch
import torch.nn.functional as F

from quivers.continuous import inline
from quivers.continuous.measure import Independent, Mixture, Normalize, Restrict
from quivers.continuous._measure_queries import log_mass
from quivers.continuous._truncated_normal import TruncatedNormal
from quivers.continuous.spaces import Euclidean
from quivers.dsl.ast_nodes.program_steps import DrawArgNamed, DrawArgScalar


def _logit_beta(y):
    return (
        math.lgamma(5)
        - math.lgamma(2)
        - math.lgamma(3)
        + 2 * F.logsigmoid(y)
        + 3 * F.logsigmoid(-y)
    )


def _log_gamma(y):
    return 2 * math.log(1.5) - math.lgamma(2) + 2 * y - 1.5 * y.exp()


def _truncated(y):
    base = torch.distributions.Normal(0.0, 1.0)
    mass = base.cdf(torch.tensor(1.5, dtype=y.dtype)) - base.cdf(
        torch.tensor(-0.5, dtype=y.dtype)
    )
    return base.log_prob(y) - mass.log()


CASES = [
    (
        "normal",
        inline.make_fixed_normal,
        (0.5, 1.25),
        lambda y: torch.distributions.Normal(0.5, 1.25).log_prob(y),
        (-0.4, 0.8),
    ),
    (
        "logitnormal",
        inline.make_fixed_logitnormal,
        (0.5, 0.75),
        lambda y: (
            torch.distributions.Normal(0.5, 0.75).log_prob(torch.logit(y))
            - y.log()
            - torch.log1p(-y)
        ),
        (0.2, 0.8),
    ),
    (
        "uniform",
        inline.make_fixed_uniform,
        (-0.5, 1.5),
        lambda y: torch.distributions.Uniform(-0.5, 1.5).log_prob(y),
        (-0.4, 0.8),
    ),
    (
        "beta",
        inline.make_fixed_beta,
        (2.0, 3.0),
        lambda y: torch.distributions.Beta(2.0, 3.0).log_prob(y),
        (0.2, 0.8),
    ),
    ("logitbeta", inline.make_fixed_logit_beta, (2.0, 3.0), _logit_beta, (-0.5, 0.7)),
    ("loggamma", inline.make_fixed_log_gamma, (2.0, 1.5), _log_gamma, (-0.4, 0.6)),
    (
        "exponential",
        inline.make_fixed_exponential,
        (1.25,),
        lambda y: torch.distributions.Exponential(1.25).log_prob(y),
        (0.2, 1.4),
    ),
    (
        "halfcauchy",
        inline.make_fixed_halfcauchy,
        (0.5,),
        lambda y: torch.distributions.HalfCauchy(0.5).log_prob(y),
        (0.2, 1.4),
    ),
    (
        "halfnormal",
        inline.make_fixed_halfnormal,
        (0.75,),
        lambda y: torch.distributions.HalfNormal(0.75).log_prob(y),
        (0.2, 1.4),
    ),
    (
        "lognormal",
        inline.make_fixed_lognormal,
        (0.5, 0.75),
        lambda y: torch.distributions.LogNormal(0.5, 0.75).log_prob(y),
        (0.2, 1.4),
    ),
    (
        "gamma",
        inline.make_fixed_gamma,
        (2.0, 1.5),
        lambda y: torch.distributions.Gamma(2.0, 1.5).log_prob(y),
        (0.2, 1.4),
    ),
    (
        "truncatednormal",
        inline.make_fixed_truncated_normal,
        (0.0, 1.0, -0.5, 1.5),
        _truncated,
        (-0.3, 0.7),
    ),
]


@pytest.mark.parametrize(
    "name,factory,parameters,oracle,interval", CASES, ids=[x[0] for x in CASES]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_scalar_factory_density_and_coordinate_gradients_preserve_each_run(
    name, factory, parameters, oracle, interval, explicit
):
    event = (2, 3) if explicit else (6,)
    carrier = Euclidean(name=name, dim=6, intrinsic_shape=event if explicit else None)
    morphism = factory(*parameters, carrier)
    value = (
        torch.linspace(*interval, 4 * 6, dtype=torch.float64)
        .reshape(4, *event)
        .requires_grad_()
    )
    x = torch.zeros(4, 1, dtype=torch.float64)
    actual = morphism.log_prob(x, value)
    expected = oracle(value).flatten(1).sum(-1)
    assert actual.shape == (4,)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    if name != "uniform":
        got = torch.autograd.grad(actual.sum(), value, retain_graph=True)[0]
        wanted = torch.autograd.grad(expected.sum(), value)[0]
        torch.testing.assert_close(got, wanted, atol=2e-6, rtol=2e-6)
    draw = morphism.rsample(x, torch.Size((2,)))
    assert draw.shape == (2, 4, *event)
    native = morphism.sampling_distribution(x)
    if explicit:
        assert native.batch_shape == (4,)
        assert native.event_shape == event


@pytest.mark.parametrize("event", [(2, 3), (1, 3), (3, 1), (1, 1), (1,), (), (0, 3)])
@pytest.mark.parametrize("rows", [0, 1, 4])
def test_normal_scalar_vector_matrix_empty_geometry_rng_moments_and_hessian(
    event, rows
):
    carrier = Euclidean(name="e", dim=math.prod(event), intrinsic_shape=event)
    morphism = inline.make_fixed_normal(0.5, 1.25, carrier)
    x = torch.zeros(rows, 1, dtype=torch.float64)
    native = morphism.sampling_distribution(x)
    assert native.mean.shape == (rows, *event)
    assert native.variance.shape == (rows, *event)
    torch.testing.assert_close(native.mean, torch.full((rows, *event), 0.5))
    torch.testing.assert_close(native.variance, torch.full((rows, *event), 1.25**2))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(83)
        actual = morphism.rsample(x, torch.Size((2,)))
        state = torch.random.get_rng_state()
        torch.manual_seed(83)
        expected = torch.distributions.Normal(
            torch.full((rows, *event), 0.5), torch.full((rows, *event), 1.25)
        ).rsample(torch.Size((2,)))
        assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    point = actual.double().requires_grad_()
    score = morphism.log_prob(x, point)
    expected_score = torch.distributions.Normal(0.5, 1.25).log_prob(point)
    if event:
        expected_score = expected_score.sum(tuple(range(-len(event), 0)))
    assert score.shape == (2, rows)
    torch.testing.assert_close(score, expected_score)
    if point.numel():
        gradient = torch.autograd.grad(score.sum(), point, create_graph=True)[0]
        second = torch.autograd.grad(gradient.sum(), point)[0]
        torch.testing.assert_close(gradient, -(point - 0.5) / (1.25**2))
        torch.testing.assert_close(second, torch.full_like(point, -1 / (1.25**2)))


def test_generated_and_alternative_factories_use_the_same_explicit_coordinate_contract():
    carrier = Euclidean(name="matrix", dim=6, intrinsic_shape=(2, 3))

    def fixed(family, arguments):
        return inline.make_inline_distribution(
            family,
            tuple(
                DrawArgNamed(parameter=k, value=DrawArgScalar(value=v))
                for k, v in arguments.items()
            ),
            carrier,
        )[0]

    x = torch.zeros(4, 1)
    student = fixed("StudentT", {"df": 5.0, "loc": 0.5, "scale": 1.25})
    value = torch.linspace(-0.4, 0.8, 24).reshape(4, 2, 3)
    torch.testing.assert_close(
        student.log_prob(x, value),
        torch.distributions.StudentT(5.0, 0.5, 1.25).log_prob(value).sum((-2, -1)),
    )
    geometric = fixed("Geometric", {"logits": -2.0})
    counts = torch.arange(24, dtype=torch.int64).reshape(4, 2, 3) % 4
    expected = (
        torch.distributions.Geometric(logits=torch.tensor(-2.0, dtype=torch.float64))
        .log_prob(counts)
        .sum((-2, -1))
    )
    torch.testing.assert_close(geometric.log_prob(x, counts), expected)
    assert geometric.rsample(x).dtype == torch.int64
    assert geometric.rsample(x).shape == (4, 2, 3)


def test_raw_measure_coordinates_normalize_once_and_keep_analytical_moments_where_defined():
    carrier = Euclidean(name="matrix", dim=6, intrinsic_shape=(2, 3))

    def constructor(batch, device):
        return Restrict(
            torch.distributions.Normal(
                torch.zeros(batch, 2, 3, device=device),
                torch.ones(batch, 2, 3, device=device),
            ),
            -0.5,
            1.5,
        )

    morphism = inline.FixedDistribution(carrier, constructor)
    value = torch.linspace(-0.3, 0.7, 24).reshape(4, 2, 3)
    actual = morphism.log_prob(torch.zeros(4, 1), value)
    expected = _truncated(value).sum((-2, -1))
    torch.testing.assert_close(actual, expected)


def test_incompatible_native_event_shape_and_flattened_queries_fail_explicitly():
    carrier = Euclidean(name="matrix", dim=6, intrinsic_shape=(2, 3))
    mismatched = inline.FixedDistribution(
        carrier,
        lambda batch, device: torch.distributions.MultivariateNormal(
            torch.zeros(batch, 6, device=device), torch.eye(6, device=device)
        ),
    )
    with pytest.raises(ValueError, match="native event shape"):
        mismatched.rsample(torch.zeros(2, 1))
    normal = inline.make_fixed_normal(0.0, 1.0, carrier)
    with pytest.raises(ValueError, match="trailing event shape"):
        normal.log_prob(torch.zeros(4, 1), torch.zeros(4, 6))


@pytest.mark.parametrize("count", [0, 1, 2])
def test_independent_retains_moments_pathwise_gradients_and_base_rng(count):
    loc = torch.arange(6, dtype=torch.float64).reshape(2, 3).requires_grad_()
    scale = torch.full((2, 3), 0.75, dtype=torch.float64, requires_grad=True)
    base = torch.distributions.Normal(loc, scale)
    law = Independent(base, count)
    assert law.has_rsample
    torch.testing.assert_close(law.mean, loc)
    torch.testing.assert_close(law.variance, scale.square())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(89)
        actual = law.rsample(torch.Size((4,)))
        torch.manual_seed(89)
        expected = base.rsample(torch.Size((4,)))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    got = torch.autograd.grad(actual.sum(), (loc, scale), retain_graph=True)
    wanted = torch.autograd.grad(expected.sum(), (loc, scale))
    for x, y in zip(got, wanted, strict=True):
        torch.testing.assert_close(x, y)


def test_discrete_independent_does_not_claim_a_pathwise_sampler():
    law = Independent(torch.distributions.Poisson(torch.ones(2, 3)), 1)
    assert not law.has_rsample
    with pytest.raises(NotImplementedError, match="no pathwise"):
        law.rsample()
    assert law.sample(torch.Size((4,))).shape == (4, 2, 3)


def test_probability_normalization_constant_is_not_raw_mass_or_its_derivative():
    parameters = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in (0.25, 1.25, -0.5, 1.5)
    )
    law = TruncatedNormal(*parameters)
    base = torch.distributions.Normal(parameters[0], parameters[1])
    expected = (base.cdf(parameters[3]) - base.cdf(parameters[2])).log()
    torch.testing.assert_close(law.log_normalizer(), expected)
    assert float(expected.detach()) < 0
    mass = log_mass(law)
    torch.testing.assert_close(mass, torch.zeros_like(mass))
    gradients = torch.autograd.grad(mass, parameters, retain_graph=True)
    for value in gradients:
        torch.testing.assert_close(value, torch.zeros_like(value))
    actual_gradient = torch.autograd.grad(
        law.log_normalizer(), parameters, retain_graph=True
    )
    expected_gradient = torch.autograd.grad(expected, parameters)
    for x, y in zip(actual_gradient, expected_gradient, strict=True):
        torch.testing.assert_close(x, y)
    joint = Independent(law.expand((2, 3)), 2)
    torch.testing.assert_close(log_mass(joint), torch.zeros_like(mass))


def test_raw_scaled_measure_mass_and_gradients_survive_independent_event_composition():
    weights = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    base = torch.distributions.Normal(
        torch.zeros(2, 3, dtype=torch.float64), torch.ones(2, 3, dtype=torch.float64)
    )
    raw = Mixture(weights, [base, base])
    joint = Independent(raw, 2)
    expected = 6 * weights.sum().log()
    mass = log_mass(joint)
    torch.testing.assert_close(mass, expected)
    got = torch.autograd.grad(mass, weights, retain_graph=True)[0]
    wanted = torch.autograd.grad(expected, weights)[0]
    torch.testing.assert_close(got, wanted)
    torch.testing.assert_close(log_mass(Normalize(joint)), torch.zeros_like(mass))


@pytest.mark.parametrize(
    ("family", "arguments"),
    [
        ("Normal", {"loc": 0, "scale": 1}),
        ("Uniform", {"low": 0, "high": 2}),
        ("HalfNormal", {"scale": 1}),
        ("Gamma", {"concentration": 2, "rate": 1}),
    ],
)
def test_integer_literals_of_real_parameters_draw_floating_values(family, arguments):
    carrier = Euclidean(name="scalar", dim=1)

    def fixed(values):
        return inline.make_inline_distribution(
            family,
            tuple(
                DrawArgNamed(parameter=k, value=DrawArgScalar(value=v))
                for k, v in values.items()
            ),
            carrier,
        )[0]

    x = torch.zeros(3, 1)
    torch.manual_seed(0)
    draw = fixed(arguments).rsample(x)
    torch.manual_seed(0)
    reference = fixed({k: float(v) for k, v in arguments.items()}).rsample(x)
    assert draw.dtype == torch.get_default_dtype()
    torch.testing.assert_close(draw, reference)
    torch.testing.assert_close(
        fixed(arguments).log_prob(x, reference),
        fixed({k: float(v) for k, v in arguments.items()}).log_prob(x, reference),
    )
