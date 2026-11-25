"""A potential evaluated at many positions in one pass equals one per row.

`PotentialFn.log_density_batch` maps the program's scoring run over the
positions, so the oracle throughout is the single-position
`PotentialFn.log_density` and `PotentialFn.value_and_grad`, row by row.

Cover:

1. Values equal the single-position ones to rounding, and gradients per
   row, on every kind of site, plate, let, score, marginalized block,
   scan, and formula model, at batch sizes equal to the extents the model
   has (a plate, its rows, one) so that no axis can be read for another.
2. A row outside the support scores ``-inf`` and leaves the others alone,
   whether a parameter validator, a value validator, a factorization, or
   a budget refuses it; a defect raises.
3. Batchability is declared: an undeclared step is refused by name with no
   evaluation, and a batched callable that loses the position axis is
   refused with the step named.
4. One call replays the plan once, whatever the number of rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions import Normal

from quivers.continuous import Draw, Euclidean, Let, MonadicProgram, Score
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.inline import make_fixed_logit_beta, make_fixed_normal
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.program_steps import Observe, mappable, reading
from quivers.core import Unit
from quivers.dsl import load, loads
from quivers.effects import interpreter
from quivers.effects.joint_plan import JointPlan
from quivers.effects.batching import UnbatchableProgramError
from quivers._positions import PositionLayoutError
from quivers.formulas import (
    BayesianFit,
    GroupCovariance,
    MultivariateFit,
    bf,
    fit,
    set_rescor,
)
from quivers.inference.mcmc import PotentialFn
from quivers.inference.registry import LatentRegistry
from quivers.inference.smc import TemperedPotential
from quivers.inference.trace import trace
from tests.benchmarks import datasets

_X = torch.zeros(1, 1)
_FIXTURE_ROOT = Path(__file__).parent / "transpile" / "fixtures"
_RTOL = 1e-12


@pytest.fixture(autouse=True)
def _double() -> Iterator[None]:
    """Score in double precision, where the tolerance is rounding."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _positions(potential: PotentialFn, rows: int, seed: int = 0) -> torch.Tensor:
    """Standard-normal positions of the potential's registry."""
    generator = torch.Generator().manual_seed(seed)
    dimension = potential.registry.total_unconstrained_dim
    return torch.randn(rows, dimension, generator=generator, dtype=torch.float64)


def _agree(
    potential: PotentialFn, positions: torch.Tensor, gradient: float = 1e-9
) -> torch.Tensor:
    """Check the batched value and gradient against one call per row.

    ``gradient`` is the relative tolerance of a gradient, looser for a
    model whose gradient passes through an ill-conditioned factorization
    and is itself determined only to that many digits.
    """
    single = [potential.value_and_grad(row) for row in positions]
    values = torch.stack([value for value, _ in single])
    grads = torch.stack([grad for _, grad in single])
    batched = potential.log_density_batch(positions)
    paired, gradients = potential.value_and_grad_batch(positions)
    assert batched.shape == (positions.shape[0],)
    assert gradients.shape == positions.shape
    assert torch.equal(torch.isfinite(batched), torch.isfinite(values))
    inside = torch.isfinite(values)
    torch.testing.assert_close(batched[inside], values[inside], rtol=_RTOL, atol=1e-12)
    torch.testing.assert_close(paired[inside], values[inside], rtol=_RTOL, atol=1e-12)
    assert bool(torch.isneginf(batched[~inside]).all())
    scale = grads.abs().amax(-1, keepdim=True).clamp(min=1.0)
    torch.testing.assert_close(
        gradients / scale, grads / scale, rtol=0.0, atol=gradient, check_dtype=True
    )
    assert torch.equal((gradients == 0).all(-1), (grads == 0).all(-1))
    return batched


def _compiled(source: str) -> MonadicProgram:
    """The program a source compiles to."""
    model = loads(source).morphism
    assert isinstance(model, MonadicProgram)
    return model


def _source_potential(
    source: str,
    data: Mapping[str, torch.Tensor],
    x: torch.Tensor = _X,
) -> PotentialFn:
    """The potential of a compiled program at its data."""
    model = _compiled(source)
    registry = LatentRegistry.from_model(model, set(data))
    return PotentialFn(model, registry, x, dict(data))


# ---------------------------------------------------------------------------
# Sites, plates, lets, and scores
# ---------------------------------------------------------------------------

_HEADER = """
object Group : FinSet 4
object Row : FinSet 12
object K : FinSet 3
object Val : Real 1
"""

_GROUP = torch.arange(12) % 4
_COVARIATE = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64)


def _program(body: str, returns: str, domain: str = "Row") -> str:
    """A program over the shared objects."""
    lines = "\n".join(f"    {line}" for line in body.strip().splitlines())
    return (
        f"{_HEADER}\nprogram p : {domain} -> Val\n{lines}\n    return {returns}\n"
        "export p\n"
    )


_LATENTS = {
    "normal": "sample a <- Normal(loc=0.0, scale=1.0)",
    "half_normal": "sample a <- HalfNormal(scale=2.0)",
    "half_cauchy": "sample a <- HalfCauchy(scale=1.0)",
    "gamma": "sample a <- Gamma(concentration=2.0, rate=1.5)",
    "inverse_gamma": "sample a <- InverseGamma(concentration=3.0, rate=2.0)",
    "beta": "sample a <- Beta(concentration1=2.0, concentration0=3.0)",
    "exponential": "sample a <- Exponential(rate=1.5)",
    "log_normal": "sample a <- LogNormal(loc=0.0, scale=0.5)",
    "student_t": "sample a <- StudentT(df=4.0, loc=0.0, scale=1.0)",
    "cauchy": "sample a <- Cauchy(loc=0.0, scale=1.0)",
    "uniform": "sample a <- Uniform(low=-1.0, high=2.0)",
    "weibull": "sample a <- Weibull(scale=1.0, concentration=1.5)",
    "truncated_normal": (
        "sample a <- TruncatedNormal(mu=0.3, sigma=1.0, low=0.0, high=2.0)"
    ),
    "dirichlet": "sample a <- Dirichlet(concentration=2.0) [over=K]",
    "lkj": "sample a : K <- LKJCholesky(concentration=2.0)",
    "vector": "sample a <- Normal(loc=0.0, scale=1.0) [over=K]",
    "plate": "sample a : Group <- Normal(loc=0.0, scale=2.0)",
    "plated_simplex": "sample a : Group <- Dirichlet(concentration=2.0) [over=K]",
    "plated_vector": "sample a : Group <- Normal(loc=0.0, scale=1.0) [over=K]",
}


@pytest.mark.parametrize("name", sorted(_LATENTS))
@pytest.mark.parametrize("rows", [1, 3, 4])
def test_each_kind_of_latent_site(name: str, rows: int) -> None:
    """The row counts are one, the event's extent, and the plate's."""
    potential = _source_potential(_program(_LATENTS[name], "a", "Val"), {})
    _agree(potential, _positions(potential, rows))


_RESPONSES = {
    "normal": ("Normal(loc=eta, scale=sigma)", torch.linspace(-1, 2, 12)),
    "student_t": ("StudentT(df=4.0, loc=eta, scale=sigma)", torch.linspace(-1, 2, 12)),
    "laplace": ("Laplace(loc=eta, scale=sigma)", torch.linspace(-1, 2, 12)),
    "log_normal": ("LogNormal(loc=eta, scale=sigma)", torch.linspace(0.2, 3, 12)),
    "gamma": ("Gamma(concentration=sigma, rate=rate)", torch.linspace(0.2, 3, 12)),
    "weibull": ("Weibull(scale=rate, concentration=sigma)", torch.linspace(0.2, 3, 12)),
    "poisson": ("Poisson(rate=rate)", torch.arange(12.0) % 5),
    "bernoulli": ("Bernoulli(logits=eta)", torch.arange(12.0) % 2),
    "binomial": ("Binomial(total_count=6, logits=eta)", torch.arange(12.0) % 7),
    "negative_binomial": (
        "NegativeBinomial(total_count=sigma, logits=eta)",
        torch.arange(12.0) % 5,
    ),
    "von_mises": ("VonMises(loc=eta, concentration=sigma)", torch.linspace(-2, 2, 12)),
    "zero_inflated": (
        "ZeroInflatedPoisson(zero_prob=prob, rate=rate)",
        torch.arange(12.0) % 3,
    ),
    "hurdle": ("HurdlePoisson(zero_prob=prob, rate=rate)", torch.arange(12.0) % 3),
    "beta": (
        "Beta(concentration1=sigma, concentration0=rate)",
        torch.linspace(0.1, 0.9, 12),
    ),
}


@pytest.mark.parametrize("name", sorted(_RESPONSES))
@pytest.mark.parametrize("rows", [1, 4, 12])
def test_each_kind_of_plated_response(name: str, rows: int) -> None:
    """The row counts are one, the latent plate's extent, and the data's."""
    family, response = _RESPONSES[name]
    body = f"""
sample mu <- Normal(loc=0.0, scale=1.0)
sample sigma <- HalfNormal(scale=2.0)
sample alpha : Group <- Normal(loc=mu, scale=1.0)
sample slope <- Normal(loc=0.0, scale=1.0)
let eta = alpha[group] + slope * x
let rate = exp(eta)
let prob = sigmoid(eta)
observe y : Row <- {family}
"""
    potential = _source_potential(
        _program(body, "mu"),
        {"y": response.double(), "group": _GROUP, "x": _COVARIATE},
    )
    _agree(potential, _positions(potential, rows))


_STRUCTURES = {
    "correlated_rows": (
        """
sample tau : K <- HalfNormal(scale=1.0)
sample corr : K <- LKJCholesky(concentration=2.0)
sample mu : K <- Normal(loc=0.0, scale=1.0)
let factors = factor i : K, j : K in tau[i] * corr[i, j]
observe y : Group <- MultivariateNormal(loc=mu, scale_tril=factors) [over=K]
""",
        "tau",
        {"y": torch.linspace(-1.0, 1.0, 12).reshape(4, 3)},
    ),
    "let_chain": (
        """
sample a <- Normal(loc=0.0, scale=1.0)
sample v : Group <- Normal(loc=0.0, scale=1.0)
let centered = v - mean(v)
let w = softmax(centered)
let table = factor g : Group in a * real(g) + centered[g]
score s = sum(table) - a * a
let loc = table[group]
observe y : Row <- Normal(loc=loc, scale=1.0)
""",
        "a",
        {"y": _COVARIATE, "group": _GROUP},
    ),
    "marginalize": (
        """
sample w <- Dirichlet(concentration=1.0) [over=K]
sample m <- Normal(loc=0.0, scale=2.0) [over=K]
marginalize c : Row <- Categorical(probs=w) [reduction=logsumexp]
    let loc = m[c]
    observe y : Row <- Normal(loc=loc, scale=1.0)
""",
        "w",
        {"y": _COVARIATE},
    ),
    "weighted": (
        """
sample a <- Normal(loc=0.0, scale=1.0)
observe y : Row <- Normal(loc=a, scale=1.0) [weights=w]
""",
        "a",
        {"y": _COVARIATE, "w": (torch.arange(12.0) % 3).double()},
    ),
}


@pytest.mark.parametrize("name", sorted(_STRUCTURES))
@pytest.mark.parametrize("rows", [1, 3, 4, 12])
def test_each_structure(name: str, rows: int) -> None:
    body, returns, data = _STRUCTURES[name]
    domain = "Row" if data else "Val"
    potential = _source_potential(_program(body, returns, domain), data)
    _agree(potential, _positions(potential, rows))


# ---------------------------------------------------------------------------
# The step and family fixtures
# ---------------------------------------------------------------------------

_FIXTURES = {
    "steps": (
        "call_lowered_pure",
        "call_step",
        "function_linear",
        "marginalize_body_let_chain",
        "marginalize_body_plated_let",
        "marginalize_factor_let_rows",
        "marginalize_latent_gather",
        "marginalize_let_vector_probs",
        "marginalize_shared_prior_scalar_let",
        "observe_bernstein_hazard_log_rate",
        "observe_class_code",
        "observe_gather_event_vector",
        "observe_gather_matrix_row",
        "observe_geometric_logits",
        "observe_matrix_normal_row_factor",
        "observe_negative_binomial_logits",
        "observe_ordered_probit",
        "observe_ordinal_code",
        "observe_over_vector_location",
        "observe_plated_vector_location",
        "observe_relaxed_bernoulli_logits",
        "observe_vector_location_let",
        "observe_vector_scale_rate",
        "sample_dirichlet_sampled_concentration",
        "sample_inverse_wishart_factor",
        "sample_matrix_normal",
        "sample_mvn_covariance",
        "sample_mvn_precision",
        "sample_wishart_covariance",
        "sample_wishart_factor",
        "scan_bidirectional",
        "scan_cell_reads_vector",
        "scan_initial_binding",
        "scan_marginal_chain",
        "scan_outputs_all",
        "scan_reverse",
    ),
    "families": (
        "continuousbernoulli",
        "fishersnedecor",
        "gp",
        "inversewishart",
        "kumaraswamy",
        "loggamma",
        "logitbeta",
        "logitnormal",
        "lowrankmvn",
        "multivariatenormal",
        "relaxedbernoulli",
        "wishart",
    ),
    "let_expressions": (
        "let_concat_slice_matrix",
        "let_expr_factor",
        "let_fold_shared_accumulator",
        "let_int_division",
        "let_plated_list_operand",
        "let_real_division",
        "let_reduce_nested",
        "let_row_gather_drawn",
        "let_select",
        "let_tensor_builtins",
    ),
}


@pytest.mark.parametrize(
    ("group", "stem"),
    [(group, stem) for group, stems in _FIXTURES.items() for stem in stems],
)
def test_each_fixture_program(group: str, stem: str) -> None:
    """The programs the transpilation suite draws its steps from.

    Every observed or discrete site is held at a draw from the program.
    """
    model = load(_FIXTURE_ROOT / group / f"{stem}.qvr").morphism
    assert isinstance(model, MonadicProgram)
    width = int(getattr(model.domain, "dim", 1))
    x = torch.zeros(1, width)
    torch.manual_seed(0)
    run = trace(model, x)
    scored = {
        name for step in model.steps if isinstance(step, Observe) for name in step.names
    }
    data = {
        label: site.value
        for label, site in run.sites.items()
        if site.morphism is not None
        and not site.is_deterministic
        and (label in scored or not site.value.is_floating_point())
    }
    registry = LatentRegistry.from_trace(model, trace(model, x, observations=data))
    potential = PotentialFn(model, registry, x, data)
    _agree(potential, _positions(potential, 4))


# ---------------------------------------------------------------------------
# Formula models
# ---------------------------------------------------------------------------

_FRAME = pd.DataFrame(
    {
        "y": np.linspace(-1.0, 2.0, 12) + np.tile([0.3, -0.2, 0.1], 4),
        "z": np.linspace(1.0, -1.0, 12) ** 2,
        "pos": np.linspace(0.2, 3.0, 12),
        "count": np.arange(12) % 4,
        "binary": np.arange(12) % 2,
        "ordinal": np.arange(12) % 3 + 1,
        "species": np.array(["a", "b", "c"] * 4),
        "x": np.linspace(-1.0, 1.0, 12),
        "g": np.array(["u", "v", "w", "q"] * 3),
        "t": np.tile(np.arange(3), 4),
    }
)

_COVARIANCE = GroupCovariance(
    levels=("u", "v", "w", "q"),
    matrix=torch.tensor(
        [
            [1.0, 0.3, 0.1, 0.0],
            [0.3, 1.2, 0.2, 0.1],
            [0.1, 0.2, 0.9, 0.4],
            [0.0, 0.1, 0.4, 1.1],
        ],
        dtype=torch.float64,
    ),
)

_FITS: dict[str, Callable[[], BayesianFit | MultivariateFit]] = {
    "gaussian": lambda: fit("y ~ x", data=_FRAME, method="svi", num_steps=0),
    "student": lambda: fit(
        "y ~ x", data=_FRAME, method="svi", num_steps=0, family="student"
    ),
    "binomial": lambda: fit(
        "binary ~ x", data=_FRAME, method="svi", num_steps=0, family="bernoulli"
    ),
    "poisson": lambda: fit(
        "count ~ x + (1 | g)", data=_FRAME, method="svi", num_steps=0, family="poisson"
    ),
    "negative binomial": lambda: fit(
        "count ~ x", data=_FRAME, method="svi", num_steps=0, family="negbinomial"
    ),
    "ordinal": lambda: fit(
        "ordinal ~ x", data=_FRAME, method="svi", num_steps=0, family="cumulative"
    ),
    "stopping ratio": lambda: fit(
        "ordinal ~ x", data=_FRAME, method="svi", num_steps=0, family="sratio"
    ),
    "categorical": lambda: fit(
        "species ~ x", data=_FRAME, method="svi", num_steps=0, family="categorical"
    ),
    "hierarchical LKJ": lambda: fit(
        "y ~ x + (1 + x | g)", data=_FRAME, method="svi", num_steps=0
    ),
    "known covariance": lambda: fit(
        "y ~ x + (1 | g)",
        data=_FRAME,
        method="svi",
        num_steps=0,
        group_covariances={"g": _COVARIANCE},
    ),
    "smooth": lambda: fit("y ~ s(x, k = 5)", data=_FRAME, method="svi", num_steps=0),
    "gaussian process": lambda: fit(
        "y ~ gp(x)", data=_FRAME, method="svi", num_steps=0
    ),
    "zero-inflated": lambda: fit(
        "count ~ x",
        data=_FRAME,
        method="svi",
        num_steps=0,
        family="zero_inflated_poisson",
    ),
    "hurdle": lambda: fit(
        "count ~ x", data=_FRAME, method="svi", num_steps=0, family="hurdle_poisson"
    ),
    "mixture": lambda: fit(
        "y ~ x", data=_FRAME, method="svi", num_steps=0, family="mixture"
    ),
    "distributional": lambda: fit(
        "y ~ x", data=_FRAME, method="svi", num_steps=0, distributional={"sigma": "~ x"}
    ),
    "autoregressive": lambda: fit(
        "y ~ x + ar(time = t, gr = g, p = 1)", data=_FRAME, method="svi", num_steps=0
    ),
    "skew normal": lambda: fit(
        "y ~ x", data=_FRAME, method="svi", num_steps=0, family="skew_normal"
    ),
    "beta binomial": lambda: fit(
        "count | trials(4) ~ x",
        data=_FRAME,
        method="svi",
        num_steps=0,
        family="beta_binomial",
    ),
    "censored": lambda: fit(
        "pos | cens(binary) ~ x",
        data=_FRAME,
        method="svi",
        num_steps=0,
        family="weibull",
    ),
    "truncated": lambda: fit(
        "pos | trunc(lb = 0) ~ x", data=_FRAME, method="svi", num_steps=0
    ),
    "truncated mixture": lambda: fit(
        "pos | trunc(lb = 0) ~ x",
        data=_FRAME,
        method="svi",
        num_steps=0,
        family="mixture",
    ),
    "monotonic": lambda: fit("y ~ mo(count)", data=_FRAME, method="svi", num_steps=0),
    "joint": lambda: fit(
        bf("y ~ x") + bf("z ~ x") + set_rescor(True),
        data=_FRAME,
        method="svi",
        num_steps=0,
    ),
    "com poisson": lambda: fit(
        "count ~ x", data=_FRAME, method="svi", num_steps=0, family="com_poisson"
    ),
    "cox": lambda: fit("pos ~ x", data=_FRAME, method="svi", num_steps=0, family="cox"),
}


@pytest.mark.parametrize("name", sorted(_FITS))
def test_each_formula_model(name: str) -> None:
    """The row counts are one, the group's levels, and the data's rows."""
    fitted = _FITS[name]()
    program = fitted.program
    assert isinstance(program, MonadicProgram)
    observations = {
        key: value.double() if value.is_floating_point() else value
        for key, value in fitted.observations.items()
    }
    registry = LatentRegistry.from_model(program, set(observations))
    potential = PotentialFn(program.double(), registry, _X, observations)
    tolerance = 1e-6 if name == "gaussian process" else 1e-9
    for rows in (1, 4, 12):
        _agree(potential, 0.3 * _positions(potential, rows), tolerance)


# ---------------------------------------------------------------------------
# Rows outside the support, and defects
# ---------------------------------------------------------------------------


def _rejecting(rows: torch.Tensor, potential: PotentialFn) -> torch.Tensor:
    """Check that some rows are refused and that the others are untouched."""
    values = _agree(potential, rows)
    refused = torch.isneginf(values)
    assert bool(refused.any()) and not bool(refused.all())
    kept = potential.log_density_batch(rows[~refused])
    torch.testing.assert_close(values[~refused], kept, rtol=_RTOL, atol=1e-12)
    return refused


def test_a_validator_refuses_its_rows_alone() -> None:
    """The scale is a latent of no sign, which the family refuses below zero."""
    potential = _source_potential(
        _program(
            """
sample a <- Normal(loc=0.0, scale=1.0)
observe y : Row <- Normal(loc=0.0, scale=a)
""",
            "a",
        ),
        {"y": _COVARIATE},
    )
    rows = torch.tensor([[0.5], [-0.5], [1.5], [0.0], [2.0]])
    refused = _rejecting(rows, potential)
    assert refused.tolist() == [False, True, False, True, False]


def test_a_density_that_is_not_finite_refuses_its_row() -> None:
    potential = _source_potential(
        _program(
            """
sample a <- Normal(loc=0.0, scale=1.0)
let b = a * a - 1.0
score s = log(b)
""",
            "a",
            "Val",
        ),
        {},
    )
    rows = torch.tensor([[0.2], [1.5], [-0.1], [-2.0]])
    refused = _rejecting(rows, potential)
    assert refused.tolist() == [True, False, True, False]


def test_a_factorization_refuses_its_rows_alone() -> None:
    """The covariance is positive definite exactly where ``|a| < 1``."""

    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        off = env["a"][..., None, None] * torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        factor = torch.linalg.cholesky(torch.eye(2) + off)
        return -factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    potential = _hand_potential(score, batched=True)
    rows = torch.tensor([[0.5], [1.5], [-0.3], [-2.0]])
    refused = _rejecting(rows, potential)
    assert refused.tolist() == [False, True, False, True]
    mapped = _hand_potential(mappable(score), batched=False)
    assert torch.equal(
        mapped.log_density_batch(rows), potential.log_density_batch(rows)
    )


def test_a_validator_inside_a_batched_callable_refuses_its_rows_alone() -> None:
    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        return Normal(0.0, env["a"]).log_prob(torch.tensor(0.3))

    rows = torch.tensor([[0.5], [-0.5], [2.0]])
    refused = _rejecting(rows, _hand_potential(score, batched=True))
    assert refused.tolist() == [False, True, False]


def test_a_budget_refuses_its_rows_alone() -> None:
    """COM-Poisson refuses a mode beyond exact counts, as a position."""
    potential = _source_potential(
        _program(
            """
sample a <- Normal(loc=0.0, scale=1.0)
let nu = exp(a)
observe y : Row <- COMPoisson(log_rate=3.0, dispersion=nu)
""",
            "a",
        ),
        {"y": torch.arange(12.0) % 4},
    )
    rows = torch.tensor([[0.1], [-30.0], [0.4]])
    refused = _rejecting(rows, potential)
    assert refused.tolist() == [False, True, False]


def test_a_refusal_no_row_can_be_read_from_raises() -> None:
    """The refused value has left the tensors that carry the positions."""

    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        values = torch.tensor(env["a"].detach().tolist())
        return Normal(0.0, values).log_prob(torch.tensor(0.3))

    potential = _hand_potential(score, batched=True)
    with pytest.raises(PositionLayoutError, match="cannot be read"):
        potential.log_density_batch(torch.tensor([[0.5], [-0.5]]))


def test_every_row_refused_scores_minus_infinity_with_no_gradient() -> None:
    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        return Normal(0.0, env["a"]).log_prob(torch.tensor(0.3))

    potential = _hand_potential(score, batched=True)
    rows = torch.tensor([[-0.5], [-2.0]])
    values, grads = potential.value_and_grad_batch(rows)
    assert bool(torch.isneginf(values).all())
    assert torch.equal(grads, torch.zeros_like(rows))


def test_a_defect_raises_as_it_does_at_one_position() -> None:
    """A parameter check that is not the sampler's raises at a failing row."""
    model = _compiled(
        _program(
            """
sample a <- Normal(loc=0.0, scale=1.0)
let p = log1mexp(a)
score s = p
""",
            "a",
            "Val",
        )
    )
    potential = PotentialFn(model, LatentRegistry.from_model(model, set()), _X, {})
    _agree(potential, torch.tensor([[-0.5], [-1.5]]))
    with pytest.raises(ValueError, match="nonpositive"):
        potential.log_density(torch.tensor([0.5]))
    with pytest.raises(ValueError, match="nonpositive"):
        potential.log_density_batch(torch.tensor([[-0.5], [0.5]]))


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def _hand_model(
    score: Callable[[dict[str, torch.Tensor]], torch.Tensor], *, batched: bool
) -> MonadicProgram:
    """A program of one Normal latent and a score step."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("a",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="A", dim=1)),
            ),
            Score(name="lik", score=score, batched=batched),
        ],
        return_vars=("a",),
    )


def _hand_potential(
    score: Callable[[dict[str, torch.Tensor]], torch.Tensor], *, batched: bool
) -> PotentialFn:
    model = _hand_model(score, batched=batched)
    return PotentialFn(model, LatentRegistry.from_model(model, set()), _X, {})


def test_a_batched_score_reads_the_positions_as_a_leading_axis() -> None:
    """Values that depend on the position lead with it; data do not."""
    counts = torch.tensor([3.0, 0.0, 5.0, 1.0])
    seen: list[dict[str, tuple[int, ...]]] = []

    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        seen.append({name: tuple(env[name].shape) for name in ("a", "b", "counts")})
        a, b = env["a"], env["b"]
        total = (env["counts"] * torch.nn.functional.logsigmoid(a)).sum(-1)
        if bool((env["counts"] > 0).any()):
            total = total - (b**2).sum(-1)
        return total

    model = MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("a",),
                morphism=make_fixed_logit_beta(1.0, 2.0, Euclidean(name="A", dim=4)),
            ),
            Draw(
                names=("b",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="B", dim=4)),
            ),
            Score(
                name="lik",
                score=reading(score, ("a", "b", "counts")),
                batched=True,
            ),
        ],
        return_vars=("a", "b"),
    )
    registry = LatentRegistry.from_model(model, {"counts"})
    potential = PotentialFn(model, registry, _X, {"counts": counts})
    for rows in (1, 4, 7):
        _agree(potential, _positions(potential, rows))
    seen.clear()
    potential.log_density_batch(_positions(potential, 7))
    assert [(call["a"], call["b"], call["counts"]) for call in seen] == [
        ((7, 4), (7, 4), (4,))
    ]


def test_a_batched_let_feeds_the_steps_after_it() -> None:
    model = MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("a",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="A", dim=3)),
            ),
            Let(name="m", value=lambda env: env["a"].cumsum(-1), batched=True),
            Let(
                name="shared",
                value=reading(lambda env: torch.arange(3.0), ()),
                batched=True,
            ),
            Score(
                name="s",
                score=lambda env: -(env["m"] * env["shared"]).square().sum(-1),
                batched=True,
            ),
        ],
        return_vars=("a",),
    )
    potential = PotentialFn(model, LatentRegistry.from_model(model, set()), _X, {})
    for rows in (1, 3, 5):
        _agree(potential, _positions(potential, rows))


def test_an_undeclared_step_is_refused_by_name_and_not_evaluated() -> None:
    calls: list[int] = []

    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        calls.append(1)
        return -env["a"].square()

    potential = _hand_potential(score, batched=False)
    with pytest.raises(UnbatchableProgramError, match="score step 'lik'") as refusal:
        potential.log_density_batch(torch.zeros(3, 1))
    assert [item.step for item in refusal.value.obstacles] == ["score step 'lik'"]
    assert "batched=True" in refusal.value.obstacles[0].reason
    with pytest.raises(UnbatchableProgramError):
        potential.value_and_grad_batch(torch.zeros(3, 1))
    assert not calls
    assert torch.isfinite(potential.log_density(torch.zeros(1)))


def test_every_undeclared_step_is_listed() -> None:
    class Opaque(ContinuousMorphism):
        """A prior that declares nothing."""

        def __init__(self) -> None:
            super().__init__(Unit, Euclidean(name="A", dim=1))

        def rsample(
            self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
        ) -> torch.Tensor:
            return torch.randn(x.shape[0], 1)

        def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return -y.square().sum(-1)

    model = MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(names=("a",), morphism=Opaque()),
            Let(name="b", value=lambda env: 2.0 * env["a"]),
            Score(name="s", score=lambda env: -env["b"].sum(-1), batched=True),
        ],
        return_vars=("a",),
    )
    potential = PotentialFn(model, LatentRegistry.from_model(model, set()), _X, {})
    with pytest.raises(UnbatchableProgramError) as refusal:
        potential.log_density_batch(torch.zeros(2, 1))
    assert [item.step for item in refusal.value.obstacles] == [
        "draw step 'a'",
        "let step 'b'",
    ]

    class Declared(Opaque):
        """The same prior, with its density declared."""

        mappable = True

    class Overridden(Declared):
        """A subclass whose own density is not the one declared."""

        def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return super().log_prob(x, y)

    assert Declared().mapping_obstacle() is None
    reason = Overridden().mapping_obstacle()
    assert reason is not None and "Overridden" in reason


def test_a_module_a_family_holds_declares_its_own_computation() -> None:
    family = ConditionalNormal(Euclidean(name="X", dim=2), Euclidean(name="Y", dim=1))
    assert family.mapping_obstacle() is None


@pytest.mark.parametrize(
    "score",
    [
        lambda env: env["a"].sum(),
        lambda env: env["a"].sum(0),
        lambda env: torch.zeros(5),
        lambda env: 0.0,
    ],
)
def test_a_batched_score_that_loses_the_positions_is_refused(
    score: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> None:
    potential = _hand_potential(score, batched=True)
    with pytest.raises(PositionLayoutError, match="score step 'lik'"):
        potential.log_density_batch(torch.zeros(3, 1))


def test_a_site_given_no_value_is_refused() -> None:
    model = _compiled(
        _program(
            """
sample a <- Normal(loc=0.0, scale=1.0)
observe y : Row <- Normal(loc=a, scale=1.0)
""",
            "a",
        )
    )
    registry = LatentRegistry.from_model(model, {"y"})
    potential = PotentialFn(model, registry, _X, {})
    with pytest.raises(UnbatchableProgramError, match="observe step 'y'"):
        potential.log_density_batch(torch.zeros(2, 1))


def test_weights_that_depend_on_a_latent_are_refused() -> None:
    """A zero weight drops its row, so which rows are scored would vary."""
    body = """
sample a <- Normal(loc=0.0, scale=1.0)
let w = exp(a * x)
observe y : Row <- Normal(loc=a, scale=1.0) [weights=w]
"""
    potential = _source_potential(
        _program(body, "a"), {"y": _COVARIATE, "x": _COVARIATE}
    )
    with pytest.raises(UnbatchableProgramError, match="'w'"):
        potential.log_density_batch(torch.zeros(2, 1))


def test_an_overridden_joint_is_refused() -> None:
    model = _hand_model(lambda env: -env["a"].square(), batched=True)
    vars(model)["log_joint"] = lambda x, values: values["a"].sum(-1)
    potential = PotentialFn(model, LatentRegistry.from_model(model, set()), _X, {})
    with pytest.raises(UnbatchableProgramError, match="overridden"):
        potential.log_density_batch(torch.zeros(2, 1))


def test_positions_of_the_wrong_shape_are_refused() -> None:
    potential = _hand_potential(lambda env: -env["a"].square(), batched=True)
    for shape in ((1,), (3, 2), (2, 3, 1)):
        with pytest.raises(ValueError, match=r"shape \(B, 1\)"):
            potential.log_density_batch(torch.zeros(shape))


def test_no_rows_and_no_coordinates() -> None:
    potential = _hand_potential(lambda env: -env["a"].square(), batched=True)
    assert potential.log_density_batch(torch.zeros(0, 1)).shape == (0,)
    values, grads = potential.value_and_grad_batch(torch.zeros(0, 1))
    assert values.shape == (0,) and grads.shape == (0, 1)
    held = _source_potential(
        _program("sample a <- Normal(loc=0.0, scale=1.0)", "a"), {}
    )
    model = held._model
    registry = LatentRegistry.from_model(model, {"a"})
    fixed = PotentialFn(model, registry, _X, {"a": torch.zeros(1, 1)})
    values, grads = fixed.value_and_grad_batch(torch.zeros(3, 0))
    assert values.shape == (3,) and grads.shape == (3, 0)
    assert torch.equal(values, fixed.log_density(torch.zeros(0)).expand(3))


# ---------------------------------------------------------------------------
# Work per call, and the potentials that share the evaluation
# ---------------------------------------------------------------------------


@contextmanager
def _replays() -> Iterator[list[int]]:
    """Count the plans replayed while the block runs."""
    original = JointPlan.evaluate
    calls = [0]

    def counted(
        self: JointPlan, x: torch.Tensor, values: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        calls[0] += 1
        return original(self, x, values)

    JointPlan.evaluate = counted
    try:
        yield calls
    finally:
        JointPlan.evaluate = original


@pytest.mark.parametrize("rows", [1, 2, 500])
def test_one_call_replays_the_plan_once(rows: int) -> None:
    body, returns, data = _STRUCTURES["marginalize"]
    potential = _source_potential(_program(body, returns), data)
    positions = _positions(potential, rows)
    potential.log_density_batch(positions)
    prepared = potential._plans.prepared
    with _replays() as calls:
        potential.log_density_batch(positions)
        assert calls[0] == 1
        potential.value_and_grad_batch(positions)
        assert calls[0] == 2
    potential.log_density(positions[0])
    assert potential._plans.prepared == prepared


def test_evaluations_are_counted_by_pass() -> None:
    potential = _hand_potential(lambda env: -env["a"].square(), batched=True)
    potential.log_density_batch(torch.zeros(9, 1))
    assert potential.num_evaluations == 1
    potential.value_and_grad_batch(torch.zeros(9, 1))
    assert potential.num_evaluations == 2
    assert potential.num_gradient_evaluations == 1


def test_a_value_of_data_alone_is_computed_once() -> None:
    shapes: list[tuple[int, ...]] = []

    def design(env: dict[str, torch.Tensor]) -> torch.Tensor:
        shapes.append(tuple(env["x"].shape))
        return env["x"].square()

    model = MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("a",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="A", dim=1)),
            ),
            Let(name="d", value=reading(design, ("x",)), batched=True),
            Score(
                name="s",
                score=lambda env: -(env["a"][..., None] * env["d"]).sum(-1),
                batched=True,
            ),
        ],
        return_vars=("a",),
    )
    registry = LatentRegistry.from_model(model, {"x"})
    potential = PotentialFn(model, registry, _X, {"x": _COVARIATE})
    potential.log_density_batch(torch.zeros(12, 1))
    shapes.clear()
    potential.log_density_batch(torch.zeros(12, 1))
    assert shapes == [(12,)]


def test_a_tempered_potential_scores_a_population() -> None:
    body, returns, data = _STRUCTURES["correlated_rows"]
    plain = _source_potential(_program(body, returns), data)
    for temperature in (0.0, 0.4, 1.0):
        tempered = TemperedPotential(
            plain._model, plain.registry, _X, dict(data), temperature
        )
        _agree(tempered, _positions(tempered, 4))


def test_a_conditioned_potential_holds_its_sites_for_every_row() -> None:
    body = """
sample a <- Normal(loc=0.0, scale=1.0)
sample v : Group <- Normal(loc=a, scale=1.0)
let loc = v[group]
observe y : Row <- Normal(loc=loc, scale=1.0)
"""
    data = {"y": _COVARIATE, "group": _GROUP}
    model = _compiled(_program(body, "a"))
    registry = LatentRegistry.from_model(model, {"y", "group", "v"})
    base = PotentialFn(model, registry, _X, data)
    held = base.condition({"v": torch.linspace(0.0, 1.0, 4)})
    _agree(held, _positions(held, 4))


@pytest.mark.parametrize("name", ["eight_schools_noncentered", "neal_funnel"])
def test_each_benchmark_model(name: str) -> None:
    data = getattr(datasets, name)()
    model = data.model.double()
    observations = {
        key: value.double() if value.is_floating_point() else value
        for key, value in data.observations.items()
    }
    registry = LatentRegistry.from_model(model, set(observations))
    potential = PotentialFn(model, registry, _X, observations)
    for rows in (1, 8, 9):
        _agree(potential, _positions(potential, rows))


# ---------------------------------------------------------------------------
# The runs of a joint
# ---------------------------------------------------------------------------


def _joint_by_rows(
    model: MonadicProgram, x: torch.Tensor, values: dict[str, torch.Tensor]
) -> torch.Tensor:
    """The joint of each run, scored one run at a time."""
    return torch.cat(
        [
            model.log_joint(
                x[index : index + 1],
                {name: value[index] for name, value in values.items()},
            )
            for index in range(x.shape[0])
        ]
    )


@contextmanager
def _machines() -> Iterator[list[int]]:
    """Count the reference machine's evaluators while the block runs."""
    original = interpreter.Evaluator
    calls = [0]

    class Counted(original):
        """An evaluator whose construction is counted."""

        def __new__(cls, *_args: object, **_kwargs: object) -> Counted:
            calls[0] += 1
            return super().__new__(cls)

    interpreter.Evaluator = Counted
    try:
        yield calls
    finally:
        interpreter.Evaluator = original


def _runs(rows: int) -> tuple[MonadicProgram, torch.Tensor, dict[str, torch.Tensor]]:
    """A plated program and one draw of its sites per run."""
    model = _compiled(
        _program(
            """
sample a <- Normal(loc=0.0, scale=1.0)
sample v : Group <- Normal(loc=a, scale=2.0)
let centered = v - mean(v)
score s = sum(centered) * a
""",
            "a",
            "Val",
        )
    )
    generator = torch.Generator().manual_seed(1)
    values = {
        "a": torch.randn(rows, 1, generator=generator),
        "v": torch.randn(rows, 4, generator=generator),
    }
    return model, torch.zeros(rows, 1), values


@pytest.mark.parametrize("rows", [2, 4, 6])
def test_the_runs_of_a_joint_are_scored_together(rows: int) -> None:
    model, x, values = _runs(rows)
    with _machines() as calls:
        joint = model.log_joint(x, values)
        assert calls[0] == 1
    torch.testing.assert_close(
        joint, _joint_by_rows(model, x, values), rtol=_RTOL, atol=1e-12
    )


def test_a_joint_given_shared_values_and_gradients() -> None:
    model, x, values = _runs(5)
    shared = {"a": values["a"].requires_grad_(), "v": torch.zeros(4)}
    joint = model.log_joint(x, shared)
    (grad,) = torch.autograd.grad(joint.sum(), shared["a"])
    own = values["a"].detach().clone().requires_grad_()
    single = torch.cat(
        [
            model.log_joint(x[:1], {"a": own[index], "v": torch.zeros(4)})
            for index in range(5)
        ]
    )
    (expected,) = torch.autograd.grad(single.sum(), own)
    torch.testing.assert_close(joint, single, rtol=_RTOL, atol=1e-12)
    torch.testing.assert_close(grad, expected, rtol=1e-12, atol=1e-12)


def test_a_failing_run_of_a_joint_raises_its_own_error() -> None:
    model = _compiled(
        _program(
            """
sample a <- Normal(loc=0.0, scale=1.0)
sample v : Group <- Normal(loc=0.0, scale=a)
""",
            "a",
            "Val",
        )
    )
    values = {"a": torch.tensor([[1.0], [-1.0], [2.0]]), "v": torch.zeros(3, 4)}
    with pytest.raises(ValueError) as together:
        model.log_joint(torch.zeros(3, 1), values)
    with pytest.raises(ValueError) as alone:
        model.log_joint(torch.zeros(1, 1), {"a": values["a"][1], "v": values["v"][1]})
    assert str(together.value) == str(alone.value)


def test_the_parameters_of_a_family_are_differentiated_in_a_batch() -> None:
    """A learned family's parameters take the gradient of one call per row."""
    body = """
sample a <- Normal(loc=0.0, scale=1.0)
sample b <- f(a)
"""
    source = (
        "object X : Real 1\nmorphism f : X -> X ~ Normal\nprogram p : X -> X\n"
        + "\n".join(f"    {line}" for line in body.strip().splitlines())
        + "\n    return b\nexport p\n"
    )
    model = _compiled(source).double()
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, _X, {})
    rows = _positions(potential, 3)
    parameters = tuple(model.parameters())
    together = torch.autograd.grad(potential.log_density_batch(rows).sum(), parameters)
    apart = torch.autograd.grad(
        torch.stack([potential.log_density(row) for row in rows]).sum(), parameters
    )
    for one, other in zip(together, apart, strict=True):
        torch.testing.assert_close(one, other, rtol=1e-10, atol=1e-12)


def test_the_log_jacobian_of_every_row_is_its_own() -> None:
    """A half-normal prior alone has the density of its exponential map."""
    potential = _source_potential(
        _program("sample a <- HalfNormal(scale=2.0)", "a", "Val"), {}
    )
    rows = torch.linspace(-2.0, 1.0, 6).unsqueeze(-1)
    constrained, log_det = potential.registry.to_constrained(
        potential.registry.unflatten_unconstrained(rows)
    )
    value = constrained["a"].reshape(6)
    expected = (
        Normal(0.0, 2.0).log_prob(value) + math.log(2.0) + log_det["a"].reshape(6)
    )
    torch.testing.assert_close(
        potential.log_density_batch(rows), expected, rtol=_RTOL, atol=1e-12
    )
