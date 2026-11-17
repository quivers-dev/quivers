"""Response-time log tails agree with independent high-precision probability laws."""

import math

import mpmath as mp
import pytest
import torch

from quivers.continuous import ExGaussian, InverseGaussian
from quivers.continuous._gaussian_tails import log_scaled_erfcx_difference
from quivers.qiec._gaussian_tails import log_scaled_erfcx_difference as host_difference
from quivers.qiec._response_time import log_response_time_tails
from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec._interval import log_interval_mass as host_interval
from quivers.continuous._interval import log_interval_mass


def _mp_difference(x, d):
    # Computing the difference independently in arbitrary precision also
    # tests the factored derivative rule used when d is below one ulp of x.
    precision = max(100, int(-math.log10(d)) + 60)
    with mp.workdps(precision):
        x, d = mp.mpf(x), mp.mpf(d)
        return float(mp.log(mp.erfc(x) - mp.exp(d * (2 * x + d)) * mp.erfc(x + d)))


@pytest.mark.parametrize(
    "x,d",
    [
        (-1000.0, 1e-300),
        (-2.0, 1e-16),
        (-2.0, 0.1),
        (0.0, 1e-30),
        (0.0, 0.1),
        (2.0, 1.0),
        (7.9, 1e-12),
        (8.0, 1e-12),
        (100.0, 1e-12),
        (1e4, 10.0),
        (-2.0, 5.0),
        (-0.2, 1.0),
    ],
)
def test_scaled_gaussian_difference_matches_high_precision(x, d):
    expected = _mp_difference(x, d)
    value = log_scaled_erfcx_difference(
        torch.tensor(x, dtype=torch.double), torch.tensor(d, dtype=torch.double)
    )
    assert value.item() == pytest.approx(expected, abs=2e-10, rel=2e-15)
    assert host_difference(x, d) == pytest.approx(expected, abs=2e-10, rel=2e-15)


@pytest.mark.parametrize(
    "x,d",
    [
        (-2.0, 0.1),
        (0.0, 0.1),
        (2.0, 1.0),
        (5.0, 0.001),
        (7.9, 0.001),
        (8.0, 0.001),
        (100.0, 0.001),
    ],
)
def test_gaussian_difference_first_second_and_forward_derivatives(x, d):
    args = (
        torch.tensor(x, dtype=torch.double, requires_grad=True),
        torch.tensor(math.log(d), dtype=torch.double, requires_grad=True),
    )

    def f(a, b):
        return log_scaled_erfcx_difference(a, b.exp())

    assert torch.autograd.gradcheck(f, args)
    assert torch.autograd.gradgradcheck(f, args)
    reverse = torch.stack(torch.autograd.grad(f(*args), args))
    forward = torch.stack(torch.func.jacfwd(f, argnums=(0, 1))(*args))
    torch.testing.assert_close(forward, reverse, rtol=1e-10, atol=1e-10)


def _oracle(family, parameters, point):
    with mp.workdps(250):
        p = list(map(mp.mpf, parameters))
        y = mp.mpf(point)

        def phi(a):
            return mp.erfc(-a / mp.sqrt(2)) / 2

        if family == "ExGaussian":
            loc, scale, rate = p
            z = (y - loc) / scale
            r = rate * scale
            term = mp.exp(r * r / 2 - r * z) * phi(z - r)
            lower = phi(z) - term
            upper = phi(-z) + term
        else:
            mu, shape = p
            a = mp.sqrt(shape / y) * (y / mu - 1)
            b = mp.sqrt(shape / y) * (y / mu + 1)
            term = mp.exp(2 * shape / mu) * phi(-b)
            lower = phi(a) + term
            upper = phi(-a) - term
        return float(mp.log(lower)), float(mp.log(upper))


_CASES = [
    ("ExGaussian", (0.0, 1.0, 0.1), -40.0),
    ("ExGaussian", (0.3, 0.7, 1.4), 0.5),
    ("ExGaussian", (0.0, 1.0, 1e-100), 1e3),
    ("ExGaussian", (0.0, 1.0, 0.1), 1e4),
    ("ExGaussian", (0.0, 1.0, 1e4), -1.0),
    ("InverseGaussian", (2.0, 3.0), 0.001),
    ("InverseGaussian", (2.0, 3.0), 1000.0),
    ("InverseGaussian", (1e10, 1.0), 1e10),
    ("InverseGaussian", (1e-3, 1e-4), 1e6),
    ("InverseGaussian", (2.0, 3.0), 1.0),
]


@pytest.mark.parametrize("family,parameters,point", _CASES)
def test_native_and_reference_paired_tails_match_independent_law(
    family, parameters, point
):
    cls = ExGaussian if family == "ExGaussian" else InverseGaussian
    law = cls(*(torch.tensor(p, dtype=torch.double) for p in parameters))
    y = torch.tensor(point, dtype=torch.double)
    actual = (law.log_cdf(y).item(), law.log_survival_function(y).item())
    names = ("loc", "scale", "rate") if family == "ExGaussian" else ("mu", "shape")
    host = log_response_time_tails(family, dict(zip(names, parameters)), point)
    expected = _oracle(family, parameters, point)
    for a, h, e in zip(actual, host, expected, strict=True):
        assert a == pytest.approx(e, abs=3e-10, rel=2e-13)
        assert h == pytest.approx(e, abs=3e-10, rel=2e-13)
    assert torch.logaddexp(
        law.log_cdf(y), law.log_survival_function(y)
    ).item() == pytest.approx(0.0, abs=2e-16)


@pytest.mark.parametrize(
    "cls,parameters,point",
    [
        (ExGaussian, (0.3, 0.7, 1.4), -0.5),
        (ExGaussian, (0.3, 0.7, 1.4), 8.0),
        (ExGaussian, (0.0, 1.0, 0.0001), 100.0),
        (InverseGaussian, (2.0, 3.0), 0.1),
        (InverseGaussian, (2.0, 3.0), 100.0),
        (InverseGaussian, (1e5, 1.0), 1e5),
    ],
)
@pytest.mark.parametrize("tail", ["log_cdf", "log_survival_function"])
def test_response_time_tail_all_head_derivatives(cls, parameters, point, tail):
    args = tuple(
        torch.tensor(p, dtype=torch.double, requires_grad=True) for p in parameters
    )
    y = torch.tensor(point, dtype=torch.double)

    def f(*p):
        return getattr(cls(*p), tail)(y)

    # Perturb positive parameters multiplicatively, especially tiny rates.
    logs = tuple(
        a if i == 0 and cls is ExGaussian else a.log().detach().requires_grad_()
        for i, a in enumerate(args)
    )

    def g(*p):
        return f(
            *(a if i == 0 and cls is ExGaussian else a.exp() for i, a in enumerate(p))
        )

    assert torch.autograd.gradcheck(g, logs, atol=2e-6, rtol=2e-4)
    assert torch.autograd.gradgradcheck(g, logs, atol=2e-5, rtol=2e-4)
    reverse = torch.stack(torch.autograd.grad(g(*logs), logs))
    forward = torch.stack(torch.func.jacfwd(g, argnums=tuple(range(len(logs))))(*logs))
    torch.testing.assert_close(forward, reverse, rtol=1e-9, atol=1e-9)
    # Independent probability-law finite differences, not differences of the
    # implementation's paired-tail expressions.
    index = 0 if tail == "log_cdf" else 1
    family = cls.__name__
    for i in range(len(parameters)):
        step = 1e-4
        up = list(parameters)
        down = list(parameters)
        if i == 0 and cls is ExGaussian:
            up[i] += step
            down[i] -= step
        else:
            up[i] *= math.exp(step)
            down[i] *= math.exp(-step)
        expected = (
            _oracle(family, up, point)[index] - _oracle(family, down, point)[index]
        ) / (2 * step)
        assert reverse[i].item() == pytest.approx(expected, abs=3e-6, rel=3e-7)


@pytest.mark.parametrize(
    "cls,parameters", [(ExGaussian, (0.3, 0.7, 1.4)), (InverseGaussian, (2.0, 3.0))]
)
def test_tail_support_endpoints_and_nan(cls, parameters):
    args = tuple(
        torch.tensor(p, dtype=torch.double, requires_grad=True) for p in parameters
    )
    law = cls(*args)
    x = torch.tensor([-math.inf, math.inf], dtype=torch.double)
    torch.testing.assert_close(
        law.log_cdf(x), torch.tensor([-math.inf, 0.0], dtype=torch.double)
    )
    torch.testing.assert_close(
        law.log_survival_function(x), torch.tensor([0.0, -math.inf], dtype=torch.double)
    )
    grads = torch.autograd.grad(
        law.log_cdf(x).sum() + law.log_survival_function(x).sum(), args
    )
    assert all(torch.equal(g, torch.zeros_like(g)) for g in grads)
    with pytest.raises(ValueError, match="NaN"):
        law.log_cdf(torch.tensor(math.nan))


@pytest.mark.parametrize(
    "family,params,lower",
    [
        ("ExGaussian", (0.0, 1.0, 0.1), -40.0),
        ("ExGaussian", (0.0, 1.0, 0.1), 100.0),
        ("InverseGaussian", (2.0, 3.0), 0.001),
        ("InverseGaussian", (2.0, 3.0), 1000.0),
    ],
)
def test_native_reference_interval_tails_retain_extreme_mass(family, params, lower):
    names = ("loc", "scale", "rate") if family == "ExGaussian" else ("mu", "shape")
    arguments = dict(zip(names, params))
    base = RuntimeDistribution(family=family, arguments=arguments)
    cls = ExGaussian if family == "ExGaussian" else InverseGaussian
    law = cls(*(torch.tensor(p, dtype=torch.double) for p in params))
    upper = lower + (0.00001 if lower < 1 else 0.01)
    with mp.workdps(250):
        lo = _oracle(family, params, lower)
        hi = _oracle(family, params, upper)
        expected = (
            float(mp.log(mp.exp(hi[0]) - mp.exp(lo[0])))
            if hi[0] < -0.5
            else float(mp.log(mp.exp(lo[1]) - mp.exp(hi[1])))
        )
    actual = log_interval_mass(
        law,
        torch.tensor(lower, dtype=torch.double),
        torch.tensor(upper, dtype=torch.double),
    ).item()
    host = host_interval(base, lower, upper)
    assert actual == pytest.approx(expected, abs=1e-8)
    assert host == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    "family,parameters,point",
    [
        ("ExGaussian", (0.0, 1.0, 0.1), 0.1),
        ("ExGaussian", (0.0, 1.0, 0.1), 0.0),
        ("ExGaussian", (0.0, 1.0, 0.1), -40.0),
        ("ExGaussian", (0.0, 1.0, 0.1), 100.0),
        ("InverseGaussian", (2.0, 3.0), 0.001),
        ("InverseGaussian", (2.0, 3.0), 0.1),
        ("InverseGaussian", (2.0, 3.0), 100.0),
    ],
)
def test_adjacent_bounds_use_shared_local_density_instead_of_tail_subtraction(
    family, parameters, point
):
    cls = ExGaussian if family == "ExGaussian" else InverseGaussian
    parameters = tuple(
        torch.tensor(p, dtype=torch.double, requires_grad=True) for p in parameters
    )
    law = cls(*parameters)
    low = torch.tensor(point, dtype=torch.double)
    high = torch.nextafter(low, torch.tensor(math.inf, dtype=torch.double))
    actual = log_interval_mass(law, low, high, lower_closed=False, upper_closed=False)
    # Over one representable step the smooth density is constant to the
    # asserted tolerance. This oracle never subtracts two CDF evaluations.
    expected = law.log_prob(low) + math.log((high - low).item())
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
    actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, parameters)
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e, rtol=2e-12, atol=2e-12)
    names = ("loc", "scale", "rate") if family == "ExGaussian" else ("mu", "shape")
    args = dict(zip(names, (p.item() for p in parameters)))
    host = host_interval(
        RuntimeDistribution(family=family, arguments=args),
        low.item(),
        high.item(),
        lower_closed=False,
    )
    assert host == pytest.approx(expected.item(), abs=3e-12)


@pytest.mark.parametrize(
    "cls,parameters", [(ExGaussian, (0.3, 0.7, 1.4)), (InverseGaussian, (2.0, 3.0))]
)
def test_batched_tails_and_expansion_keep_batch_axes_and_original_cdf_dtype(
    cls, parameters
):
    args = [
        torch.tensor([p, p * 1.1], dtype=torch.double).unsqueeze(-1) for p in parameters
    ]
    law = cls(*args).expand((2, 3))
    points = torch.tensor([0.1, 1.0, 10.0], dtype=torch.double)
    actual = law.log_cdf(points)
    sf = law.log_survival_function(points)
    assert actual.shape == sf.shape == (2, 3)
    for row in range(2):
        expected = cls(*(arg[row, 0] for arg in args))
        torch.testing.assert_close(actual[row], expected.log_cdf(points))
        torch.testing.assert_close(sf[row], expected.log_survival_function(points))
    plain = cls(*parameters)
    assert plain.cdf(points.float()).dtype == torch.float32
    assert plain.log_cdf(points.float()).dtype == torch.float64


def test_generic_interval_consumes_one_paired_tail_evaluation_per_endpoint(monkeypatch):
    from quivers.continuous._interval import _log_tails

    law = ExGaussian(0.0, 1.0, 0.1)
    original = law._log_tails
    calls = []

    def paired(point):
        calls.append(point)
        return original(point)

    def separate(_point):
        pytest.fail("the paired capability must avoid evaluating separate tails")

    monkeypatch.setattr(law, "_log_tails", paired)
    monkeypatch.setattr(law, "log_cdf", separate)
    monkeypatch.setattr(law, "log_survival_function", separate)
    lower, upper = _log_tails(law, torch.tensor(1.0))
    assert len(calls) == 1
    assert math.isfinite(lower.item()) and math.isfinite(upper.item())


@pytest.mark.parametrize(
    "cls,parameters", [(ExGaussian, (0.3, 0.7, 1.4)), (InverseGaussian, (2.0, 3.0))]
)
def test_invalid_smooth_intervals_are_not_accepted_by_variation_rule(cls, parameters):
    law = cls(*parameters)
    lo = torch.tensor([-math.inf, 1.0, 2.0, math.nan], dtype=torch.double)
    hi = torch.tensor([1.0, math.inf, 1.0, 3.0], dtype=torch.double)
    assert torch.isposinf(law.log_density_variation(lo, hi)).all()
    torch.testing.assert_close(
        log_interval_mass(law, 2.0, 1.0), torch.tensor(-math.inf, dtype=torch.double)
    )


@pytest.mark.parametrize(
    "family,parameters,low,high,point",
    [
        ("ExGaussian", {"loc": 0.0, "scale": 1.0, "rate": 0.1}, -40.0, -39.9, -39.95),
        ("InverseGaussian", {"mu": 2.0, "shape": 3.0}, 1000.0, 1001.0, 1000.5),
    ],
)
def test_checked_native_and_reference_truncation_share_tail_normalizer(
    family, parameters, low, high, point
):
    from quivers.dsl import loads
    from quivers.dsl.parser import parse
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.inference import trace
    from quivers.qiec.program_runtime import run_program

    args = ", ".join(f"{key}={value}" for key, value in parameters.items())
    source = f"""object Row : FinSet 1
program model(y:Tensor[Real]([Row])) : Row -> Row
    observe y:Row <- {family}({args}) [trunc_lower={low},trunc_upper={high}]
    return y
export model
"""
    # Keep literal construction in the declared Double numerical context;
    # fixed source constants follow Torch's configured default dtype.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.double)
        module = loads(source).morphism
        y = torch.tensor([[point]], dtype=torch.double)
        native = trace(module, torch.zeros(1, 1), {"y": y}).log_joint
    finally:
        torch.set_default_dtype(previous)
    checked = lower_qvr_to_qiec(parse(source))
    host = run_program(checked, "model", data={"y": (point,)}, sites={}).log_joint
    cls = ExGaussian if family == "ExGaussian" else InverseGaussian
    law = cls(*(torch.tensor(p, dtype=torch.double) for p in parameters.values()))
    mass = host_interval(
        RuntimeDistribution(family=family, arguments=parameters), low, high
    )
    expected = law.log_prob(torch.tensor(point, dtype=torch.double)).item() - mass
    assert native.item() == pytest.approx(expected, abs=2e-10)
    assert host == pytest.approx(expected, abs=2e-10)


@pytest.mark.parametrize("x,d", [(1e151, 1.0), (-1e151, 1.0), (0.0, 1e151)])
def test_gaussian_tail_capacity_refusal_is_explicit(x, d):
    from quivers.qiec.reference_support import DistributionError

    with pytest.raises(RuntimeError, match="capacity"):
        log_scaled_erfcx_difference(
            torch.tensor(x, dtype=torch.double), torch.tensor(d, dtype=torch.double)
        )
    with pytest.raises(DistributionError, match="capacity"):
        host_difference(x, d)


def test_short_difference_checks_bounded_rule_agreement(monkeypatch):
    import quivers.continuous._gaussian_tails as native
    import quivers.qiec._gaussian_tails as reference
    from quivers.qiec.reference_support import DistributionError

    monkeypatch.setattr(native, "_WEIGHTS_8", tuple(2 * w for w in native._WEIGHTS_8))
    monkeypatch.setattr(
        reference, "_WEIGHTS_8", tuple(2 * w for w in reference._WEIGHTS_8)
    )
    with pytest.raises(RuntimeError, match="converge"):
        native.log_scaled_erfcx_difference(torch.tensor(0.0), torch.tensor(1e-10))
    with pytest.raises(DistributionError, match="converge"):
        reference.log_scaled_erfcx_difference(0.0, 1e-10)


@pytest.mark.parametrize(
    "family,parameters,low,high,point",
    [
        ("ExGaussian", {"loc": 0.0, "scale": 1.0, "rate": 0.1}, -40.0, -39.9, -39.95),
        ("InverseGaussian", {"mu": 2.0, "shape": 3.0}, 1000.0, 1001.0, 1000.5),
    ],
)
def test_checked_truncated_response_all_parameter_gradients_match_reference(
    family, parameters, low, high, point
):
    from quivers.dsl import loads
    from quivers.dsl.parser import parse
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.inference import trace
    from quivers.qiec.program_runtime import run_program

    declarations = ", ".join(name + ":Real" for name in parameters)
    arguments = ", ".join(f"{name}={name}" for name in parameters)
    source = f"""object Row : FinSet 1
program model({declarations}, y:Tensor[Real]([Row])) : Row -> Row
    observe y:Row <- {family}({arguments}) [trunc_lower={low},trunc_upper={high}]
    return y
export model
"""
    native = loads(source).morphism
    inputs = {
        name: torch.tensor(value, dtype=torch.double, requires_grad=True)
        for name, value in parameters.items()
    }
    score = trace(
        native,
        torch.zeros(1, 1, dtype=torch.double),
        inputs | {"y": torch.tensor([[point]], dtype=torch.double)},
    ).log_joint.sum()
    grads = torch.autograd.grad(score, tuple(inputs.values()))
    checked = lower_qvr_to_qiec(parse(source))
    data = parameters | {"y": (point,)}
    for (name, value), actual in zip(parameters.items(), grads, strict=True):
        step = 1e-4 * max(abs(value), 0.1)
        plus = run_program(
            checked, "model", data=data | {name: value + step}, sites={}
        ).log_joint
        minus = run_program(
            checked, "model", data=data | {name: value - step}, sites={}
        ).log_joint
        assert actual.item() == pytest.approx(
            (plus - minus) / (2 * step), rel=5e-5, abs=1e-6
        )
