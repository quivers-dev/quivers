"""Calls the compiled runtime lowers to steps, and the calls it refuses.

A program's call of a computation is fit when the callee's body can be
stated as a fixed chain of compiled steps:

* pure calls nested inside pure callees are inlined in turn;
* an ``if`` between pure branches is ``select`` of its branches;
* a callee whose only requests are the program's ``Random`` and
  ``Score`` runs in place, its sites named ``binder$label``;
* ``traverse`` maps a pure callee as a comprehension and a drawing
  callee as one plated draw per site.

Every newly fit form is checked against a float64 oracle written here
by hand: the reference machine at rel 1e-9, the compiled runtime at rel
1e-5 (float32 arithmetic on a joint of order ten), its gradients against
autograd on the oracle, and one NUTS recovery per form against a
closed-form Gaussian posterior. Each check is shown to reject a wrong
variant of the model. Every refusal is checked on both paths: the
compiled runtime refuses to fit the program, located at the call and
saying why, and the reference machine still runs it.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.diagnostics import ess as effective_sample_size
from quivers.dsl import loads
from quivers.dsl.compiler._prelude import CompileError
from quivers.effects.checked_program import CheckedProgram
from quivers.inference import MCMC, NUTSKernel
from quivers.inference.trace import trace
from quivers.qiec.execution import ExecutionFailure

_F64 = torch.float64
_Y4 = (0.4, -0.3, 0.9, 0.2)
_Y8 = (0.5, -0.1, 0.8, 0.3, 1.1, -0.4, 0.6, 0.2)


def _normal(value: torch.Tensor, loc: torch.Tensor, scale: float) -> torch.Tensor:
    """The float64 normal log density, summed."""
    return torch.distributions.Normal(loc, scale).log_prob(value).sum()


def _compiled(source: str, values: dict[str, torch.Tensor]) -> float:
    """The compiled runtime's log joint of one run."""
    program = loads(source).morphism
    assert isinstance(program, MonadicProgram)
    return float(program.log_joint(torch.zeros(1, 1), values).sum())


def _reference(
    source: str, entry: str, sites: dict[str, object], data: dict[str, object]
) -> float:
    """The reference machine's log joint of one run, in float64."""
    return float(loads(source).run(entry, data=data, sites=sites, seed=0).log_joint)


def _draws(seed: int, count: int) -> list[float]:
    """Seeded standard normal site values."""
    generator = torch.Generator().manual_seed(seed)
    return [float(v) for v in torch.randn(count, generator=generator)]


# ---------------------------------------------------------------------------
# Nested pure calls
# ---------------------------------------------------------------------------

NESTED = """
define bump(x : Real) : Real !{} =
    return x + 1.0

define scaled_bump(x : Real, s : Real) : Real !{} =
    let once <- bump(x)
    let twice <- bump(once * s)
    return twice

object Row : FinSet 8

program nested : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample s <- Normal(loc=1.0, scale=0.5)
    let m <- scaled_bump(a, s)
    observe y : Row <- Normal(loc=m, scale=1.0)
    return y

export nested
"""


def _nested_oracle(a: torch.Tensor, s: torch.Tensor, inner_once: bool) -> torch.Tensor:
    """The nested program's joint; ``inner_once`` drops the outer bump."""
    y = torch.tensor(_Y8, dtype=_F64)
    mean = (a + 1.0) * s + (0.0 if inner_once else 1.0)
    return _normal(a, torch.tensor(0.0, dtype=_F64), 1.0) + (
        _normal(s, torch.tensor(1.0, dtype=_F64), 0.5) + _normal(y, mean, 1.0)
    )


def test_nested_pure_calls_compile_to_let_steps() -> None:
    program = loads(NESTED).morphism
    assert isinstance(program, MonadicProgram)
    names = [getattr(step, "name", None) for step in program.steps]
    assert names[-1] is None and "m" in names


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nested_calls_agree_with_the_oracle_on_both_paths(seed: int) -> None:
    a, s = _draws(seed, 2)
    oracle = float(
        _nested_oracle(torch.tensor(a, dtype=_F64), torch.tensor(s, dtype=_F64), False)
    )
    reference = _reference(NESTED, "nested", {"a": a, "s": s}, {"y": _Y8})
    compiled = _compiled(
        NESTED,
        {"a": torch.tensor([a]), "s": torch.tensor([s]), "y": torch.tensor(_Y8)},
    )
    assert reference == pytest.approx(oracle, rel=1e-9)
    assert compiled == pytest.approx(oracle, rel=1e-5)
    wrong = float(
        _nested_oracle(torch.tensor(a, dtype=_F64), torch.tensor(s, dtype=_F64), True)
    )
    # One bump fewer moves every mean by 1: eight residuals change, far
    # outside the tolerance.
    assert compiled != pytest.approx(wrong, rel=1e-3)


def test_nested_calls_have_the_oracles_gradient() -> None:
    program = loads(NESTED).morphism
    assert isinstance(program, MonadicProgram)
    a = torch.tensor([0.3], requires_grad=True)
    s = torch.tensor([0.8], requires_grad=True)
    total = program.log_joint(
        torch.zeros(1, 1), {"a": a, "s": s, "y": torch.tensor(_Y8)}
    ).sum()
    grads = torch.autograd.grad(total, (a, s))
    a64 = torch.tensor(0.3, dtype=_F64, requires_grad=True)
    s64 = torch.tensor(0.8, dtype=_F64, requires_grad=True)
    expected = torch.autograd.grad(_nested_oracle(a64, s64, False), (a64, s64))
    for grad, value in zip(grads, expected, strict=True):
        assert float(grad) == pytest.approx(float(value), rel=1e-5, abs=1e-5)


NESTED_LINE = """
define bump(x : Real) : Real !{} =
    return x + 1.0

define line(x : Real) : Real !{} =
    let once <- bump(x)
    return 2.0 * once

object Row : FinSet 8

program nested_line : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- line(a)
    observe y : Row <- Normal(loc=m, scale=1.0)
    return y

export nested_line
"""


def _gaussian_check(
    draws: torch.Tensor, mean: float, variance: float
) -> tuple[float, float]:
    """The posterior mean's error and its Monte Carlo bound, 5 sd / sqrt(ess)."""
    draws = draws.to(_F64).reshape(draws.shape[0], -1)
    ess = float(effective_sample_size(draws, method="mean"))
    return abs(float(draws.mean()) - mean), 5.0 * math.sqrt(variance / ess)


def _run_nuts(source: str, observations: dict[str, torch.Tensor]):
    program = loads(source).morphism
    assert isinstance(program, MonadicProgram)
    return MCMC(
        NUTSKernel(max_tree_depth=5),
        num_warmup=300,
        num_samples=500,
        num_chains=2,
        seed=0,
    ).run(program, torch.zeros(1, 1), observations)


def test_nuts_recovers_the_posterior_through_nested_calls() -> None:
    """y_i ~ N(2 a + 2, 1), a ~ N(0, 1): a | y is Gaussian with precision
    1 + 4 n and mean 2 sum(y_i - 2) over that precision."""
    y = torch.tensor(_Y8)
    precision = 1.0 + 4.0 * 8
    mean = 2.0 * float((y.to(_F64) - 2.0).sum()) / precision
    result = _run_nuts(NESTED_LINE, {"y": y})
    error, bound = _gaussian_check(result.samples["a"], mean, 1.0 / precision)
    assert error < bound
    # The model with the inner bump dropped (mean 2 a) has its posterior
    # mean 2 sum(y)/precision away by 16/precision, which the bound rejects.
    wrong = 2.0 * float(y.to(_F64).sum()) / precision
    assert abs(float(result.samples["a"].to(_F64).mean()) - wrong) > bound


# ---------------------------------------------------------------------------
# Conditionals lowered to select
# ---------------------------------------------------------------------------

LEAKY = """
define leaky(x : Real) : Real !{} =
    if x > 0.0 then
        return x
    else
        let small = 0.1 * x
        return small

define clamp(x : Real, low : Real, high : Real) : Real !{} =
    if x < low then
        return low
    else
        if x > high then
            return high
        else
            return x

object Row : FinSet 8

program rectified : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    let m <- leaky(a)
    let c <- clamp(b, -0.5, 0.5)
    let both = m + c
    observe y : Row <- Normal(loc=both, scale=1.0)
    return y

export rectified
"""


def _leaky_oracle(a: torch.Tensor, b: torch.Tensor, swapped: bool) -> torch.Tensor:
    """The rectified program's joint; ``swapped`` exchanges leaky's branches."""
    y = torch.tensor(_Y8, dtype=_F64)
    keep = a > 0.0
    if swapped:
        keep = ~keep
    m = torch.where(keep, a, 0.1 * a)
    c = torch.clamp(b, -0.5, 0.5)
    zero = torch.tensor(0.0, dtype=_F64)
    return _normal(a, zero, 1.0) + _normal(b, zero, 1.0) + _normal(y, m + c, 1.0)


def test_a_conditional_lowers_to_select() -> None:
    program = loads(LEAKY).morphism
    assert isinstance(program, MonadicProgram)


@pytest.mark.parametrize(
    ("a", "b"), [(0.7, 0.2), (-0.6, -1.3), (1.4, 0.9), (-2.0, 0.1)]
)
def test_select_agrees_with_the_oracle_on_both_paths(a: float, b: float) -> None:
    oracle = float(
        _leaky_oracle(torch.tensor(a, dtype=_F64), torch.tensor(b, dtype=_F64), False)
    )
    reference = _reference(LEAKY, "rectified", {"a": a, "b": b}, {"y": _Y8})
    compiled = _compiled(
        LEAKY,
        {"a": torch.tensor([a]), "b": torch.tensor([b]), "y": torch.tensor(_Y8)},
    )
    assert reference == pytest.approx(oracle, rel=1e-9)
    assert compiled == pytest.approx(oracle, rel=1e-5)
    swapped = float(
        _leaky_oracle(torch.tensor(a, dtype=_F64), torch.tensor(b, dtype=_F64), True)
    )
    # Swapping the branches moves m by 0.9 |a| >= 0.54 at every point
    # here, eight residuals at once.
    assert compiled != pytest.approx(swapped, rel=1e-3)


@pytest.mark.parametrize("a", [0.7, -0.6])
def test_select_has_the_piecewise_gradient(a: float) -> None:
    program = loads(LEAKY).morphism
    assert isinstance(program, MonadicProgram)
    a_t = torch.tensor([a], requires_grad=True)
    b_t = torch.tensor([0.2], requires_grad=True)
    total = program.log_joint(
        torch.zeros(1, 1), {"a": a_t, "b": b_t, "y": torch.tensor(_Y8)}
    ).sum()
    grads = torch.autograd.grad(total, (a_t, b_t))
    a64 = torch.tensor(a, dtype=_F64, requires_grad=True)
    b64 = torch.tensor(0.2, dtype=_F64, requires_grad=True)
    expected = torch.autograd.grad(_leaky_oracle(a64, b64, False), (a64, b64))
    for grad, value in zip(grads, expected, strict=True):
        assert float(grad) == pytest.approx(float(value), rel=1e-5, abs=1e-5)


LEAKY_ONLY = """
define leaky(x : Real) : Real !{} =
    if x > 0.0 then
        return x
    else
        let small = 0.1 * x
        return small

object Row : FinSet 8

program leaky_mean : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- leaky(a)
    observe y : Row <- Normal(loc=m, scale=1.0)
    return y

export leaky_mean
"""


def test_nuts_recovers_the_posterior_through_select() -> None:
    """With sum(y) = 16 the posterior N(16/9, 1/9) puts mass below zero
    of order 1e-8, so a | y is that Gaussian to within the test's error."""
    y = torch.full((8,), 2.0)
    mean, variance = 16.0 / 9.0, 1.0 / 9.0
    result = _run_nuts(LEAKY_ONLY, {"y": y})
    error, bound = _gaussian_check(result.samples["a"], mean, variance)
    assert error < bound
    # The swapped model (mean 0.1 a for a > 0) has posterior mean
    # 1.6 / 1.08, which the bound rejects.
    assert abs(float(result.samples["a"].to(_F64).mean()) - 1.6 / 1.08) > bound


# ---------------------------------------------------------------------------
# Callees whose only requests are the program's Random and Score
# ---------------------------------------------------------------------------

NOISY = """
object Obs : FinSet 4

instance random : Random
instance score : Score

define noisy(x : Real) : Real !{random, score} =
    let z <- perform random.sample[Real](site("noise"), Normal(x, 0.5))
    perform score.add(log_prob(Gamma(2.0, 1.0), z * z))
    perform score.add(weight(-0.25 * x * x))
    return z

define twice_noisy(x : Real) : Real !{random, score} =
    let first <- noisy(x)
    let second <- noisy(first)
    return second

program drawn : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let c <- noisy(a)
    let d <- twice_noisy(c)
    observe y : Obs <- Normal(loc=d, scale=1.0)
    return y

export drawn
"""


def _noisy_term(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """What one call of ``noisy`` adds to the joint."""
    return (
        _normal(z, x, 0.5)
        + torch.distributions.Gamma(
            torch.tensor(2.0, dtype=_F64), torch.tensor(1.0, dtype=_F64)
        ).log_prob(z * z)
        - 0.25 * x * x
    )


def _noisy_oracle(sites: dict[str, torch.Tensor], drop_score: bool) -> torch.Tensor:
    a, c, first, second = (
        sites[key] for key in ("a", "c$noise", "d$first$noise", "d$second$noise")
    )
    total = _normal(a, torch.tensor(0.0, dtype=_F64), 1.0)
    for x, z in ((a, c), (c, first), (first, second)):
        total = total + (_normal(z, x, 0.5) if drop_score else _noisy_term(x, z))
    y = torch.tensor(_Y4, dtype=_F64)
    return total + _normal(y, second, 1.0)


def test_an_effectful_call_runs_in_place_with_prefixed_sites() -> None:
    program = loads(NOISY).morphism
    assert isinstance(program, MonadicProgram)
    recorded = trace(program, torch.zeros(1, 1), {"y": torch.tensor(_Y4)})
    assert {"c$noise", "d$first$noise", "d$second$noise"} <= set(recorded.sites)
    assert float(recorded.sites["c"].value.reshape(())) == pytest.approx(
        float(recorded.sites["c$noise"].value.reshape(()))
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_an_effectful_call_agrees_with_the_oracle_on_both_paths(seed: int) -> None:
    a, c, first, second = (0.5 * v + 1.0 for v in _draws(seed, 4))
    sites = {"a": a, "c$noise": c, "d$first$noise": first, "d$second$noise": second}
    oracle = float(
        _noisy_oracle({k: torch.tensor(v, dtype=_F64) for k, v in sites.items()}, False)
    )
    # The reference machine names the callee's site by its label, once
    # per occurrence in the run.
    reference = _reference(
        NOISY,
        "drawn",
        {"a": a, "noise@0": c, "noise@1": first, "noise@2": second},
        {"y": _Y4},
    )
    compiled = _compiled(
        NOISY,
        {**{k: torch.tensor([v]) for k, v in sites.items()}, "y": torch.tensor(_Y4)},
    )
    assert reference == pytest.approx(oracle, rel=1e-9)
    assert compiled == pytest.approx(oracle, rel=1e-5)
    dropped = float(
        _noisy_oracle({k: torch.tensor(v, dtype=_F64) for k, v in sites.items()}, True)
    )
    assert compiled != pytest.approx(dropped, rel=1e-3)


def test_an_effectful_call_has_the_oracles_gradient() -> None:
    program = loads(NOISY).morphism
    assert isinstance(program, MonadicProgram)
    names = ("a", "c$noise", "d$first$noise", "d$second$noise")
    start = (0.3, 0.9, 1.2, 0.7)
    leaves = [torch.tensor([v], requires_grad=True) for v in start]
    total = program.log_joint(
        torch.zeros(1, 1),
        {**dict(zip(names, leaves, strict=True)), "y": torch.tensor(_Y4)},
    ).sum()
    grads = torch.autograd.grad(total, leaves)
    leaves64 = [torch.tensor(v, dtype=_F64, requires_grad=True) for v in start]
    expected = torch.autograd.grad(
        _noisy_oracle(dict(zip(names, leaves64, strict=True)), False), leaves64
    )
    for grad, value in zip(grads, expected, strict=True):
        assert float(grad) == pytest.approx(float(value), rel=1e-5, abs=1e-5)


JITTER = """
object Row : FinSet 8

instance random : Random

define jitter(x : Real) : Real !{random} =
    let z <- perform random.sample[Real](site("z"), Normal(x, 0.5))
    return z

program jittered : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let c <- jitter(a)
    observe y : Row <- Normal(loc=c, scale=1.0)
    return y

export jittered
"""


def test_nuts_recovers_the_posterior_through_an_effectful_call() -> None:
    """a ~ N(0, 1), c ~ N(a, 0.25), y_i ~ N(c, 1): marginally c ~ N(0, 1.25)
    and a | y is Gaussian, read off the joint of (a, c) given y, whose
    the joint precision of (a, c) is [[5, -4], [-4, 12]]."""
    y = torch.full((8,), 2.0)
    total = float(y.to(_F64).sum())
    # Joint Gaussian over (a, c): precision [[1 + 4, -4], [-4, 4 + 8]].
    precision = torch.tensor([[5.0, -4.0], [-4.0, 12.0]], dtype=_F64)
    covariance = torch.linalg.inv(precision)
    mean = covariance @ torch.tensor([0.0, total], dtype=_F64)
    result = _run_nuts(JITTER, {"y": y})
    error, bound = _gaussian_check(
        result.samples["a"], float(mean[0]), float(covariance[0, 0])
    )
    assert error < bound
    error_c, bound_c = _gaussian_check(
        result.samples["c$z"], float(mean[1]), float(covariance[1, 1])
    )
    assert error_c < bound_c
    # Without the call's draw (y_i ~ N(a, 1)) the posterior mean of a is
    # sum(y)/9, which the bound rejects.
    assert abs(float(result.samples["a"].to(_F64).mean()) - total / 9.0) > bound


# ---------------------------------------------------------------------------
# traverse over a lambda
# ---------------------------------------------------------------------------

TRAVERSE = """
object Obs : FinSet 4

instance random : Random
instance score : Score

define bump(x : Real) : Real !{} =
    return x + 1.0

define noisy(x : Real) : Real !{random, score} =
    let z <- perform random.sample[Real](site("noise"), Normal(x, 0.5))
    perform score.add(log_prob(Gamma(2.0, 1.0), z * z))
    return z

program traversed : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    sample theta : Obs <- Normal(loc=0.0, scale=1.0)
    let bumped <- traverse(theta, t -> bump(t))
    let drawn <- traverse(bumped, t -> noisy(t + a))
    observe y : Obs <- Normal(loc=drawn, scale=1.0)
    return y

export traversed
"""


def _traverse_oracle(
    a: torch.Tensor, theta: torch.Tensor, noise: torch.Tensor, shared: bool
) -> torch.Tensor:
    """The traversed program's joint; ``shared`` gives every item one draw."""
    zero = torch.tensor(0.0, dtype=_F64)
    drawn = noise[:1].expand(4) if shared else noise
    loc = theta + 1.0 + a
    gamma = torch.distributions.Gamma(
        torch.tensor(2.0, dtype=_F64), torch.tensor(1.0, dtype=_F64)
    )
    return (
        _normal(a, zero, 1.0)
        + _normal(theta, zero, 1.0)
        + _normal(drawn, loc, 0.5)
        + gamma.log_prob(drawn * drawn).sum()
        + _normal(torch.tensor(_Y4, dtype=_F64), drawn, 1.0)
    )


def test_a_drawing_traverse_is_one_plated_draw() -> None:
    program = loads(TRAVERSE).morphism
    assert isinstance(program, MonadicProgram)
    recorded = trace(program, torch.zeros(1, 1), {"y": torch.tensor(_Y4)})
    assert recorded.sites["drawn$noise"].value.reshape(-1).shape == (4,)
    assert torch.allclose(
        recorded.sites["bumped"].value.reshape(-1),
        recorded.sites["theta"].value.reshape(-1) + 1.0,
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_a_traverse_agrees_with_the_oracle_on_both_paths(seed: int) -> None:
    values = _draws(seed, 9)
    a, theta, noise = values[0], values[1:5], [0.5 * v + 1.0 for v in values[5:]]
    oracle = float(
        _traverse_oracle(
            torch.tensor(a, dtype=_F64),
            torch.tensor(theta, dtype=_F64),
            torch.tensor(noise, dtype=_F64),
            False,
        )
    )
    reference = _reference(
        TRAVERSE,
        "traversed",
        {
            "a": a,
            "theta": tuple(theta),
            **{f"noise@{k}": v for k, v in enumerate(noise)},
        },
        {"y": _Y4},
    )
    compiled = _compiled(
        TRAVERSE,
        {
            "a": torch.tensor([a]),
            "theta": torch.tensor(theta),
            "drawn$noise": torch.tensor(noise),
            "y": torch.tensor(_Y4),
        },
    )
    assert reference == pytest.approx(oracle, rel=1e-9)
    assert compiled == pytest.approx(oracle, rel=1e-5)
    shared = float(
        _traverse_oracle(
            torch.tensor(a, dtype=_F64),
            torch.tensor(theta, dtype=_F64),
            torch.tensor(noise, dtype=_F64),
            True,
        )
    )
    assert compiled != pytest.approx(shared, rel=1e-3)


def test_a_traverse_has_the_oracles_gradient() -> None:
    program = loads(TRAVERSE).morphism
    assert isinstance(program, MonadicProgram)
    a = torch.tensor([0.2], requires_grad=True)
    theta = torch.tensor([0.1, -0.4, 0.3, 0.8], requires_grad=True)
    noise = torch.tensor([1.1, 0.7, 1.4, 0.9], requires_grad=True)
    total = program.log_joint(
        torch.zeros(1, 1),
        {"a": a, "theta": theta, "drawn$noise": noise, "y": torch.tensor(_Y4)},
    ).sum()
    grads = torch.autograd.grad(total, (a, theta, noise))
    leaves = [
        t.detach().to(_F64).reshape(-1 if t.numel() > 1 else ()).requires_grad_()
        for t in (a, theta, noise)
    ]
    expected = torch.autograd.grad(_traverse_oracle(*leaves, False), leaves)
    for grad, value in zip(grads, expected, strict=True):
        assert torch.allclose(
            grad.reshape(-1).to(_F64), value.reshape(-1), rtol=1e-5, atol=1e-5
        )


TRAVERSE_LINE = """
object Group : FinSet 4

instance random : Random

define jitter(x : Real) : Real !{random} =
    let z <- perform random.sample[Real](site("z"), Normal(x, 0.5))
    return z

program grouped : Group -> Group
    sample a <- Normal(loc=0.0, scale=1.0)
    let base = factor k : Group in a
    let effects <- traverse(base, t -> jitter(t))
    observe y : Group <- Normal(loc=effects, scale=1.0)
    return y

export grouped
"""


def test_nuts_recovers_the_posterior_through_a_drawing_traverse() -> None:
    """Each group's effect is N(a, 0.25) and its datum N(effect, 1), so
    y_k ~ N(a, 1.25) independently and a | y ~ N(sum(y)/1.25 / p, 1/p)
    with p = 1 + 4/1.25."""
    y = torch.tensor([2.0, 3.0, 2.5, 3.5])
    p = 1.0 + 4.0 / 1.25
    mean = float(y.to(_F64).sum()) / 1.25 / p
    result = _run_nuts(TRAVERSE_LINE, {"y": y})
    error, bound = _gaussian_check(result.samples["a"], mean, 1.0 / p)
    assert error < bound
    # With every group sharing one effect c ~ N(a, 0.25), y ~ N(a 1, I +
    # 0.25 11'): the posterior precision is 1 + 4/2 = 3 and the mean
    # sum(y)/2/3, which the bound rejects.
    shared = float(y.to(_F64).sum()) / 6.0
    assert abs(float(result.samples["a"].to(_F64).mean()) - shared) > bound


# ---------------------------------------------------------------------------
# Refusals, on both paths
# ---------------------------------------------------------------------------

_REFUSED_HEAD = """
object Obs : FinSet 4

instance random : Random
instance score : Score

effect Echo
    echo : Real -> Real

handler identity_echo for Echo : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    echo(value : Real) resumes 1 =>
        resume(value)

define echoed(value : Real) : Real !{} =
    with instance local : Echo in
        handle local with identity_echo in
            let result <- perform local.echo(value)
            return result

define triangle(n : Int) : Int !{} =
    if n <= 0 then
        return 0
    else
        let rest <- triangle(n - 1)
        return n + rest

define branchy(x : Real) : Real !{random} =
    if x > 0.0 then
        let z <- perform random.sample[Real](site("z"), Normal(x, 1.0))
        return z
    else
        return x
"""


def _refused(call: str) -> str:
    return (
        _REFUSED_HEAD
        + f"""
program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let k <- {call}
    let m = a + k
    observe y : Obs <- Normal(loc=m, scale=1.0)
    return y

export prog
"""
    )


_REASONS = {
    "triangle(3)": (
        "reaches a recursive call (`triangle` -> `triangle`)",
        "write the repetition without recursion",
        "`scan` over a sequence",
    ),
    "echoed(a)": (
        "handles the effect `Echo` inside `echoed`",
        "not a handler and its resumptions",
    ),
    "branchy(a)": (
        "runs an `if` in `branchy` (line 30) whose branches draw or score",
        "the set of sites the program has would depend on the branch taken",
        "choose between values with `select`",
    ),
}


@pytest.mark.parametrize("call", sorted(_REASONS))
def test_the_compiled_runtime_refuses_at_the_call_saying_why(call: str) -> None:
    source = _refused(call)
    program = loads(source).morphism
    assert isinstance(program, CheckedProgram)
    line = source.splitlines().index(f"    let k <- {call}") + 1
    for fit in (
        lambda: trace(program, torch.zeros(1, 1), {"y": torch.zeros(4)}),
        lambda: MCMC(NUTSKernel(), num_warmup=2, num_samples=2, seed=0).run(
            program, torch.zeros(1, 1), {"y": torch.zeros(4)}
        ),
    ):
        with pytest.raises(CompileError) as raised:
            fit()
        message = str(raised.value)
        assert message.startswith(f"line {line}, col 4: program 'prog' cannot be fit")
        assert f"`let k <- {call.split('(')[0]}(...)`" in message
        for fragment in _REASONS[call]:
            assert fragment in message


@pytest.mark.parametrize(
    ("call", "a", "k"),
    [("triangle(3)", 0.3, 6.0), ("echoed(a)", 0.3, 0.3), ("branchy(a)", -0.4, -0.4)],
)
def test_the_reference_machine_still_runs_a_refused_program(
    call: str, a: float, k: float
) -> None:
    """``branchy`` at a < 0 takes the branch that draws nothing."""
    run = loads(_refused(call)).run(
        "prog", data={"y": (0.0,) * 4}, sites={"a": a}, seed=0
    )
    zero = torch.tensor(0.0, dtype=_F64)
    expected = _normal(torch.tensor(a, dtype=_F64), zero, 1.0) + _normal(
        torch.zeros(4, dtype=_F64), torch.tensor(a + k, dtype=_F64), 1.0
    )
    assert float(run.log_joint) == pytest.approx(float(expected), rel=1e-9)


RECURSIVE_TEMPLATE = """
define triangle(n : Int) : Int !{} =
    if n <= 0 then
        return 0
    else
        let rest <- triangle(n - 1)
        return n + rest

object Row : FinSet 4

program part (tau : Real) : Row -> Real 1
    sample effect <- Normal(loc=0.0, scale=tau)
    let k <- triangle(3)
    let lifted = effect + k
    return lifted

program caller : Row -> Row
    sample a <- part(0.7)
    observe y : Row <- Normal(loc=a, scale=1.0)
    return y

export caller
"""


def test_a_template_making_a_refused_call_is_refused_at_the_call() -> None:
    with pytest.raises(CompileError) as raised:
        loads(RECURSIVE_TEMPLATE)
    message = str(raised.value)
    line = RECURSIVE_TEMPLATE.splitlines().index("    let k <- triangle(3)") + 1
    assert message.startswith(f"line {line}, col 4: program 'part' cannot be fit")
    assert "reaches a recursive call (`triangle` -> `triangle`)" in message


SCALAR_PARAMETER = """
define standardize(x : Real, location : Real, scale : Real) : Real !{} =
    return (x - location) / scale

object Row : FinSet 8

program standardized : Row -> Row
    sample location <- Normal(loc=0.0, scale=1.0)
    sample scale <- HalfNormal(scale=1.0)
    let z <- standardize(x, location, scale)
    observe y : Row <- Normal(loc=z, scale=1.0)
    return y

export standardized
"""

_REMEDY = (
    "a call does not map a computation over a tensor's entries: declare the "
    "parameter as a tensor and `map` over it inside the computation"
)


def test_a_column_passed_to_a_scalar_parameter_is_refused_on_the_compiled_runtime() -> (
    None
):
    program = loads(SCALAR_PARAMETER).morphism
    assert isinstance(program, MonadicProgram)
    line = (
        SCALAR_PARAMETER.splitlines().index(
            "    let z <- standardize(x, location, scale)"
        )
        + 1
    )
    with pytest.raises(ValueError) as raised:
        trace(
            program,
            torch.zeros(1, 1),
            {"y": torch.zeros(8), "x": torch.randn(8)},
        )
    message = str(raised.value)
    assert f"line {line}, col 13: `let z <- standardize(...)` passes 'x'" in message
    assert "`x : Real` of `standardize`" in message
    assert _REMEDY in message


def test_a_column_passed_to_a_scalar_parameter_is_refused_on_the_reference_machine() -> (
    None
):
    line = (
        SCALAR_PARAMETER.splitlines().index(
            "    let z <- standardize(x, location, scale)"
        )
        + 1
    )
    with pytest.raises(ExecutionFailure) as raised:
        loads(SCALAR_PARAMETER).run(
            "standardized",
            data={"y": (0.0,) * 8, "x": (0.5,) * 8},
            sites={"location": 0.1, "scale": 1.0},
        )
    message = str(raised.value)
    assert f"line {line}, col 13: `let z <- standardize(...)` passes 'x'" in message
    assert _REMEDY in message


def test_a_scalar_column_still_runs_on_both_paths() -> None:
    """The refusal is about shape only: a scalar datum is a valid call."""
    compiled = _compiled(
        SCALAR_PARAMETER,
        {
            "location": torch.tensor([0.1]),
            "scale": torch.tensor([2.0]),
            "x": torch.tensor(0.5),
            "y": torch.zeros(8),
        },
    )
    reference = _reference(
        SCALAR_PARAMETER,
        "standardized",
        {"location": 0.1, "scale": 2.0},
        {"y": (0.0,) * 8, "x": 0.5},
    )
    assert compiled == pytest.approx(reference, rel=1e-5)


def test_a_plated_latent_passed_to_a_scalar_parameter_is_refused_when_checked() -> None:
    source = SCALAR_PARAMETER.replace(
        "    let z <- standardize(x, location, scale)",
        "    sample x : Row <- Normal(loc=0.0, scale=1.0)\n"
        "    let z <- standardize(x, location, scale)",
    )
    with pytest.raises(Exception) as raised:
        loads(source)
    message = str(raised.value)
    line = source.splitlines().index("    let z <- standardize(x, location, scale)") + 1
    assert f":{line}:13:" in message
    assert "expected Real, got Tensor[Real]([8])" in message
    assert (
        "declare the parameter as Tensor[Real]([8]) and `map` over it inside the "
        "computation" in message
    )
