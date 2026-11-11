"""Programs and templates that call pure computations are fitted.

A call of a pure computation (no effect, no static arguments, a chain
of pure bindings ending in a return) is inlined into let steps by the
analysis the transpile planner uses, so the program compiles to a
`MonadicProgram` with gradients. Each test compares the compiled
runtime with an independent oracle: the reference machine's log joint
at the same site values, finite differences, or a closed-form
posterior.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.dsl import loads
from quivers.dsl.compiler._prelude import CompileError
from quivers.inference import MCMC, NUTSKernel
from quivers.diagnostics import ess as effective_sample_size
from quivers.inference.trace import trace

AFFINE = """
define affine(a : Real, b : Real, t : Real) : Real !{} =
    let scaled = b * t
    let shifted = scaled + a
    return shifted

object Row : FinSet 8

program reg : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    sample t <- Normal(loc=0.0, scale=1.0)
    let mu <- affine(a, b, t * 2.0)
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return y

export reg
"""

_Y = torch.linspace(-1.0, 1.5, 8)


def _compiled_log_joint(
    source: str, sites: dict[str, float], data: dict[str, torch.Tensor]
) -> float:
    program = loads(source).morphism
    assert isinstance(program, MonadicProgram)
    values = {name: torch.tensor([value]) for name, value in sites.items()}
    return float(program.log_joint(torch.zeros(1, 1), {**values, **data}).sum())


def _reference_log_joint(
    source: str,
    entry: str,
    sites: dict[str, float],
    data: dict[str, tuple[float, ...] | int | float],
) -> float:
    run = loads(source).run(entry, data=data, sites=sites, seed=0)
    return float(run.log_joint)


def test_a_pure_call_compiles_to_a_monadic_program_with_its_bindings() -> None:
    program = loads(AFFINE).morphism
    assert isinstance(program, MonadicProgram)
    # The non-name argument is bound first, then the callee's bindings,
    # then the result, before the observation.
    names = [getattr(step, "name", None) for step in program.steps]
    assert names[-5:-1] == ["mu_t", "scaled", "shifted", "mu"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_compiled_log_joint_is_the_reference_machines(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    sites = {name: float(torch.randn((), generator=generator)) for name in "abt"}
    compiled = _compiled_log_joint(AFFINE, sites, {"y": _Y})
    reference = _reference_log_joint(AFFINE, "reg", sites, {"y": tuple(_Y.tolist())})
    assert compiled == pytest.approx(reference, rel=1e-6, abs=1e-5)


def test_the_compiled_log_joint_is_differentiable() -> None:
    program = loads(AFFINE).morphism
    assert isinstance(program, MonadicProgram)
    a, b, t = (torch.tensor([v], requires_grad=True) for v in (0.3, -0.4, 0.8))
    log_joint = program.log_joint(
        torch.zeros(1, 1), {"a": a, "b": b, "t": t, "y": _Y}
    ).sum()
    grads = torch.autograd.grad(log_joint, (a, b, t))
    # d/da of sum N(y; a + 2 b t, 1) + N(a; 0, 1), and likewise for b, t.
    residual = float((_Y - (a + 2.0 * b * t)).sum())
    expected = (
        residual - float(a),
        2.0 * float(t) * residual - float(b),
        2.0 * float(b) * residual - float(t),
    )
    for grad, value in zip(grads, expected, strict=True):
        assert float(grad) == pytest.approx(value, rel=1e-5, abs=1e-5)


LINE = """
define line(a : Real, b : Real, t : Tensor[Real]([8])) : Tensor[Real]([8]) !{} =
    let at = v -> a + b * v
    return map(t, at)

object Row : FinSet 8

program regression : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu <- line(a, b, t)
    observe y : Row <- Normal(loc=mu, scale=0.5)
    return y

export regression
"""


def test_nuts_on_a_program_calling_a_computation_recovers_the_posterior() -> None:
    """Bayesian linear regression with known noise: the posterior of
    ``(a, b)`` is Gaussian in closed form."""
    t = torch.linspace(-1.0, 1.0, 8)
    y = 0.5 + 1.5 * t + 0.5 * torch.tensor([0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.0])
    design = torch.stack([torch.ones(8), t], dim=1).to(torch.float64)
    precision = torch.eye(2, dtype=torch.float64) + design.t() @ design / 0.25
    covariance = torch.linalg.inv(precision)
    mean = covariance @ design.t() @ y.to(torch.float64) / 0.25

    program = loads(LINE).morphism
    assert isinstance(program, MonadicProgram)
    result = MCMC(
        NUTSKernel(max_tree_depth=5),
        num_warmup=300,
        num_samples=500,
        num_chains=2,
        seed=0,
    ).run(program, torch.zeros(1, 1), {"y": y, "t": t})
    for index, name in enumerate("ab"):
        draws = result.samples[name].to(torch.float64)
        ess = float(effective_sample_size(draws, method="mean"))
        sd = math.sqrt(float(covariance[index, index]))
        assert abs(float(draws.mean()) - float(mean[index])) < 5.0 * sd / math.sqrt(ess)


TEMPLATE = """
define standardize(x : Real, location : Real, scale : Real) : Real !{} =
    let centred = x - location
    return centred / scale

object Row : FinSet 8

program shifted (prior_scale : Real) : Row -> Real 1
    sample location <- Normal(loc=0.0, scale=prior_scale)
    let z <- standardize(1.0, location, 2.0)
    return z

program outer : Row -> Row
    sample v <- shifted(3.0)
    sample w <- shifted(0.5)
    let both = v + w
    observe y : Row <- Normal(loc=both, scale=1.0)
    return y

export outer
"""


def test_a_template_calling_a_pure_computation_instantiates_twice() -> None:
    program = loads(TEMPLATE).morphism
    assert isinstance(program, MonadicProgram)
    recorded = trace(program, torch.zeros(1, 1), observations={"y": _Y})
    location_v = float(recorded.sites["v$location"].value.reshape(()))
    location_w = float(recorded.sites["w$location"].value.reshape(()))
    v = (1.0 - location_v) / 2.0
    w = (1.0 - location_w) / 2.0
    assert float(recorded.sites["v"].value.reshape(())) == pytest.approx(v, rel=1e-6)
    assert float(recorded.sites["w"].value.reshape(())) == pytest.approx(w, rel=1e-6)
    expected = (
        float(torch.distributions.Normal(0.0, 3.0).log_prob(torch.tensor(location_v)))
        + float(torch.distributions.Normal(0.0, 0.5).log_prob(torch.tensor(location_w)))
        + float(torch.distributions.Normal(v + w, 1.0).log_prob(_Y).sum())
    )
    assert float(recorded.log_joint.sum()) == pytest.approx(expected, rel=1e-5)


NESTED = """
define bump(x : Real) : Real !{} =
    return x + 1.0

define twice_bumped(x : Real) : Real !{} =
    let once <- bump(x)
    return once + 1.0

object Row : FinSet 4

program part (tau : Real) : Row -> Real 1
    sample effect <- Normal(loc=0.0, scale=tau)
    let lifted <- twice_bumped(effect)
    return lifted

program caller : Row -> Row
    sample a <- part(0.7)
    observe y : Row <- Normal(loc=a, scale=1.0)
    return y

export caller
"""


def test_a_call_that_cannot_be_inlined_stays_a_located_refusal_in_a_template() -> None:
    with pytest.raises(CompileError) as raised:
        loads(NESTED)
    message = str(raised.value)
    assert "program 'part' runs `let lifted <- twice_bumped(...)`" in message
    line = NESTED.splitlines().index("    let lifted <- twice_bumped(effect)")
    assert message.startswith(f"line {line + 1}, col 4:")


HALF = """
define half(n : Int) : Int !{} =
    return n / 2

object Row : FinSet 8

program halving : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let k <- half(n0)
    let shift = a + k
    observe y : Row <- Normal(loc=shift, scale=1.0)
    return y

export halving
"""


@pytest.mark.parametrize("n0", [-7, 7, -1, 0])
def test_integer_division_in_a_callee_truncates_as_the_reference_machine(
    n0: int,
) -> None:
    program = loads(HALF).morphism
    assert isinstance(program, MonadicProgram)
    recorded = trace(
        program,
        torch.zeros(1, 1),
        observations={"y": _Y, "n0": torch.tensor(n0), "a": torch.tensor([0.25])},
    )
    assert int(recorded.sites["k"].value) == int(n0 / 2)
    reference = _reference_log_joint(
        HALF, "halving", {"a": 0.25}, {"y": tuple(_Y.tolist()), "n0": n0}
    )
    assert float(recorded.log_joint.sum()) == pytest.approx(reference, rel=1e-6)


SHADOW = """
define scaled_sum(x : Real, s : Real) : Real !{} =
    let w = x * s
    return w + 1.0

object Row : FinSet 8

program shadowing : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- scaled_sum(a, 2.0)
    let shift = m + w
    observe y : Row <- Normal(loc=shift, scale=1.0)
    return y

export shadowing
"""


def test_an_inlined_binding_does_not_capture_a_host_datum() -> None:
    """The callee's binding ``w`` is renamed, so the program's own read
    of the host datum ``w`` still reads the datum."""
    program = loads(SHADOW).morphism
    assert isinstance(program, MonadicProgram)
    recorded = trace(
        program,
        torch.zeros(1, 1),
        observations={"y": _Y, "w": torch.full((8,), 10.0), "a": torch.tensor([0.5])},
    )
    assert float(recorded.sites["m"].value.reshape(())) == pytest.approx(2.0)
    assert torch.allclose(recorded.sites["shift"].value, torch.full((8,), 12.0))
