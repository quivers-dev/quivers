"""A scan cell reads the scanning program's tensor bindings and matrix data.

A program cell reads a name it does not bind as the scanning program's
binding of that name before the scan, at the binding's own type: a scalar,
a plated vector, or a matrix-valued draw; a name the scanning program does
not bind is host data of the scanning program, a matrix included. Each case
is checked in sample position (the cell draws its next state from a mean
that reads the name) and in observe position (the cell observes through a
deterministic state that reads it).

The oracle is a hand-written torch loop over the positions. The compiled
runtime runs in float32, so it is compared with the float64 oracle at
``rel=1e-5`` (float32 rounding over a dozen terms is about 1e-6 relative);
the reference machine runs in float64 and is compared at ``rel=1e-9``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.inference import (
    ELBO,
    MCMC,
    SVI,
    AutoMultivariateNormalGuide,
    LatentRegistry,
    NUTSKernel,
    trace,
)

T = 3

HEADER = """
object In : Real 2
object Hid : FinSet 2
object State : Real 2
"""

MEANS = {
    "scalar": (
        "w * h_prev + x_t",
        "    sample w <- Normal(loc=0.0, scale=0.5)\n",
    ),
    "vector": (
        "b * h_prev + x_t",
        "    sample b : Hid <- Normal(loc=0.0, scale=0.5)\n",
    ),
    "matrix": (
        "sum(factor i : Hid, j : Hid in W[i, j] * h_prev[j]) + x_t",
        "    sample W : Hid <- Normal(loc=0.0, scale=0.5) [over=Hid, iid_over=Hid]\n",
    ),
    "host matrix": (
        "sum(factor i : Hid, j : Hid in M[i, j] * h_prev[j]) + x_t",
        "",
    ),
}

CELLS = {
    "sample": """
    sample s <- Normal(loc=m, scale=0.5)
    observe o <- Normal(loc=s, scale=0.3)
    return s
""",
    "observe": """
    let s = tanh(m)
    observe o <- Normal(loc=s, scale=0.3)
    return s
""",
}


def _source(binding: str, position: str) -> str:
    mean, declaration = MEANS[binding]
    return (
        HEADER
        + "\nprogram cell(x_t, h_prev) : In * State -> State\n"
        + f"    let m = {mean}"
        + CELLS[position]
        + "\ndefine run_all = scan(cell)\n\n"
        + "program model : In -> State\n"
        + declaration
        + "    sample h <- run_all\n    return h\n\nexport model\n"
    )


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, 2, generator=generator)
    o = torch.randn(1, T, 2, generator=generator)
    matrix = 0.5 * torch.randn(2, 2, generator=generator)
    return x, o, matrix


def _given(
    binding: str, o: torch.Tensor, matrix: torch.Tensor
) -> dict[str, torch.Tensor]:
    given = {f"o@{t}": o[:, t] for t in range(T)}
    if binding == "host matrix":
        given["M"] = matrix
    return given


def _oracle(
    binding: str,
    position: str,
    x: torch.Tensor,
    o: torch.Tensor,
    value: torch.Tensor,
    states: list[torch.Tensor],
) -> torch.Tensor:
    """The log joint by a torch loop over the positions, in float64."""
    total = torch.zeros((), dtype=torch.float64)
    if binding != "host matrix":
        total = total + D.Normal(0.0, 0.5).log_prob(value).sum()
    h = torch.zeros(2, dtype=torch.float64)
    for t in range(T):
        if binding in ("scalar", "vector"):
            mean = value * h + x[0, t]
        else:
            mean = value @ h + x[0, t]
        if position == "sample":
            s = states[t]
            total = total + D.Normal(mean, 0.5).log_prob(s).sum()
        else:
            s = torch.tanh(mean)
        total = total + D.Normal(s, 0.3).log_prob(o[0, t]).sum()
        h = s
    return total


def _name(binding: str) -> str:
    return {"scalar": "w", "vector": "b", "matrix": "W", "host matrix": "M"}[binding]


CASES = [(binding, position) for binding in MEANS for position in CELLS]


@pytest.mark.parametrize(("binding", "position"), CASES)
def test_compiled_runtime_matches_the_torch_oracle(binding: str, position: str) -> None:
    program = loads(_source(binding, position))
    x, o, matrix = _inputs(0)
    torch.manual_seed(1)
    recorded = trace(program.morphism, x, observations=_given(binding, o, matrix))
    name = _name(binding)
    value = matrix if binding == "host matrix" else recorded.sites[name].value
    shape = {"scalar": (), "vector": (2,), "matrix": (2, 2), "host matrix": (2, 2)}[
        binding
    ]
    states = (
        [recorded.sites[f"s@{t}"].value.reshape(2).double() for t in range(T)]
        if position == "sample"
        else []
    )
    expected = _oracle(
        binding, position, x.double(), o.double(), value.reshape(shape).double(), states
    )
    assert float(recorded.log_joint) == pytest.approx(float(expected), rel=1e-5)


def _host(value: list) -> tuple:
    return tuple(_host(v) if isinstance(v, list) else v for v in value)


@pytest.mark.parametrize(("binding", "position"), CASES)
def test_reference_machine_matches_the_torch_oracle(
    binding: str, position: str
) -> None:
    program = loads(_source(binding, position))
    x, o, matrix = _inputs(2)
    generator = torch.Generator().manual_seed(3)
    shape = {"scalar": (), "vector": (2,), "matrix": (2, 2), "host matrix": (2, 2)}[
        binding
    ]
    value = (
        matrix.double()
        if binding == "host matrix"
        else 0.5 * torch.randn(shape, generator=generator, dtype=torch.float64)
    )
    states = [
        torch.randn(2, generator=generator, dtype=torch.float64) for _ in range(T)
    ]
    data: dict[str, object] = {
        "in": _host(x[0].double().tolist()),
        "o": _host(o[0].double().tolist()),
    }
    sites: dict[str, object] = {}
    if binding == "host matrix":
        data["M"] = _host(value.tolist())
    elif binding == "scalar":
        sites["w"] = float(value)
    else:
        sites[_name(binding)] = _host(value.tolist())
    if position == "sample":
        sites.update({f"s@{t}": _host(states[t].tolist()) for t in range(T)})
    run = program.run("model", data=data, sites=sites, seed=0)
    expected = _oracle(binding, position, x.double(), o.double(), value, states)
    assert run.log_joint == pytest.approx(float(expected), rel=1e-9)


@pytest.mark.parametrize("binding", ["vector", "matrix"])
def test_gradients_reach_the_binding_through_every_position(binding: str) -> None:
    program = loads(_source(binding, "observe"))
    x, o, _ = _inputs(4)
    generator = torch.Generator().manual_seed(5)
    shape = (2,) if binding == "vector" else (2, 2)
    value = (0.5 * torch.randn(shape, generator=generator)).requires_grad_(True)
    name = _name(binding)
    joint = program.morphism.log_joint(x, {name: value, **_given(binding, o, value)})
    joint.sum().backward()
    reference = value.detach().double().requires_grad_(True)
    _oracle(binding, "observe", x.double(), o.double(), reference, []).backward()
    assert value.grad is not None and reference.grad is not None
    torch.testing.assert_close(
        value.grad.double(), reference.grad, rtol=1e-4, atol=1e-6
    )


def test_a_cell_reading_an_unbound_vector_name_keeps_the_host_data_rule() -> None:
    # Without a binding before the scan, the name is host data of the
    # scanning program, typed from the cell's use (a scalar here).
    source = _source("vector", "observe").replace(
        "    sample b : Hid <- Normal(loc=0.0, scale=0.5)\n", ""
    )
    program = loads(source)
    entry = next(item for item in program.entry_points() if item.name == "model")
    assert ("b", "Real") in [(p.name, str(p.type)) for p in entry.parameters]


# Inference over the per-position sites, against the exact posterior of a
# linear-Gaussian model: b ~ N(0, I_2); s_t ~ N(0.5 s_{t-1} + b . x_t, 1)
# with s_{-1} = 0; o_t ~ N(s_t, 0.5^2).

GAUSSIAN = """
object In : Real 2
object Hid : FinSet 2
object St : Real 1

program cell(x_t, h_prev) : In * St -> St
    let m = 0.5 * h_prev + sum(b * x_t)
    sample s <- Normal(loc=m, scale=1.0)
    observe o <- Normal(loc=s, scale=0.5)
    return s

define run_all = scan(cell)

program model : In -> St
    sample b : Hid <- Normal(loc=0.0, scale=1.0)
    sample h <- run_all
    return h

export model
"""
STEPS = 4


def _gaussian_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1, STEPS, 2, generator=generator)
    o = 2.0 * torch.randn(1, STEPS, 1, generator=generator)
    return x, o


def _exact_posterior(
    x: torch.Tensor, o: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviations of (b_0, b_1, s_0, ..., s_3) given o."""
    size = 2 + STEPS
    rows: list[torch.Tensor] = []
    targets: list[float] = []
    precisions: list[float] = []

    def residual(
        coefficients: dict[int, float], target: float, variance: float
    ) -> None:
        row = torch.zeros(size, dtype=torch.float64)
        for index, coefficient in coefficients.items():
            row[index] = coefficient
        rows.append(row)
        targets.append(target)
        precisions.append(1.0 / variance)

    residual({0: 1.0}, 0.0, 1.0)
    residual({1: 1.0}, 0.0, 1.0)
    for t in range(STEPS):
        coefficients = {2 + t: 1.0, 0: -float(x[0, t, 0]), 1: -float(x[0, t, 1])}
        if t > 0:
            coefficients[2 + t - 1] = -0.5
        residual(coefficients, 0.0, 1.0)
        residual({2 + t: 1.0}, float(o[0, t, 0]), 0.25)
    design = torch.stack(rows)
    weights = torch.diag(torch.tensor(precisions, dtype=torch.float64))
    precision = design.T @ weights @ design
    covariance = torch.linalg.inv(precision)
    mean = covariance @ (
        design.T @ weights @ torch.tensor(targets, dtype=torch.float64)
    )
    return mean, covariance.diagonal().sqrt()


def test_the_latent_registry_holds_every_position() -> None:
    x, o = _gaussian_data()
    model = loads(GAUSSIAN).morphism
    observations = {f"o@{t}": o[:, t] for t in range(STEPS)}
    registry = LatentRegistry.from_trace(
        model, trace(model, x, observations=observations)
    )
    assert registry.names == ("b", "s@0", "s@1", "s@2", "s@3")


@pytest.mark.slow
def test_nuts_recovers_the_exact_posterior_of_the_per_position_sites() -> None:
    # Tolerances fixed before running: with an effective sample size of
    # at least 200 (checked below) the Monte Carlo sd of a posterior mean
    # is at most sd / sqrt(200) = 0.071 sd, so 4.5 Monte Carlo sds is
    # 0.32 sd; a posterior sd estimate has relative sd about
    # 1 / sqrt(2 * 200) = 0.05, so 4.5 of those is 0.225, rounded to 0.25.
    x, o = _gaussian_data()
    model = loads(GAUSSIAN).morphism
    observations = {f"o@{t}": o[:, t] for t in range(STEPS)}
    result = MCMC(
        NUTSKernel(), num_warmup=300, num_samples=500, num_chains=2, seed=0
    ).run(model, x, observations)
    mean, sd = _exact_posterior(x, o)
    draws = torch.cat(
        [result.samples["b"].reshape(-1, 2)]
        + [result.samples[f"s@{t}"].reshape(-1, 1) for t in range(STEPS)],
        dim=1,
    ).double()
    ess = torch.cat(
        [result.ess["b"].reshape(-1)]
        + [result.ess[f"s@{t}"].reshape(-1) for t in range(STEPS)]
    )
    assert float(ess.min()) >= 200
    assert torch.all((draws.mean(0) - mean).abs() <= 4.5 * sd / math.sqrt(200))
    assert torch.all((draws.std(0) / sd - 1.0).abs() <= 0.25)


@pytest.mark.slow
def test_svi_recovers_the_exact_posterior_mean_of_the_per_position_sites() -> None:
    # A full-rank Gaussian guide contains the exact posterior. Averaging
    # the guide's mean over the last 1000 of 3000 Adam steps (lr 0.02)
    # leaves stochastic-optimization noise of a few hundredths of a
    # posterior sd; the tolerance, fixed before running, is 0.2 sd.
    x, o = _gaussian_data()
    model = loads(GAUSSIAN).morphism
    observations = {f"o@{t}": o[:, t] for t in range(STEPS)}
    torch.manual_seed(0)
    guide = AutoMultivariateNormalGuide(model, {"o"}, x=x, observations=observations)
    assert guide.registry.names == ("b", "s@0", "s@1", "s@2", "s@3")
    svi = SVI(model, guide, torch.optim.Adam(guide.parameters(), lr=0.02), ELBO())
    iterates: list[torch.Tensor] = []
    for step in range(3000):
        svi.step(x, observations)
        if step >= 2000:
            iterates.append(guide.loc.detach().clone())
    estimate = torch.stack(iterates).mean(0).double()
    mean, sd = _exact_posterior(x, o)
    assert torch.all((estimate - mean).abs() <= 0.2 * sd)


def test_a_cell_reads_a_let_bound_vector_of_the_scanning_program() -> None:
    source = _source("vector", "observe").replace(
        "    sample b : Hid <- Normal(loc=0.0, scale=0.5)\n",
        "    sample a : Hid <- Normal(loc=0.0, scale=0.5)\n    let b = 2.0 * a\n",
    )
    program = loads(source)
    x, o, _ = _inputs(6)
    torch.manual_seed(7)
    recorded = trace(program.morphism, x, observations=_given("vector", o, o))
    a = recorded.sites["a"].value.reshape(2).double()
    expected = _oracle("vector", "observe", x.double(), o.double(), 2.0 * a, [])
    # The oracle's prior term is for b ~ N(0, 0.5); the program's is for a.
    expected = expected - D.Normal(0.0, 0.5).log_prob(2.0 * a).sum()
    expected = expected + D.Normal(0.0, 0.5).log_prob(a).sum()
    assert float(recorded.log_joint) == pytest.approx(float(expected), rel=1e-5)


def test_a_cell_that_indexes_a_scalar_binding_is_refused_where_it_reads_it() -> None:
    source = _source("matrix", "observe").replace(
        "    sample W : Hid <- Normal(loc=0.0, scale=0.5) [over=Hid, iid_over=Hid]\n",
        "    sample W <- Normal(loc=0.0, scale=0.5)\n",
    )
    with pytest.raises(QiecDiagnosticError, match="Real"):
        loads(source)
