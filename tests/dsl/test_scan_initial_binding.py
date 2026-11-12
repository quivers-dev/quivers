"""A scan starts from a value its scanning program binds: ``scan(cell, init=h0)``.

The oracle is a hand-written torch loop. The compiled runtime runs in
float32 and is compared with the float64 oracle at ``rel=1e-5``; the
reference machine runs in float64 and is compared at ``rel=1e-9``.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributions as D

from quivers.continuous.scan import ScanMorphism
from quivers.continuous.spaces import Euclidean, ProductSpace
from quivers.continuous.families import ConditionalNormal
from quivers.dsl import loads
from quivers.dsl.compiler import CompileError
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.inference import MCMC, NUTSKernel, trace

T = 5

VOLATILITY = """
object Val : Real 1

program sv_step(u_t, h_prev) : Val * Val -> Val
    let m = mu + phi * (h_prev - mu)
    sample h <- Normal(loc=m, scale=sigma_h)
    let vol = exp(0.5 * h)
    observe r <- Normal(loc=0.0, scale=vol)
    return h

define run_all = scan(sv_step, init=h0)

program stochastic_volatility : Val -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample phi <- Uniform(low=-1.0, high=1.0)
    sample sigma_h <- HalfNormal(scale=0.5)
    let s0 = sigma_h / sqrt(1.0 - phi * phi)
    sample h0 <- Normal(loc=mu, scale=s0)
    sample final <- run_all
    return final

export stochastic_volatility
"""


def _volatility_oracle(
    values: dict[str, torch.Tensor], r: torch.Tensor
) -> torch.Tensor:
    mu, phi, sigma, h0 = (values[name] for name in ("mu", "phi", "sigma_h", "h0"))
    total = (
        D.Normal(0.0, 1.0).log_prob(mu)
        + D.Uniform(-1.0, 1.0).log_prob(phi)
        + D.HalfNormal(0.5).log_prob(sigma)
        + D.Normal(mu, sigma / torch.sqrt(1.0 - phi**2)).log_prob(h0)
    )
    previous = h0
    for t in range(T):
        h = values[f"h@{t}"]
        total = total + D.Normal(mu + phi * (previous - mu), sigma).log_prob(h)
        total = total + D.Normal(0.0, torch.exp(0.5 * h)).log_prob(r[t])
        previous = h
    return total


def test_volatility_compiled_runtime_matches_the_oracle() -> None:
    program = loads(VOLATILITY)
    generator = torch.Generator().manual_seed(0)
    r = torch.randn(1, T, 1, generator=generator)
    torch.manual_seed(1)
    recorded = trace(
        program.morphism,
        torch.zeros(1, T, 1),
        observations={f"r@{t}": r[:, t] for t in range(T)},
    )
    names = ["mu", "phi", "sigma_h", "h0", *(f"h@{t}" for t in range(T))]
    values = {name: recorded.sites[name].value.reshape(()).double() for name in names}
    expected = _volatility_oracle(values, r.reshape(T).double())
    assert float(recorded.log_joint) == pytest.approx(float(expected), rel=1e-5)


def test_volatility_reference_machine_matches_the_oracle() -> None:
    program = loads(VOLATILITY)
    generator = torch.Generator().manual_seed(2)
    r = torch.randn(T, generator=generator, dtype=torch.float64)
    values = {
        "mu": torch.tensor(0.3, dtype=torch.float64),
        "phi": torch.tensor(0.8, dtype=torch.float64),
        "sigma_h": torch.tensor(0.4, dtype=torch.float64),
        "h0": torch.tensor(-0.2, dtype=torch.float64),
    }
    for t in range(T):
        values[f"h@{t}"] = 0.5 * torch.randn(
            (), generator=generator, dtype=torch.float64
        )
    sites: dict[str, object] = {name: float(value) for name, value in values.items()}
    for t in range(T):
        sites[f"h@{t}"] = (float(values[f"h@{t}"]),)
    run = program.run(
        "stochastic_volatility",
        data={
            "val": tuple((0.0,) for _ in range(T)),
            "r": tuple((float(v),) for v in r),
        },
        sites=sites,
        seed=0,
    )
    assert run.log_joint == pytest.approx(
        float(_volatility_oracle(values, r)), rel=1e-9
    )


VECTOR = """
object In : Real 2
object Hid : FinSet 2
object State : Real 2

program cell(x_t, h_prev) : In * State -> State
    let m = 0.5 * h_prev + x_t
    sample s <- Normal(loc=m, scale=0.5)
    observe o <- Normal(loc=s, scale=0.3)
    return s

define run_all = scan(cell, init=start)

program model : In -> State
    sample start : Hid <- Normal(loc=0.0, scale=1.0)
    sample h <- run_all
    return h

export model
"""


def _vector_oracle(
    start: torch.Tensor, states: list[torch.Tensor], x: torch.Tensor, o: torch.Tensor
) -> torch.Tensor:
    total = D.Normal(0.0, 1.0).log_prob(start).sum()
    previous = start
    for t in range(T):
        total = total + D.Normal(0.5 * previous + x[t], 0.5).log_prob(states[t]).sum()
        total = total + D.Normal(states[t], 0.3).log_prob(o[t]).sum()
        previous = states[t]
    return total


def test_vector_initial_state_compiled_runtime_matches_the_oracle() -> None:
    program = loads(VECTOR)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, T, 2, generator=generator)
    o = torch.randn(1, T, 2, generator=generator)
    torch.manual_seed(4)
    recorded = trace(
        program.morphism, x, observations={f"o@{t}": o[:, t] for t in range(T)}
    )
    start = recorded.sites["start"].value.reshape(2).double()
    states = [recorded.sites[f"s@{t}"].value.reshape(2).double() for t in range(T)]
    expected = _vector_oracle(start, states, x[0].double(), o[0].double())
    assert float(recorded.log_joint) == pytest.approx(float(expected), rel=1e-5)


def test_vector_initial_state_reference_machine_matches_the_oracle() -> None:
    program = loads(VECTOR)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(T, 2, generator=generator, dtype=torch.float64)
    o = torch.randn(T, 2, generator=generator, dtype=torch.float64)
    start = torch.randn(2, generator=generator, dtype=torch.float64)
    states = [
        torch.randn(2, generator=generator, dtype=torch.float64) for _ in range(T)
    ]
    sites: dict[str, object] = {"start": tuple(start.tolist())}
    sites.update({f"s@{t}": tuple(states[t].tolist()) for t in range(T)})
    run = program.run(
        "model",
        data={
            "in": tuple(tuple(row) for row in x.tolist()),
            "o": tuple(tuple(row) for row in o.tolist()),
        },
        sites=sites,
        seed=0,
    )
    assert run.log_joint == pytest.approx(
        float(_vector_oracle(start, states, x, o)), rel=1e-9
    )


def test_the_initial_state_differs_from_zero_in_the_joint() -> None:
    # The same model started from zero scores the first position under a
    # different mean; a scan that ignored `init` would agree with it.
    program = loads(VECTOR)
    zero = loads(VECTOR.replace("scan(cell, init=start)", "scan(cell)"))
    generator = torch.Generator().manual_seed(6)
    x = torch.randn(1, T, 2, generator=generator)
    o = torch.randn(1, T, 2, generator=generator)
    values = {"start": torch.tensor([2.0, -2.0])}
    values.update({f"s@{t}": torch.randn(1, 2, generator=generator) for t in range(T)})
    values.update({f"o@{t}": o[:, t] for t in range(T)})
    started = program.morphism.log_joint(x, values)
    from_zero = zero.morphism.log_joint(x, values)
    first = values["s@0"][0]
    shift = (
        D.Normal(0.5 * values["start"] + x[0, 0], 0.5).log_prob(first).sum()
        - D.Normal(x[0, 0], 0.5).log_prob(first).sum()
    )
    # Both programs draw `start`, so its prior cancels.
    assert float(started - from_zero) == pytest.approx(float(shift), rel=1e-5)


def test_gradients_reach_the_initial_state() -> None:
    program = loads(VECTOR)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, T, 2, generator=generator)
    o = torch.randn(1, T, 2, generator=generator)
    start = torch.tensor([0.4, -0.3], requires_grad=True)
    states = [torch.randn(1, 2, generator=generator) for _ in range(T)]
    values = {"start": start, **{f"s@{t}": states[t] for t in range(T)}}
    values.update({f"o@{t}": o[:, t] for t in range(T)})
    program.morphism.log_joint(x, values).sum().backward()
    reference = start.detach().double().requires_grad_(True)
    _vector_oracle(
        reference, [s[0].double() for s in states], x[0].double(), o[0].double()
    ).backward()
    assert start.grad is not None and reference.grad is not None
    torch.testing.assert_close(
        start.grad.double(), reference.grad, rtol=1e-4, atol=1e-6
    )


def test_an_initial_state_the_program_does_not_bind_is_refused() -> None:
    source = VECTOR.replace("init=start", "init=missing")
    with pytest.raises((CompileError, QiecDiagnosticError), match="missing"):
        loads(source)


def test_an_initial_state_of_another_type_is_refused() -> None:
    source = VECTOR.replace(
        "    sample start : Hid <- Normal(loc=0.0, scale=1.0)\n",
        "    sample start <- Normal(loc=0.0, scale=1.0)\n",
    )
    with pytest.raises((CompileError, QiecDiagnosticError), match="start"):
        loads(source)


def test_qvr_check_reports_an_unbound_initial_state_without_a_traceback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.qvr"
    path.write_text(VECTOR.replace("init=start", "init=missing"))
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from quivers.cli import main; "
            "sys.argv = ['qvr', 'check', sys.argv[1]]; sys.exit(main())",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = done.stdout + done.stderr
    assert done.returncode == 1
    assert "Traceback" not in output
    assert "missing" in output


def test_a_standalone_scan_from_a_binding_names_the_binding_when_run() -> None:
    hidden = Euclidean(name="h", dim=2)
    cell = ConditionalNormal(
        ProductSpace(components=(Euclidean(name="x", dim=2), hidden)), hidden
    )
    scanned = ScanMorphism(cell, init="start")
    with pytest.raises(ValueError, match="'start'"):
        scanned.rsample(torch.zeros(1, 3, 2))


# Posterior of the per-position sites and the initial state, against the
# exact posterior of a Gaussian AR(1): h0 ~ N(0, 1);
# s_t ~ N(0.8 s_{t-1}, 0.5^2) with s_{-1} = h0; o_t ~ N(s_t, 0.5^2).

AR = """
object Val : Real 1

program ar_step(u_t, s_prev) : Val * Val -> Val
    let m = 0.8 * s_prev
    sample s <- Normal(loc=m, scale=0.5)
    observe o <- Normal(loc=s, scale=0.5)
    return s

define run_all = scan(ar_step, init=h0)

program ar : Val -> Val
    sample h0 <- Normal(loc=0.0, scale=1.0)
    sample last <- run_all
    return last

export ar
"""
STEPS = 4


def _ar_posterior(o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    size = 1 + STEPS
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
    for t in range(STEPS):
        residual({1 + t: 1.0, t: -0.8}, 0.0, 0.25)
        residual({1 + t: 1.0}, float(o[t]), 0.25)
    design = torch.stack(rows)
    weights = torch.diag(torch.tensor(precisions, dtype=torch.float64))
    covariance = torch.linalg.inv(design.T @ weights @ design)
    mean = covariance @ (
        design.T @ weights @ torch.tensor(targets, dtype=torch.float64)
    )
    return mean, covariance.diagonal().sqrt()


@pytest.mark.slow
def test_nuts_recovers_the_initial_state_and_the_trajectory() -> None:
    # Tolerances fixed before running: with an effective sample size of at
    # least 200 (checked) a posterior mean's Monte Carlo sd is at most
    # 0.071 posterior sd, so 4.5 of them is 0.32 sd; a posterior sd
    # estimate's relative sd is about 0.05, so 4.5 of those, rounded, 0.25.
    generator = torch.Generator().manual_seed(8)
    o = 1.5 * torch.randn(1, STEPS, 1, generator=generator)
    model = loads(AR).morphism
    result = MCMC(
        NUTSKernel(), num_warmup=300, num_samples=500, num_chains=2, seed=0
    ).run(model, torch.zeros(1, STEPS, 1), {f"o@{t}": o[:, t] for t in range(STEPS)})
    names = ["h0", *(f"s@{t}" for t in range(STEPS))]
    draws = torch.stack(
        [result.samples[name].reshape(-1) for name in names], dim=1
    ).double()
    ess = torch.stack([result.ess[name].reshape(()) for name in names])
    mean, sd = _ar_posterior(o.reshape(STEPS).double())
    assert float(ess.min()) >= 200
    assert torch.all((draws.mean(0) - mean).abs() <= 4.5 * sd / math.sqrt(200))
    assert torch.all((draws.std(0) / sd - 1.0).abs() <= 0.25)
