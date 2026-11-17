"""brms ``cores``: chains in worker processes reproduce the serial run exactly.

Every test uses at most two worker processes, the machine's memory limit.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import bf, fit
from quivers.formulas._joint_options import FormulaFitOptions, JointFitOptions
from quivers.formulas._multivariate import MultivariateFormula
from quivers.inference import MCMC, MCMCResult, NUTSKernel
from quivers.inference.mcmc import MCMCCheckpoint
from quivers.inference.mcmc.parallel import (
    ChainStates,
    assemble_chains,
    parallel_chains,
    run_chains,
)
from quivers.modeling import MCMCMethod, Model

ROWS = np.arange(16)
FRAME = pd.DataFrame({"x": np.sin(1.7 * ROWS)})
FRAME["y"] = 0.4 + 0.9 * FRAME.x + 0.2 * np.cos(4.1 * ROWS)
FRAME["y2"] = -0.3 * FRAME.x + 0.25 * np.sin(2.3 * ROWS)
BUDGET = FormulaFitOptions(num_warmup=15, num_samples=8, num_chains=3, seed=11)
SOURCE = """object Row : FinSet _
object Out : Real 1
program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""


def _assert_identical(serial: MCMCResult, parallel: MCMCResult) -> None:
    """Every recorded quantity of the two runs is bit-identical."""
    assert serial.samples.keys() == parallel.samples.keys()
    for name, value in serial.samples.items():
        assert torch.equal(value, parallel.samples[name]), name
    for name in (
        "log_densities",
        "acceptance_rates",
        "divergence_counts",
        "num_draws",
        "num_evaluations",
        "num_gradient_evaluations",
    ):
        assert torch.equal(getattr(serial, name), getattr(parallel, name)), name
    assert serial.sample_stats.keys() == parallel.sample_stats.keys()
    for name, value in serial.sample_stats.items():
        assert torch.equal(value, parallel.sample_stats[name]), name
    assert serial.stopped_by == parallel.stopped_by


def test_formula_chains_in_two_processes_equal_the_serial_run():
    serial = fit("y ~ x", data=FRAME, **BUDGET)
    two: FormulaFitOptions = {**BUDGET, "cores": 2}
    parallel = fit("y ~ x", data=FRAME, **two)
    assert isinstance(serial.posterior, MCMCResult)
    assert isinstance(parallel.posterior, MCMCResult)
    _assert_identical(serial.posterior, parallel.posterior)
    # A different seed gives a different run, so equality is not vacuous.
    reseeded: FormulaFitOptions = {**BUDGET, "cores": 2, "seed": 12}
    other = fit("y ~ x", data=FRAME, **reseeded)
    assert isinstance(other.posterior, MCMCResult)
    assert not torch.equal(
        other.posterior.samples["beta_x"], parallel.posterior.samples["beta_x"]
    )
    refit = parallel.update(data=FRAME.iloc[:12])
    assert isinstance(refit.posterior, MCMCResult)
    assert refit.posterior.num_chains == 3


def test_joint_formula_chains_in_two_processes_equal_the_serial_run():
    formula = cast(MultivariateFormula, bf("y ~ x") + bf("y2 ~ x"))
    budget = JointFitOptions(num_warmup=15, num_samples=8, num_chains=2, seed=11)
    serial = fit(formula, data=FRAME, **budget)
    joint_two: JointFitOptions = {**budget, "cores": 2}
    parallel = fit(formula, data=FRAME, **joint_two)
    assert isinstance(serial.posterior, MCMCResult)
    assert isinstance(parallel.posterior, MCMCResult)
    _assert_identical(serial.posterior, parallel.posterior)


def test_model_interface_chains_in_two_processes_equal_the_serial_run():
    x = torch.tensor(FRAME.x.to_numpy(), dtype=torch.float32)
    y = torch.tensor(FRAME.y.to_numpy(), dtype=torch.float32)

    def sampler() -> MCMC:
        return MCMC(NUTSKernel(), num_warmup=15, num_samples=8, num_chains=3, seed=5)

    serial = Model(SOURCE).fit({"x": x, "y": y}, method=MCMCMethod(sampler=sampler()))
    parallel = Model(SOURCE).fit(
        {"x": x, "y": y}, method=MCMCMethod(sampler=sampler(), cores=2)
    )
    assert isinstance(serial.posterior, MCMCResult)
    assert isinstance(parallel.posterior, MCMCResult)
    _assert_identical(serial.posterior, parallel.posterior)


def _program():
    bound = Model(SOURCE).bind(
        {
            "x": torch.tensor(FRAME.x.to_numpy(), dtype=torch.float32),
            "y": torch.tensor(FRAME.y.to_numpy(), dtype=torch.float32),
        }
    )
    return bound.require_tensor_program(), bound.input, bound.tensor_data()


def test_chain_records_assemble_into_the_serial_result_in_process():
    program, x, data = _program()

    def sampler() -> MCMC:
        return MCMC(NUTSKernel(), num_warmup=10, num_samples=6, num_chains=3, seed=2)

    serial = sampler().run(program, x, data)
    parts = [
        run_chains(sampler(), program, x, data, (2, 0)),
        run_chains(sampler(), program, x, data, (1,)),
    ]
    _assert_identical(serial, assemble_chains(sampler(), program, x, data, parts))


def test_mismatched_or_incomplete_chain_records_are_refused():
    program, x, data = _program()

    def sampler(seed: int = 2) -> MCMC:
        return MCMC(NUTSKernel(), num_warmup=5, num_samples=3, num_chains=2, seed=seed)

    first = run_chains(sampler(), program, x, data, (0,))
    second = run_chains(sampler(), program, x, data, (1,))
    # A worker that derived its streams from another seed.
    wrong_seed = run_chains(sampler(3), program, x, data, (1,))
    with pytest.raises(ValueError, match="different program, data, or seed"):
        assemble_chains(sampler(), program, x, data, [first, wrong_seed])
    # A worker that rebuilt another model.
    other = ChainStates(
        fingerprint="0" * 64, run_seed=second.run_seed, states=second.states
    )
    with pytest.raises(ValueError, match="different program, data, or seed"):
        assemble_chains(sampler(), program, x, data, [first, other])
    with pytest.raises(ValueError, match="ran twice"):
        assemble_chains(sampler(), program, x, data, [first, first, second])
    with pytest.raises(ValueError, match="expected"):
        assemble_chains(sampler(), program, x, data, [first])
    with pytest.raises(ValueError, match="outside"):
        run_chains(sampler(), program, x, data, (2,))


def test_parallel_chains_refuse_what_cannot_cross_processes(tmp_path):
    program, x, data = _program()
    worker = MCMCMethod  # any picklable callable; never called below
    unseeded = MCMC(NUTSKernel(), num_warmup=2, num_samples=2, num_chains=2)
    with pytest.raises(ValueError, match="explicit integer seed"):
        parallel_chains(unseeded, program, x, data, worker, 2)  # type: ignore[arg-type]
    checkpointed = MCMC(
        NUTSKernel(),
        num_warmup=2,
        num_samples=2,
        num_chains=2,
        seed=1,
        checkpoint=MCMCCheckpoint(directory=str(tmp_path / "run"), every=1),
    )
    with pytest.raises(ValueError, match="checkpoint or call back"):
        parallel_chains(checkpointed, program, x, data, worker, 2)  # type: ignore[arg-type]
    seeded = MCMC(NUTSKernel(), num_warmup=2, num_samples=2, num_chains=2, seed=1)
    with pytest.raises(ValueError, match="cores must be a positive integer"):
        parallel_chains(seeded, program, x, data, worker, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="picklable"):
        fit(
            "y ~ x",
            data=FRAME,
            cores=2,
            init=lambda chain: {"beta_x": 0.1 * chain},
            num_warmup=15,
            num_samples=8,
            num_chains=2,
            seed=11,
        )
    with pytest.raises(ValueError, match="cores must be a positive integer"):
        zero: FormulaFitOptions = {**BUDGET, "cores": 0}
        fit("y ~ x", data=FRAME, **zero)
