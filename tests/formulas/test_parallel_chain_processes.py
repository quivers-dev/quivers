"""Chain workers start without executing the calling script again.

A fit with several cores is called from the top level of a script that has
no ``if __name__ == "__main__":`` guard. The script appends one line to a
marker file each time it is executed, so a second execution in a worker
process is visible. The remaining tests pin the located refusal of a recipe
that refers to the main module and the cleanup after a failing worker.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from quivers.inference.mcmc._chain_process import (
    run_worker_processes,
    worker_payload,
)
from quivers.inference.mcmc.parallel import ChainStates

BUDGET = (
    "method='nuts', num_warmup=4, num_samples=4, num_chains=2, seed=3, "
    "control={'max_treedepth': 2}"
)


def _run(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a script file with this interpreter and the test's import path."""
    path = tmp_path / "analysis.py"
    path.write_text(textwrap.dedent(script))
    return subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
        check=False,
    )


def test_unguarded_script_runs_once_and_matches_the_serial_fit(tmp_path: Path) -> None:
    result = _run(
        f"""
        import pandas as pd
        import torch
        from quivers.formulas import fit

        with open("executions.txt", "a") as marker:
            marker.write("executed\\n")
        frame = pd.DataFrame(
            {{"x": [-1.0, -0.5, 0.0, 0.4, 0.9, 1.3], "y": [0.1, 0.4, 0.2, 0.9, 1.1, 1.6]}}
        )
        serial = fit("y ~ x", data=frame, {BUDGET})
        parallel = fit("y ~ x", data=frame, cores=2, {BUDGET})
        same = all(
            torch.equal(serial.posterior.samples[site], parallel.posterior.samples[site])
            for site in serial.posterior.samples
        )
        print("IDENTICAL" if same else "DIFFERENT")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == ["IDENTICAL"]
    assert (tmp_path / "executions.txt").read_text() == "executed\n"
    assert "freeze_support" not in result.stderr
    assert "RuntimeWarning" not in result.stderr


def test_recipe_from_the_main_module_is_refused_before_any_process(
    tmp_path: Path,
) -> None:
    result = _run(
        f"""
        import pandas as pd
        from quivers.formulas import fit


        def start(chain):
            return {{"beta_x": 0.1 * chain}}


        frame = pd.DataFrame(
            {{"x": [-1.0, -0.5, 0.0, 0.4, 0.9, 1.3], "y": [0.1, 0.4, 0.2, 0.9, 1.1, 1.6]}}
        )
        try:
            fit("y ~ x", data=frame, cores=2, init=start, {BUDGET})
        except ValueError as error:
            print(str(error))
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "defined in the main module (__main__)" in result.stdout
    assert "use cores=1" in result.stdout


class FailingWorker:
    """A recipe whose first chain raises and whose other chains never finish."""

    def __call__(self, chains: tuple[int, ...]) -> ChainStates:
        if 0 in chains:
            raise ValueError("the first chain cannot run")
        time.sleep(600)
        return ChainStates(fingerprint="", run_seed=0, states={})


class EchoWorker:
    """A recipe that returns its assignment as the run seed."""

    def __call__(self, chains: tuple[int, ...]) -> ChainStates:
        return ChainStates(fingerprint="echo", run_seed=sum(chains), states={})


def test_worker_records_return_in_assignment_order() -> None:
    parts = run_worker_processes(EchoWorker(), [(0, 2), (1,), (3, 4)], threads=1)
    assert [part.run_seed for part in parts] == [2, 1, 7]
    assert all(part.fingerprint == "echo" for part in parts)


def test_failing_worker_reports_its_error_and_stops_the_others() -> None:
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="the first chain cannot run") as raised:
        run_worker_processes(FailingWorker(), [(0,), (1,)], threads=1)
    assert time.monotonic() - started < 120
    assert "chains [0]" in str(raised.value)
    assert "Traceback" in str(raised.value)


def test_unpicklable_recipe_is_refused_with_its_reason() -> None:
    with pytest.raises(ValueError, match="must be picklable"):
        worker_payload(lambda chains: chains, (0,))  # type: ignore[arg-type]
