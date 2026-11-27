"""A saved program fit keeps the program's text, not its file.

Oracle: the pointwise log likelihood of the fit before saving. The fit is
loaded after the ``.qvr`` file it was read from is deleted, and scores the
same rows to the same values; the program's text, entry, response and
replaced parameters are the artifact's own.
"""

from __future__ import annotations

from pathlib import Path

import torch

from quivers.formulas import BayesianFit, fit
from quivers.formulas._program_kernel import ProgramKernelSpec
from quivers.modeling.artifact import load_fit, save_fit

from tests.formulas._program_cases import AROUND_ROWS, HALFWIDTH, frame


def test_a_saved_fit_loads_without_its_program_file(tmp_path: Path) -> None:
    program = tmp_path / "approximators.qvr"
    program.write_text(AROUND_ROWS, encoding="utf-8")
    fitted = fit(
        program=(program, "around_rows"),
        parameters={"noise": "~ x + (1 | g)", "res_logits": "~ 1"},
        response="accept",
        data=frame(),
        data2={"halfwidth": HALFWIDTH},
        method="svi",
        num_steps=3,
        seed=0,
    )
    before = fitted.log_likelihood(num_samples=5)["accept"]
    target = tmp_path / "fit"
    save_fit(fitted, target)
    program.unlink()
    loaded = load_fit(target)
    assert isinstance(loaded, BayesianFit)
    after = loaded.log_likelihood(num_samples=5)["accept"]
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
    law = loaded.family.kernel_law
    assert law == ProgramKernelSpec(
        source=AROUND_ROWS,
        entry="around_rows",
        response="accept",
        parameters=("res_logits", "noise"),
    )
    assert loaded.qvr_source == fitted.qvr_source
    assert fitted.refit_recipe is not None
    assert fitted.refit_recipe.method["program"] == AROUND_ROWS
    refitted = loaded.update()
    torch.testing.assert_close(
        refitted.log_likelihood(num_samples=5)["accept"], before, rtol=0.0, atol=0.0
    )
