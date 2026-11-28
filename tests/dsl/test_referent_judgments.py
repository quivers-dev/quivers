"""Separate bad state and resumption rules from the gallery judgment model."""

from pathlib import Path

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads
from quivers.qiec.discharge import discharge_module
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import run_program


_SOURCE = (
    Path(__file__).resolve().parents[2] / "docs/examples/source/referent_judgments.qvr"
).read_text()


@pytest.mark.parametrize(
    ("source_line", "changed_line", "missing_gradients"),
    (
        ("resume() with (value)", "resume() with (false)", (True, False)),
        ("let second <- resume(false)", "let second <- resume(true)", (True, True)),
    ),
)
def test_judgment_identifies_lost_state_or_repeated_reading(
    source_line, changed_line, missing_gradients
):
    assert _SOURCE.count(source_line) == 1
    source = _SOURCE.replace(source_line, changed_line)
    module = loads(source).qiec
    assert isinstance(module, QiecModule)
    compiled = discharge_module(module)
    assert compiled.obstacle("truth_probability") is None

    # The positive-judgment evidence when both original choices are balanced.
    expected_probability = torch.tensor(0.5, dtype=torch.float64)
    for candidate in (module, compiled.module):
        logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        run = run_program(
            candidate,
            "judgment",
            data={
                "lexical_logit": logits[0],
                "pronoun_logit": logits[1],
                "truth": True,
            },
            sites={},
            distributions=TensorBackend(),
        )
        assert not torch.isclose(run.log_joint.exp(), expected_probability)
        (gradient,) = torch.autograd.grad(run.log_joint, logits)
        assert tuple(gradient.abs() < 1e-14) == missing_gradients
