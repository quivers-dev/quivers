"""A posterior block's parameters take the types of the latents it consumes.

A program declared ``[posterior_of=model]`` names latents of ``model`` as its
parameters, so each parameter has the type of the model's site of that
name: a real for a real draw, a real vector for a vector draw. An
unannotated parameter is not an index merely because the block's
domain is a finite set.
"""

from __future__ import annotations

import math

import pytest

from quivers.dsl import loads
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.program_runtime import run_program

_SCALAR = """
object Item : FinSet 1
object R : Real 1
program scored : Item -> R
    sample z <- Normal(loc=0.0, scale=1.0)
    return z
program post({parameter}) : Item -> R [effects=[Pure], posterior_of=scored]
    let w = exp({body})
    return w
export scored
"""

_VECTOR = """
object Item : FinSet 1
object K : FinSet 4
object L4 : Real 4
program scored : Item -> L4
    sample raw : K <- Normal(loc=0.0, scale=1.0)
    return raw
program post(raw) : Item -> L4 [effects=[Pure], posterior_of=scored]
    let p = softmax(raw)
    return p
export scored
"""


def test_unannotated_real_latent_is_real() -> None:
    """``exp(z)`` of a consumed real latent checks and runs at its value."""
    program = loads(_SCALAR.format(parameter="z", body="z"))
    assert program.qiec is not None
    run = run_program(program.qiec, "post", data={"z": 0.7}, sites={})
    assert float(run.value) == pytest.approx(math.exp(0.7), abs=1e-12)  # type: ignore[arg-type]


def test_unannotated_vector_latent_is_a_real_vector() -> None:
    """``softmax(raw)`` of a consumed vector latent normalizes its entries."""
    program = loads(_VECTOR)
    assert program.qiec is not None
    raw = [0.1, -0.4, 1.2, 0.0]
    run = run_program(program.qiec, "post", data={"raw": tuple(raw)}, sites={})
    total = sum(math.exp(v) for v in raw)
    expected = [math.exp(v) / total for v in raw]
    assert [float(v) for v in run.value] == pytest.approx(expected, abs=1e-12)  # type: ignore[union-attr]


def test_parameter_naming_no_latent_is_refused() -> None:
    """A parameter that names no latent of the consumed model is located."""
    with pytest.raises(QiecDiagnosticError, match="not a latent the model"):
        loads(_SCALAR.format(parameter="q", body="q"))
