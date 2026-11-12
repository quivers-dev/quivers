"""A program step drawing from a decoder or an encoder is a located compile error.

A decoder's values are terms of its signature and an encoder is a
deterministic map from terms to vectors; neither is a kernel over the
tensor values a program's sites hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quivers.dsl import CompileError, loads

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "source"
    / "term_autoencoder.qvr"
)


def _source(step: str) -> str:
    """The example's structural declarations and a program with one step.

    Parameters
    ----------
    step : str
        The program step under test.

    Returns
    -------
    str
        QVR source.
    """
    return (
        _EXAMPLE.read_text()
        + f"""
object Code : FinSet 24
program p : Code -> Code
    sample h : Code <- Normal(loc=0.0, scale=1.0)
    {step}
    return h
export p
"""
    )


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ("sample t <- Dec(h)", "'Dec' is a decoder"),
        ("observe t <- Dec(h)", "'Dec' is a decoder"),
        ("sample t <- Enc(h)", "'Enc' is an encoder"),
    ],
)
def test_structural_draw_is_a_located_compile_error(step: str, message: str) -> None:
    """The step raises `CompileError` at its own line, not `AttributeError`."""
    source = _source(step)
    line = next(
        number
        for number, text in enumerate(source.splitlines(), start=1)
        if text.strip() == step
    )
    with pytest.raises(CompileError, match=message) as caught:
        loads(source)
    assert caught.value.line == line


def test_a_family_draw_in_the_same_program_compiles() -> None:
    """The refusal is specific to the structural kernel: a family draw compiles."""
    loads(_source("sample t <- Normal(loc=h, scale=1.0)"))
