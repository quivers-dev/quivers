"""Stable diagnostic codes for the conditions a tool dispatches on.

A code is what an editor, a CI gate, or a migration script keys off, so
it has to name the condition rather than the pass that happened to catch
it. Codes are attached where a rejection is raised, because only that
site knows which condition fired, and the lowerer preserves them rather
than overwriting with its own.

Every case here is a source module that must be rejected, paired with
the code the rejection must carry. A test that only checked rejection
would pass even if every condition collapsed to one code.
"""

from __future__ import annotations

import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.dsl.qiec_tooling import QIEC_DIAGNOSTIC_CODES


_HANDLER = """effect E
    op : Unit -> Int
handler h for E : Int -> Int [coverage=total, implementation=authored]
    return x =>
        return {answer}
    op() resumes {grade} =>
{body}
"""


def _handler(
    *, grade: str = "1", body: str = "        resume(0)", answer: str = "x"
) -> str:
    """A one-operation handler, varied where a test needs it.

    Parameters
    ----------
    grade : str
        The clause's declared resumption grade.
    body : str
        The clause body, indented to its block.
    answer : str
        What the return clause answers with.

    Returns
    -------
    str
        QVR source for the module.
    """
    return _HANDLER.format(grade=grade, body=body, answer=answer)


_CASES: dict[str, tuple[str, str]] = {
    "a clause resuming under grade zero": (
        _handler(grade="0"),
        "qiec-resumption",
    ),
    "a linear clause that never resumes": (
        _handler(grade="1", body="        return 0"),
        "qiec-resumption",
    ),
    "a return clause answering the wrong type": (
        _handler(answer='"text"'),
        "qiec-handler-body",
    ),
    "a call to an undeclared computation": (
        "define a() : Int !{} =\n    missing()\n",
        "qiec-route",
    ),
    "a call with the wrong value arity": (
        "define one(x : Int) : Int !{} =\n    return x\n"
        "define two() : Int !{} =\n    one()\n",
        "qiec-call-arity",
    ),
    "a filter with data-dependent output shape": (
        "define positives() : Tensor[Real]([2]) !{} =\n"
        "    return filter([-1.0, 1.0], x -> x > 0.0)\n",
        "qiec-collection:filter:dynamic-shape",
    ),
    "an eager-only builtin": (
        "define normalized(x : Tensor[Real]([2])) : Tensor[Real]([2]) !{} =\n"
        "    return softmin(x)\n",
        "qiec-builtin:host-only",
    ),
}


@pytest.mark.parametrize(
    ("source", "expected"), list(_CASES.values()), ids=list(_CASES)
)
def test_a_rejection_carries_the_code_for_its_condition(
    source: str, expected: str
) -> None:
    """Each condition reports its own code, not a neighbour's.

    The arity and resumption cases are the ones that regressed before:
    the lowerer wrapped a kernel rejection and replaced its code with
    whichever one the surrounding pass used, collapsing a precise
    condition into a generic one.
    """
    with pytest.raises(QiecDiagnosticError) as raised:
        lower_qvr_to_qiec(parse(source), module_name="codes")
    assert raised.value.code == expected


@pytest.mark.parametrize("expected", sorted({code for _, code in _CASES.values()}))
def test_every_code_used_here_is_published(expected: str) -> None:
    """A code a tool can receive must appear in the published set.

    `qiec_diagnostic` falls back to classifying by message text for a
    code it does not recognise, so an unpublished code would be silently
    reclassified and the dispatch it was meant to drive would not happen.
    """
    assert expected in QIEC_DIAGNOSTIC_CODES


def test_a_valid_module_raises_nothing() -> None:
    """The control: the fixtures differ from valid source only where intended.

    Without it, a mistake in the shared template could make every case
    above fail for a reason unrelated to the code being tested.
    """
    lower_qvr_to_qiec(parse(_handler()), module_name="codes")
