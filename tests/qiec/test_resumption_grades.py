"""Holding a handler clause to the resumption grade it declares.

A grade is a promise other code relies on. A handler declared `0` may be
compiled without keeping the continuation alive at all, so a clause that
resumes anyway is not a stylistic lapse but a miscompilation waiting to
happen. Equally a `1` promises the continuation always runs, which a
caller may depend on for an effect to be observed.

The analysis counts invocations along the paths through a body, as an
interval: the fewest any path performs and the most any path performs. A
`case` is where the two diverge, and that is the case worth testing
hardest, because a single-path body cannot tell an interval apart from a
count.
"""

from __future__ import annotations

import pytest

from quivers.qiec import EMPTY_ROW, INT, ConstructorId, StaticScopeId
from quivers.qiec.checking import (
    KernelError,
    ResumptionUse,
    check_resumption_grade,
    resumption_use,
)
from quivers.qiec.effects import ResumptionGrade
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.terms import (
    Bind,
    Case,
    CaseBranch,
    CaseMotive,
    LiteralValue,
    Local,
    Resume,
    Return,
    Var,
)


_ORIGIN = SourceOrigin("tests", ("path",), "resume", "qvr-source/v0.19")
_RESUME = Resume(LiteralValue(1, INT), _ORIGIN)
_PLAIN = Return(LiteralValue(1, INT))


def _branch(body, tag: str) -> CaseBranch:
    """One case branch with the given body.

    Parameters
    ----------
    body : Computation
        What the branch does.
    tag : str
        Distinguishes this branch's constructor and scope from others.

    Returns
    -------
    CaseBranch
        The branch.
    """
    return CaseBranch(
        ConstructorId.derive("tests", tag),
        (),
        (),
        body,
        StaticScopeId.derive("tests", tag),
    )


def _case(*bodies) -> Case:
    """A case whose branches run the given bodies.

    Parameters
    ----------
    bodies : Computation
        One body per branch.

    Returns
    -------
    Case
        The case term.
    """
    return Case(
        Var(Local("x", INT)),
        CaseMotive((), INT),
        tuple(_branch(body, f"C{index}") for index, body in enumerate(bodies)),
    )


def test_a_body_that_never_resumes_counts_zero() -> None:
    """The baseline the other counts are measured against."""
    assert resumption_use(_PLAIN) == ResumptionUse(minimum=0, maximum=0)


def test_a_body_that_resumes_once_counts_one() -> None:
    """A bare resumption is exactly one invocation."""
    assert resumption_use(_RESUME) == ResumptionUse(minimum=1, maximum=1)


def test_sequenced_resumptions_add() -> None:
    """Both halves of a bind run, so their counts add.

    This is what separates an affine clause from an unrestricted one, and
    a body that resumes twice in sequence is the simplest way to exceed
    an affine promise.
    """
    twice = Bind(Local("a", INT), _RESUME, Bind(Local("b", INT), _RESUME, _PLAIN))
    assert resumption_use(twice) == ResumptionUse(minimum=2, maximum=2)


def test_a_call_cannot_resume_this_clause() -> None:
    """A callee has no access to the caller's continuation.

    `resume` is lexical to the clause enclosing it, so a call contributes
    nothing however the callee is written. Without this the analysis
    would have to be conservative across every call and no clause could
    ever be shown linear.
    """
    from quivers.qiec import ComputationId
    from quivers.qiec.terms import Call

    call = Call(
        ComputationId.derive("tests", "helper"),
        "helper",
        (),
        (),
        INT,
        EMPTY_ROW,
        _ORIGIN,
    )
    assert resumption_use(call) == ResumptionUse(minimum=0, maximum=0)


def test_branches_widen_the_interval_rather_than_adding() -> None:
    """Only one branch runs, so the counts join rather than sum.

    One branch resuming and one not gives `0..1`: some path resumes and
    some does not. Reading that as a count would make it either wrongly
    linear or wrongly forbidden.
    """
    assert resumption_use(_case(_RESUME, _PLAIN)) == ResumptionUse(minimum=0, maximum=1)
    assert resumption_use(_case(_RESUME, _RESUME)) == ResumptionUse(
        minimum=1, maximum=1
    )


@pytest.mark.parametrize(
    ("grade", "use", "allowed"),
    [
        (ResumptionGrade.ZERO, ResumptionUse(minimum=0, maximum=0), True),
        (ResumptionGrade.ZERO, ResumptionUse(minimum=1, maximum=1), False),
        (ResumptionGrade.ZERO, ResumptionUse(minimum=0, maximum=1), False),
        (ResumptionGrade.AFFINE, ResumptionUse(minimum=0, maximum=0), True),
        (ResumptionGrade.AFFINE, ResumptionUse(minimum=0, maximum=1), True),
        (ResumptionGrade.AFFINE, ResumptionUse(minimum=1, maximum=1), True),
        (ResumptionGrade.AFFINE, ResumptionUse(minimum=2, maximum=2), False),
        (ResumptionGrade.AFFINE, ResumptionUse(minimum=0, maximum=None), False),
        (ResumptionGrade.LINEAR, ResumptionUse(minimum=1, maximum=1), True),
        (ResumptionGrade.LINEAR, ResumptionUse(minimum=0, maximum=1), False),
        (ResumptionGrade.LINEAR, ResumptionUse(minimum=0, maximum=0), False),
        (ResumptionGrade.LINEAR, ResumptionUse(minimum=2, maximum=2), False),
        (ResumptionGrade.UNRESTRICTED, ResumptionUse(minimum=0, maximum=None), True),
        (ResumptionGrade.UNRESTRICTED, ResumptionUse(minimum=9, maximum=9), True),
    ],
    ids=lambda value: str(value),
)
def test_each_grade_admits_exactly_its_own_uses(
    grade: ResumptionGrade, use: ResumptionUse, allowed: bool
) -> None:
    """The full table, so no grade silently admits a neighbour's uses.

    Linear is the strict one: it rejects both resuming twice and failing
    to resume on some path, and the second is easy to lose because every
    other grade tolerates it.
    """
    if allowed:
        check_resumption_grade(grade, use, subject="clause")
        return
    with pytest.raises(KernelError):
        check_resumption_grade(grade, use, subject="clause")


def test_an_unbounded_use_is_refused_by_every_bounded_grade() -> None:
    """The lattice top must not pass for a small count.

    Nothing produces an unbounded count today, since `resume` is lexical
    and calls contribute nothing. The top exists so that an analysis
    extended to a construct it cannot bound fails closed rather than
    reporting zero.
    """
    unbounded = ResumptionUse(minimum=0, maximum=None)
    for grade in (
        ResumptionGrade.ZERO,
        ResumptionGrade.AFFINE,
        ResumptionGrade.LINEAR,
    ):
        with pytest.raises(KernelError):
            check_resumption_grade(grade, unbounded, subject="clause")
    check_resumption_grade(ResumptionGrade.UNRESTRICTED, unbounded, subject="clause")
