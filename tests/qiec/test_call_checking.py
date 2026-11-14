"""Checking calls, allocations, and resumptions against declarations.

A call resolves against a registered signature rather than against the
callee's body. That is what makes a forward call ordinary: every
signature is registered before any body is checked, so `is_even` may call
`is_odd` declared below it, and the two may call each other, without a
forward declaration.

The rejections matter as much as the acceptances. A call records the
result type and row it expects, and the checker compares those against
the instantiated signature, so a stale call site fails here rather than
flowing a wrong type onward to a backend.
"""

from __future__ import annotations

import pytest

from quivers.qiec import (
    BOOL,
    EMPTY_ROW,
    INT,
    STRING,
    TYPE,
    ComputationId,
    KernelRegistry,
)
from quivers.qiec.checking import (
    CheckContext,
    ComputationSignature,
    KernelError,
    ResumptionType,
    infer_computation,
)
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.kinds import TypeBinder
from quivers.qiec.terms import Call, LiteralValue, Resume
from quivers.qiec.types import TypeVariable


_ORIGIN = SourceOrigin(
    module="tests",
    structural_path=("path",),
    role="call",
    source_protocol="qvr-source/v0.19",
)
_A = TypeVariable(name="A", kind=TYPE)


def _registry() -> tuple[KernelRegistry, ComputationId, ComputationId]:
    """A registry holding a monomorphic and a polymorphic signature.

    Returns
    -------
    tuple[KernelRegistry, ComputationId, ComputationId]
        The registry, the identity of `takes_int`, and the identity of
        `identity`.
    """
    registry = KernelRegistry()
    takes_int = ComputationId.derive("tests", "takes_int")
    polymorphic = ComputationId.derive("tests", "identity")
    registry.register_computation(
        ComputationSignature(takes_int, "takes_int", (), (INT,), INT, EMPTY_ROW)
    )
    registry.register_computation(
        ComputationSignature(
            polymorphic,
            "identity",
            (TypeBinder(name="A", kind=TYPE),),
            (_A,),
            _A,
            EMPTY_ROW,
        )
    )
    return registry, takes_int, polymorphic


def test_a_well_typed_call_infers_the_callee_result() -> None:
    """The ordinary case the rejections below are measured against."""
    registry, takes_int, _ = _registry()
    call = Call(
        callee=takes_int,
        name="takes_int",
        static_arguments=(),
        arguments=(LiteralValue(value=1, type=INT),),
        result_type=INT,
        effects=EMPTY_ROW,
        origin=_ORIGIN,
    )
    assert infer_computation(call, registry).result == INT


def test_a_polymorphic_call_instantiates_the_telescope() -> None:
    """A static argument instantiates the callee's binders.

    The result is the callee's declared result under that instantiation,
    not the declared result itself, so `identity[Int]` returns `Int`
    rather than the binder `A`.
    """
    registry, _, polymorphic = _registry()
    call = Call(
        callee=polymorphic,
        name="identity",
        static_arguments=(INT,),
        arguments=(LiteralValue(value=7, type=INT),),
        result_type=INT,
        effects=EMPTY_ROW,
        origin=_ORIGIN,
    )
    assert infer_computation(call, registry).result == INT


def test_mutual_recursion_needs_no_forward_declaration() -> None:
    """Signatures are collected before bodies, so order does not matter.

    Without this, a module would have to be written in dependency order
    and two computations could not call each other at all.
    """
    registry = KernelRegistry()
    even = ComputationId.derive("tests", "is_even")
    odd = ComputationId.derive("tests", "is_odd")
    for identity, name in ((even, "is_even"), (odd, "is_odd")):
        registry.register_computation(
            ComputationSignature(identity, name, (), (INT,), BOOL, EMPTY_ROW)
        )
    forward = Call(
        callee=odd,
        name="is_odd",
        static_arguments=(),
        arguments=(LiteralValue(value=1, type=INT),),
        result_type=BOOL,
        effects=EMPTY_ROW,
        origin=_ORIGIN,
    )
    assert infer_computation(forward, registry).result == BOOL


def test_a_duplicate_registration_is_rejected() -> None:
    """Two declarations cannot share one identity.

    A call carries an identity, so a duplicate would resolve to whichever
    declaration registered last and make the module's signature table
    lossy.
    """
    registry, takes_int, _ = _registry()
    with pytest.raises(KernelError, match="already registered"):
        registry.register_computation(
            ComputationSignature(takes_int, "takes_int", (), (), INT, EMPTY_ROW)
        )


@pytest.mark.parametrize(
    ("label", "build"),
    [
        (
            "an unregistered callee",
            lambda ids: Call(
                callee=ComputationId.derive("tests", "absent"),
                name="absent",
                static_arguments=(),
                arguments=(),
                result_type=INT,
                effects=EMPTY_ROW,
                origin=_ORIGIN,
            ),
        ),
        (
            "too few value arguments",
            lambda ids: Call(
                callee=ids[0],
                name="takes_int",
                static_arguments=(),
                arguments=(),
                result_type=INT,
                effects=EMPTY_ROW,
                origin=_ORIGIN,
            ),
        ),
        (
            "an argument of the wrong type",
            lambda ids: Call(
                callee=ids[0],
                name="takes_int",
                static_arguments=(),
                arguments=(LiteralValue(value="x", type=STRING),),
                result_type=INT,
                effects=EMPTY_ROW,
                origin=_ORIGIN,
            ),
        ),
        (
            "a stale recorded result type",
            lambda ids: Call(
                callee=ids[0],
                name="takes_int",
                static_arguments=(),
                arguments=(LiteralValue(value=1, type=INT),),
                result_type=STRING,
                effects=EMPTY_ROW,
                origin=_ORIGIN,
            ),
        ),
        (
            "too few static arguments",
            lambda ids: Call(
                callee=ids[1],
                name="identity",
                static_arguments=(),
                arguments=(LiteralValue(value=7, type=INT),),
                result_type=INT,
                effects=EMPTY_ROW,
                origin=_ORIGIN,
            ),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_malformed_call_is_rejected(label: str, build) -> None:
    """Each way a call can disagree with its callee is caught.

    The stale-result case is the subtle one. A call records what it
    expects, so a signature that changed after the call was built shows
    up here instead of as a wrong type reaching a renderer.
    """
    registry, takes_int, polymorphic = _registry()
    with pytest.raises(KernelError):
        infer_computation(build((takes_int, polymorphic)), registry)


def test_resume_outside_a_handler_clause_is_rejected() -> None:
    """There is no continuation to invoke outside a clause body.

    `resume` is a capability of the position rather than a bound name, so
    the check is that the position provides one.
    """
    registry, _, _ = _registry()
    with pytest.raises(KernelError, match="outside a handler clause"):
        infer_computation(
            Resume(value=LiteralValue(value=1, type=INT), origin=_ORIGIN), registry
        )


def test_resume_inside_a_clause_answers_the_clause_type() -> None:
    """Resuming produces the clause's answer, not the operation result."""
    registry, _, _ = _registry()
    context = CheckContext().with_resumption(ResumptionType(INT, STRING, EMPTY_ROW))
    result = infer_computation(
        Resume(value=LiteralValue(value=1, type=INT), origin=_ORIGIN), registry, context
    )
    assert result.result == STRING


def test_resume_must_carry_what_the_operation_supplies() -> None:
    """The value resumed with is the operation's result type.

    Resuming with anything else would hand the suspended computation a
    value of a type it cannot use.
    """
    registry, _, _ = _registry()
    context = CheckContext().with_resumption(ResumptionType(INT, STRING, EMPTY_ROW))
    with pytest.raises(KernelError, match="resume carries"):
        infer_computation(
            Resume(value=LiteralValue(value="x", type=STRING), origin=_ORIGIN),
            registry,
            context,
        )


def test_a_resumption_survives_entering_a_nested_binding() -> None:
    """Context extension carries the continuation through.

    A clause body almost always binds something before resuming, so a
    context method that dropped the resumption would make `resume`
    unusable in practice while every direct test still passed.
    """
    registry, _, _ = _registry()
    context = CheckContext().with_resumption(ResumptionType(INT, STRING, EMPTY_ROW))
    from quivers.qiec.terms import Local

    deeper = context.extend(Local(name="x", type=INT))
    assert deeper.resumption == context.resumption
