"""Static substitution through computation terms, and what it must not touch.

Instantiating a callee's telescope at a call site means substituting into
its result type, its effect row, and eventually its body. The property
that makes this sound is that substitution is blind to rigid variables: a
declaration binder carries no identity and is matched by name, while a
case branch's skolem carries a derived identity and is left alone.

Every test here is paired. One half shows the substitution reaches a
position, and the other shows it stops at a binder that shadows the same
name, since a rewrite that reached too far would capture.
"""

from __future__ import annotations

import pytest

from quivers.qiec import (
    EMPTY_ROW,
    INT,
    STRING,
    TYPE,
    ComputationId,
    EffectId,
    EffectRef,
    TypeVariable,
)
from quivers.qiec.identifiers import (
    EffectInstanceId,
    SourceOrigin,
    StaticVariableId,
)
from quivers.qiec.substitution import StaticSubstitution, substitute_computation
from quivers.qiec.terms import (
    Bind,
    Call,
    LiteralValue,
    Local,
    NewInstance,
    Resume,
    Return,
    Var,
)


def _origin(role: str = "call") -> SourceOrigin:
    """A source origin for a term under test.

    Parameters
    ----------
    role : str
        The role recorded on the origin.

    Returns
    -------
    SourceOrigin
        A fixed origin, so two terms differ only where a test varies
        them.
    """
    return SourceOrigin(
        module="tests",
        structural_path=("path",),
        role=role,
        source_protocol="qvr-source/v0.19",
    )


#: A declaration binder: no identity, so substitution matches it by name.
FREE = TypeVariable(name="A", kind=TYPE)

#: A rigid variable of the *same name*, as a case branch would introduce.
#: Substitution must not touch it, or a branch's refinement would leak.
RIGID = TypeVariable(
    name="A", kind=TYPE, identity=StaticVariableId.derive("branch-scope", "A")
)

#: Binds the name both variables share.
SUBSTITUTION = StaticSubstitution(types=(("A", INT),))


def test_empty_substitution_shares_a_deep_immutable_computation() -> None:
    """Identity substitution preserves a deep graph without recursive copying."""
    body = Return(value=LiteralValue(value=1, type=INT))
    for index in range(1100):
        body = Bind(
            binder=Local(name=f"x{index}", type=INT),
            first=Return(value=LiteralValue(value=index, type=INT)),
            then=body,
        )
    assert substitute_computation(body, StaticSubstitution()) is body


def test_empty_substitution_still_refuses_a_value_in_computation_position() -> None:
    """The identity path keeps the computation union closed."""
    with pytest.raises(TypeError, match="unknown computation"):
        substitute_computation(LiteralValue(value=1, type=INT), StaticSubstitution())


def _call(result_type: TypeVariable | type) -> Call:
    """A call whose static argument and result type are `result_type`."""
    return Call(
        callee=ComputationId.derive("tests", "helper"),
        name="helper",
        static_arguments=(result_type,),
        arguments=(),
        result_type=result_type,
        effects=EMPTY_ROW,
        origin=_origin(),
    )


def test_a_call_substitutes_its_static_argument_and_result_type() -> None:
    """A call is where a callee's telescope is instantiated."""
    result = substitute_computation(_call(FREE), SUBSTITUTION)
    assert result.result_type == INT
    assert result.static_arguments == (INT,)


def test_a_call_does_not_substitute_a_rigid_variable_of_the_same_name() -> None:
    """The capture case, and the reason the rewrite is sound.

    A branch skolem named `A` must survive a substitution binding `A`.
    If it did not, a type mentioning the branch's own variable would be
    rewritten to whatever the enclosing declaration bound, silently
    changing what the branch proves.
    """
    result = substitute_computation(_call(RIGID), SUBSTITUTION)
    assert result.result_type == RIGID
    assert result.static_arguments == (RIGID,)


def test_a_call_keeps_the_identity_it_points_at() -> None:
    """Substitution rewrites types, not references.

    A call that changed callee under substitution would resolve to a
    different declaration, which is how an inlining bug would present.
    """
    original = _call(FREE)
    assert substitute_computation(original, SUBSTITUTION).callee == original.callee


def test_a_resumption_substitutes_the_value_it_resumes_with() -> None:
    """`resume` carries a value whose type may mention a binder."""
    resumption = Resume(
        value=LiteralValue(value=1, type=FREE), origin=_origin("resume")
    )
    result = substitute_computation(resumption, SUBSTITUTION)
    assert result.value.type == INT
    assert result.origin == resumption.origin


def test_a_local_instance_substitutes_its_interface_and_body() -> None:
    """An allocation's interface application may be parameterised."""
    effect = EffectRef(
        id=EffectId.derive("tests", "State"), name="State", arguments=(FREE,)
    )
    allocation = NewInstance(
        instance=EffectInstanceId.derive("tests", "cell"),
        effect=effect,
        body=Return(value=LiteralValue(value=0, type=FREE)),
        origin=_origin("instance"),
    )
    result = substitute_computation(allocation, SUBSTITUTION)
    assert result.effect.arguments == (INT,)
    assert result.body.value.type == INT
    assert result.instance == allocation.instance, (
        "the allocated instance identity must not move under substitution, "
        "or a row entry would stop naming the instance that discharges it"
    )


def test_substitution_reaches_a_binder_type_inside_a_sequence() -> None:
    """A `let` binder's own type is a substitution position too.

    Easy to miss, since the binder is a value binder and the eye reads it
    as opaque, but its type mentions static variables like any other.
    """
    bound = Bind(
        binder=Local(name="x", type=FREE),
        first=Return(value=LiteralValue(value=1, type=FREE)),
        then=Return(value=Var(local=Local(name="x", type=FREE))),
    )
    result = substitute_computation(bound, SUBSTITUTION)
    assert result.binder.type == INT
    assert result.then.value.local.type == INT


def test_a_substitution_that_binds_nothing_leaves_a_term_unchanged() -> None:
    """The identity case, which guards against a rewrite that invents.

    A substitution over an unrelated name must return an equal term, so
    a structural walk that dropped or defaulted a field is caught here
    rather than in whichever pass first notices the loss.
    """
    unrelated = StaticSubstitution(types=(("B", STRING),))
    original = _call(FREE)
    assert substitute_computation(original, unrelated) == original


@pytest.mark.parametrize("term", ["call", "resume", "instance"])
def test_every_new_term_is_handled_rather_than_rejected(term: str) -> None:
    """None of the new terms falls through to the unknown-term error.

    The walk raises on an unrecognised class, which is right, but it
    means a term added to the union and not to the walk fails only when
    something substitutes into it. This checks each one directly.
    """
    terms = {
        "call": _call(FREE),
        "resume": Resume(
            value=LiteralValue(value=1, type=FREE), origin=_origin("resume")
        ),
        "instance": NewInstance(
            instance=EffectInstanceId.derive("tests", "cell"),
            effect=EffectRef(
                id=EffectId.derive("tests", "State"), name="State", arguments=(FREE,)
            ),
            body=Return(value=LiteralValue(value=0, type=FREE)),
            origin=_origin("instance"),
        ),
    }
    substitute_computation(terms[term], SUBSTITUTION)
