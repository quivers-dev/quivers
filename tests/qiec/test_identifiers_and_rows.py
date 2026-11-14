import json

import didactic.api as dx
import pytest

from quivers.qiec import (
    BOOL,
    INT,
    NAT,
    AttachmentId,
    DynamicAddressFrame,
    EffectId,
    EffectBinder,
    EffectInstanceId,
    EffectRef,
    EffectRow,
    EffectVariable,
    IndexBinder,
    IndexConstructor,
    IndexLiteral,
    OperationId,
    RowEntry,
    RowVariableId,
    RowVariable,
    SiteProvenance,
    SourceOrigin,
    TypeBinder,
    TypeConstructorRef,
    TypeId,
    TypeVariable,
    UserIndexSort,
    instantiate_telescope,
    unify_effect_rows,
)


def test_identifiers_are_namespaced_stable_and_round_trip() -> None:
    first = EffectId.derive("example", "Reader")
    second = EffectId.derive("example", "Reader")
    operation = OperationId.derive("example", "Reader", 1)

    assert first == second
    assert str(first) != str(operation)
    assert EffectId.parse(first.to_data()) == first
    assert AttachmentId.parse(str(AttachmentId.derive("host", "dist")))


def test_site_identity_uses_structural_origin_not_diagnostic_coordinates() -> None:
    before = SourceOrigin(
        module="model",
        structural_path=("declarations", 2, "body"),
        role="sample",
        source_protocol="qvr-source/v0.19",
        file="model.qvr",
        line=10,
        column=4,
    )
    after = SourceOrigin(
        module="model",
        structural_path=("declarations", 2, "body"),
        role="sample",
        source_protocol="qvr-source/v0.19",
        file="renamed.qvr",
        line=90,
        column=1,
    )
    provenance = SiteProvenance(
        origin=before,
        dynamic_path=(DynamicAddressFrame(scope="plate", key=3),),
        resumption_path=(1, 0),
    )

    assert before.site_id() == after.site_id()
    assert provenance.dynamic_key()[1] == (("plate", 3),)
    json.dumps(provenance.to_data())


def test_effect_rows_are_finite_maps_over_lexical_instances() -> None:
    effect_id = EffectId.derive("example", "State")
    effect = EffectRef(id=effect_id, name="State")
    left_id = EffectInstanceId.derive("scope", "left")
    right_id = EffectInstanceId.derive("scope", "right")
    tail_id = RowVariableId.derive("tests", "rho")
    tail = RowVariable(name="rho", identity=tail_id, lacks=(left_id, right_id))

    row = EffectRow(
        entries=(
            RowEntry(instance=right_id, effect=effect),
            RowEntry(instance=left_id, effect=effect),
        ),
        tail=tail,
    )
    assert row == EffectRow(entries=tuple(reversed(row.entries)), tail=tail)
    assert row.lookup(left_id) == effect
    assert row.without(left_id).contains(right_id)
    assert len(row.entries) == 2
    assert (
        EffectRow(entries=(), tail=tail)
        .add(RowEntry(instance=left_id, effect=effect))
        .contains(left_id)
    )

    addable = EffectRow(
        entries=(),
        tail=RowVariable(
            name="addable", identity=RowVariableId.derive("tests", "addable")
        ),
    )
    added = addable.add(RowEntry(instance=left_id, effect=effect))
    assert added.tail is not None
    assert added.tail.proves_lacks(left_id)
    assert added.add(RowEntry(instance=left_id, effect=effect)) is added
    conflicting = EffectRef(id=EffectId.derive("example", "Other"), name="Other")
    with pytest.raises(ValueError, match="two interfaces"):
        added.add(RowEntry(instance=left_id, effect=conflicting))

    with pytest.raises(dx.ValidationError, match="cannot contain"):
        EffectRow(
            entries=(
                RowEntry(instance=left_id, effect=effect),
                RowEntry(instance=left_id, effect=effect),
            )
        )

    with pytest.raises(dx.ValidationError, match="must prove"):
        EffectRow(
            entries=(RowEntry(instance=left_id, effect=effect),),
            tail=RowVariable(
                name="open", identity=RowVariableId.derive("tests", "open")
            ),
        )

    shared = RowVariableId.derive("tests", "shared")
    left_open = EffectRow(
        entries=(), tail=RowVariable(name="before", identity=shared, lacks=(left_id,))
    )
    right_open = EffectRow(
        entries=(), tail=RowVariable(name="after", identity=shared, lacks=(right_id,))
    )
    merged = left_open.union(right_open)
    assert merged.tail is not None
    assert merged.tail.proves_lacks(left_id)
    assert merged.tail.proves_lacks(right_id)

    with pytest.raises(ValueError, match="distinct row variables"):
        left_open.union(
            EffectRow(
                entries=(),
                tail=RowVariable(
                    name="other", identity=RowVariableId.derive("tests", "other")
                ),
            )
        )


def test_open_row_unification_produces_stable_capture_free_substitutions() -> None:
    effect = EffectRef(id=EffectId.derive("tests", "Unified"), name="Unified")
    left_instance = EffectInstanceId.derive("tests", "left-entry")
    right_instance = EffectInstanceId.derive("tests", "right-entry")
    left_tail = RowVariable(
        name="rho",
        identity=RowVariableId.derive("tests", "left-tail"),
        lacks=(left_instance,),
    )
    right_tail = RowVariable(
        name="rho",
        identity=RowVariableId.derive("tests", "right-tail"),
        lacks=(right_instance,),
    )
    left = EffectRow(entries=(), tail=left_tail).add(
        RowEntry(instance=left_instance, effect=effect)
    )
    right = EffectRow(entries=(), tail=right_tail).add(
        RowEntry(instance=right_instance, effect=effect)
    )

    solution = unify_effect_rows(left, right)
    assert solution.substitution.apply(left) == solution.row
    assert solution.substitution.apply(right) == solution.row
    assert solution.row.tail is not None
    assert solution.row.tail.identity == RowVariableId.derive(
        "unify",
        *sorted((left_tail.identity, right_tail.identity), key=str),
        *sorted((left_instance, right_instance), key=str),
    )

    closed = EffectRow(entries=(RowEntry(instance=left_instance, effect=effect),))
    open_empty = EffectRow(
        entries=(),
        tail=RowVariable(
            name="tail", identity=RowVariableId.derive("tests", "closed-solution")
        ),
    )
    closed_solution = unify_effect_rows(open_empty, closed)
    assert closed_solution.substitution.apply(open_empty) == closed


def test_stable_references_ignore_diagnostic_names() -> None:
    type_id = TypeId.derive("tests", "semantic-type")
    left_type = TypeConstructorRef(id=type_id, name="Before")
    right_type = TypeConstructorRef(
        id=type_id, name="After", telescope=(TypeBinder(name="ignored"),)
    )
    assert left_type == right_type
    assert hash(left_type) == hash(right_type)

    effect_id = EffectId.derive("tests", "semantic-effect")
    left_effect = EffectRef(id=effect_id, name="Before", arguments=(BOOL,))
    right_effect = EffectRef(id=effect_id, name="After", arguments=(BOOL,))
    assert left_effect == right_effect
    assert hash(left_effect) == hash(right_effect)
    # Identity is the declaration plus its static arguments, so the
    # display name is ignored on both sides while a different argument
    # separates two applications of the same interface.
    assert left_effect != EffectRef(id=effect_id, name="After", arguments=(INT,))


def test_user_index_constructors_validate_membership_and_arity() -> None:
    unary = UserIndexSort(name="Unary", constructors=("zero", "succ"), arities=(0, 1))
    zero = IndexLiteral(value="zero", sort=unary)
    assert IndexConstructor(name="succ", arguments=(zero,), sort=unary).arguments == (
        zero,
    )

    with pytest.raises(dx.ValidationError, match="expects 1 arguments"):
        IndexConstructor(name="succ", arguments=(), sort=unary)
    with pytest.raises(dx.ValidationError, match="not a constructor"):
        IndexConstructor(name="other", arguments=(), sort=unary)
    with pytest.raises(dx.ValidationError, match="not nullary"):
        IndexLiteral(value="succ", sort=unary)


def test_telescope_instantiation_preserves_static_namespaces() -> None:
    effect = EffectRef(id=EffectId.derive("example", "Reader"), name="Reader")
    telescope = (
        TypeBinder(name="a"),
        IndexBinder(name="n", sort=NAT),
        # Effect variables are kinded independently of value types.
        EffectBinder(name="e"),
    )
    substitution = instantiate_telescope(
        telescope,
        (BOOL, IndexLiteral(value=2, sort=NAT), effect),
    )

    assert substitution.type("a") == BOOL
    assert substitution.index("n") == IndexLiteral(value=2, sort=NAT)
    assert substitution.effect("e") == effect
    assert TypeVariable(name="a") != EffectVariable(name="a")

    with pytest.raises(TypeError, match="index argument"):
        instantiate_telescope((IndexBinder(name="n", sort=NAT),), (BOOL,))
