"""Recursive model identity, canonical normalization, and closed wire graphs."""

import copy

import didactic.api as dx
import pytest

from quivers.qiec import (
    TYPE,
    REAL,
    CheckContext,
    KernelRegistry,
    Local,
    EffectId,
    EffectInstanceId,
    EffectRef,
    EffectRow,
    RowEntry,
    RowVariable,
    RowVariableId,
    StaticVariableId,
    TypeApplication,
    TypeConstructorRef,
    TypeId,
    TypeVariable,
    UserIndexSort,
)
from quivers.qiec._models import QiecModel
from quivers.qiec.serialization import SerializationError, dumps, loads


def test_nominal_identity_survives_recursive_models_and_display_renames() -> None:
    """Nested equality uses rigid identities and rejects a distinct skolem."""
    identity = StaticVariableId.derive("model-test", "a")
    first = TypeVariable(name="a", kind=TYPE, identity=identity)
    renamed = first.with_(name="renamed")
    constructor = TypeConstructorRef(id=TypeId.derive("model-test", "Box"), name="Box")
    left = TypeApplication(constructor=constructor, arguments=(first,))
    right = TypeApplication(
        constructor=constructor.with_(name="DisplayBox"), arguments=(renamed,)
    )
    assert left == right
    assert hash(left) == hash(right)
    assert copy.copy(left) == left
    assert loads(dumps(left)) == left
    different = first.with_(identity=StaticVariableId.derive("model-test", "b"))
    assert left != left.with_(arguments=(different,))
    assert first != first.with_(identity=None)
    effect = EffectRef(
        id=EffectId.derive("model-test", "E"), name="E", arguments=(left,)
    )
    instance = EffectInstanceId.derive("model-test", "instance")
    row = EffectRow(entries=(RowEntry(instance=instance, effect=effect),))
    renamed_row = row.with_(
        entries=(
            RowEntry(
                instance=instance,
                effect=effect.with_(name="DisplayE", arguments=(right,)),
            ),
        )
    )
    assert row == renamed_row
    assert hash(row) == hash(renamed_row)


def test_row_and_index_sort_updates_apply_canonical_normalization() -> None:
    """Construction and updates sort row entries and infer nullary arities."""
    effect = EffectRef(id=EffectId.derive("model-test", "State"), name="State")
    instances = tuple(
        EffectInstanceId.derive("model-test", name) for name in ("left", "right")
    )
    tail = RowVariable(
        name="rho", identity=RowVariableId.derive("model-test", "rho"), lacks=instances
    )
    row = EffectRow(
        entries=tuple(RowEntry(instance=item, effect=effect) for item in instances),
        tail=tail,
    )
    reversed_row = row.with_(entries=tuple(reversed(row.entries)))
    assert row == reversed_row
    assert tuple(str(entry.instance) for entry in row.entries) == tuple(
        sorted(map(str, instances))
    )
    renamed_tail = tail.with_(name="other", lacks=tuple(reversed(instances)))
    assert row == row.with_(tail=renamed_tail)
    assert hash(row) == hash(row.with_(tail=renamed_tail))
    sort = UserIndexSort(name="Flag", constructors=("Off", "On"))
    assert sort.arities == (0, 0)
    assert sort.with_(arities=()).arities == (0, 0)
    with pytest.raises(dx.ValidationError, match="cannot contain"):
        row.with_(entries=(row.entries[0], row.entries[0]))
    with pytest.raises(dx.ValidationError, match="align"):
        sort.with_(arities=(0,))


def test_recursive_field_updates_reject_wrong_kernel_types() -> None:
    """Opaque recursive storage enforces its declared types on every update."""
    constructor = TypeConstructorRef(id=TypeId.derive("model-test", "T"), name="T")
    application = TypeApplication(constructor=constructor)
    with pytest.raises(dx.ValidationError, match="TypeConstructorRef"):
        application.with_(constructor="T")
    with pytest.raises(
        dx.ValidationError, match=r"TypeApplication.arguments\[0\].*got int"
    ):
        application.with_(arguments=(3,))


class _Unregistered(QiecModel):
    """A model outside the closed QIEC wire allowlist."""

    value: int


def test_kernel_model_inheritance_does_not_open_the_wire_allowlist() -> None:
    """A well-typed model still needs an explicit stable wire registration."""
    with pytest.raises(SerializationError, match="unregistered model"):
        dumps(_Unregistered(value=3))


def test_checker_context_updates_preserve_typed_bindings() -> None:
    """Context extension retains values and validates each recursive field."""
    local = Local(name="x", type=REAL)
    empty = CheckContext()
    context = empty.extend(local)
    assert empty.locals == ()
    assert context.locals == (local,)
    assert context == context.with_(locals=(Local(name="x", type=REAL),))
    assert hash(context) == hash(copy.copy(context))
    with pytest.raises(dx.ValidationError, match="locals"):
        context.with_(locals=("x",))


def test_kernel_registry_retains_live_tables_and_validates_replacements() -> None:
    """Registry tables stay independent and mutable with stable identity keys."""
    registry = KernelRegistry()
    other = KernelRegistry()
    copied = copy.copy(registry)
    assert registry != other
    assert registry == copied
    assert copied.type_constructors is registry.type_constructors
    before = hash(registry)
    constructor = TypeConstructorRef(id=TypeId.derive("models", "T"), name="T")
    registry.type_constructors[constructor.id] = constructor
    assert copied.type_constructors[constructor.id] is constructor
    assert other.type_constructors == {}
    assert hash(registry) == before
    with pytest.raises(dx.ValidationError, match="type_constructors.key"):
        registry.with_(type_constructors={"T": constructor})
    with pytest.raises(dx.ValidationError, match="TypeConstructorRef"):
        registry.with_(type_constructors={constructor.id: "T"})
