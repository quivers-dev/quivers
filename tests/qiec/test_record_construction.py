"""Kernel records built directly are the records didactic builds.

A record with nothing to encode is built without didactic's field pipeline,
and a decoded field is kept after its first read. Neither may change what a
record is, what it refuses, or what a caller can do with a value read from it.
"""

from __future__ import annotations

import copy
import pickle

import didactic.api as dx
import pytest

from quivers._values import (
    OPAQUE_BY_IDENTITY,
    OPAQUE_BY_VALUE,
    HeldValue,
    ValueModel,
    _direct_plan,
)
from quivers.dsl import loads
from quivers.qiec._models import QiecModel
from quivers.qiec.identifiers import DynamicAddressFrame, SourceOrigin
from quivers.qiec.program_runtime import run_deduction

GRAMMAR = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, A, B, span, a, b
    rule pair : span(I, K, A), span(K, J, B) |- span(I, J, S) #[learnable]
    lexicon
        "a" : A = a
        "b" : B = b
"""


class Pair(QiecModel):
    """Two recursive fields, the second with a default."""

    path: tuple[int, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    held: HeldValue = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)


class Positive(QiecModel):
    """A recursive field a validator constrains."""

    path: tuple[int, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _positive(self) -> None:
        if any(step < 0 for step in self.path):
            raise ValueError("a step is negative")


class Reversed(QiecModel):
    """A validator that stores a canonical form after the field was read."""

    path: tuple[int, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _order(self) -> None:
        hash(self)
        self._normalize_field("path", tuple(sorted(self.path)))


class Encoded(ValueModel):
    """Encoded fields of immutable and of mutable values."""

    name: str
    steps: tuple[int, ...]
    table: dict[str, int]


def _through_didactic(class_: type[QiecModel], fields: dict[str, HeldValue]):
    record = class_.__new__(class_)
    dx.Model.__init__(record, **fields)
    return record


def test_records_built_directly_equal_those_didactic_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every record a run builds, built both ways, agrees field by field."""
    compared: dict[type[QiecModel], int] = {}
    construct = QiecModel.__init__

    def comparing(self: QiecModel, **fields: HeldValue) -> None:
        construct(self, **fields)
        class_ = type(self)
        if _direct_plan(class_) is None or compared.get(class_, 0) >= 50:
            return
        compared[class_] = compared.get(class_, 0) + 1
        expected = _through_didactic(class_, fields)
        assert self._storage == expected._storage
        assert list(self._opaque_storage) == list(expected._opaque_storage)
        for name in class_.__field_specs__:
            left = self._opaque_storage[name]
            right = expected._opaque_storage[name]
            assert left is right or left == right

    program = loads(GRAMMAR)
    monkeypatch.setattr(QiecModel, "__init__", comparing)
    run_deduction(program.qiec, "G", tokens=["a", "b"], record_trace=True)
    assert len(compared) > 10
    assert sum(compared.values()) > 200


@pytest.mark.parametrize(
    ("class_", "fields"),
    [
        (Pair, {}),
        (Pair, {"path": (1,), "other": 2}),
        (Pair, {"path": [1]}),
        (Pair, {"path": (1, "a")}),
        (Pair, {"path": (1, True)}),
        (Positive, {"path": (1, -1)}),
        (Positive, {"path": None}),
    ],
)
def test_a_refused_record_is_refused_as_didactic_refuses_it(
    class_: type[QiecModel], fields: dict[str, HeldValue]
) -> None:
    with pytest.raises(dx.ValidationError) as direct:
        class_(**fields)
    with pytest.raises(dx.ValidationError) as expected:
        _through_didactic(class_, fields)
    assert str(direct.value) == str(expected.value)


def test_which_records_are_built_directly() -> None:
    assert _direct_plan(Pair) is not None
    assert _direct_plan(DynamicAddressFrame) is None
    assert Pair(path=(1, 2)).held is None


def test_a_field_stored_during_validation_is_the_field_later_reads_see() -> None:
    record = Reversed(path=(3, 1, 2))
    assert record.path == (1, 2, 3)
    assert record == Reversed(path=(1, 2, 3))
    assert hash(record) == hash(Reversed(path=(1, 2, 3)))
    assert record.with_(path=(9, 8)).path == (8, 9)


def test_an_immutable_field_is_read_once_and_a_mutable_one_every_time() -> None:
    record = Encoded(name="a", steps=(1, 2), table={"x": 1})
    assert record.steps is record.steps
    record.table["x"] = 5
    assert record.table == {"x": 1}
    assert record == Encoded(name="a", steps=(1, 2), table={"x": 1})


def test_an_update_a_copy_and_a_pickle_do_not_share_what_was_read() -> None:
    record = Encoded(name="a", steps=(1, 2), table={"x": 1})
    assert (record.name, record.steps) == ("a", (1, 2))
    assert record.with_(steps=(3,)).steps == (3,)
    assert record.steps == (1, 2)
    assert copy.copy(record) == record
    assert pickle.loads(pickle.dumps(record)) == record
    with pytest.raises(AttributeError, match="immutable"):
        record.name = "b"
    with pytest.raises(AttributeError, match="has no field"):
        record.missing


def test_frozen_values_keep_their_content() -> None:
    nested = (1, (2, ("a", None)), b"x")
    assert Pair(path=(1,), held=nested).held is nested
    origin = SourceOrigin(
        module="m", structural_path=("a", 1), role="r", source_protocol="p"
    )
    assert origin.site_id() is origin.site_id()
    assert (
        origin.site_id()
        == SourceOrigin(
            module="m", structural_path=("a", 1), role="r", source_protocol="p", line=3
        ).site_id()
    )
    assert origin.site_id() != origin.with_(role="s").site_id()


class _Path(tuple):
    """A tuple subclass, which a value field replaces by a plain tuple."""


def test_a_tuple_subclass_is_stored_as_a_tuple() -> None:
    assert type(Pair(path=_Path((1, 2))).path) is tuple
