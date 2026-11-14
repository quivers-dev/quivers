"""Canonical wire codec for stable QIEC graphs.

The codec is an explicit allowlist, not a generic Python object serializer.
Only frozen kernel records, concrete identifier classes, kernel enums, tuples,
bytes, and JSON scalars cross the boundary.  Runtime attachments, handler
callbacks, modules, mappings, and unknown future node tags are rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import json
import math
from typing import cast

from quivers.qiec.coverage import (
    BranchPattern,
    BranchRefinement,
    CoverageResult,
    CoverageStatus,
    Reachability,
)
from quivers.qiec._models import QiecModel, field_types, validate_shape
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl, FieldDef
from quivers.qiec.effects import (
    ArgumentDef,
    ComputationType,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerReturnClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    RowEntry,
    RowSubstitution,
    RowUnification,
    RowVariable,
)
from quivers.qiec.evidence import BranchGiven, Reflexivity
from quivers.qiec.identifiers import (
    AttachmentId,
    ConstructorId,
    DynamicAddressFrame,
    EffectId,
    EffectInstanceId,
    EqualityId,
    FamilyId,
    ComputationId,
    HandlerId,
    OperationId,
    QIEC_ABI,
    PrimitiveId,
    DistributionId,
    RowVariableId,
    SiteId,
    SiteProvenance,
    SourceOrigin,
    StableId,
    StaticScopeId,
    StaticVariableId,
    TypeId,
)
from quivers.qiec.kinds import (
    ArrowKind,
    ContextSort,
    EffectBinder,
    EffectKind,
    IndexBinder,
    NatSort,
    RowKind,
    ShapeSort,
    TypeBinder,
    TypeKind,
    UserIndexSort,
)
from quivers.qiec.module import NamedComputation, NamedEffectInstance, QiecModule
from quivers.qiec.programs import ProgramEntry, ProgramParameter, ProgramSite
from quivers.qiec.substitution import StaticSubstitution
from quivers.qiec.terms import (
    NewInstance,
    Resume,
    AttachmentRef,
    Bind,
    Call,
    Case,
    CaseBranch,
    CaseMotive,
    ConstructorValue,
    DistributionValue,
    EvidenceValue,
    Handle,
    If,
    LiteralValue,
    Local,
    LogDensity,
    Perform,
    PrimitiveApplication,
    Projection,
    Return,
    SiteValue,
    TensorValue,
    TransportValue,
    TupleValue,
    PlateAxis,
    PlateShape,
    Gather,
    WeightSum,
    SegmentSum,
    KernelMatrix,
    AffineMap,
    TableMap,
    Reduction,
    Rowwise,
    Comprehension,
    LetBinding,
    LetValue,
    Var,
)
from quivers.qiec.types import (
    EffectRef,
    EffectVariable,
    EqualityType,
    FunctionType,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    ShapeIndex,
    TypeApplication,
    TypeConstructorRef,
    TypeVariable,
)


QIEC_WIRE_FORMAT = "qiec-json/v1"
"""The format tag every encoded document carries, checked on decoding."""

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
"""A JSON document as ``json.loads`` returns it and ``json.dumps`` accepts it."""


class SerializationError(ValueError):
    """A value is not in the stable QIEC wire language."""


def _class_tag(class_: type[object]) -> str:
    """The wire tag naming one allowlisted class.

    Parameters
    ----------
    class_ : type
        The class to name.

    Returns
    -------
    str
        ``<module>.<qualname>`` with the package prefix trimmed, which is
        what a payload carries in its ``$type`` field.
    """
    module = class_.__module__.removeprefix("quivers.qiec.")
    return f"{module}.{class_.__qualname__}"


_ID_CLASSES = (
    TypeId,
    ComputationId,
    FamilyId,
    ConstructorId,
    EffectId,
    OperationId,
    EffectInstanceId,
    HandlerId,
    AttachmentId,
    SiteId,
    EqualityId,
    StaticScopeId,
    StaticVariableId,
    RowVariableId,
    PrimitiveId,
    DistributionId,
)

_ENUM_CLASSES = (
    ResumptionGrade,
    CoverageStatus,
    Reachability,
)

_NODE_CLASSES = (
    # source identity and kinds
    SourceOrigin,
    DynamicAddressFrame,
    SiteProvenance,
    TypeKind,
    EffectKind,
    RowKind,
    ArrowKind,
    NatSort,
    ShapeSort,
    ContextSort,
    UserIndexSort,
    TypeBinder,
    IndexBinder,
    EffectBinder,
    # static terms
    IndexVariable,
    IndexLiteral,
    IndexConstructor,
    ShapeIndex,
    TypeVariable,
    TypeConstructorRef,
    TypeApplication,
    FunctionType,
    EqualityType,
    EffectVariable,
    EffectRef,
    # declarations, rows, and signatures
    FieldDef,
    FamilyDecl,
    ConstructorDecl,
    RowVariable,
    RowEntry,
    EffectRow,
    RowSubstitution,
    RowUnification,
    ComputationType,
    ArgumentDef,
    OperationDef,
    EffectDef,
    HandlerClauseDef,
    HandlerReturnClauseDef,
    HandlerDef,
    EffectRequest,
    NamedEffectInstance,
    NamedComputation,
    QiecModule,
    ProgramEntry,
    ProgramParameter,
    ProgramSite,
    StaticSubstitution,
    # equality and coverage results
    Reflexivity,
    BranchGiven,
    BranchPattern,
    BranchRefinement,
    CoverageResult,
    # stable value/computation graph
    Local,
    Var,
    LiteralValue,
    ConstructorValue,
    EvidenceValue,
    AttachmentRef,
    TransportValue,
    PrimitiveApplication,
    TupleValue,
    TensorValue,
    Projection,
    PlateAxis,
    PlateShape,
    DistributionValue,
    Gather,
    WeightSum,
    SegmentSum,
    KernelMatrix,
    AffineMap,
    TableMap,
    Reduction,
    Rowwise,
    Comprehension,
    LetBinding,
    LetValue,
    LogDensity,
    SiteValue,
    Return,
    Bind,
    Perform,
    Handle,
    CaseMotive,
    CaseBranch,
    Case,
    If,
    Call,
    Resume,
    NewInstance,
)

_IDS = {_class_tag(class_): class_ for class_ in _ID_CLASSES}
_ENUMS = {_class_tag(class_): class_ for class_ in _ENUM_CLASSES}
_NODES: dict[str, type[QiecModel]] = {
    _class_tag(class_): class_ for class_ in _NODE_CLASSES
}


def _encode(value: object) -> JsonValue:
    """Encode one stable QIEC value.

    Parameters
    ----------
    value : object
        The value to encode. Only allowlisted identifiers, enums, and
        records may cross the boundary, along with the JSON scalars and
        the tuples that hold them.

    Returns
    -------
    JsonValue
        The encoded value.

    Raises
    ------
    SerializationError
        If the value is of a class not on the allowlist, or is a
        non-finite float. Refusing an unknown class is the point of the
        boundary: a Python callable or a mutable resource must not be
        able to ride across it.
    """
    if isinstance(value, StableId):
        tag = _class_tag(type(value))
        if tag not in _IDS:
            raise SerializationError(f"unregistered QIEC identifier class {tag!r}")
        return {"$id": tag, "value": value.to_data()}
    if isinstance(value, Enum):
        tag = _class_tag(type(value))
        if tag not in _ENUMS:
            raise SerializationError(f"unregistered QIEC enum class {tag!r}")
        return {"$enum": tag, "value": cast(str, value.value)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("non-finite floats have no canonical JSON form")
        return value
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if callable(value):
        raise SerializationError("host callables cannot enter stable QIEC")
    if isinstance(value, QiecModel):
        tag = _class_tag(type(value))
        if tag not in _NODES:
            raise SerializationError(
                f"runtime or unregistered model cannot enter stable QIEC: {tag!r}"
            )
        return {
            "$type": tag,
            "fields": [
                {"name": name, "value": _encode(getattr(value, name))}
                for name in type(value).__field_specs__
            ],
        }
    raise SerializationError(
        f"runtime or unregistered value of type {type(value).__module__}.{type(value).__qualname__} "
        "cannot enter stable QIEC"
    )


def _exact_keys(value: dict[str, JsonValue], expected: set[str], subject: str) -> None:
    """Require an object to carry exactly the expected keys.

    Parameters
    ----------
    value : dict[str, JsonValue]
        The decoded object.
    expected : set[str]
        The keys it must have, no more and no fewer.
    subject : str
        What is being checked, named in the diagnostic.

    Raises
    ------
    SerializationError
        If a key is missing or extra. Extra keys are rejected rather than
        ignored, because a payload from a later version carrying a field
        this one does not understand would otherwise decode into a record
        missing information the sender considered essential.
    """
    if set(value) != expected:
        raise SerializationError(
            f"malformed {subject}: expected keys {sorted(expected)!r}, "
            f"got {sorted(value)!r}"
        )


def _decode(value: JsonValue) -> object:
    """Decode one value from the wire form.

    Parameters
    ----------
    value : JsonValue
        The decoded JSON to interpret.

    Returns
    -------
    object
        The reconstructed value.

    Raises
    ------
    SerializationError
        If a tag is unknown, a record's keys do not match its fields, a
        field's contents do not inhabit its declared type, an identifier
        is malformed, or a float is non-finite.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("non-finite floats have no canonical JSON form")
        return value
    if isinstance(value, list):
        raise SerializationError("bare JSON arrays are not QIEC tuples")
    if not isinstance(value, dict):  # pragma: no cover - closed JsonValue union
        raise SerializationError("invalid QIEC JSON value")

    if "$bytes" in value:
        _exact_keys(value, {"$bytes"}, "bytes value")
        encoded = value["$bytes"]
        if not isinstance(encoded, str):
            raise SerializationError("encoded bytes must be a hexadecimal string")
        try:
            return bytes.fromhex(encoded)
        except ValueError as error:
            raise SerializationError("encoded bytes are not hexadecimal") from error

    if "$tuple" in value:
        _exact_keys(value, {"$tuple"}, "tuple")
        items = value["$tuple"]
        if not isinstance(items, list):
            raise SerializationError("$tuple must contain an array")
        return tuple(_decode(item) for item in items)

    if "$id" in value:
        _exact_keys(value, {"$id", "value"}, "identifier")
        tag = value["$id"]
        encoded = value["value"]
        if not isinstance(tag, str) or tag not in _IDS:
            raise SerializationError(f"unknown QIEC identifier tag {tag!r}")
        if not isinstance(encoded, str):
            raise SerializationError("identifier value must be a string")
        try:
            return _IDS[tag].parse(encoded)
        except ValueError as error:
            raise SerializationError(str(error)) from error

    if "$enum" in value:
        _exact_keys(value, {"$enum", "value"}, "enum")
        tag = value["$enum"]
        encoded = value["value"]
        if not isinstance(tag, str) or tag not in _ENUMS:
            raise SerializationError(f"unknown QIEC enum tag {tag!r}")
        try:
            return _ENUMS[tag](encoded)
        except (TypeError, ValueError) as error:
            raise SerializationError(
                f"invalid value {encoded!r} for QIEC enum {tag!r}"
            ) from error

    if "$type" in value:
        _exact_keys(value, {"$type", "fields"}, "kernel node")
        tag = value["$type"]
        encoded_fields = value["fields"]
        if not isinstance(tag, str) or tag not in _NODES:
            raise SerializationError(f"unknown QIEC node tag {tag!r}")
        if not isinstance(encoded_fields, list):
            raise SerializationError("kernel node fields must be an array")
        class_ = _NODES[tag]
        class_fields = class_.__field_specs__
        expected_names = list(class_fields)
        actual_names: list[str] = []
        arguments: dict[str, object] = {}
        for encoded_field in encoded_fields:
            if not isinstance(encoded_field, dict):
                raise SerializationError("kernel field entry must be an object")
            _exact_keys(encoded_field, {"name", "value"}, "kernel field")
            name = encoded_field["name"]
            if not isinstance(name, str):
                raise SerializationError("kernel field name must be a string")
            if name in arguments:
                raise SerializationError(f"duplicate kernel field {name!r}")
            actual_names.append(name)
            arguments[name] = _decode(encoded_field["value"])
        if actual_names != expected_names:
            raise SerializationError(
                f"fields for {tag!r} must be {expected_names!r}, got {actual_names!r}"
            )
        tag_field = class_fields.get("tag")
        if tag_field is not None and not tag_field.is_required:
            actual_tag = arguments["tag"]
            expected_tag = tag_field.make_default()
            if actual_tag != expected_tag:
                raise SerializationError(
                    f"invalid literal tag for {tag!r}: expected "
                    f"{expected_tag!r}, got {actual_tag!r}"
                )
        annotations = field_types(class_)
        for name, argument in arguments.items():
            try:
                validate_shape(argument, annotations[name], path=f"{tag}.{name}")
            except TypeError as error:
                raise SerializationError(str(error)) from error
        try:
            factory = cast(Callable[..., object], class_)
            return factory(**arguments)
        except Exception as error:
            raise SerializationError(f"invalid {tag!r} node: {error}") from error

    raise SerializationError("unknown object in QIEC wire data")


def to_data(value: object) -> dict[str, JsonValue]:
    """Encode a stable QIEC graph in its versioned wire envelope.

    Parameters
    ----------
    value : object
        The graph to encode.

    Returns
    -------
    dict[str, JsonValue]
        The envelope, carrying the wire format and the ABI alongside the
        graph. Both are recorded because neither can be guessed from the
        payload, and a reader that guessed wrong would misread it.

    Raises
    ------
    SerializationError
        If the graph contains a value that may not cross the boundary.
    """
    return {
        "$schema": QIEC_WIRE_FORMAT,
        "abi": QIEC_ABI,
        "root": _encode(value),
    }


def from_data(data: object) -> object:
    """Decode a QIEC graph, rejecting other ABIs and unknown node tags.

    Parameters
    ----------
    data : object
        The decoded envelope.

    Returns
    -------
    object
        The reconstructed graph.

    Raises
    ------
    SerializationError
        If the envelope is malformed, declares another wire format or
        ABI, or holds a graph that fails to decode. An ABI mismatch is
        refused rather than attempted, since a payload from another
        version may use the same tags for different shapes.
    """
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise SerializationError("QIEC wire envelope must be an object")
    typed = cast(dict[str, JsonValue], data)
    _exact_keys(typed, {"$schema", "abi", "root"}, "QIEC wire envelope")
    if typed["$schema"] != QIEC_WIRE_FORMAT:
        raise SerializationError(f"unsupported QIEC wire format {typed['$schema']!r}")
    if typed["abi"] != QIEC_ABI:
        raise SerializationError(
            f"QIEC ABI mismatch: expected {QIEC_ABI!r}, got {typed['abi']!r}"
        )
    return _decode(typed["root"])


def dumps(value: object) -> str:
    """Serialize a QIEC graph as deterministic UTF-8-compatible JSON text.

    Parameters
    ----------
    value : object
        The graph to serialize.

    Returns
    -------
    str
        Compact, key-sorted JSON. Determinism matters because identities
        downstream are content addressed: the same graph must produce the
        same bytes on every run and every platform.

    Raises
    ------
    SerializationError
        If the graph contains a value that may not cross the boundary.
    """
    return json.dumps(
        to_data(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def loads(text: str | bytes | bytearray) -> object:
    """Deserialize one canonical QIEC JSON document.

    Parameters
    ----------
    text : str or bytes or bytearray
        The document to read.

    Returns
    -------
    object
        The reconstructed graph.

    Raises
    ------
    SerializationError
        If the text is not valid JSON, repeats an object key at any
        depth, or fails any of the decoding checks.
    """
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SerializationError(f"invalid QIEC JSON: {error}") from error
    return from_data(data)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Build a JSON object while rejecting duplicate keys at every depth.

    Parameters
    ----------
    pairs : list[tuple[str, JsonValue]]
        The key and value pairs as the parser found them, in order.

    Returns
    -------
    dict[str, JsonValue]
        The object.

    Raises
    ------
    SerializationError
        If a key repeats. The JSON parser would otherwise keep the last
        occurrence silently, so a payload could carry two values for one
        field and decode to whichever came second.
    """

    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


__all__ = [
    "JsonValue",
    "QIEC_WIRE_FORMAT",
    "SerializationError",
    "dumps",
    "from_data",
    "loads",
    "to_data",
]
