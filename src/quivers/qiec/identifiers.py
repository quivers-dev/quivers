"""Stable identifiers and source provenance for the QIEC kernel.

Identifiers in the kernel are content-derived references, not process-local
object identities.  This keeps the serialized core independent of Python
objects and gives migrations a stable target when source locations move.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import total_ordering
from hashlib import sha256
from typing import Literal, Self

import didactic.api as dx

QIEC_ABI = "qiec-core/v1alpha1"
"""The kernel ABI tag a module records, so a reader can reject a module built against another kernel."""


def _canonical_value(value: object) -> object:
    """Reduce a value to the JSON shape identifier derivation hashes.

    Parameters
    ----------
    value : object
        The value to encode. Scalars pass through, bytes become a tagged
        hex string, identifiers and enums become their wire forms,
        sequences and mappings recurse, and a dataclass becomes its
        fields under a tag naming its type.

    Returns
    -------
    object
        A JSON-encodable value. Mappings are key-sorted and dataclasses
        carry their qualified type name, so structurally distinct values
        cannot collide once encoded.

    Raises
    ------
    TypeError
        If the value is of a class with no canonical encoding. This is
        deliberate: silently stringifying an unknown value would make two
        different values hash alike and give them one identity.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"$bytes": value.hex()}
    if isinstance(value, StableId):
        return value.to_data()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            **{
                field.name: _canonical_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    raise TypeError(f"value has no canonical QIEC encoding: {value!r}")


def _canonical_bytes(parts: tuple[object, ...]) -> bytes:
    """Encode identifier inputs in the canonical QIEC representation.

    Parameters
    ----------
    parts : tuple[object, ...]
        The values contributing to an identifier, in order.

    Returns
    -------
    bytes
        Their UTF-8 JSON encoding, compact and key-sorted so the same
        inputs produce the same bytes on every run and every platform.

    Raises
    ------
    TypeError
        If any part has no canonical encoding.
    """
    return json.dumps(
        _canonical_value(parts),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@total_ordering
class StableId(dx.Model):
    """A namespaced, SHA-256-derived kernel identifier.

    The human-readable declaration name is intentionally not stored in the
    identifier.  Renaming display labels must not silently alter equality;
    callers choose the declaration path supplied to :meth:`derive` and record
    renames explicitly in a migration map.

    Parameters
    ----------
    digest
        The 64-character lowercase hex SHA-256 digest; usually produced by
        :meth:`derive` rather than written by hand.

    Raises
    ------
    didactic.api.ValidationError
        If the digest is not 64 lowercase hexadecimal characters.
    """

    digest: str
    namespace = "id"

    @dx.model_validator()
    def _validate_digest(self) -> None:
        """Reject a digest that is not 64 lowercase hex characters.

        Raises
        ------
        ValueError
            If the digest is the wrong length or contains a character
            outside ``0-9a-f``.
        """
        if len(self.digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.digest
        ):
            raise ValueError(f"invalid {self.namespace} digest: {self.digest!r}")

    @classmethod
    def derive(cls, *parts: object) -> Self:
        """Derive an identifier from the values that determine it.

        Parameters
        ----------
        parts : object
            The determining values, in order. Order matters, so two
            identifiers built from the same values differently arranged
            are distinct.

        Returns
        -------
        Self
            The derived identifier. The ABI version and the namespace are
            mixed in, so an identifier from one namespace can never equal
            one from another, and a future ABI change re-derives rather
            than silently reusing.

        Raises
        ------
        TypeError
            If a part has no canonical encoding.
        """
        payload = (QIEC_ABI, cls.namespace, *parts)
        return cls(digest=sha256(_canonical_bytes(payload)).hexdigest())

    @classmethod
    def parse(cls, text: str) -> Self:
        """Read an identifier back from its wire form.

        Parameters
        ----------
        text : str
            The wire form, ``qiec:<namespace>:<digest>``.

        Returns
        -------
        Self
            The parsed identifier.

        Raises
        ------
        ValueError
            If the text carries another namespace's prefix, or the
            remainder is not a valid digest. Parsing is namespace-checked
            so a handler identifier cannot be read as an effect one.
        """
        prefix = f"qiec:{cls.namespace}:"
        if not text.startswith(prefix):
            raise ValueError(f"expected {prefix!r} identifier, got {text!r}")
        try:
            return cls(digest=text.removeprefix(prefix))
        except dx.ValidationError as error:
            raise ValueError(str(error)) from error

    def __lt__(self, other: StableId) -> bool:
        """Order identifiers by digest within the same concrete namespace.

        Parameters
        ----------
        other : StableId
            The identifier to compare.

        Returns
        -------
        bool
            Whether this digest precedes the other, or NotImplemented
            for identifiers of different classes.
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.digest < other.digest

    def __str__(self) -> str:
        """The wire form of this identifier.

        Returns
        -------
        str
            ``qiec:<namespace>:<digest>``, which `parse` accepts.
        """
        return f"qiec:{self.namespace}:{self.digest}"

    def to_data(self) -> str:
        """Return the stable wire representation.

        Returns
        -------
        str
            The same string as `__str__`, named for the serialization
            protocol the rest of the kernel uses.
        """
        return str(self)


class TypeId(StableId):
    """Stable identity of a type."""

    namespace = "type"


class FamilyId(StableId):
    """Stable identity of an indexed type family."""

    namespace = "family"


class ConstructorId(StableId):
    """Stable identity of a family constructor."""

    namespace = "constructor"


class EffectId(StableId):
    """Stable identity of an effect interface."""

    namespace = "effect"


class OperationId(StableId):
    """Stable identity of an effect operation."""

    namespace = "operation"


class EffectInstanceId(StableId):
    """Stable identity of an effect instance."""

    namespace = "effect-instance"


class ComputationId(StableId):
    """Stable identity of a named computation."""

    namespace = "computation"


class HandlerId(StableId):
    """Stable identity of a handler."""

    namespace = "handler"


class AttachmentId(StableId):
    """Stable identity of a runtime attachment."""

    namespace = "attachment"


class SiteId(StableId):
    """Stable identity of a source site."""

    namespace = "site"


class EqualityId(StableId):
    """Stable identity of an equality proposition."""

    namespace = "equality"


class StaticScopeId(StableId):
    """Stable identity of a static scope."""

    namespace = "static-scope"


class StaticVariableId(StableId):
    """Stable identity of a static variable."""

    namespace = "static-variable"


class RowVariableId(StableId):
    """Stable identity of a row variable."""

    namespace = "row-variable"


class PrimitiveId(StableId):
    """Stable identity of a pure primitive."""

    namespace = "primitive"


class DistributionId(StableId):
    """Stable identity of a distribution family."""

    namespace = "distribution"


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    """Stable structural source origin with optional diagnostic coordinates.

    ``structural_path`` is the protocol path after parsing and name
    resolution.  Line and column are diagnostic metadata and therefore do not
    participate in :meth:`site_id`.

    Parameters
    ----------
    module
        The module the site belongs to.
    structural_path
        The path of names and positions from the module root to the site.
    role
        What the site is, such as ``"sample"`` or ``"call"``.
    source_protocol
        The source language or protocol the path is expressed in.
    file
        The file the site was read from, if any.
    line
        The one-based line of the site, if known.
    column
        The one-based column of the site, if known.
    """

    module: str
    structural_path: tuple[str | int, ...]
    role: str
    source_protocol: str
    file: str | None = None
    line: int | None = None
    column: int | None = None

    def site_id(self) -> SiteId:
        """The stable identity of the source site this origin describes.

        Returns
        -------
        SiteId
            An identity derived from the protocol, module, structural
            path, and role. The file, line, and column are deliberately
            excluded, so reformatting a source file does not change the
            identity of the sites within it.
        """
        return SiteId.derive(
            self.source_protocol,
            self.module,
            self.structural_path,
            self.role,
        )

    def to_data(self) -> dict[str, object]:
        """Return the serializable form of this origin.

        Returns
        -------
        dict[str, object]
            Every field, including the diagnostic coordinates. Those do
            not enter `site_id` but are carried here so a deserialized
            origin can still point at a line.
        """
        return {
            "module": self.module,
            "structural_path": list(self.structural_path),
            "role": self.role,
            "source_protocol": self.source_protocol,
            "file": self.file,
            "line": self.line,
            "column": self.column,
        }


type AddressRelation = Literal[
    "preserve",
    "split",
    "duplicate",
    "eliminate",
]
"""How an effect request relates to its source site.

``"preserve"`` marks the request as written; ``"split"`` and
``"duplicate"`` mark a request a handler derived from it; ``"eliminate"``
marks one a handler discharged.
"""


@dataclass(frozen=True, slots=True)
class DynamicAddressFrame:
    """One serializable component of a dynamic effect address.

    Parameters
    ----------
    scope
        What kind of frame this is, such as a call, a handler clause, a
        resumption, or a local instance.
    key
        The frame's position within its scope: a name or an ordinal.
    """

    scope: str
    key: str | int


@dataclass(frozen=True, slots=True)
class SiteProvenance:
    """Static and dynamic provenance of an effect request.

    Parameters
    ----------
    origin
        The source site the request was written at.
    dynamic_path
        The address frames entered to reach this occurrence of the site,
        outermost first.
    resumption_path
        The ordinal of each resumption taken to reach this occurrence, so a
        site reached twice under a multi-shot handler yields two keys.
    relation
        How this request relates to the source site: ``"preserve"`` for the
        request as written, ``"split"`` or ``"duplicate"`` for a request a
        handler derived from it, and ``"eliminate"`` for one it discharged.
    parents
        The static sites a derived request was generated from.
    """

    origin: SourceOrigin
    dynamic_path: tuple[DynamicAddressFrame, ...] = ()
    resumption_path: tuple[int, ...] = ()
    relation: AddressRelation = "preserve"
    parents: tuple[SiteId, ...] = ()

    @property
    def static_site(self) -> SiteId:
        """The source site this request was written at.

        Returns
        -------
        SiteId
            The static identity, shared by every dynamic occurrence of
            the request.
        """
        return self.origin.site_id()

    def dynamic_key(
        self,
    ) -> tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]:
        """A hashable key distinguishing occurrences of one static site.

        Returns
        -------
        tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]
            The static site, the dynamic address frames, and the
            resumption path. One site reached twice, under a loop or
            under a resumed continuation, yields two keys, which is what
            lets a trace name each occurrence separately.
        """
        return (
            str(self.static_site),
            tuple((frame.scope, frame.key) for frame in self.dynamic_path),
            self.resumption_path,
        )

    def to_data(self) -> dict[str, object]:
        """Return the serializable form of this provenance.

        Returns
        -------
        dict[str, object]
            The origin, the static site, and the dynamic address. The
            static site is written out alongside the origin it derives
            from so a reader need not re-derive it.
        """
        return {
            "origin": self.origin.to_data(),
            "static_site": self.static_site.to_data(),
            "dynamic_path": [
                {"scope": frame.scope, "key": frame.key} for frame in self.dynamic_path
            ],
            "resumption_path": list(self.resumption_path),
            "relation": self.relation,
            "parents": [parent.to_data() for parent in self.parents],
        }


__all__ = [
    "AddressRelation",
    "AttachmentId",
    "ComputationId",
    "ConstructorId",
    "DistributionId",
    "DynamicAddressFrame",
    "EffectId",
    "EffectInstanceId",
    "EqualityId",
    "FamilyId",
    "HandlerId",
    "OperationId",
    "PrimitiveId",
    "QIEC_ABI",
    "RowVariableId",
    "SiteId",
    "SiteProvenance",
    "SourceOrigin",
    "StableId",
    "StaticScopeId",
    "StaticVariableId",
    "TypeId",
]
