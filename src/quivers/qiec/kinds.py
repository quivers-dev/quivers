"""Kinds, closed index sorts, and declaration telescopes for QIEC."""

from __future__ import annotations

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
import didactic.api as dx

from typing import Literal


class TypeKind(QiecModel):
    """The kind of types.

    Parameters
    ----------
    tag
        The serialization discriminator; always ``"type"``.
    """

    tag: Literal["type"] = "type"


class EffectKind(QiecModel):
    """The kind of effect interfaces.

    Parameters
    ----------
    tag
        The serialization discriminator; always ``"effect"``.
    """

    tag: Literal["effect"] = "effect"


class RowKind(QiecModel):
    """The kind of effect rows.

    Parameters
    ----------
    tag
        The serialization discriminator; always ``"row"``.
    """

    tag: Literal["row"] = "row"


class ArrowKind(QiecModel):
    """The kind of a constructor from one kind to another.

    Parameters
    ----------
    domain
        The kind of the argument.
    codomain
        The kind of the result.
    tag
        The serialization discriminator; always ``"arrow"``.
    """

    domain: Kind = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    codomain: Kind = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["arrow"] = "arrow"


type Kind = TypeKind | EffectKind | RowKind | ArrowKind
"""The kind of a static binder: a type, an effect, a row, or an arrow between kinds."""

TYPE = TypeKind()
"""The kind of types."""
EFFECT = EffectKind()
"""The kind of effect interface applications."""
ROW = RowKind()
"""The kind of effect rows."""


class NatSort(QiecModel):
    """The sort of natural-number indices.

    Parameters
    ----------
    tag
        The serialization discriminator; always ``"nat"``.
    """

    tag: Literal["nat"] = "nat"


class ShapeSort(QiecModel):
    """The sort of shapes, optionally restricted to a fixed rank.

    Parameters
    ----------
    rank : int or None
        The number of dimensions, or None for any rank. A binder of
        unspecified rank accepts a shape of any rank, which is what makes
        a rank-polymorphic signature expressible.
    tag : Literal["shape"]
        Discriminator for the index-sort union.
    """

    rank: int | None = None
    tag: Literal["shape"] = "shape"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Reject a negative rank.

        Raises
        ------
        ValueError
            If `rank` is negative.
        """
        if self.rank is not None and self.rank < 0:
            raise ValueError("shape rank cannot be negative")


class ContextSort(QiecModel):
    """A logic-program context indexed by a stable signature name.

    Parameters
    ----------
    signature
        The name of the signature whose contexts inhabit this sort.
    tag
        The serialization discriminator; always ``"context"``.
    """

    signature: str
    tag: Literal["context"] = "context"


class UserIndexSort(QiecModel):
    """A closed user-defined index datatype.

    Constructor names are part of the sort identity.  Open index sorts are
    deliberately absent from the first kernel because they make coverage
    unstable under separate compilation.

    Parameters
    ----------
    name : str
        The sort's name, part of its identity.
    constructors : tuple[str, ...]
        Constructor names, unique and part of the identity. Two sorts
        with the same name and different constructors are different
        sorts, which is what keeps coverage decidable.
    arities : tuple[int, ...]
        Argument count per constructor, positionally aligned with
        `constructors`. Empty defaults to all-nullary.
    tag : Literal["user"]
        Discriminator for the index-sort union.
    """

    name: str
    constructors: tuple[str, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arities: tuple[int, ...] = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["user"] = "user"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Default the arities and reject a malformed sort.

        Raises
        ------
        ValueError
            If the name is empty, there are no constructors, a
            constructor name repeats, the arities do not align with the
            constructors, or an arity is negative.
        """
        if not self.name:
            raise ValueError("user index sort name cannot be empty")
        if not self.constructors:
            raise ValueError("a closed user index sort needs a constructor")
        if len(set(self.constructors)) != len(self.constructors):
            raise ValueError(f"duplicate constructors in index sort {self.name!r}")
        if not self.arities:
            self._normalize_field("arities", (0,) * len(self.constructors))
        if len(self.arities) != len(self.constructors):
            raise ValueError("index constructor arities must align with constructors")
        if any(arity < 0 for arity in self.arities):
            raise ValueError("index constructor arities cannot be negative")

    def constructor_arity(self, name: str) -> int:
        """The number of arguments one constructor takes.

        Parameters
        ----------
        name : str
            The constructor name.

        Returns
        -------
        int
            Its arity, zero for a nullary constructor.

        Raises
        ------
        ValueError
            If the sort has no such constructor. The sort is closed, so
            an unknown name is an error rather than an open extension.
        """
        try:
            position = self.constructors.index(name)
        except ValueError as error:
            raise ValueError(
                f"{name!r} is not a constructor of index sort {self.name!r}"
            ) from error
        return self.arities[position]


type IndexSort = NatSort | ShapeSort | ContextSort | UserIndexSort
"""The sort of an index binder: naturals, shapes, contexts, or a declared sort."""

NAT = NatSort()
"""The sort of natural-number indices."""
SHAPE = ShapeSort()
"""The sort of static tensor shapes."""


class TypeBinder(QiecModel):
    """A telescope binder for a type or effect-kinded static argument.

    Parameters
    ----------
    name
        The binder's display name.
    kind
        The kind of the bound argument.
    refinable
        Whether case analysis may refine the bound argument; true for a
        family's indices and false for its uniform parameters.
    tag
        The serialization discriminator; always ``"type"``.
    """

    name: str
    kind: Kind = dx.field(default=TYPE, opaque=True, extras=OPAQUE_BY_VALUE)
    refinable: bool = False
    tag: Literal["type"] = "type"


class IndexBinder(QiecModel):
    """A telescope binder for an index argument.

    Parameters
    ----------
    name
        The binder's display name.
    sort
        The closed index sort of the bound argument.
    refinable
        Whether case analysis may refine the bound argument; true for a
        family's indices and false for its uniform parameters.
    tag
        The serialization discriminator; always ``"index"``.
    """

    name: str
    sort: IndexSort = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    refinable: bool = False
    tag: Literal["index"] = "index"


class EffectBinder(QiecModel):
    """A telescope binder for an effect argument.

    Parameters
    ----------
    name
        The binder's display name.
    refinable
        Whether case analysis may refine the bound argument.
    tag
        The serialization discriminator; always ``"effect"``.
    """

    name: str
    refinable: bool = False
    tag: Literal["effect"] = "effect"


type TelescopeBinder = TypeBinder | IndexBinder | EffectBinder
"""One binder of a declaration telescope: a type, an index, or an effect binder."""
type Telescope = tuple[TelescopeBinder, ...]
"""The static binders a declaration takes, in order; later binders may mention earlier ones."""


def validate_telescope(telescope: Telescope) -> None:
    """Reject duplicate names in one dependent telescope.

    Parameters
    ----------
    telescope : Telescope
        The binders to check, in order.

    Raises
    ------
    ValueError
        If a binder name is empty or repeats. Later binders may depend on
        earlier ones, so a repeated name would make a reference ambiguous.
    """
    seen: set[str] = set()
    for binder in telescope:
        if not binder.name:
            raise ValueError("telescope binder name cannot be empty")
        if binder.name in seen:
            raise ValueError(f"duplicate telescope binder {binder.name!r}")
        seen.add(binder.name)


__all__ = [
    "ArrowKind",
    "ContextSort",
    "EFFECT",
    "EffectBinder",
    "EffectKind",
    "IndexBinder",
    "IndexSort",
    "Kind",
    "NAT",
    "NatSort",
    "ROW",
    "RowKind",
    "SHAPE",
    "ShapeSort",
    "TYPE",
    "Telescope",
    "TelescopeBinder",
    "TypeBinder",
    "TypeKind",
    "UserIndexSort",
    "validate_telescope",
]
