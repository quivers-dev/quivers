"""Helper models shared by multiple ast_nodes submodules."""

from typing import Literal

import didactic.api as dx

# ---------------------------------------------------------------------------
# axis-role surface: per-distribution event / batch axis specification
# ---------------------------------------------------------------------------


class Statement(dx.TaggedUnion, discriminator="kind"):
    """Sum of top-level statement kinds."""


class AxisSpec(dx.Model):
    """Axis-role specification on a distribution clause.

    Surface form: ``over <axes> [iid over <axes>]``.

    ``over`` names the event axes, the axes on which the family's
    joint structure (an MVN covariance, a MatrixNormal Kronecker pair,
    a GP kernel) lives.  The axis count must match the family's
    declared event rank; the positional ordering corresponds
    positionally to the family's event-axis ordering.

    ``iid_over`` is an optional readability assertion naming the batch
    axes (the complement of ``over`` in the surrounding morphism's
    type signature).  Inconsistency with the type signature or with
    ``over`` is a compile-time error.

    Axis names resolve against the named factors of the surrounding
    morphism's dom/cod.  The reserved tokens ``dom`` and ``cod`` are
    legal shortcuts only when the corresponding side is a single
    unfactored object.
    """

    over: tuple[str, ...]
    iid_over: tuple[str, ...] = ()
    line: int = 0
    col: int = 0


# ---------------------------------------------------------------------------
# composition-level type alias used by CompositionDecl
# ---------------------------------------------------------------------------

type CompositionLevel = Literal["algebra", "semigroupoid", "magmoid", "rule"]
"""Algebraic level the file declares for its composition rule.

The four levels correspond to the
[`quivers.core.algebras.CompositionRule`][quivers.core.algebras.CompositionRule]-hierarchy:

* ``"algebra"`` requires a full `Algebra` (unit, zero,
  meet, negate, identity, dagger, cup/cap).
* ``"semigroupoid"`` requires a `Semigroupoid`
  (associative composition, no identity required); every `Semiring`
  algebra qualifies.
* ``"magmoid"`` requires a `Magmoid`, a rule declared to satisfy
  no composition law.
* ``"composition_rule"`` is permissive: any
  `CompositionRule` is accepted.
"""

# ---------------------------------------------------------------------------
# Option block: one ``[k=v, ...]`` syntax for every declaration
# ---------------------------------------------------------------------------


class OptionValue(dx.TaggedUnion, discriminator="kind"):
    """Value inhabiting one entry of an option block.

    Surface shapes:

    * Bare flag ``[role]`` -> `OptionFlag` (no value)
    * ``role=parameter`` -> `OptionName` (identifier)
    * ``depth=8`` / ``scale=0.1`` -> `OptionNumber`
    * ``path="lex.tsv"`` -> `OptionString`
    * ``over=[a, b]`` -> `OptionList`
    * ``via=product(a, b)`` -> `OptionCall`
    """


class OptionFlag(OptionValue):
    """A bare key with no value (e.g. ``[learnable]``)."""

    kind: Literal["option_flag"] = "option_flag"


class OptionName(OptionValue):
    """A key bound to a bare identifier (e.g. ``role=parameter``)."""

    value: str
    kind: Literal["option_name"] = "option_name"


class OptionNumber(OptionValue):
    """A key bound to a numeric literal (e.g. ``depth=8``, ``scale=0.1``)."""

    value: float
    kind: Literal["option_number"] = "option_number"


class OptionString(OptionValue):
    """A key bound to a string literal (e.g. ``path="lex.tsv"``)."""

    value: str
    kind: Literal["option_string"] = "option_string"


class OptionList(OptionValue):
    """A key bound to a list of identifiers / numbers / strings."""

    items: tuple[OptionValue, ...] = ()
    kind: Literal["option_list"] = "option_list"


class OptionCall(OptionValue):
    """A key bound to a function-call value (e.g. ``via=product(a, b)``)."""

    func: str
    args: tuple[OptionValue, ...] = ()
    kind: Literal["option_call"] = "option_call"


class OptionEntry(dx.Model):
    """One ``key=value`` (or bare ``key``) entry in an option block.

    The list-of-entries layout (rather than a dict) preserves source
    order; downstream code typically realises a dict on first use.
    """

    key: str
    value: OptionValue = dx.field(default_factory=OptionFlag)
    line: int = 0
    col: int = 0


__all__ = [
    "Statement",
    "AxisSpec",
    "CompositionLevel",
    "OptionCall",
    "OptionEntry",
    "OptionFlag",
    "OptionList",
    "OptionName",
    "OptionNumber",
    "OptionString",
    "OptionValue",
]
