"""Fixed schema shapes for semantic effect prefixes in chart parsers.

`class_directed_lifts` emits ``pure`` and forward ``apply`` for an
`Applicative`, ``bind`` for a `Monad`, and ``alt`` for an `Alternative`.
The base schema contributes its name to each generated name; these shapes
are fixed rather than derived from the base schema's domain and codomain.
A bare `Functor` and the arrow interfaces add no schemata.

Generated schemata use the chart's ordinary pattern matching. Distributive
laws can additionally generate swap schemata through `make_swap_schema`.
These semantic type prefixes are distinct from QIEC effect instances and
lexical handlers. Chart execution does not eliminate a prefix by invoking
a QIEC handler.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    SchemaDecl,
    SchemaParameter,
    ObjectEffectApply,
    TypeName,
    ObjectProduct,
    ObjectSlash,
)
from quivers.monadic.distributive_laws import DistributiveLaw
from quivers.monadic.typeclasses import Alternative, Applicative, Monad


def _wrap_with_effect(type_expr, effect_name: str):
    """Wrap a ObjectExpr in a ObjectEffectApply for the given effect.

    e.g. ``X -> T(X)``, ``X/Y -> T(X/Y)``, etc.
    """
    return ObjectEffectApply(effect=effect_name, args=(type_expr,))


def _effect_name(effect: object) -> str:
    """Return the user-facing name for an effect instance.

    Stdlib effects expose a ``name`` attribute (set on each dx.Model
    instance by the stdlib). User-defined effects can supply
    a ``__effect_name__`` class attribute or fall back to the class
    name.
    """
    name = getattr(effect, "name", None)
    if isinstance(name, str) and name:
        return name
    name = getattr(effect, "__effect_name__", None)
    if isinstance(name, str) and name:
        return name
    return type(effect).__name__


def _make_pure_schema(effect: object, base_schema: SchemaDecl) -> SchemaDecl:
    """Generate the ``pure_T`` schema for an Applicative effect.

    Schema shape: ``pure_T[X : Cat] : X -> T(X)``.
    """
    eff_name = _effect_name(effect)
    return SchemaDecl(
        name=f"pure_{eff_name}_{base_schema.name}",
        parameters=(
            SchemaParameter(
                names=("X",),
                type_expr=TypeName(name="Cat"),
            ),
        ),
        domain=TypeName(name="X"),
        codomain=_wrap_with_effect(TypeName(name="X"), eff_name),
    )


def _make_apply_schema(effect: object, base_schema: SchemaDecl) -> SchemaDecl:
    """Generate the ``apply_T`` lift for an Applicative effect.

    Schema shape: ``apply_T[X, Y : Cat] : T(X/Y) * T(Y) -> T(X)``.

    This is the forward-application schema. The function does not
    generate a backward-application variant.
    """
    eff_name = _effect_name(effect)
    return SchemaDecl(
        name=f"apply_{eff_name}_{base_schema.name}",
        parameters=(
            SchemaParameter(
                names=("X", "Y"),
                type_expr=TypeName(name="Cat"),
            ),
        ),
        domain=ObjectProduct(
            components=(
                _wrap_with_effect(
                    ObjectSlash(
                        result=TypeName(name="X"),
                        argument=TypeName(name="Y"),
                        direction="/",
                    ),
                    eff_name,
                ),
                _wrap_with_effect(TypeName(name="Y"), eff_name),
            )
        ),
        codomain=_wrap_with_effect(TypeName(name="X"), eff_name),
    )


def _make_bind_schema(effect: object, base_schema: SchemaDecl) -> SchemaDecl:
    """Generate the ``bind_T`` lift for a Monad effect.

    Schema shape: ``bind_T[X, Y : Cat] : T(X) * (X -> T(Y)) -> T(Y)``.

    Charlow's central scope-extruding lift: a Cont-typed expression
    can take scope over an inner argument by passing through bind.
    """
    eff_name = _effect_name(effect)
    return SchemaDecl(
        name=f"bind_{eff_name}_{base_schema.name}",
        parameters=(
            SchemaParameter(
                names=("X", "Y"),
                type_expr=TypeName(name="Cat"),
            ),
        ),
        domain=ObjectProduct(
            components=(
                _wrap_with_effect(TypeName(name="X"), eff_name),
                # The (X -> T(Y)) function-space lives in the
                # internal-hom of the residuated universe; surface
                # representation uses ObjectSlash to reuse the existing
                # pattern machinery.
                ObjectSlash(
                    result=_wrap_with_effect(TypeName(name="Y"), eff_name),
                    argument=TypeName(name="X"),
                    direction="/",
                ),
            )
        ),
        codomain=_wrap_with_effect(TypeName(name="Y"), eff_name),
    )


def _make_alt_schema(effect: object, base_schema: SchemaDecl) -> SchemaDecl:
    """Generate the ``alt_T`` lift for an Alternative effect.

    Schema shape: ``alt_T[X : Cat] : T(X) * T(X) -> T(X)``.

    Adds non-deterministic / Hamblin-style alternative branches at
    parse time.
    """
    eff_name = _effect_name(effect)
    return SchemaDecl(
        name=f"alt_{eff_name}_{base_schema.name}",
        parameters=(
            SchemaParameter(
                names=("X",),
                type_expr=TypeName(name="Cat"),
            ),
        ),
        domain=ObjectProduct(
            components=(
                _wrap_with_effect(TypeName(name="X"), eff_name),
                _wrap_with_effect(TypeName(name="X"), eff_name),
            )
        ),
        codomain=_wrap_with_effect(TypeName(name="X"), eff_name),
    )


def class_directed_lifts(
    base_schema: SchemaDecl, effect: object
) -> tuple[SchemaDecl, ...]:
    """Generate the class-driven lifts for ``base_schema`` under ``effect``.

    Membership in `Applicative` emits ``pure_T`` and forward ``apply_T``;
    `Monad` adds ``bind_T``; `Alternative` adds ``alt_T``. `MonadPlus`
    receives both sets through its parent classes. A bare `Functor` and
    arrow-only instances emit nothing.

    These are fixed schema shapes. The base schema provides the suffix
    of their names, not a general transformation of its premises.

    Parameters
    ----------
    base_schema : SchemaDecl
        The base schema being lifted.
    effect : object
        An effect instance (any class registered against the relevant
        typeclass ABCs in [`quivers.monadic.typeclasses`][quivers.monadic.typeclasses]).

    Returns
    -------
    tuple of SchemaDecl
        Zero or more schemata in the order ``pure``, ``apply``, ``bind``,
        ``alt``, retaining only those supported by the effect's classes.
    """
    lifts: list[SchemaDecl] = []

    if isinstance(effect, Applicative):
        lifts.append(_make_pure_schema(effect, base_schema))
        lifts.append(_make_apply_schema(effect, base_schema))

    if isinstance(effect, Monad):
        lifts.append(_make_bind_schema(effect, base_schema))

    if isinstance(effect, Alternative):
        lifts.append(_make_alt_schema(effect, base_schema))

    return tuple(lifts)


def make_swap_schema(
    distributive_law: object, base_schema: SchemaDecl | None = None
) -> SchemaDecl:
    """Generate a ``swap_TU`` schema from a registered distributive law.

    Given a `DistributiveLaw` ``λ : S ∘ T ⇒ T ∘ S``, the chart
    parser uses ``swap_TU`` to exchange sibling effect orderings at a
    cell whose effect stack ends in ``T·U``:

        swap_TU[X : Cat] : T(U(X)) -> U(T(X))

    The schema is consumed by the chart's *commutation firing* rule
    of [Effects §4.4](../../semantics/effects.md#4-joint-type-and-effect-dispatch).

    Parameters
    ----------
    distributive_law : DistributiveLaw
        The law whose outer and inner monads the schema exchanges.
    base_schema : SchemaDecl or None
        The schema whose name the swap schema's name extends; the
        name ``swap`` is used when omitted.

    Returns
    -------
    SchemaDecl
        The ``swap_TU`` schema ``T(U(X)) -> U(T(X))``.

    Raises
    ------
    TypeError
        If ``distributive_law`` is not a `DistributiveLaw`.
    """
    if not isinstance(distributive_law, DistributiveLaw):
        raise TypeError(
            "make_swap_schema requires a DistributiveLaw instance; "
            f"got {type(distributive_law).__name__}"
        )
    # The naming convention for the swap schema reuses the monad
    # class names; the schema's directionality (S∘T → T∘S) matches
    # the underlying λ : S(T(-)) → T(S(-)).
    outer = type(distributive_law.outer_monad).__name__
    inner = type(distributive_law.inner_monad).__name__
    base_name = base_schema.name if base_schema is not None else "swap"
    return SchemaDecl(
        name=f"swap_{outer}_{inner}_{base_name}",
        parameters=(
            SchemaParameter(
                names=("X",),
                type_expr=TypeName(name="Cat"),
            ),
        ),
        domain=_wrap_with_effect(_wrap_with_effect(TypeName(name="X"), inner), outer),
        codomain=_wrap_with_effect(_wrap_with_effect(TypeName(name="X"), outer), inner),
    )


def swap_rule_set(
    distributive_laws: tuple[object, ...],
) -> tuple[SchemaDecl, ...]:
    """Generate one ``swap_TU`` schema per registered distributive law.

    The resulting tuple is appended to the chart parser's rule set
    alongside the class-driven lifts; the chart's commutation-firing
    dispatch picks each swap schema up by name.

    Parameters
    ----------
    distributive_laws : tuple of DistributiveLaw
        The registered distributive laws.

    Returns
    -------
    tuple of SchemaDecl
        One swap schema per law, in the order given.

    Raises
    ------
    TypeError
        If an entry is not a `DistributiveLaw`.
    """
    return tuple(make_swap_schema(law) for law in distributive_laws)


def lift_rule_set(
    base_schemas: tuple[SchemaDecl, ...], effects: tuple[object, ...]
) -> tuple[SchemaDecl, ...]:
    """Apply `class_directed_lifts` over a rule-set and effect-stack.

    Returns the union of base schemas and all lifts produced for each
    (base_schema, effect) pair. The chart parser consumes the
    resulting tuple directly.

    Parameters
    ----------
    base_schemas : tuple of SchemaDecl
        The base rule set.
    effects : tuple
        The effect instances to lift each base schema under.

    Returns
    -------
    tuple of SchemaDecl
        The base schemas followed by every lift, grouped by base
        schema and then by effect.
    """
    out: list[SchemaDecl] = list(base_schemas)
    for base in base_schemas:
        for eff in effects:
            out.extend(class_directed_lifts(base, eff))
    return tuple(out)


__all__ = [
    "class_directed_lifts",
    "make_swap_schema",
    "swap_rule_set",
    "lift_rule_set",
]
