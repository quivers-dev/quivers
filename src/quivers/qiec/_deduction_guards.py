"""The portable fragment for checked matched-field rule guards."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec.declarations import ConstructorDecl, FamilyDecl
from quivers.qiec.effects import EMPTY_ROW, ComputationType
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.identifiers import ComputationId, ConstructorId, SourceOrigin
from quivers.qiec.kinds import NAT, IndexBinder, Telescope, UserIndexSort
from quivers.qiec.module import NamedComputation
from quivers.qiec.rules import PRESENTATION_PROTOCOL, Ground, Rule, RuleError
from quivers.qiec.terms import (
    Bind,
    Case,
    Computation,
    If,
    LiteralValue,
    Local,
    PrimitiveApplication,
    Return,
    Var,
)
from quivers.qiec.types import BOOL, INT, TypeApplication

_OPERATIONS = frozenset(
    {
        "add_int",
        "sub_int",
        "mul_int",
        "eq_int",
        "ne_int",
        "lt_int",
        "le_int",
        "gt_int",
        "ge_int",
        "eq_bool",
        "ne_bool",
        "and",
        "or",
        "not",
    }
)


class _TermGuard(NamedComputation):
    """A named condition with its exact owner's closed item declarations.

    The named computation keeps its existing identity, matched parameters,
    and pure Boolean body. The capture stores the generated unindexed item
    family and its constructor declarations in family order.
    """

    id: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    telescope: Telescope = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameters: tuple[Local, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    body: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    type: ComputationType = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SourceOrigin = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    item_family: FamilyDecl = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    item_constructors: tuple[ConstructorDecl, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

    @cached_property
    def item_type(self) -> TypeApplication:
        """Get the exact nominal type of a matched constructor value.

        Returns
        -------
        TypeApplication
            The declaration's closed item type.
        """
        return TypeApplication(
            constructor=self.item_family.type_constructor, arguments=()
        )

    @cached_property
    def _constructors(self) -> Mapping[str, ConstructorDecl]:
        """Index the owner's constructor declarations by their local symbols.

        Returns
        -------
        Mapping of str to ConstructorDecl
            The validated declarations for converting matched ground fields.
        """
        prefix = self.item_family.name.removesuffix("Item")
        return MappingProxyType(
            {item.name.removeprefix(prefix): item for item in self.item_constructors}
        )

    @dx.model_validator()
    def _closed_capture(self) -> None:
        """Require the exact shape of a generated unindexed item family.

        Raises
        ------
        RuleError
            If a capture includes another family, an unbound constructor,
            static indices, or non-item and non-integer data.
        """
        family = self.item_family
        if (
            not family.name.endswith("__Item")
            or not family.closed
            or family.parameters
            or family.indices
            or tuple(item.id for item in self.item_constructors) != family.constructors
            or len(self._constructors) != len(self.item_constructors)
            or any(
                item.family != family.id
                or item.id != ConstructorId.derive(family.id, "constructor", item.name)
                or not item.name.startswith(family.name.removesuffix("Item"))
                or item.telescope
                or item.result_indices
                or any(field.type not in (INT, self.item_type) for field in item.fields)
                for item in self.item_constructors
            )
        ):
            raise RuleError(
                "term guards require the deduction's closed item declarations"
            )
        if all(parameter.type != self.item_type for parameter in self.parameters):
            raise RuleError("a term guard must read a matched item parameter")
        _check_guard_body(self, self.item_type)

    def ground_item(self, value: Ground, sort: UserIndexSort) -> RuntimeConstructor:
        """Reify a matched term using its owned constructor declarations.

        Parameters
        ----------
        value
            The existing grounding's term value.
        sort
            Its schematic variable's closed index sort.

        Returns
        -------
        RuntimeConstructor
            The corresponding unindexed constructor value.

        Raises
        ------
        RuleError
            If the ground term cannot inhabit the captured nominal family.
        """
        if isinstance(value, str):
            label, fields = value, ()
        elif isinstance(value, tuple) and value and isinstance(value[0], str):
            label, fields = value[0], value[1:]
        else:
            raise RuleError("a matched term guard argument must be a constructor")
        constructor = self._constructors.get(label)
        if (
            label not in sort.constructors
            or sort.constructor_arity(label) != len(fields)
            or (isinstance(value, tuple) and not fields)
            or constructor is None
            or tuple(field.type for field in constructor.fields)
            != (self.item_type,) * len(fields)
        ):
            raise RuleError(
                f"guard term {label!r} does not match its nominal declaration"
            )
        return RuntimeConstructor(
            constructor=constructor.id,
            static_arguments=(),
            fields=tuple(self.ground_item(field, sort) for field in fields),
            result_type=self.item_type,
        )


def _coordinate_arguments(rule: Rule) -> tuple[str, ...]:
    """Read the matched Nat fields in the rule telescope's order.

    Parameters
    ----------
    rule
        The typed rule whose fields are matched.

    Returns
    -------
    tuple of str
        The matched coordinate names in order.
    """
    return tuple(
        binder.name
        for binder in rule.telescope
        if isinstance(binder, IndexBinder) and binder.sort == NAT
    )


def _matched_arguments(rule: Rule, guard: NamedComputation) -> tuple[str, ...]:
    """Select the declared argument fragment in the existing telescope order.

    Parameters
    ----------
    rule
        The range-restricted rule.
    guard
        Its checked condition.

    Returns
    -------
    tuple of str
        The coordinate-only projection or the whole matched telescope.
    """
    if isinstance(guard, _TermGuard):
        return tuple(binder.name for binder in rule.telescope)
    return _coordinate_arguments(rule)


def _matched_value(
    guard: NamedComputation, rule: Rule, name: str, parameter: Local, value: Ground
) -> Ground | RuntimeConstructor:
    """Pass the existing binding across the guard's checked value boundary.

    Parameters
    ----------
    guard
        The owning condition and its optional item declarations.
    rule
        The rule whose binding supplied the argument.
    name
        The schematic variable's name.
    parameter
        The condition's corresponding value parameter.
    value
        The ground matching value.

    Returns
    -------
    Ground or RuntimeConstructor
        The existing ground value, or its exact checked item representation.

    Raises
    ------
    RuleError
        If a declared term guard argument has no closed term sort.
    """
    if not isinstance(guard, _TermGuard) or parameter.type != guard.item_type:
        return value
    for binder in rule.telescope:
        if (
            binder.name == name
            and isinstance(binder, IndexBinder)
            and isinstance(binder.sort, UserIndexSort)
        ):
            return guard.ground_item(value, binder.sort)
    raise RuleError(f"term guard argument {name!r} has no closed term sort")


def _check_guard_body(
    computation: NamedComputation, item_type: TypeApplication | None = None
) -> None:
    """Check the same closed source fragment for scalar and grounded execution.

    Parameters
    ----------
    computation
        The ordinary checked computation authoring a guard.
    item_type
        The owned item type when the guard can analyze matched terms.

    Raises
    ------
    RuleError
        If the computation needs terms outside this portable scalar fragment.
    """
    if (
        computation.telescope
        or computation.type.effects != EMPTY_ROW
        or computation.type.result != BOOL
        or any(local.type not in (INT, item_type) for local in computation.parameters)
    ):
        carriers = "Ints" if item_type is None else "Ints and the deduction's Item type"
        raise RuleError(
            f"a rule guard must be a closed pure Bool computation of {carriers}"
        )
    local_types = (INT, BOOL, item_type)
    pending: list[object] = [computation.body]
    visited: set[int] = set()
    while pending:
        term = pending.pop()
        if id(term) in visited:
            continue
        visited.add(id(term))
        match term:
            case Return(value=value):
                pending.append(value)
            case Bind(binder=binder, first=first, then=then) if (
                binder.type in local_types
            ):
                pending.extend((first, then))
            case If(condition=condition, then=then, otherwise=otherwise):
                pending.extend((condition, then, otherwise))
            case PrimitiveApplication(name=name, arguments=arguments) if (
                name in _OPERATIONS
            ):
                pending.extend(arguments)
            case Case(scrutinee=Var(local=local), motive=motive, branches=branches) if (
                item_type is not None
                and local.type == item_type
                and not motive.indices
                and motive.result_type in local_types
                and all(not branch.static_arguments for branch in branches)
            ):
                pending.extend(branch.body for branch in branches)
            case Var(local=local) if local.type in local_types:
                continue
            case LiteralValue(type=type_) if type_ in (INT, BOOL):
                continue
            case _:
                fragment = (
                    "scalar Int and Bool operations"
                    if item_type is None
                    else "scalar Int and Bool operations and case on its Item type"
                )
                raise RuleError(
                    f"a rule guard uses only {fragment}, without named calls or handlers"
                )


def _guard_in_scope(
    computation: NamedComputation, presentation: str
) -> NamedComputation:
    """Retain the guard in a rule presentation's private module envelope.

    Parameters
    ----------
    computation
        The authored guard, with its stable computation identity.
    presentation
        The lexical rule presentation's module name.

    Returns
    -------
    NamedComputation
        The same signature, parameters, identity, and body, with a
        declaration origin in the nested module.
    """
    return computation.with_(
        origin=computation.origin.with_(
            module=presentation, source_protocol=PRESENTATION_PROTOCOL
        )
    )


def _term_guard_in_scope(
    computation: NamedComputation,
    presentation: str,
    family: FamilyDecl,
    constructors: tuple[ConstructorDecl, ...],
) -> _TermGuard:
    """Capture exactly the owner's item declarations alongside its checked guard.

    Parameters
    ----------
    computation
        The original computation from the owning QIEC module.
    presentation
        The name of the private rule module checking the guard.
    family, constructors
        The closed item family and its constructor declarations.

    Returns
    -------
    _TermGuard
        The same computation, scoped to the rule module and carrying its
        exact type dependencies.
    """
    return _TermGuard(
        id=computation.id,
        name=computation.name,
        telescope=computation.telescope,
        parameters=computation.parameters,
        body=computation.body,
        type=computation.type,
        origin=computation.origin.with_(
            module=presentation, source_protocol=PRESENTATION_PROTOCOL
        ),
        item_family=family,
        item_constructors=constructors,
    )


__all__: list[str] = []
