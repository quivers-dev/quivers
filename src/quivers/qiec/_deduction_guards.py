"""The portable matched-coordinate fragment for lexical rule conditions."""

from __future__ import annotations

from quivers.qiec.effects import EMPTY_ROW
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.module import NamedComputation
from quivers.qiec.rules import PRESENTATION_PROTOCOL, Rule, RuleError
from quivers.qiec.terms import Bind, If, LiteralValue, PrimitiveApplication, Return, Var
from quivers.qiec.types import BOOL, INT

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


def _check_guard_body(computation: NamedComputation) -> None:
    """Require a closed scalar computation shared by checked and grounded runs.

    Parameters
    ----------
    computation
        The ordinary checked computation authoring a guard.

    Raises
    ------
    RuleError
        If the computation needs terms outside this portable scalar fragment.
    """
    if (
        computation.telescope
        or computation.type.effects != EMPTY_ROW
        or computation.type.result != BOOL
        or any(local.type != INT for local in computation.parameters)
    ):
        raise RuleError("a rule guard must be a closed pure Bool computation of Ints")
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
            case Bind(binder=binder, first=first, then=then) if binder.type in (
                INT,
                BOOL,
            ):
                pending.extend((first, then))
            case If(condition=condition, then=then, otherwise=otherwise):
                pending.extend((condition, then, otherwise))
            case PrimitiveApplication(name=name, arguments=arguments) if (
                name in _OPERATIONS
            ):
                pending.extend(arguments)
            case Var(local=local) if local.type in (INT, BOOL):
                continue
            case LiteralValue(type=type_) if type_ in (INT, BOOL):
                continue
            case _:
                raise RuleError(
                    "a rule guard uses only scalar Int and Bool operations, "
                    "without named calls or handlers"
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


__all__: list[str] = []
