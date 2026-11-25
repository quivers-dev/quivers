"""The data variables a design term reads."""

from __future__ import annotations

from collections.abc import Iterator

from formulae.terms.call_resolver import LazyCall, LazyOperator, LazyVariable
from formulae.terms.call import Call
from formulae.terms.terms import Term
from formulae.terms.variable import Variable

from quivers._values import ValueModel
from quivers.formulas._formula_options import BRMS_INTERCEPT
from quivers.formulas._smooth import term_constants

__all__: list[str] = []

type Argument = LazyCall | LazyOperator | LazyVariable
"""A part of an authored call that can hold variables."""


class DesignVariables(ValueModel):
    """The data variables of one design term.

    Parameters
    ----------
    names : tuple[str, ...]
        Variables in order of appearance.
    by : tuple[str, ...]
        Variables given as a smooth's or Gaussian process's ``by``.
    integer : tuple[str, ...]
        Variables a monotonic effect codes.
    """

    names: tuple[str, ...]
    by: tuple[str, ...] = ()
    integer: tuple[str, ...] = ()


def _calls(argument: Argument) -> Iterator[LazyCall]:
    """Yield a call and the calls nested in it.

    Parameters
    ----------
    argument : Argument
        Authored argument.

    Yields
    ------
    LazyCall
        Every call, outermost first.
    """
    if isinstance(argument, LazyCall):
        yield argument
        for part in (*argument.args, *argument.kwargs.values()):
            yield from _calls(part)
    elif isinstance(argument, LazyOperator):
        for part in argument.args:
            yield from _calls(part)


def _names(argument: Argument) -> Iterator[str]:
    """Yield the variables of an authored argument.

    Parameters
    ----------
    argument : Argument
        Authored argument.

    Yields
    ------
    str
        Variable names in order of appearance.
    """
    if isinstance(argument, LazyVariable):
        yield argument.name
    elif isinstance(argument, LazyCall):
        for part in (*argument.args, *argument.kwargs.values()):
            yield from _names(part)
    elif isinstance(argument, LazyOperator):
        for part in argument.args:
            yield from _names(part)


def design_variables(term: Term) -> DesignVariables:
    """List the data variables of a design term.

    Parameters
    ----------
    term : formulae.terms.terms.Term
        Term of a retained design.

    Returns
    -------
    DesignVariables
        Its variables, each once. R's logical literals, the names a smooth's
        ``xt`` reads, and the reserved intercept column are no data variables.
    """
    constants = term_constants(term) | {BRMS_INTERCEPT, ""}
    names: list[str] = []
    by: list[str] = []
    integer: list[str] = []
    for component in getattr(term, "components", ()):
        if isinstance(component, Variable):
            names.append(component.name)
            continue
        if not isinstance(component, Call):
            continue
        names.extend(_names(component.call))
        for call in _calls(component.call):
            if "by" in call.kwargs:
                by.extend(_names(call.kwargs["by"]))
            if call.callee == "mo":
                integer.extend(name for part in call.args for name in _names(part))

    def distinct(values: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for name in values if name not in constants))

    return DesignVariables(
        names=distinct(names), by=distinct(by), integer=distinct(integer)
    )
