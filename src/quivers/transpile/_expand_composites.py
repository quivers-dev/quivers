"""The transpilers' view of composite let expansion.

Expansion lives with the source language in
:mod:`quivers.dsl.composite_lets`; this module turns its failures into
the transpile API's
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct].
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    LetExprCall,
    LetExprVar,
    LetStep,
    MarginalizeStep,
    Module,
    ProgramDecl,
    ProgramStep,
)
from quivers.dsl.composite_lets import expand_composite_lets as _expand_composite_lets
from quivers.dsl.composite_lets import scan_let_call
from quivers.dsl.step_resolution import StepResolutionError
from quivers.transpile._api import UnsupportedConstruct


def expand_composite_lets(module: Module, *, target: str) -> Module:
    """Expand composite let aliases, reporting failure as unsupported.

    A `scan` in a chain expands to a step program applied at every
    position of a sequence whose length arrives with the data. Writing
    that out for a target needs a loop whose bound is the sequence
    length and one sample site per position, and the sequence axis is
    not an object the module declares, so no target has a lowering
    that keeps the measure; the scan is refused here rather than
    approximated.

    Parameters
    ----------
    module : Module
        The parsed module.
    target : str
        The transpile target, for the diagnostic.

    Returns
    -------
    Module
        The module with every composite draw unfolded into steps.

    Raises
    ------
    UnsupportedConstruct
        If a composite cannot be expanded, or a draw scans a cell over
        a sequence.
    """
    try:
        expanded = _expand_composite_lets(module, target=target)
    except StepResolutionError as error:
        raise UnsupportedConstruct(error.target, error.kinds) from error
    for statement in expanded.statements:
        if isinstance(statement, ProgramDecl):
            _refuse_scans(statement.draws)
    return expanded


def _refuse_scans(steps: tuple[ProgramStep, ...]) -> None:
    """Refuse every scan among a program's steps.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps, searched through marginalization scopes.

    Raises
    ------
    UnsupportedConstruct
        If a let step calls ``scan``: a marginalized chain under
        ``scan:marginal:no-lowering:<cell>``, any other scan under
        ``scan:no-lowering:<cell>``.
    """
    for step in steps:
        if isinstance(step, MarginalizeStep):
            _refuse_scans(step.scope)
        elif (
            isinstance(step, LetStep)
            and isinstance(step.value, LetExprCall)
            and step.value.func == "scan"
        ):
            scan = scan_let_call(step.value)
            if scan is not None and scan.marginal:
                raise UnsupportedConstruct(
                    "qvr-transpile", [f"scan:marginal:no-lowering:{scan.step}"]
                )
            cell = (
                step.value.args[0].name.split("__scan_step", 1)[0]
                if step.value.args and isinstance(step.value.args[0], LetExprVar)
                else "scan"
            )
            raise UnsupportedConstruct("qvr-transpile", [f"scan:no-lowering:{cell}"])


__all__ = ["expand_composite_lets"]
