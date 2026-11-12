"""Scan cells whose state is a marginalized finite latent.

A program ``cell(x_t, z_prev) : X * S -> S`` whose body binds lets, then
marginalizes a latent ``z : S`` drawn from a ``Categorical`` that reads
``z_prev``, observes what ``z`` emits inside the block, and returns
``z``, is a step of a discrete Markov chain. Returning the marginalized
latent carries it to the next position, so its sum extends over the rest
of the chain: ``scan(cell, init=p)`` with ``p`` a distribution over
``S`` denotes the chain with every state summed out, which the forward
algorithm computes in ``O(T K^2)`` for ``T`` positions and ``K`` states.

This module recognizes such a cell from its syntax and refuses the forms
the forward algorithm does not cover, for every consumer of the
expanded module: the compiler, the checked elaboration, and the
transpilers.
"""

from __future__ import annotations

from typing import NoReturn

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    LetStep,
    MarginalizeStep,
    ObjectExpr,
    ObjectProduct,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    TypeName,
)
from quivers.dsl.step_resolution import StepResolutionError

#: The family a chain's transition draws its next state from.
TRANSITION_FAMILY = "Categorical"


class MarkovCell(dx.Model):
    """A scan cell that marginalizes and returns its next state.

    Parameters
    ----------
    name : str
        The cell program's name.
    entry : str
        The parameter holding one position's entry.
    previous : str
        The parameter holding the previous state.
    latent : str
        The marginalized next state, which the cell returns.
    entry_object : ObjectExpr
        The object of one position's entry, the domain's first factor.
    state_object : ObjectExpr
        The object of the state, the cell's codomain.
    prefix : tuple[LetStep, ...]
        The lets before the block, which may read the entry and the
        previous state.
    block : MarginalizeStep
        The block marginalizing the next state.
    """

    name: str
    entry: str
    previous: str
    latent: str
    entry_object: ObjectExpr
    state_object: ObjectExpr
    prefix: tuple[LetStep, ...]
    block: MarginalizeStep


def returns_marginalized_latent(declaration: ProgramDecl) -> bool:
    """Whether a program returns the latent a top-level block marginalizes.

    Such a program has no value on its own, since the latent it returns
    is summed out; it is a step of a marginalized chain and runs only as
    the cell of a scan.

    Parameters
    ----------
    declaration : ProgramDecl
        The program.

    Returns
    -------
    bool
        ``True`` when the program returns one name and a ``marginalize``
        block among its top-level steps binds it.
    """
    if len(declaration.return_vars) != 1:
        return False
    returned = declaration.return_vars[0]
    return any(
        isinstance(step, MarginalizeStep) and step.var == returned
        for step in declaration.draws
    )


def markov_cell(declaration: ProgramDecl) -> MarkovCell:
    """Read a cell that marginalizes and returns its next state.

    Parameters
    ----------
    declaration : ProgramDecl
        A program for which
        [`returns_marginalized_latent`][quivers.dsl.markov_chains.returns_marginalized_latent]
        holds.

    Returns
    -------
    MarkovCell
        The cell's parts.

    Raises
    ------
    StepResolutionError
        If the program is not of the form the forward algorithm sums
        out: two parameters over ``X * S`` returning ``S``; lets before
        the block and nothing after it; a block over ``S`` drawing from
        a ``Categorical``, ungrouped, reduced by log-sum-exp (the
        default), holding only lets and observations of one value per
        position.
    """
    name = declaration.name
    factors = (
        declaration.domain.components
        if isinstance(declaration.domain, ObjectProduct)
        else (declaration.domain,)
    )
    if declaration.params is None or len(declaration.params) != 2 or len(factors) != 2:
        _refuse(
            "state",
            name,
            f"the cell {name!r} of a marginalized chain takes an entry and the "
            f"previous state, `program {name}(x_t, z_prev) : X * S -> S`",
        )
    if not _same_object(factors[1], declaration.codomain):
        _refuse(
            "state",
            name,
            f"the cell {name!r} returns a state of another object than the "
            f"previous state it reads; its type is `X * S -> S`",
        )
    latent = declaration.return_vars[0]
    index = next(
        position
        for position, step in enumerate(declaration.draws)
        if isinstance(step, MarginalizeStep) and step.var == latent
    )
    block = declaration.draws[index]
    assert isinstance(block, MarginalizeStep)
    prefix: list[LetStep] = []
    for step in declaration.draws[:index]:
        if not isinstance(step, LetStep):
            _refuse(
                "step",
                name,
                f"the cell {name!r} of a marginalized chain binds only lets "
                f"before the block over {latent!r}; {_describe(step)} at line "
                f"{_line(step)} has no place in the forward algorithm",
            )
        prefix.append(step)
    if declaration.draws[index + 1 :]:
        _refuse(
            "step",
            name,
            f"the cell {name!r} of a marginalized chain ends with the block over "
            f"{latent!r}; {_describe(declaration.draws[index + 1])} at line "
            f"{_line(declaration.draws[index + 1])} follows it",
        )
    if block.morphism != TRANSITION_FAMILY:
        _refuse(
            "family",
            name,
            f"the cell {name!r} draws its next state {latent!r} from "
            f"{block.morphism!r}; a marginalized chain draws it from a "
            f"`{TRANSITION_FAMILY}` over the state object",
        )
    if block.over is not None or block.over_objs:
        _refuse(
            "grouped",
            name,
            f"the block over {latent!r} in the cell {name!r} is grouped; a "
            f"marginalized chain sums one state per position",
        )
    if block.index is not None and not _same_object(block.index, declaration.codomain):
        _refuse(
            "state",
            name,
            f"the block over {latent!r} in the cell {name!r} ranges over another "
            f"object than the state the cell returns",
        )
    if block.reduction not in (None, "logsumexp"):
        _refuse(
            "reduction",
            name,
            f"the block over {latent!r} in the cell {name!r} reduces by "
            f"{block.reduction!r}; a marginalized chain sums its states, "
            f"`reduction=logsumexp`",
        )
    for step in block.scope:
        if not isinstance(step, LetStep | ObserveStep):
            _refuse(
                "step",
                name,
                f"the block over {latent!r} in the cell {name!r} holds lets and "
                f"observations; {_describe(step)} at line {_line(step)} has no "
                f"place in the forward algorithm",
            )
        if isinstance(step, ObserveStep) and (
            len(step.vars) != 1
            or step.index is not None
            or step.axes is not None
            or step.via is not None
        ):
            _refuse(
                "observe",
                name,
                f"the cell {name!r} observes one value per position; the "
                f"observation at line {step.line} is plated, fibred, or "
                f"destructured",
            )
    entry, previous = declaration.params
    return MarkovCell(
        name=name,
        entry=entry,
        previous=previous,
        latent=latent,
        entry_object=factors[0],
        state_object=declaration.codomain,
        prefix=tuple(prefix),
        block=block,
    )


def _refuse(reason: str, cell: str, message: str) -> NoReturn:
    """Refuse a marginalized chain under ``scan:marginal:<reason>:<cell>``.

    Parameters
    ----------
    reason : str
        What the refusal turns on.
    cell : str
        The cell's name.
    message : str
        The diagnostic.

    Raises
    ------
    StepResolutionError
        Always.
    """
    raise StepResolutionError("qvr-expand", [f"scan:marginal:{reason}:{cell}", message])


def _same_object(left: ObjectExpr, right: ObjectExpr) -> bool:
    """Whether two object expressions name one object.

    Parameters
    ----------
    left : ObjectExpr
        One expression.
    right : ObjectExpr
        The other.

    Returns
    -------
    bool
        ``True`` when both name the same declared object or literal.
    """
    if isinstance(left, TypeName) and isinstance(right, TypeName):
        return left.name == right.name
    return left == right


def _describe(step: ProgramStep) -> str:
    """A step's kind as the source spells it.

    Parameters
    ----------
    step : ProgramStep
        The step.

    Returns
    -------
    str
        The keyword that opens the step.
    """
    return {
        "sample_step": "a `sample` step",
        "observe_step": "an `observe` step",
        "marginalize_step": "a `marginalize` block",
        "let_step": "a `let` step",
        "call_step": "a call",
        "score_step": "a `score` step",
    }.get(str(step.kind), "a step")


def _line(step: ProgramStep) -> int:
    """The line a step is written on.

    Parameters
    ----------
    step : ProgramStep
        The step.

    Returns
    -------
    int
        Its line, or zero when it carries none.
    """
    line = getattr(step, "line", 0)
    return int(line) if isinstance(line, int) else 0


__all__ = []
