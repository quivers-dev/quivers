"""Emit the terms of a predictor into a program.

`ProgramEmission` is what the terms ask of the program being emitted.
`emit_terms` appends the steps of a sequence of terms, `emit_predictor` the
scalar part of one predictor, and `emit_axis_part` its part over an output axis,
without knowing which terms they are.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from quivers.dsl.ast_nodes import LetExprNode, ProgramStep, SampleStep, Statement
from quivers.formulas._ast import _add, _factor, _let, _var
from quivers.formulas._group_emission import emit_group_block
from quivers.formulas._group_law import GroupBlockLaw
from quivers.formulas._predictor import Predictor
from quivers.formulas._term import (
    COORDINATE,
    AxisDecl,
    Contribution,
    OutputAxis,
    SiteDecl,
    Term,
)

if TYPE_CHECKING:
    from quivers.formulas.compile import FormulaToQVRModule

__all__: list[str] = []


class ProgramEmission:
    """The program a compiler is emitting, as its terms see it."""

    def __init__(
        self,
        lens: FormulaToQVRModule,
        statements: list[Statement],
        gathered: Mapping[tuple[str, str, str], LetExprNode],
    ) -> None:
        """Wrap one emission.

        Parameters
        ----------
        lens : FormulaToQVRModule
            Compiler carrying the prior configuration.
        statements : list[Statement]
            Module declarations receiving axes.
        gathered : Mapping[tuple[str, str, str], LetExprNode]
            Group-level coefficients emitted ahead of the predictors, gathered
            at the rows' levels.
        """
        self._lens = lens
        self._statements = statements
        self._gathered = dict(gathered)

    def draw(self, site: SiteDecl) -> SampleStep:
        """Draw a site from its configured or default prior.

        Parameters
        ----------
        site : SiteDecl
            Site with a default prior.

        Returns
        -------
        SampleStep
            The draw over the site's axis.
        """
        return self._lens.draw_site(site)

    def gathered(
        self, key: tuple[str, str, str], law: GroupBlockLaw | None
    ) -> tuple[tuple[ProgramStep, ...], LetExprNode]:
        """Read a group-level coefficient at the rows' levels.

        Parameters
        ----------
        key : tuple[str, str, str]
            Predictor namespace, grouping factor, and coefficient label.
        law : GroupBlockLaw or None
            Law of the coefficient's block; None when another program emits it.

        Returns
        -------
        tuple[tuple[ProgramStep, ...], LetExprNode]
            The steps of the block, empty once it is emitted, and the gathered
            coefficient.

        Raises
        ------
        ValueError
            If neither this program nor another emits the block.
        """
        steps: list[ProgramStep] = []
        if key not in self._gathered:
            if law is None:
                raise ValueError(f"no program emits the group coefficient {key!r}")
            self._gathered.update(
                emit_group_block(self._lens, law, self._statements, steps)
            )
        return tuple(steps), self._gathered[key]

    def correlation(
        self, site: str, axis: AxisDecl, result: str
    ) -> tuple[ProgramStep, ...]:
        """Draw a correlation factor over a finite axis.

        Parameters
        ----------
        site : str
            Factor site.
        axis : AxisDecl
            Axis of the factor.
        result : str
            Binding of the factor, constant over an axis of one element.

        Returns
        -------
        tuple[ProgramStep, ...]
            The draw and the binding.
        """
        steps: list[ProgramStep] = []
        self._lens._append_residual_correlation(
            steps, axis.name, axis.size, result=result, site=site
        )
        return tuple(steps)


def emit_terms(
    terms: Iterable[Term], emission: ProgramEmission, steps: list[ProgramStep]
) -> tuple[Contribution, ...]:
    """Append the steps of a sequence of terms.

    Parameters
    ----------
    terms : Iterable[Term]
        Terms in emission order.
    emission : ProgramEmission
        Program being emitted.
    steps : list[ProgramStep]
        Program steps receiving the draws and bindings.

    Returns
    -------
    tuple[Contribution, ...]
        The bound summands of the terms.
    """
    contributions: list[Contribution] = []
    for term in terms:
        steps.extend(term.steps(emission))
        contributions.extend(term.contributions())
    return tuple(contributions)


def emit_predictor(
    predictor: Predictor, emission: ProgramEmission, steps: list[ProgramStep]
) -> LetExprNode:
    """Append the scalar part of one predictor.

    Parameters
    ----------
    predictor : Predictor
        Planned predictor.
    emission : ProgramEmission
        Program being emitted.
    steps : list[ProgramStep]
        Program steps receiving the draws and bindings.

    Returns
    -------
    LetExprNode
        Sum of the contributions over the rows, or zero without terms.
    """
    summands = emit_terms(predictor.part(None), emission, steps)
    return _add(*(_var(summand.name) for summand in summands))


def emit_axis_part(
    predictor: Predictor,
    output: OutputAxis,
    emission: ProgramEmission,
    steps: list[ProgramStep],
    name: str,
) -> None:
    """Append the part of one predictor over an output axis.

    Parameters
    ----------
    predictor : Predictor
        Planned predictor.
    output : OutputAxis
        Axis of the part.
    emission : ProgramEmission
        Program being emitted.
    steps : list[ProgramStep]
        Program steps receiving the draws and bindings.
    name : str
        Binding of the sum of the contributions over the rows and the axis.
    """
    row, coordinate = _var("response"), _var(COORDINATE)
    summands = emit_terms(predictor.part(output), emission, steps)
    steps.append(
        _let(
            name,
            _factor(
                (("response", "Resp"), (COORDINATE, output.name)),
                body=_add(*(summand.cell(row, coordinate) for summand in summands)),
            ),
        )
    )
