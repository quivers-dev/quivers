"""Collapse handler: integrate a conjugate parent out of the joint.

`CollapseHandler` takes parent-to-child site pairs. The parent is drawn
from its prior so the program downstream of it runs, but contributes
nothing to the joint; the child is answered with the run's observation
for it and scored under the closed-form marginal the pair's conjugacy
gives, with the parent integrated out. The joint the run accumulates is
therefore the marginal joint, exactly, for the identity-link pairs of
:mod:`quivers.effects.conjugate`. Each collapsed parent site is annotated
with ``metadata["collapse"]`` naming its child in every trace of the run.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.effects.conjugate import (
    CONJUGATE_SOLVERS,
    conjugate_solver,
    family_of,
)
from quivers.effects.sites import TorchSampleable
from quivers.qiec.builtins import (
    RANDOM,
    RANDOM_SAMPLE,
    _emit_score,
    _handler_def,
    _site_and_sampleable,
)
from quivers.qiec.effects import EffectRow, ResumptionGrade, RowEntry
from quivers.qiec.evaluator import (
    ClauseContext,
    Forward,
    InvalidHandlerError,
    Resumption,
    RuntimeClause,
    RuntimeHandler,
    RuntimeRequest,
)
from quivers.qiec.builtins import SCORE


class CollapseHandler(EffectHandler):
    """Integrate conjugate parents out of the joint.

    Parameters
    ----------
    pairs : dict[str, str] or None
        Parent site name to child site name. ``None`` collapses nothing
        and only carries the solver registry for introspection.
    """

    def __init__(self, pairs: dict[str, str] | None = None) -> None:
        self.pairs = dict(pairs) if pairs is not None else {}
        self.registry = CONJUGATE_SOLVERS

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install the collapsing handler of the ``random`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared, whose observations supply each
            child's value.

        Returns
        -------
        tuple[Installation, ...]
            The handler, which draws parents without scoring them and
            scores children under the pair's marginal.
        """
        kernel = run.kernel
        children = {child: parent for parent, child in self.pairs.items()}
        parents: dict[str, tuple[TorchSampleable, torch.Tensor]] = {}
        score_instance = kernel.score.entry.instance
        definition = _handler_def(
            "Random.collapse",
            RANDOM,
            ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
            key=f"collapse-{id(self):x}",
            input_type=run.result_type,
            output_type=run.result_type,
            introduced=EffectRow(
                entries=(RowEntry(instance=score_instance, effect=SCORE),)
            ),
            total=False,
        )

        def sample(
            request: RuntimeRequest, resume: Resumption, context: ClauseContext
        ) -> object:
            """Answer a parent by an unscored draw and a child by its marginal.

            Parameters
            ----------
            request : RuntimeRequest
                The request being answered and its arguments.
            resume : Resumption
                The continuation.
            context : ClauseContext
                Runtime services available to the clause.

            Returns
            -------
            object
                What the resumed computation produced, or a `Forward`
                for a site the pairs do not name.

            Raises
            ------
            InvalidHandlerError
                If a child is reached before its parent, or the run has
                no observation for it.
            """
            site, sampleable = _site_and_sampleable(request, definition.name)
            assert isinstance(site, str)
            assert isinstance(sampleable, TorchSampleable)
            if site in self.pairs:
                value = sampleable.rsample()
                parents[site] = (sampleable, value)
                run.annotations.setdefault(site, {})["collapse"] = self.pairs[site]
                run.interventions.add(run.key_of(request))
                return resume(value)
            if site in children:
                parent_name = children[site]
                if parent_name not in parents:
                    raise InvalidHandlerError(
                        f"collapse reached child {site!r} before its parent "
                        f"{parent_name!r}"
                    )
                if site not in run.observations:
                    raise InvalidHandlerError(
                        f"collapse needs an observation for child {site!r}"
                    )
                parent_sampleable, parent_value = parents[parent_name]
                solver = conjugate_solver(
                    family_of(parent_sampleable), family_of(sampleable)
                )
                value = run.observations[site]
                weight = solver(
                    parent_sampleable.parameters(), parent_value, sampleable, value
                )
                _emit_score(
                    context,
                    request,
                    score_instance,
                    weight,
                    run.validator,
                    role="condition-score",
                )
                return resume(value)
            return Forward()

        runtime = RuntimeHandler(
            definition=definition,
            clauses={
                RANDOM_SAMPLE: RuntimeClause(
                    invoke=sample, result_validator=run.validator
                )
            },
        )
        return (Installation(instance=kernel.random, runtime=runtime),)


def collapse(pairs: dict[str, str] | None = None) -> CollapseHandler:
    """Return a `CollapseHandler` for the given parent-child pairs.

    Parameters
    ----------
    pairs : dict[str, str] or None
        Parent site name to child site name.

    Returns
    -------
    CollapseHandler
        The handler.
    """
    return CollapseHandler(pairs)


__all__ = ["CollapseHandler", "collapse"]
