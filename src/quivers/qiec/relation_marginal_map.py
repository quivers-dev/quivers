"""Checked sparse alignment-marginal scoring on explicit output candidates."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    MarginalOutputChoice,
    RationalRelation,
    Words,
    best_marginal_output,
)
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.relations import RationalRelationSpec, RelationWeightBindings
from quivers.qiec.types import TypeExpr


class FixedRelationMarginalMapRequest(QiecModel):
    """A checked request for the greatest alignment sum on declared outputs.

    The original Markov coefficients define exact nonnegative rational
    path sums on acyclic productive observations. The first maximum in
    the candidate order wins a tie. Gradients through the selected
    log pair weight keep that selection fixed.

    Parameters
    ----------
    relation : RationalRelationSpec
        The typed original Markov topology and weight identities.
    inputs : Words
        One fixed word per input tape.
    candidates : tuple of Words
        Output words in candidate order, one word per output tape.
    max_states : int
        Cumulative reached state budget over all candidate lattices.
    max_work : int
        Cumulative examined arc budget over all candidate lattices.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    inputs: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    candidates: tuple[Words, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_states: int
    max_work: int

    @dx.model_validator()
    def _validate(self) -> None:
        """Check budgets, algebra, tape split, and every declared word.

        Raises
        ------
        ValueError
            If the limits, algebra, or observation tape values are invalid.
        """
        _validate_limits(self.max_states, self.max_work)
        if self.relation.algebra != MARKOV.subject:
            raise ValueError("exact marginal output ranking requires Markov weights")
        topology = self.relation.topology
        split = topology.input_tapes
        output_count = len(topology.tapes) - split
        if split == 0 or output_count == 0:
            raise ValueError("marginal output ranking requires input and output tapes")
        if len(self.inputs) != split:
            raise ValueError("marginal output ranking requires one word per input tape")
        topology._validate_words((*self.inputs, *((),) * output_count))
        for output in self.candidates:
            topology._validate_words((*self.inputs, *output))

    @property
    def operation(self) -> Literal["marginal_map_log_weight"]:
        """Return the operation recognized by the scalar score provider.

        Returns
        -------
        str
            The bounded alignment-marginal MAP objective.
        """
        return "marginal_map_log_weight"

    @property
    def result_type(self) -> TypeExpr:
        """Return the QIEC result type of the selected log pair weight.

        Returns
        -------
        TypeExpr
            The canonical LogWeight carrier.
        """
        return LOG_WEIGHT

    def choose(self, parameters: RelationWeightBindings) -> MarginalOutputChoice | None:
        """Return the selected output words and differentiable log pair weight.

        Parameters
        ----------
        parameters : RelationWeightBindings
            Live scalar weights owned by their original parameters.

        Returns
        -------
        MarginalOutputChoice or None
            The positive candidate with greatest alignment sum, if any.
        """
        return self._choose(self.relation.bind(parameters))

    def _choose(self, relation: RationalRelation) -> MarginalOutputChoice | None:
        """Select an output for an already bound instance of this relation.

        Parameters
        ----------
        relation : RationalRelation
            The original relation with stable live parameter identities.

        Returns
        -------
        MarginalOutputChoice or None
            The selected positive candidate if the support is nonempty.
        """
        return best_marginal_output(
            relation,
            self.inputs,
            self.candidates,
            max_states=self.max_states,
            max_work=self.max_work,
        )


__all__ = ["FixedRelationMarginalMapRequest"]
