"""Checked pair weights after a finite capped ranked preference filter."""

from __future__ import annotations

from hashlib import sha256

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import FixedStringQuery, Words, compile_ranked_ot
from quivers.core.relations.queries import _validate_limits
from quivers.core.relations.ranked import _validate_ranks
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.relations import (
    RationalRelationSpec,
    RelationWeightBindings,
    _validate_fixed_observation,
)
from quivers.qiec.types import REAL, TypeExpr


_RANKED_OT_HANDLER_PREFIX = "__relation_ranked_ot_"


class FixedRankedOTRelationRequest(QiecModel):
    """Score an observed pair among the paths that minimize capped ranks.

    The minimum capped lexicographic profile is computed separately for
    every input over the structural candidate paths. The scalar result
    sums all winning paths at the observed pair in the declared algebra.
    Ranking takes place before reading the numerical weights, and the
    result is an unnormalized pair weight.

    Parameters
    ----------
    relation : RationalRelationSpec
        The original one-input, one-output Markov or LogProb transducer.
    words : Words
        The input and output observations in nominal tape order.
    violations : tuple of tuples of int
        One nonnegative profile per original arc occurrence. Initial and
        final marks have zero violations.
    caps : tuple of int
        A nonnegative finite saturation limit per rank. A zero cap
        ignores the corresponding coordinate.
    max_rank_states : int
        Maximum reached states during compilation for all inputs.
    max_rank_work : int
        Maximum combined weighted product and competitor construction work.
    max_states : int
        Maximum visited states during the fixed-word query.
    max_work : int
        Maximum examined arcs during the fixed-word query.
    max_closure_size : int or None
        Largest supported productive closure in the fixed-word query.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    violations: tuple[tuple[int, ...], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    caps: tuple[int, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_rank_states: int
    max_rank_work: int
    max_states: int
    max_work: int
    max_closure_size: int | None = None

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the two tapes, algebra, rank data and explicit limits.

        Raises
        ------
        TypeError
            The relation lacks one input and one output tape.
        ValueError
            An observation, limit, algebra, rank vector or cap is invalid.
        """
        _validate_fixed_observation(
            self.relation,
            self.words,
            self.max_states,
            self.max_work,
            self.max_closure_size,
        )
        _validate_limits(self.max_rank_states, self.max_rank_work)
        _validate_ranks(self.relation.topology, self.violations, len(self.caps))
        if any(type(cap) is not int or cap < 0 for cap in self.caps):
            raise ValueError(
                "ranked relation requests require nonnegative integer caps"
            )

    @property
    def handler_name(self) -> str:
        """Return the foreign name for the exact ranked observation.

        Returns
        -------
        str
            A digest of the ordered original topology, words, violations,
            and caps. Limits are independent invocation settings.
        """
        from quivers.qiec.serialization import dumps

        owner = (self.relation, self.words, self.violations, self.caps)
        digest = sha256(dumps(owner).encode("utf-8")).hexdigest()
        return f"{_RANKED_OT_HANDLER_PREFIX}{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Return the scalar type of the winning-path pair weight.

        Returns
        -------
        TypeExpr
            `LogWeight` for LogProb and `Real` for Markov.
        """
        return LOG_WEIGHT if self.relation.algebra == LOG_PROB.subject else REAL

    def plan(self, parameters: RelationWeightBindings) -> FixedStringQuery:
        """Compile the capped filter and read live weights in the pair plan.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The original scalar tensors by stable identity.

        Returns
        -------
        FixedStringQuery
            The reusable observation plan for the winning paths.

        Raises
        ------
        TypeError
            A parameter fails the original scalar binding contract.
        ValueError
            A parameter, compilation or fixed-word limit fails.
        """
        winner = compile_ranked_ot(
            self.relation.bind(parameters),
            self.violations,
            self.caps,
            max_states=self.max_rank_states,
            max_work=self.max_rank_work,
        )
        return winner.query(
            self.words,
            max_states=self.max_states,
            max_work=self.max_work,
            max_closure_size=self.max_closure_size,
        )


__all__ = ["FixedRankedOTRelationRequest"]
