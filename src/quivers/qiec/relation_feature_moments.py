"""Checked fixed-word requests for sparse log feature moments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from hashlib import sha256
from math import isfinite

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import FixedStringQuery, RelationLimitError, Words
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import RationalRelationSpec
from quivers.qiec.kinds import NAT
from quivers.qiec.types import IndexLiteral, TypeExpr

type _LogTable = tuple[tuple[float | None, ...], ...]


class FixedRelationFeatureObservation(QiecModel):
    """A fixed typed relation observation and nonnegative local features.

    The original topology fixes the order of initial, arc, and final
    occurrences. Feature columns remain ordered even when some are
    identically zero. Each entry is a finite log increment or `None`
    for log zero. An omitted boundary table is all log zero.

    Parameters
    ----------
    relation : RationalRelationSpec
        The original typed topology and stable weight identities.
    words : Words
        The fixed words in declared tape order.
    feature_extent : int
        The positive number of feature columns.
    arc_logs, initial_logs, final_logs : tuple or None
        Log features by original arc or boundary occurrence.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    feature_extent: int
    arc_logs: _LogTable = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    initial_logs: _LogTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    final_logs: _LogTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_features(self) -> None:
        """Check the scalar algebra, observation and finite feature tables.

        Raises
        ------
        ValueError
            The observation, feature extent or a log increment is invalid.
        """
        if self.relation.algebra not in (MARKOV.subject, LOG_PROB.subject):
            raise ValueError("fixed feature observations require Markov or LogProb")
        self.relation.topology._validate_words(self.words)
        if type(self.feature_extent) is not int or self.feature_extent <= 0:
            raise ValueError("relation features require a positive feature extent")
        for name, table, size in (
            ("initial", self.initial_logs, len(self.relation.initial)),
            ("arc", self.arc_logs, len(self.relation.arcs)),
            ("final", self.final_logs, len(self.relation.final)),
        ):
            if table is None:
                continue
            if len(table) != size or any(
                len(row) != self.feature_extent
                or any(value is not None and not isfinite(value) for value in row)
                for row in table
            ):
                raise ValueError(
                    f"invalid {name} log feature table for this observation"
                )

    @property
    def handler_name(self) -> str:
        """Return the normal foreign handler name for this exact observation.

        Returns
        -------
        str
            A canonical name derived from the original topology, words
            and feature data.
        """
        from quivers.qiec.serialization import dumps

        digest = sha256(dumps(self).encode("utf-8")).hexdigest()
        return f"__relation_feature_moments_{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Return the fixed tensor type for the total and ordered moments.

        Returns
        -------
        TypeExpr
            A log-weight tensor with the total followed by each moment.
        """
        return tensor_type(
            LOG_WEIGHT, (IndexLiteral(value=self.feature_extent + 1, sort=NAT),)
        )

    @property
    def _occurrences(self) -> int:
        """Return the size of the original feature table in rows.

        Returns
        -------
        int
            The number of initial, arc and final occurrences.
        """
        relation = self.relation
        return len(relation.initial) + len(relation.arcs) + len(relation.final)


def _feature_tensor(
    values: _LogTable | None, width: int, anchor: torch.Tensor
) -> torch.Tensor | None:
    """Convert a checked wire log table in the original weight domain.

    Parameters
    ----------
    values : tuple or None
        Checked log rows in original occurrence order.
    width : int
        The number of feature columns, including when there are no rows.
    anchor : torch.Tensor
        The weight dtype and device.

    Returns
    -------
    torch.Tensor or None
        A constant torch log table or the omitted boundary group.
    """
    if values is None:
        return None
    return anchor.new_tensor(
        tuple(
            tuple(-torch.inf if value is None else value for value in row)
            for row in values
        )
    ).reshape(len(values), width)


def _moment_vector(
    query: FixedStringQuery,
    arcs: torch.Tensor,
    initial: torch.Tensor | None,
    final: torch.Tensor | None,
    *,
    allow_cyclic: bool,
) -> torch.Tensor:
    """Evaluate a bounded feature query with live original weights.

    Parameters
    ----------
    query : FixedStringQuery
        The checked reusable observation.
    arcs, initial, final : torch.Tensor or None
        Constant log features of the original occurrences.
    allow_cyclic : bool
        Whether the request permits certified scalar component closure.

    Returns
    -------
    torch.Tensor
        The log total followed by the ordered log weighted sums.
    """
    total, moments = query.log_feature_moments(
        arcs, initial_logs=initial, final_logs=final, allow_cyclic=allow_cyclic
    )
    return torch.cat((total.reshape(1), moments))


class FixedRelationFeatureMomentRequest(QiecModel):
    """Collect packed nonnegative feature moments for a fixed observation.

    The observation owns the foreign handler identity. Work limits may
    change without changing that identity. A feature-cell budget separately
    bounds the product of feature width and sparse graph records.

    Parameters
    ----------
    observation : FixedRelationFeatureObservation
        The exact typed topology, words, and feature tables.
    max_states : int
        The reached-state budget for this fixed observation.
    max_work : int
        The examined-arc budget for this fixed observation.
    max_feature_cells : int
        The budget for chart cells. This counts one mass plus the feature
        extent, multiplied by original occurrences and productive chart
        states and arcs.
    """

    observation: FixedRelationFeatureObservation = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_states: int
    max_work: int
    max_feature_cells: int

    @dx.model_validator()
    def _validate_limits(self) -> None:
        """Check the independent finite execution budgets.

        Raises
        ------
        ValueError
            A budget is negative or not an integer.
        """
        _validate_limits(self.max_states, self.max_work)
        if type(self.max_feature_cells) is not int or self.max_feature_cells < 0:
            raise ValueError("max_feature_cells must be a nonnegative integer")

    @property
    def handler_name(self) -> str:
        """Return the requested observation's canonical foreign handler name.

        Returns
        -------
        str
            The name the checked source declares for the observation.
        """
        return self.observation.handler_name

    @property
    def result_type(self) -> TypeExpr:
        """Return the log tensor type of the requested feature moments.

        Returns
        -------
        TypeExpr
            One log total and one log moment per feature.
        """
        return self.observation.result_type

    def plan(
        self, parameters: Mapping[ParameterId, torch.Tensor]
    ) -> Callable[[], torch.Tensor]:
        """Compile the bounded feature evaluator with original weight tensors.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The live primitive scalar weights by original identity.

        Returns
        -------
        callable
            A reusable evaluator of the log total and moments.

        Raises
        ------
        RelationLimitError
            The fixed observation exceeds a declared resource budget.
        """
        return self._plan(parameters, max_closure_size=None)

    def _plan(
        self,
        parameters: Mapping[ParameterId, torch.Tensor],
        *,
        max_closure_size: int | None,
    ) -> Callable[[], torch.Tensor]:
        """Compile the fixed query under the requested closure capability.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The original live scalar weights.
        max_closure_size : int or None
            The per-component order limit for a cyclic request.

        Returns
        -------
        callable
            The reusable vector of log total and unnormalized feature moments.

        Raises
        ------
        RelationLimitError
            The typed feature table or active chart exceeds its cell budget.
        """
        observation = self.observation
        columns = observation.feature_extent + 1
        original_cells = columns * observation._occurrences
        if original_cells > self.max_feature_cells:
            raise RelationLimitError("original feature cells exceed max_feature_cells")
        relation = observation.relation.bind(parameters)
        query = relation.query(
            observation.words,
            max_states=self.max_states,
            max_work=self.max_work,
            max_closure_size=max_closure_size,
        )
        work = query.work
        if (
            original_cells + columns * (work.active_states + work.active_arcs)
            > self.max_feature_cells
        ):
            raise RelationLimitError("packed feature cells exceed max_feature_cells")
        anchor = relation._weights()
        converted = tuple(
            _feature_tensor(table, observation.feature_extent, anchor)
            for table in (
                observation.arc_logs,
                observation.initial_logs,
                observation.final_logs,
            )
        )
        arcs, initial, final = converted
        assert arcs is not None
        return partial(
            _moment_vector,
            query,
            arcs,
            initial,
            final,
            allow_cyclic=max_closure_size is not None,
        )


class FixedCyclicRelationFeatureMomentRequest(QiecModel):
    """Collect unnormalized nonnegative moments with certified scalar closure.

    The original observation owns the same foreign name and result type
    in either execution mode. This request supplies the extra operation
    that a productive cyclic component requires.

    Parameters
    ----------
    request : FixedRelationFeatureMomentRequest
        The observation and its fixed state, arc and feature-cell bounds.
    max_closure_size : int
        The maximum order of each productive scalar component closure.
    """

    request: FixedRelationFeatureMomentRequest = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_closure_size: int

    @dx.model_validator()
    def _validate_closure_size(self) -> None:
        """Require the explicit additional scalar closure budget.

        Raises
        ------
        ValueError
            The per-component order is negative or is not an integer.
        """
        if type(self.max_closure_size) is not int or self.max_closure_size < 0:
            raise ValueError("max_closure_size must be a nonnegative integer")

    @property
    def observation(self) -> FixedRelationFeatureObservation:
        """Return the unchanged typed observation.

        Returns
        -------
        FixedRelationFeatureObservation
            The original topology, words and fixed feature increments.
        """
        return self.request.observation

    @property
    def handler_name(self) -> str:
        """Return the canonical foreign handler of the original observation.

        Returns
        -------
        str
            The checked handler name, independent of closure limits.
        """
        return self.request.handler_name

    @property
    def result_type(self) -> TypeExpr:
        """Return the unchanged closed log tensor result type.

        Returns
        -------
        TypeExpr
            One log total followed by one unnormalized log moment per feature.
        """
        return self.request.result_type

    def plan(
        self, parameters: Mapping[ParameterId, torch.Tensor]
    ) -> Callable[[], torch.Tensor]:
        """Bind live weights and prepare the certified cyclic feature query.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The original live scalar weights.

        Returns
        -------
        callable
            The reusable result evaluator. Divergent or uncertifiable
            scalar closures report their failure at evaluation.
        """
        return self.request._plan(parameters, max_closure_size=self.max_closure_size)


__all__ = [
    "FixedCyclicRelationFeatureMomentRequest",
    "FixedRelationFeatureMomentRequest",
    "FixedRelationFeatureObservation",
]
