"""Checked records for two distinct string factor graph objectives."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    RegularStringEPResult,
    StringAlignmentMapResult,
    dual_alignment_map,
    regular_string_ep,
)
from quivers.core.relations.beliefs import _validate_sweeps
from quivers.core.relations.projection import _validate_projection_settings
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import RationalRelationSpec
from quivers.qiec.string_graphs import StringFactorGraphSpec


__all__ = ["JointAlignmentMapConfig", "RegularStringEPConfig"]


class JointAlignmentMapConfig(QiecModel):
    """Configure exact local decoders for joint alignment and word MAP.

    The record travels separately from the factor graph and its parameters.
    Its decoder capability requires acyclic productive LogProb factors.
    Only an execution whose exact local paths agree on complete words can
    return an optimum certificate. A finite budget can end in disagreement.

    Parameters
    ----------
    max_iterations : int
        Positive bound on updates of the dual variables.
    max_ngram_order : int
        Highest order of a distinguishing feature.
    max_features : int
        Maximum number of active variable-feature pairs.
    max_states : int
        Configuration limit per exact factor decode.
    max_work : int
        Transition examinations allowed per exact factor decode.
    step_ratio : tuple of int
        Positive numerator and denominator of the rational multiplier step.
    """

    max_iterations: int
    max_ngram_order: int
    max_features: int
    max_states: int
    max_work: int
    step_ratio: tuple[int, int] = dx.field(
        default=(1, 1), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_settings(self) -> None:
        """Check the exact decoder and rational update settings.

        Raises
        ------
        ValueError
            If an iteration, n-gram, resource, or step bound is invalid.
        """
        if self.max_iterations < 1 or self.max_ngram_order < 1:
            raise ValueError("MAP requires positive iteration and n-gram order bounds")
        if self.max_features < 0:
            raise ValueError("MAP requires a nonnegative feature bound")
        if any(value <= 0 for value in self.step_ratio):
            raise ValueError("MAP requires a positive step numerator and denominator")
        _validate_limits(self.max_states, self.max_work)

    @property
    def objective(self) -> Literal["joint_alignment_map"]:
        """Identify the objective selected by this record.

        Returns
        -------
        Literal["joint_alignment_map"]
            Joint maximization over words and local paths.
        """
        return "joint_alignment_map"

    def solve(
        self,
        model: StringFactorGraphSpec,
        parameters: Mapping[ParameterId, torch.Tensor],
    ) -> StringAlignmentMapResult:
        """Run exact local decoding on a bound graph and return its search result.

        Parameters
        ----------
        model : StringFactorGraphSpec
            The checked string factors, variables, and observations.
        parameters : Mapping of ParameterId to torch.Tensor
            The original scalar tensors used by the model.

        Returns
        -------
        StringAlignmentMapResult
            The feasible assignment, bound, and certificate when available.
        """
        return dual_alignment_map(
            model.bind(parameters),
            max_iterations=self.max_iterations,
            max_ngram_order=self.max_ngram_order,
            max_features=self.max_features,
            max_states=self.max_states,
            max_work=self.max_work,
            step_size=Fraction(*self.step_ratio),
        )


class RegularStringEPConfig(QiecModel):
    """Configure synchronous inclusive-KL projection of regular string messages.

    Each variable has a one-tape LogProb reference language and an ordered
    feature vocabulary. Reference weights enter the target distribution.
    Unit reference weights give a support restriction of the original
    factor model. The solver snapshots weights and returns approximate
    beliefs, without a derivative through the schedule or its fixed point.

    Parameters
    ----------
    bases : tuple
        Variable names paired with finite regular reference specifications.
    features : tuple
        Variable names paired with their candidate n-grams in coefficient order.
    penalty : float
        Nonnegative strength of the n-gram prefix-group penalty.
    max_sweeps : int
        Maximum accepted synchronous sweeps.
    max_states : int
        State limit per sparse relation operation.
    max_work : int
        Work limit per sparse relation operation.
    max_closure_size : int or None
        Maximum size of a productive closed component, if permitted.
    tolerance : float or None
        Upper bound on the undamped site change, or None for the full budget.
    damping : float
        Fraction of the old site retained at each update.
    projection_tolerance : float
        Tolerance of the local proximal-gradient solver.
    projection_max_iterations : int
        Accepted iteration limit for each local projection.
    """

    bases: tuple[tuple[str, RationalRelationSpec], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    features: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    penalty: float
    max_sweeps: int
    max_states: int
    max_work: int
    max_closure_size: int | None = None
    tolerance: float | None = 1e-8
    damping: float = 0.0
    projection_tolerance: float = 1e-8
    projection_max_iterations: int = 512

    @dx.model_validator()
    def _validate_settings(self) -> None:
        """Check projection settings and the declarations of reference support.

        Raises
        ------
        ValueError
            If declarations, projection settings, or resource limits are invalid.
        TypeError
            If a reference fails the one-tape LogProb contract.
        """
        _validate_sweeps(self.max_sweeps, self.tolerance, self.damping)
        _validate_projection_settings(
            self.projection_tolerance, self.projection_max_iterations, 0
        )
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)
        if not 0 <= self.penalty < float("inf"):
            raise ValueError("EP requires a nonnegative finite penalty")
        bases = dict(self.bases)
        features = dict(self.features)
        if (
            not bases
            or "" in bases
            or len(bases) != len(self.bases)
            or len(features) != len(self.features)
            or bases.keys() != features.keys()
        ):
            raise ValueError("EP requires one base and vocabulary per named variable")
        for variable, base in bases.items():
            if len(base.tapes) != 1 or base.algebra != LOG_PROB.subject:
                raise TypeError("EP requires one-tape LogProb reference languages")
            _, size = base.tapes[0]
            grams = features[variable]
            if len(grams) != len(set(grams)) or any(
                not gram or any(not 0 <= symbol < size for symbol in gram)
                for gram in grams
            ):
                raise ValueError(
                    f"EP features for {variable!r} must be distinct nonempty words in the reference alphabet"
                )

    @property
    def objective(self) -> Literal["regular_string_ep"]:
        """Identify the objective selected by this record.

        Returns
        -------
        Literal["regular_string_ep"]
            Inclusive-KL projection of regular string messages.
        """
        return "regular_string_ep"

    def infer(
        self,
        model: StringFactorGraphSpec,
        parameters: Mapping[ParameterId, torch.Tensor],
    ) -> RegularStringEPResult:
        """Run projected sum-product on a bound graph and its reference languages.

        Parameters
        ----------
        model : StringFactorGraphSpec
            The checked string factors, variables, and observations.
        parameters : Mapping of ParameterId to torch.Tensor
            The original scalar tensors used by the graph and its references.

        Returns
        -------
        RegularStringEPResult
            The approximate marginals, site scales, and convergence diagnostics.
        """
        graph = model.bind(parameters)
        return regular_string_ep(
            graph,
            {name: spec.bind(parameters) for name, spec in self.bases},
            dict(self.features),
            penalty=self.penalty,
            max_sweeps=self.max_sweeps,
            max_states=self.max_states,
            max_work=self.max_work,
            max_closure_size=self.max_closure_size,
            tolerance=self.tolerance,
            damping=self.damping,
            projection_tolerance=self.projection_tolerance,
            projection_max_iterations=self.projection_max_iterations,
        )
