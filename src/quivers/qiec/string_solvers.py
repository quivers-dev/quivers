"""Checked records for distinct string factor graph objectives."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    FiniteStringBeliefPlan,
    PairedAlignmentMapResult,
    PrunedStringBeliefResult,
    RegularStringEPResult,
    StringAlignmentMapResult,
    dual_alignment_map,
    paired_dual_alignment_map,
    regular_string_ep,
)
from quivers.core.relations.alignment_groups import _validate_groups
from quivers.core.relations.beliefs import _validate_pruned_settings, _validate_sweeps
from quivers.core.relations.projection import _validate_projection_settings
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import RationalRelationSpec, RelationWeightBindings
from quivers.qiec.string_graphs import StringFactorGraphSpec


__all__ = [
    "JointAlignmentMapConfig",
    "JointAlignmentMapDecomposition",
    "JointAlignmentMapGapConfig",
    "PrunedStringBPConfig",
    "RegularStringEPConfig",
]


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
        Transition examinations allowed per exact factor decode or
        individual verification scan. The independent final check
        has two such scans and a separate original trace check.
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
        *,
        absolute_gap: Fraction | int | None = None,
        relative_gap: Fraction | int | None = None,
    ) -> StringAlignmentMapResult:
        """Run exact local decoding on a bound graph and return its search result.

        Parameters
        ----------
        model : StringFactorGraphSpec
            The checked string factors, variables, and observations.
        parameters : Mapping of ParameterId to torch.Tensor
            The original scalar tensors used by the model.
        absolute_gap, relative_gap : Fraction or int or None
            Optional exact objective gap targets. Serialize these using
            `JointAlignmentMapGapConfig` when they form part of the policy.

        Returns
        -------
        StringAlignmentMapResult
            The feasible assignment, bound, and certificate when available.
        """
        return dual_alignment_map(
            model.bind(parameters, max_states=self.max_states, max_work=self.max_work),
            max_iterations=self.max_iterations,
            max_ngram_order=self.max_ngram_order,
            max_features=self.max_features,
            max_states=self.max_states,
            max_work=self.max_work,
            step_size=Fraction(*self.step_ratio),
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
        )


class JointAlignmentMapGapConfig(QiecModel):
    """Configure proved objective-gap stopping for exact local alignment MAP.

    The plain MAP configuration retains its original encoding. This
    record adds one or both exact tolerances alongside it. Reaching
    the requested bounds can end a disagreeing run without producing
    an agreement certificate.

    Parameters
    ----------
    solver : JointAlignmentMapConfig
        The exact local decoder, feature schedule and resource settings.
    absolute_gap_ratio : tuple of int or None
        Nonnegative numerator and positive denominator for absolute
        suboptimality. `None` specifies no absolute constraint.
    relative_gap_ratio : tuple of int or None
        Nonnegative ratio for suboptimality relative to the optimal
        objective's magnitude. `None` specifies no relative constraint.
    """

    solver: JointAlignmentMapConfig = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    absolute_gap_ratio: tuple[int, int] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    relative_gap_ratio: tuple[int, int] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_targets(self) -> None:
        """Check each exact tolerance and require at least one target.

        Raises
        ------
        ValueError
            No target was supplied, or a target is not a nonnegative
            rational represented with a positive denominator.
        """
        if self.absolute_gap_ratio is None and self.relative_gap_ratio is None:
            raise ValueError("alignment MAP gap settings require at least one target")
        for ratio in (self.absolute_gap_ratio, self.relative_gap_ratio):
            if ratio is not None:
                numerator, denominator = ratio
                if (
                    type(numerator) is not int
                    or type(denominator) is not int
                    or numerator < 0
                    or denominator <= 0
                ):
                    raise ValueError(
                        "MAP objective gap targets must be nonnegative rational ratios"
                    )

    @property
    def objective(self) -> Literal["joint_alignment_map"]:
        """Identify the unchanged objective of the exact local solver.

        Returns
        -------
        Literal["joint_alignment_map"]
            Joint maximization over words and local paths.
        """
        return self.solver.objective

    def solve(
        self,
        model: StringFactorGraphSpec,
        parameters: Mapping[ParameterId, torch.Tensor],
    ) -> StringAlignmentMapResult:
        """Run alignment MAP until agreement, the proved target, or a limit.

        Parameters
        ----------
        model : StringFactorGraphSpec
            The checked graph, separate from the solver configuration.
        parameters : Mapping of ParameterId to torch.Tensor
            Original scalar parameters used by the graph's factors.

        Returns
        -------
        StringAlignmentMapResult
            The feasible assignment and upper bound, whether the gap
            target was established, and any agreement certificate.
        """
        absolute, relative = self._targets()
        return self.solver.solve(
            model, parameters, absolute_gap=absolute, relative_gap=relative
        )

    def _targets(self) -> tuple[Fraction | None, Fraction | None]:
        """Decode the stored ratios without floating-point conversion.

        Returns
        -------
        tuple of Fraction or None
            The absolute and relative targets in that order.
        """
        return (
            None
            if self.absolute_gap_ratio is None
            else Fraction(*self.absolute_gap_ratio),
            None
            if self.relative_gap_ratio is None
            else Fraction(*self.relative_gap_ratio),
        )


class JointAlignmentMapDecomposition(QiecModel):
    """Configure exact singleton and one-shared-tape paired subproblems.

    Factor indices refer to occurrences in the separate checked graph.
    Equal factor definitions can appear at different positions. The
    group order specifies multiplier scope order. Returned original
    traces follow the graph's original factor order.

    Parameters
    ----------
    solver : JointAlignmentMapConfig or JointAlignmentMapGapConfig
        The search, local bounds, and optional gap policy.
    groups : tuple of tuples of int
        An ordered partition of all original factor occurrence indices
        into singletons and pairs sharing exactly one tape variable.
    max_group_states : int
        Maximum states constructed by each filtered pair join.
    max_group_work : int
        Maximum label probes and arcs constructed by each pair join.
    """

    solver: JointAlignmentMapConfig | JointAlignmentMapGapConfig = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    groups: tuple[tuple[int, ...], ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_group_states: int
    max_group_work: int

    @dx.model_validator()
    def _validate_decomposition(self) -> None:
        """Check the portable policy before binding a specific graph.

        Raises
        ------
        ValueError
            A factor group, occurrence index or resource limit is invalid.
        """
        _validate_groups(self.groups)
        _validate_limits(self.max_group_states, self.max_group_work)

    @property
    def objective(self) -> Literal["joint_alignment_map"]:
        """Identify the exact objective shared with the singleton solver.

        Returns
        -------
        Literal["joint_alignment_map"]
            Joint maximization over full words and original local paths.
        """
        return self.solver.objective

    def solve(
        self,
        model: StringFactorGraphSpec,
        parameters: Mapping[ParameterId, torch.Tensor],
    ) -> PairedAlignmentMapResult:
        """Bind the graph and run the verified grouped local search.

        Parameters
        ----------
        model : StringFactorGraphSpec
            The unchanged checked graph and original occurrence order.
        parameters : Mapping of ParameterId to torch.Tensor
            Original live LogProb scalar owners for the factor graph.

        Returns
        -------
        PairedAlignmentMapResult
            Grouped local evidence and feasible traces in original factor order.
        """
        solver = self.solver
        absolute = relative = None
        if isinstance(solver, JointAlignmentMapGapConfig):
            absolute, relative = solver._targets()
            solver = solver.solver
        return paired_dual_alignment_map(
            model.bind(
                parameters, max_states=solver.max_states, max_work=solver.max_work
            ),
            groups=self.groups,
            max_group_states=self.max_group_states,
            max_group_work=self.max_group_work,
            max_iterations=solver.max_iterations,
            max_ngram_order=solver.max_ngram_order,
            max_features=solver.max_features,
            max_states=solver.max_states,
            max_work=solver.max_work,
            step_size=Fraction(*solver.step_ratio),
            absolute_gap=absolute,
            relative_gap=relative,
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
        graph = model.bind(
            parameters, max_states=self.max_states, max_work=self.max_work
        )
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


class PrunedStringBPConfig(QiecModel):
    """Configure a beam-pruned schedule on declared finite string supports.

    Supports describe the finite candidate pool and are kept separate
    from the typed graph. The configuration contains only inputs and
    limits. A run derives the convergence status and raw, retained and
    omitted message masses from the current specified graph and weights.

    Parameters
    ----------
    supports : tuple
        Variable names and their candidate words in result order.
    beam : int
        Positive limit on the number of retained words per message.
    max_sweeps : int
        Positive number of permitted synchronous updates.
    max_entries : int
        Maximum total entries in the bound finite factor tables.
    max_states : int
        Cumulative observed configuration budget when building the tables.
    max_work : int
        Cumulative arc examination budget when building the tables.
    max_closure_size : int or None
        Optional size bound for productive cyclic closure components.
    tolerance : float or None
        Threshold for undamped pruned residual, or None for fixed sweeps.
    damping : float
        Old-message mixing fraction after restriction to the new support.
    """

    supports: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    beam: int
    max_sweeps: int
    max_entries: int
    max_states: int
    max_work: int
    max_closure_size: int | None = None
    tolerance: float | None = 1e-9
    damping: float = 0.0

    @dx.model_validator()
    def _validate_settings(self) -> None:
        """Check declared finite words and the beam and construction limits.

        Raises
        ------
        ValueError
            The supports, sweep parameters, or resource limits are invalid.
        """
        _validate_pruned_settings(
            self.beam, self.max_sweeps, self.tolerance, self.damping
        )
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)
        if type(self.max_entries) is not int or self.max_entries < 0:
            raise ValueError("max_entries must be a nonnegative integer")
        names: set[str] = set()
        if not isinstance(self.supports, tuple) or not self.supports:
            raise ValueError("pruned string BP requires declared finite supports")
        for row in self.supports:
            if not isinstance(row, tuple) or len(row) != 2:
                raise ValueError(
                    "finite supports require ordered name and word records"
                )
            name, words = row
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(
                    "finite supports require distinct nonempty variable names"
                )
            names.add(name)
            if (
                not isinstance(words, tuple)
                or not words
                or any(
                    not isinstance(word, tuple)
                    or any(type(symbol) is not int or symbol < 0 for symbol in word)
                    for word in words
                )
            ):
                raise ValueError(
                    "each finite support requires words of nonnegative integers"
                )
            if len(words) != len(set(words)):
                raise ValueError("each finite support requires distinct declared words")

    @property
    def objective(self) -> Literal["pruned_string_bp"]:
        """Identify beam-pruned finite sum-product inference.

        Returns
        -------
        Literal["pruned_string_bp"]
            The declared approximate schedule.
        """
        return "pruned_string_bp"

    def infer(
        self,
        model: StringFactorGraphSpec,
        parameters: RelationWeightBindings,
    ) -> PrunedStringBeliefResult:
        """Bind the checked graph and evaluate the declared beam schedule.

        Parameters
        ----------
        model : StringFactorGraphSpec
            The original typed graph, variable identities, and observations.
        parameters : RelationWeightBindings
            Original weight owners and permitted Markov complements.

        Returns
        -------
        PrunedStringBeliefResult
            Derived approximate beliefs and measured diagnostics.
        """
        return self._plan(model, parameters).run_pruned(
            beam=self.beam,
            max_sweeps=self.max_sweeps,
            tolerance=self.tolerance,
            damping=self.damping,
        )

    def _plan(
        self,
        model: StringFactorGraphSpec,
        parameters: RelationWeightBindings,
    ) -> FiniteStringBeliefPlan:
        """Compile the declared finite model and prepare live factor queries.

        Parameters
        ----------
        model : StringFactorGraphSpec
            The checked graph whose topology is retained.
        parameters : RelationWeightBindings
            Original tensors and permitted Markov complements.

        Returns
        -------
        FiniteStringBeliefPlan
            A reusable finite pruned schedule with the configured limits.
        """
        return FiniteStringBeliefPlan(
            model.bind(parameters, max_states=self.max_states, max_work=self.max_work),
            dict(self.supports),
            max_entries=self.max_entries,
            max_states=self.max_states,
            max_work=self.max_work,
            max_closure_size=self.max_closure_size,
        )
