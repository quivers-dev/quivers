"""Stable sparse relation specifications and their process-local weight bindings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from typing import Literal, cast

import didactic.api as dx
import torch
from torch import nn

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB, MARKOV, Semiring
from quivers.core.laws import ClaimSubject
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import (
    Arc,
    Boundary,
    FixedStringQuery,
    Label,
    RationalRelation,
    RelationTopology,
    StaticMinimizationWork,
    Words,
    minimize_static_acceptor_with_work,
)
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.kinds import NAT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.foreign import KERNEL_ALGEBRAS
from quivers.qiec.types import REAL, IndexLiteral, TypeExpr


@dataclass(frozen=True, slots=True)
class RelationWeightComplement:
    """Use one minus an original probability in a Markov weight slot.

    Parameters
    ----------
    owner : ParameterId
        The original scalar leaf in the same binding mapping.
    """

    owner: ParameterId


type RelationWeightBindings = Mapping[
    ParameterId, torch.Tensor | RelationWeightComplement
]
"""Process-local bindings of sparse relation weight identities."""


def _weight_bindings(
    identities: Iterable[ParameterId],
    parameters: RelationWeightBindings,
    *,
    description: str,
) -> tuple[dict[ParameterId, torch.Tensor], dict[ParameterId, ParameterId]]:
    """Resolve original leaves and explicit unit probability complements.

    Parameters
    ----------
    identities : Iterable of ParameterId
        Weight identities used by the checked topology or collection.
    parameters : RelationWeightBindings
        The common store of original leaves and complement declarations.
    description : str
        The checked relation or collection named in refusal diagnostics.

    Returns
    -------
    tuple
        Original owner tensors and a map from dependent slots to owners.

    Raises
    ------
    TypeError
        If an original parameter lacks the checked tensor representation.
    ValueError
        If a binding is missing, an owner is itself a complement, or
        original owners alias or disagree on dtype or device.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{description} weights require an explicit parameter mapping")
    owners: dict[ParameterId, torch.Tensor] = {}
    complements: dict[ParameterId, ParameterId] = {}
    for identity in dict.fromkeys(identities):
        if identity not in parameters:
            raise ValueError(f"missing {description} parameter {identity}")
        value = parameters[identity]
        owner = identity
        if isinstance(value, RelationWeightComplement):
            owner = value.owner
            if owner not in parameters or isinstance(
                parameters[owner], RelationWeightComplement
            ):
                raise ValueError(
                    f"complement {identity} requires an original relation owner {owner}"
                )
            complements[identity] = owner
            value = parameters[owner]
        if (
            not isinstance(value, torch.Tensor)
            or type(value) not in (torch.Tensor, nn.Parameter)
            or value.ndim != 0
            or not value.is_leaf
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.is_quantized
        ):
            raise TypeError(
                f"relation parameter {owner} requires a dense scalar leaf tensor"
            )
        if value.requires_grad and not isinstance(value, nn.Parameter):
            raise TypeError(
                f"trainable relation parameter {owner} requires torch.nn.Parameter"
            )
        owners[owner] = value
    if len({id(value) for value in owners.values()}) != len(owners):
        raise ValueError(
            f"shared {description} weights must use the same parameter identity"
        )
    if len({(value.dtype, value.device) for value in owners.values()}) > 1:
        raise ValueError(f"{description} parameters require a common dtype and device")
    return owners, complements


class _RelationWeights(nn.Module):
    """Expose live scalar tensors in the relation's weight-slot order.

    Parameters
    ----------
    slots : tuple of ParameterId
        The parameter identity at each weight slot.
    values : Mapping of ParameterId to torch.Tensor
        The original tensor for each distinct identity.
    complements : Mapping of ParameterId to ParameterId
        Explicit dependent slots and their original owners.
    """

    def __init__(
        self,
        slots: tuple[ParameterId, ...],
        values: Mapping[ParameterId, torch.Tensor],
        complements: Mapping[ParameterId, ParameterId],
    ) -> None:
        super().__init__()
        self._slots = tuple(
            (f"weight_{complements.get(id_, id_).digest}", id_ in complements)
            for id_ in slots
        )
        for id_, value in values.items():
            name = f"weight_{id_.digest}"
            if isinstance(value, nn.Parameter):
                self.register_parameter(name, value)
            else:
                self.register_buffer(name, value)

    def forward(self) -> torch.Tensor:
        """Read current weights without changing their parameter identities.

        Returns
        -------
        torch.Tensor
            One value per slot, including repeated uses of shared parameters.

        Raises
        ------
        ValueError
            If a complement's current owner falls outside the unit interval.
        """
        if not self._slots:
            return torch.empty(0)
        derived: dict[str, torch.Tensor] = {}
        values = []
        for name, complement in self._slots:
            value = cast(torch.Tensor, getattr(self, name))
            if complement:
                if name not in derived:
                    if not value.is_floating_point():
                        raise TypeError(
                            "a relation weight complement requires a floating-point owner"
                        )
                    if not bool((0 <= value) & (value <= 1)):
                        raise ValueError(
                            "a relation weight complement requires an owner in [0, 1]"
                        )
                    derived[name] = 1 - value
                value = derived[name]
            values.append(value)
        return torch.stack(values)


class RationalRelationSpec(QiecModel):
    """An ordered sparse topology with a stable parameter identity for each slot.

    Tape records contain the alphabet's nominal name and cardinality,
    in tape order. Arc records contain source, target, labels, and a
    weight slot, in occurrence order. Initial and final records contain
    a state and a weight slot. A label of None denotes epsilon.

    The order of `weight_names` names the local slots. The corresponding
    entries of `weight_parameters` can repeat a ParameterId. This ties
    weight values without identifying arc or boundary occurrences.
    `algebra` states the trusted claim subject of a shipped scalar semiring.
    Live values are supplied to `bind` after decoding.

    Parameters
    ----------
    tapes : tuple of pairs of str and int
        The nominal alphabet name and cardinality of each tape.
    input_tapes : int
        The number of leading tapes in the domain.
    n_states : int
        The number of states, indexed from zero.
    arcs : tuple
        Source, target, tape labels, and weight slot for each arc occurrence.
    initial : tuple of pairs of int
        Initial-state and weight-slot records.
    final : tuple of pairs of int
        Final-state and weight-slot records.
    weight_names : tuple of str
        The distinct names of the relation's local weight slots.
    weight_parameters : tuple of ParameterId
        The stable parameter identity for each slot.
    algebra : ClaimSubject
        The trusted identity of the scalar semiring.
    """

    tapes: tuple[tuple[str, int], ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    input_tapes: int
    n_states: int
    arcs: tuple[tuple[int, int, tuple[Label, ...], int], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    initial: tuple[tuple[int, int], ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    final: tuple[tuple[int, int], ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    weight_names: tuple[str, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    weight_parameters: tuple[ParameterId, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    algebra: ClaimSubject

    @dx.model_validator()
    def _validate_spec(self) -> None:
        """Validate decoded topology and parameter identities.

        Raises
        ------
        ValueError
            If the algebra is unrecognized or the slot and parameter counts differ.
        """
        if self._algebra is None:
            raise ValueError(
                "relation requires a trusted scalar semiring claim subject"
            )
        if len(self.weight_parameters) != len(self.weight_names):
            raise ValueError("relation requires one parameter identity per weight slot")
        _ = self.topology

    @property
    def _algebra(self) -> Semiring | None:
        """Resolve the complete claim subject against the kernel registry.

        Returns
        -------
        Semiring or None
            The registered scalar semiring, if its claim subject matches.
        """
        member = KERNEL_ALGEBRAS.get(self.algebra.name)
        return member if member is not None and member.subject == self.algebra else None

    @classmethod
    def of(
        cls,
        topology: RelationTopology,
        algebra: Semiring,
        weight_parameters: tuple[ParameterId, ...],
    ) -> RationalRelationSpec:
        """Describe a sparse topology and its weight identities as QIEC data.

        The topology uses the shipped string sets, finite alphabets, arcs
        and boundaries. The algebra must be a shipped scalar semiring
        singleton from the kernel registry.

        Parameters
        ----------
        topology : RelationTopology
            The ordered sparse topology to encode.
        algebra : Semiring
            The registered scalar semiring of the relation.
        weight_parameters : tuple of ParameterId
            One stable identity per weight slot.

        Returns
        -------
        RationalRelationSpec
            The validated specification.

        Raises
        ------
        TypeError
            If the topology representation or algebra is unsupported.
        """
        if (
            type(topology) is not RelationTopology
            or any(
                type(tape) is not StringSet or type(tape.alphabet) is not FinSet
                for tape in topology.tapes
            )
            or any(type(arc) is not Arc for arc in topology.arcs)
            or any(
                type(mark) is not Boundary
                for mark in (*topology.initial, *topology.final)
            )
        ):
            raise TypeError("relation topology uses an unsupported representation")
        if (
            not isinstance(algebra, Semiring)
            or KERNEL_ALGEBRAS.get(algebra.subject.name) is not algebra
        ):
            raise TypeError("relation requires a shipped scalar semiring")
        return cls(
            tapes=tuple(
                (tape.alphabet.name, tape.alphabet.cardinality)
                for tape in topology.tapes
            ),
            input_tapes=topology.input_tapes,
            n_states=topology.n_states,
            arcs=tuple(
                (arc.source, arc.target, arc.labels, arc.weight)
                for arc in topology.arcs
            ),
            initial=tuple((mark.state, mark.weight) for mark in topology.initial),
            final=tuple((mark.state, mark.weight) for mark in topology.final),
            weight_names=topology.weight_names,
            weight_parameters=weight_parameters,
            algebra=algebra.subject,
        )

    @cached_property
    def topology(self) -> RelationTopology:
        """Reconstruct the topology and its sparse indexes.

        Returns
        -------
        RelationTopology
            The validated, indexed topology of the specification.
        """
        return RelationTopology(
            tapes=tuple(
                StringSet(alphabet=FinSet(name=name, cardinality=size))
                for name, size in self.tapes
            ),
            input_tapes=self.input_tapes,
            n_states=self.n_states,
            arcs=tuple(Arc(*record) for record in self.arcs),
            initial=tuple(Boundary(*record) for record in self.initial),
            final=tuple(Boundary(*record) for record in self.final),
            weight_names=self.weight_names,
        )

    def bind(self, parameters: RelationWeightBindings) -> RationalRelation:
        """Bind original scalar leaves by stable identity and return a live relation.

        Values must be dense scalar leaf tensors in the algebra's
        carrier, with a common dtype and device. Trainable values must
        be original torch.nn.Parameter objects. Each distinct identity
        binds one object, and shared slots must use the same identity.
        The mapping may also provide identities for other relations.
        A Markov slot may explicitly use one minus an original unit
        probability. Its weight is recomputed from that leaf on every read.

        Parameters
        ----------
        parameters : RelationWeightBindings
            Original leaves and explicit complements by stable identity.

        Returns
        -------
        RationalRelation
            The sparse relation with live access to its bound weights.

        Raises
        ------
        TypeError
            If the mapping or its tensors do not meet the binding contract.
        ValueError
            If a required identity is missing, two different identities bind
            one object, or tensors disagree on dtype or device.
        """
        values, complements = _weight_bindings(
            self.weight_parameters, parameters, description="relation"
        )
        algebra = self._algebra
        assert algebra is not None  # Checked when constructing the specification.
        if complements and algebra is not MARKOV:
            raise ValueError("relation weight complements require the Markov algebra")
        weights = _RelationWeights(self.weight_parameters, values, complements)
        with torch.no_grad():
            algebra.validate(weights())
        return RationalRelation(self.topology, weights, algebra=algebra)


class FixedRelationRequest(QiecModel):
    """A fixed observation and exact-work contract over a sparse topology.

    The words are in tape order. For `conditional_log_prob`, the
    topology's input split separates conditioning words from scored
    words. The `path_entropy` operation returns the conditional
    entropy of the accepting paths in nats. For `best_path_weight`,
    the highest-weight positive path supplies its original scalar
    weight. This objective requires an acyclic productive graph.
    The wire contains the
    request and stable relation specification. The provider binds
    its live scalar parameters.

    Parameters
    ----------
    relation : RationalRelationSpec
        The typed sparse topology and original weight identities.
    operation : str
        The fixed-pair weight, conditional log probability, path
        entropy, or best path weight objective.
    words : Words
        One observed word per tape, in the topology's order.
    max_states : int
        The budget for fixed-word state exploration.
    max_work : int
        The budget for fixed-word arc exploration.
    max_closure_size : int or None
        The largest permitted productive closure component, or `None`
        to require an acyclic query.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operation: Literal[
        "pair_weight", "conditional_log_prob", "path_entropy", "best_path_weight"
    ]
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_states: int
    max_work: int
    max_closure_size: int | None = None

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the budgets, tape observations, algebra, and objective.

        Raises
        ------
        ValueError
            An observation or resource limit is invalid, the algebra is
            unsupported, or a conditional request lacks input or output tapes.
        """
        _validate_fixed_observation(
            self.relation,
            self.words,
            self.max_states,
            self.max_work,
            self.max_closure_size,
        )
        if self.operation == "best_path_weight" and self.max_closure_size is not None:
            raise ValueError(
                "best path weight requires an acyclic query without closure"
            )
        topology = self.relation.topology
        if self.operation == "conditional_log_prob" and not (
            0 < topology.input_tapes < len(topology.tapes)
        ):
            raise ValueError(
                "conditional relation requests require input and output tapes"
            )

    @property
    def result_type(self) -> TypeExpr:
        """The checked scalar type of the request result.

        Returns
        -------
        TypeExpr
            `LogWeight` for a conditional or log-domain path or pair
            score, or `Real` for entropy and Markov path or pair scores.
        """
        if self.operation == "path_entropy":
            return REAL
        if (
            self.operation == "conditional_log_prob"
            or self.relation.algebra == LOG_PROB.subject
        ):
            return LOG_WEIGHT
        return REAL


class FixedMinimizedRelationRequest(QiecModel):
    """Score one declared acceptor through an exact minimized snapshot.

    The observation keeps the original authored topology and weight
    identities. Its fixed-word state and work limits apply to the
    quotient. The separate minimization limits apply to the source
    acceptor. Binding requires nontrainable weights and an exactly
    representable quotient in the dtype supplied by the caller.

    Parameters
    ----------
    observation : FixedRelationRequest
        The original static Markov acceptor and its fixed pair weight
        observation. The acceptor has one tape, deterministic symbols,
        and at most one initial state.
    max_minimization_states : int
        The source's productive state limit.
    max_minimization_work : int
        The source input record and refinement work limit.
    """

    observation: FixedRelationRequest = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_minimization_states: int
    max_minimization_work: int

    @dx.model_validator()
    def _validate_minimization(self) -> None:
        """Check the objective, static acceptor capability, and limits.

        Raises
        ------
        ValueError
            A limit is invalid or the observation lacks the required
            one-tape Markov pair weight or has epsilon closure settings.
        """
        _validate_limits(self.max_minimization_states, self.max_minimization_work)
        source = self.observation
        topology = source.relation.topology
        if (
            source.operation != "pair_weight"
            or source.relation.algebra != MARKOV.subject
            or len(topology.tapes) != 1
            or len(topology.initial) > 1
            or source.max_closure_size is not None
        ):
            raise ValueError(
                "minimized relation requests require a one-tape Markov pair_weight "
                "with at most one initial mark and no closure setting"
            )

    @property
    def result_type(self) -> TypeExpr:
        """Return the checked `Real` type of the unchanged pair-weight objective.

        Returns
        -------
        TypeExpr
            The scalar `Real` result type.
        """
        return self.observation.result_type

    def plan(
        self, parameters: RelationWeightBindings
    ) -> tuple[FixedStringQuery, StaticMinimizationWork]:
        """Snapshot the source, minimize it, and plan the declared observation.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The original nontrainable scalar leaves indexed by identity.

        Returns
        -------
        tuple[FixedStringQuery, StaticMinimizationWork]
            The fixed-word plan over detached quotient weights and the
            source minimization counters. Rebinding takes a new snapshot.

        Raises
        ------
        TypeError
            The source weights are unsupported or trainable.
        ValueError
            A binding is invalid, the acceptor is ambiguous or has
            epsilon arcs, a budget is exhausted, or a quotient weight
            is unrepresentable in the input dtype.
        """
        source = self.observation
        quotient, work = minimize_static_acceptor_with_work(
            source.relation.bind(parameters),
            max_states=self.max_minimization_states,
            max_work=self.max_minimization_work,
        )
        return quotient.query(
            source.words, max_states=source.max_states, max_work=source.max_work
        ), work


class FixedRelationCostRequest(QiecModel):
    """A fixed-word expectation of a signed additive path cost.

    Costs follow the order of original arc and boundary occurrences.
    Omitted groups contribute zero. The scalar Real result is the
    expectation under the normalized distribution of accepting paths.
    Weight parameters are bound live. The cost literals are fixed
    data in the wire record.

    Parameters
    ----------
    relation : RationalRelationSpec
        The typed topology and original scalar weight identities.
    words : Words
        One fixed word per tape, in topology order.
    max_states : int
        The fixed-word state budget.
    max_work : int
        The fixed-word arc examination budget.
    max_closure_size : int or None
        The largest permitted productive closure component.
    initial_costs : tuple of float or None
        One signed real cost per original initial mark, if supplied.
    arc_costs : tuple of float or None
        One signed real cost per original arc, if supplied.
    final_costs : tuple of float or None
        One signed real cost per original final mark, if supplied.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_states: int
    max_work: int
    max_closure_size: int | None = None
    initial_costs: tuple[float, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    arc_costs: tuple[float, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    final_costs: tuple[float, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_cost(self) -> None:
        """Check the finite signed costs against original occurrence counts.

        Raises
        ------
        ValueError
            If the observation is invalid, every cost group is absent,
            or a supplied group has invalid size or a nonfinite entry.
        """
        _validate_fixed_observation(
            self.relation,
            self.words,
            self.max_states,
            self.max_work,
            self.max_closure_size,
        )
        topology = self.relation.topology
        costs = self.initial_costs, self.arc_costs, self.final_costs
        if all(group is None for group in costs):
            raise ValueError("expected path cost requires at least one cost group")
        counts = len(topology.initial), len(topology.arcs), len(topology.final)
        for group, count in zip(costs, counts, strict=True):
            if group is not None and (
                len(group) != count or any(not isfinite(value) for value in group)
            ):
                raise ValueError(
                    "path costs require one finite value per original occurrence"
                )

    @property
    def operation(self) -> Literal["expected_path_cost"]:
        """Name the checked signed scalar objective.

        Returns
        -------
        str
            The expected additive path cost objective.
        """
        return "expected_path_cost"

    @property
    def result_type(self) -> TypeExpr:
        """Name the checked scalar result carrier.

        Returns
        -------
        TypeExpr
            Real, since additive costs may have either sign.
        """
        return REAL


class FixedRelationArcCountsRequest(QiecModel):
    """Return conditional traversal counts of selected original occurrences.

    Each tensor position names one original arc occurrence. Repeated
    request IDs repeat the corresponding expectation. Parallel arcs
    that share a parameter remain separate occurrences. Traversals of
    one original occurrence at several observed offsets, or repeatedly
    along a walk, all contribute to that occurrence's expectation.

    Parameters
    ----------
    relation : RationalRelationSpec
        The typed original topology and stable scalar weight identities.
    words : Words
        One observed word per tape in topology order.
    arc_ids : tuple of int
        The ordered original arc IDs whose counts the result exposes.
    max_states : int
        The reached-state limit for the observation.
    max_work : int
        The examined-arc limit for the observation.
    max_closure_size : int or None
        The permitted productive closure component size.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arc_ids: tuple[int, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_states: int
    max_work: int
    max_closure_size: int | None = None

    @dx.model_validator()
    def _validate_occurrences(self) -> None:
        """Validate the request and each ordered original occurrence ID.

        Raises
        ------
        ValueError
            The observation, limits, or original arc identities are invalid.
        """
        _validate_fixed_observation(
            self.relation,
            self.words,
            self.max_states,
            self.max_work,
            self.max_closure_size,
        )
        arc_count = len(self.relation.arcs)
        if not isinstance(self.arc_ids, tuple) or any(
            type(index) is not int or not 0 <= index < arc_count
            for index in self.arc_ids
        ):
            raise ValueError(
                "arc counts require an ordered tuple of valid original arc IDs"
            )

    @property
    def operation(self) -> Literal["expected_arc_counts"]:
        """Return the operation recognized by the reference machine.

        Returns
        -------
        str
            The checked traversal-count objective.
        """
        return "expected_arc_counts"

    @property
    def result_type(self) -> TypeExpr:
        """Return the fixed-length real tensor type for the requested positions.

        Returns
        -------
        TypeExpr
            One Real count per requested occurrence ID.
        """
        extent = IndexLiteral(value=len(self.arc_ids), sort=NAT)
        return tensor_type(REAL, (extent,))


def _bind_relation_operands(
    left_spec: RationalRelationSpec,
    right_spec: RationalRelationSpec,
    parameters: RelationWeightBindings,
    *,
    description: str,
) -> tuple[RationalRelation, RationalRelation]:
    """Bind two checked relations to one common parameter store.

    Parameters
    ----------
    left_spec, right_spec : RationalRelationSpec
        The ordered original operand specifications.
    parameters : Mapping of ParameterId to torch.Tensor
        The original scalar tensor owners.
    description : str
        The operation name for binding diagnostics.

    Returns
    -------
    tuple of RationalRelation
        Both bound operands, sharing the runtime object when the
        specifications are identical.

    Raises
    ------
    ValueError
        Distinct parameter IDs alias, or the tensor domains disagree.
    """
    left = left_spec.bind(parameters)
    right = left if left_spec == right_spec else right_spec.bind(parameters)
    _weight_bindings(
        (*left_spec.weight_parameters, *right_spec.weight_parameters),
        parameters,
        description=description,
    )
    return left, right


def _validate_fixed_observation(
    relation: RationalRelationSpec,
    words: Words,
    max_states: int,
    max_work: int,
    max_closure_size: int | None,
) -> None:
    """Check the shared observation and resource contract.

    Parameters
    ----------
    relation : RationalRelationSpec
        The validated sparse relation specification.
    words : Words
        The fixed observed words.
    max_states : int
        The state-exploration budget.
    max_work : int
        The arc-examination budget.
    max_closure_size : int or None
        The productive closure limit.

    Raises
    ------
    ValueError
        If a word or limit is invalid, or the algebra is unsupported.
    """
    _validate_limits(max_states, max_work, max_closure_size)
    relation.topology._validate_words(words)
    if relation.algebra not in (MARKOV.subject, LOG_PROB.subject):
        raise ValueError("fixed relation requests require Markov or LogProb")


__all__ = [
    "FixedMinimizedRelationRequest",
    "FixedRelationArcCountsRequest",
    "FixedRelationCostRequest",
    "FixedRelationRequest",
    "RationalRelationSpec",
    "RelationWeightBindings",
    "RelationWeightComplement",
]
