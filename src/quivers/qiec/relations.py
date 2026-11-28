"""Stable sparse relation specifications and their process-local weight bindings."""

from __future__ import annotations

from collections.abc import Mapping
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
    Label,
    RationalRelation,
    RelationTopology,
    Words,
)
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.foreign import KERNEL_ALGEBRAS
from quivers.qiec.types import REAL, TypeExpr


class _RelationWeights(nn.Module):
    """Expose live scalar tensors in the relation's weight-slot order.

    Parameters
    ----------
    slots : tuple of ParameterId
        The parameter identity at each weight slot.
    values : Mapping of ParameterId to torch.Tensor
        The original tensor for each distinct identity.
    """

    def __init__(
        self,
        slots: tuple[ParameterId, ...],
        values: Mapping[ParameterId, torch.Tensor],
    ) -> None:
        super().__init__()
        self._slots = tuple(f"weight_{id_.digest}" for id_ in slots)
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
        """
        if not self._slots:
            return torch.empty(0)
        return torch.stack(
            [cast(torch.Tensor, getattr(self, name)) for name in self._slots]
        )


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

    def bind(self, parameters: Mapping[ParameterId, torch.Tensor]) -> RationalRelation:
        """Bind original scalar leaves by stable identity and return a live relation.

        Values must be dense scalar leaf tensors in the algebra's
        carrier, with a common dtype and device. Trainable values must
        be original torch.nn.Parameter objects. Each distinct identity
        binds one object, and shared slots must use the same identity.
        The mapping may also provide identities for other relations.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The live scalar tensor for each stable identity.

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
        if not isinstance(parameters, Mapping):
            raise TypeError("relation weights require an explicit parameter mapping")
        values: dict[ParameterId, torch.Tensor] = {}
        for id_ in dict.fromkeys(self.weight_parameters):
            if id_ not in parameters:
                raise ValueError(f"missing relation parameter {id_}")
            value = parameters[id_]
            if (
                type(value) not in (torch.Tensor, nn.Parameter)
                or value.ndim != 0
                or not value.is_leaf
                or value.layout != torch.strided
                or value.device.type == "meta"
                or value.is_quantized
            ):
                raise TypeError(
                    f"relation parameter {id_} requires a dense scalar leaf tensor"
                )
            if value.requires_grad and not isinstance(value, nn.Parameter):
                raise TypeError(
                    f"trainable relation parameter {id_} requires torch.nn.Parameter"
                )
            values[id_] = value
        if len({id(value) for value in values.values()}) != len(values):
            raise ValueError(
                "shared relation weights must use the same parameter identity"
            )
        if len({(value.dtype, value.device) for value in values.values()}) > 1:
            raise ValueError("relation parameters require a common dtype and device")
        algebra = self._algebra
        assert algebra is not None  # Checked when constructing the specification.
        weights = _RelationWeights(self.weight_parameters, values)
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


__all__ = ["FixedRelationCostRequest", "FixedRelationRequest", "RationalRelationSpec"]
