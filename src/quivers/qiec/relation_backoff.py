"""Checked fixed-word backoff requests with portable weight identities."""

from __future__ import annotations

from functools import cached_property
from hashlib import sha256

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import FailureArc, FixedStringQuery, backoff_query
from quivers.core.relations.backoff import _failure_index, _ordinary_index
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import (
    RationalRelationSpec,
    _validate_fixed_observation,
    RelationWeightBindings,
)
from quivers.qiec.types import REAL, TypeExpr


_BACKOFF_HANDLER_PREFIX = "__relation_backoff_"


class FixedBackoffRelationRequest(QiecModel):
    """Check one word against a deterministic acceptor with failure arcs.

    Ordinary arcs have structural priority, including when their current
    weights are zero. Each missing symbol follows an acyclic failure
    chain until it finds an ordinary match or runs out of failures. The
    failure leaves the symbol unconsumed and never supplies a final mark.

    Parameters
    ----------
    relation : RationalRelationSpec
        The original deterministic, epsilon-free one-tape topology,
        with Markov or LogProb weights and at most one initial mark.
    failures : tuple of triples of int, int, ParameterId
        Ordered failure arc sources, targets and original scalar weight
        identities. Identities may be shared with the ordinary topology
        or supplied only for failure arcs.
    word : tuple of int
        The word on the original nominal alphabet.
    max_states : int
        The limit on visited configurations, including failures.
    max_work : int
        The limit on ordinary symbol probes, including misses. Topology
        validation runs separately in the size of the supplied graph.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    failures: tuple[tuple[int, int, ParameterId], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    word: tuple[int, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_states: int
    max_work: int

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check typed words, deterministic priority, failure links and limits.

        Raises
        ------
        ValueError
            The topology lacks the deterministic one-tape capability,
            a failure occurrence is invalid, or a word or limit fails.
        """
        topology = self.relation.topology
        if len(topology.tapes) != 1 or len(topology.initial) > 1:
            raise ValueError(
                "backoff requires a one-tape acceptor with one initial mark at most"
            )
        _validate_fixed_observation(
            self.relation, (self.word,), self.max_states, self.max_work, None
        )
        _ordinary_index(topology)
        prepared, failures = self._binding
        _failure_index(prepared.topology, failures)

    @cached_property
    def _binding(self) -> tuple[RationalRelationSpec, tuple[FailureArc, ...]]:
        """Assign failure-only identities to additional local weight slots.

        Returns
        -------
        tuple[RationalRelationSpec, tuple[FailureArc, ...]]
            The original specification with any added weight slots, and
            failure occurrences addressing those or existing slots.
        """
        original = self.relation
        slots = {identity: i for i, identity in enumerate(original.weight_parameters)}
        identities = list(original.weight_parameters)
        names = list(original.weight_names)
        used_names = set(names)
        arcs = []
        for source, target, identity in self.failures:
            if identity not in slots:
                slots[identity] = len(identities)
                identities.append(identity)
                name = f"failure:{identity.digest}"
                while name in used_names:
                    name += ":"
                used_names.add(name)
                names.append(name)
            arcs.append(FailureArc(source, target, slots[identity]))
        if len(identities) != len(original.weight_parameters):
            original = original.with_(
                weight_names=tuple(names), weight_parameters=tuple(identities)
            )
        return original, tuple(arcs)

    @property
    def handler_name(self) -> str:
        """Return the checked foreign name for the exact backoff observation.

        Returns
        -------
        str
            A portable digest of the original specification, ordered
            failure arcs and fixed word. Limits are invocation settings.
        """
        from quivers.qiec.serialization import dumps

        owner = (self.relation, self.failures, self.word)
        digest = sha256(dumps(owner).encode("utf-8")).hexdigest()
        return f"{_BACKOFF_HANDLER_PREFIX}{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Return the scalar type of the ordinary algebra's pair weight.

        Returns
        -------
        TypeExpr
            `LogWeight` for LogProb and `Real` for Markov.
        """
        return LOG_WEIGHT if self.relation.algebra == LOG_PROB.subject else REAL

    def plan(self, parameters: RelationWeightBindings) -> FixedStringQuery:
        """Plan the selected compact path with the original live tensors.

        Parameters
        ----------
        parameters : RelationWeightBindings
            All the original scalar owners, including failure-only owners.

        Returns
        -------
        FixedStringQuery
            The reusable path plan for the declared word.

        Raises
        ------
        TypeError
            A scalar binding fails its carrier or original leaf contract.
        ValueError
            A scalar binding or a construction limit fails.
        """
        relation, failures = self._binding
        return backoff_query(
            relation.bind(parameters),
            failures,
            self.word,
            max_states=self.max_states,
            max_work=self.max_work,
        )


__all__ = ["FixedBackoffRelationRequest"]
