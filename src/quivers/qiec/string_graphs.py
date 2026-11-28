"""Checked wire records and live weight bindings for sparse string factor graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cached_property

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.laws import ClaimSubject
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import StringFactor, StringFactorGraph, StringVariable
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import RationalRelationSpec
from quivers.qiec.types import REAL, TypeExpr


class StringFactorGraphSpec(QiecModel):
    """A checked string factor graph with named reusable relation specifications.

    Factor records are ordered occurrences. Two factors can refer to the
    same relation name, which binds one runtime relation and shares its
    weight source. Distinct relations can also share stable parameter
    identities. The wire record contains these identities and topology.
    Runtime binding supplies the tensor values.

    Parameters
    ----------
    variables : tuple
        The variable name, alphabet name, alphabet cardinality, and
        observed word or None for each variable.
    relations : tuple
        Distinct relation names and their sparse relation specifications.
    factors : tuple
        A relation name and its ordered variable scope for each factor
        occurrence. Each variable in the scope corresponds to one tape.
    """

    variables: tuple[tuple[str, str, int, tuple[int, ...] | None], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    relations: tuple[tuple[str, RationalRelationSpec], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    factors: tuple[tuple[str, tuple[str, ...]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

    @cached_property
    def _variables(self) -> tuple[StringVariable, ...]:
        """Reconstruct the typed variable declarations.

        Returns
        -------
        tuple of StringVariable
            The variables in wire order.

        Raises
        ------
        TypeError
            If an observation or nominal alphabet has an invalid representation.
        """
        variables = []
        for name, alphabet, size, word in self.variables:
            if not isinstance(alphabet, str) or type(size) is not int:
                raise TypeError(
                    "string graph alphabets require a name and integer size"
                )
            if word is not None and not isinstance(word, tuple):
                raise TypeError("string graph observations require a tuple of symbols")
            variables.append(
                StringVariable(
                    name,
                    StringSet(alphabet=FinSet(name=alphabet, cardinality=size)),
                    word,
                )
            )
        return tuple(variables)

    @dx.model_validator()
    def _validate_spec(self) -> None:
        """Validate scopes, tape identities, and the graph algebra.

        Raises
        ------
        ValueError
            If variable or relation names conflict, a factor is missing
            its relation, or a factor scope has the wrong shape.
        TypeError
            If a record has an invalid representation, a factor variable
            has the wrong tape, or the factor algebras disagree or are unsupported.
        """
        for records in (self.variables, self.relations, self.factors):
            if not isinstance(records, tuple) or not all(
                isinstance(record, tuple) for record in records
            ):
                raise TypeError(
                    "string graph declarations require ordered tuple records"
                )
        variables = {variable.name: variable.tape for variable in self._variables}
        if len(variables) != len(self.variables):
            raise ValueError("string graph variable names must be distinct")
        for name, spec in self.relations:
            if not isinstance(name, str) or not isinstance(spec, RationalRelationSpec):
                raise TypeError(
                    "string graph relations require names and relation specs"
                )
        relations = dict(self.relations)
        if len(relations) != len(self.relations) or "" in relations:
            raise ValueError(
                "string graph relation names must be nonempty and distinct"
            )
        if not self.factors:
            raise ValueError("a string factor graph requires at least one factor")

        common_algebra = None
        for index, (relation_name, scope) in enumerate(self.factors):
            if not isinstance(relation_name, str) or not isinstance(scope, tuple):
                raise TypeError(
                    "string graph factors require a relation name and scope"
                )
            if not all(isinstance(variable, str) for variable in scope):
                raise TypeError("string graph factor scopes require variable names")
            spec = relations.get(relation_name)
            if spec is None:
                raise ValueError(
                    f"unknown relation {relation_name!r} at factor {index}"
                )
            tapes = spec.topology.tapes
            if len(scope) != len(tapes) or len(scope) != len(set(scope)):
                raise ValueError(
                    f"factor {index} requires one distinct variable per tape"
                )
            for variable, tape in zip(scope, tapes, strict=True):
                if variables.get(variable) != tape:
                    raise TypeError(
                        f"unknown or mistyped variable {variable!r} at factor {index}"
                    )
            if spec.algebra not in (MARKOV.subject, LOG_PROB.subject):
                raise TypeError(
                    "string graph factors require Markov or LogProb algebra"
                )
            if common_algebra is not None and spec.algebra != common_algebra:
                raise TypeError("string graph factors require one shared algebra")
            common_algebra = spec.algebra

    @property
    def algebra(self) -> ClaimSubject:
        """Return the common checked algebra of the referenced relations.

        Returns
        -------
        ClaimSubject
            The nominal identity of the graph's scalar algebra.
        """
        return dict(self.relations)[self.factors[0][0]].algebra

    def bind(self, parameters: Mapping[ParameterId, torch.Tensor]) -> StringFactorGraph:
        """Bind the graph's referenced relations to their live scalar leaves.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The live scalar tensor for each used stable identity. All
            used tensors share one dtype and device, and distinct
            identities bind distinct tensor objects.

        Returns
        -------
        StringFactorGraph
            The typed graph with original tensor objects and reusable
            sparse relations.

        Raises
        ------
        TypeError
            If the parameter store or a required tensor is invalid.
        ValueError
            If a binding is missing, ambiguous, incompatible with the
            algebra, or inconsistent in dtype or device.
        """
        if not isinstance(parameters, Mapping):
            raise TypeError(
                "string graph weights require an explicit parameter mapping"
            )
        relations = dict(self.relations)
        used = dict.fromkeys(name for name, _ in self.factors)
        identities = dict.fromkeys(
            parameter
            for name in used
            for parameter in relations[name].weight_parameters
        )
        values_by_id = {}
        for identity in identities:
            if identity not in parameters:
                raise ValueError(f"missing string graph parameter {identity}")
            values_by_id[identity] = parameters[identity]
        bound = {name: relations[name].bind(values_by_id) for name in used}
        values = values_by_id.values()
        if len({id(value) for value in values}) != len(values):
            raise ValueError(
                "shared graph weights must use the same parameter identity"
            )
        if len({(value.dtype, value.device) for value in values}) > 1:
            raise ValueError(
                "string graph parameters require a common dtype and device"
            )
        return StringFactorGraph(
            self._variables,
            tuple(StringFactor(scope, bound[name]) for name, scope in self.factors),
        )


class TreeStringGraphRequest(QiecModel):
    """A bounded exact tree query over a serialized string factor graph.

    A partition uses the graph's scalar algebra. Supplying a word selects
    its normalized log probability at the named variable. An empty tuple
    queries the empty word. The exact executor checks the forest topology
    and the sparse construction and closure limits when bound.

    Parameters
    ----------
    graph : StringFactorGraphSpec
        The checked graph, including any observed words.
    variable : str
        The root variable for the marginal schedule.
    word : tuple of int or None
        The word whose normalized log probability is requested. None
        requests the full graph partition.
    max_states : int
        The maximum states for each intermediate construction or query.
    max_work : int
        The maximum candidate arc examinations for each construction or query.
    max_closure_size : int or None
        The maximum productive closure component, or None to require an
        acyclic query.
    """

    graph: StringFactorGraphSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    variable: str
    word: tuple[int, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_states: int
    max_work: int
    max_closure_size: int | None = None

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the requested variable, word, and construction budgets.

        Raises
        ------
        TypeError
            If the word or graph has an invalid representation.
        ValueError
            If the variable, word, or resource limits are invalid.
        """
        if not isinstance(self.graph, StringFactorGraphSpec):
            raise TypeError(
                "a tree string query requires a checked graph specification"
            )
        if self.max_states is None or self.max_work is None:
            raise ValueError(
                "tree string requests require explicit state and work limits"
            )
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)
        variable = next(
            (item for item in self.graph._variables if item.name == self.variable), None
        )
        if variable is None:
            raise ValueError(f"unknown string variable {self.variable!r}")
        if self.word is not None:
            if not isinstance(self.word, tuple):
                raise TypeError("a tree string query word must be a tuple")
            StringVariable(variable.name, variable.tape, self.word)

    @property
    def operation(self) -> str:
        """Return the objective identified by this request.

        Returns
        -------
        str
            Partition or normalized log probability.
        """
        return "partition" if self.word is None else "log_prob"

    @property
    def result_type(self) -> TypeExpr:
        """Return the closed QIEC scalar type of the objective.

        Returns
        -------
        TypeExpr
            LogWeight for log-domain results, or Real for a Markov partition.
        """
        if self.word is not None or self.graph.algebra == LOG_PROB.subject:
            return LOG_WEIGHT
        return REAL

    def bind(
        self, parameters: Mapping[ParameterId, torch.Tensor]
    ) -> Callable[[], torch.Tensor]:
        """Compile a reusable query over live parameters.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The live scalar tensor for each parameter identity in the graph.

        Returns
        -------
        Callable
            A zero-argument objective that reads the original parameters.

        Raises
        ------
        TypeError
            If the live parameter binding is invalid.
        ValueError
            If the graph is cyclic or the construction exceeds a limit.
        """
        marginal = self.graph.bind(parameters).tree_marginal(
            self.variable,
            max_states=self.max_states,
            max_work=self.max_work,
            max_closure_size=self.max_closure_size,
        )
        word = self.word
        if word is None:
            return marginal.partition
        return marginal.query(word)


__all__ = ["StringFactorGraphSpec", "TreeStringGraphRequest"]
