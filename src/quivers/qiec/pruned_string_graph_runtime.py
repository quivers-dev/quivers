"""Checked finite-support pruned string inference through linear Compute."""

from __future__ import annotations

from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.relations import PrunedStringBeliefResult, RelationLimitError
from quivers.core.relations.messages import _candidate_log_mass
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.effects import HandlerDef
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import RelationWeightBindings
from quivers.qiec.string_graph_runtime import _tree_string_graph_handler
from quivers.qiec.string_graphs import StringFactorGraphSpec
from quivers.qiec.string_solvers import PrunedStringBPConfig
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    IndexLiteral,
    TypeExpr,
    product_type,
)

type _Report = tuple[
    str,
    str,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class PrunedStringGraphRequest(QiecModel):
    """A bounded approximate query over one checked graph and finite support.

    The same authored graph can use other solvers with their own checked
    capabilities. The ordered candidate pools and requested variable
    identify this foreign declaration. Operational settings can change
    within the same declaration.

    Parameters
    ----------
    graph : StringFactorGraphSpec
        The original typed factor declarations and parameter owners.
    config : PrunedStringBPConfig
        The ordered supports and existing beam inference settings.
    variable : str
        The variable whose approximate belief the query returns.
    """

    graph: StringFactorGraphSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    config: PrunedStringBPConfig = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    variable: str

    @dx.model_validator()
    def _validate_supports(self) -> None:
        """Check the typed candidate pools and the selected result variable.

        Raises
        ------
        ValueError
            A graph variable is missing, a word is outside its nominal
            alphabet, or the candidates omit a graph observation.
        """
        variables = {name: (size, word) for name, _, size, word in self.graph.variables}
        supports = dict(self.config.supports)
        if variables.keys() != supports.keys():
            raise ValueError(
                "finite string supports must name every graph variable exactly once"
            )
        if self.variable not in variables:
            raise ValueError("the pruned query variable is not declared in its graph")
        for name, (size, observed) in variables.items():
            words = supports[name]
            if any(symbol >= size for word in words for symbol in word):
                raise ValueError(
                    f"finite support for {name!r} contains an invalid symbol"
                )
            if observed is not None and observed not in words:
                raise ValueError(
                    f"finite support for {name!r} excludes its observation"
                )

    @property
    def candidate_words(self) -> tuple[tuple[int, ...], ...]:
        """Return the query variable's checked output coordinate order.

        Returns
        -------
        tuple of tuples of int
            Its ordered candidates, or the observed singleton.
        """
        observed = next(
            word for name, _, _, word in self.graph.variables if name == self.variable
        )
        return (
            dict(self.config.supports)[self.variable]
            if observed is None
            else (observed,)
        )

    @property
    def _proposal_count(self) -> int:
        """Return the number of factor-tape proposal positions.

        Returns
        -------
        int
            The sum of factor arities in original scope order.
        """
        return sum(len(scope) for _, scope in self.graph.factors)

    @property
    def handler_name(self) -> str:
        """Return the declared foreign name bound to the graph and candidate axes.

        Returns
        -------
        str
            A name derived from the exact graph, ordered supports, and query variable.
        """
        from quivers.qiec.serialization import dumps

        payload = (self.graph, self.config.supports, self.variable)
        digest = sha256(dumps(payload).encode("utf-8")).hexdigest()
        return f"__string_graph_pruned_{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Return the typed product of measured approximate beliefs and diagnostics.

        Returns
        -------
        TypeExpr
            Status, reason, sweeps, residual, truncation, log belief,
            proposal presence, and factor-tape log masses in that order.
        """
        extent = IndexLiteral(value=len(self.candidate_words), sort=NAT)
        proposal_rows = IndexLiteral(value=self._proposal_count, sort=NAT)
        mass_columns = IndexLiteral(value=3, sort=NAT)
        return product_type(
            STRING,
            STRING,
            INT,
            REAL,
            BOOL,
            tensor_type(LOG_WEIGHT, (extent,)),
            BOOL,
            tensor_type(LOG_WEIGHT, (proposal_rows, mass_columns)),
        )


def pruned_string_graph_handler(
    module: str, request: PrunedStringGraphRequest
) -> HandlerDef:
    """Declare the checked linear foreign Compute capability for a pruned query.

    Parameters
    ----------
    module : str
        The checked module name.
    request : PrunedStringGraphRequest
        The graph, candidate axes and typed result.

    Returns
    -------
    HandlerDef
        The checked declaration with the canonical observation name.

    Raises
    ------
    ValueError
        The module name is empty.
    """
    if not module:
        raise ValueError("a pruned string graph handler requires a module name")
    return _tree_string_graph_handler(
        module,
        request.handler_name,
        request.config.objective,
        request.result_type,
        graph=request.graph,
    )


class PrunedStringGraphRuntimeProvider:
    """Serve one checked bounded approximate string belief query.

    Parameters
    ----------
    request : PrunedStringGraphRequest
        The authored graph, ordered candidates, selected variable and limits.
    parameters : RelationWeightBindings
        Original live scalar leaves and any permitted Markov complements.
    """

    name = "pruned-string-graphs"

    def __init__(
        self,
        request: PrunedStringGraphRequest,
        parameters: RelationWeightBindings,
    ) -> None:
        self._request = request
        self._resource_failure: str | None = None
        try:
            self._plan = request.config._plan(request.graph, parameters)
        except RelationLimitError as error:
            self._resource_failure = str(error)
            self._plan = None

    def _report(
        self, result: PrunedStringBeliefResult | None, resource_failure: str | None
    ) -> _Report:
        """Pack measured diagnostics and raw proposal scales in typed order.

        Parameters
        ----------
        result : PrunedStringBeliefResult or None
            The approximate schedule's result, when it ran.
        resource_failure : str or None
            The unavailable-plan resource diagnostic.

        Returns
        -------
        tuple
            The declared product, including explicit availability of proposal rows.
        """
        request = self._request
        rows = (
            ()
            if result is None
            else tuple(item for group in result.proposals for item in group)
        )
        belief = (
            None
            if result is None or result.log_marginals is None
            else result.log_marginals[request.variable]
        )
        like = (
            belief
            if belief is not None
            else (
                rows[0].log_total_mass if rows else torch.zeros((), dtype=torch.float64)
            )
        )
        if belief is None:
            belief = like.new_full((len(request.candidate_words),), -torch.inf)
        if rows:
            masses = torch.stack(
                tuple(
                    torch.stack(
                        (
                            row.log_total_mass,
                            _candidate_log_mass(row.log_weights),
                            row.log_omitted_mass,
                        )
                    )
                    for row in rows
                )
            )
        else:
            masses = like.new_full((request._proposal_count, 3), -torch.inf)
        status = "resource_limit" if result is None else result.status
        reason = resource_failure if result is None else result.diagnostic
        sweeps = 0 if result is None else result.sweeps
        residual = None if result is None else result.message_residual
        return (
            status,
            reason or "",
            like.new_tensor(sweeps, dtype=torch.int64),
            like.new_tensor(-1.0 if residual is None else residual),
            like.new_tensor(
                False if result is None else result.truncated, dtype=torch.bool
            ),
            belief,
            like.new_tensor(bool(rows), dtype=torch.bool),
            masses,
        )

    def _evaluate(self, argument: object, _context: ClauseContext) -> _Report:
        """Run the existing bounded beam schedule with the live graph weights.

        Parameters
        ----------
        argument : object
            The unit Compute argument.
        _context : ClauseContext
            The checked foreign call context.

        Returns
        -------
        tuple
            The approximate belief, measured proposals, and execution status.

        Raises
        ------
        TypeError
            The argument is outside the unit carrier.
        """
        if argument is not None:
            raise TypeError("a pruned graph query takes a unit argument")
        if self._plan is None:
            return self._report(None, self._resource_failure)
        config = self._request.config
        try:
            result = self._plan.run_pruned(
                beam=config.beam,
                max_sweeps=config.max_sweeps,
                tolerance=config.tolerance,
                damping=config.damping,
            )
        except RelationLimitError as error:
            return self._report(None, str(error))
        return self._report(result, None)

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach the evaluator to its exact typed foreign declaration.

        Parameters
        ----------
        module : QiecModule
            The module declaring the pruned graph handler.
        attachments : RuntimeAttachments
            The invocation's runtime handler table.

        Raises
        ------
        TypeError
            The expected graph, supports, or linear Compute contract is absent.
        """
        expected = pruned_string_graph_handler(module.module, self._request)
        if expected not in module.handlers:
            try:
                expected = _fixed_compute_definition(
                    module, expected.name, expected.output_type
                )
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "incompatible pruned string handler for the graph, supports, or result type"
                ) from error
        validator = self.validator_for(self._request.result_type)
        assert validator is not None
        runtime = compute_handler(
            self._evaluate,
            effect=expected.effect,
            result_validator=validator,
            answer_type=expected.output_type,
            key=expected.name,
        )
        attachments.bind_handler(runtime.with_(definition=expected))

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Return the validator for this query's typed report.

        Parameters
        ----------
        type_ : TypeExpr
            The closed type requested by the reference machine.

        Returns
        -------
        RuntimeValidator or None
            The canonical product predicate when the type matches.
        """
        if type_ != self._request.result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


__all__ = [
    "PrunedStringGraphRequest",
    "PrunedStringGraphRuntimeProvider",
    "pruned_string_graph_handler",
]
