"""Checked finite-support EP with differentiation through executed updates."""

from __future__ import annotations

from functools import cached_property
from hashlib import sha256
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.relations import (
    FiniteStringBeliefPlan,
    ProductiveEpsilonCycleError,
    RelationLimitError,
)
from quivers.core.relations.beliefs import (
    _FiniteEPInitialSites,
    _validate_finite_ep_initial_sites,
    _validate_sweeps,
)
from quivers.core.relations.projection import _validate_projection_settings
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import RelationWeightBindings
from quivers.qiec.string_graphs import StringFactorGraphSpec
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    IndexLiteral,
    TypeExpr,
    product_type,
)

type _Word = tuple[int, ...]
type _Supports = tuple[tuple[str, tuple[_Word, ...]], ...]
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


class FiniteStringEPRequest(QiecModel):
    """Request fixed-sweep, unpenalized unigram EP on declared finite supports.

    Accepted updates retain their autodiff graph. Branch decisions use
    the actual working-precision tests. The request concerns a finite
    program's derivative on its executed branch.

    Parameters
    ----------
    graph : StringFactorGraphSpec
        The exact authored factors, observations, and original weight owners.
    supports : tuple
        One ordered finite support per graph variable.
    variable : str
        The selected marginal.
    words : tuple
        Distinct probe words in result order.
    sweeps : int
        The exact number of accepted synchronous sweeps required.
    damping : float
        The old site's fraction in log-space damping.
    projection_tolerance : float
        The local infinity-norm moment residual tolerance.
    projection_max_iterations, projection_max_backtracks : int
        Bounds for accepted Newton steps and backtracks per step.
    max_entries, max_states, max_work : int
        Cumulative factor table entries, reached states, and candidate arc
        examinations for the observed-pair tables. Productive cycles refuse.
    max_feature_cells : int
        Bound on the summed full unigram matrix sizes after observations.
    max_site_fits : int
        Bound on the number of local site attempts, including diagnostics
        on the returned state.
    initial_sites : tuple or None
        Ordered factor occurrence, variable, affine basis, and finite
        coefficient vector for each warm initial normalized site.
    """

    graph: StringFactorGraphSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    supports: _Supports = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    variable: str
    words: tuple[_Word, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    sweeps: int
    damping: float
    projection_tolerance: float
    projection_max_iterations: int
    projection_max_backtracks: int
    max_entries: int
    max_states: int
    max_work: int
    max_feature_cells: int
    max_site_fits: int
    initial_sites: _FiniteEPInitialSites | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the finite schedule, nominal supports, and resource bounds.

        Raises
        ------
        ValueError
            The schedule, supports, probes, warm data, or limits are invalid.
        """
        _validate_sweeps(self.sweeps, None, self.damping)
        _validate_projection_settings(
            self.projection_tolerance,
            self.projection_max_iterations,
            self.projection_max_backtracks,
        )
        _validate_finite_ep_initial_sites(self.initial_sites)
        limits = (
            self.max_entries,
            self.max_states,
            self.max_work,
            self.max_feature_cells,
            self.max_site_fits,
        )
        largest = torch.iinfo(torch.int64).max
        if any(type(limit) is not int or not 0 <= limit <= largest for limit in limits):
            raise ValueError("finite EP resource limits must be nonnegative Int values")
        variables = {
            name: (size, observed) for name, _, size, observed in self.graph.variables
        }
        supports = dict(self.supports)
        if (
            len(supports) != len(self.supports)
            or supports.keys() != variables.keys()
            or self.variable not in variables
        ):
            raise ValueError("finite EP requires one support per named graph variable")
        for name, (size, observed) in variables.items():
            words = supports[name]
            _check_words(words, size)
            if observed is not None and observed not in words:
                raise ValueError("finite EP support excludes its observation")
        _check_words(self.words, variables[self.variable][0])
        if self.initial_sites is not None:
            identities = tuple(
                (index, name)
                for index, (_, scope) in enumerate(self.graph.factors)
                for name in scope
            )
            if tuple(record[:2] for record in self.initial_sites) != identities:
                raise ValueError("finite EP initial sites must match graph order")

    @property
    def gradient_policy(self) -> Literal["finite_unrolled"]:
        """Give the derivative scope of the requested program.

        Returns
        -------
        Literal["finite_unrolled"]
            Autodiff follows the accepted finite sequence of operations.
        """
        return "finite_unrolled"

    @property
    def handler_name(self) -> str:
        """Bind the exact requested finite function to its foreign Compute.

        Returns
        -------
        str
            The handler identity, including the solver and initialization choices.
        """
        from quivers.qiec.serialization import dumps

        static = (
            self.graph,
            self.supports,
            self.variable,
            self.words,
            self.sweeps,
            self.damping,
            self.projection_tolerance,
            self.projection_max_iterations,
            self.projection_max_backtracks,
            self.initial_sites,
        )
        return "__string_graph_finite_ep_" + sha256(dumps(static).encode()).hexdigest()

    @property
    def result_type(self) -> TypeExpr:
        """Give the typed report and aggregate work counters.

        Returns
        -------
        TypeExpr
            Status, diagnostic, sweeps, residual presence and value,
            belief presence, ordered log scores, and three work counters.
        """
        size = IndexLiteral(value=len(self.words), sort=NAT)
        work = IndexLiteral(value=3, sort=NAT)
        return product_type(
            STRING,
            STRING,
            INT,
            BOOL,
            REAL,
            BOOL,
            tensor_type(LOG_WEIGHT, (size,)),
            tensor_type(INT, (work,)),
        )


def _check_words(words: tuple[_Word, ...], size: int) -> None:
    """Validate an ordered nonempty finite word axis.

    Parameters
    ----------
    words : tuple
        The support or probe words.
    size : int
        The nominal alphabet cardinality.

    Raises
    ------
    ValueError
        A word is duplicated, malformed, or outside the alphabet.
    """
    if (
        not words
        or any(
            not isinstance(word, tuple)
            or any(type(symbol) is not int or not 0 <= symbol < size for symbol in word)
            for word in words
        )
        or len(set(words)) != len(words)
    ):
        raise ValueError("finite EP requires distinct words in its declared alphabet")


class FiniteStringEPRuntimeProvider:
    """Expose the accepted finite EP schedule through checked native Compute.

    Parameters
    ----------
    request : FiniteStringEPRequest
        The exact finite program, graph, probes, and resource limits.
    parameters : RelationWeightBindings
        Original live scalar owners supplying a new evaluation snapshot.
    """

    name = "finite-string-ep"

    def __init__(
        self, request: FiniteStringEPRequest, parameters: RelationWeightBindings
    ) -> None:
        self._request = request
        self._parameters = parameters
        self._handler_name = request.handler_name
        self._result_type = request.result_type

    @cached_property
    def _plan(self) -> FiniteStringBeliefPlan:
        """Compile bounded topology and retain original sources for future calls.

        Returns
        -------
        FiniteStringBeliefPlan
            The finite factor-table plan, before evaluation of its scores.

        Raises
        ------
        RelationLimitError
            The static feature, fitting, or graph budgets cannot admit the plan.
        """
        request = self._request
        supports = dict(request.supports)
        cells = 0
        for name, _, _, observed in request.graph.variables:
            words = supports[name] if observed is None else (observed,)
            cells += len(words) * len({symbol for word in words for symbol in word})
            if cells > request.max_feature_cells:
                raise RelationLimitError("finite EP exceeds max_feature_cells")
        attempts = (request.sweeps + 1) * sum(
            len(scope) for _, scope in request.graph.factors
        )
        if attempts > request.max_site_fits:
            raise RelationLimitError("finite EP exceeds max_site_fits")
        trials = (
            attempts
            * request.projection_max_iterations
            * (request.projection_max_backtracks + 1)
        )
        if trials > torch.iinfo(torch.int64).max:
            raise RelationLimitError("finite EP projection trial count exceeds Int")
        graph = request.graph.bind(
            self._parameters, max_states=request.max_states, max_work=request.max_work
        )
        return FiniteStringBeliefPlan(
            graph,
            supports,
            max_entries=request.max_entries,
            max_states=request.max_states,
            max_work=request.max_work,
        )

    def _evaluate(self, argument: object, _context: ClauseContext) -> _Report:
        """Evaluate one live table snapshot and differentiate the accepted schedule.

        Parameters
        ----------
        argument : object
            Unit supplied to the native handler.
        _context : ClauseContext
            The checked runtime's clause context.

        Returns
        -------
        tuple
            Measured work and a live score when the entire schedule succeeds.

        Raises
        ------
        TypeError
            The Compute argument is outside Unit.
        """
        if argument is not None:
            raise TypeError("a finite EP report takes a unit argument")
        request = self._request
        try:
            plan = self._plan
            result = plan.expectation_propagation(
                order=1,
                max_sweeps=request.sweeps,
                tolerance=None,
                damping=request.damping,
                projection_tolerance=request.projection_tolerance,
                projection_max_iterations=request.projection_max_iterations,
                projection_max_backtracks=request.projection_max_backtracks,
                initial_sites=request.initial_sites,
            )
        except RelationLimitError as error:
            return self._unavailable("resource_limit", str(error))
        except ProductiveEpsilonCycleError as error:
            return self._unavailable("unsupported_closure", str(error))
        counts = (
            result.projection_attempts,
            result.projection_iterations,
            result.line_search_trials,
        )
        if result.status == "projection_failed":
            diagnostic = result.site_projections[-1].diagnostic
            assert diagnostic is not None
            return self._unavailable(
                "projection_failed", diagnostic, result.sweeps, counts
            )
        assert result.status == "limit" and result.sweeps == request.sweeps
        marginal = result.log_marginals[request.variable]
        positions = {
            word: index for index, word in enumerate(plan.supports[request.variable])
        }
        scores = torch.stack(
            tuple(
                marginal[positions[word]]
                if word in positions
                else marginal.new_full((), -torch.inf)
                for word in request.words
            )
        )
        residual = result.message_residual
        assert residual is not None
        return (
            "completed",
            "",
            scores.new_tensor(result.sweeps, dtype=torch.int64),
            scores.new_tensor(True, dtype=torch.bool),
            scores.new_tensor(residual),
            scores.new_tensor(True, dtype=torch.bool),
            scores,
            scores.new_tensor(counts, dtype=torch.int64),
        )

    def _unavailable(
        self,
        status: str,
        reason: str,
        sweeps: int = 0,
        counts: tuple[int, int, int] = (0, 0, 0),
    ) -> _Report:
        """Encode an incomplete schedule or unavailable graph.

        Parameters
        ----------
        status, reason : str
            The unavailable category and its diagnostic.
        sweeps : int
            The number of completed sweeps before a local failure.
        counts : tuple of int
            The measured fit attempts, accepted steps, and line-search trials.

        Returns
        -------
        tuple
            A report with absent scores and residual.
        """
        missing = torch.full(
            (len(self._request.words),), -torch.inf, dtype=torch.float64
        )
        absent = missing.new_tensor(False, dtype=torch.bool)
        return (
            status,
            reason,
            missing.new_tensor(sweeps, dtype=torch.int64),
            absent,
            missing.new_zeros(()),
            absent,
            missing,
            missing.new_tensor(counts, dtype=torch.int64),
        )

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach only to the exact declared finite EP program.

        Parameters
        ----------
        module : QiecModule
            The module containing the nominal foreign Compute.
        attachments : RuntimeAttachments
            The current invocation's handler table.

        Raises
        ------
        ValueError
            The named handler is absent or ambiguous.
        TypeError
            The declaration has an incompatible Compute contract.
        """
        definition = _fixed_compute_definition(
            module, self._handler_name, self._result_type
        )
        validator = self.validator_for(self._result_type)
        assert validator is not None
        runtime = compute_handler(
            self._evaluate,
            effect=definition.effect,
            result_validator=validator,
            answer_type=definition.output_type,
            key=definition.name,
        )
        attachments.bind_handler(runtime.with_(definition=definition))

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Supply the exact checked report carrier validator.

        Parameters
        ----------
        type_ : TypeExpr
            The runtime type to validate.

        Returns
        -------
        RuntimeValidator or None
            The canonical product validator when the carrier matches.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


__all__ = ["FiniteStringEPRequest", "FiniteStringEPRuntimeProvider"]
