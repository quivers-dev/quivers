"""Interpret bounded binary choice over the exact loss of a QIEC continuation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from quivers._values import HeldValue
from quivers.qiec._random_site_graph import _Address, _check_first_order_graph
from quivers.qiec.differentiation import (
    ExactJvp,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    _bound,
    _Dual,
    _dual,
    _fraction,
    _run_derivative,
    _ScalarDerivativeEvaluator,
)
from quivers.qiec.effects import EffectRequest
from quivers.qiec.evaluator import RuntimeRequest, _Continue, _Frame, _Returned
from quivers.qiec.identifiers import ComputationId, EffectInstanceId, OperationId
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.numbers import decide
from quivers.qiec.scope import Scope
from quivers.qiec.terms import Local
from quivers.qiec.types import BOOL, REAL, UNIT


@dataclass(frozen=True, slots=True)
class FiniteSelectionDecision:
    """The alternatives and the chosen answer at one realized occurrence.

    Parameters
    ----------
    address : tuple
        The original checked request's dynamic address.
    maximize : bool
        Whether the choice operation asks to maximize loss.
    losses : tuple[Fraction, Fraction]
        Total source losses for the two policies' optimal continuations,
        in false, true order. Loss before the choice is common to both.
    choice : bool
        The reply issued to the selected continuation.
    """

    address: _Address
    maximize: bool
    losses: tuple[Fraction, Fraction]
    choice: bool


@dataclass(frozen=True, slots=True)
class FiniteSelectionResult:
    """Return, loss, and derivative observations for a checked finite choice.

    Parameters
    ----------
    value : ExactJvp
        The returned value and tangent of the selected source path.
    loss : ExactJvp
        The sum of the selected path's loss writes and its tangent.
    decisions : tuple[FiniteSelectionDecision, ...]
        Decisions in the selected trace, in execution order.
    ties : int
        Equal comparisons across the explored tree. When this is
        positive, the tangents are of the selected path at the point.
    paths : int
        The number of completed alternatives.
    source_steps : int
        Actual reference machine steps, including speculative replays.
    """

    value: ExactJvp
    loss: ExactJvp
    decisions: tuple[FiniteSelectionDecision, ...]
    ties: int
    paths: int
    source_steps: int


@dataclass(frozen=True, slots=True)
class _Completed:
    """The exact returned scalar and writer loss on one complete path.

    Parameters
    ----------
    value : ExactJvp
        The path's returned scalar and tangent.
    loss : ExactJvp
        The path's accumulated loss and tangent.
    """

    value: ExactJvp
    loss: ExactJvp


class _Choice(Exception):
    """An unresolved nominal choice encountered by the source replay.

    Parameters
    ----------
    address : tuple
        The checked dynamic request address.
    maximize : bool
        Whether the request asks for the greater continuation loss.
    """

    def __init__(self, address: _Address, maximize: bool) -> None:
        super().__init__(address)
        self.address = address
        self.maximize = maximize


class _SelectionEvaluator(_ScalarDerivativeEvaluator):
    """Replay one path using the exact scalar machine and a fresh writer state.

    Parameters
    ----------
    module : QiecModule
        The checked module containing the entry and its callees.
    fuel : int
        Maximum source steps on this replay.
    choose : OperationId
        The checked operation that asks to select a Boolean answer.
    prefix : tuple[bool, ...]
        Replies to the choice occurrences already resolved on this path.
    """

    def __init__(
        self,
        module: QiecModule,
        fuel: int,
        choose: OperationId,
        prefix: tuple[bool, ...],
    ) -> None:
        super().__init__(module=module, fuel=fuel)
        self.choose = choose
        self.prefix = prefix
        self.position = 0
        self.loss = ExactJvp(Fraction(), Fraction())
        self.addresses: set[_Address] = set()

    def _dispatch(
        self,
        request: RuntimeRequest,
        environment: Scope[Local, HeldValue],
        stack: list[_Frame],
        search_index: int,
    ) -> _Continue:
        """Resume a checked request or expose its unresolved choice.

        Parameters
        ----------
        request : RuntimeRequest
            The already evaluated nominal request.
        environment : Scope
            The reference machine's current scope.
        stack : list
            The reference machine's stack, preserved across the reply.
        search_index : int
            The reference handler search position. This fragment uses
            its own selection and writer at the boundary.

        Returns
        -------
        _Continue
            The original continuation with the chosen reply or unit.

        Raises
        ------
        JvpUnsupported
            If the reference machine repeats a reached request address.
        _Choice
            If the source reaches a choice beyond the replay prefix.
        """
        address = request.address
        if address in self.addresses:
            raise JvpUnsupported(f"repeated selection or loss address {address!r}")
        self.addresses.add(address)
        (argument,) = request.arguments
        if request.operation == self.choose:
            maximize = decide(argument, "a selection direction")
            if self.position == len(self.prefix):
                raise _Choice(address, maximize)
            reply = self.numbers.number("truth", self.prefix[self.position])
            self.position += 1
        else:
            assert isinstance(argument, _Dual)
            value = self.loss.value + argument.value
            tangent = self.loss.tangent + argument.tangent
            _bound(value, tangent)
            self.loss = ExactJvp(value, tangent)
            reply = None
        return _Continue(
            current=_Returned(value=reply),
            environment=environment,
            resumption_path=request.resumption_path,
        )


def _resolve(
    records: dict[tuple[bool, ...], _Choice | _Completed],
    choose_first_at_ties: bool,
    paths: int,
    source_steps: int,
) -> FiniteSelectionResult:
    """Fold later choices first so each candidate uses its own selected future.

    Parameters
    ----------
    records : dict
        The full choice tree in parent-before-child order.
    choose_first_at_ties : bool
        Whether to select false on equal continuation losses.
    paths : int
        The observed number of complete paths.
    source_steps : int
        Total reference machine steps.

    Returns
    -------
    FiniteSelectionResult
        Only the realized value, loss, and sequence of decisions.

    Raises
    ------
    JvpBoundaryError
        If any comparison ties without an explicit branch convention.
    """
    completed: dict[tuple[bool, ...], _Completed] = {}
    decisions: dict[tuple[bool, ...], FiniteSelectionDecision] = {}
    ties = 0
    for prefix in reversed(records):
        event = records[prefix]
        if isinstance(event, _Completed):
            completed[prefix] = event
            continue
        no = completed.pop((*prefix, False))
        yes = completed.pop((*prefix, True))
        losses = (no.loss.value, yes.loss.value)
        if losses[0] == losses[1]:
            ties += 1
            if not choose_first_at_ties:
                raise JvpBoundaryError(
                    f"equal continuation losses at selection address {event.address!r}"
                )
        choice = losses[1] > losses[0] if event.maximize else losses[1] < losses[0]
        completed[prefix] = yes if choice else no
        decisions[prefix] = FiniteSelectionDecision(
            event.address, event.maximize, losses, choice
        )

    result = completed[()]
    selected: list[FiniteSelectionDecision] = []
    prefix = ()
    while (decision := decisions.get(prefix)) is not None:
        selected.append(decision)
        prefix = (*prefix, decision.choice)
    return FiniteSelectionResult(
        result.value, result.loss, tuple(selected), ties, paths, source_steps
    )


def finite_selection_jvp(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    *,
    choose: OperationId,
    loss: OperationId,
    choose_first_at_ties: bool = False,
    max_paths: int = 1_024,
    fuel: int = 65_536,
) -> FiniteSelectionResult:
    """Handle authored binary selection over the loss of the remaining entry.

    The selected entry has one closed nominal effect instance. Its
    bound choice, of type `Bool -> Bool`, takes a direction (true
    maximizes). Its bound writer, of type `Real -> Unit`, accumulates
    exact loss. Traversal interprets later decisions before comparing the
    losses offered to an earlier decision. Candidate evaluations have
    private writer state. The returned loss counts the selected writes
    once. The scalar derivative is exact wherever all explored source
    comparisons and continuation choices are locally fixed.

    Parameters
    ----------
    module : QiecModule
        The module supplying checked source and its effect interface.
    computation : ComputationId
        The scalar real entry whose remaining computation defines the
        loss scope for all requests in it.
    arguments : tuple[int or Fraction, ...]
        Exact input coordinates in source position order.
    direction : tuple[int or Fraction, ...]
        Exact tangent coordinates in the same order.
    choose : OperationId
        The nominal `Bool -> Bool` operation of the entry's interface.
    loss : OperationId
        The nominal `Real -> Unit` operation of the same interface.
    choose_first_at_ties : bool
        Allow a pointwise branch tangent by choosing false at equal
        losses. By default equality refuses a derivative claim.
    max_paths : int
        Maximum number of complete alternatives.
    fuel : int
        Reference machine steps allowed for each replay.

    Returns
    -------
    FiniteSelectionResult
        The exact selected source value, writer loss, and decisions.

    Raises
    ------
    JvpUnsupported
        If the source or either operation is outside the checked fragment.
    JvpBoundaryError
        If the source comparison is unstable or a selection ties.
    JvpLimit
        If the path, reference machine, or exact arithmetic bounds are hit.
    ValueError
        If an argument or resource limit is malformed.
    """
    for name, limit in (("max_paths", max_paths), ("fuel", fuel)):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(choose_first_at_ties) is not bool:
        raise ValueError("choose_first_at_ties must be Boolean")
    registry = validate_module(module)
    owner = next((item for item in module.computations if item.id == computation), None)
    if owner is None:
        raise JvpUnsupported(f"unknown computation for finite selection {computation}")
    row = owner.type.effects
    if (
        owner.telescope
        or owner.type.result != REAL
        or any(parameter.type != REAL for parameter in owner.parameters)
        or row.tail is not None
        or len(row.entries) != 1
    ):
        raise JvpUnsupported(
            "finite selection needs a scalar real entry with one nominal effect instance"
        )
    if not len(arguments) == len(direction) == len(owner.parameters):
        raise ValueError("arguments and direction must match the computation's arity")
    effect = row.entries[0].effect
    declaration = next((item for item in module.effects if item.ref == effect), None)
    operations = (
        {} if declaration is None else {op.id: op for op in declaration.operations}
    )
    for identity, arg, result in ((choose, BOOL, BOOL), (loss, REAL, UNIT)):
        operation = operations.get(identity)
        if (
            operation is None
            or operation.telescope
            or tuple(parameter.type for parameter in operation.arguments) != (arg,)
            or operation.result_type != result
        ):
            raise JvpUnsupported(
                "finite selection operation does not match its nominal interface and signature"
            )

    def check_request(request: EffectRequest, instance: EffectInstanceId) -> None:
        """Restrict all reachable requests to the two checked operations.

        Parameters
        ----------
        request : EffectRequest
            A request found in the checked source graph.
        instance : EffectInstanceId
            The unique instance selected by the entry.

        Raises
        ------
        JvpUnsupported
            If the source requests another instance or operation.
        """
        if request.instance != instance or request.operation not in (choose, loss):
            raise JvpUnsupported("request is outside the finite selection interface")

    _check_first_order_graph(module, owner, "finite selection", check_request)
    environment = {
        parameter: _dual(_fraction(value), _fraction(tangent), True)
        for parameter, value, tangent in zip(
            owner.parameters, arguments, direction, strict=True
        )
    }
    records: dict[tuple[bool, ...], _Choice | _Completed] = {}
    pending: list[tuple[bool, ...]] = [()]
    paths = 1
    source_steps = 0
    while pending:
        prefix = pending.pop()
        evaluator = _SelectionEvaluator(module, fuel, choose, prefix)
        try:
            value = _run_derivative(evaluator, owner, registry, environment)
        except _Choice as event:
            records[prefix] = event
            paths += 1
            if paths > max_paths:
                raise JvpLimit(
                    f"finite selection exceeds {max_paths} complete paths"
                ) from None
            pending.extend(((*prefix, True), (*prefix, False)))
        else:
            assert isinstance(value, _Dual) and evaluator.position == len(prefix)
            records[prefix] = _Completed(
                ExactJvp(value.value, value.tangent), evaluator.loss
            )
        finally:
            source_steps += evaluator._steps
    return _resolve(records, choose_first_at_ties, paths, source_steps)


__all__ = ["FiniteSelectionDecision", "FiniteSelectionResult", "finite_selection_jvp"]
