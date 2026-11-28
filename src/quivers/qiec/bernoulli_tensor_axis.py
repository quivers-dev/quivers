"""One-pass tensor execution for a checked independent Bernoulli row axis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from quivers._values import HeldValue
from quivers.qiec._random_site_graph import (
    _Address,
    _check_first_order_graph,
    _random_computation,
)
from quivers.qiec.bernoulli_estimators import _check_request, _probability_argument
from quivers.qiec.differentiation import JvpBoundaryError, JvpLimit, JvpUnsupported
from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec.effects import EMPTY_ROW
from quivers.qiec.evaluator import (
    Evaluator,
    FuelExhaustedError,
    RuntimeAttachments,
    RuntimeRequest,
    Scope,
    _Continue,
    _Frame,
    _Returned,
)
from quivers.qiec.finite_bernoulli_batch import BernoulliRowReplay
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.numbers import NumberDomain
from quivers.qiec.primitives import PRIMITIVES
from quivers.qiec.terms import (
    Bind,
    Call,
    CallAddress,
    Case,
    Computation,
    ConditionalValue,
    ConstructorValue,
    DistributionValue,
    If,
    LetValue,
    LiteralValue,
    Local,
    Perform,
    PrimitiveApplication,
    Return,
    Value,
    Var,
    WithTraceAddresses,
)
from quivers.qiec.types import BOOL, REAL, UNIT

_ELEMENTWISE = frozenset(
    PRIMITIVES[name].id
    for name in ("add_real", "sub_real", "mul_real", "neg_real", "bool_to_real")
)
_CASE_DIVISION = PRIMITIVES["div_real"].id
_CASE_PREDICATE = PRIMITIVES["gt_real"].id


@dataclass(frozen=True, slots=True)
class _CaseAxis:
    """The single selector, case and draw sites of a bounded tensor plan.

    Parameters
    ----------
    parameter, threshold : int, int or float
        Source parameter and literal of the strict greater-than split.
    selector, comparison : Call, PrimitiveApplication
        The dominating call and its sole row-dependent comparison.
    constructors, case : tuple, Case
        The constructor results of the selector and their case analysis.
    draws : tuple of Call
        The two marked invocations of the same sampling helper.
    """

    parameter: int
    threshold: int | float
    selector: Call
    comparison: PrimitiveApplication
    constructors: tuple[ConstructorValue, ConstructorValue]
    case: Case
    draws: tuple[Call, Call]


def _scoped_call(term: Computation) -> Call | None:
    """Read a call beneath an optional linear trace scope.

    Parameters
    ----------
    term : Computation
        The already checked computation in a bind prefix.

    Returns
    -------
    Call or None
        The underlying call, if any.
    """
    if isinstance(term, WithTraceAddresses):
        term = term.body
    return term if isinstance(term, Call) else None


def _branch_draw(
    body: Computation, declarations: dict[ComputationId, NamedComputation]
) -> Call | None:
    """Find the single direct sampling helper in a case arm's bind prefix.

    Parameters
    ----------
    body : Computation
        The checked branch body.
    declarations : dict
        The available named computations.

    Returns
    -------
    Call or None
        Its one direct sampling call, if present. The full graph check
        separately excludes any other effectful call or request.
    """
    draw = None
    while isinstance(body, Bind):
        call = _scoped_call(body.first)
        if call is not None and declarations[call.callee].type.effects != EMPTY_ROW:
            if draw is not None or not isinstance(
                declarations[call.callee].body, Perform
            ):
                return None
            draw = call
        body = body.then
    return draw


def _case_axis(module: QiecModule, owner: NamedComputation) -> _CaseAxis | None:
    """Recognize a two-constructor case with a common direct sampling helper.

    Parameters
    ----------
    module : QiecModule
        The validated module containing the selected source.
    owner : NamedComputation
        The scalar Bernoulli entry.

    Returns
    -------
    _CaseAxis or None
        The actual pure selector and call sites of the supported shape.
    """
    body = owner.body
    if not isinstance(body, Bind) or (select := _scoped_call(body.first)) is None:
        return None
    declarations = {item.id: item for item in module.computations}
    maker = declarations[select.callee]
    term = maker.body
    if not (
        maker.type.effects == EMPTY_ROW
        and isinstance(term, If)
        and isinstance(comparison := term.condition, PrimitiveApplication)
        and comparison.primitive == _CASE_PREDICATE
        and isinstance(term.then, Return)
        and isinstance(term.otherwise, Return)
        and isinstance(left := term.then.value, ConstructorValue)
        and isinstance(right := term.otherwise.value, ConstructorValue)
        and left.constructor != right.constructor
        and not left.static_arguments
        and not right.static_arguments
    ):
        return None
    operand, literal = comparison.arguments
    if not (
        isinstance(operand, Var)
        and operand.local in maker.parameters
        and isinstance(literal, LiteralValue)
        and isinstance(threshold := literal.value, int | float)
        and not isinstance(threshold, bool)
        and isinstance(
            argument := select.arguments[maker.parameters.index(operand.local)], Var
        )
        and argument.local in owner.parameters
    ):
        return None
    after = body.then
    while isinstance(after, Bind) and isinstance(after.first, CallAddress):
        after = after.then
    if not (
        isinstance(after, Case)
        and isinstance(after.scrutinee, Var)
        and after.scrutinee.local == body.binder
        and len(after.branches) == 2
        and {branch.constructor for branch in after.branches}
        == {left.constructor, right.constructor}
    ):
        return None
    first = _branch_draw(after.branches[0].body, declarations)
    second = _branch_draw(after.branches[1].body, declarations)
    if first is None or second is None or first.callee != second.callee:
        return None
    return _CaseAxis(
        owner.parameters.index(argument.local),
        threshold,
        select,
        comparison,
        (left, right),
        after,
        (first, second),
    )


def _literal_branch(condition: Value) -> bool:
    """Require a scalar Boolean that is independent of every lane.

    Parameters
    ----------
    condition : Value
        A source decision in an active term.

    Returns
    -------
    bool
        The literal decision made by every lane.

    Raises
    ------
    JvpUnsupported
        If the branch may read a parameter or random choice.
    """
    if isinstance(condition, LiteralValue) and type(condition.value) is bool:
        return condition.value
    raise JvpUnsupported("a Bernoulli tensor axis needs a literal Boolean branch")


def _check_tensor_terms(
    module: QiecModule, owner: NamedComputation, case: _CaseAxis | None = None
) -> None:
    """Check the terms acted on by a physical row axis.

    Parameters
    ----------
    module : QiecModule
        The module whose typed rows and acyclic calls have already checked.
    owner : NamedComputation
        Its selected source entry.
    case : _CaseAxis or None
        The sole admitted dynamic selector, case and draw calls, if present.

    Raises
    ------
    JvpUnsupported
        If an active term could inspect the row axis or has no tensor rule.
    """
    declarations = {item.id: item for item in module.computations}
    seen = {owner.id}
    pending: list[Computation | Value] = [owner.body]
    selectors = draws = 0
    while pending:
        term = pending.pop()
        if isinstance(term, Bind):
            pending.extend((term.then, term.first))
        elif isinstance(term, Return):
            pending.append(term.value)
        elif isinstance(term, If | ConditionalValue):
            condition = term.condition
            if case is not None and (
                condition is case.comparison
                or isinstance(condition, Var)
                and condition.local.type == BOOL
            ):
                pending.extend((condition, term.then, term.otherwise))
            else:
                pending.append(
                    term.then if _literal_branch(condition) else term.otherwise
                )
        elif isinstance(term, Call):
            if case is not None:
                if term.callee == case.selector.callee:
                    if term is not case.selector:
                        raise JvpUnsupported("tensor cases need one selector call")
                    selectors += 1
                if declarations[term.callee].type.effects != EMPTY_ROW:
                    if all(term is not draw for draw in case.draws):
                        raise JvpUnsupported("tensor cases need one common draw")
                    draws += 1
            pending.extend(term.arguments)
            if term.callee not in seen:
                seen.add(term.callee)
                pending.append(declarations[term.callee].body)
        elif isinstance(term, Perform):
            if case is not None and term is not declarations[case.draws[0].callee].body:
                raise JvpUnsupported("tensor cases need one common draw")
            sampleable = term.request.arguments[1]
            assert isinstance(sampleable, DistributionValue)
            pending.append(_probability_argument(sampleable))
        elif isinstance(term, LetValue):
            pending.append(term.body)
            pending.extend(item.value for item in term.bindings)
        elif isinstance(term, PrimitiveApplication) and (
            term.primitive in _ELEMENTWISE
            or case is not None
            and (term is case.comparison or term.primitive == _CASE_DIVISION)
        ):
            pending.extend(term.arguments)
        elif case is not None and isinstance(term, Case) and term is case.case:
            pending.extend(branch.body for branch in term.branches)
        elif (
            case is not None
            and isinstance(term, ConstructorValue)
            and any(term is constructor for constructor in case.constructors)
        ):
            pending.extend(term.fields)
        elif case is not None and isinstance(term, WithTraceAddresses):
            pending.append(term.body)
        elif case is not None and isinstance(term, CallAddress):
            continue
        elif not (
            isinstance(term, Var)
            and term.local.type in (REAL, BOOL, UNIT)
            or isinstance(term, LiteralValue)
            and term.type in (REAL, BOOL, UNIT)
        ):
            if isinstance(term, PrimitiveApplication):
                subject = f"primitive `{term.name}`"
            else:
                subject = type(term).__name__
            raise JvpUnsupported(f"{subject} has no Bernoulli tensor axis rule")
    if case is not None and (selectors != 1 or draws != 2):
        raise JvpUnsupported("tensor cases need one selector and one draw per arm")


@dataclass(frozen=True, slots=True)
class BernoulliTensorResult:
    """Values and addressed log scores from one tensor-axis invocation.

    Parameters
    ----------
    values : torch.Tensor
        The source results in row order.
    log_scores : torch.Tensor
        The fixed-outcome log probability of each request, laid out as
        rows by request occurrence. Differentiation of these scores holds
        the Boolean outcomes fixed.
    tapes : tuple[BernoulliRowReplay, ...]
        The original source addresses and choices lifted to the row axis.
    source_passes : int
        The number of checked evaluator invocations.
    source_steps : int
        The sum of machine steps across the executed passes.
    lane_steps : int
        The sum of each pass's machine steps times its active row count.
    """

    values: torch.Tensor
    log_scores: torch.Tensor
    tapes: tuple[BernoulliRowReplay, ...]
    source_passes: int
    source_steps: int
    lane_steps: int = 0


class _CaseDraw(Exception):
    """A checked vector prefix suspended just before its one common draw.

    Parameters
    ----------
    address : _Address
        The complete canonical source address of the request.
    probability : torch.Tensor
        The probabilities in the active rows before sampling.
    """

    def __init__(self, address: _Address, probability: torch.Tensor) -> None:
        super().__init__(address)
        self.address = address
        self.probability = probability


def _require_finite_score(mass: torch.Tensor) -> None:
    """Check representability of each local score slope before using it.

    Parameters
    ----------
    mass : torch.Tensor
        The probability of the observed atom, or the smaller atom mass
        when a prefix has yet to draw.

    Raises
    ------
    JvpUnsupported
        If the working dtype cannot hold a reciprocal of the mass.
    """
    if not bool(torch.isfinite(mass.detach().reciprocal()).all()):
        raise JvpUnsupported(
            "a tensor case score derivative is outside working precision"
        )


class _BernoulliTensorMachine(Evaluator):
    """Run the usual source machine with row tensors at Bernoulli requests.

    Parameters
    ----------
    module : QiecModule
        The checked source and its callees.
    rows : torch.Tensor
        The local input table, which fixes the number domain and row extent.
    replay : tuple or None
        The original common request shape and Boolean replay columns, if given.
    generator : torch.Generator or None
        The generator used by each vectorized Bernoulli draw.
    fuel : int
        The source step bound of the pass.
    case : _CaseAxis or None
        The checked case whose selector and draw answers must be uniform
        within this pass. Without replay, suspend on that common draw.
    """

    def __init__(
        self,
        module: QiecModule,
        rows: torch.Tensor,
        replay: tuple[tuple[_Address, ...], torch.Tensor] | None,
        generator: torch.Generator | None,
        fuel: int,
        *,
        case: _CaseAxis | None = None,
    ) -> None:
        from quivers.continuous.tensor_backend import TensorBackend

        super().__init__(
            RuntimeAttachments(
                numbers=NumberDomain.of(rows), distributions=TensorBackend()
            ),
            module=module,
            fuel=fuel,
        )
        self.count = rows.shape[0]
        self.replay = replay
        self.generator = generator
        self.case = case
        self.addresses: list[_Address] = []
        self.seen: set[_Address] = set()
        self.choices: list[torch.Tensor] = []
        self.scores: list[torch.Tensor] = []

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
        """Read the normal tensor value or the uniform selector of this pass.

        Parameters
        ----------
        value : Value
            The reached expression.
        environment : Mapping
            The source bindings for the expression.

        Returns
        -------
        object
            The tensor value or a scalar Boolean for the checked comparison.

        Raises
        ------
        JvpUnsupported
            If the reached comparison differs among the rows of this pass.
        """
        result = super()._value(value, environment)
        # A cached static invocation can share an equivalent immutable body.
        if (
            self.case is not None
            and isinstance(value, PrimitiveApplication)
            and value.primitive == _CASE_PREDICATE
        ):
            assert isinstance(result, torch.Tensor)
            if result.ndim:
                if not bool((result == result[0]).all()):
                    raise JvpUnsupported(
                        "a tensor case selector differs within its cohort"
                    )
                return result[0]
        return result

    def _dispatch(
        self,
        request: RuntimeRequest,
        environment: Scope[Local, HeldValue],
        stack: list[_Frame],
        search_index: int,
    ) -> _Continue:
        """Draw or replay one independent row vector at the current site.

        Parameters
        ----------
        request : RuntimeRequest
            The evaluated request in its original source scope.
        environment : Scope[Local, HeldValue]
            The unchanged scope for the continuation.
        stack : list
            The unchanged source continuation stack.
        search_index : int
            The reference handler search position.

        Returns
        -------
        _Continue
            The continuation after supplying one Boolean per row.

        Raises
        ------
        JvpUnsupported
            If a runtime probability rounds outside the interior domain
            or a source address repeats. Constructor-case scores also
            need a finite reciprocal of the scored probability mass.
        ValueError
            If the request disagrees with the given tape.
        """
        address = request.address
        if address in self.seen:
            raise JvpUnsupported("a Bernoulli tensor request repeats a source address")
        sampleable = request.arguments[1]
        assert isinstance(sampleable, RuntimeDistribution)
        probability = sampleable.arguments["probs"]
        assert isinstance(probability, torch.Tensor)
        probs = probability.expand(self.count)
        if not bool(((probs > 0) & (probs < 1)).detach().all()):
            raise JvpUnsupported(
                "tensor Bernoulli probabilities must be strictly between zero and one"
            )
        if self.replay is None:
            if self.case is not None:
                _require_finite_score(torch.minimum(probs, 1 - probs))
                raise _CaseDraw(address, probs)
            bit = torch.bernoulli(probs.detach(), generator=self.generator).to(
                torch.bool
            )
        else:
            addresses, bits = self.replay
            position = len(self.addresses)
            if position == len(addresses) or addresses[position] != address:
                raise ValueError(
                    "tensor replay differs from the original request order"
                )
            bit = bits[:, position]
        if self.case is not None:
            _require_finite_score(torch.where(bit, probs, 1 - probs))
        distribution = sampleable.with_(batch=(self.count,), arguments={"probs": probs})
        score = distribution.log_prob(bit, keep_batch=True)
        self.seen.add(address)
        self.addresses.append(address)
        self.choices.append(bit)
        self.scores.append(score)
        answer = bit
        if self.case is not None:
            if not bool((bit == bit[0]).all()):
                raise JvpUnsupported("a tensor case replay differs within its cohort")
            answer = bit[0]
        return _Continue(
            current=_Returned(value=answer),
            environment=environment,
            resumption_path=request.resumption_path,
        )


class BernoulliTensorAxis:
    """Check source for native tensor execution over independent rows.

    Only scalar Real entries with one nominal Random row, acyclic calls,
    direct unplated Boolean Bernoulli draws and an elementwise polynomial
    fragment are admitted. Literal Boolean decisions use one pass. A
    two-constructor case with one strict scalar selector and a common direct
    Bernoulli helper may use bounded passes over the reached row subsets.
    Shared inputs are scalars and local inputs are tensor columns. Draws
    and log scores keep the row axis even when a probability is shared.

    Parameters
    ----------
    module : QiecModule
        The actual source module to validate and execute.
    computation : ComputationId
        The nominal scalar source entry.
    axis : str
        The label used with the ordinal in each row's addresses.
    shared_arity : int
        How many initial source parameters are shared across rows.
    max_rows, fuel : int
        The maximum number of rows and source steps in one invocation.

    Raises
    ------
    ValueError
        If the entry, axis label, arity or a bound is malformed.
    JvpUnsupported
        If the checked source leaves the tensor fragment.
    """

    def __init__(
        self,
        module: QiecModule,
        computation: ComputationId,
        axis: str,
        shared_arity: int,
        *,
        max_rows: int = 256,
        fuel: int = 50_000,
    ) -> None:
        _validate_limit("max_rows", max_rows)
        _validate_limit("fuel", fuel)
        if not isinstance(axis, str) or not axis.strip():
            raise ValueError("a Bernoulli tensor axis needs a nonempty label")
        fragment = "Bernoulli tensor axis"
        self._registry, self._owner = _random_computation(module, computation, fragment)
        if type(shared_arity) is not int or not 0 <= shared_arity <= len(
            self._owner.parameters
        ):
            raise ValueError("shared arity must be within the source entry arity")
        self._case = None
        try:
            _check_first_order_graph(
                module, self._owner, fragment, _check_request, require_acyclic=True
            )
            _check_tensor_terms(module, self._owner)
        except JvpUnsupported:
            self._case = _case_axis(module, self._owner)
            if self._case is None:
                raise
            _check_first_order_graph(
                module,
                self._owner,
                fragment,
                _check_request,
                require_acyclic=True,
                allow_closed_cases=True,
            )
            _check_tensor_terms(module, self._owner, self._case)
        self._module = module
        self._axis = axis
        self._shared_arity = shared_arity
        self._max_rows = max_rows
        self._fuel = fuel

    def _inputs(self, shared: torch.Tensor, rows: torch.Tensor) -> None:
        """Check the physical row and parameter axes before executing source.

        Parameters
        ----------
        shared, rows : torch.Tensor
            The common prefix and the row-local input table.

        Raises
        ------
        ValueError
            If the shapes, numbers, dtype or devices are incompatible.
        JvpLimit
            If the row bound is exceeded.
        """
        local_arity = len(self._owner.parameters) - self._shared_arity
        if (
            shared.shape != (self._shared_arity,)
            or rows.ndim != 2
            or rows.shape[0] < 1
            or rows.shape[1] != local_arity
        ):
            raise ValueError("shared and row tensors must match the source arity")
        if rows.shape[0] > self._max_rows:
            raise JvpLimit(f"Bernoulli tensor axis exceeds {self._max_rows} rows")
        if (
            rows.dtype not in (torch.float32, torch.float64)
            or shared.dtype != rows.dtype
            or shared.device != rows.device
            or rows.layout != torch.strided
            or shared.layout != torch.strided
            or not bool(torch.isfinite(shared.detach()).all())
            or not bool(torch.isfinite(rows.detach()).all())
        ):
            raise ValueError(
                "tensor axis inputs need finite dense floats in one domain"
            )

    def _replay_columns(
        self, tapes: tuple[BernoulliRowReplay, ...], rows: torch.Tensor
    ) -> tuple[tuple[_Address, ...], torch.Tensor]:
        """Make one fixed Boolean vector per original chronological request.

        Parameters
        ----------
        tapes : tuple[BernoulliRowReplay, ...]
            The supplied original addressed tapes in row order.
        rows : torch.Tensor
            The local inputs, determining row count and device.

        Returns
        -------
        tuple
            The common local address sequence and the row Boolean table.

        Raises
        ------
        ValueError
            If any axis, ordinal, address order, outcome type or count disagrees.
        """
        if len(tapes) != rows.shape[0]:
            raise ValueError("tensor replay needs one addressed tape per row")
        shape = tuple(address for address, _ in tapes[0].choices)
        for ordinal, tape in enumerate(tapes):
            if (
                tape.axis != self._axis
                or type(tape.ordinal) is not int
                or tape.ordinal != ordinal
                or tuple(address for address, _ in tape.choices) != shape
                or any(type(bit) is not bool for _, bit in tape.choices)
            ):
                raise ValueError(
                    "tensor replay tapes must match the axis and address order"
                )
        bits = torch.tensor(
            tuple(tuple(bit for _, bit in tape.choices) for tape in tapes),
            device=rows.device,
            dtype=torch.bool,
        )
        return shape, bits

    def _run(
        self,
        shared: torch.Tensor,
        rows: torch.Tensor,
        tapes: tuple[BernoulliRowReplay, ...] | None,
        generator: torch.Generator | None,
    ) -> BernoulliTensorResult:
        """Execute the checked source over its admitted row axis.

        Parameters
        ----------
        shared, rows : torch.Tensor
            The deterministic shared parameters and row-local inputs.
        tapes : tuple[BernoulliRowReplay, ...] or None
            The original addressed choices, or `None` to sample.
        generator : torch.Generator or None
            The generator for new draws.

        Returns
        -------
        BernoulliTensorResult
            Values, scores, original row tapes and measured source work.

        Raises
        ------
        JvpLimit
            If execution exceeds the source work bound.
        ValueError
            If the result, domain or replay is invalid.
        """
        self._inputs(shared, rows)
        replay = None if tapes is None else self._replay_columns(tapes, rows)
        if self._case is not None:
            return self._run_case(self._case, shared, rows, tapes, replay, generator)
        evaluator = _BernoulliTensorMachine(
            self._module, rows, replay, generator, self._fuel
        )
        args = (*shared.unbind(), *rows.unbind(dim=1))
        environment = dict(zip(self._owner.parameters, args, strict=True))
        try:
            result = evaluator.evaluate_validated(
                self._owner.body, self._registry, environment
            )
        except FuelExhaustedError as error:
            raise JvpLimit(str(error)) from error
        assert isinstance(result, torch.Tensor)
        values = result.expand(rows.shape[0])
        if not bool(torch.isfinite(values.detach()).all()):
            raise ValueError("tensor axis outputs must be finite")
        if replay is not None and len(replay[0]) != len(evaluator.addresses):
            raise ValueError("tensor replay contains unused requests")
        if evaluator.scores:
            scores = torch.stack(evaluator.scores, dim=1)
        else:
            scores = rows.new_empty((rows.shape[0], 0))
        if tapes is None:
            bits = (
                torch.stack(evaluator.choices, dim=1).tolist()
                if evaluator.choices
                else [() for _ in range(rows.shape[0])]
            )
            tapes = tuple(
                BernoulliRowReplay(
                    self._axis,
                    ordinal,
                    tuple(zip(evaluator.addresses, row, strict=True)),
                )
                for ordinal, row in enumerate(bits)
            )
        return BernoulliTensorResult(
            values, scores, tapes, 1, evaluator._steps, rows.shape[0] * evaluator._steps
        )

    def _run_case(
        self,
        case: _CaseAxis,
        shared: torch.Tensor,
        rows: torch.Tensor,
        tapes: tuple[BernoulliRowReplay, ...] | None,
        replay: tuple[tuple[_Address, ...], torch.Tensor] | None,
        generator: torch.Generator | None,
    ) -> BernoulliTensorResult:
        """Execute only the nonempty subsets that select each constructor and bit.

        Parameters
        ----------
        case : _CaseAxis
            The bounded case shape checked in this source graph.
        shared, rows : torch.Tensor
            The validated shared and row-local inputs.
        tapes : tuple of BernoulliRowReplay or None
            The original complete tapes, when replaying.
        replay : tuple or None
            Their common address and Boolean vector, when replaying.
        generator : torch.Generator or None
            The generator for a single source-ordered Bernoulli vector.

        Returns
        -------
        BernoulliTensorResult
            Values, scores and tapes gathered in original row order.

        Raises
        ------
        JvpBoundaryError
            If a row lies on the source comparison's threshold.
        JvpLimit
            If the passes together exhaust the source step budget.
        JvpUnsupported
            If the reached prefixes do not share one source address.
        ValueError
            If the replay differs from the selected computations.
        """
        count = rows.shape[0]
        coordinate = (
            shared[case.parameter]
            if case.parameter < self._shared_arity
            else rows[:, case.parameter - self._shared_arity]
        ).detach()
        try:
            threshold = coordinate.new_tensor(case.threshold)
        except (OverflowError, RuntimeError) as error:
            raise JvpUnsupported(
                "the tensor case threshold is outside working precision"
            ) from error
        if not bool(torch.isfinite(threshold)):
            raise JvpUnsupported(
                "the tensor case threshold is outside working precision"
            )
        if bool((coordinate == threshold).any()):
            raise JvpBoundaryError(
                "the tensor case selector is at its equality boundary"
            )
        selected = (coordinate > threshold).expand(count)
        tags = (selected, ~selected)
        passes = steps = lane_steps = 0

        def at(
            index: torch.Tensor,
            local_replay: tuple[tuple[_Address, ...], torch.Tensor] | None,
        ) -> tuple[torch.Tensor | _CaseDraw, _BernoulliTensorMachine]:
            """Run one nonempty tensor subset under the remaining aggregate fuel.

            Parameters
            ----------
            index : torch.Tensor
                Its original row ordinals.
            local_replay : tuple or None
                The fixed uniform answer vector, or a request to suspend.

            Returns
            -------
            tuple
                The completed row values or pending draw, and the pass.

            Raises
            ------
            JvpLimit
                If the aggregate work budget leaves too few steps.
            """
            nonlocal passes, steps, lane_steps
            remaining = self._fuel - steps
            if remaining <= 0:
                raise JvpLimit("tensor case passes exhaust the aggregate source fuel")
            local = rows.index_select(0, index)
            prefix = shared
            if local_replay is None:
                local = local.detach()
                prefix = prefix.detach()
            evaluator = _BernoulliTensorMachine(
                self._module, local, local_replay, None, remaining, case=case
            )
            args = (*prefix.unbind(), *local.unbind(dim=1))
            environment = dict(zip(self._owner.parameters, args, strict=True))
            try:
                value = evaluator.evaluate_validated(
                    self._owner.body, self._registry, environment
                )
            except _CaseDraw as pending:
                value = pending
            except FuelExhaustedError as error:
                raise JvpLimit(str(error)) from error
            assert isinstance(value, torch.Tensor | _CaseDraw)
            passes += 1
            steps += evaluator._steps
            lane_steps += index.numel() * evaluator._steps
            return value, evaluator

        if replay is None:
            probs = rows.new_zeros(count)
            address = None
            for tag in tags:
                index = tag.nonzero().squeeze(1)
                if index.numel() == 0:
                    continue
                pending, _ = at(index, None)
                if not isinstance(pending, _CaseDraw) or (
                    address is not None and address != pending.address
                ):
                    raise JvpUnsupported(
                        "tensor case rows need one common original draw address"
                    )
                address = pending.address
                probs = probs.index_copy(0, index, pending.probability)
            assert address is not None
            shape = (address,)
            bits = torch.bernoulli(probs, generator=generator).to(torch.bool)
        else:
            shape, columns = replay
            if len(shape) != 1:
                raise ValueError("tensor case replay needs one common addressed draw")
            bits = columns[:, 0]

        values = rows.new_zeros(count)
        scores = rows.new_zeros((count, 1))
        for tag in tags:
            for bit in (False, True):
                index = (tag & (bits == bit)).nonzero().squeeze(1)
                if index.numel() == 0:
                    continue
                value, evaluator = at(
                    index, (shape, bits.index_select(0, index)[:, None])
                )
                assert isinstance(value, torch.Tensor)
                if tuple(evaluator.addresses) != shape:
                    raise ValueError(
                        "tensor case replay differs from the original request order"
                    )
                values = values.index_copy(0, index, value.expand(index.numel()))
                scores = scores.index_copy(0, index, evaluator.scores[0][:, None])
        if not bool(torch.isfinite(values.detach()).all()):
            raise ValueError("tensor axis outputs must be finite")
        if tapes is None:
            tapes = tuple(
                BernoulliRowReplay(self._axis, ordinal, ((shape[0], bit),))
                for ordinal, bit in enumerate(bits.tolist())
            )
        return BernoulliTensorResult(values, scores, tapes, passes, steps, lane_steps)

    def replay(
        self,
        shared: torch.Tensor,
        rows: torch.Tensor,
        tapes: tuple[BernoulliRowReplay, ...],
    ) -> BernoulliTensorResult:
        """Run all rows on fixed addressed tapes, keeping scores and derivatives.

        Parameters
        ----------
        shared : torch.Tensor
            The shared input prefix, with one axis of source parameters.
        rows : torch.Tensor
            The local inputs, laid out as rows by local source parameters.
        tapes : tuple[BernoulliRowReplay, ...]
            One complete tape per row in ordinal order.

        Returns
        -------
        BernoulliTensorResult
            Finite tensor values and per-request log scores. PyTorch JVPs
            and VJPs differentiate both with the supplied choices fixed.

        Raises
        ------
        ValueError
            If the input tensors, probabilities or replay addresses fail a check.
        JvpLimit
            If the row or machine step limit is exceeded.
        """
        return self._run(shared, rows, tapes, None)

    def sample(
        self,
        shared: torch.Tensor,
        rows: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> BernoulliTensorResult:
        """Sample one independent Boolean vector at each source request.

        Parameters
        ----------
        shared : torch.Tensor
            The shared input prefix.
        rows : torch.Tensor
            The rows by local-parameters table.
        generator : torch.Generator or None
            Optional torch generator on the input device. Requests are
            visited in source order and each draw fills all row ordinals.

        Returns
        -------
        BernoulliTensorResult
            The sampled values, log scores and addressed replay tapes.
            Derivatives through selection require a separate estimator.

        Raises
        ------
        ValueError
            If the input tensors or probabilities fail a check.
        JvpLimit
            If the row or machine step limit is exceeded.
        """
        return self._run(shared, rows, None, generator)


__all__ = ["BernoulliTensorAxis", "BernoulliTensorResult"]
