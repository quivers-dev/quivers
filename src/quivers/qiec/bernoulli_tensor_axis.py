"""One-pass tensor execution for a checked independent Bernoulli row axis."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quivers._values import HeldValue
from quivers.qiec._random_site_graph import (
    _Address,
    _check_first_order_graph,
    _random_computation,
)
from quivers.qiec.bernoulli_estimators import _check_request, _probability_argument
from quivers.qiec.differentiation import JvpLimit, JvpUnsupported
from quivers.qiec.distributions import RuntimeDistribution
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
    Computation,
    ConditionalValue,
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
)
from quivers.qiec.types import BOOL, REAL, UNIT

_ELEMENTWISE = frozenset(
    PRIMITIVES[name].id
    for name in ("add_real", "sub_real", "mul_real", "neg_real", "bool_to_real")
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


def _check_tensor_terms(module: QiecModule, owner: NamedComputation) -> None:
    """Check the terms acted on by a physical row axis.

    Parameters
    ----------
    module : QiecModule
        The module whose typed rows and acyclic calls have already checked.
    owner : NamedComputation
        Its selected source entry.

    Raises
    ------
    JvpUnsupported
        If an active term could inspect the row axis or has no tensor rule.
    """
    declarations = {item.id: item for item in module.computations}
    seen = {owner.id}
    pending: list[Computation | Value] = [owner.body]
    while pending:
        term = pending.pop()
        if isinstance(term, Bind):
            pending.extend((term.then, term.first))
        elif isinstance(term, Return):
            pending.append(term.value)
        elif isinstance(term, If | ConditionalValue):
            pending.append(
                term.then if _literal_branch(term.condition) else term.otherwise
            )
        elif isinstance(term, Call):
            pending.extend(term.arguments)
            if term.callee not in seen:
                seen.add(term.callee)
                pending.append(declarations[term.callee].body)
        elif isinstance(term, Perform):
            sampleable = term.request.arguments[1]
            assert isinstance(sampleable, DistributionValue)
            pending.append(_probability_argument(sampleable))
        elif isinstance(term, LetValue):
            pending.append(term.body)
            pending.extend(item.value for item in term.bindings)
        elif isinstance(term, PrimitiveApplication) and term.primitive in _ELEMENTWISE:
            pending.extend(term.arguments)
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
        The machine steps executed once over the row axis.
    """

    values: torch.Tensor
    log_scores: torch.Tensor
    tapes: tuple[BernoulliRowReplay, ...]
    source_passes: int
    source_steps: int


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
    """

    def __init__(
        self,
        module: QiecModule,
        rows: torch.Tensor,
        replay: tuple[tuple[_Address, ...], torch.Tensor] | None,
        generator: torch.Generator | None,
        fuel: int,
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
        self.addresses: list[_Address] = []
        self.seen: set[_Address] = set()
        self.choices: list[torch.Tensor] = []
        self.scores: list[torch.Tensor] = []

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
            or a source address repeats.
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
        distribution = sampleable.with_(batch=(self.count,), arguments={"probs": probs})
        score = distribution.log_prob(bit, keep_batch=True)
        self.seen.add(address)
        self.addresses.append(address)
        self.choices.append(bit)
        self.scores.append(score)
        return _Continue(
            current=_Returned(value=bit),
            environment=environment,
            resumption_path=request.resumption_path,
        )


class BernoulliTensorAxis:
    """Check source for a single native tensor pass over independent rows.

    Only scalar Real entries with one nominal Random row, acyclic calls,
    direct unplated Boolean Bernoulli draws and an elementwise polynomial
    fragment are admitted. Conditionals require literal Boolean decisions
    and execute only the selected branch. Shared input values are scalars
    and local inputs are tensor columns. Draws and log scores keep the row
    axis even when a probability is shared.

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
        _check_first_order_graph(
            module, self._owner, fragment, _check_request, require_acyclic=True
        )
        _check_tensor_terms(module, self._owner)
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
        """Execute the checked source once on all rows.

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
        return BernoulliTensorResult(values, scores, tapes, 1, evaluator._steps)

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
