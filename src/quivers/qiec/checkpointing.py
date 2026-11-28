"""Checkpointed reverse differentiation of deterministic scalar iterations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from quivers.qiec.checking import KernelRegistry
from quivers.qiec.contracts import DerivativeSemantics
from quivers.qiec.differentiation import (
    JvpLimit,
    _bound,
    _fraction,
    _prepare_scalar,
    _ReverseEvaluator,
)
from quivers.qiec.module import NamedComputation, QiecModule


@dataclass(frozen=True, slots=True)
class ExactCheckpointedGradient:
    """A finite iteration's output, exact gradient, and replay counts.

    Parameters
    ----------
    value : Fraction
        The state after all transitions.
    gradient : tuple[Fraction, ...]
        Derivatives with respect to the initial state and shared parameters.
    replay_steps : int
        Transitions executed for a primal checkpoint without a reverse graph.
    reverse_steps : int
        Transitions executed with their reverse graph, once per iterate.
    peak_saved_states : int
        The maximum number of simultaneously saved state checkpoints,
        including the initial state for a nonempty iteration.
    """

    value: Fraction
    gradient: tuple[Fraction, ...]
    replay_steps: int
    reverse_steps: int
    peak_saved_states: int

    @property
    def semantics(self) -> DerivativeSemantics:
        """Report the exact finite composition that was differentiated.

        Returns
        -------
        DerivativeSemantics
            The iteration count, or exact identity for the empty iteration.
        """
        if self.reverse_steps:
            return DerivativeSemantics.unrolled_iterations(self.reverse_steps)
        return DerivativeSemantics.exact()


class _CheckpointSweep:
    """Recompute deterministic state checkpoints in reverse segment order.

    Parameters
    ----------
    module : QiecModule
        The transition's checked module.
    transition : NamedComputation
        The checked scalar transition.
    registry : KernelRegistry
        The registry used to check that transition.
    initial : Fraction
        The exact initial state.
    parameters : tuple[Fraction, ...]
        The exact parameters shared by every transition.
    fuel : int
        The reference step bound per transition.
    max_work : int
        The bound on transition evaluations, including replays.
    """

    def __init__(
        self,
        module: QiecModule,
        transition: NamedComputation,
        registry: KernelRegistry,
        initial: Fraction,
        parameters: tuple[Fraction, ...],
        fuel: int,
        max_work: int,
    ) -> None:
        self.module = module
        self.transition = transition
        self.registry = registry
        self.parameters = parameters
        self.fuel = fuel
        self.max_work = max_work
        self.checkpoints = [initial]
        self.peak_saved_states = 1
        self.final: Fraction | None = None
        self.parameter_gradient = [Fraction()] * len(parameters)
        self.replay_steps = 0
        self.reverse_steps = 0

    def _evaluate(
        self, state: Fraction, *, record: bool
    ) -> tuple[Fraction, tuple[Fraction, ...]]:
        """Run a single transition and charge its actual evaluation.

        Parameters
        ----------
        state : Fraction
            The transition's input state.
        record : bool
            Whether to differentiate the transition.

        Returns
        -------
        tuple[Fraction, tuple[Fraction, ...]]
            The resulting state and the ordered gradient if recorded.

        Raises
        ------
        JvpLimit
            If the run cannot afford another transition evaluation.
        """
        if self.replay_steps + self.reverse_steps == self.max_work:
            raise JvpLimit("checkpointed gradient transition limit exhausted")
        evaluator = _ReverseEvaluator(self.module, self.fuel)
        point = (state, *self.parameters)
        output = evaluator.run(self.transition, self.registry, point, record=record)
        if record:
            self.reverse_steps += 1
            return output.value, evaluator.gradient(output, len(point))
        self.replay_steps += 1
        return output.value, ()

    def pullback(self, steps: int, cotangent: Fraction) -> Fraction:
        """Propagate a segment-end seed back to its saved initial state.

        Parameters
        ----------
        steps : int
            The number of transitions in the nonempty segment.
        cotangent : Fraction
            The adjoint at the end of the segment.

        Returns
        -------
        Fraction
            The adjoint of its starting state. Shared parameter
            contributions are accumulated across the segment.
        """
        if steps == 1:
            after, gradient = self._evaluate(self.checkpoints[-1], record=True)
            if self.final is None:
                self.final = after
            for index, partial in enumerate(gradient[1:]):
                self.parameter_gradient[index] += cotangent * partial
            before = cotangent * gradient[0]
            _bound(before, *self.parameter_gradient)
            return before

        left = steps // 2
        middle = self.checkpoints[-1]
        for _ in range(left):
            middle, _ = self._evaluate(middle, record=False)
        self.checkpoints.append(middle)
        del middle
        self.peak_saved_states = max(self.peak_saved_states, len(self.checkpoints))
        middle_cotangent = self.pullback(steps - left, cotangent)
        self.checkpoints.pop()
        return self.pullback(left, middle_cotangent)


def exact_checkpointed_gradient(
    module: QiecModule,
    computation: str,
    initial: int | Fraction,
    arguments: tuple[int | Fraction, ...],
    steps: int,
    *,
    max_work: int = 65_536,
    fuel: int = 65_536,
) -> ExactCheckpointedGradient:
    """Differentiate a finite scalar iteration using deterministic replay.

    The transition takes its current state followed by the parameters
    shared across all steps. It runs in the closed regular scalar fragment
    of :func:`~quivers.qiec.differentiation.exact_gradient`. The schedule
    saves a midpoint state, reverses the second segment, and restores the
    earlier state to reverse the first segment. Every visited transition
    must satisfy the derivative fragment's strict arithmetic and branch
    conditions, including transitions with zero output cotangent.

    For positive ``steps``, the schedule saves at most one state per
    bisection level plus the initial state. Each step is differentiated
    once and may also replay once at each containing bisection level.
    These counts are in units of transitions and saved scalar states.

    Parameters
    ----------
    module : QiecModule
        The checked module declaring the transition.
    computation : str
        The name of the scalar transition.
    initial : int or Fraction
        The initial state of the iteration.
    arguments : tuple[int or Fraction, ...]
        The shared parameters in positional order.
    steps : int
        The nonnegative number of iterations.
    max_work : int
        The maximum number of transition evaluations, including replays.
    fuel : int
        The maximum reference machine steps per transition.

    Returns
    -------
    ExactCheckpointedGradient
        The output, ordered gradient and measured replay counts.

    Raises
    ------
    JvpUnsupported
        If the transition is outside the closed scalar fragment.
    JvpBoundaryError
        If an input-dependent comparison is at equality.
    JvpLimit
        If bounded work, fuel or arithmetic is exhausted.
    ValueError
        If an argument or resource limit is malformed.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if type(max_work) is not int or max_work < 1:
        raise ValueError("max_work must be a positive integer")
    registry, transition = _prepare_scalar(module, computation, fuel)
    if len(transition.parameters) != len(arguments) + 1:
        raise ValueError("state and arguments must match the transition's arity")
    state = _fraction(initial)
    parameters = tuple(map(_fraction, arguments))
    _bound(state, *parameters)
    if steps == 0:
        return ExactCheckpointedGradient(
            state, (Fraction(1), *((Fraction(),) * len(parameters))), 0, 0, 0
        )
    if steps > max_work:
        raise JvpLimit("checkpointed gradient transition limit exhausted")
    sweep = _CheckpointSweep(
        module, transition, registry, state, parameters, fuel, max_work
    )
    try:
        cotangent = sweep.pullback(steps, Fraction(1))
    except RecursionError as error:
        raise JvpLimit("checkpoint nesting limit exhausted") from error
    assert sweep.final is not None
    return ExactCheckpointedGradient(
        sweep.final,
        (cotangent, *sweep.parameter_gradient),
        sweep.replay_steps,
        sweep.reverse_steps,
        sweep.peak_saved_states,
    )


__all__ = ["ExactCheckpointedGradient", "exact_checkpointed_gradient"]
