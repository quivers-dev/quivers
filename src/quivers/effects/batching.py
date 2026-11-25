"""Which programs can be scored at many positions in one pass.

A program's log joint is evaluated at ``B`` positions by running its
steps once, on values that carry the position axis
(`quivers._positions`). Whether a step can run this way is declared, by
the step for a Python callable the program cannot inspect and by the
morphism for a distribution, and is never found out by running it.
`batching_obstacles` reads the declarations of a program and of the
programs it calls and scans, and
[`require_batchable`][quivers.effects.joint_plan.require_batchable] raises
`UnbatchableProgramError` when any of them stands in the way.
"""

from __future__ import annotations

from collections.abc import Collection

from quivers._positions import PositionLayoutError
from quivers._values import ValueModel
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.program_steps import (
    Call,
    Draw,
    Indexed,
    Let,
    Observe,
    Scan,
    Score,
    Step,
    StepArgument,
    StepMapping,
    mapping_of,
    reads_of,
)
from quivers.continuous.programs import MonadicProgram


class BatchingObstacle(ValueModel):
    """A step that cannot be evaluated at many positions, and why.

    Attributes
    ----------
    step : str
        The step: its kind and the names it binds.
    reason : str
        Why it cannot.
    """

    step: str
    reason: str


class UnbatchableProgramError(ValueError):
    """A program that cannot be scored at many positions in one pass.

    Parameters
    ----------
    obstacles : tuple[BatchingObstacle, ...]
        The steps standing in the way, in program order, and why each
        does.

    Attributes
    ----------
    obstacles : tuple[BatchingObstacle, ...]
        The same.
    """

    def __init__(self, obstacles: tuple[BatchingObstacle, ...]) -> None:
        self.obstacles = obstacles
        listed = "; ".join(f"{item.step}: {item.reason}" for item in obstacles)
        super().__init__(
            f"the program cannot be scored at many positions in one pass: {listed}"
        )


def _bound_names(step: Step) -> tuple[str, ...]:
    """The names a step binds.

    Parameters
    ----------
    step : Step
        The step.

    Returns
    -------
    tuple[str, ...]
        The names.
    """
    if isinstance(step, Draw | Observe):
        return step.names
    assert isinstance(step, Let | Score | Call | Scan)
    return (step.name,)


def _argument_names(argument: float | StepArgument | None) -> frozenset[str]:
    """The bindings one argument of a step reads.

    Parameters
    ----------
    argument : float | StepArgument | None
        The argument.

    Returns
    -------
    frozenset[str]
        The names, empty for a number.
    """
    if isinstance(argument, Indexed):
        return frozenset({argument.name, *argument.indices})
    return frozenset({argument} if isinstance(argument, str) else ())


def _declaration(program: MonadicProgram, step: Step) -> StepMapping:
    """The mapping a step declares.

    Parameters
    ----------
    program : MonadicProgram
        The program the step belongs to.
    step : Step
        The step.

    Returns
    -------
    StepMapping
        Its declaration: the morphism's for a draw or an observation,
        with the names of an observation's weights shared, since a weight
        of zero removes its row from the score; the callable's for a let
        or a score, unless the step declares it ``batched``.
    """
    if isinstance(step, Draw | Observe):
        morphism = program.step_module(step)
        return StepMapping(
            obstacle=morphism.mapping_obstacle()
            if isinstance(morphism, ContinuousMorphism)
            else None,
            shared=_argument_names(step.weights)
            if isinstance(step, Observe)
            else frozenset(),
        )
    if isinstance(step, Let | Score) and not step.batched:
        function = step.value if isinstance(step, Let) else step.score
        if callable(function):
            return mapping_of(function)
    return StepMapping()


def _positional(step: Step, depends: frozenset[str]) -> bool:
    """Whether what a let, score, call, or scan binds depends on a latent.

    Parameters
    ----------
    step : Step
        The step.
    depends : frozenset[str]
        The names bound before it that do.

    Returns
    -------
    bool
        ``True`` when a name it reads does, or when the names it reads
        are not declared and some name does.
    """
    function = (
        step.value
        if isinstance(step, Let)
        else step.score
        if isinstance(step, Score)
        else None
    )
    if isinstance(function, str):
        return function in depends
    reads = reads_of(function) if callable(function) else None
    if reads is None:
        constant = isinstance(step, Let) and not callable(function)
        return bool(depends) and not constant
    return bool(reads & depends)


def batching_obstacles(
    program: MonadicProgram,
    latents: Collection[str],
    given: Collection[str],
) -> tuple[BatchingObstacle, ...]:
    """The steps of a program that cannot be evaluated at many positions.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    latents : Collection[str]
        The sites whose values the position gives.
    given : Collection[str]
        The sites and host data whose values every position shares.

    Returns
    -------
    tuple[BatchingObstacle, ...]
        One entry per step, in the order the program runs them, those of
        the programs it calls and scans included. Empty for a program
        that can be scored at many positions.
    """
    found: list[BatchingObstacle] = []
    depends = frozenset(latents)
    bound = frozenset({*latents, *given})
    for step in program.steps:
        names = _bound_names(step)
        inner = (
            step.program
            if isinstance(step, Call)
            else step.step
            if isinstance(step, Scan)
            else None
        )
        if isinstance(inner, MonadicProgram):
            own = inner_names(inner)
            found.extend(batching_obstacles(inner, depends | own, own))
        declared = _declaration(program, step)
        reason = declared.obstacle
        varying = sorted(declared.shared & depends)
        if reason is None and varying:
            reason = (
                f"it needs the same value of {varying} at every position, and that "
                "value depends on a latent"
            )
        if (
            reason is None
            and isinstance(step, Draw | Observe)
            and not bound.issuperset(names)
        ):
            reason = "its site is given no value, so every evaluation draws one"
        if reason is not None:
            found.append(
                BatchingObstacle(
                    step=f"{step.kind} step {', '.join(map(repr, names))}",
                    reason=reason,
                )
            )
        bound = bound.union(names)
        if not isinstance(step, Draw | Observe) and _positional(step, depends):
            depends = depends.union(names)
    return tuple(found)


def inner_names(program: MonadicProgram) -> frozenset[str]:
    """The sites of a program another one runs.

    Parameters
    ----------
    program : MonadicProgram
        The called or scanned program.

    Returns
    -------
    frozenset[str]
        The names its draws bind. A site of it has a label of its own in
        the run, so whether it is given a value is checked when the run
        reaches it.
    """
    return frozenset(
        name
        for step in program.steps
        if isinstance(step, Draw | Observe)
        for name in step.names
    )


def draws_at_random(program: MonadicProgram) -> bool:
    """Whether a program draws at random even when every site has a value.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    bool
        ``True`` when a let or score step of it, or of a program it calls
        or scans, declares that it draws.
    """
    for step in program.steps:
        inner = (
            step.program
            if isinstance(step, Call)
            else step.step
            if isinstance(step, Scan)
            else None
        )
        if isinstance(inner, MonadicProgram) and draws_at_random(inner):
            return True
        if _declaration(program, step).draws:
            return True
    return False


__all__ = [
    "BatchingObstacle",
    "PositionLayoutError",
    "UnbatchableProgramError",
    "batching_obstacles",
]
