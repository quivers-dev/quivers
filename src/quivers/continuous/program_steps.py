"""The steps a `MonadicProgram` is a sequence of.

A program is built from six kinds of step, each a record of the
[`Step`][quivers.continuous.program_steps.Step] union:

- [`Draw`][quivers.continuous.program_steps.Draw] binds names to a
  sample from a morphism, applied to the program input or to earlier
  bindings;
- [`Observe`][quivers.continuous.program_steps.Observe] is a draw whose
  value is supplied as data and scored rather than sampled;
- [`Let`][quivers.continuous.program_steps.Let] binds a name
  deterministically, to a constant, an alias, or a function of earlier
  bindings;
- [`Score`][quivers.continuous.program_steps.Score] binds a name to a
  function of earlier bindings whose value also joins the log joint;
- [`Call`][quivers.continuous.program_steps.Call] runs a sub-program's
  steps in place and binds what it returns;
- [`Scan`][quivers.continuous.program_steps.Scan] runs a step program
  at every position of a sequence, threading a state, and binds the
  final state.

These are the records the DSL compiler emits, so a program assembled
from them in Python is the program the compiler would build from the
equivalent source.

A step function reads the environment by name. One that declares the
names it reads, through [`reading`][quivers.continuous.program_steps.reading],
is encoded as a step over those values alone; one that declares nothing
is encoded over every value bound before it, which is correct but makes
a long program's encoding grow with its length.
"""

from __future__ import annotations

import collections.abc
from typing import Literal, TypeVar

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.core.morphisms import Morphism

StepCallable = TypeVar("StepCallable", bound=collections.abc.Callable[..., object])

type Environment = dict[str, torch.Tensor]
"""The bindings a step function reads, by name."""

type StepFunction = collections.abc.Callable[[Environment], torch.Tensor]
"""A let or score step's function of the environment."""


def _check_indices(indices: tuple[str, ...]) -> tuple[str, ...]:
    if not indices:
        raise ValueError("an Indexed argument needs at least one index")
    return tuple(indices)


class Indexed(dx.Model):
    """A binding gathered at index bindings, as ``mu[group]`` in source.

    With ``mu`` of shape ``(K, ...)`` and an integer ``group`` of shape
    ``(N,)``, the argument has shape ``(N, ...)``. Several indices
    gather left to right.

    Attributes
    ----------
    name : str
        The binding gathered from.
    indices : tuple[str, ...]
        The integer bindings it is gathered at, at least one.
    """

    name: str
    indices: tuple[str, ...] = dx.field(converter=_check_indices)


type StepArgument = str | Indexed
"""One argument of a draw or observe step: a binding's name, or an
[`Indexed`][quivers.continuous.program_steps.Indexed] gather of one."""

READS = "reads"
"""The attribute a step function declares the names it reads under."""


def reading(
    function: StepCallable, names: collections.abc.Iterable[str]
) -> StepCallable:
    """Declare the environment names a step callable reads.

    Parameters
    ----------
    function : StepCallable
        The callable, a let or score step's value.
    names : Iterable[str]
        The names it reads from the environment it is called with.

    Returns
    -------
    StepCallable
        The same callable, carrying the names under ``reads``.
    """
    setattr(function, READS, frozenset(names))
    return function


def reads_of(function: collections.abc.Callable[..., object]) -> frozenset[str] | None:
    """The environment names a step callable declares it reads.

    Parameters
    ----------
    function : Callable
        A let or score step's value.

    Returns
    -------
    frozenset[str] | None
        The declared names, or ``None`` when the callable declares none
        and may read any bound value.
    """
    names = getattr(function, READS, None)
    if names is None:
        return None
    assert isinstance(names, frozenset)
    return names


class Step(dx.TaggedUnion, ValueModel, discriminator="kind"):
    """One step of a [`MonadicProgram`][quivers.continuous.MonadicProgram].

    The variants are [`Draw`][quivers.continuous.program_steps.Draw],
    [`Observe`][quivers.continuous.program_steps.Observe],
    [`Let`][quivers.continuous.program_steps.Let],
    [`Score`][quivers.continuous.program_steps.Score],
    [`Call`][quivers.continuous.program_steps.Call], and
    [`Scan`][quivers.continuous.program_steps.Scan].

    Steps compare by value over their names, tags, and arguments, and
    by identity over the morphisms, functions, constants, and tensors
    they hold: two steps are equal exactly when they agree on the
    former and share the latter objects.
    """


class Draw(Step):
    """Bind names to a sample from a morphism.

    In source, ``sample x <- f(a, b)``. With no arguments the morphism
    reads the program input; with one, it reads that binding; with
    several, it reads their concatenation along the feature axis, each
    placed by the declared parameter shape when the morphism is an
    inline distribution.

    Attributes
    ----------
    names : tuple[str, ...]
        The names bound. One name binds the sample; several destructure
        a tuple-returning sub-program, or split a product-codomain
        sample along its components.
    morphism : ContinuousMorphism | Morphism
        The kernel sampled from. A [`Morphism`][quivers.core.Morphism]
        is applied as a deterministic step through its tensor.
    args : tuple[StepArgument, ...] | None
        The bindings the morphism reads, or ``None`` for the program
        input.
    kind : Literal["draw"]
        The variant tag.
    """

    names: tuple[str, ...]
    morphism: ContinuousMorphism | Morphism = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    args: tuple[StepArgument, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    kind: Literal["draw"] = "draw"


class Observe(Step):
    """Score supplied data under a morphism.

    In source, ``observe y ~ f(a, b)``. The step reads its arguments as
    [`Draw`][quivers.continuous.program_steps.Draw] does; its value is
    supplied under its name through ``observations`` when sampling and
    through the intermediates when scoring, and its log density joins
    the log joint.

    Attributes
    ----------
    names : tuple[str, ...]
        The names bound to the observed value.
    morphism : ContinuousMorphism | Morphism
        The kernel the data are scored under.
    args : tuple[StepArgument, ...] | None
        The bindings the morphism reads, or ``None`` for the program
        input.
    kind : Literal["observe"]
        The variant tag.
    """

    names: tuple[str, ...]
    morphism: ContinuousMorphism | Morphism = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    args: tuple[StepArgument, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    kind: Literal["observe"] = "observe"


class Let(Step):
    """Bind a name deterministically.

    In source, ``let w = expression``.

    Attributes
    ----------
    name : str
        The name bound.
    value : float | str | StepFunction
        A constant, broadcast over the batch; the name of an earlier
        binding, which the step aliases; or a function of the
        environment, which may declare what it reads through
        [`reading`][quivers.continuous.program_steps.reading].
    kind : Literal["let"]
        The variant tag.
    """

    name: str
    value: float | str | StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    kind: Literal["let"] = "let"


class Score(Step):
    """Bind a name to a log-density contribution.

    The function's value is bound, as a let step's would be, and added
    to the log joint. The DSL compiler emits one for a ``marginalize``
    block, whose function sums the block's scores over the
    marginalized variable.

    Attributes
    ----------
    name : str
        The name bound.
    score : StepFunction
        The contribution, of shape ``(batch,)``.
    kind : Literal["score"]
        The variant tag.
    """

    name: str
    score: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    kind: Literal["score"] = "score"


class Call(Step):
    """Bind a name to what a sub-program returns, its steps run in place.

    In source, ``let y <- p(a, b)`` with ``p`` a program. The callee's
    steps run as steps of the caller: its parameters are bound to the
    arguments, its sites are sites of the caller under their own labels,
    and a name it reads without binding it is read from the caller's
    bindings, else from the caller's host data. Nothing the callee binds
    is visible to the caller apart from the value bound to ``name``.

    Attributes
    ----------
    name : str
        The name bound to the callee's return value.
    program : ContinuousMorphism
        The callee, a [`MonadicProgram`][quivers.continuous.MonadicProgram]
        with a single return variable.
    args : tuple[str, ...]
        The bindings passed for the callee's parameters, in order: one
        per declared parameter, or one for a callee without named
        parameters, which reads it as its input.
    kind : Literal["call"]
        The variant tag.
    """

    name: str
    program: ContinuousMorphism = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    args: tuple[str, ...] = ()
    kind: Literal["call"] = "call"


class Scan(Step):
    """Bind a name to the final state of a step program run along a sequence.

    In source, ``sample h <- f >> scan(cell)`` with ``cell`` a program.
    The step program takes one position's entry and the previous state,
    ``(x_t, h_prev)``, and returns the next state; the scan runs it at
    every position of the sequence, left to right, from the initial
    state, and binds the last state. The step program's steps are steps
    of the enclosing program, reached once per position: the ``n``-th
    time a run reaches a site labelled ``s`` it is the site ``"s@n"``,
    which a trace records and an observation, a clamp, or a joint
    density addresses. A name the step program reads without binding
    it is the enclosing program's binding of that name before the scan,
    else host data.

    Attributes
    ----------
    name : str
        The name bound to the final state.
    step : ContinuousMorphism
        The step program, a
        [`MonadicProgram`][quivers.continuous.MonadicProgram] with two
        parameters, the entry and the previous state, over
        ``A * H -> H``.
    source : str | None
        The binding holding the sequence, or ``None`` for the program
        input. A discrete entry's sequence is ``(batch, seq_len)``; a
        continuous entry's is ``(batch, seq_len, dim)`` or the folded
        ``(batch, seq_len * dim)``.
    initial : torch.Tensor | None
        The initial state, of shape ``(dim H,)`` and shared by every
        row, or ``None`` to start from zero. A learned initial state is
        an ``nn.Parameter``, which the enclosing program registers.
    initial_from : str | None
        A binding of the enclosing program, made before the scan, whose
        value is the initial state (``scan(cell, init=h0)``), or
        ``None``. A scalar binding is the one entry of a one-wide state.
        At most one of ``initial`` and ``initial_from`` is set.
    kind : Literal["scan"]
        The variant tag.
    """

    name: str
    step: ContinuousMorphism = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    source: str | None = None
    initial: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    initial_from: str | None = None
    kind: Literal["scan"] = "scan"


__all__ = [
    "READS",
    "Call",
    "Draw",
    "Environment",
    "Indexed",
    "Let",
    "Observe",
    "Scan",
    "Score",
    "Step",
    "StepArgument",
    "StepFunction",
    "reading",
    "reads_of",
]
