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
from torch.distributions import Distribution

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers._observation import ObservationEvent
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


MAPPING = "mapping"
"""The attribute a step function declares its mapping under."""


class StepMapping(dx.Model):
    """Whether a step function can be mapped over many positions.

    Attributes
    ----------
    obstacle : str | None
        Why it cannot be mapped, or ``None`` when it can.
    shared : frozenset[str]
        The names it reads whose values must be the same at every
        position, because what it computes is laid out by them.
    draws : bool
        Whether it draws at random, so that each position takes a draw of
        its own.
    """

    obstacle: str | None = None
    shared: frozenset[str] = frozenset()
    draws: bool = False


def mappable(
    function: StepCallable,
    obstacle: str | None = None,
    shared: collections.abc.Iterable[str] = (),
    draws: bool = False,
) -> StepCallable:
    """Declare whether a step callable can be mapped over many positions.

    A mappable callable computes its value with tensor operations alone:
    it branches on no value, and reads the shapes of one run. It is then
    evaluated at many positions by [`torch.func.vmap`][torch.func.vmap],
    with no batch axis for it to handle. The DSL compiler declares it of
    the functions it builds. A callable written for an explicit batch
    axis declares ``batched=True`` on its step instead.

    Parameters
    ----------
    function : StepCallable
        The callable, a let or score step's value.
    obstacle : str | None
        Why it cannot be mapped, or ``None`` when it can.
    shared : Iterable[str]
        The names it reads whose values must be the same at every
        position: the number of entries it computes depends on them.
    draws : bool
        Whether it draws at random.

    Returns
    -------
    StepCallable
        The same callable, carrying the declaration.
    """
    setattr(
        function,
        MAPPING,
        StepMapping(obstacle=obstacle, shared=frozenset(shared), draws=draws),
    )
    return function


def mapping_of(function: collections.abc.Callable[..., object]) -> StepMapping:
    """The mapping a step callable declares.

    Parameters
    ----------
    function : Callable
        A let or score step's value.

    Returns
    -------
    StepMapping
        The declaration; for a callable that makes none, an obstacle.
    """
    declared = getattr(function, MAPPING, None)
    if declared is None:
        return StepMapping(
            obstacle="its callable is opaque and the step does not declare batched=True"
        )
    assert isinstance(declared, StepMapping)
    return declared


class RunConditions(ValueModel):
    """What a run does to the factors of a score besides evaluating them.

    Parameters
    ----------
    fixed : frozenset[str]
        The responses an intervention fixes. The conditional factor of a
        fixed response is removed from the joint; its value is read by
        everything after it.
    subsets : dict[str, torch.Tensor]
        The positions kept of each subsampled plate, by name, distinct and
        in increasing order.
    """

    fixed: frozenset[str] = frozenset()
    subsets: dict[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def unconditioned(self) -> bool:
        """Whether the run evaluates every factor in full.

        Returns
        -------
        bool
            ``True`` when no response is fixed and no plate subsampled.
        """
        return not self.fixed and not self.subsets


class PlateMembers(ValueModel):
    """The elements of a plate that a factored score reads, and their units.

    Parameters
    ----------
    plate : str
        The plate's name: the object of a response's rows, or of the
        grouping plate of a block inside the one that scores.
    extent : int
        The plate's extent.
    units : StepFunction
        From the environment to the unit of each element, ``(elements,)``.
    positions : StepFunction
        From the environment to the position of each element along the
        plate, ``(elements,)``.
    """

    plate: str
    extent: int
    units: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    positions: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


class UnitRows(ValueModel):
    """Which unit of a factored score each row of a response belongs to.

    Parameters
    ----------
    response : str
        The observed name whose rows the map sends to units.
    units : StepFunction
        The row map: from the environment to a long tensor of shape
        ``(rows,)`` holding each row's unit, counted row-major over the
        unit axes. It declares what it reads through
        [`reading`][quivers.continuous.program_steps.reading].
    one_row_per_unit : str
        The change to the program that would put each row in a unit of its
        own, as a sentence for a refusal to state. The changed program is
        another model.
    """

    response: str
    units: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    one_row_per_unit: str


class UnitFactorization(ValueModel):
    """How a score is a sum of one log weight per unit.

    A score ``S`` factors over a set of units ``U`` when
    ``S = sum_u w(u)``, with ``w(u)`` reading the rows that the row maps
    send to ``u`` and no other row. The units of a grouped ``marginalize``
    block are the positions of its grouping plate; a block with one latent
    for all its rows has one unit, and its ``axes`` are empty.

    A row of a response has a log weight of its own exactly when its unit
    holds that row and no other row of any response. The weights of the
    units exist in every case.

    Parameters
    ----------
    latent : str
        The name the score sums out.
    axes : tuple[str, ...]
        The objects whose product indexes the units, outermost first.
    sizes : tuple[int, ...]
        The extent of each axis.
    weights : StepFunction
        From the environment to the log weight of each unit, of shape
        ``(prod(sizes),)``, row-major over the axes. Their sum is the
        score's contribution.
    raw_weights : StepFunction
        The same weights with every likelihood power of the score's
        observations left out; ``weights`` itself when there is none.
    rows : tuple[UnitRows, ...]
        The row map of each response the score reads.
    density_obstacle : str | None
        Why the weight of a unit is not the log density of its rows, or
        ``None`` when it is.
    members : tuple[PlateMembers, ...]
        The plates whose elements lie in the units, besides the unit axes.
    under : Callable[[RunConditions], UnitFactorization] or None
        The factorization in a run under conditions; ``None`` for a score
        that no condition changes.
    """

    latent: str
    axes: tuple[str, ...]
    sizes: tuple[int, ...]
    weights: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    raw_weights: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    rows: tuple[UnitRows, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    density_obstacle: str | None = None
    members: tuple[PlateMembers, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    under: collections.abc.Callable[[RunConditions], UnitFactorization] | None = (
        dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    )

    @property
    def reads(self) -> frozenset[str] | None:
        """The names the factorization reads.

        Returns
        -------
        frozenset[str] | None
            The names the weights and the row maps read, or ``None`` when
            one of them declares none and may read any bound value.
        """
        names: set[str] = set()
        for function in (
            self.weights,
            self.raw_weights,
            *(item.units for item in self.rows),
            *(item.units for item in self.members),
            *(item.positions for item in self.members),
        ):
            declared = reads_of(function)
            if declared is None:
                return None
            names |= declared
        return frozenset(names)


class ResponseLaw(ValueModel):
    """The law of a response whose density a score step carries.

    The response of a ``marginalize`` block has no density of its own in
    the joint: the block's score holds it. The response still has a law
    given the values the block does not sum out, and a run that is given
    no value for it draws from that law.

    Parameters
    ----------
    marginal : Callable[[Environment], Distribution]
        From the environment to the law of each row of the response with
        the block's latents summed out.
    draw : StepFunction
        From the environment to one joint draw of all the rows. Rows that
        share a latent share its drawn value, and a response that follows
        another of the same block is drawn given that one's value.
    row_axis : str or None
        The plate of the response's rows.
    under : Callable[[RunConditions], ResponseLaw] or None
        The law in a run under conditions; ``None`` for a law that no
        condition changes.
    """

    marginal: collections.abc.Callable[[Environment], Distribution] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    draw: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    row_axis: str | None = None
    under: collections.abc.Callable[[RunConditions], ResponseLaw] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def reads(self) -> frozenset[str] | None:
        """The names the law reads.

        Returns
        -------
        frozenset[str] | None
            The names its two functions read, or ``None`` when one of them
            declares none and may read any bound value.
        """
        marginal, draw = reads_of(self.marginal), reads_of(self.draw)
        if marginal is None or draw is None:
            return None
        return marginal | draw


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
    the log joint. If an earlier binding already supplies the name,
    the observation scores that value. It adds no latent coordinate;
    a computed plated value retains the plate's row and event axes.

    Attributes
    ----------
    names : tuple[str, ...]
        The names bound to the observed value.
    morphism : ContinuousMorphism | Morphism
        The kernel the data are scored under.
    args : tuple[StepArgument, ...] | None
        The bindings the morphism reads, or ``None`` for the program
        input.
    weights : float | StepArgument | None
        A nonnegative finite power-likelihood weight, or a binding with one
        weight per observation row. It changes scoring and leaves sampling
        and the underlying response density unchanged.
    event : ObservationEvent or None
        Recording and truncation roles, resolved from the same environment as arguments.
    law : ResponseLaw or None
        The law a run draws the response from when a score step carries
        its density; ``None`` for an observation that draws from its
        morphism at its arguments.
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
    weights: float | StepArgument | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    event: ObservationEvent | None = None
    law: ResponseLaw | None = dx.field(
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
    batched : bool
        Whether a function ``value`` accepts a leading batch axis. Over
        many positions it is then called once, with a batch axis of
        length ``B`` leading every value it reads that depends on the
        position, ahead of the axes the value has in one run, and a value
        every position shares at its own shape; its result leads with
        that axis too. A function that declares neither this nor
        [`mappable`][quivers.continuous.program_steps.mappable] makes the
        program unbatchable.
    under : Callable[[RunConditions], StepFunction] or None
        The function in a run under conditions, for a binding that holds a
        factor of a score; ``None`` for one that no condition changes. It
        reads no name the function of ``value`` does not read.
    kind : Literal["let"]
        The variant tag.
    """

    name: str
    value: int | float | str | StepFunction = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    batched: bool = False
    under: collections.abc.Callable[[RunConditions], StepFunction] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
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
    role : {"potential", "likelihood"}
        Explicit potentials are retained under prior-only inference. A
        compiler-generated marginalized response likelihood is classified
        separately so the common observation policy can omit it.
    batched : bool
        Whether ``score`` accepts a leading batch axis, as
        [`Let.batched`][quivers.continuous.program_steps.Let] declares it
        of a let step's function. Over ``B`` positions the contribution
        then has shape ``(B, ...)``, and each row's trailing entries are
        summed as one run's are.
    units : UnitFactorization or None
        How the contribution is a sum of one log weight per unit, or
        ``None`` for a score that declares no factorization.
    under : Callable[[RunConditions], StepFunction] or None
        The function in a run under conditions; ``None`` for a score that
        no condition changes. It reads no name the step does not read.
    kind : Literal["score"]
        The variant tag.
    """

    name: str
    score: StepFunction = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    role: Literal["potential", "likelihood"] = "potential"
    batched: bool = False
    units: UnitFactorization | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    under: collections.abc.Callable[[RunConditions], StepFunction] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    kind: Literal["score"] = "score"

    @property
    def reads(self) -> frozenset[str] | None:
        """The names the step reads.

        Returns
        -------
        frozenset[str] | None
            The names its function and its factorization read, or ``None``
            when one of them declares none and may read any bound value.
        """
        declared = reads_of(self.score)
        if self.units is None or declared is None:
            return declared
        factored = self.units.reads
        return None if factored is None else declared | factored


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
    """Bind a name to the state of a step program run along a sequence.

    In source, ``sample h <- f >> scan(cell)`` with ``cell`` a program.
    The step program takes one position's entry and the previous state,
    ``(x_t, h_prev)``, and returns the next state; the scan runs it at
    every position of the sequence, left to right (or right to left,
    for a reversed scan), from the initial state, and binds the last
    state, or every state for ``outputs="all"``. The step program's
    steps are steps of the enclosing program, reached once per position
    visited: the ``n``-th time a run reaches a site labelled ``s`` it is
    the site ``"s@n"``,
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
    outputs : Literal["last", "all"]
        ``"last"`` binds the state after the last position visited;
        ``"all"`` binds the state after every position, ``(batch,
        seq_len, dim H)`` in input order.
    reverse : bool
        Whether the scan visits the positions from the last to the
        first. Its sites are numbered in the order it visits them, so
        ``"s@0"`` is the site at the last position; a sequence of
        values given under ``"s"`` is still read in input order.
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
    outputs: Literal["last", "all"] = "last"
    reverse: bool = False
    kind: Literal["scan"] = "scan"


__all__ = [
    "MAPPING",
    "READS",
    "Call",
    "Draw",
    "Environment",
    "Indexed",
    "Let",
    "Observe",
    "PlateMembers",
    "ResponseLaw",
    "RunConditions",
    "Scan",
    "Score",
    "Step",
    "StepArgument",
    "StepFunction",
    "StepMapping",
    "UnitFactorization",
    "UnitRows",
    "mappable",
    "mapping_of",
    "reading",
    "reads_of",
]
