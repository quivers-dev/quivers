"""The kernel encoding of a `MonadicProgram`.

A `MonadicProgram` is a sequence of host steps: a draw from a torch
morphism at an input built from earlier bindings, a deterministic binding
computed by a host closure, or a score contributed by one. The kernel has
no term for a host function, and does not need one: each step's host part
is an instance of the prelude's ``Compute`` interface, performed on the
values it reads (the arguments of a draw, the names a closure declares,
or every value bound so far when it declares none), and each draw is the
canonical ``Random.sample`` request on the sampleable the host part
returns.

The computation this module builds therefore makes the program's data flow
and effect structure explicit while every tensor operation stays in the
host, behind a handler a runtime provider installs. The user's effect
handlers are lexical handlers of the same ``random`` and ``score``
instances, so conditioning, intervention, replay, reweighting, and tracing
compose by the kernel's own rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from quivers.continuous.program_steps import (
    Call as CallStep,
    Indexed,
    StepArgument,
    Draw,
    Let,
    Observe,
    Scan,
    Score,
    Step,
    reads_of,
)

from quivers.continuous.programs import MonadicProgram
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import ContinuousSpace, ProductSpace
from quivers.core.morphisms import extract_morphism
from quivers.core.objects import SetObject
from quivers.qiec.builtins import (
    COMPUTE,
    COMPUTE_APPLY,
    COMPUTE_EFFECT,
    PARAM,
    PARAM_EFFECT,
    RANDOM,
    RANDOM_EFFECT,
    RANDOM_SAMPLE,
    SCORE,
    SCORE_ADD,
    SCORE_EFFECT,
)
from quivers.qiec.canonical import LOG_WEIGHT, sampleable_type, site_type, tensor_type
from quivers.qiec.effects import (
    ComputationType,
    EffectRequest,
    EffectRow,
    RowEntry,
    instantiate_effect,
)
from quivers.qiec.identifiers import (
    ComputationId,
    EffectInstanceId,
    SiteProvenance,
    SourceOrigin,
)
from quivers.qiec.module import NamedComputation, NamedEffectInstance, QiecModule
from quivers.qiec.primitives import primitive
from quivers.qiec.terms import (
    Bind,
    Call,
    Computation,
    If,
    LiteralValue,
    Local,
    Perform,
    PrimitiveApplication,
    Projection,
    Return,
    SiteValue,
    TupleValue,
    Value,
    Var,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    UNIT,
    EffectRef,
    IndexLiteral,
    TypeExpr,
    product_type,
)
from quivers.qiec.kinds import NAT

#: The source protocol the encoding claims; there is no source text.
SOURCE_PROTOCOL = "quivers-program/host"

#: The names of the canonical instances the encoding allocates.
RANDOM_INSTANCE = "random"
SCORE_INSTANCE = "score"
PARAM_INSTANCE = "param"

#: The name of the computation holding the program.
ENTRY = "program"

#: The local the program's input is bound to, which no step variable may
#: reuse; step callables read the input from the environment under it.
INPUT_NAME = "_x_input"


@dataclass(frozen=True, slots=True)
class HostStep:
    """One host part of the program, as the runtime provider serves it.

    Parameters
    ----------
    instance
        The ``Compute`` instance the step performs on.
    effect
        The applied ``Compute[X, A]`` interface of the instance.
    kind
        ``"input"`` for the program's parameter split, ``"draw"`` for a
        morphism applied to its input, ``"apply"`` for a categorical
        morphism's tensor contracted against its input, ``"bind"`` for
        a destructuring split, ``"let"`` for a deterministic binding,
        ``"score"`` for a scored binding, ``"pack"`` for a called
        program's input assembled from its parameters, ``"site"`` for
        the address of a site reached once per scan position,
        ``"scan_length"`` and ``"scan_initial"`` for a scan's sequence
        length and initial state, ``"scan_entry"`` for the entry at a
        scan position, and ``"scan_state"`` for the state a scan's step
        returns.
    spec
        The step record the host part serves, or ``None`` for the input
        split and an input pack.
    environment
        The names of the locals the step reads, in the order the
        environment tuple carries them.
    answer_type
        The type the host part returns.
    site
        The site label of a draw step or a site address, else ``None``.
    program
        The program ``spec`` belongs to: the encoded program, or one a
        call or a scan runs in place.
    scanned
        Whether the step runs once per position of a scan.
    position
        The step's position among the kernel's host steps.
    """

    instance: EffectInstanceId
    effect: EffectRef
    kind: str
    spec: Step | None
    environment: tuple[str, ...]
    answer_type: TypeExpr
    site: str | None
    program: MonadicProgram
    scanned: bool
    position: int


@dataclass(frozen=True, slots=True)
class ProgramKernel:
    """A program's computation with the record the runtime needs to serve it.

    Parameters
    ----------
    module
        The kernel module: the interfaces, the instances, the entry
        computation first, and the helper computation of each scan,
        without handlers; a run adds its own.
    program
        The program encoded.
    steps
        The host steps, in body order.
    sites
        The site label of every draw, in order; a site reached once per
        scan position is listed once, under its label.
    site_types
        The value type of each site, by label.
    data
        The names of the host data inputs after the program's input, in
        parameter order.
    random
        The canonical ``Random`` instance.
    score
        The canonical ``Score`` instance.
    param
        The canonical ``Param`` instance.
    result_type
        The entry computation's result type.
    locals
        The entry computation's local for each name the program binds.
    scanned
        The labels of the sites reached once per scan position, whose
        occurrences are addressed ``"<label>@<n>"``.
    site_morphisms
        The morphism each site draws from, by label.
    """

    module: QiecModule
    program: MonadicProgram
    steps: tuple[HostStep, ...]
    sites: tuple[str, ...]
    site_types: dict[str, TypeExpr]
    data: tuple[str, ...]
    random: NamedEffectInstance
    score: NamedEffectInstance
    param: NamedEffectInstance
    result_type: TypeExpr
    locals: dict[str, Local] = field(default_factory=dict)
    scanned: frozenset[str] = frozenset()
    site_morphisms: dict[str, ContinuousMorphism] = field(default_factory=dict)


def space_type(space: SetObject | ContinuousSpace) -> TypeExpr:
    """The kernel type of a value of a space.

    Parameters
    ----------
    space : SetObject | ContinuousSpace
        A program's domain, codomain, or a morphism's codomain.

    Returns
    -------
    TypeExpr
        ``Int`` for a finite set, ``Tensor[Real]([dim])`` for a
        continuous space of that dimension, and the product of the
        component types for a product space.
    """
    if isinstance(space, ProductSpace):
        return product_type(*(space_type(component) for component in space.components))
    if isinstance(space, ContinuousSpace):
        return tensor_type(REAL, (IndexLiteral(int(space.dim), NAT),))
    return INT


def _origin(path: tuple[str | int, ...], role: str, module: str) -> SourceOrigin:
    """A source origin inside the encoding.

    Parameters
    ----------
    path : tuple[str | int, ...]
        The structural path.
    role : str
        The role of the node.
    module : str
        The module name.

    Returns
    -------
    SourceOrigin
        The origin, carrying no file position.
    """
    return SourceOrigin(module, path, role, SOURCE_PROTOCOL)


def _arg_reads(args: tuple[StepArgument, ...] | None) -> tuple[str, ...]:
    """The environment names a draw's arguments read.

    Parameters
    ----------
    args : tuple[StepArgument, ...] | None
        The draw's arguments: bound names, indexed references, or
        ``None`` for the program input.

    Returns
    -------
    tuple[str, ...]
        The names read, the program input's included when the draw
        takes it.
    """
    if args is None:
        return (INPUT_NAME,)
    names: list[str] = []
    for arg in args:
        if isinstance(arg, Indexed):
            names.append(arg.name)
            names.extend(arg.indices)
        else:
            names.append(arg)
    return tuple(names)


def _let_reads(spec: Let) -> tuple[str, ...] | None:
    """The environment names a let step reads.

    Parameters
    ----------
    spec : Let
        The step.

    Returns
    -------
    tuple[str, ...] | None
        The aliased name, nothing for a constant, a closure's declared
        names, or ``None`` when the closure declares none.
    """
    if isinstance(spec.value, str):
        return (spec.value,)
    if callable(spec.value):
        names = reads_of(spec.value)
        return None if names is None else tuple(sorted(names))
    return ()


def _step_sites(spec: Draw | Observe) -> str:
    """The site label of a draw step.

    Parameters
    ----------
    spec : Draw | Observe
        The step.

    Returns
    -------
    str
        The bound name, or the bound names joined by commas for a
        destructuring draw, which is one draw of the joint.
    """
    return ",".join(spec.names)


def _module_name(program: MonadicProgram) -> str:
    """The module name the encoding of a program takes.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    str
        A name derived from the program's identity, so two programs
        encode to distinct modules.
    """
    return f"program_{id(program):x}"


class _Body:
    """The bind chain of one kernel computation under construction.

    A computation's locals are named apart: a step program run in place
    may bind a name its caller binds too, and the caller's later steps
    must still read the caller's binding.
    """

    def __init__(self) -> None:
        self.binds: list[tuple[Local, Computation]] = []
        self.used: set[str] = set()

    def local(self, name: str, type_: TypeExpr) -> Local:
        """A local of this computation no other local shares a name with.

        Parameters
        ----------
        name : str
            The name the local stands for.
        type_ : TypeExpr
            Its type.

        Returns
        -------
        Local
            ``name`` itself when it is unused, else ``name#k`` for the
            first unused ``k``.
        """
        candidate = name
        serial = 0
        while candidate in self.used:
            candidate = f"{name}#{serial}"
            serial += 1
        self.used.add(candidate)
        return Local(candidate, type_)


class _Frame:
    """The names visible at one point of the encoding, and where they bind.

    A frame binds into one computation's chain. A call's frame sits
    inside its caller's, in the same computation, so a name the callee
    does not bind reads the caller's binding; a scan's step runs in a
    helper computation, whose frame starts from the bindings the scan
    passes it.

    Parameters
    ----------
    body : _Body
        The computation the frame binds into.
    parent : _Frame | None
        The enclosing frame of the same computation, or ``None``.
    scanned : bool
        Whether the frame's sites are reached once per position of a
        scan.
    """

    def __init__(self, body: _Body, parent: _Frame | None, scanned: bool) -> None:
        self.body = body
        self.parent = parent
        self.scanned = scanned
        self.names: dict[str, Local] = {}
        self.order: list[str] = []

    def lookup(self, name: str) -> Local | None:
        """The local a name reads here.

        Parameters
        ----------
        name : str
            The name.

        Returns
        -------
        Local | None
            The innermost binding, or ``None`` when nothing binds it.
        """
        local = self.names.get(name)
        if local is not None:
            return local
        return self.parent.lookup(name) if self.parent is not None else None

    def visible(self) -> list[str]:
        """Every name bound here or in an enclosing frame, in binding order.

        Returns
        -------
        list[str]
            The names, each once, in the order they were first bound.
        """
        names = self.parent.visible() if self.parent is not None else []
        seen = set(names)
        names.extend(name for name in self.order if name not in seen)
        return names

    def bind(self, name: str, local: Local) -> None:
        """Make a name read a local from here on.

        Parameters
        ----------
        name : str
            The name.
        local : Local
            The local it reads.
        """
        if name not in self.names:
            self.order.append(name)
        self.names[name] = local


#: The reserved names a scan's helper reads its position and sequence under.
SCAN_POSITION = "_scan_position"
SCAN_SEQUENCE = "_scan_sequence"


def _row(binds: Sequence[tuple[Local, Computation]]) -> list[RowEntry]:
    """The effect row entries a bind chain performs.

    Parameters
    ----------
    binds : Sequence[tuple[Local, Computation]]
        The chain.

    Returns
    -------
    list[RowEntry]
        The entry of every request performed and of every computation
        called, each once, in order.
    """
    performed: list[RowEntry] = []
    for _local, computation in binds:
        if isinstance(computation, Perform):
            entries: tuple[RowEntry, ...] = (
                RowEntry(computation.request.instance, computation.request.effect),
            )
        elif isinstance(computation, Call):
            entries = computation.effects.entries
        else:
            entries = ()
        for entry in entries:
            if entry not in performed:
                performed.append(entry)
    return performed


def _chain(
    binds: Sequence[tuple[Local, Computation]], tail: Computation
) -> Computation:
    """Sequence a bind chain before a computation.

    Parameters
    ----------
    binds : Sequence[tuple[Local, Computation]]
        The chain, first bind first.
    tail : Computation
        What runs after it.

    Returns
    -------
    Computation
        The nested binds.
    """
    body = tail
    for local, computation in reversed(binds):
        body = Bind(local, computation, body)
    return body


def _reads_input(program: MonadicProgram) -> bool:
    """Whether some draw of a program reads the program's input whole.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    bool
        ``True`` when a draw or observe step takes no arguments.
    """
    return any(
        isinstance(spec, Draw | Observe) and spec.args is None for spec in program.steps
    )


def scanned_sites(program: MonadicProgram) -> frozenset[str]:
    """The labels of the sites a program reaches once per scan position.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    frozenset[str]
        The labels of the sites its scans' step programs reach, through
        the programs those call, at any depth.
    """
    found: set[str] = set()
    for spec in program.steps:
        if isinstance(spec, Scan):
            assert isinstance(spec.step, MonadicProgram)
            found |= _reached_sites(spec.step)
        elif isinstance(spec, CallStep):
            assert isinstance(spec.program, MonadicProgram)
            found |= scanned_sites(spec.program)
    return frozenset(found)


def _reached_sites(program: MonadicProgram) -> frozenset[str]:
    """The labels of every site a program reaches.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    frozenset[str]
        Its own draw and observe labels and those of the programs its
        calls and scans run.
    """
    found: set[str] = set()
    for spec in program.steps:
        if isinstance(spec, Draw | Observe):
            found.add(_step_sites(spec))
        elif isinstance(spec, CallStep):
            assert isinstance(spec.program, MonadicProgram)
            found |= _reached_sites(spec.program)
        elif isinstance(spec, Scan):
            assert isinstance(spec.step, MonadicProgram)
            found |= _reached_sites(spec.step)
    return frozenset(found)


def once_sites(program: MonadicProgram) -> frozenset[str]:
    """The labels of the sites a program reaches at most once per run.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    frozenset[str]
        Its own draw and observe labels and those of the programs its
        calls run, but none a scan reaches.
    """
    found: set[str] = set()
    for spec in program.steps:
        if isinstance(spec, Draw | Observe):
            found.add(_step_sites(spec))
        elif isinstance(spec, CallStep):
            assert isinstance(spec.program, MonadicProgram)
            found |= once_sites(spec.program)
    return frozenset(found)


def _bindings(program: MonadicProgram) -> frozenset[str]:
    """Every name a program and the programs it runs in place bind.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    frozenset[str]
        The names its let, score, call, and scan steps bind, with those
        of its callees and step programs.
    """
    found: set[str] = set()
    for spec in program.steps:
        if isinstance(spec, Let | Score | CallStep | Scan):
            found.add(spec.name)
        if isinstance(spec, CallStep):
            assert isinstance(spec.program, MonadicProgram)
            found |= _bindings(spec.program)
        elif isinstance(spec, Scan):
            assert isinstance(spec.step, MonadicProgram)
            found |= _bindings(spec.step)
    return frozenset(found)


def derived_bindings(program: MonadicProgram) -> tuple[frozenset[str], frozenset[str]]:
    """The names a run computes and a caller may not supply a value for.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    tuple[frozenset[str], frozenset[str]]
        The names the program's own call and scan steps bind, which the
        run computes from the sites those reach; and the names bound
        inside its scans' step programs, which the run computes once
        per position.
    """
    once: set[str] = set()
    per_position: set[str] = set()
    for spec in program.steps:
        if isinstance(spec, CallStep):
            once.add(spec.name)
            assert isinstance(spec.program, MonadicProgram)
            _, inner_scanned = derived_bindings(spec.program)
            per_position |= inner_scanned
        elif isinstance(spec, Scan):
            once.add(spec.name)
            assert isinstance(spec.step, MonadicProgram)
            per_position |= _bindings(spec.step)
    return frozenset(once), frozenset(per_position)


def site_label(label: str, occurrence: int) -> str:
    """The address of one occurrence of a site reached once per position.

    Parameters
    ----------
    label : str
        The site's label.
    occurrence : int
        How many times the run reached the site before, counted from zero.

    Returns
    -------
    str
        ``"<label>@<occurrence>"``.
    """
    return f"{label}@{occurrence}"


class _Builder:
    """Accumulates the encoding of one program.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    data : Sequence[str]
        The names of the host data inputs a run supplies beside the
        program's input.
    """

    def __init__(self, program: MonadicProgram, data: Sequence[str]) -> None:
        self.program = program
        self.module = _module_name(program)
        self.data = tuple(data)
        self.instances: list[NamedEffectInstance] = []
        self.steps: list[HostStep] = []
        self.sites: list[str] = []
        self.site_types: dict[str, TypeExpr] = {}
        self.site_morphisms: dict[str, ContinuousMorphism] = {}
        self.scanned: set[str] = set()
        self.unscanned: set[str] = set()
        self.helpers: list[NamedComputation] = []
        self.random = self._instance(RANDOM_INSTANCE, RANDOM)
        self.score = self._instance(SCORE_INSTANCE, SCORE)
        self.param = self._instance(PARAM_INSTANCE, PARAM)

    def _instance(self, name: str, effect: EffectRef) -> NamedEffectInstance:
        """Allocate a named lexical instance.

        Parameters
        ----------
        name : str
            The instance's name.
        effect : EffectRef
            The applied interface.

        Returns
        -------
        NamedEffectInstance
            The instance, added to the module.
        """
        entry = instantiate_effect(
            effect, module=self.module, lexical_path=("instances", name)
        )
        instance = NamedEffectInstance(
            name, entry, _origin(("instances", name), "effect-instance", self.module)
        )
        self.instances.append(instance)
        return instance

    @staticmethod
    def _bind(
        frame: _Frame, name: str, type_: TypeExpr, computation: Computation
    ) -> Local:
        """Bind a name to a computation's result.

        Parameters
        ----------
        frame : _Frame
            The frame the name binds in.
        name : str
            The name.
        type_ : TypeExpr
            Its type.
        computation : Computation
            The computation whose result it takes.

        Returns
        -------
        Local
            The local, fresh in its computation; a step that rebinds a
            name, as a nested marginalize block does for its placeholder
            lets, gets a new local, and later steps read the new one.
        """
        local = frame.body.local(name, type_)
        frame.bind(name, local)
        frame.body.binds.append((local, computation))
        return local

    @staticmethod
    def _environment(
        frame: _Frame, reads: Sequence[str] | None
    ) -> tuple[tuple[str, ...], TupleValue, TypeExpr]:
        """The tuple of the locals a step reads.

        Parameters
        ----------
        frame : _Frame
            The frame the step runs in.
        reads : Sequence[str] | None
            The names the step reads, of which those visible are passed
            in binding order; ``None`` passes every visible name.

        Returns
        -------
        tuple[tuple[str, ...], TupleValue, TypeExpr]
            The names in order, the tuple value, and its type.
        """
        visible = frame.visible()
        if reads is None:
            names = tuple(visible)
        else:
            wanted = set(reads)
            names = tuple(name for name in visible if name in wanted)
        locals_: list[Local] = []
        for name in names:
            local = frame.lookup(name)
            assert local is not None
            locals_.append(local)
        type_ = product_type(*(local.type for local in locals_))
        value = TupleValue(tuple(Var(local) for local in locals_), type_)
        return names, value, type_

    def _host(
        self,
        frame: _Frame,
        name: str,
        kind: str,
        spec: Step | None,
        program: MonadicProgram,
        answer_type: TypeExpr,
        site: str | None = None,
        reads: Sequence[str] | None = None,
    ) -> Local:
        """Add a host step and bind its result.

        Parameters
        ----------
        frame : _Frame
            The frame the step runs in.
        name : str
            The name the result binds.
        kind : str
            The step's kind.
        spec : Step | None
            The step record the host part serves.
        program : MonadicProgram
            The program the record belongs to.
        answer_type : TypeExpr
            The type the host part returns.
        site : str | None
            The site label of a draw step.
        reads : Sequence[str] | None
            The names the step reads, or ``None`` for every visible name.

        Returns
        -------
        Local
            The bound result.
        """
        environment_names, environment, environment_type = self._environment(
            frame, reads
        )
        effect = EffectRef(COMPUTE.id, COMPUTE.name, (environment_type, answer_type))
        position = len(self.steps)
        instance = self._instance(f"step_{position}", effect)
        self.steps.append(
            HostStep(
                instance.entry.instance,
                effect,
                kind,
                spec,
                environment_names,
                answer_type,
                site,
                program,
                frame.scanned,
                position,
            )
        )
        request = EffectRequest(
            instance.entry.instance,
            effect,
            COMPUTE_APPLY,
            (),
            (environment,),
            answer_type,
            SiteProvenance(
                _origin(("steps", position, "host"), "effect-request", self.module)
            ),
        )
        return self._bind(frame, name, answer_type, Perform(request))

    def _sample(
        self,
        frame: _Frame,
        label: str,
        sampleable: Local,
        type_: TypeExpr,
        program: MonadicProgram,
        spec: Draw | Observe,
    ) -> Local:
        """Add a ``Random.sample`` request at a site.

        Parameters
        ----------
        frame : _Frame
            The frame the draw runs in.
        label : str
            The site label.
        sampleable : Local
            The local holding the sampleable.
        type_ : TypeExpr
            The sampled value's type.
        program : MonadicProgram
            The program the draw belongs to.
        spec : Draw | Observe
            The draw.

        Returns
        -------
        Local
            The bound draw.

        Raises
        ------
        ValueError
            If the label is reached both once and once per scan
            position, or by two draws from different morphisms.
        """
        morphism = program.step_module(spec)
        assert isinstance(morphism, ContinuousMorphism)
        known = self.site_morphisms.get(label)
        if known is not None and known is not morphism:
            raise ValueError(
                f"site {label!r} is drawn by two steps from different "
                f"morphisms; give one of them another name"
            )
        self.site_morphisms[label] = morphism
        (self.scanned if frame.scanned else self.unscanned).add(label)
        if label in self.scanned and label in self.unscanned:
            raise ValueError(
                f"site {label!r} is reached once per scan position and also "
                f"outside the scan, so its occurrences have no stable "
                f"addresses; rename one of them"
            )
        site: Value
        if frame.scanned:
            # The site is reached once per position: its address is the
            # label with the occurrence the run counts, which a host step
            # supplies.
            site = Var(
                self._host(
                    frame,
                    f"__site_{label}",
                    "site",
                    spec,
                    program,
                    site_type(type_),
                    site=label,
                    reads=(),
                )
            )
        else:
            site = SiteValue(label, site_type(type_))
        request = EffectRequest(
            self.random.entry.instance,
            self.random.entry.effect,
            RANDOM_SAMPLE,
            (type_,),
            (site, Var(sampleable)),
            type_,
            SiteProvenance(_origin(("sites", label), "effect-request", self.module)),
        )
        if label not in self.site_types:
            self.sites.append(label)
        self.site_types[label] = type_
        return self._bind(frame, label, type_, Perform(request))

    def _add_score(self, frame: _Frame, weight: Local, position: int) -> None:
        """Add a ``Score.add`` of a bound weight.

        Parameters
        ----------
        frame : _Frame
            The frame the score runs in.
        weight : Local
            The local holding the weight.
        position : int
            The score's host step's position, for the request's origin.
        """
        request = EffectRequest(
            self.score.entry.instance,
            self.score.entry.effect,
            SCORE_ADD,
            (),
            (Var(weight),),
            UNIT,
            SiteProvenance(
                _origin(("steps", position, "score"), "effect-request", self.module)
            ),
        )
        unit = frame.body.local(f"__score_{position}", UNIT)
        frame.body.binds.append((unit, Perform(request)))

    def build(self) -> ProgramKernel:
        """Encode the program.

        Returns
        -------
        ProgramKernel
            The kernel module and the record the runtime serves it by.
        """
        program = self.program
        body = _Body()
        frame = _Frame(body, None, scanned=False)
        parameters: list[Local] = []
        x = body.local(INPUT_NAME, space_type(program.domain))
        frame.bind(INPUT_NAME, x)
        parameters.append(x)
        for name in self.data:
            local = body.local(name, REAL)
            frame.bind(name, local)
            parameters.append(local)
        if program._params is not None and program._param_dims is not None:
            components = tuple(
                _component_type(program, index) for index in range(len(program._params))
            )
            split = self._host(
                frame,
                "__params",
                "input",
                None,
                program,
                product_type(*components),
                reads=(),
            )
            for index, name in enumerate(program._params):
                self._bind(
                    frame,
                    name,
                    components[index],
                    Return(Projection(Var(split), index, components[index])),
                )
        self._steps(frame, program)
        result, result_type = _result(frame, program)
        computation = NamedComputation(
            ComputationId.derive(self.module, "computation", ENTRY),
            ENTRY,
            (),
            tuple(parameters),
            _chain(body.binds, Return(result)),
            ComputationType(EffectRow(tuple(_row(body.binds))), result_type),
            _origin(("computations", ENTRY), "computation", self.module),
        )
        module = QiecModule(
            self.module,
            SOURCE_PROTOCOL,
            effects=(RANDOM_EFFECT, SCORE_EFFECT, PARAM_EFFECT, COMPUTE_EFFECT),
            instances=tuple(self.instances),
            computations=(computation, *self.helpers),
        )
        return ProgramKernel(
            module,
            program,
            tuple(self.steps),
            tuple(self.sites),
            dict(self.site_types),
            self.data,
            self.random,
            self.score,
            self.param,
            result_type,
            {name: local for name in frame.order if (local := frame.lookup(name))},
            frozenset(self.scanned),
            dict(self.site_morphisms),
        )

    def _steps(self, frame: _Frame, program: MonadicProgram) -> None:
        """Encode a program's steps in a frame.

        Parameters
        ----------
        frame : _Frame
            The frame.
        program : MonadicProgram
            The program the steps belong to.
        """
        for spec in program.steps:
            match spec:
                case Let():
                    self._host(
                        frame,
                        spec.name,
                        "let",
                        spec,
                        program,
                        REAL,
                        reads=_let_reads(spec),
                    )
                case Score():
                    position = len(self.steps)
                    weight = self._host(
                        frame,
                        spec.name,
                        "score",
                        spec,
                        program,
                        LOG_WEIGHT,
                        reads=reads_of(spec.score),
                    )
                    self._add_score(frame, weight, position)
                case Draw() | Observe():
                    self._draw(frame, program, spec)
                case CallStep():
                    self._call(frame, spec)
                case Scan():
                    self._scan(frame, program, spec)

    def _draw(
        self, frame: _Frame, program: MonadicProgram, spec: Draw | Observe
    ) -> None:
        """Encode a draw step: its host sampleable and its sample request.

        Parameters
        ----------
        frame : _Frame
            The frame the draw runs in.
        program : MonadicProgram
            The program the draw belongs to.
        spec : Draw | Observe
            The step.
        """
        morphism = program.step_module(spec)
        assert morphism is not None
        label = _step_sites(spec)
        reads = _arg_reads(spec.args)
        categorical = extract_morphism(morphism)
        if categorical is not None and not isinstance(morphism, ContinuousMorphism):
            # A categorical morphism is deterministic: its tensor is
            # contracted against the input, and the result is bound
            # like a let.
            shape = tuple(
                IndexLiteral(int(extent), NAT) for extent in categorical.codomain.shape
            )
            self._host(
                frame,
                label,
                "apply",
                spec,
                program,
                tensor_type(REAL, shape),
                reads=reads,
            )
            return
        codomain = getattr(morphism, "codomain")
        type_ = space_type(codomain)
        sampleable = self._host(
            frame,
            f"__sampleable_{label}",
            "draw",
            spec,
            program,
            sampleable_type(type_),
            site=label,
            reads=reads,
        )
        self._sample(frame, label, sampleable, type_, program, spec)
        if len(spec.names) > 1:
            parts = self._host(
                frame,
                f"__parts_{label}",
                "bind",
                spec,
                program,
                product_type(*(REAL for _ in spec.names)),
                reads=(label,),
            )
            for index, name in enumerate(spec.names):
                self._bind(
                    frame, name, REAL, Return(Projection(Var(parts), index, REAL))
                )

    def _enter(
        self,
        frame: _Frame,
        callee: MonadicProgram,
        arguments: Sequence[Local],
    ) -> _Frame:
        """Open a frame running a program on given arguments.

        Parameters
        ----------
        frame : _Frame
            The frame the program is run from.
        callee : MonadicProgram
            The program.
        arguments : Sequence[Local]
            One local per parameter, or one for a program without named
            parameters, which reads it as its input.

        Returns
        -------
        _Frame
            The program's frame, inside ``frame``, with its parameters
            bound, and its input too when one of its draws reads it.
        """
        inner = _Frame(frame.body, frame, frame.scanned)
        if callee._params is None:
            (argument,) = arguments
            self._bind(inner, INPUT_NAME, argument.type, Return(Var(argument)))
            return inner
        for name, argument in zip(callee._params, arguments, strict=True):
            self._bind(inner, name, argument.type, Return(Var(argument)))
        if _reads_input(callee):
            self._host(
                inner,
                INPUT_NAME,
                "pack",
                None,
                callee,
                space_type(callee.domain),
                reads=callee._params,
            )
        return inner

    def _call(self, frame: _Frame, spec: CallStep) -> None:
        """Encode a call: the callee's steps in a frame of their own.

        Parameters
        ----------
        frame : _Frame
            The frame the call runs in.
        spec : CallStep
            The step.

        Raises
        ------
        ValueError
            If an argument names nothing bound.
        """
        callee = spec.program
        assert isinstance(callee, MonadicProgram)
        arguments: list[Local] = []
        for name in spec.args:
            local = frame.lookup(name)
            if local is None:
                raise ValueError(
                    f"call {spec.name!r} passes {name!r}, which nothing binds"
                )
            arguments.append(local)
        inner = self._enter(frame, callee, arguments)
        self._steps(inner, callee)
        result, result_type = _result(inner, callee)
        self._bind(frame, spec.name, result_type, Return(result))

    def _scan(self, frame: _Frame, program: MonadicProgram, spec: Scan) -> None:
        """Encode a scan: a recursive helper running the step per position.

        The helper takes the position, the sequence, its length, the
        state, and every binding visible at the scan; it answers the
        state once the position reaches the length, and otherwise runs
        the step program at the position's entry and recurs at the
        next position with the state the step returned.

        Parameters
        ----------
        frame : _Frame
            The frame the scan runs in.
        program : MonadicProgram
            The program the scan belongs to.
        spec : Scan
            The step.

        Raises
        ------
        ValueError
            If the sequence names nothing bound.
        """
        step = spec.step
        assert isinstance(step, MonadicProgram)
        source_name = INPUT_NAME if spec.source is None else spec.source
        source = frame.lookup(source_name)
        if source is None:
            raise ValueError(
                f"scan {spec.name!r} runs over {source_name!r}, which nothing binds"
            )
        state_type = space_type(step.codomain)
        entry_type = _component_type(step, 0)
        length = self._host(
            frame,
            f"__scan_length_{spec.name}",
            "scan_length",
            spec,
            program,
            INT,
            reads=(source_name,),
        )
        initial = self._host(
            frame,
            f"__scan_initial_{spec.name}",
            "scan_initial",
            spec,
            program,
            state_type,
            reads=(source_name,),
        )
        carried_names = [
            name
            for name in frame.visible()
            if name not in (SCAN_POSITION, SCAN_SEQUENCE)
        ]
        carried: list[Local] = []
        for name in carried_names:
            local = frame.lookup(name)
            assert local is not None
            carried.append(local)
        helper_name = f"scan_{len(self.helpers)}_{spec.name}"
        identity = ComputationId.derive(self.module, "computation", helper_name)
        body = _Body()
        inner = _Frame(body, None, scanned=True)
        position = body.local(SCAN_POSITION, INT)
        sequence = body.local(SCAN_SEQUENCE, source.type)
        count = body.local("_scan_length", INT)
        state = body.local("_scan_state", state_type)
        passed = [
            body.local(name, local.type)
            for name, local in zip(carried_names, carried, strict=True)
        ]
        for name, local in zip(carried_names, passed, strict=True):
            inner.bind(name, local)
        inner.bind(SCAN_POSITION, position)
        inner.bind(SCAN_SEQUENCE, sequence)
        entry = self._host(
            inner,
            "__scan_entry",
            "scan_entry",
            spec,
            program,
            entry_type,
            reads=(SCAN_SEQUENCE, SCAN_POSITION),
        )
        step_frame = self._enter(inner, step, (entry, state))
        self._steps(step_frame, step)
        _result(step_frame, step)
        # A let's local is typed nominally; the state the step returns is
        # read back at the state's type, which the recursion carries.
        following = Var(
            self._host(
                step_frame,
                "__scan_next",
                "scan_state",
                spec,
                program,
                state_type,
                reads=step._return_vars,
            )
        )
        # The helper's row is what one position performs, which is also
        # what the recursion performs.
        row = EffectRow(tuple(_row(body.binds)))
        origin = _origin(("scans", helper_name), "call", self.module)
        recur = Call(
            identity,
            helper_name,
            (),
            (
                PrimitiveApplication(
                    primitive("add_int").id,
                    "add_int",
                    (Var(position), LiteralValue(1, INT)),
                    INT,
                    _origin(
                        ("scans", helper_name, "successor"), "primitive", self.module
                    ),
                ),
                Var(sequence),
                Var(count),
                following,
                *(Var(local) for local in passed),
            ),
            state_type,
            row,
            origin,
        )
        finished = PrimitiveApplication(
            primitive("eq_int").id,
            "eq_int",
            (Var(position), Var(count)),
            BOOL,
            _origin(("scans", helper_name, "finished"), "primitive", self.module),
        )
        self.helpers.append(
            NamedComputation(
                identity,
                helper_name,
                (),
                (position, sequence, count, state, *passed),
                If(finished, Return(Var(state)), _chain(body.binds, recur)),
                ComputationType(row, state_type),
                _origin(("computations", helper_name), "computation", self.module),
            )
        )
        self._bind(
            frame,
            spec.name,
            state_type,
            Call(
                identity,
                helper_name,
                (),
                (
                    LiteralValue(0, INT),
                    Var(source),
                    Var(length),
                    Var(initial),
                    *(Var(local) for local in carried),
                ),
                state_type,
                row,
                origin,
            ),
        )


def _component_type(program: MonadicProgram, index: int) -> TypeExpr:
    """The type of one named parameter of a product-domain program.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    index : int
        The parameter's position.

    Returns
    -------
    TypeExpr
        ``Int`` for a finite factor, else the real vector of the
        factor's width.
    """
    assert program._param_dims is not None
    assert program._param_is_continuous is not None
    if program._param_is_continuous[index]:
        return tensor_type(REAL, (IndexLiteral(program._param_dims[index], NAT),))
    return INT


def _result(frame: _Frame, program: MonadicProgram) -> tuple[Value, TypeExpr]:
    """The value a program's frame returns.

    Parameters
    ----------
    frame : _Frame
        The frame the program's steps ran in.
    program : MonadicProgram
        The program.

    Returns
    -------
    tuple[Value, TypeExpr]
        The returned value and its type: the single return variable,
        or the tuple of the return variables.

    Raises
    ------
    ValueError
        If a return variable is not bound.
    """
    locals_: list[Local] = []
    for name in program._return_vars:
        local = frame.lookup(name)
        if local is None:
            raise ValueError(f"return variable {name!r} is not bound in the program")
        locals_.append(local)
    if program._return_is_single:
        return Var(locals_[0]), locals_[0].type
    type_ = product_type(*(local.type for local in locals_))
    return TupleValue(tuple(Var(local) for local in locals_), type_), type_


def program_kernel(program: MonadicProgram, data: Sequence[str] = ()) -> ProgramKernel:
    """Encode a program as a kernel module.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    data : Sequence[str]
        The names of the host data inputs a run supplies beside the
        program's input, in parameter order.

    Returns
    -------
    ProgramKernel
        The module and the record the runtime serves it by.
    """
    return _Builder(program, data).build()


__all__ = [
    "HostStep",
    "ProgramKernel",
    "program_kernel",
]
