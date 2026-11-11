"""Prepared log joints: a program's scoring run compiled once per layout.

`MonadicProgram.log_joint` runs the program on the reference machine:
every call encodes nothing new (the kernel encoding is cached per
program), but it builds the run's handlers, validates their module, and
drives the machine through every request, deriving each site's dynamic
address and instantiating each handler clause as it goes. None of that
depends on the values the sites take. It depends on the program, on the
names and layouts of the values given, and on the handlers installed.

A `JointPlan` is that part done once. `prepare_joint` walks the entry
computation of the program's kernel, a chain of binds, and compiles it
to a straight-line list of instructions over numbered slots: each host
step is called through the same `_HostSteps` function the machine's
``Compute`` handler calls, each sample site is answered with its given
value and scored by its sampleable's ``log_prob`` as the machine's
observation handler does, and each ``Score.add`` is collected as the
run's accumulator collects it. The per-site densities are then summed
in the order and with the operations `TraceHandler` uses, so the joint
and its gradient are the machine's, bit for bit.

The plan is exact for the runs it describes and refuses the others. It
is prepared for one layout, the shape, dtype, and device of the input
and of every value given (`joint_signature`), and evaluating it at any
other layout raises; within a layout, the bind chain fixes the sites
reached and the order they are reached in, since the chain has no
branch. `prepare_joint` declines, naming the reason, a program whose
entry computation is not such a chain (one with a scan, whose
recursion depends on the sequence it runs over), a site that is given
no value, an input of more than one run, host data that is not a
tensor, and a call made with handlers on the stack; those are evaluated
on the reference machine.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.program_steps import Draw, Let, Observe, Score
from quivers.continuous.programs import MonadicProgram
from quivers.effects.base import RunContext, _handler_stack
from quivers.effects.interpreter import (
    _canonical,
    _cached_kernel,
    _HostSteps,
    _run_slice,
    _RunAxis,
    _site_carries,
    _site_morphism,
    _split_observations,
    check_host_data,
    host_value,
)
from quivers.effects.program_module import HostStep, ProgramKernel
from quivers.qiec.builtins import COMPUTE_APPLY, RANDOM_SAMPLE, SCORE_ADD
from quivers.qiec.evaluator import EvaluationError, RuntimeTypeMismatch
from quivers.qiec.terms import (
    Bind,
    Computation,
    Local,
    Perform,
    Projection,
    Return,
    SiteValue,
    TupleValue,
    Value,
    Var,
)

type TensorLayout = tuple[tuple[int, ...], torch.dtype, torch.device]
"""The shape, dtype, and device of one tensor."""

type JointSignature = tuple[TensorLayout, tuple[tuple[str, TensorLayout], ...]]
"""The layout of a scoring call: the input's, and each given value's by
name, in the order given."""


def _layout(value: torch.Tensor) -> TensorLayout:
    """The layout of a tensor.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    TensorLayout
        Its shape, dtype, and device.
    """
    return (tuple(value.shape), value.dtype, value.device)


def joint_signature(
    x: torch.Tensor, values: Mapping[str, torch.Tensor]
) -> JointSignature:
    """The layout a prepared joint is specific to.

    Parameters
    ----------
    x : torch.Tensor
        The program input.
    values : Mapping[str, torch.Tensor]
        The site values and host data, by name.

    Returns
    -------
    JointSignature
        The input's layout and every value's, by name, in order.

    Raises
    ------
    TypeError
        If a value is not a tensor.
    """
    entries: list[tuple[str, TensorLayout]] = []
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"joint_signature: {name!r} is a {type(value).__name__}, not a tensor"
            )
        entries.append((name, _layout(value)))
    return (_layout(x), tuple(entries))


def _checked(value: object, subject: str) -> object:
    """Refuse a value no host step may produce, as the machine's validator does.

    Parameters
    ----------
    value : object
        The value a host step returned or a site was answered with.
    subject : str
        What the value is, for the diagnostic.

    Returns
    -------
    object
        The value.

    Raises
    ------
    RuntimeTypeMismatch
        If the value is not a host value.
    """
    if not host_value(value):
        raise RuntimeTypeMismatch(f"{subject} is not a host value: {value!r}")
    return value


class _Evaluation:
    """The state of one evaluation of a plan.

    Parameters
    ----------
    slots : list[object]
        The value of each local, by slot; the input and the host data
        fill the first slots.
    steps : _HostSteps
        The host functions of the run.
    parameters : Callable[[str, ContinuousMorphism], dict[str, torch.Tensor]]
        Reads a draw's parameters from the run's parameter store.
    clamped : dict[str, torch.Tensor]
        Each site's given value, in the shape its morphism scores.
    """

    __slots__ = ("slots", "steps", "parameters", "clamped", "events", "scores", "total")

    def __init__(
        self,
        slots: list[object],
        steps: _HostSteps,
        parameters: Callable[[str, ContinuousMorphism], dict[str, torch.Tensor]],
        clamped: dict[str, torch.Tensor],
    ) -> None:
        self.slots = slots
        self.steps = steps
        self.parameters = parameters
        self.clamped = clamped
        # The trace's events in order: a site's label and its density, or
        # a let or score step and its value.
        self.events: list[tuple[str, HostStep | None, torch.Tensor | object]] = []
        # The contributions of each score step, by the step's position.
        self.scores: dict[int, list[torch.Tensor]] = {}
        # The accumulator's running total, folded as the machine folds it.
        self.total: object = 0.0


class _Instruction(ABC):
    """One bind of the compiled chain, writing its result to a slot.

    Parameters
    ----------
    target : int
        The slot the bind's local takes.
    """

    __slots__ = ("target",)

    def __init__(self, target: int) -> None:
        self.target = target

    @abstractmethod
    def execute(self, state: _Evaluation) -> None:
        """Run the bind.

        Parameters
        ----------
        state : _Evaluation
            The evaluation.
        """


class _Compute(_Instruction):
    """A host step applied to the locals it reads.

    Parameters
    ----------
    target : int
        The slot of the step's result.
    step : HostStep
        The step.
    reads : tuple[int, ...]
        The slots of its environment tuple, in order.
    """

    __slots__ = ("step", "reads", "traced")

    def __init__(self, target: int, step: HostStep, reads: tuple[int, ...]) -> None:
        super().__init__(target)
        self.step = step
        self.reads = reads
        self.traced = step.kind in ("let", "score")

    def execute(self, state: _Evaluation) -> None:
        """Call the step's host function and bind its result.

        Parameters
        ----------
        state : _Evaluation
            The evaluation.
        """
        slots = state.slots
        argument = tuple(slots[read] for read in self.reads)
        step = self.step
        if step.kind == "draw":
            result = state.steps.draw(step, argument, state.parameters)
        else:
            result = state.steps.pure(step)(argument)
        slots[self.target] = _checked(result, f"host step {step.position}")
        if self.traced:
            state.events.append(("step", step, result))


class _Sample(_Instruction):
    """A site answered with its given value and scored.

    Parameters
    ----------
    target : int
        The slot of the site's value.
    label : str
        The site's label.
    sampleable : int
        The slot of the sampleable the site draws from.
    """

    __slots__ = ("label", "sampleable")

    def __init__(self, target: int, label: str, sampleable: int) -> None:
        super().__init__(target)
        self.label = label
        self.sampleable = sampleable

    def execute(self, state: _Evaluation) -> None:
        """Score the given value under the sampleable and bind it.

        Parameters
        ----------
        state : _Evaluation
            The evaluation.

        Raises
        ------
        EvaluationError
            If the site's sampleable has no ``log_prob``.
        """
        value = _checked(state.clamped[self.label], f"observation {self.label!r}")
        sampleable = state.slots[self.sampleable]
        log_prob = getattr(sampleable, "log_prob", None)
        if not callable(log_prob):
            raise EvaluationError(
                "a scored observation/replay needs a runtime sampleable with log_prob"
            )
        weight = log_prob(value)
        state.total = operator.add(state.total, weight)
        state.slots[self.target] = value
        state.events.append(("site", None, (self.label, torch.as_tensor(weight))))


class _ScoreAdd(_Instruction):
    """A score step's weight added to the run's accumulator.

    Parameters
    ----------
    target : int
        The slot of the bind's unit result.
    weight : int
        The slot of the weight.
    position : int
        The position of the score's host step.
    """

    __slots__ = ("weight", "position")

    def __init__(self, target: int, weight: int, position: int) -> None:
        super().__init__(target)
        self.weight = weight
        self.position = position

    def execute(self, state: _Evaluation) -> None:
        """Accumulate the weight.

        Parameters
        ----------
        state : _Evaluation
            The evaluation.
        """
        weight = state.slots[self.weight]
        state.total = operator.add(state.total, weight)
        state.scores.setdefault(self.position, []).append(torch.as_tensor(weight))
        state.slots[self.target] = None


class _Copy(_Instruction):
    """A local bound to another's value.

    Parameters
    ----------
    target : int
        The slot bound.
    source : int
        The slot read.
    """

    __slots__ = ("source",)

    def __init__(self, target: int, source: int) -> None:
        super().__init__(target)
        self.source = source

    def execute(self, state: _Evaluation) -> None:
        """Bind the value.

        Parameters
        ----------
        state : _Evaluation
            The evaluation.
        """
        state.slots[self.target] = state.slots[self.source]


class _Project(_Instruction):
    """A local bound to one component of a product.

    Parameters
    ----------
    target : int
        The slot bound.
    source : int
        The slot of the product.
    position : int
        The component.
    """

    __slots__ = ("source", "position")

    def __init__(self, target: int, source: int, position: int) -> None:
        super().__init__(target)
        self.source = source
        self.position = position

    def execute(self, state: _Evaluation) -> None:
        """Bind the component.

        Parameters
        ----------
        state : _Evaluation
            The evaluation.

        Raises
        ------
        EvaluationError
            If the value is not a product with that component.
        """
        source = state.slots[self.source]
        if not isinstance(source, tuple) or self.position >= len(source):
            raise EvaluationError(
                f"projection {self.position} from a non-product runtime value"
            )
        state.slots[self.target] = source[self.position]


class _Tuple(_Instruction):
    """A local bound to a tuple of locals.

    Parameters
    ----------
    target : int
        The slot bound.
    sources : tuple[int, ...]
        The slots of the components.
    """

    __slots__ = ("sources",)

    def __init__(self, target: int, sources: tuple[int, ...]) -> None:
        super().__init__(target)
        self.sources = sources

    def execute(self, state: _Evaluation) -> None:
        """Bind the tuple.

        Parameters
        ----------
        state : _Evaluation
            The evaluation.
        """
        state.slots[self.target] = tuple(state.slots[source] for source in self.sources)


class _Compiler:
    """Compiles a kernel's entry computation to instructions over slots.

    Parameters
    ----------
    kernel : ProgramKernel
        The program's kernel encoding.
    """

    def __init__(self, kernel: ProgramKernel) -> None:
        self.kernel = kernel
        self.slots: dict[Local, int] = {}
        self.instructions: list[_Instruction] = []
        self.labels: list[str] = []
        self.steps = {step.instance: step for step in kernel.steps}

    def slot(self, local: Local) -> int:
        """A fresh slot for a local.

        Parameters
        ----------
        local : Local
            The local.

        Returns
        -------
        int
            Its slot.
        """
        index = len(self.slots)
        self.slots[local] = index
        return index

    def read(self, value: Value) -> int | None:
        """The slot a value reads, when it is a local.

        Parameters
        ----------
        value : Value
            The value.

        Returns
        -------
        int | None
            The slot of the local, or ``None`` when the value is not one
            bound earlier in the chain.
        """
        if isinstance(value, Var) and value.local in self.slots:
            return self.slots[value.local]
        return None

    def compile(self) -> str | None:
        """Compile the entry computation.

        Returns
        -------
        str | None
            ``None`` when the whole computation compiled, else why it is
            not a straight bind chain the plan can run.
        """
        entry = self.kernel.module.computations[0]
        for parameter in entry.parameters:
            self.slot(parameter)
        term: Computation = entry.body
        while isinstance(term, Bind):
            reason = self.bind(term.binder, term.first)
            if reason is not None:
                return reason
            term = term.then
        if not isinstance(term, Return):
            return f"the program's body ends in a {type(term).__name__}, not a return"
        return None

    def bind(self, binder: Local, first: Computation) -> str | None:
        """Compile one bind.

        Parameters
        ----------
        binder : Local
            The local bound.
        first : Computation
            The computation whose result it takes.

        Returns
        -------
        str | None
            ``None`` when compiled, else why it cannot be.
        """
        if isinstance(first, Return):
            return self.bind_value(binder, first.value)
        if not isinstance(first, Perform):
            return f"the program performs a {type(first).__name__} (a scan)"
        request = first.request
        if request.operation == COMPUTE_APPLY and request.instance in self.steps:
            step = self.steps[request.instance]
            (environment,) = request.arguments
            if step.scanned or not isinstance(environment, TupleValue):
                return f"host step {step.position} runs per scan position"
            reads = tuple(self.read(item) for item in environment.items)
            if any(read is None for read in reads):
                return f"host step {step.position} reads a value outside the chain"
            self.instructions.append(
                _Compute(self.slot(binder), step, tuple(r for r in reads if r is not None))
            )
            return None
        if (
            request.operation == RANDOM_SAMPLE
            and request.instance == self.kernel.random.entry.instance
        ):
            site, sampleable = request.arguments
            source = self.read(sampleable)
            if not isinstance(site, SiteValue) or source is None:
                return "a site is reached once per scan position"
            self.labels.append(site.label)
            self.instructions.append(_Sample(self.slot(binder), site.label, source))
            return None
        if (
            request.operation == SCORE_ADD
            and request.instance == self.kernel.score.entry.instance
        ):
            (weight,) = request.arguments
            source = self.read(weight)
            path = request.origin.origin.structural_path
            if source is None or len(path) < 2 or not isinstance(path[1], int):
                return "a score is added from outside the chain"
            self.instructions.append(_ScoreAdd(self.slot(binder), source, path[1]))
            return None
        return f"the program performs {request.operation} outside its host steps"

    def bind_value(self, binder: Local, value: Value) -> str | None:
        """Compile a bind of a value.

        Parameters
        ----------
        binder : Local
            The local bound.
        value : Value
            The value: a local, a projection of one, or a tuple of them.

        Returns
        -------
        str | None
            ``None`` when compiled, else why it cannot be.
        """
        if isinstance(value, Var):
            source = self.read(value)
            if source is None:
                return f"{binder.name!r} binds a value outside the chain"
            self.instructions.append(_Copy(self.slot(binder), source))
            return None
        if isinstance(value, Projection):
            source = self.read(value.value)
            if source is None:
                return f"{binder.name!r} projects a value outside the chain"
            self.instructions.append(_Project(self.slot(binder), source, value.position))
            return None
        if isinstance(value, TupleValue):
            sources = tuple(self.read(item) for item in value.items)
            if any(source is None for source in sources):
                return f"{binder.name!r} pairs a value outside the chain"
            self.instructions.append(
                _Tuple(self.slot(binder), tuple(s for s in sources if s is not None))
            )
            return None
        return f"{binder.name!r} binds a {type(value).__name__}"


class JointPlan:
    """A program's log joint, compiled once for one layout of its values.

    Built by `prepare_joint`; `evaluate` scores the program at values of
    the layout it was prepared for, with the operations
    `MonadicProgram.log_joint` performs, in the same order.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    kernel : ProgramKernel
        Its kernel encoding at the host data given.
    instructions : Sequence[_Instruction]
        The compiled bind chain.
    slots : int
        How many locals the chain binds, its parameters included.
    signature : JointSignature
        The layout prepared for.
    sites : tuple[str, ...]
        The labels of the given site values, in the order given.
    supplied : tuple[str, ...]
        The names of the given values standing in for let bindings.
    axis : _RunAxis
        The classification of the run's bindings before it runs.
    """

    def __init__(
        self,
        program: MonadicProgram,
        kernel: ProgramKernel,
        instructions: Sequence[_Instruction],
        slots: int,
        signature: JointSignature,
        sites: tuple[str, ...],
        supplied: tuple[str, ...],
        axis: _RunAxis,
    ) -> None:
        self._program = program
        self._kernel = kernel
        self._instructions = tuple(instructions)
        self._slots = slots
        self._signature = signature
        self._sites = sites
        self._supplied = supplied
        self._axis = axis
        self._draws = tuple(
            step
            for step in kernel.steps
            if step.kind == "draw" and step.site is not None
        )

    @property
    def signature(self) -> JointSignature:
        """The layout of the input and values the plan scores."""
        return self._signature

    def evaluate(
        self, x: torch.Tensor, values: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Score the program at values of the prepared layout.

        Parameters
        ----------
        x : torch.Tensor
            The program input, one run.
        values : Mapping[str, torch.Tensor]
            A value for every site and host datum, by name, of the layout
            prepared for.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            The density of each site and each let or score binding, by
            label, in the order the run reaches them, each of shape
            ``(1,)``, as `quivers.inference.trace.trace` records them;
            and their sum, the joint of the run, of shape ``(1,)``, as
            `MonadicProgram.log_joint` returns it.

        Raises
        ------
        ValueError
            If the input or a value is not of the prepared layout, or
            handlers are on the stack, which the plan does not install.
        Exception
            Any error a host step or a site's density raises.
        """
        if joint_signature(x, values) != self._signature:
            raise ValueError(
                "JointPlan.evaluate: the input and values are not of the layout "
                "the plan was prepared for"
            )
        if _handler_stack():
            raise ValueError(
                "JointPlan.evaluate: handlers are on the stack; the plan scores "
                "the program under none"
            )
        kernel = self._kernel
        run_sites = {
            label: _run_slice(kernel, label, values[label], 1, 0)
            for label in self._sites
        }
        axis = self._axis.copy()
        run = RunContext(
            kernel,
            host_value,
            observations=dict(run_sites),
            supplied={name: values[name] for name in self._supplied},
            run_index=0,
            runs=1,
            slice_value=lambda label, value: _run_slice(kernel, label, value, 1, 0),
            carries_run_axis=axis.carries,
            site_carries_run_axis=axis.site,
        )
        store: dict[str, torch.Tensor] = {}
        for step in self._draws:
            assert isinstance(step.spec, Draw | Observe)
            morphism = step.program.step_module(step.spec)
            assert isinstance(morphism, ContinuousMorphism)
            for name, parameter in morphism.named_parameters():
                store[f"{step.site}.{name}"] = parameter
        run.parameters = store

        def parameters(
            site: str, morphism: ContinuousMorphism
        ) -> dict[str, torch.Tensor]:
            """Read a draw's parameters from the run's store.

            Parameters
            ----------
            site : str
                The site the morphism draws at.
            morphism : ContinuousMorphism
                The morphism.

            Returns
            -------
            dict[str, torch.Tensor]
                Its parameters, by its own names.
            """
            return {
                name: _parameter(store, f"{site}.{name}")
                for name, _ in morphism.named_parameters()
            }

        slots: list[object] = [None] * self._slots
        slots[0] = x
        for index, name in enumerate(kernel.data, start=1):
            slots[index] = values[name]
        state = _Evaluation(
            slots,
            _HostSteps(run, x, axis),
            parameters,
            {
                label: _canonical(_site_morphism(kernel, label), value)
                for label, value in run_sites.items()
            },
        )
        for instruction in self._instructions:
            instruction.execute(state)
        return self._densities(state, x.device)

    def _densities(
        self, state: _Evaluation, device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Gather the run's densities as `TraceHandler` gathers a trace's.

        Parameters
        ----------
        state : _Evaluation
            The completed evaluation.
        device : torch.device
            The device the joint's accumulator seeds on.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            Each binding's density, shape ``(1,)``, and their sum.
        """
        densities: dict[str, torch.Tensor] = {}
        for kind, step, payload in state.events:
            if kind == "site":
                assert isinstance(payload, tuple)
                label, weight = payload
                densities[label] = weight
                continue
            assert step is not None and isinstance(step.spec, Let | Score)
            if step.kind != "score" and isinstance(payload, tuple):
                continue
            assert isinstance(payload, torch.Tensor)
            if step.kind == "score":
                contributions = state.scores.get(step.position, [])
                if contributions:
                    density = contributions[0]
                    for item in contributions[1:]:
                        density = density + item
                else:
                    density = torch.zeros_like(payload)
            else:
                density = torch.zeros((), device=payload.device)
            densities[step.spec.name] = density
        per_run = {
            label: torch.stack([density.sum()]) for label, density in densities.items()
        }
        total = torch.zeros(1, device=device)
        for density in per_run.values():
            total = total + density
        return per_run, total


def _parameter(store: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """A parameter from the run's store, as the parameter handler answers.

    Parameters
    ----------
    store : Mapping[str, torch.Tensor]
        The store, by qualified name.
    name : str
        The qualified name.

    Returns
    -------
    torch.Tensor
        The parameter.

    Raises
    ------
    RuntimeTypeMismatch
        If the stored value is not a host value.
    """
    value = store[name]
    _checked(value, f"parameter {name!r}")
    return value


def prepare_joint(
    program: MonadicProgram,
    x: torch.Tensor,
    values: Mapping[str, torch.Tensor],
) -> JointPlan | str:
    """Compile a program's log joint for the layout of given values.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    x : torch.Tensor
        The program input.
    values : Mapping[str, torch.Tensor]
        Values of the layout to prepare for: a value for every site the
        program reaches, and its host data, by name.

    Returns
    -------
    JointPlan | str
        The plan, or why the program at these values is scored on the
        reference machine instead: an input of more than one run, a
        value that is not a tensor, handlers on the stack, a site given
        no value, or an entry computation that is not a straight bind
        chain (a program with a scan).

    Raises
    ------
    ValueError
        If the values are refused as `MonadicProgram.log_joint` refuses
        them: host data of the wrong shape, or a value under a name the
        program computes.
    TypeError
        If host data has the wrong element type.
    """
    if x.dim() and int(x.shape[0]) != 1:
        return f"the input has {int(x.shape[0])} runs"
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            return f"{name!r} is a {type(value).__name__}, not a tensor"
    if _handler_stack():
        return "handlers are on the stack"
    sites, supplied, data = _split_observations(program, values)
    per_run_data = check_host_data(program, data, runs=1 if x.dim() else None)
    kernel = _cached_kernel(program, tuple(sorted(data)))
    compiler = _Compiler(kernel)
    reason = compiler.compile()
    if reason is not None:
        return reason
    missing = sorted(label for label in compiler.labels if label not in sites)
    if missing:
        return f"the sites {missing} are given no value"
    axis = _RunAxis(program, {name: per_run_data.get(name, False) for name in data})
    for label, value in sites.items():
        if "@" not in label:
            axis.given(
                program,
                label,
                _site_carries(kernel, label, _run_slice(kernel, label, value, 1, 0)),
            )
    return JointPlan(
        program,
        kernel,
        compiler.instructions,
        len(compiler.slots),
        joint_signature(x, values),
        tuple(sites),
        tuple(supplied),
        axis,
    )


class JointPlans:
    """The plans of one program at one input, one per layout of the values.

    A consumer that scores the same program at many values of a few
    layouts, an MCMC potential for instance, holds one: each layout is
    prepared the first time it is scored, and the reason a layout cannot
    be prepared is kept with it, so the decision is made once.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    x : torch.Tensor
        The input every evaluation runs the program on.
    """

    def __init__(self, program: MonadicProgram, x: torch.Tensor) -> None:
        self._program = program
        self._x = x
        self._plans: dict[JointSignature, JointPlan | str] = {}

    @property
    def prepared(self) -> int:
        """How many layouts have been prepared or declined so far."""
        return len(self._plans)

    def plan(self, values: Mapping[str, torch.Tensor]) -> JointPlan | str:
        """The plan for the layout of given values.

        Parameters
        ----------
        values : Mapping[str, torch.Tensor]
            A value for every site and host datum, by name.

        Returns
        -------
        JointPlan | str
            The layout's plan, prepared on first use; or why values of
            this layout, or any call made while handlers are on the
            stack, are scored on the reference machine.

        Raises
        ------
        ValueError
            If the values are refused as `prepare_joint` refuses them.
        TypeError
            If host data has the wrong element type.
        """
        if _handler_stack():
            return "handlers are on the stack"
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                return f"{name!r} is a {type(value).__name__}, not a tensor"
        signature = joint_signature(self._x, values)
        plan = self._plans.get(signature)
        if plan is None:
            plan = prepare_joint(self._program, self._x, values)
            self._plans[signature] = plan
        return plan

    def log_joint(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The program's joint log density at given values.

        Parameters
        ----------
        values : Mapping[str, torch.Tensor]
            A value for every site and host datum, by name.

        Returns
        -------
        torch.Tensor
            `MonadicProgram.log_joint` at the input and ``values``, shape
            ``(1,)``: the layout's plan evaluated, or the program run on
            the reference machine for a layout no plan describes.

        Raises
        ------
        Exception
            Any error ``log_joint`` raises at these values.
        """
        plan = self.plan(values)
        if isinstance(plan, JointPlan):
            return plan.evaluate(self._x, values)[1]
        return self._program.log_joint(self._x, dict(values))


__all__ = [
    "JointPlan",
    "JointPlans",
    "JointSignature",
    "joint_signature",
    "prepare_joint",
]
