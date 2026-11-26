"""A program that runs its checked computation on the reference machine.

A ``program`` that calls a named computation the compiler cannot reduce
to compiled runtime steps (a recursive call, a branch whose arms draw
different sites, a handler that is not discharged, a ``case``) compiles to
a [`CheckedProgram`][quivers.effects.checked_program.CheckedProgram]. It is
a [`MonadicProgram`][quivers.continuous.MonadicProgram] with no host steps,
whose body is the program's computation in its checked module, run on the
reference machine with every number a tensor.

A run installs the handlers on the effect handler stack exactly as a run
of a compiled program does: each handler's installations go on the
module's canonical ``random``, ``score`` and ``Param`` instances, around
the computation, under a drawing handler that draws and scores every site
nobody else answers and a replay that conditions every site given a value.
Draws and densities go through the tensor backend, so the log joint, a
reparameterized draw and the value keep the gradient of every tensor the
run was given, through calls, recursion and handler clauses. A trace,
`log_joint`, `rsample`, and every inference consumer built on them read
the program as they read a compiled one.

A site reached more than once in a run, inside a recursion or a repeated
call, is one site per occurrence; a trace names its occurrences
``"<label>@<n>"`` in the order the run reaches them, as a replay takes
them.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import math

import torch
from torch import nn

from quivers.continuous.morphisms import AnySpace
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.tensor_backend import TensorBackend
from quivers.effects.base import (
    Contribution,
    EffectHandler,
    Installation,
    RunContext,
    _handler_stack,
    handle_chain,
)
from quivers.effects.program_module import PARAM_INSTANCE, ProgramKernel
from quivers.effects.trace_handler import TraceHandler
from quivers.qiec.builtins import (
    PARAM,
    ExtraValuePolicy,
    MissingValuePolicy,
    condition_handler,
    draw_scoring_handler,
    param_handler,
    score_handler,
)
from quivers.qiec.checking import KernelRegistry
from quivers.qiec.effects import ComputationType, EffectRow, instantiate_effect
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec.evaluator import Evaluator, RuntimeAttachments, RuntimeRequest
from quivers.qiec.execution import CoreRuntimeProvider
from quivers.qiec.identifiers import ComputationId, SourceOrigin
from quivers.qiec.module import (
    NamedComputation,
    NamedEffectInstance,
    QiecModule,
    validate_module,
)
from quivers.qiec.numbers import NumberDomain, machine_value, number_kind
from quivers.qiec.program_runtime import (
    broadcast_shared_data,
    program_entry,
    static_arguments_for,
)
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.terms import Call, Local, Var
from quivers.qiec.types import IndexTerm, TypeExpr

RUN = "__checked_run"
"""The name of the computation a run wraps the program in."""


def machine_value_admitted(value: object) -> bool:
    """Whether a handler's answer is a value the machine holds.

    Every number the machine holds is a tensor, so an answer that is a
    Python number would enter the computation without the run's dtype and
    without a gradient; it is refused where it enters.

    Parameters
    ----------
    value : object
        The answer.

    Returns
    -------
    bool
        ``False`` for a Python number or a tuple holding one.
    """
    if isinstance(value, bool | int | float):
        return False
    if isinstance(value, tuple):
        return all(machine_value_admitted(item) for item in value)
    return True


def _run_axis(value: torch.Tensor, runs: int) -> bool:
    """Whether a value is given once per run, along a leading axis.

    Parameters
    ----------
    value : torch.Tensor
        The value.
    runs : int
        The number of runs.

    Returns
    -------
    bool
        Whether it leads with an axis of length ``runs``.
    """
    return value.dim() >= 1 and int(value.shape[0]) == runs


def _at_run(value: object, sampleable: object, *, row: int, runs: int) -> object:
    """A given site value's entry for one run of a call.

    A value has one axis more than a draw of the site, leading with the
    run axis, when it is given once per run; otherwise it is one value
    every run shares. A draw's shape is the plate's axes followed by the
    family's own, which the site's distribution states.

    Parameters
    ----------
    value : object
        The value as the call gives it.
    sampleable : object
        The site's distribution.
    row : int
        The run.
    runs : int
        The number of runs of the call.

    Returns
    -------
    object
        The run's value.
    """
    if not isinstance(value, torch.Tensor) or not isinstance(
        sampleable, RuntimeDistribution
    ):
        return value
    rank = len(sampleable.batch) + len(sampleable.event) + len(sampleable.natural)
    if value.dim() == rank + 1 and int(value.shape[0]) == runs:
        return value[row]
    return value


def _computation(module: QiecModule, entry: ProgramEntry) -> NamedComputation:
    """The computation a program's entry names.

    Parameters
    ----------
    module : QiecModule
        The module.
    entry : ProgramEntry
        The program's entry.

    Returns
    -------
    NamedComputation
        The computation.

    Raises
    ------
    KeyError
        If the module has no such computation.
    """
    for computation in module.computations:
        if computation.id == entry.computation:
            return computation
    raise KeyError(f"module {module.module!r} has no computation {entry.computation!r}")


def _instance(module: QiecModule, effect: str) -> NamedEffectInstance | None:
    """The module's canonical instance of a prelude interface.

    Parameters
    ----------
    module : QiecModule
        The module.
    effect : str
        ``"Random"``, ``"Score"`` or ``"Param"``.

    Returns
    -------
    NamedEffectInstance | None
        The instance, or ``None`` when the module declares none.
    """
    for instance in module.instances:
        if instance.entry.effect.name == effect:
            return instance
    return None


class CheckedProgram(MonadicProgram):
    """A program run as its checked computation on the reference machine.

    Parameters
    ----------
    domain : AnySpace
        The program's declared domain.
    codomain : AnySpace
        The program's declared codomain.
    module : QiecModule
        The checked module holding the program's entry point.
    name : str
        The program's name.
    refusal : Exception | None
        Why the compiled runtime does not run the program: the located
        diagnostic of the call it could not lower to steps; ``None`` gives
        one naming the program.

    Attributes
    ----------
    weights : torch.nn.ParameterDict
        The learned weights the program's ``Param`` instance answers, by
        name, registered so an optimizer sees them. A name it does not
        hold reads as zero.
    numbers : NumberDomain
        The dtype and device a run carries its numbers in; the input's
        domain when a run is given a floating input.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        module: QiecModule,
        name: str,
        refusal: Exception | None = None,
    ) -> None:
        super().__init__(domain, codomain, (), (name,))
        self._module = module
        self._name = name
        self._entry: ProgramEntry = program_entry(module, name)
        self._computation = _computation(module, self._entry)
        self._refusal = refusal
        self.input_contract = self._entry
        self.weights = nn.ParameterDict()
        random = _instance(module, "Random")
        score = _instance(module, "Score")
        if random is None or score is None:
            raise ValueError(
                f"program {name!r} has no canonical random and score instances"
            )
        self._random = random
        self._score = score
        self._param = _instance(module, "Param")
        # A program that reads no learned weight performs no Param
        # instance; the kernel's is then one the program never performs,
        # so nothing installed on it reaches the run.
        param = self._param or NamedEffectInstance(
            name=PARAM_INSTANCE,
            entry=instantiate_effect(
                PARAM, module=module.module, lexical_path=("instances", PARAM_INSTANCE)
            ),
            origin=SourceOrigin(
                module=module.module,
                structural_path=("instances", PARAM_INSTANCE),
                role="effect-instance",
                source_protocol=module.source_protocol,
            ),
        )
        self._kernel = ProgramKernel(
            module=module,
            program=self,
            steps=(),
            sites=tuple(site.name for site in self._entry.sites),
            site_types={},
            data=tuple(parameter.name for parameter in self._entry.parameters),
            random=random,
            score=score,
            param=param,
            result_type=self._computation.type.result,
        )
        self._run_modules: dict[
            tuple[tuple[object, ...], ...], tuple[QiecModule, KernelRegistry]
        ] = {}
        self._provider = CoreRuntimeProvider()

    @property
    def refusal(self) -> Exception:
        """Why the compiled runtime does not run the program.

        Returns
        -------
        Exception
            The located diagnostic the compiler recorded for the call it
            could not lower, or one naming the program.
        """
        if self._refusal is not None:
            return self._refusal
        return TypeError(
            f"program {self._name!r} has no compiled steps; it runs on the "
            f"reference machine"
        )

    @property
    def steps_list_sites(self) -> bool:
        """No: the sites are those a run of the checked computation reaches.

        Returns
        -------
        bool
            ``False``; how many sites a run reaches, and which, may depend
            on the values it draws.
        """
        return False

    @property
    def module(self) -> QiecModule:
        """The checked module the program runs in."""
        return self._module

    @property
    def name(self) -> str:
        """The program's name."""
        return self._name

    @property
    def entry(self) -> ProgramEntry:
        """The program's elaborated entry point."""
        return self._entry

    def observed_names(self) -> set[str]:
        """The names the program observes.

        Returns
        -------
        set[str]
            The labels of its ``observe`` sites.
        """
        return {site.name for site in self._entry.sites if site.kind == "observe"}

    def _numbers(self, x: torch.Tensor) -> NumberDomain:
        """The numbers of a run.

        Parameters
        ----------
        x : torch.Tensor
            The input.

        Returns
        -------
        NumberDomain
            The input's dtype and device when it is floating, else torch's
            default floating dtype on its device.
        """
        return NumberDomain.of(x)

    def _split(
        self,
        x: torch.Tensor,
        row: int,
        runs: int,
        given: Mapping[str, torch.Tensor],
        numbers: NumberDomain,
    ) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        """One run's data and its given sites.

        A datum is given once per run when it leads with the run axis and
        has one axis more than its checked type, and is one value every
        run shares otherwise.

        Parameters
        ----------
        x : torch.Tensor
            The input, one row per run.
        row : int
            The run.
        runs : int
            The number of runs.
        given : Mapping[str, torch.Tensor]
            The data by parameter name and the values of sites, by label
            or by ``"<label>@<n>"``.
        numbers : NumberDomain
            The run's numbers.

        Returns
        -------
        tuple[dict[str, object], dict[str, torch.Tensor]]
            The entry's arguments by parameter name, and the values of the
            given sites as the call gives them; which entry of a site's
            value is the run's is read when the run reaches the site.
        """
        data: dict[str, object] = {}
        domain = [p for p in self._entry.parameters if p.role == "domain"]
        if domain:
            value = x[row]
            parts = (
                [value]
                if len(domain) == 1
                else [value[..., index] for index in range(len(domain))]
            )
            for parameter, part in zip(domain, parts, strict=True):
                scalar = number_kind(parameter.type) is not None and not _dimensions(
                    parameter.type
                )
                data[parameter.name] = machine_value(
                    parameter.type,
                    part.reshape(()) if scalar and part.numel() == 1 else part,
                    numbers,
                )
        types = {parameter.name: parameter.type for parameter in self._entry.parameters}
        sites: dict[str, torch.Tensor] = {}
        for name, value in given.items():
            if name in types and name not in data:
                rank = len(_dimensions(types[name]))
                per_run = (
                    isinstance(value, torch.Tensor)
                    and value.dim() == rank + 1
                    and _run_axis(value, runs)
                )
                data[name] = machine_value(
                    types[name], value[row] if per_run else value, numbers
                )
            elif name not in types:
                sites[name] = value
        return data, sites

    def _run_module(
        self, installations: list[Installation]
    ) -> tuple[QiecModule, KernelRegistry, NamedComputation]:
        """The module of a run: the program's computation under the handlers.

        The module depends only on the declarations of the installations,
        so it is built and validated once per set of them and reused.

        Parameters
        ----------
        installations : list[Installation]
            The run's installations, outermost first.

        Returns
        -------
        tuple[QiecModule, KernelRegistry, NamedComputation]
            The module, its validated registry, and the wrapper.
        """
        key = tuple(
            (installation.instance, installation.runtime.definition)
            for installation in installations
        )
        cached = self._run_modules.get(key)
        computation = self._computation
        parameters = tuple(
            Local(name=parameter.name, type=parameter.type)
            for parameter in computation.parameters
        )
        origin = SourceOrigin(
            module=self._module.module,
            structural_path=("programs", self._name, "run"),
            role="computation",
            source_protocol=self._module.source_protocol,
        )
        if cached is None:
            call = Call(
                callee=computation.id,
                name=computation.name,
                static_arguments=(),
                arguments=tuple(Var(local=parameter) for parameter in parameters),
                result_type=computation.type.result,
                effects=computation.type.effects,
                origin=origin.with_(role="call"),
            )
            wrapper = NamedComputation(
                id=ComputationId.derive(self._module.module, "computation", RUN),
                name=RUN,
                telescope=(),
                parameters=parameters,
                body=handle_chain(installations, call),
                type=ComputationType(
                    effects=EffectRow(), result=computation.type.result
                ),
                origin=origin,
            )
            module = self._module.with_(
                handlers=(
                    *self._module.handlers,
                    *(item.runtime.definition for item in installations),
                ),
                computations=(*self._module.computations, wrapper),
            )
            cached = (module, validate_module(module))
            self._run_modules[key] = cached
        module, registry = cached
        return module, registry, module.computations[-1]

    def _installations(
        self, run: RunContext, stack: list[EffectHandler], numbers: NumberDomain
    ) -> list[Installation]:
        """The installations of one run, outermost first.

        Parameters
        ----------
        run : RunContext
            The run.
        stack : list[EffectHandler]
            The handlers on the stack, outermost first.
        numbers : NumberDomain
            The run's numbers.

        Returns
        -------
        list[Installation]
            The accumulator, the stack's score handlers, the drawing
            handler, the replay of the given sites, the stack's random
            handlers, the parameter store and the stack's parameter
            handlers, each on an instance the program performs.

        Raises
        ------
        ValueError
            If a handler on the stack installs on an instance the run does
            not serve.
        """
        result_type = run.result_type
        performed = {entry.instance for entry in self._computation.type.effects.entries}

        def record(request: RuntimeRequest, weight: object) -> None:
            """Record a contribution reaching the accumulator.

            Parameters
            ----------
            request : RuntimeRequest
                The score request.
            weight : object
                The contribution.
            """
            assert isinstance(weight, torch.Tensor)
            run.contributions.append(
                Contribution(
                    key=run.key_of(request),
                    role=request.core.origin.origin.role,
                    weight=weight,
                    path=request.core.origin.origin.structural_path,
                    derived=bool(request.core.origin.parents),
                )
            )

        accumulator, _ = score_handler(
            identity=numbers.zero(),
            answer_type=result_type,
            observer=record,
            key="checked-accumulator",
        )
        scores = [Installation(instance=self._score, runtime=accumulator)]
        randoms: list[Installation] = [
            Installation(
                instance=self._random,
                runtime=draw_scoring_handler(
                    score_instance=self._score.entry.instance,
                    result_validator=machine_value_admitted,
                    answer_type=result_type,
                    key="checked-draw",
                ),
            ),
            Installation(
                instance=self._random,
                runtime=condition_handler(
                    run.observations,
                    score_instance=self._score.entry.instance,
                    result_validator=machine_value_admitted,
                    missing=MissingValuePolicy.FORWARD,
                    extra=ExtraValuePolicy.ERROR,
                    answer_type=result_type,
                    key="checked-given",
                    at_run=partial(_at_run, row=run.run_index, runs=run.runs),
                ),
            ),
        ]
        params: list[Installation] = []
        if self._param is not None:
            weights = self.weights

            def lookup(name: str, request: RuntimeRequest) -> object:
                """Read one learned weight.

                Parameters
                ----------
                name : str
                    The weight's name.
                request : RuntimeRequest
                    The request, whose result type the weight is read at.

                Returns
                -------
                object
                    The weight, or zero when the program holds none.
                """
                value = weights[name] if name in weights else 0.0
                return machine_value(request.core.result_type, value, numbers)

            params.append(
                Installation(
                    instance=self._param,
                    runtime=param_handler(
                        lookup,
                        result_validator=machine_value_admitted,
                        answer_type=result_type,
                        key="checked-weights",
                    ),
                )
            )
        for handler in stack:
            for installation in handler.install(run):
                if installation.instance is self._score:
                    scores.append(installation)
                elif installation.instance is self._random:
                    randoms.append(installation)
                elif self._param is not None and installation.instance is self._param:
                    params.append(installation)
                else:
                    raise ValueError(
                        f"{type(handler).__name__} installs on an instance the run "
                        "does not serve"
                    )
        installations = [
            *(scores if self._score.entry.instance in performed else ()),
            *(randoms if self._random.entry.instance in performed else ()),
            *(
                params
                if self._param is not None and self._param.entry.instance in performed
                else ()
            ),
        ]
        return [
            installation.named(f"{installation.runtime.definition.name}#{position}")
            for position, installation in enumerate(installations)
        ]

    def _run(
        self,
        x: torch.Tensor,
        row: int,
        runs: int,
        given: Mapping[str, torch.Tensor],
    ) -> object:
        """One run of the program under the handler stack.

        Parameters
        ----------
        x : torch.Tensor
            The input, one row per run.
        row : int
            The run.
        runs : int
            The number of runs of the call.
        given : Mapping[str, torch.Tensor]
            The data and the values of the given sites.

        Returns
        -------
        object
            The program's value, as the machine holds it.
        """
        numbers = self._numbers(x)
        data, sites = self._split(x, row, runs, given, numbers)
        data = broadcast_shared_data(self._entry, data)
        statics = static_arguments_for(self._entry, data)
        if statics:
            raise ValueError(
                f"program {self._name!r} binds static extents, which a checked "
                "program run fixes from its data; give them as arguments of the "
                "entry"
            )
        run = RunContext(
            kernel=self._kernel,
            validator=machine_value_admitted,
            observations=dict(sites),
            run_index=row,
            runs=runs,
        )
        stack = list(_handler_stack())
        installations = self._installations(run, stack, numbers)
        module, registry, wrapper = self._run_module(installations)
        attachments = RuntimeAttachments(numbers=numbers, distributions=TensorBackend())
        for installation in installations:
            attachments.bind_handler(installation.runtime)
        evaluator = Evaluator(
            attachments, module=module, type_validator=self._provider.validator_for
        )
        missing = [p.name for p in wrapper.parameters if p.name not in data]
        if missing:
            raise ValueError(
                f"program {self._name!r} is not given its parameter(s) "
                f"{', '.join(repr(name) for name in missing)}: a checked program "
                "scores an observation of a datum it is given and does not draw it"
            )
        environment = {
            parameter: machine_value(parameter.type, data[parameter.name], numbers)
            for parameter in wrapper.parameters
        }
        active = _handler_stack()
        entered = active[:]
        active.clear()
        try:
            output = evaluator.evaluate_validated(wrapper.body, registry, environment)
        finally:
            active.extend(entered)
        for handler in stack:
            handler.finish(run, output)
        return output

    def rsample(  # type: ignore[override]
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
        observations: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the program forward under the handler stack.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``: one run per row.
        sample_shape : torch.Size
            Additional leading sample dimensions; each element is one
            independent call.
        observations : Mapping[str, torch.Tensor] or None
            The program's data by parameter name, and the values of the
            sites the runs are given, by label or ``"<label>@<n>"``; a
            value leading with the run axis is one per run. Every other
            site is drawn.

        Returns
        -------
        torch.Tensor
            The returned values, shape ``(*sample_shape, batch, ...)``,
            with the gradient of every tensor the runs were given.

        Raises
        ------
        TypeError
            If the program returns a value that is not a tensor.
        """
        given = dict(observations or {})
        runs = int(x.shape[0]) if x.dim() >= 1 else 1
        calls: list[torch.Tensor] = []
        for _ in range(max(math.prod(sample_shape), 1)):
            values: list[torch.Tensor] = []
            for row in range(runs):
                value = self._run(x, row, runs, given)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"program {self._name!r} returns a "
                        f"{type(value).__name__}, which has no tensor form"
                    )
                values.append(value)
            calls.append(torch.stack(values))
        stacked = torch.stack(calls)
        if not sample_shape:
            return stacked[0]
        return stacked.reshape((*sample_shape, *stacked.shape[1:]))

    def has_conditional_density(self) -> bool:
        """No: a program's density at its output marginalizes its draws.

        Returns
        -------
        bool
            Always ``False``;
            [`log_joint`][quivers.effects.checked_program.CheckedProgram.log_joint]
            scores the draws themselves.
        """
        return False

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability is not defined for a program's output alone.

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        y : torch.Tensor
            Program output.

        Raises
        ------
        NotImplementedError
            Always: the density at the output marginalizes every draw.
        """
        del x, y
        raise NotImplementedError(
            "log_prob is not supported for programs; computing p(y | x) "
            "requires marginalizing over all intermediate draws. Use "
            "log_joint with a value for every site."
        )

    def log_joint(
        self, x: torch.Tensor, intermediates: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """The joint log density at given values for every site.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``: one run per row.
        intermediates : Mapping[str, torch.Tensor]
            The program's data by parameter name and a value for every
            site the runs reach, by label, or by ``"<label>@<n>"`` for a
            site a run reaches more than once.

        Returns
        -------
        torch.Tensor
            The log joint of each run, shape ``(batch,)``, with the
            gradient of every value given.

        Raises
        ------
        KeyError
            If a site a run reaches is given no value.
        """
        with TraceHandler() as handler:
            self.rsample(x, observations=intermediates)
        for name, site in handler.trace.sites.items():
            if (
                site.sampleable is not None
                and not site.is_observed
                and not site.is_deterministic
            ):
                raise KeyError(name)
        return handler.total_log_joint(x.device)

    def __repr__(self) -> str:
        return f"CheckedProgram({self._name!r}: {self.domain!r} -> {self.codomain!r})"


def _dimensions(type_: TypeExpr) -> tuple[IndexTerm, ...]:
    """The dimensions of a tensor type, none for any other type.

    Parameters
    ----------
    type_ : TypeExpr
        A type.

    Returns
    -------
    tuple[IndexTerm, ...]
        The dimensions.
    """
    shape = tensor_shape(type_)
    return () if shape is None else shape[1]


__all__ = ["CheckedProgram", "machine_value_admitted"]
