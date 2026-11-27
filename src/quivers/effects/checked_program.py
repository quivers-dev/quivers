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
from typing import cast

import torch
from torch import nn

from quivers._positions import depends_on_position, map_positions
from quivers.continuous.morphisms import AnySpace
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.tensor_backend import TensorBackend
from quivers.effects.base import (
    Contribution,
    EffectHandler,
    Installation,
    RunContext,
    _handler_stack,
    fixed_site_values,
    handle_chain,
)
from quivers.effects._observation_laws import ScoreRoles, observation_variant
from quivers.effects.observation_scoring import observations_scored
from quivers.effects.program_module import PARAM_INSTANCE, ProgramKernel
from quivers.effects.trace_handler import TraceHandler
from quivers.qiec.builtins import (
    PARAM,
    ExtraValuePolicy,
    MissingSiteValue,
    MissingValuePolicy,
    ScoreAccumulator,
    condition_handler,
    draw_scoring_handler,
    param_handler,
    score_handler,
)
from quivers.qiec.checking import KernelRegistry
from quivers.qiec.effects import (
    ComputationType,
    EffectRow,
    instantiate_effect,
)
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec.evaluator import Evaluator, RuntimeAttachments, RuntimeRequest
from quivers.qiec.execution import CoreRuntimeProvider
from quivers.qiec.identifiers import ComputationId, SourceOrigin
from quivers.qiec.module import (
    NamedComputation,
    NamedEffectInstance,
    QiecModule,
    reached_handlers,
    validate_module,
)
from quivers.qiec.contracts import Phase, WorkField, WorkReport, measuring
from quivers.qiec.numbers import (
    NumberDomain,
    PositionDependentControl,
    machine_value,
    number_kind,
)
from quivers.qiec.program_runtime import (
    broadcast_shared_data,
    foreign_handler_configuration,
    program_entry,
    static_arguments_for,
)
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.terms import (
    Call,
    Local,
    Var,
)
from quivers.qiec.substitution import (
    instantiate_telescope,
    substitute_row,
    substitute_type,
)
from quivers.qiec.types import IndexTerm, StaticArgument, TypeExpr

RUN = "__checked_run"
"""The name of the computation a run wraps the program in."""

OBSERVED_APART = (
    "a handler on the stack is installed in each run and may watch its sites as "
    "the run reaches them, and in a pass over every run at once a site's value "
    "is every run's"
)
"""Why the forward runs of a call under handlers are made one by one."""


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

    Raises
    ------
    PositionDependentControl
        If, in a pass over every run at once, a value mapped along the run
        axis does not have the site's shape: it was one value every run
        shares whose leading axis happens to have the runs' length.
    """
    if not isinstance(value, torch.Tensor) or not isinstance(
        sampleable, RuntimeDistribution
    ):
        return value
    rank = len(sampleable.batch) + len(sampleable.event) + len(sampleable.natural)
    if depends_on_position(value):
        # A pass over every run at once mapped the value along the run
        # axis; it is the run's only when it has the site's shape.
        if value.dim() == rank:
            return value
        raise PositionDependentControl(
            "a site value shared by every run whose leading axis has the runs' length"
        )
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
    prior : bool
        Whether the program drops the scores of its observations, which
        leaves its prior; otherwise each observation with a law is a site,
        conditioned when it is given and drawn when it is not.

    Attributes
    ----------
    weights : torch.nn.ParameterDict
        The learned weights the program's ``Param`` instance answers, by
        name, registered so an optimizer sees them. A name it does not
        hold reads as zero.
    work : WorkReport
        The work of the last call: its ``iterations`` are the machine
        passes it made, one for a call whose runs were made together and
        one per run otherwise.
    vectorization_obstacle : str or None
        Why the last call made one pass per run, or ``None`` when it made
        one pass, or had one run.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        module: QiecModule,
        name: str,
        refusal: Exception | None = None,
        *,
        prior: bool = False,
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
        self._prior = prior
        # Each observation scored at the top of the body under a law of its
        # own becomes a draw of that law at its site, or, for the prior,
        # leaves the program; the prior, and a run whose likelihood a
        # handler turns off, drop every other observation density too.
        self._laws = observation_variant(
            module,
            self._computation,
            self._entry,
            random,
            score.entry.instance,
            drop=prior,
            likelihoods=not prior,
        )
        self._unscored = observation_variant(
            module,
            self._computation,
            self._entry,
            random,
            score.entry.instance,
            drop=prior,
            likelihoods=False,
        )
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
            tuple[object, ...], tuple[QiecModule, KernelRegistry]
        ] = {}
        self._provider = CoreRuntimeProvider()
        self._foreign = CoreRuntimeProvider(
            options={
                "handlers": foreign_handler_configuration(
                    module, reached=reached_handlers(module, self._computation.id)
                )
            }
        )
        self._kernels: dict[tuple[StaticArgument, ...], ProgramKernel] = {}
        self.work = WorkReport(phase=Phase.EXECUTION)
        self.vectorization_obstacle: str | None = None

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
        """The checked module the program runs in.

        Returns
        -------
        QiecModule
            The module.
        """
        return self._module

    @property
    def name(self) -> str:
        """The program's name.

        Returns
        -------
        str
            The name.
        """
        return self._name

    @property
    def entry(self) -> ProgramEntry:
        """The program's elaborated entry point.

        Returns
        -------
        ProgramEntry
            The entry.
        """
        return self._entry

    def prior_program(self) -> CheckedProgram:
        """The program without the scores of its observations.

        Returns
        -------
        CheckedProgram
            The program over the same module and learned weights, every
            observation's density dropped and every draw kept.

        Raises
        ------
        ValueError
            If an observation's density is added where no bind of the
            program's body can drop it: inside a computation it calls, or
            under a branch.
        """
        if self._unscored.unremoved:
            raise ValueError(_unremoved(self._name, self._unscored.unremoved))
        prior = CheckedProgram(
            self.domain,
            self.codomain,
            self._module,
            self._name,
            self._refusal,
            prior=True,
        )
        prior.weights = self.weights
        return prior

    @property
    def kept_data(self) -> Mapping[str, str]:
        """The observations that stay data the program must be given.

        Returns
        -------
        Mapping[str, str]
            Each observation the program scores but cannot draw, with why.
        """
        return self._laws.kept

    @property
    def potentials(self) -> tuple[str, ...]:
        """The scores of the program that no law of a site accounts for.

        Returns
        -------
        tuple[str, ...]
            The structural path, dotted, of every ``Score`` request a run
            makes, in the program's computation or one it calls, that adds
            neither the density of an ``observe`` site's law nor a value
            computed from such densities, as a marginalization block's
            total is. Empty when the computation performs no score at all.
        """
        computation = self._laws.computation
        instance = self._score.entry.instance
        if all(item.instance != instance for item in computation.type.effects.entries):
            return ()
        return tuple(
            path
            for path, likelihood in ScoreRoles(self._module).scores(
                computation.body, instance
            )
            if not likelihood
        )

    @property
    def potential_sites(self) -> frozenset[str]:
        """The trace names of the program's own scores, every one a potential.

        Returns
        -------
        frozenset[str]
            ``"score:<path>"`` for each score the program adds that is not
            an observation's density: a trace records such a score as a
            deterministic site of that name, and it belongs to the prior.
        """
        return frozenset(f"score:{path}" for path in self.potentials)

    @property
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
        x_row: torch.Tensor,
        given: Mapping[str, torch.Tensor],
        numbers: NumberDomain,
        *,
        row: int,
        runs: int,
    ) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        """One run's data and its given sites.

        A datum is given once per run when it leads with the run axis and
        has one axis more than its checked type, and is one value every
        run shares otherwise; in a pass over every run at once, a datum
        given per run is already the run's.

        Parameters
        ----------
        x_row : torch.Tensor
            The run's input.
        given : Mapping[str, torch.Tensor]
            The data by parameter name and the values of sites, by label
            or by ``"<label>@<n>"``.
        numbers : NumberDomain
            The run's numbers.
        row : int
            The run.
        runs : int
            The number of runs.

        Returns
        -------
        tuple[dict[str, object], dict[str, torch.Tensor]]
            The entry's arguments by parameter name, and the values of the
            given sites as the call gives them, an observation with a law
            among them; which entry of a site's value is the run's is read
            when the run reaches the site.
        """
        data: dict[str, object] = {}
        domain = [p for p in self._entry.parameters if p.role == "domain"]
        parts = (
            [x_row]
            if len(domain) == 1
            else [x_row[..., index] for index in range(len(domain))]
        )
        for parameter, part in zip(domain, parts, strict=False):
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
        observations = set(self._laws.observations)
        for name, value in given.items():
            if name in observations:
                # An observation with a law is a site: its value conditions
                # the draw, and the prior has no such site.
                if not self._prior:
                    sites[name] = value
            elif name in types and name not in data:
                rank = len(_dimensions(types[name]))
                per_run = (
                    isinstance(value, torch.Tensor)
                    and not depends_on_position(value)
                    and value.dim() == rank + 1
                    and _run_axis(value, runs)
                )
                data[name] = machine_value(
                    types[name], value[row] if per_run else value, numbers
                )
            elif name not in types:
                sites[name] = value
        return data, sites

    def _kernel_at(self, statics: tuple[StaticArgument, ...]) -> ProgramKernel:
        """The program's kernel at its static extents.

        Parameters
        ----------
        statics : tuple[StaticArgument, ...]
            One closed index per binder of the program's telescope.

        Returns
        -------
        ProgramKernel
            The kernel whose result type is the program's at the extents,
            built once per extents.
        """
        kernel = self._kernels.get(statics)
        if kernel is None:
            substitution = instantiate_telescope(self._computation.telescope, statics)
            kernel = self._kernel.with_(
                result_type=substitute_type(self._computation.type.result, substitution)
            )
            self._kernels[statics] = kernel
        return kernel

    def _run_module(
        self,
        installations: list[Installation],
        statics: tuple[StaticArgument, ...],
        computation: NamedComputation,
    ) -> tuple[QiecModule, KernelRegistry, NamedComputation]:
        """The module of a run: the program's computation under the handlers.

        The module depends only on the declarations of the installations
        and the static extents, so it is built and validated once per set
        of them and reused.

        Parameters
        ----------
        installations : list[Installation]
            The run's installations, outermost first.
        statics : tuple[StaticArgument, ...]
            The program's static extents.
        computation : NamedComputation
            The variant of the program's computation the run makes.

        Returns
        -------
        tuple[QiecModule, KernelRegistry, NamedComputation]
            The module, its validated registry, and the wrapper.
        """
        key = (
            statics,
            computation.id,
            *(
                (installation.instance, installation.runtime.definition)
                for installation in installations
            ),
        )
        cached = self._run_modules.get(key)
        if cached is None:
            substitution = instantiate_telescope(computation.telescope, statics)
            parameters = tuple(
                Local(
                    name=parameter.name,
                    type=substitute_type(parameter.type, substitution),
                )
                for parameter in computation.parameters
            )
            origin = SourceOrigin(
                module=self._module.module,
                structural_path=("programs", self._name, "run"),
                role="computation",
                source_protocol=self._module.source_protocol,
            )
            result_type = substitute_type(computation.type.result, substitution)
            call = Call(
                callee=computation.id,
                name=computation.name,
                static_arguments=statics,
                arguments=tuple(Var(local=parameter) for parameter in parameters),
                result_type=result_type,
                effects=substitute_row(computation.type.effects, substitution),
                origin=origin.with_(role="call"),
            )
            wrapper = NamedComputation(
                id=ComputationId.derive(self._module.module, "computation", RUN),
                name=RUN,
                telescope=(),
                parameters=parameters,
                body=handle_chain(installations, call),
                type=ComputationType(effects=EffectRow(), result=result_type),
                origin=origin,
            )
            module = self._module.with_(
                handlers=(
                    *self._module.handlers,
                    *(item.runtime.definition for item in installations),
                ),
                computations=(*self._module.computations, computation, wrapper),
            )
            cached = (module, validate_module(module))
            self._run_modules[key] = cached
        module, registry = cached
        return module, registry, module.computations[-1]

    def _installations(
        self,
        run: RunContext,
        stack: list[EffectHandler],
        numbers: NumberDomain,
        *,
        joint: bool,
    ) -> tuple[list[Installation], ScoreAccumulator]:
        """The installations of one run, outermost first.

        Parameters
        ----------
        run : RunContext
            The run.
        stack : list[EffectHandler]
            The handlers on the stack, outermost first.
        numbers : NumberDomain
            The run's numbers.
        joint : bool
            Whether the run scores given values only: a site no handler
            answers is then refused rather than drawn.

        Returns
        -------
        tuple[list[Installation], ScoreAccumulator]
            The accumulator, the stack's score handlers, the drawing
            handler (unless ``joint``), the conditioning on the given
            sites, the stack's random handlers, the parameter store and the
            stack's parameter handlers, each on an instance the program
            performs; and the accumulator the run's total reaches.

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

        accumulator, total = score_handler(
            identity=numbers.zero(),
            answer_type=result_type,
            observer=record,
            key="checked-accumulator",
        )
        scores = [Installation(instance=self._score, runtime=accumulator)]
        stack_scores: list[Installation] = []
        stack_randoms: list[Installation] = []
        stack_params: list[Installation] = []
        for handler in stack:
            for installation in handler.install(run):
                if installation.instance is self._score:
                    stack_scores.append(installation)
                elif installation.instance is self._random:
                    stack_randoms.append(installation)
                elif self._param is not None and installation.instance is self._param:
                    stack_params.append(installation)
                else:
                    raise ValueError(
                        f"{type(handler).__name__} installs on an instance the run "
                        "does not serve"
                    )
        scores.extend(stack_scores)
        # Scoring given values, a site no handler answers is refused by
        # name rather than drawn, which is how a run that reaches more sites
        # than its values describe is refused.
        randoms: list[Installation] = [
            Installation(
                instance=self._random,
                runtime=draw_scoring_handler(
                    score_instance=self._score.entry.instance,
                    result_validator=machine_value_admitted,
                    draw=_refuse_draw if joint else None,
                    answer_type=result_type,
                    key="checked-draw",
                ),
            ),
            Installation(
                instance=self._random,
                runtime=condition_handler(
                    cast(Mapping[object, object], run.observations),
                    score_instance=self._score.entry.instance,
                    result_validator=machine_value_admitted,
                    missing=MissingValuePolicy.FORWARD,
                    extra=ExtraValuePolicy.ERROR,
                    answer_type=result_type,
                    key="checked-given",
                    at_run=run.select_run,
                    # A handler on the stack may turn the likelihood off:
                    # an observation is then given but not scored.
                    unscored=(
                        () if observations_scored(run) else self._laws.observations
                    ),
                ),
            ),
            *stack_randoms,
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
        params.extend(stack_params)
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
        ], total

    def _run(
        self,
        x_row: torch.Tensor,
        given: Mapping[str, torch.Tensor],
        *,
        row: int,
        runs: int,
        joint: bool,
    ) -> object:
        """One run of the program under the handler stack.

        Parameters
        ----------
        x_row : torch.Tensor
            The run's input.
        given : Mapping[str, torch.Tensor]
            The data and the values of the given sites.
        row : int
            The run.
        runs : int
            The number of runs of the call.
        joint : bool
            Whether to score the given values rather than run forward.

        Returns
        -------
        object
            The program's value, as the machine holds it; with ``joint``,
            the total log weight the run accumulated.

        Raises
        ------
        ValueError
            If a parameter of the program is not given, or the program
            binds a static extent.
        """
        numbers = self._numbers(x_row)
        data, sites = self._split(x_row, given, numbers, row=row, runs=runs)
        data = broadcast_shared_data(self._entry, data)
        statics = static_arguments_for(self._entry, data)
        run = RunContext(
            kernel=self._kernel_at(statics),
            validator=machine_value_admitted,
            observations=dict(sites),
            run_index=row,
            runs=runs,
            select_run=partial(_at_run, row=row, runs=runs),
        )
        stack = list(_handler_stack())
        installations, total = self._installations(run, stack, numbers, joint=joint)
        laws = self._laws if observations_scored(run) else self._unscored
        if laws.unremoved:
            raise ValueError(_unremoved(self._name, laws.unremoved))
        module, registry, wrapper = self._run_module(
            installations, statics, laws.computation
        )
        attachments = RuntimeAttachments(numbers=numbers, distributions=TensorBackend())
        for installation in installations:
            attachments.bind_handler(installation.runtime)
        # The module's own foreign handlers (the enumerations of its
        # marginalization blocks, the searches and collectors of its
        # deductions) are served by what their declarations state.
        self._foreign.attach(module, attachments)
        evaluator = Evaluator(
            attachments, module=module, type_validator=self._provider.validator_for
        )
        missing = [p.name for p in wrapper.parameters if p.name not in data]
        if missing:
            reasons = "; ".join(
                self._laws.kept.get(name, f"{name!r} is a datum of the program")
                for name in missing
            )
            raise ValueError(
                f"program {self._name!r} is not given its parameter(s) "
                f"{', '.join(repr(name) for name in missing)}: {reasons}"
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
        if joint:
            weight = total.total
            assert isinstance(weight, torch.Tensor)
            return weight
        return output

    def _calls(
        self,
        x: torch.Tensor,
        given: Mapping[str, torch.Tensor],
        *,
        joint: bool,
        apart: str | None,
    ) -> torch.Tensor:
        """Every run of one call, in one pass when control allows it.

        With ``together``, the runs are one pass of the machine over all of
        them at once: every value given once per run (a datum by its
        checked type, a site value by its leading run axis) is mapped along
        the run axis by `quivers._positions.map_positions`, so no shape the
        computation reads sees the axis. A branch, an index into structured
        values or an item key that differs between the runs cannot be taken
        once for all of them; the pass stops and the runs are made one by
        one. `work` and `vectorization_obstacle` report which happened.

        Parameters
        ----------
        x : torch.Tensor
            The input, one row per run.
        given : Mapping[str, torch.Tensor]
            The data and the values of the given sites.
        joint : bool
            Whether to score the given values rather than run forward.
        apart : str or None
            Why the runs cannot be made in one pass whatever their values,
            or ``None`` to try.

        Returns
        -------
        torch.Tensor
            The runs' values, or joints, stacked along the run axis.

        Raises
        ------
        TypeError
            If the program returns a value that is not a tensor.
        """
        runs = int(x.shape[0]) if x.dim() >= 1 else 1
        rows = x if x.dim() >= 1 else x.reshape(1)
        obstacle = apart if runs > 1 else None
        if apart is None and runs > 1:
            mapped = {
                name: value
                for name, value in given.items()
                if isinstance(value, torch.Tensor) and self._per_run(name, value, runs)
            }
            shared = {
                name: value for name, value in given.items() if name not in mapped
            }

            def one(
                position: tuple[torch.Tensor, dict[str, torch.Tensor]],
            ) -> torch.Tensor:
                """One run, at a position of the pass.

                Parameters
                ----------
                position : tuple[torch.Tensor, dict[str, torch.Tensor]]
                    The run's input and its values given per run.

                Returns
                -------
                torch.Tensor
                    The run's value or joint.
                """
                row, own = position
                found = self._run(row, {**shared, **own}, row=0, runs=runs, joint=joint)
                return _tensor_of(found, self._name)

            try:
                values = map_positions(one, (rows, mapped))
            except PositionDependentControl as error:
                obstacle = str(error)
            else:
                self._report(passes=1, obstacle=None)
                if joint:
                    return torch.where(
                        values.accepted,
                        values.values,
                        torch.full_like(values.values, -torch.inf),
                    )
                if not bool(values.accepted.all()):
                    index = int((~values.accepted).nonzero()[0])
                    raise ValueError(
                        f"run {index} of program {self._name!r} fails a check of "
                        "its values"
                    )
                return values.values
        outputs = [
            _tensor_of(
                self._run(rows[row], given, row=row, runs=runs, joint=joint),
                self._name,
            )
            for row in range(runs)
        ]
        self._report(passes=runs, obstacle=obstacle)
        return torch.stack(outputs)

    def _per_run(self, name: str, value: torch.Tensor, runs: int) -> bool:
        """Whether a given value is one value per run of a call.

        Parameters
        ----------
        name : str
            A parameter's name, or a site's label.
        value : torch.Tensor
            The value.
        runs : int
            The number of runs.

        Returns
        -------
        bool
            For a datum, whether it has one axis more than its checked type
            and leads with the run axis; for a site, whether it leads with
            the run axis, which the run checks against the site's shape when
            it reaches the site.
        """
        for parameter in self._entry.parameters:
            if parameter.name == name:
                rank = len(_dimensions(parameter.type))
                return value.dim() == rank + 1 and _run_axis(value, runs)
        return _run_axis(value, runs)

    def _report(self, *, passes: int, obstacle: str | None) -> None:
        """Record how the last call ran its runs.

        Parameters
        ----------
        passes : int
            The machine passes the call made.
        obstacle : str | None
            Why the runs were not one pass, or ``None``.
        """
        self.work = WorkReport(
            phase=Phase.EXECUTION,
            iterations=passes,
            measured=measuring((WorkField.ITERATIONS,)),
        )
        self.vectorization_obstacle = obstacle

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
        # Draws differ between the runs of one pass, so the runs of a call
        # are one pass unless a handler on the stack may watch each run.
        apart = OBSERVED_APART if _handler_stack() else None
        calls = [
            self._calls(x, given, joint=False, apart=apart)
            for _ in range(max(math.prod(sample_shape), 1))
        ]
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
        if fixed_site_values() is not None:
            # The stack only fixes sites, so the joint is what each run's
            # accumulator receives, and no run's values are recorded.
            return self._calls(x, dict(intermediates), joint=True, apart=None)
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
        """The program's name and spaces.

        Returns
        -------
        str
            The representation.
        """
        return f"CheckedProgram({self._name!r}: {self.domain!r} -> {self.codomain!r})"


def _unremoved(name: str, paths: tuple[str, ...]) -> str:
    """Why a program's observation densities cannot be left out.

    Parameters
    ----------
    name : str
        The program.
    paths : tuple[str, ...]
        The densities' structural paths.

    Returns
    -------
    str
        The diagnostic.
    """
    return (
        f"program {name!r} adds the density of an observation at "
        f"{', '.join(paths)}, inside a computation it calls or under a branch, "
        "where its run cannot leave it out; its prior, and a run without its "
        "likelihood, have no such run"
    )


def _refuse_draw(site: object, sampleable: object, request: RuntimeRequest) -> object:
    """Refuse to draw a site a run scoring given values reaches.

    Parameters
    ----------
    site : object
        The site.
    sampleable : object
        Its distribution, unused.
    request : RuntimeRequest
        The request, unused.

    Returns
    -------
    object
        Never returns.

    Raises
    ------
    MissingSiteValue
        Always, naming the site.
    """
    del sampleable, request
    raise MissingSiteValue(site)


def _tensor_of(value: object, name: str) -> torch.Tensor:
    """A run's value as a tensor.

    Parameters
    ----------
    value : object
        The value.
    name : str
        The program, for the diagnostic.

    Returns
    -------
    torch.Tensor
        The value.

    Raises
    ------
    TypeError
        If the value is not a tensor.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"program {name!r} returns a {type(value).__name__}, which has no "
            "tensor form"
        )
    return value


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
