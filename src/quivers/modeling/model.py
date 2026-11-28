"""Compile, bind, and inspect QVR models through checked input metadata."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol, Literal, cast

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous import ContinuousSpace, MonadicProgram
from quivers.dsl import Compiler, CompileError, Module, parse
from quivers.dsl.qiec_lowering import _lower_qvr_with_companions
from quivers.dsl._scoped_interpretations import selected_scoped_interpretations
from quivers.family_catalog import FamilyCatalog
from quivers.dsl.ast_nodes import (
    ExportDecl,
    ExprIdent,
    ProgramDecl,
)
from quivers.effects import CheckedProgram
from quivers.effects.interpreter import check_host_data
from quivers.effects.program_module import once_sites, scanned_sites
from quivers.inference import (
    PriorDraws,
    PriorMode,
    Guide,
    LatentRegistry,
    MCMC,
    MCMCResult,
    SMC,
    SMCResult,
    Predictive,
    Trace,
    trace,
)
from quivers.diagnostics import ConvergenceSummary, summary
from quivers.program import Program
from quivers.qiec import (
    ClosedInterpretationBinding,
    ComputationId,
    IndexLiteral,
    ProgramEntry,
    ProgramParameter,
    QiecModule,
    ScopedInterpretationBinding,
    TypeExpr,
    render_static,
    tensor_shape,
)
from quivers.qiec.module import _execution_reachability


from quivers.diagnostics.posterior import PosteriorAnalysis
from quivers.diagnostics.conditional import ConditionalEffects, PredictionGrid
from quivers.diagnostics.cross_validation import FoldPrediction, FoldSplit
from quivers.modeling._replay import (
    ScoreUnits,
    source_analysis,
    source_unit_log_likelihood,
)
from quivers.modeling.refit import RefitRecipe, SnapshotPolicy

# Pathfinder draws report variational provenance.
from quivers.inference.pathfinder import draw_provenance
from quivers.inference.units import UnitLogLikelihood


class ModelSchema(ValueModel):
    """The checked input contract of one QVR program.

    Parameters
    ----------
    entry : ProgramEntry
        Typed parameters, site roles, plates, and static binders. The entry
        is immutable and compared by value, without rendering away its types.
    scoped_interpretations : tuple of ScopedInterpretationBinding
        The checked scoped contracts reachable from this entry, including
        calls from installed authored handler clauses.
    closed_interpretations : tuple of ClosedInterpretationBinding
        The reachable indexed lexical proof interpretations and their
        exact carrier and effect requirements.
    """

    entry: ProgramEntry = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    scoped_interpretations: tuple[ScopedInterpretationBinding, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    closed_interpretations: tuple[ClosedInterpretationBinding, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def inputs(self) -> tuple[ProgramParameter, ...]:
        """External data and observations, in checked parameter order.

        Returns
        -------
        tuple[ProgramParameter, ...]
            Typed external parameters in declaration order.
        """
        return tuple(
            p
            for p in self.entry.parameters
            if p.role in {"data", "index_map", "observation", "scalar"}
        )

    @property
    def observations(self) -> tuple[str, ...]:
        """The names of the values scored by observation steps.

        Returns
        -------
        tuple[str, ...]
            Observation names, excluding covariates and index maps.
        """
        return tuple(p.name for p in self.entry.parameters if p.role == "observation")

    @property
    def types(self) -> dict[str, TypeExpr]:
        """External symbols mapped to their original checked types.

        Returns
        -------
        dict[str, TypeExpr]
            The typed external input contract.
        """
        return {p.name: p.type for p in self.inputs}

    def describe(self) -> dict[str, dict[str, str]]:
        """Render the external input contract for display.

        Returns
        -------
        dict[str, dict[str, str]]
            Each input's role, readable type, and optional value restriction.
        """
        return {
            p.name: {
                "role": p.role,
                "type": render_static(p.type),
                **({"constraint": p.constraint} if p.constraint is not None else {}),
            }
            for p in self.inputs
        }


def _model_schema(
    source: Module,
    module: QiecModule,
    entry: ProgramEntry,
    root: ComputationId,
    closed: tuple[ClosedInterpretationBinding, ...],
) -> ModelSchema:
    """Select interpretation companions by checked execution reachability."""
    reached = _execution_reachability(module, root)[0] if closed else None
    return ModelSchema(
        entry=entry,
        scoped_interpretations=selected_scoped_interpretations(
            source, module, root, reached=reached
        ),
        closed_interpretations=(
            ()
            if reached is None
            else tuple(binding for binding in closed if binding.computation in reached)
        ),
    )


class InferenceMethod[Result](Protocol):
    """An inference implementation whose result type is chosen by the method.

    Implementations may return draws, messages, a MAP certificate, or another
    result. They must check their own execution and differentiation requirements.
    """

    def run(self, model: BoundModel) -> Result:
        """Run inference on a bound model.

        Parameters
        ----------
        model : BoundModel
            Checked program, data, input, and schema.

        Returns
        -------
        Result
            The method's own result type.
        """
        ...


class Model:
    """A reusable QVR source whose finite dimensions specialize at data binding.

    Parameters
    ----------
    source : str
        QVR source, optionally declaring objects as ``FinSet _``.
    entry : str or None
        Program to select. Defaults to the first exported program, or the
        only program when none is exported.
    families : FamilyCatalog or None
        Immutable native family definitions for this model's compilation.
    """

    def __init__(
        self,
        source: str,
        *,
        entry: str | None = None,
        families: FamilyCatalog | None = None,
    ) -> None:
        self.families = families
        self.source = source
        self.syntax: Module = parse(source)
        names = [s.name for s in self.syntax.statements if isinstance(s, ProgramDecl)]
        if entry is None:
            exported = [
                s.expr.name
                for s in self.syntax.statements
                if isinstance(s, ExportDecl)
                and isinstance(s.expr, ExprIdent)
                and s.expr.name in names
            ]
            if exported:
                entry = exported[0]
            elif len(names) == 1:
                entry = names[0]
            else:
                raise ValueError(f"select an entry from {names}")
        if entry not in names:
            raise ValueError(f"unknown program {entry!r}; choose from {names}")
        self.entry = entry
        artifacts = _lower_qvr_with_companions(self.syntax, families=families)
        checked = artifacts.module
        self.checked_module = checked
        self._checked_source = (self.syntax, checked, artifacts.closed_interpretations)
        self._checked_families = families
        typed_entry = next((e for e in checked.entries if e.name == entry), None)
        if typed_entry is None:
            raise CompileError(f"entry {entry!r} has no checked input contract")
        self.schema = _model_schema(
            self.syntax,
            checked,
            typed_entry,
            typed_entry.computation,
            artifacts.closed_interpretations,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        entry: str | None = None,
        families: FamilyCatalog | None = None,
    ) -> Model:
        """Read a QVR model from a file.

        Parameters
        ----------
        path : str or Path
            UTF-8 source file.
        entry : str or None
            Program to select.
        families : FamilyCatalog or None
            Immutable native family definitions for this compilation.

        Returns
        -------
        Model
            Parsed reusable model.
        """
        return cls(
            Path(path).read_text(encoding="utf-8"), entry=entry, families=families
        )

    def fit[Result](
        self,
        data: Mapping[str, object],
        *,
        method: InferenceMethod[Result],
        parameters: Mapping[str, object] | None = None,
        input: torch.Tensor | None = None,
        seed: int = 0,
        sample_prior: PriorMode = "no",
        num_prior_samples: int = 100,
        prior_seed: int = 1,
        snapshot_policy: SnapshotPolicy = "clone",
    ) -> Result:
        """Bind external data and apply an inference method in one call.

        Parameters
        ----------
        data : Mapping[str, object]
            External values, with roles inferred from QVR.
        method : InferenceMethod[Result]
            Inference configuration, including its own random seed and budgets.
        parameters : Mapping[str, object] or None
            Static specialization arguments passed to binding.
        input : torch.Tensor or None
            One-run domain input, if the program reads one.
        seed : int
            Compilation seed, independent of the method's inference seed.
        sample_prior : {"no", "yes", "only"}
            Ordinary fitting, ordinary fitting plus independent prior draws,
            or inference with observation likelihoods disabled. Explicit score
            potentials remain active in prior-only inference.
        num_prior_samples : int
            Positive independent draw budget for mode "yes".
        prior_seed : int
            Isolated prior stream, independent of the inference method.
        snapshot_policy : {"clone", "reference"}
            Refit-data ownership. Clone retains one independent training-data
            copy; reference explicitly shares later caller mutations.

        Returns
        -------
        Result
            The method's result with its own type and semantics.
        """
        return self.bind(data, parameters=parameters, input=input, seed=seed).infer(
            method,
            sample_prior=sample_prior,
            num_prior_samples=num_prior_samples,
            prior_seed=prior_seed,
            snapshot_policy=snapshot_policy,
        )

    def bind(
        self,
        data: Mapping[str, object],
        *,
        parameters: Mapping[str, object] | None = None,
        input: torch.Tensor | None = None,
        seed: int = 0,
    ) -> BoundModel:
        """Infer dimensions, compile, and validate one data binding.

        Parameters
        ----------
        data : Mapping[str, object]
            External data and observations. Roles and types come from QVR.
        parameters : Mapping[str, object] or None
            Explicit static arguments. Runtime scalars belong in ``data``.
        input : torch.Tensor or None
            Explicit one-run domain input. Required for a real-valued domain.
        seed : int
            Compilation seed; the caller's CPU random state is restored.

        Returns
        -------
        BoundModel
            Compiled entry and its checked binding.

        Raises
        ------
        CompileError
            If sizes cannot be inferred or the source cannot be checked.
        ValueError
            If input names, shapes, or required values disagree with the schema.
        TypeError
            If supplied element types disagree with their checked types.
        """
        return self._bind(
            data,
            parameters=parameters,
            input=input,
            seed=seed,
            require_observations=True,
        )

    def _bind(
        self,
        data: Mapping[str, object],
        *,
        parameters: Mapping[str, object] | None,
        input: torch.Tensor | None,
        seed: int,
        require_observations: bool,
    ) -> BoundModel:
        """Specialize one binding, optionally leaving responses to prediction."""
        values = dict(data)
        arguments = {} if parameters is None else dict(parameters)
        declaration = next(
            statement
            for statement in self.syntax.statements
            if isinstance(statement, ProgramDecl) and statement.name == self.entry
        )
        from quivers.dsl.program_templates import static_program_parameters

        static_names = {
            name
            for parameter in static_program_parameters(declaration)
            if isinstance(name := getattr(parameter, "name", None), str)
        }
        if extra := set(arguments) - static_names:
            raise ValueError(
                f"unknown static parameters: {sorted(extra)}; supply runtime values in data"
            )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            compiler = Compiler(self.syntax, families=self.families)
            if self.families is self._checked_families:
                compiler._reuse_checked_source(*self._checked_source)
            compiler.bind_data({**values, **arguments})
            module = compiler.compile()
            qiec = compiler.qiec_module
            if qiec is None:
                raise CompileError("the model has no checked input contract")
            entry = next((e for e in qiec.entries if e.name == self.entry), None)
            if entry is None:
                raise CompileError(
                    f"entry {self.entry!r} has no checked input contract: {qiec.gap or 'not elaborated'}"
                )
            declaration = next(
                s
                for s in self.syntax.statements
                if isinstance(s, ProgramDecl) and s.name == self.entry
            )
            if self.entry in compiler.morphisms:
                program = compiler.morphisms[self.entry]
            else:
                invoker = compiler._make_template_invoker(self.entry)
                template = compiler._program_templates[self.entry]
                application = {
                    parameter.name: arguments.get(
                        parameter.name, values.get(parameter.name)
                    )
                    for parameter in static_program_parameters(template)
                    if parameter.name in arguments or parameter.name in values
                }
                selected = invoker(**application)
                program = selected.morphism
        if not isinstance(program, (MonadicProgram, CheckedProgram)):
            raise TypeError(f"entry {self.entry!r} is not a probabilistic program")
        root = entry.computation
        if program.input_contract is not None:
            entry = program.input_contract
        schema = _model_schema(
            self.syntax, qiec, entry, root, compiler._closed_interpretations
        )
        expected = set(schema.types)
        required = (
            expected if require_observations else expected - set(schema.observations)
        )
        if missing := required - set(values):
            raise ValueError(f"missing external inputs: {sorted(missing)}")
        if extra := set(values) - expected:
            raise ValueError(
                f"unknown external inputs: {sorted(extra)}; expected {sorted(expected)}"
            )
        if input is None:
            if any(p.role == "domain" for p in entry.parameters) or isinstance(
                program.domain, ContinuousSpace
            ):
                raise ValueError(
                    "this entry reads a domain value; provide input explicitly"
                )
            input = torch.zeros(1, 1, dtype=torch.long)
        if input.ndim < 2 or input.shape[0] != 1:
            raise ValueError(
                "model fitting requires exactly one execution; observation rows belong to QVR plates"
            )
        check_host_data(program, {**values, **arguments})
        return BoundModel(
            module=module,
            program=program,
            schema=schema,
            data=values,
            input=input,
            specification=self,
            template_arguments=arguments,
            compilation_seed=seed,
        )


class BoundModel(ValueModel):
    """A compiled model with validated external values and one execution input.

    Parameters
    ----------
    specification : Model
        Original source and entry selection, retained by identity for rebinding.
    template_arguments : Mapping[str, object]
        Static arguments retained by identity for recompilation.
    compilation_seed : int
        Seed used to initialize deterministic parameters when binding.
    module : Program
        Compiled module container, held by identity.
    program : MonadicProgram or CheckedProgram
        Selected entry, held by identity.
    schema : ModelSchema
        Checked parameter and site metadata, compared by value.
    data : Mapping[str, object]
        External values, held by identity to preserve gradients and structured values.
    input : torch.Tensor
        One run's domain input, held by identity.
    """

    specification: Model = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    template_arguments: Mapping[str, object] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    compilation_seed: int
    module: Program = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    program: MonadicProgram | CheckedProgram = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    schema: ModelSchema = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    data: Mapping[str, object] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    input: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def rebind(
        self, data: Mapping[str, object], *, input: torch.Tensor | None = None
    ) -> BoundModel:
        """Bind new data while retaining the fitted parameter interpretation.

        Parameters
        ----------
        data : Mapping[str, object]
            Replacement covariates and optional responses. Inferred dimensions
            specialize again; explicit cardinalities remain fixed.
        input : torch.Tensor or None
            One execution input; the original input when omitted.

        Returns
        -------
        BoundModel
            Independent binding with copied deterministic parameter state.

        Raises
        ------
        ValueError
            If the new data change latent site names or shapes.
        """
        rebound = self.specification._bind(
            data,
            parameters=self.template_arguments,
            input=self.input if input is None else input,
            seed=self.compilation_seed,
            require_observations=False,
        )
        original_parameters = dict(self.require_tensor_program().named_parameters())
        replacement_parameters = dict(
            rebound.require_tensor_program().named_parameters()
        )
        if original_parameters.keys() != replacement_parameters.keys():
            raise ValueError(
                "new data change deterministic parameter names; refit the model"
            )
        with torch.no_grad():
            for name, parameter in replacement_parameters.items():
                original = original_parameters[name]
                if parameter.shape != original.shape:
                    raise ValueError(
                        f"new data change deterministic parameter {name!r}; refit the model"
                    )
                parameter.copy_(original)

        def shapes(bound: BoundModel) -> dict[str, tuple[int, ...]]:
            observations = set(bound.schema.observations)
            recorded = bound.trace()
            return {
                name: tuple(site.value.shape)
                for name, site in recorded.sites.items()
                if not site.is_deterministic
                and name.partition("@")[0] not in observations
            }

        # Ordinary models expose their latent layout without executing a prior
        # draw. Input-dependent scans and non-bijective supports use the trace
        # fallback, which checks both bindings with the same representation.
        try:
            layouts = [
                {
                    name: site.constrained_shape
                    for name, site in LatentRegistry.from_model(
                        bound.require_tensor_program(),
                        set(bound.schema.observations),
                    ).sites.items()
                }
                for bound in (self, rebound)
            ]
        except ValueError, NotImplementedError:
            with torch.random.fork_rng(devices=[]):
                layouts = [shapes(self), shapes(rebound)]
        if layouts[0] != layouts[1]:
            raise ValueError(
                "new data change the latent site layout; refit the model for that layout"
            )
        return rebound

    def infer[Result](
        self,
        method: InferenceMethod[Result],
        *,
        sample_prior: PriorMode = "no",
        num_prior_samples: int = 100,
        prior_seed: int = 1,
        snapshot_policy: SnapshotPolicy = "clone",
    ) -> Result:
        """Apply a method without imposing a posterior-draw result contract.

        Parameters
        ----------
        method : InferenceMethod[Result]
            Solver responsible for its capability checks and result metadata.
        sample_prior : {"no", "yes", "only"}
            Fit the posterior, additionally retain independent priors, or
            disable declared observation and marginalized likelihood scores.
        num_prior_samples : int
            Positive independent prior draw budget for mode "yes".
        prior_seed : int
            Isolated CPU stream for independent prior draws.
        snapshot_policy : {"clone", "reference"}
            Retained refit-data ownership for methods returning ModelFit.

        Returns
        -------
        Result
            The solver's result without conversion.

        Raises
        ------
        ValueError
            If the policy is invalid or global score potentials prevent
            independent prior sampling.
        TypeError
            If mode "yes" is requested from a solver not returning ModelFit.
        """
        from quivers.effects.observation_scoring import observation_scoring
        from quivers.inference.prior import _ancestral_program, _prior_mode

        mode = _prior_mode(sample_prior)
        if snapshot_policy not in {"clone", "reference"}:
            raise ValueError("snapshot_policy must be 'clone' or 'reference'")
        if mode == "yes":
            from quivers.inference._posterior import sample_count

            sample_count(num_prior_samples)
            _ancestral_program(self.require_tensor_program())
        with observation_scoring(mode != "only"):
            result = method.run(self)
        if isinstance(result, ModelFit):
            from quivers.modeling._refit import source_recipe

            prior = (
                self.prior_draws(num_samples=num_prior_samples, seed=prior_seed)
                if mode == "yes"
                else None
            )
            recipe = source_recipe(
                result.model,
                cast(InferenceMethod[ModelFit], method),
                snapshot_policy=snapshot_policy,
                sample_prior=mode,
                num_prior_samples=num_prior_samples,
                prior_seed=prior_seed,
            )
            return cast(
                Result,
                result.with_(sample_prior=mode, prior=prior, refit_recipe=recipe),
            )
        if mode == "yes":
            raise TypeError(
                "sample_prior='yes' requires an inference method returning ModelFit"
            )
        return result

    def prior_draws(
        self,
        *,
        num_samples: int = 100,
        seed: int = 0,
        method: Literal["ancestral"] | MCMC = "ancestral",
    ) -> PriorDraws | MCMCResult:
        """Draw retained conditional priors, or use explicit factor-conditioned MCMC.

        Parameters
        ----------
        num_samples : int
            Positive independent ancestral draw count.
        seed : int
            Isolated CPU random stream.
        method : {"ancestral"} or MCMC
            Independent sampling, or a configured driver for global potentials.

        Returns
        -------
        PriorDraws or MCMCResult
            Independent constrained draws or genuine factor-conditioned chains.
        """
        from quivers.inference.prior import prior_draws

        return prior_draws(
            self.require_tensor_program(),
            self.input,
            self.tensor_data(),
            num_samples=num_samples,
            seed=seed,
            method=method,
        )

    def sample(self, sampler: MCMC) -> ModelFit:
        """Fit using a configured MCMC driver, including its budgets and hooks.

        Parameters
        ----------
        sampler : MCMC
            Driver with kernel, chains, seed, adaptation, and optional checkpoint.

        Returns
        -------
        ModelFit
            Bound model and posterior draws.

        Raises
        ------
        TypeError
            If the selected entry cannot use the tensor inference runtime.
        """
        program = self.require_tensor_program()
        posterior = sampler.run(program, self.input, self.tensor_data())
        return ModelFit(model=self, posterior=posterior)

    def tensor_data(self) -> dict[str, torch.Tensor]:
        """Adapt external values to the native tensor executor's one-run layout.

        Scanned observations acquire a leading run axis through tensor views.
        The stored binding retains its checked shapes and original objects.

        Returns
        -------
        dict[str, torch.Tensor]
            Values for a custom tensor inference method or guide constructor.

        Raises
        ------
        TypeError
            If the entry requires reference execution or an input has no tensor form.
        """
        repeated = scanned_sites(self.require_tensor_program())
        values = {name: torch.as_tensor(value) for name, value in self.data.items()}
        for name in set(self.schema.observations) & repeated:
            values[name] = values[name].unsqueeze(0)
        return values

    def require_tensor_program(self) -> MonadicProgram:
        """Select the tensor runtime or explain its current capability limit.

        Returns
        -------
        MonadicProgram
            Tensor-executable program.

        Raises
        ------
        TypeError
            If only reference execution is available (enhancement #232).
        """
        if not isinstance(self.program, MonadicProgram):
            raise TypeError(
                "this checked entry cannot use tensor inference yet; see enhancement #232"
            )
        return self.program

    def trace(self, sites: Mapping[str, torch.Tensor] | None = None) -> Trace:
        """Trace one run with optional latent-site conditioning.

        Parameters
        ----------
        sites : Mapping[str, torch.Tensor] or None
            Latent values to condition on; external data come from the binding.
            A scanned position can be addressed as ``name@position``.

        Returns
        -------
        Trace
            The execution trace, including differentiable scores.

        Raises
        ------
        ValueError
            If conditioning names an external input or an unknown sample site.
        """
        supplied = {} if sites is None else dict(sites)
        program = self.require_tensor_program()
        repeated = scanned_sites(program)
        allowed = (set(once_sites(program)) | repeated) - set(self.schema.observations)
        unknown = set()
        for name in supplied:
            base, at, occurrence = name.rpartition("@")
            if name not in allowed and not (
                at and occurrence.isdigit() and base in allowed and base in repeated
            ):
                unknown.add(name)
        if unknown:
            raise ValueError(f"unknown latent sites: {sorted(unknown)}")
        return trace(program, self.input, {**self.tensor_data(), **supplied})


class ModelFit(ValueModel):
    """A bound model and fitted posterior, held by identity.

    Parameters
    ----------
    model : BoundModel
        Data, checked schema, and compiled model used by this fit.
    posterior : MCMCResult or PriorDraws or Guide
        Genuine MCMC chains, independent prior draws, or a variational guide.
        Live posterior objects are held by identity.
    sample_prior : {"no", "yes", "only"}
        Observation-scoring and retained-prior policy used for this fit.
    prior : PriorDraws or MCMCResult or None
        Retained independent priors or explicitly factor-conditioned chains.
    losses : tuple[float, ...]
        Optimization losses for a variational fit; empty for MCMC.
    refit_recipe : RefitRecipe or None
        Live fitting adapter and explicit training-data snapshot. Attached by
        BoundModel.infer; manually constructed results may omit it.
    """

    model: BoundModel = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    posterior: MCMCResult | PriorDraws | Guide = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    sample_prior: PriorMode = "no"
    prior: PriorDraws | MCMCResult | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    losses: tuple[float, ...] = ()
    refit_recipe: (
        RefitRecipe[Mapping[str, object], InferenceMethod[ModelFit], ModelFit] | None
    ) = dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)

    def update(
        self,
        data: Mapping[str, object] | None = None,
        *,
        method: InferenceMethod[ModelFit] | None = None,
        seed: int | None = None,
    ) -> ModelFit:
        """Refit retained or replacement data without changing this posterior.

        Parameters
        ----------
        data : Mapping[str, object] or None
            Replacement training data, or the recipe's retained snapshot.
        method : InferenceMethod[ModelFit] or None
            Explicit inference override, otherwise the retained method and seed.
        seed : int or None
            Compilation seed override. Inference seeds belong to the method.

        Returns
        -------
        ModelFit
            Fresh fitting result with the same source, entry, static arguments,
            domain input, and prior-scoring policy. New latent dimensions are
            permitted in a refit; prediction rebind remains stricter.

        Raises
        ------
        ValueError
            If this manually constructed result has no retained recipe.
        """
        if self.refit_recipe is None:
            raise ValueError(
                "this fit has no retained refit recipe; fit through Model.fit or BoundModel.infer"
            )
        return self.refit_recipe.run(data, method=method, seed=seed)

    def refit_callback(
        self,
        *,
        method_factory: Callable[
            [InferenceMethod[ModelFit], int], InferenceMethod[ModelFit]
        ],
        row_axis: str | None = None,
        partition: Callable[
            [Mapping[str, object], tuple[int, ...]], Mapping[str, object]
        ]
        | None = None,
        num_samples: int | None = None,
    ) -> Callable[[FoldSplit], FoldPrediction]:
        """Build an adapter for the shared explicit cross-validation runner.

        Parameters
        ----------
        method_factory : Callable[[InferenceMethod, int], InferenceMethod]
            Explicit fold-specific inference configuration. The seed comes
            from FoldSplit; return the method unchanged to retain its seed.
        row_axis : str or None
            Checked leading nominal axis to partition. Unrelated equal-size
            inputs remain constants. Supply this or an explicit partitioner.
        partition : Callable[[Mapping[str, object], tuple[int, ...]], Mapping[str, object]] or None
            Custom row selection when a single leading axis is insufficient.
        num_samples : int or None
            Per-chain draw budget passed to existing raw held-out scoring.

        Returns
        -------
        Callable[[FoldSplit], FoldPrediction]
            Fresh fitting and raw held-out scoring, preserving actual results.

        Raises
        ------
        ValueError
            If the recipe or a unique partition policy is missing.
        """
        from quivers.modeling._refit import partition_rows
        from quivers.modeling.refit import refit_callback

        if self.refit_recipe is None:
            raise ValueError("this fit has no retained refit recipe")
        if (row_axis is None) == (partition is None):
            raise ValueError("supply exactly one row_axis or explicit partition")
        partitioner = partition
        if partitioner is None:
            assert row_axis is not None
            axis = row_axis

            def select_rows(
                data: Mapping[str, object], rows: tuple[int, ...]
            ) -> Mapping[str, object]:
                return partition_rows(self.model, data, rows, axis)

            partitioner = select_rows

        return refit_callback(
            self.refit_recipe,
            partitioner,
            lambda fitted, data, seed: fitted.log_likelihood(
                data, num_samples=num_samples, seed=seed
            ),
            method_factory=method_factory,
        )

    def posterior_epred(
        self,
        data: Mapping[str, object] | None = None,
        *,
        input: torch.Tensor | None = None,
        sites: Sequence[str] | None = None,
        num_samples: int | None = None,
        mc_samples: int | None = None,
        seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Read native conditional response expectations at retained joint draws.

        Parameters
        ----------
        data : Mapping[str, object] or None
            New checked predictors and optional conditioning responses;
            None retains the original fitted binding.
        input : torch.Tensor or None
            One-run execution input; the fitted input by default.
        sites : Sequence[str] or None
            Response names, or None for all declared observations. Exact Scan
            occurrence labels are supported by the generic collector.
        num_samples : int or None
            Leading draws per recorded chain, or fresh variational draw count.
        mc_samples : int or None
            Explicit approximation budget for certified finite response means
            without an analytic expression.
        seed : int
            Isolated posterior and response seed; caller RNG is restored.

        Returns
        -------
        dict[str, torch.Tensor]
            Conditional means preserving chain/draw/checked observation axes.
            Categorical and ordinal responses append a probability axis.
            Upstream observed and lexical values retain their conditioning;
            this is not an analytic joint integration over other responses.

        Raises
        ------
        ValueError, TypeError
            If binding, latent geometry, selected sites, or response moments
            violate the existing checked prediction contract.
        """
        from quivers.inference.predictive import posterior_expectations

        bound = (
            self.model
            if data is None and input is None
            else self.model.rebind(
                self.model.data if data is None else data, input=input
            )
        )
        return posterior_expectations(
            bound.require_tensor_program(),
            self.posterior,
            bound.input,
            bound.tensor_data(),
            sites=tuple(
                site.name for site in bound.schema.entry.sites if site.kind == "observe"
            )
            or bound.schema.observations
            if sites is None
            else sites,
            num_samples=num_samples,
            mc_samples=mc_samples,
            seed=seed,
        )

    def conditional_effects(
        self,
        grid: PredictionGrid,
        *,
        site: str | None = None,
        num_samples: int | None = None,
        mc_samples: int | None = None,
        seed: int = 0,
    ) -> ConditionalEffects:
        """Evaluate expected source responses on one checked predictor grid.

        Parameters
        ----------
        grid : PredictionGrid
            Explicit row-axis effect and conditioning coordinates.
        site : str or None
            Declared observation; may be omitted when exactly one exists.
        num_samples : int or None
            Leading draws per recorded chain or fresh variational draw count.
        mc_samples : int or None
            Explicit finite-moment approximation budget when required.
        seed : int
            Isolated posterior and response random stream.

        Returns
        -------
        ConditionalEffects
            Joint expected responses with explicit conditional or empirical
            reduction. Source grouping inputs remain checked numeric maps;
            the fitted latent-layout guard prevents accidental new identities.

        Raises
        ------
        ValueError, TypeError
            If observation selection, row-axis identity, conditioning data,
            latent layout, or response moments are invalid.
        """
        from quivers.modeling._prediction import grid_data

        if site is None:
            observations = tuple(
                site.name
                for site in self.model.schema.entry.sites
                if site.kind == "observe"
            )
            if len(observations) != 1:
                raise ValueError(
                    "select site explicitly when the model has multiple observations"
                )
            site = observations[0]
        data = grid_data(self.model, grid, site)
        predictions = self.posterior_epred(
            data,
            sites=(site,),
            num_samples=num_samples,
            mc_samples=mc_samples,
            seed=seed,
        )[site]
        return grid.result(
            predictions,
            kind=draw_provenance(self.posterior),
        )

    def evidence(self, sampler: SMC) -> SMCResult:
        """Estimate the fitted target's normalizing constant with existing SMC.

        Parameters
        ----------
        sampler : SMC
            Configured likelihood-tempered driver, including its particle
            budget, rejuvenation kernel, and random stream.

        Returns
        -------
        SMCResult
            Independent particles and the estimated log normalizing constant.
            This is ordinary data evidence only for a normalized prior and
            likelihood. Observation weights instead define a powered target.

        Raises
        ------
        TypeError
            If the retained program has no tensor executor.
        ValueError
            If SMC cannot partition the target, normalize its prior, or use
            an exact likelihood without observation subsampling.
        RuntimeError
            If no temperature step meets the driver's ESS requirement.

        Notes
        -----
        The retained program, input, and data are reused. The fitted posterior
        is neither sampled nor modified; the driver's own guards and random
        stream govern this separate computation.
        """
        return sampler.run(
            self.model.require_tensor_program(),
            self.model.input,
            self.model.tensor_data(),
        )

    @property
    def samples(self) -> dict[str, torch.Tensor]:
        """Constrained draws with separate chain and draw axes.

        Returns
        -------
        dict[str, torch.Tensor]
            Recorded MCMC chains by latent site.

        Raises
        ------
        TypeError
            If the posterior is variational and has no recorded chains.
        """
        if not isinstance(self.posterior, MCMCResult):
            raise TypeError(
                "this fit has no recorded MCMC chains; use draws(num_samples=...)"
            )
        return self.posterior.samples

    def draws(
        self, *, num_samples: int | None = None, seed: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Draw a variational posterior, or return recorded MCMC chains.

        Parameters
        ----------
        num_samples : int or None
            Maximum draws per recorded chain, or fresh guide draws. Omission
            retains all MCMC draws or samples 100 guide draws.
        seed : int or None
            Optional CPU seed for guide sampling, restored on exit.

        Returns
        -------
        dict[str, torch.Tensor]
            Draws with leading chain and draw axes. Variational samples have
            one pseudo-chain; it cannot diagnose MCMC convergence.

        Raises
        ------
        ValueError
            If num_samples is not positive.
        """
        from quivers.inference._posterior import constrained_draws

        return constrained_draws(
            self.posterior,
            self.model.input,
            num_samples=num_samples,
            seed=seed,
        )

    def prior_view(
        self,
        *,
        num_samples: int | None = None,
        seed: int = 0,
        method: Literal["ancestral"] | MCMC = "ancestral",
    ) -> ModelFit:
        """Reuse prediction and analysis with retained or newly sampled prior draws.

        Parameters
        ----------
        num_samples : int or None
            Fresh independent count, or None to reuse retained priors. Without
            retained draws, None generates 100 independent draws.
        seed : int
            Isolated CPU stream for fresh ancestral samples.
        method : {"ancestral"} or MCMC
            Conditional sampling or explicit factor-conditioned inference.

        Returns
        -------
        ModelFit
            A view sharing the bound model and its original priors.
        """
        prior = self.prior
        if prior is None or num_samples is not None or method != "ancestral":
            prior = self.model.prior_draws(
                num_samples=100 if num_samples is None else num_samples,
                seed=seed,
                method=method,
            )
        return self.with_(posterior=prior, sample_prior="only", prior=prior, losses=())

    def predict(
        self,
        data: Mapping[str, object] | None = None,
        *,
        input: torch.Tensor | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample responses at the fitted data or a compatible new binding.

        Parameters
        ----------
        data : Mapping[str, object] or None
            New covariates and optional responses; fitted data when omitted.
            Observation values are released before predictive sampling.
        input : torch.Tensor or None
            Execution input for new data; the fitted input when omitted.
        num_samples : int or None
            Maximum posterior draws; all available MCMC draws when omitted.
        seed : int or None
            Optional CPU random seed, restored on exit.

        Returns
        -------
        dict[str, torch.Tensor]
            Predictive samples by site at the requested data layout.

        Raises
        ------
        ValueError
            If new data change the latent parameter layout.
        """
        context = (
            torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
        )
        with context:
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            bound = (
                self.model
                if data is None and input is None
                else self.model.rebind(
                    self.model.data if data is None else data, input=input
                )
            )
            covariates = {
                name: value
                for name, value in bound.tensor_data().items()
                if name not in bound.schema.observations
            }
            predictive = Predictive(
                bound.require_tensor_program(), self.posterior, num_samples=num_samples
            )
            predicted = predictive(bound.input, observations=covariates)
            for name in bound.schema.observations:
                if name not in predicted:
                    continue
                split = tensor_shape(bound.schema.types[name])
                dimensions = () if split is None else split[1]
                if all(
                    isinstance(dimension, IndexLiteral)
                    and isinstance(dimension.value, int)
                    for dimension in dimensions
                ):
                    shape = tuple(dimension.value for dimension in dimensions)
                    predicted[name] = predicted[name].reshape(
                        predicted[name].shape[0], *shape
                    )
            return predicted

    def log_likelihood(
        self,
        data: Mapping[str, object] | None = None,
        *,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate observation densities separately at posterior draws.

        Parameters
        ----------
        data : Mapping[str, object] or None
            Replacement covariates and observed responses; fitted data by default.
        num_samples : int or None
            Maximum draws per MCMC chain; 100 draws for a guide when omitted.
        seed : int or None
            Optional seed for variational draws and participating native streams, restored on exit.

        Returns
        -------
        dict[str, torch.Tensor]
            Raw observation log densities with leading chain and draw axes.
            Event dimensions are reduced; observation plate axes are retained.

        Raises
        ------
        ValueError
            If responses are missing or the draw count is not positive.
        UnfactoredObservationError
            If the rows of a marginalized observation share a latent, so they
            have no scores of their own. The message says which program gives
            each row its own; `unit_log_likelihood` reads the joint scores.
        """
        bound = self.model if data is None else self.model.rebind(data)
        return dict(
            source_analysis(
                bound, self.posterior, num_samples=num_samples, seed=seed
            ).log_likelihood
        )

    def unit_log_likelihood(
        self,
        data: Mapping[str, object] | None = None,
        *,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> dict[str, UnitLogLikelihood]:
        """Evaluate the density of each independent unit at posterior draws.

        The units are the rows of an observation with a density of its own,
        the units of a grouped ``marginalize`` block, and the whole of a block
        whose rows share one latent. Given a draw, the responses' joint log
        density is the sum over the units.

        Parameters
        ----------
        data : Mapping[str, object] or None
            Replacement covariates and observed responses; fitted data by default.
        num_samples : int or None
            Maximum draws per MCMC chain; 100 draws for a guide when omitted.
        seed : int or None
            Optional seed for variational draws and participating native streams, restored on exit.

        Returns
        -------
        dict[str, UnitLogLikelihood]
            One entry per set of responses whose rows share units, keyed by
            their names joined by commas.

        Raises
        ------
        ValueError
            If responses are missing, the draw count is not positive, or the
            units change between draws.
        TypeError
            If the weights of a block's units are not log densities, or nothing
            scores an observation.
        """
        bound = self.model if data is None else self.model.rebind(data)
        return source_unit_log_likelihood(
            bound, self.posterior, num_samples=num_samples, seed=seed
        )

    def analysis(
        self,
        *,
        num_samples: int | None = None,
        seed: int = 0,
        units: ScoreUnits = "rows",
    ) -> PosteriorAnalysis:
        """Prepare reusable tensor posterior analysis on the fitted data.

        Parameters
        ----------
        num_samples : int or None
            Maximum recorded draws per chain, or fresh variational draws.
        seed : int
            Guide sampling seed; participating caller RNG streams are preserved.
        units : ScoreUnits
            ``"rows"`` for one score per response row, and ``"factors"`` for
            one per independent unit of the likelihood, which
            `unit_log_likelihood` describes.

        Returns
        -------
        quivers.diagnostics.PosteriorAnalysis
            Posterior draws, raw observation scores, observed responses,
            and recorded sampler statistics on their original axes.

        Raises
        ------
        UnfactoredObservationError
            If scores per row are asked for and the rows of a marginalized
            observation share a latent.
        ValueError
            If fitted response data are missing or the draw count is invalid.
        """
        return source_analysis(
            self.model, self.posterior, num_samples=num_samples, seed=seed, units=units
        )

    def summary(self) -> dict[str, ConvergenceSummary]:
        """Summarize MCMC chains with the package's convergence estimators.

        Returns
        -------
        dict[str, ConvergenceSummary]
            Means, quantiles, R-hat, ESS, and Monte Carlo errors by site.

        Raises
        ------
        TypeError
            If this is a variational fit, which has no MCMC chains.
        """
        return summary(self.samples)


__all__ = ["BoundModel", "InferenceMethod", "Model", "ModelFit", "ModelSchema"]
