"""Compile, bind, and inspect QVR models through checked input metadata."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous import ContinuousSpace, MonadicProgram
from quivers.dsl import Compiler, CompileError, Module, parse, lower_qvr_to_qiec
from quivers.dsl.ast_nodes import ProgramDecl
from quivers.effects import CheckedProgram
from quivers.effects.interpreter import check_host_data
from quivers.inference import Guide, MCMC, MCMCResult, Predictive, Trace, trace
from quivers.diagnostics import ConvergenceSummary, summary
from quivers.program import Program
from quivers.qiec import ProgramEntry, ProgramParameter, TypeExpr, render_static


class ModelSchema(ValueModel):
    """The checked input contract of one QVR program.

    Parameters
    ----------
    entry : ProgramEntry
        Typed parameters, site roles, plates, and static binders. The entry
        is immutable and compared by value, without rendering away its types.
    """

    entry: ProgramEntry = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

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
            Each input's role and readable type.
        """
        return {
            p.name: {"role": p.role, "type": render_static(p.type)} for p in self.inputs
        }


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


class BoundModel(ValueModel):
    """A compiled model with validated external values and one execution input.

    Parameters
    ----------
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

    module: Program = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    program: MonadicProgram | CheckedProgram = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    schema: ModelSchema = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    data: Mapping[str, object] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    input: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def infer[Result](self, method: InferenceMethod[Result]) -> Result:
        """Apply a method without imposing a posterior-draw result contract.

        Parameters
        ----------
        method : InferenceMethod[Result]
            Solver responsible for its capability checks and result metadata.

        Returns
        -------
        Result
            The solver's result without conversion.
        """
        return method.run(self)

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
        posterior = sampler.run(program, self.input, dict(self.data))
        return ModelFit(model=self, posterior=posterior)

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
        allowed = {s.name for s in self.schema.entry.sites if s.kind == "sample"}
        if unknown := set(supplied) - allowed:
            raise ValueError(f"unknown latent sites: {sorted(unknown)}")
        return trace(
            self.require_tensor_program(), self.input, {**self.data, **supplied}
        )


class ModelFit(ValueModel):
    """A bound model and fitted posterior, held by identity.

    Parameters
    ----------
    model : BoundModel
        Data, checked schema, and compiled model used by this fit.
    posterior : MCMCResult or Guide
        MCMC draws and diagnostics, or a variational guide, held by identity.
    losses : tuple[float, ...]
        Optimization losses for a variational fit; empty for MCMC.
    """

    model: BoundModel = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    posterior: MCMCResult | Guide = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    losses: tuple[float, ...] = ()

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
                "a variational posterior has no recorded chains; use draws(num_samples=...)"
            )
        return self.posterior.samples

    def draws(self, *, num_samples: int = 100) -> dict[str, torch.Tensor]:
        """Draw a variational posterior, or return recorded MCMC chains.

        Parameters
        ----------
        num_samples : int
            Number of fresh guide draws. Ignored for recorded MCMC chains.

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
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if isinstance(self.posterior, MCMCResult):
            return self.samples
        with torch.no_grad():
            draws = [
                self.posterior.rsample(self.model.input) for _ in range(num_samples)
            ]
        return {
            name: torch.stack(
                [
                    draw[name].reshape(
                        self.posterior.registry.sites[name].constrained_shape
                    )
                    for draw in draws
                ]
            ).unsqueeze(0)
            for name in draws[0]
        }

    def predict(self, *, num_samples: int | None = None) -> dict[str, torch.Tensor]:
        """Sample observations conditional on fitted latents and external covariates.

        Parameters
        ----------
        num_samples : int or None
            Maximum posterior draws; all available draws when omitted.

        Returns
        -------
        dict[str, torch.Tensor]
            Predictive samples by site. Observations are released while
            covariates and index maps remain bound.
        """
        data = {
            name: value
            for name, value in self.model.data.items()
            if name not in self.model.schema.observations
        }
        predictive = Predictive(
            self.model.require_tensor_program(), self.posterior, num_samples=num_samples
        )
        return predictive(self.model.input, observations=data)

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


class Model:
    """A reusable QVR source whose finite dimensions specialize at data binding.

    Parameters
    ----------
    source : str
        QVR source, optionally declaring objects as ``FinSet _``.
    entry : str or None
        Program to select; a module with several programs requires a name.
    """

    def __init__(self, source: str, *, entry: str | None = None) -> None:
        self.source = source
        self.syntax: Module = parse(source)
        names = [s.name for s in self.syntax.statements if isinstance(s, ProgramDecl)]
        if entry is None:
            if len(names) != 1:
                raise ValueError(f"select an entry from {names}")
            entry = names[0]
        if entry not in names:
            raise ValueError(f"unknown program {entry!r}; choose from {names}")
        self.entry = entry
        checked = lower_qvr_to_qiec(self.syntax)
        typed_entry = next((e for e in checked.entries if e.name == entry), None)
        if typed_entry is None:
            raise CompileError(f"entry {entry!r} has no checked input contract")
        self.schema = ModelSchema(entry=typed_entry)

    @classmethod
    def from_file(cls, path: str | Path, *, entry: str | None = None) -> Model:
        """Read a QVR model from a file.

        Parameters
        ----------
        path : str or Path
            UTF-8 source file.
        entry : str or None
            Program to select.

        Returns
        -------
        Model
            Parsed reusable model.
        """
        return cls(Path(path).read_text(encoding="utf-8"), entry=entry)

    def fit[Result](
        self,
        data: Mapping[str, object],
        *,
        method: InferenceMethod[Result],
        parameters: Mapping[str, object] | None = None,
        input: torch.Tensor | None = None,
        seed: int = 0,
    ) -> Result:
        """Bind external data and apply an inference method in one call.

        Parameters
        ----------
        data : Mapping[str, object]
            External values, with roles inferred from QVR.
        method : InferenceMethod[Result]
            Inference configuration, including its own random seed and budgets.
        parameters : Mapping[str, object] or None
            Template arguments passed to binding.
        input : torch.Tensor or None
            One-run domain input, if the program reads one.
        seed : int
            Compilation seed, independent of the method's inference seed.

        Returns
        -------
        Result
            The method's result with its own type and semantics.
        """
        return self.bind(data, parameters=parameters, input=input, seed=seed).infer(
            method
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
            Arguments of a scalar-parameterized program, such as LDA's priors.
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
        values = dict(data)
        arguments = {} if parameters is None else dict(parameters)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            compiler = Compiler(self.syntax)
            compiler.bind_data(values)
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
            if declaration.type_params is not None:
                invoker = compiler._make_template_invoker(self.entry)
                selected = invoker(**arguments)
                program = selected.morphism
            else:
                if arguments:
                    raise ValueError("this entry has no template parameters")
                program = compiler.morphisms[self.entry]
        if not isinstance(program, (MonadicProgram, CheckedProgram)):
            raise TypeError(f"entry {self.entry!r} is not a probabilistic program")
        if program.input_contract is not None:
            entry = program.input_contract
        schema = ModelSchema(entry=entry)
        expected = set(schema.types) - set(arguments)
        if missing := expected - set(values):
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
            module=module, program=program, schema=schema, data=values, input=input
        )


__all__ = ["BoundModel", "InferenceMethod", "Model", "ModelFit", "ModelSchema"]
