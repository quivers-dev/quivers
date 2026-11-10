"""User-facing `fit` entry point and the
`BayesianFit` result wrapper.

The compilation path is fully AST-driven: the formula lens emits a
[`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module], the existing
[`quivers.dsl.compiler.Compiler`][quivers.dsl.compiler.Compiler] consumes it directly (no
source-string round-trip), and inference runs on the resulting
[`quivers.continuous.programs.MonadicProgram`][quivers.continuous.programs.MonadicProgram].  Source text
is generated only when the user requests it via
`formula_to_qvr` or `BayesianFit.dump_qvr`, in which
case [`quivers.dsl.emit.module_to_source`][quivers.dsl.emit.module_to_source] walks the same AST
to produce canonical ``.qvr`` source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import didactic.api as dx
import narwhals as nw
import torch
from torch import nn
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.dsl.emit import module_to_source
from quivers.formulas.compile import FormulaToQVRModule
from quivers.formulas.family import Family, families
from quivers.formulas.formula import Formula, _qvr_name, formula_from_data
from quivers.inference import (
    ELBO,
    HMCKernel,
    MCMC,
    NUTSKernel,
    SVI,
    AutoNormalGuide,
)
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult


class BayesianFit(ValueModel):
    """A fitted Bayesian regression: the compiled program, the
    parsed formula, the family, the user-supplied data, and the
    posterior samples.

    Attributes
    ----------
    formula : Formula
        Parsed formula IR.
    family : Family
        Response family used at compile time.
    program : MonadicProgram
        The compiled program.
    posterior : MCMCResult or Guide
        [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult] from NUTS / HMC, or
        a [`quivers.inference.guides.base.Guide`][quivers.inference.guides.base.Guide] from SVI.
    observations : Mapping[str, torch.Tensor]
        Inference-time observations dict (response + per-column
        covariates + per-group plate indices).

    Fits compare by value over the formula, family, and settings, and
    by identity over the live objects: the program, the posterior, the
    observations mapping, the predictor, and its data. Two fits are
    thus equal exactly when they share those objects and agree on
    everything else.
    """

    formula: Formula = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    family: Family
    program: MonadicProgram = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    posterior: MCMCResult | Guide = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    observations: Mapping[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    reparameterize: Literal["centered", "noncentered"] = "noncentered"
    binomial_trials: int | str = 1
    thresholds_by: str | None = None
    mixture_components: int = 2
    fixed_prior: str = "Normal(0.0, 5.0)"
    random_scale_prior: str = "HalfNormal(1.0)"
    priors: dict[str, str] = dx.field(default_factory=dict)
    predictor: nn.Module | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    predictor_data: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    predictor_name: str | None = None

    @property
    def qvr_source(self) -> str:
        """Lazily emit the AST-equivalent ``.qvr`` source for display."""
        lens = FormulaToQVRModule(
            self.family,
            fixed_prior=self.fixed_prior,
            random_scale_prior=self.random_scale_prior,
            user_priors=self.priors,
            reparameterize=self.reparameterize,
            binomial_trials=self.binomial_trials,
            thresholds_by=self.thresholds_by,
            mixture_components=self.mixture_components,
            predictor_name=self.predictor_name,
        )
        module, _ = lens.forward(self.formula)
        return module_to_source(module)

    def dump_qvr(self, path: str | Path) -> Path:
        """Write the AST-equivalent ``.qvr`` source to ``path`` and
        return the resulting `Path`.
        """
        out = Path(path)
        out.write_text(self.qvr_source)
        return out


def fit(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    method: Literal["nuts", "hmc", "svi"] = "nuts",
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 4,
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    priors: Mapping[str, str] | None = None,
    guide: type | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str = 1,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor: nn.Module | None = None,
    predictor_data: torch.Tensor | None = None,
    seed: int = 0,
) -> BayesianFit:
    """Compile a brms-style formula, fit it, and return the result.

    See [`quivers.formulas`][quivers.formulas] for surface details.  This entry
    point composes `formula_from_data`, `FormulaToQVRModule`,
    `Compiler`, and the inference layer in one call.

    ``family="categorical"`` infers contiguous labels ``0..K-1`` and uses
    label zero as the reference. An attached categorical predictor may return
    ``(N, K-1)`` reference logits or ``(N, K)`` full logits. For
    ``family="mixture"``, ``mixture_components`` selects any finite component
    count of at least two; the likelihood integrates assignments exactly.

    Parameters
    ----------
    formula : str
        brms/lme4-style response formula.
    data : IntoDataFrame
        Pandas, Polars, or another Narwhals-compatible dataframe.
    family : str or Family
        Registered response family or a family value.
    method : {"nuts", "hmc", "svi"}
        Inference algorithm.
    mixture_components : int
        Number of Gaussian components for the mixture family.
    predictor : nn.Module or None
        Optional differentiable contribution on the linear-predictor scale.
    predictor_data : torch.Tensor or None
        Input passed to ``predictor`` on each SVI step.

    Returns
    -------
    BayesianFit
        Parsed formula, compiled program, fitted posterior, and observations.
    """
    if isinstance(family, str):
        if family not in families:
            raise ValueError(
                f"fit: unknown family {family!r}; choices are {sorted(families)}"
            )
        family_obj = families[family]
    else:
        family_obj = family

    parsed = formula_from_data(formula, data)
    predictor_name = "neural_eta" if predictor is not None else None
    if predictor is not None and predictor_data is None:
        raise ValueError("fit: predictor_data is required when predictor is supplied")
    if predictor is None and predictor_data is not None:
        raise ValueError("fit: predictor_data requires predictor")
    if (
        predictor is not None
        and method != "svi"
        and any(parameter.requires_grad for parameter in predictor.parameters())
    ):
        raise ValueError(
            "fit: a trainable external predictor requires method='svi'; "
            "freeze its parameters before using NUTS or HMC"
        )
    lens = FormulaToQVRModule(
        family_obj,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        user_priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
    )
    module, _ = lens.forward(parsed)
    compiler = Compiler(module)
    program_runtime = compiler.compile()
    morphism = program_runtime.morphism
    if not isinstance(morphism, MonadicProgram):
        raise TypeError(
            f"fit: compiled morphism has type "
            f"{type(morphism).__name__}, expected MonadicProgram"
        )
    program = morphism

    observations: dict[str, torch.Tensor] = {}
    observations.update(lens.fixed_column_observations(parsed))
    response_name = parsed.response_name
    observations[response_name] = torch.as_tensor(
        parsed.response_values.copy(), dtype=torch.float32
    ).reshape(-1)
    for group, codes in parsed.group_indices.items():
        observations[f"{_qvr_name(group)}_idx"] = torch.as_tensor(
            list(codes), dtype=torch.long
        )
    if family_obj.name == "binomial":
        trials = _binomial_trial_tensor(
            data, binomial_trials, n_obs=parsed.response_values.shape[0]
        )
        response = observations[response_name]
        _validate_binomial_response(response, trials, caller="fit")
        if isinstance(binomial_trials, str):
            observations[_qvr_name(binomial_trials)] = trials
    if predictor is not None:
        assert predictor_data is not None
        assert predictor_name is not None
        predictor_categories = (
            FormulaToQVRModule._category_count(parsed, family="categorical")
            if family_obj.name == "categorical"
            else None
        )
        observations[predictor_name] = _predictor_output(
            predictor,
            predictor_data,
            n_obs=parsed.response_values.shape[0],
            n_categories=predictor_categories,
        ).detach()
    else:
        predictor_categories = None

    torch.manual_seed(seed)
    if method == "svi":
        posterior = _fit_svi(
            program,
            observations,
            num_samples,
            guide_cls=guide,
            predictor=predictor,
            predictor_data=predictor_data,
            predictor_name=predictor_name,
            predictor_categories=predictor_categories,
        )
    else:
        posterior = _fit_mcmc(
            program,
            observations,
            sampler=method,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )

    return BayesianFit(
        formula=parsed,
        family=family_obj,
        program=program,
        posterior=posterior,
        observations=observations,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        priors=dict(priors or {}),
        predictor=predictor,
        predictor_data=predictor_data,
        predictor_name=predictor_name,
    )


def formula_to_qvr(
    formula: str,
    *,
    data: IntoDataFrame,
    family: str | Family = "gaussian",
    fixed_prior: str = "Normal(0.0, 5.0)",
    random_scale_prior: str = "HalfNormal(1.0)",
    priors: Mapping[str, str] | None = None,
    reparameterize: Literal["centered", "noncentered"] = "noncentered",
    binomial_trials: int | str = 1,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    predictor_name: str | None = None,
    path: str | Path | None = None,
) -> str:
    """Emit ``.qvr`` source for a brms-style formula without fitting.

    Builds the formula AST → QVR Module via `FormulaToQVRModule`,
    then serialises the module via [`quivers.dsl.emit.module_to_source`][quivers.dsl.emit.module_to_source].
    Optionally writes the result to ``path``. ``mixture_components`` selects
    any integer component count of at least two for ``family="mixture"``.
    """
    if isinstance(family, str):
        if family not in families:
            raise ValueError(
                f"formula_to_qvr: unknown family {family!r}; choices are "
                f"{sorted(families)}"
            )
        family_obj = families[family]
    else:
        family_obj = family
    parsed = formula_from_data(formula, data)
    if family_obj.name == "binomial":
        trials = _binomial_trial_tensor(
            data, binomial_trials, n_obs=parsed.response_values.shape[0]
        )
        response = torch.as_tensor(parsed.response_values.copy())
        _validate_binomial_response(response, trials, caller="formula_to_qvr")
    lens = FormulaToQVRModule(
        family_obj,
        fixed_prior=fixed_prior,
        random_scale_prior=random_scale_prior,
        user_priors=priors,
        reparameterize=reparameterize,
        binomial_trials=binomial_trials,
        thresholds_by=thresholds_by,
        mixture_components=mixture_components,
        predictor_name=predictor_name,
    )
    module, _ = lens.forward(parsed)
    source = module_to_source(module)
    if path is not None:
        Path(path).write_text(source)
    return source


def _fit_mcmc(program, observations, *, sampler, num_warmup, num_samples, num_chains):
    """Run NUTS / HMC on the compiled program."""
    kernel = NUTSKernel() if sampler == "nuts" else HMCKernel()
    mcmc = MCMC(
        kernel=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    n_obs = int(observations[next(iter(observations))].shape[0])
    x = torch.zeros(n_obs, 1, dtype=torch.long)
    return mcmc.run(program, x, observations)


def _fit_svi(
    program,
    observations,
    num_steps,
    *,
    guide_cls=None,
    predictor: nn.Module | None = None,
    predictor_data: torch.Tensor | None = None,
    predictor_name: str | None = None,
    predictor_categories: int | None = None,
):
    """Run an SVI fit + ELBO.

    Default guide is `AutoNormalGuide`: a mean-field
    diagonal-Normal that scales well across model shapes. Mean-field
    is known to underestimate posterior variance components in
    hierarchical / mixed-effects models; for serious analysis of
    those models, use ``method="nuts"``. Users can swap in any other
    ``Guide`` class via ``fit(..., guide=SomeGuide)``; the class is
    constructed with ``(program, observed_names=...)``.
    """
    if guide_cls is None:
        guide_cls = AutoNormalGuide
    guide = guide_cls(program, observed_names=set(observations.keys()))
    parameters = list(program.parameters()) + list(guide.parameters())
    if predictor is not None:
        parameters.extend(predictor.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    svi = SVI(program, guide, optimizer, ELBO())
    n_obs = int(observations[next(iter(observations))].shape[0])
    x = torch.zeros(n_obs, 1, dtype=torch.long)
    for _ in range(num_steps):
        step_observations = observations
        if predictor is not None:
            assert predictor_data is not None and predictor_name is not None
            step_observations = dict(observations)
            step_observations[predictor_name] = _predictor_output(
                predictor,
                predictor_data,
                n_obs=n_obs,
                n_categories=predictor_categories,
            )
        svi.step(x, step_observations)
    if predictor is not None:
        assert predictor_data is not None and predictor_name is not None
        observations[predictor_name] = _predictor_output(
            predictor,
            predictor_data,
            n_obs=n_obs,
            n_categories=predictor_categories,
        ).detach()
    return guide


def _binomial_trial_tensor(
    data: IntoDataFrame,
    trials: int | str,
    *,
    n_obs: int,
) -> torch.Tensor:
    if isinstance(trials, int):
        if trials < 1:
            raise ValueError("binomial_trials must be at least one")
        return torch.full((n_obs,), float(trials))
    frame = nw.from_native(data, eager_only=True)
    if trials not in frame.columns:
        raise ValueError(f"binomial_trials column {trials!r} is not in the data")
    values = torch.as_tensor(
        frame[trials].to_numpy().copy(), dtype=torch.float32
    ).reshape(-1)
    if values.shape[0] != n_obs:
        raise ValueError(
            f"binomial_trials column has {values.shape[0]} rows, expected {n_obs}"
        )
    if not torch.isfinite(values).all() or torch.any(values < 1):
        raise ValueError("binomial_trials must contain positive finite integers")
    if not torch.equal(values, values.round()):
        raise ValueError("binomial_trials must contain integers")
    return values


def _predictor_output(
    predictor: nn.Module,
    data: torch.Tensor,
    *,
    n_obs: int,
    n_categories: int | None = None,
) -> torch.Tensor:
    value = predictor(data)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "fit: predictor must return a torch.Tensor on the linear-predictor scale"
        )
    if n_categories is not None:
        if value.shape == (n_obs,) and n_categories == 2:
            return value
        if value.shape == (n_obs, n_categories - 1):
            return value.reshape(-1)
        if value.shape == (n_obs, n_categories):
            return (value[:, 1:] - value[:, :1]).reshape(-1)
        raise ValueError(
            "fit: categorical predictor returned shape "
            f"{tuple(value.shape)}, expected ({n_obs}, {n_categories - 1}) "
            f"or ({n_obs}, {n_categories})"
        )
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.shape != (n_obs,):
        raise ValueError(
            f"fit: predictor returned shape {tuple(value.shape)}, expected ({n_obs},)"
        )
    return value


def _validate_binomial_response(
    response: torch.Tensor,
    trials: torch.Tensor,
    *,
    caller: str,
) -> None:
    if not torch.isfinite(response).all() or not torch.equal(
        response, response.round()
    ):
        raise ValueError(f"{caller}: binomial responses must be finite integers")
    if torch.any(response < 0) or torch.any(response > trials):
        raise ValueError(
            f"{caller}: binomial responses must lie between zero and "
            "binomial_trials for every row"
        )
