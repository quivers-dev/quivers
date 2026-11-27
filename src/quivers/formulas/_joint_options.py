"""Keyword contracts of the `fit` entry point for its two formula kinds.

`fit` accepts a single-response formula string or a `MultivariateFormula`.
These typed keyword sets give its two overloads exact result types without a
second fitting function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict

import torch
from torch import nn

from quivers.formulas._autocor import AutocorMatrix
from quivers.formulas._fit_controls import (
    Algorithm,
    FileRefit,
    InitSpec,
    SamplerControl,
)

# Pathfinder variational inference.
from quivers.formulas._fit_controls import FitMethod, PathfinderOptions
from quivers.formulas._data_options import Knots
from quivers.formulas._group_covariance import GroupCovariance
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas._priors import PriorInput
from quivers.formulas.family import Family
from quivers.inference.prior import PriorMode
from quivers.modeling.refit import SnapshotPolicy


class JointFitOptions(TypedDict, total=False):
    """Keywords of `fit` that apply to a joint formula.

    Each key has the meaning of the `fit` keyword of the same name.
    Response-level options (family, distributional and nonlinear formulas,
    weights, trials, thresholds) belong to each response's `bf` call.
    ``data2`` holds named matrices shared by every response, as in brms.
    """

    data2: Mapping[str, AutocorMatrix] | None
    method: FitMethod
    num_warmup: int | None
    num_samples: int | None
    num_chains: int | None
    num_steps: int | None
    learning_rate: float
    fixed_prior: str
    random_scale_prior: str
    correlation_prior: str
    group_covariances: Mapping[str, GroupCovariance] | None
    priors: PriorInput | None
    guide: type | None
    initial_values: Mapping[str, torch.Tensor | float] | None
    reparameterize: Literal["centered", "noncentered"]
    seed: int
    sample_prior: PriorMode
    num_prior_samples: int
    prior_seed: int | None
    snapshot_policy: SnapshotPolicy
    # brms default priors and fitting controls.
    default_priors: Literal["quivers", "brms"]
    control: SamplerControl | Mapping[str, float | int | str] | None
    thin: int
    init: InitSpec | None
    init_r: float
    algorithm: Algorithm | None
    file: str | Path | None
    file_refit: FileRefit
    pathfinder: PathfinderOptions | Mapping[str, float | int | bool] | None
    cores: int
    knots: Knots | None
    drop_unused_levels: bool


class FormulaFitOptions(JointFitOptions, total=False):
    """Keywords of `fit` for a single-response formula.

    Each key has the meaning of the `fit` keyword of the same name.
    """

    family: str | Family
    parameters: Mapping[str, str | NonlinearExpression] | None
    weights: str | None
    decision: str | None
    distributional: Mapping[str, str | NonlinearExpression] | None
    nonlinear: Mapping[str, str | NonlinearExpression] | None
    fixed_parameters: Mapping[str, float] | None
    binomial_trials: int | str | None
    thresholds_by: str | None
    mixture_components: int
    predictor: nn.Module | None
    predictor_data: torch.Tensor | None
    # brms bf(center, decomp) for a single response.
    center: bool | None
    decomp: Literal["none", "QR"]
    cmc: bool
    sparse: bool
    unused: Sequence[str]


__all__: list[str] = []
