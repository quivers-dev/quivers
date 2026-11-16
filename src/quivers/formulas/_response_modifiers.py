"""Typed row modifiers shared by formula parsing and response adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import didactic.api as dx
import torch
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.formulas._group_prediction import NewLevelPolicy

if TYPE_CHECKING:
    from quivers.formulas.formula import Formula


class PredictionOptions(TypedDict, total=False):
    """Optional prediction keywords shared by native and decoded response draws.

    Each key follows `BayesianFit.posterior_predict`: retained predictor
    encodings, group policies, trial counts, draw count, and random seed.
    Omitted keys retain that method's defaults.
    """

    new_data: IntoDataFrame | None
    predictor_data: torch.Tensor | None
    trials: int | str | None
    re_formula: str | None
    allow_new_levels: bool
    sample_new_levels: NewLevelPolicy
    num_samples: int | None
    seed: int


class ExpectationOptions(PredictionOptions, total=False):
    """Optional expectation keywords shared by response and decision means.

    Prediction keys follow `BayesianFit.posterior_epred`. The additional
    mc_samples key requests Monte Carlo moments when no closed mean exists.
    Decision probabilities always use their closed joint-model probabilities.
    """

    mc_samples: int | None


class ResponseDraws(ValueModel):
    """Posterior responses decoded into their original measurement coordinates.

    Parameters
    ----------
    value : torch.Tensor
        Response draws preserving chain, draw, row, and event dimensions. Wiener
        responses are positive reaction times, including nondecision delay.
    decision : torch.Tensor or None
        Joint binary decisions on the same sample axes, zero lower and one
        upper; None for a response without a decision modifier.
    decision_labels : tuple[str, ...]
        Semantic labels of the binary codes, or an empty tuple.
    """

    value: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    decision: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    decision_labels: tuple[str, ...] = ()


def decode_response_draws(formula: Formula, draws: torch.Tensor) -> ResponseDraws:
    """Decode a lossless native response tensor without another posterior replay.

    Parameters
    ----------
    formula : Formula
        Response measurement and modifier metadata.
    draws : torch.Tensor
        Native signed or ordinary response draws.

    Returns
    -------
    ResponseDraws
        Positive reaction times with joint decisions, or unchanged response
        coordinates for an ordinary scalar or vector family.
    """
    if formula.modifiers.decision is not None:
        return ResponseDraws(
            value=draws.abs(),
            decision=(draws > 0).long(),
            decision_labels=("lower", "upper"),
        )
    return ResponseDraws(value=draws)


def response_input_name(formula: Formula) -> str:
    """Name the raw response input before any checked response adaptation.

    Parameters
    ----------
    formula : Formula
        Retained response and modifier metadata.

    Returns
    -------
    str
        Compiler-owned raw input for a decision response, or its original name.
    """
    from quivers.formulas.formula import _qvr_name

    return (
        "formula_response"
        if formula.modifiers.decision is not None
        else _qvr_name(formula.response_name)
    )
