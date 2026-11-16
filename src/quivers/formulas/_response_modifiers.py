"""Typed row modifiers shared by formula parsing and response adapters."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

import didactic.api as dx
import numpy as np
import pandas as pd
import torch
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.formulas._group_prediction import NewLevelPolicy

if TYPE_CHECKING:
    from quivers.formulas.formula import FixedColumn, Formula, ResponseModifiers


# Role aliases remain independent of any particular response family.
_ROLE_CALLS = {"weights": "weights", "dec": "decision"}
_ROLE_INPUTS = {"weights": "formula_weights", "decision": "formula_decision"}


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


def modifier_names(
    formula: str, explicit: Mapping[str, str | None]
) -> tuple[str, dict[str, str]]:
    """Read named left-hand response modifiers without evaluating expressions.

    Parameters
    ----------
    formula : str
        Response and predictor formula; modifiers follow a response-side bar.
    explicit : Mapping[str, str or None]
        Equivalent keyword column names by semantic role.

    Returns
    -------
    tuple of str and dict
        Formula for the design evaluator and resolved modifier-column names.

    Raises
    ------
    ValueError
        If modifier syntax, role, arguments, or duplicate declarations disagree.
    """
    lhs, separator, rhs = formula.partition("~")
    names = {role: name for role, name in explicit.items() if name is not None}
    if not separator or "|" not in lhs:
        return formula, names
    text = lhs.strip()
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError as error:
        raise ValueError(
            "response modifiers require response | dec(column) + weights(column)"
        ) from error
    if not isinstance(expression, ast.BinOp) or not isinstance(
        expression.op, ast.BitOr
    ):
        return formula, names
    calls = []
    pending = [expression.right]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            pending.extend((node.right, node.left))
        else:
            calls.append(node)
    for call in calls:
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id not in _ROLE_CALLS
        ):
            raise ValueError(
                "response modifiers support dec(column) and weights(column)"
            )
        if (
            len(call.args) != 1
            or call.keywords
            or not isinstance(call.args[0], ast.Name)
        ):
            raise ValueError(f"{call.func.id} requires one dataframe column name")
        role, name = _ROLE_CALLS[call.func.id], call.args[0].id
        if role in names and names[role] != name:
            raise ValueError(
                f"conflicting {role} modifier columns {names[role]!r} and {name!r}"
            )
        names[role] = name
    response = ast.get_source_segment(text, expression.left)
    assert response is not None
    return f"{response} ~ {rhs}", names


def modifier_columns(
    frame: pd.DataFrame, names: Mapping[str, str]
) -> ResponseModifiers:
    """Validate named modifier columns and retain immutable typed row values.

    Parameters
    ----------
    frame : pandas.DataFrame
        A dataframe exposing column names and named arrays.
    names : Mapping[str, str]
        Columns keyed by checked semantic role.

    Returns
    -------
    ResponseModifiers
        Likelihood weights and normalized binary decisions.

    Raises
    ------
    ValueError
        If names, numeric domains, labels, or column shapes are invalid.
    """
    from quivers.formulas.formula import FixedColumn, ResponseModifiers

    columns: dict[str, FixedColumn] = {}
    for role, name in names.items():
        if role not in _ROLE_INPUTS:
            raise ValueError(f"unknown response modifier role {role!r}")
        if not isinstance(name, str) or name not in frame.columns:
            raise ValueError(f"{role} must name an existing dataframe column")
        values = np.asarray(frame[name])
        if values.ndim != 1:
            raise ValueError(f"{role} requires one value per response row")
        if role == "weights":
            if values.dtype.kind not in "iuf":
                raise ValueError("weights require a real numeric column")
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("weights must be finite and nonnegative")
            values = values.astype(np.float64)
        elif values.dtype.kind in "iuf":
            if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
                raise ValueError("decisions must be exactly zero/lower or one/upper")
            values = values.astype(np.int64)
        else:
            if not all(
                isinstance(value, str) and value in {"lower", "upper"}
                for value in values
            ):
                raise ValueError("decisions require numeric 0/1 or lower/upper labels")
            values = np.asarray(
                [int(value == "upper") for value in values], dtype=np.int64
            )
        columns[role] = FixedColumn(
            term=name, name=name, qvr_name=_ROLE_INPUTS[role], data=values
        )
    return ResponseModifiers(**columns)
