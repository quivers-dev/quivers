"""Response roles bind once, retain values, and structurally select rows."""

import pickle

import numpy as np
import pandas as pd
import pytest

from quivers.formulas._response_roles import (
    bind_roles,
    parse_roles,
    validate_event_response,
)


def test_shared_role_syntax_preserves_predictors_and_binds_keyword_aliases():
    normalized, roles = parse_roles(
        "y | cens(kind, y2=end) + trunc(lb=-1.0, ub=upper) + se(sdy, sigma=False) + subset(include) + weights(2.0) + trials(n) ~ x + (1 | g)"
    )
    assert normalized.strip() == "y ~  x + (1 | g)"
    assert roles == {
        "censoring": "kind",
        "censoring_upper": "end",
        "trunc_lower": -1.0,
        "trunc_upper": "upper",
        "standard_error": "sdy",
        "se_sigma": False,
        "subset": "include",
        "weights": 2.0,
        "trials": "n",
    }


@pytest.mark.parametrize(
    "modifier",
    [
        "cens()",
        "trunc()",
        "se(s, sigma=flag)",
        "weights(open('secret'))",
        "weights(*x)",
        "trunc(lb=x, lower=y)",
        "cens(c, x, y)",
        "trunc(x, lb=y)",
        "module.weights(w)",
    ],
)
def test_modifier_syntax_and_constant_policies_are_checked_without_evaluation(modifier):
    with pytest.raises(ValueError):
        normalized, roles = parse_roles(f"y | {modifier} ~ x")
        bind_roles(pd.DataFrame({"x": [0.0], "s": [0.2]}), roles)


def test_conflicts_are_checked_by_semantic_role_and_equal_declarations_agree():
    assert parse_roles("y | dec(choice) ~ x", {"decision": "choice"})[1] == {
        "decision": "choice"
    }
    with pytest.raises(ValueError, match="conflicting"):
        parse_roles("y | weights(w) ~ x", {"weights": "other"})


def test_subset_removes_invalid_rows_before_any_other_role_or_design_reading():
    frame = pd.DataFrame(
        {
            "y": [1.0, np.nan, 2.0],
            "x": [0.2, np.nan, 0.4],
            "keep": [True, False, True],
            "w": [1.0, -99.0, 2.0],
            "se": [0.1, np.nan, 0.3],
            "cens": [0, 99, 2],
            "end": [None, "invalid", 3.0],
            "n": [0, -5, 10],
        },
        index=["a", "excluded", "b"],
    )
    _, sources = parse_roles(
        "y | subset(keep) + weights(w) + se(se) + cens(cens,end) + trials(n) ~ x"
    )
    selected, roles = bind_roles(frame, sources)
    assert list(selected.index) == ["a", "b"]
    assert roles["subset"].positions == (0, 2) and roles["subset"].size == 3
    np.testing.assert_array_equal(roles["weights"].data, [1.0, 2.0])
    np.testing.assert_array_equal(roles["censoring"].status.data, [0, 2])
    np.testing.assert_array_equal(roles["censoring"].upper.data, [0.0, 3.0])
    np.testing.assert_array_equal(roles["trials"].data, [0, 10])
    validate_event_response(selected.y.to_numpy(), roles["censoring"])
    assert roles["censoring"] == pickle.loads(pickle.dumps(roles["censoring"]))
    frame.loc["a", "w"] = 4.0
    assert roles["weights"].data[0] == 1.0 and not roles["weights"].data.flags.writeable


@pytest.mark.parametrize(
    "statuses,expected",
    [([False, True], [0, 1]), (["left", "interval"], [-1, 2]), ([-1.0, 0.0], [-1, 0])],
)
def test_censoring_encodings_have_one_canonical_integer_carrier(statuses, expected):
    frame = pd.DataFrame({"code": statuses, "end": [3.0, 4.0]})
    _, roles = bind_roles(frame, {"censoring": "code", "censoring_upper": "end"})
    actual = roles["censoring"].status.data
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("code", [3, -2, 0.5, np.nan, "maybe"])
def test_invalid_censoring_status_is_rejected(code):
    with pytest.raises(ValueError, match="censoring"):
        bind_roles(pd.DataFrame({"code": [code]}), {"censoring": "code"})


def test_interval_missingness_is_permitted_only_on_inactive_rows():
    frame = pd.DataFrame({"y": [1.0, 2.0], "c": [0, 2], "end": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="upper endpoint"):
        bind_roles(frame, {"censoring": "c", "censoring_upper": "end"})
    frame["end"] = [np.nan, 1.0]
    _, roles = bind_roles(frame, {"censoring": "c", "censoring_upper": "end"})
    with pytest.raises(ValueError, match="exceed"):
        validate_event_response(frame.y.to_numpy(), roles["censoring"])


@pytest.mark.parametrize("n", [-1, True, 0.5, 2**53 + 1, 2**63 - 1])
def test_trial_role_rejects_invalid_or_inexact_counts_before_float_conversion(n):
    with pytest.raises(ValueError, match="trials"):
        bind_roles(pd.DataFrame({"n": [n]}), {"trials": "n"})


def test_constant_bounds_and_errors_retain_inputs_and_policy():
    _, roles = bind_roles(
        pd.DataFrame({"x": [0.0, 1.0]}),
        {
            "trunc_lower": -float("inf"),
            "trunc_upper": 3.0,
            "standard_error": 0.3,
            "se_sigma": False,
        },
    )
    np.testing.assert_array_equal(roles["truncation"].lower.data, [-float("inf")] * 2)
    np.testing.assert_array_equal(roles["truncation"].upper.data, [3.0, 3.0])
    assert not roles["standard_error"].sigma
    np.testing.assert_array_equal(roles["standard_error"].column.data, [0.3, 0.3])


def test_scaled_weights_normalize_included_rows_without_overflow():
    frame = pd.DataFrame({"keep": [True, False, True], "w": [1e308, -1.0, 1e308]})
    _, sources = parse_roles("y | subset(keep) + weights(w, scale=True) ~ x")
    _, roles = bind_roles(frame, sources)
    assert roles["weights_scale"] is True
    np.testing.assert_array_equal(roles["weights"].data, [1.0, 1.0])
    with pytest.raises(ValueError, match="positive total"):
        bind_roles(frame, {"weights": 0.0, "weights_scale": True})


@pytest.mark.parametrize(
    "sources",
    [
        {"subset": False},
        {"weights": True},
        {"standard_error": 0.0, "se_sigma": False},
        {"trunc_lower": 2.0, "trunc_upper": 1.0},
        {"se_sigma": True},
        {"censoring_upper": 1.0},
        {"unknown": 1.0},
    ],
)
def test_role_dependencies_and_numeric_domains_fail_before_model_execution(sources):
    with pytest.raises(ValueError):
        bind_roles(pd.DataFrame({"y": [1.0]}), sources)
