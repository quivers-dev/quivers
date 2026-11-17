"""The data behind bayesplot's ppc plots, against bayesplot itself.

Oracle: ``oracles/bayesplot_ppc_data.R`` (bayesplot, loo, rstantools and
survival on exported inputs). Tolerance 1e-9 relative, from the 15
significant digits of the export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quivers.diagnostics import (
    km_data,
    loo_pit_values,
    pit_ecdf_data,
    pit_values,
    ppc_bars_data,
    ppc_data,
    ppc_error_binned_data,
    ppc_error_data,
    ppc_intervals_data,
    ppc_rootogram_data,
    ppc_scatter_avg_data,
    ppc_scatter_data,
    ppc_stat_data,
)

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "bayesplot_ppc_data.json").read_text()
)
INPUTS = {k: np.asarray(v) for k, v in ORACLE["inputs"].items()}
Y, YREP, GROUP, X = INPUTS["y"], INPUTS["yrep"], INPUTS["group"], INPUTS["x"]


def assert_frame(actual: pd.DataFrame, expected: dict) -> None:
    """Compare a frame with an exported R data frame column by column."""
    assert list(actual.columns) == list(expected), (
        list(actual.columns),
        list(expected),
    )
    for name, values in expected.items():
        column = actual[name].tolist()
        assert len(column) == len(values), name
        for a, e in zip(column, values, strict=True):
            if e is None:
                assert a is None or (isinstance(a, float) and np.isnan(a)), (name, a)
            elif isinstance(e, (int, float)) and not isinstance(e, bool):
                assert float(a) == pytest.approx(e, rel=1e-9, abs=1e-12), (name, a, e)
            else:
                assert a == e, (name, a, e)


def test_long_tables_match_ppc_and_ppd_data():
    assert_frame(ppc_data(Y, YREP, group=GROUP), ORACLE["ppc_data"])
    assert_frame(ppc_data(None, YREP), ORACLE["ppd_data"])


def test_statistics_match_ppc_stat_data():
    assert_frame(ppc_stat_data(Y, YREP), ORACLE["stat"])
    assert_frame(
        ppc_stat_data(Y, YREP, stat="median", group=GROUP), ORACLE["stat_grouped"]
    )
    assert_frame(ppc_stat_data(Y, YREP, stat=("mean", "sd")), ORACLE["stat_2d"])
    assert_frame(ppc_stat_data(None, YREP, stat="sd"), ORACLE["ppd_stat"])
    with pytest.raises(ValueError, match="unknown statistic"):
        ppc_stat_data(Y, YREP, stat="mode")


def test_intervals_match_ppc_intervals_data():
    assert_frame(ppc_intervals_data(Y, YREP, x=X, group=GROUP), ORACLE["intervals"])
    assert_frame(
        ppc_intervals_data(Y, YREP, prob=0.6, prob_outer=0.95),
        ORACLE["intervals_plain"],
    )
    with pytest.raises(ValueError, match="prob"):
        ppc_intervals_data(Y, YREP, prob=1.5)


def test_errors_match_ppc_error_data_and_binned_errors():
    assert_frame(ppc_error_data(Y, YREP, group=GROUP), ORACLE["error"])
    assert_frame(
        ppc_error_binned_data(INPUTS["ybin"], INPUTS["epred"]), ORACLE["error_binned"]
    )


def test_bars_and_rootogram_match_bayesplot():
    counts, rep_counts = INPUTS["ycount"], INPUTS["yrepcount"]
    assert_frame(ppc_bars_data(counts, rep_counts), ORACLE["bars"])
    assert_frame(
        ppc_bars_data(counts, rep_counts, group=GROUP, freq=False),
        ORACLE["bars_grouped"],
    )
    assert_frame(ppc_rootogram_data(counts, rep_counts), ORACLE["rootogram"])
    with pytest.raises(ValueError, match="discrete"):
        ppc_bars_data(Y, YREP)
    with pytest.raises(ValueError, match="counts"):
        ppc_rootogram_data(Y, YREP)


def test_scatter_tables_match_bayesplot():
    assert_frame(ppc_scatter_data(Y, YREP), ORACLE["scatter"])
    assert_frame(ppc_scatter_avg_data(Y, YREP, group=GROUP), ORACLE["scatter_avg"])


def test_loo_pit_matches_rstantools_with_psis_weights():
    np.testing.assert_allclose(
        loo_pit_values(Y, INPUTS["yrep_loo"], INPUTS["lw"]),
        ORACLE["loo_pit"],
        rtol=1e-9,
    )
    with pytest.raises(ValueError, match="log weights"):
        loo_pit_values(Y, INPUTS["yrep_loo"], INPUTS["lw"][:3])


def test_kaplan_meier_matches_survfit():
    assert_frame(
        km_data(INPUTS["ypos"], INPUTS["yreppos"], status=INPUTS["status"]),
        ORACLE["km"],
    )
    with pytest.raises(ValueError, match="status"):
        km_data(INPUTS["ypos"], INPUTS["yreppos"], status=np.full(Y.size, 2.0))


def test_pit_ecdf_matches_bayesplot():
    case = ORACLE["pit_ecdf"]
    np.testing.assert_allclose(pit_values(Y, YREP), case["pit"], rtol=1e-12)
    frame = pit_ecdf_data(Y, YREP)
    assert len(frame) == case["K"]
    np.testing.assert_allclose(frame["ecdf"], case["ecdf"], rtol=1e-12)
