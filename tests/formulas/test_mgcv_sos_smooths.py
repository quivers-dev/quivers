"""mgcv splines on the sphere ``s(lat, lon, bs = "sos")`` equal brms's Stan data.

Oracle: ``oracles/mgcv_sos_smooths.R`` (brms 2.20.4, mgcv 1.8-42) records, for
every penalty choice of ``m``, the unpenalized design ``Xs`` and penalized
design ``Zs`` (checked there against ``make_standata``), the matrices brms
evaluates on fresh rows, mgcv's kernel ``makeR`` at fixed points, mgcv's raw
constructor output, and the text order of ``uniquecombs``. Penalized columns
are compared up to sign. Unpenalized columns of a one-dimensional null space
are compared up to sign, and those of a larger null space by their span,
because brms takes them from a repeated zero eigenvalue and gives them
exchangeable flat priors.

Tolerances, fixed before comparing:

- Designs, ``RTOL = 1e-6`` of each matrix's largest entry. mgcv stops its
  Lanczos eigensolver at a relative residual of ``eps ** 0.5`` (1.5e-8), so
  eigenvector error is that residual over the relative eigengap. The kernels
  of ``m = -2`` and ``m = 1`` also have a square-root cusp at coincident
  points, where the rounded cosine of a zero angle is one or one minus one
  unit of least precision, an angle of 1.5e-8. The tolerance covers both
  while rejecting any change of kernel, knots, constraint, or scale.
- Kernel values, ``1e-12`` for angles above 1e-3 radians, where the
  arccosine amplifies a rounding error of the cosine by at most 1e3, and
  ``1e-7`` at smaller angles because of the cusp.
- Closed forms against quadrature and the dilogarithm, ``1e-10``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.integrate import quad
from scipy.special import eval_legendre, spence

from quivers.formulas import BayesianFit, conditional_smooths, fit
from quivers.formulas._data_options import formula_data_options
from quivers.formulas._mgcv_smooth import (
    SphereBasis,
    _r_number_text,
    _sphere_dilogarithm_kernel,
    _sphere_kernel,
    _sphere_null,
    _unique_rows_as_text,
    sphere_order,
    sphere_spline,
)
from quivers.formulas._smooth import SmoothTransform
from quivers.formulas.formula import _formula_and_design
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "mgcv_sos_smooths.json").read_text()
)
RTOL = 1e-6
KERNEL_ATOL = 1e-12
CUSP_ATOL = 1e-7
CLOSED_FORM_ATOL = 1e-10


def _frames() -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.DataFrame(ORACLE[name]) for name in ("train", "fresh", "repeated")
    }
    levels = sorted(set(frames["train"]["g"]))
    for frame in frames.values():
        frame["g"] = pd.Categorical(frame["g"], categories=levels)
    return frames


def _matrix(rows: list, count: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(count, -1)


def _assert_columns_up_to_sign(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    if not expected.size:
        return
    signs = np.sign(np.sum(actual * expected, axis=0))
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual * signs, expected, rtol=0, atol=RTOL * scale)


def _assert_same_span(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    if not expected.size:
        return
    left = np.linalg.qr(actual)[0]
    right = np.linalg.qr(expected)[0]
    np.testing.assert_allclose(left @ left.T, right @ right.T, rtol=0, atol=RTOL)


def _unit_vectors(points: np.ndarray) -> np.ndarray:
    latitude, longitude = np.deg2rad(points[:, 0]), np.deg2rad(points[:, 1])
    return np.column_stack(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )


def _kernel_grid() -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack(
        (ORACLE["kernel_points"]["la"], ORACLE["kernel_points"]["lo"])
    ).astype(np.float64)
    knots = np.column_stack(
        (ORACLE["kernel_knots"]["la"], ORACLE["kernel_knots"]["lo"])
    ).astype(np.float64)
    return points, knots


@pytest.mark.parametrize(
    "entry",
    ORACLE["entries"],
    ids=lambda e: f"{e['spec']}|{e['data']}|{'knots' if 'knots' in e else 'data'}",
)
def test_training_and_fresh_designs_equal_brms(entry):
    frames = _frames()
    data, fresh = frames[entry["data"]], frames["fresh"]
    with formula_data_options(
        knots=entry.get("knots"), drop_unused_levels=True, data2=None
    ):
        _, design = _formula_and_design("y ~ " + entry["spec"], data)
        [(name, term)] = [
            (name, term)
            for name, term in design.common.terms.items()
            if name != "Intercept"
        ]
        transform = term.components[0].call.stateful_transform
        assert isinstance(transform, SmoothTransform)
        state = transform.state
        columns = design.common.slices[name]
        trained = np.asarray(design.common.design_matrix)[:, columns]
        evaluated = np.asarray(design.common.evaluate_new_data(fresh).design_matrix)[
            :, columns
        ]
    assert isinstance(state.basis, SphereBasis)
    assert len(state.layout) == len(entry["blocks"])
    for layout, block in zip(state.layout, entry["blocks"], strict=True):
        fixed = []
        for values, prefix, rows in (
            (trained, "", len(data)),
            (evaluated, "fresh_", len(fresh)),
        ):
            expected = block[prefix + "fixed"]
            fixed.append(_matrix(expected, rows) if expected else np.zeros((rows, 0)))
            assert len(layout.random) == len(block[prefix + "random"])
            for group, expected in zip(
                layout.random, block[prefix + "random"], strict=True
            ):
                _assert_columns_up_to_sign(
                    values[:, list(group)], _matrix(expected, rows)
                )
        # One rotation of the unpenalized columns relates training and fresh
        # rows, so their span is compared on the stacked rows.
        ours = np.vstack(
            (trained[:, list(layout.fixed)], evaluated[:, list(layout.fixed)])
        )
        theirs = np.vstack(fixed)
        if theirs.shape[1] <= 1:
            _assert_columns_up_to_sign(ours, theirs)
        else:
            assert np.linalg.matrix_rank(theirs) == theirs.shape[1]
            _assert_same_span(ours, theirs)


@pytest.mark.parametrize("entry", ORACLE["kernels"], ids=lambda e: f"m={e['m']}")
def test_kernel_equals_mgcv_make_r(entry):
    points, knots = _kernel_grid()
    expected = np.asarray(entry["R"], dtype=np.float64)
    actual = _sphere_kernel(points, knots, entry["m"])
    cosine = np.clip(_unit_vectors(points) @ _unit_vectors(knots).T, -1.0, 1.0)
    near = np.arccos(cosine) < 1e-3
    assert near.sum() == 4
    np.testing.assert_allclose(actual[~near], expected[~near], rtol=0, atol=KERNEL_ATOL)
    np.testing.assert_allclose(actual[near], expected[near], rtol=0, atol=CUSP_ATOL)
    np.testing.assert_allclose(
        _sphere_null(points, entry["m"]),
        _matrix(entry["null"], len(points)),
        rtol=0,
        atol=KERNEL_ATOL,
    )


@pytest.mark.parametrize("entry", ORACLE["raw"], ids=lambda e: f"k={e['k']},m={e['m']}")
def test_raw_constructor_equals_mgcv(entry):
    frames = _frames()
    x = frames["train"][["la", "lo"]].to_numpy(dtype=np.float64)
    fresh = frames["fresh"][["la", "lo"]].to_numpy(dtype=np.float64)
    raw = sphere_spline(x, entry["k"], entry["m"])
    basis = raw.basis
    assert isinstance(basis, SphereBasis)
    # The knots are bit-for-bit mgcv's, in the text order of uniquecombs.
    np.testing.assert_array_equal(basis.knots, np.asarray(entry["knots"]))
    assert raw.rank == (entry["rank"],) and raw.null_dim == entry["null_dim"]
    expected = np.asarray(entry["X"], dtype=np.float64)
    signs = np.sign(np.sum(raw.design * expected, axis=0))
    np.testing.assert_allclose(raw.design * signs, expected, rtol=0, atol=RTOL)
    penalty = np.asarray(entry["S"], dtype=np.float64)
    np.testing.assert_allclose(
        raw.penalties[0] * np.outer(signs, signs),
        penalty,
        rtol=0,
        atol=RTOL * float(np.max(np.abs(penalty))),
    )
    np.testing.assert_allclose(basis.scale, entry["scale"], rtol=RTOL)
    np.testing.assert_allclose(
        basis.evaluate(fresh) * signs,
        np.asarray(entry["fresh"], dtype=np.float64),
        rtol=0,
        atol=RTOL,
    )
    # The penalized columns are orthogonal to the null space at the knots.
    np.testing.assert_allclose(
        _sphere_null(basis.knots, entry["m"]).T @ basis.transform, 0.0, atol=1e-10
    )


def test_second_order_kernel_is_the_dilogarithm_sum_of_wahba_series():
    z = np.array([-1.0, -0.8, -0.3, 0.0, 1e-9, 0.2, 0.75, 0.999, 1.0])
    closed = 1 - math.pi**2 / 6 + spence(1 - (1 + z) / 2)
    np.testing.assert_allclose(
        _sphere_dilogarithm_kernel(z), closed, rtol=0, atol=CLOSED_FORM_ATOL
    )
    # Wahba (1981), equation 2.12 with m = 2. The Legendre polynomials are
    # bounded by one, so the tail after N terms is at most 1 / (N + 1) ** 2,
    # attained at z = 1. The added 1e-12 covers the rounding of 2000 terms.
    order = np.arange(1, 2001)[:, None]
    series = np.sum(
        (2 * order + 1) / (order * (order + 1)) ** 2 * eval_legendre(order, z[None, :]),
        axis=0,
    )
    np.testing.assert_allclose(
        _sphere_dilogarithm_kernel(z), series, rtol=0, atol=1 / 2001**2 + 1e-12
    )
    # The first-order series coefficients give a different function.
    wrong = np.sum(
        (2 * order + 1) / (order * (order + 1)) * eval_legendre(order, z[None, :]),
        axis=0,
    )
    assert np.max(np.abs(_sphere_dilogarithm_kernel(z) - wrong)) > 0.1


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_pseudospline_kernels_equal_wahba_integral(m):
    # Wahba (1981), equations 3.3 and 3.4: R = (q_m / m! - 1 / (m + 1)!) / (2 pi)
    # with q_m(z) the integral of (1 - h) ** m / sqrt(1 - 2 h z + h ** 2).
    latitudes = np.array([-80.0, -41.0, -3.0, 0.0, 27.5, 66.0, 90.0])
    points = np.column_stack((latitudes, np.linspace(-170.0, 340.0, 7)))
    knots = np.array([[12.0, 40.0], [-55.0, 200.0]])
    cosine = _unit_vectors(points) @ _unit_vectors(knots).T
    expected = np.empty_like(cosine)
    wrong = np.empty_like(cosine)
    for index, z in np.ndenumerate(cosine):
        for order, target in ((m, expected), (m + 1, wrong)):
            q = quad(
                lambda h, z=z, order=order: (
                    (1 - h) ** order / math.sqrt(1 - 2 * h * z + h * h)
                ),
                0.0,
                1.0,
                epsabs=1e-13,
                epsrel=1e-13,
            )[0]
            target[index] = (
                q / math.factorial(order) - 1 / math.factorial(order + 1)
            ) / (2 * math.pi)
    actual = _sphere_kernel(points, knots, m)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=CLOSED_FORM_ATOL)
    assert np.max(np.abs(actual - wrong)) > 1e-4
    # Swapped latitude and longitude give a different kernel.
    assert (
        np.max(np.abs(_sphere_kernel(points[:, ::-1] / 4, knots, m) - expected)) > 1e-4
    )


def test_embedded_sphere_kernels_are_semi_kernels_of_the_chord():
    points, knots = _kernel_grid()
    difference = _unit_vectors(points)[:, None, :] - _unit_vectors(knots)[None, :, :]
    chord = np.sqrt(np.sum(difference * difference, axis=2))
    far = chord > 1e-3
    np.testing.assert_allclose(
        _sphere_kernel(points, knots, -2)[far], -chord[far], rtol=0, atol=KERNEL_ATOL
    )
    np.testing.assert_allclose(
        _sphere_kernel(points, knots, -1)[far],
        chord[far] ** 2 * np.log(chord[far]) / (8 * math.pi),
        rtol=0,
        atol=KERNEL_ATOL,
    )
    null = _sphere_null(points, -1)
    np.testing.assert_allclose(null[:, 0], 1.0)
    # The three null columns are the coordinates of the embedding.
    np.testing.assert_allclose(
        np.sum(null[:, 1:] ** 2, axis=1), 1.0, rtol=0, atol=1e-14
    )
    assert _sphere_null(points, 2).shape == (len(points), 1)


def test_unique_rows_follow_mgcv_text_order():
    recorded = ORACLE["row_text"]
    for value, text in zip(recorded["values"], recorded["text"], strict=True):
        assert _r_number_text(float(value)) == text
    rows = np.asarray(recorded["rows"], dtype=np.float64)
    unique = _unique_rows_as_text(rows)
    np.testing.assert_array_equal(unique, np.asarray(recorded["unique"]))
    # Numeric order differs from mgcv's order of the pasted text.
    assert not np.array_equal(unique, np.unique(rows, axis=0)[: len(unique)])


@pytest.mark.parametrize(
    ("authored", "read"),
    [(-3, -1), (-7.2, -1), (-2, -2), (0, 0), (1.4, 1), (2.5, 2), (3.5, 4), (7, 4)],
)
def test_order_is_rounded_and_clamped_as_mgcv_does(authored, read):
    assert sphere_order(authored) == read
    assert ORACLE["accepted"] == {"out_of_range": True, "one_knot_vector": True}


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_sphere_splines_fit_predict_refit_and_restore(method, tmp_path):
    frames = _frames()
    train = frames["train"]
    fresh = frames["fresh"].assign(y=np.linspace(0.0, 1.0, len(frames["fresh"])))
    options = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 15, "num_samples": 8, "num_chains": 1}
    )
    formula = "y ~ s(la, lo, bs = 'sos', k = 7, m = -1) + w"
    fitted = fit(formula, data=train, method=method, seed=2, **options)
    names = fitted.variables()
    assert "sds_slalo_1" in names and "bs_slalo_3" in names
    assert sum(name.startswith("s_slalo_1[") for name in names) == 3
    linpred = fitted.posterior_linpred(new_data=fresh, num_samples=3)
    assert linpred.shape[-1] == len(fresh) and bool(torch.isfinite(linpred).all())
    assert fitted.posterior_epred(new_data=fresh, num_samples=3).shape[-1] == 8
    assert fitted.posterior_predict(new_data=fresh, num_samples=3).shape[-1] == 8
    assert bool(
        torch.isfinite(fitted.log_likelihood(new_data=fresh, num_samples=3)["y"]).all()
    )
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=train, num_samples=3, seed=1),
        fitted.posterior_linpred(num_samples=3, seed=1),
        rtol=1e-6,
        atol=1e-6,
    )
    term = next(iter(t.term for t in fitted.formula.design_terms))
    effect = conditional_smooths(fitted, term, fresh, num_samples=3, seed=1)
    assert effect.shape[-1] == len(fresh) and bool(torch.isfinite(effect).all())
    path = tmp_path / "sos_smooth.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    torch.testing.assert_close(
        restored.posterior_linpred(new_data=fresh, num_samples=3, seed=1),
        fitted.posterior_linpred(new_data=fresh, num_samples=3, seed=1),
    )
    with pytest.raises(ValueError, match=r"latitude outside \[-90, 90\]"):
        fitted.posterior_linpred(new_data=fresh.assign(la=95.0), num_samples=2)
    if method == "svi":
        updated = fitted.update(num_steps=3)
        assert set(updated.draws()) == set(fitted.draws())
        assert np.isfinite(fitted.kfold(K=2, seed=1).estimate.elpd)


def test_user_knots_and_distributional_predictors_fit():
    train = _frames()["train"]
    knots = {"la": train["la"][::3].tolist(), "lo": train["lo"][::3].tolist()}
    fitted = fit(
        "y ~ s(la, lo, bs = 'sos', k = 6, m = 2)",
        data=train,
        knots=knots,
        method="svi",
        seed=3,
        num_steps=10,
    )
    fresh = _frames()["fresh"]
    assert bool(
        torch.isfinite(fitted.posterior_linpred(new_data=fresh, num_samples=2)).all()
    )
    scale = fit(
        "y ~ w",
        distributional={"sigma": "~ s(la, lo, bs = 'sos', k = 5)"},
        data=train,
        method="svi",
        seed=3,
        num_steps=10,
    )
    assert "sds_sigma_slalo_1" in scale.variables()


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("y ~ s(la, lo, bs = 'sos', k = 61)", "fewer unique covariate combinations"),
        ("y ~ s(la, lo, bs = 'sos', k = 2)", "k of at least 3"),
        ("y ~ s(la, lo, bs = 'sos', k = 1)", "k of at least 3"),
        ("y ~ s(la, lo, bs = 'sos', k = 5, m = -1)", "k of at least 6"),
        ("y ~ s(la, lo, bs = 'sos', k = 10, m = c(1, 2))", "one m from -2 to 4"),
        ("y ~ s(la, lo, bs = 'sos', k = 10, m = 'a')", "one m from -2 to 4"),
        ("y ~ s(la, bs = 'sos')", "exactly two covariates"),
        ("y ~ s(la, lo, w, bs = 'sos')", "exactly two covariates"),
        ("y ~ t2(la, lo, bs = 'sos')", "margins must use bases"),
        ("y ~ s(la, lo, bs = 'sos', xt = list(max.knots = 20))", "knot subsample"),
        ("y ~ s(la, g, bs = 'sos')", "finite real"),
        ("y ~ s(la, lo, g, bs = 'fs', xt = 'sos')", "base"),
        ("y ~ s(bad_la, lo, bs = 'sos', k = 6)", r"latitude outside \[-90, 90\]"),
        ("y ~ s(la, bad_lo, bs = 'sos', k = 6)", r"longitude outside \[-180, 360\]"),
        ("y ~ s(lo, la, bs = 'sos', k = 6)", r"latitude outside \[-90, 90\]"),
        ("y ~ s(flat, lo, bs = 'sos', k = 8, m = -1)", "lie on one circle"),
    ],
)
def test_invalid_sphere_splines_are_refused(formula, message):
    train = _frames()["train"]
    train = train.assign(
        bad_la=train["la"].where(train.index != 4, 120.0),
        bad_lo=train["lo"].where(train.index != 7, 700.0),
        flat=30.0,
    )
    with pytest.raises(ValueError, match=message):
        _formula_and_design(formula, train)


def test_mgcv_failures_are_refused_here():
    train = _frames()["train"]
    assert len(ORACLE["failures"]) == 8
    for failure in ORACLE["failures"]:
        assert failure["error"]
        with pytest.raises(ValueError):
            _formula_and_design(
                "y ~ " + failure["spec"].replace("max.knots", "k"), train
            )


@pytest.mark.parametrize(
    ("knots", "message"),
    [
        ({"la": [0.0, 10.0, 20.0, 30.0]}, "both covariates"),
        ({"la": [0.0, 10.0, 20.0], "lo": [0.0, 10.0]}, "same length"),
        ({"la": [0.0, 10.0, 95.0, 30.0], "lo": [0.0, 1.0, 2.0, 3.0]}, "the knots"),
        (
            {
                "la": np.linspace(-80, 80, 70).tolist(),
                "lo": np.linspace(0, 300, 70).tolist(),
            },
            "more knots than rows",
        ),
        ({"la": [0.0, 10.0, 20.0, 30.0], "lo": [0.0, 1.0, 2.0, 3.0]}, "fewer unique"),
    ],
)
def test_invalid_sphere_knots_are_refused(knots, message):
    train = _frames()["train"]
    with (
        formula_data_options(knots=knots, drop_unused_levels=True, data2=None),
        pytest.raises(ValueError, match=message),
    ):
        _formula_and_design("y ~ s(la, lo, bs = 'sos', k = 6)", train)


def test_fresh_rows_outside_the_sphere_are_refused():
    train = _frames()["train"]
    raw = sphere_spline(train[["la", "lo"]].to_numpy(dtype=np.float64), 6, 0)
    with pytest.raises(ValueError, match="longitude outside"):
        raw.basis.evaluate(np.array([[10.0, 361.0]]))
    with pytest.raises(ValueError, match="must be finite"):
        raw.basis.evaluate(np.array([[np.nan, 10.0]]))
    with pytest.raises(ValueError, match="one finite m"):
        sphere_order(float("nan"))
    with pytest.raises(ValueError, match="one m from -2 to 4"):
        sphere_spline(train[["la", "lo"]].to_numpy(dtype=np.float64), 6, 5)
