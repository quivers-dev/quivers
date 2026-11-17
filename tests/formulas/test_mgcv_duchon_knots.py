"""Duchon splines and user knots of thin-plate bases equal brms's Stan data.

Oracle: ``oracles/mgcv_duchon_knots.R`` (brms 2.20.4, mgcv 1.8-42) records,
for Duchon splines ``s(..., bs = "ds")`` and for ``brm(knots = )`` on
thin-plate bases and ``t2()`` margins, the unpenalized design ``Xs`` and the
penalized designs ``Zs`` (checked there against ``make_standata``) and the
matrices brms evaluates on fresh rows.

Comparison. Penalized columns are compared up to sign. Unpenalized columns
of a null space of dimension at least two come from a repeated zero
eigenvalue of the penalty, whose basis LAPACK fixes only up to rounding
before mgcv rescales each column, so QVR fixes that basis canonically
(``_canonical_null``): training and fresh matrices must equal brms's through
one common invertible map of the null space. The shrunken thin-plate penalty
of a ``ts`` basis over two covariates gives its former null space one
repeated positive eigenvalue, whose eigenvectors brms's iid standard-Normal
coefficients do not distinguish; those columns must equal brms's through
one common orthogonal rotation.

Tolerance, fixed before comparing: mgcv stops the Lanczos eigensolver of a
Duchon spline at a relative residual of ``eps ** 0.5`` (1.5e-8) and that of
a thin-plate basis at ``eps ** 0.7``, so eigenvector error is the residual
over the relative eigengap. A relative tolerance of 1e-6 of each matrix's
largest entry covers both while rejecting any change of kernel, order,
knots, constraint, or scale. Kernels and spans are also checked against
closed forms written here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.formulas import (
    BayesianFit,
    FormulaToQVRModule,
    conditional_smooths,
    families,
    fit,
    formula_from_data,
)
from quivers.formulas._data_options import formula_data_options
from quivers.formulas._mgcv_smooth import (
    DuchonBasis,
    ThinPlateBasis,
    duchon_spline,
    thin_plate,
)
from quivers.formulas._smooth import SmoothState, SmoothTransform
from quivers.formulas.formula import _formula_and_design
from quivers.inference.trace import trace
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "mgcv_duchon_knots.json").read_text()
)
RTOL = 1e-6
#: Entries whose penalized block holds a repeated positive eigenvalue: the
#: count of leading columns (the smallest penalty eigenvalue comes first in
#: brms's block) fixed only up to a rotation.
REPEATED_HEAD = {"s(x, z, bs = 'ts', k = 9)": 2}


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(ORACLE["train"])
    fresh = pd.DataFrame(ORACLE["fresh"])
    levels = sorted(set(train["g"]))
    train["g"] = pd.Categorical(train["g"], categories=levels)
    fresh["g"] = pd.Categorical(fresh["g"], categories=levels)
    return train, fresh


def _matrix(rows: list, count: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(count, -1)


def _assert_columns_up_to_sign(
    actual: tuple[np.ndarray, np.ndarray], expected: tuple[np.ndarray, np.ndarray]
) -> None:
    """Training and fresh columns equal brms's with one sign per column."""
    assert actual[0].shape == expected[0].shape
    assert actual[1].shape == expected[1].shape
    if not expected[0].size:
        return
    signs = np.sign(np.sum(actual[0] * expected[0], axis=0))
    assert bool(np.all(signs != 0))
    for ours, theirs in zip(actual, expected, strict=True):
        scale = max(1.0, float(np.max(np.abs(theirs))))
        np.testing.assert_allclose(ours * signs, theirs, rtol=0, atol=RTOL * scale)


def _assert_one_map(
    actual: tuple[np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray],
    *,
    orthogonal: bool,
) -> None:
    """Training and fresh columns equal brms's through one invertible map."""
    assert actual[0].shape == expected[0].shape
    assert actual[1].shape == expected[1].shape
    if not expected[0].size:
        return
    rotation = np.linalg.lstsq(actual[0], expected[0], rcond=None)[0]
    if orthogonal:
        np.testing.assert_allclose(
            rotation.T @ rotation, np.eye(rotation.shape[0]), rtol=0, atol=RTOL
        )
    assert float(np.linalg.cond(rotation)) < 10.0
    for ours, theirs in zip(actual, expected, strict=True):
        scale = max(1.0, float(np.max(np.abs(theirs))))
        np.testing.assert_allclose(ours @ rotation, theirs, rtol=0, atol=RTOL * scale)


def _state(spec: str, train: pd.DataFrame, knots: dict | None = None):
    with formula_data_options(knots):
        formula, design = _formula_and_design("y ~ " + spec, train)
    assert design.common is not None
    [(name, term)] = [
        (name, term)
        for name, term in design.common.terms.items()
        if name != "Intercept"
    ]
    transform = term.components[0].call.stateful_transform
    assert isinstance(transform, SmoothTransform)
    state = transform.state
    assert isinstance(state, SmoothState)
    return formula, design.common, name, state


def _entry_id(entry: dict) -> str:
    knots = entry.get("knots")
    if not knots:
        return entry["spec"]
    sizes = ",".join(f"{name}:{len(values)}" for name, values in knots.items())
    return f"{entry['spec']} knots {sizes}"


@pytest.mark.parametrize(
    "entry", ORACLE["entries"] + ORACLE["knot_entries"], ids=_entry_id
)
def test_training_and_fresh_designs_equal_brms(entry):
    train, fresh = _frames()
    _, common, name, state = _state(entry["spec"], train, entry.get("knots"))
    columns = common.slices[name]
    trained = np.asarray(common.design_matrix)[:, columns]
    # Fresh rows are evaluated outside the knots scope, from retained state.
    evaluated = np.asarray(common.evaluate_new_data(fresh).design_matrix)[:, columns]
    assert len(state.layout) == len(entry["blocks"])
    for layout, block in zip(state.layout, entry["blocks"], strict=True):
        fixed = list(layout.fixed)
        expected = tuple(
            _matrix(block[prefix + "fixed"], rows)
            if block[prefix + "fixed"]
            else np.zeros((rows, 0))
            for prefix, rows in (("", len(train)), ("fresh_", len(fresh)))
        )
        ours_fixed = (trained[:, fixed], evaluated[:, fixed])
        if len(fixed) <= 1:
            _assert_columns_up_to_sign(ours_fixed, (expected[0], expected[1]))
        else:
            _assert_one_map(ours_fixed, (expected[0], expected[1]), orthogonal=False)
        assert len(layout.random) == len(block["random"])
        for group, train_rows, fresh_rows in zip(
            layout.random, block["random"], block["fresh_random"], strict=True
        ):
            index = list(group)
            ours = (trained[:, index], evaluated[:, index])
            theirs = (_matrix(train_rows, len(train)), _matrix(fresh_rows, len(fresh)))
            head = REPEATED_HEAD.get(entry["spec"], 0)
            _assert_columns_up_to_sign(
                (ours[0][:, head:], ours[1][:, head:]),
                (theirs[0][:, head:], theirs[1][:, head:]),
            )
            if head:
                _assert_one_map(
                    (ours[0][:, :head], ours[1][:, :head]),
                    (theirs[0][:, :head], theirs[1][:, :head]),
                    orthogonal=True,
                )


def test_a_wrong_fourier_weight_or_order_is_detected():
    """The pinned matrices reject a neighbouring Duchon kernel."""
    train, fresh = _frames()
    entry = next(
        e for e in ORACLE["entries"] if e["spec"].endswith("k = 12, m = c(1, 0.5))")
    )
    for wrong in (
        "s(x, z, bs = 'ds', k = 12)",
        "s(x, z, bs = 'ds', k = 12, m = c(2, 0.5))",
    ):
        with pytest.raises(AssertionError):
            test_training_and_fresh_designs_equal_brms({**entry, "spec": wrong})
    knotted = next(e for e in ORACLE["knot_entries"] if e["spec"] == "s(x, z, k = 8)")
    # The same smooth without the user knots is a different basis.
    with pytest.raises(AssertionError):
        test_training_and_fresh_designs_equal_brms({**knotted, "knots": None})


@pytest.mark.parametrize(
    ("dimension", "m", "s", "closed_form"),
    [
        # 2 m + 2 s - d = 3: the cubic kernel of the univariate spline.
        (1, 2, 0.0, lambda r: r**3),
        # Power 2 in two dimensions: the thin-plate kernel r^2 log r.
        (2, 2, 0.0, lambda r: np.where(r > 0, r**2 * np.log(np.where(r > 0, r, 1)), 0)),
        # Power 1, with mgcv's negative sign for a conditionally negative kernel.
        (2, 1, 0.5, lambda r: -r),
        (2, 2, -0.5, lambda r: -r),
        # Power 2 in three dimensions.
        (3, 2, 0.5, lambda r: np.where(r > 0, r**2 * np.log(np.where(r > 0, r, 1)), 0)),
        (3, 1, 1.0, lambda r: -r),
        # Power 5 in one dimension.
        (1, 3, 0.0, lambda r: -(r**5)),
    ],
)
def test_kernels_equal_their_closed_forms(dimension, m, s, closed_form):
    rng = np.random.default_rng(5)
    x = rng.uniform(-1.0, 1.5, size=(14, dimension))
    raw = duchon_spline(x, 14, m, s)
    basis = raw.basis
    assert isinstance(basis, DuchonBasis)
    centered = x - x.mean(axis=0)
    # The knots are the covariate rows, in mgcv's text order.
    np.testing.assert_allclose(
        np.unique(basis.knots, axis=0), np.unique(centered, axis=0), atol=1e-14
    )
    distance = np.sqrt(
        np.sum((centered[:, None, :] - basis.knots[None, :, :]) ** 2, axis=-1)
    )
    rank = raw.rank[0]
    np.testing.assert_allclose(
        raw.design[:, :rank],
        closed_form(distance) @ basis.transform,
        rtol=0,
        atol=1e-12,
    )
    # The polynomials of degree below m follow, the constant first.
    np.testing.assert_allclose(raw.design[:, rank], 1.0)
    assert raw.null_dim == raw.design.shape[1] - rank
    # The kernel coefficients are orthogonal to those polynomials (Duchon's
    # side condition), which makes the kernel quadratic form a seminorm.
    polynomials = np.column_stack(
        [np.ones(len(basis.knots))]
        + ([basis.knots] if m > 1 else [])
        + ([basis.knots**2] if m > 2 else [])
    )
    np.testing.assert_allclose(polynomials.T @ basis.transform, 0.0, atol=1e-9)
    knot_distance = np.sqrt(
        np.sum((basis.knots[:, None, :] - basis.knots[None, :, :]) ** 2, axis=-1)
    )
    quadratic = basis.transform.T @ closed_form(knot_distance) @ basis.transform
    np.testing.assert_allclose(raw.penalties[0][:rank, :rank], quadratic, atol=1e-10)
    assert float(np.linalg.eigvalsh(quadratic).min()) > 0
    # The opposite sign is not a seminorm.
    assert float(np.linalg.eigvalsh(-quadratic).min()) < 0


@pytest.mark.parametrize("dimension", [1, 2])
def test_duchon_spline_without_weight_spans_the_thin_plate_spline(dimension):
    """With ``s = 0`` and one basis function per knot both are the full spline."""
    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, size=(12, dimension))
    fresh = rng.uniform(-0.2, 1.2, size=(9, dimension))
    duchon = duchon_spline(x, 12, 2, 0.0)
    plate = thin_plate(x, 12, 2, shrink=None)
    assert isinstance(duchon.basis, DuchonBasis)
    assert isinstance(plate.basis, ThinPlateBasis)
    stacked_d = np.vstack((duchon.design, duchon.basis.evaluate(fresh)))
    stacked_t = np.vstack((plate.design, plate.basis.evaluate(fresh)))
    left = np.linalg.qr(stacked_d)[0]
    right = np.linalg.qr(stacked_t)[0]
    np.testing.assert_allclose(left @ left.T, right @ right.T, rtol=0, atol=1e-8)
    # A half-integer weight is a different function space.
    if dimension == 2:
        other = duchon_spline(x, 12, 2, 0.5)
        assert isinstance(other.basis, DuchonBasis)
        stacked_o = np.vstack((other.design, other.basis.evaluate(fresh)))
        q = np.linalg.qr(stacked_o)[0]
        assert float(np.max(np.abs(q @ q.T - right @ right.T))) > 1e-3


@pytest.mark.parametrize("shrink", [None, 0.1])
def test_thin_plate_knots_at_the_data_reproduce_the_default_basis(shrink):
    """Knots equal to the covariate rows are mgcv's default knots."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, size=(15, 2))
    fresh = rng.uniform(0.0, 1.0, size=(6, 2))
    for k in (15, 7):
        default = thin_plate(x, k, 0, shrink=shrink)
        knotted = thin_plate(x, k, 0, shrink=shrink, knots=x[::-1].copy())
        assert isinstance(knotted.basis, ThinPlateBasis)
        assert isinstance(default.basis, ThinPlateBasis)
        tolerance = 1e-10 if k == 15 else 1e-7
        np.testing.assert_allclose(knotted.design, default.design, atol=tolerance)
        np.testing.assert_allclose(
            knotted.penalties[0], default.penalties[0], atol=tolerance
        )
        np.testing.assert_allclose(
            knotted.basis.evaluate(fresh), default.basis.evaluate(fresh), atol=tolerance
        )
    # A proper subset of the rows is a different basis.
    subset = thin_plate(x, 7, 0, shrink=shrink, knots=x[:9].copy())
    assert not np.allclose(subset.design, thin_plate(x, 7, 0, shrink=shrink).design)


def test_knots_are_retained_for_prediction_outside_their_scope():
    train, fresh = _frames()
    knots = {"x": np.linspace(0.05, 0.95, 9).tolist()}
    _, default, name, _ = _state("s(x, k = 6)", train)
    _, knotted, _, state = _state("s(x, k = 6)", train, knots)
    assert isinstance(state.basis, ThinPlateBasis)
    assert state.basis.knots.shape == (9, 1)
    columns = knotted.slices[name]
    assert not np.allclose(
        np.asarray(default.design_matrix)[:, columns],
        np.asarray(knotted.design_matrix)[:, columns],
    )
    inside = np.asarray(knotted.evaluate_new_data(fresh).design_matrix)
    with formula_data_options({"x": np.linspace(0.0, 1.0, 9).tolist()}):
        again = np.asarray(knotted.evaluate_new_data(fresh).design_matrix)
    np.testing.assert_array_equal(inside, again)
    replay = np.asarray(knotted.evaluate_new_data(train).design_matrix)
    np.testing.assert_allclose(
        replay[:, columns], np.asarray(knotted.design_matrix)[:, columns], atol=1e-10
    )


@pytest.mark.parametrize(
    ("formula", "knots", "message"),
    [
        ("y ~ s(x, z, bs = 'ds', m = 1)", None, "choose s explicitly.*c\\(1, 0.5\\)"),
        ("y ~ s(x, bs = 'ds', m = c(2, 0.5))", None, "violates Duchon's conditions"),
        ("y ~ s(x, z, bs = 'ds', m = c(2, 1))", None, "violates Duchon's conditions"),
        ("y ~ s(x, z, bs = 'ds', m = c(2, 0.3))", None, "multiple of 0.5"),
        ("y ~ s(x, z, bs = 'ds', m = c(1.5, 0))", None, "integer penalty order"),
        ("y ~ s(x, z, bs = 'ds', m = c(0, 0.5))", None, "integer m of at least 1"),
        ("y ~ s(x, z, bs = 'ds', m = c(2, 0, 1))", None, "m = c\\(m, s\\)"),
        ("y ~ s(x, z, bs = 'ds', k = 3)", None, "must exceed its penalty null-space"),
        ("y ~ s(x, z, bs = 'ds', k = 41)", None, "fewer unique covariate"),
        ("y ~ s(x, z, bs = 'ds', xt = list(max.knots = 10))", None, "reads no xt"),
        ("y ~ s(g, bs = 'ds')", None, "finite real row vector"),
        (
            "y ~ s(x, z, bs = 'ds', k = 10)",
            {"x": [0.1, 0.5, 0.9]},
            "must be given for every covariate",
        ),
        (
            "y ~ s(x, z, bs = 'ds', k = 10)",
            {"x": [0.1, 0.5, 0.9], "z": [0.0, 1.0]},
            "must be of same length",
        ),
        (
            "y ~ s(x, z, bs = 'ds', k = 10)",
            {"x": [0.1, 0.5, 0.9], "z": [0.0, 0.5, 1.0]},
            "needs at least 10 knots, got 3",
        ),
        (
            "y ~ s(x, bs = 'ds', k = 6)",
            {"x": np.linspace(0, 1, 41).tolist()},
            "more knots than data rows",
        ),
        ("y ~ s(x, k = 6)", {"x": [0.1, 0.5, 0.9]}, "needs at least 6 knots, got 3"),
        (
            "y ~ s(x, k = 6)",
            {"x": np.linspace(0, 1, 41).tolist()},
            "more knots than data rows",
        ),
        (
            "y ~ s(x, k = 6)",
            {"x": [0.1, 0.1, 0.1, 0.5, 0.5, 0.9, 0.9]},
            "fewer unique covariate combinations",
        ),
        ("y ~ s(x, z, k = 8)", {"z": [0.1, 0.5, 0.9]}, "given for every covariate"),
        (
            "y ~ s(x, z, bs = 'gp', k = 8)",
            {"x": [0.1, 0.5, 0.9]},
            "multivariate bs='gp' smooth are not supported",
        ),
        (
            "y ~ t2(x, z, v, d = c(2, 1))",
            {"x": [0.1, 0.5, 0.9]},
            "given for every covariate",
        ),
    ],
)
def test_invalid_duchon_splines_and_knots_are_refused(formula, knots, message):
    train, _ = _frames()
    with formula_data_options(knots), pytest.raises(ValueError, match=message):
        _formula_and_design(formula, train)


def test_large_duchon_spline_needs_knots():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(2001, 2))
    with pytest.raises(ValueError, match="subsamples knots with R's random number"):
        duchon_spline(x, 10, 2, 0.0)
    # User knots replace the subsample, as in mgcv, for both constructors.
    knots = rng.uniform(size=(20, 2))
    assert duchon_spline(x, 10, 2, 0.0, knots=knots).design.shape == (2001, 10)
    assert thin_plate(x, 10, 0, shrink=None, knots=knots).design.shape == (2001, 10)
    with pytest.raises(ValueError, match="subsamples knots above 2000"):
        thin_plate(x, 10, 0, shrink=None)


CASES: dict[str, tuple[str, dict[str, list[float]] | None]] = {
    "duchon": ("y ~ s(x, z, bs = 'ds', k = 10, m = c(1, 0.5)) + v", None),
    "duchon_knots": (
        "y ~ s(x, z, bs = 'ds', k = 8) + v",
        {
            "x": [0.05, 0.3, 0.55, 0.8, 0.95, 0.15, 0.45, 0.7, 0.2, 0.9],
            "z": [-0.4, 1.2, 0.3, 0.8, -0.1, 0.6, 1.4, 0.0, 1.0, 0.4],
        },
    ),
    "thin_plate_knots": (
        "y ~ s(x, k = 6) + t2(z, v, bs = c('tp', 'cr'), k = c(5, 4))",
        {
            "x": np.linspace(0.05, 0.95, 9).tolist(),
            "z": np.linspace(-0.45, 1.45, 8).tolist(),
            "v": [0.05, 0.35, 0.65, 0.95],
        },
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_fit_predict_refit_and_restore(
    case: str, method: Literal["svi", "hmc", "nuts"], tmp_path: Path
):
    train, fresh = _frames()
    fresh = fresh.assign(y=np.linspace(0.0, 1.0, len(fresh)))
    formula, knots = CASES[case]
    if method == "svi":
        fitted = fit(
            formula, data=train, knots=knots, method="svi", seed=2, num_steps=30
        )
    else:
        fitted = fit(
            formula,
            data=train,
            knots=knots,
            method=method,
            seed=2,
            num_warmup=15,
            num_samples=8,
            num_chains=1,
        )
    linpred = fitted.posterior_linpred(new_data=fresh, num_samples=3)
    assert linpred.shape[-1] == len(fresh) and bool(torch.isfinite(linpred).all())
    assert fitted.posterior_epred(new_data=fresh, num_samples=3).shape[-1] == 7
    assert fitted.posterior_predict(new_data=fresh, num_samples=3).shape[-1] == 7
    assert bool(
        torch.isfinite(fitted.log_likelihood(new_data=fresh, num_samples=3)["y"]).all()
    )
    # Training rows given as new data replay the training predictor.
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=train, num_samples=3, seed=1),
        fitted.posterior_linpred(num_samples=3, seed=1),
        rtol=1e-6,
        atol=1e-6,
    )
    term = next(iter(t.term for t in fitted.formula.design_terms))
    effect = conditional_smooths(fitted, term, fresh, num_samples=3, seed=1)
    assert effect.shape[-1] == len(fresh) and bool(torch.isfinite(effect).all())
    path = tmp_path / "smooth.qvrfit"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    torch.testing.assert_close(
        restored.posterior_linpred(new_data=fresh, num_samples=3, seed=1),
        fitted.posterior_linpred(new_data=fresh, num_samples=3, seed=1),
    )
    if method != "svi":
        return
    # A refit on fewer rows rebuilds the basis with the retained knots.
    subset = train.iloc[:30]
    updated = fitted.update(data=subset, num_steps=3)
    plan = updated.prediction_plan
    assert plan is not None
    refit_common = plan.designs["mu"].common
    assert refit_common is not None
    with formula_data_options(knots):
        _, expected = _formula_and_design(formula, subset)
    assert expected.common is not None
    np.testing.assert_allclose(
        np.asarray(refit_common.design_matrix),
        np.asarray(expected.common.design_matrix),
        atol=1e-12,
    )
    if knots is not None:
        with formula_data_options(None):
            _, unknotted = _formula_and_design(formula, subset)
        assert unknotted.common is not None
        assert not np.allclose(
            np.asarray(unknotted.common.design_matrix),
            np.asarray(expected.common.design_matrix),
        )
    assert np.isfinite(fitted.kfold(K=2, seed=1).estimate.elpd)


def test_joint_density_and_derivatives_equal_a_torch_reference():
    """A Duchon and a knotted thin-plate term score as their mixed model."""
    train, _ = _frames()
    knots = {"v": np.linspace(0.05, 0.95, 8).tolist()}
    spec = "y ~ s(x, z, bs = 'ds', k = 7, m = c(1, 0.5)) + s(v, k = 5)"
    with formula_data_options(knots):
        formula = formula_from_data(spec, train)
    duchon, plate = formula.design_terms
    assert duchon.data.shape == (40, 6) and plate.data.shape == (40, 3)
    [column] = formula.fixed_columns[1:]
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    values = lens.fixed_column_observations(formula)
    values["y"] = torch.tensor(
        np.array(formula.response_values, dtype=np.float64), dtype=torch.float64
    )
    point = torch.tensor(
        [0.1, 0.3, 0.8, 0.1, -0.2, 0.3, -0.1, 0.4, 0.2, 0.6, -0.3, 0.5, 0.1, 0.7],
        dtype=torch.float64,
        requires_grad=True,
    )

    def actual(v):
        observations = {
            **values,
            "intercept": v[0],
            f"beta_{column.qvr_name}": v[1],
            duchon.qvr_name + "_scale": v[2],
            duchon.qvr_name + "_z": v[3:9],
            plate.qvr_name + "_scale": v[9],
            plate.qvr_name + "_z": v[10:13],
            "sigma": v[13],
        }
        return trace(program, v.new_zeros(1, 1), observations).log_joint.sum()

    def oracle(v):
        normal = torch.distributions.Normal
        fixed = torch.tensor(np.array(column.data), dtype=torch.float64)
        first = torch.tensor(np.array(duchon.data), dtype=torch.float64)
        second = torch.tensor(np.array(plate.data), dtype=torch.float64)
        mu = v[0] + v[1] * fixed + first @ (v[2] * v[3:9]) + second @ (v[9] * v[10:13])
        return (
            normal(0.0, 5.0).log_prob(v[0])
            + normal(0.0, 5.0 / fixed.square().mean().sqrt()).log_prob(v[1])
            + torch.distributions.HalfNormal(1.0).log_prob(v[2])
            + normal(0.0, 1.0).log_prob(v[3:9]).sum()
            + torch.distributions.HalfNormal(1.0).log_prob(v[9])
            + normal(0.0, 1.0).log_prob(v[10:13]).sum()
            + torch.distributions.HalfCauchy(2.0).log_prob(v[13])
            + normal(mu, v[13]).log_prob(values["y"]).sum()
        )

    torch.testing.assert_close(actual(point), oracle(point))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(oracle, point),
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(oracle, point),
    )
    scales = {t.site: t.class_ for t in lens.prior_targets if t.site.endswith("_scale")}
    assert set(scales.values()) == {"sds"} and len(scales) == 2
