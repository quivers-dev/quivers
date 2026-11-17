"""mgcv smooth bases equal brms's Stan data and its fresh-row matrices.

The oracle ``oracles/mgcv_smooths.json`` is written by ``oracles/mgcv_smooths.R``
with brms 2.20.4 and mgcv 1.8-42. For each smooth it records the training
unpenalized design ``Xs`` and penalized designs ``Zs`` (checked inside the
script against ``make_standata``) and the matrices brms evaluates on fresh
rows. Columns are compared up to sign, because every reparameterization
step brms runs fixes each column only up to sign. Unpenalized columns are
also compared up to order: brms orders them by the rounding noise of the
numerically zero penalty eigenvalues, and their population coefficients are
exchangeable under every isotropic or flat prior.

Tolerance: thin-plate bases come from mgcv's Lanczos eigensolver, stopped at
a residual of ``eps ** 0.7`` (about 1e-11) relative to the largest
eigenvalue; eigenvector error is that residual divided by the relative
eigengap, which exceeds 1e-2 for every basis here. A relative tolerance of
1e-9 of each matrix's largest entry therefore covers solver error with two
orders of margin while rejecting any change of basis, constraint, or scale.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

import quivers.formulas._mgcv_smooth as mgcv_smooth
from quivers.formulas.formula import _formula_and_design
from quivers.formulas._smooth import SmoothTransform

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "mgcv_smooths.json").read_text()
)
RTOL = 1e-9
ROUNDING_DEPENDENT = frozenset({"s(x, bs = 'tp', m = 3, k = 8)", "s(x, z, v, k = 15)"})


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(ORACLE["train"])
    fresh = pd.DataFrame(ORACLE["fresh"])
    levels = sorted(set(train["g"]))
    train["g"] = pd.Categorical(train["g"], categories=levels)
    fresh["g"] = pd.Categorical(fresh["g"], categories=levels)
    return train, fresh


def _matrix(rows: list, count: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(count, -1)


def _assert_columns_up_to_sign(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    if not expected.size:
        return
    signs = np.sign(np.sum(actual * expected, axis=0))
    signs[signs == 0] = 1.0
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual * signs, expected, rtol=0, atol=RTOL * scale)


def _assert_columns_up_to_signed_order(
    actual: np.ndarray, expected: np.ndarray
) -> None:
    assert actual.shape == expected.shape
    if not expected.size:
        return
    unit_a = actual / np.linalg.norm(actual, axis=0)
    unit_e = expected / np.linalg.norm(expected, axis=0)
    _, order = linear_sum_assignment(-np.abs(unit_e.T @ unit_a))
    _assert_columns_up_to_sign(actual[:, order], expected)


def _assert_same_span(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    left = np.linalg.qr(actual)[0]
    right = np.linalg.qr(expected)[0]
    np.testing.assert_allclose(left @ left.T, right @ right.T, rtol=0, atol=RTOL)


def _smooth(spec: str, train: pd.DataFrame):
    formula, design = _formula_and_design("y ~ " + spec, train)
    [(name, term)] = [
        (name, term)
        for name, term in design.common.terms.items()
        if name != "Intercept"
    ]
    transform = term.components[0].call.stateful_transform
    assert isinstance(transform, SmoothTransform)
    return formula, design, name, transform.state


@pytest.mark.parametrize("entry", ORACLE["entries"], ids=lambda e: e["spec"])
def test_training_and_fresh_designs_equal_brms(entry):
    train, fresh = _frames()
    formula, design, name, state = _smooth(entry["spec"], train)
    columns = design.common.slices[name]
    trained = np.asarray(design.common.design_matrix)[:, columns]
    evaluated = np.asarray(design.common.evaluate_new_data(fresh).design_matrix)[
        :, columns
    ]
    assert len(state.layout) == len(entry["blocks"])
    for layout, block in zip(state.layout, entry["blocks"], strict=True):
        for values, prefix, rows in (
            (trained, "", len(train)),
            (evaluated, "fresh_", len(fresh)),
        ):
            expected = block[prefix + "fixed"]
            compare = (
                _assert_same_span
                if entry["spec"] in ROUNDING_DEPENDENT
                else _assert_columns_up_to_signed_order
            )
            compare(
                values[:, list(layout.fixed)],
                _matrix(expected, rows) if expected else np.zeros((rows, 0)),
            )
            assert len(layout.random) == len(block[prefix + "random"])
            for group, expected in zip(
                layout.random, block[prefix + "random"], strict=True
            ):
                _assert_columns_up_to_sign(
                    values[:, list(group)], _matrix(expected, rows)
                )
    fixed = [column for column in formula.fixed_columns if column.term == name]
    assert len(fixed) == sum(len(layout.fixed) for layout in state.layout)
    blocks = [term for term in formula.design_terms if term.term == name]
    assert [term.column_indices for term in blocks] == [
        group for layout in state.layout for group in layout.random
    ]


def test_fresh_rows_reproduce_training_rows_exactly_through_retained_state():
    train, _ = _frames()
    for entry in ORACLE["entries"]:
        _, design, name, _ = _smooth(entry["spec"], train)
        columns = design.common.slices[name]
        replay = np.asarray(design.common.evaluate_new_data(train).design_matrix)
        np.testing.assert_allclose(
            replay[:, columns],
            np.asarray(design.common.design_matrix)[:, columns],
            rtol=0,
            atol=1e-10,
        )


def test_another_valid_null_space_basis_is_detected(monkeypatch):
    """mgcv's t2 matrices depend on LAPACK's null-space basis; a rotation fails."""
    entry = next(e for e in ORACLE["entries"] if e["spec"] == "t2(x, z)")
    exact = mgcv_smooth._canonical_null

    def rotated(design, vectors, rank):
        fixed = exact(design, vectors, rank)
        if fixed.shape[1] - rank < 2:
            return fixed
        angle = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
        out = fixed.copy()
        out[:, rank : rank + 2] = fixed[:, rank : rank + 2] @ angle
        return out

    monkeypatch.setattr(mgcv_smooth, "_canonical_null", rotated)
    with pytest.raises(AssertionError):
        test_training_and_fresh_designs_equal_brms(entry)
