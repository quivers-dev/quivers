"""Independent formulas retain strict checks when density terms cancel."""

from __future__ import annotations

import math

import pytest
import torch
from torch import distributions as td

from tests.transpile._analytic_references import kumaraswamy_terms
from tests.transpile.test_gallery_numeric_equivalence import reference_pin_atol
from tests.transpile.test_oracle_reference_strength import _fixture


_EXAMPLE = "kumaraswamy_bounded_outcome"


@pytest.mark.parametrize("index", range(6))
def test_analytic_reference_matches_every_compiled_site(index: int) -> None:
    """All four sites and the joint agree at the existing eight-ULP budget."""
    fixture = _fixture(_EXAMPLE)
    terms = kumaraswamy_terms(fixture.dataset, fixture.points[index])
    assert set(terms) | {"eta", "a"} == set(fixture.sites[index])
    # Deterministic let bindings are point masses and add no log density.
    assert fixture.sites[index]["eta"] == 0.0
    assert fixture.sites[index]["a"] == 0.0
    for name, expected in terms.items():
        value = float(expected.item())
        assert abs(fixture.sites[index][name] - value) <= reference_pin_atol(value)
    expected_joint = float(sum(terms.values()).item())
    assert abs(fixture.joints[index] - expected_joint) <= reference_pin_atol(
        expected_joint
    )


@pytest.mark.parametrize("index", range(6))
def test_analytic_reference_agrees_with_scalar_closed_form(index: int) -> None:
    """Check the distribution calls against the density on (0, 1)."""
    fixture = _fixture(_EXAMPLE)
    point = fixture.points[index]
    # JSON values retain the fixture's represented float32 numbers.
    b, g0, g1 = (
        float(torch.as_tensor(point.params[name], dtype=torch.float64).item())
        for name in ("b", "gamma_0", "gamma_1")
    )
    likelihood = []
    for x, y in zip(point.data["x"], point.data["y"], strict=True):
        a = math.exp(g0 + g1 * x)
        likelihood.append(
            math.log(a)
            + math.log(b)
            + (a - 1) * math.log(y)
            + (b - 1) * math.log1p(-(y**a))
        )
    expected = {
        "b": math.log(2)
        - math.log(2)
        - 0.5 * math.log(2 * math.pi)
        - 0.5 * (b / 2) ** 2,
        "gamma_0": -0.5 * (math.log(2 * math.pi) + g0**2),
        "gamma_1": -0.5 * (math.log(2 * math.pi) + g1**2),
        "y": math.fsum(likelihood),
    }
    terms = kumaraswamy_terms(fixture.dataset, point, dtype=torch.float64)
    for name, value in expected.items():
        assert float(terms[name]) == pytest.approx(value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "defect", ("prior_dropped", "shapes_transposed", "data_term_dropped")
)
def test_analytic_reference_rejects_wrong_densities(defect: str) -> None:
    """The reference detects missing terms and a proper but different law."""
    fixture = _fixture(_EXAMPLE)
    gaps = []
    for index, point in enumerate(fixture.points):
        terms = kumaraswamy_terms(fixture.dataset, point)
        b, g0, g1 = (
            torch.as_tensor(point.params[name]) for name in ("b", "gamma_0", "gamma_1")
        )
        x, y = (torch.as_tensor(point.data[name]) for name in ("x", "y"))
        a = (g0 + g1 * x).exp()
        if defect == "prior_dropped":
            terms["b"] = torch.zeros_like(terms["b"])
        elif defect == "shapes_transposed":
            terms["y"] = td.Kumaraswamy(b, a).log_prob(y).sum()
        else:
            # Omit the density's (a-1) log y term.
            terms["y"] = terms["y"] - ((a - 1) * y.log()).sum()
        wrong = float(sum(terms.values()).item())
        expected = fixture.joints[index]
        gap = abs(wrong - expected)
        assert gap > reference_pin_atol(expected)
        gaps.append(gap)
    assert max(gaps) > 0.1
