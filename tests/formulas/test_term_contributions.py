"""A term's contribution is its row design times its coefficients."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from quivers._values import ValueModel
from quivers.dsl.ast_nodes import (
    DrawArgNamed,
    DrawArgScalar,
    LetExprVar,
    LetStep,
    ProgramStep,
    SampleStep,
)
from quivers.dsl.emit import _emit_let_expr
from quivers.formulas import BayesianFit, FormulaToQVRModule, fit, families
from quivers.formulas._ast import _let, _var
from quivers.formulas._predictor import HostTerm, OffsetTerm, Predictor
from quivers.formulas._term import (
    AxisDecl,
    Contribution,
    Emission,
    InputDecl,
    OutputAxis,
    SiteDecl,
)
from quivers.formulas._term_emission import (
    ProgramEmission,
    emit_axis_part,
    emit_predictor,
    emit_terms,
)
from quivers.formulas.family import links
from quivers.formulas.formula import FixedColumn

type Array = npt.NDArray[np.float64]

ROWS = 24
X = np.random.default_rng(3).normal(size=ROWS).round(3)
W = np.linspace(0.5, 1.5, ROWS)
FRAME = pd.DataFrame(
    {
        "y": np.random.default_rng(1).normal(size=ROWS).round(3),
        "c": ["A", "B", "C"] * 8,
        "x": X,
        "w": W,
        "z": [0, 1, 2, 3, 2, 1] * 4,
        "g": ["a", "b", "c"] * 8,
        "h": ["a", "c", "b", "b", "a", "c"] * 4,
    }
)
G = np.arange(ROWS) % 3
H = np.array(["abc".index(label) for label in FRAME["h"]])


def _smooth(draw: Mapping[str, Array], data: Mapping[str, Array]) -> Array:
    return data["smooth_s_x_k_5"] @ (
        draw["smooth_s_x_k_5_scale"] * draw["smooth_s_x_k_5_z"]
    )


def _spectral(draw: Mapping[str, Array], data: Mapping[str, Array]) -> Array:
    scale, length = draw["sdgp_gpx"], draw["lscale_gpx"]
    density = (
        scale**2
        * np.sqrt(2 * np.pi)
        * length
        * np.exp(-0.5 * length**2 * data["gp_gpx_frequency_sq_1"])
    )
    return data["gp_gpx"] @ (np.sqrt(density) * draw["gp_gpx_z"])


def _exact(draw: Mapping[str, Array], data: Mapping[str, Array]) -> Array:
    scale, length = draw["sdgp_gpx"], draw["lscale_gpx"]
    distance = data["gp_gpx_distance_1"]
    kernel = scale**2 * np.exp(-0.5 * (distance / length) ** 2)
    factor = np.linalg.cholesky(kernel + 1e-12 * np.eye(len(kernel)))
    return data["gp_gpx"] @ (factor @ draw["gp_gpx_z"])


def _monotonic(draw: Mapping[str, Array], data: Mapping[str, Array]) -> Array:
    steps = np.concatenate([[0.0], np.cumsum(draw["simo_moz1"])])
    return draw["bsp_mo_z"] * 3.0 * steps[np.array(FRAME["z"])]


def _member(draw: Mapping[str, Array], data: Mapping[str, Array]) -> Array:
    levels = draw["sigma_mmgh_Intercept"] * draw["z_mmgh_Intercept"]
    return 0.5 * levels[G] + 0.5 * levels[H]


SCALAR: dict[
    str, tuple[str, Callable[[Mapping[str, Array], Mapping[str, Array]], Array]]
] = {
    "x + offset(w)": ("beta_x", lambda draw, data: draw["beta_x"] * X + W),
    "s(x, k = 5)": (
        "smooth",
        lambda draw, data: (
            sum(
                (
                    draw[name] * data[name.removeprefix("beta_")]
                    for name in draw
                    if name.startswith("beta_")
                ),
                start=np.zeros(ROWS),
            )
            + _smooth(draw, data)
        ),
    ),
    "offset(w) + mo(z)": (
        "offset_monotonic",
        lambda draw, data: W + _monotonic(draw, data),
    ),
    "offset(w) + gp(x)": (
        "offset_exact",
        lambda draw, data: W + _exact(draw, data),
    ),
    "gp(x, k = 5, c = 1.25)": ("spectral", _spectral),
    "gp(x)": ("exact", _exact),
    "mo(z)": ("monotonic", _monotonic),
    "(0 + x | g)": (
        "group",
        lambda draw, data: (draw["sigma_g_x"] * draw["z_g_x"])[G] * X,
    ),
    "(1 | mm(g, h))": ("membership", _member),
}


def _fit(formula: str, family: str) -> BayesianFit:
    fitted = fit(formula, data=FRAME, family=family, method="svi", num_steps=2, seed=0)
    assert isinstance(fitted, BayesianFit)
    return fitted


def _draws(fitted: BayesianFit, index: int) -> dict[str, Array]:
    return {
        name: value[0, index].double().numpy()
        for name, value in fitted.draws(num_samples=2, seed=1).items()
    }


@pytest.mark.parametrize("terms", SCALAR)
def test_scalar_contributions(terms):
    """The linear predictor is the intercept plus the term's contribution."""
    fitted = _fit(f"y ~ {terms}", "gaussian")
    data = {name: value.double().numpy() for name, value in fitted.observations.items()}
    value = fitted.posterior_linpred(num_samples=2, seed=1)
    for index in range(2):
        draw = _draws(fitted, index)
        expected = draw["intercept"] + SCALAR[terms][1](draw, data)
        np.testing.assert_allclose(
            value[0, index].double().numpy(), expected, rtol=1e-4, atol=1e-4
        )


def test_component_contributions():
    """Every coordinate has its own coefficients over the same row designs."""
    fitted = _fit("c ~ x + offset(w) + s(x, k = 5) + (0 + x || g)", "categorical")
    data = {name: value.double().numpy() for name, value in fitted.observations.items()}
    value = fitted.posterior_linpred(num_samples=2, seed=1)
    for index in range(2):
        draw = _draws(fitted, index)
        for component in range(2):
            coefficients = draw["smooth_s_x_k_5_z"].reshape(-1, 2)[:, component]
            expected = (
                draw["intercept"][component]
                + W
                + sum(
                    draw[name][component] * data[name.removeprefix("beta_")]
                    for name in draw
                    if name.startswith("beta_") and draw[name].shape == (2,)
                )
                + data["smooth_s_x_k_5"]
                @ (draw["smooth_s_x_k_5_scale"][component] * coefficients)
                + draw["sigma_g_x"][component]
                * draw["z_g_x"].reshape(3, 2)[G, component]
                * X
            )
            np.testing.assert_allclose(
                value[0, index, :, component + 1].double().numpy()
                if value.shape[-1] == 3
                else value[0, index, :, component].double().numpy(),
                expected,
                rtol=1e-4,
                atol=1e-4,
            )


def _column(name: str) -> FixedColumn:
    return FixedColumn(term=name, name=name, qvr_name=name, data=np.zeros(2))


class _Stub(ValueModel):
    """A term with one coefficient and no design."""

    name: str
    output: OutputAxis | None = None

    @property
    def label(self) -> str:
        return self.name

    def contributions(self) -> tuple[Contribution, ...]:
        return (
            Contribution(
                name=self.name + "_per_row",
                rows=False,
                coordinates=self.output is not None,
            ),
        )

    def axes(self) -> tuple[AxisDecl, ...]:
        return ()

    def inputs(self) -> tuple[InputDecl, ...]:
        return ()

    def sites(self) -> tuple[SiteDecl, ...]:
        return ()

    def catalog(self) -> None:
        return None

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        return (
            emission.draw(
                SiteDecl(
                    name=self.name,
                    class_="b",
                    default="Normal(0.0, 6.0)",
                    divisor=2.0,
                )
            ),
            _let(self.name + "_per_row", _var(self.name)),
        )


def _emission(**options) -> ProgramEmission:
    lens = FormulaToQVRModule(families["gaussian"], **options)
    return ProgramEmission(lens, [], {})


def test_predictors_sum_the_terms_of_each_part_in_order():
    """The steps of a part precede its sum, and an empty part is zero."""
    axis = OutputAxis(name="K", size=2)
    predictor = Predictor(
        parameter="mu",
        role="dpar",
        prefix="",
        link=links["identity"],
        terms=(
            _Stub(name="a"),
            _Stub(name="vector", output=axis),
            _Stub(name="b"),
        ),
    )
    steps: list[ProgramStep] = []
    value = emit_predictor(predictor, _emission(), steps)
    assert _emit_let_expr(value) == "a_per_row + b_per_row"
    assert [
        step.vars[0] if isinstance(step, SampleStep) else step.name
        for step in steps
        if isinstance(step, SampleStep | LetStep)
    ] == ["a", "a_per_row", "b", "b_per_row"]
    emit_axis_part(predictor, axis, _emission(), steps, "eta")
    last = steps[-1]
    assert isinstance(last, LetStep)
    assert last.name == "eta"
    assert _emit_let_expr(last.value) == (
        "factor response : Resp, output_coordinate : K in "
        "vector_per_row[output_coordinate]"
    )
    empty = predictor.with_(terms=())
    assert _emit_let_expr(emit_predictor(empty, _emission(), [])) == "0.0"


def test_default_scales_follow_the_design_and_explicit_priors_do_not():
    """Only a default prior is divided by the column's root mean square."""
    for priors, scale in ((None, 3.0), ({"a": "Normal(0.0, 6.0)"}, 6.0)):
        steps: list[ProgramStep] = []
        emit_terms([_Stub(name="a")], _emission(user_priors=priors), steps)
        (draw,) = [step for step in steps if isinstance(step, SampleStep)]
        values = {
            item.parameter: item.value.value
            for item in draw.args or ()
            if isinstance(item, DrawArgNamed) and isinstance(item.value, DrawArgScalar)
        }
        assert values == {"loc": 0.0, "scale": scale}


def test_contributions_are_indexed_by_the_axes_they_vary_over():
    """Offsets are shared by coordinates, and host values are stored by row."""
    axis = OutputAxis(name="K", size=2)
    row, component = LetExprVar(name="n"), LetExprVar(name="k")
    offsets = OffsetTerm(
        prefix="p_", columns=(_column("first"), _column("second")), output=axis
    )
    host = HostTerm(name="net", output=axis)
    cells = [
        _emit_let_expr(summand.cell(row, component))
        for term in (offsets, host, _Stub(name="t", output=axis), _Stub(name="s"))
        for summand in term.contributions()
    ]
    assert cells == [
        "p_formula_offset_0[n]",
        "p_formula_offset_1[n]",
        "net_by_coordinate[k, n]",
        "t_per_row[k]",
        "s_per_row",
    ]
    (binding,) = host.steps(_emission())
    assert isinstance(binding, LetStep)
    assert _emit_let_expr(binding.value) == (
        "factor output_coordinate : K, response : Resp in "
        "net[response * 2 + output_coordinate]"
    )


def test_an_unplanned_group_coefficient_is_refused():
    """A coefficient no program emits cannot be read."""
    with pytest.raises(ValueError, match="no program emits"):
        _emission().gathered(("", "g", "Intercept"), None)
