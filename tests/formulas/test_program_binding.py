"""How a program's data, response and parameters are bound, and the refusals.

Oracles: the program text itself. Data the program reads at each row are data
columns of the same name; data on another axis and scalars come from
``data2``; index maps are integer columns; the response carries the carrier
its observation declares. Each refusal is matched on its reason, and each is
paired with the closest call that binds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quivers.formulas import fit, formula_to_qvr, get_prior

from tests.formulas._program_cases import (
    AROUND_ROWS,
    DECAY,
    GAUSS,
    HALFWIDTH,
    frame,
)


def _emit(source: str, parameters: dict[str, str], response: str, **extra):
    keywords = {"data": frame(), "data2": {"halfwidth": HALFWIDTH}} | extra
    return formula_to_qvr(
        program=source, parameters=parameters, response=response, **keywords
    )


def test_inputs_are_bound_by_role_axis_and_carrier() -> None:
    fitted = fit(
        program=AROUND_ROWS,
        parameters={"noise": "~ x"},
        response="accept",
        data=frame(),
        data2={"halfwidth": HALFWIDTH},
        method="svi",
        num_steps=1,
        seed=0,
    )
    observations = fitted.observations
    rows = frame()
    torch.testing.assert_close(
        observations["dist"], torch.as_tensor(rows["dist"].to_numpy())
    )
    assert observations["dist"].dtype == torch.float64
    assert observations["trial_idx"].dtype == torch.int64
    assert observations["accept"].dtype == torch.bool
    torch.testing.assert_close(observations["halfwidth"], torch.as_tensor(HALFWIDTH))
    # The row axis is the rows of the data; the other axes keep their sizes.
    source = _emit(AROUND_ROWS, {"noise": "~ x"}, "accept")
    assert "object Resp : FinSet 12" in source
    assert "object Res : FinSet 3" in source
    assert "Trial" not in source
    assert "trial_idx : Tensor[Int]([Resp])" in source


def test_rows_with_a_missing_program_column_are_left_out() -> None:
    rows = frame()
    rows.loc[3, "dist"] = np.nan
    fitted = fit(
        program=AROUND_ROWS,
        parameters={},
        response="accept",
        data=rows.assign(trial_idx=np.arange(12)),
        data2={"halfwidth": HALFWIDTH},
        method="svi",
        num_steps=1,
        seed=0,
    )
    assert fitted.observations["dist"].shape == (11,)
    assert fitted.included_rows == tuple(index for index in range(12) if index != 3)


@pytest.mark.parametrize(
    ("source", "parameters", "response", "extra", "reason"),
    [
        (GAUSS, {"nope": "~ x"}, "y", {}, r"no parameter 'nope'.*\['mu', 'sigma'\]"),
        (GAUSS, {"y": "~ x"}, "y", {}, "'y' is the response"),
        (
            AROUND_ROWS,
            {"r": "~ x"},
            "accept",
            {},
            "'r' is not drawn at the program's top level",
        ),
        (GAUSS, {"mu": "~ x"}, "nope", {}, r"does not observe 'nope'.*\['y'\]"),
        (
            DECAY,
            {"a": "~ 1"},
            "y",
            {"data": frame().drop(columns="x")},
            "reads 'x' at each row, and the data has no column 'x'",
        ),
        (
            AROUND_ROWS,
            {"noise": "~ x"},
            "accept",
            {"data": frame().assign(halfwidth=1.0), "data2": {}},
            "'halfwidth', which is not indexed by its rows; give it in data2; it is "
            "a column of the data",
        ),
        (AROUND_ROWS, {"noise": "~ x"}, "accept", {"data2": {}}, "give it in data2"),
        (
            GAUSS,
            {"mu": "~ x"},
            "y | se(s)",
            {"data": frame().assign(s=1.0)},
            r"se\(\) is not supported for family 'gauss'",
        ),
        (
            GAUSS,
            {"mu": "~ x"},
            "y",
            {"weights": "w", "data": frame().assign(w=1.0)},
            "row_weights",
        ),
        (GAUSS, {"mu": "~ x"}, "y", {"family": "poisson"}, "family= does not apply"),
        (GAUSS, {"mu": "~ x"}, None, {}, "needs response="),
    ],
)
def test_refusals_state_their_reason(
    source, parameters, response, extra, reason
) -> None:
    with pytest.raises(ValueError, match=reason):
        _emit(source, parameters, response, **extra)


def test_a_path_is_not_a_program() -> None:
    with pytest.raises(TypeError, match="takes the QVR source text"):
        _emit(("gauss.qvr", "gauss"), {"mu": "~ x"}, "y")  # type: ignore[arg-type]


def test_a_program_that_couples_its_rows_is_refused() -> None:
    coupled = GAUSS.replace(
        "program gauss : Trial -> Val",
        "program gauss(w : Tensor[Real]([Trial, Trial])) : Trial -> Val",
    )
    with pytest.raises(ValueError, match="uses its row axis 'Trial' twice"):
        _emit(coupled, {"mu": "~ x"}, "y")


def test_a_second_observation_is_refused() -> None:
    two = GAUSS.replace(
        "    return mu",
        "    observe y2 : Trial <- Normal(loc=mu, scale=sigma)\n    return mu",
    )
    with pytest.raises(ValueError, match=r"observes \['y2'\] besides the response"):
        _emit(two, {"mu": "~ x"}, "y")


def test_static_parameters_are_refused() -> None:
    static = GAUSS.replace("program gauss :", "program gauss[k : Nat] :")
    with pytest.raises(ValueError, match="static parameters"):
        _emit(static, {"mu": "~ x"}, "y")


def test_priors_follow_the_formula_conventions() -> None:
    priors = get_prior(
        program=GAUSS,
        parameters={"mu": "~ x + (1 | g)", "sigma": "~ z"},
        response="y",
        data=frame(),
    )
    assert set(priors) == {
        "intercept",
        "beta_x",
        "sigma_g_Intercept",
        "dpar_sigma_intercept",
        "dpar_sigma_beta_z",
    }
    # The program's own sites are not overridable: the program states them.
    left = get_prior(
        program=GAUSS, parameters={"mu": "~ x"}, response="y", data=frame()
    )
    assert "sigma" not in left
    with pytest.raises(ValueError, match="unused prior overrides"):
        get_prior(
            program=GAUSS,
            parameters={"mu": "~ x"},
            response="y",
            data=frame(),
            priors={"sigma": "HalfNormal(2.0)"},
        )
