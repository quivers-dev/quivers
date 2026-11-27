"""How a program's data, response and parameters are bound, and the refusals.

Oracles: the program text itself. Data the program reads at each row are data
columns of the same name; data on another axis and scalars come from
``data2``; index maps are integer columns; the response carries the carrier
its observation declares. Each refusal is matched on its reason, and each is
paired with the closest call that binds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from quivers.formulas import fit, formula_to_qvr, get_prior

from tests.formulas._program_cases import (
    AROUND_ROWS,
    DECAY,
    GAUSS,
    HALFWIDTH,
    MIX2,
    frame,
    log_joint,
    pin,
    run,
)


def _emit(
    source: str | tuple[str | Path, str],
    parameters: dict[str, str],
    response: str,
    **extra,
):
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
            {"mu": "~ x", "sigma": "~ z"},
            "y | se(s)",
            {"data": frame().assign(s=1.0)},
            r"se\(\) without sigma=True excludes a separate residual sigma",
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


def test_a_program_file_selects_the_named_program(tmp_path: Path) -> None:
    other = GAUSS.replace("gauss", "other").replace("observe y", "observe z")
    path = tmp_path / "programs.qvr"
    path.write_text(
        GAUSS + "".join(other.splitlines(keepends=True)[3:]), encoding="utf-8"
    )
    assert _emit((path, "gauss"), {"mu": "~ x"}, "y") == _emit(
        GAUSS, {"mu": "~ x"}, "y"
    )
    assert _emit((str(path), "other"), {"mu": "~ x"}, "z") == _emit(
        other, {"mu": "~ x"}, "z"
    )
    with pytest.raises(ValueError, match="no program 'missing'"):
        _emit((str(path), "missing"), {"mu": "~ x"}, "y")


def test_weights_do_not_raise_the_conditional_scores_of_a_marginalization() -> None:
    with pytest.raises(ValueError, match="after the marginalization"):
        _emit(MIX2, {}, "y", weights="w", data=frame().assign(w=1.0))


def test_a_program_that_couples_its_rows_is_refused() -> None:
    coupled = GAUSS.replace(
        "program gauss : Trial -> Val",
        "program gauss(w : Tensor[Real]([Trial, Trial])) : Trial -> Val",
    )
    with pytest.raises(ValueError, match="uses its row axis 'Trial' twice"):
        _emit(coupled, {"mu": "~ x"}, "y")


def test_a_second_observation_conditions_the_named_response() -> None:
    source = GAUSS.replace(
        "    observe y : Trial <- Normal(loc=mu, scale=sigma)",
        "    observe y2 : Trial <- Normal(loc=mu, scale=sigma)\n"
        "    observe y : Trial <- Normal(loc=y2, scale=sigma)",
    )
    rows = frame().assign(y=np.arange(-6, 6) / 8, y2=np.arange(12) / 4)
    fitted = fit(
        program=source,
        parameters={"mu": "~ 1"},
        response="y",
        data=rows,
        method="svi",
        num_steps=1,
        seed=0,
    )
    location = torch.tensor(0.5, dtype=torch.float64)
    scale = torch.tensor(1.25, dtype=torch.float64)
    values = {"intercept": location, "sigma": scale}
    normal = torch.distributions.Normal(location, scale)
    y = torch.tensor(rows["y"].to_numpy())
    y2 = torch.tensor(rows["y2"].to_numpy())
    prior = torch.distributions.Normal(0.0, 5.0).log_prob(location) + (
        torch.distributions.HalfNormal(1.0).log_prob(scale)
    )
    first = torch.distributions.Normal(y2, scale).log_prob(y)
    joint = log_joint(run(fitted, values))
    assert joint == pytest.approx(
        float(prior + first.sum() + normal.log_prob(y2).sum()), rel=1e-10
    )
    assert abs(joint - float(prior + first.sum())) > 1.0
    draws = {name: value.numpy()[None] for name, value in values.items()}
    pinned = pin(fitted, draws)
    scores = pinned.log_likelihood()
    assert set(scores) == {"y"}
    torch.testing.assert_close(scores["y"].reshape(-1), first)
    fresh = rows.iloc[:4].drop(columns="y")
    expected = pinned.posterior_epred(new_data=fresh).reshape(-1).double()
    torch.testing.assert_close(expected, y2[:4])
    with pytest.raises(ValueError, match="data has no column 'y2'"):
        _emit(source, {"mu": "~ x"}, "y", data=frame())


@pytest.mark.parametrize("kind", ["Nat", "FinSet"])
def test_static_extents_specialize_the_program(kind: str) -> None:
    static = (
        GAUSS.replace("program gauss :", f"program gauss[k : {kind}] :")
        .replace("HalfNormal(scale=1.0)", "HalfNormal(scale=1.0) [over=k]")
        .replace("    observe y", "    let first_sigma = sigma[0]\n    observe y")
        .replace("scale=sigma)", "scale=first_sigma)")
    )
    with pytest.raises(ValueError, match="static parameters"):
        _emit(static, {"mu": "~ x"}, "y")
    with pytest.raises(ValueError, match="nonnegative integer"):
        _emit(static, {"mu": "~ x"}, "y", static={"k": -1})
    for size in (2, 3):
        source = _emit(static, {"mu": "~ x"}, "y", static={"k": size})
        explicit = static.replace(f"[k : {kind}]", "").replace("[over=k]", "[over=K]")
        explicit = f"object K : FinSet {size}\n" + explicit
        expected = _emit(explicit, {"mu": "~ x"}, "y")
        assert source.replace("__qvr_gauss_k", "K") == expected


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
