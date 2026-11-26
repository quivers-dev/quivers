"""The planned law of each block of group-level coefficients.

`plan_group_blocks` is the one description of a model's group blocks. These
tests pin the identities, sites and factors it declares for each combination
of level covariance, coefficient correlation, scale indexing, mixing, prior
weight and row map, and check that the emitted program draws exactly the
sites the laws declare.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl.ast_nodes import ProgramDecl, SampleStep
from quivers.formulas import FormulaToQVRModule, families, formula_from_data
from quivers.formulas._group_covariance import GroupCovariance
from quivers.formulas._group_law import (
    column_deviations,
    GroupBlockLaw,
    IdentityCovariance,
    IndependentCoordinates,
    KnownCovariance,
    LKJCorrelation,
    MembershipRows,
    SingleLevelRows,
)

Parameterization = Literal["centered", "noncentered"]

COVARIANCE = GroupCovariance(
    levels=("a", "b", "c"),
    matrix=((1.0, 0.3, 0.1), (0.3, 1.0, 0.2), (0.1, 0.2, 1.0)),
)
DEFICIENT = GroupCovariance(
    levels=("a", "b", "c"),
    matrix=((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    rank=2,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = 12
    groups = ["a", "b", "c"] * 4
    return pd.DataFrame(
        {
            "y": rng.normal(size=rows),
            "o": [0, 1, 2, 3] * 3,
            "k3": ["A", "B", "C"] * 4,
            "x": rng.normal(size=rows),
            "g": groups,
            "h": ["u", "v", "w", "z"] * 3,
            "fb": [{"a": "m", "b": "m", "c": "n"}[level] for level in groups],
            "pw": [{"a": 0.5, "b": 1.0, "c": 1.5}[level] for level in groups],
        }
    )


def _laws(
    text: str,
    *,
    family: str = "gaussian",
    parameterization: Parameterization = "noncentered",
    covariance: GroupCovariance | None = None,
    distributional: dict[str, str] | None = None,
) -> tuple[FormulaToQVRModule, tuple[GroupBlockLaw, ...]]:
    frame = _frame()
    lens = FormulaToQVRModule(
        families[family],
        reparameterize=parameterization,
        group_covariances={"g": covariance} if covariance is not None else None,
        distributional={
            name: formula_from_data(f"y {value}", frame)
            for name, value in (distributional or {}).items()
        },
    )
    lens.forward(formula_from_data(text, frame))
    return lens, lens.group_laws


def _sampled(lens: FormulaToQVRModule, text: str, family: str) -> set[str]:
    module, _ = lens.forward(formula_from_data(text, _frame()))
    return {
        step.vars[0]
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, SampleStep)
    }


def test_independent_scalar_coefficient_is_an_inline_identity_block():
    _, (law,) = _laws("y ~ x + (1 | g)")
    assert law.placement == "inline"
    assert law.identity == "independent_g_Intercept"
    assert isinstance(law.level.covariance, IdentityCovariance)
    assert isinstance(law.correlation, IndependentCoordinates)
    assert law.coordinate_axis is None
    (member,) = law.members
    assert (member.coefficient, member.raw_site) == ("alpha_g", "z_g_Intercept")
    assert member.scale.site == "sigma_g_Intercept"
    assert member.scale.by_axis is None and member.scale.component_axis is None
    assert member.rows == SingleLevelRows(index="g_idx")


def test_centered_independent_coefficient_draws_the_coefficient():
    _, (law,) = _laws("y ~ x + (0 + x | g)", parameterization="centered")
    (member,) = law.members
    assert member.raw_site == member.coefficient == "beta_g_x"
    assert member.rows == SingleLevelRows(index="g_idx", covariate="random_x")


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_correlated_coefficients_form_one_joint_lkj_block(
    parameterization: Parameterization,
):
    _, (law,) = _laws("y ~ x + (1 + x | g)", parameterization=parameterization)
    assert law.placement == "joint"
    assert law.correlation == LKJCorrelation(
        factors=(f"cor_{law.identity}",),
        axis=f"FormulaCorrelation_{law.identity}",
        size=2,
    )
    assert law.coordinate_axis == f"FormulaCorrelation_{law.identity}"
    assert [member.offset for member in law.members] == [0, 1]
    raw = [member.raw_site for member in law.members]
    if parameterization == "centered":
        assert raw == [f"group_coefficients_{law.identity}"] * 2
    else:
        assert raw == ["z_g_Intercept", "z_g_x"]
    assert law.layout.correlation_site == f"cor_{law.identity}"


def test_one_scalar_coordinate_with_an_identifier_is_independent():
    _, (law,) = _laws("y ~ x + (1 | p | g)")
    assert law.placement == "inline"
    assert isinstance(law.correlation, IndependentCoordinates)


def test_shared_identifier_joins_location_and_scale_coefficients():
    _, (law,) = _laws(
        "y ~ x + (1 | p | g)", distributional={"sigma": "~ 1 + (1 | p | g)"}
    )
    assert law.identity == "shared_p"
    assert [member.prefix for member in law.members] == ["", "dpar_sigma_"]
    assert law.correlation == LKJCorrelation(
        factors=("cor_shared_p",), axis="FormulaCorrelation_shared_p", size=2
    )


@pytest.mark.parametrize("covariance", [COVARIANCE, DEFICIENT])
@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_known_covariance_blocks_share_one_support_site(
    covariance: GroupCovariance, parameterization: Parameterization
):
    _, laws = _laws(
        "y ~ x + (1 + x | g) + (1 | h)",
        covariance=covariance,
        parameterization=parameterization,
    )
    by_group = {law.group: law for law in laws}
    known = by_group["g"]
    assert isinstance(known.level.covariance, KnownCovariance)
    assert known.level.covariance.basis.rank == (covariance.rank or 3)
    assert known.placement == "joint"
    assert {member.raw_site for member in known.members} == {
        f"group_covariance_{known.identity}"
    }
    assert isinstance(known.correlation, LKJCorrelation)
    assert isinstance(by_group["h"].level.covariance, IdentityCovariance)
    assert by_group["h"].placement == "inline"


def test_uncorrelated_known_covariance_coefficient_is_its_own_block():
    _, laws = _laws("y ~ x + (1 + x || g)", covariance=COVARIANCE)
    assert [law.identity for law in laws] == [
        "independent_g_Intercept",
        "independent_g_x",
    ]
    assert all(isinstance(law.correlation, IndependentCoordinates) for law in laws)
    assert all(law.coordinate_axis is not None for law in laws)


def test_by_student_and_prior_weight_are_declared_level_factors():
    _, (law,) = _laws("y ~ x + (1 + x | gr(g, by = fb, dist = 'student', pw = pw))")
    level = law.level
    assert level.by is not None and level.by.variable == "fb"
    assert level.by.levels == ("m", "n") and level.by.codes == (0, 0, 1)
    assert level.mixing is not None
    assert (level.mixing.df_site, level.mixing.mixing_site) == ("df_g", "udf_g")
    assert level.prior_weight is not None
    assert level.prior_weight.weights == (0.5, 1.0, 1.5)
    assert law.correlation == LKJCorrelation(
        factors=(f"cor_{law.identity}_by1", f"cor_{law.identity}_by2"),
        axis=f"FormulaCorrelation_{law.identity}",
        size=2,
        cell_axis=level.by.axis,
    )
    assert all(member.scale.by_axis == level.by.axis for member in law.members)


def test_multi_membership_is_a_row_map():
    _, (law,) = _laws("y ~ x + (1 | mm(g, h))")
    (member,) = law.members
    rows = member.rows
    assert isinstance(rows, MembershipRows)
    assert rows.indices == ("mmgh_idx_member1", "mmgh_idx_member2")
    assert rows.covariates == (
        "mmgh_Intercept_member1",
        "mmgh_Intercept_member2",
    )


def test_category_specific_coefficient_has_the_transition_axis():
    _, (law,) = _laws("o ~ x + (0 + cs(x) || g)", family="acat")
    (member,) = law.members
    assert law.placement == "inline"
    assert member.axis == law.coordinate_axis == "FormulaCutpoint"
    assert member.width == 3 and member.scale.component_axis == "FormulaCutpoint"


def test_component_valued_predictor_has_the_category_axis():
    _, laws = _laws("k3 ~ x + (1 + x | g)", family="categorical")
    (law,) = laws
    assert law.placement == "joint" and law.width == 4
    assert [member.axis for member in law.members] == ["FormulaCategoryLogit"] * 2
    assert [member.offset for member in law.members] == [0, 2]
    correlation = law.correlation
    assert isinstance(correlation, LKJCorrelation)
    assert (correlation.size, correlation.components) == (2, 2)
    assert len(correlation.factors) == 2


def test_categories_are_independent_unless_a_term_has_an_identifier():
    """brms draws one block per category without an identifier."""
    _, (single,) = _laws("k3 ~ x + (1 | g)", family="categorical")
    assert isinstance(single.correlation, IndependentCoordinates)
    assert single.placement == "inline"
    _, (shared,) = _laws("k3 ~ x + (1 | p | g)", family="categorical")
    correlation = shared.correlation
    assert isinstance(correlation, LKJCorrelation)
    assert (correlation.size, correlation.components) == (2, 1)


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize("covariance", [None, COVARIANCE], ids=["iid", "known"])
def test_block_coefficients_correlate_within_each_category(
    parameterization: Parameterization, covariance: GroupCovariance | None
):
    """The covariance of one level's coefficients is block diagonal by category."""
    _, (law,) = _laws(
        "k3 ~ x + (1 + x | g)",
        family="categorical",
        covariance=covariance,
        parameterization=parameterization,
    )
    correlation = law.correlation
    assert isinstance(correlation, LKJCorrelation)
    factors = {
        site: torch.tensor(
            [[1.0, 0.0], [value, (1 - value**2) ** 0.5]], dtype=torch.float64
        )
        for site, value in zip(correlation.factors, (0.6, -0.3), strict=True)
    }
    scales = {
        member.scale.site: torch.tensor(value, dtype=torch.float64)
        for member, value in zip(law.members, ([0.7, 0.5], [1.1, 0.9]), strict=True)
    }
    identity = torch.eye(law.width, dtype=torch.float64)
    column = column_deviations(identity, factors | scales, law, sample_ndims=0)
    covariance_matrix = column.T @ column
    for category, site in enumerate(correlation.factors):
        index = [category, 2 + category]
        block = covariance_matrix[index][:, index]
        sd = torch.stack(
            [scales[member.scale.site][category] for member in law.members]
        )
        rho = factors[site] @ factors[site].T
        torch.testing.assert_close(block, sd[:, None] * rho * sd)
    assert float(covariance_matrix[0, 1].abs()) == 0.0
    assert float(covariance_matrix[0, 3].abs()) == 0.0


CASES = [
    ("y ~ x + (1 | g)", "gaussian", None),
    ("y ~ x + (1 + x | g) + (1 | h)", "gaussian", None),
    ("y ~ x + (1 + x || g)", "gaussian", COVARIANCE),
    ("y ~ x + (1 + x | g)", "gaussian", DEFICIENT),
    ("o ~ x + (1 + cs(x) | g)", "acat", None),
    ("k3 ~ x + (1 | g)", "categorical", None),
    ("k3 ~ x + (1 + x | g)", "categorical", COVARIANCE),
]


@pytest.mark.parametrize("text,family,covariance", CASES)
@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_program_samples_exactly_the_sites_the_laws_declare(
    text: str,
    family: str,
    covariance: GroupCovariance | None,
    parameterization: Parameterization,
):
    lens, laws = _laws(
        text, family=family, covariance=covariance, parameterization=parameterization
    )
    declared = set()
    for law in laws:
        declared.update(law.correlation.sites)
        for member in law.members:
            declared.add(member.scale.site)
            if member.raw_site is not None:
                declared.add(member.raw_site)
    sampled = _sampled(lens, text, family)
    assert declared <= sampled
    group_sites = {
        site
        for site in sampled
        if site.startswith(("sigma_g", "sigma_h", "z_", "cor_", "group_", "centered_"))
        or site in {law.members[0].coefficient for law in laws}
    }
    assert group_sites <= declared | {"sigma"}
    assert tuple(law.layout for law in laws) == lens.group_layouts
