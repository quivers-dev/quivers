"""Every target spells every family it names, with its own conventions."""

from __future__ import annotations

import pytest

from quivers.qiec.families import FAMILIES
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.family_spelling import (
    DYNAMIC_TARGET_LANGUAGES,
    can_spell,
    helper_families,
    helper_roots,
    spell_distribution,
)


def _arguments(family: str) -> dict[str, str]:
    """Placeholder host expressions for every parameter of a family.

    Parameters
    ----------
    family : str
        The family.

    Returns
    -------
    dict[str, str]
        Parameter name to a distinct identifier.
    """
    return {name: f"a_{name}" for name in FAMILIES[family].parameter_names}


@pytest.mark.parametrize("target", sorted(DYNAMIC_TARGET_LANGUAGES))
def test_every_named_family_has_a_spelling_or_is_declared_unspellable(
    target: str,
) -> None:
    for family, meta in FAMILY_META.items():
        if target not in meta.target_names:
            assert not can_spell(target, family)
            continue
        if not can_spell(target, family):
            with pytest.raises(UnsupportedConstruct):
                spell_distribution(target, family, _arguments(family), (3, 3))
            continue
        spelled = spell_distribution(target, family, _arguments(family), (3, 3))
        assert spelled.expression
        for name in FAMILIES[family].parameter_names:
            if name in (
                "logits",
                "precision_matrix",
                "scale_tril",
                "transforms",
            ):
                continue
            if (
                family == "MultivariateNormal"
                and name != "loc"
                and name != "covariance_matrix"
            ):
                continue
            assert f"a_{name}" in spelled.expression, (target, family, name)
        assert set(spelled.helpers) <= helper_roots(target, frozenset({family}))


@pytest.mark.parametrize(
    ("target", "family", "arguments", "fragment"),
    [
        ("turing", "Gamma", {"concentration": "a", "rate": "b"}, "Gamma(a, inv(b))"),
        ("gen", "Gamma", {"concentration": "a", "rate": "b"}, "gamma, a, inv(b)"),
        ("webppl", "Gamma", {"concentration": "a", "rate": "b"}, "scale: 1 / b"),
        (
            "turing",
            "NegativeBinomial",
            {"total_count": "r", "probs": "p"},
            "NegativeBinomial(r, 1 - p)",
        ),
        ("pymc", "NegativeBinomial", {"total_count": "r", "probs": "p"}, "p=(1.0 - p)"),
        (
            "numpyro",
            "NegativeBinomial",
            {"total_count": "r", "probs": "p"},
            "mean=(r * p) / (1 - p)",
        ),
        (
            "turing",
            "Categorical",
            {"probs": "p"},
            "_qvr_qiec_shifted(Categorical(_qvr_qiec_array(p)), 1)",
        ),
        ("gen", "Categorical", {"probs": "p"}, "_qvr_qiec_shifted("),
        (
            "pymc",
            "Geometric",
            {"probs": "p"},
            "_qvr_qiec_shifted(pymc.Geometric.dist(p=p), 1)",
        ),
        ("turing", "HalfNormal", {"scale": "s"}, "truncated(Normal(0, s), 0, Inf)"),
        ("gen", "HalfNormal", {"scale": "s"}, "truncated_normal, 0, s, 0, Inf"),
        (
            "webppl",
            "HalfNormal",
            {"scale": "s"},
            "_qvr_qiec_half(Gaussian({mu: 0, sigma: s}))",
        ),
        ("church", "HalfNormal", {"scale": "s"}, "(half (gaussian 0 s))"),
        ("church", "Pareto", {"alpha": "a", "scale": "s"}, "(pareto s a)"),
        ("turing", "Weibull", {"scale": "s", "concentration": "k"}, "Weibull(k, s)"),
        (
            "turing",
            "StudentT",
            {"df": "d", "loc": "m", "scale": "s"},
            "(m + s * TDist(d))",
        ),
        ("edward2", "HalfCauchy", {"scale": "s"}, "HalfCauchy(loc=0.0, scale=s)"),
        ("edward2", "Pareto", {"alpha": "a", "scale": "s"}, "concentration=a"),
        (
            "pyro",
            "LKJCholesky",
            {"concentration": "c"},
            "LKJCorrCholesky(3, concentration=c)",
        ),
        (
            "numpyro",
            "MatrixNormal",
            {"loc": "m", "row_covariance": "u", "col_covariance": "v"},
            "scale_tril_row=jnp.linalg.cholesky(_qvr_qiec_array(u))",
        ),
    ],
)
def test_target_conventions_are_applied(
    target: str, family: str, arguments: dict[str, str], fragment: str
) -> None:
    assert fragment in spell_distribution(target, family, arguments, (3, 3)).expression


def test_matrix_dimension_families_need_a_square_event_shape() -> None:
    with pytest.raises(UnsupportedConstruct, match="no-square-event-shape"):
        spell_distribution("pyro", "LKJCholesky", {"concentration": "c"}, None)
    with pytest.raises(UnsupportedConstruct, match="missing:scale"):
        spell_distribution("turing", "HalfNormal", {}, None)


def test_helper_families_are_the_ones_each_runtime_file_defines() -> None:
    assert helper_families("pyro") == frozenset(
        {
            "TruncatedNormal",
            "LogitNormal",
            "HalfStudentT",
            "MatrixNormal",
            "InverseWishart",
            "LogGamma",
            "LogitBeta",
        }
    )
    assert "Categorical" in helper_families("webppl")
    assert helper_roots("pymc", frozenset({"Normal", "LKJCholesky"})) == frozenset(
        {"LKJCholesky"}
    )
