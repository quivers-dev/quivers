"""Keyword schemas for QVR distribution-family construction.

A family call matches one complete schema. Most families have one canonical
schema supplied by their registry record; families with mutually exclusive
parameterizations declare each alternative here.
"""

from __future__ import annotations


_FAMILY_PARAMETERS: dict[str, tuple[str, ...]] = {
    "Normal": ("loc", "scale"),
    "LogitNormal": ("mu", "sigma"),
    "Beta": ("concentration1", "concentration0"),
    "TruncatedNormal": ("mu", "sigma", "low", "high"),
    "Dirichlet": ("concentration",),
    "Cauchy": ("loc", "scale"),
    "Laplace": ("loc", "scale"),
    "Gumbel": ("loc", "scale"),
    "LogNormal": ("loc", "scale"),
    "StudentT": ("df", "loc", "scale"),
    "Exponential": ("rate",),
    "Gamma": ("concentration", "rate"),
    "Chi2": ("df",),
    "HalfCauchy": ("scale",),
    "HalfNormal": ("scale",),
    "InverseGamma": ("concentration", "rate"),
    "Weibull": ("scale", "concentration"),
    "Pareto": ("scale", "alpha"),
    "Kumaraswamy": ("concentration1", "concentration0"),
    "ContinuousBernoulli": ("probs",),
    "FisherSnedecor": ("df1", "df2"),
    "Uniform": ("low", "high"),
    "MultivariateNormal": ("loc", "scale_tril"),
    "LowRankMVN": ("loc", "cov_factor", "cov_diag"),
    "RelaxedBernoulli": ("temperature", "probs"),
    "RelaxedOneHotCategorical": ("temperature", "probs"),
    "Wishart": ("df", "scale_tril"),
    "InverseWishart": ("df", "scale_tril"),
    "MatrixNormal": ("loc", "row_covariance", "col_covariance"),
    "GP": ("length_scale", "amplitude"),
    "Horseshoe": ("scale",),
    "Bernoulli": ("probs",),
    "Categorical": ("probs",),
    "Poisson": ("rate",),
    "NegativeBinomial": ("total_count", "probs"),
    "Geometric": ("probs",),
    "Binomial": ("total_count", "probs"),
    "VonMises": ("loc", "concentration"),
    "LogisticNormal": ("loc", "scale"),
    "OneHotCategorical": ("probs",),
    "LKJCholesky": ("concentration",),
    "Mixture": ("weights", "components"),
    "MixtureNormal": ("weights", "loc", "scale"),
    "Independent": ("base", "reinterpreted_batch_ndims"),
    "Transformed": ("base", "transforms"),
    "Truncated": ("base", "low", "high"),
    "ZeroInflatedPoisson": ("zero_prob", "rate"),
    "HurdlePoisson": ("zero_prob", "rate"),
    "ZeroOneInflatedBeta": ("mu", "phi", "zoi", "coi"),
    "Restrict": ("base", "low", "high"),
    "Normalize": ("base",),
    "PointMass": ("value",),
    "Pushforward": ("base", "bijector"),
    "LKJCorrelationFactor": ("concentration",),
    "BetaBinomial": ("total_count", "concentration1", "concentration0"),
    "OrderedLogistic": ("predictor", "cutpoints"),
    "OrderedProbit": ("eta", "cutpoints"),
    "Logistic": ("loc", "scale"),
    "GeneralizedPareto": ("loc", "scale", "concentration"),
    "HalfStudentT": ("df", "scale"),
}


_ALTERNATIVE_SCHEMAS: dict[str, tuple[tuple[str, ...], ...]] = {
    "Categorical": (("probs",), ("logits",)),
    "MultivariateNormal": (
        ("loc", "scale_tril"),
        ("loc", "covariance_matrix"),
        ("loc", "precision_matrix"),
    ),
    "Wishart": (
        ("df", "scale_tril"),
        ("df", "covariance_matrix"),
        ("df", "precision_matrix"),
    ),
    "Bernoulli": (("probs",), ("logits",)),
    "Binomial": (("total_count", "probs"), ("total_count", "logits")),
    "Geometric": (("probs",), ("logits",)),
    "NegativeBinomial": (("total_count", "probs"), ("total_count", "logits")),
    "Restrict": (
        ("base", "low", "high"),
        ("base", "low"),
        ("base", "high"),
    ),
    "Truncate": (
        ("base", "low", "high"),
        ("base", "low"),
        ("base", "high"),
    ),
}


FAMILY_ALIASES: dict[str, str] = {
    "Pushforward": "Transformed",
    "Truncate": "Restrict",
}
"""Surface family names that resolve to another family's registry record.

A key is a name a source may write; its value is the family whose
specification builds the distribution.
"""

PARAMETER_ALIASES: dict[str, dict[str, str]] = {
    "LogitNormal": {"mu": "loc", "sigma": "scale"},
    "TruncatedNormal": {"mu": "loc", "sigma": "scale"},
    "Mixture": {"components": "component"},
    "OrderedLogistic": {"predictor": "eta"},
    "Pushforward": {"bijector": "transforms"},
}
"""Surface keywords that name a registry parameter under another name.

A family's entry maps each keyword a source may write to the registry
parameter it fills, which means the same quantity. A keyword absent here
and from the registry record names no parameter of the family.
"""

DISTRIBUTION_FAMILIES: frozenset[str] = frozenset(
    (*_FAMILY_PARAMETERS, *FAMILY_ALIASES)
)
"""Every distribution-family name accepted in QVR source.

Highlighters consume this set so their surface vocabulary follows the same
schemas used by validation and lowering.
"""


def family_parameterizations(
    family: str, canonical: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Return every complete keyword schema accepted by ``family``.

    Parameters
    ----------
    family : str
        The family name as written in source.
    canonical : tuple[str, ...]
        The parameter names the family's registry record declares,
        used when no schema for ``family`` is declared here.

    Returns
    -------
    tuple[tuple[str, ...], ...]
        One tuple of keyword names per accepted parameterization, empty
        when the family takes no parameters.
    """
    alternatives = _ALTERNATIVE_SCHEMAS.get(family)
    if alternatives is not None:
        return alternatives
    parameters = _FAMILY_PARAMETERS.get(family, canonical)
    return (parameters,) if parameters else ()


def family_parameter_names(family: str) -> tuple[str, ...] | None:
    """Return the canonical QVR keyword order for ``family``.

    Parameters
    ----------
    family : str
        The family name as written in source.

    Returns
    -------
    tuple[str, ...] | None
        The keyword names in canonical order, or None when no keyword
        schema is declared for ``family``.
    """
    return _FAMILY_PARAMETERS.get(family)


def render_parameterizations(schemas: tuple[tuple[str, ...], ...]) -> str:
    """Render schemas compactly for diagnostics."""
    return " or ".join("(" + ", ".join(schema) + ")" for schema in schemas)


__all__ = [
    "DISTRIBUTION_FAMILIES",
    "FAMILY_ALIASES",
    "PARAMETER_ALIASES",
    "family_parameter_names",
    "family_parameterizations",
]
