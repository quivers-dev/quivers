"""Shared constructor schemas for QVR and its checked distribution core.

A family call matches one complete schema. Most families have one canonical
schema supplied by their registry record; families with mutually exclusive
parameterizations declare each alternative here.
"""

from __future__ import annotations


_FAMILY_PARAMETERS: dict[str, tuple[str, ...]] = {
    "Normal": ("loc", "scale"),
    "LogitNormal": ("mu", "sigma"),
    "Beta": ("concentration1", "concentration0"),
    "ExtendedBeta": ("mu", "phi", "kappa"),
    "TruncatedNormal": ("mu", "sigma", "low", "high"),
    "Dirichlet": ("concentration",),
    "Multinomial": ("total_count", "probs"),
    "DirichletMultinomial": ("total_count", "concentration"),
    "Cauchy": ("loc", "scale"),
    "Laplace": ("loc", "scale"),
    "BernsteinHazard": ("breaks", "controls", "rate"),
    "COMPoisson": ("rate", "dispersion"),
    "DiscreteWeibull": ("probs", "concentration"),
    "Wiener": ("boundary", "ndt", "bias", "drift"),
    "ZeroInflatedAsymmetricLaplace": ("zero_prob", "loc", "scale", "quantile"),
    "AsymmetricLaplace": ("loc", "scale", "quantile"),
    "ExGaussian": ("loc", "scale", "rate"),
    "HurdleGamma": ("zero_prob", "concentration", "rate"),
    "HurdleLogNormal": ("zero_prob", "loc", "scale"),
    "InverseGaussian": ("mu", "shape"),
    "SkewNormal": ("loc", "scale", "skewness"),
    "ShiftedLogNormal": ("loc", "scale", "shift"),
    "Frechet": ("scale", "concentration"),
    "GeneralizedExtremeValue": ("loc", "scale", "concentration"),
    "Gumbel": ("loc", "scale"),
    "LogNormal": ("loc", "scale"),
    "StudentT": ("df", "loc", "scale"),
    "Exponential": ("rate",),
    "Gamma": ("concentration", "rate"),
    "LogGamma": ("concentration", "rate"),
    "LogitBeta": ("concentration1", "concentration0"),
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
    "MultivariateStudentT": ("df", "loc", "scale_tril"),
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
    "AdditiveLogRatioNormal": ("loc", "scale_tril", "reference"),
    "LogisticNormal": ("loc", "scale"),
    "OneHotCategorical": ("probs",),
    "LKJCholesky": ("concentration",),
    "Mixture": ("weights", "components"),
    "MixtureNormal": ("weights", "loc", "scale"),
    "Independent": ("base", "reinterpreted_batch_ndims"),
    "Transformed": ("base", "transforms"),
    "Truncated": ("base", "low", "high"),
    "ZeroInflatedNegativeBinomial": ("zero_prob", "total_count", "probs"),
    "ZeroInflatedBinomial": ("zero_prob", "total_count", "probs"),
    "ZeroInflatedBetaBinomial": (
        "zero_prob",
        "total_count",
        "concentration1",
        "concentration0",
    ),
    "HurdleNegativeBinomial": ("zero_prob", "total_count", "probs"),
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
    "AdditiveLogRatioNormal": (
        ("loc", "scale_tril"),
        ("loc", "scale_tril", "reference"),
    ),
    "BernsteinHazard": (
        ("breaks", "controls", "rate"),
        ("breaks", "controls", "log_rate"),
    ),
    "COMPoisson": (("rate", "dispersion"), ("log_rate", "dispersion")),
    "Categorical": (("probs",), ("logits",)),
    "OneHotCategorical": (("probs",), ("logits",)),
    "RelaxedBernoulli": (("temperature", "probs"), ("temperature", "logits")),
    "RelaxedOneHotCategorical": (
        ("temperature", "probs"),
        ("temperature", "logits"),
    ),
    "Multinomial": (("total_count", "probs"), ("total_count", "logits")),
    "MultivariateNormal": (
        ("loc", "scale_tril"),
        ("loc", "covariance_matrix"),
        ("loc", "precision_matrix"),
    ),
    "MultivariateStudentT": (
        ("df", "loc", "scale_tril"),
        ("df", "loc", "covariance_matrix"),
        ("df", "loc", "precision_matrix"),
    ),
    "MatrixNormal": (
        ("loc", "row_covariance", "col_covariance"),
        ("loc", "row_scale_tril", "col_covariance"),
        ("loc", "row_covariance", "col_scale_tril"),
        ("loc", "row_scale_tril", "col_scale_tril"),
    ),
    "Wishart": (
        ("df", "scale_tril"),
        ("df", "covariance_matrix"),
        ("df", "precision_matrix"),
    ),
    "Bernoulli": (("probs",), ("logits",)),
    "ContinuousBernoulli": (("probs",), ("logits",)),
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
        if family in ("Restrict", "Truncate"):
            # Endpoint policies are independent optional Boolean arguments.
            # Retain the legacy positional schema and accept every explicit
            # subset without inventing a second operator construction path.
            return tuple(
                (*schema, *flags)
                for schema in alternatives
                for flags in (
                    (),
                    ("lower_closed",),
                    ("upper_closed",),
                    ("lower_closed", "upper_closed"),
                )
            )
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


def _semantic_parameterizations(
    family: str, canonical: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Translate complete surface schemas to semantic parameter identities.

    Parameters
    ----------
    family : str
        Canonical semantic family name.
    canonical : tuple[str, ...]
        Semantic parameter names, also the fallback for custom families.

    Returns
    -------
    tuple[tuple[str, ...], ...]
        Complete constructor alternatives using semantic names. GP's surface
        kernel hyperparameters are elaborated to a mean vector and kernel
        matrix before this boundary.
    """
    if family == "GP":
        return (canonical,)
    aliases = PARAMETER_ALIASES.get(family, {})
    return tuple(
        tuple(aliases.get(name, name) for name in schema)
        for schema in family_parameterizations(family, canonical)
    )


def _select_parameterization(
    family: str,
    supplied: tuple[str, ...],
    schemas: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    """Select one complete constructor schema without supplying defaults.

    Parameters
    ----------
    family : str
        Family name for diagnostics.
    supplied : tuple[str, ...]
        Authored argument names, before ordering.
    schemas : tuple[tuple[str, ...], ...]
        Complete alternatives at this interface.

    Returns
    -------
    tuple[str, ...]
        Matching canonical order.

    Raises
    ------
    TypeError
        If an argument repeats or no complete alternative matches.
    """
    seen = set()
    for name in supplied:
        if name in seen:
            raise TypeError(f"family {family!r} is given parameter {name!r} twice")
        seen.add(name)
    selected = next((schema for schema in schemas if set(schema) == seen), None)
    if selected is None:
        raise TypeError(
            f"family {family!r} requires exactly one complete parameter schema "
            f"{render_parameterizations(schemas)}; got ({', '.join(supplied)})"
        )
    return selected


def render_parameterizations(schemas: tuple[tuple[str, ...], ...]) -> str:
    """Render schemas compactly for diagnostics.

    Parameters
    ----------
    schemas : tuple[tuple[str, ...], ...]
        Complete alternatives in constructor order.

    Returns
    -------
    str
        Parenthesized alternatives joined by ``or``.
    """
    return " or ".join("(" + ", ".join(schema) + ")" for schema in schemas)


def _family_signature(family: str) -> str | None:
    """Render the canonical constructor schemas for interactive tooling.

    Parameters
    ----------
    family : str
        Name resolved after authored declarations and local bindings.

    Returns
    -------
    str or None
        Family name and complete parameter alternatives, or None for other names.
    """
    schemas = family_parameterizations(family, ())
    return family + " " + render_parameterizations(schemas) if schemas else None


__all__ = [
    "DISTRIBUTION_FAMILIES",
    "FAMILY_ALIASES",
    "PARAMETER_ALIASES",
    "family_parameter_names",
    "family_parameterizations",
]
