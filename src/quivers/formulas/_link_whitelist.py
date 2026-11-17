"""Admissible links per family and distributional parameter, as in brms.

Location links follow the ``links`` entries of the brms family lists
(``R/family-lists.R``) and auxiliary links follow ``links_dpars``
(``R/families.R``), both at brms 2.23.0. QVR names brms's ``1/mu^2`` link
``inverse_square`` and the compositional ``logit`` link ``softmax``; both
spellings are admitted. `LOCATION_EXTENSIONS` records the stated QVR
extensions beyond brms. A family missing from `LOCATION_LINKS` is refused
until a decision is recorded.
"""

from __future__ import annotations

from collections.abc import Mapping

_REAL = (
    "identity",
    "log",
    "inverse",
    "softplus",
    "squareplus",
    "logit",
    "probit",
    "probit_approx",
    "cloglog",
    "cauchit",
    "softit",
)
_POSITIVE = ("log", "identity", "inverse", "softplus", "squareplus")
_COUNT = ("log", "identity", "sqrt", "softplus", "squareplus")
_UNIT = (
    "logit",
    "probit",
    "probit_approx",
    "cloglog",
    "cauchit",
    "softit",
    "identity",
    "log",
)
_ORDINAL = ("logit", "probit", "probit_approx", "cloglog", "cauchit", "softit")
_SHIFTED = ("identity", "log", "inverse", "softplus", "squareplus")
_COMPOSITIONAL = ("logit", "softmax")

LOCATION_LINKS: Mapping[str, tuple[str, ...]] = {
    "gaussian": _REAL,
    "student_t": _REAL,
    "skew_normal": _REAL,
    "mixture": _REAL,
    "binomial": _UNIT,
    "bernoulli": _UNIT,
    "beta": _UNIT,
    "xbeta": _UNIT,
    "zero_inflated_beta": _UNIT,
    "zero_one_inflated_beta": _UNIT,
    "zero_inflated_binomial": _UNIT,
    "zero_inflated_beta_binomial": _UNIT,
    "beta_binomial": _UNIT[:-1],
    "discrete_weibull": _UNIT[:-1],
    "categorical": _COMPOSITIONAL,
    "multinomial": _COMPOSITIONAL,
    "dirichlet": _COMPOSITIONAL,
    "dirichlet_multinomial": _COMPOSITIONAL,
    "dirichlet2": ("log", "softplus", "squareplus", "identity", "logm1"),
    "logistic_normal": ("identity",),
    "poisson": _COUNT,
    "negative_binomial": _COUNT,
    "negbinomial2": _COUNT,
    "geometric": _COUNT,
    "com_poisson": _COUNT,
    "hurdle_poisson": _COUNT,
    "hurdle_negbinomial": _COUNT,
    "zero_inflated_poisson": _COUNT,
    "zero_inflated_negbinomial": _COUNT,
    "gamma": _POSITIVE,
    "weibull": _POSITIVE,
    "exponential": _POSITIVE,
    "frechet": _POSITIVE,
    "hurdle_gamma": _POSITIVE,
    "inverse_gaussian": (
        "inverse_square",
        "inverse",
        "identity",
        "log",
        "softplus",
        "squareplus",
    ),
    "lognormal": ("identity", "inverse"),
    "shifted_lognormal": ("identity", "inverse"),
    "hurdle_lognormal": ("identity", "inverse"),
    "exgaussian": _SHIFTED,
    "gen_extreme_value": _SHIFTED,
    "asym_laplace": _SHIFTED,
    "zero_inflated_asym_laplace": _SHIFTED,
    "cox": ("log", "identity", "softplus", "squareplus"),
    "von_mises": ("tan_half", "identity"),
    "wiener": ("identity", "log", "softplus", "squareplus"),
    "cumulative": _ORDINAL,
    "acat": _ORDINAL,
    "hurdle_cumulative": _ORDINAL,
    "sratio": _ORDINAL[:-1],
    "cratio": _ORDINAL[:-1],
}
"""Location (``mu``) links each registered family admits, as in brms."""

_REAL_EXTENSIONS = ("log1p", "sqrt", "inverse_square", "logm1", "tan_half")

LOCATION_EXTENSIONS: Mapping[str, tuple[str, ...]] = {
    "gaussian": _REAL_EXTENSIONS,
    "student_t": _REAL_EXTENSIONS,
    "skew_normal": _REAL_EXTENSIONS,
}
"""Location links QVR admits beyond brms, a stated extension.

The location of these families ranges over the whole real line, so every
link whose inverse is real-valued gives a valid mean. brms lists only a
subset; QVR also admits the remaining real-valued links of its registry.
"""

_LOG_LIKE = ("log", "identity", "softplus", "squareplus")
_PROBABILITY = ("logit", "identity")

AUXILIARY_LINKS: Mapping[str, tuple[str, ...]] = {
    "sigma": _LOG_LIKE,
    "shape": _LOG_LIKE,
    "nu": ("logm1", "identity"),
    "phi": _LOG_LIKE,
    "kappa": _LOG_LIKE,
    "beta": _LOG_LIKE,
    "zi": _PROBABILITY,
    "hu": _PROBABILITY,
    "zoi": _PROBABILITY,
    "coi": _PROBABILITY,
    "disc": _LOG_LIKE,
    "bs": _LOG_LIKE,
    "ndt": _LOG_LIKE,
    "bias": _PROBABILITY,
    "quantile": _PROBABILITY,
    "xi": ("log1p", "identity"),
    "alpha": ("identity", "log", "softplus", "squareplus"),
}
"""Links each distributional parameter class admits (brms ``links_dpars``)."""

AUXILIARY_CLASSES: Mapping[str, str] = {"disp": "shape"}
"""QVR auxiliary names whose brms parameter class has another name."""


def admissible_links(family_name: str, parameter: str) -> tuple[str, ...]:
    """Return the links brms admits for one family parameter.

    Parameters
    ----------
    family_name : str
        Registered family name.
    parameter : str
        ``mu`` for the location or an auxiliary parameter name.

    Returns
    -------
    tuple[str, ...]
        Admissible link names.

    Raises
    ------
    ValueError
        If the family or parameter has no recorded link decision.
    """
    if parameter == "mu":
        if family_name not in LOCATION_LINKS:
            raise ValueError(
                f"family {family_name!r} has no recorded link decision; "
                f"families with decisions are {sorted(LOCATION_LINKS)}"
            )
        return LOCATION_LINKS[family_name] + LOCATION_EXTENSIONS.get(family_name, ())
    parameter_class = AUXILIARY_CLASSES.get(parameter, parameter)
    if parameter_class not in AUXILIARY_LINKS:
        raise ValueError(
            f"parameter {parameter!r} of family {family_name!r} has no recorded "
            "link decision"
        )
    return AUXILIARY_LINKS[parameter_class]


def check_family_links(family_name: str, requested: Mapping[str, str]) -> None:
    """Refuse a link brms does not admit for a family parameter.

    Parameters
    ----------
    family_name : str
        Registered family name.
    requested : Mapping[str, str]
        Link names keyed by ``mu`` or an auxiliary parameter name.

    Raises
    ------
    ValueError
        If a link is not admitted, naming the family, the parameter, and the
        admissible links.
    """
    for parameter, link in requested.items():
        allowed = admissible_links(family_name, parameter)
        if link not in allowed:
            raise ValueError(
                f"link {link!r} is not supported for parameter {parameter!r} of "
                f"family {family_name!r}; admissible links are {list(allowed)}"
            )


__all__ = [
    "AUXILIARY_CLASSES",
    "AUXILIARY_LINKS",
    "LOCATION_EXTENSIONS",
    "LOCATION_LINKS",
    "admissible_links",
    "check_family_links",
]
