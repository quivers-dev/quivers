"""Category-specific linear predictors of categorical and simplex families.

brms gives every non-reference category ``c`` of a categorical, multinomial,
Dirichlet, Dirichlet-multinomial or logistic-normal response its own
distributional parameter ``mu<c>`` (``muB``, ``muC`` for levels ``A``, ``B``,
``C`` with reference ``A``). The main formula predicts every ``mu<c>`` that
has no formula of its own, and an explicit ``mu<c> ~ ...`` replaces it for
that category. Each category then has its own coefficients, group effects and
priors (``dpar = "muB"``); the reference category's logit is zero.

The shared-coefficient emission (one coefficient vector over the
``K - 1`` free categories) is the same model when every category has the
same formula; per-category formulas are emitted as ordinary scalar
distributional predictors ``dpar_mu<c>_`` and stacked into the free logits.

The logistic-normal family also has one scale ``sigma<c>`` per free category:
the standard deviation of that category's log ratio to the reference. A
``sigma<c> ~ ...`` formula predicts it through the log link with its own
coefficients, and a category without a formula keeps one scalar scale.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
import re
from typing import TYPE_CHECKING, TypeVar

from quivers.formulas.family import Family
from quivers.formulas.formula import Formula

if TYPE_CHECKING:
    pass

_Value = TypeVar("_Value")


def category_parameter(label: str) -> str:
    """Name the location parameter of one response category.

    Parameters
    ----------
    label : str
        Category or component label.

    Returns
    -------
    str
        ``mu`` followed by the label with characters other than letters,
        digits and underscores removed, as brms forms ``mu<cat>``.

    Raises
    ------
    ValueError
        If the label has no letter, digit or underscore.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", str(label))
    if not cleaned:
        raise ValueError(f"category label {label!r} cannot name a parameter")
    return f"mu{cleaned}"


def is_category_parameter(family: Family, name: str) -> bool:
    """Decide whether a distributional name is a per-category location.

    Parameters
    ----------
    family : Family
        Response family.
    name : str
        Distributional parameter name.

    Returns
    -------
    bool
        True for ``mu<c>`` names of a categorical or simplex family.
    """
    return (
        family.response is not None
        and name.startswith("mu")
        and name != "mu"
        and name not in {aux.name for aux in family.aux_params}
    )


def scale_parameter(scale: str, label: str) -> str:
    """Name the scale parameter of one response category.

    Parameters
    ----------
    scale : str
        Name of the per-component scale parameter.
    label : str
        Category or component label.

    Returns
    -------
    str
        The scale name followed by the label with characters other than letters,
        digits and underscores removed, as brms forms ``sigma<cat>``.

    Raises
    ------
    ValueError
        If the label has no letter, digit or underscore.
    """
    return scale + category_parameter(label)[2:]


def is_scale_parameter(family: Family, name: str) -> bool:
    """Decide whether a distributional name is a per-category scale.

    Parameters
    ----------
    family : Family
        Response family.
    name : str
        Distributional parameter name.

    Returns
    -------
    bool
        True for a name extending that of the parameter a family declares
        `AuxParam.per_component`, as ``sigma<c>`` of the logistic-normal
        family.
    """
    scale = family.component_scale
    return scale is not None and name.startswith(scale.name) and name != scale.name


def free_scales(family: Family, labels: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
    """List the free categories and their scale parameter names.

    Parameters
    ----------
    family : Family
        Logistic-normal family.
    labels : tuple[str, ...]
        Retained component labels in response order.

    Returns
    -------
    tuple[tuple[int, str], ...]
        Coordinate positions and ``sigma<c>`` names in retained order.
    """
    assert family.response is not None and family.component_scale is not None
    scale = family.component_scale.name
    return tuple(
        (index, scale_parameter(scale, labels[index]))
        for index in family.response.coordinate_indices(labels)
    )


def check_scale_formulas(
    family: Family, labels: tuple[str, ...], names: Collection[str]
) -> None:
    """Check per-category scale formulas against the free categories.

    Parameters
    ----------
    family : Family
        Response family.
    labels : tuple[str, ...]
        Retained component labels of the response.
    names : Collection[str]
        Names of the distributional and fixed parameters.

    Raises
    ------
    ValueError
        If a ``sigma<c>`` names no free category (the reference category has
        no scale), or if ``sigma`` is given beside a ``sigma<c>``.
    """
    requested = {name for name in names if is_scale_parameter(family, name)}
    if not requested or family.component_scale is None:
        return
    scale = family.component_scale.name
    known = {name for _, name in free_scales(family, labels)}
    unknown = requested - known
    if unknown:
        raise ValueError(
            f"unknown scale parameters {sorted(unknown)}; the free categories "
            f"have {sorted(known)} (the reference category has no scale)"
        )
    if scale in names:
        raise ValueError(
            f"{family.name} takes either {scale!r}, which predicts every "
            f"category's scale, or per-category scales {sorted(known)}, not both"
        )


def free_categories(
    family: Family, labels: tuple[str, ...]
) -> tuple[tuple[int, str], ...]:
    """List the free categories and their location parameter names.

    Parameters
    ----------
    family : Family
        Categorical or simplex family.
    labels : tuple[str, ...]
        Retained category or component labels in response order.

    Returns
    -------
    tuple[tuple[int, str], ...]
        Coordinate positions and ``mu<c>`` names in retained order.
    """
    assert family.response is not None
    return tuple(
        (index, category_parameter(labels[index]))
        for index in family.response.coordinate_indices(labels)
    )


def category_distributional(
    family: Family,
    labels: tuple[str, ...],
    authored: str,
    distributional: Mapping[str, _Value] | None,
    *,
    nonlinear: bool,
) -> dict[str, _Value | str] | None:
    """Copy the main formula to every free category without a formula.

    Parameters
    ----------
    family : Family
        Response family.
    labels : tuple[str, ...]
        Retained category or component labels of the response.
    authored : str
        Authored main formula.
    distributional : Mapping or None
        Authored distributional formulas.
    nonlinear : bool
        Whether the main formula is a nonlinear expression, which a category
        without a predictor of its own reads as it is.

    Returns
    -------
    dict or None
        The distributional formulas, completed with every ``mu<c>`` when any
        per-category formula was given; otherwise unchanged.

    Raises
    ------
    ValueError
        If a ``mu<c>`` names no free category (the reference category has no
        predictor).
    """
    if distributional is None:
        return None
    result: dict[str, _Value | str] = dict(distributional)
    requested = {name for name in result if is_category_parameter(family, name)}
    if not requested:
        return result
    names = {name for _, name in free_categories(family, labels)}
    unknown = requested - names
    if unknown:
        raise ValueError(
            f"unknown category parameters {sorted(unknown)}; the free categories "
            f"are {sorted(names)} (the reference category has no predictor)"
        )
    if nonlinear:
        return result
    rhs = authored.split("~", 1)[1]
    for _, name in free_categories(family, labels):
        result.setdefault(name, f"{name} ~ {rhs}")
    return result


def main_without_terms(formula: Formula) -> Formula:
    """Strip the main formula's predictor terms when every category has its own.

    Parameters
    ----------
    formula : Formula
        Main formula.

    Returns
    -------
    Formula
        The response and modifiers without coefficients or group effects.
    """
    return formula.with_(
        fixed_columns=(),
        design_terms=(),
        random_terms=(),
        offsets=(),
        latent_terms=(),
        monotonic_terms=(),
    )


__all__ = [
    "check_scale_formulas",
    "category_distributional",
    "category_parameter",
    "free_categories",
    "free_scales",
    "is_category_parameter",
    "is_scale_parameter",
    "main_without_terms",
    "scale_parameter",
]
