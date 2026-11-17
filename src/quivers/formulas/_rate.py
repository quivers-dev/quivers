"""Response denominators of count families: the brms ``rate()`` addition term.

``y | rate(denom) ~ ...`` models the count of ``denom`` exposure units:
the mean becomes ``mu * denom`` for every link, and the negative binomial
shape is multiplied by ``denom`` as well, so the count is a sum of ``denom``
independent unit-exposure counts. brms writes these as
``poisson_log_lpmf(Y | mu + log_denom)`` and
``neg_binomial_2_log_lpmf(Y | mu + log_denom, shape * denom)``; the geometric
family becomes the negative binomial with shape ``denom``. Expected values
include the denominator, linear predictors and ``mu`` do not.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import LetExprLiteral, ProgramStep
from quivers.formulas._ast import _add, _div, _let, _mul, _var
from quivers.formulas.family import Family
from quivers.formulas.formula import ResponseModifiers

#: Families whose brms ``ad`` list includes ``rate``.
RATE_FAMILIES = frozenset({"poisson", "negative_binomial", "negbinomial2", "geometric"})


def rate_arguments(
    family: Family,
    observe_family: str,
    arguments: tuple[str | float, ...],
    steps: list[ProgramStep],
    modifiers: ResponseModifiers,
) -> tuple[str, tuple[str | float, ...]]:
    """Scale a count observation by its retained response denominators.

    Scoring costs O(n) beyond the unscaled law.

    Parameters
    ----------
    family : Family
        Response family.
    observe_family : str
        Native observation kernel.
    arguments : tuple[str or float, ...]
        Native constructor arguments, in schema order.
    steps : list[ProgramStep]
        Program steps receiving the scaled bindings.
    modifiers : ResponseModifiers
        Response roles, possibly with ``rate`` denominators.

    Returns
    -------
    tuple[str, tuple[str or float, ...]]
        The observation kernel and its arguments, unchanged without ``rate``.

    Raises
    ------
    ValueError
        If ``rate`` is used with a family outside poisson, negative_binomial,
        negbinomial2 and geometric.
    """
    if modifiers.rate is None:
        return observe_family, arguments
    if family.name not in RATE_FAMILIES:
        raise ValueError(
            f"rate() is not supported for the {family.name} family; it applies to "
            f"{', '.join(sorted(RATE_FAMILIES))}"
        )
    denominator = _var(modifiers.rate.qvr_name)
    if family.name == "poisson":
        steps.append(_let("rate_mu", _mul(_var("mu"), denominator)))
        return observe_family, ("rate_mu",)
    if family.name == "geometric":
        # NB(shape = denom, mean = mu * denom): success odds mu / (1 + mu).
        steps.append(
            _let(
                "rate_probs",
                _div(_var("mu"), _add(LetExprLiteral(value=1.0), _var("mu"))),
            )
        )
        return "NegativeBinomial", (modifiers.rate.qvr_name, "rate_probs")
    # Negative binomial: the shape scales by denom, which leaves the success
    # probability mu / (mu + shape) unchanged.
    shape, probs = arguments
    steps.append(_let("rate_shape", _mul(_var(str(shape)), denominator)))
    return observe_family, ("rate_shape", probs)


__all__ = ["RATE_FAMILIES", "rate_arguments"]
