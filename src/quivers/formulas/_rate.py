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

from quivers.dsl.ast_nodes import ProgramStep
from quivers.formulas._ast import _bind, _let, _var
from quivers.formulas.family import Family, families
from quivers.formulas.formula import ResponseModifiers


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
        If ``rate`` is used with a family that declares no exposure law.
    """
    if modifiers.rate is None:
        return observe_family, arguments
    exposure = family.exposure
    if exposure is None:
        admitted = sorted(
            name for name, entry in families.items() if entry.exposure is not None
        )
        raise ValueError(
            f"rate() is not supported for the {family.name} family; it applies to "
            f"{', '.join(admitted)}"
        )
    denominator = modifiers.rate.qvr_name
    steps.extend(
        _let(binding.name, _bind(binding.expr, exposure=_var(denominator)))
        for binding in exposure.observe_lets
    )
    return exposure.observe_family or observe_family, tuple(
        denominator if argument == "exposure" else argument
        for argument in exposure.observe_args
    )


__all__ = ["rate_arguments"]
