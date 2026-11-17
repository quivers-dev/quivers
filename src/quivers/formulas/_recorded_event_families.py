"""Which response families admit the ``cens`` and ``trunc`` addition terms.

The decision for every registered family follows the ``ad`` entries of the
brms family lists (``R/family-lists.R`` at brms 2.23.0): a family admits
``cens`` and ``trunc`` exactly when brms lists them for it. A family missing
from `RECORDED_EVENT_DECISIONS` is refused until a decision is recorded, so a
new family cannot silently inherit either behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.formulas.formula import CensoringSpec, TruncationSpec

RECORDED_EVENT_DECISIONS: Mapping[str, bool] = {
    "gaussian": True,
    "student_t": True,
    "skew_normal": True,
    "binomial": True,
    "beta_binomial": True,
    "beta": True,
    "poisson": True,
    "negative_binomial": True,
    "negbinomial2": True,
    "geometric": True,
    "discrete_weibull": True,
    "com_poisson": True,
    "gamma": True,
    "weibull": True,
    "exponential": True,
    "frechet": True,
    "inverse_gaussian": True,
    "lognormal": True,
    "shifted_lognormal": True,
    "exgaussian": True,
    "gen_extreme_value": True,
    "von_mises": True,
    "asym_laplace": True,
    "zero_inflated_asym_laplace": True,
    "cox": True,
    "hurdle_poisson": True,
    "hurdle_negbinomial": True,
    "hurdle_gamma": True,
    "hurdle_lognormal": True,
    "zero_inflated_poisson": True,
    "zero_inflated_negbinomial": True,
    "zero_inflated_binomial": True,
    "zero_inflated_beta_binomial": True,
    "zero_inflated_beta": True,
    "mixture": True,
    "wiener": False,
    "xbeta": False,
    "bernoulli": False,
    "categorical": False,
    "multinomial": False,
    "dirichlet_multinomial": False,
    "dirichlet": False,
    "dirichlet2": False,
    "logistic_normal": False,
    "cumulative": False,
    "sratio": False,
    "cratio": False,
    "acat": False,
    "hurdle_cumulative": False,
    "zero_one_inflated_beta": False,
}
"""Whether each registered family admits ``cens`` and ``trunc``, as in brms.

``mixture`` is the Gaussian location mixture, whose components admit both
terms in brms. Custom families are decided by their native law and are not
listed here.
"""


def check_recorded_event_family(
    family_name: str,
    *,
    custom: bool,
    censoring: CensoringSpec | None,
    truncation: TruncationSpec | None,
) -> None:
    """Refuse ``cens`` or ``trunc`` on a family brms refuses them for.

    Parameters
    ----------
    family_name : str
        Registered family name.
    custom : bool
        Whether the family is a custom family bound to a native law.
    censoring : CensoringSpec or None
        Bound ``cens`` term, if any.
    truncation : TruncationSpec or None
        Bound ``trunc`` term, if any.

    Raises
    ------
    ValueError
        If a requested term is not admitted by the family, naming the family,
        the term, and the families that admit it.
    """
    terms = [
        name
        for name, bound in (("cens", censoring), ("trunc", truncation))
        if bound is not None
    ]
    if not terms or custom:
        return
    decision = RECORDED_EVENT_DECISIONS.get(family_name)
    if decision is True:
        return
    admitted = sorted(
        name for name, allowed in RECORDED_EVENT_DECISIONS.items() if allowed
    )
    reason = (
        "has no recorded cens/trunc decision"
        if decision is None
        else "does not admit it in brms"
    )
    raise ValueError(
        f"response term {' and '.join(f'{term}()' for term in terms)} is not "
        f"supported for family {family_name!r}: the family {reason}; "
        f"families admitting cens() and trunc() are {admitted}"
    )


__all__ = ["RECORDED_EVENT_DECISIONS", "check_recorded_event_family"]
