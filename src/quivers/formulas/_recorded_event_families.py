"""Admission of the ``cens`` and ``trunc`` addition terms.

A family admits the terms exactly when it declares
`Family.recorded_events`. The registered families declare it where the
``ad`` entries of the brms family lists (``R/family-lists.R``) include
``cens`` and ``trunc``; a custom family is decided by its native law.
"""

from __future__ import annotations

from quivers.formulas.family import Family, families
from quivers.formulas.formula import CensoringSpec, TruncationSpec


def check_recorded_event_family(
    family: Family,
    *,
    censoring: CensoringSpec | None,
    truncation: TruncationSpec | None,
) -> None:
    """Refuse ``cens`` or ``trunc`` on a family that does not admit them.

    Parameters
    ----------
    family : Family
        Response family.
    censoring : CensoringSpec or None
        Bound ``cens`` term, if any.
    truncation : TruncationSpec or None
        Bound ``trunc`` term, if any.

    Raises
    ------
    ValueError
        If a requested term is not admitted by the family, naming the family,
        the term, the missing declaration, and the registered families that
        admit it.
    """
    terms = [
        name
        for name, bound in (("cens", censoring), ("trunc", truncation))
        if bound is not None
    ]
    if not terms or family.recorded_events:
        return
    admitted = sorted(name for name, entry in families.items() if entry.recorded_events)
    raise ValueError(
        f"response term {' and '.join(f'{term}()' for term in terms)} is not "
        f"supported for family {family.name!r}: the family does not declare "
        f"recorded events (Family.recorded_events); families admitting cens() "
        f"and trunc() are {admitted}"
    )


__all__ = ["check_recorded_event_family"]
