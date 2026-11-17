"""brms ``brmsfamily()``: a family from its brms name, link, and options."""

from __future__ import annotations

from typing import Literal

from quivers.formulas.family import Family, families


def brmsfamily(
    family: str,
    link: str | None = None,
    *,
    threshold: Literal["flexible", "equidistant", "sum_to_zero"] | None = None,
    refcat: int | str | None = None,
    **links: str,
) -> Family:
    """Build a response family as brms ``brmsfamily()`` does.

    Parameters
    ----------
    family : str
        brms family name, in any spelling brms reads: ``"negbinomial"``,
        ``"student"``, ``"Gamma"``, ``"zi_poisson"``, ``"cumulative"``.
    link : str or None
        Link of the location parameter. None keeps the family's default.
    threshold : {"flexible", "equidistant", "sum_to_zero"} or None
        Threshold structure of an ordinal family.
    refcat : int or str or None
        Reference category of a categorical or vector family.
    **links : str
        Links of auxiliary parameters, keyed ``link_<parameter>`` as in
        brms (``link_sigma="log"``).

    Returns
    -------
    Family
        The configured family, for ``fit(..., family=...)`` or added to a
        `bf` response with ``+``.

    Raises
    ------
    ValueError
        If the family is unknown, an option does not apply to it, a keyword
        is not ``link_<parameter>``, or a link is not admissible.

    Notes
    -----
    A family that keeps the default links, thresholds and reference is the
    registered family itself. Cost is constant.
    """
    if family not in families:
        raise ValueError(
            f"brmsfamily: unknown family {family!r}; the families are "
            f"{sorted(families)!r}"
        )
    resolved = families[family]
    misnamed = sorted(name for name in links if not name.startswith("link_"))
    if misnamed:
        raise ValueError(
            f"brmsfamily: {misnamed!r} are not link arguments; name the link of "
            "an auxiliary parameter link_<parameter>"
        )
    selected = {name.removeprefix("link_"): value for name, value in links.items()}
    if link is not None:
        selected["mu"] = link
    if selected:
        resolved = resolved.with_links(**selected)
    if threshold is not None:
        resolved = resolved.with_thresholds(threshold)
    if refcat is not None:
        resolved = resolved.with_reference(refcat)
    return resolved


__all__: list[str] = []
