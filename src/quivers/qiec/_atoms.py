"""Scalar singleton-atom capabilities shared with semantic family metadata."""

from __future__ import annotations

import math

from quivers.qiec.families import family
from quivers.qiec.reference_support import DistributionError


def atomicity(distribution):
    """Read a homogeneous carrier proof through reference compositions.

    Parameters
    ----------
    distribution : HostLaw
        Reference distribution or measure construction.

    Returns
    -------
    str
        Atomless, discrete, or mixed carrier classification.

    Raises
    ------
    DistributionError
        If a custom composition supplies no carrier classification.
    """
    record = family(distribution.family)
    if distribution.family in {
        "Restrict",
        "Truncated",
        "Normalize",
        "Transformed",
        "Independent",
    }:
        return atomicity(distribution.arguments["base"])
    if distribution.family == "Mixture":
        from quivers.qiec.reference_families import _components

        kinds = {atomicity(c) for c in _components(distribution.arguments, "Mixture")}
        return kinds.pop() if len(kinds) == 1 else "mixed"
    if distribution.family == "PointMass":
        return "discrete"
    if record.effective_atomicity == "composite":
        raise DistributionError(f"{distribution.family} has no atomicity proof")
    return record.effective_atomicity


def atom_reference(distribution, value):
    """Identify declared scalar atom carriers, independent of current weights.

    Parameters
    ----------
    distribution : HostLaw
        Unplated scalar reference distribution. Compositions supply their own hook.
    value : float
        Scalar observation.

    Returns
    -------
    bool
        Membership in the fixed atomic reference carrier.

    Raises
    ------
    DistributionError
        If the law is plated, coupled, or requires a composition callback.
    """
    from quivers.qiec import reference_families as rf

    record = family(distribution.family)
    if distribution.plated or record.event_rank:
        raise DistributionError("scalar atom queries require an unplated scalar law")
    name, arguments = distribution.family, distribution.arguments
    if name in {"Restrict", "Truncated", "Normalize"}:
        return atom_reference(arguments["base"], value)
    if name == "Mixture":
        return any(atom_reference(c, value) for c in rf._components(arguments, name))
    if name == "Transformed":
        point = float(value)
        try:
            for _, inverse, _ in reversed(rf._transform_chain(arguments)):
                point = inverse(point)
        except ValueError, OverflowError, ZeroDivisionError:
            return False
        return atom_reference(arguments["base"], point)
    point = float(value)
    if math.isnan(point):
        raise DistributionError("atom queries cannot contain NaN")
    if record.effective_atomicity == "atomless":
        return False
    if record.atom_locations is not None:
        if record.effective_atomicity == "mixed":
            return point in record.atom_locations
    if record.effective_atomicity == "discrete":
        # An integer counting reference remains fixed when individual PMF cells
        # vanish at parameter boundaries. Unsupported cells simply score -inf.
        from quivers.qiec.types import BOOL

        return (
            point in (0, 1)
            if record.element == BOOL
            else math.isfinite(point) and point.is_integer()
        )
    if distribution.family == "PointMass":
        return value == distribution.arguments["value"]
    raise DistributionError(
        f"{distribution.family} requires a compositional atom carrier"
    )


def log_atom_mass(distribution, value):
    """Score scalar singleton mass under the declared reference carrier.

    Parameters
    ----------
    distribution : HostLaw
        Unplated scalar reference distribution.
    value : float
        Scalar observation.

    Returns
    -------
    float
        Raw log atom mass or negative infinity.
    """
    from quivers.qiec import reference_families as rf

    if not atom_reference(distribution, value):
        return -math.inf
    name, arguments = distribution.family, distribution.arguments
    if name == "Normalize":
        base = arguments["base"]
        return log_atom_mass(base, value) - rf._log_mass_of(base)
    if name == "Mixture":
        return rf._logsumexp(
            [
                math.log(w) + log_atom_mass(c, value)
                for w, c in zip(
                    rf._mixture_weights(arguments),
                    rf._components(arguments, name),
                    strict=True,
                )
                if w > 0
            ]
        )
    if name == "Transformed":
        point = float(value)
        for _, inverse, _ in reversed(rf._transform_chain(arguments)):
            point = inverse(point)
        return log_atom_mass(arguments["base"], point)
    return float(distribution.log_prob(value))
