"""Retained custom-law constructor bindings over ordinary formula predictors."""

from __future__ import annotations

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.family_catalog import FamilyCatalog


class CustomFamilySpec(ValueModel):
    """One immutable custom-law binding, with a compilation-local implementation.

    Parameters
    ----------
    catalog : FamilyCatalog
        Checked semantic contracts and native implementations, held by identity.
    bindings : tuple[tuple[str, str], ...]
        Native constructor parameters paired with formula predictor bindings.
    data : tuple[tuple[str, str], ...]
        Deterministic distributional parameters paired with source data columns.
    """

    catalog: FamilyCatalog = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    bindings: tuple[tuple[str, str], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    data: tuple[tuple[str, str], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
