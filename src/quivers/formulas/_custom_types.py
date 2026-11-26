"""The compilation-local law and data columns of a custom family."""

from __future__ import annotations

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.family_catalog import FamilyCatalog


class CustomFamilySpec(ValueModel):
    """The compilation-local law of a custom family and its data parameters.

    The family's `Family.observe_args` name the formula binding passed to
    each native parameter.

    Parameters
    ----------
    catalog : FamilyCatalog
        Checked semantic contracts and native implementations, held by identity.
    data : tuple[tuple[str, str], ...]
        Deterministic distributional parameters paired with source data columns.
    """

    catalog: FamilyCatalog = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    data: tuple[tuple[str, str], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
