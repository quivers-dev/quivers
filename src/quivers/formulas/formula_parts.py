"""Names of the authored elements of a formula.

A `FormulaPart` identifies one element of a `quivers.formulas.Formula`: the
field that holds it and its name within the field. The terms of a compiled
formula record the part they come from, so a program binding can be read
back to the element that wrote it.
"""

from __future__ import annotations

from typing import Literal

from quivers._values import ValueModel

type FormulaField = Literal[
    "fixed_columns",
    "design_terms",
    "random_terms",
    "latent_terms",
    "monotonic_terms",
    "offsets",
]
"""A field of `Formula` whose elements terms contribute."""


class FormulaPart(ValueModel):
    """One authored element of a formula.

    Parameters
    ----------
    field : str
        Field of `Formula` that holds the element.
    key : str
        Name of the element within the field.
    """

    field: FormulaField
    key: str


__all__ = ["FormulaField", "FormulaPart"]
