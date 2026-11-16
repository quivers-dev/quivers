"""QVR adapter for the dependency-free shared constructor schemas.

The core and source boundaries consume the same complete alternatives. This
module retains the source tooling import path without maintaining another table.
"""

from quivers._family_schemas import (
    DISTRIBUTION_FAMILIES,
    FAMILY_ALIASES,
    PARAMETER_ALIASES,
    family_parameter_names,
    family_parameterizations,
    render_parameterizations as render_parameterizations,
)

__all__ = [
    "DISTRIBUTION_FAMILIES",
    "FAMILY_ALIASES",
    "PARAMETER_ALIASES",
    "family_parameter_names",
    "family_parameterizations",
]
