"""Names of the fit types for modules that the fit modules import.

`quivers.formulas._fit` and `quivers.formulas._joint_fit` import the
modules that define their methods, so those modules cannot import the fit
classes by name. The aliases here read the classes from the package when an
annotation is evaluated, after the package has bound them.
"""

from __future__ import annotations

import quivers.formulas as _formulas

type SingleFit = _formulas.BayesianFit
"""A single-response formula fit."""

type JointFit = _formulas.MultivariateFit
"""A joint formula fit."""

type Fit = _formulas.BayesianFit | _formulas.MultivariateFit
"""A single-response or joint formula fit."""
