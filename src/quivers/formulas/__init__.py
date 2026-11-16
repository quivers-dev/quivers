"""Brms-style formula frontend over the QVR DSL.

A user writes a regression formula (Wilkinson notation, extended with
brms-style random-effect groups) and gets a fitted Bayesian model
without hand-writing ``.qvr`` source.  The compiler emits programs
that route through the existing QVR DSL surface:

* fixed-effect linear predictors as morphism composition
  ``X >> beta`` over ``observed`` design-matrix morphisms,
* random-effect groups as plate-gather of per-level latent draws,
* responses as ``observe`` sites with the family-link kernel
  (Gaussian / Bernoulli / Binomial / Categorical / Poisson /
  NegativeBinomial / Cumulative / Beta / Gamma / Student-t /
  ZeroInflatedPoisson / HurdlePoisson / Mixture)
  registered in [`quivers.formulas.family`][quivers.formulas.family].

The implementation reuses [`formulae`](https://bambinos.github.io/formulae/)
for formula parsing (the Bambi team's pure-Python parser; supports
brms-style ``(slope | group)`` random effects, smooth terms, and
custom contrasts) and lifts the resulting `DesignMatrices`
into a typed `Formula` `didactic.api.Model`.

The frontend is the formula→QVR direction of a panproto lens; the
QVR DSL is the canonical source of truth, and the formula compiler
is a structure-preserving translation from the smaller formula
language to the QVR DSL.
"""

from __future__ import annotations

from quivers.formulas.formula import (
    FixedColumn,
    RandomTerm,
    Formula,
    FormulaData,
    ResponseLayout,
    ResponseModifiers,
    CensoringSpec,
    TruncationSpec,
    StandardErrorSpec,
    RowSelection,
    formula_from_data,
)
from quivers.formulas.family import (
    Link,
    AuxParam,
    OrdinalSpec,
    ResponseSpec,
    BaselineSpec,
    Family,
    links,
    families,
)
from quivers.formulas.compile import (
    FormulaToQVRModule,
)
from quivers.formulas._group_layout import GroupBlockLayout, GroupMemberLayout
from quivers.formulas._group_covariance import GroupCovariance, GroupCovarianceBasis
from quivers.formulas._predictor_graph import (
    NonlinearExpression,
    ComponentExpression,
    PredictorGraph,
)
from quivers.formulas._priors import (
    PriorSpec,
    PriorSet,
    PriorTarget,
    PriorInput,
    PriorDistribution,
    RegularizedHorseshoe,
    R2D2,
    normalize_priors,
)
from quivers.formulas._fit import BayesianFit, fit, formula_to_qvr, get_prior
from quivers.formulas._response_modifiers import (
    ResponseDraws,
    PredictionOptions,
    ExpectationOptions,
)

__all__ = [
    # formula
    "FixedColumn",
    "RandomTerm",
    "Formula",
    "FormulaData",
    "ResponseLayout",
    "ResponseModifiers",
    "CensoringSpec",
    "TruncationSpec",
    "StandardErrorSpec",
    "RowSelection",
    "ResponseDraws",
    "PredictionOptions",
    "ExpectationOptions",
    "formula_from_data",
    # family
    "Link",
    "AuxParam",
    "OrdinalSpec",
    "ResponseSpec",
    "BaselineSpec",
    "Family",
    "links",
    "families",
    "PriorSpec",
    "PriorSet",
    "PriorTarget",
    "PriorInput",
    "PriorDistribution",
    "RegularizedHorseshoe",
    "R2D2",
    "normalize_priors",
    # compile
    "FormulaToQVRModule",
    "GroupBlockLayout",
    "GroupCovariance",
    "GroupCovarianceBasis",
    "GroupMemberLayout",
    "NonlinearExpression",
    "ComponentExpression",
    "PredictorGraph",
    # _fit
    "BayesianFit",
    "fit",
    "formula_to_qvr",
    "get_prior",
]
