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
    GaussianDesignTerm,
    SplineBasis,
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
    MixtureSpec,
    Family,
    links,
    families,
)

# mixtures: brms mixture() families.
from quivers.formulas._mixture import mixture

# brms update_adterms(): edit response addition terms.
from quivers.formulas._addition_terms import update_adterms
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
from quivers.formulas._custom_family import custom_family
from quivers.formulas._custom_types import CustomFamilySpec
from quivers.formulas._fit import BayesianFit, fit, formula_to_qvr, get_prior
from quivers.formulas._smooth_effects import conditional_smooths
from quivers.formulas._gp import GaussianProcessBlock, GaussianProcessConditional
from quivers.formulas._multivariate import (
    MecorSetting,
    MultivariateFormula,
    RescorSetting,
    ResponseModel,
    bf,
    mvbf,
    set_mecor,
    set_rescor,
)
from quivers.formulas._joint_fit import MultivariateFit
from quivers.formulas._multiple import MultipleFit, combine_models, fit_multiple
from quivers.formulas._response_modifiers import (
    ResponseDraws,
    PredictionOptions,
    ExpectationOptions,
)
from quivers.formulas._postprocess_loo import KFoldPrediction, KFoldResult
from quivers.formulas._postprocess_methods import (
    JointPredictionOptions,
    SinglePredictionOptions,
    kfold_predict,
    loo_compare,
    loo_model_weights,
    model_weights,
)
from quivers.formulas._postprocess_summary import FitSummary, VarCorrBlock

# brms evidence, model averaging, conditional effects and predictive checks.
from quivers.formulas._brms_effects import (
    BrmsConditionalEffects,
    EffectFrame,
    make_conditions,
)
from quivers.formulas._evidence import (
    BayesFactor,
    bayes_factor,
    bridge_sampler,
    post_prob,
)
from quivers.formulas._mcmc_plot import mcmc_plot
from quivers.formulas._model_averaging import (
    PosteriorAverage,
    PredictionAverage,
    posterior_average,
    pp_average,
)
from quivers.formulas._pp_check import PPCheck, pp_check

# brms prior tables, flat and constant priors, and fitting controls.
from quivers.formulas._brms_priors import PriorRow, PriorTable
from quivers.formulas._fit_controls import PathfinderOptions, SamplerControl
from quivers.formulas._priors import ConstantPrior, ImproperFlat
from quivers.formulas._prior_tables import default_prior, prior_summary, validate_prior

# Keyword types that public signatures name.
from quivers.formulas._brms_effects import (
    ConditionalEffectsOptions,
    JointConditionalEffectsOptions,
)
from quivers.formulas._joint_options import FormulaFitOptions
from quivers.formulas._prior_tables import PriorTableOptions

__all__ = [
    "ConditionalEffectsOptions",
    "JointConditionalEffectsOptions",
    "FormulaFitOptions",
    "PriorTableOptions",
    # formula
    "FixedColumn",
    "GaussianDesignTerm",
    "GaussianProcessBlock",
    "GaussianProcessConditional",
    "SplineBasis",
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
    "CustomFamilySpec",
    "custom_family",
    "links",
    "families",
    "mixture",
    "MixtureSpec",
    "update_adterms",
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
    "conditional_smooths",
    # joint (multivariate) formulas
    "MultivariateFormula",
    "MultivariateFit",
    "ResponseModel",
    "RescorSetting",
    "bf",
    "mvbf",
    "set_rescor",
    "set_mecor",
    "MecorSetting",
    # multiple imputation (brm_multiple, combine_models)
    "MultipleFit",
    "combine_models",
    "fit_multiple",
    # brms post-processing (summaries, criteria, k-fold)
    "FitSummary",
    "JointPredictionOptions",
    "KFoldPrediction",
    "KFoldResult",
    "SinglePredictionOptions",
    "VarCorrBlock",
    "kfold_predict",
    "loo_compare",
    "loo_model_weights",
    "model_weights",
    # brms evidence, model averaging, conditional effects and predictive checks
    "BayesFactor",
    "BrmsConditionalEffects",
    "EffectFrame",
    "PPCheck",
    "PosteriorAverage",
    "PredictionAverage",
    "bayes_factor",
    "bridge_sampler",
    "make_conditions",
    "mcmc_plot",
    "post_prob",
    "posterior_average",
    "pp_average",
    "pp_check",
    # brms prior tables, flat and constant priors, and fitting controls
    "ConstantPrior",
    "ImproperFlat",
    "PriorRow",
    "PriorTable",
    "PathfinderOptions",
    "SamplerControl",
    "default_prior",
    "prior_summary",
    "validate_prior",
]
