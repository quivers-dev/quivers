"""Inference algorithms for probabilistic programs.

This module provides the inference layer for quivers: execution
tracing, conditioning on observations, the
`quivers.inference.registry.LatentRegistry` introspection
helper, automatic variational guide construction, variational
objectives, optimisation drivers, MCMC and hybrid samplers,
normalizing-flow transforms, and posterior-predictive sampling.

Submodules
----------
trace : Execution trace and sample site recording.
conditioning : Observation marking and ``condition()`` factory.
registry : `LatentRegistry`, the per-site introspection
    helper every guide and MCMC kernel consumes.
guides : Variational guide families (mean-field, full-rank /
    low-rank Gaussian, normalizing-flow, structured, mixture,
    Laplace). Re-exported here at top level.
estimators : Gradient estimators (reparameterized, sticking-the-
    landing, DReG, score-function).
objectives : Variational objectives (ELBO, IWAEBound, RenyiBound,
    VRIWAEBound, ChiVI, RWS, DReGsBound).
svi : Stochastic variational inference training loop.
predictive : Posterior predictive sampling.
transforms : Learnable bijections that normalizing-flow guides stack.
dais : Differentiable annealed importance sampling guide.
warmup : Variational warmup before HMC.
lifts : Bayesian lifts of non-Bayesian models.
"""

from __future__ import annotations

from quivers.inference.trace import Trace, SampleSite, trace
from quivers.inference.conditioning import Conditioned, condition
from quivers.inference.registry import LatentRegistry, LatentSite
from quivers.inference.guides import (
    AutoDelta,
    AutoDeltaGuide,
    AmortizedGuide,
    AutoGuideList,
    AutoIAFGuide,
    AutoIAFNormal,
    AutoLaplace,
    AutoLaplaceApproximation,
    AutoLowRankMVN,
    AutoLowRankMultivariateNormalGuide,
    AutoMixtureGuide,
    AutoMultivariateNormal,
    AutoMultivariateNormalGuide,
    AutoNeuralSplineGuide,
    AutoNormal,
    AutoNormalGuide,
    AutoNormalizingFlow,
    AutoStructured,
    Guide,
    GuideDraw,
)
from quivers.inference.estimators import (
    DoublyReparameterized,
    GradientEstimator,
    ParticleLogDensities,
    Reparameterized,
    ScoreFunction,
    StickingTheLanding,
    leave_one_out_log_scale,
    power_mean_bound,
)
from quivers.inference.objectives import (
    ELBO,
    RWS,
    ChiVI,
    DReGsBound,
    IWAEBound,
    Objective,
    RenyiBound,
    VRIWAEBound,
)
from quivers.inference.mcmc import (
    AdaptationSchedule,
    CycleKernel,
    GibbsKernel,
    HMCKernel,
    IndependenceProposal,
    MetropolisKernel,
    MixtureKernel,
    Proposal,
    RandomWalkProposal,
    MCMC,
    MCMCCheckpoint,
    MCMCKernel,
    MCMCResult,
    NUTSKernel,
)
from quivers.inference.svi import SVI
from quivers.inference.smc import (
    SMC,
    ResamplingScheme,
    SMCResult,
    TemperedPotential,
    resample,
)
from quivers.inference.predictive import Predictive
from quivers.inference.dais import AutoDAIS
from quivers.inference.warmup import WarmupThenHMC
from quivers.inference.transforms import (
    MADE,
    AffineCouplingTransform,
    BatchNormTransform,
    InverseAutoregressiveTransform,
    LULinearTransform,
    MaskedAutoregressiveTransform,
    NeuralSplineCouplingTransform,
    TransformModule,
    alternating_mask,
    half_mask,
    make_coupling_mlp,
)
from quivers.inference.lifts import (
    bayesian_lift_parameters,
    lift_to_bayesian_program,
    lift_from_log_prob,
    monte_carlo_log_joint,
)

__all__ = [
    "Trace",
    "SampleSite",
    "trace",
    "Conditioned",
    "condition",
    "LatentRegistry",
    "LatentSite",
    "Guide",
    "GuideDraw",
    "AutoNormalGuide",
    "AutoMultivariateNormalGuide",
    "AutoLowRankMultivariateNormalGuide",
    "AutoDeltaGuide",
    "AutoLaplaceApproximation",
    "AutoNormalizingFlow",
    "AutoIAFGuide",
    "AutoNeuralSplineGuide",
    "AutoMixtureGuide",
    "AmortizedGuide",
    "AutoGuideList",
    "AutoStructured",
    "AutoNormal",
    "AutoMultivariateNormal",
    "AutoLowRankMVN",
    "AutoDelta",
    "AutoLaplace",
    "AutoIAFNormal",
    "MCMCKernel",
    "AdaptationSchedule",
    "HMCKernel",
    "MetropolisKernel",
    "Proposal",
    "RandomWalkProposal",
    "IndependenceProposal",
    "GibbsKernel",
    "CycleKernel",
    "MixtureKernel",
    "NUTSKernel",
    "MCMC",
    "MCMCResult",
    "MCMCCheckpoint",
    "AutoDAIS",
    "WarmupThenHMC",
    "SMC",
    "SMCResult",
    "TemperedPotential",
    "ResamplingScheme",
    "resample",
    "Objective",
    "ELBO",
    "IWAEBound",
    "RenyiBound",
    "ChiVI",
    "RWS",
    "DReGsBound",
    "VRIWAEBound",
    "GradientEstimator",
    "ParticleLogDensities",
    "power_mean_bound",
    "leave_one_out_log_scale",
    "Reparameterized",
    "StickingTheLanding",
    "DoublyReparameterized",
    "ScoreFunction",
    "SVI",
    "Predictive",
    "bayesian_lift_parameters",
    "lift_to_bayesian_program",
    "lift_from_log_prob",
    "monte_carlo_log_joint",
    "TransformModule",
    "AffineCouplingTransform",
    "MaskedAutoregressiveTransform",
    "InverseAutoregressiveTransform",
    "NeuralSplineCouplingTransform",
    "LULinearTransform",
    "BatchNormTransform",
    "MADE",
    "make_coupling_mlp",
    "alternating_mask",
    "half_mask",
]
