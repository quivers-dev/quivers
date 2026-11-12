# Trace

An [execution trace](../../guides/inference-foundations.md#trace-and-sample-sites) records the
random choices of [one run](../../tutorials/python/05-variational-inference.md#tracing-the-model) of
a program, with the value and log density of each site [@wingate2011lightweight]. The sum of those
log densities is the run's log joint density, which inference uses to weight, optimize, or accept
proposed latent values. `trace(program, x, observations)` stacks a `TraceHandler` and runs the
program.

::: quivers.inference.trace

## Package-level exports

`quivers.inference` also re-exports the variational guides of
`quivers.inference.guides` and the kernels and runner of
`quivers.inference.mcmc`, so that `from quivers.inference import
AutoNormalGuide, NUTSKernel` works. The entries below record those import
paths with their signatures; the [Guides](guide.md) and [MCMC](mcmc.md)
pages document each class in full.

::: quivers.inference
    options:
      show_root_heading: false
      show_docstring_description: false
      show_docstring_parameters: false
      show_docstring_attributes: false
      show_docstring_examples: false
      show_docstring_raises: false
      show_docstring_returns: false
      show_source: false
      show_bases: false
      filters: ["!.*"]
      members:
        - Guide
        - GuideDraw
        - AutoNormalGuide
        - AutoMultivariateNormalGuide
        - AutoLowRankMultivariateNormalGuide
        - AutoDeltaGuide
        - AutoLaplaceApproximation
        - AutoNormalizingFlow
        - AutoIAFGuide
        - AutoNeuralSplineGuide
        - AutoMixtureGuide
        - AutoGuideList
        - AutoStructured
        - AutoNormal
        - AutoMultivariateNormal
        - AutoLowRankMVN
        - AutoDelta
        - AutoLaplace
        - AutoIAFNormal
        - MCMCKernel
        - HMCKernel
        - NUTSKernel
        - MCMC
        - MCMCResult
