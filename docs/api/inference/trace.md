# Trace

A trace records the random choices and scores from one execution. Adding the relevant
log-density contributions gives that execution’s joint score. Inference procedures use
such scores to weight, optimize, or accept proposed latent values when approximating a
posterior.

See [what a trace records](../../guides/inference-foundations.md#trace-and-sample-sites) and
[inspecting a model execution](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For the distinction between a model and posterior inference, see [@gelman2013bayesian].

Program trace data structures and trace-based inference.

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
