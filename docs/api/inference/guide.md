# Variational Guides

A [variational guide](https://pyro.ai/examples/svi_part_i.html#Guide) is a family of distributions
over the unobserved sites whose parameters
[optimization adjusts](../../tutorials/python/05-variational-inference.md#building-a-model) to
approximate the posterior [@pyroSVI; @blei2017variational; @hoffman2013svi]. The guides live in
submodules of `quivers.inference.guides` and share the `Guide` abstract base class and the
`LatentRegistry`. They are `AutoNormalGuide`, `AutoDeltaGuide`, `AutoMultivariateNormalGuide`,
`AutoLowRankMultivariateNormalGuide`, `AutoLaplaceApproximation`, `AutoNormalizingFlow`,
`AutoIAFGuide`, `AutoNeuralSplineGuide`, `AutoMixtureGuide`, `AutoGuideList`, `AutoStructured`, and
`AmortizedGuide`, which runs [declared encoders](../../guides/inference-svi.md#amortizedguide) on
the data. [Independent Normal guides](../../guides/inference-svi.md#guides-variational-families)
represent marginal locations and scales, and covariance and flow guides also represent
dependence.

`quivers.inference` re-exports every guide, and `quivers.inference.guides`
adds the short names that Pyro uses (`AutoNormal`, `AutoMultivariateNormal`,
`AutoLowRankMVN`, `AutoDelta`, `AutoLaplace`, `AutoIAFNormal`), each bound
to the guide class it names, and the type aliases that configure
`AutoStructured`.

The Laplace approximation uses a quadratic expansion around a mode
[@mackay2003information; @tierney1986laplace]. The autoregressive flow
implementations use masked networks [@germain2015made].
The invertible linear transform follows the construction used in Glow
[@kingma2018glow].

::: quivers.inference.guides

## Latent registry

Every guide and MCMC kernel reads the model's latent sites through a
`LatentRegistry`.

::: quivers.inference.registry

## Flow transforms

The learnable bijections that `AutoNormalizingFlow` and its subclasses
stack. A custom flow subclasses `TransformModule`.

::: quivers.inference.transforms

## Annealed importance sampling

::: quivers.inference.dais
