# Variational Guides

A variational guide is an adjustable distribution over unobserved variables.
Optimization changes its parameters to approximate the posterior. Independent Normal
guides represent marginal locations and scales; covariance and flow guides can also
represent dependence. Compare fits from several initializations when assessing a more
flexible guide.

Pyro's [guide introduction](https://pyro.ai/examples/svi_part_i.html#Guide) explains the role of the adjustable distribution used to approximate a posterior [@pyroSVI].

See [variational families and their assumptions](../../guides/inference-svi.md#guides-variational-families) and
[a regression fitted with variational inference](../../tutorials/python/05-variational-inference.md#building-a-model).

For variational approximation and its limits, see [@blei2017variational; @hoffman2013svi].

Variational guide distributions for approximate inference. The guides
live in submodules of `quivers.inference.guides` and share the `Guide`
ABC and `LatentRegistry` introspection layer. They are
`AutoNormalGuide`, `AutoDeltaGuide`, `AutoMultivariateNormalGuide`,
`AutoLowRankMultivariateNormalGuide`, `AutoLaplaceApproximation`,
`AutoNormalizingFlow`, `AutoIAFGuide`, `AutoNeuralSplineGuide`,
`AutoMixtureGuide`, `AutoGuideList`, and `AutoStructured`.

`quivers.inference` re-exports every guide, and `quivers.inference.guides`
adds the Pyro-style short names (`AutoNormal`, `AutoMultivariateNormal`,
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
