# Variational Inference: Guides, Objectives, and SVI

[Variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
approximates the posterior by the closest member of a tractable family of
distributions [@blei2017variational]. In quivers the family is a *guide*, the closeness
is measured by an *objective* whose gradient an *estimator* computes, and
[`SVI`](../api/inference/svi.md#quivers.inference.svi.SVI) optimizes the guide by
stochastic gradient steps. Every guide reads the
[trace](inference-foundations.md#trace-and-sample-sites),
[conditioning](inference-foundations.md#conditioning-on-observations), and latent
registry of [Inference Foundations](inference-foundations.md), and
[MCMC and Hybrid Samplers](inference-mcmc.md) covers sampling instead.

## Guides: variational families

A *guide* $q_\phi(z \mid x, y)$ is a distribution over the latent sites with
adjustable parameters $\phi$, fitted to approximate the posterior, as in Pyro
[@pyroSVI]. If `z` is an unknown mean, a simple guide learns a centre and a spread for
`z`. The model still gives the prior and the likelihood, the guide gives candidate
latent values and their density, and [observed sites](inference-foundations.md#trace-and-sample-sites)
stay fixed to the data throughout fitting.

A diagonal, or *mean-field*, normal guide treats the latent coordinates as independent
in their unconstrained representation. A full-covariance normal represents linear
correlations at the cost of more parameters, a flow can bend the density, and a mixture
can put mass near several separated modes. Comparing families means fitting each from
several initializations and comparing their ELBOs and posterior predictive draws.

The `Auto*` guides are built from a model and a set of observed site names
([Variational Guides](../api/inference/guide.md)). The single-family guides are below.
[`AutoGuideList`](../api/inference/guide.md#quivers.inference.guides.AutoGuideList)
and [`AutoStructured`](../api/inference/guide.md#quivers.inference.guides.AutoStructured)
combine guides for different sites into one family, and
[`AmortizedGuide`](#amortizedguide) computes each datum's approximation with an
encoder.

| Guide | Posterior structure | When to use |
|---|---|---|
| [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) | Diagonal Normal (mean-field) | Default; an [identifiable](https://en.wikipedia.org/wiki/Identifiability) posterior with weak correlation |
| [`AutoMultivariateNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoMultivariateNormalGuide) | Full-rank Normal (Cholesky) | Strong posterior correlations; $D \lesssim 1000$ |
| [`AutoLowRankMultivariateNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoLowRankMultivariateNormalGuide) | Low-rank + diagonal | [Hierarchical models](programs-hierarchical.md#hierarchical-models-with-parametric-templates) with localized correlations |
| [`AutoLaplaceApproximation`](../api/inference/guide.md#quivers.inference.guides.AutoLaplaceApproximation) | Gaussian centred at the maximum a posteriori (MAP) point with covariance $-H^{-1}$, the negative inverse Hessian of the log joint density | [Laplace approximation](https://en.wikipedia.org/wiki/Laplace%27s_approximation) at a mode [@tierney1986laplace] |
| [`AutoNormalizingFlow`](../api/inference/guide.md#quivers.inference.guides.AutoNormalizingFlow) | Composed bijectors over a normal base [@rezende2015flows] | Skewed or curved posterior densities |
| [`AutoIAFGuide`](../api/inference/guide.md#quivers.inference.guides.AutoIAFGuide) | Inverse autoregressive flow [@kingma2016iaf] | Autoregressive transformed approximation |
| [`AutoNeuralSplineGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNeuralSplineGuide) | Rational-quadratic spline coupling [@durkan2019splines] | Flexible spline transformations |
| [`AutoMixtureGuide`](../api/inference/guide.md#quivers.inference.guides.AutoMixtureGuide) | K-component mixture of guides | Multimodal posteriors |
| [`AutoDeltaGuide`](../api/inference/guide.md#quivers.inference.guides.AutoDeltaGuide) | Dirac at MAP | Quick MAP; no uncertainty |

Every guide maps each site through the bijector `biject_to(support)` of its prior's
support ([Inference Foundations](inference-foundations.md#the-variational-pipeline)),
so its samples lie inside that support, and its `log_prob` includes the log absolute
determinant of the
[Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).

### AutoNormalGuide

A diagonal Gaussian approximation with one bijector per site, which maps
unconstrained normal draws into the prior's support:

$$q_\phi(z_i \mid x, y) = T_i\bigl(\,\mathcal{N}(\mu_i, \sigma_i)\,\bigr)$$

where $T_i = \mathsf{biject\_to}(\mathrm{support}(p_i))$ is the
bijector for site $i$'s prior support: the identity on the real
line for `Normal`, $\exp$ for `HalfNormal` / `Gamma` /
`Exponential` / `LogNormal`, sigmoid for `Beta` / `Uniform(0, 1)`
/ `LogitNormal`, an affine-shifted sigmoid for arbitrary
`Uniform(low, high)` / `TruncatedNormal`, and
[`StickBreakingTransform`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.StickBreakingTransform)
for `Dirichlet`. The learnable parameters $(\mu_i, \sigma_i)$ live in the
unconstrained space, and the constrained draw $v_i = T_i(z_i)$ is always inside the
prior's support, so `prior.log_prob(v_i)` never raises a support error.

<!-- python: skip -->
```python
from quivers.inference import AutoNormalGuide

model = ...  # MonadicProgram
conditioned = condition(model, observations)

guide = AutoNormalGuide(conditioned.model, observed_names=set(observations))

# Sample latents from guide (each lives in its prior's support)
latents = guide.rsample(x)  # {name: (batch, *constrained_shape)}; plates: constrained_shape

# Log probability under guide (with Jacobian correction)
log_q = guide.log_prob(x, latents)
```

### AutoDeltaGuide

A delta guide puts all its mass on one fitted value, a
[Dirac measure](https://en.wikipedia.org/wiki/Dirac_measure), so repeated draws return
that value, and fitting it gives a MAP estimate. The point lives in the unconstrained
space and passes through `biject_to(support)` when it is evaluated, so it lies inside
the prior's support:

$$q_\phi(z_i \mid x, y) = \delta_{T_i(\zeta_i)}(z_i)$$

where $\zeta_i$ is the learnable unconstrained point and $T_i$ the
same bijector as for `AutoNormalGuide`. A point mass has no density with respect to
Lebesgue measure, so the identity between the ELBO and the KL divergence in
[ELBO](#elbo-evidence-lower-bound) does not apply to it.

<!-- python: skip -->
```python
from quivers.inference import AutoDeltaGuide

guide = AutoDeltaGuide(conditioned.model, observed_names=set(observations))

# Point estimate (deterministic, inside the prior's support)
z_map = guide.rsample(x)

# Delta log probability (zero; the delta term cancels in the ELBO)
log_q = guide.log_prob(x, z_map)
```

### AutoMixtureGuide

A mixture guide combines $K$ component guides with learnable weights
$\pi = \mathrm{softmax}(\texttt{mixture\_logits})$:

$$q_\phi(z) = \sum_{k=1}^{K} \pi_k\, q_k(z).$$

Each component may be any guide, so a mixture of `AutoNormalGuide`
instances can place one component on each of several separated modes.
`log_prob` evaluates the mixture density with a log-sum-exp over the
components. `rsample` draws from that density: it picks a component
index $k \sim \mathrm{Categorical}(\pi)$ and returns component $k$'s
draw, so predictive sampling from a fitted mixture guide reproduces the
distribution that `log_prob` scores. Its draws carry gradient to the
chosen component's parameters but not to the mixture logits, since the
component choice is discrete.

Training does not go through `rsample`. Every objective calls
[`Guide.rsample_strata`](../api/inference/guide.md#quivers.inference.guides.Guide.rsample_strata),
which a mixture implements by enumerating its components: one
reparameterized draw $z_k \sim q_k$ per component, weighted by $\pi_k$.
The ELBO is thus estimated as the stratified ELBO
[@morningstar2021mixtures]

$$\mathcal{L}(\phi) = \sum_{k=1}^{K} \pi_k\,
\mathbb{E}_{q_k}\bigl[\log p(y, z \mid x) - \log q_\phi(z)\bigr],$$

which equals the ELBO of the mixture density exactly. Gradients reach the
component parameters through each reparameterized draw and reach the
mixture logits through the finite sum. The importance-weighted objectives
use the same weights: `IWAEBound` becomes the stratified IWAE bound
$\log \frac{1}{K'}\sum_{j}\sum_{k} \pi_k\, p(y, z_{jk})/q_\phi(z_{jk})$
over $K'$ particles, and `RenyiBound`, `VRIWAEBound`, `ChiVI`, and `RWS`
weight each draw the same way. A mixture nested inside another mixture or
inside an `AutoGuideList` part is enumerated as well (an `AutoGuideList`
takes the product of its parts' strata), so each step costs $K$ model
evaluations per particle.

The `DoublyReparameterized` estimator rewrites the score of the density
each particle was drawn from, and stratified draws come from the
components rather than from $q_\phi$. It thus raises an error with a
mixture guide. `IWAEBound(num_particles=K, estimator=Reparameterized())` and
`StickingTheLanding()` work with one.

<!-- python: skip -->
```python
from quivers.inference import ELBO, SVI, AutoMixtureGuide, AutoNormalGuide

components = [
    AutoNormalGuide(conditioned.model, observed_names=set(observations))
    for _ in range(3)
]
guide = AutoMixtureGuide(components)
svi = SVI(conditioned.model, guide, optimizer, ELBO(num_particles=4))

# After fitting: draws from the mixture density q
z = guide.rsample(x)
weights = guide.mixture_weights()
```

### AmortizedGuide

In amortized inference the variational posterior of a latent is a
kernel applied to the data, $q_\phi(z \mid y) = e_\phi(y)$, as in the
encoder of a variational autoencoder [@kingma2014autoencoding]. One set
of encoder weights then serves every datum, new ones included.
[`AmortizedGuide`](../api/inference/guide.md#quivers.inference.guides.AmortizedGuide)
runs such kernels: each binding names a latent site, an encoder (any
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism) with an exact
density, such as a declared morphism or deterministic layers followed by
one stochastic kernel), and the datum it reads.

The encoder is applied to every row of its datum. A site drawn once per
run reads a datum with a leading batch axis, so an input of `B` rows (`B`
[independent runs](inference-foundations.md#input-rows-are-independent-runs))
gives each run its own posterior. A local latent in a plate reads one datum
row per plate row. When the encoder's support is the site's support its
draw is the site's value. When the encoder is real-valued and the site
is not, the draw passes through the site's bijector $T$ and the density
carries the Jacobian,
$\log q(v) = \log e(T^{-1}(v) \mid y) + \log |\det J_{T^{-1}}(v)|$.
The guide checks each binding against the registry and refuses an
unknown site, an encoder whose support or event size does not fit, a
datum that is missing or whose rows do not line up with the site, and a
composition with a stochastic hidden layer, whose `log_prob` scores a
reference path rather than a density.

The data reach the guide through
[`Guide.observe`](../api/inference/guide.md#quivers.inference.guides.Guide.observe),
which every objective calls with the step's observations. Calling it
directly samples the posterior of held-out data. Sites without an
encoder go to another guide in an `AutoGuideList`, and
`uncovered_sites` lists them. Every objective and estimator applies,
since the guide has a reparameterized sampler and a density at any
point.

<!-- python: skip -->
```python
import torch

from quivers.dsl import loads
from quivers.inference import ELBO, SVI, AmortizedGuide

source = """
object UnitSpace : Real 1
object Latent : Real 2
object ObsSpace : Real 4

morphism prior : UnitSpace -> Latent ~ Normal
morphism decoder : Latent -> ObsSpace [param_source=mlp(32)] ~ Normal

program vae : UnitSpace -> ObsSpace
    sample z <- prior
    observe y <- decoder(z)
    return y

export vae
"""
encoder_source = """
object ObsSpace : Real 4
object Latent : Real 2
morphism encoder : ObsSpace -> Latent [param_source=mlp(32)] ~ Normal
export encoder
"""
model = loads(source).morphism
encoder = loads(encoder_source).morphism

y = torch.randn(256, 4)                 # one datum per run
x = torch.zeros(256, 1)                 # 256 independent runs
guide = AmortizedGuide(model, {"y"}, {"z": (encoder, "y")}, x=x, observations={"y": y})
params = list(model.parameters()) + list(guide.parameters())
svi = SVI(model, guide, torch.optim.Adam(params, lr=1e-2), ELBO(num_particles=4))
losses = [svi.step(x, {"y": y}) for _ in range(200)]

y_new = torch.randn(10, 4)
guide.observe({"y": y_new})             # the posterior of new data
z_new = guide.rsample(torch.zeros(10, 1))["z"]
print(losses[0], losses[-1])   # the negative ELBO falls
print(z_new.shape)             # (10, 2)
```

## Objectives and gradient estimators

The *objective* is the quantity the fit optimizes, and the *gradient estimator* is how
its derivatives are estimated from random draws. They are separate choices: two
estimators can target one objective with different variance and applicability. A
*particle* is one set of latent draws in a Monte Carlo estimate, and more particles
reduce sampling noise at more cost. The
[reparameterization (pathwise) estimator](https://en.wikipedia.org/wiki/Reparameterization_trick)
differentiates through a draw written as `mu + sigma * epsilon`
[@kingma2014autoencoding], and the score-function estimator uses the derivative of the
sampling log density, which also works when the draw is not differentiable
[@ranganath2014bbvi]. The ELBO with its default estimator is the usual starting point.

[`SVI`](../api/inference/svi.md) accepts any [`Objective`](../api/inference/elbo.md)
subclass. The objectives are:

| Objective | Bound | Use case |
|---|---|---|
| `ELBO(num_particles=K)` | $\mathbb{E}_q[\log p - \log q]$ | Default |
| `IWAEBound(K, estimator=...)` | $\mathbb{E}[\log \tfrac{1}{K}\sum_k (p/q)_k]$ | Tighter than ELBO for $K > 1$ [@burda2016iwae] |
| `RenyiBound(alpha, K)` | $\alpha$-divergence bound [@li2016renyi] | $\alpha = 0$ recovers IWAE; $\alpha \to 1$ recovers the ELBO in the limit (the constructor rejects `alpha == 1.0` as numerically singular, and `ELBO` computes that case) |
| `VRIWAEBound(alpha, K)` | Variational Rényi-IWAE [@daudel2023vriwae] | Controls the tradeoff between computation and approximation |
| `ChiVI(n, K)` | $\chi^n$ upper bound $\mathrm{CUBO}_n = \tfrac{1}{n}\log \mathbb{E}_q[(p/q)^n]$ [@dieng2017chi] | Over-covering approximation; minimizes $\exp(n\,\mathrm{CUBO}_n)$ with an unbiased gradient and reports the $K$-sample CUBO estimate for the sandwich $\mathrm{ELBO} \le \log p(y) \le \mathrm{CUBO}_n$ |
| `RWS(K)` | Reweighted wake-sleep, wake phase [@bornschein2015rws] | Importance-weighted wake updates with the draws held fixed; not a bound, takes no estimator |
| `DReGsBound(K)` | IWAE bound with a doubly reparameterized gradient [@tucker2019dreg] | Switch the IWAE bound and DReG estimator together |

`ELBO`, `IWAEBound`, `RenyiBound`, `VRIWAEBound`, and `DReGsBound` are Monte Carlo
estimates of the variational Rényi bound family [@li2016renyi], computed as a power
mean of the log importance ratios $\ell_k = \log p(z_k, y) - \log q_\phi(z_k)$ of the
draws:

$$B_a = \frac{1}{a}\log \frac{1}{K}\sum_{k=1}^{K} e^{a \ell_k},$$

with exponent $a = 0$ (the ELBO, read as the mean of $\ell_k$), $a = 1$
(IWAE), or $a = 1 - \alpha$ (Rényi and VR-IWAE). Each accepts an
`estimator=` strategy, which estimates the gradient of $\mathbb{E}[B_a]$
for both the guide and the model parameters. `DReGsBound` fixes the
strategy to `DoublyReparameterized`, and `RWS` takes none.

`ChiVI` is an upper bound, and the log of a Monte Carlo average is
biased *downward* by Jensen's inequality, so minimizing the $K$-sample
estimate $\tfrac{1}{n}\log \tfrac{1}{K}\sum_k w_k^n$ would neither
keep the bounding property nor give an unbiased gradient. Following
Dieng et al. [@dieng2017chi], `ChiVI` minimizes the exponentiated bound
$L_n = \exp(n\,\mathrm{CUBO}_n) = \mathbb{E}_q[w^n]$, which has the same
minimizers and the unbiased estimate $\tfrac{1}{K}\sum_k w_k^n$. To
avoid overflow, particle $k$'s term is divided by $e^{m_{-k}}$, where
$m_{-k}$ is the largest $n \log w_j$ among the *other* particles. That
scale is independent of particle $k$'s draws, so the expected update is
a positive multiple of $\nabla L_n$ and the stationary points are
unchanged. `ChiVI` thus needs `num_particles >= 2` and an order
$n > 1$. The value it returns is the $K$-sample CUBO estimate, directly
comparable with the negated `ELBO` loss. It is biased downward at finite
$K$, so for small $K$ it can fall below $\log p(y)$, and the bias
vanishes as $K$ grows. Start `ChiVI` from a guide wider than the
posterior: the gradient's variance is finite only when
$\mathbb{E}_q[w^{2n}]$ is.

| Estimator | What it does |
|---|---|
| [`Reparameterized`](../api/inference/estimators.md#quivers.inference.estimators.Reparameterized) | Pathwise gradient of $B_a$ (default except for `IWAEBound`) |
| [`StickingTheLanding`](../api/inference/estimators.md#quivers.inference.estimators.StickingTheLanding) | Detaches the variational parameters in `log_q` [@roeder2017sticking], whose variance goes to 0 as $q \to p^*$ |
| [`DoublyReparameterized`](../api/inference/estimators.md#quivers.inference.estimators.DoublyReparameterized) | DReG [@tucker2019dreg; @daudel2023vriwae]: rewrites the score term with the reparameterization identity, weighting draw $k$ by $(1-a)\bar w_k + a \bar w_k^2$ with $\bar w = \mathrm{softmax}(a\ell)$; default for `IWAEBound` |
| [`ScoreFunction`](../api/inference/estimators.md#quivers.inference.estimators.ScoreFunction) | REINFORCE [@ranganath2014bbvi]: the objective draws from the guide and holds the draws fixed, so the guide's `rsample` need not be differentiable |

Not every pair has an unbiased estimator. Dropping the score term, as
sticking the landing does, is unbiased only when that term has mean zero:
for the ELBO, or for a single draw. The objectives check the pair at
construction and on every call, and raise `ValueError` naming a valid
estimator instead of silently using another one:

| Objective | `Reparameterized` | `StickingTheLanding` | `DoublyReparameterized` | `ScoreFunction` |
|---|---|---|---|---|
| `ELBO` | yes | yes | yes (equals sticking the landing) | yes |
| `IWAEBound`, `RenyiBound`, `VRIWAEBound` | yes | only with `num_particles=1` | yes | yes |
| `ChiVI` | yes (Dieng et al., Eq. 3) | no | no | yes (their Algorithm 1) |
| any of the above with an `AutoMixtureGuide` | yes | `ELBO` only | `ELBO` only | yes (independent draws) |

<!-- python: skip -->
```python
from quivers.inference import IWAEBound, DoublyReparameterized

iwae = IWAEBound(num_particles=16, estimator=DoublyReparameterized())
loss = iwae(model, guide, x, observations)
```

Multi-particle objectives run a Python `for` loop over
`num_particles`, calling `guide.rsample_strata` once per particle and
`model.log_joint` once per stratum, and stack the
log-densities into a leading `(K * S, batch)` axis with a log-weight
for each draw. An ordinary guide returns a single stratum ($S = 1$, equal
weights $1/K$), and an `AutoMixtureGuide` returns one stratum per component,
weighted by the mixture weights. `torch.distributions` and the model
runtime are not vectorized over a Monte Carlo dimension, so the cost
scales linearly in the number of draws.

## ELBO: evidence lower bound

The [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) is

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z \mid x, y)} [\log p(y, z \mid x) - \log q_\phi(z \mid x, y)].$$

Its first term rewards latent values that the prior and the data support, and the
second, the guide's entropy, rewards spreading mass over them [@pyroSVI]. For a guide
with a density and the required finite expectations,

$$
\log p(y\mid x)-\mathcal L(\phi)
=\mathrm{KL}\bigl(q_\phi(z\mid x,y)\,\|\,p(z\mid x,y)\bigr)\ge 0,
$$

so $\mathcal L$ is a lower bound on the log marginal likelihood, equal to it when $q_\phi$
is the posterior, and maximizing it minimizes the
[KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) to the
posterior. The optimizer minimizes the *negative* ELBO. A diagonal guide can reach its
best ELBO while missing a correlation between two latent variables, which a
full-covariance guide would show, and replicated observations compared with the data
show predictive fit. The observed values reach the objective in the
[`observations` dict](programs.md#the-observations-dict), passed to
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) and
[`SVI.step`](../api/inference/svi.md#quivers.inference.svi.SVI.step) with the input.

The `ELBO` class computes:

<!-- python: skip -->
```python
from quivers.inference import ELBO

model = ...  # MonadicProgram (joint p)
guide = ...  # variational q

elbo = ELBO(num_particles=10)

# Compute loss
x = torch.zeros(1, 1)
observations = {"y": y_obs}
loss = elbo(model, guide, x, observations)  # negative ELBO (for minimization)

loss.backward()  # backprop through both model and guide
```

Internally, the ELBO:

1. Draws latent variables with `guide.rsample_strata()`: one draw
   $z \sim q_\phi(\cdot \mid x, y)$ with weight $\omega = 1$ for an
   ordinary guide, or one draw per component with weight $\pi_k$ for an
   [`AutoMixtureGuide`](#automixtureguide).
2. Computes $\log p(y, z \mid x)$ via
   [`model.log_joint()`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint).
3. Computes $\log q_\phi(z \mid x, y)$ via `guide.log_prob()`.
4. Returns $\frac{1}{n}\sum_i \sum_s \omega_{s} [\log q - \log p]$ over
   the $n$ particles and their strata $s$.

## SVI: stochastic variational inference

Stochastic variational inference optimizes the objective with gradients estimated from
draws [@hoffman2013svi]. The
[`SVI`](../api/inference/svi.md#quivers.inference.svi.SVI) loop updates the model's and
the guide's parameters together:

<!-- python: skip -->
```python
from quivers.inference import ELBO, SVI
import torch.optim as optim

model = ...   # MonadicProgram
guide = ...   # Guide
elbo  = ELBO(num_particles=5)

optimizer = optim.Adam(
    list(model.parameters()) + list(guide.parameters()),
    lr=1e-3,
)

svi = SVI(model, guide, optimizer, elbo)

# Training loop
for step in range(100):
    loss = svi.step(x, observations)
    print(f"Step {step}: loss={loss:.4f}")
```

`step` computes the loss, backpropagates, and steps the optimizer. Each input row is
an [independent run](inference-foundations.md#input-rows-are-independent-runs) with its
own guide draws, so a minibatch of input rows is a minibatch of runs, and a program
whose observations live in a plate takes an input with one row.

## Minibatches: subsampled plates

A program whose records live in a plate scores every record on every
step. Stochastic variational inference [@hoffman2013svi] scores a random
subset of the records instead and multiplies their contribution by the
number of records over the subset's size. The minibatch estimate of the
ELBO is then unbiased for the full-data ELBO, and so is its gradient, so
SVI converges to the same posterior while each step reads only the
subset's observations.

[`subsample`][quivers.effects.subsample.subsample] does this for one plate.
Inside its block, every site whose plate axes include the plate, a plate
latent or an observation, is scored at the index's rows, times the plate's
size `N` over the index's size `B`. The global sites, the latents outside
the plate, are scored in full and unscaled, so the prior and the guide's
global entropy count once per step rather than once per record. Several
blocks compose: a site in two subsampled plates takes the product of the
two factors, and a site in one of two sibling plates takes its own
plate's factor alone. A trace records each scaled site's factor as
`metadata["subsample_scale"]`.

<!-- compile: standalone -->
```qvr
object Item : FinSet 40

program mean_model : Item -> Item
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Item <- Normal(loc=mu, scale=1.0)
    return mu

export mean_model
```

<!-- python: skip -->
```python
import torch
from quivers.dsl import load
from quivers.effects import subsample
from quivers.inference import ELBO, SVI, AutoNormalGuide

model = load("mean_model.qvr").morphism   # the program above
y = torch.randn(40) + 1.0                 # one observation per Item

guide = AutoNormalGuide(model, {"y"})
svi = SVI(model, guide, torch.optim.Adam(guide.parameters(), lr=0.02), ELBO())
x = torch.zeros(1, 1)
for step in range(1500):
    index = torch.randperm(40)[:8].sort().values
    with subsample("Item", index):
        svi.step(x, {"y": y[index]})
```

The index lists distinct rows in increasing order. An observation may be
given at the subset's size, as above, its rows read in the order of the
index, or at the plate's full extent. Within a run the plate keeps its
declared extent: its latents are drawn at every row and the program's
lets compute every row, so a row outside the subset keeps a well-defined
value and only its score is left out. The saving is thus in the scored
terms and in the data a step reads, not in the lets of the program.

A local latent inside the subsampled plate, one variable per record, is
handled by [`AutoNormalGuide`][quivers.inference.guides.AutoNormalGuide],
whose density factors over the plate's rows: it scores the latent at the
subset's rows with the same factor, and a row outside the subset receives
no gradient from the step. [`AutoDeltaGuide`][quivers.inference.guides.AutoDeltaGuide],
whose density is constant, also admits it. A guide whose density couples
the rows, such as
[`AutoMultivariateNormalGuide`][quivers.inference.guides.AutoMultivariateNormalGuide],
refuses a local latent in a subsampled plate rather than score a biased
objective. It remains usable when the plate holds only observations.

Only an objective linear in the log joint stays unbiased under
subsampling: the ELBO, with any number of particles and any gradient
estimator. The importance-weighted and Rényi bounds, `ChiVI`, and `RWS`
exponentiate or self-normalize the log weights and refuse a subsampled
plate, and so do the consumers of the exact posterior, MCMC and
`PotentialFn`, the Laplace approximation, `AutoDAIS`, and
`lift_to_bayesian_program`. A site in the plate that is neither a plate draw nor a plated
observation (a grouped marginalization, for instance) has no density for each row,
density and is refused, as is a plate whose extent the program leaves to
its data.

## Predictive sampling

The posterior predictive distribution has two sources of variation, uncertainty about
`z` and response noise given `z`, and drawing from both gives replicated or future
observations:

$$p(y_\text{new} \mid x_\text{new}, \text{observations}) = \int p(y_\text{new} \mid z, x_\text{new}) \, p(z \mid x, y_\text{obs}) \, dz$$

With a guide, the fitted $q$ replaces the exact posterior in this integral, and the
result inherits the guide's limitations.

<!-- python: skip -->
```python
from quivers.inference import Predictive

predictive = Predictive(
    model=conditioned.model,
    posterior=guide,
    num_samples=1000,
)

# Sample from posterior predictive
x_new = torch.zeros(1, 1)
samples = predictive(x_new)              # dict[str, torch.Tensor]
y_new_samples = samples["y"]             # shape (num_samples, batch, ...)

# Posterior mean and credible intervals
y_mean = y_new_samples.mean(dim=0)
y_low = y_new_samples.quantile(0.025, dim=0)
y_high = y_new_samples.quantile(0.975, dim=0)
```

The [`Predictive`](../api/inference/predictive.md) driver:

1. Samples latents from the guide: $z \sim q_\phi(\cdot \mid x, y_\text{obs})$.
2. Samples outcomes: $y_\text{new} \sim p(\cdot \mid z, x_\text{new})$.
3. Returns the ensemble.

## Full example: Bayesian linear regression

The slope $w$ is one random variable shared by all responses, and each response has its
own known predictor: $y_n \sim \mathcal N(w x_n, 0.1^2)$. The binding
`let mu = w * predictor` computes every response's mean before the observation is
scored, and the input is one row, since the 100 responses come from the plate `Row`.

<!-- python: skip -->
```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

program = loads("""
object Row : FinSet 100
object Value : Real 1
program regression : Row -> Value
    sample w <- Normal(loc=0.0, scale=1.0)
    let mu = w * predictor
    observe y : Row <- Normal(loc=mu, scale=0.1)
    return y
export regression
""").morphism

x_obs = torch.linspace(-1.0, 1.0, 100)
y_obs = 2.0 * x_obs + 0.1 * torch.randn(100)
observations = {"predictor": x_obs, "y": y_obs}
domain_input = torch.zeros(1, 1)   # one run; its 100 rows come from Row
guide = AutoNormalGuide(program, observed_names={"y"})
optimizer = torch.optim.Adam(
    list(program.parameters()) + list(guide.parameters()), lr=1e-2
)
svi = SVI(program, guide, optimizer, ELBO(num_particles=5))
for _ in range(500):
    loss = svi.step(domain_input, observations)

# Summarize the slope site itself, not the returned response.
w_draws = torch.stack([guide.rsample(domain_input)["w"] for _ in range(500)])
print(w_draws.mean(), w_draws.quantile(0.025), w_draws.quantile(0.975))
```

`predictor` is a data input and `y` the observed responses, so `w` is the only latent
site. The intervals summarize the fitted guide,
so inspect fit quality before interpreting them as posterior uncertainty.

## Custom guides

A custom guide subclasses
[`Guide`](../api/inference/guide.md#quivers.inference.guides.Guide) and gives
`latent_names`, `rsample`, which returns one draw per input row of every latent site it
covers, and `log_prob`, which returns one log density per row. A normal guide for the
slope of the regression above:

<!-- python: skip -->
```python
import torch
from torch.distributions import Normal
from quivers.inference.guides import Guide

class SlopeGuide(Guide):
    def __init__(self, site: str):
        super().__init__()
        self._site = site
        self.loc = torch.nn.Parameter(torch.zeros(1))
        self.log_scale = torch.nn.Parameter(torch.zeros(1))

    @property
    def latent_names(self) -> list[str]:
        return [self._site]

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        q = Normal(self.loc, self.log_scale.exp())
        return {self._site: q.rsample((x.shape[0],))}   # (batch, 1)

    def log_prob(self, x: torch.Tensor, sites: dict[str, torch.Tensor]) -> torch.Tensor:
        q = Normal(self.loc, self.log_scale.exp())
        return q.log_prob(sites[self._site]).sum(-1)     # (batch,)

guide = SlopeGuide("w")
svi = SVI(program, guide, torch.optim.Adam(guide.parameters(), lr=2e-2), ELBO(num_particles=5))
for _ in range(800):
    svi.step(domain_input, observations)
```

The objectives draw through `Guide.rsample_strata`, which returns
[`GuideDraw`](../api/inference/guide.md#quivers.inference.guides.GuideDraw)
records: the sites, the guide log-density the objective pairs with
them, and a stratum log-weight. Its default returns one draw from
`rsample`, scored by `log_prob`, with weight one. A guide whose sampler
makes a discrete choice that `rsample` cannot reparameterize overrides
it to return weighted strata whose weighted average estimates
expectations under the guide, as `AutoMixtureGuide` does. A guide whose density
exists only along the path that produced a draw returns that density with the draw,
as `AutoDAIS` does.
