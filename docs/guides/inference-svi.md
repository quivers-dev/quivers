# Variational Inference: Guides, Objectives, and SVI

This page covers the variational layer: the `Auto*Guide` family,
seven objectives and four gradient estimators, the SVI training
loop, and predictive sampling from a trained guide. The MCMC layer
([HMC](inference-mcmc.md#mcmc-hmc-and-nuts), NUTS, hybrid samplers) lives in [the MCMC
guide](inference-mcmc.md); the [trace](inference-foundations.md#trace-and-sample-sites) / [conditioning](inference-foundations.md#conditioning-on-observations) / registry
primitives that every guide consumes live in
[Inference Foundations](inference-foundations.md).

## Guides: variational families

A **guide** is a [distribution with adjustable parameters](https://pyro.ai/examples/svi_part_i.html#Guide) used to
approximate the posterior [@pyroSVI]. If `z` is an unknown mean, a simple guide
might learn a center and spread for `z`. The model still supplies the
prior and likelihood; the guide supplies candidate latent values and
their density under the approximation. [Observed sites](inference-foundations.md#trace-and-sample-sites) remain fixed to the
supplied data throughout fitting.

A diagonal, or *mean-field*, Normal treats latent coordinates as
independent in its unconstrained representation. A full-covariance Normal
can represent linear correlations but costs more parameters. Flows can bend the density, and mixtures can place mass around several
separated modes. To compare these families, fit each from several
initializations and compare their ELBOs and posterior predictive draws.

A guide $q_\phi(z \mid x, y)$ is a variational family approximating
the posterior. Eleven concrete `Auto*` guide classes cover the
available families, all documented under [Variational
Guides](../api/inference/guide.md); each is constructed from the
model and a set of observed site names. The nine single-family
guides are summarized in the table below; two combinators,
[`AutoGuideList`](../api/inference/guide.md#quivers.inference.guides.AutoGuideList)
and
[`AutoStructured`](../api/inference/guide.md#quivers.inference.guides.AutoStructured),
compose per-site guides into a joint variational family.

| Guide | Posterior structure | When to use |
|---|---|---|
| [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) | Diagonal Normal (mean-field) | Default; identifiable posterior, weak correlation |
| [`AutoMultivariateNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoMultivariateNormalGuide) | Full-rank Normal (Cholesky) | Strong posterior correlations; $D \lesssim 1000$ |
| [`AutoLowRankMultivariateNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoLowRankMultivariateNormalGuide) | Low-rank + diagonal | [Hierarchical models](programs-hierarchical.md#hierarchical-models-with-parametric-templates) with localized correlations |
| [`AutoLaplaceApproximation`](../api/inference/guide.md#quivers.inference.guides.AutoLaplaceApproximation) | Gaussian centered at MAP with covariance $-H^{-1}$ (negative inverse Hessian of the log-joint) | Local Gaussian approximation after finding a mode |
| [`AutoNormalizingFlow`](../api/inference/guide.md#quivers.inference.guides.AutoNormalizingFlow) | Composed bijector over Normal base | Skewed or curved posterior densities |
| [`AutoIAFGuide`](../api/inference/guide.md#quivers.inference.guides.AutoIAFGuide) | Inverse autoregressive flow [@kingma2016iaf] | Autoregressive transformed approximation |
| [`AutoNeuralSplineGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNeuralSplineGuide) | Rational-quadratic spline coupling [@durkan2019splines] | Flexible spline transformations |
| [`AutoMixtureGuide`](../api/inference/guide.md#quivers.inference.guides.AutoMixtureGuide) | K-component mixture of guides | Multimodal posteriors |
| [`AutoDeltaGuide`](../api/inference/guide.md#quivers.inference.guides.AutoDeltaGuide) | Dirac at MAP | Quick MAP; no uncertainty |

Every guide uses `biject_to(support)` per site, so samples always
lie inside the prior's constrained support; `log_prob` carries the
corresponding [log-det
Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).

### AutoNormalGuide

A diagonal Gaussian approximation to the posterior, with a per-site
bijector that maps unconstrained Normal samples to the prior's
constrained support:

$$q_\phi(z_i \mid x, y) = T_i\bigl(\,\mathcal{N}(\mu_i, \sigma_i)\,\bigr)$$

where $T_i = \mathsf{biject\_to}(\mathrm{support}(p_i))$ is the
bijector for site $i$'s prior support: the identity on the real
line for `Normal`, $\exp$ for `HalfNormal` / `Gamma` /
`Exponential` / `LogNormal`, sigmoid for `Beta` / `Uniform(0, 1)`
/ `LogitNormal`, an affine-shifted sigmoid for arbitrary
`Uniform(low, high)` / `TruncatedNormal`, and
[`StickBreakingTransform`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.StickBreakingTransform)
for `Dirichlet`. The learnable parameters $(\mu_i, \sigma_i)$ live
in the unconstrained space; the constrained sample $v_i = T_i(z_i)$
is always inside the prior's support, so `prior.log_prob(v_i)`
evaluates without raising `Expected value to be within the support
of ...`.

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

A delta guide puts all its mass at one fitted value, so repeated draws
return that value. Its scoring convention optimizes a MAP estimate.
The ELBO/KL identity below assumes a density with respect to the model
measure; a point mass in a continuous space violates that assumption.

A delta (point mass) approximation, i.e. a single best estimate.
The point lives in the unconstrained space and is pushed through
`biject_to(support)` at evaluation time, so it always lies inside
the prior's support:

$$q_\phi(z_i \mid x, y) = \delta_{T_i(\zeta_i)}(z_i)$$

where $\zeta_i$ is the learnable unconstrained point and $T_i$ the
same per-site bijector as `AutoNormalGuide`.

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
components. `rsample` draws from that same density: it picks a component
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
components rather than from $q_\phi$. It therefore raises an error with a
mixture guide; use `IWAEBound(num_particles=K, estimator=Reparameterized())`
or `StickingTheLanding()` instead.

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

## Objectives and gradient estimators

The **objective** says what numerical quantity the fit optimizes. The
**gradient estimator** says how to estimate its derivatives using random
draws. These are separate choices: two estimators can target the same
objective but differ in variance and applicability. A *particle* is one
set of latent draws used in a Monte Carlo estimate; increasing particles
reduces some sampling noise at additional computational cost.

The pathwise estimator differentiates through a sample such as
`mu + sigma * epsilon`. A score-function estimator instead uses
derivatives of the sampling log probability, which is useful when the
sample itself is not differentiable. The table records the advanced
choices; ordinary ELBO with a supported estimator is a simpler starting
point for understanding the loop.

[`SVI`](../api/inference/svi.md) accepts any
[`Objective`](../api/inference/elbo.md) subclass. The seven shipped
objectives include [`ELBO`](../api/inference/elbo.md#quivers.inference.objectives.ELBO):

| Objective | Bound | Use case |
|---|---|---|
| `ELBO(num_particles=K)` | $\mathbb{E}_q[\log p - \log q]$ | Default |
| `IWAEBound(K, estimator=...)` | $\mathbb{E}[\log \tfrac{1}{K}\sum_k (p/q)_k]$ | Tighter than ELBO for $K > 1$ [@burda2016iwae] |
| `RenyiBound(alpha, K)` | $\alpha$-divergence bound [@li2016renyi] | $\alpha = 0$ recovers IWAE; $\alpha \to 1$ recovers the ELBO in the limit (the constructor rejects `alpha == 1.0` as numerically singular; use `ELBO` directly) |
| `VRIWAEBound(alpha, K)` | Variational Rényi-IWAE [@daudel2023vriwae] | Controls the tradeoff between computation and approximation |
| `ChiVI(n, K)` | Chi-squared upper bound (CUBO; [@dieng2017chi]) | Over-covering approximation; the $K$-sample estimate's expectation lies below the CUBO, so treat it as an upper bound only as $K$ grows |
| `RWS(K)` | Reweighted wake-sleep, wake phase [@bornschein2015rws] | Importance-weighted wake updates with the draws held fixed; not a bound, takes no estimator |
| `DReGsBound(K)` | IWAE bound with a doubly reparameterized gradient [@tucker2019dreg] | Switch the IWAE bound and DReG estimator together |

The first five are Monte Carlo *power-mean bounds* of the per-draw log
importance ratios $\ell_k = \log p(z_k, y) - \log q_\phi(z_k)$,

$$B_a = \frac{1}{a}\log \frac{1}{K}\sum_{k=1}^{K} e^{a \ell_k},$$

with exponent $a = 0$ (the ELBO, read as the mean of $\ell_k$), $a = 1$
(IWAE), $a = 1 - \alpha$ (Rényi and VR-IWAE), or $a = n$ (`ChiVI`). Each
accepts an `estimator=` strategy, which estimates the gradient of
$\mathbb{E}[B_a]$ for both the guide and the model parameters.
`DReGsBound` fixes the strategy to `DoublyReparameterized`, and `RWS`
takes none:

| Estimator | What it does |
|---|---|
| [`Reparameterized`](../api/inference/estimators.md#quivers.inference.estimators.Reparameterized) | Pathwise gradient of $B_a$ (default except for `IWAEBound`) |
| [`StickingTheLanding`](../api/inference/estimators.md#quivers.inference.estimators.StickingTheLanding) | Detaches the variational parameters in `log_q` [@roeder2017sticking]; variance to 0 as $q \to p^*$ |
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
| `IWAEBound`, `RenyiBound`, `VRIWAEBound`, `ChiVI` | yes | only with `num_particles=1` | yes | yes |
| any of the above with an `AutoMixtureGuide` | yes | `ELBO` only | `ELBO` only | yes (independent draws) |

<!-- python: skip -->
```python
from quivers.inference import IWAEBound, DoublyReparameterized

iwae = IWAEBound(num_particles=16, estimator=DoublyReparameterized())
loss = iwae(model, guide, x, observations)
```

Multi-particle objectives run a Python `for` loop over
`num_particles`, calling `guide.rsample_strata` once per particle and
`model.log_joint` once per stratum, and stack the per-draw
log-densities into a leading `(K * S, batch)` axis with a log-weight
for each draw. An ordinary guide returns a single stratum ($S = 1$, equal
weights $1/K$); an `AutoMixtureGuide` returns one stratum per component,
weighted by the mixture weights. `torch.distributions` and the model
runtime are not vectorized over a Monte Carlo dimension, so the cost
scales linearly in the number of draws.

## ELBO: evidence lower bound

The **[evidence lower bound (ELBO)](https://pyro.ai/examples/svi_part_i.html#ELBO)** balances two terms [@pyroSVI]. `log p` rewards
latent values that the prior and observed data support. The expected
negative `log q` is the guide’s entropy, which rewards spreading mass
across the values supported by the model. For a fixed model and an
absolutely continuous guide with the required finite expectations,

$$
\log p(y\mid x)-\mathcal L(\phi)
=\mathrm{KL}\bigl(q_\phi(z\mid x,y)\,\|\,p(z\mid x,y)\bigr)\ge 0.
$$

Thus maximizing the ELBO minimizes this divergence to the posterior.
The optimizer below minimizes the *negative* ELBO. A diagonal guide, for
instance, can reach its best ELBO while missing a correlation between two
latent variables. Fit a full-covariance guide to assess that correlation,
and compare replicated observations with the data to assess predictive fit.

The [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) is
the variational objective:

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z \mid x, y)} [\log p(y, z \mid x) - \log q_\phi(z \mid x, y)]$$

It lower bounds the log marginal likelihood $\log p(y \mid x)$ and
equals it when $q_\phi = p(\cdot \mid x, y)$.

Indexed-observe steps (`observe r : N <- F(args)`) read their
response tensors from a runtime
`observations: dict[str, torch.Tensor]` keyed by the
observed-variable name. The dict is threaded through
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO)
and
[`SVI.step`](../api/inference/svi.md#quivers.inference.svi.SVI.step)
via the `observations` kwarg, alongside the domain input.

The `ELBO` class computes:

<!-- python: skip -->
```python
from quivers.inference import ELBO

model = ...  # MonadicProgram (joint p)
guide = ...  # variational q

elbo = ELBO(num_particles=10)

# Compute loss
x = torch.randn(5)
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

Variational inference chooses a tractable family of distributions and
optimizes its parameters to approximate the posterior. Stochastic
optimization estimates the objective or its gradient from samples
[@blei2017variational; @hoffman2013svi].

The
[`SVI`](../api/inference/svi.md#quivers.inference.svi.SVI) training
loop optimizes both model and guide parameters:

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
for epoch in range(100):
    x = next(data_loader)  # minibatch
    observations = {"y": x[:, -1]}
    x_input = x[:, :-1]

    loss = svi.step(x_input, observations)
    print(f"Epoch {epoch}: loss={loss:.4f}")
```

The `step` method computes the ELBO loss, backpropagates gradients,
and steps the optimizer.

## Predictive sampling

The posterior predictive has two sources of variation: uncertainty about
`z`, and response noise conditional on `z`. Drawing from both produces
replicated or future observations. With a guide, replace the exact
posterior in the integral below by the fitted approximation `q`; the
result inherits that approximation's limitations.

After training, sample from the posterior predictive:

$$p(y_\text{new} \mid x_\text{new}, \text{observations}) = \int p(y_\text{new} \mid z, x_\text{new}) \, p(z \mid x, y_\text{obs}) \, dz$$

<!-- python: skip -->
```python
from quivers.inference import Predictive

predictive = Predictive(
    model=conditioned.model,
    posterior=guide,
    num_samples=1000,
)

# Sample from posterior predictive
x_new = torch.randn(5)
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

The slope is a global random variable. Each response uses that same slope
and its own known predictor: $y_n \sim \mathcal N(w x_n, 0.1^2)$.
The QVR binding `let mu = w * predictor` computes the mean for every
response before the observation is scored.

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
domain_input = torch.arange(100)
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

`predictor` supplies known inputs; `y` supplies observed responses. Only `w`
is an unobserved sample site here. The intervals summarize the fitted guide,
so inspect fit quality before interpreting them as posterior uncertainty.

## Custom guides

Implement a custom guide by subclassing
[`Guide`](../api/inference/guide.md#quivers.inference.guides.Guide):

<!-- python: skip -->
```python
from quivers.inference.guides import Guide

class MyGuide(Guide):
    def __init__(self, model, latent_names: list[str]):
        super().__init__()
        self._latent_names = list(latent_names)
        self.mu_net = torch.nn.Linear(5, 10)
        self.sigma_net = torch.nn.Linear(5, 10)

    @property
    def latent_names(self) -> list[str]:
        """Names of latent sites this guide covers."""
        return self._latent_names

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample latent sites z ~ q(. | x). Returns {site_name: tensor}."""
        raise NotImplementedError()

    def log_prob(
        self, x: torch.Tensor, sites: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute log q(sites | x), summed across latent sites."""
        raise NotImplementedError()
```

The objectives draw through `Guide.rsample_strata`, whose default
returns the single stratum `[self.rsample(x)]` with weight one. A guide
whose sampler makes a discrete choice that `rsample` cannot
reparameterize overrides it to return weighted strata whose weighted
average estimates expectations under `log_prob`'s density, as
`AutoMixtureGuide` does.

## Further reading

- [@durkan2019splines]
- [@roeder2017sticking]
- [@tucker2019dreg]
- [@daudel2023vriwae]
- [@dieng2017chi]
- [@bornschein2015rws]
- [@li2016renyi]
- [@burda2016iwae]
- [@morningstar2021mixtures]
