# 3. Hierarchical models

The eight-schools data [@rubin1981parallel] give one estimated treatment effect $y_j$ and its known standard error $\sigma_j$ for each of eight schools. A [hierarchical model](https://en.wikipedia.org/wiki/Multilevel_model) draws the school effects from a shared population distribution, so each estimate is shrunk toward the population mean by an amount that depends on its standard error (partial pooling) [@gelman2013bayesian]. Write the model in QVR with a plate over schools, fit it with stochastic variational inference (SVI) and with the [No-U-Turn Sampler](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) (NUTS) [@hoffman2014nuts], and compare the centered and non-centered parameterizations.

## Eight schools, centered

Let $\mu$ be the population mean, $\tau$ the scale of the variation between schools, and $\theta_j$ the effect of school $j$:

$$
\mu \sim \mathrm{Normal}(0, 5), \qquad
\tau \sim \mathrm{HalfNormal}(5), \qquad
\theta_j \sim \mathrm{Normal}(\mu, \tau), \qquad
y_j \sim \mathrm{Normal}(\theta_j, \sigma_j).
$$

=== "QVR"

    ```qvr
    object School : FinSet 8
    program eight_schools_centered : School -> School [effects=[Sample, Score]]
        sample mu  <- Normal(loc=0.0, scale=5.0)
        sample tau <- HalfNormal(scale=5.0)
        sample theta : School <- Normal(loc=mu, scale=tau)
        observe y : School <- Normal(loc=theta, scale=sigma_j)
        return theta

    export eight_schools_centered
    ```

=== "NumPyro"

    ```python
    def eight_schools(sigma_j, y=None):
        mu  = numpyro.sample("mu",  dist.Normal(0., 5.))
        tau = numpyro.sample("tau", dist.HalfNormal(5.))
        with numpyro.plate("schools", 8):
            theta = numpyro.sample("theta", dist.Normal(mu, tau))
            numpyro.sample("y", dist.Normal(theta, sigma_j), obs=y)
    ```

`sample theta : School <- Normal(...)` is a [plate](https://en.wikipedia.org/wiki/Plate_notation): it draws one value for each of the eight elements of `School`, independently given `mu` and `tau`, and later `let` steps may index the result as `theta[j]`. `observe y : School <- ...` observes one value for each school in the same way. The standard errors `sigma_j` are a [data input](01-first-model.md#data-inputs-and-the-observations-dict).

## Centered parameterization and mean-field VI

In the centered form each $\theta_j$ is drawn around $\mu$ with scale $\tau$. As $\tau$ approaches zero, the conditional distribution of the $\theta_j$ narrows, and the posterior of $(\theta, \tau)$ takes the shape of a funnel ([@neal2003slice], §8). A mean-field guide has no correlation between $\theta$ and $\tau$ and may place too much mass near $\tau = 0$:

```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

CENTERED_SRC = """
object School : FinSet 8

program eight_schools_centered : School -> School
    sample mu  <- Normal(loc=0.0, scale=5.0)
    sample tau <- HalfNormal(scale=5.0)
    sample theta : School <- Normal(loc=mu, scale=tau)
    observe y : School <- Normal(loc=theta, scale=sigma_j)
    return theta

export eight_schools_centered
"""

program = loads(CENTERED_SRC)
model   = program.morphism

sigma_j = torch.tensor([[15., 10., 16., 11., 9., 11., 10., 18.]])   # (1, 8)
y_obs   = torch.tensor([[28., 8., -3., 7., -1., 1., 18., 12.]])     # (1, 8)

guide = AutoNormalGuide(model, observed_names={"y", "sigma_j"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)
x_tensor = torch.zeros(1, 1)
observations = {"sigma_j": sigma_j, "y": y_obs}
for _ in range(50):                              # bump to ~3000 for real fits
    svi.step(x_tensor, observations)
```

To see the effect, fit both parameterizations from several initializations and compare the guide's distribution of `tau` with a well-mixed NUTS fit.

## Non-centered parameterization

The non-centered form draws a standardized effect $\eta_j \sim \mathrm{Normal}(0, 1)$ and sets $\theta_j = \mu + \tau \eta_j$ deterministically [@papaspiliopoulos2007parametrization]. The latent variables $\eta_j$ are then independent of $\tau$ a priori, and the funnel moves out of the sampled coordinates.

```qvr
object School : FinSet 8
program eight_schools_noncentered : School -> School [effects=[Sample, Score]]
    sample mu  <- Normal(loc=0.0, scale=5.0)
    sample tau <- HalfNormal(scale=5.0)
    sample eta : School <- Normal(loc=0.0, scale=1.0)
    let theta = mu + tau * eta
    observe y : School <- Normal(loc=theta, scale=sigma_j)
    return theta

export eight_schools_noncentered
```

A mean-field guide often fits this form better. Whether it agrees with NUTS is a question for the diagnostics below.

## NUTS

NUTS is a [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) (HMC) sampler that chooses its trajectory length adaptively [@neal2011mcmc; @hoffman2014nuts]:

```python
from quivers.inference import NUTSKernel, MCMC

kernel = NUTSKernel(
    target_accept=0.95,              # high target -> smaller step -> fewer divergences
    max_tree_depth=10,
)
# Small budget for documentation; bump warmup/samples for real fits.
mcmc = MCMC(
    kernel,
    num_warmup=50,
    num_samples=100,
    num_chains=2,
    init_strategy="prior",
)

result = mcmc.run(model, x_tensor, {"sigma_j": sigma_j, "y": y_obs})
print("posterior mean tau:", result.samples["tau"].mean().item())
print("R-hat tau:", result.r_hat["tau"].item())
print("ESS tau:", result.ess["tau"].item())
print("divergences:", result.total_divergences)
```

`result.r_hat` is the rank-normalized split R-hat [@vehtari2021rhat], which compares the variance between chains with the variance within them. `result.ess` is the effective sample size, and `result.total_divergences` counts trajectories whose numerical integration error blew up [@betancourt2015hierarchical]. With 50 warmup steps and 100 draws, R-hat for `tau` is usually above 1.01 and its effective sample size is a few dozen at most. Thresholds and remedies are in [chapter 8](08-diagnostics-and-comparison.md#convergence-diagnostics).

### Centered or non-centered?

The better [parameterization](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#hierarchical-models-and-the-non-centered-parameterization) depends on how informative each group's likelihood is [@betancourt2015hierarchical; @stanEfficiency]. The centered form tends to work when $\sigma_j$ is small relative to $\tau$, and the non-centered form when the group likelihoods are weak and the posterior narrows sharply as $\tau$ approaches zero. In eight schools $\sigma_j$ ranges from 9 to 18, while the prior puts substantial mass on values of $\tau$ below 10, so the group likelihoods are weak. Compare divergences and effective sample sizes under both forms on the data at hand.

### Initialization and mass matrix

`init_strategy` takes three values. `"prior"` (the default) starts each chain at a small random point near the origin of the unconstrained space, that is, $0.1\,z$ with $z$ standard Normal. `"zero"` starts at the origin, and `"guide"` draws the starting point from a fitted guide passed to `MCMC.run`.

The mass matrix is the metric that HMC adapts during warmup to rescale the latent coordinates [@neal2011mcmc]. `NUTSKernel(mass_matrix="diagonal")`, the default, adapts one variance for each coordinate. `"dense"` adapts a full covariance, which can help when latent variables are strongly correlated, at memory quadratic in their number. `"identity"` does not adapt.

## Posterior predictive

[`Predictive`](../../api/inference/predictive.md) accepts an `MCMCResult` as well as a [`Guide`](../../api/inference/guide.md):

```python
from quivers.inference import Predictive

pred = Predictive(model, posterior=result, num_samples=100)
y_hat = pred(x_tensor, {"sigma_j": sigma_j})["y"]  # (100, ..., 8)
print("predictive school 1:", y_hat[..., 0].mean().item(),
      "+/-", y_hat[..., 0].std().item())
```

## Try this

- Fit the centered form with [`AutoMultivariateNormalGuide`](../../api/inference/guide.md), whose full covariance can represent part of the funnel's correlation.
- Lower `target_accept` from 0.95 to 0.8 with longer warmup, and count divergences. A lower target gives a larger step size.
- Add a covariate $x_j$ for each school and give each school its own intercept and slope.

## Next

[Chapter 4](04-marginalize.md) sums discrete latent variables out of mixtures with the `marginalize` block.
