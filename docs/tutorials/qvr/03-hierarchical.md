# 3. Hierarchical models

The eight-schools dataset [@rubin1981parallel] is a common stress test for hierarchical Bayesian inference: eight schools, one observed treatment effect and standard error per school, and the question of how much each school's effect borrows strength from the global mean. The model is small, but its centered parameterization can produce funnel-shaped posterior geometry that is difficult for mean-field VI and Hamiltonian Monte Carlo.

This chapter covers:

- The plate-draw syntax (`v : G <- Normal(loc=0, scale=sigma)`) for per-group random effects.
- Centered vs non-centered parameterizations and how to write each in QVR.
- Running NUTS with R-hat, ESS, and divergence diagnostics.

Partial pooling expresses a compromise between estimating every school
separately and assuming all schools have one effect. The population mean
`mu` describes their shared location; `tau` describes between-school
variation; each known `sigma_j` describes uncertainty in that school's measured
effect. A noisy school estimate provides less evidence for a distinct effect
and tends to be pulled more strongly toward the population mean.

When `tau` is near zero, all `theta_j` must lie very close to `mu`; when `tau`
is large, they can spread out. This changing width produces the funnel geometry
mentioned above. Noncentering expresses each school as a standardized deviation
before scaling it, which can make weakly informed models easier to fit.

## Eight schools, centered

$$
\mu \sim \mathrm{Normal}(0, 5), \qquad
\tau \sim \mathrm{HalfNormal}(5), \qquad
\theta_j \sim \mathrm{Normal}(\mu, \tau), \qquad
y_j \sim \mathrm{Normal}(\theta_j, \sigma_j).
$$

The eight $\theta_j$ are a per-group random effect over the group object `School`.

=== "QVR"

    ```qvr
    object School : FinSet 8
    program eight_schools_centred : School -> School [effects=[Sample, Score]]
        sample mu  <- Normal(loc=0.0, scale=5.0)
        sample tau <- HalfNormal(scale=5.0)
        sample theta : School <- Normal(loc=mu, scale=tau)
        observe y : School <- Normal(loc=theta, scale=sigma_j)
        return theta

    export eight_schools_centred
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

The `theta : School <- Normal(...)` line is a *plate-draw*: it samples one value per element of `School`. The plate index is the object's cardinality (8). The `observe y : School <- ...` line is a *vectorized observe* over the same index.

The compiler synthesizes a `PlateDraw` morphism whose codomain is the product space `School ⊗ Real`; you can index into it like `theta[j]` inside subsequent `let` arithmetic.

## Centered parameterization and mean-field VI

The centered parameterization draws each `theta_j` directly around `mu` with
scale `tau`. When `tau` approaches zero, the conditional distribution of the
group effects becomes narrow, coupling `theta`, `mu`, and `tau` into a
funnel-shaped posterior ([@neal2003slice],
§8). A diagonal variational family cannot represent that dependence and may
put too much mass near `tau = 0`:

```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

CENTRED_SRC = """
object School : FinSet 8

program eight_schools_centred : School -> School
    sample mu  <- Normal(loc=0.0, scale=5.0)
    sample tau <- HalfNormal(scale=5.0)
    sample theta : School <- Normal(loc=mu, scale=tau)
    observe y : School <- Normal(loc=theta, scale=sigma_j)
    return theta

export eight_schools_centred
"""

program = loads(CENTRED_SRC)
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

A [diagonal guide](../python/05-variational-inference.md#creating-a-guide) can concentrate near `tau = 0` because the funnel couples the group effects to their shared scale. Compare the distribution of `tau` under centered and non-centered fits, using multiple initializations and a well-mixed MCMC fit as a reference.

## Non-centered parameterization

The standard fix is to reparameterize [@papaspiliopoulos2007parametrization]: draw $\eta_j \sim \mathrm{Normal}(0, 1)$ and define $\theta_j = \mu + \tau \cdot \eta_j$ deterministically.

```qvr
object School : FinSet 8
program eight_schools_noncentred : School -> School [effects=[Sample, Score]]
    sample mu  <- Normal(loc=0.0, scale=5.0)
    sample tau <- HalfNormal(scale=5.0)
    sample eta : School <- Normal(loc=0.0, scale=1.0)
    let theta = mu + tau * eta
    observe y : School <- Normal(loc=theta, scale=sigma_j)
    return theta

export eight_schools_noncentred
```

Re-running with the non-centered parameterization can improve the diagonal guide's geometry. Whether it agrees with NUTS depends on optimization and MCMC diagnostics.

## NUTS

To approximate the posterior with sampling rather than a fixed variational family, use the No-U-Turn Sampler [@hoffman2014nuts]:

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

Inspect rank-normalized split-R-hat [@vehtari2021rhat] for disagreement between chains and effective sample sizes for the precision of each posterior summary. The following adjustments address high R-hat, divergences, and low effective sample sizes separately.

If R-hat is above 1.01 anywhere, give NUTS more warmup steps (`num_warmup=2000`). If divergences are nonzero, raise `target_accept` to 0.99 or reparameterize to non-centered. If ESS is small (<100 per chain), the chain is mixing slowly: either run longer or reparameterize.

### Centered or non-centered?

A useful rule of thumb from Betancourt and Girolami [@betancourt2015hierarchical]: the centered parameterization is good when the data is informative about $\theta_j$ relative to the prior, that is when the per-group likelihood scale $\sigma_j$ is small relative to the population scale $\tau$. The non-centered parameterization is good in the opposite regime: weak per-group likelihoods, where the posterior pinches toward the prior funnel. Eight schools sits in the weak-likelihood regime ($\sigma_j$ ranges 9-18 against a prior $\tau$ in single digits), so a non-centered form is often useful. Compare divergences and effective sample sizes under both parameterizations for the data you use. Stan’s [hierarchical reparameterization discussion](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html#hierarchical-models-and-the-non-centered-parameterization) follows the same choice from centered parameters to standard Normal offsets [@stanEfficiency].

### Init strategy and mass matrix

`init_strategy="prior"` draws the chain's initial state from the model prior. The other shipped options are `"uniform"` (uniform on the unconstrained space) and `"value"` (you supply a dict). `"prior"` is the safe default: it starts the chain in regions the prior places mass on.

`NUTSKernel(mass_matrix="diagonal")` adapts a diagonal mass matrix during warmup, which works for most hierarchical models. `"dense"` adapts a full matrix and is worth a try when latents are strongly correlated, at quadratic memory cost. `"identity"` skips adaptation entirely and only makes sense for already-well-scaled models.

## Posterior predictive

[`Predictive`](../../api/inference/predictive.md) accepts either a [`Guide`](../../api/inference/guide.md) or an `MCMCResult`:

```python
from quivers.inference import Predictive

pred = Predictive(model, posterior=result, num_samples=100)
y_hat = pred(x_tensor, {"sigma_j": sigma_j})["y"]  # (100, ..., 8)
print("predictive school 1:", y_hat[..., 0].mean().item(),
      "+/-", y_hat[..., 0].std().item())
```

## What you've seen

- **Plate-draws.** `v : G <- F(...)` declares one draw per index of object `G`, producing a vector-valued latent.
- **Non-centered parameterization.** Draw a standard Normal variable and multiply it by the group scale; the model below writes these as separate steps.
- **Diagnostics on `MCMCResult`.** R-hat, ESS, and divergences are available as fields for inspecting the fit.

## Try this

- Run `AutoMultivariateNormal` on the centered parameterization. It can sometimes recover the funnel where mean-field can't.
- Change `target_accept` from 0.95 to 0.8 and watch divergences appear. The trade-off is step size vs trajectory length.
- Add a per-school covariate $x_j$ and lift the model to a varying-intercepts-and-slopes regression.

## Next

[Chapter 4](04-marginalize.md) introduces the `marginalize` block, which gives
discrete latents and mixtures a typed scope.


## Sources and further reading

- *Estimation in Parallel Randomized Experiments* [@rubin1981parallel].
- *Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC* [@vehtari2021rhat].
- *Hamiltonian Monte Carlo for Hierarchical Models* [@betancourt2015hierarchical].
- *A General Framework for the Parametrization of Hierarchical Models* [@papaspiliopoulos2007parametrization].
- *Slice sampling* [@neal2003slice].
