# Beta-Binomial A/B Test

## Overview

A two-arm [A/B test](https://en.wikipedia.org/wiki/A/B_testing) on a binary conversion outcome with per-arm [Beta-Binomial](https://en.wikipedia.org/wiki/Beta-binomial_distribution) likelihood. Each arm carries an independent positive concentration pair $(c^{(1)}_j, c^{(0)}_j)$ under a heavy-tailed [HalfCauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Truncation_and_folding) prior; each measurement batch reports the successes out of a shared trial count $N$ under the concentrations of the arm it was assigned to.

Generative structure:

$$
\begin{aligned}
c^{(1)}_j &\sim \mathrm{HalfCauchy}(2),\\
c^{(0)}_j &\sim \mathrm{HalfCauchy}(2),\\
y_n \mid c^{(1)}, c^{(0)} &\sim \mathrm{BetaBinomial}\!\bigl(N,\, c^{(1)}_{j(n)},\, c^{(0)}_{j(n)}\bigr).
\end{aligned}
$$

The Beta-Binomial integrates the latent per-batch conversion rate $p_n$ analytically: under the prior $p_n \sim \mathrm{Beta}(c^{(1)}_{j(n)}, c^{(0)}_{j(n)})$ the marginal $\int \mathrm{Binomial}(y_n; N, p_n)\,\mathrm{Beta}(p_n; c^{(1)}_{j(n)}, c^{(0)}_{j(n)})\,dp_n$ is closed-form, so the observed counts carry direct evidence about the concentration parameters without introducing an intermediate rate variable.

## QVR source

```qvr
# Beta-Binomial A/B Test
#
# A two-arm A/B test on a binary conversion outcome. Each arm has
# its own Beta(conc1, conc0) prior on the latent conversion rate;
# the observed successes in each measurement batch are
# Beta-Binomial draws over a shared trial count with that arm's
# concentration parameters.
#
# Generative structure:
#
#   conc1_j ~ HalfCauchy(2)              per-arm success concentration
#   conc0_j ~ HalfCauchy(2)              per-arm failure concentration
#   y_n     ~ BetaBinomial(n_trials_n, conc1_{j(n)}, conc0_{j(n)})
#
# The Beta-Binomial marginalises the latent conversion rate
# analytically. Each batch has an implicit Beta-distributed rate,
# so the observed counts carry direct evidence about the arm's
# concentration parameters without introducing that intermediate
# rate as a sample site.
#
# Arm is the plate the concentrations are allocated over and Batch
# is the plate the counts are observed over; the per-row arm
# assignment reaches the model through the standard plate-gather
# idiom (conc1[arm_idx]). The per-batch trial count n_trials is exogenous
# data supplied at fit time through the observations dict, and Val
# is the value space of the returned per-row success
# concentration.
#
# Reference: [Skellam 1948](https://doi.org/10.1111/j.2517-6161.1948.tb00014.x).

object Arm : FinSet 2
object Batch : FinSet 12
object Val : Real 1

program ab_test : Batch -> Val
    sample conc1 : Arm <- HalfCauchy(scale=2.0)
    sample conc0 : Arm <- HalfCauchy(scale=2.0)

    let a = conc1[arm_idx]
    let b = conc0[arm_idx]

    observe y : Batch <- BetaBinomial(total_count=n_trials, concentration1=a, concentration0=b)
    return a

export ab_test
```

## Understanding the model

Think of each batch as a day of trials. One arm can have a higher average
conversion rate while still varying substantially from day to day. The ratio
`conc1 / (conc1 + conc0)` is its mean rate, while the sum of concentrations
controls how tightly batch rates cluster around that mean. A Binomial with
one fixed rate would omit this extra batch variation.

After fitting, compute the mean-rate difference between arms in each [posterior draw](../tutorials/qvr/01-first-model.md#inspect-the-posterior). Its distribution answers how uncertain the comparison remains; the
returned success concentration alone is not a conversion probability. Since
the arm priors are independent, any shared behavior must come from the
observations rather than a population-level pooling term.

## Walkthrough

`object Arm : FinSet 2` is the plate the concentrations are allocated over, `object Batch : FinSet 12` is the plate the counts are observed over, and `object Val : Real 1` is the value space of what the program returns, so the signature reads `Batch -> Val`. The domain names the observation plate; the codomain names the space the returned value lives in, which for a real scalar is `Real 1`.

The per-arm concentration parameters `conc1` and `conc0` have independent HalfCauchy priors with scale $2$. `let a = conc1[arm_idx]` and `let b = conc0[arm_idx]` gather each batch's pair through the exogenous arm-assignment vector, the same plate-gather idiom the [2PL IRT example](irt-2pl.md) uses for its cross-classified design. The `observe` step then scores the batch's successes under a Beta-Binomial likelihood with each batch's trial count, `n_trials`, also supplied through host data: a name that fills a parameter of the `Batch`-plated `observe` directly holds one value per batch. `return a` exposes the fitted per-batch success concentration; conversion probabilities can be summarized from `conc1 / (conc1 + conc0)` after drawing both sites.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Pick ground truth for both concentration sites, draw each batch's latent conversion rate from the Beta its arm's concentrations define, and forward-generate the successes as Binomial draws at that rate. The composition of the two steps is exactly the Beta-Binomial the model scores, so the generated counts and the clamped concentrations describe one self-consistent point.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/beta_binomial_ab_test.qvr")
model = prog.morphism

n_arm, n_batch = 2, 12
n_trials = torch.randint(150, 251, (n_batch,))

true_conc1 = torch.tensor([12.0, 18.0])
true_conc0 = torch.tensor([28.0, 22.0])

arm_idx = torch.arange(n_arm).repeat(n_batch // n_arm)
rate = torch.distributions.Beta(
    true_conc1[arm_idx], true_conc0[arm_idx],
).sample()
y = torch.distributions.Binomial(n_trials.float(), rate).sample()

observations = {
    "y": y,
    "arm_idx": arm_idx,
    "n_trials": n_trials,
}
x_in = torch.zeros(n_batch, 1)
```

The trial count `n_trials` and the arm assignment `arm_idx` are both exogenous: the program never declares them, so the runtime resolves them from the observations dict at trace time and every backend takes them as data inputs.

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "arm_idx", "n_trials"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(300)]

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=6, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Use cases

This model is useful when binary successes are summarized per measurement batch and an explicit per-trial latent rate is unnecessary. The arms have independent concentration priors: there is no shared population-level hyperprior, so the specification does not shrink one arm toward the other.

## Sources and further reading

- *A Probability Distribution Derived from the Binomial Distribution by Regarding the Probability of Success as Variable between the Sets of Trials* [@skellam1948binomial].
