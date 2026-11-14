# Beta-Binomial A/B Test

## Overview

A two-arm [A/B test](https://en.wikipedia.org/wiki/A/B_testing) on a binary conversion outcome with per-arm [Beta-Binomial](https://en.wikipedia.org/wiki/Beta-binomial_distribution) likelihood. Each arm carries an independent positive concentration pair $(c^{(1)}_j, c^{(0)}_j)$ under a heavy-tailed [HalfCauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Truncation_and_folding) prior. Each measurement batch reports the successes out of a shared trial count $N$ under the concentrations of the arm it was assigned to. [@skellam1948binomial]

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
#   y_n     ~ BetaBinomial(n_trials, conc1_{j(n)}, conc0_{j(n)})
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
# idiom (conc1[arm_idx]). The trial count n_trials is exogenous
# data provided at fit time through the observations dict, and Val
# is the value space of the returned per-row success
# concentration.
#
# Reference: [Skellam 1948](https://doi.org/10.1111/j.2517-6161.1948.tb00014.x).

object Arm : FinSet 2
object Batch : FinSet 40
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

After fitting, compute the mean-rate difference between arms in each [posterior draw](../tutorials/qvr/01-first-model.md#inspect-the-posterior). Its distribution answers how uncertain the comparison remains. The
returned success concentration alone is not a conversion probability. Since
the arm priors are independent, any shared behavior must come from the
observations rather than a population-level pooling term.

## Walkthrough

`object Arm : FinSet 2` is the [plate](https://en.wikipedia.org/wiki/Plate_notation) the concentrations are allocated over, `object Batch : FinSet 40` is the plate the counts are observed over (20 batches per arm), and `object Val : Real 1` is the value space of what the program returns, so the signature reads `Batch -> Val`. The domain names the observation plate. The codomain names the space the returned value lives in, which for a real scalar is `Real 1`.

The per-arm concentration parameters `conc1` and `conc0` have independent HalfCauchy priors with scale $2$. `let a = conc1[arm_idx]` and `let b = conc0[arm_idx]` gather each batch's pair through the exogenous arm-assignment vector, the same indexed lookup the [2PL IRT example](irt-2pl.md) uses for its cross-classified design. The `observe` step then scores the batch's successes under a Beta-Binomial likelihood with shared trial count `n_trials`, also provided through input data. `return a` exposes the fitted per-batch success concentration. Conversion probabilities can be summarized from `conc1 / (conc1 + conc0)` after drawing both sites.

## Try it

The blocks below simulate batches from the two arms, check the compiled log
joint against a hand-written density, and then compare the arms: the quantity
an A/B test reports is the posterior of the difference in mean conversion rate,
`conc1 / (conc1 + conc0)` in arm B minus the same in arm A. The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

Pick ground truth for both concentration sites, draw each batch's latent conversion rate from the Beta its arm's concentrations define, and forward-generate the successes as Binomial draws at that rate. The composition of the two steps is exactly the Beta-Binomial the model scores. The true mean rates are `12 / 40 = 0.30` in arm A and `18 / 40 = 0.45` in arm B.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/beta_binomial_ab_test.qvr")
model = prog.morphism

n_arm, n_batch, n_trials = 2, 40, 200

true_conc1 = torch.tensor([12.0, 18.0])
true_conc0 = torch.tensor([28.0, 22.0])
true_mean = true_conc1 / (true_conc1 + true_conc0)
true_lift = float(true_mean[1] - true_mean[0])

arm_idx = torch.arange(n_arm).repeat(n_batch // n_arm)
rate = torch.distributions.Beta(
    true_conc1[arm_idx], true_conc0[arm_idx],
).sample()
y = torch.distributions.Binomial(n_trials, rate).sample()

observations = {"y": y, "arm_idx": arm_idx, "n_trials": torch.tensor(n_trials)}
x_in = torch.zeros(1, 1)
```

The trial count `n_trials` and the arm assignment `arm_idx` are both exogenous: the program never declares them, so the runtime resolves them from the observations mapping at trace time and every backend takes them as data inputs.

### Checking the log joint

The Beta-Binomial log mass is written out with `lgamma`, in double precision
because its terms reach several hundred nats before they cancel.

```python
D = torch.distributions
truth = {"conc1": true_conc1, "conc0": true_conc0}
a, b = true_conc1[arm_idx].double(), true_conc0[arm_idx].double()
k, n = y.double(), torch.tensor(float(n_trials), dtype=torch.float64)
lg = torch.lgamma
log_beta_binomial = (
    lg(n + 1) - lg(k + 1) - lg(n - k + 1)
    + lg(k + a) + lg(n - k + b) - lg(n + a + b)
    - lg(a) - lg(b) + lg(a + b)
)
oracle = (
    D.HalfCauchy(2.0).log_prob(true_conc1).sum().double()
    + D.HalfCauchy(2.0).log_prob(true_conc0).sum().double()
    + log_beta_binomial.sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum().double()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-6)
```

### SVI fit and arm comparison

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md) (stochastic variational inference), and the lift is computed in every guide draw.
Batch rates vary with a standard deviation near 0.07 around each arm's mean, so
with 20 batches per arm the lift has a standard error near 0.025. The check asks
that the posterior mean lift lie within 0.07 of the true 0.15 (about three
standard errors) and that the posterior put at least 99% of its mass on arm B
converting better.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "arm_idx", "n_trials"})
optim = torch.optim.Adam(guide.parameters(), lr=5e-2)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
for _ in range(1500):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(2000)]
c1 = torch.stack([d["conc1"] for d in draws])
c0 = torch.stack([d["conc0"] for d in draws])
lift = (c1 / (c1 + c0)) @ torch.tensor([-1.0, 1.0])
print(f"posterior mean rates: {(c1 / (c1 + c0)).mean(0).numpy().round(3)}")
print(f"posterior lift: {float(lift.mean()):.3f} (truth {true_lift:.3f})")
print(f"P(arm B > arm A) = {float((lift > 0).float().mean()):.3f}")

assert abs(float(lift.mean()) - true_lift) < 0.07
assert float((lift > 0).float().mean()) > 0.99
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
40 batches, and the lift computed from its draws is held to the same checks.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=5, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=150, num_chains=1)
result = mc.run(model, x_in, observations)

c1 = result.samples["conc1"].reshape(-1, n_arm)
c0 = result.samples["conc0"].reshape(-1, n_arm)
lift_nuts = (c1 / (c1 + c0)) @ torch.tensor([-1.0, 1.0])
print(f"posterior lift: {float(lift_nuts.mean()):.3f} (truth {true_lift:.3f})")
print(f"P(arm B > arm A) = {float((lift_nuts > 0).float().mean()):.3f}")
print(f"divergences: {int(result.divergence_counts.sum())}")

assert abs(float(lift_nuts.mean()) - true_lift) < 0.07
assert float((lift_nuts > 0).float().mean()) > 0.99
```

## Use cases

This model is useful when binary successes are summarized per measurement batch and an explicit per-trial latent rate is unnecessary. The arms have independent concentration priors: there is no shared population-level hyperprior, so the specification does not shrink one arm toward the other.
