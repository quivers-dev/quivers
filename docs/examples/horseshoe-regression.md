# Horseshoe Sparse Regression

## Overview

A multiple linear regression with the horseshoe prior [@carvalho2010horseshoe] on its coefficients. Each of 100 responses depends on all twenty columns of a dense design matrix, and only three of the twenty true coefficients are nonzero. The horseshoe is a [global-local scale mixture of Normals](https://en.wikipedia.org/wiki/Sparsity-inducing_prior): a global scale `tau` shared by all coefficients and local scales `lambda_p`, one per coefficient, jointly set each coefficient's prior scale `tau * lambda_p`. Conditional on the global scale, integrating over the local scale gives a density unbounded at zero with Cauchy-like tails. This prior encourages small coefficients while allowing large effects.

## QVR source

```qvr
# Horseshoe Sparse Linear Regression
#
# A multiple linear regression with a horseshoe prior on its twenty
# coefficients. One global scale tau and per-coefficient local
# scales lambda_p define a global-local scale mixture of Normals:
# a small tau pulls every coefficient toward zero, while a large
# lambda_p lets an individual coefficient escape the shrinkage.
#
# Generative structure:
#
#   tau        ~ HalfCauchy(1)                    global scale
#   lambda_p   ~ HalfCauchy(1)                    local scales, p = 0..19
#   z_p        ~ Normal(0, 1)                     standard-Normal raw
#   beta_p     = tau * lambda_p * z_p             so beta_p ~ N(0, (tau * lambda_p)**2)
#   alpha      ~ Normal(0, 5)                     intercept, not shrunk
#   sigma      ~ HalfCauchy(2)                    noise scale
#   y_n        ~ Normal(alpha + sum_p beta_p * x_{n,p}, sigma)
#
# The design matrix arrives as its twenty columns x_0, ..., x_19,
# each a host vector over Resp, so the linear predictor is the
# dense inner product of every row with the full coefficient
# vector. The non-centered product tau * lambda * z has the
# horseshoe marginal for beta_p without a closed-form density.
#
# Reference: [Carvalho, Polson, and Scott 2010](https://doi.org/10.1093/biomet/asq017).

object Coef : FinSet 20
object Resp : FinSet 100
object Val : Real 1

program horseshoe_regression : Resp -> Val
    sample tau <- HalfCauchy(scale=1.0)
    sample lambda_local : Coef <- HalfCauchy(scale=1.0)
    sample z_raw : Coef <- Normal(loc=0.0, scale=1.0)
    sample alpha <- Normal(loc=0.0, scale=5.0)
    sample sigma <- HalfCauchy(scale=2.0)

    let beta = tau * lambda_local * z_raw
    let signal_a = beta[0] * x_0 + beta[1] * x_1 + beta[2] * x_2 + beta[3] * x_3 + beta[4] * x_4
    let signal_b = beta[5] * x_5 + beta[6] * x_6 + beta[7] * x_7 + beta[8] * x_8 + beta[9] * x_9
    let signal_c = beta[10] * x_10 + beta[11] * x_11 + beta[12] * x_12 + beta[13] * x_13 + beta[14] * x_14
    let signal_d = beta[15] * x_15 + beta[16] * x_16 + beta[17] * x_17 + beta[18] * x_18 + beta[19] * x_19
    let mu = alpha + signal_a + signal_b + signal_c + signal_d

    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return tau

export horseshoe_regression
```

## Understanding the model

The coefficient is built as `tau * lambda * z`, a [non-centered parameterization](../tutorials/qvr/03-hierarchical.md#non-centered-parameterization) of a Normal with mean zero and standard deviation `tau * lambda_p`. For fixed positive `tau`, integrating a half-Cauchy local scale gives the horseshoe density, which is unbounded at zero and has tails proportional to `beta**-2`. Integrating over the additional prior on `tau` gives a further mixture. The prior does not place a point mass at zero.

The global scale has a half-Cauchy prior with scale 1. Scaling it by the observation noise `sigma` is another specification [@piironen2016hyperprior], with a different joint prior. The global and local scales can trade off even when their product is constrained, so the fit is assessed through `beta_p = tau * lambda_p * z_p`.

## Walkthrough

`tau ~ HalfCauchy(1)` is the global scale. `lambda_local : Coef <- HalfCauchy(1)`
is the [plate](https://en.wikipedia.org/wiki/Plate_notation) of twenty local scales and `z_raw : Coef <- Normal(0, 1)` the plate
of standard-Normal raw draws. The deterministic product `beta = tau * lambda * z`
gives the twenty coefficients. The design matrix arrives as its twenty columns
`x_0`, ..., `x_19`, each a host vector over the 100 response rows, and the
let-steps `signal_a` through `signal_d` form the dense inner product
`sum_p beta_p * x_{n,p}` for every row `n`, five columns at a time. Each column
is a separate host vector because a free name in a program body is typed by the
plate of the step that reads it, which gives it one axis. A single `(N, P)`
host matrix would need a second. The intercept `alpha` has a Normal prior with standard deviation 5. It is regularized separately from the coefficients and is not subject to the horseshoe scales.

`return tau` exposes the positive global shrinkage scale. The coefficient posterior is recovered from draws of `tau`, `lambda_local`, and `z_raw`.

## Try it

The blocks below simulate a sparse regression, check the compiled log joint
against a hand-written density, and then show shrinkage under [SVI](../guides/inference-svi.md) (stochastic variational inference) and [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler): the
seventeen null coefficients are pulled toward zero while the three signals are
recovered, and the coefficients come out closer to the truth than least
squares puts them. The input `x_in` has one row, representing one [run over the full data set](../guides/inference-foundations.md#input-rows-are-independent-runs).

### Generating synthetic data

The true coefficients are `3.0`, `-2.0`, and `1.5` on columns 0, 3, and 7 and
exactly zero elsewhere. The ground-truth latents are one horseshoe
decomposition of those coefficients: a small global scale, large local scales on
the three signals, and a raw draw of zero on every null coefficient.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/horseshoe_regression.qvr")
model = prog.morphism

N, P = 100, 20
X = torch.randn(N, P)

true_tau = torch.tensor(0.1)
true_lambda_local = torch.ones(P)
true_lambda_local[[0, 3, 7]] = torch.tensor([30.0, 20.0, 15.0])
true_z_raw = torch.zeros(P)
true_z_raw[[0, 3, 7]] = torch.tensor([1.0, -1.0, 1.0])
true_alpha = torch.tensor(0.5)
true_sigma = torch.tensor(1.0)

true_beta = true_tau * true_lambda_local * true_z_raw
mu_true = true_alpha + X @ true_beta
y = torch.distributions.Normal(mu_true, true_sigma).sample()

columns = {f"x_{p}": X[:, p] for p in range(P)}
observations = {**columns, "y": y}
x_in = torch.zeros(1, 1)
signal = true_beta != 0
print(f"true coefficients: {true_beta.numpy()}")
```

### Checking the log joint

```python
D = torch.distributions
truth = {
    "tau": true_tau,
    "lambda_local": true_lambda_local,
    "z_raw": true_z_raw,
    "alpha": true_alpha,
    "sigma": true_sigma,
}
oracle = (
    D.HalfCauchy(1.0).log_prob(true_tau)
    + D.HalfCauchy(1.0).log_prob(true_lambda_local).sum()
    + D.Normal(0.0, 1.0).log_prob(true_z_raw).sum()
    + D.Normal(0.0, 5.0).log_prob(true_alpha)
    + D.HalfCauchy(2.0).log_prob(true_sigma)
    + D.Normal(mu_true, true_sigma).log_prob(y).sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

A [mean-field](../guides/inference-svi.md) [`AutoNormalGuide`](../api/inference/guide.md) is fit by
[`SVI`](../api/inference/svi.md), and every summary is computed from guide draws
of the latents. Least squares on the same data is the baseline. Three checks
follow from what the horseshoe is for:

1. **Signals are kept.** Each nonzero coefficient's posterior mean lies within
   0.3 of its true value, about three least-squares standard errors.
2. **Nulls are shrunk.** For standardized columns, the shrinkage factor
   `kappa_p = 1 / (1 + N * tau**2 * lambda_p**2 / sigma**2)` is the weight the
   posterior mean puts on zero [@piironen2016hyperprior]: near 0 for a
   coefficient left alone, near 1 for one shrunk away. The posterior mean of
   `kappa_p` should stay below 0.05 for every signal and average above 0.4
   over the seventeen nulls.
3. **Estimation improves.** The squared error of the posterior-mean
   coefficients, summed over all twenty, is at most 0.6 times that of least
   squares.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

design = torch.cat([torch.ones(N, 1), X], dim=1)
ols = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)[1:]
ols_sse = float(((ols - true_beta) ** 2).sum())


def summarize(tau, lam, z, sigma, check_error=True):
    tau, sigma = tau.reshape(-1, 1), sigma.reshape(-1, 1)
    beta = (tau * lam * z).mean(0)
    kappa = (1.0 / (1.0 + N * (tau * lam) ** 2 / sigma**2)).mean(0)
    sse = float(((beta - true_beta) ** 2).sum())
    print(f"posterior mean beta: {beta.numpy().round(2)}")
    print(f"kappa: signals {kappa[signal].numpy().round(3)}, null mean {float(kappa[~signal].mean()):.2f}")
    print(f"squared error: {sse:.3f} (least squares {ols_sse:.3f})")
    assert (beta[signal] - true_beta[signal]).abs().max() < 0.3
    assert kappa[signal].max() < 0.05
    assert kappa[~signal].mean() > 0.4
    if check_error:
        assert sse <= 0.6 * ols_sse


torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={*columns, "y"})
optim = torch.optim.Adam(guide.parameters(), lr=2e-2)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
for _ in range(3000):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(1000)]


def stack(name):
    return torch.stack([d[name].reshape(-1) for d in draws])


summarize(stack("tau"), stack("lambda_local"), stack("z_raw"), stack("sigma"))
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the same posterior over all
100 rows, with a high target acceptance because the horseshoe's funnel between
the scales and the raw draws calls for small steps. Its draws are held to the
first two checks. The third compares posterior means, and with 100 draws their
Monte Carlo error is of the same order as the gap between the horseshoe and
least squares, so it is printed but not asserted here.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=5, target_accept=0.9)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)
print(f"divergences: {int(result.divergence_counts.sum())}")

s = result.samples
summarize(
    s["tau"].reshape(-1),
    s["lambda_local"].reshape(-1, P),
    s["z_raw"].reshape(-1, P),
    s["sigma"].reshape(-1),
    check_error=False,
)
```

## Categorical perspective

A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. The sampling construction draws the global and local scales and the standardized coefficients, applies the elementwise product `tau * lambda * z`, maps the coefficients through the design matrix, and draws Normal responses. A deterministic function defines a [Dirac kernel](https://en.wikipedia.org/wiki/Dirac_measure), which puts all probability at its output. The `lambda_local` and `z_raw` plates are independent draws indexed by `Coef`, conditional on the shared parameters.

Observing data weights this sampling distribution by a likelihood. The resulting measure need not have total mass one, so the scored program uses weighted-measure semantics [@staton2017commutative].
