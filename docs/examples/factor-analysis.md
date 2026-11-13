# Bayesian Factor Analysis

## Overview

[Factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) ([@spearman1904intelligence]; [@bartholomew2011latent]) explains a $D$-dimensional observation as a linear transformation of a $K$-dimensional latent factor plus noise that is independent across coordinates and has its own scale in each coordinate:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}\bigl(W z_i, \mathrm{diag}(\psi_1^2, \ldots, \psi_D^2)\bigr).
$$

The scale $\psi_d$ is the uniqueness of coordinate $d$: the part of its variation that the common factors do not explain. Integrating the factors out gives $y_i \sim \mathcal{N}(0, W W^\top + \mathrm{diag}(\psi^2))$. The loadings are identified only up to a $K \times K$ rotation, but the implied covariance and the uniquenesses themselves are identified. [Probabilistic PCA](ppca.md) is the special case $\psi_1 = \cdots = \psi_D = \sigma$; the free diagonal is what lets factor analysis treat a precise measurement and a noisy one differently.

## QVR source

```qvr
# Bayesian Factor Analysis
#
# Factor analysis explains each item's D observed coordinates by a
# K-dimensional latent code and a shared loading matrix, and gives
# every observed coordinate its own residual scale (its uniqueness):
#
#   z_i     ~ Normal(0, I_K)                 per-item factor scores
#   W[d]    ~ Normal(0, I_K)                 loading row of coordinate d
#   psi[d]  ~ HalfCauchy(2.5)                uniqueness of coordinate d
#   y[i, d] ~ Normal(z_i . W[d], psi[d])     observed coordinate
#
# The data matrix is flattened into response rows: item_idx and
# obs_idx give each row's item and coordinate, so the mean of a row
# is the inner product of two gathered K-vectors and its scale is
# the gathered uniqueness. Integrating the factor scores out gives
# y_i ~ Normal(0, W W^T + diag(psi^2)). Probabilistic PCA is the
# special case in which every psi[d] equals one shared sigma.
#
# References: [Spearman 1904](https://doi.org/10.2307/1412107),
# [Bartholomew, Knott and Moustaki 2011](https://doi.org/10.1002/9781119970583).

object LatentDim : FinSet 2
object ObsDim : FinSet 6
object Item : FinSet 400
object Resp : FinSet 2400
object Val : Real 1

program factor_analysis : Resp -> Val
    sample psi : ObsDim <- HalfCauchy(scale=2.5)
    sample Z : Item <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=Item]
    sample W : ObsDim <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=ObsDim]

    let z_row = Z[item_idx]
    let w_row = W[obs_idx]
    let mu = sum(z_row * w_row)
    let psi_row = psi[obs_idx]

    observe y : Resp <- Normal(loc=mu, scale=psi_row)
    return y

export factor_analysis
```

## Understanding the model

Suppose several observed measurements rise and fall together. A small latent vector can explain those shared patterns, and a loading matrix maps its coordinates back to measurement space. With two factors, the predicted value of coordinate $d$ for item $i$ is `Z[i, 0] * W[d, 0] + Z[i, 1] * W[d, 1]`. What is left over is the coordinate's unique variation, and factor analysis lets each coordinate carry its own amount: a carefully measured coordinate may have $\psi_d$ near zero, so the factors must account for almost all of its variance, while a noisy one may have a large $\psi_d$ and a weak connection to the factors.

A factor coordinate has no unique orientation: rotating the scores and undoing that rotation in the loadings leaves every predicted measurement unchanged. Fitted loading entries thus need not match their synthetic counterparts, and the fit below is compared on the implied covariance and on $\psi$, neither of which depends on the rotation.

## Walkthrough

The same object declaration can identify an axis in several roles. Read the position where it is used:

| Use | Meaning in this model |
|---|---|
| `sample Z : Item <- ... [over=LatentDim]` | A table with one row per item and one column per factor. |
| `sample W : ObsDim <- ... [over=LatentDim]` | A table with one row per observed coordinate: the loading matrix. |
| `sample psi : ObsDim <- ...` | One positive uniqueness per observed coordinate. |
| `observe y : Resp <- Normal(...)` | One scored response per cell of the flattened data matrix. |
| `program ... : Resp -> Val` | A response-indexed program returning real values; `Val` is `Real 1`. |

`object Item : FinSet 400` and `object ObsDim : FinSet 6` give the data matrix its $400 \times 6$ shape, and `object Resp : FinSet 2400` is that matrix flattened into one response per cell. The host data `item_idx` and `obs_idx` record which item and which coordinate each response belongs to.

`sample psi : ObsDim <- HalfCauchy(scale=2.5)` draws one uniqueness per coordinate. The two gathers `Z[item_idx]` and `W[obs_idx]` return `Resp`-by-`LatentDim` tables, and `sum(z_row * w_row)` reduces their last axis, giving each response's mean $\sum_k Z_{i,k} W_{d,k}$. The third gather, `psi[obs_idx]`, gives each response the uniqueness of its own coordinate, and that per-row scale is the only line in which this program differs from the [PPCA program](ppca.md#walkthrough).

## Try it

The blocks below simulate a data matrix whose coordinates have quite different uniquenesses, check the compiled joint against a hand-written density, fit the model by SVI and check that it recovers $\psi$ and the implied covariance, fit the PPCA program to the same data for contrast, and run a short NUTS chain.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

D = torch.distributions

torch.manual_seed(0)
prog = load("docs/examples/source/factor_analysis.qvr")
model = prog.morphism

N, K, Dn = 400, 2, 6
item_idx = torch.arange(N).repeat_interleave(Dn)
obs_idx = torch.arange(Dn).repeat(N)

true_psi = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
true_W = torch.tensor([
    [1.2, 0.0], [1.0, 0.3], [0.8, -0.5],
    [0.0, 1.1], [0.4, 0.9], [-0.3, 0.7],
])
true_Z = torch.randn(N, K)
mu_true = (true_Z[item_idx] * true_W[obs_idx]).sum(dim=-1)
y = D.Normal(mu_true, true_psi[obs_idx]).sample()

observations = {"y": y, "item_idx": item_idx, "obs_idx": obs_idx}
x_in = torch.zeros(1, 1)
```

The program input `x_in` is a single row; the extent of the `Resp` plate comes from the observed responses and their indices. The uniquenesses range over a factor of six, so a single shared noise scale cannot describe these data well.

### Checking the joint density

At the true values, [`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) should equal the half-Cauchy and Normal prior terms plus the Normal likelihood of every cell under its own coordinate's uniqueness, written out by hand.

```python
truth = {"psi": true_psi, "Z": true_Z, "W": true_W, **observations}
log_joint = model.log_joint(x_in, truth)[0]

by_hand = (
    D.HalfCauchy(2.5).log_prob(true_psi).sum()
    + D.Normal(0.0, 1.0).log_prob(true_Z).sum()
    + D.Normal(0.0, 1.0).log_prob(true_W).sum()
    + D.Normal(mu_true, true_psi[obs_idx]).log_prob(y).sum()
)
assert torch.allclose(log_joint, by_hand, rtol=1e-5)
print(f"log joint at the truth: {float(log_joint):.2f}")
```

### SVI fit

Fit an [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) by [`SVI`](../api/inference/svi.md#svi) and summarize $W$ and $\psi$ by the mean of 100 guide draws. The helper is reused below for the PPCA contrast.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI


def fit_svi(program, sites, steps=2000):
    torch.manual_seed(1)
    guide = AutoNormalGuide(program, observed_names={"y", "item_idx", "obs_idx"})
    optim = torch.optim.Adam(
        list(program.parameters()) + list(guide.parameters()), lr=5e-2,
    )
    svi = SVI(program, guide, optim, ELBO(num_particles=1))
    for _ in range(steps):
        svi.step(x_in, observations)
    with torch.no_grad():
        draws = [guide.rsample(x_in) for _ in range(100)]
    return {name: torch.stack([d[name] for d in draws]).mean(0) for name in sites}


true_cov = true_W @ true_W.T + torch.diag(true_psi**2)


def covariance_error(cov):
    return float((cov - true_cov).norm() / true_cov.norm())


fa_fit = fit_svi(model, ("W", "psi"))
fa_W = fa_fit["W"].reshape(Dn, K)
fa_psi = fa_fit["psi"].reshape(Dn)
fa_error = covariance_error(fa_W @ fa_W.T + torch.diag(fa_psi**2))

print(f"psi fitted: {fa_psi.numpy().round(2)}")
print(f"psi true:   {true_psi.numpy()}")
print(f"relative covariance error: {fa_error:.3f}")

assert (fa_psi - true_psi).abs().max() < 0.15
assert fa_error < 0.2
```

With 400 items, the sample covariance is itself about 11% (in relative Frobenius norm) away from the true one on these data, so the covariance tolerance leaves room for sampling error and little else. The smallest uniqueness is the hardest to pin down, since a coordinate with $\psi_d = 0.2$ and loadings of size one has nearly all of its variance explained by the factors.

### Contrast with PPCA

The [PPCA program](ppca.md) declares the same objects and host data, so it can be fitted to the same responses. Its single noise scale has to compromise between the precise and the noisy coordinates.

```python
ppca = load("docs/examples/source/ppca.qvr").morphism
ppca_fit = fit_svi(ppca, ("W", "sigma"))
ppca_W = ppca_fit["W"].reshape(Dn, K)
ppca_sigma = float(ppca_fit["sigma"].reshape(-1)[0])
ppca_error = covariance_error(ppca_W @ ppca_W.T + ppca_sigma**2 * torch.eye(Dn))

print(f"PPCA sigma: {ppca_sigma:.2f}")
print(f"relative covariance error: FA {fa_error:.3f}, PPCA {ppca_error:.3f}")
assert fa_error < ppca_error
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) runs on the same joint, with the factor scores, the loadings, and the uniquenesses all latent. The chain below is short; the rotation invariance makes $W$ itself mix poorly, so a real analysis would monitor rotation-invariant summaries such as $\psi$ and the implied covariance across several chains.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

assert torch.isfinite(result.log_densities).all()
print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

As in [PPCA](ppca.md#categorical-perspective), the response mean at $(i, d)$ is the $(i, d)$ entry of the real-algebra composite $Z\,W^\top : \mathsf{Item} \to \mathsf{ObsDim}$, computed row by row after both arrows are pulled back to the response plate along `item_idx` and `obs_idx`. The two models differ only in the Gaussian kernel that follows the mean: PPCA uses one scale for every coordinate, while factor analysis uses the scale $\psi$ pulled back along `obs_idx`, a diagonal kernel on $\mathbb{R}^{\mathsf{ObsDim}}$. Integrating $Z$ out then gives $\mathcal{N}(0, W W^\top + \mathrm{diag}(\psi^2))$ for each item.

## See also

- [Probabilistic PCA](ppca.md), the special case with one shared noise scale.
- [DSL Guide: Hierarchical Bayesian Models](../guides/programs-hierarchical.md) for further plate-indexed models.

## Sources and further reading

- *"General Intelligence," Objectively Determined and Measured* [@spearman1904intelligence].
- *Latent Variable Models and Factor Analysis: A Unified Approach* [@bartholomew2011latent].
