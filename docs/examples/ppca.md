# Probabilistic Principal Component Analysis

## Overview

[Probabilistic PCA](https://en.wikipedia.org/wiki/Probabilistic_principal_component_analysis) (PPCA) [@tipping1999ppca] explains a $D$-dimensional observation by a $K$-dimensional latent code $z_i$, a loading matrix $W$, and one isotropic noise scale $\sigma$:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \sigma^2 I_D).
$$

Integrating the code out gives $y_i \sim \mathcal{N}(0, W W^\top + \sigma^2 I_D)$. Tipping and Bishop show that the maximum-likelihood $W$ spans the leading $K$ [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) of the sample covariance, with the $k$-th direction scaled by $\sqrt{\lambda_k - \sigma^2}$, where $\lambda_k$ is the $k$-th largest eigenvalue, and that $\sigma^2$ is the mean of the $D - K$ discarded eigenvalues. $W$ is identified only up to a $K \times K$ rotation, so a fit is checked through rotation-invariant quantities: the implied covariance and the subspace $W$ spans. [Factor analysis](factor-analysis.md) is the same model with a separate noise scale for every observed coordinate.

## QVR source

```qvr
# Probabilistic Principal Component Analysis
#
# Probabilistic PCA explains each item's D observed coordinates by a
# K-dimensional latent code, a shared loading matrix, and one
# isotropic noise scale:
#
#   z_i     ~ Normal(0, I_K)                 per-item latent code
#   W[d]    ~ Normal(0, I_K)                 loading row of coordinate d
#   sigma   ~ HalfCauchy(2.5)                shared noise scale
#   y[i, d] ~ Normal(z_i . W[d], sigma)      observed coordinate
#
# The data matrix is flattened into response rows: item_idx and
# obs_idx give each row's item and coordinate, so the mean of a
# row is the inner product of two gathered K-vectors. Integrating
# the codes out gives y_i ~ Normal(0, W W^T + sigma^2 I), and the
# maximum-likelihood W spans the leading K principal components of
# the data. W is identified only up to a K x K rotation.
#
# Reference: [Tipping and Bishop 1999](https://doi.org/10.1111/1467-9868.00196).

object LatentDim : FinSet 2
object ObsDim : FinSet 6
object Item : FinSet 400
object Resp : FinSet 2400
object Val : Real 1

program ppca : Resp -> Val
    sample sigma <- HalfCauchy(scale=2.5)
    sample Z : Item <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=Item]
    sample W : ObsDim <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=ObsDim]

    let z_row = Z[item_idx]
    let w_row = W[obs_idx]
    let mu = sum(z_row * w_row)

    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export ppca
```

## Understanding the model

PPCA describes many correlated measurements using a shorter latent code. The loading matrix expands that code back to observation space, and one shared noise scale describes the residual variation in every direction. With fewer latent than observed coordinates, the reconstructed means lie in a $K$-dimensional subspace, and the noise accounts for everything off it.

Rotating the latent coordinates and applying the inverse rotation to the loadings leaves every reconstructed mean, and thus the likelihood, unchanged. Two fits can thus have quite different loading entries and still describe the same distribution. The quantities that do not move under rotation are the implied covariance $W W^\top + \sigma^2 I$ and the column space of $W$, and these are what the fit below is compared on.

## Walkthrough

The same object declaration can identify an axis in several roles. Read the position where it is used:

| Use | Meaning in this model |
|---|---|
| `sample Z : Item <- ... [over=LatentDim]` | A table with one row per item and one column per latent coordinate. |
| `sample W : ObsDim <- ... [over=LatentDim]` | A table with one row per observed coordinate: the transposed loading matrix. |
| `observe y : Resp <- Normal(...)` | One scored response per cell of the flattened data matrix. |
| `program ... : Resp -> Val` | A response-indexed program returning real values; `Val` is `Real 1`. |

`object Item : FinSet 400` and `object ObsDim : FinSet 6` give the data matrix its $400 \times 6$ shape, and `object Resp : FinSet 2400` is that matrix flattened into one response per cell. The host data `item_idx` and `obs_idx` record which item and which coordinate each response belongs to.

The three `sample` steps draw the noise scale, the codes, and the loadings. The loading table is drawn with one row per observed coordinate, $W_{d, \cdot} \in \mathbb{R}^K$, so that both gathers in the `let` steps return `Resp`-by-`LatentDim` tables: `z_row` holds the code of each response's item and `w_row` holds the loading row of its coordinate. `sum(z_row * w_row)` reduces the last axis, giving each response's mean $\sum_k Z_{i,k} W_{d,k}$, which is the $(i, d)$ entry of $Z W^\top$. Every response then shares the one noise scale `sigma`; that sharing is what makes this PPCA rather than factor analysis.

## Try it

The blocks below simulate a data matrix from the model, check the compiled joint against a hand-written density, fit by SVI and compare the fit with the Tipping and Bishop solution, and run a short NUTS chain on the same joint.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

D = torch.distributions

torch.manual_seed(0)
prog = load("docs/examples/source/ppca.qvr")
model = prog.morphism

N, K, Dn = 400, 2, 6
item_idx = torch.arange(N).repeat_interleave(Dn)
obs_idx = torch.arange(Dn).repeat(N)

true_sigma = 0.5
true_W = torch.tensor([
    [1.2, 0.0], [1.0, 0.3], [0.8, -0.5],
    [0.0, 1.1], [0.4, 0.9], [-0.3, 0.7],
])
true_Z = torch.randn(N, K)
mu_true = (true_Z[item_idx] * true_W[obs_idx]).sum(dim=-1)
y = D.Normal(mu_true, true_sigma).sample()

observations = {"y": y, "item_idx": item_idx, "obs_idx": obs_idx}
x_in = torch.zeros(1, 1)
```

The program input `x_in` is a single row. The extent of the `Resp` plate comes from the observed responses and their indices, and the scalar site `sigma` is drawn once per run rather than once per input row.

### Checking the joint density

At the true values, [`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) should equal the half-Cauchy and Normal prior terms plus the Normal likelihood of every cell, written out by hand.

```python
truth = {"sigma": torch.tensor(true_sigma), "Z": true_Z, "W": true_W, **observations}
log_joint = model.log_joint(x_in, truth)[0]

by_hand = (
    D.HalfCauchy(2.5).log_prob(torch.tensor(true_sigma))
    + D.Normal(0.0, 1.0).log_prob(true_Z).sum()
    + D.Normal(0.0, 1.0).log_prob(true_W).sum()
    + D.Normal(mu_true, true_sigma).log_prob(y).sum()
)
assert torch.allclose(log_joint, by_hand, rtol=1e-5)
print(f"log joint at the truth: {float(log_joint):.2f}")
```

### SVI fit

Fit an [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) by [`SVI`](../api/inference/svi.md#svi) and summarize $W$ and $\sigma$ by the mean of 100 guide draws. The checks are rotation-invariant: the noise scale, the implied covariance against the true one, and the [principal angles](https://en.wikipedia.org/wiki/Angles_between_flats) between the column space of the fitted $W$ and the top-$K$ eigenvectors of the sample covariance, which is the subspace the Tipping and Bishop solution spans.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "item_idx", "obs_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
losses = [svi.step(x_in, observations) for _ in range(2000)]

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(100)]
fit_W = torch.stack([d["W"] for d in draws]).mean(0).reshape(Dn, K)
fit_sigma = float(torch.stack([d["sigma"] for d in draws]).mean())

true_cov = true_W @ true_W.T + true_sigma**2 * torch.eye(Dn)
fit_cov = fit_W @ fit_W.T + fit_sigma**2 * torch.eye(Dn)
cov_error = float((fit_cov - true_cov).norm() / true_cov.norm())

sample_cov = torch.cov(y.reshape(N, Dn).T)
eigvals, eigvecs = torch.linalg.eigh(sample_cov)
leading = eigvecs[:, -K:]
basis, _ = torch.linalg.qr(fit_W)
cosines = torch.linalg.svdvals(leading.T @ basis).clamp(max=1.0)
angles = torch.rad2deg(torch.arccos(cosines))

ml_sigma = float(eigvals[:-K].mean().sqrt())
print(f"ELBO loss: {losses[0]:.1f} -> {sum(losses[-100:]) / 100:.1f}")
print(f"sigma: fitted {fit_sigma:.3f}, closed-form ML {ml_sigma:.3f}, true {true_sigma}")
print(f"relative covariance error: {cov_error:.3f}")
print(f"principal angles (degrees): {angles.numpy().round(2)}")

assert abs(fit_sigma - true_sigma) < 0.1
assert cov_error < 0.2
assert angles.max() < 5.0
```

The covariance tolerance is set against sampling error: with 400 items, the sample covariance itself is about 12% (in relative Frobenius norm) away from the true one here, and a fit can do no better than the data allow. The subspace check needs no truth at all; it asks whether the fitted loadings span the principal subspace of the data, which is the content of Tipping and Bishop's result.

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) runs on the same joint, with the codes, the loadings, and the noise scale all latent. The chain below is short; the rotation invariance makes $W$ itself mix poorly, so a real analysis would monitor rotation-invariant summaries such as the implied covariance across several chains.

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

Read as arrows in the real algebra, the code table is a map $Z : \mathsf{Item} \to \mathsf{LatentDim}$ and the transposed loading table a map $W^\top : \mathsf{LatentDim} \to \mathsf{ObsDim}$, and the mean of the response at $(i, d)$ is the $(i, d)$ entry of their composite $Z\,W^\top : \mathsf{Item} \to \mathsf{ObsDim}$. The program never materializes that composite: the two gathers pull both arrows back along `item_idx` and `obs_idx` to the response plate, where the contraction over `LatentDim` is done row by row. Following the mean by the isotropic Normal kernel and integrating $Z$ against its standard Normal prior gives, for each item, the Gaussian measure $\mathcal{N}(0, W W^\top + \sigma^2 I)$ on $\mathbb{R}^{\mathsf{ObsDim}}$. Factor analysis keeps the same composite and changes only the noise kernel.

## See also

- [Bayesian Factor Analysis](factor-analysis.md), the same low-rank mean with a separate uniqueness for every observed coordinate.
- [Probabilistic Matrix Factorization](pmf.md), the same per-row inner-product idiom applied to a sparse rating matrix.

## Sources and further reading

- *Probabilistic Principal Component Analysis* [@tipping1999ppca].
