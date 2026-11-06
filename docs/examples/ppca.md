# Probabilistic Principal Component Analysis

## Overview

[Probabilistic PCA](https://en.wikipedia.org/wiki/Probabilistic_principal_component_analysis) [@tipping1999ppca] factors a data matrix through a low-rank loading matrix $W$ acting on a per-item latent code $z$:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \sigma^2 I_D).
$$

The model is identifiable up to a $K \times K$ orthogonal rotation of $W$; the maximum-likelihood $W$ recovers the leading-$K$ [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) scaled by $\sqrt{\lambda_k - \sigma^2}$, where $\lambda_k$ are the data covariance eigenvalues. In general, PPCA differs from classical factor analysis in the observation noise: PPCA uses a single isotropic scalar $\sigma$, while classical factor analysis uses a free diagonal $\psi$. The gallery's [isotropic factor-model page](factor-analysis.md) intentionally has the same noise structure as this program and emphasizes the shared low-rank composition instead.

In quivers, the loading matrix is a [`LatentMorphism`](../api/core/morphisms.md) $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ and the per-item code is $Z : \mathsf{Item} \to \mathsf{LatentDim}$. The model mean is the composition $Z \mathbin{>>} W$. The exported program below supplies explicit Normal priors through `Z_mat` and `W_mat`; the top-level morphisms themselves have no family declaration.

## QVR source

```qvr
# Probabilistic Principal Component Analysis
#
# Probabilistic PCA factors a data matrix through a low-rank
# loading matrix and a per-item latent code, so the model mean
# is a morphism composition under the real algebra.
#
# Structural form:
#
#   Z     : Item      -> LatentDim       per-item latent code
#   W     : LatentDim -> ObsDim          learned loadings
#   ppca  = Z >> W                       low-rank linear mean
#
# The structural W arrow has no family declaration. The exported
# program below supplies independent Normal priors for its loading
# entries, the identity-covariance special case of a matrix-Normal
# prior. The model is identifiable up to a K x K orthogonal
# rotation of W; the maximum-likelihood W recovers the leading-K
# principal components scaled by sqrt(lambda_k - sigma^2).
#
# Reference: [Tipping and Bishop 1999](https://doi.org/10.1111/1467-9868.00196).

composition real [level=algebra]

object LatentDim : FinSet 2
object ObsDim : FinSet 5
object Item : FinSet 32
object Resp : FinSet 160
object Val : Real 1

morphism Z : Item -> LatentDim [role=latent]

morphism W : LatentDim -> ObsDim [role=latent]

define ppca = Z >> W

# Probabilistic model: every entry of the loading matrix and
# per-item latent code carries an independent Normal(0, 1) prior
# (the matrix-normal special case with V = U = I), the isotropic
# noise scale sigma carries a HalfCauchy(2.5) prior, and the
# observed response is scored under Normal(mu, sigma) where mu is
# the per-row inner product Z[i] . W[d]. The loading matrix is
# sampled transposed (ObsDim -> LatentDim) so the per-Resp inner
# product is two compatible Resp-by-LatentDim gathers.
program ppca_program : Resp -> Val
    sample sigma <- HalfCauchy(scale=2.5)
    sample Z_mat : Item <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=Item]
    sample W_mat : ObsDim <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=ObsDim]

    let z_row = Z_mat[item_idx]
    let w_row = W_mat[obs_idx]
    let mu = sum(z_row * w_row)

    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export ppca_program
```

## Understanding the model

PPCA describes many correlated measurements using a shorter latent code.
The [loading matrix](factor-analysis.md#understanding-the-model) expands that code back to observation space, and one
shared noise scale describes residual variation in every direction. With
fewer latent than observed coordinates, the mean reconstruction lies in a
lower-dimensional subspace.

Rotating the latent coordinates and applying the inverse rotation to the
loadings preserves the reconstructed covariance. Compare fits through their
reconstructed means and `W.T @ W + sigma**2 * I`, using the row-oriented
loading convention here. Equivalent fits can have different loading entries.

## Walkthrough

The same object declaration can identify an axis in several roles. Read the
position where it is used:

| Use | Meaning in this model |
|---|---|
| `morphism Z : Item -> LatentDim` | A table with one row per item and one column per latent coordinate. |
| `[over=LatentDim]` | The coordinates making up one vector-valued draw. |
| `observe y : Resp <- Normal(...)` | One scored response per element of the flattened response index. |
| `program ... : Resp -> Val` | A response-indexed program returning real values; `Val` is `Real 1`. |

A row index tells you which observation is being scored; the Normal family
tells you that its value is real. These are separate roles even if their
sizes happen to coincide. Keep the return-space declaration consistent with
the returned values rather than assuming an index name is also a value type.

The two latent declarations introduce the per-item code and the loading matrix as first-class arrows. The composition `Z >> W` is real-algebra matmul: under `composition real [level=algebra]` the `(i, d)` entry of the resulting `Item x ObsDim` tensor is exactly $\sum_k Z_{i,k} W_{k,d}$, the PPCA model mean.

The following matrix-normal declaration is an optional alternative, not part of the source above:

<!-- compile: false -->
```qvr
morphism W : LatentDim -> ObsDim [role=latent] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0) over (dom, cod)
```

It would place a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix. The runnable `ppca_program` uses indexed Normal draws instead.

The PPCA / factor analysis distinction lies in the downstream observation kernel applied to the matrix-product mean: a single shared scalar `sigma` for PPCA or a free diagonal `psi_d` for factor analysis. Both models use the same morphism composition, `Z >> W`.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ppca.qvr")
model = prog.morphism

N, K, Dn = 32, 2, 5
ND = N * Dn
item_idx = torch.arange(N).repeat_interleave(Dn)
obs_idx = torch.arange(Dn).repeat(N)

true_sigma = 0.2
true_Z_mat = torch.randn(N, K)
true_W_mat = torch.randn(Dn, K)
mu_true = (true_Z_mat[item_idx] * true_W_mat[obs_idx]).sum(dim=-1)
y = torch.distributions.Normal(mu_true, true_sigma).sample()

observations = {
    "y": y,
    "item_idx": item_idx,
    "obs_idx": obs_idx,
}
x_in = torch.zeros(ND, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "item_idx", "obs_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(200)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

The recovered factorization `Z @ W` matches the data up to the $K \times K$ rotation invariance of PPCA.

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical perspective

PPCA is a pair of arrows in a real-algebra category: the per-item code $Z : \mathsf{Item} \to \mathsf{LatentDim}$ and the loading $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$. Their composition $Z \mathbin{>>} W$ is a [`ComposedMorphism`](../api/core/morphisms.md) $\mathsf{Item} \to \mathsf{ObsDim}$ whose tensor is the model mean. Marginalizing the latent factor under an isotropic noise kernel recovers the closed-form covariance $W^\top W + \sigma^2 I$ on the observation side.

The optional morphism-valued [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) declaration shown above would place a measure on the hom-object $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$. The loaded program instead uses indexed independent-Normal draws for the loading entries.

## See also

- [Isotropic Bayesian Factor Model](factor-analysis.md), the same low-rank mean and shared-noise structure presented as a factor model.
- [DSL Guide](../guides/dsl-overview.md) for the morphism-valued prior syntax.


## Sources and further reading

- *Probabilistic Principal Component Analysis* [@tipping1999ppca].
