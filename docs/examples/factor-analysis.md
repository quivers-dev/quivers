# Isotropic Bayesian Factor Model

## Overview

Classical [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) ([@spearman1904intelligence]; [@bartholomew2011latent]) decomposes a $D$-dimensional observation as a linear-Gaussian transformation of a $K$-dimensional latent factor plus diagonal idiosyncratic noise:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \mathrm{diag}(\psi)).
$$

The runnable QVR program below uses one scalar `sigma`, not a free diagonal $\psi$. Its observation-noise structure is thus isotropic and matches [probabilistic PCA](ppca.md), despite the example's historical name. The top-level arrows $Z : \mathsf{Item} \to \mathsf{LatentDim}$ and $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ demonstrate the corresponding low-rank composition.

## QVR source

```qvr
# Isotropic Bayesian Factor Model
#
# This low-rank Gaussian factor model decomposes each observation
# through a K-dimensional latent code and a shared loading matrix.
# Its observation noise uses one scalar sigma, so the runnable
# program has the isotropic noise structure of probabilistic PCA,
# not the free diagonal uniquenesses of classical factor analysis.
#
# Structural form:
#
#   Z     : Item      -> LatentDim       per-item latent code
#   W     : LatentDim -> ObsDim          learned loadings
#   model = Z >> W                       linear-Gaussian mean
#
# The structural arrows are learned tensors without family
# declarations. The exported program supplies independent
# Normal priors for the code and loading entries and a
# HalfCauchy prior for the shared observation scale.

composition real [level=algebra]

object LatentDim : FinSet 2
object ObsDim : FinSet 5
object Item : FinSet 32
object Resp : FinSet 160
object Val : Real 1

morphism Z : Item -> LatentDim [role=latent]

morphism W : LatentDim -> ObsDim [role=latent]

define factor_analysis = Z >> W

# Probabilistic model: every entry of the loading matrix and
# per-item latent code carries an independent Normal(0, 1) prior
# (the matrix-normal special case with V = U = I), the noise
# scale carries a HalfCauchy(2.5) prior, and the observed
# response is scored under Normal(mu, sigma) where mu is the
# per-row inner product Z[i] . W[d]. The loading matrix is
# sampled transposed (ObsDim -> LatentDim) so the per-Resp
# inner product is two compatible Resp-by-LatentDim gathers.
program factor_analysis_program : Resp -> Val
    sample sigma <- HalfCauchy(scale=2.5)
    sample Z_mat : Item <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=Item]
    sample W_mat : ObsDim <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=ObsDim]

    let z_row = Z_mat[item_idx]
    let w_row = W_mat[obs_idx]
    let mu = sum(z_row * w_row)

    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export factor_analysis_program
```

## Understanding the model

Suppose several observed measurements rise and fall together. A small latent
vector can explain those shared patterns, and a loading matrix maps its
coordinates back to measurement space. With two factors, one predicted cell
is `z[i, 0] * W[0, d] + z[i, 1] * W[1, d]`. Observation noise accounts for
variation left over after that reconstruction.

A factor coordinate has no unique orientation: rotating the code and undoing
that rotation in the loadings leaves every predicted measurement unchanged.
Thus fitted factor entries need not match their synthetic counterparts.
Compare reconstructed means and the implied observation covariance. This
source uses one shared residual scale, so its noise is the [isotropic PPCA](ppca.md#understanding-the-model)
special case of factor analysis.

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

The two latent declarations introduce the per-item factor and the loading matrix as arrows. Under `composition real [level=algebra]` the composition `Z >> W` is a real-valued matmul: the `(i, d)` entry of the `Item x ObsDim` model tensor is $\sum_k Z_{i,k} W_{k,d}$, the factor analysis model mean.

The following block is an optional extension, not part of the source shown above:

<!-- compile: false -->
```qvr
morphism W : LatentDim -> ObsDim [role=latent] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0) over (dom, cod)
```

It would place a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix. The actual `factor_analysis_program` instead samples `W_mat` through an indexed Normal plate.

A free-diagonal variant would replace scalar `sigma` with a positive `ObsDim`-indexed noise vector and gather the appropriate element for each response row.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/factor_analysis.qvr")
model = prog.morphism

N, K, Dn = 32, 2, 5
ND = N * Dn
item_idx = torch.arange(N).repeat_interleave(Dn)
obs_idx = torch.arange(Dn).repeat(N)

true_sigma = 0.3
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

Factor analysis is identifiable only up to a $K \times K$ orthogonal rotation of $W$, so the rotation-invariant covariance $W^\top W$ is the natural recovery target rather than $W$ itself.

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

The top-level mean is the real-algebra composition $Z \mathbin{>>} W$, whose tensor is the low-rank matrix product. In the exported program, `Z_mat` and `W_mat` are explicit Normal plate draws and the scalar `sigma` supplies isotropic observation noise.

## See also

- [Probabilistic PCA](ppca.md), the isotropic-noise special case.
- [DSL Guide: Hierarchical Bayesian Models](../guides/programs-hierarchical.md#hierarchical-models-with-parametric-templates) for the morphism-valued prior syntax.


## Sources and further reading

- *"General Intelligence," Objectively Determined and Measured* [@spearman1904intelligence].
- *Latent Variable Models and Factor Analysis: A Unified Approach* [@bartholomew2011latent].
