# Probabilistic Matrix Factorization

## Overview

[Probabilistic matrix factorization](https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html) (PMF) [@salakhutdinov2007pmf] completes a sparse user-item rating matrix. Each user $u$ and each item $m$ has a $K$-dimensional latent vector with a zero-mean [Normal](https://en.wikipedia.org/wiki/Normal_distribution) prior, and each observed rating is Normal noise around the inner product of its user's and its item's vectors:

$$
U_u \sim \mathcal{N}(0, I_K), \quad V_m \sim \mathcal{N}(0, I_K), \quad
r_{u, m} \mid U, V \sim \mathcal{N}(U_u^\top V_m, \sigma^2) \text{ for each observed } (u, m).
$$

Only observed cells enter the likelihood. The unobserved cells are predicted through the same inner product, so the quality of a fit is measured by how well it predicts ratings it never saw. Salakhutdinov and Mnih fix the prior and noise precisions or adapt them. This page places a weakly informative half-Cauchy prior on $\sigma$ and unit-scale priors on the factors.

## QVR source

```qvr
# Probabilistic Matrix Factorization
#
# A user-item latent-factor model for completing a sparse rating
# matrix. Every user and every item carries a K-dimensional latent
# vector with a Normal prior, and each observed rating is Normal
# around the inner product of its user's and its item's vectors:
#
#   U[u]    ~ Normal(0, I_K)                    user factors
#   V[m]    ~ Normal(0, I_K)                    item factors
#   sigma   ~ HalfCauchy(2.5)                   rating noise scale
#   r_n     ~ Normal(U[u_n] . V[m_n], sigma)    observed rating n
#
# Only the observed (user, item) pairs are scored: the host data
# user_idx and movie_idx give the user and item of each observed
# rating, so the rows of the Rating plate are a sparse subset of
# the User x Movie grid. The posterior over U and V predicts every
# unobserved cell through the same inner product.
#
# Reference: [Salakhutdinov and Mnih 2007](https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html).

object LatentDim : FinSet 2
object User : FinSet 50
object Movie : FinSet 40
object Rating : FinSet 800
object Val : Real 1

program pmf : Rating -> Val
    sample sigma <- HalfCauchy(scale=2.5)
    sample U : User <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=User]
    sample V : Movie <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=Movie]

    let u_row = U[user_idx]
    let v_row = V[movie_idx]
    let score = sum(u_row * v_row)

    observe r : Rating <- Normal(loc=score, scale=sigma)
    return r

export pmf
```

## Understanding the model

A user vector describes preferences and an item vector describes features. Their dot product is a predicted rating: matching coordinates reinforce one another, and opposite signs lower it. The latent coordinates are learned jointly, so their interpretation need not be fixed in advance.

The data are a list of observed triples $(u_n, m_n, r_n)$ rather than a full matrix. Missing cells are not zeros. They are simply absent from the likelihood, and filling them with zeros would treat those zeros as observed ratings. Since the factors are shared across all of a user's ratings and all of an item's ratings, a user who rated a few items still gets a usable vector, and that is what makes completion possible.

Different factor coordinates can produce the same predicted ratings: for any invertible $K \times K$ matrix $A$, replacing $U$ by $U A$ and $V$ by $V A^{-\top}$ leaves every inner product unchanged (the prior breaks this symmetry only partly). Fits are thus compared on predicted ratings, not on factor entries.

## Walkthrough

`object User : FinSet 50` and `object Movie : FinSet 40` declare the two sides of the rating matrix, `object LatentDim : FinSet 2` the latent dimension $K$, and `object Rating : FinSet 800` the [plate](https://en.wikipedia.org/wiki/Plate_notation) of observed ratings: 800 of the $50 \times 40 = 2000$ cells. `object Val : Real 1` is the value space of a rating.

The three `sample` steps draw the noise scale and the two factor tables. `sample U : User <- Normal(loc=0.0, scale=1.0) [over=LatentDim, iid_over=User]` draws one $K$-vector per user, and the `Movie` table is drawn the same way. The input data `user_idx` and `movie_idx` are integer vectors over the `Rating` plate, giving the user and item of each observed rating. The two gathers `U[user_idx]` and `V[movie_idx]` return `Rating`-by-`LatentDim` tables, and `sum(u_row * v_row)` reduces their last axis, giving each observed rating's score $U_{u_n}^\top V_{m_n}$. `observe r : Rating <- Normal(loc=score, scale=sigma)` scores the observed ratings and nothing else. The same per-row inner-product idiom appears in the [PPCA](ppca.md#walkthrough) example, where every cell of the matrix is observed.

## Try it

The blocks below simulate a rating matrix from the model, hide 60% of its cells, check the compiled joint against a hand-written density, fit by [SVI](../guides/inference-svi.md) (stochastic variational inference) and measure prediction error on the hidden cells against a global-mean baseline, and run a short [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler) chain on the same joint.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

D = torch.distributions

torch.manual_seed(0)
prog = load("docs/examples/source/pmf.qvr")
model = prog.morphism

K, n_user, n_movie, n_obs = 2, 50, 40, 800
true_sigma = 0.3
true_U = torch.randn(n_user, K)
true_V = torch.randn(n_movie, K)
full = true_U @ true_V.T + true_sigma * torch.randn(n_user, n_movie)

cells = torch.randperm(n_user * n_movie)
observed, held_out = cells[:n_obs], cells[n_obs:]
user_idx, movie_idx = observed // n_movie, observed % n_movie
r = full[user_idx, movie_idx]

observations = {"r": r, "user_idx": user_idx, "movie_idx": movie_idx}
x_in = torch.zeros(1, 1)
```

The program input `x_in` is a single row. The extent of the `Rating` plate comes from the observed ratings and their indices, and the scalar site `sigma` is drawn once in that run. Each [input row is an independent run](../guides/inference-foundations.md#input-rows-are-independent-runs). Every cell of `full` is a draw from the model, but only the 800 cells in `observed` reach the program. The other 1200 are kept aside as a test set.

### Checking the joint density

At the true values, [`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) should equal the prior terms plus the Normal likelihood of the observed ratings only, written out by hand.

```python
truth = {"sigma": torch.tensor(true_sigma), "U": true_U, "V": true_V, **observations}
log_joint = model.log_joint(x_in, truth)[0]

score = (true_U[user_idx] * true_V[movie_idx]).sum(dim=-1)
by_hand = (
    D.HalfCauchy(2.5).log_prob(torch.tensor(true_sigma))
    + D.Normal(0.0, 1.0).log_prob(true_U).sum()
    + D.Normal(0.0, 1.0).log_prob(true_V).sum()
    + D.Normal(score, true_sigma).log_prob(r).sum()
)
assert torch.allclose(log_joint, by_hand, rtol=1e-5)
print(f"log joint at the truth: {float(log_joint):.2f}")
```

### SVI fit

Fit an [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) by [`SVI`](../api/inference/svi.md#svi), average the predicted rating matrix $U V^\top$ over 100 guide draws, and compute the root-mean-square error on the held-out cells. The baseline predicts every held-out cell by the mean observed rating. The noise scale $\sigma = 0.3$ is the floor no predictor can beat on average.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"r", "user_idx", "movie_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
losses = [svi.step(x_in, observations) for _ in range(1500)]

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(100)]
predicted = torch.stack([
    d["U"].reshape(n_user, K) @ d["V"].reshape(n_movie, K).T for d in draws
]).mean(0)
fit_sigma = float(torch.stack([d["sigma"] for d in draws]).mean())

held_u, held_m = held_out // n_movie, held_out % n_movie
target = full[held_u, held_m]
model_rmse = float((predicted[held_u, held_m] - target).pow(2).mean().sqrt())
baseline_rmse = float((r.mean() - target).pow(2).mean().sqrt())

print(f"ELBO loss: {losses[0]:.1f} -> {sum(losses[-100:]) / 100:.1f}")
print(f"held-out RMSE: PMF {model_rmse:.3f}, global mean {baseline_rmse:.3f}")
print(f"sigma: fitted {fit_sigma:.3f}, true {true_sigma}")

assert model_rmse < 0.5 * baseline_rmse
assert model_rmse < 1.5 * true_sigma
assert abs(fit_sigma - true_sigma) < 0.1
```

The second assertion requires prediction error within 50% of the noise scale on the held-out cells. It assesses the product of the factors at those cells, not recovery of individual factor entries.

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) runs on the same joint, with both factor tables and the noise scale latent. The chain below is short. A real analysis would run several chains and monitor predicted ratings rather than factor entries.

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

In the category of real matrices, whose composition is matrix multiplication, the factor tables are maps $U : \mathsf{User} \to \mathsf{LatentDim}$ and $V^\top : \mathsf{LatentDim} \to \mathsf{Movie}$, and the dense score matrix is their composite

$$
\mathsf{User} \xrightarrow{U} \mathsf{LatentDim} \xrightarrow{V^\top} \mathsf{Movie},
$$

whose $(u, m)$ entry is $\sum_k U_{u,k} V_{m,k}$. The program never materializes that composite. The observed ratings come with a map $\mathsf{Rating} \to \mathsf{User} \times \mathsf{Movie}$, given by `user_idx` and `movie_idx`, and the two gathers pull the factor tables back along its components, so the contraction over `LatentDim` is done only on the observed cells. Prediction for an unobserved cell evaluates the same composite at a pair outside the image of that map.

## See also

- [Probabilistic PCA](ppca.md), the same inner-product mean with every cell observed and the per-item codes treated as latent factors of a covariance model.
- [Bilinear Tensor Contraction](tensor-contraction.md), a three-way analogue of the bilinear score.
