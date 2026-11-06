# Probabilistic Matrix Factorization

## Overview

[Probabilistic matrix factorization](https://en.wikipedia.org/wiki/Probabilistic_matrix_factorization) [@salakhutdinov2007pmf] is the standard latent-factor recommender model: each user and each item has a $K$-dimensional latent vector, the expected rating is their inner product, and observed ratings are Normal noise around that inner product:

$$
r_{u, m} \mid U_{:, u}, V_{:, m} \sim \mathcal{N}(\langle U_{:, u}, V_{:, m} \rangle, \sigma_{\text{obs}}^2).
$$

In quivers, the two factor matrices are arrows $U : \mathsf{LatentDim} \to \mathsf{User}$ and $V : \mathsf{LatentDim} \to \mathsf{Movie}$. The bilinear score is the composition $U^\dagger \mathbin{>>} V : \mathsf{User} \to \mathsf{Movie}$, whose `(u, m)` entry is $\sum_k U_{k,u}V_{k,m}$. The source declares learnable latent parameters but no explicit priors; the fits below supply the Normal priors that turn the deterministic score into a joint density.

## QVR source

```qvr
# Probabilistic Matrix Factorization
#
# A user-item latent-factor recommender model. Every user and
# every item carries a K-dimensional latent vector, and the
# expected rating is their inner product. In categorical form,
# the two factor matrices are morphisms and the bilinear score
# is the composite User -> Movie obtained by transposing U and
# contracting along LatentDim.
#
# Structural form:
#
#   U     : LatentDim -> User            user latent factors
#   V     : LatentDim -> Movie           movie latent factors
#   pmf   = U.dagger >> V                bilinear rating score
#
# The (u, m) entry of the resulting tensor is sum_k U[k, u] *
# V[k, m], the canonical PMF rating mean. The `[role=latent]`
# declarations create learned factor tensors but do not declare
# probability distributions; the runnable fit on the page adds
# Normal priors and a Normal observation model in Python.

composition real [level=algebra]

object LatentDim : FinSet 2
object User, Movie : FinSet 8

morphism U : LatentDim -> User [role=latent]
morphism V : LatentDim -> Movie [role=latent]

define pmf = U.dagger >> V

export pmf
```

## Understanding the model

A user vector describes preferences and an item vector describes features.
Their [dot product](../tutorials/python/07-composition-rules.md#operadic-n-ary-contractions) is a predicted rating: matching coordinates reinforce
one another, and opposite signs can lower it. The latent coordinates are
learned jointly, so their interpretation need not be fixed in advance.

`U.dagger` transposes the table to put users on the input side and latent
features on the shared axis. Its entries stay the same under this transpose.
Different factor coordinates can produce the same rating matrix, so compare
predicted ratings when evaluating the fit. The code below scores a dense
synthetic matrix. With missing ratings, score only the observed cells: filling
missing cells with zero would treat those zeros as observed ratings.

## Walkthrough

The two top-level declarations introduce the factor matrices as first-class arrows. The following `MatrixNormal` declarations illustrate optional priors and are not present in the loaded source:

<!-- compile: false -->
```qvr
morphism U : LatentDim -> User [role=latent, over=[dom, cod]] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0)
morphism V : LatentDim -> Movie [role=latent, over=[dom, cod]] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0)
```

The `.dagger` modifier on $U$ transposes the morphism to $\mathsf{User} \to \mathsf{LatentDim}$. The composition `U.dagger >> V` contracts along `LatentDim` and recovers the full `(User, Movie)` score matrix; under `composition real [level=algebra]` this is a real matmul and the resulting tensor entry at `(u, m)` is exactly $\sum_k U_{k, u} V_{k, m}$.

Working over discrete `User` and `Movie` plates materialises the full dense score matrix. For very large catalogues the dense materialisation is wasteful and a per-rating gather is preferable; the morphism API can lift that gather as a separate fibration $\mathsf{Rating} \to \mathsf{User} \times \mathsf{Movie}$ composed with the bilinear PMF morphism.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


The QVR source declares $U^\dagger \mathbin{>>} V$ as a deterministic morphism, so it defines a mean tensor rather than a probability measure. The snippets below add the standard PMF probability model: every entry of each factor matrix carries an independent $\mathcal{N}(0, 1)$ prior, and every cell of the rating matrix is scored under $r_{u, m} \sim \mathcal{N}(S_{u, m}, \sigma^2)$ with $S = U^\top V$. The mean is rebuilt from the sampled factors on each evaluation by composing them exactly as the source does, which keeps $U$ and $V$ latent variables of the model rather than fixed tensors.

### Generating synthetic data

Draw ground-truth factor matrices, push them through $U^\dagger \mathbin{>>} V$ to get the dense $8 \times 8$ score matrix, then add Normal observation noise to every cell. The two `sample` sites of the [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) built here are named after the arrows the source declares, and the ratings are generated from the very factors bound as ground truth, so the snippet leaves one self-consistent point of the joint behind. Object cardinalities and the active algebra are read off the compiled module through the [`Compiler`](../api/dsl/compiler.md#quivers.dsl.compiler.Compiler) environment rather than restated, so the block tracks the source if the source changes.

```python
import torch
import torch.distributions as D

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous import Draw, Let, MonadicProgram, Observe
from quivers.continuous.spaces import Euclidean
from quivers.core.morphisms import ObservedMorphism
from quivers.core.objects import Unit
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse_file

torch.manual_seed(0)

compiler = Compiler(parse_file("docs/examples/source/pmf.qvr"))
compiler.compile()
U_arrow = compiler.morphisms["U"]
V_arrow = compiler.morphisms["V"]
algebra = compiler.algebra

K = int(U_arrow.domain.cardinality)
n_user = int(U_arrow.codomain.cardinality)
n_movie = int(V_arrow.codomain.cardinality)
sigma = 0.5


def bilinear_score(U, V):
    """Dense (User, Movie) tensor of `U.dagger >> V` under `real`."""
    left = ObservedMorphism(U_arrow.domain, U_arrow.codomain, U, algebra=algebra)
    right = ObservedMorphism(V_arrow.domain, V_arrow.codomain, V, algebra=algebra)
    return (left.dagger >> right).tensor


class FactorPrior(ContinuousMorphism):
    """Independent Normal(0, 1) prior over one factor matrix's entries."""

    def __init__(self, rows, cols):
        super().__init__(Unit, Euclidean(name="Factor", dim=rows * cols))
        self._shape = (rows, cols)

    def rsample(self, x, sample_shape=torch.Size()):
        return D.Normal(torch.zeros(self._shape), 1.0).rsample()

    def log_prob(self, x, y):
        return D.Normal(0.0, 1.0).log_prob(y.reshape(self._shape)).sum()


class RatingLikelihood(ContinuousMorphism):
    """Normal rating around the bilinear score, one per (user, movie) cell."""

    def __init__(self, sigma):
        super().__init__(
            Euclidean(name="Score", dim=n_movie),
            Euclidean(name="Rating", dim=n_movie),
        )
        self._sigma = sigma

    def rsample(self, x, sample_shape=torch.Size()):
        return D.Normal(x, self._sigma).rsample()

    def log_prob(self, x, y):
        return D.Normal(x, self._sigma).log_prob(y.reshape(x.shape)).sum()


model = MonadicProgram(
    domain=Euclidean(name="Ix", dim=1),
    codomain=Euclidean(name="Rating", dim=n_movie),
    steps=[
        Draw(names=("U",), morphism=FactorPrior(K, n_user)),
        Draw(names=("V",), morphism=FactorPrior(K, n_movie)),
        Let(
            name="mu",
            value=lambda env: bilinear_score(
                env["U"].reshape(K, n_user),
                env["V"].reshape(K, n_movie),
            ),
        ),
        Observe(
            names=("rating",), morphism=RatingLikelihood(sigma), args=("mu",)
        ),
    ],
    return_vars=("rating",),
)

true_U = torch.randn(K, n_user)
true_V = torch.randn(K, n_movie)
score = bilinear_score(true_U, true_V)
rating = D.Normal(score, sigma).sample()

observations = {"rating": rating}
x_in = torch.zeros(1, 1)

print("score row 0:", score[0].round(decimals=2).tolist())
```

The program input `x_in` carries no coordinate of the model: every site is either a global factor matrix or the full rating plate, so the input is a single bracket row the steps read past. A gather-based likelihood over a sparse `(user, movie)` sample would instead thread the index pairs through it.

### SVI fit

Fit with [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) + [`ELBO`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) + [`SVI`](../api/inference/svi.md#svi). Both factor matrices are `sample` sites. The guide places an independent Normal distribution on each unconstrained coordinate and optimizes the ELBO using draws of both matrices.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"rating"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(200)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

The factorization is identifiable only up to a $K \times K$ invertible reparameterization, so the fitted $U$, $V$ tend to differ from the truths even once the score matrix has converged.

### NUTS posterior

Because the priors are declared inside the program, [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) targets it directly; a model that instead exposed $U$ and $V$ as bare `[role=latent]` parameters would first need [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) to give them a prior.

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

The two factor matrices $U : \mathsf{LatentDim} \to \mathsf{User}$ and $V : \mathsf{LatentDim} \to \mathsf{Movie}$ are arrows in the discrete-object category whose priors live on the hom-objects $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{User})$ and $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{Movie})$. The rating likelihood couples them through the bilinear pairing realized as the composition

$$
\mathsf{User} \xrightarrow{U^\dagger} \mathsf{LatentDim} \xrightarrow{V} \mathsf{Movie}
$$

in the real algebra, whose tensor is the dense score matrix $U^\top V$. The loaded source treats $U$ and $V$ as learned tensors; prior measures enter only through an explicit prior declaration or through the priors the fits on this page declare around the composition.

## See also

- [Isotropic Bayesian Factor Model](factor-analysis.md) for a single-side morphism-valued loading.
- [DSL Guide](../guides/dsl-overview.md) for morphism-valued prior syntax and the [`.dagger`](../api/core/morphisms.md) transpose.
