# Correlation-Matrix Prior via LKJCholesky

## Overview

A prior over $K \times K$ correlation matrices through their Cholesky factor. The [LKJ family](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution) uses a shape parameter $\eta > 0$ to control concentration around the identity: $\eta > 1$ favours weaker correlations, $\eta = 1$ is uniform over correlation matrices, and $\eta < 1$ favours stronger correlations. Its Cholesky factor is lower triangular with positive diagonal and unit row norms. It is not unit-lower-triangular. [@lewandowski2009lkj]

Generative structure:

$$
\begin{aligned}
\eta &\sim \mathrm{HalfNormal}(2),\\
L &\sim \mathrm{LKJCholesky}(\eta),
\end{aligned}
$$

where $L L^\top$ is the implied correlation matrix.

## QVR source

```qvr
# Correlation-Matrix Prior via LKJCholesky
#
# A prior over correlation matrices, given in its Cholesky-factor
# form (named after Lewandowski-Kurowicka-Joe). The Cholesky
# factorization gives a direct representation for Gaussian likelihoods.
# Both the factor and the positive-definite correlation matrix have
# unconstrained coordinates, related through the Cholesky transform.
#
# Generative structure:
#
#   eta     ~ HalfNormal(2)                          shape parameter
#   chol    ~ LKJCholesky(eta)                       Cholesky factor
#
# The program carries no observe step: it is the prior model a
# multivariate model composes with, and Sigma = chol * chol^T is
# the correlation matrix a downstream likelihood would consume.
#
# Dim is the matrix dimension the LKJCholesky family reads off its
# index annotation, and Factor is the value space of one row of
# the returned factor, a point of R^K.
#
# Reference: [Lewandowski et al. 2009](https://doi.org/10.1016/j.jmva.2009.04.008).

object Dim : FinSet 4
object Factor : Real 4

program correlation_model : Dim -> Factor
    sample eta <- HalfNormal(scale=2.0)
    sample chol : Dim <- LKJCholesky(concentration=eta)
    return chol

export correlation_model
```

## Understanding the model

Choosing each pairwise correlation separately can produce an invalid matrix. The entries must jointly define a positive-definite matrix. A Cholesky factor enforces this constraint through a triangular matrix with positive diagonal and unit row norms. Both this factor and the correlation matrix admit [unconstrained parameterizations](https://mc-stan.org/docs/reference-manual/variable-transforms.html) [@stanReparameterization]. Multiplying `L` by its transpose gives the correlation matrix, with diagonal entries one.

For marginal standard deviations `s`, the covariance is
`diag(s) @ (L @ L.T) @ diag(s)`. The example draws from the [prior](../getting-started/concepts.md#4-a-probabilistic-program-describes-how-data-could-arise) with no
observations. Varying `eta` lets you compare its distributions over correlation
matrices before adding a likelihood.

## Walkthrough

`object Dim : FinSet 4` declares the matrix dimension. The `sample chol : Dim <- ...` line carries that as the family's matrix axis, so `LKJCholesky(eta)` produces a $4 \times 4$ lower-triangular Cholesky factor whose rows have unit Euclidean norm. `object Factor : Real 4` is the value space of one row of that factor, and the program's signature `Dim -> Factor` names it as the codomain, since the codomain of a program is the space its returned value lives in rather than an index set. The shape parameter $\eta$ is drawn from a half-Normal, giving a weakly informative prior over how strongly the correlation matrix concentrates around the identity. Downstream consumers reconstruct the correlation matrix as $\Sigma = L L^\top$ when they need the dense form.

## Try it

> The runs below demonstrate the API on the prior alone. A downstream likelihood would condition the factor. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Drawing a factor

The program declares a prior and no likelihood, so a forward run draws the shape parameter and a factor under it. What a consumer needs from the factor is the dense correlation matrix, recovered by one multiplication.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/lkj_cholesky_correlation.qvr")
model = prog.morphism

x_in = torch.zeros(1, 1)
chol = model.rsample(x_in)
factor = chol[0]
correlation = factor @ factor.T

print("factor:  ", tuple(chol.shape))
print("row norms:", factor.norm(dim=-1))
print("diagonal: ", correlation.diagonal())
```

The factor has shape `(1, 4, 4)`, one $4 \times 4$ lower-triangular matrix for the single input row. Both printed vectors are ones: the factor's rows are unit-norm by construction, so the implied $\Sigma = L L^\top$ is a correlation matrix rather than a general covariance. A model with per-dimension scales $\tau$ turns it into a covariance as $\mathrm{diag}(\tau)\,\Sigma\,\mathrm{diag}(\tau)$, which is the form a multivariate-Normal likelihood consumes.

### Scoring the prior

`log_joint` scores the two sites at given values: the half-Normal density of $\eta$ plus the LKJ density of the factor under it, which is the sum torch's own distributions give.

```python
eta = torch.tensor([[2.0]])
log_joint = model.log_joint(x_in, {"eta": eta, "chol": chol})

closed_form = (
    torch.distributions.HalfNormal(2.0).log_prob(torch.tensor(2.0))
    + torch.distributions.LKJCholesky(4, torch.tensor(2.0)).log_prob(factor)
)
assert torch.allclose(log_joint, closed_form.reshape(1), atol=1e-5)
print("log joint:", float(log_joint))
```

### Tracing a run

A `trace` records every site a forward run visits with its value and log density, so the prior's joint at the drawn point is the sum over its two sites.

```python
from quivers.inference import trace

torch.manual_seed(1)
run = trace(model, x_in)
print(sorted(run.sites))
print("log joint:", float(run.log_joint))
assert set(run.sites) == {"eta", "chol"}
assert torch.isfinite(run.log_joint).all()
```

## Use cases

The factor can parameterize multivariate Gaussian likelihoods, hierarchical regressions with correlated effects, and [copulas](https://en.wikipedia.org/wiki/Copula_(statistics)) with separately modeled marginals. Hamiltonian Monte Carlo operates on unconstrained Euclidean coordinates through a transform of the factor [@stanReparameterization].
