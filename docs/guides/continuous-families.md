# Continuous Distribution Families

quivers ships more than forty built-in conditional distribution families. Each
takes an input (the domain) and produces learnable parameters via a
parameter network. This page groups the registry by output shape
and details the families whose event structure interacts with the
[axis-role syntax](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over).
The spaces and base
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
interface live in [continuous spaces](continuous-spaces.md).

## Family registry

Choose a family by what one outcome represents and which values it may
take. A Normal can model an unconstrained measurement; a Gamma can model
a positive duration; a Bernoulli models a binary outcome; a Dirichlet
models a whole probability vector. Among families with the same support,
tail behavior can guide the choice: a Student-t places more probability
on extreme deviations than a Normal.

`ConditionalNormal(domain, codomain)` learns a function from inputs to
Normal parameters, including a learned mean `mu(x)`. To specify a mean
exactly, write QVR `Normal(loc=mu, scale=sigma)`: the runtime uses the
values of `mu` and `sigma` supplied at that point in the program.

### Hand-written core families

```python
from quivers.continuous.families import (
    ConditionalNormal,
    ConditionalLogitNormal,
    ConditionalBeta,
    ConditionalTruncatedNormal,
    ConditionalDirichlet,
)
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
import torch

# Input: finite set X
X = FinSet(name="X", cardinality=5)
domain = Euclidean(name="X", dim=5)        # or FinSet
codomain = Euclidean(name="Y", dim=3)

# Conditional normal: learns linear maps mu(x), log_sigma(x)
normal = ConditionalNormal(domain, codomain)

# Sample from p(y | x)
x = torch.randn(8, 5)                              # batch of 8 inputs
samples = normal.rsample(x, sample_shape=torch.Size((100,)))  # (100, 8, 3)

# Log probability
y = torch.randn(8, 3)
log_p = normal.log_prob(x, y)  # shape (8,)
```

### Loc-scale families

Standard reparameterizable distributions with learned location and
scale:

```python
from quivers.continuous.families import (
    ConditionalCauchy,
    ConditionalLaplace,
    ConditionalGumbel,
    ConditionalLogNormal,
    ConditionalStudentT,
)
```

### Positive-valued families

For $\mathbb{R}_{>0}$ output:

```python
import torch
from quivers.continuous.families import (
    ConditionalExponential,
    ConditionalGamma,
    ConditionalWeibull,
    ConditionalPareto,
    ConditionalInverseGamma,
    ConditionalHalfCauchy,
    ConditionalHalfNormal,
)
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
codomain = Euclidean(name="Y", dim=3)
x = torch.randn(8, 5)

gamma = ConditionalGamma(domain, codomain)
samples = gamma.rsample(x)  # positive
```

### Unit-interval families

For $(0, 1)$ output:

```python
from quivers.continuous.families import (
    ConditionalBeta,
    ConditionalKumaraswamy,
    ConditionalContinuousBernoulli,
)
```

### Multivariate families

```python
from quivers.continuous.families import (
    ConditionalMultivariateNormal,
    ConditionalLowRankMVN,
    ConditionalDirichlet,
    ConditionalWishart,
    ConditionalInverseWishart,
    ConditionalMatrixNormal,
    ConditionalLKJCholesky,
    ConditionalGaussianProcess,
    ConditionalHorseshoe,
)
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)

# Multivariate normal with learned mean and cov
mvn = ConditionalMultivariateNormal(domain, Euclidean(name="Y", dim=5))
```

### Discrete / categorical families

```python
from quivers.continuous.families import (
    ConditionalBernoulli,
    ConditionalCategorical,
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
)
```

## Event rank and axis-role syntax

[Three axes can appear in a sampled tensor](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape) [@pyroTensorShapes]. A **sample axis** counts repeated
Monte Carlo draws. A **batch axis** counts separate distribution instances.
An **event axis** holds components scored jointly as one outcome. In the
example above, `(100, 8, 3)` means 100 draws for eight inputs with three
output coordinates; `log_prob` reduces the output coordinates and retains
one score per input. The [axis-role clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over)
assigns these roles to named QVR axes.

Event *rank* is the number of event axes, not their length. A
100-dimensional multivariate Normal has event rank one; a matrix-valued
family has rank two. Scalar Normal draws across three coordinates can be
independent even when stored together, whereas a multivariate Normal can
correlate those coordinates. The `over` and `iid_over` declarations make
that modeling choice visible.

`event_rank` per family controls the
[axis-role syntax](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over):

| Family | Event rank | Categorical reading |
|---|---|---|
| `Normal`, `Beta`, `Gamma`, `Exponential`, `LogNormal`, `LogitNormal`, `Cauchy`, `Laplace`, `Gumbel`, `StudentT`, `Weibull`, `Pareto`, `Kumaraswamy`, `ContinuousBernoulli`, `HalfNormal`, `HalfCauchy`, `InverseGamma`, `TruncatedNormal`, `Uniform`, `Bernoulli` | 0 | Scalar; every codomain axis is iid by default |
| `MultivariateNormal`, `LowRankMVN`, `Dirichlet`, `OneHotCategorical`, `RelaxedOneHotCategorical`, `LogisticNormal`, `GP`, `Horseshoe` | 1 | Vector; one named event axis carries the joint distribution |
| `Wishart`, `InverseWishart`, `MatrixNormal`, `LKJCholesky` | 2 | Matrix; two named event axes carry the joint distribution |

The QVR form `~ Family over <axes>` requires the axis count to
match the family's event rank exactly; mismatch is a compile-time
error. In particular, a flat MVN over $\dim(A) \cdot \dim(B)$
(dense covariance, event_rank 1 with a single named axis whose dim
equals the product) is categorically distinct from a `MatrixNormal`
over `(A, B)` (Kronecker structure $V \otimes U$, event_rank 2);
the declaration thus records both the event axes and the covariance
structure.

## Structured priors over weight matrices

### MatrixNormal: Kronecker-covariance matrix prior

A weight matrix can have two kinds of similarity: rows may covary because
their inputs are related, and columns may covary because their outputs are
related. A matrix-Normal separates those patterns into two covariance
matrices. This reduces parameters but also restricts the correlations
available; it is not equivalent to an arbitrary covariance over every
flattened entry.

```python
from quivers.continuous.families import ConditionalMatrixNormal
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)

# Matrix-valued kernel: domain -> R^(rows*cols), with samples
# reshaped to (rows, cols) and Kronecker covariance Sigma = V (x) U.
mn = ConditionalMatrixNormal(
    domain, Euclidean(name="W", dim=4 * 8), rows=4, cols=8
)
```

The
[matrix-Normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
`MN(M, U, V)` is the natural prior for a weight matrix $W :
\mathbb{R}^d \to \mathbb{R}^k$ whose row and column correlations
factor separately. When used as a latent-morphism prior in the
DSL — `morphism W : Euclidean(D) -> Euclidean(K) [role=latent,
over=[dom, cod]] ~ MatrixNormal(loc, row_scale, col_scale)` — the
first axis listed in `over` binds the row covariance and the
second the column covariance.

### InverseWishart: conjugate covariance prior

```python
from quivers.continuous.families import ConditionalInverseWishart
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)

# Conjugate prior on a d-dim covariance matrix.  Realized as the
# inversion of a Wishart sample with the correct symmetric-matrix
# change-of-variables Jacobian.
d = 4
iw = ConditionalInverseWishart(domain, Euclidean(name="Sigma", dim=d))
```

Conjugate prior for the covariance of a multivariate normal
([@gelman2013bayesian], §3.6).
The change-of-variables Jacobian for inverting a positive-definite
symmetric matrix contributes a `-(d + 1) log det(Sigma)` term to
the log-density.

### LKJCholesky: correlation-matrix prior

A correlation matrix describes association independently of each variable's scale. A Cholesky factor is a triangular representation used to build
a valid matrix and compute with it efficiently. Modeling a factor avoids
proposing an arbitrary array that might fail to be a correlation matrix;
separate scale parameters can then supply marginal variability.

```python
from quivers.continuous.families import ConditionalLKJCholesky
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
d = 4

lkj = ConditionalLKJCholesky(
    domain, Euclidean(name="L", dim=d),
)
```

LKJ-distributed [@lewandowski2009lkj]
Cholesky factors on the manifold of correlation matrices. The
matrix dimension comes from `codomain.dim`; the concentration
parameter is learned from the input by the family's parameter
network (concentration 1 gives the uniform distribution over
correlation matrices).

Draws use the onion method of section 3.2 of
[@lewandowski2009lkj]: row
$k$ of the factor (counting from zero) is
$(\sqrt{y}\, u, \sqrt{1 - y})$ with
$y \sim \mathrm{Beta}(k/2, \eta + (d - 1 - k)/2)$ and $u$ uniform on
the unit sphere in $\mathbb{R}^k$. Every off-diagonal correlation $r$
of a $d \times d$ draw thus has
$(r + 1)/2 \sim \mathrm{Beta}(\eta - 1 + d/2, \eta - 1 + d/2)$. The
compiled runtime, every factory over the family (the fixed and
variable-concentration forms, a plate over it, and
[`LKJCorrelationFactor`][quivers.continuous.families.LKJCorrelationFactor]),
and the QIEC reference machine all draw through one implementation of
this construction, so a program's prior draws follow the same law in
both runtimes. `rsample` is differentiable in the concentration
through implicitly reparameterized Beta draws; the directions $u$ do
not depend on it. The log density is torch's `LKJCholesky.log_prob`.

### Wishart

```python
from quivers.continuous.families import ConditionalWishart
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
d = 4

wishart = ConditionalWishart(
    domain, Euclidean(name="Sigma", dim=d),
)
```

## Non-parametric and structured families

### ConditionalGaussianProcess

A Gaussian process assigns a joint distribution to function values at
chosen inputs. Its covariance kernel determines which inputs should have
similar outputs. With an RBF kernel, nearby inputs tend to receive more
strongly correlated values; the length scale controls how quickly this
relationship decays. One draw is a collection of function values, not a
separate unrelated regression coefficient at every location.

A [Gaussian
process](https://en.wikipedia.org/wiki/Gaussian_process) on
$\mathbb{R}^D$ is a Markov kernel $X^N \to \mathcal{G}(\mathbb{R}^N)$
whose value at any finite collection of input locations
$x_1, \dots, x_N \in \mathbb{R}^D$ is a joint multivariate Normal
with covariance $K(x_i, x_j)$. Each evaluation uses the entire collection
of input locations to construct this covariance matrix. Changing one
location changes its covariance with the other locations.
[`ConditionalGaussianProcess`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess)
exposes three covariance kernels (RBF, Matern 5/2, linear) with
learnable length scale and amplitude.

```python
from quivers.continuous.families import ConditionalGaussianProcess
from quivers.continuous.spaces import Euclidean
import torch

D, N = 2, 8
gp = ConditionalGaussianProcess(
    Euclidean(name="X", dim=D),
    Euclidean(name="Y", dim=N),
    kernel="rbf",
    length_scale=0.5,
    amplitude=1.0,
)
x = torch.randn(N, D)            # N input locations in R^D
f = gp.rsample(x)                # one GP draw at those inputs, shape (N,)
log_p = gp.log_prob(x, f)        # MVN(0, K(x, x) + jitter * I) density
```

The DSL registers the family as `GP` with event rank 1; one named
axis (the input-location count $N$) carries the joint distribution.
For the statistical foundations and worked examples, see [@rasmussen2006gp]. The [open-access edition](https://www.gaussianprocess.org/gpml/) is available from the authors.

### ConditionalHorseshoe

Shrinkage priors express a preference for small coefficients while allowing
some large effects. In the hierarchy below, the global scale `tau` controls
overall shrinkage, while each local scale `lambda_d` can let its coefficient
escape that shrinkage. The heavy tails of the scale priors distinguish this
behavior from choosing a small Normal variance for every coefficient.

The horseshoe prior [@carvalho2010horseshoe] is a
sparse-signal prior whose hierarchical form is

$$
\begin{aligned}
\tau &\sim \mathrm{HalfCauchy}(\text{scale}) \\
\lambda_d &\sim \mathrm{HalfCauchy}(1) \quad \text{for } d = 1, \dots, D \\
\beta_d \mid \tau, \lambda_d &\sim \mathcal{N}\!\bigl(0, (\tau \lambda_d)^2\bigr).
\end{aligned}
$$

The marginal density of $\beta_d$ after integrating the local scale
$\lambda_d$ has no elementary closed form.
[`ConditionalHorseshoe`](../api/continuous/families.md#quivers.continuous.families.ConditionalHorseshoe)
approximates the marginal with 16-point
[Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature)
after the change of variables $\lambda = \tan(\pi t / 2)$ which
maps $(0, \infty)$ to $(0, 1)$ with Jacobian
$(\pi / 2) \sec^2(\pi t / 2)$.

```python
import torch
from quivers.continuous.families import ConditionalHorseshoe
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet

hs = ConditionalHorseshoe(
    FinSet(name="A", cardinality=4),
    Euclidean(name="beta", dim=10),
    scale=0.1,
)
x = torch.tensor([0, 1, 2, 3])
beta = hs.rsample(x)             # shape (4, 10)
log_p = hs.log_prob(x, beta)     # shape (4,), summed over coordinates
```

The DSL registers the family as `Horseshoe` with event rank 1; the
one named axis carries the per-coordinate marginal product.

### Choosing among multivariate priors

| Goal | Family | Why |
|---|---|---|
| Generic multivariate posterior with dense covariance | `ConditionalMultivariateNormal` | Standard; learnable Cholesky factor. |
| Weight matrix with separable row / column correlations | `ConditionalMatrixNormal` | Kronecker structure $V \otimes U$ has $r^2 + c^2$ params instead of $(rc)^2$. |
| Covariance prior for a downstream MVN | `ConditionalInverseWishart` | Places a distribution on positive-definite covariance matrices. |
| Correlation matrix in a hierarchical model | `ConditionalLKJCholesky` | Prior on correlations alone, leaving scales free. |
| Non-parametric regression with smoothness control | `ConditionalGaussianProcess` | Prior over functions; bandwidth from kernel length-scale. |
| Sparse signal selection (e.g. variable selection in regression) | `ConditionalHorseshoe` | Strong shrinkage toward zero with heavy tails for large signals. |

## See also

- [Continuous Spaces and Morphisms](continuous-spaces.md): the
  surrounding space hierarchy and composition operations.
- [DSL Programs and Let-Expressions](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over):
  the `over` / `iid over` axis-role clause that depends on
  `event_rank`.
- [Examples Gallery: Horseshoe Regression](../examples/horseshoe-regression.md):
  end-to-end use of `ConditionalHorseshoe` in a regression.

## Further reading

- [@gelman2013bayesian]
- [@carvalho2010horseshoe]
