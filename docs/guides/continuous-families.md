# Continuous Distribution Families

A conditional distribution family in quivers is a
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
whose distribution's parameters a parameter source, a lookup table or a network,
computes from each input. More than fifty families are registered, and in QVR the same
families are written inline in program steps with given parameters
([Inline distributions](dsl-programs-and-lets.md#inline-distributions)).
[Continuous Spaces](continuous-spaces.md) describes the spaces they live on.

## Family registry

A family is chosen by what one outcome represents and which values it may take: a
Normal for an unconstrained measurement, a Gamma for a positive duration, a Bernoulli
for a binary outcome, a Dirichlet for a whole probability vector. Among families with
the same support, the tails decide: a Student-t puts more probability on extreme
deviations than a Normal.

`ConditionalNormal(domain, codomain)` learns a map from inputs to the normal's
parameters, including its mean. A mean given exactly is written in QVR as
`Normal(loc=mu, scale=sigma)`, which uses the values bound to `mu` and `sigma` at that
step.

### Core families {#hand-written-core-families}

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

# Conditional normal: learns maps x -> mu(x) and x -> log sigma(x)
normal = ConditionalNormal(domain, codomain)

# Sample from p(y | x)
x = torch.randn(8, 5)                              # batch of 8 inputs
samples = normal.rsample(x, sample_shape=torch.Size((100,)))  # (100, 8, 3)

# Log probability
y = torch.randn(8, 3)
log_p = normal.log_prob(x, y)  # shape (8,)
```

### Loc-scale families

Families with a learned location and scale, all with reparameterized draws:

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

For positive outputs, $(0, \infty)$:

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

For outputs in $(0, 1)$:

```python
from quivers.continuous.families import (
    ConditionalBeta,
    ConditionalKumaraswamy,
    ConditionalContinuousBernoulli,
)
```

### Log-space families for small shapes

A [Beta](https://en.wikipedia.org/wiki/Beta_distribution) or
[Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) latent reaches
the samplers and the guides on the real line: its unconstrained value is
mapped into the support by a sigmoid or an exponential and scored there,
with the map's Jacobian. When the shape is small, as it tends to be in
sparse and stick-breaking priors, that round trip loses the tails. The
sigmoid of `-800` is already the smallest normal float, so
`Beta(0.001, 1)` scores the logit `-800` at about `-99` where its exact
log density is about `-7.71`, and a Gamma sampler at shape `0.001` returns the smallest
normal float for roughly half its draws.

[`LogitBeta`][quivers.continuous.log_space.LogitBeta] and
[`LogGamma`][quivers.continuous.log_space.LogGamma] put the variable on
the real line from the start. `LogitBeta(concentration1=a,
concentration0=b)` is the law of $x = \operatorname{logit}(p)$ for
$p \sim \mathrm{Beta}(a, b)$, and `LogGamma(concentration=a, rate=r)` is
the law of $y = \log g$ for $g \sim \mathrm{Gamma}(a, r)$:

$$
\log p(x) = a \log \sigma(x) + b \log \sigma(-x) - \log \mathrm{B}(a, b),
\qquad
\log p(y) = a (\log r + y) - r e^{y} - \log \Gamma(a).
$$

Both are exact for every finite argument, since no quantity is formed on
the constrained scale. Sampling stays in log space through the shape
augmentation of Marsaglia and Tsang [@marsaglia2000gamma]: for independent
$G' \sim \mathrm{Gamma}(a + 1)$ and $U \sim \mathrm{Uniform}(0, 1)$,
$\log G' + \log(U) / a$ has the law of $\log G$ with
$G \sim \mathrm{Gamma}(a)$, and a logit-Beta draw is the difference of two
such draws. $G'$ has shape at least one, so it never comes near the
underflow threshold, and its implicit reparameterization gradient, which
differentiates the distribution function instead of the sampler, carries through
[@figurnov2018implicit]. Draws are finite for every shape at or above 64 times the
smallest normal number of the dtype, and at shape `1e-8` the median draw of `LogGamma`
is near `-6.9e7`.

Because the support is the real line, a site drawn from either family
takes the identity as its bijector, so HMC, NUTS, and the guides see the exact
density, and prior draws, `rsample`, and initialization from the prior
all sample the declared prior. A downstream quantity on the constrained
scale is a `let`:

<!-- compile: standalone -->
```qvr
object Item : FinSet 20

program sticks : Item -> Item
    sample weight_logit : Item <- LogitBeta(concentration1=1.0, concentration0=0.01)
    sample log_rate <- LogGamma(concentration=0.001, rate=1.0)
    let weight = sigmoid(weight_logit)
    let rate = exp(log_rate)
    observe y : Item <- Normal(loc=weight, scale=1.0)
    return weight

export sticks
```

From Python, the distributions are ordinary `torch.distributions`
classes, and
[`make_fixed_logit_beta`][quivers.continuous.inline.make_fixed_logit_beta]
and [`make_fixed_log_gamma`][quivers.continuous.inline.make_fixed_log_gamma]
build the fixed program steps the compiler emits for literal parameters. The inline
paths pass small literal shapes through unchanged, and a shape that is not positive
raises.

```python
import torch
from quivers.continuous import LogGamma, LogitBeta

shape = torch.tensor(1e-3, dtype=torch.float64)
prior = LogitBeta(shape, torch.tensor(1.0, dtype=torch.float64))
prior.log_prob(torch.tensor(-800.0, dtype=torch.float64))  # about -7.71

draws = LogGamma(shape, torch.tensor(1.0, dtype=torch.float64)).rsample((5,))
assert torch.isfinite(draws).all()  # near -1000 on average, never clamped
```

The conditional forms
[`ConditionalLogitBeta`][quivers.continuous.families.ConditionalLogitBeta]
and [`ConditionalLogGamma`][quivers.continuous.families.ConditionalLogGamma]
learn their shapes as the softplus of a network output. The conditional Beta and Gamma
families add a positive shift to that softplus (0.1 for `ConditionalBeta`), so their
learned shapes cannot fall below it. [Transpiler support](../transpile-support.md)
lists the targets that accept the two log-space families.

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

Three kinds of axis appear in a drawn tensor [@pyroTensorShapes]. A *sample* axis counts
repeated draws, a *batch* axis counts separate distributions, and an *event* axis
holds coordinates scored jointly as one outcome. In the example above, `(100, 8, 3)` is
100 draws for eight inputs with three output coordinates, and `log_prob` sums over the
output coordinates and keeps one score per input. The
[`over` clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over) assigns
these roles to named QVR axes.

A family's number of event dimensions, `event_rank` in the code, counts its event axes,
not their length: a 100-dimensional multivariate normal has one and a matrix family has
two. Scalar normal draws over three coordinates are independent even when stored
together, while a multivariate normal can correlate them, and the `over` clause makes
that choice explicit:

| Family | Event dimensions | Interpretation |
|---|---|---|
| `Normal`, `Beta`, `Gamma`, `Exponential`, `LogNormal`, `LogitNormal`, `Cauchy`, `Laplace`, `Gumbel`, `StudentT`, `Weibull`, `Pareto`, `Kumaraswamy`, `ContinuousBernoulli`, `HalfNormal`, `HalfCauchy`, `InverseGamma`, `TruncatedNormal`, `Uniform`, `Bernoulli`, `Horseshoe` | 0 | Scalar; every codomain axis is iid by default |
| `MultivariateNormal`, `LowRankMVN`, `Dirichlet`, `OneHotCategorical`, `RelaxedOneHotCategorical`, `LogisticNormal`, `GP` | 1 | Vector; one named event axis carries the joint distribution |
| `Wishart`, `InverseWishart`, `MatrixNormal`, `LKJCholesky` | 2 | Matrix; two named event axes carry the joint distribution |

On a `morphism` declaration, `[over=<axes>] ~ Family` must list exactly as many axes as
the family has event dimensions, and a mismatch is a compile error. A `sample` or
`observe` step may list more, the extra leading axes indexing independent draws
([`over` clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over)). A
multivariate normal over all $|A| \cdot |B|$ entries, with a dense covariance and one
event axis whose length is the product, is a different model from a `MatrixNormal` over
`(A, B)`, whose covariance is the Kronecker product $V \otimes U$ and which has two
event axes, so the declaration records both the event axes and the covariance
structure.

## Structured priors over weight matrices

### MatrixNormal: Kronecker-covariance matrix prior

The rows of a weight matrix may covary because their inputs are related, and its
columns because their outputs are. A
[matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
$\mathcal{MN}(M, U, V)$ has a row covariance $U$ and a column covariance $V$, and the
covariance of its flattened entries is their
[Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product). This needs far
fewer parameters than a covariance over every entry and also allows fewer
correlations.

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

As a prior on a learnable table in QVR,
`morphism W : D -> K [role=latent, over=[dom, cod]] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0)`,
the first axis in `over` takes the row covariance and the second the column
covariance ([Latent morphism prior](dsl-declarations.md#latent-morphism-prior)).

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

The inverse Wishart is the [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
for the covariance of a multivariate normal, so with normal data its posterior is again
an inverse Wishart [@gelman2013bayesian]. The change-of-variables Jacobian for inverting a positive-definite
symmetric matrix contributes a `-(d + 1) log det(Sigma)` term to
the log-density.

### LKJCholesky: correlation-matrix prior

A correlation matrix describes association apart from each variable's scale, and its
[Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition) is the
lower-triangular $L$ with $L L^\top$ equal to it. Placing the prior on the factor means
every draw is a valid correlation matrix, and separate scale parameters then give the
marginal variances.

```python
from quivers.continuous.families import ConditionalLKJCholesky
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
d = 4

lkj = ConditionalLKJCholesky(
    domain, Euclidean(name="L", dim=d),
)
```

The family draws Cholesky factors of correlation matrices from the
[LKJ distribution](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution)
[@lewandowski2009lkj], with density proportional to $\det(R)^{\eta - 1}$ for the
correlation matrix $R$ and concentration $\eta$, so $\eta = 1$ is uniform over
correlation matrices. The matrix dimension is `codomain.dim`, and the parameter source
learns $\eta$ from the input.

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
and the [reference machine](../developer/qiec.md#reference-evaluator), the library's interpreter for its core language, all draw through one implementation of
this construction, so a program's prior draws follow the same law in
both runtimes. `rsample` is differentiable in the concentration
through implicitly reparameterized Beta draws, and the directions $u$ do not depend
on it. The log density is torch's `LKJCholesky.log_prob`.

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

A [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) on $\mathbb R^D$
gives the function values at any finite set of inputs $x_1, \dots, x_N \in \mathbb R^D$
a joint multivariate normal distribution with covariance $K(x_i, x_j)$
[@rasmussen2006gp]. As a family it is a kernel from the $N$ input locations to
distributions on $\mathbb R^N$, and each evaluation builds the covariance matrix from
all the locations at once, so moving one location changes its covariance with the
others. With an RBF covariance kernel, nearby inputs get more strongly correlated
values, and the length scale sets how fast the correlation decays.
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

QVR registers the family as `GP`, with one event dimension: the axis of the $N$ input
locations carries the joint distribution.

### ConditionalHorseshoe

The horseshoe prior [@carvalho2010horseshoe] prefers small coefficients and still
allows some large ones. In its hierarchical form below, the global scale $\tau$ sets
the overall shrinkage, and each local scale $\lambda_d$ can let its coefficient escape
it. The heavy tails of the scale priors are what distinguish it from a small normal
variance for every coefficient:

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

QVR registers the family as `Horseshoe` with no event dimensions: each coefficient is
one draw, scored by its marginal density, and draws over several coefficients are
independent.

### Choosing among multivariate priors

| Goal | Family | Why |
|---|---|---|
| Multivariate prior with a dense covariance | `ConditionalMultivariateNormal` | Standard; learnable Cholesky factor. |
| Weight matrix with separable row and column correlations | `ConditionalMatrixNormal` | The Kronecker covariance $V \otimes U$ has on the order of $r^2 + c^2$ parameters instead of $(rc)^2$. |
| Covariance prior for a downstream MVN | `ConditionalInverseWishart` | Places a distribution on positive-definite covariance matrices. |
| Correlation matrix in a hierarchical model | `ConditionalLKJCholesky` | Prior on correlations alone, leaving scales free. |
| Non-parametric regression with smoothness control | `ConditionalGaussianProcess` | Prior over functions; bandwidth from kernel length-scale. |
| Sparse signals, such as variable selection in regression | `ConditionalHorseshoe` | Strong shrinkage toward zero with heavy tails for large signals. |

## See also

- [Continuous Spaces and Morphisms](continuous-spaces.md): the
  surrounding space hierarchy and composition operations.
- [DSL Programs and Let-Expressions](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over):
  the `over` and `iid_over` clause, which depends on the number of event dimensions.
- [Examples Gallery: Horseshoe Regression](../examples/horseshoe-regression.md):
  end-to-end use of `ConditionalHorseshoe` in a regression.
