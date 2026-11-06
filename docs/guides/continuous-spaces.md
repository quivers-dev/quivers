# Continuous Spaces and Morphisms

This page covers the `ContinuousSpace` hierarchy, the
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
interface, composition operations, the boundary between discrete and
continuous models, and normalizing flows. The full registry of
parameterized distribution families lives in
[continuous families](continuous-families.md).

## ContinuousSpace hierarchy

A finite object lists possible indices. A continuous space specifies
possible numerical values: a point in `Real 3` has three real coordinates,
such as `(0.2, -1.0, 4.5)`. The space also records constraints. A simplex
contains probability vectors whose entries sum to one; positive reals
can represent a scale that must stay above zero. These constraints define
the *support* on which a distribution may place mass.


Continuous morphisms act between continuous measurable spaces.
quivers provides a hierarchy of standard spaces:

```
ContinuousSpace (abstract)
├── Euclidean         R^n with Lebesgue measure
├── UnitInterval      (0, 1) with Lebesgue measure (defined via Euclidean)
├── Simplex           {x in R^dim : x >= 0, sum x = 1}
├── PositiveReals     (0, inf) with Lebesgue measure
├── CholeskyFactor    K x K lower-triangular Cholesky factors with unit-norm rows
├── ProductSpace      Cartesian product of spaces (and discrete objects)
├── Sphere            unit sphere S^{N-1} in R^N
├── Ball              closed ball of radius r in R^N
├── Covariance        cone of D x D symmetric positive-definite matrices
├── Correlation       D x D correlation matrices (unit diagonal, PD)
├── Orthogonal        orthogonal group O(D)
├── Stiefel           Stiefel manifold V_K(R^N) of N x K orthonormal-column matrices
├── LowerTriangular   D x D lower-triangular matrices
└── Diagonal          D x D diagonal matrices (identified with R^D)
```

### Euclidean space

```python
from quivers.continuous.spaces import Euclidean

R3 = Euclidean(name="R3", dim=3)                                  # R^3
R2_bounded = Euclidean(name="R2", dim=2, low=0.0, high=1.0)        # [0,1]^2
```

### Unit interval

```python
from quivers.continuous.spaces import UnitInterval

U = UnitInterval(name="U")     # (0, 1); pass dim=k for [0, 1]^k
```

### Simplex

```python
from quivers.continuous.spaces import Simplex

S3 = Simplex(name="S3", dim=3)   # 2-simplex in R^3 (3 categories with sum = 1)
```

### PositiveReals

```python
from quivers.continuous.spaces import PositiveReals

P2 = PositiveReals(name="P2", dim=2)   # (0, inf)^2
```

### ProductSpace

```python
from quivers.continuous.spaces import ProductSpace, Euclidean, UnitInterval

R3 = Euclidean(name="R3", dim=3)
U = UnitInterval(name="U")
P = ProductSpace(components=(R3, U))   # R^3 x (0, 1); also R3 * U
```

## ContinuousMorphism

Given an input `x`, a kernel supplies a distribution over outputs `y`.
A probability density describes mass per unit volume. A uniform density
on an interval of width `0.1`, for instance, is `10`; integrating over
the interval gives `10 * 0.1 = 1`. Its `log_prob` is `log(10)`, a
positive observation score.

For a Normal sample, reparameterization writes `y = mu + sigma * epsilon`
with `epsilon` drawn from a standard Normal. Gradients then pass through
`mu` and `sigma` for the realized noise. Discrete sampling does not generally
admit the same pathwise derivative.


A continuous morphism $f : X \to Y$ between spaces $X$ and $Y$
defines a conditional distribution $p(y \mid x)$ via two operations:

- `log_prob(x, y)`: evaluate $\log p(y \mid x)$.
- `rsample(x)`: generate samples from $p(\cdot \mid x)$, with a
  pathwise differentiable representation when the family supports one.

```python
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
import torch

class MyMorphism(ContinuousMorphism):
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log density log p(y | x)."""
        raise NotImplementedError()

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Reparameterized samples; shape (*sample_shape, batch, codomain.dim)."""
        raise NotImplementedError()
```

Morphisms are
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
subclasses, so they are trainable.

## Composition: SampledComposition

The composite distribution is
$p(z\mid x)=\int p(z\mid y)p(y\mid x)\,dy$. Drawing `y` and then
`z` produces a draw from this distribution. Evaluating its density at a
specified `z` requires integrating over the possible values of `y`, using
an analytic formula or an approximation. A finite intermediate allows
this calculation to use a finite sum.


When composing continuous morphisms, the intermediate is
continuous. The result is computed via ancestral sampling:

```python
import torch
from quivers.continuous.spaces import Euclidean
from quivers.continuous.morphisms import SampledComposition
from quivers.continuous.families import ConditionalNormal

# f: X -> Y (continuous), g: Y -> Z (continuous)
X = Euclidean(name="X", dim=3)
Y = Euclidean(name="Y", dim=4)
Z = Euclidean(name="Z", dim=5)
f = ConditionalNormal(X, Y)
g = ConditionalNormal(Y, Z)

# Composition: (g . f)(x) samples from g(f(x))
composed = f >> g
assert isinstance(composed, SampledComposition)

# rsample: sample from intermediate
x = torch.randn(2, 3)
z_samples = composed.rsample(x, sample_shape=torch.Size((50,)))  # (50, 2, 5)
```

When an intermediate is discrete (a
[`FinSet`](../api/core/objects.md)), composition uses exact
[marginalization](morphisms.md#marginalization):

$$p(z \mid x) = \sum_y p(y \mid x) \, p(z \mid y)$$

## ProductContinuousMorphism

Tensor product of continuous morphisms:

```python
from quivers.continuous.spaces import Euclidean, UnitInterval
from quivers.continuous.morphisms import ProductContinuousMorphism
from quivers.continuous.families import ConditionalNormal, ConditionalBeta

X = Euclidean(name="X", dim=3)
f = ConditionalNormal(X, Euclidean(name="Y1", dim=2))
g = ConditionalBeta(X, UnitInterval(name="Y2"))

# Product takes two inputs: p(y_1, y_2 | x_1, x_2)
# = p_f(y_1 | x_1) * p_g(y_2 | x_2).
fg = f @ g

# Domain and codomain are products
assert fg.domain == f.domain * g.domain   # X x X
assert fg.codomain == f.codomain * g.codomain  # R^2 x (0,1)
```

## Discrete / continuous boundary

### Embed: embed a discrete domain into a continuous codomain

Embed a finite-set value into a continuous space:

```python
import torch
from quivers.continuous.spaces import Euclidean
from quivers.continuous.boundaries import Embed
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

X = FinSet(name="X", cardinality=4)
Y = FinSet(name="Y", cardinality=5)
discrete_f = morphism(X, Y)

# Give each element of X a learnable Gaussian embedding in R^5
Y_continuous = Euclidean(name="Y", dim=5)
embedded_f = Embed(domain=X, codomain=Y_continuous)

# Now can compose with continuous morphisms
y_cont = torch.randn(4, 5)
log_p = embedded_f.log_prob(torch.arange(4), y_cont)
```

### Discretize: bin a continuous space

Discretize a continuous space into a finite set:

```python
import torch
from quivers.continuous.spaces import Euclidean
from quivers.continuous.boundaries import Discretize

# Discretize a bounded interval into 20 bins
# (Discretize requires bounded Euclidean)
U = Euclidean(name="U", dim=1, low=0.0, high=1.0)
discretized = Discretize(U, n_bins=20)

# Discretize bins the continuous domain into the codomain FinSet.
assert discretized.domain == U
assert discretized.codomain.cardinality == 20
```

### Integration with discrete morphisms

Continuous and discrete morphisms integrate transparently via the
`>>` and `@` operators:

```python
# Discrete to Continuous
discrete_f = morphism(X, Y)  # X -> Y (finite)
cont_g = ConditionalNormal(
    domain=Y,                        # discrete domain, indexed lookup
    codomain=Euclidean(name="Z", dim=3),
)

composed = discrete_f >> cont_g  # exact marginalization over Y

# Continuous to Discrete: not directly composable without an
# explicit Discretize boundary; use Discretize to materialize a
# FinSet target first.
```

## Normalizing flows

Stan’s [multivariate change-of-variables explanation](https://mc-stan.org/docs/stan-users-guide/reparameterization.html#multivariate-changes-of-variables) gives the determinant correction needed when transforming a density. [@stanReparameterization] A flow uses a sequence of such transformations; [variational guides](inference-svi.md#guides-variational-families) can learn them as a posterior approximation.

Normalizing flows combine invertible transformations with density
corrections [@rezende2015flows].

An invertible transformation can stretch or compress regions of a simple
distribution. If `y = 2*z` in one dimension, the transformed density at
`y` is half the original density at `y/2`, because the same mass now
occupies twice the width. The log-determinant Jacobian records this volume
change. Invertibility lets the model find the corresponding base value
and score it by including the Jacobian term below.


A [normalizing
flow](https://en.wikipedia.org/wiki/Flow-based_generative_model)
transforms a simple base distribution via a chain of invertible
transformations.

### AffineCouplingLayer

An affine coupling layer [@dinh2017realnvp] partitions the
input and applies an invertible affine transformation to each
partition:

```python
import torch
from quivers.continuous.flows import AffineCouplingLayer
from quivers.continuous.spaces import Euclidean

dim = 4
domain = Euclidean(name="X", dim=2)
layer = AffineCouplingLayer(domain, dim, hidden_dim=32)

# Forward (base to target), conditioned on x
x = torch.randn(10, 2)
z = torch.randn(10, dim)
z_out, log_det = layer.forward(x, z)

# Inverse (target to base)
z_back, log_det_inv = layer.inverse(x, z_out)
```

### ConditionalFlow

A full normalizing flow conditioned on input:

```python
import torch
from quivers.continuous.flows import ConditionalFlow
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
codomain = Euclidean(name="Y", dim=4)

flow = ConditionalFlow(
    domain=domain,
    codomain=codomain,
    n_layers=6,
    hidden_dim=64,
)

# Sample
x = torch.randn(8, 5)
y_samples = flow.rsample(x, sample_shape=torch.Size((50,)))

# Log probability
y = torch.randn(8, 4)
log_p = flow.log_prob(x, y)
```

The [`quivers.continuous.flows`](../api/continuous/flows.md) module
exposes two public building blocks:
[`AffineCouplingLayer`](../api/continuous/flows.md#quivers.continuous.flows.AffineCouplingLayer),
a single invertible affine coupling layer, and
[`ConditionalFlow`](../api/continuous/flows.md#quivers.continuous.flows.ConditionalFlow),
a stack of coupling layers wrapped as a `ContinuousMorphism`.

## See also

- [Continuous Families](continuous-families.md): the more than forty
  parameterized distribution families used to build
  `ContinuousMorphism`s and the event-rank table that drives the
  axis-role syntax.
- [Monadic Programs](programs.md): how `ContinuousMorphism`s
  compose into probabilistic programs.
- [Stochastic Morphisms](stochastic.md): the finite-state
  counterpart and the
  [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) construction.

## Further reading

- [@durkan2019splines]
- [@kingma2016iaf]
- [@papamakarios2017maf]
- [@dinh2017realnvp]
