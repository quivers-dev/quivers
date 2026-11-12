# Continuous Spaces and Morphisms

A finite object lists possible indices, and a continuous space lists possible
numerical values: a point of `Real 3` has three real coordinates, such as
`(0.2, -1.0, 4.5)`. A space also records constraints, which define the *support* on
which a distribution may put mass: the entries of a simplex point are nonnegative and
sum to one, and a positive real stays above zero. Continuous morphisms are conditional
distributions between such spaces. [Continuous Families](continuous-families.md) lists
the registered distribution families.

## ContinuousSpace hierarchy {#continuousspace-hierarchy}

Each space is a [measurable space](https://en.wikipedia.org/wiki/Measurable_space), a
set with the collection of subsets to which probabilities are assigned, and densities
on the Euclidean spaces are taken with respect to
[Lebesgue measure](https://en.wikipedia.org/wiki/Lebesgue_measure), ordinary length,
area, or volume. The spaces are subclasses of `ContinuousSpace`:

| Class | Points |
|---|---|
| `Euclidean` | $\mathbb R^n$, optionally bounded by `low` and `high` |
| `Simplex` | $\{x \in \mathbb R^n : x \ge 0, \sum_i x_i = 1\}$, probability vectors |
| `PositiveReals` | $(0, \infty)^n$ |
| `CholeskyFactor` | $K \times K$ lower-triangular Cholesky factors of correlation matrices, with rows of unit norm |
| `ProductSpace` | Cartesian products of spaces and finite objects |
| `Sphere` | the unit sphere $S^{N-1}$ in $\mathbb R^N$ |
| `Ball` | the closed ball of radius $r$ in $\mathbb R^N$ |
| `Covariance` | $D \times D$ symmetric positive-definite matrices |
| `Correlation` | $D \times D$ correlation matrices, positive definite with unit diagonal |
| `Orthogonal` | the [orthogonal group](https://en.wikipedia.org/wiki/Orthogonal_group) $O(D)$ of $D \times D$ orthogonal matrices |
| `Stiefel` | the [Stiefel manifold](https://en.wikipedia.org/wiki/Stiefel_manifold) $V_K(\mathbb R^N)$ of $N \times K$ matrices with orthonormal columns |
| `LowerTriangular` | $D \times D$ lower-triangular matrices |
| `Diagonal` | $D \times D$ diagonal matrices, identified with $\mathbb R^D$ |

`UnitInterval(name, dim=1)` is a function that returns the bounded `Euclidean` space
$[0, 1]^{\mathrm{dim}}$, not a separate class.

### Euclidean space

```python
from quivers.continuous.spaces import Euclidean

R3 = Euclidean(name="R3", dim=3)                                  # R^3
R2_bounded = Euclidean(name="R2", dim=2, low=0.0, high=1.0)        # [0,1]^2
```

### Unit interval

```python
from quivers.continuous.spaces import UnitInterval

U = UnitInterval(name="U")     # [0, 1]; pass dim=k for [0, 1]^k
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
P = ProductSpace(components=(R3, U))   # R^3 x [0, 1]; also R3 * U
```

## ContinuousMorphism {#continuousmorphism}

A continuous morphism $f : X \to Y$ gives a conditional distribution $p(y \mid x)$, a
[Markov kernel](https://en.wikipedia.org/wiki/Markov_kernel), through two operations:

- `log_prob(x, y)` evaluates the log density $\log p(y \mid x)$;
- `rsample(x)` draws from $p(\cdot \mid x)$, as a differentiable function of fixed
  noise when the family allows.

A density is probability per unit volume and can exceed one: a uniform density on an
interval of width `0.1` is `10`, and its `log_prob` is `log(10)`. A draw from a normal
written as $y = \mu + \sigma \varepsilon$, with $\varepsilon$ standard normal, lets
gradients pass through $\mu$ and $\sigma$ for the realized noise, the
[reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick)
[@kingma2014autoencoding]. Discrete draws do not in general have such a pathwise
derivative.

Morphisms are
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
subclasses, so their parameters train with PyTorch. A new morphism subclasses
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
and implements the two operations, as this normal kernel with a fixed scale does:

```python
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
import torch

class ShiftedNormal(ContinuousMorphism):
    """y ~ Normal(x, 0.5), coordinate by coordinate."""

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Normal(x, 0.5).log_prob(y).sum(-1)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        noise = torch.randn(*sample_shape, *x.shape)
        return x + 0.5 * noise    # (*sample_shape, batch, codomain.dim)

R2 = Euclidean(name="R2", dim=2)
k = ShiftedNormal(R2, R2)
draws = k.rsample(torch.zeros(3, 2), torch.Size((10,)))   # (10, 3, 2)
```

## Composition: SampledComposition {#composition-sampledcomposition}

Composing $f : X \to Y$ with $g : Y \to Z$ gives the kernel

$$
p(z \mid x) = \int p(z \mid y)\, p(y \mid x)\, dy,
$$

the [Chapman–Kolmogorov equation](https://en.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation).
Drawing $y$ and then $z$ is a draw from it, but its density at a given $z$ needs the
integral over $y$. When the intermediate is continuous, `f >> g` is a
`SampledComposition`. Its `rsample` draws ancestrally, first from `f` and then from
`g`, and is an exact draw from the composite. Its `log_prob` does not compute the
integral: it scores $z$ along one deterministic path through the intermediate, which
is not the composite density.

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

When the intermediate is a [`FinSet`](../api/core/objects.md), composition sums over
it exactly, $p(z \mid x) = \sum_y p(y \mid x)\, p(z \mid y)$
([marginalization](morphisms.md#marginalization)), and `log_prob` is the composite
density.

## ProductContinuousMorphism

The parallel product `f @ g` of two continuous morphisms takes a pair of inputs and
draws the two outputs independently:

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
assert fg.codomain == f.codomain * g.codomain  # R^2 x [0,1]
```

## Discrete / continuous boundary {#discrete-continuous-boundary}

### Embed: embed a discrete domain into a continuous codomain

`Embed` maps each element $i$ of a finite set to a normal distribution
$\mathcal N(\mu_i, \sigma_i)$ in a continuous space, with a learnable centre and scale
per element:

```python
import torch
from quivers.continuous.spaces import Euclidean
from quivers.continuous.boundaries import Embed
from quivers.core.objects import FinSet

X = FinSet(name="X", cardinality=4)

# A learnable normal distribution in R^5 for each element of X
Y_continuous = Euclidean(name="Y", dim=5)
embedded_f = Embed(domain=X, codomain=Y_continuous)

y_cont = torch.randn(4, 5)
log_p = embedded_f.log_prob(torch.arange(4), y_cont)   # one log density per element
```

### Discretize: bin a continuous space

`Discretize` maps a bounded one-dimensional space to a finite set of equal-width bins:

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

`>>` composes a finite morphism with a continuous one whose domain is that finite set,
summing over the intermediate exactly. The other direction needs an explicit
`Discretize` to produce a finite target first.

```python
from quivers.core.morphisms import morphism

Y = FinSet(name="Y", cardinality=5)

# Discrete to Continuous
discrete_f = morphism(X, Y)  # X -> Y (finite)
cont_g = ConditionalNormal(
    domain=Y,                        # discrete domain, indexed lookup
    codomain=Euclidean(name="Z", dim=3),
)

composed = discrete_f >> cont_g  # exact marginalization over Y
```

## Normalizing flows

A [normalizing flow](https://en.wikipedia.org/wiki/Flow-based_generative_model)
pushes a simple base distribution through a chain of invertible maps
[@rezende2015flows]. A map can stretch or compress regions: if $y = 2z$ in one
dimension, the density at $y$ is half the base density at $y / 2$, since the same mass
occupies twice the width. The log absolute determinant of the map's Jacobian records
this change of volume, and adding it to the base log density gives the flow's density
by the change of variables formula [@stanReparameterization]. A
[flow guide](inference-svi.md#guides-variational-families) learns the maps to
approximate a posterior.

### AffineCouplingLayer

An affine coupling layer [@dinh2017realnvp] splits the coordinates in two, leaves one
part unchanged, and shifts and scales the other by amounts computed from the first
part and the conditioning input, so its Jacobian is triangular and its determinant
cheap:

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

`ConditionalFlow` stacks coupling layers into a conditional distribution:

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

[`quivers.continuous.flows`](../api/continuous/flows.md) exports these two:
[`AffineCouplingLayer`](../api/continuous/flows.md#quivers.continuous.flows.AffineCouplingLayer)
and
[`ConditionalFlow`](../api/continuous/flows.md#quivers.continuous.flows.ConditionalFlow),
which wraps a stack of coupling layers as a `ContinuousMorphism`. The flow guides of
[Variational Inference](inference-svi.md#guides-variational-families) use further
transforms, such as inverse autoregressive and spline flows.

## See also

- [Continuous Families](continuous-families.md): the registered distribution families
  and their numbers of event dimensions.
- [Monadic Programs](programs.md): how continuous morphisms compose into probabilistic
  programs.
- [Stochastic Morphisms](stochastic.md): the finite counterpart and the Giry monad.
