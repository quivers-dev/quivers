# Stochastic Morphisms

## The FinStoch Category {#the-finstoch-category}

A [Markov kernel](https://en.wikipedia.org/wiki/Markov_kernel) from a finite set $A$
to a finite set $B$ assigns a probability distribution over $B$ to each element of
$A$. It is a [stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix) of
shape $(|A|, |B|)$: nonnegative, with every row summing to one. Given today's
weather, for instance, a row assigns probabilities to tomorrow's weather. A function
$A \to B$ is the special case with a single `1` in each row.

Composing two kernels gives the distribution of the final value after passing through
an intermediate value that is not kept, which is matrix multiplication (the
[Chapman–Kolmogorov equation](https://en.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation)).
Finite sets with Markov kernels, matrix multiplication as composition, and identity
matrices as identities form the [category](https://ncatlab.org/nlab/show/category)
[**FinStoch**](https://ncatlab.org/nlab/show/FinStoch) [@fritz2020synthetic]. The
library's class is `FinStoch`.

```python
from quivers.stochastic import StochasticMorphism, FinStoch
from quivers.core.objects import FinSet
import torch

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# Create a stochastic morphism (Markov kernel)
f = StochasticMorphism(X, Y)

# Check that rows sum to 1
assert (f.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5

# Composition preserves row-stochasticity
g = StochasticMorphism(Y, FinSet(name="Z", cardinality=5))
h = f >> g
assert (h.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5
```

## MarkovAlgebra {#markovalgebra}

Stochastic morphisms compose under the `MARKOV` instance of
[`Algebra`](core.md#algebras-enrichment-algebras), the sum-product semiring
[@goodman1999semiring]: $a \otimes b = a b$, $\bigoplus_i x_i = \sum_i x_i$, unit
$1$, and zero $0$. Composition is then

$$
(g \circ f)[a, c] = \sum_b f[a, b] \, g[b, c],
$$

ordinary matrix multiplication. With the two tensors in the code below, the
probability of output 0 from input 0 is `0.7 * 0.5 + 0.3 * 0.8 = 0.59`. The two terms
concern different intermediate values, so they add. The sum is not idempotent, as
`0.7 + 0.7` is `1.4`, but composed kernels stay in $[0, 1]$, since each output entry
is an average of conditional probabilities weighted by a distribution. `MARKOV`'s
`meet` is the product and its `negate` is $1 - a$.

```python
import torch
from quivers.stochastic import MARKOV

assert MARKOV.name == "Markov"

# Composition uses sum-product (matrix mult)
f_tensor = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
g_tensor = torch.tensor([[0.5, 0.5], [0.8, 0.2]])

composed = MARKOV.compose(f_tensor, g_tensor, n_contract=1)
expected = f_tensor @ g_tensor
assert torch.allclose(composed, expected)
```

## Stochastic Morphisms

A `StochasticMorphism` is a learnable Markov kernel: an unconstrained weight matrix
whose rows pass through [softmax](https://en.wikipedia.org/wiki/Softmax_function).
`CategoricalMorphism` is another name for the same class, since each row is a
categorical distribution.

```python
from quivers.stochastic import StochasticMorphism
from quivers.core.objects import FinSet

X = FinSet(name="X", cardinality=4)
Y = FinSet(name="Y", cardinality=6)

f = StochasticMorphism(X, Y)

# Tensor is softmax applied to raw weights
# f.tensor has shape (4, 6) with rows summing to 1
assert f.tensor.shape == (4, 6)
assert (f.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5

# Gradient descent updates via raw weights
module = f.module()
params = list(module.parameters())
assert len(params) == 1  # one weight matrix
```

## Discretized Distribution Families

A discretized family approximates a continuous distribution on an interval by a
categorical distribution over equal bins. The codomain's cardinality is the number of
bins. `DiscretizedNormal` learns one mean and one log scale per element of its domain
and scores the bin centres between `low` and `high`.

```python
from quivers.stochastic.families import (
    DiscretizedNormal,
    DiscretizedLogitNormal,
    DiscretizedBeta,
    DiscretizedTruncatedNormal,
)
from quivers.core.objects import FinSet

X = FinSet(name="X", cardinality=10)

# Discretized normal: the codomain `FinSet` cardinality is the bin count.
Bins20 = FinSet(name="Bins20", cardinality=20)
normal = DiscretizedNormal(X, Bins20, low=0.0, high=1.0)

# One learnable mean and log scale per element of X
tensor = normal.tensor  # (10, 20) stochastic matrix

# Discretized beta on (0, 1)
Bins15 = FinSet(name="Bins15", cardinality=15)
beta = DiscretizedBeta(X, Bins15)
```

## Conditioning and Bayesian Updates

`condition(f, e)` reweights each row of a kernel $f : A \to B$ by a nonnegative
likelihood vector $e$ over $B$ and renormalizes:

$$
f_{\mid e}(a, b) = \frac{f(a, b)\, e(b)}{\sum_{b'} f(a, b')\, e(b')}.
$$

If a row is `[0.6, 0.4]` and the weights are `[0.2, 0.8]`, the reweighted entries are
`[0.12, 0.32]` and the conditioned row is approximately `[0.273, 0.727]`. A vector of
zeros and ones conditions on an event, and a general vector is likelihood (virtual)
evidence [@pearl1988probabilistic]. A row whose reweighted total is zero has no
conditional distribution.

```python
import torch
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism, condition, ConditionedMorphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# A kernel X → Y
f = StochasticMorphism(X, Y)

# Evidence over Y that rules out y = 1
evidence = torch.tensor([1.0, 0.0, 1.0, 1.0])

# f|e(a, b) ∝ f(a, b) · e(b), row-renormalized
posterior = condition(f, evidence)
assert isinstance(posterior, ConditionedMorphism)
assert (posterior.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5
```

Conditioning reweights the outputs of each row separately. Recovering a distribution
over inputs from an observed output also needs a prior over the inputs. Given a prior
$\pi$ on $A$, the Bayesian inverse of $f$ is the kernel $B \to A$ with

$$
f^{\dagger}_\pi(a \mid b) = \frac{f(b \mid a)\, \pi(a)}{\sum_{a'} f(b \mid a')\, \pi(a')},
$$

the conditional distribution of $a$ given $b$ under the joint $\pi(a) f(b \mid a)$
[@cho2019disintegration]. The transformation
[`bayes_invert`](transformations.md#constructors) computes it:

```python
from quivers.core.morphism_transformations import bayes_invert

prior = torch.tensor([0.5, 0.3, 0.2])
inverse = f.change_base(bayes_invert(prior))   # Y → X
assert inverse.tensor.shape == (4, 3)
```

## Mixture Morphisms

`mix(f, g)` is the convex combination $w f + (1 - w) g$ of two kernels with the same
domain and codomain:

```python
import torch
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism, mix, MixtureMorphism

X = FinSet(name="X", cardinality=4)
Y = FinSet(name="Y", cardinality=6)
f = StochasticMorphism(X, Y)
g = StochasticMorphism(X, Y)

# Mixture with a learnable weight (sigmoid of init_logit=0.0, so initial weight 0.5)
mixture = mix(f, g, learnable=True, init_logit=0.0)
assert isinstance(mixture, MixtureMorphism)

# Initial tensor is 0.5 * f + 0.5 * g
expected = 0.5 * f.tensor + 0.5 * g.tensor
assert torch.allclose(mixture.tensor, expected)
```

## Factored Morphisms

`factor(f, w)` multiplies each column of a kernel by a nonnegative weight over the
codomain, $f_w(a, b) = f(a, b)\, w(b)$, and does not renormalize:

```python
import torch
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism, factor, FactoredMorphism

X = FinSet(name="X", cardinality=4)
Y = FinSet(name="Y", cardinality=6)
f = StochasticMorphism(X, Y)
weights = torch.tensor([1.0, 0.5, 2.0, 1.0, 0.0, 1.0])  # over Y (|Y|=6)

# factor(f, w)(a, b) = f(a, b) · w(b)
factored = factor(f, weights)
assert isinstance(factored, FactoredMorphism)
```

## Normalization

`normalize(f)` rescales each row to sum to one. Applied after `factor`, it gives the
same result as `condition`:

```python
import torch
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism, factor, normalize, NormalizedMorphism

X = FinSet(name="X", cardinality=4)
Y = FinSet(name="Y", cardinality=6)
weights = torch.tensor([1.0, 0.5, 2.0, 1.0, 0.0, 1.0])
factored = factor(StochasticMorphism(X, Y), weights)

# `factored` is a morphism whose tensor need not be row-stochastic
normalized = normalize(factored)
assert isinstance(normalized, NormalizedMorphism)
assert (normalized.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5
```

## Queries: Probability and Expectation

`prob(f, a, b)` reads conditional probabilities $f(b \mid a)$ at index pairs.
`marginal_prob(f, b)` averages the rows of `f`, giving the probability of each listed
output under a uniform distribution over the inputs. `expectation(f, v)` returns
$\sum_b f(b \mid a)\, v(b)$ for each input $a$.

```python
import torch
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism, prob, marginal_prob, expectation

X = FinSet(name="X", cardinality=4)
Y = FinSet(name="Y", cardinality=6)
f = StochasticMorphism(X, Y)

# Query: P(y=2 | x=0)
p = prob(f, torch.tensor([0]), torch.tensor([2]))   # shape (1,)
assert (0 <= p).all() and (p <= 1).all()

# P(y) for every y in Y, under a uniform distribution over X
m = marginal_prob(f, torch.arange(6))                # shape (6,)

# Expectation of a real-valued function over the codomain, per domain element
values = torch.arange(6, dtype=torch.float)
expect = expectation(f, values)                      # shape (|X|,)
```

## The Giry Monad and FinStoch {#the-giry-monad-and-finstoch}

The [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) $\mathcal G$ sends a
measurable space $X$ to the space $\mathcal G(X)$ of probability measures on $X$
[@giry1982categorical]. A [monad](https://ncatlab.org/nlab/show/monad) of this kind
has a *unit* $\eta_X : X \to \mathcal G(X)$, here the map sending $x$ to the point
mass $\delta_x$, and a *multiplication* $\mu_X : \mathcal G(\mathcal G(X)) \to
\mathcal G(X)$, here the map sending a distribution over distributions to its mixture.
Choosing equally between Bernoulli distributions with success probabilities `0.2` and
`0.8`, for instance, gives success probability `0.5`. A Markov kernel $A \to B$ is a
map $A \to \mathcal G(B)$, an arrow of the
[Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of $\mathcal G$,
and Kleisli composition integrates over the intermediate value. On finite sets the
finite-support analogue, the [distribution monad](https://ncatlab.org/nlab/show/distribution+monad)
$\mathcal D$, gives Kleisli arrows that are exactly stochastic matrices, and Kleisli
composition is matrix multiplication. FinStoch is the Kleisli category of
$\mathcal D$ restricted to finite sets.

The class `GiryMonad` represents these Kleisli arrows by their matrices and never
builds $\mathcal D(X)$ as a separate object. `unit(X)` returns the identity matrix on
`X`, which is the kernel $x \mapsto \delta_x$. `kleisli_compose(f, g)` and
`FinStoch().compose(f, g)` return the matrix product. `join(X)` also returns the
identity matrix on `X`, because in this representation the mixing step of $\mu$ is
already part of the matrix product, so it does not compute the multiplication on a
space of distributions over distributions.

```python
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism
from quivers.stochastic.giry import GiryMonad, FinStoch

giry = GiryMonad()

# The unit, represented as the identity kernel X → X (x ↦ point mass at x)
X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
unit_X = giry.unit(X)

# Kleisli composition of two stochastic morphisms via the monad
f = StochasticMorphism(X, Y)
g = StochasticMorphism(Y, FinSet(name="Z", cardinality=2))
h = giry.kleisli_compose(f, g)

# The same composite through the category class
fs = FinStoch()
h2 = fs.compose(f, g)
```
