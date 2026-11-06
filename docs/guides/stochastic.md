# Stochastic Morphisms

## The FinStoch Category

A stochastic map assigns a probability distribution to each possible input.
For instance, given today's weather, a row might assign probabilities to
tomorrow's weather. An ordinary deterministic map is the special case with
one `1` per row and zeros elsewhere. [Composition](morphisms.md#composition-the-operator) asks for the distribution
after passing through an intermediate state whose value is not retained.


**FinStoch** is the category of finite sets and stochastic maps (Markov kernels):

- **Objects**: [finite sets](core.md#finset) (as in the base category)
- **Morphisms** $A \to B$: stochastic matrices of shape $(|A|, |B|)$ with rows summing to 1
- **Composition**: standard matrix multiplication
- **Identity**: Kronecker delta

FinStoch is the [Kleisli category](monadic.md#kleisli-category) of the Giry monad, restricted to finite sets.

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

## MarkovAlgebra

Using the two tables below, the chance of output zero from input zero is
`0.7 * 0.5 + 0.3 * 0.8 = 0.59`. The two terms concern mutually exclusive
intermediate states, so they add. Row sums remain one because each second
stage row is a complete conditional distribution.

Here the API operation `join` computes ordinary sums. For example,
`join(0.7, 0.7)` is `1.4`, whereas their lattice supremum is `0.7`.
The row-normalization constraints make composed kernels stay in `[0,1]`:
each output entry is a weighted average of conditional probabilities.


The enrichment for FinStoch uses the Markov algebra:

$$
\begin{align}
\mathcal{L} &= [0, 1] \\
a \otimes b &= a \cdot b \quad \text{(pointwise product)} \\
\bigvee_i x_i &= \sum_i x_i \quad \text{(sum)} \\
\bigwedge_i x_i &= \min_i x_i \\
I &= 1, \quad \perp = 0
\end{align}
$$

Composition of stochastic matrices uses this structure:

$$(g \circ f)[a, c] = \sum_b f[a, b] \cdot g[b, c]$$

which is standard matrix multiplication.

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

A stochastic morphism is a learnable Markov kernel parameterized by an unconstrained weight matrix and a [softmax normalization](transformations.md#first-class-transformations):

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

An alias `CategoricalMorphism` is provided with explicit categorical semantics.

## Discretized Distribution Families

For approximating continuous-valued distributions with finite tables, Quivers provides discretized distribution families. These map a finite input set to a bucketed approximation of the distribution.

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

# Input-conditioned parameters via learned linear maps
# μ, σ computed from input
tensor = normal.tensor  # (10, 20) stochastic matrix

# Discretized beta on (0, 1)
Bins15 = FinSet(name="Bins15", cardinality=15)
beta = DiscretizedBeta(X, Bins15)
```

## Conditioning and Bayesian Updates

Evidence reweights possibilities and then normalizes what remains. If a row
is `[0.6, 0.4]` and evidence weights are `[0.2, 0.8]`, its weighted entries
are `[0.12, 0.32]`, and its normalized posterior row is approximately
`[0.273, 0.727]`. This operation updates outputs conditional on each input;
recovering a distribution over inputs from observed outputs additionally
requires a prior over those inputs. Zero total evidence has no ordinary
Bayesian normalization and must be treated as a support problem.


Bayesian conditioning via Bayes' rule. Given a joint distribution $p(x, y)$ and observation $\text{obs}(y)$, compute $p(x | \text{obs}(y))$:

$$p(x | \text{obs}) \propto p(x, \text{obs}) = \sum_y p(x, y) \cdot \mathbb{1}_{\text{obs}}(y)$$

```python
import torch
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism, condition, ConditionedMorphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# Prior: X → Y
prior = StochasticMorphism(X, Y)

# Evidence: a likelihood vector over Y (e.g., a soft observation "Y is unlikely to be 1")
evidence = torch.tensor([1.0, 0.0, 1.0, 1.0])

# Posterior: f|e(a, b) ∝ f(a, b) · e(b), row-renormalized
posterior = condition(prior, evidence)
assert isinstance(posterior, ConditionedMorphism)
assert (posterior.tensor.sum(dim=-1) - 1.0).abs().max() < 1e-5
```

## Mixture Morphisms

Convex combination of stochastic morphisms:

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

Pointwise reweighting by a non-negative weight vector over the codomain:

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

# Note: result is unnormalized; use normalize() after
```

## Normalization

Renormalize rows to sum to 1:

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

Extract probabilities from stochastic morphisms:

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

# Marginal P(y) under a uniform prior over X
m = marginal_prob(f, torch.tensor([0, 1, 2, 3, 4, 5]))  # |Y|=6

# Expectation of a real-valued function over the codomain, per domain element
values = torch.arange(6, dtype=torch.float)
expect = expectation(f, values)                      # shape (|X|,)
```

## The Giry Monad and FinStoch

For the measure-theoretic construction see [@giry1982categorical]; for
probability kernels in a broader categorical account see [@fritz2020synthetic].

The unit sends an outcome to certainty at that outcome, called a point
mass. Flattening a distribution of distributions forms their mixture.
For instance, choosing equally between Bernoulli probabilities `0.2` and
`0.8` gives success probability `0.5`. The Giry monad states these
operations for probability measures on measurable spaces; finite tables
represent its kernels without enumerating the whole space of measures.


The Giry monad $\mathcal{G}$ sends a space $X$ to its space of probability measures. On finite sets, this becomes:

$$\mathcal{G}(X) = \{\text{probability distributions on } X\}$$

The Kleisli category of $\mathcal{G}$ restricted to finite sets **is** FinStoch.

```python
from quivers.core.objects import FinSet
from quivers.stochastic import StochasticMorphism
from quivers.stochastic.giry import GiryMonad, FinStoch

giry = GiryMonad()

# Unit η_X: X → X is the Kronecker delta (each element to its point mass)
X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
unit_X = giry.unit(X)

# Kleisli composition of two stochastic morphisms via the monad
f = StochasticMorphism(X, Y)
g = StochasticMorphism(Y, FinSet(name="Z", cardinality=2))
h = giry.kleisli_compose(f, g)

# FinStoch is the Kleisli category of the Giry monad
fs = FinStoch()
h2 = fs.compose(f, g)
```

Matrix multiplication is precisely Kleisli composition for finite probability
kernels: it multiplies probabilities along a route and sums over intermediate
outcomes. These are two descriptions of the same operation.
