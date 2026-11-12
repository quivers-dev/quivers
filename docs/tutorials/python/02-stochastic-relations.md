# 2. Stochastic relations

A [Markov kernel](https://en.wikipedia.org/wiki/Markov_kernel) between finite sets $X$ and $Y$ is a [row-stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix): row $x$ is the conditional distribution $p(y \mid x)$. Kernels compose by the matrix product under sum and product, which sums over the intermediate value:

$$
(\kappa_1 \mathbin{>\!\!>} \kappa_2)(x, z) = \sum_{y} \kappa_1(x, y)\, \kappa_2(y, z).
$$

Finite sets with Markov kernels form the category [FinStoch](https://ncatlab.org/nlab/show/FinStoch) [@fritz2020synthetic]. This chapter builds kernels with the `MARKOV` algebra, which is the sum-product semiring on row-stochastic matrices, composes them, discretizes continuous families onto a grid, conditions a kernel on evidence about its output, and queries probabilities and expectations [@bishop2006pattern].

## Setup

```python
import torch
from quivers.core.objects import FinSet
from quivers.core.morphisms import identity, observed
from quivers.stochastic import (
    MARKOV,
    stochastic,
    DiscretizedNormal,
    DiscretizedBeta,
    condition,
    prob,
    marginal_prob,
    expectation,
)
from quivers.program import Program
```

## Creating stochastic morphisms

```python
X = FinSet(name="Latent", cardinality=3)
Y = FinSet(name="Observed", cardinality=4)
```

[`stochastic`](../../api/stochastic/morphisms.md) creates a learnable kernel, a [`StochasticMorphism`](../../api/stochastic/morphisms.md):

```python
kern = stochastic(X, Y)
print(kern.tensor.shape)  # [3, 4]
```

Its parameters are unconstrained logits, and each row of the tensor is their softmax (with the `temperature` argument, default 1), so every row sums to one:

```python
row_sums = kern.tensor.sum(dim=1)
print(row_sums)  # tensor([1., 1., 1.])
```

Row $x$ holds $p(y \mid x)$:

```python
print(kern.tensor)
# Each row: P(y | x) for a fixed x
```

A kernel with fixed entries is an `observed` morphism under `MARKOV`:

```python
Z = FinSet(name="Output", cardinality=2)
data = torch.tensor([
    [0.7, 0.3],
    [0.4, 0.6],
    [0.5, 0.5],
    [0.2, 0.8],
])
kern_obs = observed(Y, Z, data, algebra=MARKOV)
```

```python
print(kern_obs.tensor.sum(dim=1))  # all 1.0
```

## Composition of Markov kernels

```python
composed = kern >> kern_obs
print(composed.tensor.shape)  # [3, 2]
print(composed.tensor.sum(dim=1))  # all 1.0
```

The composite is again row-stochastic. A [`Program`](../../api/program.md) wraps it for training:

```python
program = Program(composed)
output = program()
print(output.shape)  # [3, 2]
```

## Discretized distributions

A model with discrete and continuous variables can discretize a continuous family onto a finite grid. [`DiscretizedNormal`](../../api/stochastic/families.md) evaluates a Normal density at the centers of equal bins spanning `[low, high]` and normalizes the values to a probability vector. Its mean and log scale are learnable, one pair for each element of the domain:

```python
Z_bin = FinSet(name="DiscretizedValue", cardinality=10)
Unit = FinSet(name="Unit", cardinality=1)

disc_normal = DiscretizedNormal(Unit, Z_bin, low=0.0, high=1.0)
print(disc_normal.tensor.shape)        # [1, 10]
print(disc_normal.tensor.sum(dim=-1))  # ~[1.0]
```

This is a kernel from a set with one element to the grid. The mass of each bin is the normalized density at its center, not the integral of the density over the bin. [`DiscretizedBeta`](../../api/stochastic/families.md) does the same for a Beta density:

```python
disc_beta = DiscretizedBeta(Unit, Z_bin)
```

Discretized families compose with other kernels, for instance with a learnable kernel from the grid to a coarser set:

```python
Output = FinSet(name="Output", cardinality=3)
kern_to_output = stochastic(Z_bin, Output)

combined = disc_normal >> kern_to_output
print(combined.tensor.shape)        # [1, 3]
print(combined.tensor.sum(dim=-1))  # ~[1.0]
```

## Observations and conditioning

Given nonnegative evidence $e$ over the codomain, [`condition`](../../api/stochastic/transforms.md) reweights each row of the kernel and renormalizes it:

$$
f|_e(a, b) \;\propto\; f(a, b)\, e(b).
$$

Evidence given as a likelihood vector over the values of $Y$, rather than as an observed value, is *virtual* (likelihood) evidence [@pearl1988probabilistic]:

```python
evidence = torch.tensor([0.1, 0.4, 0.3, 0.2])

conditioned = condition(kern, evidence)
print(conditioned.tensor.shape)        # [3, 4]
print(conditioned.tensor.sum(dim=-1))  # ~[1, 1, 1]
```

An observed value $Y = 2$ is the vector that is one at index 2 and zero elsewhere:

```python
hard = torch.zeros(4)
hard[2] = 1.0
posterior_kernel = condition(kern, hard)
```

Each row with positive evidence mass becomes a distribution on $Y$ concentrated on the support of the evidence. A row with zero evidence mass has no conditional distribution, and the implementation, which clamps the denominator, leaves it all zeros.

Conditioning the output does not update a distribution over the input $X$. With a prior $p(x)$, the posterior over the input is proportional to $p(x) \sum_y f(x, y)\, e(y)$, and the row normalization above discards exactly these row totals. [Bayes inversion](06-first-class-trans.md#bayes-inversion) in chapter 6 computes the posterior over the input from the original kernel and the prior.

## Probabilities and expectations

[`prob`](../../api/stochastic/queries.md) reads entries of a kernel at pairs of domain and codomain indices:

```python
x_idx = torch.tensor([0, 1, 2])
y_idx = torch.tensor([3, 1, 0])
print(prob(kern, x_idx, y_idx))  # shape [3]
```

[`marginal_prob`](../../api/stochastic/queries.md) computes $p(y) = \frac{1}{|X|} \sum_x f(x, y)$, the marginal of the codomain under a **uniform** distribution on the domain, at the requested indices. It takes no prior argument. For another prior, compose a kernel from a set with one element that holds the prior with the kernel, and read the row of the composite.

```python
y_idx = torch.tensor([0, 1, 2, 3])
print(marginal_prob(kern, y_idx))  # shape [4], sums to 1
```

[`expectation`](../../api/stochastic/queries.md) computes $\mathbb E[v(Y) \mid X = x]$ of a real function $v$ on the codomain, for each element $x$ of the domain:

```python
values = torch.tensor([10.0, 5.0, 1.0, 0.5])
print(expectation(kern, values))  # shape [3]: E[v | x]
```

## Multiple stages

```python
# X -> Y -> Z
kern1 = stochastic(X, Y)
kern2 = stochastic(Y, Z)
composed = kern1 >> kern2

print(composed.tensor.shape)        # [3, 2]
print(composed.tensor.sum(dim=-1))  # ~[1, 1, 1]

# Marginal of Z under uniform X
z_idx = torch.tensor([0, 1])
print(marginal_prob(composed, z_idx))  # shape [2], sums to 1
```

Since sum and product form a semiring, composition under `MARKOV` is associative, unlike the `PRODUCT_FUZZY` composition of [chapter 1](01-first-quiver.md#composition), and a chain of kernels may be bracketed either way.

## Next

[Chapter 3](03-probabilistic-programs.md) builds probabilistic programs over continuous spaces.
