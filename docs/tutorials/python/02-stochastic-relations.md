# Tutorial 2: Stochastic Relations

The [FinStoch](https://ncatlab.org/nlab/show/FinStoch) category has finite sets as objects and [Markov kernels](https://en.wikipedia.org/wiki/Markov_kernel) as morphisms. Their entries are [conditional probabilities](../../getting-started/concepts.md#4-a-probabilistic-program-describes-how-data-could-arise). This chapter constructs and composes kernels, conditions them on observations, and queries probabilities and expectations.

Think of `X` as weather, `Y` as a travel choice, and `Z` as arrival status.
A row of the first table gives the probabilities of travel choices for one
weather condition. If a sunny day leads to walking with probability `0.25`
and taking a bus with probability `0.75`, and their on-time probabilities are
`0.8` and `0.6`, then the composed on-time probability is
`0.25 * 0.8 + 0.75 * 0.6 = 0.65`.

The sum accounts for all intermediate choices. Unlike the fuzzy scores in
chapter 1, each kernel row allocates total probability one across its possible
outputs. This is the concrete content of the term *Markov kernel* on finite sets.

Bishop develops conditional probability and marginalization from the
statistical side [@bishop2006pattern]. Fritz gives a categorical treatment
of stochastic maps, including finite stochastic matrices [@fritz2020synthetic].
The calculation above is the finite sum that connects those two accounts.

## Concepts

- **Markov Algebra**: the algebra where the tensor product is multiplication and the join is summation
- **[`StochasticMorphism`](../../api/stochastic/morphisms.md)**: a morphism in FinStoch whose rows are probability distributions (each row sums to 1)
- **Discretized families**: continuous distributions discretized into finite bins
- **Conditioning**: updating a kernel given evidence
- **Probabilities and expectations**: extracting entries and computing expected values

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

The FinStoch category uses the Markov algebra: composition is Markov kernel composition (matrix multiplication), and the join is summation.

Create two finite sets:

```python
X = FinSet(name="Latent", cardinality=3)
Y = FinSet(name="Observed", cardinality=4)
```

Create a latent (learnable) stochastic morphism from X to Y using [`stochastic`](../../api/stochastic/morphisms.md):

```python
kern = stochastic(X, Y)
print(kern.tensor.shape)  # [3, 4]
```

Verify it is row-stochastic (each row sums to 1):

```python
row_sums = kern.tensor.sum(dim=1)
print(row_sums)  # tensor([1., 1., 1.])
```

The entries represent conditional probabilities:

```python
print(kern.tensor)
# Each row: P(y | x) for a fixed x
```

The morphism is learnable; parameters are adjusted via softmax normalization internally.

Create an observed stochastic morphism from Y to a third set Z:

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

Verify the data is valid:

```python
print(kern_obs.tensor.sum(dim=1))  # all 1.0
```

## Composition of Markov Kernels

Compose stochastic morphisms with `>>`:

```python
composed = kern >> kern_obs
print(composed.tensor.shape)  # [3, 2]
print(composed.tensor.sum(dim=1))  # all 1.0
```

The composition is standard Markov kernel composition (matrix multiplication):

$$
(\kappa_2 \circ \kappa_1)(x, z) = \sum_y \kappa_1(x, y) \cdot \kappa_2(y, z)
$$

Wrap in a Program for training:

```python
program = Program(composed)
output = program()
print(output.shape)  # [3, 2]
```

## Discretized distributions

For models mixing discrete and continuous variables, discretize continuous distributions into finite bins.

Create a [`DiscretizedNormal`](../../api/stochastic/families.md): a normal density evaluated at bin centers spanning the interval `[low, high]`, normalized to a probability vector. The mean and log-scale are learnable parameters, one pair per domain element:

```python
Z_bin = FinSet(name="DiscretizedValue", cardinality=10)
Unit = FinSet(name="Unit", cardinality=1)

disc_normal = DiscretizedNormal(Unit, Z_bin, low=0.0, high=1.0)
print(disc_normal.tensor.shape)        # [1, 10]
print(disc_normal.tensor.sum(dim=-1))  # ~[1.0]
```

This is a stochastic morphism from the one-element input object to a finite
grid. Each entry is a normalized weight at a bin center, used as the mass of
that discrete outcome. It is not the exact integral of the continuous density
over the corresponding interval.

Similarly, create a [`DiscretizedBeta`](../../api/stochastic/families.md):

```python
disc_beta = DiscretizedBeta(Unit, Z_bin)
```

These discretized morphisms compose with other stochastic morphisms. For instance, chain a discretized normal into a learnable kernel from `Z_bin` to a coarser output set:

```python
Output = FinSet(name="Output", cardinality=3)
kern_to_output = stochastic(Z_bin, Output)

combined = disc_normal >> kern_to_output
print(combined.tensor.shape)        # [1, 3]
print(combined.tensor.sum(dim=-1))  # ~[1.0]
```

## Observations and Conditioning

Given non-negative evidence over the codomain, [`condition`](../../api/stochastic/transforms.md) reweights each row of the kernel and renormalizes:

$$
f|e(a, b) \;\propto\; f(a, b)\, e(b)
$$

For instance, soft evidence over Y:

```python
evidence = torch.tensor([0.1, 0.4, 0.3, 0.2])

conditioned = condition(kern, evidence)
print(conditioned.tensor.shape)        # [3, 4]
print(conditioned.tensor.sum(dim=-1))  # ~[1, 1, 1]
```

Hard evidence (only Y = 2 is allowed) is just a zero-one mask:

```python
hard = torch.zeros(4)
hard[2] = 1.0
posterior_kernel = condition(kern, hard)
```

For every row with positive evidence mass, the result is a distribution on
`Y` concentrated on the evidence support. A row whose evidence mass is zero
remains all zeros under the implementation's clamped denominator; no conditional
distribution is defined by Bayes' rule in that case.

This operation does not infer `X`. Given a prior `p(x)`, the [input posterior](06-first-class-trans.md#bayes-inversion) is
proportional to `p(x) * sum_y f(x, y) * e(y)`, normalized over `x` when that
total is positive. The row normalization above discards precisely those
row-specific evidence totals. Use the original kernel and prior for Bayes
inversion; chapter 6 develops that separate operation.

## Probabilities and Expectations

[`prob`](../../api/stochastic/queries.md) reads off entries of a kernel at specific `(domain, codomain)` index pairs:

```python
x_idx = torch.tensor([0, 1, 2])
y_idx = torch.tensor([3, 1, 0])
print(prob(kern, x_idx, y_idx))  # shape [3]
```

[`marginal_prob`](../../api/stochastic/queries.md) marginalizes a kernel over its domain under a uniform prior, returning the codomain marginal at the requested indices:

```python
y_idx = torch.tensor([0, 1, 2, 3])
print(marginal_prob(kern, y_idx))  # shape [4], sums to 1
```

[`expectation`](../../api/stochastic/queries.md) computes the conditional expectation of a real-valued function on the codomain, for each domain element:

```python
values = torch.tensor([10.0, 5.0, 1.0, 0.5])
print(expectation(kern, values))  # shape [3]: E[v | x]
```

## Multiple stages

Build a longer chain:

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

## Summary

You have:

- Created stochastic morphisms (Markov kernels)
- Verified row-stochasticity
- Composed kernels with `>>`
- Used `DiscretizedNormal` and `DiscretizedBeta` to mix discrete and continuous randomness
- Applied conditioning to update kernels given evidence
- Computed entries, marginals, and expectations
- Built multi-stage inference chains

Next, learn how to work with continuous spaces and probabilistic programs in [Tutorial 3](03-probabilistic-programs.md).
