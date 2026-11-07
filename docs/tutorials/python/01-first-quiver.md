# Tutorial 1: Your First Quiver

Begin with two tables: one relates positions to colors, and another relates
colors to outcomes. We want to connect them into a table from positions to
outcomes. Quivers calls each [named set](../../getting-started/concepts.md#1-an-object-names-the-possible-values) an *object* and each typed relation a
*morphism*. This chapter builds those relations as tensors and learns their
entries with PyTorch. The default algebra combines [fuzzy membership values](04-fuzzy-factorization.md#background)
using multiplication and noisy-OR. These operations have different
[algebraic properties](../../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine)
from ordinary matrix multiplication, as the numerical example below shows.

If you're coming from Stan, PyMC, or Pyro: the Python API is the lower-level interface that the [QVR DSL](../qvr/01-first-model.md) compiles to. You write models as compositions of typed tensors instead of as `program` blocks. Use this track when you want to build new categorical constructs, write libraries on top of quivers, or read the type errors emitted by the DSL compiler. For day-to-day modeling, the DSL track is the right entry point.

The example uses three finite sets and two tables. Their typed signatures identify the shared axis that `>>` contracts.

## Concepts

- **Objects**: finite sets ([`FinSet`](../../api/core/objects.md))
- **Morphisms**: $\mathcal{V}$-relations represented as tensors
- **Latent morphisms**: parameters (learnable)
- **Observed morphisms**: fixed tensors
- **Composition**: tensor contraction according to the algebra's operations

## Setup

```python
import torch
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism, observed
from quivers.core.algebras import PRODUCT_FUZZY
from quivers.program import Program
```

## Creating objects

Create three finite sets: one for positions (X), one for colors (Y), and one for outcomes (Z):

```python
X = FinSet(name="Position", cardinality=3)
Y = FinSet(name="Color", cardinality=4)
Z = FinSet(name="Outcome", cardinality=2)

print(X.size)   # 3
print(Y.size)   # 4
print(Z.size)   # 2
```

Each object has a shape and size. These define the dimensions of the tensors representing morphisms.

## Creating morphisms

### Latent morphism

A latent morphism has learnable tensor entries. Create one from X to Y using [`morphism`](../../api/core/morphisms.md):

```python
f = morphism(X, Y)
print(f.domain.name)     # Position
print(f.codomain.name)   # Color
print(f.tensor)          # shape [3, 4], values in (0, 1)
print(f.tensor.shape)
```

The tensor is initialized with a sigmoid activation, so all entries lie in $(0, 1)$. The entries are PyTorch parameters: they will be updated during backpropagation if you include the morphism in a training loop.

Inspect the underlying module:

```python
mod = f.module()
params = list(mod.parameters())
print(len(params))  # 1
print(params[0].shape)  # torch.Size([3, 4])
```

### Observed morphism

An observed morphism has a fixed, non-learnable tensor. Create one from Y to Z with explicit data using [`observed`](../../api/core/morphisms.md):

```python
data = torch.tensor([
    [1.0,  0.1],
    [0.8,  0.2],
    [0.5,  0.5],
    [0.0,  0.9],
])
g = observed(Y, Z, data)
print(g.tensor)
```

The tensor shape must match $(|Y|, |Z|) = (4, 2)$. If you pass the wrong shape, an error is raised.

Verify it has no learnable parameters:

```python
mod_g = g.module()
params_g = list(mod_g.parameters())
print(len(params_g))  # 0
```

## Composition

Compose two morphisms using the `>>` operator. Composition applies the algebra's operations: the tensor product $\otimes$ (multiplication) and the join $\bigvee$ (noisy-OR).

Create a third latent morphism from Z to a new object:

```python
W = FinSet(name="Result", cardinality=5)
h = morphism(Z, W)
```

Compose f and g: X -> Y -> Z

```python
fg = f >> g
print(fg.domain.name)    # Position
print(fg.codomain.name)  # Outcome
print(fg.tensor.shape)   # [3, 2]
```

Composition is lazy: the tensor is not computed until evaluation. Verify this is a [`ComposedMorphism`](../../api/core/morphisms.md):

```python
from quivers.core.morphisms import ComposedMorphism
print(isinstance(fg, ComposedMorphism))  # True
```

Compose again: (X -> Y -> Z) -> W

```python
fgh = fg >> h
print(fgh.tensor.shape)  # [3, 5]
```

The composition operation uses the algebra's tensor product (here, pointwise multiplication) and join (noisy-OR):

$$
(g \circ f)(x, z) = \bigvee_y f(x, y) \otimes g(y, z)
$$

In PRODUCT_FUZZY, $\otimes$ is multiplication and $\bigvee$ is $1 - \prod_i (1 - x_i)$.

To read the equation, fix one position `x` and outcome `z`. For each color
`y`, multiply the strength of the position-to-color relation by the strength
of the color-to-outcome relation. Then combine those alternatives. If two
routes contribute `0.4` and `0.3`, noisy-OR gives `1 - 0.6 * 0.7 = 0.58`.
A third route of strength zero changes nothing; a route of strength one makes
the combined value one.

Each outcome receives a fuzzy membership score independently, so a row can
sum to more than one. Product with noisy-OR can also make bracketing affect
the result. Keep the written order of composition when extending this example
to three or more tables.

## Accessing tensor values

Once composed, access the materialized tensor:

```python
tensor_value = fg.tensor
print(tensor_value.dtype)  # torch.float32
print(tensor_value.min(), tensor_value.max())  # check range
```

For a relation in PRODUCT_FUZZY, values should lie in $[0, 1]$.

## Working as a Differentiable Module

PyTorch records the operations on parameters with `requires_grad=True`.
Calling [`.backward()`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients)
on the loss accumulates its derivatives in each parameter's `.grad` field
[@pytorchAutogradTutorial]. The optimizer uses those derivatives to update
the morphism entries.

Wrap a morphism in a [`Program`](../../api/program.md) to make it a differentiable `nn.Module`:

```python
prog = Program(f)
output = prog()
print(output.shape)  # [3, 4]
print(output)
```

Integrate with PyTorch training:

```python
import torch.optim as optim

program = Program(fgh)  # a complex composition
optimizer = optim.Adam(program.parameters(), lr=0.01)

for epoch in range(5):
    optimizer.zero_grad()

    # Forward pass
    result = program()

    # Define a loss (e.g., sum for illustration)
    loss = result.sum()

    # Backward and step
    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

The parameters of all latent morphisms in the composition will be updated.

## Algebras

The examples so far use [`PRODUCT_FUZZY`](../../api/core/algebras.md) as the enrichment:

```python
print(PRODUCT_FUZZY.name)  # "ProductFuzzy"
```

Other algebras are available ([`BOOLEAN`](../../api/core/algebras.md), [`LUKASIEWICZ`](../../api/core/algebras.md), [`GODEL`](../../api/core/algebras.md)). The choice of algebra affects:

1. How morphisms are initialized
2. How they compose
3. The semantics of the resulting values

The examples use `PRODUCT_FUZZY` because their entries denote fuzzy membership and compose by product and noisy-OR.

## Summary

You have:

- Created finite sets (objects)
- Constructed latent (learnable) and observed (fixed) morphisms
- Composed morphisms with `>>`
- Inspected tensor shapes and values
- Wrapped morphisms as differentiable modules
- Learned that composition uses the algebra's operations (tensor product and join)

Next, explore how these ideas extend to stochastic morphisms (Markov kernels) in [Tutorial 2](02-stochastic-relations.md).
