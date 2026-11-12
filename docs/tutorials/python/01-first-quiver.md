# 1. Your first quiver

The Python track builds models from the classes that QVR compiles to: the compiler checks a module in its core language, the [Quivers Indexed Effect Core](../../reference/qvr/index.md) (QIEC), and builds its compiled runtime, the PyTorch modules that `loads` returns, from [`FinSet`](../../api/core/objects.md), [`Morphism`](../../api/core/morphisms.md), and the other classes of this track. Use it to write new constructs, build libraries on quivers, or inspect what a QVR module compiles to. To write and fit a statistical model, start with the [QVR track](../qvr/01-first-model.md).

An *object* is a named finite set, and a *morphism* `f : A -> B` is a matrix of weights of shape `(|A|, |B|)`. Composing `f : X -> Y` with `g : Y -> Z` gives a morphism `X -> Z` by a matrix product over two operations on the weights, a product $\otimes$ and a sum $\oplus$:

$$
(f \mathbin{>\!\!>} g)(x, z) = \bigoplus_{y \in Y} f(x, y) \otimes g(y, z).
$$

The default operations, `PRODUCT_FUZZY`, treat each weight as a degree of membership in $[0, 1]$ in the sense of [fuzzy sets](https://en.wikipedia.org/wiki/Fuzzy_set): $\otimes$ is the product and $\oplus$ the probabilistic sum $1 - \prod_i (1 - x_i)$, the noisy-OR of the alternatives [@klement2000triangular; @pearl1988probabilistic]. The example relates positions to colors and colors to outcomes, and learns the entries of one of the two matrices with PyTorch.

## Setup

```python
import torch
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism, observed
from quivers.core.algebras import PRODUCT_FUZZY
from quivers.program import Program
```

## Creating objects

Create three finite sets: positions `X`, colors `Y`, and outcomes `Z`.

```python
X = FinSet(name="Position", cardinality=3)
Y = FinSet(name="Color", cardinality=4)
Z = FinSet(name="Outcome", cardinality=2)

print(X.size)   # 3
print(Y.size)   # 4
print(Z.size)   # 2
```

An object's `size` is its number of elements, and its `shape`, here `(3,)`, gives the axes it contributes to a tensor.

## Creating morphisms

### Learnable morphism

[`morphism`](../../api/core/morphisms.md) creates a morphism whose entries are learned:

```python
f = morphism(X, Y)
print(f.domain.name)     # Position
print(f.codomain.name)   # Color
print(f.tensor)          # shape [3, 4], values in (0, 1)
print(f.tensor.shape)
```

The PyTorch parameter is an unconstrained tensor of logits, and `f.tensor` is its sigmoid, so every entry lies in $(0, 1)$. Training updates the logits.

```python
mod = f.module()
params = list(mod.parameters())
print(len(params))  # 1
print(params[0].shape)  # torch.Size([3, 4])
```

### Observed morphism

[`observed`](../../api/core/morphisms.md) creates a morphism with fixed entries:

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

The data must have shape $(|Y|, |Z|) = (4, 2)$. Data of shape `(2, 4)` raises `ValueError: data shape torch.Size([2, 4]) does not match expected (4, 2)`. An observed morphism has no parameters:

```python
mod_g = g.module()
params_g = list(mod_g.parameters())
print(len(params_g))  # 0
```

## Composition

`>>` composes two morphisms whose codomain and domain agree. Add a learnable morphism out of `Z`:

```python
W = FinSet(name="Result", cardinality=5)
h = morphism(Z, W)
```

Compose `f` and `g`, `X -> Y -> Z`:

```python
fg = f >> g
print(fg.domain.name)    # Position
print(fg.codomain.name)  # Outcome
print(fg.tensor.shape)   # [3, 2]
```

Composition is lazy: the result is a [`ComposedMorphism`](../../api/core/morphisms.md), whose tensor is computed when it is read.

```python
from quivers.core.morphisms import ComposedMorphism
print(isinstance(fg, ComposedMorphism))  # True
```

Compose again, `X -> Y -> Z -> W`:

```python
fgh = fg >> h
print(fgh.tensor.shape)  # [3, 5]
```

For a fixed position `x` and outcome `z`, each color `y` gives a route of strength $f(x, y)\, g(y, z)$, and noisy-OR combines the routes: two routes of strengths `0.4` and `0.3` give `1 - 0.6 * 0.7 = 0.58`. Each outcome gets its own degree of membership, so a row may sum to more than one.

The product does not distribute over the probabilistic sum, so `PRODUCT_FUZZY` is not a [semiring](https://en.wikipedia.org/wiki/Semiring), and composition under it is not associative: the two bracketings of a composite of three morphisms differ.

```python
left = ((f >> g) >> h).tensor
right = (f >> (g >> h)).tensor
print(float((left - right).abs().max()))
```

With random initial weights the largest entrywise difference is typically near 0.1, so a composite of three or more morphisms under `PRODUCT_FUZZY` must keep its written bracketing.

## Accessing tensor values

```python
tensor_value = fg.tensor
print(tensor_value.dtype)  # torch.float32
print(tensor_value.min(), tensor_value.max())  # check range
```

Under `PRODUCT_FUZZY` every entry lies in $[0, 1]$.

## Working as a differentiable module

A [`Program`](../../api/program.md) wraps a morphism as a PyTorch `nn.Module` whose forward pass returns the morphism's tensor:

```python
prog = Program(f)
output = prog()
print(output.shape)  # [3, 4]
print(output)
```

Train it like any module. [`.backward()`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients) accumulates the gradient of the loss in each parameter [@pytorchAutogradTutorial], and the optimizer updates the logits of every learnable morphism in the composite:

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

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Other algebras

The operations of a morphism are held by an [`Algebra`](../../api/core/algebras.md) object:

```python
print(PRODUCT_FUZZY.name)  # "ProductFuzzy"
```

Others include `BOOLEAN` (AND and OR), `LUKASIEWICZ` (Łukasiewicz t-norm with bounded sum), and `GODEL` (minimum and maximum). The algebra fixes how a learnable morphism's entries are constrained, how morphisms compose, and what the entries mean. [Chapter 7 of the QVR track](../qvr/07-categorical.md#composition-rules-and-their-laws) lists every built-in algebra with its standard name and its laws.

## Next

[Chapter 2](02-stochastic-relations.md) composes Markov kernels: row-stochastic matrices under sum and product.
