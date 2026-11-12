# Morphisms & Composition

## What is a Morphism?

A morphism `f : A -> B` between finite objects is a typed table. The domain `A` names
its input choices, the codomain `B` names its output choices, and the entry `f[a, b]`
scores the pair: whether a relation holds, how compatible the two are, or how likely
`b` is given `a`. The morphism's [`Algebra`](core.md#algebras-enrichment-algebras)
fixes which of these the numbers mean. A Boolean row can relate one input to several
outputs, while a row of a probability kernel gives the outputs weights that sum to
one.

The library stores a morphism as a tensor of shape `(*A.shape, *B.shape)` with
entries in the value set $\mathcal V$ of its `Algebra`. For
[`FinSet`](core.md#finset) objects with $|A| = m$ and $|B| = n$, this is an
$m \times n$ matrix. A [product](core.md#productset-and-coproductset) domain
contributes one axis per component, so a morphism $X \times Y \to Z$ has shape
`(|X|, |Y|, |Z|)`.

## The Morphism Hierarchy

| Class | What it holds | Built by |
|---|---|---|
| `ObservedMorphism` | a fixed tensor with no learnable parameters | `observed(A, B, data)`, `identity(A)` |
| `LatentMorphism` | a learnable table: an unconstrained parameter passed through a sigmoid | `morphism(A, B)` |
| `ComposedMorphism` | the composite of two morphisms, computed on access | `f >> g` |
| `ProductMorphism` | the parallel product of two morphisms, computed on access | `f @ g` |
| `MarginalizedMorphism` | a morphism with codomain components reduced away | `f.marginalize(Z)` |
| `TransformedMorphism` | a fixed rearrangement of another morphism's tensor | `f.dagger`, `f.trace(A)` |
| `RepeatMorphism` | the $n$-fold composite $f \gg \dots \gg f$ of an endomorphism, by repeated squaring, with $n$ settable at run time | [`repeat`](dsl-declarations.md#repeat-iterated-composition) |
| `CurriedMorphism` | the same tensor as `f : X * Y -> Z`, read as a morphism into a categorial (slash) type | [`curry`](dsl-declarations.md) |
| `FunctorMorphism` | the image of a morphism under a [`Functor`](categorical.md#functors) | `functor(f)` |

All of them are subclasses of the abstract `Morphism`. Composites, products,
marginals, and transforms keep a reference to their inputs and recompute the tensor
on each access, so gradients reach the parameters of every learnable input.

### ObservedMorphism

`observed()` wraps a fixed tensor:

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
import torch

torch.manual_seed(0)

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
data = torch.tensor([
    [0.9, 0.1, 0.0, 0.0],
    [0.2, 0.6, 0.1, 0.1],
    [0.0, 0.0, 0.8, 0.2],
])

f = observed(X, Y, data)
assert f.tensor.shape == (3, 4)
assert list(f.module().parameters()) == []  # no learnable params
```

### LatentMorphism

`morphism()` builds a learnable table. Each entry is the sigmoid of its own raw
weight, so every entry lies in $(0, 1)$ and a row need not sum to one: three zero raw
weights give `[0.5, 0.5, 0.5]`, whose sum is `1.5`. The class name says "latent" in
the sense of "learned", not in the sense of a latent random variable. Random draws
belong in a [program's sample steps](programs.md#bind-steps), and learnable rows that
sum to one are [stochastic morphisms](stochastic.md#stochastic-morphisms).

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

f = morphism(X, Y)
print(f.tensor.shape)  # (3, 4)
print((f.tensor > 0).all() and (f.tensor < 1).all())  # True
```

The raw parameter is `f.raw`:

```python
params = list(f.module().parameters())
assert len(params) == 1
assert torch.allclose(f.tensor, torch.sigmoid(f.raw))
```

It can be set in place before the tensor is read:

```python
f = morphism(X, Y)
with torch.no_grad():
    f.raw.fill_(0.0)  # modify f.raw in place
```

## Composition: The >> Operator {#composition-the-operator}

To connect `f : A -> B` to `g : B -> C`, composition considers every middle value
`b`, combines the two link scores with $\otimes$, and reduces over the alternatives
with $\bigoplus$:

$$
(g \circ f)[a, c] = \bigoplus_b f[a, b] \otimes g[b, c].
$$

The middle axis disappears, since the result reports only the input `a` and the output
`c`. Source syntax reads left to right (`f >> g`), and mathematical notation right to
left ($g \circ f$).

The choice of `Algebra` changes the number. Suppose the two paths into one output
entry score `0.8 * 0.5 = 0.4` and `0.2 * 0.5 = 0.1`. Under the default noisy-OR
reduction the entry is `1 - (1 - 0.4) * (1 - 0.1) = 0.46`, and under the
sum-product `MARKOV` instance it is `0.4 + 0.1 = 0.5`. Composition is ordinary matrix
multiplication only for sum and product, and it is associative only when $\otimes$
and $\bigoplus$ form a semiring ([Core Types](core.md#algebras-enrichment-algebras)).

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)

f = morphism(X, Y)
g = morphism(Y, Z)

# Compose with >>
h = f >> g
assert h.domain == X
assert h.codomain == Z
assert h.tensor.shape == (3, 2)
```

The tensor of a composite is computed when it is read, and composites chain:

```python
W = FinSet(name="W", cardinality=5)
k = morphism(Z, W)
pipeline = f >> g >> k
assert pipeline.tensor.shape == (3, 5)
```

Both morphisms must have the same `Algebra`:

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism
from quivers.core.algebras import BOOLEAN, GodelAlgebra

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)

f_bool = morphism(X, Y, algebra=BOOLEAN)
g_godel = morphism(Y, Z, algebra=GodelAlgebra())

# Raises TypeError: incompatible algebras
try:
    _ = f_bool >> g_godel
except TypeError as e:
    print(e)
```

## Tensor Product: The @ Operator {#tensor-product-the-operator}

The parallel product of $f : A \to B$ and $g : C \to D$ is the morphism
$f \otimes g : A \times C \to B \times D$ that runs both on a pair of inputs and
returns both outputs. Its tensor is the outer product of the two tensors under the
`Algebra`'s $\otimes$, which is why one symbol names both operations. `f @ g` thus
expects a pair `(a, c)` even when `A` and `C` are the same object, and using one input
twice requires an explicit copy ([`fan`](dsl-declarations.md#fan-out-diagonal-morphism)).

```python
A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
C = FinSet(name="C", cardinality=4)
D = FinSet(name="D", cardinality=5)

f = morphism(A, B)
g = morphism(C, D)

# Tensor product
h = f @ g
assert h.domain == A * C
assert h.codomain == B * D
# Axes are grouped as (domain..., codomain...) = (|A|, |C|, |B|, |D|)
assert h.tensor.shape == (2, 4, 3, 5)
```

Domains and codomains that are already products flatten as usual:

```python
P = A * B
Q = C * D

f_prod = morphism(P, Q)  # (A × B) → (C × D)
```

## Marginalization

`f.marginalize(Z)` reduces the codomain axes of `Z` with the `Algebra`'s
$\bigoplus$. A table over color and shape becomes one value per color. For a
probability kernel the reduction is a sum, giving the
[marginal distribution](https://en.wikipedia.org/wiki/Marginal_distribution). For a
Boolean relation it asks whether any shape works, and for fuzzy scores it is
noisy-OR.

```python
X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=5)

f = morphism(X, Y * Z)  # X → Y × Z

# Marginalize over Z (sum/join over Z dimension)
g = f.marginalize(Z)
assert g.domain == X
assert g.codomain == Y
```

## Transpose, Trace, Cup, and Cap

Four further operations require the module's composition rule to be an `Algebra`
([levels](transformations.md#the-composition-rule-hierarchy)):

- `f.dagger` swaps the domain and codomain axes of `f : A -> B`, giving `B -> A`. It
  is a transpose for every `Algebra` and does not renormalize, so the transpose of a
  probability kernel is not its Bayesian inverse ([`bayes_invert`](transformations.md#constructors)
  computes that).
- `f.trace(A)` takes `f : A * X -> A * Y` and returns `X -> Y` with entries
  $\bigoplus_a f[(a, x), (a, y)]$, the [trace](https://ncatlab.org/nlab/show/traced+monoidal+category)
  [@joyal1996traced]. `A` must be the first component of both the domain and the
  codomain.
- `cup(A)` is the morphism `1 -> A * A` and `cap(A)` the morphism `A * A -> 1` whose
  tensors are the identity tensor of `A`: the unit on the diagonal and the zero
  elsewhere.

With these, every object is its own dual in the sense of a
[compact closed category](https://ncatlab.org/nlab/show/compact+closed+category)
[@kelly1980coherence], and `cup`, `cap`, and `trace` satisfy the expected equations
whenever the `Algebra` is a commutative semiring, so that composition is associative
and unital.

## Operations Summary

| Operation | Syntax | Input | Output |
|-----------|--------|-------|--------|
| Composition | `f >> g` | $f: A \to B$, $g: B \to C$ | $g \circ f: A \to C$ |
| Parallel product | `f @ g` | $f: A \to B$, $g: C \to D$ | $f \otimes g: A \times C \to B \times D$ |
| Marginalization | `f.marginalize(Z)` | $f: X \to Y \times Z$ | $X \to Y$, $\bigoplus$ over $Z$ |
| Identity | `identity(X)` | object $X$ | $\text{id}_X: X \to X$, unit on the diagonal and zero elsewhere |
| Transpose | `f.dagger` | $f: A \to B$ | $B \to A$ |
| Trace | `f.trace(A)` | $f: A \times X \to A \times Y$ | $X \to Y$ |

## Learning and Gradients

Every morphism exposes its parameters as a PyTorch module through `.module()`:

```python
f = morphism(X, Y)
module = f.module()

# Collect parameters for optimization
optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

# Compute loss and backpropagate
loss = f.tensor.sum()
loss.backward()
optimizer.step()
```

Gradients flow through composition, products, and the other derived morphisms, so a
pipeline of morphisms trains as one model.
