# Morphisms & Composition

## What is a Morphism?

In the finite relation representation, `f : A -> B` is a typed table: the domain `A` names its input
choices and the codomain `B` its output choices. An entry `f[a, b]` answers
a question about that pair, such as whether a relation holds, how compatible
it is, or how likely `b` is given `a`. The selected [algebra](core.md#algebras-enrichment-algebras) determines which
question the numbers answer. A Boolean row can relate one input to several
outputs; a probability row assigns those outputs weights that sum to one.


In Quivers' finite relation representation, a morphism from domain $A$ to codomain $B$ is a tensor $M \in \mathcal{V}^{|A| \times |B|}$, where $\mathcal{V}$ is the selected composition algebra. Concretely, it is a multi-dimensional tensor with shape `(*A.shape, *B.shape)` whose entries combine through that algebra's operations.

For a simple [`FinSet`](core.md#finset) morphism $f: X \to Y$ with $|X| = m$ and $|Y| = n$, the tensor is an $m \times n$ matrix.

For a [product domain](core.md#productset-and-coproductset) like $X \times Y$, the tensor has shape `(|X|, |Y|, |Z|)`.

## The Morphism Hierarchy

```
Morphism (abstract)
├── ObservedMorphism      , fixed tensor, not learnable
├── LatentMorphism        , learnable parameter with sigmoid output
├── ComposedMorphism      , lazy composition f >> g
├── ProductMorphism       , lazy tensor product f @ g
├── MarginalizedMorphism  , lazy marginalization (join reduction)
├── TransformedMorphism   , lazy tensor transform (e.g., dagger)
├── RepeatMorphism        , independent product f^⊗n
├── CurriedMorphism       , currying of a tensor-product domain
└── FunctorMorphism       , lazy functor image
```

### ObservedMorphism

A fixed, non-learnable morphism. Use `observed()` to construct:

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

Here *latent* means a learnable table whose entries are optimized.
A sigmoid bounds each entry separately: three zero raw weights become
`[0.5, 0.5, 0.5]`, whose sum is `1.5`.
For random draws, use [program sample sites](programs.md#bind-steps); for normalized
finite kernels, use [stochastic morphisms](stochastic.md#stochastic-morphisms).


A learnable morphism parameterized by a weight matrix, with sigmoid output to ensure values in $(0, 1)$. Construct with `morphism()`:

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

f = morphism(X, Y)
print(f.tensor.shape)  # (3, 4)
print((f.tensor > 0).all() and (f.tensor < 1).all())  # True
```

Access the underlying parameter via `f.raw`:

```python
params = list(f.module().parameters())
assert len(params) == 1
assert torch.allclose(f.tensor, torch.sigmoid(f.raw))
```

Initialize with a specific scale:

```python
# Manual initialization of f.raw before calling f.tensor
f = morphism(X, Y)
with torch.no_grad():
    f.raw.fill_(0.0)  # modify f.raw in place
```

## Composition: The >>  Operator

To connect `A -> B` to `B -> C`, consider every possible middle value `b`.
Combine the two link scores for that value, then aggregate the alternatives.
The middle axis disappears because the result reports only the external
input `a` and output `c`. Source syntax reads left to right (`f >> g`);
mathematical composition reads right to left ($g \circ f$).

For one output entry, suppose the two path scores are `0.8 * 0.5 = 0.4`
and `0.2 * 0.5 = 0.1`. Under the default noisy-OR rule the entry is
`1 - (1 - 0.4) * (1 - 0.1) = 0.46`. Under Markov sum-product it is
`0.4 + 0.1 = 0.5`. Ordinary matrix multiplication is thus correct only
for rules that use sum and product. See [Core Types](core.md) for the
law assumptions and the default rule's associativity limitation.


Composition of two morphisms uses the algebra's operations. If $f: A \to B$ and $g: B \to C$, then $g \circ f: A \to C$ is computed as:

$$(g \circ f)[a, c] = \bigvee_b f[a, b] \otimes g[b, c]$$

where $\bigvee$ is the algebra's join and $\otimes$ is its tensor operation.

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

The composition is lazy: the tensor is not materialized until accessed. Chain compositions:

```python
W = FinSet(name="W", cardinality=5)
k = morphism(Z, W)
pipeline = f >> g >> k
assert pipeline.tensor.shape == (3, 5)
```

Compositions must have compatible algebras:

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

## Tensor Product: The @ Operator

Sequential composition feeds an output into the next operation. Parallel
composition takes two inputs, runs the two operations, and returns both
outputs. Thus `f @ g` expects a pair `(a, c)`, even when `A` and `C` have
the same type; sharing one input requires an explicit copying/fan operation.
The symbol $\otimes$ names both this operation on morphisms and the
entrywise combining operation of the algebra; their types distinguish them.


The tensor (or parallel) product $f \otimes g$ combines two morphisms $f: A \to B$ and $g: C \to D$ into a morphism $f \otimes g: A \times C \to B \times D$. The tensor is the outer product via the algebra's $\otimes$:

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

The @ operator works even if domains or codomains already have products; ProductSet automatically flattens:

```python
P = A * B
Q = C * D

f_prod = morphism(P, Q)  # (A × B) → (C × D)
```

## Marginalization

If a result tracks both color and shape but only color matters to the
query, aggregate over shape. A color-by-shape table becomes one value
per color.
For a probability table the aggregation is a sum; for a Boolean relation
it asks whether any shape works; for fuzzy scores it uses the selected
soft aggregation. Thus the same axis reduction answers a different
question for each algebra.


Join-reduce the codomain over specified components:

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

The tensor is computed by applying the algebra's join operation over the codomain dimensions corresponding to $Z$.

## Operations Summary

| Operation | Syntax | Input | Output |
|-----------|--------|-------|--------|
| Composition | `f >> g` | $f: A \to B$, $g: B \to C$ | $g \circ f: A \to C$ |
| Tensor | `f @ g` | $f: A \to B$, $g: C \to D$ | $f \otimes g: A \times C \to B \times D$ |
| Marginalize | `f.marginalize(A)` | $f: X \to Y \times A$ | $f: X \to Y$ (join over $A$ in codomain) |
| Identity | `identity(X)` | object $X$ | $\text{id}_X: X \to X$ |

## Learning and Gradients

All morphisms expose an `nn.Module` tree via `.module()`:

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

The gradient flows through composition and tensor operations, so you can train entire pipelines end-to-end.
