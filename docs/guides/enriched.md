# Enriched Category Theory

The `quivers.enriched` package provides finite tensor operations motivated by $\mathcal{V}$-enriched categories: ends and coends, weighted limits, profunctors, Yoneda, Kan extensions, Day convolution, and optics. Every operation is parameterized by an [algebra](core.md#algebras-enrichment-algebras) $\mathcal{V}$ (defaulting to [`PRODUCT_FUZZY`](../api/core/algebras.md)); the implementation contracts and joins/meets against the algebra's primitive operations.

## Ends and coends

The general enriched constructions, beyond the finite reductions here,
are treated in [@kelly1982basic].

Start with a square table whose two axes refer to the same choices. Matching
those choices means keeping the diagonal, a special case of
[tensor contraction](dsl-contractions.md#type-driven-wiring-inference). The finite `end` operation then
combines the diagonal entries with `meet`; `coend` uses `join`. Under Boolean
AND/OR these read as "all matched cases" and "some matched case."
*Contravariant* and *covariant* mark the two argument directions;
$\mathcal C^{\mathrm{op}}$ reverses the arrows of $\mathcal C$.

For the matrix below the selected values are `0.9`, `0.8`, and `0.7`.
The default product meet gives `0.504`; noisy-OR gives `0.994`.
The implementation performs this diagonal reduction, without separately
enforcing the dinaturality conditions of a general non-discrete category.
Use the general terminology only under hypotheses that justify that reduction.

An [end](https://ncatlab.org/nlab/show/end) $\int_X F(X, X)$ is the universal dinatural wedge for a bifunctor $F : \mathcal{C}^{\mathrm{op}} \times \mathcal{C} \to \mathcal{V}$. Dually, a [coend](https://ncatlab.org/nlab/show/coend) $\int^X F(X, X)$ is the universal dinatural cowedge. In a general category these constructions impose compatibility along morphisms, via an equalizer or coequalizer when the required products or coproducts exist. In the finite tensor representation used by quivers, an end is computed by restricting a tensor to its diagonal in matched contravariant/covariant axes and reducing with the algebra's meet ($\bigwedge$); a coend reduces with the join ($\bigvee$).

The functions [`end`](../api/enriched/ends_coends.md) and [`coend`](../api/enriched/ends_coends.md) consume a tensor representing $F$ and the index tuples of its contravariant / covariant occurrences:

```python
import torch
from quivers.enriched.ends_coends import end, coend
from quivers.core.algebras import PRODUCT_FUZZY

# F: C^op × C → V with C of size 3, so F is shape (3, 3).
F = torch.tensor([
    [0.9, 0.4, 0.1],
    [0.5, 0.8, 0.2],
    [0.3, 0.6, 0.7],
])

# Coend ∫^X F(X, X) joins along the diagonal pair (axis 0, axis 1).
ce = coend(F, contra_dims=(0,), co_dims=(1,), algebra=PRODUCT_FUZZY)

# End ∫_X F(X, X) meets along the same diagonal.
e = end(F, contra_dims=(0,), co_dims=(1,), algebra=PRODUCT_FUZZY)
```

## Yoneda

The Yoneda idea is to describe an object by all the ways other objects can
map into it. That collection is its *representable presheaf*
$y_A = \mathcal C(-,A)$. A presheaf is a [functor](categorical.md#functors) that assigns information to objects and
pulls it backward along arrows. A transformation from $y_A$ to `F` is
determined by what it does to the identity arrow at `A`: all other
components follow by [naturality](categorical.md#natural-transformations). Conversely, an element of `F(A)` supplies
those components. This explains the correspondence in the theorem below [@kelly1982basic].

In the finite discrete case, the identity relation acts like a lookup
selector: only the matching index contributes. With non-identity arrows,
the presheaf’s action along those arrows and the transformation’s naturality
equations ensure the same correspondence.

For a $\mathcal{V}$-presheaf $F : \mathcal{C}^{\mathrm{op}} \to \mathcal{V}$ and an object $A \in \mathcal{C}$, the [Yoneda lemma](https://ncatlab.org/nlab/show/Yoneda+lemma) asserts $[\mathcal{C}^{\mathrm{op}}, \mathcal{V}](y_A, F) \cong F(A)$, where $y_A = \mathcal{C}(-, A)$ is the representable presheaf at $A$. [`yoneda_lemma`](../api/enriched/yoneda.md) computes a finite internal-hom/meet expression. Tests compare selected instances numerically; arbitrary tensors need not define a presheaf satisfying the theorem's hypotheses:

```python
import torch
from quivers.enriched.yoneda import (
    yoneda_lemma, yoneda_embedding, verify_yoneda_fully_faithful, Presheaf,
)
from quivers.core.objects import FinSet

X1 = FinSet(name="X1", cardinality=2)
X2 = FinSet(name="X2", cardinality=3)

# Presheaf F: C^op → V; values at each object.
F = Presheaf(
    objects=(X1, X2),
    values={0: torch.tensor([0.6, 0.2]), 1: torch.tensor([0.1, 0.5, 0.9])},
)

hom_tensors = [
    torch.eye(2),                    # C(X1, X1)
    torch.zeros(3, 2),               # C(X2, X1)
]
fa = yoneda_lemma(presheaf=F, obj_index=0, hom_tensors=hom_tensors)
```

The [Yoneda embedding](../api/enriched/yoneda.md) $y : \mathcal{C} \to [\mathcal{C}^{\mathrm{op}}, \mathcal{V}]$ sends a morphism $f : A \to B$ to the profunctor view $y(f)$. [`yoneda_embedding`](../api/enriched/yoneda.md) takes a `Morphism` and returns a [`Profunctor`](../api/enriched/profunctors.md):

<!-- python: skip -->
```python
y_f = yoneda_embedding(f)  # f : Morphism
```

[`verify_yoneda_fully_faithful`](../api/enriched/yoneda.md) checks that the embedding preserves composition by comparing $y(g) \circ y(f)$ to $y(g \circ f)$ pointwise:

<!-- python: skip -->
```python
is_ff = verify_yoneda_fully_faithful(f, g, atol=1e-5)
```

## Kan extensions

A map of indices can merge several detailed cases into one coarser case.
The *fiber* over a coarse index is the set of detailed indices sent there.
For a map from observations to participants, a fiber contains one
participant's observations. A left extension aggregates each fiber with
`join`; a right extension uses `meet`. Under Boolean operations they answer
"does any observation qualify?" and "do all observations qualify?"
The example uses an inclusion instead: old entries retain their positions,
and previously absent positions receive the relevant empty reduction.
These fiber formulas describe the finite discrete/order-based setting;
general Kan extensions involve the base category's morphisms as well.

[Left and right Kan extensions](https://ncatlab.org/nlab/show/Kan+extension) extend a morphism along an [`ObjectMap`](../api/enriched/kan_extensions.md) between objects. `quivers.enriched.kan_extensions` provides two concrete `ObjectMap` subclasses, [`Projection`](../api/enriched/kan_extensions.md) (projection from a product onto selected factors) and [`Inclusion`](../api/enriched/kan_extensions.md) (the canonical $A_k \hookrightarrow A_1 + \ldots + A_n$):

```python
import torch
from quivers.enriched.kan_extensions import Inclusion, left_kan, right_kan
from quivers.core.objects import FinSet, CoproductSet
from quivers.core.morphisms import observed

A0 = FinSet(name="A0", cardinality=2)
A1 = FinSet(name="A1", cardinality=3)
S = CoproductSet(components=(A0, A1))

iota = Inclusion(coproduct=S, component_index=0)  # A0 ↪ A0 + A1

B = FinSet(name="B", cardinality=4)
R = observed(A0, B, torch.rand(2, 4))  # R: A0 → B

LanR = left_kan(R, along=iota)         # Lan_iota R: S → B
RanR = right_kan(R, along=iota)        # Ran_iota R: S → B
```

[`left_kan`](../api/enriched/kan_extensions.md) computes $(\mathrm{Lan}_p R)(a', b) = \bigvee\{R(a, b) : p(a) = a'\}$ (the algebra join over the fiber); [`right_kan`](../api/enriched/kan_extensions.md) is the meet-dual.

## Weighted limits

A weight specifies how strongly each part of a diagram constrains an
answer. For Boolean values, the internal hom `[w, d]` is implication:
when a requirement `w` is true, `d` must hold; a false requirement imposes
no constraint. Taking the meet combines all these requirements.
In a residuated ordered algebra, the internal hom is defined by
$a \otimes b \le c$ iff $b \le [a,c]$. This adjoint relation, together
with the diagram’s action on morphisms, supplies the weighted-limit
interpretation of the reduction.

A [weighted limit](https://ncatlab.org/nlab/show/weighted+limit) $\{W, D\}$ pairs a [`Weight`](../api/enriched/weighted_limits.md) $W$ (a presheaf valued in $\mathcal{V}$) with a [`Diagram`](../api/enriched/weighted_limits.md) $D$ (a tuple of objects, possibly extended to a true functor in richer settings). For discrete diagrams the value is a meet over the residuated internal-hom of $W$ and $D$:

```python
from quivers.enriched.weighted_limits import (
    Diagram, Weight, weighted_limit, representable_weight,
)
from quivers.core.objects import FinSet

J = FinSet(name="J", cardinality=3)

# Representable weight at index k = 0 (Yoneda-style).
W = representable_weight(index_set=J, represented_at=0)

# Discrete diagram of objects.
A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=4)
C = FinSet(name="C", cardinality=3)
D = Diagram(objects=(A, B, C))

limit_tensor = weighted_limit(W, D)  # torch.Tensor
```

[`weighted_limit`](../api/enriched/weighted_limits.md) returns a `torch.Tensor`; the shape is the product of the diagram-object shapes joined under the weight.

## Profunctors

A profunctor is a relation-like object between two categories. In the
finite discrete case it is a table with one input index and one output
index, much like a morphism already encountered in [Morphisms](morphisms.md).
Composing two tables aggregates over a hidden middle index, which is the
coend formula below. For non-discrete categories, a profunctor also has
compatible actions of the source and target morphisms. Those actions
specify how changing either object along an arrow changes the profunctor
value; the matrix representation covers the discrete case.

A [profunctor](https://ncatlab.org/nlab/show/profunctor) (or distributor) $P : \mathcal{A} \nrightarrow \mathcal{B}$ is a $\mathcal{V}$-valued bimodule, represented in quivers as a [`Profunctor`](../api/enriched/profunctors.md) wrapping a tensor of shape `(*contra.shape, *co.shape)`:

```python
import torch
from quivers.enriched.profunctors import Profunctor
from quivers.core.objects import FinSet

A = FinSet(name="A", cardinality=3)
B = FinSet(name="B", cardinality=4)
data = torch.rand(3, 4)
P = Profunctor(contra=A, co=B, tensor=data)
```

A morphism in $\mathcal{V}\text{-}\mathbf{Rel}$ embeds as a profunctor via `Profunctor.from_morphism(f)`. Composition of profunctors uses the coend formula $(Q \circ P)(c, a) = \int^b P(b, a) \otimes Q(c, b)$ and is provided as `Profunctor.compose`.

## Day convolution

Ordinary convolution combines contributions from every decomposition of an
output index: for sequence lengths, an output length `n` can come from
`i + j = n`. Day convolution replaces addition of lengths by the base
category's monoidal product. For each `C`, enumerate pairs `A, B` whose
product is `C`, combine their scores, and aggregate the alternatives.
The chosen object inventory must contain the decompositions you want to
represent; a truncated inventory omits the others.

[Day convolution](https://ncatlab.org/nlab/show/Day+convolution) lifts the monoidal structure of a base category to its presheaves. For a finite discrete category and presheaf values $F, G$, [`day_convolution`](../api/enriched/day_convolution.md) computes

$$
(F \circledast G)(C) \;=\; \bigvee_{A, B : A \otimes B = C} F(A) \otimes G(B).
$$

```python
import torch
from quivers.enriched.day_convolution import day_convolution, day_unit
from quivers.core.objects import FinSet
from quivers.categorical.monoidal import CartesianMonoidal

objects = (FinSet(name="A", cardinality=2), FinSet(name="B", cardinality=3))
monoidal = CartesianMonoidal()

# F and G are presheaves indexed by `objects`, one value per object.
f_values = torch.tensor([0.6, 0.4])
g_values = torch.tensor([0.5, 0.7])

FG = day_convolution(f_values, g_values, objects=objects, monoidal=monoidal)
unit = day_unit(objects=objects, unit_index=0)
```

## Optics

A lens focuses on one field of a record, such as the color in a
(color, shape) pair. Reading extracts color; updating replaces color while
retaining shape. A prism focuses on one alternative of a tagged union: it
can match a color case, but a shape case does not contain a color to read.
This explains why products and coproducts lead to different optics.
An adapter changes representations in both directions; a grate packages a
more general way to combine access through a family of contexts.

Optics are bidirectional morphisms that decompose into a forward `get` / `match` and a backward `put` / `build`. The package provides four kinds, all subclassing [`Optic`](../api/enriched/optics.md):

- [`Lens`](../api/enriched/optics.md) on a [`ProductSet`](../api/core/objects.md) focuses on one component.
- [`Prism`](../api/enriched/optics.md) on a [`CoproductSet`](../api/core/objects.md) focuses on one case.
- [`Adapter`](../api/enriched/optics.md) is an isomorphism optic given by a `from` / `to` pair of morphisms.
- [`Grate`](../api/enriched/optics.md) cotraverses through a coindexing object.

```python
import torch
from quivers.enriched.optics import (
    Lens, Prism, Adapter, Grate, compose_optics,
)
from quivers.core.objects import FinSet, ProductSet, CoproductSet
from quivers.core.morphisms import observed

A = FinSet(name="A", cardinality=2)
C = FinSet(name="C", cardinality=3)

# Lens on a product S = A × C, focusing on A (index 0).
S = ProductSet(components=(A, C))
lens = Lens(whole=S, focus_index=0)

# Prism on a coproduct T = A + C, focusing on A.
T = CoproductSet(components=(A, C))
prism = Prism(whole=T, focus_index=0)

# Adapter is an isomorphism A ↔ A' specified by two morphisms.
A_prime = FinSet(name="A_prime", cardinality=2)
fwd = observed(A, A_prime, torch.eye(2))
bwd = observed(A_prime, A, torch.eye(2))
adapter = Adapter(from_morph=fwd, to_morph=bwd)

# Grate from S through coindex I to focus A.
I = FinSet(name="I", cardinality=2)
cotraverse = torch.zeros(6, 2)  # (|S|, |A|)
grate = Grate(source=S, focus=A, index=I, cotraverse_tensor=cotraverse)
```

Optics are composed via [`compose_optics`](../api/enriched/optics.md):

```python
combined = compose_optics(outer=lens, inner=adapter)
```

Each optic exposes `.forward()` and `.backward()` returning `Morphism`s
(typically `ObservedMorphism`). Laws depend on the optic: a lens must preserve
the unfocused part when updating, while a prism distinguishes a successful
match from another variant. For a lens into the color of `(color, shape)`,
writing back the color just read must preserve the pair, and reading after
writing a new color must return that new color. A prism needs laws about
which variant matches and how a successful match is rebuilt.
