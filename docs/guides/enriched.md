# Enriched Category Theory

In an [enriched category](https://ncatlab.org/nlab/show/enriched+category) the
morphisms from one object to another form an object of a monoidal category
$\mathcal V$ rather than a set [@kelly1982basic]. For a finite morphism with scores in
an [`Algebra`](core.md#algebras-enrichment-algebras), each entry is such a value, and
the constructions of enriched category theory reduce to sums and products of tensor
entries. `quivers.enriched` implements the finite versions of ends and coends, the
Yoneda lemma, Kan extensions, weighted limits, profunctors, Day convolution, and
optics. Each function takes an `Algebra` (default `PRODUCT_FUZZY`) and computes with
its $\otimes$, its reduction $\bigoplus$ (`join`), its reduction $\bigwedge$ (`meet`),
and, where needed, an internal hom.

Two cautions apply throughout. First, the functions reduce over finite index sets
and treat the index categories as discrete, with only identity morphisms, so they
check no naturality or dinaturality condition along other morphisms. Second, the
general theorems hold when $\mathcal V$ is a complete closed monoidal structure such as
a [quantale](https://ncatlab.org/nlab/show/quantale), and most built-in instances are
not ([Core Types](core.md#algebras-enrichment-algebras)).

## Ends and coends

For a functor $F : \mathcal C^{\mathrm{op}} \times \mathcal C \to \mathcal V$, where
$\mathcal C^{\mathrm{op}}$ is $\mathcal C$ with its arrows reversed, the
[end](https://ncatlab.org/nlab/show/end) $\int_X F(X, X)$ and the coend
$\int^X F(X, X)$ are the universal ways of reducing $F$ along its diagonal, compatibly
with every morphism of $\mathcal C$ [@kelly1982basic]. When $\mathcal C$ is a finite
discrete category, the end is the meet of the diagonal entries and the coend is their
join. [`end`](../api/enriched/ends_coends.md) and
[`coend`](../api/enriched/ends_coends.md) compute exactly these reductions: they keep
the diagonal of the matched contravariant and covariant axes and reduce it with
$\bigwedge$ or $\bigoplus$. Under the Boolean semiring the two read as "every matched
case" and "some matched case".

For the matrix below the diagonal entries are `0.9`, `0.8`, and `0.7`. The default
product meet gives `0.504`, and noisy-OR gives `0.994`.

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

A [presheaf](https://ncatlab.org/nlab/show/presheaf) on $\mathcal C$ is a functor
$F : \mathcal C^{\mathrm{op}} \to \mathcal V$. The representable presheaf at $A$ is
$y_A = \mathcal C(-, A)$, which records every way of mapping into $A$. The
[Yoneda lemma](https://ncatlab.org/nlab/show/Yoneda+lemma) states that natural
transformations $y_A \Rightarrow F$ correspond to elements of $F(A)$. In the enriched
form, the object of such transformations is the end
$\int_X [\mathcal C(X, A), F(X)] \cong F(A)$, where $[-, -]$ is the internal hom of
$\mathcal V$ [@kelly1982basic].

[`yoneda_lemma`](../api/enriched/yoneda.md) computes the left side,
$\bigwedge_X \bigwedge_x [\mathcal C(x, a), F(x)]$, from a `Presheaf` and the hom
tensors $\mathcal C(X_i, A)$. It does not check that the inputs form a presheaf, so the
isomorphism holds only for inputs that meet the lemma's hypotheses. The internal hom
is the residuum $[w, x] = \sup\{v : w \otimes v \le x\}$ for `PRODUCT_FUZZY`, where it
equals $\min(1, x / w)$, and for `BOOLEAN`, where it equals $\max(1 - w, x)$. For any
other `Algebra` the function uses $\neg(w \otimes \neg x)$, which is not the residuum
in general.

```python
import torch
from quivers.enriched.yoneda import (
    yoneda_lemma, yoneda_embedding, verify_yoneda_preserves_composition, Presheaf,
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

The [Yoneda embedding](https://ncatlab.org/nlab/show/Yoneda+embedding)
$y : \mathcal C \to [\mathcal C^{\mathrm{op}}, \mathcal V]$ sends an object $A$ to the
representable presheaf $y_A$ and a morphism $f : A \to B$ to the natural
transformation $y_A \Rightarrow y_B$ given by composing with $f$. It is fully faithful:
every natural transformation $y_A \Rightarrow y_B$ comes from exactly one morphism
$A \to B$. In the library, [`yoneda_embedding(f)`](../api/enriched/yoneda.md) returns
$f$'s tensor as a [`Profunctor`](../api/enriched/profunctors.md), the same value as
`Profunctor.from_morphism(f)`.

[`verify_yoneda_preserves_composition(f, g)`](../api/enriched/yoneda.md) compares
`yoneda_embedding(f >> g)` with the profunctor composite of `yoneda_embedding(f)` and
`yoneda_embedding(g)`. This tests one instance of functoriality, preservation of
composition. Despite its name, it tests neither fullness nor faithfulness, which are
statements about bijectivity on hom-sets.

<!-- python: skip -->
```python
y_f = yoneda_embedding(f)                          # f : Morphism
preserves = verify_yoneda_preserves_composition(f, g, atol=1e-5)
```

## Kan extensions

A function $p : A \to A'$ between finite sets groups the elements of $A$ by their
image: the [fiber](programs-hierarchical.md#grouped-marginalization-fibred-discrete-latents) of $p$ over $a'$ is the set
$p^{-1}(a')$ of elements sent to $a'$. When $A$ and $A'$ are discrete categories, the
left [Kan extension](https://ncatlab.org/nlab/show/Kan+extension) of
$R : A \to B$ along $p$ aggregates each fiber with $\bigoplus$, and the right Kan
extension with $\bigwedge$ [@maclane1998categories]:

$$
(\mathrm{Lan}_p R)(a', b) = \bigoplus_{a \in p^{-1}(a')} R(a, b), \qquad
(\mathrm{Ran}_p R)(a', b) = \bigwedge_{a \in p^{-1}(a')} R(a, b).
$$

For a map from observations to participants, each fiber holds one participant's
observations, and under Boolean operations the two extensions ask whether any, or
every, observation qualifies. An empty fiber gives the zero of $\bigoplus$ on the left
and the unit of $\bigwedge$ on the right. General Kan extensions also involve the
morphisms of the categories, which these functions do not model.

[`left_kan`](../api/enriched/kan_extensions.md) and
[`right_kan`](../api/enriched/kan_extensions.md) take $p$ as an
[`ObjectMap`](../api/enriched/kan_extensions.md). Two are provided:
[`Projection`](../api/enriched/kan_extensions.md), from a product onto chosen factors,
and [`Inclusion`](../api/enriched/kan_extensions.md), of one component
$A_k \hookrightarrow A_1 + \dots + A_n$ of a coproduct. Both functions read $R$'s tensor
without gradients and return a fixed `ObservedMorphism`.

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

Along an inclusion each fiber has at most one element, so both extensions copy $R$ onto
the `A0` rows and fill the `A1` rows with the empty reductions.

## Weighted limits

A [weighted limit](https://ncatlab.org/nlab/show/weighted+limit) $\{W, D\}$ of a
diagram $D$ by a weight $W$ generalizes a limit by letting $W(j)$ say how strongly the
$j$-th part of the diagram constrains the result [@kelly1982basic]. It is built from
the internal hom: in a residuated structure, $[a, c]$ is the largest $b$ with
$a \otimes b \le c$, and for Boolean values $[w, d]$ is implication, so a weight of 0
imposes no constraint.

For a discrete diagram, with no morphisms between its objects, the weighted limit is
the product of the cotensors of each $D(j)$ by $W(j)$.
[`weighted_limit`](../api/enriched/weighted_limits.md) represents each cotensor by
applying $[W(j), -]$ entrywise to the identity tensor of $D(j)$, with the internal hom
described under [Yoneda](#yoneda), and combines the results across $j$ by the outer
$\otimes$-product. The result for objects $A$, $B$, $C$ thus has shape
`(*A.shape, *A.shape, *B.shape, *B.shape, *C.shape, *C.shape)`, here `(2, 2, 4, 4, 3, 3)`.
A [`Weight`](../api/enriched/weighted_limits.md) holds one value per index, and a
[`Diagram`](../api/enriched/weighted_limits.md) holds the tuple of objects.

```python
from quivers.enriched.weighted_limits import (
    Diagram, Weight, weighted_limit, representable_weight,
)
from quivers.core.objects import FinSet

J = FinSet(name="J", cardinality=3)

# Representable weight at index 0: values (1, 0, 0)
W = representable_weight(index_set=J, represented_at=0)

# Discrete diagram of objects.
A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=4)
C = FinSet(name="C", cardinality=3)
D = Diagram(objects=(A, B, C))

limit_tensor = weighted_limit(W, D)  # shape (2, 2, 4, 4, 3, 3)
```

## Profunctors

A [profunctor](https://ncatlab.org/nlab/show/profunctor)
$P : \mathcal A \nrightarrow \mathcal B$ is a functor
$\mathcal B^{\mathrm{op}} \times \mathcal A \to \mathcal V$, a relation between two
categories that is compatible with the morphisms of both. Profunctors compose by a
coend over the middle category, $(Q \circ P)(c, a) = \int^b P(b, a) \otimes Q(c, b)$.
Between discrete categories the compatibility conditions are empty, a profunctor is a
table with one input index and one output index, and composition is the matrix
product of [Morphisms](morphisms.md#composition-the-operator).

[`Profunctor`](../api/enriched/profunctors.md) wraps a tensor of shape
`(*contra.shape, *co.shape)`. `Profunctor.from_morphism(f)` wraps a morphism's
tensor, and `Profunctor.compose` computes the coend formula.

```python
import torch
from quivers.enriched.profunctors import Profunctor
from quivers.core.objects import FinSet

A = FinSet(name="A", cardinality=3)
B = FinSet(name="B", cardinality=4)
data = torch.rand(3, 4)
P = Profunctor(contra=A, co=B, tensor=data)
```

## Day convolution

Ordinary convolution sums contributions from every decomposition of an output index:
a sequence of length $n$ can come from lengths $i + j = n$.
[Day convolution](https://ncatlab.org/nlab/show/Day+convolution) replaces addition of
lengths by the monoidal product of a base category and so lifts that product to
presheaves [@kelly1982basic]. For a finite discrete category and presheaf values
$F$, $G$, [`day_convolution`](../api/enriched/day_convolution.md) computes

$$
(F \circledast G)(C) = \bigoplus_{A, B \,:\, A \otimes B = C} F(A) \otimes G(B),
$$

where $A$, $B$, and $C$ range over a given tuple of objects. A decomposition counts
only when the product of two listed objects is itself listed, so the tuple must
include the products of interest. In the example the only such product is `A * B`,
and the result is `0.6 * 0.7 = 0.42` at that index and zero elsewhere.

```python
import torch
from quivers.enriched.day_convolution import day_convolution, day_unit
from quivers.core.objects import FinSet
from quivers.categorical.monoidal import CartesianProductMonoidal

A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
objects = (A, B, A * B)
monoidal = CartesianProductMonoidal()

# F and G are presheaves indexed by `objects`, one value per object.
f_values = torch.tensor([0.6, 0.4, 0.1])
g_values = torch.tensor([0.5, 0.7, 0.2])

FG = day_convolution(f_values, g_values, objects=objects, monoidal=monoidal)
# tensor([0.0000, 0.0000, 0.4200])
unit = day_unit(objects=objects, unit_index=0)
```

## Optics

A [lens](https://ncatlab.org/nlab/show/lens+%28in+computer+science%29) reads one
component of a product and writes it back, such as the color in a `(color, shape)`
pair: reading returns the color, and writing replaces it while keeping the shape. A
prism does the same for one case of a disjoint union: it matches a color case and
can build one, and a shape case has no color to read. Lenses and prisms are kinds of
optic, bidirectional accessors with a forward and a backward part. The package
provides four kinds, all subclasses of [`Optic`](../api/enriched/optics.md):

- [`Lens`](../api/enriched/optics.md) focuses on one component of a
  [`ProductSet`](../api/core/objects.md);
- [`Prism`](../api/enriched/optics.md) focuses on one case of a
  [`CoproductSet`](../api/core/objects.md);
- [`Adapter`](../api/enriched/optics.md) is a pair of morphisms meant to be mutually
  inverse, with `verify_isomorphism` to check this;
- [`Grate`](../api/enriched/optics.md) is given by a tensor representing a map from
  functions $I \to A$ into the whole $S$, for an index object $I$.

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

[`compose_optics`](../api/enriched/optics.md) composes two optics:

```python
combined = compose_optics(outer=lens, inner=adapter)
```

Each optic's `.forward()` and `.backward()` return morphisms. A lens with read map
$\mathrm{get}$ and write map $\mathrm{put}$ is expected to satisfy the lens laws:
$\mathrm{put}(s, \mathrm{get}(s)) = s$ (writing back what was read changes nothing)
and $\mathrm{get}(\mathrm{put}(s, a)) = a$ (reading after writing returns what was
written). A prism is expected to satisfy the analogous laws for matching and building.
The library provides no check of either set of laws.
