# Enriched Category Theory

In an [enriched category](https://ncatlab.org/nlab/show/enriched+category) the
morphisms from one object to another form an object of a monoidal category
$\mathcal V$ rather than a set [@kelly1982basic]. For a finite morphism with scores in
an [`Algebra`](core.md#algebras-enrichment-algebras), each entry is such a value, and
the constructions of enriched category theory reduce to joins, meets, and products of
tensor entries. `quivers.enriched` implements the finite versions of ends and coends, the
Yoneda lemma, Kan extensions, weighted limits, profunctors, Day convolution, and optics.

These constructions are joins, meets, and internal homs in a
[complete lattice](https://en.wikipedia.org/wiki/Complete_lattice), and they have their
universal properties only when the `Algebra`'s $\bigoplus$ is the lattice join, its
$\bigwedge$ (`meet`) the infimum, and its internal hom the residuum. Every function here
therefore takes an [`IdempotentSemiring`](../api/core/algebras.md) (`BOOLEAN`, `GODEL`,
`TROPICAL`, `MAX_PLUS`, or the duals of `BOOLEAN` and `GODEL`) and raises `TypeError` for any
other `Algebra`. On such a semiring the *residuum* $[w, x]$ is the greatest $v$ with
$v \otimes w \le x$, the internal hom of the lattice (`IdempotentSemiring.residuum`), and
$\top$ (`IdempotentSemiring.top`) is its greatest element. The index categories are finite
and discrete, with only identity morphisms, so no naturality or dinaturality condition
along other morphisms arises.

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

For the matrix below the diagonal entries are `0.9`, `0.8`, and `0.7`. Under `GODEL`, whose
join is the maximum and whose meet is the minimum, the end is `0.7` and the coend `0.9`.

```python
import torch
from quivers.enriched.ends_coends import end, coend
from quivers.core.algebras import GODEL

# F: C^op × C → V with C of size 3, so F is shape (3, 3).
F = torch.tensor([
    [0.9, 0.4, 0.1],
    [0.5, 0.8, 0.2],
    [0.3, 0.6, 0.7],
])

ce = coend(F, contra_dims=(0,), co_dims=(1,), algebra=GODEL)  # 0.9
e = end(F, contra_dims=(0,), co_dims=(1,), algebra=GODEL)     # 0.7
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
tensors $\mathcal C(X_i, A)$, with the residuum of the `IdempotentSemiring` as internal hom.
It does not check that the inputs form a presheaf, so the isomorphism holds for inputs that meet
the lemma's hypotheses.

```python
import torch
from quivers.enriched.yoneda import (
    yoneda_lemma, yoneda_embedding, verify_yoneda_preserves_composition, Presheaf,
)
from quivers.core.algebras import GODEL
from quivers.core.objects import FinSet

X1 = FinSet(name="X1", cardinality=2)
X2 = FinSet(name="X2", cardinality=3)

# Presheaf F: C^op → V; values at each object.
F = Presheaf(
    objects=(X1, X2),
    values={0: torch.tensor([0.6, 0.2]), 1: torch.tensor([0.1, 0.5, 0.9])},
    algebra=GODEL,
)

hom_tensors = [
    torch.eye(2),                    # C(X1, X1)
    torch.zeros(3, 2),               # C(X2, X1)
]
fa = yoneda_lemma(presheaf=F, obj_index=0, hom_tensors=hom_tensors, algebra=GODEL)
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
composition, at the given morphisms. Fullness and faithfulness, which are statements about
bijectivity on hom-sets, are not tested.

```python
from quivers.core.morphisms import morphism

f = morphism(X1, X2, algebra=GODEL)
g = morphism(X2, X1, algebra=GODEL)
y_f = yoneda_embedding(f)
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
and the top $\top$ of the lattice on the right. Kan extensions along functors between
categories with more than identity morphisms also involve those morphisms, which these
functions do not model.

[`left_kan`](../api/enriched/kan_extensions.md) and
[`right_kan`](../api/enriched/kan_extensions.md) take $p$ as an
[`ObjectMap`](../api/enriched/kan_extensions.md). Two are provided:
[`Projection`](../api/enriched/kan_extensions.md), from a product onto chosen factors,
and [`Inclusion`](../api/enriched/kan_extensions.md), of one component
$A_k \hookrightarrow A_1 + \dots + A_n$ of a coproduct. Both functions use $R$'s
`IdempotentSemiring`, read its tensor without gradients, and return a fixed `ObservedMorphism`.

```python
import torch
from quivers.enriched.kan_extensions import Inclusion, left_kan, right_kan
from quivers.core.algebras import GODEL
from quivers.core.objects import FinSet, CoproductSet
from quivers.core.morphisms import observed

A0 = FinSet(name="A0", cardinality=2)
A1 = FinSet(name="A1", cardinality=3)
S = CoproductSet(components=(A0, A1))

iota = Inclusion(coproduct=S, component_index=0)  # A0 ↪ A0 + A1

B = FinSet(name="B", cardinality=4)
R = observed(A0, B, torch.rand(2, 4), algebra=GODEL)  # R: A0 → B

LanR = left_kan(R, along=iota)         # Lan_iota R: S → B
RanR = right_kan(R, along=iota)        # Ran_iota R: S → B
```

Along an inclusion each fiber has at most one element, so both extensions copy $R$ onto
the `A0` rows and fill the `A1` rows with the empty reductions.

## Weighted limits

A [weighted limit](https://ncatlab.org/nlab/show/weighted+limit) $\{W, D\}$ of a
diagram $D$ by a weight $W$ generalizes a limit by letting $W(j)$ say how strongly the
$j$-th part of the diagram constrains the result [@kelly1982basic]. It is built from
the internal hom: $[w, d]$ is the largest $v$ with $v \otimes w \le d$, and for Boolean
values it is implication, so a weight of 0 imposes no constraint.

For a family of parallel morphisms $f_j : A \to B$ indexed by a finite set $J$, each an
object of the lattice $\mathcal V^{A \times B}$, the weighted limit and colimit are computed
entrywise,

$$
\{W, f\} = \bigwedge_{j \in J} [W(j), f_j], \qquad W \otimes_J f = \bigoplus_{j \in J} W(j) \otimes f_j,
$$

and they have the universal properties $g \le \{W, f\} \iff W(j) \otimes g \le f_j$ for every
$j$, and $W \otimes_J f \le g \iff W(j) \otimes f_j \le g$ for every $j$.
[`weighted_limit_morphisms`](../api/enriched/weighted_limits.md) and
[`weighted_colimit_morphisms`](../api/enriched/weighted_limits.md) compute the two tensors. A
[`Weight`](../api/enriched/weighted_limits.md) holds one value per index, and
`representable_weight` gives the weight that is the unit at one index and the zero elsewhere.

```python
import torch
from quivers.enriched.weighted_limits import (
    Weight, weighted_limit_morphisms, weighted_colimit_morphisms,
)
from quivers.core.algebras import GODEL
from quivers.core.morphisms import observed
from quivers.core.objects import FinSet

A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
f0 = observed(A, B, torch.tensor([[0.9, 0.2, 0.4], [0.1, 0.8, 0.6]]), algebra=GODEL)
f1 = observed(A, B, torch.tensor([[0.3, 0.7, 0.5], [0.6, 0.4, 0.2]]), algebra=GODEL)

W = Weight(values=torch.tensor([1.0, 0.5]), algebra=GODEL)
lim = weighted_limit_morphisms(W, [f0, f1], algebra=GODEL)      # shape (2, 3)
colim = weighted_colimit_morphisms(W, [f0, f1], algebra=GODEL)  # shape (2, 3)
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
`(*contra.shape, *co.shape)` over an `IdempotentSemiring`. `Profunctor.from_morphism(f)` wraps a morphism's
tensor, and `Profunctor.compose` computes the coend formula.

```python
import torch
from quivers.enriched.profunctors import Profunctor
from quivers.core.algebras import GODEL
from quivers.core.objects import FinSet

A = FinSet(name="A", cardinality=3)
B = FinSet(name="B", cardinality=4)
data = torch.rand(3, 4)
P = Profunctor(contra=A, co=B, tensor=data, algebra=GODEL)
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

where $A$, $B$, and $C$ range over a given tuple of objects. Without further data the tuple
must be closed under the product, and the function raises `ValueError` otherwise. A
`product_table` lists the decompositions instead: it maps a pair of indices $(i, j)$ to the index
$k$ of their product and the hom value $\mathcal C(C_k, A_i \otimes B_j)$. In the example the only
decomposition is $A \otimes B = A \times B$, and under `GODEL` the result is
$\min(0.6, 0.7) = 0.6$ at that index and zero elsewhere.

```python
import torch
from quivers.enriched.day_convolution import day_convolution, day_unit
from quivers.core.algebras import GODEL
from quivers.core.objects import FinSet
from quivers.categorical.monoidal import CartesianProductMonoidal

A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
objects = (A, B, A * B)
monoidal = CartesianProductMonoidal()

# F and G are presheaves indexed by `objects`, one value per object.
f_values = torch.tensor([0.6, 0.4, 0.1])
g_values = torch.tensor([0.5, 0.7, 0.2])

table = {(0, 1): (2, torch.tensor(1.0))}   # A ⊗ B is objects[2]
FG = day_convolution(
    f_values, g_values, objects=objects, monoidal=monoidal, algebra=GODEL, product_table=table,
)
# tensor([0.0000, 0.0000, 0.6000])
unit = day_unit(objects=objects, unit_index=0, algebra=GODEL)
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
- [`Adapter`](../api/enriched/optics.md) is a pair of mutually inverse morphisms, and its
  constructor raises `ValueError` when the round trips differ from the identities;
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

[`compose_optics`](../api/enriched/optics.md) composes two optics into a
[`ComposedOptic`](../api/enriched/optics.md):

```python
combined = compose_optics(outer=lens, inner=adapter)
```

Each optic's `.forward()` and `.backward()` return morphisms. A lens's `.forward()` is its read
map $\mathrm{get} : S \to A$, and `Lens.put()` is its write map
$\mathrm{put} : S \times A \to S$, which replaces the focus and keeps the rest. Both are
functions, so the lens laws hold under every `Algebra`: GetPut,
$\mathrm{put}(s, \mathrm{get}(s)) = s$ (writing back what was read changes nothing); PutGet,
$\mathrm{get}(\mathrm{put}(s, a)) = a$ (reading after writing returns what was written); and
PutPut, $\mathrm{put}(\mathrm{put}(s, a), a') = \mathrm{put}(s, a')$ [@foster2007combinators]. A
prism is expected to satisfy the analogous laws for matching and building. A `Grate` wraps a
tensor it is given and checks no law.
