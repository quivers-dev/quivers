# Categorical Structures

`quivers.categorical` implements functors, natural transformations, adjunctions,
monoidal products, change of base, and traces over the finite morphisms of
[Core Types](core.md) and [Morphisms](morphisms.md). Each class computes the
structure on tensors. Most also carry a `verify_*` method that checks the defining
equation numerically on given inputs, which tests that equation at those inputs and
does not prove it in general. Mac Lane is the standard reference for every definition
below [@maclane1998categories].

## Functors

A [functor](https://ncatlab.org/nlab/show/functor) $F : \mathcal C \to \mathcal D$
sends each object $X$ to an object $F(X)$ and each morphism $f : X \to Y$ to a
morphism $F(f) : F(X) \to F(Y)$, so that identities go to identities and
$F(g \circ f) = F(g) \circ F(f)$. The free monoid construction is the standard
example: it sends a set $X$ to the words over $X$ and a function to the map that
applies it letter by letter.

[`Functor`](../api/categorical/functors.md) is an abstract class with three methods:

- `map_object(obj)`: the object $F(X)$;
- `map_morphism(morph)`: the morphism $F(f)$, a
  [`FunctorMorphism`](../api/core/morphisms.md) that recomputes its tensor from `f`;
- `map_tensor(tensor, algebra)`: the action on a morphism's tensor.

The library includes `IdentityFunctor`, `ComposedFunctor`, and `FreeMonoidFunctor`:

<!-- python: skip -->
```python
from quivers.categorical.functors import IdentityFunctor, ComposedFunctor, FreeMonoidFunctor
from quivers.core.algebras import MARKOV
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# Identity endofunctor
I = IdentityFunctor()
assert I.map_object(X) == X

# Free monoid functor on FinSet
F = FreeMonoidFunctor(max_length=3)
FX = F.map_object(X)            # FreeMonoid(generators=X, max_length=3)

f = morphism(X, Y, algebra=MARKOV)  # the word functor needs a Semiring
Ff = F.map_morphism(f)          # FunctorMorphism with domain F(X), codomain F(Y)

# Composition of functors
FF = ComposedFunctor(F, F)      # F ∘ F applied as outer ∘ inner
```

A new functor subclasses `Functor` and implements the three methods.

## Natural Transformations

A [natural transformation](https://ncatlab.org/nlab/show/natural+transformation)
$\alpha : F \Rightarrow G$ between functors $F, G : \mathcal C \to \mathcal D$ is a
family of morphisms $\alpha_X : F(X) \to G(X)$, one for each object $X$, such that
$G(f) \circ \alpha_X = \alpha_Y \circ F(f)$ for every $f : X \to Y$:

$$
\begin{array}{ccc}
F(X) & \xrightarrow{\;\alpha_X\;} & G(X) \\
\Big\downarrow{\scriptstyle F(f)} & & \Big\downarrow{\scriptstyle G(f)} \\
F(Y) & \xrightarrow{\;\alpha_Y\;} & G(Y)
\end{array}
$$

Embedding each element as a word of length one is a natural transformation from the
identity functor to the free monoid functor.
[`NaturalTransformation`](../api/categorical/natural_transformations.md) is an
abstract class, and [`ComponentwiseNT`](../api/categorical/natural_transformations.md)
builds one from a function that returns each component. `verify_naturality(f)`
compares the two sides of the square for one morphism `f`.

```python
import torch
from quivers.categorical.functors import IdentityFunctor, FreeMonoidFunctor
from quivers.categorical.natural_transformations import ComponentwiseNT
from quivers.core.algebras import MARKOV
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism, observed

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

F = IdentityFunctor()
G = FreeMonoidFunctor(max_length=3)

def length_one_words(A):
    """η_A : A → A^{≤3}, each element as a word of length one."""
    words = G.map_object(A)
    tensor = torch.zeros(A.size, words.size)
    for a in range(A.size):
        tensor[a, words.encode((a,))] = 1.0
    return observed(A, words, tensor, algebra=MARKOV)

eta = ComponentwiseNT(F, G, length_one_words)

# check the naturality square on one morphism
f = morphism(X, Y, algebra=MARKOV)
assert eta.verify_naturality(f)
```

`FreeMonoidFunctor` maps each word of length $k$ to words of length $k$ by applying a
morphism letter by letter, and it satisfies the functor laws when the `Algebra` is a
`Semiring`.

## Adjunctions

An [adjunction](https://ncatlab.org/nlab/show/adjunction) $F \dashv G$ between
functors $F : \mathcal C \to \mathcal D$ and $G : \mathcal D \to \mathcal C$ is a
bijection $\mathcal D(F(A), B) \cong \mathcal C(A, G(B))$, natural in $A$ and $B$. It
is equivalently given by a *unit* $\eta : \mathrm{id}_{\mathcal C} \Rightarrow GF$ and
a *counit* $\varepsilon : FG \Rightarrow \mathrm{id}_{\mathcal D}$ satisfying the two
triangle identities $\varepsilon_{F(A)} \circ F(\eta_A) = \mathrm{id}_{F(A)}$ and
$G(\varepsilon_B) \circ \eta_{G(B)} = \mathrm{id}_{G(B)}$. Every adjunction determines
a [monad](monadic.md#monads) $GF$, whose multiplication is
$G \varepsilon_F : GFGF \Rightarrow GF$.

The standard example is the free monoid adjunction between sets and monoids. Giving
each letter of $A$ a value in a monoid $M$ determines a unique map of monoids
$A^* \to M$: evaluate the letters and multiply. The unit $\eta_A : A \to A^*$ embeds
letters as words of length one, and the counit $\varepsilon_M : M^* \to M$ multiplies
out a word of elements of $M$.

[`Adjunction`](../api/categorical/adjunctions.md) is the abstract interface: a subclass gives the
left and right functors and the components `unit_component(A)` and `counit_component(B)`, and
`verify_triangle_left(A)` and `verify_triangle_right(B)` compare the two triangle composites with
the identity at one object. The free monoid adjunction itself has no finite implementation: $A^*$
is infinite, and words of bounded length are not closed under concatenation, so a bounded counit
fails the triangle identities.

## Monoidal Structures

A [monoidal category](https://ncatlab.org/nlab/show/monoidal+category)
$(\mathcal C, \otimes, I)$ has a product $\otimes$ on objects and morphisms, a unit
object $I$, and isomorphisms that regroup $(X \otimes Y) \otimes Z \cong
X \otimes (Y \otimes Z)$ (the associator) and remove a unit factor (the left and right
unitors). The [coherence theorem](https://ncatlab.org/nlab/show/coherence+theorem+for+monoidal+categories)
states that any two composites of these isomorphisms between the same objects agree,
so a wiring of parallel morphisms does not depend on how it is parenthesized. A
symmetric monoidal category also has a braiding $X \otimes Y \cong Y \otimes X$.

### Cartesian product as a monoidal product {#cartesian-monoidal}

[`CartesianProductMonoidal`](../api/categorical/monoidal.md) uses the Cartesian product of
finite sets, with the one-element set `Unit` as $I$, and gives the `associator`, the
`left_unitor` and `right_unitor`, and the `braiding`. On functions this product is
the categorical product. On relations and Markov kernels it is a symmetric monoidal
product and not a categorical one: a kernel into a pair is not determined by its two marginals, so there is no
product in the categorical sense. Markov kernels with this product, together with
copying and discarding, form a
[Markov category](https://ncatlab.org/nlab/show/Markov+category) [@fritz2020synthetic].

<!-- python: skip -->
```python
from quivers.categorical.monoidal import CartesianProductMonoidal
from quivers.core.objects import FinSet, Unit

monoidal = CartesianProductMonoidal()

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# product on objects
XY = monoidal.product(X, Y)
assert XY == X * Y
assert monoidal.unit == Unit

# coherence morphisms
swap = monoidal.braiding(X, Y)              # X × Y → Y × X
assoc = monoidal.associator(X, Y, X)        # (X×Y)×X → X×(Y×X)
```

On morphisms the product is the `@` operator:

<!-- python: skip -->
```python
f = morphism(X, Y)
g = morphism(X, Y)
fg = f @ g                                   # ProductMorphism: X×X → Y×Y
```

### Coproduct Monoidal

`CoproductMonoidal` uses the disjoint union, with the empty set `EMPTY` (exported by
`quivers.categorical.monoidal`) as unit:

<!-- python: skip -->
```python
from quivers.categorical.monoidal import CoproductMonoidal
from quivers.core.objects import FinSet

monoidal = CoproductMonoidal()

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

XY = monoidal.product(X, Y)
assert XY == X + Y
```

## Change of Base {#base-change}

A [`BaseChange`](../api/categorical/base_change.md) applies an entrywise map from one
`Algebra` to another to a morphism and keeps its domain and codomain. When the map is
monotone and preserves the combining operations up to inequality, this is
[change of base](https://ncatlab.org/nlab/show/change+of+enriching+category)
[@kelly1982basic]. `BoolToFuzzy` includes $\{0, 1\}$ in $[0, 1]$, and it acts like the QVR
transformation `boolean_embedding` ([Transformations](transformations.md#singletons)). The
inclusion preserves both operations and both units, so it commutes with composition. The reverse
direction, thresholding, is the QVR transformation `threshold`. It does not commute with
composition: two scores of `0.6` pass a threshold at `0.5`, while their product `0.36` does not.

```python
from quivers.categorical.base_change import BoolToFuzzy
from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

bool_morph = morphism(X, Y, algebra=BOOLEAN)
fuzzy_morph = BoolToFuzzy().apply_to_morphism(bool_morph)

assert fuzzy_morph.algebra == PRODUCT_FUZZY
assert fuzzy_morph.domain == bool_morph.domain
```

## Traced Monoidal Structure

A [traced monoidal category](https://ncatlab.org/nlab/show/traced+monoidal+category)
has an operation

$$
\mathrm{tr}^X_{Y,Z} : \mathcal C(Y \otimes X, Z \otimes X) \to \mathcal C(Y, Z)
$$

that feeds the output $X$ back into the input $X$ [@joyal1996traced]. On finite
tensors, the trace of $f : Y \times X \to Z \times X$ fixes an input $y$ and output
$z$, matches the two $X$ indices, and reduces $f[(y, x), (z, x)]$ over $x$ with the
`Algebra`'s $\bigoplus$, as a matrix trace sums the diagonal. This is the trace that
compact closure induces, and it satisfies the trace axioms when the `Algebra` is a `Semiring`.
The function [`trace`](../api/categorical/traced.md) wraps a
[`CompactClosedTrace`](../api/categorical/traced.md), which requires a `Semiring`. It takes the object fed back last
in the domain and codomain, while the method `Morphism.trace` takes it first
([Morphisms](morphisms.md#transpose-trace-cup-and-cap)).

<!-- python: skip -->
```python
from quivers.categorical.traced import trace
from quivers.core.algebras import MARKOV
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=2)
Y = FinSet(name="Y", cardinality=3)
Z = FinSet(name="Z", cardinality=4)

# f: (Y × X) → (Z × X), over a semiring
f = morphism(Y * X, Z * X, algebra=MARKOV)

# Trace over X: returns Y → Z with the feedback loop closed
traced_f = trace(f, feedback=X, domain=Y, codomain=Z)
assert traced_f.domain == Y
assert traced_f.codomain == Z
```

## Summary

| Structure | What it gives | Library entry point |
|-----------|---------------|--------|
| Functor | maps of objects and morphisms | `Functor`, `F.map_morphism(f)` |
| Natural transformation | one morphism per object, commuting with every $F(f)$ | `ComponentwiseNT(F, G, component)` |
| Adjunction | $\mathcal D(F(A), B) \cong \mathcal C(A, G(B))$ | `Adjunction` (abstract) |
| Monoidal product | parallel composition with coherent regrouping | `CartesianProductMonoidal`, `f @ g` |
| Change of base | entrywise map between `Algebra`s | `BoolToFuzzy().apply_to_morphism(f)` |
| Trace | feedback of one wire | `trace(f, feedback=X, domain=Y, codomain=Z)` |
