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

f = morphism(X, Y)
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

<!-- python: skip -->
```python
from quivers.categorical.functors import IdentityFunctor, FreeMonoidFunctor
from quivers.categorical.natural_transformations import ComponentwiseNT
from quivers.categorical.adjunctions import FreeForgetfulAdjunction
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

F = IdentityFunctor()
G = FreeMonoidFunctor(max_length=3)

# η: Id ⇒ Free, the length-1 embedding, taken from the adjunction below
adj = FreeForgetfulAdjunction(max_length=3)
eta = ComponentwiseNT(F, G, adj.unit_component)

# check the naturality square on one morphism
f = morphism(X, Y)
assert eta.verify_naturality(f)
```

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

[`FreeForgetfulAdjunction`](../api/categorical/adjunctions.md) implements this
adjunction on words of bounded length. Its `unit_component(A)` is the length-one
embedding. Its `counit_component` at a free monoid $A^*$ is concatenation
$(A^*)^* \to A^*$, which is $\varepsilon$ at the free monoid and the multiplication of
the monad $GF$. A plain `FinSet` carries no monoid operation, so
`counit_component(A)` at a `FinSet` cannot be a counit. The class returns there the
partial map that sends each length-one word to its letter and every other word to an
all-zero row. Concatenations longer than `max_length` also go to an all-zero row, so
the bounded construction agrees with the adjunction only on words that fit. The
methods `verify_triangle_left(A)` and `verify_triangle_right(A)` compare the two
triangle composites with the identity at one object.

<!-- python: skip -->
```python
from quivers.categorical.adjunctions import FreeForgetfulAdjunction
from quivers.core.objects import FinSet, FreeMonoid

A = FinSet(name="A", cardinality=2)
adjunction = FreeForgetfulAdjunction(max_length=2)

# Left and right functors
F = adjunction.left        # FreeMonoidFunctor
U = adjunction.right       # ForgetfulFunctor

# T(A) = U(F(A)) is the bounded free monoid on A
FA = F.map_object(A)
assert isinstance(U.map_object(FA), FreeMonoid)

eta_A = adjunction.unit_component(A)        # A → A*, length-one words
mu_A = adjunction.counit_component(FA)      # (A*)* → A*, concatenation

assert adjunction.verify_triangle_left(A)
assert adjunction.verify_triangle_right(A)
```

## Monoidal Structures

A [monoidal category](https://ncatlab.org/nlab/show/monoidal+category)
$(\mathcal C, \otimes, I)$ has a product $\otimes$ on objects and morphisms, a unit
object $I$, and isomorphisms that regroup $(X \otimes Y) \otimes Z \cong
X \otimes (Y \otimes Z)$ (the associator) and remove a unit factor (the left and right
unitors). The [coherence theorem](https://ncatlab.org/nlab/show/coherence+theorem+for+monoidal+categories)
states that any two composites of these isomorphisms between the same objects agree,
so a wiring of parallel morphisms does not depend on how it is parenthesized. A
symmetric monoidal category also has a braiding $X \otimes Y \cong Y \otimes X$.

### Cartesian Monoidal

[`CartesianMonoidal`](../api/categorical/monoidal.md) uses the Cartesian product of
finite sets, with the one-element set `Unit` as $I$, and gives the `associator`, the
`left_unitor` and `right_unitor`, and the `braiding`. On functions this product is
the categorical product. On relations and Markov kernels it is only a monoidal
product: a kernel into a pair is not determined by its two marginals, so there is no
product in the categorical sense. Markov kernels with this product, together with
copying and discarding, form a
[Markov category](https://ncatlab.org/nlab/show/Markov+category) [@fritz2020synthetic].

<!-- python: skip -->
```python
from quivers.categorical.monoidal import CartesianMonoidal
from quivers.core.objects import FinSet, Unit

monoidal = CartesianMonoidal()

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

## Base Change

A [`BaseChange`](../api/categorical/base_change.md) applies an entrywise map from one
`Algebra` to another to a morphism and keeps its domain and codomain. When the map is
monotone and preserves the combining operations up to inequality, this is
[change of base](https://ncatlab.org/nlab/show/change+of+enriching+category)
[@kelly1982basic]. `FuzzyToBool(threshold)` sends scores at or above the threshold to
1 and the rest to 0, and `BoolToFuzzy` includes $\{0, 1\}$ in $[0, 1]$. They act like
the QVR transformations `threshold` (which uses a strict inequality at 0.5) and
`boolean_embedding` ([Transformations](transformations.md#singletons)).

Thresholding does not commute with composition: two scores of `0.6` pass a threshold
at `0.5`, while their product `0.36` does not, so thresholding a composite can differ
from composing thresholded morphisms.

<!-- python: skip -->
```python
from quivers.categorical.base_change import BoolToFuzzy, FuzzyToBool
from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# fuzzy → boolean via thresholding
fuzzy_morph = morphism(X, Y, algebra=PRODUCT_FUZZY)
to_bool = FuzzyToBool(threshold=0.5)
bool_morph = to_bool.apply_to_morphism(fuzzy_morph)

assert bool_morph.algebra == BOOLEAN
assert bool_morph.domain == fuzzy_morph.domain
assert bool_morph.codomain == fuzzy_morph.codomain

# boolean → fuzzy via inclusion
to_fuzzy = BoolToFuzzy()
fuzzy_again = to_fuzzy.apply_to_morphism(bool_morph)
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
`Algebra`'s $\bigoplus$, as a matrix trace sums the diagonal. The function
[`trace`](../api/categorical/traced.md) wraps a
[`CartesianTrace`](../api/categorical/traced.md). It takes the fed-back object last
in the domain and codomain, while the method `Morphism.trace` takes it first
([Morphisms](morphisms.md#transpose-trace-cup-and-cap)).

<!-- python: skip -->
```python
from quivers.categorical.traced import trace
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=2)
Y = FinSet(name="Y", cardinality=3)
Z = FinSet(name="Z", cardinality=4)

# f: (Y × X) → (Z × X)
f = morphism(Y * X, Z * X)

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
| Adjunction | $\mathcal D(F(A), B) \cong \mathcal C(A, G(B))$ | `FreeForgetfulAdjunction(max_length)` |
| Monoidal product | parallel composition with coherent regrouping | `CartesianMonoidal`, `f @ g` |
| Change of base | entrywise map between `Algebra`s | `FuzzyToBool().apply_to_morphism(f)` |
| Trace | feedback of one wire | `trace(f, feedback=X, domain=Y, codomain=Z)` |
