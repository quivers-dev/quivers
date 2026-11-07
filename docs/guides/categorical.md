# Categorical Structures

Category theory studies which ways of assembling typed operations agree.
Functors transform whole pipelines, natural transformations compare
functors, and monoidal and traced structures describe parallel wiring
and feedback. [Core Types](core.md)
and [Morphisms](morphisms.md) supply the finite table interpretation.
For each construction, the equations specify the properties required of
the underlying operations.

## Functors

A functor transforms an entire typed system consistently. As an ordinary
set-theoretic example, send a set of letters to lists of letters and a
letter-renaming function to the function that renames every letter in a list.
Renaming letters with `f` and then `g` gives the same list as renaming once
with their composite: `map(g, map(f, xs)) = map(g ∘ f, xs)`.
Mapping the identity similarly leaves the list unchanged. These are the
identity and composition laws that also govern functors on Quivers’
[typed morphisms](morphisms.md#composition-the-operator) [@maclane1998categories].

A functor $F: \mathcal{C} \to \mathcal{D}$ between $\mathcal{V}$-enriched categories maps objects to objects and morphisms to morphisms, preserving composition and identity.

In quivers, [`Functor`](../api/categorical/functors.md) is an abstract base class with three abstract methods:

- `map_object(obj)`: the object map $F_\text{obj}: \text{Ob}(\mathcal{C}) \to \text{Ob}(\mathcal{D})$
- `map_morphism(morph)`: the morphism map, returning a [`FunctorMorphism`](../api/core/morphisms.md)
- `map_tensor(tensor, algebra)`: the tensor-level action

Concrete subclasses ship with the library:

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

To define your own functor, subclass `Functor` and implement the three abstract methods.

## Natural Transformations

A natural transformation changes representations uniformly across types.
For instance, place a value in a singleton list. Renaming the value and then
making its singleton list agrees with making the list and renaming its
element. The diagram below expresses that two-route agreement:
$G(f) \circ \alpha_X = \alpha_Y \circ F(f)$. A component $\alpha_X$
is the conversion at type `X`; naturality requires all such conversions to
satisfy the displayed equation for every eligible `f`.

A natural transformation $\alpha: F \Rightarrow G$ between functors $F, G: \mathcal{C} \to \mathcal{D}$ is a family of morphisms $\{\alpha_X: F(X) \to G(X)\}$ indexed by objects of $\mathcal{C}$, such that the naturality square commutes:

$$
\begin{array}{cccc}
F(X) \xrightarrow{\alpha_X} & G(X) \\
\Big\downarrow F(f) & & \Big\downarrow G(f) \\
F(Y) \xrightarrow{\alpha_Y} & G(Y)
\end{array}
$$

[`NaturalTransformation`](../api/categorical/natural_transformations.md) is an ABC; use [`ComponentwiseNT`](../api/categorical/natural_transformations.md) to construct one from a callable that produces each component:

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

# unit η: Id ⇒ Free (length-1 embedding) reuses the adjunction's unit
adj = FreeForgetfulAdjunction(max_length=3)
eta = ComponentwiseNT(F, G, adj.unit_component)

# verify naturality on a chosen morphism
f = morphism(X, Y)
assert eta.verify_naturality(f)
```

## Adjunctions

An adjunction relates two ways of specifying the same map. In the ordinary
free-monoid example, assigning each letter an element of a monoid uniquely
determines a concatenation-preserving map on all words: evaluate the letters
and multiply their values. Thus maps from a free monoid correspond to maps
from its generators into the underlying set. Formally,
$\mathcal D(F(A), B) \cong \mathcal C(A, G(B))$, naturally in `A` and `B`.
The unit and counit encode this correspondence; they need not be inverses.
An adjunction also determines a [monad](monadic.md#monads): its unit inserts
a value into the composite functor, and its counit supplies the map used to
flatten two applications of that functor [@maclane1998categories].

Quivers stores words only up to a chosen length. Concatenations exceeding
that bound are dropped, which changes the maps used in the unbounded
free-monoid adjunction. Triangle-identity checks must thus use the
actual bounded operations and specify which input lengths they cover.

An adjunction $F \dashv G$ is a pair of functors with morphisms $\eta: \text{id}_\mathcal{C} \Rightarrow GF$ (unit) and $\varepsilon: FG \Rightarrow \text{id}_\mathcal{D}$ (counit) satisfying triangle identities.

The free-forgetful adjunction $\text{Free} \dashv \text{Forget}$ is shipped as [`FreeForgetfulAdjunction`](../api/categorical/adjunctions.md):

<!-- python: skip -->
```python
from quivers.categorical.adjunctions import FreeForgetfulAdjunction
from quivers.core.objects import FinSet, FreeMonoid

A = FinSet(name="A", cardinality=5)
adjunction = FreeForgetfulAdjunction(max_length=3)

# Left and right functors
F = adjunction.left        # FreeMonoidFunctor
U = adjunction.right       # ForgetfulFunctor

# Monad image T(A) = U(F(A)) is the free monoid on A
T_A = U.map_object(F.map_object(A))
assert isinstance(T_A, FreeMonoid)

# Unit and counit components
eta_A = adjunction.unit_component(A)        # A → A*
eps_A = adjunction.counit_component(A)      # A* → A

# Triangle identities
assert adjunction.verify_triangle_right(A)
```

## Monoidal Structures

A monoidal product describes doing two things side by side. With pairs,
`(a, (b, c))` and `((a, b), c)` contain the same information; the associator
converts between their groupings. A unitor removes a redundant singleton
component. A braiding swaps the two components. Coherence requires that
different routes through these regroupings agree, so wiring a larger
pipeline does not depend on arbitrary parentheses.

A monoidal category $(\mathcal{C}, \otimes, I)$ has:

- A bifunctor $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$ (the product)
- A unit object $I$
- Natural associativity, left identity, right identity (with coherence)

### Cartesian Monoidal

The standard monoidal structure on finite sets. [`CartesianMonoidal`](../api/categorical/monoidal.md) exposes the object-level `product`, the `unit` (terminal object), and the coherence morphisms `associator`, `left_unitor`, `right_unitor`, `braiding`:

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

Morphism-level tensor product is provided directly on `Morphism` via the `@` operator:

<!-- python: skip -->
```python
f = morphism(X, Y)
g = morphism(X, Y)
fg = f @ g                                   # ProductMorphism: X×X → Y×Y
```

### Coproduct Monoidal

Coproduct as the monoidal operation:

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

The unit here is the initial object (empty set), exposed as `EMPTY` from `quivers.categorical.monoidal`.

## Base Change

A threshold turns a soft score into a yes/no decision. The table keeps its
row and column types, but subsequent composition uses Boolean operations.
Thresholding can discard a weak link before another operation uses its
score. Thus thresholding a composite need not agree with composing the
thresholded tables; consult [Transformations](transformations.md) before
using a preservation law.

Change the enriching algebra without changing the structure of morphisms. [`BaseChange`](../api/categorical/base_change.md) is an ABC; concrete instances are `BoolToFuzzy` and `FuzzyToBool`:

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

For `f : Y * X -> Z * X`, fix an external input `y` and output `z`,
match the two `X` indices, and aggregate `f[y, x, z, x]` over `x`.
The matching condition is the feedback wire; the remaining axes are `Y`
and `Z`, leaving exactly the axes shown in the displayed type.

A traced monoidal category adds a trace operator:

$$
\text{tr}^X_{Y,Z}: \mathcal{C}(Y \otimes X, Z \otimes X) \to \mathcal{C}(Y, Z)
$$

The finite tensor trace matches an input index with an output index and
reduces over their shared value, just as a matrix trace sums diagonal entries. The free function [`trace`](../api/categorical/traced.md) wraps a [`CartesianTrace`](../api/categorical/traced.md):

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

# Trace over X: returns A → B with the feedback loop closed
traced_f = trace(f, feedback=X, domain=Y, codomain=Z)
assert traced_f.domain == Y
assert traced_f.codomain == Z
```

The trace contracts the matching $X$ wire, leaving the external domain $Y$ and codomain $Z$.

## Summary

| Structure | Concept | Syntax |
|-----------|---------|--------|
| Functor | Object and morphism map | `F: C → D` |
| Natural Transform | Family of morphisms | `α: F ⇒ G` |
| Adjunction | Natural correspondence between two kinds of morphism | `F ⊣ G` |
| Monoidal | Product with coherence | `(C, ⊗, I)` |
| Base Change | Enrichment transformation | `FuzzyToBool().apply_to_morphism(f)` |
| Traced | Feedback/iteration | `trace(f, feedback=X, domain=Y, codomain=Z)` |
