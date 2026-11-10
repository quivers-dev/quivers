# Monads & Comonads

A function can return more than a bare value: it might return several
alternatives, a random outcome, or a value together with updated state.
A **monad** supplies compatible operations for building and sequencing
computations of one such kind. A **comonad** describes operations that
consume values with context. We first connect these ideas to ordinary
functions, then give their laws and Quivers interfaces.

## Monads

For the ordinary list monad, `pure(a)` produces `[a]`; `fmap(f)` applies
`f` to each element; `join` flattens one layer of lists. For instance,
`[[1, 2], [3]]` becomes `[1, 2, 3]`. The type constructor `T` says what
extra structure a result carries. An *[endofunctor](categorical.md#functors)* maps objects and arrows
within the same category; here it maps element types to list types and
functions to elementwise functions [@moggi1991notions; @wadler1992essence].

The laws below say that flattening three layers in either order agrees,
and that adding then removing a singleton layer changes nothing. They
justify regrouping a [sequence of computations](dsl-programs-and-lets.md#kleisli-bind-syntax). They are obligations on
an implementation: each side of an equation must give the same result.

A monad $T: \mathcal{C} \to \mathcal{C}$ is an endofunctor with two [natural transformations](categorical.md#natural-transformations):

- **Unit** $\eta: \text{id} \Rightarrow T$ (embedding into the monad)
- **Multiply** $\mu: T \circ T \Rightarrow T$ (flattening nested applications)

satisfying associativity and unitality (monad laws):

$$
\mu \circ \mu T = \mu \circ T \mu, \quad \mu \circ \eta T = \text{id}, \quad \mu \circ T \eta = \text{id}
$$

[`Monad`](../api/monadic/typeclasses.md) (in `quivers.monadic.typeclasses`) is an abstract base class. Concrete implementations must provide `endofunctor`, `fmap_obj`, `fmap`, `pure`, and `join`; the Eilenberg-Moore aliases `unit` / `multiply` are usually exposed too.

## Kleisli Category

Suppose `f(x)` returns a list of possible intermediate values and `g(y)`
returns possible outputs for one intermediate. To compose them, apply `g`
to every value from `f`, then flatten the resulting list of lists. For
probabilities, the corresponding operation weights each route and sums
over the intermediate. A Kleisli arrow packages this familiar sequencing
pattern as an arrow `X -> Y` whose actual result has type `T(Y)`.

Every monad $T$ induces a Kleisli category $\mathcal{C}_T$ where:

- Objects are the same as $\mathcal{C}$
- Morphisms $X \rightsquigarrow Y$ are morphisms $X \to T(Y)$ in $\mathcal{C}$
- Composition is Kleisli composition: $(g \circ_T f)(x) = \mu(T(g)(f(x)))$

In quivers, [`KleisliCategory`](../api/monadic/monads.md) wraps any `Monad` and exposes `identity` and `compose`:

```python
from quivers.monadic.monads import KleisliCategory, FuzzyPowersetMonad
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad()
kleisli = KleisliCategory(monad)

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)

f = morphism(X, Y)              # interpreted as Kleisli arrow X ⇝ Y
g = morphism(Y, Z)

h = kleisli.compose(f, g)       # X ⇝ Z via μ ∘ T(g) ∘ f
```

## Fuzzy Powerset Monad

The fuzzy powerset monad on the V-enriched category of finite sets:

```python
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.core.objects import FinSet
from quivers.core.algebras import PRODUCT_FUZZY

X = FinSet(name="X", cardinality=3)

monad = FuzzyPowersetMonad(algebra=PRODUCT_FUZZY)

# At the set level, T(X) = X
TX = monad.fmap_obj(X)
assert TX == X

# η_X : X → T(X) is the identity morphism with PRODUCT_FUZZY enrichment
eta_X = monad.unit(X)
```

A fuzzy subset is stored as one membership value for each element of `X`,
so `fmap_obj(X) == X` retains the object that indexes those values.
Composition uses the selected relation operation. With `PRODUCT_FUZZY`,
product fails to distribute over noisy-OR, so different parenthesizations
can give different results; see the numerical calculation in
[Core Types](core.md#productfuzzyalgebra).

## Free Monoid Monad

The ordinary list monad includes lists of every finite length. A finite
implementation must bound that collection. Here `pure` inserts a
length-one word and `join` concatenates nested words; flattenings longer
than `max_length` are dropped rather than clipped to a prefix. This
can lose weight: with `max_length=2`, flattening `[[a, b], [c]]`
drops the length-three result. The usual list-monad equations thus
require a separate check for this truncated representation.

The `FreeMonoidMonad` API stores words up to a maximum length:

```python
from quivers.monadic.monads import FreeMonoidMonad
from quivers.core.objects import FinSet, FreeMonoid

monad = FreeMonoidMonad(max_length=3)

X = FinSet(name="X", cardinality=5)

# T(X) = FreeMonoid(generators=X, max_length=3)
TX = monad.fmap_obj(X)
assert isinstance(TX, FreeMonoid)
assert TX.generators == X
assert TX.max_length == 3
```

## Comonads

The direction of the extra structure changes. A monadic operation produces
`T(Y)`; a contextual operation consumes `W(X)`. For a standard product
comonad `W(X) = E * X`, a fixed environment `E` travels with the value:
`counit(e, x) = x` extracts it, and `comultiply(e, x) = (e, (e, x))`
retains the environment for another contextual operation. This ordinary
example motivates the interface; it is different from the pair operation
implemented by `DiagonalComonad` below.

A comonad $W: \mathcal{C} \to \mathcal{C}$ is a dual monad with:

- **Counit** $\varepsilon: W \Rightarrow \text{id}$ (projection from comonad)
- **Comultiply** $\delta: W \Rightarrow W \circ W$ (extend / duplicate)

[`Comonad`](../api/monadic/comonads.md) is an ABC; subclasses provide `endofunctor`, `counit(obj)`, and `comultiply(obj)`.

## CoKleisli Category

The coKleisli category $\mathcal{C}^W$ has:

- Objects as in $\mathcal{C}$
- Morphisms $X \rightleftharpoons Y$ are morphisms $W(X) \to Y$ in $\mathcal{C}$
- Composition $f \,=\!\!>\!\!>\, g = g \circ W(f) \circ \delta$

```python
from quivers.monadic.comonads import CoKleisliCategory, DiagonalComonad

comonad = DiagonalComonad()
cokleisli = CoKleisliCategory(comonad)
```

## Diagonal Comonad

With the current `DiagonalComonad` pair operations, $(a,b)$ duplicates to $((a,b),(a,b))$.
Applying the first projection inside each pair then gives $(a,a)$, not
$(a,b)$ in general; thus $W(\varepsilon)\circ\delta=\mathrm{id}$
fails whenever `a` and `b` differ. The example below constructs these
pair operations.

The `DiagonalComonad` implementation stores $W(A) = A \times A$, takes the
first projection as `counit`, and duplicates the pair in `comultiply`:

```python
from quivers.monadic.comonads import DiagonalComonad
from quivers.core.objects import FinSet

X = FinSet(name="X", cardinality=3)
comonad = DiagonalComonad()

# Endofunctor action: W(X) = X × X
WX = comonad.endofunctor.map_object(X)
assert WX == X * X

# Counit ε_X : X × X → X (first projection)
eps = comonad.counit(X)

# Comultiplication δ_X : X × X → (X × X) × (X × X)
delta = comonad.comultiply(X)
```

## Cofree Comonad

The `CofreeComonad` class provides a store-based representation:

```python
from quivers.monadic.comonads import CofreeComonad
from quivers.core.objects import FinSet

store = FinSet(name="S", cardinality=4)
comonad = CofreeComonad(store=store)
```

## Algebras and Coalgebras

A monad describes structured values; a **monad algebra** specifies how to
evaluate that structure into a value of one carrier `A`. For lists of
numbers, summation is an example: a singleton evaluates to its number,
and summing flattened lists agrees with summing each sublist and then
summing the results. These are exactly the two laws below. This use of
*algebra* is distinct from the entry-combination `Algebra` in [Core Types](core.md).

A $T$-algebra is a pair $(A, \alpha: T(A) \to A)$ satisfying:

$$
\alpha \circ \eta_A = \text{id}_A, \quad \alpha \circ \mu_A = \alpha \circ T(\alpha)
$$

```python
import torch
from quivers.monadic.algebras import FreeAlgebra, ObservedAlgebra
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad()
X = FinSet(name="X", cardinality=3)

# Free algebra: carrier is T(A), structure map is μ_A
free_alg = FreeAlgebra(monad, X)
assert free_alg.carrier == monad.fmap_obj(X)

# Verify the algebra laws
assert free_alg.verify_unit_law()
assert free_alg.verify_associativity()

# Algebra from an explicit structure tensor α : T(A) → A
TA = monad.fmap_obj(X)
alpha_tensor = torch.eye(TA.size, X.size)[: TA.size, : X.size]
explicit_alg = ObservedAlgebra(monad, X, alpha_tensor)
```

A $W$-coalgebra is the dual notion $(A, \gamma: A \to W(A))$.

## Eilenberg-Moore Category

This category organizes the evaluators just described. A map between two
carriers is allowed only when mapping after evaluation agrees with
evaluating after mapping the structured inputs. For list summation, this
is the requirement that the map preserve the chosen addition and zero.
The homomorphism test below checks that agreement for supplied tensors.

The Eilenberg-Moore category of $T$ has $T$-algebras as objects and algebra homomorphisms as morphisms:

```python
from quivers.monadic.algebras import EilenbergMooreCategory
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad()
X = FinSet(name="X", cardinality=3)
em = EilenbergMooreCategory(monad)

# Free algebra on X
free_alg = em.free_algebra(X)

# Check whether a base morphism f : A → B is a homomorphism between two algebras
f = morphism(X, X)
is_hom = em.is_homomorphism(f, free_alg, free_alg)
```

## Distributive Laws

Effects do not generally commute. A list of distributions and a distribution
over lists store different structures; combining them requires a specified
rule for moving one layer through the other. A distributive law supplies
that rule and its compatibility with both units and multiplications.
The type $T(S(A)) \to S(T(A))$ records the direction of the exchange.

A distributive law $\lambda: T \circ S \Rightarrow S \circ T$ between monads allows lifting one through the other:

```python
from quivers.monadic.distributive_laws import FreeMonoidPowersetLaw

law = FreeMonoidPowersetLaw(max_length=3)
```

The lifted monad combines word-formation with fuzzy aggregation; see [`distributive_laws`](../api/monadic/distributive_laws.md) for the precise construction.
