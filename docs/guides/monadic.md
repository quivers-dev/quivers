# Monads & Comonads

A function can return more than a bare value: several alternatives, a random outcome,
or a value together with updated state. A [monad](https://ncatlab.org/nlab/show/monad)
packages one such kind of result with the operations needed to build and sequence
computations that return it, and a [comonad](https://ncatlab.org/nlab/show/comonad)
packages values that carry a context to be consumed
[@moggi1991notions; @wadler1992essence]. `quivers.monadic` implements both over the
finite morphisms of [Morphisms](morphisms.md).

## Monads

A monad on a category $\mathcal C$ is an
[endofunctor](categorical.md#functors) $T : \mathcal C \to \mathcal C$ with two
[natural transformations](categorical.md#natural-transformations), the *unit*
$\eta : \mathrm{id} \Rightarrow T$ and the *multiplication*
$\mu : T \circ T \Rightarrow T$, satisfying

$$
\mu \circ \mu T = \mu \circ T \mu, \qquad \mu \circ \eta T = \mathrm{id}, \qquad \mu \circ T \eta = \mathrm{id}.
$$

For the list monad, $T(X)$ is the set of lists over $X$, $\eta$ makes the singleton
list `[a]`, $T$ applies a function to every element, and $\mu$ flattens one layer, so
`[[1, 2], [3]]` becomes `[1, 2, 3]`. The first law says that flattening three layers
gives the same list in either order, and the other two say that adding and then
removing a singleton layer changes nothing. Together they make
[sequencing](programs.md#bind-steps) independent of how the steps are grouped.

[`Monad`](../api/monadic/typeclasses.md) (in `quivers.monadic.typeclasses`) is an
abstract class. A subclass provides `endofunctor`, `fmap_obj` (the object map),
`fmap` (the morphism map), `pure` (the unit), and `join` (the multiplication). The
concrete monads also expose `unit` and `multiply` as other names for `pure` and
`join`.

## Kleisli Category {#kleisli-category}

Suppose `f(x)` returns a list of possible intermediate values and `g(y)` returns a
list of possible outputs. To compose them, apply `g` to every value from `f` and
flatten the resulting list of lists. In general, the
[Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category)
$\mathcal C_T$ of a monad $T$ has the objects of $\mathcal C$, has as arrows
$X \rightsquigarrow Y$ the morphisms $X \to T(Y)$ of $\mathcal C$, and composes them by

$$
g \circ_T f = \mu_Z \circ T(g) \circ f.
$$

[`KleisliCategory`](../api/monadic/monads.md) wraps a `Monad` and exposes `identity`
and `compose`:

```python
from quivers.monadic.monads import KleisliCategory, FuzzyPowersetMonad
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad()
kleisli = KleisliCategory(monad)

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)

f = morphism(X, Y)              # read as a Kleisli arrow X ⇝ Y
g = morphism(Y, Z)

h = kleisli.compose(f, g)       # X ⇝ Z
```

## Fuzzy Powerset Monad

On sets, the $\mathcal V$-valued powerset monad sends $X$ to the set $\mathcal V^X$ of
functions from $X$ to $\mathcal V$, read as fuzzy subsets of $X$. A Kleisli arrow
$X \to \mathcal V^Y$ is then a $\mathcal V$-valued relation, which the library stores
as a matrix. `FuzzyPowersetMonad(algebra)` works directly with these matrices:

- `fmap_obj(X)` returns `X`, the object that indexes the membership values;
- `fmap(f)` returns `f`;
- `pure(X)` and `join(X)` both return `identity(X)` for the chosen `Algebra`;
- Kleisli composition is `f >> g`.

Its Kleisli category is thus the category of morphisms of
[Morphisms](morphisms.md) under the chosen `Algebra`, and Kleisli composition is
associative exactly when that composition is. Under the default `PRODUCT_FUZZY`,
product does not distribute over noisy-OR, and regrouping a chain of Kleisli
composites can change the result ([Core Types](core.md#productfuzzyalgebra)).

```python
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.core.objects import FinSet
from quivers.core.algebras import PRODUCT_FUZZY

X = FinSet(name="X", cardinality=3)

monad = FuzzyPowersetMonad(algebra=PRODUCT_FUZZY)

# fmap_obj returns the indexing object itself
TX = monad.fmap_obj(X)
assert TX == X

# The unit is the identity morphism of the chosen Algebra
eta_X = monad.unit(X)
```

## Free Monoid Monad

The list monad has lists of every finite length, and a finite implementation has to
bound them. `FreeMonoidMonad(max_length)` takes `fmap_obj(X)` to the
[`FreeMonoid`](core.md#freemonoid) of words of length at most `max_length`. Its
`pure` makes a word of length one, and its `join` concatenates a word of words. A
concatenation longer than `max_length` is dropped, as an all-zero row, rather than
cut to a prefix: with `max_length=2`, flattening `[[a, b], [c]]` gives nothing. The
monad equations thus hold as for lists only on inputs whose flattenings fit within
the bound.

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

A comonad reverses the direction: an endofunctor $W$ with a *counit*
$\varepsilon : W \Rightarrow \mathrm{id}$ and a *comultiplication*
$\delta : W \Rightarrow W \circ W$, satisfying

$$
W\delta \circ \delta = \delta W \circ \delta, \qquad \varepsilon W \circ \delta = \mathrm{id}, \qquad W \varepsilon \circ \delta = \mathrm{id}.
$$

A monadic operation produces $T(Y)$, while a comonadic one consumes $W(X)$. In the
environment comonad $W(X) = X \times S$, a fixed environment $s$ travels with each
value: $\varepsilon(x, s) = x$ extracts the value, and
$\delta(x, s) = ((x, s), s)$ keeps the environment for the next operation.

[`Comonad`](../api/monadic/comonads.md) is an abstract class whose subclasses provide
`endofunctor`, `counit(obj)`, and `comultiply(obj)`.

## CoKleisli Category

The coKleisli category of $W$ has the objects of $\mathcal C$, has as arrows from
$X$ to $Y$ the morphisms $W(X) \to Y$, and composes $f : W(X) \to Y$ with
$g : W(Y) \to Z$ as $g \circ W(f) \circ \delta_X$. `CoKleisliCategory` wraps a
`Comonad` and computes this composite:

```python
from quivers.monadic.comonads import CoKleisliCategory, DiagonalComonad

comonad = DiagonalComonad()
cokleisli = CoKleisliCategory(comonad)
```

## Diagonal Comonad {#diagonal-comonad}

`DiagonalComonad` has $W(A) = A \times A$ and $W(f) = f \times f$. Its `counit` is the
first projection $(a, b) \mapsto a$, and its `comultiply` copies the pair,
$(a, b) \mapsto ((a, b), (a, b))$. With these maps the first counit law holds, but
$W\varepsilon \circ \delta$ sends $(a, b)$ to $(a, a)$, so the second counit law
fails whenever $a \neq b$. The class thus does not define a comonad, and its
coKleisli composite need not be associative or unital.

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

## Environment Comonad {#cofree-comonad}

`CofreeComonad(store=S)` is the environment comonad $W(A) = A \times S$ described
above: `counit` is the projection $(a, s) \mapsto a$ and `comultiply` is
$(a, s) \mapsto ((a, s), s)$. On functions these maps satisfy the comonad laws.
Despite the class name, this is neither the cofree comonad on a functor nor the store
comonad $W(A) = S \times A^S$.

```python
from quivers.monadic.comonads import CofreeComonad
from quivers.core.objects import FinSet

store = FinSet(name="S", cardinality=4)
comonad = CofreeComonad(store=store)
```

## Algebras and Coalgebras {#algebras-and-coalgebras}

An [algebra for a monad](https://ncatlab.org/nlab/show/algebra+over+a+monad) $T$
evaluates structured values into one carrier: it is a pair
$(A, \alpha : T(A) \to A)$ with

$$
\alpha \circ \eta_A = \mathrm{id}_A, \qquad \alpha \circ \mu_A = \alpha \circ T(\alpha).
$$

Summation is an algebra for the list monad on numbers: a singleton sums to its
element, and summing a flattened list equals summing the sums of its sublists. This
sense of "algebra" is unrelated to the `Algebra` class of [Core Types](core.md). The
*free algebra* on $A$ has carrier $T(A)$ and structure map $\mu_A$. Dually, a
coalgebra for a comonad $W$ is a pair $(A, \gamma : A \to W(A))$ satisfying the dual
equations.

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

# Check the algebra laws at this object
assert free_alg.verify_unit_law()
assert free_alg.verify_associativity()

# Algebra from an explicit structure tensor α : T(A) → A
TA = monad.fmap_obj(X)
alpha_tensor = torch.eye(TA.size, X.size)[: TA.size, : X.size]
explicit_alg = ObservedAlgebra(monad, X, alpha_tensor)
```

## Eilenberg-Moore Category

The [Eilenberg-Moore category](https://ncatlab.org/nlab/show/Eilenberg-Moore+category)
of $T$ has the $T$-algebras as objects. A morphism from $(A, \alpha)$ to
$(B, \beta)$ is a map $h : A \to B$ with $h \circ \alpha = \beta \circ T(h)$, so that
evaluating and then mapping agrees with mapping the structured input and then
evaluating. For list summation, this asks that $h$ preserve addition and zero.
`EilenbergMooreCategory.is_homomorphism(h, alg_a, alg_b)` compares the two sides of
that equation for given tensors.

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

## Distributive Laws {#distributive-laws}

Two monads do not compose into a monad in general. A list of distributions and a
distribution over lists are different structures, and moving one layer through the
other needs a rule. A [distributive law](https://ncatlab.org/nlab/show/distributive+law)
$\lambda : S \circ T \Rightarrow T \circ S$ is such a rule, a natural transformation
compatible with both units and both multiplications, and it makes $T \circ S$ a monad
[@beck1969distributive].

`FreeMonoidPowersetLaw(max_length)` has components
$\lambda_A : \mathrm{Free}(P(A)) \to P(\mathrm{Free}(A))$ that read a word of fuzzy
sets $(S_1, \dots, S_k)$ as the fuzzy set of words with membership
$\lambda(S_1, \dots, S_k)(a_1, \dots, a_k) = \bigotimes_i S_i(a_i)$, computed with
the chosen `Algebra`'s $\otimes$.

```python
from quivers.monadic.distributive_laws import FreeMonoidPowersetLaw

law = FreeMonoidPowersetLaw(max_length=3)
```

[`distributive_laws`](../api/monadic/distributive_laws.md) documents the
construction.
