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
from quivers.core.algebras import GODEL
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad(algebra=GODEL)
kleisli = KleisliCategory(monad)

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)

f = morphism(X, Y, algebra=GODEL)   # read as a Kleisli arrow X ⇝ Y
g = morphism(Y, Z, algebra=GODEL)

h = kleisli.compose(f, g)       # X ⇝ Z
```

## Fuzzy Powerset Monad

On sets, the $\mathcal V$-valued powerset monad sends $X$ to the set $\mathcal V^X$ of
functions from $X$ to $\mathcal V$, read as fuzzy subsets of $X$. A Kleisli arrow
$X \to \mathcal V^Y$ is then a $\mathcal V$-valued relation, and the Kleisli category is the
category $\mathcal V$-$\mathbf{Rel}$ of these relations under the matrix product
[@hofmann2014monoidal]. The monad laws need that product to be associative and unital, so
$\mathcal V$ must be a semiring, and reading a fuzzy subset as the join of its points needs
$\bigoplus$ to be idempotent. `FuzzyPowersetMonad(algebra)` thus takes an `IdempotentSemiring`,
such as `BOOLEAN` (the ordinary powerset monad, whose Kleisli category is $\mathbf{Rel}$),
`GODEL`, or `TROPICAL`, and raises `TypeError` for any other `Algebra`, `PRODUCT_FUZZY`
included. It works directly with the matrices:

- `fmap_obj(X)` returns `X`, the object that indexes the membership values;
- `fmap(f)` returns `f`;
- `pure(X)` and `join(X)` both return `identity(X)` for the chosen `Algebra`;
- Kleisli composition is `f >> g`, and it refuses a morphism over another `Algebra`.

```python
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.core.objects import FinSet
from quivers.core.algebras import BOOLEAN

X = FinSet(name="X", cardinality=3)

monad = FuzzyPowersetMonad(algebra=BOOLEAN)

# fmap_obj returns the indexing object itself
TX = monad.fmap_obj(X)
assert TX == X

# The unit is the identity morphism of the chosen Algebra
eta_X = monad.unit(X)
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
from quivers.monadic.comonads import CoKleisliCategory, CofreeComonad
from quivers.core.algebras import MARKOV
from quivers.core.objects import FinSet

store = FinSet(name="S", cardinality=4)
comonad = CofreeComonad(store=store, algebra=MARKOV)
cokleisli = CoKleisliCategory(comonad)
```

## Environment Comonad {#cofree-comonad}

`CofreeComonad(store=S, algebra=V)` is the environment comonad $W(A) = A \times S$
described above: `counit` is the projection $(a, s) \mapsto a$, `comultiply` is
$(a, s) \mapsto ((a, s), s)$, and $W(f) = f \times \mathrm{id}_S$. It is the cofree comonad on
the constant functor at $S$, and not the store comonad $W(A) = S \times A^S$. Since only the
environment is copied and every morphism acts on it as the identity, the counit and the
comultiplication are natural for every $\mathcal V$-valued morphism, and the comonad laws hold.
The constructor requires a `Semiring`, so that the coKleisli composite is associative.

```python
from quivers.monadic.comonads import CofreeComonad
from quivers.core.algebras import MARKOV
from quivers.core.objects import FinSet

store = FinSet(name="S", cardinality=4)
comonad = CofreeComonad(store=store, algebra=MARKOV)
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
from quivers.core.algebras import BOOLEAN
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad(algebra=BOOLEAN)
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
explicit_alg = ObservedAlgebra(monad, X, alpha_tensor, algebra=BOOLEAN)  # checks both laws
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
from quivers.core.algebras import BOOLEAN
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad(algebra=BOOLEAN)
X = FinSet(name="X", cardinality=3)
em = EilenbergMooreCategory(monad)

# Free algebra on X
free_alg = em.free_algebra(X)

# Check whether a base morphism f : A → B is a homomorphism between two algebras
f = morphism(X, X, algebra=BOOLEAN)
is_hom = em.is_homomorphism(f, free_alg, free_alg)
```

## Distributive Laws {#distributive-laws}

Two monads do not compose into a monad in general. A list of distributions and a
distribution over lists are different structures, and moving one layer through the
other needs a rule. A [distributive law](https://ncatlab.org/nlab/show/distributive+law)
$\lambda : S \circ T \Rightarrow T \circ S$ is such a rule, a natural transformation
compatible with both units and both multiplications, and it makes $T \circ S$ a monad
[@beck1969distributive].

[`DistributiveLaw`](../api/monadic/distributive_laws.md) is the abstract interface: a subclass
gives the outer monad $S$, the inner monad $T$, and the component
$\lambda_A : S(T(A)) \to T(S(A))$ at each object (`distribute(A)`).
