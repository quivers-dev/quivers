# Core Types & Algebras

A finite model in quivers is a table of scores. Its rows and columns are labeled by
named finite sets, called *objects*. A [*morphism*](morphisms.md#what-is-a-morphism)
from one object to another stores one score for every row and column, and an
[`Algebra`](../api/core/algebras.md) fixes how scores combine when morphisms are
composed. [Concepts](../getting-started/concepts.md) relates these
pieces to probabilistic programs, and [Continuous Spaces](continuous-spaces.md)
covers objects that are not finite.

## Objects: Finite Sets {#objects-finite-sets}

Each object labels one or more tensor axes. The finite objects are:

| Class | Elements | Size |
|---|---|---|
| `FinSet` | a named set of $n$ elements, indexed $0, \dots, n - 1$ | $n$ |
| `ProductSet` | tuples, one element from each component (Cartesian product) | product of the sizes |
| `CoproductSet` | one element of one component, tagged with the component (disjoint union) | sum of the sizes |
| `FreeMonoid` | words over a generator set $G$, up to a maximum length $n$ | $\sum_{k=0}^{n} \lvert G \rvert^k$ |

All four are subclasses of the abstract `SetObject`.

### FinSet

`FinSet(name="phoneme", cardinality=40)` is a set of forty elements, and a tensor axis labeled by it
has length 40. The one-element set is exported as `Unit`.

```python
from quivers.core.objects import FinSet

X = FinSet(name="phoneme", cardinality=40)
print(X.size)   # 40
print(X.shape)  # (40,)
```

```python
from quivers.core.objects import Unit
print(Unit.size)  # 1
```

### ProductSet and CoproductSet

The operators `*` and `+` build products and coproducts, and both flatten nested
operands. If `X` has three colors and `Y` four shapes, `X * Y` has twelve
pairs of a color and a shape and keeps one axis per component. `X + Y` has seven elements on
one flat axis, and `CoproductSet` records the offset at which each component starts,
so a color and a shape with the same index are never confused.

```python
X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

P = X * Y          # ProductSet(components=(X, Y))
print(P.size)      # 12
print(P.shape)     # (3, 4)

C = X + Y          # CoproductSet(components=(X, Y))
print(C.size)      # 7
print(C.shape)     # (7,), flat representation
```

### FreeMonoid

The [free monoid](https://en.wikipedia.org/wiki/Free_monoid) on a generator set $G$
is the set of all finite words over $G$, with concatenation as its operation and the
empty word as its unit. `FreeMonoid(generators=G, max_length=n)` stores only the
words of length $0$ through $n$, as the coproduct $1 + G + G^2 + \dots + G^n$.
Concatenating two stored words can exceed the bound, and the
[monad interface](monadic.md) drops any result longer than `max_length`.

```python
from quivers.core.objects import FinSet, FreeMonoid

G = FinSet(name="letter", cardinality=26)
FM = FreeMonoid(generators=G, max_length=3)

# Components: Unit + G + G^2 + G^3
print(FM.size)  # 1 + 26 + 676 + 17576 = 18279

# Encode a word
idx = FM.encode((0, 1, 2))    # first three letters
word = FM.decode(idx)          # back to tuple
```

## Algebras {#algebras-enrichment-algebras}

Suppose a route from `a` to `c` passes through some intermediate `b`. Composing two
morphisms has to say how the scores of successive links combine and how the scores
of alternative routes combine. An [`Algebra`](../api/core/algebras.md) gives the
first as a binary operation $\otimes$ (the method `tensor_op`) and the second as a
reduction $\bigoplus$ (the method `join`). Composition is then the matrix product
over these two operations:

$$
(g \circ f)(a, c) = \bigoplus_b f(a, b) \otimes g(b, c).
$$

Boolean relations use AND then OR, shortest paths use addition then minimum, and
probability kernels use multiplication then sum. The wiring is the same in each case,
and only the meaning of the resulting number changes.

An `Algebra` also gives a unit $\mathbf 1$ for $\otimes$ (the diagonal of the
identity morphism), a zero $\mathbf 0$ for $\bigoplus$ (the score of "no route"), a
second reduction $\bigwedge$ (`meet`), used for universal quantification, and a
negation $\neg$ (`negate`). The class checks none of the laws these names suggest.
When $\otimes$ and $\bigoplus$ form a
[semiring](https://en.wikipedia.org/wiki/Semiring), that is, both are associative,
$\bigoplus$ is commutative, $\otimes$ distributes over $\bigoplus$, and $\mathbf 0$
annihilates, composition is associative and the identity morphism is a two-sided
unit [@goodman1999semiring; @mohri2002semiring]. When $\otimes$ fails to distribute
over $\bigoplus$, regrouping a chain of compositions can change its result.

The built-in instances, with the standard name of each structure:

| Instance | Values | $\otimes$ | $\bigoplus$ | $\mathbf 1$, $\mathbf 0$ | Standard structure | Semiring |
|---|---|---|---|---|---|---|
| `PRODUCT_FUZZY` (default) | $[0, 1]$ | $a b$ | $1 - \prod_i (1 - x_i)$ | $1$, $0$ | product t-norm with probabilistic sum | no |
| `BOOLEAN` | $\{0, 1\}$ | $a \land b$ | $\lor$ | $1$, $0$ | Boolean semiring | yes |
| `LUKASIEWICZ` | $[0, 1]$ | $\max(a + b - 1, 0)$ | $\min(1, \sum_i x_i)$ | $1$, $0$ | Łukasiewicz t-norm with bounded sum | no |
| `GODEL` | $[0, 1]$ | $\min(a, b)$ | $\max$ | $1$, $0$ | minimum t-norm with maximum | yes |
| `TROPICAL` | $[0, \infty]$ | $a + b$ | $\min$ | $0$, $\infty$ | tropical (min-plus) semiring | yes |
| `MAX_PLUS` | $[-\infty, \infty)$ | $a + b$ | $\max$ | $0$, $-\infty$ | max-plus semiring | yes |
| `LOG_PROB` | $[-\infty, \infty)$ | $a + b$ | $\log \sum_i e^{x_i}$ | $0$, $-\infty$ | log semiring | yes |
| `MARKOV` | $[0, \infty)$ | $a b$ | $\sum$ | $1$, $0$ | sum-product semiring, for row-stochastic matrices | yes |
| `REAL` | $\mathbb R$ | $a b$ | $\sum$ | $1$, $0$ | the ring $(\mathbb R, +, \times)$ | yes |
| `PROBABILITY` | $[0, 1]$ | $a b$ | $\min(1, \sum_i x_i)$ | $1$, $0$ | product with bounded sum (no standard name) | no |
| `COUNTING` | $\mathbb N$ | $a b$ | $\sum$ | $1$, $0$ | counting semiring | yes |

A [t-norm](https://en.wikipedia.org/wiki/T-norm) is an associative, commutative,
monotone operation on $[0, 1]$ with unit $1$, used as fuzzy conjunction, and its dual
t-conorm is the matching fuzzy disjunction [@klement2000triangular]. The class names
in code are `ProductFuzzyAlgebra`, `BooleanAlgebra`, `LukasiewiczAlgebra`,
`GodelAlgebra`, `TropicalAlgebra`, `MaxPlusAlgebra`, `LogProbAlgebra`,
`MarkovAlgebra`, `RealAlgebra`, `ProbabilityAlgebra`, and `CountingAlgebra`. In QVR
the same instances are named `product_fuzzy`, `boolean`, `lukasiewicz`, `godel`,
`tropical`, `max_plus`, `log_prob`, `markov`, `real`, `probability`, and `counting`
([`composition`](dsl-declarations.md#algebra)).

Some of these structures are also [quantales](https://ncatlab.org/nlab/show/quantale),
complete lattices with an associative product that preserves all joins
[@rosenthal1996quantaloids]. The Boolean, minimum t-norm, and tropical cases are
quantales whose $\bigoplus$ is the lattice join. Sum, log-sum-exp, noisy-OR, and
bounded sum are not lattice joins, so the remaining instances are not quantales in
this sense.

### ProductFuzzyAlgebra

The default instance scores fuzzy relations with the product t-norm:

$$
\begin{align}
a \otimes b &= a \cdot b, &
\bigoplus_i x_i &= 1 - \prod_i (1 - x_i), \\
\bigwedge_i x_i &= \prod_i x_i, &
\neg a &= 1 - a.
\end{align}
$$

The reduction is noisy-OR, the probability
that at least one of several independent causes fires [@pearl1988probabilistic].
Rows of a fuzzy relation need not sum to one, so a fuzzy relation is not a
probability kernel. Conditional probabilities use the
[sum-product `MARKOV` instance](stochastic.md).

Noisy-OR is not idempotent: combining `0.5` with itself gives `0.75`. Product also
fails to distribute over it, since `0.5 * (0.5 OR 0.5)` is `0.375` while
`(0.5 * 0.5) OR (0.5 * 0.5)` is `0.4375`. Composition under this instance is thus
not associative, and a chain of compositions should be parenthesized the way the
model intends.

### BooleanAlgebra

The Boolean semiring on $\{0, 1\}$ composes crisp relations: $\otimes$ is AND,
$\bigoplus$ is OR (maximum), $\bigwedge$ is AND over a set (minimum), and
$\neg a = 1 - a$. Its morphisms are the ordinary relations between finite sets.

### LukasiewiczAlgebra

The [Łukasiewicz t-norm](https://en.wikipedia.org/wiki/T-norm) is
$a \otimes b = \max(a + b - 1, 0)$, so two weak premises can combine to a score of
zero. The reduction is the bounded sum $\min(1, \sum_i x_i)$, $\bigwedge$ is the
minimum, and $\neg a = 1 - a$. The t-norm does not distribute over the bounded sum,
so composition under this instance is not associative in general.

### GodelAlgebra

The minimum (Gödel) t-norm takes $a \otimes b = \min(a, b)$ and reduces by maximum,
so composition returns the best path scored by its weakest link. Its negation is
$\neg a = 1$ when $a = 0$ and $0$ otherwise.

### TropicalAlgebra

In the tropical (min-plus) semiring a smaller number is a better route. If two routes
cost $2 + 5 = 7$ and $4 + 1 = 5$, composition returns $5$:

$$
(g \circ f)(a, c) = \min_b \bigl[f(a, b) + g(b, c)\bigr].
$$

The score of "no route" is $\infty$ and a step that changes nothing costs $0$, so the
identity morphism has $0$ on the diagonal and $\infty$ elsewhere. The reduction
$\bigwedge$ is the supremum, and `negate` raises an error. With this structure,
$[0, \infty]$ is the base over which Lawvere reads metric spaces as enriched
categories [@lawvere1973metric].

## Creating and Using Algebras

```python
from quivers.core.algebras import (
    PRODUCT_FUZZY, BOOLEAN,
    LukasiewiczAlgebra, GodelAlgebra, TropicalAlgebra
)

# Use singletons
qnt = PRODUCT_FUZZY

# Or instantiate custom algebras
luka = LukasiewiczAlgebra()
godel = GodelAlgebra()
tropical = TropicalAlgebra()
```

A morphism takes its `Algebra` at construction:

```python
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism
from quivers.core.algebras import BOOLEAN, GodelAlgebra

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# default: PRODUCT_FUZZY
f_fuzzy = morphism(X, Y)

# explicit algebra
f_bool = morphism(X, Y, algebra=BOOLEAN)
f_godel = morphism(X, Y, algebra=GodelAlgebra())
```

Composition, [marginalization](morphisms.md#marginalization), and quantification on
that morphism then use the operations of its `Algebra`.
