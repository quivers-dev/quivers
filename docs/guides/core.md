# Core Types & Algebras

A finite model often starts with a table: rows might be words and columns
might be meanings. **Objects** name the row and column possibilities;
**[morphisms](morphisms.md#what-is-a-morphism)** hold scores between them; an **[algebra](#algebras-enrichment-algebras)** says what those scores
mean and how to combine them. This guide builds those ingredients in that
order. The [concepts introduction](../getting-started/concepts.md) connects
them to probabilistic programs.


## Objects: Finite Sets

The finite tensor layer uses named finite sets to label tensor axes. Continuous
spaces and probability kernels are covered in [Continuous Spaces](continuous-spaces.md).
The finite object hierarchy is:

```
SetObject (abstract)
├── FinSet             , atomic named set with fixed cardinality
├── ProductSet         , Cartesian product A × B × C ...
├── CoproductSet       , tagged union (coproduct) A + B + C ...
└── FreeMonoid         , free monoid on generators, truncated by length
```

### FinSet

A finite set specifies how many distinct possibilities an axis has.
`FinSet("phoneme", 40)` labels forty choices. An index from `0` through
`39` selects one of them; a tensor axis stores a value for each choice.


A `FinSet` is the most basic object:

```python
from quivers.core.objects import FinSet

X = FinSet(name="phoneme", cardinality=40)
print(X.size)   # 40
print(X.shape)  # (40,)
```

The terminal object is provided as a singleton:

```python
from quivers.core.objects import Unit
print(Unit.size)  # 1
```

### ProductSet and CoproductSet

Read a product as **both** and a coproduct as **either, with a tag**. If
`X` has three colors and `Y` has four shapes, `X * Y` has twelve color-shape
pairs. `X + Y` has seven alternatives: a color tagged as a color or a shape
tagged as a shape. The tag prevents two equal-looking component values from
being confused.

Milewski’s [products and coproducts chapter](https://bartoszmilewski.com/2015/01/07/products-and-coproducts/), especially its sections on product and sum types, develops these constructions from pairs and tagged alternatives. [@milewski2015products] The [monoidal structures guide](categorical.md#monoidal-structures) explains how maps act on these combined objects.


Products and coproducts are formed with the `*` and `+` operators, which automatically flatten nesting:

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

Products represent the Cartesian product: elements are pairs. Coproducts are tagged unions: an element belongs to exactly one component, and `CoproductSet` tracks offsets for indexing.

### FreeMonoid

A word is a sequence of generators; the empty word is the sequence of length
zero. In the ordinary free monoid, multiplication concatenates words and the
empty word changes nothing. The class stores only words up to `max_length`,
so concatenating two stored words can exceed the bound. The associated
[monad interface](monadic.md) drops results longer than `max_length`.


A free monoid on a generator set $G$ (truncated by length) represents all strings of length 0 through $n$:

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

## Algebras: Enrichment Algebras

For enriched categories and their algebraic assumptions, see [@kelly1982basic].

Suppose a route from `a` to `c` passes through an intermediate `b`. There are
two questions: how should the scores on successive links combine, and how
should we combine alternative routes? `tensor_op` answers the first; `join`
answers the second. Boolean relations use AND then OR; shortest paths use
addition then minimum; probability kernels use multiplication then sum.
The wiring is the same, but the meaning of a resulting number changes.


An [`Algebra`](../api/core/algebras.md) $(\mathcal{L}, \otimes, \bigvee, \bigwedge, \neg, I, \perp)$ specifies allowed entry values and operations for combining them. It defines the structure in which morphism values live and how morphisms compose. A quantale requires a complete lattice and an associative multiplication that preserves arbitrary joins in each argument. Ordinary sum-product and log-sum-exp aggregation are not lattice joins, though distributivity can still support associative matrix composition. See [Algebras §2](../semantics/algebras.md#2-algebraic-scope) for the assumptions each case needs.

The six primitive operations:

- $\otimes$ (tensor): monoidal product (associative, unital with $I$)
- $\bigvee$ (`join`): aggregates alternative paths; a least upper bound only in the lattice-based cases
- $\bigwedge$ (`meet`): combines constraints; a greatest lower bound only in the lattice-based cases
- $\neg$ (negation): the selected negation operation
- $I$ (unit): identity for $\otimes$
- $\perp$ (zero): identity for the aggregation operation

The finite tensor composition operation uses these two operations:

$$
(g \circ f)(a, c) = \bigvee_b f(a, b) \otimes g(b, c)
$$

### ProductFuzzyAlgebra

The enrichment for fuzzy relations with product t-norm:

$$
\begin{align}
\mathcal{L} &= [0, 1] \\
a \otimes b &= a \cdot b \\
\bigvee_i x_i &= 1 - \prod_i (1 - x_i) \quad \text{(noisy-OR)} \\
\bigwedge_i x_i &= \prod_i x_i \\
\neg a &= 1 - a \\
I &= 1, \quad \perp = 0
\end{align}
$$

This is the default rule for soft relation scores. Its entries need not sum to
one along a row, so a fuzzy relation is not automatically a probability kernel.
Use [Markov sum-product composition](stochastic.md) for conditional probabilities.

Noisy-OR is not a lattice join: combining `0.5` with itself gives `0.75`.
Product also fails to distribute over noisy-OR: `0.5 * (0.5 OR 0.5)`
is `0.375`, while `(0.5 * 0.5) OR (0.5 * 0.5)` is `0.4375`.
Consequently, regrouping a chain of compositions can change its tensor;
preserve the intended parentheses when using this algebra.

### BooleanAlgebra

Crisp binary relations:

$$
\begin{align}
\mathcal{L} &= \{0, 1\} \\
a \otimes b &= a \land b \\
\bigvee_i x_i &= \max_i x_i \quad \text{(OR)} \\
\bigwedge_i x_i &= \min_i x_i \quad \text{(AND)} \\
\neg a &= 1 - a \\
I &= 1, \quad \perp = 0
\end{align}
$$

### LukasiewiczAlgebra

Łukasiewicz t-norm:

$$
\begin{align}
a \otimes b &= \max(a + b - 1, 0) \\
\bigvee_i x_i &= \min(1, \sum_i x_i) \quad \text{(bounded sum)} \\
\bigwedge_i x_i &= \min_i x_i \\
\neg a &= 1 - a
\end{align}
$$

Useful for resource-sensitive reasoning where evidence can "cancel out."

### GodelAlgebra

Gödel (min) t-norm:

$$
\begin{align}
a \otimes b &= \min(a, b) \\
\bigvee_i x_i &= \max_i x_i \\
\bigwedge_i x_i &= \min_i x_i \\
\neg a &= \begin{cases} 1 & \text{if } a = 0 \\ 0 & \text{otherwise} \end{cases}
\end{align}
$$

Composition computes max-min paths: maximize the weakest link along each path.

### TropicalAlgebra

Here a smaller number is a better route. If two routes cost `2 + 5 = 7` and
`4 + 1 = 5`, composition returns `5`. The no-route value must be infinity,
while a link that changes nothing costs zero. This explains why this
algebra's identity differs from an ordinary matrix identity.


The tropical semiring enrichment for generalized metric spaces:

$$
\begin{align}
\mathcal{L} &= [0, \infty] \\
a \otimes b &= a + b \\
\bigvee_i x_i &= \inf_i x_i \quad \text{(shortest path)} \\
\bigwedge_i x_i &= \sup_i x_i \\
I &= 0, \quad \perp = \infty
\end{align}
$$

The identity tensor has $0$ on the diagonal and $\infty$ elsewhere. Composition computes shortest paths via $(g \circ f)(a, c) = \inf_b [f(a, b) + g(b, c)]$.

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

Pass an algebra to a morphism to set its enrichment:

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

Once set, all operations on the morphism (composition, [marginalization](morphisms.md#marginalization), etc.) use that algebra's operations.
