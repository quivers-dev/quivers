# Semirings, t-norms, and change of base

Composition of finite weighted morphisms is the matrix product over a structure
$\mathcal V = (V, \otimes, \bigoplus, \mathbf 1)$ ([Setting §1](setting.md#1-weights-and-their-operations)):
$\otimes$ combines the weights along one path through the intermediate value, and $\bigoplus$
combines the paths. In the library this structure is an instance of
[`Algebra`](../api/core/algebras.md), whose methods `tensor_op`, `join`, `unit`, and `zero` give
$\otimes$, $\bigoplus$, $\mathbf 1$, and $\mathbf 0$, and which also gives a meet and a negation.
Changing the instance changes what the same composite computes, as in semiring parsing, where one
dynamic program yields a total probability, a best derivation, or a count of derivations
depending on the semiring [@goodman1999semiring]. A *change of base* applies a map
$h : V \to W$ to every entry of a matrix [@kelly1982basic].

## 1. The built-in instances {#1-the-eleven-algebras}

Each row gives the carrier, the two operations, and the unit and zero that the class
implements. The last column names the structure. Each class states the laws of that structure
as claims with a status, and the law grid of the evidence harness checks every claim of every
shipped instance ([§6](#6-capabilities-and-the-status-of-laws)). The library checks none of
these laws at run time.

| Class (`name`) | Carrier | $a \otimes b$ | $\bigoplus_i a_i$ | $\mathbf 1$ | $\mathbf 0$ | Structure |
|---|---|---|---|---|---|---|
| `ProductFuzzyAlgebra` (`ProductFuzzy`) | $[0, 1]$ | $ab$ | $1 - \prod_i (1 - a_i)$ | $1$ | $0$ | product t-norm with probabilistic sum, not a semiring |
| `BooleanAlgebra` (`Boolean`) | $\{0, 1\}$ | $a \wedge b$ | $\bigvee_i a_i$ | $1$ | $0$ | Boolean semiring and quantale |
| `StandardMVAlgebra` (`Lukasiewicz`) | $[0, 1]$ | $\max(0, a + b - 1)$ | $\min(1, \sum_i a_i)$ | $1$ | $0$ | Łukasiewicz t-norm with bounded sum, not a semiring |
| `GodelAlgebra` (`Godel`) | $[0, 1]$ | $\min(a, b)$ | $\max_i a_i$ | $1$ | $0$ | minimum t-norm with maximum, a semiring and a quantale |
| `TropicalAlgebra` (`Tropical`) | $[0, \infty]$ | $a + b$ | $\min_i a_i$ | $0$ | $\infty$ | tropical (min-plus) semiring and quantale |
| `MaxPlusAlgebra` (`MaxPlus`) | $[-\infty, \infty)$ | $a + b$ | $\max_i a_i$ | $0$ | $-\infty$ | max-plus semiring |
| `LogProbAlgebra` (`LogProb`) | $[-\infty, \infty)$ | $a + b$ | $\log \sum_i e^{a_i}$ | $0$ | $-\infty$ | log semiring |
| `MarkovAlgebra` (`Markov`) | $[0, \infty)$ | $ab$ | $\sum_i a_i$ | $1$ | $0$ | probability (sum-product) semiring |
| `RealAlgebra` (`Real`) | $\mathbb R$ | $ab$ | $\sum_i a_i$ | $1$ | $0$ | the ring of real numbers |
| `BoundedSumProductAlgebra` (`Probability`) | $[0, 1]$ | $ab$ | $\min(1, \sum_i a_i)$ | $1$ | $0$ | product with bounded sum, not a semiring |
| `CountingAlgebra` (`Counting`) | $\mathbb N$ | $ab$ | $\sum_i a_i$ | $1$ | $0$ | counting semiring |

The default for a finite weighted morphism is `ProductFuzzyAlgebra`. The classes record the
structure: the eight semirings are subclasses of `Semiring`, the four idempotent ones (Boolean,
Gödel, tropical, and max-plus) of `IdempotentSemiring`, and the three that are not semirings of
`Algebra` only. Further properties are capabilities that a class also inherits, listed in
[§6](#6-capabilities-and-the-status-of-laws).

The [t-norms](https://en.wikipedia.org/wiki/T-norm) in the table are the product, Łukasiewicz,
and minimum (Gödel) t-norms, each paired with a t-conorm: the probabilistic sum
$a + b - ab$, the bounded sum $\min(1, a + b)$, and the maximum [@klement2000triangular].
Iterated over a finite family, the probabilistic sum is the noisy-OR $1 - \prod_i (1 - a_i)$, the
probability that at least one of several independent causes with probabilities $a_i$ is present
[@pearl1988probabilistic].

The [tropical semiring](https://en.wikipedia.org/wiki/Tropical_semiring) composes by the min-plus
matrix product and so computes shortest paths. Ordered by $\ge$, so that its join is the infimum,
$[0, \infty]$ is a quantale, and categories enriched in it are Lawvere's generalized metric spaces
[@lawvere1973metric]. The max-plus semiring computes the scores of best paths, as in the
[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm). The
[log semiring](https://en.wikipedia.org/wiki/Log_semiring) is the probability semiring carried to
log space by $a \mapsto \log a$, with [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) as
its addition [@mohri2002semiring]. It is not idempotent, since $\log(e^a + e^a) = a + \log 2$.
The probability semiring composes stochastic matrices, and a matrix over it with rows summing to
one is a morphism of $\mathbf{FinStoch}$. The counting semiring counts paths or derivations
[@goodman1999semiring].

## 2. Which laws hold {#2-algebraic-scope}

Associativity of the matrix product needs distributivity. Expanding $((r ; s) ; t)(w, z)$ gives
$\bigoplus_y \bigl(\bigoplus_x r(w, x) \otimes s(x, y)\bigr) \otimes t(y, z)$, and moving
$t(y, z)$ inside the inner reduction is the step
$(\bigoplus_i a_i) \otimes b = \bigoplus_i (a_i \otimes b)$. The full calculation is in
[Morphisms §1.1](morphisms.md#11-composition-tensor-identity).

Eight of the eleven instances are semirings: Boolean, Gödel, tropical, max-plus, log, probability,
real, and counting. Their matrix products are associative, with the identity matrix of
[Setting §2](setting.md#2-v-enriched-relations) as unit. The other three fail distributivity, so
their matrix products may depend on the parenthesization:

- Product t-norm with probabilistic sum: with $a = b_1 = b_2 = \tfrac12$,
  $a \otimes (b_1 \oplus b_2) = \tfrac12 \cdot \tfrac34 = \tfrac38$, while
  $(a \otimes b_1) \oplus (a \otimes b_2) = 1 - \tfrac34 \cdot \tfrac34 = \tfrac{7}{16}$.
- Łukasiewicz t-norm with bounded sum: with $a = b_1 = b_2 = \tfrac12$,
  $a \otimes (b_1 \oplus b_2) = \max(0, \tfrac12 + 1 - 1) = \tfrac12$, while
  $(a \otimes b_1) \oplus (a \otimes b_2) = \min(1, 0 + 0) = 0$.
- Product with bounded sum: with $a = \tfrac12$ and $b_1 = b_2 = 1$,
  $a \otimes \min(1, b_1 + b_2) = \tfrac12$, while $\min(1, ab_1 + ab_2) = 1$.

The quantale results of [Setting §2](setting.md#2-v-enriched-relations), which need arbitrary
joins, apply to the Boolean, Gödel, and tropical instances. The real numbers have no least or
greatest element, and the log and probability semirings are not idempotent, so neither is a
quantale.

## 3. Change of base {#3-base-change}

A function $h : V \to W$ induces the entrywise map

$$
h_* r = h \circ r,
\qquad (h_* r)(x, y) = h(r(x, y)),
$$

which keeps the shape of every matrix. If $\mathcal V$ and $\mathcal W$ are semirings and $h$ is a
semiring homomorphism, preserving $\otimes$, $\bigoplus$, $\mathbf 1$, and $\mathbf 0$, then
$h_*$ is a functor $\mathbf{Mat}(\mathcal V) \to \mathbf{Mat}(\mathcal W)$: converting before
composition gives the same matrix as converting after. For quantales, a map that is only lax
monoidal, with $h(a) \otimes h(b) \le h(a \otimes b)$ and $\mathbf 1 \le h(\mathbf 1)$, still
induces a [change of base](https://ncatlab.org/nlab/show/change+of+enriching+category) of enriched
categories [@kelly1982basic]. A map that preserves neither may still be applied entrywise, but
converting before and after composition then give different results. Thresholding at $\tfrac12$
is such a map: two entries of $0.6$ each pass the threshold, but their product $0.36$ does not.

The class [`CarrierMap`](../api/core/transformations.md) is the library's type of entrywise maps
between instances, whatever laws they satisfy, and its subclass `AlgebraHomomorphism` holds the maps
that preserve $\otimes$, $\bigoplus$, $\mathbf 1$, and $\mathbf 0$ (`IdentityHom`, `Embedding`, and
`CountingToReal`). `lookup_carrier_map(source, target)` returns the
registered map for a pair of instances, or the identity when the two are the same. The registered
maps are:

| Source to target | Class | Entry map | Laws |
|---|---|---|---|
| Boolean to ProductFuzzy | `Embedding` | inclusion $\{0, 1\} \hookrightarrow [0, 1]$ | preserves $\otimes$, $\bigoplus$, units |
| ProductFuzzy to Boolean | `Threshold` | $a \mapsto [a > \tfrac12]$ | preserves neither operation |
| ProductFuzzy to Gödel | `ProductFuzzyToGodel` | clamp to $[0, 1]$, the identity on the carrier | preserves neither operation |
| Markov to ProductFuzzy | `MarkovToProductFuzzy` | clamp to $[0, 1]$ | preserves $\otimes$ on $[0, 1]$, not $\bigoplus$ |
| ProductFuzzy to LogProb | `LogProb` | $a \mapsto \log \max(a, 10^{-30})$ | preserves $\otimes$ away from $0$, not $\bigoplus$ |
| ProductFuzzy to MaxPlus | `MaxPlus` | $a \mapsto \log \max(a, 10^{-30})$ | preserves $\otimes$ away from $0$, not $\bigoplus$ |
| Real to Probability | `ProbabilityClamp` | clamp to $[0, 1]$ | preserves neither operation |
| Probability to Real | `ProbabilityToReal` | inclusion $[0, 1] \hookrightarrow \mathbb R$ | preserves $\otimes$, not $\bigoplus$ |
| Real to Counting | `CountingFromReal` | $a \mapsto \lfloor \max(a, 0) \rfloor$ | preserves neither operation |
| Counting to Real | `CountingToReal` | inclusion $\mathbb N \hookrightarrow \mathbb R$ | a semiring homomorphism |

Only `CountingToReal` and the identity are semiring homomorphisms between semirings, and only for
them is $h_*$ a functor. `Embedding` preserves both operations, but its target is not a semiring.
The two logarithm maps send $0$ to $\log 10^{-30} \approx -69$ instead of to the zero $-\infty$ of
the target.

`SupportMap(source)` sends every entry other than the zero to true and the zero to false. It is a
homomorphism onto the Boolean algebra exactly when sums and products of the source are the zero
only when they must be: a sum only when every term is the zero (the source is *zero-sum-free*), a
product only when a factor is (it has *no zero divisors*) [@golan1999semirings]. The map accepts a
source by these two capabilities, and only when it establishes both claims, by an argument or a decision procedure, so a subclass
that replaces an operation without its own proof is refused. The accepted sources are the
Boolean, Gödel, tropical, max-plus, log, probability and counting semirings and the two algebras
with a product t-norm (`ProductFuzzy` and `Probability`). The reals are refused, since
$1 + (-1) = 0$, and so is the Łukasiewicz algebra, since
$\tfrac12 \otimes \tfrac12 = 0$. `Morphism.change_base` applies either a `CarrierMap` or a
[morphism transformation](composition-rules.md#5-first-class-transformations), such as softmax
row normalization, which rewrites a whole matrix and is not an entrywise map.

## 4. Change of base and the denotation of phrases

Write $\llbracket \cdot \rrbracket_{\mathcal V}$ for the denotation of finite weighted phrases
when the active instance is $\mathcal V$. For a semiring homomorphism $h : \mathcal V \to \mathcal W$,

$$
h_* \circ \llbracket \cdot \rrbracket_{\mathcal V} = \llbracket \cdot \rrbracket_{\mathcal W}
$$

holds for phrases built from declared matrices by composition, tensor product, and
marginalization, provided that the declared matrices of the $\mathcal W$ program are the images
under $h$ of those of the $\mathcal V$ program. The proof is an induction on the phrase, using
that $h$ commutes with every operation the clauses use. For the other registered maps the
equation fails in general. The finite stochastic and continuous fragments
([Setting §5](setting.md#5-fragments)) are interpreted by probability measures directly and do
not depend on the active instance.

## 5. Implementation

Each instance implements `join` as a reduction over tensor axes. A reduction over an axis of size
zero is the zero of the remaining shape, in the dtype and device of its input, and its derivative
is zero ([§7](#7-carriers)). The log semiring computes $\log \sum_i e^{a_i}$ with
`torch.logsumexp`, and its binary sum with `torch.logaddexp`. When every $a_i$ is $-\infty$, an
unguarded `torch.logsumexp` returns $-\infty$ with a gradient of `NaN`. The library replaces such
rows by zeros before the reduction and returns the constant $-\infty$, with a gradient of zero,
because cells of a dynamic program that no derivation reaches are represented by such rows: the
log of a sum of absent alternatives is $-\infty$ near them, so no mass moves.

The semirings of chart parsing in `quivers.stochastic` are members of the same hierarchy.
`ChartSemiring` is a `Semiring`, and `LogProbSemiring`, `ViterbiSemiring`, `BooleanSemiring` and
`CountingSemiring` are the log, max-plus, Boolean and counting semirings read through the chart
names of their operations: `times` is `tensor_op`, `plus` is `join`, `plus_pair` is `join_pair`,
and `one` is `unit`. A chart therefore has the empty reductions, the impossible sums and the tie
rule of the core instances.

## 6. Capabilities and the status of laws

The classes form one hierarchy. Above `Semiring`, whose product commutes, is
`OrderPreservingSemiring`: a semiring whose product need not commute [@golan1999semirings].
Matrices over it compose associatively and unitally, but a composite keeps the order of its
factors, so a computation over such weights may reorder sums and may not exchange factors.
`MatrixSemiring(base, n)` is the shipped member: $n \times n$ matrices over a scalar semiring,
with the entrywise sum and the matrix product, which do not commute for $n \ge 2$.

A *capability* is a further property that an executor asks for by type, never by the name of an
instance. Each is a class with the operation it adds and the law it claims:

| Capability | Operation | Law | Shipped members |
|---|---|---|---|
| `Carrier` | `zero_like`, `contains`, `element_shape`, `join` | the zero in the requested shape, dtype and device; a join over no terms is the zero | every algebra, `MatrixSemiring`, `FreeSemimodule` |
| `UnitalCarrier` | `unit_like`, `tensor_op` | the unit in the requested shape, dtype and device | every algebra, `MatrixSemiring` |
| `Selective` | `select`, `rank`, `tie_rule` | $a \oplus b$ is $a$ or $b$; the first best operand receives the derivative | Boolean, Gödel, tropical, max-plus |
| `ZeroSumFree` | none | $a \oplus b = 0$ implies $a = b = 0$ | Boolean, Gödel, tropical, max-plus, log, Markov, counting, ProductFuzzy, Probability, Łukasiewicz |
| `NoZeroDivisors` | none | $a \otimes b = 0$ implies $a = 0$ or $b = 0$ | Boolean, Gödel, tropical, max-plus, log, Markov, real, counting, ProductFuzzy, Probability |
| `Semifield` | `divide` | $(a \otimes b) \oslash b = a$ for $b \ne 0$ | Boolean, max-plus, log, Markov, real |
| `Ring` | `subtract` | $a \oplus (0 \ominus a) = 0$ | real |
| `Star` | `star`, `star_with_work`, `star_domain` | `star(A)` is the least solution of $X = I \oplus A X$ on the declared domain; outside it `star` raises `DivergentClosureError`, and where rounding cannot decide `UndecidedClosureError` | Boolean, Gödel, tropical, max-plus, log, Markov, counting, and matrices over any of them (`StarMatrixSemiring`) |
| `Semimodule` | `act`, `scalars` | the action is associative, unital, distributes over both sums and is annihilated by either zero | `FreeSemimodule` |

A semifield has no zero divisors, so `Semifield` is a refinement of `NoZeroDivisors`. The
tropical semiring on $[0, \infty]$ is not a semifield: the inverse of $1$ would be $-1$, which is
not in its carrier. The counting semiring is not one either, since the naturals have no
inverses; the nonnegative reals with division are the Markov semifield. `FreeSemimodule(K, n)`
is the module of vectors of length $n$ over $K$, with the entrywise sum and $K$ acting on the left
of every entry [@golan1999semirings]. It is the construction on which carriers of expected
statistics are built, whose second component is acted on by probabilities.

A class lists the laws it adds, and `laws()` collects them along the hierarchy. The laws, the
records of evidence and the claims live in `quivers.core.laws`. For each law a member claims,
`claim(law)` returns a `LawClaim` whose status is read from its evidence, so the two cannot
disagree:

- `Proved`: a written argument with the published results it uses. Each shipped class points to
  its argument in [§8](#8-proofs-of-the-claimed-laws). A floating point implementation satisfies
  no algebraic law exactly, so such a claim is about the carrier, and rounding is a separate
  matter.
- `Decided`: a procedure that decides the law on a stated domain, such as every tuple of a
  finite carrier.
- `Tested`: finitely many cases of a larger domain, which can refute a law and cannot establish
  it. The laws about the implementation (the form of the zero and the unit, the sum of no terms,
  the tie rule) quantify over every shape and dtype, so the law grid tests them; the evidence
  names the test, its domain and the number of cases, and the grid checks that number.
- `Assumed`: membership in the class is the only ground.

Proved and decided claims are *established*. The evidence is read from declarations of the
member's own class and never inherited: a subclass may replace an operation, so its claims are
assumed until it states its own argument. A structure built from another one, such as the
matrices over a semiring, holds its laws by a theorem when the laws of its base are established,
and is assumed otherwise. A `CustomAlgebra` claims every law of an algebra, those about its
carrier assumed. `establishes(*laws)` asks whether a member claims laws and establishes them,
and is how `SupportMap` decides. `weakest(claims)` gives the status that a computation using
several claims can state, in the order assumed, tested, decided, proved. A `Requirement` pairs a
law with the subject it must hold of, which is what a plan asks for; the subject of a member's
claims is the qualified name of its class, with the base for a construction over one.

The law grid checks every claim of every shipped member, in float64 and float32, on every pair and
triple of sample elements and on batches with axes of size zero. It compares the operations with
a reference written in plain numbers, which catches a sum replaced by a maximum and a product
whose factors are exchanged: both satisfy every law. Each law also has a member that claims it
and breaks it, which the grid has to reject.

## 7. Carriers

One element of a carrier occupies the trailing `element_shape` axes of a tensor, and the leading
axes index elements. A scalar carrier has `element_shape == ()`; a matrix of `MatrixSemiring(K, n)`
has shape `(n, n)`. The scalar `zero` and `unit` of an `Algebra` are the constants; the general
form is `zero_like(shape, like)` and `unit_like(shape, like)`, which give the constant at every
index of `shape` in the dtype and device of `like`, without a gradient. A dtype that cannot hold
the constant is refused: the zero $-\infty$ of the log semiring has no integer form.
`identity_tensor(shape, dtype=..., device=...)` builds the identity matrix from them.

`contains(t)` decides membership by the carrier's own predicate, and `validate(t)` refuses
complex entries and elements outside the carrier. A `CustomAlgebra` decides membership by the
`contains` it was given, and refuses to decide without one.

A reduction over no terms is the zero. The reduction stays on the autograd graph of its input,
so a derivative with respect to an empty input exists and is zero, rather than `None` or `NaN`.
The meet of no terms is the top of the order where the carrier has one ($1$ on $[0, 1]$, $0$ in
the tropical order); the reals, the log weights and the naturals have no greatest element, so
their meet of no terms is refused.

A selective sum returns one of its operands, and one rule decides which operand receives the
derivative at a tie: the first best operand, in row-major order of the reduced axes, which is
also the witness that `select` returns. In a binary sum `join_pair(a, b)` the first operand is
`a`. Splitting the derivative over tied operands is another element of the subdifferential, but
using it in one operation and not in another would make a derivative depend on how a sum is
bracketed. The rule holds for reductions over any axes and for the chart semirings, which are the
same operations.

## 8. Proofs of the claimed laws

The arguments below are the `Proved` evidence of the shipped classes. For every semiring, that
its operations stay in the carrier is part of its definition, and a sum over several axes is the
iterated sum by the generalized associativity and commutativity of a commutative monoid
[@golan1999semirings]. The chart semirings of `quivers.stochastic` use the argument of the core
class they are.

### Boolean {#proof-boolean}

$(\{0, 1\}, \vee, \wedge, 0, 1)$ is the two-element Boolean algebra, a bounded distributive
lattice, hence a commutative semiring whose sum is idempotent and returns one of its operands
[@golan1999semirings]. A disjunction is $0$ only when both operands are, a conjunction only when a
factor is, and $1$, the only element other than $0$, is its own inverse, so the algebra is a
semifield. Every loop weight is at most $1$, so $a^* = 1$ and the closure of every matrix is the reflexive, transitive closure of its relation [@lehmann1977algebraic].

### Gödel {#proof-godel}

$([0, 1], \max, \min, 0, 1)$ is a bounded chain, hence a bounded distributive lattice and a
commutative semiring whose sum returns one of its operands. $\max(a, b) = 0$ forces $a = b = 0$,
and $\min(a, b) = 0$ forces $a = 0$ or $b = 0$. Every loop weight is at most $1$, so $a^* = 1$ and every matrix has a closure, computed by elimination of states [@lehmann1977algebraic].

### Tropical {#proof-tropical}

$([0, \infty], \min, +, \infty, 0)$, with $\infty + a = \infty$, is the min-plus semiring on the
extended nonnegative reals: $+$ distributes over $\min$ since $a + \min(b, c) = \min(a + b, a + c)$,
and $\infty$ is the unit of $\min$ and absorbs $+$. $\min$ returns one of its operands.
$\min(a, b) = \infty$ forces $a = b = \infty$, and $a + b = \infty$ forces $a = \infty$ or
$b = \infty$. It is not a semifield, since the inverse of $1$ would be $-1$. No distance is below $0$, so $a^* = 0$ for every loop and every matrix has a closure, the matrix of shortest distances [@lehmann1977algebraic].

### Max-plus {#proof-max-plus}

$(\mathbb R \cup \{-\infty\}, \max, +, -\infty, 0)$ is the image of the min-plus semifield on
$\mathbb R \cup \{\infty\}$ under $a \mapsto -a$, which exchanges $\min$ and $\max$ and preserves
$+$. $\max$ returns one of its operands, every finite $a$ has the inverse $-a$, $\max(a, b) = -\infty$
forces $a = b = -\infty$, and $a + b = -\infty$ forces $a = -\infty$ or $b = -\infty$. A loop of score at most $0$ has $a^* = 0$; one of positive score has no closure. Elimination of states [@lehmann1977algebraic] finds the best loop through each state once, so the closure exists exactly when no cycle has positive score, which `star` decides from the exact rational values of the weights when rounding cannot.

### Log {#proof-log-prob}

$\log$ is a bijection from $[0, \infty)$ onto $[-\infty, \infty)$ that sends $+$ to
$\log(e^a + e^b)$, $\cdot$ to $+$, $0$ to $-\infty$ and $1$ to $0$, so it is an isomorphism of
semirings from the nonnegative reals, and every law of those transfers along it: a sum of
nonnegative reals is $0$ only when every term is, a product only when a factor is, and every
positive real has an inverse, which is $a - b$ in the log domain for $b > -\infty$. The closure is the logarithm of $(I - A)^{-1}$ of the probabilities, which exists exactly when their spectral radius is below one: elimination of states [@lehmann1977algebraic] meets loop weights below one exactly then, the pivots of the M-matrix $I - A$ being positive. A loop weight within the rounding bound of one refuses as undecided. The
inverse isomorphism $\exp$ embeds the log weights in the nonnegative reals, preserving the order,
which `to_reals` computes and `from_reals` inverts.

### Real {#proof-real}

$(\mathbb R, +, \cdot, 0, 1)$ is a field: a commutative ring, so $a + (-a) = 0$, in which every
element other than $0$ has an inverse, hence a semifield without zero divisors.

### Counting {#proof-counting}

$(\mathbb N, +, \cdot, 0, 1)$ is a commutative semiring. A sum of naturals is $0$ only when every
term is, and a product only when a factor is. The closure counts the paths, which are finite in number exactly when no cycle has a nonzero count; elimination of states [@lehmann1977algebraic] meets a loop count of zero exactly then, and adds and multiplies counts exactly below $2/\varepsilon$ of the dtype.

### Markov {#proof-markov}

$([0, \infty), +, \cdot, 0, 1)$ is the semifield of nonnegative reals: every positive element has a
positive inverse, a sum of nonnegative reals is $0$ only when every term is, and a product only
when a factor is. The closure is $(I - A)^{-1}$, which exists for a nonnegative matrix exactly when its spectral radius is below one; elimination of states [@lehmann1977algebraic] meets loop weights below one exactly then, and it computes no difference other than $1 - a_{kk}$, so every entry has a small relative error and an unreachable entry is an exact zero. The identity
embeds it in the nonnegative reals.

### Product t-norm with probabilistic sum {#proof-product-fuzzy}

On $[0, 1]$, $1 - (1 - a)(1 - b)$ and $ab$ lie in $[0, 1]$. The noisy-OR over several axes is one
minus a product over them, which is the iterated noisy-OR since the product is associative and
commutative. $1 - \prod_i (1 - x_i) = 0$ forces every $x_i = 0$, and $ab = 0$ forces $a = 0$ or
$b = 0$.

### Product with bounded sum {#proof-probability}

On $[0, 1]$, $ab$ and $\min(1, a + b)$ lie in $[0, 1]$. For nonnegative terms
$\min(1, a + b + c) = \min(1, \min(1, a + b) + c)$, so the bounded sum over several axes is
iterated. $\min(1, a + b) = 0$ forces $a = b = 0$, and $ab = 0$ forces $a = 0$ or $b = 0$.

### Łukasiewicz {#proof-lukasiewicz}

On $[0, 1]$, $\min(1, a + b)$ and $\max(a + b - 1, 0)$ lie in $[0, 1]$, the bounded sum over
several axes is iterated as above, and $\min(1, a + b) = 0$ forces $a = b = 0$.

### Dual of the product t-norm algebra {#proof-dual-product-fuzzy}

It has the probabilistic sum $a + b - ab$ as its product and the
folded product as its sum; both map $[0, 1]$ into itself, and the product is associative and
commutative, so its fold over several axes is iterated.

### Dual of the Łukasiewicz algebra {#proof-dual-lukasiewicz}

Its product is the bounded sum and its sum the folded Łukasiewicz t-norm; both map $[0, 1]$ into
itself, and the t-norm is associative and commutative, so its fold over several axes is iterated.

### Dual of the Boolean algebra {#proof-dual-boolean}

Exchanging the join and the meet of the Boolean algebra gives the Boolean algebra with its order
reversed, a bounded distributive lattice, hence an idempotent commutative semiring.

### Dual of the Gödel algebra {#proof-dual-godel}

Exchanging $\max$ and $\min$ on $[0, 1]$ gives the chain with its order reversed, a bounded
distributive lattice, hence an idempotent commutative semiring.

### Matrices over a semiring {#proof-matrix}

Square matrices of order $n$ over a semiring, with the entrywise sum and the matrix product, form a
semiring whose zero is the zero matrix and whose unit is the identity matrix
[@golan1999semirings]: associativity and distributivity of the matrix product reduce, entry by
entry, to those of the base and to the generalized associativity and commutativity of its sum.
The product does not commute for $n \ge 2$. The claims hold when the laws of the base are
established.

The closure of a matrix of blocks. Let `MatrixSemiring(S, k)` have a base $S$ with a closure, and
write $\varphi : M_n(M_k(S)) \to M_{nk}(S)$ for the flattening
$\varphi(A)[(i, a), (j, b)] = A[i, j][a, b]$. It is a bijection, it sends the entrywise sum to the
entrywise sum and the zero to the zero, and it sends the block product to the product of the flat
matrices, since
$(AB)[i, j][a, b] = \bigoplus_l \bigoplus_c A[i, l][a, c] \otimes B[l, j][c, b]$ is the
$((i, a), (j, b))$ entry of $\varphi(A)\varphi(B)$ by the generalized associativity and
commutativity of $\oplus$; the block identity goes to the identity. So $\varphi$ is an isomorphism
of semirings, and it preserves the order of sums, $x \le y$ when $x \oplus z = y$ for some $z$,
because it preserves sums. A matrix $X$ solves $X = I \oplus A X$ exactly when $\varphi(X)$ solves
$\varphi(X) = I \oplus \varphi(A)\varphi(X)$, and $X$ is least among the solutions exactly when
$\varphi(X)$ is. Hence the closure of $A$ exists exactly when the base's closure of $\varphi(A)$
does, the domain of the closure is the set of matrices whose flat matrix lies in the base's domain,
and $A^* = \varphi^{-1}(\varphi(A)^*)$ is the least solution of $X = I \oplus A X$ and of
$X = I \oplus X A$; the base's least-solution law transfers along $\varphi$. `StarMatrixSemiring`
computes it so, and it is the closure of a semiring whose product does not commute.

### Vectors over a semiring {#proof-free-semimodule}

$K^n$ with the entrywise sum and $k \cdot (m_1, \ldots, m_n) = (k m_1, \ldots, k m_n)$ is a left
$K$-semimodule [@golan1999semirings]: each law of the action is the corresponding law of $K$,
entry by entry. The claims hold when the laws of $K$ are established.

