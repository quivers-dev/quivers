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
implements. The last column names the structure, with the laws checked by hand against the
definitions on [Setting §1](setting.md#1-weights-and-their-operations). The library checks none of
these laws at run time.

| Class (`name`) | Carrier | $a \otimes b$ | $\bigoplus_i a_i$ | $\mathbf 1$ | $\mathbf 0$ | Structure |
|---|---|---|---|---|---|---|
| `ProductFuzzyAlgebra` (`ProductFuzzy`) | $[0, 1]$ | $ab$ | $1 - \prod_i (1 - a_i)$ | $1$ | $0$ | product t-norm with probabilistic sum; not a semiring |
| `BooleanAlgebra` (`Boolean`) | $\{0, 1\}$ | $a \wedge b$ | $\bigvee_i a_i$ | $1$ | $0$ | Boolean semiring; a quantale |
| `LukasiewiczAlgebra` (`Lukasiewicz`) | $[0, 1]$ | $\max(0, a + b - 1)$ | $\min(1, \sum_i a_i)$ | $1$ | $0$ | Łukasiewicz t-norm with bounded sum; not a semiring |
| `GodelAlgebra` (`Godel`) | $[0, 1]$ | $\min(a, b)$ | $\max_i a_i$ | $1$ | $0$ | minimum t-norm with maximum; a semiring and a quantale |
| `TropicalAlgebra` (`Tropical`) | $[0, \infty]$ | $a + b$ | $\min_i a_i$ | $0$ | $\infty$ | tropical (min-plus) semiring; a quantale |
| `MaxPlusAlgebra` (`MaxPlus`) | $[-\infty, \infty)$ | $a + b$ | $\max_i a_i$ | $0$ | $-\infty$ | max-plus semiring |
| `LogProbAlgebra` (`LogProb`) | $[-\infty, \infty)$ | $a + b$ | $\log \sum_i e^{a_i}$ | $0$ | $-\infty$ | log semiring |
| `MarkovAlgebra` (`Markov`) | $[0, \infty)$ | $ab$ | $\sum_i a_i$ | $1$ | $0$ | probability (sum-product) semiring |
| `RealAlgebra` (`Real`) | $\mathbb R$ | $ab$ | $\sum_i a_i$ | $1$ | $0$ | the ring of real numbers |
| `ProbabilityAlgebra` (`Probability`) | $[0, 1]$ | $ab$ | $\min(1, \sum_i a_i)$ | $1$ | $0$ | product with bounded sum; not a semiring |
| `CountingAlgebra` (`Counting`) | $\mathbb N$ | $ab$ | $\sum_i a_i$ | $1$ | $0$ | counting semiring |

The default for a finite weighted morphism is `ProductFuzzyAlgebra`.

The [t-norms](https://en.wikipedia.org/wiki/T-norm) in the table are the product, Łukasiewicz,
and minimum (Gödel) t-norms, each paired with a t-conorm: the probabilistic sum
$a + b - ab$, the bounded sum $\min(1, a + b)$, and the maximum [@klement2000triangular].
Iterated over a finite family, the probabilistic sum is the noisy-OR $1 - \prod_i (1 - a_i)$, the
probability that at least one of several independent causes with probabilities $a_i$ is present
[@pearl1988probabilistic].

The [tropical semiring](https://en.wikipedia.org/wiki/Tropical_semiring) composes by the min-plus
matrix product and so computes shortest paths. Ordered by $\ge$, so that its join is the infimum,
$[0, \infty]$ is a quantale, and categories enriched in it are Lawvere's generalized metric spaces
[@lawvere1973metric]. The max-plus semiring computes best-path scores, as in the
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

The class [`AlgebraHomomorphism`](../api/core/transformations.md) is the library's type of entrywise maps
between instances, whatever laws they satisfy. `lookup_homomorphism(source, target)` returns the
registered map for a pair of instances, or the identity when the two are the same. The registered
maps are:

| Source to target | Class | Entry map | Laws |
|---|---|---|---|
| Boolean to ProductFuzzy | `Embedding` | inclusion $\{0, 1\} \hookrightarrow [0, 1]$ | preserves $\otimes$, $\bigoplus$, units |
| ProductFuzzy to Boolean | `Threshold` | $a \mapsto [a > \tfrac12]$ | preserves neither operation |
| ProductFuzzy to Gödel | `MaterialImplication` | clamp to $[0, 1]$, the identity on the carrier | preserves neither operation |
| Markov to ProductFuzzy | `Expectation` | clamp to $[0, 1]$ | preserves $\otimes$ on $[0, 1]$, not $\bigoplus$ |
| ProductFuzzy to LogProb | `LogProb` | $a \mapsto \log \max(a, 10^{-30})$ | preserves $\otimes$ away from $0$, not $\bigoplus$ |
| ProductFuzzy to MaxPlus | `MaxPlus` | $a \mapsto \log \max(a, 10^{-30})$ | preserves $\otimes$ away from $0$, not $\bigoplus$ |
| Real to Probability | `ProbabilityClamp` | clamp to $[0, 1]$ | preserves neither operation |
| Probability to Real | `ProbabilityToReal` | inclusion $[0, 1] \hookrightarrow \mathbb R$ | preserves $\otimes$, not $\bigoplus$ |
| Real to Counting | `CountingFromReal` | $a \mapsto \lfloor \max(a, 0) \rfloor$ | preserves neither operation |
| Counting to Real | `CountingToReal` | inclusion $\mathbb N \hookrightarrow \mathbb R$ | a semiring homomorphism |

Only `CountingToReal` and the identity are semiring homomorphisms between semirings, and only for
them is $h_*$ a functor. `Embedding` preserves both operations, but its target is not a semiring.
The two logarithm maps send $0$ to $\log 10^{-30} \approx -69$ instead of to the zero $-\infty$ of
the target. `Morphism.change_base` applies either an `AlgebraHomomorphism` or a
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

Each instance implements `join` as a reduction over tensor axes. The log semiring computes
$\log \sum_i e^{a_i}$ by subtracting the maximum first. When every $a_i$ is $-\infty$, an
unguarded `torch.logsumexp` returns $-\infty$ with a gradient of `NaN`. The library returns
$-\infty$ with a gradient of zero, because cells of a dynamic program that no derivation reaches
are represented by such rows.
