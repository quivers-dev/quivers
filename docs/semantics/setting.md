# Setting and notation

QVR phrases are interpreted in three categories: $\mathcal V$-matrices between finite sets,
stochastic matrices between finite sets, and Markov kernels between standard Borel spaces.
Programs that condition on data are interpreted as s-finite kernels.

## 0. Categories, functors, and monads

A [*category*](https://ncatlab.org/nlab/show/category) consists of objects, a set of
[*morphisms*](https://ncatlab.org/nlab/show/morphism) $f : A \to B$ for each pair of objects
(the [*hom-set*](https://ncatlab.org/nlab/show/hom-set)), an associative composition, and an
identity morphism on each object [@maclane1998categories]. We write $g \circ f$ for "first $f$,
then $g$", and $f ; g$ for the same composite in diagrammatic order. A
[*functor*](https://ncatlab.org/nlab/show/functor) is a map between categories that preserves
composition and identities, and it is
[*faithful*](https://ncatlab.org/nlab/show/faithful+functor) when it is injective on each
hom-set. A [*natural transformation*](https://ncatlab.org/nlab/show/natural+transformation)
between two functors is a family of morphisms, one per object, that commutes with the image of
every morphism.

A [*monad*](https://ncatlab.org/nlab/show/monad) on a category $\mathcal C$ is a functor
$T : \mathcal C \to \mathcal C$ with a unit $\eta_X : X \to T(X)$ and a multiplication
$\mathrm{m}_X : T(T(X)) \to T(X)$, natural in $X$, that satisfy a left unit law, a right unit law,
and an associativity law [@maclane1998categories]. Moggi interprets effectful programs by
morphisms $X \to T(Y)$ [@moggi1991notions]. These morphisms form the
[*Kleisli category*](https://ncatlab.org/nlab/show/Kleisli+category) of $T$, with the same
objects as $\mathcal C$, the unit $\eta_X$ as identity on $X$, and the composite of
$f : X \to T(Y)$ and $g : Y \to T(Z)$ given by $\mathrm{m}_Z \circ T(g) \circ f$. A monad on a
category with products is [*strong*](https://ncatlab.org/nlab/show/strong+monad) when it has a
natural map $X \times T(Y) \to T(X \times Y)$ compatible with the unit and multiplication, and
it is [*commutative*](https://ncatlab.org/nlab/show/commutative+monad) when the two ways of
pairing $T(X) \times T(Y) \to T(X \times Y)$ agree, so that independent effects may be performed
in either order [@kock1972strong].

A [*monoidal category*](https://ncatlab.org/nlab/show/monoidal+category) has a tensor product
$\boxtimes$ of objects and of morphisms with a unit object, associative and unital up to
isomorphisms that satisfy the coherence conditions [@maclane1998categories]. It is
[*symmetric*](https://ncatlab.org/nlab/show/symmetric+monoidal+category) when there is also a
swap $A \boxtimes B \to B \boxtimes A$ that squares to the identity. A
[*Markov category*](https://ncatlab.org/nlab/show/Markov+category) is a symmetric monoidal
category in which every object carries a copy map $A \to A \boxtimes A$ and a discard map
$A \to I$, compatible with the tensor product. Kernels form Markov categories, and Fritz uses
this structure to state conditional independence and conditionals without measure theory
[@fritz2020synthetic].

For a monoidal category $\mathbf W$, a
[*$\mathbf W$-enriched category*](https://ncatlab.org/nlab/show/enriched+category) has an
object of $\mathbf W$ in place of each hom-set, with composition and identities given by
morphisms of $\mathbf W$ [@kelly1982basic].

## 1. Weights and their operations

Composition of finite weighted relations is computed with two operations on a set of weights.
Let $\mathcal V = (V, \otimes, \bigoplus, \mathbf 1)$ be a set $V$ of weights with a binary
product $\otimes$, a reduction $\bigoplus$ that combines any finite family of weights, and a unit
$\mathbf 1$ for $\otimes$. In the library, these are the methods `tensor_op`, `join`, and `unit`
of the class [`Algebra`](../api/core/algebras.md), which also gives a zero, a meet, and a
negation. The method name `join` does not mean that $\bigoplus$ is a lattice join: it is the
sum for the sum-product instances and log-sum-exp for the log semiring.

The laws below hold for some instances and fail for others, and the results that need them say
so. [Semirings, t-norms, and change of base](algebras.md) lists which laws each built-in
instance satisfies.

- $\mathcal V$ is a [*semiring*](https://en.wikipedia.org/wiki/Semiring) when $\bigoplus$ is an
  associative and commutative addition with a zero $\mathbf 0$, $\otimes$ is associative with
  unit $\mathbf 1$, $\otimes$ distributes over $\bigoplus$ on both sides, and $\mathbf 0$
  annihilates: $a \otimes \mathbf 0 = \mathbf 0 \otimes a = \mathbf 0$
  [@goodman1999semiring; @mohri2002semiring]. It is *commutative* when $\otimes$ is.
- A [*quantale*](https://ncatlab.org/nlab/show/quantale) is a
  [complete lattice](https://en.wikipedia.org/wiki/Complete_lattice) $V$, that is, a partial
  order in which every subset has a least upper bound $\bigvee$, with an associative product
  $\otimes$ that distributes over arbitrary joins on both sides
  [@rosenthal1996quantaloids]:

    $$
    a \otimes \bigvee_{i \in I} b_i = \bigvee_{i \in I} (a \otimes b_i),
    \qquad
    \Bigl(\bigvee_{i \in I} a_i\Bigr) \otimes b = \bigvee_{i \in I} (a_i \otimes b).
    $$

    A quantale with a unit for $\otimes$ is *unital*. When $\bigoplus$ is the join $\bigvee$, a
    unital quantale is a semiring whose addition is idempotent.
- A [*t-norm*](https://en.wikipedia.org/wiki/T-norm) is an associative, commutative, monotone
  operation on $[0, 1]$ with unit $1$, and a *t-conorm* is the same with unit $0$
  [@klement2000triangular]. They generalize logical conjunction and disjunction. A t-norm
  paired with a t-conorm need not satisfy distributivity, and so need not be a semiring.

## 2. V-matrices {#2-v-enriched-relations}

Let $X$ and $Y$ be finite sets. A *$\mathcal V$-matrix* (or $\mathcal V$-valued relation) from
$X$ to $Y$ is a function

$$
r : X \times Y \to V.
$$

The composite of $r : X \to Y$ and $s : Y \to Z$ is the matrix product over $\mathcal V$:

$$
(r ; s)(x, z) = \bigoplus_{y \in Y} r(x, y) \otimes s(y, z).
$$

When $\mathcal V$ is a semiring, this composition is associative, and the matrices
$1_X(x, x') = \mathbf 1$ for $x = x'$ and $\mathbf 0$ otherwise are identities. Finite sets and
$\mathcal V$-matrices then form a category, which we write $\mathbf{Mat}(\mathcal V)$
[@hofmann2014monoidal]. For the Boolean semiring it is the category $\mathbf{Rel}$ of relations
between finite sets. When $\mathcal V$ is a quantale, each hom-set $V^{X \times Y}$ is a complete
lattice under the pointwise order and composition preserves joins in each argument, so
$\mathbf{Mat}(\mathcal V)$ is a
[*quantaloid*](https://ncatlab.org/nlab/show/quantaloid), a category enriched in complete lattices
[@rosenthal1996quantaloids].

When the semiring $\mathcal V$ is also commutative, $\mathbf{Mat}(\mathcal V)$ is symmetric
monoidal. The
tensor of objects is the Cartesian product, and the tensor of matrices is the Kronecker product

$$
(r \boxtimes s)((x_1, x_2), (y_1, y_2)) = r(x_1, y_1) \otimes s(x_2, y_2).
$$

Given the semiring laws, commutativity of $\otimes$ is what makes $\boxtimes$ a functor, that is, what makes
$(r ; r') \boxtimes (s ; s') = (r \boxtimes s) ; (r' \boxtimes s')$.

If $\mathcal V$ is not a semiring, the formula above still defines a composite, and the runtime
computes it, but the composite need not be associative and the identity matrices need not be
identities. [Composition rules](composition-rules.md) gives the levels of law that a rule may
claim.

## 3. Standard Borel spaces and Markov kernels {#3-standard-borel-spaces-and-markov-kernels}

A [*measurable space*](https://en.wikipedia.org/wiki/Measurable_space) $(S, \Sigma)$ is a set
$S$ with a [σ-algebra](https://en.wikipedia.org/wiki/%CE%A3-algebra) $\Sigma$ of subsets, the
events to which probabilities are assigned. It is a
[*standard Borel space*](https://en.wikipedia.org/wiki/Standard_Borel_space) when it is
isomorphic to a Borel subset of a [Polish space](https://en.wikipedia.org/wiki/Polish_space),
that is, of a separable, completely metrizable space. Every finite set with all subsets as events,
every $\mathbb R^n$, and every countable product of these is standard Borel
[@kechris1995classical]. We write $\mathbf{SBor}$ for the category of standard Borel spaces and
[measurable functions](https://en.wikipedia.org/wiki/Measurable_function).

A [*Markov kernel*](https://en.wikipedia.org/wiki/Markov_kernel) $k$ from $S$ to $T$ assigns to
each $s \in S$ a probability measure $k(s, -)$ on $T$, such that $s \mapsto k(s, B)$ is
measurable for every event $B$ of $T$. On finite sets, a Markov kernel is a
[stochastic matrix](https://en.wikipedia.org/wiki/Stochastic_matrix). Kernels compose by
integrating over the intermediate value, as in the
[Chapman–Kolmogorov equation](https://en.wikipedia.org/wiki/Chapman%E2%80%93Kolmogorov_equation):

$$
(k ; l)(s, C) = \int_T l(t, C) \, k(s, \mathrm{d}t).
$$

A measurable function $f : S \to T$ gives the *deterministic kernel* $s \mapsto \delta_{f(s)}$,
where $\delta_t$ is the [Dirac measure](https://en.wikipedia.org/wiki/Dirac_measure) at $t$.

The [*Giry monad*](https://ncatlab.org/nlab/show/Giry+monad) $\mathcal G$ sends a space $S$ to
the space $\mathcal G(S)$ of probability measures on $S$, with the least σ-algebra that makes
each evaluation $\mu \mapsto \mu(B)$ measurable [@giry1982categorical]. If $S$ is standard Borel,
so is $\mathcal G(S)$ [@kechris1995classical], so $\mathcal G$ is a monad on $\mathbf{SBor}$. Its
unit sends $s$ to $\delta_s$, and its multiplication averages a measure over measures. A Markov
kernel from $S$ to $T$ is the same thing as a measurable map $S \to \mathcal G(T)$, and the
Kleisli composite of two such maps is the integral above, so Markov kernels are the morphisms of
the Kleisli category of $\mathcal G$ [@giry1982categorical; @panangaden1999markov]. The Giry monad
is commutative: this is [Fubini's theorem](https://en.wikipedia.org/wiki/Fubini%27s_theorem) for
probability measures.

Following Fritz, we write $\mathbf{BorelStoch}$ for the category of standard Borel spaces and
Markov kernels, and $\mathbf{FinStoch}$ for its full subcategory on finite sets, whose morphisms
are stochastic matrices [@fritz2020synthetic]. Both are Markov categories. On a standard Borel
space every joint distribution has a measurable family of conditional distributions, which fails
for general measurable spaces [@fritz2020synthetic]. On finite sets, $\mathcal G$ restricts to
the [*distribution monad*](https://ncatlab.org/nlab/show/distribution+monad) $\mathcal D$, which
sends $X$ to the probability vectors on $X$. Since $\mathcal D(X)$ is a simplex, an infinite set,
$\mathcal D$ is not a monad on $\mathbf{FinSet}$, but its Kleisli morphisms between finite sets
are exactly the stochastic matrices. The class [`FinStoch`](../api/stochastic/giry.md) implements
$\mathbf{FinStoch}$ as the Kleisli category of the class `GiryMonad`.

## 4. Finite and s-finite kernels {#4-finite-and-s-finite-kernels}

A program that conditions on data multiplies probabilities by likelihoods, so its denotation is a
kernel into measures that need not have mass one. A *kernel* from $S$ to $T$ in this sense assigns
to each $s$ a measure $k(s, -)$ on $T$ with values in $[0, \infty]$, measurably in $s$. It is
*finite* when there is a single bound $r < \infty$ with $k(s, T) < r$ for every $s$, and
*s-finite* when it is a countable sum $\sum_{i} k_i$ of finite kernels [@staton2017commutative].
A subprobability kernel has $k(s, T) \le 1$ for every $s$.

Staton shows that s-finite kernels compose by the same integral as Markov kernels, that this
composition is associative, and that iterated integrals against s-finite kernels may be taken in
either order [@staton2017commutative]. The last fact is what makes two independent `sample` or
`score` steps of a program commute. Kernels whose weight $\exp(w)$ comes from an unbounded
`score` expression $w$ are s-finite but in general not finite, which is why the finite kernels
alone do not suffice [@staton2017commutative]. Staton conjectures that s-finite kernels are not
the Kleisli category of a commutative monad on measurable spaces, so the pages that use them speak
of the category of s-finite kernels, not of a measure monad.

## 5. Fragments {#5-fragments}

QVR morphisms fall into three fragments, each interpreted in its own category.

| Fragment | Declared by | Category |
|---------|--------------------------------|-------------------|
| Finite weighted | `morphism` with `role=parameter` or `role=observed` | $\mathbf{Mat}(\mathcal{V})$ |
| Finite stochastic | `morphism` with `role=kernel` between finite sets and no `~ Family` initializer | $\mathbf{FinStoch}$ |
| Continuous | `morphism` with a `~ Family` initializer and a continuous codomain, and `object` declarations of continuous spaces | $\mathbf{BorelStoch}$ |

The fragments are related by embeddings. Giving a finite set all its subsets as events makes it a
standard Borel space, and this inclusion $\mathbf{FinSet} \hookrightarrow \mathbf{SBor}$ extends
to a full and faithful functor $\mathbf{FinStoch} \hookrightarrow \mathbf{BorelStoch}$. A Boolean
matrix with exactly one true entry in each row is a function, and these functions embed into
$\mathbf{FinStoch}$ as the deterministic kernels. A `discretize` declaration denotes a
kernel from a bounded interval to a finite set of bins, and an `embed` declaration denotes a
kernel in the opposite direction ([Morphisms §5](morphisms.md#5-fragment-transitions)).

## 6. Environments {#5-environments}

A *semantic environment* $\rho$ is a partial function from identifiers to denotations. It has
one part for each kind of name:

- $\rho_{\mathrm{obj}}$ maps object names to finite sets.
- $\rho_{\mathrm{spc}}$ maps space names to standard Borel spaces.
- $\rho_{\mathrm{mor}}$ maps morphism names to $\mathcal V$-matrices, stochastic matrices, or
  Markov kernels.
- $\rho_{\mathrm{cat}}$ maps the atoms of syntactic categories to their denotations in
  [weighted deduction](grammar.md).
- $\rho_{\mathrm{var}}$ maps the variables bound earlier in a `program` body to points of their
  spaces ([Programs as kernels](programs.md)).

We write $\rho[x \mapsto v]$ for $\rho$ extended with the binding of $x$ to $v$, and
$\llbracket \phi \rrbracket_{\rho}$ for the denotation of the phrase $\phi$ in $\rho$. We drop
$\rho$ when it is clear from the context.

## 7. Typing judgments

A [*typing judgment*](https://en.wikipedia.org/wiki/Typing_rule) $\Gamma \vdash \phi : \tau$
states that in the typing context $\Gamma$ the phrase $\phi$ has type $\tau$. The context
$\Gamma$ records, for each name in scope, its sort (object, space, or morphism with its domain and
codomain), where $\rho$ records its denotation. The compiler checks these judgments before it
builds any runtime object, and the denotation $\llbracket \phi \rrbracket_\rho$ is defined only
for well-typed phrases. [Typing and checking](typing.md#1-the-two-judgments) gives the judgments
of the source checker and of the checker of the
[Quivers Indexed Effect Core](../reference/qvr/index.md) (QIEC), the typed core language that QVR
elaborates to.
