# Setting and Notation

This page fixes the semantic universe in which every QVR phrase will be interpreted.

Start with a table $f[x,y]$ whose rows name inputs and columns name outputs.
An entry might mean a probability, a compatibility score, or a travel cost.
To follow $f$ by a second table $g$, first combine the two entries along each
route $x\to y\to z$, then combine all routes through the possible $y$ values.
Those two choices are the operations $\otimes$ and $\bigoplus$ below.

For probabilities, a row $f[x,-]=(0.7,0.3)$ and a column
$g[-,z]=(0.2,0.8)$ give $0.7\cdot0.2+0.3\cdot0.8=0.38$.
For travel costs, the corresponding calculation would add the costs along
each route and take the minimum across routes. The table shape is the same;
the interpretation of its entries determines the calculation.

The general categorical account of composition and structure-preserving maps
is developed in [@maclane1998categories]; enriched categories replace ordinary
hom-sets with objects of a specified enriching category [@kelly1982basic].

## 1. Algebras as enrichment bases

The [weather-table calculation](../getting-started/concepts.md#2-composition-eliminates-an-intermediate-value) supplies a finite example of the operations below. Milewski’s [“Preorders” and “Metric Spaces” examples](https://bartoszmilewski.com/2017/05/13/enriched-categories/) explain how truth values and distances can take the place of ordinary hom-sets [@milewski2017enriched].

The implementation calls any instance of `quivers.core.algebras.Algebra` a *QVR algebra*. The class supplies `tensor_op`, `join`, `meet`, `negate`, `unit`, and `zero`; the equations below state the laws required of those operations. A unital [quantale](https://ncatlab.org/nlab/show/quantale) supplies the stronger structure needed for the usual enriched-category results: a complete lattice $V$, a monoid product $\otimes$, and arbitrary joins $\bigvee$ over which $\otimes$ distributes on both sides:

$$
a \otimes \bigvee_{i \in I} b_i \;=\; \bigvee_{i \in I} (a \otimes b_i),
\qquad
\Bigl(\bigvee_{i \in I} a_i\Bigr) \otimes b \;=\; \bigvee_{i \in I} (a_i \otimes b).
$$

We write $\bigoplus$ for the implementation's reduction operation. It is a lattice join only for the built-ins whose reduction is idempotent. Results that invoke arbitrary joins, bottom elements, or distributivity thus carry an explicit quantale hypothesis.

For the symmetric monoidal relation interpretation below, assume that the
quantale is commutative and unital. Quantales are standard enrichment bases. QVR's eleven built-ins are cataloged in [Algebras and base change](algebras.md), but several are finite-reduction semirings or t-norm/t-conorm pairs rather than unital quantales. QVR evaluates their tensor contractions through the shared `Algebra` interface. For a categorical result, use the hypotheses stated with that result and check them against the operations in the catalog.

## 2. V-enriched relations

Read $\mathcal V$ as the collection of allowed entry values together with
the operations for combining them. A relation valued in $\mathcal V$ is thus
a weighted table. The term *enriched* records that relationships carry these
values instead of only a yes/no answer. The categorical laws require the
algebraic hypotheses stated above; the table calculation itself needs fewer
assumptions.

Let $X, Y$ be finite sets. A *$\mathcal{V}$-relation* from $X$ to $Y$ is a function

$$
r : X \times Y \to V.
$$

We denote by $\mathcal{V}\text{-}\mathbf{Rel}$ the category whose objects are finite sets and whose hom-sets $\mathcal{V}\text{-}\mathbf{Rel}(X, Y)$ consist of the matrices $V^{X\times Y}$. With pointwise order, these hom-sets are complete lattices, and composition preserves joins in each argument; this is the quantaloid structure [@rosenthal1996quantaloids]. Composition is the $\mathcal{V}$-matrix product:

$$
(r ; s)(x, z) \;=\; \bigoplus_{y \in Y} r(x, y) \otimes s(y, z),
\qquad r : X \to Y,\ s : Y \to Z.
$$

Identities are the indicator $\mathcal{V}$-relations $1_X(x, x') = \mathbf{1}$ if $x = x'$ and $\bot$ otherwise, where $\bot$ is the bottom of the lattice. The category $\mathcal{V}\text{-}\mathbf{Rel}$ is symmetric monoidal: tensor product is given by

$$
(r \boxtimes s)((x_1, x_2), (y_1, y_2)) \;=\; r(x_1, y_1) \otimes s(x_2, y_2).
$$

## 3. Standard Borel spaces and Markov kernels

The primer’s [sample, density, and interval-probability example](../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample) explains what a kernel assigns to an input. The [continuous composition guide](../guides/continuous-spaces.md#composition-sampledcomposition) shows the corresponding sampling and density calls.

A table cannot list every possible real-valued output. A kernel instead
answers questions about sets of outputs: given input $s$, what is the
probability that the output lies in an interval $B$? For a conditional Normal
distribution, $k(s,[0,1])$ is the Normal probability between zero and one at
the parameters selected by $s$. The measurable-space conditions below make
these probabilities and their integrals well defined. They cover familiar
finite sets, Euclidean spaces, and the structured spaces used by QVR.

A *standard Borel space* is a measurable space $(S, \Sigma)$ isomorphic to a Borel subset of a Polish space. Write $\mathbf{SBor}$ for the category of standard Borel spaces with measurable maps.

A *Markov kernel* from $S$ to $T$ is a function $k : S \times \Sigma_T \to [0, 1]$ such that $k(s, \cdot)$ is a probability measure on $T$ for every $s$, and $k(\cdot, B)$ is measurable for every $B \in \Sigma_T$. The category $\mathbf{Kern}$ has standard Borel spaces as objects and Markov kernels as morphisms, with composition

$$
(k_1 ; k_2)(s, C) \;=\; \int_T k_2(t, C) \, k_1(s, \mathrm{d}t).
$$

It is the Kleisli category of the *Giry monad* $\mathcal{G} : \mathbf{SBor} \to \mathbf{SBor}$, $S \mapsto \mathcal{G}(S) = \{\mu \mid \mu \text{ probability measure on } S\}$.

We write $\mathbf{Stoch}$ for the full subcategory of $\mathbf{Kern}$ on finite discrete spaces. Its morphisms are stochastic matrices. Its objects are finite, while the distributions they support form continuous families: on a two-element set, a distribution is determined by any $p\in[0,1]$. Thus the probability-measure construction takes that finite set outside $\mathbf{FinSet}$.

Giry develops the measure-theoretic construction [@giry1982categorical].
Fritz supplies a complementary account in terms of compositional probability
and conditional independence [@fritz2020synthetic].

## 4. The three semantic strata

QVR phrases inhabit three strata, each interpreted in a distinct ambient category.

| Stratum | Source of morphism declaration | Ambient category |
|---------|--------------------------------|-------------------|
| Discrete $\mathcal{V}$-enriched | `latent`, `observed` (no `~ Family` clause) | $\mathcal{V}\text{-}\mathbf{Rel}$ |
| Stochastic | `kernel` between finite-set types (no `~ Family` clause) | $\mathbf{Stoch}$ |
| Continuous | `morphism ... ~ Family` with a continuous codomain; continuous `object` declarations | $\mathbf{Kern}$ |

The three strata are not independent: the inclusion $\iota : \mathbf{FinSet} \hookrightarrow \mathbf{SBor}$ (every finite set is canonically a standard Borel space with the discrete $\sigma$-algebra) lifts to a faithful embedding $\mathbf{Stoch} \hookrightarrow \mathbf{Kern}$, and the *functional* sub-category $\mathcal{V}_{\mathbb{B}}\text{-}\mathbf{Rel}_{\mathrm{fun}}$ of row-deterministic Boolean relations embeds into $\mathbf{Stoch}$. The `discretize` and `embed` declarations denote the Giry-monad–level transition between strata; see [Morphisms §5](morphisms.md#5-stratum-transitions).

## 5. Environments

An environment is a lookup table for names. If the source declares `State`,
then $\rho_{\mathrm{obj}}(\mathsf{State})$ is the set assigned to that name.
The double brackets $\llbracket e\rrbracket_\rho$ mean “interpret expression
$e$ using these bindings.” Subscripts distinguish which kind of name is being
looked up.

A *semantic environment* $\rho$ is a partial function from identifiers to denotations, partitioned into:

- $\rho_{\mathrm{obj}}$: finite-set objects;
- $\rho_{\mathrm{spc}}$: standard Borel spaces;
- $\rho_{\mathrm{mor}}$: morphisms (discrete, stochastic, or continuous);
- $\rho_{\mathrm{cat}}$: category atoms in the grammar fragment;
- $\rho_{\mathrm{rv}}$: random variables bound earlier in a `program` body, each carrying its current Kleisli arrow (see [Programs §1](programs.md#1-the-giry-monad)).

Following established practice, we write $\rho[x \mapsto v]$ for the environment obtained by extending $\rho$ with the binding $x = v$. The denotation of a phrase $\phi$ in environment $\rho$ is written $\llbracket \phi \rrbracket_{\rho}$; we elide $\rho$ when the binding context is clear.

## 6. Well-typedness

The [two checking judgments](typing.md#1-the-two-judgments) identify which environment supplies each name and which conditions the compiler checks. The primer’s [notation table](../getting-started/concepts.md#7-reading-the-notation-and-finding-the-next-explanation) explains the brackets, turnstile, and hom-set notation.

Read $\Gamma\vdash\phi:\tau$ as “given the declarations in $\Gamma$, the
phrase $\phi$ has type $\tau$.” A typing context records what a name may
refer to; a semantic environment records its meaning. For composition, the
practical question is whether the outputs of the first operation match the
inputs of the next.

The QVR type system is given by judgments of the form

$$
\Gamma \vdash \phi : \tau,
$$

where $\Gamma$ is a typing context analogous to $\rho$ but tracking only sorts (object / space / morphism with domain–codomain types), and $\tau$ is the assigned sort. The compiler resolves and checks the supported phrase forms before constructing runtime objects. The judgments in these pages specify those resolution and checking steps.

The [Typing](typing.md) chapter records the declarative judgments and relates them to checks in `quivers.dsl.compiler`. [Correspondence and limitations](adequacy.md) separates equations exercised by tests from broader mathematical assumptions.
