# Denotational Semantics of Compositional Effects

QVR layers monadic and applicative effects over the residuated category
universe through two parallel typeclass towers (monads and arrows)
and a class-driven schema-lifting system. This page assigns denotations to
these constructs, states the assumptions for the lifting argument, and explains their conservative extension of the bare deduction fragment of
[Weighted Deduction Fragment](grammar.md).

An effect separates a request from the interpretation of that request. A
computation may request a random value; a handler can answer by drawing,
replaying a supplied value, or exploring several possibilities. The
computation then continues using the answer. This separation allows one
program description to participate in different execution strategies.

Two related uses appear here. The parser layer wraps grammatical types in
effect constructors and lifts composition rules to work on those wrapped
types (§§1–7). QIEC tracks explicit requests to lexical instances and checks
the handlers that answer them (§8). A typeclass interface lists available
operations; its mathematical laws state when rearranging those operations
preserves meaning.

For monads as a programming abstraction, see [@wadler1992essence]; for arrows,
[@hughes2000arrows]; for the interpretation of operation requests by handlers,
[@plotkin2009handlers]. These sources motivate the constructions, while the
implementation-specific contracts are stated below.

## 1. Setting

We work in the V-enriched semantic universe of
[Setting and Notation](setting.md). For an effect-typed parser we fix:

- A residuated category universe $\mathcal{C}$ realized as a finite
  set $\llbracket \mathrm{Cat} \rrbracket$. The carrier is built by
  the `FreeResiduated(generators, depth, ops)` object class: starting
  from a finite alphabet of atomic categories, it freely closes the
  carrier under the constructors `ops` (which include the residuation
  slashes $/$ and $\backslash$, the monoidal product $\otimes$, and
  the diamond modality $\diamondsuit$) up to a finite `depth` bound,
  yielding a finite-set object whose elements are the well-formed
  category expressions of bounded complexity.
- A finite list of declared effects $T_1, \dots, T_n$, each registered against
  one or more typeclass ABCs: `Functor`, `Applicative`, `Monad`, `Alternative`,
  `MonadPlus`, `Traversable`, `ArrowApply`, or `ArrowLoop`.

The *effect-typed category universe* is

$$
\widehat{\mathcal{C}} \;=\; \mathcal{C} \times \mathrm{EffectStack}_{\le d}(T_1, \dots, T_n),
$$

where $\mathrm{EffectStack}_{\le d}$ is the set of finite stacks of
effects of length at most $d$ (the `effect_depth` parameter of
`chart_fold`). Each chart cell at span $(i, j)$ is indexed by an
element of $\widehat{\mathcal{C}}$.

## 2. Typeclass denotations

The [typeclass hierarchy](../guides/effects.md#typeclass-hierarchy) distinguishes how a computation may use earlier results.
A functor maps a pure function over a context-bearing value, such as mapping
an increment over a possibly missing integer. An applicative combines
context-bearing inputs using a fixed computation shape. A monad additionally
allows the next computation to depend on the value just produced. For
probability distributions, that is the distinction between combining draws
with a fixed structure and choosing the next distribution after seeing an
earlier draw. The operations and equations below make those distinctions
precise for the representations supported here.

Each typeclass $\mathfrak{C}$ in
[`quivers.monadic.typeclasses`](../api/index.md#monadic-package) is
modeled by a panproto theory $\mathrm{Th}\mathfrak{C}$ in
[`quivers.monadic.theories`](../api/index.md#monadic-package); class
extension corresponds to theory inclusion via
$\mathrm{colimit}$. The sorts and operations of each theory are:

| $\mathrm{Th}\mathfrak{C}$ | Sorts | Operations | Equations |
|---------------------------|-------|------------|-----------|
| $\mathrm{ThFunctor}$ | $\mathrm{Carrier}$, $\mathrm{Hom}$ | $\mathrm{fmap\_obj}$, $\mathrm{fmap}$ | identity, composition |
| $\mathrm{ThApplicative}$ | (extends Functor) | $\mathrm{pure}$, $\mathrm{apply}$ | identity, homomorphism, interchange, composition |
| $\mathrm{ThMonad}$ | (extends Applicative) | $\mathrm{join}$ | left/right unit, associativity |
| $\mathrm{ThAlternative}$ | (extends Applicative) | $\mathrm{empty}$, $\mathrm{alt}$ | identity, associativity |
| $\mathrm{ThMonadPlus}$ | (extends Monad + Alternative) | (none new) | left zero |
| $\mathrm{ThTraversable}$ | (extends Functor + Foldable) | $\mathrm{traverse}$ | naturality, identity, composition |

An effect instance $T$ that registers against $\mathfrak{C}$ corresponds
to a panproto theory morphism $\mathrm{Th}\mathfrak{C} \to \mathrm{ThQVR}$
whose vertex map sends $\mathrm{Carrier}$ to the runtime carrier sort
of the residuated universe and whose operation map realizes the
class operations as morphisms of $\mathcal{V}\text{-}\mathbf{Rel}$ /
$\mathbf{Stoch}$ / $\mathbf{Kern}$.

## 3. Effect-typed schema denotations

A `SchemaDecl` whose domain or codomain mentions an `ObjectEffectApply`
node $T(X)$ is *parametric* in its free pattern variables. For each
choice of instantiation $X, Y \in \mathcal{C}$ the schema denotes a
morphism

$$
\llbracket r \rrbracket_{X, Y}
\;:\; T(X/Y) \otimes T(Y) \;\to\; T(X)
\quad \text{in } \widehat{\mathcal{C}};
$$

the family $\bigl(\llbracket r \rrbracket_{X, Y}\bigr)_{X, Y}$ is a
natural transformation of bifunctors (an applicative-style lifted
application). The applicative
*composition law* requires that combining lifted applications in different
bracketings give the same result. We assume a lawful applicative
interpretation in using this equation.

For Monad-typed schemas the denotation is the *Kleisli composition*:

$$
\llbracket \mathrm{bind}_T \rrbracket_{X, Y}
\;:\; T(X) \otimes (X \backslash T(Y)) \;\to\; T(Y),
$$

a morphism in the Kleisli category $\mathrm{Kl}(T)$ realizing the
canonical bind. The two presentations agree via the
[`kleisli` / `arrow_monad` bridges](../api/index.md#monadic-package).

## 4. Joint type-and-effect dispatch

A chart parser over a residuated universe $\mathcal{C}$ with declared
effects $\bar T = (T_1, \dots, T_n)$ has cells

$$
\mathrm{Chart}[i, j] \;\in\; \mathcal{V}^{|\widehat{\mathcal{C}}|}
$$

: a $\mathcal{V}$-valued vector indexed by effect-typed categories.
Schema firings populate cells according to four rules:

1. **Base firing.** A schema $r : \pi_1 \otimes \pi_2 \to \pi$ fires
   between cells $\mathrm{Chart}[i, k]$ and $\mathrm{Chart}[k, j]$ at
   the type coordinate when the effect-stack coordinates of both
   cells are empty (or when the schema has been auto-lifted to admit
   the cells' effect prefix; see rule 2).
2. **Lift firing.** When child cells carry effect prefixes
   $\bar{S}_1, \bar{S}_2$ that the schema's domain doesn't, the
   class-driven lifts of
   [`class_directed_lifts`](../api/index.md#stochastic-package) emit a
   schema whose domain wraps the base premises in the appropriate
   effect prefixes. The denotation of this lifted firing is exactly
   the natural-transformation composition of [§3](#3-effect-typed-schema-denotations)
   above.
3. **Handler firing.** When a cell holds a $T(\alpha)$ distribution
   and a $\mathrm{Handler}_{T \to S}$ is registered (a panproto theory
   morphism $\mathrm{Th}T \to \mathrm{Th}S$), applying the handler
   produces a fresh cell at $S(\alpha)$. In the special case
   $S = \mathrm{Identity}$, the handler eliminates the effect entirely.
4. **Commutation firing.** When a $\mathrm{DistributiveLaw}(T, U)$ is
   registered, the $\mathrm{swap}_{T,U}$ schema realizes the natural
   transformation $T \circ U \Rightarrow U \circ T$ on cells whose
   effect-stack ends with $T \cdot U$.

The four firings compose freely; the chart's CKY enumeration computes
the inside score over the joint search space.

### 4a. Inference rules for effect-typed firing

The four firings are inference rules in the proof system of the
chart. Let $\sigma : \mathrm{Var}(r) \to \mathcal{C}$ range over
substitutions of the schema's free pattern variables and let
$\bar S \cdot \bar T$ denote effect-stack concatenation. We write
$\mathrm{Chart}[i, j] \ni (\alpha; \bar T)$ for "the cell at span
$(i, j)$ holds an entry at type $\alpha \in \mathcal{C}$ stacked
under effect prefix $\bar T \in \mathrm{EffectStack}_{\le d}$".

$$
\frac{r : \pi_1 \otimes \pi_2 \to \pi \in \mathrm{Rules}
       \qquad \mathrm{Chart}[i, k] \ni (\sigma\pi_1; \varepsilon)
       \qquad \mathrm{Chart}[k, j] \ni (\sigma\pi_2; \varepsilon)}
     {\mathrm{Chart}[i, j] \ni (\sigma\pi; \varepsilon)}\ \textsc{Base}
$$

$$
\frac{r : \pi_1 \otimes \pi_2 \to \pi \in \mathrm{Rules}
       \qquad T \in \mathbf{Applicative}
       \qquad \mathrm{Chart}[i, k] \ni (\sigma\pi_1; \bar S \cdot T)
       \qquad \mathrm{Chart}[k, j] \ni (\sigma\pi_2; \bar S \cdot T)}
     {\mathrm{Chart}[i, j] \ni (\sigma\pi; \bar S \cdot T)}\ \textsc{Lift}_T
$$

$$
\frac{H : \mathrm{Th}T \to \mathrm{Th}S \in \mathrm{Handlers}
       \qquad S \neq \mathrm{Identity}
       \qquad \mathrm{Chart}[i, j] \ni (\alpha; \bar S \cdot T)}
     {\mathrm{Chart}[i, j] \ni (\alpha; \bar S \cdot S)}\ \textsc{Handle}_{T \to S}
$$

$$
\frac{H : \mathrm{Th}T \to \mathrm{Th}\,\mathrm{Identity} \in \mathrm{Handlers}
       \qquad \mathrm{Chart}[i, j] \ni (\alpha; \bar S \cdot T)}
     {\mathrm{Chart}[i, j] \ni (\alpha; \bar S)}\ \textsc{Eliminate}_T
$$

$$
\frac{\mathrm{swap}_{T, U} : T \circ U \Rightarrow U \circ T \in \mathrm{DistributiveLaws}
       \qquad \mathrm{Chart}[i, j] \ni (\alpha; \bar S \cdot T \cdot U)}
     {\mathrm{Chart}[i, j] \ni (\alpha; \bar S \cdot U \cdot T)}\ \textsc{Swap}_{T \mid U}
$$

In $\textsc{Base}$ and $\textsc{Lift}_T$ the substitution $\sigma$ is fixed by the cells' type coordinates (which must match $\pi_1$ and $\pi_2$ structurally); we elide its construction. $\textsc{Lift}_T$ requires only that $T$ inhabit $\mathbf{Applicative}$; every $\mathbf{Monad}$ extends $\mathbf{Applicative}$, so the rule covers monadic effects as a special case. $\textsc{Handle}_{T \to S}$ and $\textsc{Eliminate}_T$ are syntactic variants of the same handler firing, separated so each rule has a single conclusion shape.

The denotation of every derivation built from these four rules is
the corresponding composite natural transformation in
$\widehat{\mathcal{C}}$; Theorem [Lifting Adequacy](#5-adequacy-of-effect-lifting)
below identifies the chart's inside score with that composite.

## 5. Adequacy of effect lifting

**Conditional lifting claim.** *Let $r$ be a base SchemaDecl and
$T$ an effect instance implementing the typeclasses
$\mathfrak{C}_1, \dots, \mathfrak{C}_k$. Assume their stated laws hold,
the base interpretation and schema lifts preserve the specified operations,
and the chart aggregation satisfies the applicable semiring and termination
or convergence conditions. Then for every cell
configuration over $\widehat{\mathcal{C}}$ that admits a derivation
under the lifted schemas
$\mathrm{class\_directed\_lifts}(r, T)$, the chart parser's
inside score equals the categorical denotation of the composite
natural transformation through $\mathrm{Th}\mathfrak{C}_1, \dots,
\mathrm{Th}\mathfrak{C}_k$ in the appropriate enriched category.*

*Argument by construction.* For the principal lifting cases:

- For $\mathfrak{C} = \mathbf{Applicative}$, the lifted schema is the
  applicative *composition law*; the chart firing realizes the
  natural transformation
  $T(\pi_1) \otimes T(\pi_2) \Rightarrow T(\pi)$ obtained by
  postcomposing the base $r$ with $T$'s `apply`. The applicative
  laws (identity, homomorphism, interchange, composition) imply
  that successive applicative lifts compose to the same denotation
  irrespective of bracketing; the chart's left-associative scheduling
  is thus one valid realization among many equivalent choices.
- For $\mathfrak{C} = \mathbf{Monad}$, the bind-lift is the Kleisli
  composition of the base schema; the monad laws imply
  $\mathrm{join} \circ \mathrm{fmap}(\mathrm{join}) = \mathrm{join} \circ \mathrm{join}$,
  so iterated bind-lifts compose to the iterated Kleisli of the base
  rule.
- For $\mathfrak{C} = \mathbf{Alternative}$, the alt-lift realizes
  $T \otimes T \Rightarrow T$ via $T$'s `alt`; the alternative laws
  give the equational theory of the resulting non-deterministic
  derivation tree.

The inside-score recurrence aggregates these derivations under the conditions
in [Weighted Deduction Fragment §7](grammar.md#7-strategy-independence).
Each lifting step uses the laws of the selected typeclass; the resulting
derivation also requires the recurrence conditions just cited.

## 6. Free-monad denotation

A free computation records requests before deciding how to answer them.
A leaf contains a completed value. An operation node contains a request and
a [continuation for each possible response](https://dippl.org/chapters/03-enumeration.html#exploring-a-random-computation),
describing what happens next [@goodman2014design].
A handler interprets the tree by replacing each request with its chosen
behavior. This is a fold over the request structure, analogous to evaluating
an expression tree with a chosen interpretation of its operators.

For an effect signature
$\Sigma = \{\mathrm{op}_1 : P_1 \to R_1, \dots, \mathrm{op}_m : P_m \to R_m\}$,
the free monad $\mathrm{FreeMonad}(\Sigma)$ has the denotation of the
*finitely-branched signature trees* over $\Sigma$:

$$
\llbracket \mathrm{FreeMonad}(\Sigma)(A) \rrbracket
\;=\; \bigcup_{n \ge 0} \mathrm{Tree}_{\Sigma, n}(A),
$$

where $\mathrm{Tree}_{\Sigma, n}(A)$ is the set of $\Sigma$-shaped
trees of depth $\le n$ whose leaves are values in $A$, with trees identified across the nested depth bounds. The runtime
realization truncates at the `depth` parameter of the enclosing
`deduction { … }` block to keep the carrier finite.

A handler $H : \Sigma \to S$ is a panproto theory morphism
$\mathrm{Th}\Sigma \to \mathrm{Th}S$; the denotation of
$H.\mathrm{run} : \mathrm{FreeMonad}(\Sigma)(A) \to S(A)$ is the
universal map of free monads (the *fold* over the operation tree
that interprets each internal node through the corresponding
clause of $H$).

The operation-tree view separates a computation from its interpretation
[@plotkin2009handlers]. Binding a continuation can increase a tree's depth beyond the runtime
bound. Thus the bounded representation covers those substitutions whose
resulting trees remain within that bound.

## 7. Bridges and conservativity

The `kleisli` and `arrow_monad` bridges of
[`quivers.monadic.bridges`](../api/index.md#monadic-package) realize
the canonical theory morphisms:

$$
\mathrm{kleisli} \;:\; \mathrm{ThMonad} \;\to\; \mathrm{ThArrowApply},
\qquad
\mathrm{arrow\_monad} \;:\; \mathrm{ThArrowApply} \;\to\; \mathrm{ThMonad}
$$

These functions implement the ArrowApply/Monad constructions described by
[@hughes2000arrows]. In Quivers,
$\mathrm{arrow\_monad} \circ \mathrm{kleisli} \;\cong\; \mathrm{id}_{\mathrm{ThMonad}}$
and $\mathrm{kleisli} \circ \mathrm{arrow\_monad} \;\cong\; \mathrm{id}_{\mathrm{ThArrowApply}}$
are naturally isomorphic via the canonical
$1 \otimes A \cong A$ unitor that absorbs the
``ArrowMonad(a)(A) = a(1, A)`` boxing. Consequently the
arrow-side and monad-side presentations of any effect agree on
denotation up to the canonical isomorphism. When the underlying
monad is also a *monad-fix*, admitting a least-fixed-point
operator, the `chart_fold` runtime's `loop_arr` construction agrees
denotationally with the Kleisli-iteration of the binary step; for
monads without `mfix`, the arrow-side trace is the well-defined
realization and the Kleisli side is undefined.

**Conservativity.** When the declared effects list $\bar T$ is empty,
the joint type-and-effect dispatch of [§4](#4-joint-type-and-effect-dispatch)
collapses to the classical residuated-grammar dispatch of
[Weighted Deduction Fragment §6](grammar.md#6-chart-denotation), and
the inside score is the standard Lambek / CCG / multimodal-TLG
chart-parser denotation. The effects framework is thus a strict
extension over the bare grammar fragment.

## 8. QIEC rows and lexical handlers

Consider two state cells, `counter` and `total`, both storing integers.
Their interfaces match, but their identities differ: handling a request to
`counter` must leave a request to `total` untouched. A row lists these
instance identities, so a handler can select the intended cell. The
[typing rules](typing.md#7-handlers-and-scoped-instances) track which requests
remain after handling and how often each resumption may be used.

When a handler answers a request, its resumption represents the rest of the
computation. A draw handler can resume once with a random value; an
enumeration handler can resume once for each finite alternative. “Deep”
means the handler is reinstated for requests in the resumed computation;
requests made by the handler clause itself are offered outward. These rules
explain why both nesting order and the identity of the requested instance
matter.

QIEC is the explicitly typed account of effects every computation of a module
is checked under, the elaborated `program` included. An applied
interface $E[\bar a]$ describes an operation family, while a lexical instance
$i:E[\bar a]$ supplies the identity that appears in a row. The cells
`counter` and `total` thus occupy separate entries. A row has the schematic form
$\{i_1,\ldots,i_n\mid\rho\;\mathsf{lacks}\;j_1,\ldots,j_m\}$; its tail supports
effect polymorphism, and its lacks set records negative constraints needed by
row extension and handler elimination.

The reference evaluator gives QIEC computations a deep, lexical handler
semantics. Resuming a captured continuation reinstalls the matched handler,
while an effect performed by the clause body is offered to outer handlers.
Dispatch compares the lexical instance, applied interface, and operation.
Thus an unrelated instance of the same interface cannot be intercepted by
accident. Dynamic grade checks enforce the declared number of resumptions, and
multi-shot resumptions extend their trace address with a branch path.

This account of rows connects the source contracts of state, abort, random choice,
scoring, nondeterministic choice, and weighted accumulation. A handler's
clauses are authored in the source or remain process-local attachments. Pyro,
NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church lower the same typed
computation graph through corresponding implementations of the stable-ID
runtime ABI. Conformance tests exercise these implementations in the target
runtimes against the reference machine at clamped points. Each dispatches by
stable instance and operation identifiers, preserving the separate state cells
in the example above. Stan lowers a computation with an empty closed
effect row, an empty static telescope, and a scalar result as a user-defined
function, conditionals, calls, and recursion included; BUGS and JAGS refuse
every call. The static targets issue precise capability diagnostics for the
rest. The
[QIEC developer note](../developer/qiec.md) gives the executable boundary.

The effect handlers of [`quivers.effects`](../api/effects/index.md) are
lexical handlers of this calculus. A `MonadicProgram` runs on the reference
machine as the kernel computation
[`program_kernel`](../api/effects/program_module.md) encodes it to: each host
step is an instance of the prelude's `Compute` interface performed on the
environment of every value bound so far, each draw is the canonical
`Random.sample` request on the program's `random` instance, and each site's
density is a `Score.add` the answering handler emits on the program's `score`
instance. A handler on the `with` stack installs a lexical handler of one of
those instances; the run nests them in a fixed order, the score accumulator
outermost, then every score transformer, then the scoring draw and the run's
observations, then every handler of `random` in stack order, then the
parameter store and the `param` handlers layered over it, then the host
steps. Two consequences follow. An inner handler of `random` sees a site's
request before any outer one and sees the outer answer flow back through its
resumption, so a trace inside a clamp records the clamped value. A site's
density is emitted outward by whichever handler answered it, so every score
transformer reaches every site, and a trace reads each site's density from
what reached the accumulator under the site's provenance, which the kernel
carries on every derived request as the site's static identity and dynamic
address.

## References

For algebraic operations and their interpretation, see [@plotkin2003algebraic]. Handlers are developed mathematically in [@plotkin2009handlers] and through programming examples in [@bauer2015programming]. Arrows generalize the composition interface of monads [@hughes2000arrows]. For applications to natural-language interpretation, see the forthcoming book [@bumford2026effect] and its [author preprint](https://arxiv.org/abs/2504.00316).
