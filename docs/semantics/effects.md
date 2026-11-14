# Effects and type classes

Effects enter QVR in two places. A chart parser may type expressions with effects, such as a
continuation for a quantifier or a set of alternatives for an indefinite, and lift its rules to
these types by the operations of a functor, an applicative functor, or a monad, as in
effect-driven interpretation [@bumford2026effect; @charlow2025scope]. The Quivers Indexed Effect
Core (QIEC) checks every computation of a module against an effect row and runs it with
[effect handlers](https://en.wikipedia.org/wiki/Effect_system) [@plotkin2009handlers] (§8). The
library's Python packages `quivers.monadic` and `quivers.arrows` provide the
[type classes](https://en.wikipedia.org/wiki/Type_class) behind the first [@wadler1989adhoc].

An *algebraic effect* separates a request from its interpretation [@plotkin2003algebraic]: a
computation requests a random value, and a handler answers by drawing one, by replaying a given
value, or by trying every value. The computation then continues with the answer, so one program
can be run under different strategies. A type class lists the operations a type provides, and its
laws say which rearrangements of those operations preserve meaning.

## 1. Setting {#1-setting}

The definitions of [Setting and notation](setting.md) apply. A parser whose categories carry effect types fixes:

- a finite set $\mathcal C = \llbracket \mathrm{Cat} \rrbracket$ of categorial types, built by
  `FreeResiduated(generators, depth, ops)` ([Schemas §4](schemas.md#4-the-free-residuated-universe));
- a finite list of declared effects $T_1, \dots, T_n$, each an instance of one or more of the type
  classes `Functor`, `Applicative`, `Monad`, `Alternative`, `MonadPlus`, `Traversable`,
  `ArrowApply`, and `ArrowLoop`.

A chart entry has a categorial type and a stack of effects applied to it, of length at most $k$,
the `effect_depth` of `chart_fold`. Write

$$
\widehat{\mathcal{C}} = \mathcal{C} \times \mathrm{Stack}_{\le k}(T_1, \dots, T_n)
$$

for the set of such pairs, where $\mathrm{Stack}_{\le k}$ is the set of lists of at most $k$ effects.
Each chart cell at a span $(i, j)$ holds weights indexed by $\widehat{\mathcal C}$.

## 2. Type classes {#2-typeclass-denotations}

The [type class guide](../guides/effects.md#typeclass-hierarchy) describes the classes in use. A
[functor](https://en.wikipedia.org/wiki/Functor_(functional_programming)) maps a function over a value
in a context. An [applicative functor](https://en.wikipedia.org/wiki/Applicative_functor) combines
several values in contexts by a fixed shape of computation [@mcbride2008applicative]. A
[monad](https://en.wikipedia.org/wiki/Monad_(functional_programming)) also lets the next computation
depend on the value just produced [@wadler1992essence]. For probability distributions, this is the
difference between combining independent draws and choosing the next distribution after seeing a
draw.

Each class in [`quivers.monadic.typeclasses`](../api/monadic/typeclasses.md) is an abstract Python
class. Beside each, [`quivers.monadic.theories`](../api/monadic/theories.md) has a record
(`TypeclassTheory`) that lists the sorts, operations, and laws of the class as text, with the class
it extends:

| Record | Extends | Operations | Laws |
|---------------------------|-------|------------|-----------|
| `ThFunctor` | | `fmap` | identity, composition |
| `ThApplicative` | `ThFunctor` | `pure`, `apply` | identity, homomorphism, interchange, composition |
| `ThMonad` | `ThApplicative` | `join` | left unit, right unit, associativity |
| `ThAlternative` | `ThApplicative` | `empty`, `alt` | identity, associativity |
| `ThMonadPlus` | `ThMonad`, `ThAlternative` | | left zero |
| `ThTraversable` | `ThFunctor`, `ThFoldable` | `traverse` | naturality, identity, composition |

An instance of a class is a Python object that implements its methods, with its carrier a finite set
and its operations morphisms of [Morphisms](morphisms.md). The library does not check the laws when
an instance is registered. [`quivers.monadic.laws`](../api/monadic/laws.md) has functions that test
the functor and monad laws of an instance at given objects.

## 3. Lifted rules {#3-effect-typed-schema-denotations}

For each declared effect $T$ and each rule of the parser, the function
[`class_directed_lifts`](../api/stochastic/effect_lifts.md) produces lifted rule schemata according to
the classes $T$ belongs to. In QVR's slash notation, with the result first
([Schemas §3.1](schemas.md#31-residuation-slashes)):

| Class of $T$ | Schema | Shape |
|---|---|---|
| `Applicative` | `pure_T` | $X \to T(X)$ |
| `Applicative` | `apply_T` | $T(X / Y) \otimes T(Y) \to T(X)$ |
| `Monad` | `bind_T` | $T(X) \otimes (T(Y) / X) \to T(Y)$ |
| `Alternative` | `alt_T` | $T(X) \otimes T(X) \to T(X)$ |

for all categorial types $X$ and $Y$. `pure_T` and `apply_T` are the unit and the lifted application
of an applicative functor, the lifting of function application that effect-driven interpretation uses
[@mcbride2008applicative; @bumford2026effect]. `bind_T` is monadic bind, whose second premise
$T(Y) / X$ is the categorial type of a function from $X$ to $T(Y)$. It is the bind of $T$, not a
morphism of the Kleisli category of $T$. A `MonadPlus` instance gets the lifts of `Monad` and of
`Alternative`. A bare `Functor`, `ArrowApply`, and `ArrowLoop` add no schemata. The base
schema contributes the generated names, while the four displayed shapes remain fixed.

The chart combines lifted applications in the order it builds spans. The applicative laws, in
particular the composition law, make the result independent of that order when the instance satisfies
them [@mcbride2008applicative]. For an instance that does not, the chart's order fixes the value.

## 4. Firing rules {#4-joint-type-and-effect-dispatch}

A chart parser over $\mathcal C$ with declared effects $T_1, \dots, T_n$ has cells

$$
\mathrm{Chart}[i, j] \in V^{\widehat{\mathcal{C}}},
$$

vectors of weights indexed by categories with effect types. Write $\mathrm{Chart}[i, j] \ni (\alpha ; \bar T)$
when the cell at span $(i, j)$ has an entry at the type $\alpha \in \mathcal C$ under the effect stack
$\bar T$, write $\bar S \cdot \bar T$ for the concatenation of stacks and $\epsilon$ for the empty
stack, and let $\sigma$ range over substitutions of categorial types for the variables of a schema.
Cells are filled by three kinds of firing, each an inference rule of a
[weighted deduction](grammar.md):

$$
\frac{r : \pi_1 \otimes \pi_2 \to \pi \in \mathrm{Rules}
       \qquad \mathrm{Chart}[i, k] \ni (\sigma\pi_1 ; \epsilon)
       \qquad \mathrm{Chart}[k, j] \ni (\sigma\pi_2 ; \epsilon)}
     {\mathrm{Chart}[i, j] \ni (\sigma\pi ; \epsilon)}\ \textsc{Base}
$$

$$
\frac{r : \pi_1 \otimes \pi_2 \to \pi \in \mathrm{Rules}
       \qquad T \in \mathbf{Applicative}
       \qquad \mathrm{Chart}[i, k] \ni (\sigma\pi_1 ; \bar S \cdot T)
       \qquad \mathrm{Chart}[k, j] \ni (\sigma\pi_2 ; \bar S \cdot T)}
     {\mathrm{Chart}[i, j] \ni (\sigma\pi ; \bar S \cdot T)}\ \textsc{Lift}_T
$$

$$
\frac{\lambda : T \circ U \Rightarrow U \circ T \in \mathrm{DistributiveLaws}
       \qquad \mathrm{Chart}[i, j] \ni (\alpha ; \bar S \cdot T \cdot U)}
     {\mathrm{Chart}[i, j] \ni (\alpha ; \bar S \cdot U \cdot T)}\ \textsc{Swap}_{T, U}
$$

In $\textsc{Base}$ and $\textsc{Lift}_T$ the substitution $\sigma$ is determined by matching the types
of the two cells against $\pi_1$ and $\pi_2$. $\textsc{Lift}_T$ is the firing of the lifted schemata of
§3, and since every monad is an applicative functor it covers monadic effects. $\textsc{Swap}_{T, U}$
fires the schema `swap_TU : T(U(X)) -> U(T(X))` that
[`make_swap_schema`](../api/stochastic/effect_lifts.md) builds from a registered
[distributive law](https://ncatlab.org/nlab/show/distributive+law) $\lambda$, a natural transformation
that exchanges the two monads compatibly with both [@beck1969distributive]. The chart sums the weights
of all derivations built from these firings under the conditions of
[Weighted deduction §7](grammar.md#7-strategy-independence).

A chart does not eliminate an effect prefix by invoking a handler. These prefixes annotate
semantic types; QIEC effect instances and their lexical handlers belong to the computation
runtime described in §8.

## 5. When lifting preserves the base grammar

Let $r$ be a rule and $T$ an effect instance of the classes $\mathfrak C_1, \dots, \mathfrak C_k$. If the
instance satisfies the laws of these classes and the chart's semiring and convergence conditions hold
([Weighted deduction §5](grammar.md#5-goal-depth-and-convergence)), then the weight the chart assigns to
an entry whose category has an effect type is the sum, over its derivations, of the composites of base rules with the
operations of $T$ that the derivation uses, and it does not depend on the order in which the chart
combined them. The argument is by cases on the class:

- For an applicative $T$, a lifted firing composes the base rule with `apply` of $T$. By the applicative
  laws, successive lifts give the same composite under every bracketing [@mcbride2008applicative], so
  the chart's left-to-right order is one of several equal choices.
- For a monad $T$, a bind firing is Kleisli composition with the base rule, and the associativity law
  $\mathrm{join} \circ \mathrm{fmap}(\mathrm{join}) = \mathrm{join} \circ \mathrm{join}$ makes iterated binds
  equal to the iterated Kleisli composite.
- For an alternative $T$, `alt` combines two analyses of one span, and its identity and associativity
  laws make the combination independent of grouping.

This is a statement about the denotation the chart computes, given lawful instances. It is not
adequacy in the sense of programming language semantics, which relates an operational semantics to a
denotational one [@plotkin1977lcf].

## 6. Free monads and handlers

A free computation records its requests before deciding how to answer them: a leaf holds a finished
value, and an operation node holds a request and a continuation for each possible answer
[@swierstra2008data]. A handler interprets the tree by replacing each request with its own behavior,
a fold over the tree [@plotkin2009handlers].

For an effect signature $\Sigma = \{\mathrm{op}_1 : P_1 \to R_1, \dots, \mathrm{op}_m : P_m \to R_m\}$ with
finite parameter sets $P_i$ and finite answer sets $R_i$, the
[free monad](https://ncatlab.org/nlab/show/free+monad) on $\Sigma$ sends a set $A$ to the trees with
leaves in $A$ whose operation nodes $\mathrm{op}_i(p, \kappa)$ have a parameter $p \in P_i$ and a
continuation $\kappa : R_i \to$ trees, so a node branches $|R_i|$ ways:

$$
\mathrm{Free}_\Sigma(A) = \bigcup_{n \ge 0} \mathrm{Tree}_{\Sigma, n}(A),
$$

with $\mathrm{Tree}_{\Sigma, n}(A)$ the trees of depth at most $n$. The class
[`FreeMonad`](../api/monadic/algebraic.md) represents only the trees of depth at most its parameter
`depth` (default $3$), as one finite set. That set is not closed under bind, which can make a tree
deeper, so the representation covers the binds whose results stay within the bound.

A handler of $\Sigma$ into a monad $S$ gives a return clause $A \to S(A)$ and, for each operation, a
clause that answers a request using the interpreted continuation. It makes $S(A)$ a model of the
operations of $\Sigma$, and running the handler, `Handler.run`, is the unique homomorphism from the free
model $\mathrm{Free}_\Sigma(A)$ to it, the fold over the tree [@plotkin2009handlers].

## 7. Arrows and monads {#7-bridges-and-conservativity}

An [arrow](https://en.wikipedia.org/wiki/Arrow_(computer_science)) generalizes the composition of
functions [@hughes2000arrows]. Arrows with an application operation (`ArrowApply`) correspond to monads:
the Kleisli arrows of a monad form an `ArrowApply`, and an `ArrowApply` defines a monad whose values of
type $A$ are its arrows from the unit to $A$ [@hughes2000arrows]. The functions `kleisli` and
`arrow_monad` of [`quivers.monadic.bridges`](../api/monadic/bridges.md) implement the two directions.

`ArrowLoop` is the class of arrows with a feedback operation `loop`. For the Kleisli arrows of a monad,
`loop` requires the monad to support value recursion, the class `MonadFix` [@erkok2000recursive;
@paterson2001arrows], so an `ArrowLoop` instance for Kleisli arrows exists only for such monads.

When the list of declared effects is empty, only $\textsc{Base}$ fires, and the parser is the
categorial grammar parser of [Weighted deduction §6](grammar.md#6-chart-denotation) with the rules it is
given. The effects add firings and change no weight that the empty list would compute, so the
parser with effect types is a [conservative extension](https://en.wikipedia.org/wiki/Conservative_extension) of
the plain one.

## 8. QIEC rows and handlers

Two state cells `counter` and `total`, both holding integers, have the same interface and different
identities: handling a request to `counter` must leave a request to `total` alone. An *effect instance*
is one named occurrence of an effect interface, and its handler intercepts only its own requests
[@bauer2015programming; @biernacki2020binders]. An *effect row* lists the instances a computation may
use [@leijen2014koka; @hillerstrom2016liberating]. The [typing rules](typing.md#7-handlers-and-scoped-instances)
track which requests remain after handling and how many times each continuation may be resumed.

When a handler answers a request, its continuation is the rest of the computation. A handler for draws
resumes once with a random value, and a handler for enumeration resumes once for each alternative. A
*deep* handler is reinstalled around the resumed computation, while requests made by the handler's own
clause go to the handlers outside it [@kammar2013handlers]. So the order in which handlers nest, and the
instance a request names, both change the result.

Every computation of a module, including each elaborated `program`, is checked under QIEC. An applied
interface $E[\bar a]$ describes a family of operations, and an instance $i : E[\bar a]$ gives the identity
that appears in a row, so `counter` and `total` are separate entries. A row has the form
$\{i_1, \ldots, i_n \mid \rho\ \mathsf{lacks}\ j_1, \ldots, j_m\}$: the tail $\rho$ is a row variable, so
that functions may be polymorphic in the rest of the row
([row polymorphism](https://en.wikipedia.org/wiki/Row_polymorphism)), and the `lacks` list records the
instances the rest of the row may not contain, which extending a row and eliminating a handler need.

### 8.1 Implementation

The [reference machine](../reference/qvr/index.md), the library's interpreter for QIEC, implements deep
handlers bound to instances. Resuming a captured continuation reinstalls the handler that matched, and an
effect performed in a clause body goes to the outer handlers. A request matches a handler by its instance,
its applied interface, and its operation, so another instance of the same interface is never intercepted.
The machine checks at run time that a clause resumes no more often than its declared resumption grade
allows, and a continuation resumed more than once extends the address of its trace with a branch.

These rules cover the interfaces of state, abort, random choice, scoring, nondeterministic choice, and
weighted accumulation. A handler's clauses are written in QVR, or held only in the running Python process
and given by a provider. Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and Church run the same typed
computations through implementations of the QIEC runtime for each target, which dispatch by the stable
identifiers of instances and operations, so the two state cells above stay separate. Tests compare these
implementations with the reference machine at fixed parameter values. Stan expresses a computation with an
empty closed row, no static parameters, and a scalar result as a user-defined function, including
conditionals, calls, and recursion, and BUGS and JAGS refuse every call. For the other forms the static
targets report a `qiec:capability:*` diagnostic. The [QIEC developer note](../developer/qiec.md) gives the
details.

The effect handlers of [`quivers.effects`](../api/effects/index.md) are handlers of this calculus, in the
style of inference by effect handlers for probabilistic programs [@scibior2018functional; @nguyen2023effect]. A
`MonadicProgram` runs on the reference machine as the computation
[`program_kernel`](../api/effects/program_module.md) builds from it: each step computed in Python is a
request to the `Compute` interface of the prelude on the values bound so far, each draw is a request
`Random.sample` to the program's `random` instance, and each density is a request `Score.add` to its `score`
instance, made by the handler that answered the draw. A handler on the `with` stack is installed as a
handler of one of these instances, and the run nests the handlers in a fixed order, from the outside in:
the score accumulator, the score transformers, the scoring draw and the observations, the handlers of
`random` in the order of the stack, the parameter store with the `param` handlers over it, and the steps.

Two consequences follow. An inner handler of `random` sees a request before any outer one and receives the
outer answer back through its continuation, so a trace inside a clamp records the clamped value. And the
handler that answers a draw emits its density outward, so every score transformer sees every site, and a
trace reads the density of each site from what reaches the accumulator, under the static identity and
dynamic address that every derived request carries.
