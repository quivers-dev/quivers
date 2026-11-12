# Compositional Effects

An [*algebraic effect*](https://en.wikipedia.org/wiki/Effect_system) is an operation
a computation requests from its context, such as drawing a random value, reading
state, or adding a score, and a *handler* decides how each request is answered and
how the computation resumes
[@plotkin2003algebraic; @plotkin2009handlers; @bauer2015programming]. The same random request can draw a fresh value during
simulation or replay a given value during scoring, as in Pyro's effect handlers
[@pyroEffects].

Linguistic meanings can carry similar extra structure: several alternatives, a
discourse update, or a scope-taking operation. Effect-driven interpretation composes
such meanings with the functor, applicative, and monad interfaces of functional
programming [@bumford2026effect]. quivers provides these interfaces as Python
abstract classes, a function that lifts grammar rules through them, and a free-monad
construction for effects given by a signature. QVR programs declare effects and
handlers in the [Quivers Indexed Effect Core](#qiec-effect-interfaces-and-rows),
described at the end of the page. [Denotational Semantics](../semantics/effects.md)
gives the denotations.

## Typeclass hierarchy

A [type class](https://en.wikipedia.org/wiki/Type_class) is an interface of
operations, with laws, that a type implements [@wadler1989adhoc]. Of the classes
here, `fmap` applies an ordinary function inside an effect, `pure` brings a plain
value into it, `apply` combines effectful arguments, and `bind` lets the next step
depend on a value produced by the previous one [@mcbride2008applicative]. *Lifting*
a grammar rule means extending it to arguments that carry an effect. If an adjective
has two possible meanings and a noun three, lifting their combination through the
alternatives effect considers the six pairs, applies the base rule to each, and
collects the results.

`quivers.monadic.typeclasses` and `quivers.arrows.typeclasses` define the classes as
Python abstract base classes. An effect is an instance of the classes whose
operations it implements, and the lifting function below chooses lifts by those
classes, never by the effect's identity. The [monad](monadic.md#monads) classes:

| Class | Adds | Linguistic use |
|-------|------|----------------|
| `Functor` | `fmap` | lifting a function through an effect |
| `Applicative` | `pure`, `apply` | lifting a rule through an effect, argument by argument |
| `Monad` | `join` (and the derived `bind`) | letting an effectful argument take scope over its context |
| `Alternative` | `empty`, `alt` | sets of alternatives |
| `MonadPlus` | both `Monad` and `Alternative` | alternatives with scope-taking |
| `Foldable` / `Traversable` | `foldr`, `traverse` | running an applicative action across a structure |
| `MonadTrans` | `lift` | stacking one monad on another |

The [arrow](https://en.wikipedia.org/wiki/Arrow_(computer_science)) classes follow
Hughes [@hughes2000arrows]. An arrow generalizes a function. Its `first` runs an
arrow on one component of a pair, which gives a premonoidal (Freyd) structure rather
than a symmetric monoidal one [@jacobs2009categorical].

| Class | Adds | Use |
|-------|------|-----|
| `Category_` | `id_arr`, `compose` | identity and composition |
| `Arrow` | `arr`, `first` | lifting functions and acting on one component of a pair |
| `ArrowChoice` | `left_arr` | acting on one case of a disjoint union |
| `ArrowApply` | `app` | applying an arrow given as input |
| `ArrowLoop` | `loop_arr` | feedback, for recursive definitions |
| `ArrowZero` / `ArrowPlus` | `zero_arr`, `alt_arr` | failure and choice between arrows |

## Stdlib effect instances

`quivers.monadic.instances` provides eight effects, each a `dx.Model` that holds its
parameters (`Continuation(answer=S)` holds its answer type) and registers with its
classes through `ABC.register`. The linguistic uses follow Bumford and Charlow
[@bumford2026effect].

| Effect | Registered as | Linguistic use |
|--------|------------------|----------------|
| `Identity` | `Monad` | no effect |
| `Maybe` | `MonadPlus` | presupposition failure |
| `Alternative_` | `MonadPlus`, `Foldable`, `Traversable` | alternatives, focus, questions |
| `Continuation(answer)` | `Monad` | scope-taking, generalized quantifiers |
| `State(state)` | `Monad` | dynamic anaphora, discourse referents |
| `Reader(env)` | `Monad` | assignment functions, indexicals |
| `Writer(monoid)` | `Monad` | appositives, nonrestrictive relative clauses |
| `ListMonad(max_length)` | `MonadPlus`, `Foldable`, `Traversable` | sets of readings |

Every `Monad` is also an `Applicative` and a `Functor`, since the classes are
subclasses of one another.

## Lifting rule schemas by type class {#class-driven-schema-lifting}

`class_directed_lifts(base_schema, effect)` in `quivers.stochastic.effect_lifts`
checks which classes an effect is an instance of and returns one lifted
[rule schema](deduction.md) per applicable operation. For a base rule named `r` and an
effect `T`, the lifts are:

| Effect class | Lifted schema | Shape |
|--------------|---------------|-------|
| `Applicative` | `pure_T_r` | $X \vdash T(X)$ |
| `Applicative` | `apply_T_r` | $T(X/Y),\ T(Y) \vdash T(X)$ |
| `Monad` | `bind_T_r` | $T(X),\ T(Y)/X \vdash T(Y)$ |
| `Alternative` | `alt_T_r` | $T(X),\ T(X) \vdash T(X)$ |

Each lift is a `SchemaDecl`, the same value a QVR `schema` declaration produces, so a
chart parser treats lifted and written schemas alike.

```python
from quivers.dsl.ast_nodes import (
    SchemaDecl,
    SchemaParameter,
    TypeName,
    ObjectProduct,
    ObjectSlash,
)
from quivers.monadic.instances import Continuation, Alternative_
from quivers.stochastic.effect_lifts import class_directed_lifts
from quivers.core.objects import FinSet

S = FinSet(name="S", cardinality=2)

forward_app = SchemaDecl(
    name="forward_app",
    parameters=(
        SchemaParameter(names=("X", "Y"), type_expr=TypeName(name="Cat")),
    ),
    domain=ObjectProduct(components=(
        ObjectSlash(result=TypeName(name="X"), argument=TypeName(name="Y"), direction="/"),
        TypeName(name="Y"),
    )),
    codomain=TypeName(name="X"),
)

cont_lifts = class_directed_lifts(forward_app, Continuation(answer=S))
# pure_Continuation_forward_app, apply_Continuation_forward_app,
# bind_Continuation_forward_app

alt_lifts = class_directed_lifts(forward_app, Alternative_())
# the same three for Alternative, plus alt_Alternative_forward_app
```

`lift_rule_set(base_schemas, effects)` returns the base schemas followed by every
lift of each one under each effect. `swap_rule_set(laws)` returns, for each
[distributive law](monadic.md#distributive-laws) $\lambda : S \circ T \Rightarrow
T \circ S$, a schema $S(T(X)) \vdash T(S(X))$ that exchanges the order of two
effects. Both results are tuples of schemas that a chart parser takes as ordinary
rules. The parser does not add them by itself, and no step of the parser applies a
handler.

## Algebraic effects + handlers

A signature lists the operations a computation may request without fixing how they
run. The [free monad](https://ncatlab.org/nlab/show/free+monad) over a signature
represents a computation as a tree of requests waiting for answers, and a handler
interprets that tree in some target monad [@swierstra2008data; @plotkin2009handlers].
For a `Choose` request, one handler might pick an alternative and another might
explore all of them, while the operation's type stays the same.
`quivers.monadic.algebraic` provides:

- `Operation(name, parameter, result)`: one operation;
- `EffectSignature(name, operations)`: a signature, whose `to_theory()` returns it as
  a [panproto](https://github.com/panproto/panproto) theory;
- `FreeMonad(signature)`: the free monad over the signature, an instance of `Monad`;
- `Handler(signature, target, return_clause, operation_clauses)`: an interpretation
  of `FreeMonad` computations in the monad `target`, whose `as_theory_morphism()`
  returns a panproto theory with one operation for the return clause and one for each
  operation clause.

## QIEC effect interfaces and rows

The Quivers Indexed Effect Core (QIEC) is the typed core language to which every
executable QVR declaration elaborates ([language reference](../reference/qvr/index.md)).
Each computation in it has an
[effect row](../reference/qvr/effects-and-handlers.md#effect-rows), the list of effect
instances it may still request. `!{random, score}` permits random draws and scoring
through those two instances, and `!{}` permits no unhandled request. A probabilistic
`program`'s `sample` and `observe` steps request operations of the built-in `random`
and `score` instances, and its `[effects=[...]]` option summarizes that row.

An `effect` declaration names an interface and its operations, and an `instance`
declaration creates a named occurrence of it [@bauer2015programming]. Two instances
of `State[Int]` are thus different row entries, like two different memory cells, and
handling one leaves the other. A
[handler](../reference/qvr/effects-and-handlers.md#handler-declarations) matches one
instance. A total handler answers every operation of its instance and removes the
instance from the row, and a partial handler leaves it there for an enclosing handler.
Each handler clause also declares a
[resumption grade](../reference/qvr/effects-and-handlers.md#resumption-grades), how many
times it may resume the computation.
[Effects and handlers](../reference/qvr/effects-and-handlers.md) specifies the
source forms, and [Transpiler support](../transpile-support.md) lists which targets
accept which effects.

## Bridges between monads and arrows {#bridges-between-the-two-towers}

`quivers.monadic.bridges` converts between the two kinds of interface:

- `kleisli(monad)` wraps a `Monad` as its Kleisli arrow, an instance of `Arrow` and
  `ArrowApply`;
- `arrow_monad(arrow)` wraps an `ArrowApply` as a `Monad`.

Only arrows with `app` correspond to monads in this way [@hughes2000arrows], so an
arrow without `ArrowApply` has no monad counterpart.

## Worked example: scope-taking via Continuation

The gallery grammar `docs/examples/source/quantifier_scope.qvr` derives quantifier
scope with continuation-typed categories [@bumford2026effect]. Its rules write
categories in Steedman notation, with the result leftmost: `Fwd(A, B)` is `A/B`,
which takes a `B` on its right to give `A`, and `Bwd(A, B)` is `A\B`, which takes a
`B` on its left to give `A` [@steedman2000syntactic]. `Cont(X)` is a category of
continuation type.

```qvr
object Term : FinSet 16

deduction QScope : Term -> Term [semiring=LogProb, start=S, depth=4, tolerance=1e-5]
    atoms S, NP, N, Fwd, Bwd, Cont, span, every, dog, barks
    rule fwd_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule bwd_app : span(I, K, B), span(K, J, Bwd(A, B)) |- span(I, J, A) #[learnable]
    rule fwd_app_cont : span(I, K, Cont(Fwd(A, B))), span(K, J, Cont(B)) |- span(I, J, Cont(A)) #[learnable]
    rule pure_cont : span(I, J, A) |- span(I, J, Cont(A)) #[learnable, bounded]
    rule scope_take : span(I, K, Cont(A)), span(K, J, Bwd(B, A)) |- span(I, J, Cont(B)) #[learnable]
    rule cont_elim : span(I, J, Cont(S)) |- span(I, J, S) #[learnable, bounded]
    lexicon
        "every" : Cont(Fwd(NP, N)) = every #[learnable]
        "dog"   : N                = dog   #[learnable]
        "barks" : Bwd(S, NP)       = barks #[learnable]
```

`fwd_app_cont` is the applicative lift of forward application,
`Cont(A/B), Cont(B) ⊢ Cont(A)`, and has the same shape as the schema `apply_Continuation_r`
that `class_directed_lifts` generates. `pure_cont` is the unit, `A ⊢ Cont(A)`.
`scope_take`, `Cont(A), B\A ⊢ Cont(B)`, lets a continuation-typed argument take scope
over the function that consumes it, and `cont_elim` lowers a `Cont(S)` to an `S`.
These two are written by hand: the generated `bind` lift has the different shape
`T(X), T(Y)/X ⊢ T(Y)`.
