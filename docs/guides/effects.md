# Compositional Effects

An **effect** is an operation a computation requests from its execution
context, such as drawing a random value, reading state, or adding a score.
A **handler** decides how that request is interpreted. The same random
request can draw a fresh value during simulation or replay a supplied
value during scoring. A [resumption](../reference/qvr/effects-and-handlers.md#resumption-grades) continues the computation after the
request has received its answer.

Pyro’s [example of conditioning and tracing](https://pyro.ai/examples/effect_handlers.html#A-first-look-at-Poutine:-Pyro%E2%80%99s-library-of-algorithmic-building-blocks) demonstrates two interpretations of a sample request. [@pyroEffects] The QVR [handler declarations](../reference/qvr/effects-and-handlers.md#handler-declarations) give an example that follows a request through its resumption.

In linguistic semantics, an expression can likewise carry more than its
ordinary denotation: several alternatives, a discourse update, or a
scope-taking operation. The monad and arrow interfaces below describe how
to compose that extra structure. The later QIEC section describes how
source programs declare requests and [handlers](../reference/qvr/effects-and-handlers.md#handler-declarations). These related presentations
serve different levels of the implementation.


QVR provides two parallel typeclass towers, Haskell-style [monads](monadic.md#monads) and
Hughes-style arrows, together with a class-driven schema-lifting system.
These components let linguistic effects (scope-taking, anaphora, focus and
alternatives, presupposition, supplements, plurality) compose with the
residuated category universe through the same declarations as ordinary
categorial grammar rules.

This guide describes the interfaces and their interaction. The corresponding
denotations are in [Denotational Semantics](../semantics/effects.md).

## Typeclass hierarchy

For arrows as a generalization of monadic interfaces, see [@hughes2000arrows].

A *typeclass* specifies operations an implementation supports. `fmap`
applies an ordinary function while retaining its surrounding effect;
`pure` introduces an ordinary value into that effect. `apply` combines
effectful arguments, while `bind` lets the next computation depend on a
value obtained from the previous one. *Lifting* a grammar rule means
extending it to inputs that carry such extra structure.

For instance, if an adjective has two possible meanings and a noun has
three, lifting their combination over alternatives considers six pairs.
The base adjective-noun rule remains the operation applied to each pair;
the effect interface supplies the branching and aggregation.


Both towers live as Python ABCs in `quivers.monadic.typeclasses` and
`quivers.arrows.typeclasses`. Effects (instances) implement the
operations a typeclass requires; the lifting system dispatches on
which classes an effect inhabits, never on the effect's identity.

The monadic hierarchy:

| Class | Adds | Linguistic use |
|-------|------|----------------|
| `Functor` | `fmap` | structure-preserving lifts |
| `Applicative` | `pure`, `apply` | uniform-lift of base rules through an effect |
| `Monad` | `join` (and derived `bind`) | scope-extruding lifts [@bumford2026effect] |
| `Alternative` | `empty`, `alt` | non-deterministic / Hamblin-style branches |
| `MonadPlus` | combines `Monad` and `Alternative` | branching effects with substitution |
| `Foldable` / `Traversable` | `foldr`, `traverse` | distribute Applicative actions through a structure |
| `MonadTrans` | `lift` | stack one monad on top of another |

The arrow hierarchy [@hughes2000arrows]:

| Class | Adds | Use |
|-------|------|-----|
| `Category_` | `id_arr`, `compose` | the bare composition law |
| `Arrow` | `arr`, `first` | symmetric-monoidal arrow shape |
| `ArrowChoice` | `left_arr` | sum-elimination |
| `ArrowApply` | `app` | equivalent power to `Monad` via `arrow_monad` |
| `ArrowLoop` | `loop_arr` | feedback / fixed-point (the chart-fold) |
| `ArrowZero` / `ArrowPlus` | `zero_arr`, `alt_arr` | pointwise alternatives on hom-sets |

The `kleisli` and `arrow_monad` bridges in
`quivers.monadic.bridges` translate between the two towers freely;
ArrowApply and Monad are interconvertible.

## Stdlib effect instances

Eight stdlib effects ship in `quivers.monadic.instances`:

| Effect | Class instances | Linguistic use |
|--------|------------------|----------------|
| `Identity` | `Monad`, `Functor` | trivial / no-effect base case |
| `Maybe` | `MonadPlus`, `Functor` | partiality / presupposition failure |
| `Alternative_` | `MonadPlus`, `Foldable`, `Traversable` | Hamblin alternatives, focus, questions |
| `Continuation(answer)` | `Monad`, `Functor` | scope-taking, generalized quantifiers |
| `State(state)` | `Monad`, `Functor` | dynamic anaphora, discourse referents |
| `Reader(env)` | `Monad`, `Functor` | assignment functions, indexicality |
| `Writer(monoid)` | `Monad`, `Functor` | supplements, nonrestrictive content |
| `List(max_length)` | `MonadPlus`, `Foldable`, `Traversable` | bag-of-readings parsers |

Each effect is a `dx.Model` carrying the relevant parameters (e.g.
`Continuation(answer=S)` is parameterized by the answer type), and
each registers against its appropriate ABC(s) via `ABC.register(...)`.

## Class-driven schema lifting

The `class_directed_lifts(base_schema, effect)` function in
`quivers.stochastic.effect_lifts` introspects which typeclasses an
effect inhabits and emits one lifted schema per class:

| Effect class | Lifts emitted |
|--------------|---------------|
| `Applicative` | `pure_T`, `apply_T` |
| `Monad` | adds `bind_T` (scope-extruding) |
| `Alternative` | adds `alt_T` |
| `MonadPlus` | union of Monad + Alternative |

Each lift is a real `SchemaDecl` and feeds into the existing
`PatternBinarySchema` / `PatternUnarySchema` runtime; the chart parser
consumes lifted and base schemas uniformly.

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
# 3 lifts: pure_Continuation, apply_Continuation, bind_Continuation

alt_lifts = class_directed_lifts(forward_app, Alternative_())
# 4 lifts: pure_Alternative, apply_Alternative, bind_Alternative, alt_Alternative
```

## Algebraic effects + handlers

An operation signature describes what may be requested without choosing
how requests run. A free monad records those requests and the remaining
computation as syntax; a handler interprets that syntax. For a `Choose`
request, one handler might select an alternative and another might explore
all alternatives. The operation's type stays fixed while the handling
policy changes.


For effects that don't fit a closed-form typeclass instance,
`quivers.monadic.algebraic` provides a free-monad-over-signature
construction:

- `Operation(name, parameter, result)`: one operation in a signature.
- `EffectSignature(name, operations)`: a signature; lifts to a
  panproto theory via `to_theory()`.
- `FreeMonad(signature)`: the free monad over the signature; a
  `Monad` instance automatically.
- `Handler(signature, target, return_clause, operation_clauses)` :
  interprets a `FreeMonad`-valued computation in a target monad.
  Equivalently: a panproto theory morphism from
  `signature.to_theory()` into the target monad's theory.

Handlers compose with a `deduction` block to produce parsers
that interpret their effect-typed denotation through registered
handlers, ending in an effect-pure target.

## QIEC effect interfaces and rows

Read an [effect row](../reference/qvr/effects-and-handlers.md#effect-rows) as a list of named services a computation may still
request. `!{random, score}` permits random draws and scoring through
those two instances; `!{}` leaves no unhandled requests. Two state
instances are like two distinct cells even if both hold integers.
Handling one cell does not handle the other. A total handler removes its
instance from the row because it accounts for every request on that
instance; a partial handler leaves an obligation for the surrounding
context.


QVR's **QIEC effect calculus** is the source-level interface every
computation of a module is checked under, a probabilistic `program` included:
its `sample` and `observe` steps perform `Random.sample` and `Score.add` on the
module's canonical instances, and a `program [effects=[...]]` signature is a
summary of that row. An `effect` may take heterogeneous static
parameters and declares operations. An `instance` applies that
interface and receives a lexical identity, so two instances of `State[Int]`
remain different row entries. A typed computation records those entries in an
exact or open row, including row-tail `lacks` constraints.

Handlers match an applied interface and one lexical instance. Their stable
declarations record total or partial coverage, forwarding policy, introduced
effects, and a resumption grade for each operation. A total handler removes the
matched instance from the residual row; a partial handler retains it. An
explicitly forwarding partial handler passes structurally uncovered operations
to an outer handler.

This calculus unifies the types used by logic-style `Choose`, weighted
accumulation, state, abort, random choice, and scoring. A handler's clauses
are authored in the source, each resuming as its grade admits, or supplied as
process-local attachments; the effect handlers of `quivers.effects` are
themselves lexical handlers of the kernel's `Random`, `Score`, `Compute`, and
`Param` interfaces. Pyro, NumPyro, PyMC, Edward2, Turing, Gen, WebPPL, and
Church run the same computation graph through corresponding implementations
of the stable-ID runtime ABI. Tests compare their scores with the
reference machine at specified clamped values. Stan lowers a
computation with a closed empty effect row, an empty static telescope, and a
scalar result as a user-defined function; BUGS and JAGS refuse every call.
These targets reject unsupported control or value features explicitly, under
`qiec:capability:<feature>:<name>` and `call:graph:<name>`. The [effects and
handlers reference](../reference/qvr/effects-and-handlers.md) specifies the
source and operational contract. The [QIEC developer note](../developer/qiec.md)
specifies the checker, serialization boundary, evaluator, and transpiler ABI.

## Bridges between the two towers

`quivers.monadic.bridges` contains:

- `kleisli(monad)`: wraps a `Monad` as a `Kleisli` arrow registered
  against `Arrow` and `ArrowApply`.
- `arrow_monad(arrow)`: wraps an `ArrowApply` as an `ArrowMonad`
  registered against `Monad`.

The pair gives a free choice of presentation: write effects as monads
or as arrows; the framework treats them uniformly.

## Joint type-and-effect dispatch in the parser

A chart cell records what a span can mean and which effects that meaning
still carries. An ordinary rule combines plain denotations. A lifted rule
combines denotations inside an effect. A handler interprets an effect, and
a registered distributive law permits a particular change in effect order.
The parser searches these choices alongside the syntactic category
choices. Swapping two effects requires a registered distributive law
for that pair and direction.


When the chart parser fires at span `(i, j)`, each cell carries a
distribution over `(Cat, EffectStack)` pairs (the residuated universe
×️ a stack of declared effects). At each cell the parser considers:

1. **Base firings**, when both children are effect-pure, the base
   schema's pattern unifies on the type coordinate as in classical
   Lambek.
2. **Lift firings**, when child effect-stacks differ, the
   class-driven lifts of `class_directed_lifts` interpose `pure_T` /
   `fmap_T` / `apply_T` / `bind_T` etc. as appropriate.
3. **Handler firings**, when the cell holds a `T(α)` distribution and
   a `Handler` from `T` is registered, applying the handler produces
   a fresh cell at a strictly-shorter effect-stack.
4. **Commutation firings**, when a `DistributiveLaw(T, U)` (or arrow
   analogue) is registered, the `swap_TU` schema exchanges sibling
   effect orderings.

The four kinds compose freely; the chart's CKY enumeration explores
the joint search space.

## Worked example: scope-taking via Continuation

The example file `docs/examples/source/quantifier_scope.qvr`
illustrates a grammar that composes expressions with scope-taking effects [@bumford2026effect]:

```text
object Term : FinSet 16
deduction QScope : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, VP, PP, Fwd, Bwd, Cont, span

    # Base forward / backward application.
    rule fwd_app
        : span(I, K, Fwd(A, B)), span(K, J, B)
        |- span(I, J, A)
    rule bwd_app
        : span(I, K, B), span(K, J, Bwd(A, B))
        |- span(I, J, A)

    # Applicative lift of forward application under Cont:
    #   Cont(A/B) * Cont(B) ⊢ Cont(A)
    rule fwd_app_cont
        : span(I, K, Cont(Fwd(A, B))), span(K, J, Cont(B))
        |- span(I, J, Cont(A))

    # Pure: A ⊢ Cont(A)
    rule pure_cont
        : span(I, J, A) |- span(I, J, Cont(A))

    # Scope-extruding bind: a Cont-typed expression takes scope
    # over an inner argument by absorbing the surrounding context.
    #   Cont(A) * (A\B) ⊢ Cont(B)
    rule scope_take
        : span(I, K, Cont(A)), span(K, J, Bwd(B, A))
        |- span(I, J, Cont(B))
```

In production code, the lifted schemas (`apply_Cont_fwd`,
`scope_take`) are auto-generated by `class_directed_lifts`; the
explicit declaration above demonstrates what a user would write to enumerate
the lifts manually.

## Further reading

- [@bauer2015programming]
- [@charlow2025scope]
- [@bumford2026effect]
- [@plotkin2003algebraic]
- [@hughes2000arrows]
- [@mcbride2008applicative]
