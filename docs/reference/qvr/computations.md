# Computations

A `define` declaration introduces a named computation that the checker verifies. Its signature
gives its [static binders](types-and-indexed-families.md#kinds-and-binders), its runtime
parameters, its result type, and its [effect row](effects-and-handlers.md#effect-rows):

<!-- compile: qiec -->
```qvr
define clamp(x : Real, low : Real, high : Real) : Real !{} =
    if x < low then
        return low
    else
        if x > high then
            return high
        else
            return x
```

A call is checked against the signature, and the body is checked to return the declared type and
to perform no effect outside the row.

## Declaration form

```text
define NAME[STATIC_BINDERS](VALUE_PARAMETERS) : RESULT !{EFFECT_ROW} =
    COMPUTATION
```

The brackets hold the binders that specialization closes, the parentheses hold runtime values, the
type after `:` is the type of the returned value, and `!{...}` lists the effect instances the body
may leave unhandled. Both telescopes are optional, but a computation with no value parameters still
writes `()`. A static binder ranges over `Type`, an index type, or `Effect`, and a `Nat` binder may
receive an integer literal at a call site.

<!-- compile: qiec -->
```qvr
define identity[A : Type](value : A) : A !{} =
    return value

define four() : Int !{} =
    let value <- identity[Int](4)
    return value
```

## Pure and effectful binding

| Form | Right side | Meaning |
| --- | --- | --- |
| `let x = value` | pure expression | bind a value, sequencing no effect |
| `let x <- computation` | call, request, or [resumption](effects-and-handlers.md#resumption-grades) | run a computation, then bind its result |

A call to a computation always uses `<-`, even when the callee's row is empty, and there is no
implicit coercion from a call to a value expression. If `helper` is declared with `define`, the
binding is `let result <- helper(args)`. A caller thus depends on the callee's declared interface,
and a change to the callee's body does not change how the caller is written.

A computation whose result is `Unit` may appear as a statement, without a binding:

<!-- compile: qiec -->
```qvr
instance score : Score

define penalize(x : Real) : Real !{score} =
    let squared = x * x
    perform score.add(log_prob(Normal(0.0, 1.0), x))
    return squared
```

The same distinction governs [collections](collection-expressions.md#effect-aware-traverse).
`map(xs, x -> value)` is pure and uses `=`, and `traverse(xs, x -> helper(x))` sequences calls,
uses `<-`, and joins `helper`'s effects into the caller's row. Folds,
[normalization](../../guides/stochastic.md#normalization), and the shape restrictions are
specified in [Finite collection expressions](collection-expressions.md).

## Calls and row propagation

The *residual row* of a computation is the effect row left after its own
[handlers](effects-and-handlers.md#handler-declarations) have run. A caller's row must include the
callee's residual row after the static arguments are substituted. If a helper may request
`random`, a caller either lists `random` in its own row or handles it. One call syntax covers
ordinary application and recursion:

<!-- compile: qiec -->
```qvr
define triangle(n : Int) : Int !{} =
    if n <= 0 then
        return 0
    else
        let rest <- triangle(n - 1)
        return n + rest
```

The checker follows the reachable call graph, so a wrapper with an empty body row around an
effectful callee is effectful unless it handles the effect. The row bounds which requests a
computation may make, and the body determines how many it makes and in what order.

Recursion runs on the reference machine's explicit stack, not on the Python stack. A run may be
given *fuel*, a bound on its number of evaluation steps (`--fuel` in
[execution controls](execution-and-tooling.md#execution-controls)), and a run that exhausts it stops
with `qiec-run-fuel`.

## Requests

A request names an instance and an operation and may give the operation's static arguments:

<!-- compile: qiec -->
```qvr
instance random : Random

define draw_normal(location : Real, scale : Real) : Real !{random} =
    let value <- perform random.sample[Real](site("x"), Normal(location, scale))
    return value
```

The `[Real]` argument specializes the operation's result type `A`. The interface's own parameters
are fixed by the `instance` declaration. The checker substitutes both telescopes before it checks
the request's arguments and result.

## Handling and scoped instances

`handle INSTANCE with HANDLER[STATICS] in` installs a handler around its indented body, and
`with instance NAME : EFFECT in` creates an instance that exists only in its body. Both contain
computations, so both take part in row checking:

<!-- compile: qiec -->
```qvr
effect Echo
    echo : String -> String

handler identity_echo for Echo : String -> String [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    echo(value : String) resumes 1 =>
        resume(value)

define echoed(value : String) : String !{} =
    with instance local : Echo in
        handle local with identity_echo in
            let result <- perform local.echo(value)
            return result
```

Coverage, forwarding, and resumption grades are specified in
[Effects and handlers](effects-and-handlers.md).

## Conditionals and indexed cases

`if` branches at the level of computations and evaluates only the selected branch. Both branches
return the same type, and their [effect rows](effects-and-handlers.md#effect-rows) are joined.

`case` eliminates a value of an indexed family and states a motive. A constructor pattern may bind
static indices and runtime fields, and the result of every reachable branch must have the motive's
type under that branch's refinement
([indexed case analysis](types-and-indexed-families.md#indexed-case-analysis)).

## Constructor values

Three forms that look like applications stay distinct in the syntax:

| Form | Position | Example |
| --- | --- | --- |
| computation call | computation | `helper[Real](x)` |
| effect request | computation | `perform random.sample[Real](...)` |
| constructor | value | `construct Present(x) as Measurement(Observed)` |

An operation without its instance, or a constructor, is thus never resolved as an ordinary call.

## Built-in values and distributions

A `define` body and a `program` share one registry of pure primitives
([pure expressions](types-and-indexed-families.md#pure-expressions)). A distribution application
such as `Normal(mu, sigma)`, `Dirichlet([1.0, 2.0])`, `PointMass(0)`, or
`Mixture([0.3, 0.7], [PointMass(0), Poisson(rate)])` builds a typed `Sampleable` value. The
constructions on distributions are families as well, so they stay explicit in the checked term:
`Restrict` (or `Truncate`) restricts a distribution to an interval, `Normalize` rescales a measure
to total mass one, `Transformed` (or `Pushforward`) takes the
[pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) along a bijection, and
`Independent` reinterprets batch axes as event axes.

The reference machine implements every family in the
[distribution family registry](../../api/qiec/families.md). Parameter order and support follow the
[continuous family reference](../../guides/continuous-families.md), and each target renderer
converts parameters to the target's own conventions when it emits code. Inside a `program` step,
family arguments are named ([fixed and conditional families](probabilistic-programs.md#fixed-and-conditional-families)).

## Entry points

Every `define` in the source is an [entry point](execution-and-tooling.md#list-entry-points). The
helpers that elaboration generates are not. The entries of a file are listed with:

```bash
qvr run model.qvr --list
```

An entry with a nonempty residual row needs handlers for its remaining instances from the selected
[runtime provider](execution-and-tooling.md#runtime-providers), and a handler written in the body
needs nothing external. Static binders are closed at invocation with one
`--static NAME=TERM` option each. The invocation forms and the stable diagnostics are specified in
[Execution and tooling](execution-and-tooling.md).
