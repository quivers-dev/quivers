# Computations

A `define` declaration introduces a named checked computation. Its signature
separates [static binders](types-and-indexed-families.md#kinds-and-binders), runtime parameters, result type, and [effect row](effects-and-handlers.md#effect-rows):

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

Calls are checked against this four-part signature; the body is then checked to ensure that it
returns the declared type and performs no effects outside the row.

## Declaration form

Read the signature from left to right: brackets supply information
needed for specialization, parentheses supply runtime values, the type
after `:` describes the returned value, and `!{...}` lists unhandled
effect instances. In `clamp`, all arguments and the result are real
values and `!{}` permits no unhandled effects. With `x=8`, `low=0`,
and `high=5`, only the second condition succeeds and the result is `5`.


```text
define NAME[STATIC_BINDERS](VALUE_PARAMETERS) : RESULT !{EFFECT_ROW} =
    COMPUTATION
```

Both telescopes are optional. A computation with no value parameters still
writes `()`. Static binders may range over `Type`, an index sort, or `Effect`.
A `Nat` binder may receive an integer literal at a call site.

<!-- compile: qiec -->
```qvr
define identity[A : Type](value : A) : A !{} =
    return value

define four() : Int !{} =
    let value <- identity[Int](4)
    return value
```

## Pure and effectful binding

`=` names a value expression. `<-` sequences a computation and names
what it returns. This syntax tracks the distinction even when that
computation currently has no effects; the caller uses its declared
computation interface. Thus changing a helper's implementation does not
require pretending a computation call was an ordinary value expression.


QVR deliberately distinguishes two forms:

| Form | Right-hand side | Meaning |
| --- | --- | --- |
| `let x = value` | pure expression | bind a value without sequencing an effect |
| `let x <- computation` | call, request, or [resumption](effects-and-handlers.md#resumption-grades) | run a computation, then bind its result |

An inline computation may appear without a binding when its result is `Unit`:

<!-- compile: qiec -->
```qvr
instance score : Score

define penalize(x : Real) : Real !{score} =
    let squared = x * x
    perform score.add(log_prob(Normal(0.0, 1.0), x))
    return squared
```

There is no implicit coercion from a call to a value expression. If `helper`
is declared with `define`, write `let result <- helper(args)`, even when its
row is empty.

The same distinction governs [collections](collection-expressions.md#effect-aware-traverse). `map(xs, x -> value)` is pure and
uses `=`, while `traverse(xs, x -> helper(x))` sequences computation calls and
uses `<-`. The latter joins `helper`'s effects into the caller's row. See
[Finite collection expressions](collection-expressions.md) for folds,
[normalization](../../guides/stochastic.md#normalization), and shape restrictions.

## Calls and row propagation

A **residual row** lists effects still unhandled after local [handlers](effects-and-handlers.md#handler-declarations)
have run. If a helper may request `random`, a caller must also account
for `random`, either by exposing it in its own row or handling it.
The row lists the requests a computation may make. A row containing
`random` permits repeated random requests; their number and order are
determined by the computation body.


One call syntax covers ordinary application and recursion:

<!-- compile: qiec -->
```qvr
define triangle(n : Int) : Int !{} =
    if n <= 0 then
        return 0
    else
        let rest <- triangle(n - 1)
        return n + rest
```

A caller's row must include the callee's residual row after static arguments
are substituted. The checker follows the reachable call graph, so a pure
wrapper around an effectful callee remains effectful unless it handles the
effect.

Recursion uses the reference machine's explicit stack. Set an execution fuel
limit while developing a recursive computation; exhaustion reports
`qiec-run-fuel` rather than overflowing the host stack.

## Requests

A request names an instance and an operation and may supply operation-level
static arguments:

<!-- compile: qiec -->
```qvr
instance random : Random

define draw_normal(location : Real, scale : Real) : Real !{random} =
    let value <- perform random.sample[Real](site("x"), Normal(location, scale))
    return value
```

The `[Real]` argument specializes the operation's result type `A`. The
instance's interface parameters were already fixed by its `instance`
declaration. The checker substitutes both telescopes before it verifies the
request arguments and result.

## Handling and scoped instances

`handle INSTANCE with HANDLER[STATICS] in` installs a lexical handler around
its indented body. `with instance NAME : EFFECT in` creates an instance for
its body. Both forms contain computations and thus participate in row
checking:

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

See [Effects and handlers](effects-and-handlers.md) for coverage, forwarding,
and resumption grades.

## Conditionals and indexed cases

`if` branches at computation level and evaluates only the selected branch.
Both branches must return the same type; their [effect rows](effects-and-handlers.md#effect-rows) are joined.

`case` eliminates a family value and carries an explicit motive. Constructor
patterns may bind static indices and runtime fields. The result of every
reachable branch must inhabit the motive under that branch's refinement. See
[Types and indexed families](types-and-indexed-families.md#indexed-case-analysis)
for the full form.

## Constructor values

Three application-shaped forms remain syntactically distinct:

| Form | Position | Example |
| --- | --- | --- |
| computation call | computation | `helper[Real](x)` |
| effect request | computation | `perform random.sample[Real](...)` |
| constructor | value | `construct Present(x) as Measurement(Observed)` |

This separation prevents an unqualified operation or a constructor from being
resolved as an ordinary call.

## Built-in values and distributions

The same pure-expression registry is available here and in a `program`.
Distribution applications such as `Normal(mu, sigma)`,
`Dirichlet([1.0, 2.0])`, `PointMass(0)`, or
`Mixture([0.3, 0.7], [PointMass(0), Poisson(rate)])` produce typed
`Sampleable` values. `Restrict`/`Truncate`, `Normalize`,
`Transformed`/`Pushforward`, and `Independent` are distribution families too,
so the measure algebra remains explicit in the checked term.

The reference backend implements every family in the semantic registry.
Parameter order and support follow the [continuous-family
reference](../../guides/continuous-families.md); target renderers apply any
host-specific parameter conversion at emission.

## Entry-point status

Every user-authored `define` is executable unless it is an internal helper
generated by elaboration. List entries with:

```bash
qvr run model.qvr --list
```

An entry with a nonempty residual row also needs handlers supplied by the
selected runtime provider. Authored handlers inside the body need no external
attachment. Static binders must be closed at invocation with repeated
`--static NAME=TERM` options.

For the invocation forms and stable diagnostics, continue to [Execution and
tooling](execution-and-tooling.md).
