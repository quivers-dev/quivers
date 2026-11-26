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
may leave unhandled. Both telescopes are optional; `()` explicitly gives an empty
value telescope. A static binder ranges over `Type`, an index type, or `Effect`, and a `Nat` binder may
receive an integer literal at a call site.

Programs use the same staged header convention, with object and morphism
specialization in brackets and checked external values in parentheses. Both
headers elaborate to the same core distinction between a static telescope and
typed value parameters. Family declarations use parentheses for refinable static
indices, a separate distinction explained in the
[parameter reference](index.md#parameter-telescopes).

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

Recursion runs on the explicit stack of the
[reference machine](../../developer/qiec.md#reference-evaluator), the library's interpreter for
its core language, not on the Python stack. A run may be
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

`handle INSTANCE with HANDLER[STATICS](VALUES) in` installs a handler around its indented body, the
values starting the handler's [parameters](effects-and-handlers.md#handler-parameters), and
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

`Independent(base, n)` requires an actual nonnegative integer literal `n`,
no larger than the base batch rank. It reserves those trailing base batch axes
before an enclosing program consumes its row axes. Zero retains the base
partition. This rule also applies in pure computations:

<!-- compile: qiec -->
```qvr
define product_law() : Sampleable[Tensor[Real]([2])] !{} =
    return Independent(Normal([0.0, 0.3], 0.8), 1)
```

This computation constructs a two-coordinate event; it performs no sampling
or scoring. A declaration's static `[...]` telescope and runtime `(...)`
parameters retain their ordinary staging rules. The integer argument here is
a distribution-constructor argument, rather than a new binder form.

A constructor must give one complete parameter schema. Pure computation
calls use positional arguments in the semantic registry's order; draw and
observe constructors use named arguments. Both routes check the same complete
alternatives. Thus `Normal(0.0)` is a compile error, and a Bernstein hazard must
give exactly one of `rate` and `log_rate`. Optional parameters are accepted
only where the family declares an alternative schema; host-library defaults
do not fill in missing arguments in a checked program.

`log_mass(d)` returns the total log mass of a `Sampleable` value as a
`LogWeight`. This **mass query** is pure: returning a mass does not score
the model. `log_prob(d, x)` instead reads the density or probability mass
at the given outcome. Both queries use the same distribution value;
in particular, querying a raw restriction preserves its normalizer.

<!-- compile: qiec -->
```qvr
define restricted_mass() : LogWeight !{} =
    let restricted = Restrict(Uniform(0.0, 1.0), 0.2, 0.7)
    return log_mass(restricted)

define probability_mass() : LogWeight !{} =
    return log_mass(Normalize(Restrict(Uniform(0.0, 1.0), 0.2, 0.7)))
```

The first computation returns `log(0.5)`; the second returns zero.
Ordinary probability distributions also have zero log mass. Empty
restrictions have negative infinite log mass. Bounds are inclusive by
default. The optional fourth and fifth positional arguments of a pure
`Restrict` expression set `lower_closed` and `upper_closed`: for instance,
`Restrict(PointMass(0.0), 0.0, 0.0, false, true)` excludes the atom and
has zero mass. Endpoint policy matters for discrete and mixed laws as
well as for point masses. Named arguments remain the syntax of draw
statements; pure expression calls take positional arguments.

The checked core also represents mass queries retaining declared batch
axes. It sums independent event axes and preserves intrinsic vector or
matrix events according to the family's checked shape. The reference
machine and native compiler share these distinctions. External targets
check mass-query capability and family mappings separately: the eight
dynamic targets query mapped probability distributions, while a raw
measure without a target mapping is explicitly refused. Stan, BUGS,
and JAGS refuse mass queries.

The reference machine implements every family in the
[distribution family registry](../../api/qiec/families.md). Parameter order and support follow the
[continuous family reference](../../guides/continuous-families.md), and each target renderer
converts parameters to the target's own conventions when it emits code. Inside a `program` step,
family arguments are named ([fixed and conditional families](probabilistic-programs.md#fixed-and-conditional-families)).

The checked weight primitives are available in authored computations as well.
`observation_weight : Real -> Real` validates a finite nonnegative power, and
`scale_weight : Real * LogWeight -> LogWeight` returns the zero log score for
a zero coefficient, including when the raw log weight is negative infinity:

<!-- compile: qiec -->
```qvr
define weighted_log_density(w : Real, ell : Real) : Real !{} =
    return weight_value(scale_weight(observation_weight(w), weight(ell)))
```

These calls perform pure value operations. An observation's
[`weights` option](probabilistic-programs.md#observation-weights) uses their
result in a scoring step; a computation can return it without adding a score.
Support for authored primitive calls is checked separately from direct
observation rendering. BUGS, JAGS, and Stan refuse these computation calls
with capability diagnostics.

## Checked scalar domains

`log_probability(p)` computes `log(p)` for a finite probability, including
`-inf` at zero. `log1mexp(x)` computes `log(1-exp(x))` for `x <= 0`,
including `-inf`; `log_ndtr(x)` computes the standard Normal log CDF,
including both infinite endpoints. `positive_parameter(x)` checks a
finite strictly positive real and returns it unchanged. All four reject
Boolean and NaN arguments. Their signatures are `Real -> Real`; these
runtime domain checks do not introduce static indices or effects.

<!-- compile: qiec -->
```qvr
define checked_scale(x : Real) : Real !{} =
    return positive_parameter(x)

define guarded_log(x : Real) : Real !{} =
    if x < 0.0 then
        return 0.0
    else
        return log_probability(x)
```

Only the selected conditional branch executes. Thus `guarded_log(-1)`
returns zero, while an active invalid probability such as
`guarded_log(1.2)` raises. The compiler retains control flow when eager
selection would evaluate an invalid inactive branch. Unsupported symbolic
branching is refused explicitly. A primitive's `total` contract certifies that
every well-typed argument satisfies its domain. The compiler may use eager
selection when both branches have this proof, including their local bindings;
otherwise it preserves lazy branch evaluation. A checked computation need not have
a tensor-fitting executor. The
[primitive API](../../api/qiec/primitives.md#checked-probability-operations)
states external target capabilities and endpoint limits.

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
