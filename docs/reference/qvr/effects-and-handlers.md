# Effects and handlers

An effect declaration gives names and types to requests. An instance gives one
application of that interface a lexical identity. A computation's effect row
then records the instances it may request from, and a handler interprets one
instance. Two instances of `State[Int]` have the same interface but remain distinct effects.

## Declare an interface

An interface specifies the requests a service accepts, an
instance as one particular service, and a handler as its implementation.
Two integer state cells support the same `get` and `put` requests but
hold distinct identities. Qualifying `left.get()` makes it clear which
cell the computation asks about. An effect row naming both cells thus
records two separate state dependencies.


<!-- compile: qiec -->
```qvr
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

instance left : State[Int]
instance right : State[Int]

define exchange(next : Int) : Int !{left, right} =
    let previous <- perform left.get()
    perform right.put(next)
    return previous
```

Square brackets bind static parameters on the interface or an operation. An
operation's argument types precede `->`; an operation with no arguments may
write either `Unit -> T` or the corresponding nullary form admitted by the
grammar. A request is always qualified by an instance:
`perform left.get()`. This qualification is what keeps `left` and `right`
separate in the row.

## Prelude interfaces

These interfaces are available without a source declaration:

| Interface | Principal operation | Purpose |
| --- | --- | --- |
| `Random` | `sample[A](Site[A], Sampleable[A]) -> A` | draw or replay a named random value |
| `Score` | `add(LogWeight) -> Unit` | add a density or factor |
| `State[S]` | `get`, `put` | lexical mutable state under a handler |
| `Abort[E]` | `abort[A](E) -> A` | typed early exit |
| `Choose` | `choose[A](Tensor[A]) -> A` | nondeterministic search |
| `Weight[K]` | `add(K) -> Unit` | semiring-parametric accumulation |
| `Compute[X, A]` | `apply(X) -> A` | process-local host computation |
| `Param` | `get[A](String) -> A` | named learned-parameter lookup |

`Search` is generated for a [deduction](../../guides/deduction.md#the-framework) or schema-backed parser because its
choice arity depends on the elaborated system. Structural declarations create
`Compute` instances; a caller normally encounters them through
`Program.run`, not by writing the requests manually.

## Effect rows

A computation signature ends in an effect row:

| Row | Meaning |
| --- | --- |
| `!{}` | closed and pure |
| `!{left, right}` | exactly these lexical instances |
| `!{robust | rho}` | `robust` plus an open tail `rho` |
| `!{robust | rho lacks robust}` | open tail that cannot already contain `robust` |

Rows contain instances, not interface names. A call contributes its callee's
row after static substitution. A `perform` contributes the named instance. An
`if` or `case` joins the rows of every reachable branch. Conservative case
coverage also means that an unproved-impossible branch contributes its
effects.

`lacks` constraints prevent accidental duplication when a row-polymorphic
computation adds an instance of its own. Use them when an abstraction handles
or introduces the same lexical name that a caller's tail might otherwise
contain.

## Handler declarations

In `robustify(8.0)` below, execution reaches `robust.shrink(8.0)` and
hands that request to `half_weight`. The clause computes `4.0`, then
`resume(4.0)` continues the suspended computation with `adjusted = 4.0`.
The program returns `4.0`, and the handler's return clause passes it
through. *Deep* handling means later requests in that resumed computation
are handled by the same handler. [@plotkin2009handlers]


A handler declares the interface it handles, an input and answer type, options,
one return clause, and operation clauses:

<!-- compile: qiec -->
```qvr
effect Robust
    shrink : Real -> Real

instance robust : Robust

handler half_weight for Robust : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    shrink(x : Real) resumes 1 =>
        resume(0.5 * x)

define robustify(x : Real) : Real !{} =
    handle robust with half_weight in
        let adjusted <- perform robust.shrink(x)
        return adjusted
```

The `return` clause handles a computation that finishes without another
request. Each operation clause receives the request's fields and may call
`resume`. Handlers are deep: a resumed continuation runs beneath the same
handler. Effects performed by the clause body itself are offered only to
outer handlers.

### Handler options

| Option | Values | Effect |
| --- | --- | --- |
| `coverage` | `total`, `partial` | whether every operation is handled |
| `forwards` | `none`, `unknown` | whether a partial handler passes uncovered requests outward |
| `implementation` | `authored`, `foreign` | whether clauses have QVR bodies or runtime attachments |
| `introduces` | instance names | additional effects a handler body may perform |

A total handler removes the matched instance from the body's row. A partial
handler retains it. Forwarding and coverage are separate: an uncovered
operation in a non-forwarding partial handler is an error rather than an
implicit outer request.

### Resumption grades

The [compositional effects guide](../../guides/effects.md#typeclass-hierarchy) relates sequencing to `bind`; the [computation reference](computations.md#handling-and-scoped-instances) shows where a handler enters the execution. Pyro’s [handler implementation walkthrough](https://pyro.ai/examples/effect_handlers.html#Implementing-new-effect-handlers-with-the-Messenger-API) provides a Python implementation to compare with QVR’s explicit resumption clauses. [@pyroEffects]

A handler can stop the computation, continue it once, or explore
several possible continuations. The grade records which behavior the
clause promises. An abort has no continuation; a normal state read
continues once; exhaustive search can continue once per alternative.
Duplicating a continuation requires the corresponding permission because
it may also duplicate subsequent requests.


| Grade | Contract |
| --- | --- |
| `0` | the clause must not resume |
| `aff` | at most once |
| `1` | exactly once |
| `omega` | any number of times when the continuation is duplicable |

The checker records the grade, and the evaluator enforces it. `omega` is used
by search handlers that explore several alternatives; a state update that
continues once normally uses `1`; an abort clause uses `0`.

## Scoped instances

The following handler always answers `get` with `7` and ignores the
value passed to `put`, so `local_read()` returns `7` even after `put(3)`.
To implement a mutable cell, the `put` clause would instead pass its new
value into the resumed computation for subsequent `get` requests.


`with instance` creates a fresh lexical identity inside a computation:

<!-- compile: qiec -->
```qvr
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

handler fixed_reply for State[Int] : Int -> Int [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(7)
    put(s : Int) resumes 1 =>
        resume(unit)

define local_read() : Int !{} =
    with instance cell : State[Int] in
        handle cell with fixed_reply in
            perform cell.put(3)
            let value <- perform cell.get()
            return value
```

The instance does not exist outside its block. A total handler must eliminate
it before control leaves; otherwise the checker reports
`qiec-instance-escape`. Nesting two `with instance cell ...` blocks creates two
identities even though the source spelling is shadowed.

## Foreign handlers and providers

A clause with no `=>` body is a signature for a process-local implementation,
and the declaration must say `implementation=foreign`. The stable module
contains the handler type, operation coverage, and grades, but no callable.
At invocation, the selected runtime provider binds the handler's stable ID to
an implementation. Runtime configuration files contain provider names and
data only; they cannot inject code.

The built-in `core` provider supplies the reference implementations used by
program execution, deductions, and the standard prelude. Structural models
compose `core` with a structural provider holding their compiled PyTorch
modules. See [Execution and tooling](execution-and-tooling.md#runtime-providers).

## Common failures

- **Request absent from the row.** Add the lexical instance or make the row
  polymorphic when the abstraction is intended to preserve caller effects.
- **Wrong applied interface.** `State[Int]` and `State[String]` are distinct;
  a handler must match the instance's exact application.
- **Instance escape.** Handle a scoped instance before leaving its block.
- **Grade violation.** Make the number of `resume` calls match the declared
  grade.
- **Missing provider.** Attach a provider that implements the foreign handler,
  or use an authored handler.

The [effects tutorial](../../tutorials/qvr/10-effects-and-handlers.md) builds
and traces an authored handler end to end.
