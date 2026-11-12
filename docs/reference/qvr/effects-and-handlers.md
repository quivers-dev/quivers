# Effects and handlers

QVR has [algebraic effects](https://en.wikipedia.org/wiki/Effect_system) with handlers
[@plotkin2003algebraic; @plotkin2009handlers]. An `effect` declaration names a set of operations,
the requests a computation may make, and gives their types. An *effect instance* is one named
occurrence of an effect interface, so two instances of one interface are handled separately
[@bauer2015programming]. A computation's *effect row* lists the instances it may request from, and
a *handler* answers the requests made to one instance. Two instances of `State[Int]` thus share an
interface and remain distinct effects.

## Declare an interface

`left` and `right` below are two integer state cells. Both accept the requests `get` and `put`,
but a request names its instance, so the row of `exchange` records two separate state
dependencies:

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

Square brackets bind static parameters of the interface or of an operation. An operation's
argument types precede `->`. An operation with no arguments may be written `Unit -> T` or in the
nullary form the grammar admits. A request always names its instance, as in
`perform left.get()`, and this is what keeps `left` and `right` apart in the row.

## Prelude interfaces

These interfaces need no source declaration:

| Interface | Principal operation | Purpose |
| --- | --- | --- |
| `Random` | `sample[A](Site[A], Sampleable[A]) -> A` | draw or replay a random value at a named site |
| `Score` | `add(LogWeight) -> Unit` | add a log density or a log factor to the run's weight |
| `State[S]` | `get`, `put` | mutable state under a handler |
| `Abort[E]` | `abort[A](E) -> A` | typed early exit |
| `Choose` | `choose[A](Tensor[A]) -> A` | nondeterministic choice among alternatives |
| `Weight[K]` | `add(K) -> Unit` | accumulate a weight in the semiring `K` |
| `Compute[X, A]` | `apply(X) -> A` | call a Python value held by the running process |
| `Param` | `get[A](String) -> A` | look up a learned parameter by name |

Every probabilistic `program` performs through two built-in instances, `random : Random` and
`score : Score`. A `Search` interface is generated for each
[deduction](../../guides/deduction.md#the-framework) and each parser built from rule schemata, because the
number of alternatives at a choice depends on the elaborated system. Structural declarations create
`Compute` instances, which a caller reaches through `Program.run` rather than by writing the
requests.

## Effect rows

A computation's signature ends in its effect row. A row with a *row variable* as its tail is open,
and a computation whose row is open can be used under any instances the tail stands for, which is
[row polymorphism](https://en.wikipedia.org/wiki/Row_polymorphism) over effects [@leijen2014koka]:

| Row | Meaning |
| --- | --- |
| `!{}` | closed and pure |
| `!{left, right}` | exactly these instances |
| `!{robust | rho}` | `robust` and whatever instances the row variable `rho` stands for |
| `!{robust | rho lacks robust}` | the same, with `rho` constrained not to contain `robust` |

Rows contain instances, not interface names. A call contributes its callee's row after static
substitution, and a `perform` contributes the instance it names. An `if` or a `case` joins the rows
of all its reachable branches, and since coverage is conservative, a branch the checker cannot
prove impossible still contributes its effects.

A `lacks` constraint keeps a row polymorphic computation that adds an instance of its own from
duplicating one already in the caller's tail. It is needed when an abstraction handles or
introduces an instance name that the tail could otherwise contain.

## Handler declarations

A handler declares the interface it handles, its input and answer types, its options, one return
clause, and one clause per operation:

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

The `return` clause receives the value of a computation that finishes. An operation clause
receives the request's arguments and may call `resume`, which continues the suspended computation
with a value. In `robustify(8.0)`, the request `robust.shrink(8.0)` reaches `half_weight`, whose
clause resumes with `4.0`, so `adjusted` is `4.0`, and the return clause passes `4.0` through.

QVR handlers are *deep*: the handler is reinstalled around the computation it resumes, so later
requests in that computation reach the same handler [@plotkin2009handlers; @kammar2013handlers].
Effects that a clause body performs itself go only to outer handlers.

### Handler options

| Option | Values | Meaning |
| --- | --- | --- |
| `coverage` | `total`, `partial` | whether every operation of the interface has a clause |
| `forwards` | `none`, `unknown` | whether a partial handler passes requests without a clause to an outer handler |
| `implementation` | `authored`, `foreign` | whether the clauses have QVR bodies or are given by a runtime provider |
| `introduces` | instance names | the effects the handler's own clauses perform, which the row of the handled computation gains |

A total handler removes the instance it handles from the row of its body, and a partial handler
leaves it there. Forwarding and coverage are separate options: a request for an operation that a
partial, nonforwarding handler has no clause for is an error, not an implicit request to an outer
handler.

### Resumption grades

A *resumption grade* says how many times a handler clause may resume its continuation, that is,
the rest of the computation after the request. The grades are the usage multiplicities of
[substructural type systems](https://en.wikipedia.org/wiki/Substructural_type_system), as in
quantitative type theory [@atkey2018syntax]:

| Grade | Contract | Typical clause |
| --- | --- | --- |
| `0` | never resumes | an abort |
| `aff` | resumes at most once | a clause that may stop the computation |
| `1` | resumes exactly once | a state read or update |
| `omega` | resumes any number of times | a search clause that explores each alternative |

The checker records each clause's grade, and the reference machine enforces it. A grade of `omega`
lets the continuation run more than once, which also repeats the requests that follow. Each handler installed in that continuation must thus be one the
runtime can copy for a second resumption.

## Scoped instances

`with instance` creates an instance that exists only inside its block:

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

`fixed_reply` answers every `get` with `7` and ignores the value given to `put`, so `local_read()`
returns `7` after `put(3)`. A handler that implements a mutable cell would instead pass the value of
`put` to the resumed computation for later `get` requests.

A total handler must remove the scoped instance before control leaves the block, or the checker
reports `qiec-instance-escape`. Two nested `with instance cell ...` blocks create two instances,
though the inner name shadows the outer.

## Foreign handlers and providers

A handler declared `implementation=foreign` gives only the signature of each clause, with no
`=>` body. A *runtime provider* binds such a handler to an implementation held in the running
Python process (see [runtime providers](execution-and-tooling.md#runtime-providers)), the closest
standard notion being a [foreign function interface](https://en.wikipedia.org/wiki/Foreign_function_interface).
The serialized module holds the handler's type, its operation coverage, and its grades, but no
callable. At invocation, the selected provider binds the handler's stable identifier to an
implementation. A provider configuration file holds provider names and data only, and it cannot
inject code.

The built-in `core` provider gives the implementations that program execution, deductions, and the
prelude use. A structural model composes `core` with a provider that holds its compiled PyTorch
modules.

## Common failures

- **Request absent from the row.** Add the instance to the row, or make the row polymorphic when
  the abstraction should keep the caller's effects.
- **Wrong applied interface.** `State[Int]` and `State[String]` are distinct, and a handler must
  match the instance's exact application.
- **Instance escape.** Handle a scoped instance before leaving its block.
- **Grade violation.** Make the number of `resume` calls agree with the declared grade.
- **Missing provider.** Attach a provider that implements the foreign handler, or write an authored
  handler.

The [effects tutorial](../../tutorials/qvr/10-effects-and-handlers.md) builds and traces an authored
handler.
