# Typing and checking

QVR has two related type systems. The surface checker assigns objects, spaces,
axes, and family shapes to the categorical and probabilistic declarations a
user writes. Elaboration then translates every executable declaration into
the Quivers Indexed Effect Core (QIEC), whose checker assigns kinds, indexed
types, and lexical effect rows to computations. This page states the current
contract between those layers.

The checker asks two kinds of question. First, do the pieces connect: does
a morphism return the input shape the next morphism expects, and does a
family receive suitable arguments? Second, what may executing a computation
request: random draws, scoring, state access, or another declared effect?
The surface language emphasizes the first view; the core makes both views
explicit so that execution and translation can inspect them.

An inference rule is read from top to bottom: if its premises above the
line hold, its conclusion below the line is justified. A context such as
$\Gamma$ is the list of names and their types currently in scope. The symbols
compress a check that can also be stated in ordinary language.

## 1. The two judgments

The [notation key](../getting-started/concepts.md#7-reading-the-notation-and-finding-the-next-explanation) explains how to read a judgment, and [semantic environments](setting.md#5-environments) specify which names its context contains.

The first judgment below says that expression $e$ connects boundary $A$ to
boundary $B$. In the second, $\Phi$ is the list of values available before a
statement and $\Phi'$ is the list afterward: `sample x` extends that list,
while `observe` scores a supplied value. In the third, $T$ is the returned
value's type and $\epsilon$ records the effect instances the computation may
request. The semicolon separates different collections of assumptions.

The surface judgment has the schematic form

$$
\Gamma \vdash e : A \rightsquigarrow B,
$$

where $\Gamma$ contains module declarations and $A \rightsquigarrow B$ is a
categorical or probabilistic morphism. A program body additionally threads a
left-to-right value context $\Phi$:

$$
\Gamma;\Phi \vdash s \dashv \Phi'.
$$

The QIEC computation judgment is

$$
\Delta;\Gamma_v \vdash c : T\;!\epsilon.
$$

Here $\Delta$ is a static telescope, $\Gamma_v$ is a runtime-value context,
$T$ is the result type, and $\epsilon$ is a row of lexical effect-instance
identities. The QIEC judgment is the execution boundary used by `qvr run`,
the LSP, serialization, and target-capability analysis.

## 2. Surface types and objects

An `object` declaration is the sole object/space declaration form. Its
initializer determines the stratum:

```qvr
object Row : FinSet 100
object State : {cold, warm, hot}
object Weight : Real 4
object Probability : Simplex 4
```

`FinSet` and enumerations denote finite axes. `Real`, `Simplex`, `Sphere`,
`Ball`, and the matrix-space constructors denote continuous spaces. Products
use `*`; coproducts and the residuated forms are available in the declaration
contexts described by [Types and spaces](types-and-spaces.md) and
[Schemas](schemas.md).

The current grammar has no separate top-level `space` declaration. It also
does not use `atom`, `sort`, or `binder` as general-purpose object
declarations. Category atoms belong to `category` or deduction declarations;
structural sorts and binders belong inside a `signature`.

At the surface, a morphism type is determined by its declared domain and
codomain:

$$
\frac{A,B\ \mathrm{objects}}
     {\Gamma \vdash \mathsf{morphism}\ f:A\to B : A\rightsquigarrow B}.
$$

Composition requires the codomain of the left operand to match the domain of
the right operand, and both operands must carry compatible composition rules.
Tensoring combines domains and codomains componentwise. See
[Expressions](expressions.md) for the operator laws and
[Composition rules](composition-rules.md) for the algebraic preconditions.

## 3. Program contexts

A program declaration fixes a categorical boundary and introduces its body:

```qvr
object Row : FinSet 8

program regression : Row -> Row
    sample location <- Normal(loc=0.0, scale=2.0)
    sample scale <- HalfNormal(scale=1.0)
    let mean = location + offset
    observe y : Row <- Normal(loc=mean, scale=scale)
    return y
```

The surface checker classifies names in the body as follows:

| Source of a name | Role |
| --- | --- |
| declared program parameter | explicit value or scalar parameter |
| sample, call, or `let` result | local binding in $\Phi$ |
| observed response | observation input |
| `via` name | integer-valued fibration input |
| otherwise-free expression name | host-data input |

Elaboration records `offset` as a typed entry parameter. The host supplies
its value when the program runs, and tensor extents supplied by host data
are checked then.

### 3.1 Program steps

The principal surface rules are:

$$
\frac{\Gamma;\Phi\vdash F(\bar a):\mathrm{Sampleable}[T]}
     {\Gamma;\Phi\vdash \mathsf{sample}\ x\leftarrow F(\bar a)
      \dashv \Phi,x:T}
$$

$$
\frac{\Gamma;\Phi\vdash F(\bar a):\mathrm{Sampleable}[T]
      \qquad y:T\ \mathrm{is\ supplied}}
     {\Gamma;\Phi\vdash \mathsf{observe}\ y\leftarrow F(\bar a)
      \dashv \Phi}
$$

$$
\frac{\Gamma;\Phi\vdash e:T}
     {\Gamma;\Phi\vdash \mathsf{let}\ x=e\dashv \Phi,x:T}.
$$

`let x <- f(args)` instead checks a computation or program call. The callee's
instantiated result type becomes the type of `x`, and its residual QIEC row is
joined with the caller's row. This effectful binding is distinct from the pure
`let x = e` form.

`score name = e` requires a scalar log weight. It binds the value for tracing
and performs `Score.add` in the elaborated computation. `return e` checks
against the declared codomain, though the elaborated entry may also record
host inputs that do not appear in the categorical boundary.

### 3.2 Plates, events, and fibrations

An annotation after the bound name introduces a batch axis:

```qvr
object Group : FinSet 8

program hierarchical : Group -> Group
    sample tau <- HalfNormal(scale=1.0)
    sample group_effect : Group <- Normal(loc=0.0, scale=tau)
    observe y : Group <- Normal(loc=group_effect, scale=sigma)
    return y
```

Family options such as `[over=Category, iid_over=Group]` distinguish event
axes from independent batch axes. The checker verifies declared finite
extents and records the result as a `PlateShape`; a target renderer receives
that typed shape rather than reconstructing it from source text.

Inside grouped marginalization, `via=group_idx` requires one fibration input
per group factor. Product fibrations are flattened row-major, with the last
factor varying fastest. The fibration values themselves arrive from host data
and must be valid integer indices at execution.

### 3.3 Surface effect summaries

The optional `[effects=[...]]` program option accepts `Sample`, `Score`,
`Marginal`, and `Pure`. It is a checked surface summary:

- a sample contributes `Sample`;
- an observe or explicit score contributes `Score`;
- a marginalization block contributes `Marginal`; and
- `let` and `return` contribute no capability.

`Pure` is a sentinel requiring the actual set to be empty. It is not an
effect that a pure statement produces. These four names do not form the
program's precise effect type; QIEC elaboration supplies that type as a row of
lexical instances.

## 4. QIEC kinds and types

A *kind* classifies a static argument, just as a type classifies a runtime
value. A *telescope* is an ordered list of binders in which later entries may
refer to earlier ones. For instance, declaring an element type `A` before
`xs : Vec[A](n)` makes `A` available when checking `xs`. “Static” means used
in type specialization and checking; “runtime” means a value passed when
the computation executes.

An indexed family adds information to a type. `Vec[A](n)` records both the
element type and a length index. The constructors below say that `Nil` has
length zero, and putting a head in front of a length-`m` tail produces length
`S(m)`, the successor of `m`.

The core distinguishes static and runtime phases. Static binders use square
brackets and range over `Type`, an index sort, `Nat`, or `Effect`. Runtime
parameters use parentheses and carry value types.

The principal QIEC kinds are:

| Kind | Inhabitants |
| --- | --- |
| `Type` | runtime value types and indexed family applications |
| an `index` sort | closed static constructor terms |
| `Effect` | effect interfaces and their applications |
| `Nat` | static natural-number extents |

Primitive runtime types include `Unit`, `Bool`, `Int`, `Real`, `String`,
`LogWeight`, `Tensor[A]([d1, ...])`, `Sampleable[A]`, and `Site[A]`. Products
use `A * B`. An indexed-family application is also a runtime type:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))
```

`A` is a uniform static parameter. `n` is a refinable index. Constructor
checking substitutes both telescopes and verifies the explicit result family
application.

## 5. Indexed elimination

Pattern matching can reveal facts about an index. A value known to have
length zero cannot have been constructed by `Cons`, whose result has a
successor length. The checker uses that incompatibility to justify omitting
the branch. A *motive* specifies the result type as a function of the
indices; in the example below it is always `Int`. “Rigid branch-local” means
the branch's newly introduced static names cannot be freely chosen or leak
out into the result.

A `case` supplies a motive over the family's refinable indices. Matching a
constructor introduces rigid branch-local static variables and equations
between the scrutinee indices and the constructor result:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define length[A : Type, n : Nat](xs : Vec[A](n)) : Int !{} =
    case xs motive (m : Nat) => Int
        Nil =>
            return 0
        Cons[k](head, tail) =>
            let rest <- length[A, k](tail)
            return 1 + rest
```

Every reachable branch must return the motive under its refinement. Coverage
may omit a constructor only when equality of closed index constructors proves
the branch impossible. An unresolved equality remains an obligation; the
checker does not assume that an inconvenient branch is unreachable.

## 6. Computations and rows

The [two-state-cell example](effects.md#8-qiec-rows-and-lexical-handlers) explains why rows track instance identities. The [QIEC guide](../guides/effects.md#qiec-effect-interfaces-and-rows) places those rows in executable declarations.

An effect row records requests a computation may make. A trace records
requests it actually made. A conditional can mention a request in only one
branch and still require that request in its row. Two state instances, such
as a counter and an accumulator, can both have interface `State[Int]` while
remaining distinct. A handler for the counter handles only the counter's
requests; requests to the accumulator continue to its own handler.

A computation signature separates its static telescope, runtime parameters,
result, and effect row:

```text
define f[STATIC](VALUES) : RESULT !{INSTANCES | TAIL lacks INSTANCES} =
    COMPUTATION
```

The core checks values and computations separately. `let x = value` extends
the value context without sequencing. `let x <- computation` checks the
right-hand computation, binds its result, and joins its row with the
continuation. A `perform i.op[...]()` contributes the lexical instance `i` to
the row after static substitution.

Rows identify individual instances. Two instances of `State[Int]`
remain distinct. Open tails support effect polymorphism, while `lacks`
constraints prevent a tail from silently reintroducing an instance that an
abstraction handles or adds.

Conditionals join branch rows. Indexed cases join every reachable branch row.
Calls substitute the callee's static arguments before joining its residual
row. Recursive calls are checked against the declared signature, so recursion
does not require unrolling during type checking.

## 7. Handlers and scoped instances

DIPPL’s [continuation examples](https://dippl.org/chapters/03-enumeration.html#continuations) explain how the remainder of a computation can be passed as a function [@goodman2014design]. A handler’s resumption uses this idea; the grade below controls how often the function may be invoked.

A handler supplies an interpretation for a request and may resume the
computation that made it. The *resumption* is the remaining computation with
the request's answer supplied. Grade `0` allows no resumption, `aff` at most
one, `1` exactly one, and `omega` unrestricted use. An abort handler can use
grade `0`; an enumeration handler that explores several answers needs
`omega`.

`with instance` introduces a fresh lexical identity. That identity may not
escape its scope. `handle i with h in c` verifies that the handler is defined
for the exact applied interface of `i`:

- a total handler removes `i` from the body's row;
- a partial handler retains it;
- effects introduced by handler clauses join the residual row; and
- open tails retain their `lacks` constraints.

Each operation clause declares a resumption grade: `0`, `aff`, `1`, or
`omega`. Static checking records the grade and rejects evident authored-clause
violations; the reference evaluator also enforces the dynamic number of
resumptions. Branch-local indices, scoped instances, and their evidence may
not occur in the result type or row that leaves their scope.

Algebraic-effect handlers provide the general separation between operation
requests and their interpretations [@plotkin2009handlers]. QIEC adds the
specific lexical identities, indexed types, and resumption checks described
on this page.

## 8. Program elaboration

Every `program` lowers to a named QIEC computation over canonical lexical
instances `random : Random` and `score : Score`:

| Surface step | Core form |
| --- | --- |
| `sample` | `perform random.sample(...)`, then score the chosen value |
| `observe` | `perform score.add(log_prob(...))` at supplied data |
| `score` | `perform score.add(...)` |
| pure `let` | pure binding |
| `let x <- f(...)` | checked call and bind |
| finite `marginalize` | fresh choice and weight instances under enumeration/collection handlers |
| non-enumerable `marginalize` | one ordinary random sample; no exact integral |
| `scan` | generated recursive helper over an open sequence extent |

Data, observations, fibrations, explicit program parameters, and discovered
host names become typed entry parameters with recorded roles. Generated
helpers are ordinary checked computations and thus remain visible to
capability analysis, serialization, and debugging.

Finite marginalization requires an enumerable registry family. The QIEC path
handles categorical and Bernoulli-family supports exactly. A requested
`logsumexp`, `sum`, or `mean` reduction on a family without finite support is
rejected. A sample inside a finite marginal block is hoisted when it is
independent of the enumerated latent; a dependent draw is rejected because it
would denote a different stochastic program for each support value.

## 9. Implementation correspondence

The source protocol `qvr-source/v0.20` lowers to `qiec-core/v1alpha1`. These
identify the serialization protocols. The principal
implementation boundaries are:

| Boundary | Implementation |
| --- | --- |
| source parsing | `quivers.dsl.parser` and `grammars/qvr/grammar.js` |
| surface resolution and shapes | `quivers.dsl.compiler` and `step_resolution` |
| program-to-core elaboration | `quivers.dsl.program_elaboration` |
| QIEC checking | `quivers.qiec.checking` and `quivers.qiec.module` |
| indexed coverage | `quivers.qiec.coverage` |
| reference execution | `quivers.qiec.evaluator` and `quivers.qiec.execution` |
| target capability analysis | `quivers.transpile.qiec_ir` |

`qvr check` runs parsing, surface constraints, classic compilation where
applicable, and QIEC validation. `qvr check --target` adds core-capability
analysis. Full transpilation runs the renderer as well, which can report
additional unsupported surface constructs.

## 10. Guarantees and limits

The implementation and tests establish a checked correspondence: accepted
terms have well-kinded types, calls substitute their telescopes, lexical rows
track requests, handlers satisfy coverage, and generated computations validate
as a module. The evaluator then checks provider attachments and resumption
grades at the runtime boundary.

Further checks occur when concrete inputs and providers are available: host
tensors supply dynamic shapes, foreign handlers require attached providers,
and target renderers enforce their supported subsets of QIEC. Neural
attachments also remain outside the stable core. See
[Implementation correspondence and limits](adequacy.md) and the
[transpilation-correctness contract](transpile-correctness/index.md) for those
boundaries.
