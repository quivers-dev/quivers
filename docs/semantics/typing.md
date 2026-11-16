# Typing and checking

QVR is checked twice. The source checker assigns objects, spaces, axes, and family shapes to the
categorical and probabilistic declarations. Elaboration then translates every executable declaration
into the Quivers Indexed Effect Core (QIEC), the typed core language of quivers, whose checker assigns
kinds, indexed types, and effect rows to computations. The first check asks whether the pieces connect:
whether a morphism returns what the next one expects, and whether a family receives suitable arguments.
The second also asks what running a computation may request: draws, scores, state, or another declared
effect.

A [typing judgment](setting.md#7-typing-judgments) is derived by
[inference rules](https://en.wikipedia.org/wiki/Rule_of_inference): when the premises above the line
hold, the conclusion below it holds. A context such as $\Gamma$ lists the names in scope with their
types.

## 1. The two judgments {#1-the-two-judgments}

The source judgment

$$
\Gamma \vdash e : A \rightsquigarrow B
$$

states that in the context $\Gamma$ of module declarations, the expression $e$ is a morphism from $A$ to
$B$, weighted, stochastic, or continuous ([Morphisms](morphisms.md)). A program body also threads a
context $\Phi$ of the values bound so far, from left to right:

$$
\Gamma ; \Phi \vdash s \dashv \Phi',
$$

which states that the step $s$, checked with the bindings $\Phi$, leaves the bindings $\Phi'$: a `sample x`
adds $x$, and an `observe` adds nothing. The semicolon separates the two contexts.

The QIEC judgment

$$
\Delta ; \Gamma_v \vdash c : T\ !\ \epsilon
$$

states that the computation $c$, with the static parameters $\Delta$ and the run-time values $\Gamma_v$,
returns a value of type $T$ and may make requests to the effect instances in the row $\epsilon$. $\Delta$ is a
*telescope*, an ordered list of binders in which later types may mention earlier variables
[@debruijn1991telescopic]. The QIEC judgment is the one that `qvr run`, the language server, serialization,
and the analysis of what each transpilation target supports all rely on.

## 2. Source types and objects

An `object` declaration declares finite sets and spaces alike, and its value decides which:

```qvr
object Row : FinSet 100
object State : {cold, warm, hot}
object Weight : Real 4
object Probability : Simplex 4
```

`FinSet` and enumerations denote finite sets, and `Real`, `Simplex`, `Sphere`, `Ball`, and the matrix
constructors denote spaces ([Types and spaces](types-and-spaces.md)). Products use `*`. Coproducts and
categorial types are allowed in the contexts given on [Types and spaces](types-and-spaces.md) and
[Schemas](schemas.md). There is no separate `space` declaration, and `atom`, `sort`, and `binder` do not
declare objects: atomic syntactic categories come from `category` or a deduction, and sorts and binders
belong inside a `signature`.

The type of a declared morphism is fixed by its domain and codomain:

$$
\frac{A, B\ \text{objects}}
     {\Gamma \vdash \mathsf{morphism}\ f : A \to B \ :\ A \rightsquigarrow B}.
$$

Composition requires the codomain of the left operand to equal the domain of the right one, and the two
operands to have the same composition rule. The tensor product combines domains and codomains
componentwise. [Expressions](expressions.md) gives the equations, and
[Composition rules](composition-rules.md) the laws they need.

## 3. Program contexts {#3-program-contexts}

A program declaration fixes a domain and a codomain and introduces its body:

```qvr
object Row : FinSet 8

program regression(offset : Real, y : Tensor[Real]([Row])) : Row -> Row
    sample location <- Normal(loc=0.0, scale=2.0)
    sample scale <- HalfNormal(scale=1.0)
    let mean = location + offset
    observe y : Row <- Normal(loc=mean, scale=scale)
    return y
```

The source checker classifies the names in a body:

| A name that is | Becomes |
| --- | --- |
| declared in the static telescope `[...]` | a type, effect, index, object, or morphism parameter in $\Delta$ |
| declared with a type in the runtime telescope `(...)` | an external value parameter in $\Gamma_v$ |
| declared as a bare domain parameter | a projection of the declared domain |
| bound by `sample`, a call, or `let` | a local binding in $\Phi$ |
| the response of an `observe` | an observation input |
| given by `via` | an index map, an integer input (role `index_map` in the API) |
| read and bound nowhere | a data input |

Elaboration records `offset` and `y` above as typed parameters of the entry point.
Their annotations are checked against every use. The value of `offset` is given
when the program runs; `observe y` assigns `y` its observation role. The same
checked parameter is inferred when its annotation is omitted and the body
determines its type. An input whose type remains ambiguous requires an annotation.

The static and runtime telescopes have different judgments. If $\Delta$ is
well formed and $\Delta\vdash T:\mathsf{Type}$, a runtime binder extends
$\Gamma_v$ with $x:T$. It does not extend the effect row. The body is then
checked under $\Delta;\Gamma_v$, and specialization substitutes a static
valuation into both its input types and result type. Thus declarations of
external values and ordinary computation parameters use the same rule.

A static `[n : Nat]` may index `Tensor[Real]([n])` and act as the finite
ordinal in a domain or plate annotation. These occurrences share one binder,
so specialization at $n=k$ substitutes the same $k$ into every shape and
finite-object position. A module-level object cannot refer to a binder local
to one program; the scope is the program signature and its body.

Natural-number addition has the static rule

$$
\frac{\Delta\vdash a:\mathsf{Nat}\qquad\Delta\vdash b:\mathsf{Nat}}
     {\Delta\vdash\mathsf{nat\_add}(a,b):\mathsf{Nat}}.
$$

It introduces no term into $\Gamma_v$ and no effect. Under valuation
$\delta$, its denotation is $\llbracket a\rrbracket_\delta+
\llbracket b\rrbracket_\delta$. Canonical sums collect constants and
nominal summands, including repeated variables. Substitution rebuilds
that normal form while leaving rigid skolems untouched; a sum cannot
hide an escaping skolem from the ordinary scope checks. Unproved Nat
subtraction is absent. An exact affine solution must have nonnegative
integer coefficients and constant before it becomes a symbolic Nat term.

Binding solves exact affine equations from checked shapes. It rejects
conflicting evidence and forced negative or fractional extents, while
underdetermined extents remain parameters requiring further evidence.
A shape-only zero extent does not require an inhabitable categorical
finite-set object. The
[index reference](../reference/qvr/types-and-indexed-families.md#natural-number-extent-addition)
distinguishes this arithmetic from user-defined index constructors.

Distribution shape signatures use the same dependent context. Their local
Nat telescope quantifies parameter and sample-event dimensions; each
application freshens those binders before collecting exact shape equations.
For instance, an additive log-ratio normal with Gaussian coordinate extent
$d$ takes a location tensor of shape $[d]$ and a Cholesky tensor of shape
$[d,d]$, and returns an event of shape $[d+1]$. The optional reference
component changes coordinate order without changing these extents.

These are dimensions of one parameter value and one sample event, before
leading plate batch axes. Parameter-to-parameter relations apply even when
the sample is scalar. Source inference may solve its own open extents,
while the kernel solves only fresh signature variables; neither operation
binds a caller's rigid skolem by its display name. Existing event-axis
metadata is shorthand for this contract, rather than a separate typing rule.

A runtime `(count : Nat)` has a different judgment: it extends $\Gamma_v$
with an integer-valued input subject to the restriction $count\geq 0$. Its
lowered parameter uses the arithmetic carrier `Int` and carries the checked
constraint `nonnegative`. This constraint survives serialization and is
validated at native and reference execution boundaries; it does not turn the
value into a static index.
At a program call, a nonnegative integer literal or an input carrying this
constraint satisfies a runtime `Nat` formal. A computed or sampled `Int` does
not establish the restriction; without a refinement proof mechanism, the
checker conservatively refuses that actual argument.

The common rule across declarations concerns the stage of a binding,
rather than one universal interpretation of parentheses. A `define` or
`program` static telescope extends $\Delta$; its value telescope extends
$\Gamma_v$. Rule and schema brackets instead bind static pattern variables,
whose substitution fixes a morphism's types before its arguments are used.
In `family Vec[A : Type](n : Nat) : Type`, both `A` and `n` belong to
static information: `n` is an index that constructors and case analysis
may refine. Constructor fields are values. Likewise, an effect interface's
type and index parameters are static, while operation arguments and resumed
values are runtime terms. Contraction inputs and encoder attachments supply
components under their own elaboration rules; they do not quantify new
index variables merely because their names occur in a header.

Distribution parameter names also bind no static indices. In
`Frechet(scale=s, concentration=nu)`, `nu` is a runtime real parameter,
even though it changes the response's tail and moment domains. An
annotation `(nu : Real)` checks its carrier; family validation checks the
admissible parameter range. Inferring a carrier or axis from use does not
infer whether a real parameter denotes a mean, a log-location, or a scale.
The selected family and any explicit regression parameter map determine
that interpretation. None of these declarations alone extends an effect row.

Checked scalar operations refine execution domains without introducing
new static binders. `positive_parameter(x)` checks finite positivity;
`log_probability(p)` checks a finite probability; `log1mexp(x)` checks
a nonpositive logarithm; and `log_ndtr(x)` computes a Normal log CDF.
Their checked signatures are `Real -> Real`; a carrier judgment alone
does not prove their domains. An active invalid argument raises explicitly.
A conditional that protects a partial operation must retain its branch
boundary: eager evaluation of both branches would change its semantics.
An executor that cannot preserve symbolic branching must refuse it.

### 3.1 Program steps

The main rules are

$$
\frac{\Gamma ; \Phi \vdash F(\bar a) : \mathrm{Sampleable}[T]}
     {\Gamma ; \Phi \vdash \mathsf{sample}\ x \leftarrow F(\bar a) \dashv \Phi, x : T}
\qquad
\frac{\Gamma ; \Phi \vdash F(\bar a) : \mathrm{Sampleable}[T]
      \qquad y : T\ \text{is given}}
     {\Gamma ; \Phi \vdash \mathsf{observe}\ y \leftarrow F(\bar a) \dashv \Phi}
$$

$$
\frac{\Gamma ; \Phi \vdash e : T}
     {\Gamma ; \Phi \vdash \mathsf{let}\ x = e \dashv \Phi, x : T}.
$$

`let x <- f(args)` checks a call of a computation or a program. The callee's result type, with its
arguments substituted, becomes the type of `x`, and the effects the callee leaves unhandled join the
caller's row. It is a different form from the pure `let x = e`.

`score name = e` requires $e$ to be a scalar log weight. It binds the value, which the trace records, and
performs `Score.add` in the elaborated computation. `return e` is checked against the declared codomain.
The elaborated entry point may also record data inputs that do not appear in the domain.

### 3.2 Plates, events, and index maps {#32-plates-events-and-fibrations}

An annotation after the bound name introduces a [plate](https://en.wikipedia.org/wiki/Plate_notation), a
batch axis of independent draws:

```qvr
object Group : FinSet 8

program hierarchical : Group -> Group
    sample tau <- HalfNormal(scale=1.0)
    sample group_effect : Group <- Normal(loc=0.0, scale=tau)
    observe y : Group <- Normal(loc=group_effect, scale=sigma)
    return y
```

The step options `[over=Category, iid_over=Group]` separate the event axes of a draw from its batch axes
([Morphisms §6](morphisms.md#6-axis-role-specifications)). The checker verifies the declared finite extents
and records the result as a `PlateShape`, which a transpilation target receives as a typed value.

Inside a grouped `marginalize`, `via=group_idx` requires one index map for each factor of the group set
([Programs as kernels §2.7](programs.md#27-grouped-marginalize-with-multi-observe-fibration)). For a product of
groups the positions are flattened in row major order, the last factor varying fastest. The values of an
index map are data inputs and must be valid integer indices when the program runs.

#### Observation weight checking

For `observe y : Row <- F(...) [weights=w]`, the weight has type `Real` or
`Tensor[Real]` over exactly the observation's batch axes. A declared scalar
broadcasts across those axes. The observation's event coordinates are excluded
from the weight shape: one multivariate observation receives one power after its
joint log density is formed. An otherwise unbound weight name is inferred as
a tensor over the batch axes, or a scalar `Real` when no batch axis is present.

The shape and element type are checked before execution; finiteness and
nonnegativity are checked on the supplied or computed values when the program
runs. Booleans are refused. The checked pure primitive `observation_weight`
retains the value after validation, and `scale_weight` applies it to the
observation's `LogWeight` before `WeightSum` or `SegmentSum`.
`scale_weight` implements the zero-weight branch defined in the
[observation semantics](programs.md#observation-weights).

A weight parameter is an ordinary input annotation and adds no effect. Its use
by `observe` retains the observation's `Score` effect; weighting does not
introduce another sampling request or modify the prior.

### 3.3 Effect annotations

The program option `[effects=[...]]` accepts `Sample`, `Score`, `Marginal`, and `Pure`. A `sample`
contributes `Sample`, an `observe` or `score` contributes `Score`, a `marginalize` block contributes
`Marginal`, and `let` and `return` contribute nothing. `Pure` requires the set of contributed effects to be
empty. These four names summarize the body. The precise effect type is the QIEC row that elaboration
computes ([Programs as kernels §2.8](programs.md#28-effect-signatures)).

## 4. QIEC kinds and types {#4-qiec-kinds-and-types}

A *kind* classifies static arguments as a type classifies run-time values. Static parameters, written in
square brackets, are used in checking and specialization, and range over `Type`, an index sort, `Nat`, or
`Effect`. Programs additionally abstract over objects and morphisms in their static
telescopes. Run-time parameters, in parentheses on a program or computation, are
values passed when it runs. Family declarations use parentheses for refinable
static indices instead; punctuation is interpreted with its declaration form.
In a telescope
such as `[A : Type, n : Nat](xs : Vec[A](n))`, both the element type `A`
and index `n` are in scope when `xs` is checked.

| Kind | Inhabitants |
| --- | --- |
| `Type` | run-time value types and applications of indexed families |
| an `index` sort | closed terms of its constructors |
| `Effect` | effect interfaces and their applications |
| `Nat` | static natural numbers, used as extents |

The primitive run-time types are `Unit`, `Bool`, `Int`, `Real`, `String`, `LogWeight`,
`Tensor[A]([d1, ...])`, `Sampleable[A]`, and `Site[A]`, and products are written `A * B`. `LogWeight` is a
[nominal type](https://en.wikipedia.org/wiki/Nominal_type_system), distinct from `Real` by its name.

An *indexed family* is a datatype whose type carries extra information, an index, that its constructors
fix: an [inductive family](https://ncatlab.org/nlab/show/inductive+family) [@dybjer1994inductive], called a
[generalized algebraic datatype](https://en.wikipedia.org/wiki/Generalized_algebraic_data_type) (GADT) in
programming [@xi2003guarded]. `Vec[A](n)` records an element type and a length:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))
```

`Nil` has length zero, and `Cons` of a head and a tail of length `m` has length `S(m)`, the successor of `m`.
`A` is a *parameter* of the family, the same in every constructor's result, and `n` is an *index*, which each
constructor fixes [@dybjer1994inductive]. Checking a constructor substitutes both telescopes and verifies
the family application in its result.

## 5. Case analysis on indexed families

A `case` over an indexed family has a *motive*, the result type as a function of the scrutinee's indices
[@mcbride2002elimination]. Matching a constructor introduces its static variables as *rigid* variables,
fresh constants that stand for unknown indices inside the branch and may not escape it
[@peytonjones2006simple], together with equations between the scrutinee's indices and the constructor's
result:

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

Here the motive is `Int` for every index. Every reachable branch must return the motive at its refined
index. A branch may be omitted only when the equations between closed index constructors show it
impossible: a vector of length zero cannot be a `Cons`, whose length is a successor. An equation the checker
cannot decide remains an obligation, and the checker does not assume that a branch is unreachable.

## 6. Computations and rows {#6-computations-and-rows}

An *effect row* lists the effect instances a computation may make requests to
[@leijen2014koka; @hillerstrom2016liberating]. A trace records the requests one run actually made, so a
conditional that makes a request in one branch still needs it in its row. Two instances of `State[Int]`,
such as a counter and an accumulator, are different entries of a row, and a handler of the counter handles
only the counter's requests ([Effects and type classes §8](effects.md#8-qiec-rows-and-handlers)). The
[effects guide](../guides/effects.md#qiec-effect-interfaces-and-rows) shows rows in declarations.

A computation's signature separates its static telescope, its run-time parameters, its result, and its row:

```text
define f[STATIC](VALUES) : RESULT !{INSTANCES | TAIL lacks INSTANCES} =
    COMPUTATION
```

Values and computations are checked separately. `let x = value` extends the value context and sequences
nothing. `let x <- computation` checks the computation on the right, binds its result, and joins its row
with that of the rest. `perform i.op[...]()` adds the instance `i` to the row, after its static arguments are
substituted.

A row with a tail variable is open, so a computation may be polymorphic in the rest of its row
([row polymorphism](https://en.wikipedia.org/wiki/Row_polymorphism)). A `lacks` constraint on the tail
forbids it from containing an instance that an abstraction handles or adds. A conditional joins the rows of
its branches, and a `case` joins the rows of every reachable branch. A call substitutes the callee's static
arguments and then joins the effects it leaves unhandled. A recursive call is checked against the declared
signature, so recursion needs no unrolling during checking.

## 7. Handlers and scoped instances {#7-handlers-and-scoped-instances}

A handler interprets a request and may resume the computation that made it. The *continuation* is the rest
of that computation with the request's answer filled in
([continuation](https://en.wikipedia.org/wiki/Continuation)). Each operation clause declares a
*resumption grade*, the number of times it may resume the continuation: `0` never, `aff` at most once, `1`
exactly once, and `omega` any number of times. These are the multiplicities of
[substructural typing](https://en.wikipedia.org/wiki/Substructural_type_system), as in quantitative type
theory [@atkey2018syntax]. A handler for abort can use grade `0`, and a handler for enumeration that tries
several answers needs `omega`.

`with instance` introduces a fresh instance, which may not escape its scope, in the manner of lexically
scoped handlers [@biernacki2020binders]. `handle i with h in c` checks that the handler is defined for the
exact applied interface of `i`, and then:

- a handler with a clause for every operation removes `i` from the row of the body;
- a handler without one keeps `i` in the row;
- the effects of the handler's clauses join the remaining row;
- a row variable keeps its `lacks` constraints.

The checker records each clause's grade and rejects an authored clause that visibly resumes more often than
its grade allows. The [reference machine](../developer/qiec.md#reference-evaluator), the library's interpreter for QIEC, also
counts resumptions at run time. Rigid indices, scoped
instances, and the evidence for them may not occur in the type or row that leaves their scope. The
separation of requests from their interpretation is that of algebraic effects and handlers
[@plotkin2009handlers]. QIEC adds named instances, indexed types, and resumption grades.

## 8. Program elaboration {#8-program-elaboration}

Every `program` elaborates to a named QIEC computation over the built-in instances `random : Random` and
`score : Score`:

| Step | QIEC form |
| --- | --- |
| `sample` | `perform random.sample(...)`, then a score of the chosen value |
| `observe` | `perform score.add(log_prob(...))` at the given value |
| `score` | `perform score.add(...)` |
| pure `let` | pure binding |
| `let x <- f(...)` | checked call and binding |
| `marginalize` over a finite family | fresh choice and weight instances under handlers that enumerate and collect |
| `marginalize` over another family | one ordinary draw, with no integral |
| `scan` | a generated recursive helper over a sequence of open extent |

Data inputs, observations, index maps, the program's parameters, and the free names found in the body
become typed parameters of the entry point, each with its role. Generated helpers are ordinary checked
computations, which the analysis of target support, serialization, and debugging all see.

Summing out a latent variable requires a registered family with finite support, and the QIEC path
enumerates categorical and Bernoulli supports exactly. A `reduction` on a family without finite support is
rejected. A `sample` inside a block that does not depend on the summed variable is moved before the block,
and one that does depend on it is rejected, since it would be a different draw for each value of the
variable.

## 9. Implementation {#9-implementation-correspondence}

| Stage | Modules |
| --- | --- |
| parsing | `quivers.dsl.parser`, `grammars/qvr/grammar.js` |
| source resolution and shapes | `quivers.dsl.compiler`, `quivers.dsl.step_resolution` |
| elaboration of programs | `quivers.dsl.program_elaboration` |
| QIEC checking | `quivers.qiec.checking`, `quivers.qiec.module` |
| coverage of indexed cases | `quivers.qiec.coverage` |
| reference execution | `quivers.qiec.evaluator`, `quivers.qiec.execution` |
| analysis of target support | `quivers.transpile.qiec_ir` |

`qvr check` parses, checks the source constraints, builds the
[compiled runtime](../guides/dsl-overview.md#compilation-pipeline), the module's PyTorch implementation, where it applies, and checks
the QIEC module. `qvr check --target` also analyzes which QIEC features the target supports. Transpilation
runs the target's renderer as well, which can report further unsupported constructs.

## 10. Guarantees and limits {#10-guarantees-and-limits}

The checker guarantees that accepted terms have well-kinded types, that calls substitute their telescopes,
that rows record every request, that handlers cover the operations they claim, and that generated
computations form a valid module. The reference machine checks attachments and resumption grades when a
program runs.

Some checks need concrete inputs: data inputs fix dynamic shapes, foreign handlers need a provider, and
each target renderer accepts its own subset of QIEC. An *attachment*, a value or handler held only in the
running Python process such as a PyTorch module, is outside the serialized core. The limits of the runtime
are on [Implementation correspondence and limits](adequacy.md), and those of transpilation on
[Transpilation correctness](transpile-correctness/index.md).
