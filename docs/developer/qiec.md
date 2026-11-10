# Quivers Indexed Effect Core

!!! tip "Looking for the language reference?"

    This page documents kernel internals, serialization, and the target runtime
    ABI. For source syntax, typing rules, execution commands, tutorials, and
    user-facing failure modes, start with the [QVR language
    reference](../reference/qvr/index.md).

The Quivers Indexed Effect Core (QIEC) is the typed core every executable
declaration of a QVR module elaborates to. Indexed families, effect
interfaces, instances, handlers, and typed computations are its surface
directly; a probabilistic `program` is domain-specific notation for a named
computation over the module's canonical `random` and `score` instances. The
parser reads the whole surface into one source AST, Quivers projects the
indexed declarations through Didactic's public `GADT` API, and the module
lowers through the exact `qvr-source/v0.20` to `qiec-core/v1alpha1` route,
checked against both Didactic's compiled first-order theory and the Python
reference kernel's deterministic serialization contract.

QIEC makes two boundaries explicit. First, the **stable core boundary** contains
only typed, deterministic data: kinds, telescopes, indexed-family declarations,
effect interfaces, rows, handler signatures, value and computation terms,
source provenance, and equality evidence. Second, the **[runtime attachment](../tutorials/qvr/10-effects-and-handlers.md#5-decide-between-authored-and-foreign-handlers)
boundary** associates stable identifiers with process-local values and handler
implementations. Host callables and mutable resources cannot cross the stable
core boundary. This note first describes the static kernel; it then turns to
serialization and evaluation, gives four complete examples, and closes
with the typed IR and target-capability contract used by the transpilers.

## Read the core through a small program

A source program might draw `mu`, compute a Normal density for an observed
`y`, and return `mu`. Its core representation spells out the requests:
construct a distribution, ask `Random` for a draw, ask `Score` to accumulate
a log weight, and return the value. A sampling runtime, a replaying runtime,
and a target-language implementation can interpret those same requests in
different operational contexts while retaining the checked structure.

The stable core stores descriptions that can be serialized, such as an
operation's identity and argument types. A runtime attachment supplies a
process-local implementation, such as a sampler or neural module. Serializing
the description retains the operation's identity and types. A process loading
that module must obtain the implementation from a matching runtime provider.

Several terms below describe how those checks remain precise:

| Term | Role in the check |
|---|---|
| Telescope | An ordered list of typed binders; later entries may depend on earlier ones. |
| Refinable index | Type-level information, such as a vector length, that a constructor branch can make more specific. |
| Equality evidence | A checked justification for treating two index expressions as equal within a scope. |
| Effect row | The effect instances a computation may still request. |
| Continuation | The remaining computation after an operation request. |
| Resumption grade | The permitted number of times a handler can continue that remaining computation. |
| Stable identifier | A persistent identity used for dispatch, independent of a user-facing display name. |

The [indexed-data](../tutorials/qvr/09-indexed-data.md) and
[handler](../tutorials/qvr/10-effects-and-handlers.md) tutorials work these
ideas through concrete values before the kernel details below.

## Static kernel

The kernel separates family parameters from refinable indices. A
`FamilyDecl` fixes both telescopes, while each `ConstructorDecl` specifies its
fields and result indices. Case analysis uses an explicit `CaseMotive`; branch
refinement introduces rigid skolems and local equality evidence that cannot
escape the branch in which it was established.

Coverage is conservative. QIEC omits a constructor only when distinct closed
constructors certify that the branch is impossible. If refinement is unknown,
the branch remains part of both the coverage obligation and the inferred effect
row. This policy may reject some valid programs until a stronger solver is
available, but it does not erase effects on the basis of an unproved
contradiction.

Effects are parameterized interfaces. Thus `State[Int]` and `State[String]`
share a declaration but remain distinct applied interfaces. Each lexical
instance also receives its own stable `EffectInstanceId`, so two state cells of
the same type remain distinguishable in an effect row. Operations may bind
additional static arguments, and request checking substitutes both the
interface and operation telescopes before checking argument and result types.

Handlers state their coverage and resumption contracts in serializable data.
The kernel records coverage (`total` or `partial`) separately from
unknown-operation forwarding, as well as each clause's resumption grade `0`,
`aff`, `1`, or `omega`; the reference evaluator enforces these grades
dynamically. A total handler removes exactly the matched lexical instance from
the row; a partial handler retains it; and an explicitly forwarding partial
handler passes structurally uncovered operations to an outer handler.

## Stable serialization

`quivers.qiec.dumps` and `quivers.qiec.loads` encode the stable core using the
versioned `qiec-json/v1` envelope and the `qiec-core/v1alpha1` ABI identifier.
The codec is an allowlist over QIEC records, identifiers, enums, tuples, bytes,
and JSON scalars. It rejects unknown node tags, malformed records, non-finite
floats, runtime attachments, and host callables. Encoding is deterministic, so
the same core graph yields the same serialized JSON text; UTF-8 encoding thus
yields the same bytes.

These format and ABI identifiers belong to the serialization boundary. An
authored effect declaration contains only its name, static telescope, and
operation signatures.

This format is intentionally independent of Didactic's internal model and
Panproto's generic carrier. Didactic checks an ephemeral GADT projection and
negotiates the exact source and target versions before QVR lowering completes,
while Panproto can carry the canonical envelope without inspecting its nodes.
QIEC retains ownership of stable identities, its uniform-parameter/refinable-
index distinction, the typed allowlist, malformed-node rejection, and reference
semantics.

## Reference evaluator

The evaluator gives `Return`, `Bind`, `Perform`, `Handle`, and indexed `Case` a
small CEK-style operational semantics. Handlers are lexical and deep: a resumed
continuation reinstalls the matching handler, while effects performed by the
clause body itself are offered only to outer handlers. Dispatch checks the exact
applied interface as well as the lexical instance and operation identifier.

Resumption grades are enforced dynamically at the attachment boundary. A
grade-`0` clause cannot resume; an affine clause may resume at most once; a
linear clause must resume exactly once; and an unrestricted clause may resume
multiple times only when the captured context is declared duplicable. Each
multi-shot branch extends the dynamic address with a resumption path, preserving
the identity needed by traces and replay.

The built-in prelude currently supplies declarations and reference handlers for
the following interfaces:

- `Random` and `Score`, including drawing, conditioning, intervention/replay,
  and trace forwarding;
- `State[S]` and `Abort[E]`;
- `Choose` for multi-shot nondeterminism; and
- `Weight[K]` for caller-supplied accumulation operations.

These handlers specify the reference behavior in executable form; backends
can implement that behavior with their own optimized routines. Runtime samplers, observation tables, semiring
operations, state cells, and trace recorders remain process-local attachments.

A distribution construction evaluates to a
[`RuntimeDistribution`](../api/qiec/distributions.md), which samples and
scores through an installed
[`DistributionBackend`](../api/qiec/distributions.md). The reference backend
implements every family of the semantic registry in plain Python: the scalar
core beside the backend and the structured and compositional families in
[`quivers.qiec.reference_families`](../api/qiec/reference_families.md), with
the small linear algebra the matrix families need written out rather than
taken from a host library. Each density is held to the torch definition the
transpile probes use, or to the torch runtime's own definition where torch has
no class, so the reference machine is an oracle a host can be checked against.
A family whose draw has a size no parameter fixes, the LKJ families, reads it
from the constructed type's event extents. The `Transformed` chain names its
transforms as a comma-separated string over `exp`, `log`, `sigmoid`, `logit`,
`softplus`, `tanh`, and `neg`, applied left to right.

### Entry-point execution boundary

A checked module's executable entry points are its `define` computations and
its programs, listed by [`entry_points`][quivers.qiec.entries.entry_points]
and invoked by [`invoke_entry`][quivers.qiec.entries.invoke_entry]. `qvr run
FILE ENTRY [JSON ...]`, the REPL's `:run`, and
[`Program.run`][quivers.program.Program.run] all call it, so an entry
validates its arguments, selects its providers, traces, and fails with the
same `qiec-run-*` codes however it is reached.

A computation is selected as a `NamedComputation`, its static telescope
specialized (`--static NAME=TERM` supplies a closed type, index, or effect
application for each binder), its value arguments validated, and its body
evaluated with a fresh attachment table under the configured providers
(`--runtime FILE.json` selects the built-in `core` runtime or installed
`quivers.qiec_runtime` providers without placing executable objects in the
configuration file).

A program is run by [`sample_program`][quivers.qiec.program_runtime.sample_program]:
the module is extended with a wrapper computation that handles the program's
`random` instance with a conditioning replay of the given sites (`--site
NAME=JSON`) inside a scoring draw of every other site, and its `score`
instance with an accumulator, and returns the program's value paired with the
accumulated log weight. The wrapper is checked like any other computation
before it runs. The program's parameters are its data, observations,
fibrations, and scalars, given positionally or by name (`--data NAME=JSON`);
its open extents are read off the data, and an extent no data fixes, a
template's object parameter, is supplied as a static argument. Each
invocation owns its state:
observations and conditioned sites are arguments of the run, the module's
parameter store answers `Param` requests from the values the run is given,
draws use the reference generator (`--seed N` reseeds it), the score
accumulator and the trace belong to the wrapper's attachment table, and no
handler stack outlives the call. [`run_program`][quivers.qiec.program_runtime.run_program]
is the replaying form, which requires every site and scores the log joint.

`--fuel STEPS` bounds the number of evaluation steps, since a recursive
computation may otherwise never return; exhaustion is the `qiec-run-fuel`
diagnostic. `--trace` writes stable events in plain mode, while `--json`
includes the result, specialized result type, runtime label, trace, the
entry's kind, and a program's log joint in one document.

The REPL exposes the same boundary through `:runtime`, `:run`, and `:detach`.
The Textual status bar shows the active runtime and the most recent result.
Argument, specialization, provider, validator, evaluation, and result failures
retain their `qiec-run-*` diagnostic codes across the CLI, REPL, and TUI.

### Programs on the reference machine

A compiled `MonadicProgram` is a facade over the same machine. Its
[`rsample`][quivers.continuous.programs.MonadicProgram.rsample] and
[`log_joint`][quivers.continuous.programs.MonadicProgram.log_joint] run the
program's kernel encoding under the active effect handlers through the
installed [`ProgramEvaluator`][quivers.continuous.programs.ProgramEvaluator],
so a `clamp`, `do`, `trace`, or reweighting in scope applies to every draw
and every score, and the joint `log_joint` reports is the one `trace`
accumulates. Each host step is encoded over the values it reads (a draw's
arguments, the names a closure declares through
[`reading`][quivers.continuous.program_steps.reading], or every value bound
so far when it declares none), and every handler the run installs resumes in
tail position, so a program's encoding and its evaluation grow linearly with
its length. A run owns its handlers: a program a host step runs in turn (a
score closing over another program's joint, a sub-program drawn as a step) is
a separate invocation, whose sites the enclosing run's handlers neither
observe nor condition.

A draw from another program, `sample theta <- school_effects(0.6, School)`
or `sample (u, v) <- sub`, runs that program's steps in place under names of
the caller's own: the objects and scalars the draw names substitute for the
callee's template parameters, its local `z` becomes the site `theta$z`, and
the name it returns becomes `theta` itself, on the reference machine and the
torch runtime alike; a target spells such a site `theta__z`. A program over
object parameters is a computation whose telescope binds one index variable
per object, so `qvr run` invokes it with its statics.

A program that calls a named computation has no step table the runtime
compiler can build, so it compiles to a
[`CheckedProgram`][quivers.effects.checked_program.CheckedProgram]: a
morphism whose `rsample` and `log_joint` are `sample_program` and
`run_program` on the module's own entry point. The machine scores numbers,
so such a program keeps no gradient path to the tensors it is given; fitting
it is the transpile targets' work.

## QVR surface

The following examples use the normative grammar. Each block parses,
lowers to the core, validates against the kernel, and reaches a
canonical parse–emit fixed point in the documentation tests.

A length-indexed vector illustrates constructor refinement:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (m : Nat) => A
        Cons[n](head, tail) =>
            return head
```

The scrutinee type `Vec[A](S(n))` rules out `Nil`; matching `Cons` then
introduces a rigid local predecessor index. This refinement makes the single
branch exhaustive, but the local index cannot escape into the type of the
complete case.

Computations call one another by ordinary application, recursion included,
and branch on Booleans with `if`. Pure expressions share the `program` let
grammar: operators resolve to primitives of a closed registry by operand
type, an integer operand beside a real one is converted first (the kernel
term carries the `int_to_real`), tuples build finite products, and `t[i]`
projects a component.

<!-- compile: qiec -->
```qvr
define triangle(n : Int) : Int !{} =
    if n <= 0 then
        return 0
    else
        let rest <- triangle(n - 1)
        return n + rest

define split(x : Int, y : Real) : Real !{} =
    let pair = (real(x / 2) * y, x % 3)
    let scaled = pair[0] + real(pair[1])
    return max(scaled, 0.5)
```

Recursion runs on the reference machine's explicit stack, and on a
trampoline in every generated host runtime, so its depth is bounded by memory
rather than the host call stack. A run may be given a step budget, which
turns divergence into the stable `qiec-run-fuel` diagnostic.

A parameterized state effect separates an interface from its lexical
instances. The `left` and `right` declarations below have the same applied
interface but different lexical identities:

<!-- compile: qiec -->
```qvr
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

instance left : State[Int]
instance right : State[Int]

handler run_state[S : Type, A : Type] for State[S] : A -> A [coverage=total, implementation=foreign]
    get resumes 1
    put resumes 0

define exchange(next : Int) : Int !{left, right} =
    let previous <- perform left.get()
    perform right.put(next)
    return previous

define read_left() : Int !{} =
    handle left with run_state[Int, Int] in
        let current <- perform left.get()
        return current
```

The row on `exchange` retains both instances. By contrast, the total
`run_state` handler removes `left` from the row of `read_left`. The handler
declaration records a stable signature and resumption grades; its executable
clauses remain runtime attachments keyed by the derived handler identifier.

Probabilistic operations become instances of the same effect calculus:

<!-- compile: qiec -->
```qvr
effect Random
    draw : Real -> Real

effect Score
    score : Real -> Unit

instance random : Random
instance score : Score

handler replay for Random : Real -> Real [coverage=total, implementation=foreign]
    draw resumes 1

define replayed_model() : Real !{score} =
    handle random with replay in
        let x <- perform random.draw(1.0)
        perform score.score(x)
        return x
```

`replay` is total and handles the exact `random` instance. The remaining
`score` entry makes the residual effect explicit. Runtime replay tables,
samplers, and trace recorders do not enter the stable module.

Distributions are values of the same calculus. Applying a family of the
semantic registry, such as `Normal(mu, 1.0)` or `Dirichlet([1.0, 2.0, 3.0])`,
builds a `Sampleable[A]` whose element type the registry fixes; a list
literal is a `Tensor` whose leading dimension is its entry count, so a
nested literal is a matrix; `site("x")` names a sample site at the `Site[A]`
type its position expects; and `log_prob(d, x)` evaluates a distribution's
log density as a `LogWeight`. The prelude's `Random` and `Score` interfaces
need no declaration:

<!-- compile: qiec -->
```qvr
instance random : Random
instance score : Score

handler draw for Random : Real -> Real [coverage=total, implementation=foreign]
    sample[A : Type] resumes 1

handler accumulate for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

define model(mu : Real) : Real !{random, score} =
    let x <- perform random.sample[Real](site("x"), Normal(mu, 1.0))
    perform score.add(log_prob(Categorical([0.2, 0.8]), 1))
    return x

define run(mu : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle random with draw in
            model(mu)
```

On the reference machine a distribution value is a
[`RuntimeDistribution`][quivers.qiec.RuntimeDistribution] that samples and
scores through the installed
[`DistributionBackend`][quivers.qiec.DistributionBackend]; the core runtime
provider's `draw` and `score` handler kinds implement `draw` and `accumulate`
above. Each dynamic target spells the construction in its own library through
[`spell_distribution`][quivers.transpile.family_spelling.spell_distribution],
applying the target's parameterization conventions (a rate against a scale,
a complemented probability, a shifted support, a folded half-line family) so
that `log_prob` agrees with the registry's density on every host.

A `program` is domain-specific notation for a named computation, and
elaborates to one. Its data, observations, and fibrations become the
computation's typed parameters; a `sample` step performs `Random.sample`
on the module's canonical `random` instance, an `observe` step scores
`log_prob` of its family at the observation on the canonical `score`
instance, a `score` step adds its value as a weight, a `let` step binds a
pure expression, and a `marginalize` block becomes a helper computation
that allocates a `Random` instance for the latent, handles it with the
enumeration handler, collects the scope's weights under a `Weight`
instance, and answers the log marginal, which the enclosing scope scores.
Plates are typed: `xs : N <- Normal(0.0, 1.0)` samples a
`Tensor[Real]([|N|])` from a construction plated over `N`, and a grouped
marginalization's `via` fibration re-indexes the latent's arguments to the
observation rows and segments the weights back to the groups. A product
group `[over=[G, H]]` is one flat axis of extent `|G| * |H|`, and the
product fibration `[via=[g, h]]` its row-major flattening, the last
factor varying fastest. The block's `reduction` selects the enumeration
handler: `enumerate_marginal` answers the log-sum-exp of the weighted
shots, `enumerate_marginal_sum` their sum, and `enumerate_marginal_mean`
their average; each has an `enumerate_grouped_` counterpart that answers
one aggregate per group position, which is what a block nested inside a
grouped block adds to the enclosing group's weights. The nested block's
group must be the enclosing group's extent, identified with it position
by position, or a product with the enclosing axis as a factor, whose
per-position marginals are segment-summed onto the enclosing positions
and whose arguments shaped by the enclosing group are gathered by that
projection. A `sample` inside a block that reads nothing the block binds
is drawn once, before the block, since every shot shares it; a draw that
reads the latent is refused. A `sample` of a vector family written with
its entries spread, `Dirichlet(1.0, 2.0, 3.0)`, gathers them into one
concentration, and a single literal is the symmetric concentration at the
dimension the step's plate or the program's codomain fixes. Programs and
computations call each other by ordinary application: a program body
binds a call with `let x <- helper(args)`, and a computation calls a
program by its name.

<!-- compile: qiec -->
```qvr
object Obs : FinSet 4

define noisy(x : Real) : Real !{random} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    return y

program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let c <- noisy(a)
    observe y <- Normal(loc=c, scale=0.5)
    return c
export prog

define twice(y : Real) : Real !{random, score} =
    let first <- prog(y)
    let second <- prog(y)
    return first + second
```

The elaborated module records each program as a
[`ProgramEntry`][quivers.qiec.ProgramEntry]: the roles of its parameters,
its sites, and its return names, beside the computation itself.
[`run_program`][quivers.qiec.program_runtime.run_program] wraps the entry
point in a scoring replay of `random` and an accumulating `score`, so the
reference machine applied to data and a value for every site yields the
program's value and its log joint density; the tests hold that number to
the torch runtime's trace on the same models.

An input whose shape nothing in the program fixes, such as the
concentration vector of `sample probs <- Dirichlet(alpha)` written without
a plate, is typed over an index variable the program's computation binds:
the computation gains one `Nat` binder per open extent, recorded on the
entry's `telescope`, and its parameter reads `Tensor[Real]([alpha_extent])`.
A run reads each extent off the data, a call step from another program
takes the extent as its own, and an authored computation names it with an
index literal, `prog[3](alpha)`.

A draw through a declared kernel morphism reads the family's parameters
off the morphism's parameter map, whose learned tensors are typed
program inputs with the roles `weight`, `bias`, and `table`: a kernel
with a real domain applies an affine map to its conditioning row, the
step's arguments or the program's domain inputs, and the heads of the
family read their row blocks; a kernel over a finite domain reads a row
of a table at the element the step names. A morphism carrying
`[param_source=mlp]` interposes hidden layers, each an affine map
followed by `tanh`, with the widths the option spells (`mlp(a, b)`, or
`hidden_dim`, or two layers of sixty-four), and its inputs
`<m>_param_layer<i>_weight` and `<m>_param_layer<i>_bias` match the shapes
of the torch runtime's `MLPSource` layer by layer. An embedding
morphism, `[role=embed]`, is a Normal kernel over its finite domain whose
table concatenates the centres and log scales the torch runtime learns.
Under a plate the map applies at every row: an unbound conditioning name
of `observe y : Resp <- net(x)` becomes a `Tensor[Real]([|Resp|, width])`
data input, an unbound index of an embedding a `Tensor[Int]([|Resp|])`,
and each head is a comprehension over the rows. A composite drawn from
in one step, `sample h <- embed >> fan(head) >> out`, expands into one
site per factor before elaboration: `fan` over a `[replicate=k]`
morphism draws through each replica `head_0` through `head_{k-1}` and
bundles the draws, a kernel conditioned on a bundle reads its rows in
order, `stack(block, n)` composes `n` copies whose morphisms past the
first are declared afresh under `_copy<i>` names with parameters of
their own, and the branches of a tensor product `encoder @ decoder` at
the head of a chain each read their own factor of the program's domain.
The expanded module declares every replica and copy, so the entry's
inputs name each one.

The heads a kernel morphism's map fills follow the torch runtime's
parameterization of the same kernel, family by family: a location is
read as it is, a scale exponentiated and clamped at the scale floor, a
rate or concentration through a softplus lifted off zero or shifted by
a tenth, a probability through a sigmoid, and the logits of
`Categorical` or `Bernoulli` as one row over the codomain's elements.
A scalar family applied to a vector, `Normal(mu, 0.5)` with `mu` a
`Tensor[Real]([3])`, draws one coordinate per entry, the argument's
axes being the plate. A program calling a program with its domain
arguments alone, `let y <- inner(x)`, passes the callee's remaining
inputs through: each becomes an input of the caller under the callee's
name and role.

A `scan` in a chain, `sample h <- tok_embed >> scan(cell)`, expands to a
step program `<cell>__scan_step(x_t, h_prev)`, which draws the chain's
elements before the scan at one position's entry and applies the cell
to the result and the previous state, and to `let h = scan(<step>, xs)`
over the chain's input. The scanned input is a sequence, one entry per
position of an open extent the program binds, so a scanned domain
factor `Token` reads as `Tensor[Int]([token_extent])`. The let
elaborates to a helper computation `<program>__<h>_scan(t, xs, h, ...)`
that answers `h` once `t` reaches the sequence's length and otherwise
calls the step at `xs[t]` and recurs at `t + 1`, carrying the step's
inputs along; the initial state is zero, or, under `init=learned`, a
`weight` input `<cell>_scan_init`. The step's sites are reached
once per position, and `run_program` replays the `n`-th occurrence of
a site from the value keyed `"<site>@<n>"`, so the reference machine
scores the trajectory the torch runtime's `ScanMorphism.log_joint`
scores. No transpile target has a lowering that keeps the measure, since
the sequence axis is not an object the module declares, and the
boundary refuses the scan with `scan:no-lowering:<cell>`.

Logic programming uses the same handler and resumption rules:

<!-- compile: qiec -->
```qvr
effect Choose
    choose[A : Type] : A -> A

effect Weight[K : Type]
    add : K -> Unit

instance choice : Choose
instance weight : Weight[Real]

handler depth_first[A : Type] for Choose : A -> A [coverage=total, implementation=foreign]
    choose resumes omega

define searched() : Int !{weight} =
    handle choice with depth_first[Int] in
        let selected <- perform choice.choose[Int](1)
        perform weight.add(0.0)
        return selected
```

The `Choose` clause has an unrestricted resumption because a runtime attachment
may explore several alternatives. Handling `choice` leaves the parameterized
`weight` instance in the row. This factorization lets a search attachment and a
weight attachment compose without introducing another source calculus.

A `deduction` declaration is domain-specific notation for exactly this
factorization, and elaborates to it. Its items become a closed family
`<name>__Item` with one nullary constructor per atom and one constructor
per applied symbol the rules and lexicon mention, a slot typed `Int` where
the system uses it as a position and `<name>__Item` otherwise; a lexicon's
logical forms number their bound variables canonically, so alpha-equivalent
forms are one item. The module gains a `Search` effect, the operation
`choose[a : Type, n : Nat] : Tensor[a]([n]) -> a`, a `<name>__choice`
instance of it, a
`<name>__weight` instance of `Weight[K]` for the carrier `K` of the declared
semiring (`LogWeight` for `LogProb` and `Viterbi`, `Bool` for `Boolean`,
`Int` for `Counting`), and a `params` instance of `Param` through which
learned weights are read under the names the agenda engine keys its
parameters by: `<name>.lex.<index>` for a lexicon entry and
`<name>.rule.<rule>:<bindings>` for a rule, the bindings of its category
variables (every variable that is not a chart position) rendered in name
order by the `<name>__show` computation and joined by `|`. A slot holds a
position exactly when it is typed `Int`, so positions never enter a weight's
name and one weight serves a rule at every position. The recursive `<name>__derive`
computation chooses an axiom or a group of rules with premises of one
shape, derives each premise by recursion, matches it against the rule's
pattern by `case` analysis (a shared pattern variable is checked by the
family's `<name>__eq`), instantiates the conclusion, and adds the rule's
weight; a failed match performs an empty choice. A system whose rules all
relate `span` items derives by span, choosing split positions for the
premises and deriving each over its own span; any other system takes its
axioms and their weights as input. The entry `<name>__run` handles the
weight instance with a collecting handler and the search instance with a
handler that resumes once per alternative and combines the shots by the
semiring's addition, so its answer is the semiring sum over derivations of
the goal of the product of the weights along each: the inside weight the
agenda engine tabulates, to the depth the `depth` option bounds. A program
calls the deduction with `let chart = parse(D, sentence)`, which types
`sentence` as a tensor of tokens over one of the program's open extents,
and reads `chart.goal_weight()` as a real to score.

<!-- compile: qiec -->
```qvr
object Term : FinSet 16
object LogWeight : Real 1

deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the  #[learnable]
        "dog"  : N          = dog  #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]

program fit : Term -> LogWeight
    let chart = parse(AB, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit
```

Two runtime facilities make the search compose with the weights. A foreign
clause may answer with `TailResume(value)`, which resumes its continuation
on the machine's own stack, as an authored `resume` in tail position does,
rather than in a nested host call that stays pending; the collecting
`Weight` handler answers this way, so a multi-shot resumption captured
later inside its scope may run the continuation any number of times. And
a collecting handler configured `forkable` gives each shot its own copy of
the total so far, which is the context cloning an unrestricted resumption
needs.

### The fixture corpus

Six source files under `tests/fixtures/qiec/` exercise the surface end to end,
and every claim above about calls, handlers, programs, marginalization,
deductions, and parameter maps is tested against them by
`tests/test_qiec_fixture_corpus.py` and, on the transpile targets, by
`tests/transpile/test_qiec_fixture_equivalence.py`:

- `recursive_indexed_traversal.qvr`: `Nat` and `Vec[A](n)`, a recursive fold
  whose constructor branch refines the tail's length, called from another
  computation and specialized at a closed length.
- `authored_state_handler.qvr`: `State[S]`, an authored handler with `get`,
  `put`, and an explicit return clause, nested scoped instances, and a
  computation called from a clause.
- `effectful_probabilistic_helper.qvr`: an index sort of observation status, a
  GADT of measurements, a helper that scores a `Present` value and draws an
  `Absent` one under a site name, and a hierarchical program that calls it
  with both constructors.
- `grouped_marginalization.qvr`: a per-item class integrated out of a Normal
  mixture whose rows fibre into the items, against an independent `logsumexp`
  oracle.
- `ambiguous_deduction.qvr`: a prepositional-attachment grammar whose two
  derivations are summed under an unrestricted `Choose` handler and added to
  a program's score.
- `network_decoder.qvr`: a morphism parameterized by a network, observed
  through the canonical effects and backpropagated to the attached weights.

## Integration contract

The integration establishes five results. First, QVR has one typed surface
for indexed families, effect interfaces and instances, handler signatures,
stable computation terms, and programs. Second, Didactic and Panproto check
the first-order indexed-family projection, after which the Quivers kernel
checks branch refinement, effect rows, handler coverage, and resumption
contracts and lowers the surface, programs included, into a serializable
`QiecModule`. Third, the compiler attaches that checked module to its
environment and produces a `Program` whose execution runs the module's
computations. Fourth, `Lower` derives each program's target plan from its
checked computation and projects the complete module into `IRQiecModule`,
whose typed nodes retain declarations, rows, provenance, values, evidence,
computations, and resumption grades, so a renderer receives one lowered root.
Fifth, the CLI, TUI, REPL, language server, Pygments and tree-sitter
highlighters, TextMate grammar, Panproto migration assets, and transpilers
share the surface and diagnostic vocabulary.

A potential worry is that a common IR implies identical target capabilities.
The **QIEC capability boundary** blocks that inference. Pyro, NumPyro, PyMC,
Edward2, Turing, Gen, WebPPL, and Church lower the complete stable computation
graph through corresponding target-language implementations of the QIEC
runtime ABI. That lowering retains stable operation and instance identifiers,
dynamic addresses, lexical handler scope, indexed constructors, equality
transport, and the four resumption grades.

A program that calls a module computation, `let c <- noisy(b)`, reaches these
eight targets as a call in the model body. A pure callee whose body is a chain
of bindings is inlined by the plan before any renderer runs, so it costs the
target nothing; every other call is rendered against the callee's generated
entry point with the program's canonical `Random` and `Score` instances bound
to a **native operation table** the model body builds once, so a draw inside
the helper is a site of the model under the label the helper gives it and a
scored weight is a term of the joint. Each host supplies the table from its
own primitives: Pyro's `pyro.sample` and `pyro.factor`, NumPyro's
`numpyro.sample` and `numpyro.factor`, PyMC's `register_rv` and
`pymc.Potential`, Edward2's traced random variables and a one-point factor
distribution, Turing's `tilde_assume!!` and `@addlogprob!` reached through
closures placed in the model body, Gen's `@trace` and a factor distribution
traced under the `:qvr_factor` namespace, WebPPL's `sample` and `factor`,
and Church's `sample` and `factor`. A label the helper draws under more than
once in one run is counted, `noise`, `noise@1`, and so on, as the reference
machine replays it. The Python bridges also supply the host's own spellings of
the scalar primitives, so a helper's arithmetic on a drawn value stays on the
host's array and keeps its gradient.

The WebPPL target's runtime is written in WebPPL's own functional subset: a
name is bound once, loops are recursion, `try` is absent, and the only mutable
state is the global store, which holds the call and instance serials, the
ambient handler stack, and one cell per handler installation. WebPPL
trampolines every call, so a deep recursion consumes no host stack. A host
attachment written for WebPPL follows the same discipline: a mutable binding
holds its value in a global-store cell named by its `cell` field rather than
in a `value` field, and every function the runtime calls out of an object
field is reached through `apply`, since WebPPL compiles a member call as a
native one.

Stan, BUGS, and JAGS have no run-time call, so the plan they render is judged
by what it needs: the computations the program's calls reach, transitively,
or every computation when the module declares no program. Stan defines a
needed pure computation as a user-defined function when it is closed and
monomorphic, returns a scalar, and is composed of scalar bindings,
primitives with a Stan spelling, conditionals, calls, and recursion; the
model body then calls it as a transformed parameter. BUGS and JAGS refuse a
call outright as `call:graph:<callee>`, since a graph language has no
statement that runs a computation. Every computation a static target can
represent is defined in its output, called or not, so a module's entry points
stay where the target has a form for them; one the program never calls and
the target cannot represent is left out, since it is no part of the program
the output denotes. A needed computation the target cannot represent is
reported feature by feature as `qiec:capability:<feature>:<computation>`
before anything is placed in the target schema.

### Target runtime ABI

Each host-language renderer exports a QVR computation named `f` as
`qiec_f`. Its source parameters come first. The entry point also accepts a
static-specialization argument and three invocation-local mappings: `qiec_attachments`,
`qiec_handlers`, and `qiec_operations`. The specialization argument instantiates
the checked type, index, and effect telescope; a polymorphic entry point rejects
an omitted argument, a wrong arity, or a static argument of the wrong kind.
Python and Julia expose this ABI through optional or keyword arguments.
JavaScript uses trailing positional arguments, and Church accepts an optional
rest list. The generated `model` or `build_model` entry point is unchanged.

The renderer tests reparse every target's emitted QIEC source and run external
syntax checks when the target toolchain is available. Runtime tests exercise
Python, Julia, JavaScript, and Scheme entry points, covering stable-ID dispatch,
indexed cases, attachments, resumption grades, and handler lifecycle behavior.
The evidence covers those tested programs and operations; adequacy for all
well-typed programs would require a separate proof against the reference evaluator.

Attachment entries are keyed by their `qiec:attachment:*` identifier and must
be typed binding descriptors. The runtime checks the descriptor's structural
type against the checked reference and invokes its validator before exposing
its value; a raw host value is not an attachment. A mutable duplicable binding
also declares a `fork` callback, which produces a fresh value from the pristine
capture seed for each unrestricted shot.

Handler entries are keyed by `qiec:handler:*`. The generated manifest, rather
than the host attachment, fixes the handler's effect, coverage, forwarding
policy, clause set, and resumption grades. The attachment's `operations` table
is keyed by `qiec:operation:*`, and each clause receives `(request, resume,
context)`. The request contains the stable instance, effect, operation, static
arguments, value arguments, result type, source origin, and dynamic address.
The context contains the specialized handler manifest, static arguments, and
the current resumption-use count. A return clause receives
`(value, context)`. Result validators check values passed into `resume`, while
the handler output validator checks the clause or return-clause answer.

A stateful handler attachment uses `context_factory` to construct one context
per installation. Optional `on_enter`, `on_exit`, and `on_drop` callbacks form
an exact-once lifecycle. An unrestricted resumption may capture a handler only
when it declares `duplicable_context`; a mutable captured context must also
provide `fork_context`. Each shot, including the first, forks from an untouched
seed. Partial handlers forward structurally uncovered operations to an outer
handler without changing the checked forwarding policy.

Unhandled operations are keyed by the pair of `qiec:effect-instance:*` and
`qiec:operation:*` identifiers and receive the complete `request`. Thus neither
dispatch path assigns semantics to display names such as `Random`, `Score`, or
`Choose`. An address contains the stable source-site identifier, dynamic
frames, and the resumption path; each unrestricted shot appends its branch
number before evaluating the captured continuation. Python stores active
handler state in an invocation-local context, Julia stores it per task, and
Scheme uses a dynamically scoped parameter. JavaScript evaluation is
synchronous, so its scoped stack cannot overlap independent invocations.

This boundary represents QIEC-only and mixed modules without requiring a
synthetic probabilistic `program`; a mixed module retains both its probabilistic
nodes and its typed `IRQiecModule`. Acceptance still depends on the selected
target. Declaration-only modules remain valid on every target. The eight
host-language targets accept executable QIEC-only modules, while the three
static targets accept them only when every computation satisfies the subset
above. Mixed modules must additionally satisfy the selected target's ordinary
probabilistic support boundary. A renderer may omit declarations that have no
runtime effect, but it cannot drop a computation body or erase an unsupported
effect to make the module fit.

Recursive calls, scoped instances, and authored handler clauses are represented
in the checked core and exercised by the examples above. Their presence in the
core does not imply support on every target: the target-capability checks
apply to the reachable computation graph. Optimized handler implementations
and larger static-target subsets remain possible extensions, each requiring
preservation of the relevant execution and scoring behavior.

Panproto validates persisted objects written by earlier releases against
the version they were written under. The migration gate reads the original QVR history without
rewriting its content-addressed object IDs, so those fixtures continue to test
the compatibility contract rather than a regenerated approximation of it.
