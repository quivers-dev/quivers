# Quivers Indexed Effect Core

The Quivers Indexed Effect Core (QIEC) is the typed core language that every executable QVR
declaration elaborates to. It has indexed families of types, in the manner of
[generalized algebraic data types](https://en.wikipedia.org/wiki/Generalized_algebraic_data_type)
(GADTs) [@xi2003guarded], [effect rows](https://en.wikipedia.org/wiki/Row_polymorphism), and
[effect handlers](https://en.wikipedia.org/wiki/Effect_system) [@plotkin2009handlers]. Both the
[reference machine](#reference-evaluator), the library's interpreter for QIEC, and the transpilers
read it.

*Elaboration* is the compiler step that rewrites a source declaration into explicit QIEC terms.
Indexed families, effect interfaces, instances, handlers, and typed
computations are written in QIEC directly. A probabilistic `program` is notation for a named
computation over the two *effect instances* `random` and `score` that every module has built in,
where an effect instance is one named occurrence of an effect interface.
The source syntax, typing rules, and diagnostics of each form are specified in the
[QVR language reference](../reference/qvr/index.md).

A module passes through three tools on its way to QIEC. The [tree-sitter](https://tree-sitter.github.io/)
grammar parses the whole surface into one source AST. Quivers then projects the indexed
declarations through the public `GADT` API of [didactic](https://github.com/panproto/didactic),
the library whose `Model` types hold the data values of quivers, and didactic checks that
projection against its compiled first-order theory. The module finally lowers along the route
from the source protocol `QVR_SOURCE_VERSION` (in `quivers.dsl.qiec_lowering`) to
`qiec-core/v1alpha1`, the identifier of the core's
[application binary interface](https://en.wikipedia.org/wiki/Application_binary_interface) (ABI),
and the result is held to the deterministic serialization contract of the Python core.

## Terms

| Term | Meaning |
|---|---|
| Kind | The type of a static argument, such as a type or an index. |
| [Telescope](https://ncatlab.org/nlab/show/telescope) | An ordered list of typed binders in which later types may mention earlier binders [@debruijn1991telescopic]. |
| Parameter and index | The arguments of an [inductive family](https://ncatlab.org/nlab/show/inductive+family) that stay fixed across constructors (parameters) and those a constructor may refine, such as a vector length (indices) [@dybjer1994inductive]. |
| Equality evidence | A checked justification for treating two index expressions as equal within a scope. |
| Effect row | The effect instances a computation may still request [@leijen2014koka]. |
| [Continuation](https://en.wikipedia.org/wiki/Continuation) | The rest of the computation after an operation request. |
| Resumption grade | How many times a handler clause may resume its continuation: never (`0`), at most once (`aff`), exactly once (`1`), or any number of times (`omega`). These are the zero, affine, linear, and unrestricted multiplicities of [substructural typing](https://en.wikipedia.org/wiki/Substructural_type_system) [@atkey2018syntax]. |
| Stable identifier | A persistent identity used for dispatch, independent of the name shown to a user. |

## Serialized data and runtime attachments {#two-boundaries}

The serialized core contains only typed, deterministic data: kinds, telescopes,
declarations of indexed families, effect interfaces, rows, handler signatures, value and computation
terms, source provenance, and equality evidence. Python callables and mutable resources are
never part of it.

The runtime associates stable identifiers with values and handler implementations that live
only in the running Python process. An *attachment* is such a value
or handler, for instance a PyTorch module for a neural component, addressed by a stable QIEC
identifier and never serialized with the module (`RuntimeAttachments` in
[`quivers.qiec.evaluator`](../api/qiec/evaluator.md)). A *provider* is an object that binds
attachments and type validators into a run ([`RuntimeProvider`](../api/qiec/execution.md)).
The closest standard notion is a [foreign function interface](https://en.wikipedia.org/wiki/Foreign_function_interface).

Consider a source program that draws `mu`, computes a Normal density for an observed `y`, and
returns `mu`. Its core term constructs a distribution, asks `Random` for a draw, asks `Score` to
add a log weight, and returns the value. A sampling runtime, a replaying runtime, and a target
language can each interpret those requests in their own way while the checked structure stays
the same. Serializing the term keeps each operation's identity and argument types. A process
that loads the module must obtain the implementations from a matching provider.

## The static checker {#static-kernel}

The checker separates family parameters from refinable indices. A `FamilyDecl` fixes both
telescopes, and each `ConstructorDecl` gives its fields and result indices. Case analysis states
an explicit `CaseMotive`, the result type as a function of the scrutinee's indices
[@mcbride2002elimination]. Refining a branch introduces *rigid type variables*, fresh constants
that stand for an unknown index inside that one branch [@peytonjones2006simple], together with
local equality evidence. Neither may escape the branch in which it was introduced.

Coverage is conservative. QIEC omits a constructor only when distinct closed constructors prove
that the branch is impossible. If refinement is unknown, the branch stays in both the coverage
obligation and the inferred effect row. This policy may reject some valid programs, but it never
erases effects on the basis of an unproved contradiction.

Effects are parameterized interfaces, so `State[Int]` and `State[String]` share a declaration and
remain distinct applied interfaces. Each declared instance also receives its own stable
`EffectInstanceId`, which keeps two state cells of the same type apart in an effect row.
Operations may bind further static arguments. Checking a request substitutes both the interface
telescope and the operation telescope before it checks argument and result types.

Handlers state their coverage and resumption contracts as serializable data. The checker records
coverage (`total` or `partial`) separately from forwarding of unknown operations, and records the
resumption grade of each clause. The reference machine enforces the grades at run time. A total
handler removes exactly the matched instance from the row. A partial handler keeps it. A partial
handler that declares forwarding passes operations it does not cover to an outer handler.

## Stable serialization

[`quivers.qiec.dumps`](../api/qiec/serialization.md) and `quivers.qiec.loads` encode the
stable core in the `qiec-json/v1` envelope under the ABI identifier `qiec-core/v1alpha1`. The
codec is an allowlist over QIEC records, identifiers, enums, tuples, bytes, and JSON scalars. It
rejects unknown node tags, malformed records, non-finite floats, runtime attachments, and Python
callables. Encoding is deterministic: the same core graph yields the same JSON text, and thus
the same UTF-8 bytes.

The format and ABI identifiers belong to the serialized format. An authored effect
declaration contains only its name, its static telescope, and its operation signatures.

The format does not depend on the internal model of didactic or on the generic carrier of
[panproto](https://github.com/panproto/panproto), the schema and migration toolkit that versions
the QVR grammar. Didactic checks a temporary GADT projection and settles the exact source and
target versions before lowering completes. Panproto can carry the envelope without reading its
nodes. QIEC itself owns stable identities, the distinction between parameters and indices, the
typed allowlist, the rejection of malformed nodes, and the reference semantics.

## Reference machine {#reference-evaluator}

The *reference machine* is the library's interpreter for QIEC ([`quivers.qiec.evaluator`](../api/qiec/evaluator.md)).
It is a small-step [CEK machine](https://en.wikipedia.org/wiki/CEK_Machine) for `Return`, `Bind`,
`Perform`, `Handle`, and indexed `Case`, and it fixes what a checked program means operationally,
in the manner of a [definitional interpreter](https://en.wikipedia.org/wiki/Interpreter_(computing))
[@reynolds1972definitional]. Handlers are lexically scoped and deep [@kammar2013handlers]. A deep
handler is reinstalled around the continuation it resumes, while effects that the clause body
itself performs go only to outer handlers. Dispatch checks the exact applied interface as well as the instance and the
operation identifier.

The machine enforces resumption grades when it runs a handler clause. A grade `0` clause cannot
resume, an affine clause resumes at most once, and a linear clause resumes exactly once. An
unrestricted clause may resume several times only when the captured context is declared
duplicable. Each additional resumption extends the dynamic address with a resumption path, which
keeps the identity that traces and replay need.

The built-in prelude declares the following interfaces, each with reference handlers:

- `Random` and `Score`, with drawing, conditioning, intervention and replay, and trace forwarding;
- `State[S]` and `Abort[E]`;
- `Choose`, for nondeterminism that resumes more than once;
- `Weight[K]`, for accumulation with an operation the caller gives;
- `Compute[X, A]`, through which a program reaches a Python computation it depends on, such as a
  morphism or a closure;
- `Param`, through which a computation reads a learned parameter by name.

These handlers state the reference behavior in executable form. A backend may implement the same
behavior with its own optimized routines. Samplers, observation tables, semiring operations,
state cells, and trace recorders remain attachments.

A distribution construction evaluates to a [`RuntimeDistribution`](../api/qiec/distributions.md),
which samples and scores through an installed `DistributionBackend`. The reference backend
implements every family of the [distribution family registry](../api/qiec/families.md) in plain
Python. The scalar families sit beside the backend, and the structured and compositional families
are in [`quivers.qiec.reference_families`](../api/qiec/reference_families.md). The small amount of
linear algebra that the matrix families need is written out rather than taken from a library.

Each reference density is held to the PyTorch definition that the transpile probes use, or, where
PyTorch has no class for the family, to the definition in the compiled runtime. The reference
machine is thus a [test oracle](https://en.wikipedia.org/wiki/Test_oracle) against which a host
runtime can be checked. Here and below, the *compiled runtime* is the PyTorch implementation of a
module (`Program`, `MonadicProgram`), used for gradients, stochastic variational inference (SVI), and Markov chain Monte Carlo (MCMC).

Two further conventions hold in the reference backend. The LKJ families, whose draw has a size
that no parameter fixes, read that size from the event extents of the constructed type. A
`Transformed` chain names its transforms as a comma-separated string over `exp`, `log`,
`sigmoid`, `logit`, `softplus`, `tanh`, and `neg`, applied from left to right.

### Running entry points

A checked module's entry points are its `define` computations and its programs.
[`entry_points`][quivers.qiec.entries.entry_points] lists them and
[`invoke_entry`][quivers.qiec.entries.invoke_entry] runs one. `qvr run FILE ENTRY [JSON ...]`,
the REPL command `:run`, and [`Program.run`][quivers.program.Program.run] all call
`invoke_entry`. An entry thus validates its arguments, selects its providers, records its trace,
and fails with the same `qiec-run-*` codes whichever of these reaches it.

A computation is run in four steps:

1. it is selected as a `NamedComputation`;
2. its static telescope is specialized, with `--static NAME=TERM` giving a closed type, index, or
   effect application for each binder;
3. its value arguments are validated;
4. its body is evaluated with a fresh attachment table under the configured providers.

`--runtime FILE.json` selects the built-in `core` runtime or the installed `quivers.qiec_runtime`
providers, without placing executable objects in the configuration file.

A program is run by [`sample_program`][quivers.qiec.program_runtime.sample_program]. It extends the
module with a wrapper computation, which is checked like any other computation before it runs.
The wrapper handles the program's `random` instance by replaying the sites given with `--site
NAME=JSON` and drawing and scoring every other site. It handles the `score` instance with an
accumulator, and it returns the program's value paired with the accumulated log weight.

The program's parameters are its data inputs (names it reads but does not bind), observations, index maps (role `index_map` in the
API), and scalars, given by position or by name (`--data NAME=JSON`). An *index map* is an integer
array that assigns each observation row to a group. The run reads each open extent, an axis length
that no declared object fixes, from the data. An extent that no data fixes, such as an object
parameter of a template, is passed as a static argument.

Each invocation owns its state:

- observations and conditioned sites are arguments of the run;
- the module's parameter store answers `Param` requests from the values the run is given;
- draws use the reference generator, which `--seed N` reseeds;
- the score accumulator and the trace belong to the wrapper's attachment table;
- no handler stack outlives the call.

[`run_program`][quivers.qiec.program_runtime.run_program] is the replaying form. It requires a
value for every site and scores the log joint density.

`--fuel STEPS` bounds the number of evaluation steps, since a recursive computation may never
return. Running out of this fuel raises the `qiec-run-fuel` diagnostic. `--trace` writes stable
events in plain text. `--json` writes one document holding the result, the specialized result
type, the runtime label, the trace, the kind of the entry, and, for a program, its log joint.

The REPL exposes the same boundary through `:runtime`, `:run`, and `:detach`, and the status bar of
the Textual interface shows the active runtime and the latest result. Failures of arguments,
specialization, providers, validators, evaluation, and results keep their `qiec-run-*` codes
across the CLI, the REPL, and the TUI.

### Programs on the reference machine

A compiled `MonadicProgram` runs on the same machine.
[`rsample`][quivers.continuous.programs.MonadicProgram.rsample] and
[`log_joint`][quivers.continuous.programs.MonadicProgram.log_joint] run the program's encoding as a
QIEC computation under the active effect handlers, through the installed
[`ProgramEvaluator`][quivers.continuous.programs.ProgramEvaluator]. A `clamp`, `do`, `trace`, or
reweighting handler in scope thus applies to every draw and every score, and the joint that
`log_joint` reports is the one that `trace` accumulates.

Each Python step is encoded over the values it reads. These are a draw's arguments, the names a
closure declares through [`reading`][quivers.continuous.program_steps.reading], or, when a closure
declares none, every value bound so far. Every handler the run installs resumes in tail position,
so the encoding of a program and its evaluation grow linearly with its length.

A run owns its handlers. A program that a Python step runs in turn, such as a score that closes
over another program's joint or a subprogram drawn as a step, is a separate invocation. The
handlers of the enclosing run neither observe nor condition its sites.

#### Prepared joints

Some consumers score one program at many points: the MCMC
[`PotentialFn`][quivers.inference.mcmc.PotentialFn], the tempered potential that sequential Monte Carlo (SMC) moves under
[@delmoral2006sequential], and [`AutoDAIS`][quivers.inference.dais.AutoDAIS]
[@zhang2021dais; @geffner2021uha]. These score through a prepared joint
(`quivers.effects.joint_plan`) rather than one run of the machine per point.

The work of a run depends on the program, on the names and layouts of the values given, and on
the handlers in scope, but never on the values themselves. That work is building the run's
handlers, validating their module, and driving the machine through each request. A prepared joint
does it once per layout. It compiles the entry computation, a chain of binds, to a straight-line
list of instructions. These instructions call the same Python step functions that the run's
`Compute` handlers call, score each site's given value as the run's observation handler does, and
sum the densities in the order and with the operations that `TraceHandler` uses. The joint and
its gradient are thus equal to the machine's, bit for bit.

The following leave a program to the machine:

- an entry computation that is not a straight chain, such as the recursive helper of a scan;
- a site given no value;
- an input with more than one row (one run per row);
- a data input that is not a tensor;
- checked at every call: a handler in scope, an overridden `log_joint`, or another installed
  evaluator.

A prepared joint refuses values of any layout but its own, so a stale preparation raises rather
than scores.

#### Calls between programs

A draw from another program, `sample theta <- school_effects(0.6, School)` or
`sample (u, v) <- sub`, runs that program's steps in place under names of the caller. The objects
and scalars the draw names replace the callee's template parameters. The callee's local `z`
becomes the site `theta$z`, and the name it returns becomes `theta` itself. This holds on the
reference machine and on the compiled runtime alike, and a transpile target spells such a site
`theta__z`. A program over object parameters is a computation whose telescope binds one index
variable per object, so `qvr run` invokes it with its static arguments.

A program's calls of computations are lowered to compiled steps before compilation
(`quivers.dsl.compiler._call_lowering`):

- A pure callee becomes `let` steps. The pure computations it calls are inlined in turn, and each
  `if` between pure branches is read as `select`. The analysis is the one the transpile planner
  uses to inline such calls (`quivers.transpile._pure_calls`, `inline_call`).
- A callee whose only requests are the program's `Random` and `Score` runs in place. Its site
  `label` is named `y$label` under the binder `y`, and a log density it scores is evaluated by the
  internal let call `$log_prob`.
- `traverse` maps a pure callee as a comprehension and a drawing callee as one
  [plated](https://en.wikipedia.org/wiki/Plate_notation) draw per site.

A program that keeps any other call (recursion, an `if` whose branches draw, a handled effect,
`case`, static arguments) has no step table the compiler can build. It compiles to a
[`CheckedProgram`][quivers.effects.checked_program.CheckedProgram], a morphism whose `rsample`
and `log_joint` are `sample_program` and `run_program` on the module's own entry point. Its
`steps`, which every fitting entry point reads, raise the located refusal that the lowering
recorded. The machine scores Python numbers, so such a program keeps no gradient path to the
tensors it is given.

## How source declarations elaborate {#qvr-surface}

Each source form below elaborates to QIEC terms: helper computations, generated instances,
handler kinds, input names, and site labels. The syntax, typing, and runtime behavior of each
form are specified in the [QVR language reference](../reference/qvr/index.md).

### Indexed families and computations

[Indexed case analysis](../reference/qvr/types-and-indexed-families.md#indexed-case-analysis)
refines the scrutinee's indices in each branch, and the rigid index a branch introduces cannot
escape into the type of the whole case.

[Computations](../reference/qvr/computations.md) call one another by ordinary application,
recursion included, and branch on Booleans with `if`. Pure expressions share the `let` grammar of
`program`. An operator resolves by operand type to a primitive of a closed registry. An integer
operand beside a real one is converted first, and the core term carries the `int_to_real`
conversion. Recursion runs on the reference machine's explicit stack and on a
[trampoline](https://en.wikipedia.org/wiki/Trampoline_(computing)) in every generated host
runtime, so its depth is bounded by memory rather than by the host call stack.

### Effects, instances, and handlers

Two [instances](../reference/qvr/effects-and-handlers.md#scoped-instances) of one applied
interface, such as `left` and `right` of `State[Int]`, have distinct identities, and a total
handler of one removes only that instance from the row. A handler declaration records a stable
signature and resumption grades. Its executable clauses are attachments keyed by the derived
handler identifier.

Probabilistic operations are instances of the same effect calculus. A total `replay` handler of
the `random` instance leaves the `score` instance in the residual row, and replay tables,
samplers, and trace recorders never enter the stable module. Distributions are values. Applying a
family of the registry builds a `Sampleable[A]` whose element type the registry fixes. A list
literal is a `Tensor` whose leading dimension is its number of entries, so a nested literal is a
matrix. `site("x")` names a sample site at the `Site[A]` type its position expects, and
`log_prob(d, x)` evaluates a log density as a `LogWeight`.

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

The `draw` and `score` handler kinds of the core runtime provider implement `draw` and
`accumulate` above. Each dynamic target spells a distribution construction in its own library
through [`spell_distribution`][quivers.transpile.family_spelling.spell_distribution]. It applies
the target's parameterization conventions (a rate in place of a scale, a complemented probability,
a shifted support, a folded family on the half line) so that `log_prob` agrees with the registry's
density on every host.

### Programs

A `program` elaborates to a named computation. Its data inputs, observations, and index maps
become the computation's typed parameters. The elaborated module records each program as a
[`ProgramEntry`][quivers.qiec.ProgramEntry]: the roles of its parameters, its sites, and its
return names, beside the computation itself. Each step elaborates as follows.

| Step | Elaborates to |
|---|---|
| `sample` | `Random.sample` on the built-in `random` instance |
| `observe` | `log_prob` of the family at the observation, added on the built-in `score` instance |
| `score` | its value added as a weight |
| `let` | a binding of a pure expression |
| `marginalize` | a helper computation, described below |

[`run_program`][quivers.qiec.program_runtime.run_program] wraps the entry point in a scoring
replay of `random` and an accumulating `score`. Given data and a value for every site, the
reference machine thus yields the program's value and its log joint density, and the tests hold
that number to the trace of the compiled runtime on the same models.

A program body binds a call of a computation with `let x <- helper(args)`, and a computation calls
a program by its name.

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

#### Marginalization

A [`marginalize`](../reference/qvr/probabilistic-programs.md#exact-marginalization) block becomes
a helper computation. The helper allocates a `Random` instance for the latent, handles it with the
enumeration handler, collects the weights of the block's scope under a `Weight` instance, and
returns the log marginal, which the enclosing scope scores. The block's `reduction` selects the
handler:

| `reduction` | Handler | Answer |
|---|---|---|
| `logsumexp` (the default) | `enumerate_marginal` | the [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) of the weighted shots |
| `sum` | `enumerate_marginal_sum` | their sum |
| `mean` | `enumerate_marginal_mean` | their average |

Each handler has a grouped counterpart (`enumerate_grouped_marginal`, with the same suffixes)
that returns one aggregate per group position. A block nested inside a grouped block uses it to add to the enclosing group's weights.

Plates are typed. `xs : N <- Normal(loc=0.0, scale=1.0)` samples a `Tensor[Real]([|N|])` from a
construction plated over `N`. In a grouped block, the `via` index map reindexes the latent's
arguments to the observation rows and sums the weights back into the groups. A product group
`[over=[G, H]]` is one flat axis of extent `|G| * |H|`, and the product index map `[via=[g, h]]`
is its [row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order) flattening, with the
last factor varying fastest.

A nested block's group must be one of two things. It may be the enclosing group's extent,
identified with it position by position. Or it may be a product with the enclosing axis as a
factor. In that case the marginals at each position are summed by segment onto the enclosing
positions, and the arguments shaped by the enclosing group are gathered by that projection.

Three further rules apply inside a block. A `sample` that reads nothing the block binds is drawn
once, before the block, since every shot shares it, and a draw that reads the latent is refused.
A vector family takes its vector in one named parameter, as in
`Dirichlet(concentration=[1.0, 2.0, 3.0])`, and a positional argument is refused. A single literal is the symmetric concentration at the dimension that
the step's plate or the program's codomain fixes.

#### Open extents

An input whose shape nothing in the program fixes, such as the concentration vector of
`sample probs <- Dirichlet(concentration=alpha)` written without a plate, is typed over an index variable that
the program's computation binds. The computation gains one `Nat` binder per open extent, recorded
on the entry's `telescope`, and its parameter reads `Tensor[Real]([alpha_extent])`. A run reads
each extent from the data. A call step from another program takes the extent as its own, and an
authored computation names it with an index literal, `prog[3](alpha)`.

#### Network kernels

A draw through a declared kernel morphism reads the family's parameters from the morphism's
parameter map. The learned tensors of that map are typed program inputs with the roles `weight`,
`bias`, and `table`.

- A kernel with a real domain applies an affine map to its conditioning row (the step's arguments
  or the program's domain inputs), and each head of the family reads its block of rows.
- A kernel over a finite domain reads the row of a table at the element the step names.
- A morphism with `[param_source=mlp]` adds hidden layers, each an affine map followed by `tanh`,
  with the widths the option gives (`mlp(a, b)`, or `hidden_dim`, or two layers of sixty-four).
  Its inputs `<m>_param_layer<i>_weight` and `<m>_param_layer<i>_bias` match the shapes of the
  compiled runtime's `MLPSource`, layer by layer.
- An embedding morphism, `[role=embed]`, is a Normal kernel over its finite domain whose table
  concatenates the centers and log scales that the compiled runtime learns.

Under a plate, the map applies at every row. An unbound conditioning name of
`observe y : Resp <- net(x)` becomes a `Tensor[Real]([|Resp|, width])` data input, an unbound
index of an embedding becomes a `Tensor[Int]([|Resp|])`, and each head is a comprehension over the
rows.

A composite drawn in one step, `sample h <- embed >> fan(head) >> out`, expands into one site per
factor before elaboration:

- `fan` over a `[replicate=k]` morphism draws through each replica `head_0` to `head_{k-1}` and
  bundles the draws, and a kernel conditioned on a bundle reads its rows in order;
- `stack(block, n)` composes `n` copies, and each morphism after the first is declared again under
  a `_copy<i>` name with parameters of its own;
- the branches of a tensor product `encoder @ decoder` at the head of a chain each read their own
  factor of the program's domain.

The expanded module declares every replica and copy, so the entry's inputs name each one.

The heads of a kernel follow the compiled runtime's parameterization of the same kernel, family
by family:

- a location is read as it is;
- a scale is exponentiated and clamped at the scale floor;
- a rate or a concentration passes through a softplus lifted off zero or shifted by a tenth;
- a probability passes through a sigmoid;
- the logits of `Categorical` or `Bernoulli` are one row over the codomain's elements.

A scalar family applied to a vector, `Normal(mu, 0.5)` with `mu` a `Tensor[Real]([3])`, draws one
coordinate per entry, and the argument's axes act as the plate. A program that calls a program
with its domain arguments alone, `let y <- inner(x)`, passes the callee's remaining inputs
through. Each becomes an input of the caller under the callee's name and role.

#### Scans

A [`scan`](../reference/qvr/probabilistic-programs.md#scans) in a chain,
`sample h <- tok_embed >> scan(cell)`, expands to two things. The first is a step program
`<cell>__scan_step(x_t, h_prev)`, which draws the chain's elements before the scan at one
position's entry and applies the cell to the result and the previous state. The second is
`let h = scan(<step>, xs)` over the chain's input. The scanned input is a sequence with one entry
per position of an open extent the program binds, so a scanned domain factor `Token` reads as
`Tensor[Int]([token_extent])`.

The `let` elaborates to a helper computation `<program>__<h>_scan(t, xs, h, ...)`. It returns `h`
once `t` reaches the length of the sequence, and otherwise calls the step at `xs[t]` and recurs at
`t + 1`, carrying the step's inputs along. The initial state is zero, or, under `init=learned`, a
`weight` input `<cell>_scan_init`.

A step input that the step reads without binding resolves to the scanning program's binding of
that name at the scan. Otherwise it becomes an input of the scanning program under the step's name
and role. An `observation` input is the exception. The cell observes one value per position, so
the scanning program takes `Tensor[A]([<input>_extent, ...])` and the helper passes the step the
entry `obs[t]`.

The step's sites are reached once per position. `run_program` replays the `n`th occurrence of a
site from the value keyed `"<site>@<n>"`, and refuses a value keyed by the bare label of a site
reached more than once. The labels that the expansion creates derive from the scan's binding:

| Site | Label |
|---|---|
| a morphism cell | the binding itself |
| the chain's prefix | `"<binding>$x_in"` |
| the step's parameters | `"<binding>$x_t"` and `"<binding>$h_prev"` |
| a scan inside a longer chain or a parallel branch | `"<binding>$<intermediate>"` |

A program cell keeps the labels of its source. A label that two sites of one program would share
is refused, since its occurrences would not number the positions.

The compiled runtime compiles the same expansion for every scan, whatever its cell. The scanning
program's `let h = scan(<step>, xs)` becomes a [`Scan`][quivers.continuous.Scan] step over the
compiled step program, whose call of the cell becomes a [`Call`][quivers.continuous.Call] step. The
program's encoding runs a scan as a recursive helper computation and a call as the callee's steps
in a nested frame of the caller. A cell's free names thus read the caller's bindings, and its sites
are requests of the same run. A site inside a scan carries the address `"<site>@<n>"`, which a
Python step computes from the run's occurrence count. Handlers, traces, and conditioning see the
keys that the reference machine replays, and the two paths score one trajectory identically.

No transpile target has a lowering that keeps the measure of a scan, since the sequence axis is not
an object the module declares. The boundary refuses the scan with `scan:no-lowering:<cell>`.

### Search and weighted deduction

Logic programming uses the same handler and resumption rules. In the module below, the `Choose`
clause resumes without restriction because an attachment may explore several alternatives.
Handling `choice` leaves the `weight` instance in the row. A search attachment and a weight
attachment thus compose without a second source calculus.

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

A [`deduction`](../reference/qvr/generated-computations.md#deduction-declarations) declaration
elaborates to exactly this arrangement for a
[weighted deduction system](../guides/deduction.md) [@shieber1995principles; @goodman1999semiring].

**Items.** The items become a closed family `<name>__Item`, with one nullary constructor per atom
and one constructor per applied symbol that the rules and lexicon mention. A slot is typed `Int`
where the system uses it as a position and `<name>__Item` otherwise. A lexicon's logical forms
number their bound variables in a fixed order, so
[alpha-equivalent](https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B1-conversion) forms are one
item.

**Generated instances.** The module gains:

- a `Search` effect with the operation `choose[a : Type, n : Nat] : Tensor[a]([n]) -> a`, and a
  `<name>__choice` instance of it;
- a `<name>__weight` instance of `Weight[K]`, where `K` is the carrier of the declared semiring
  (`LogWeight` for `LogProb` and `Viterbi`, `Bool` for `Boolean`, `Int` for `Counting`);
- a `params` instance of `Param`, through which learned weights are read.

Learned weights carry the names under which the agenda algorithm of
[`quivers.stochastic`](../api/stochastic/agenda.md) keys its parameters: `<name>.lex.<index>` for a
lexicon entry and `<name>.rule.<rule>:<bindings>` for a rule. The bindings are those of the rule's
category variables (every variable that is not a chart position), rendered in name order by the
`<name>__show` computation and joined by `|`. A slot holds a position exactly when it is typed
`Int`, so positions never enter a weight's name and one weight serves a rule at every position.

**Derivation.** The recursive `<name>__derive` computation proceeds as follows:

1. it chooses an axiom, or a group of rules whose premises have one shape;
2. it derives each premise by recursion;
3. it matches each premise against the rule's pattern by `case` analysis, checking a shared
   pattern variable with the family's `<name>__eq`;
4. it instantiates the conclusion and adds the rule's weight.

A failed match performs an empty choice.

A system whose rules all relate `span` items derives by span: it chooses split positions for the
premises and derives each premise over its own span. A system with a lexicon whose items are not
all spans, such as a multiple context-free grammar [@seki1991mcfg],
derives an item from its head symbol and its frame. The frame is the values of the item's position
slots, read in pairs as the components of its yield. A rule binds its conclusion's positions to the
frame, chooses in order the positions inside each component where its premises' components meet,
and derives each premise from its own frame.

The lowering accepts such a system when two conditions hold, and refuses it otherwise:

- its rules are linear: each conclusion component is the concatenation of premise components, and
  every nonempty premise component is used once, so premises are derived over fewer tokens than
  their conclusion;
- its unary rules form no cycle.

Its goal spans cover the whole sentence, as the agenda algorithm's do. Any other system takes its
axioms and their weights as input.

**The run entry.** `<name>__run` handles the weight instance with a collecting handler and the
search instance with a handler that resumes once per alternative and combines the shots by the
semiring's addition. Its answer is the semiring sum, over derivations of the goal, of the product
of the weights along each derivation. This is the
[inside weight](https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm) that the agenda
algorithm tabulates [@goodman1999semiring].

**Unary chains.** In a span system, unary rules extend a derived item in chains through
`<name>__close`. When the unary rules may cycle, a chain carries the items it visited
(`<name>__Chain`, counted by `<name>__count`), and `<name>__inside` bounds how often a chain
revisits an item. `<name>__converge` raises that bound until two successive inside weights agree
within the declared `tolerance`, or exactly when no tolerance is declared. This is the search's
counterpart of the chart's fixed-point iteration, which repeats until weights stop changing.

**Depth.** The `depth` option bounds the constructor depth of each rule's conclusion category
through `<name>__depth`, the measure the agenda algorithm uses, and never the length of a
derivation. A system without a lexicon bounds the height of a derivation by the number of its
axioms and rules.

A program calls the deduction with `let chart = parse(D, sentence)`, which types `sentence` as a
tensor of tokens over one of the program's open extents, and reads `chart.goal_weight()` as a real
to score.

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

Two runtime facilities let the search compose with the weights. First, a foreign clause may answer
with `TailResume(value)`. This resumes its continuation on the machine's own stack, as an authored
`resume` in tail position does, rather than in a nested Python call that stays pending. The
collecting `Weight` handler answers this way, so a resumption captured later inside its scope may
run the continuation any number of times. Second, a collecting handler configured `forkable` gives
each shot its own copy of the total so far, which is the context cloning that an unrestricted
resumption needs.

### The fixture corpus

Six source files under `tests/fixtures/qiec/` exercise the surface end to end.
`tests/test_qiec_fixture_corpus.py` tests every statement above about calls, handlers, programs,
marginalization, deductions, and parameter maps against them, and
`tests/transpile/test_qiec_fixture_equivalence.py` does the same on the transpile targets.

| File | Contents |
|---|---|
| `recursive_indexed_traversal.qvr` | `Nat` and `Vec[A](n)`, a recursive fold whose constructor branch refines the tail's length, called from another computation and specialized at a closed length |
| `authored_state_handler.qvr` | `State[S]`, an authored handler with `get`, `put`, and an explicit return clause, nested scoped instances, and a computation called from a clause |
| `effectful_probabilistic_helper.qvr` | an index sort of observation status, a GADT of measurements, a helper that scores a `Present` value and draws an `Absent` one under a site name, and a hierarchical program that calls it with both constructors |
| `grouped_marginalization.qvr` | a class for each item summed out of a Normal mixture whose rows are grouped by item through an index map, checked against an independent `logsumexp` reference |
| `ambiguous_deduction.qvr` | a grammar with prepositional attachment ambiguity, whose two derivations are summed under an unrestricted `Choose` handler and added to a program's score |
| `network_decoder.qvr` | a morphism parameterized by a network, observed through the built-in effects, with gradients reaching the attached weights |

## Integration contract

The integration has five parts.

1. QVR has one typed surface for indexed families, effect interfaces and instances, handler
   signatures, computation terms, and programs.
2. Didactic and panproto check the first-order projection of the indexed families. The QIEC
   checker then checks branch refinement, effect rows, handler coverage, and resumption contracts,
   and lowers the surface, programs included, into a serializable `QiecModule`.
3. The compiler attaches that checked module to its environment and produces a `Program` whose
   execution runs the module's computations.
4. `Lower` derives each program's target plan from its well-typed computation and projects the whole
   module into `IRQiecModule`. Its typed nodes keep declarations, rows, provenance, values,
   evidence, computations, and resumption grades, so a renderer receives one lowered root.
5. The CLI, TUI, REPL, language server, Pygments and tree-sitter highlighters, TextMate grammar,
   panproto migration assets, and transpilers share the surface and the diagnostic vocabulary.

A common IR does not mean that every target accepts every module. A *capability* is a QIEC
feature, such as search or neural attachments, that a transpile target must support to export a
program, and a target that lacks one reports a `qiec:capability:*` diagnostic. Pyro, NumPyro,
PyMC, Edward2, Turing, Gen, WebPPL, and Church lower the whole computation graph through their own
implementations of the QIEC runtime ABI, the
ABI described below. That lowering keeps stable operation and instance identifiers, dynamic addresses,
lexical handler scope, indexed constructors, equality transport, and the four resumption grades.

### Calls on dynamic targets

A program that calls a module computation, `let c <- noisy(b)`, reaches these eight targets as a
call in the model body. The plan inlines a pure callee whose body is a chain of bindings before
any renderer runs, so it costs the target nothing. Every other call is rendered against the
callee's generated entry point. The program's built-in `Random` and `Score` instances are bound to
a *native operation table*, a table of the host's own sampling and scoring primitives that the
model body builds once. A draw inside the helper is thus a site of the model under the label the
helper gives it, and a scored weight is a term of the joint.

Each host builds the table from its own primitives:

| Target | Draw | Score |
|---|---|---|
| Pyro | `pyro.sample` | `pyro.factor` |
| NumPyro | `numpyro.sample` | `numpyro.factor` |
| PyMC | `register_rv` | `pymc.Potential` |
| Edward2 | traced random variables | a factor distribution at one point |
| Turing | `tilde_assume!!`, through closures in the model body | `@addlogprob!`, likewise |
| Gen | `@trace` | a factor distribution traced under the `:qvr_factor` namespace |
| WebPPL | `sample` | `factor` |
| Church | `sample` | `factor` |

A label that the helper draws under more than once in one run is counted, `noise`, `noise@1`, and
so on, as the reference machine replays it. The Python bridges also give the host's own spellings
of the scalar primitives, so a helper's arithmetic on a drawn value stays on the host's arrays and
keeps its gradient.

The runtime of the WebPPL target is written in WebPPL's functional subset. A name is bound once,
loops are recursion, `try` is absent, and the only mutable state is the global store. That store
holds the call and instance serial numbers, the ambient handler stack, and one cell per handler
installation. WebPPL trampolines every call, so a deep recursion uses no host stack. An attachment
written for WebPPL follows the same discipline. A mutable binding holds its value in a cell of the
global store named by its `cell` field, not in a `value` field, and every function the runtime
calls from an object field is reached through `apply`, since WebPPL compiles a member call as a
native one.

### Calls on static targets

Stan, BUGS, and JAGS have no call at run time, so the plan they render is judged by what it needs:
the computations that the program's calls reach, transitively, or every computation when the
module declares no program. Stan defines a needed pure computation as a user-defined function when
it is closed and monomorphic, returns a scalar, and is built from scalar bindings, primitives with
a Stan spelling, conditionals, calls, and recursion. The model body then calls it as a transformed
parameter. BUGS and JAGS refuse a call outright as `call:graph:<callee>`, since a graph language
has no statement that runs a computation.

Every computation that a static target can represent is defined in its output, whether it is
called or not, so a module's entry points stay where the target has a form for them. A computation
that the program never calls and the target cannot represent is left out, since it is no part of
the program the output denotes. A needed computation that the target cannot represent is reported,
feature by feature, as `qiec:capability:<feature>:<computation>` before anything is placed in the
target schema.

### Target runtime ABI

Each renderer for a host language exports a QVR computation named `f` as `qiec_f`. Its source
parameters come first. The entry point also accepts an argument for static specialization and
three mappings local to the invocation: `qiec_attachments`, `qiec_handlers`, and
`qiec_operations`. The specialization argument instantiates the checked type, index, and effect
telescope. A polymorphic entry point rejects an omitted argument, a wrong arity, or a static
argument of the wrong kind. Python and Julia expose this ABI through optional or keyword
arguments, JavaScript through trailing positional arguments, and Church through an optional rest
list. The generated `model` or `build_model` entry point is unchanged.

The renderer tests reparse every target's emitted QIEC source and run external syntax checks when
the target's toolchain is installed. Runtime tests exercise the entry points in Python, Julia,
JavaScript, and Scheme, covering dispatch by stable identifier, indexed cases, attachments,
resumption grades, and handler lifecycles. This evidence covers the tested programs and
operations. Agreement with the reference machine for all well-typed programs would need a separate
proof.

Attachment entries are keyed by their `qiec:attachment:*` identifier and must be typed binding
descriptors. The runtime checks the descriptor's structural type against the checked reference and
calls its validator before exposing its value. A raw host value is not an attachment. A mutable
duplicable binding also declares a `fork` callback, which produces a fresh value from the
untouched capture seed for each unrestricted shot.

Handler entries are keyed by `qiec:handler:*`. The generated manifest, not the host attachment,
fixes the handler's effect, coverage, forwarding policy, clause set, and resumption grades. The
attachment's `operations` table is keyed by `qiec:operation:*`, and each clause receives
`(request, resume, context)`.

- The request holds the stable instance, effect, operation, static arguments, value arguments,
  result type, source origin, and dynamic address.
- The context holds the specialized handler manifest, the static arguments, and the number of
  times the clause has resumed so far.
- A return clause receives `(value, context)`.

Result validators check the values passed to `resume`, and the handler's output validator checks
the answer of a clause or return clause.

A stateful handler attachment uses `context_factory` to build one context per installation. The
optional `on_enter`, `on_exit`, and `on_drop` callbacks each run exactly once. An unrestricted
resumption may capture a handler only when the handler declares `duplicable_context`, and a mutable
captured context must also give `fork_context`. Each shot, the first included, forks from an
untouched seed. Partial handlers forward the operations they do not cover to an outer handler
without changing the checked forwarding policy.

Unhandled operations are keyed by the pair of a `qiec:effect-instance:*` and a `qiec:operation:*`
identifier and receive the whole `request`. Neither dispatch path thus gives meaning to display
names such as `Random`, `Score`, or `Choose`. An address contains the stable identifier of the
source site, the dynamic frames, and the resumption path, and each unrestricted shot appends its
branch number before it evaluates the captured continuation. Python stores active handler state in
a context local to the invocation, Julia stores it per task, and Scheme uses a dynamically scoped
parameter. JavaScript evaluation is synchronous, so its scoped stack cannot overlap independent
invocations.

### Which modules a target accepts

This boundary represents modules written only in QIEC, and mixed modules, without a synthetic
probabilistic `program`. A mixed module keeps both its probabilistic nodes and its typed
`IRQiecModule`. Acceptance depends on the target:

- modules that only declare are valid on every target;
- the eight targets that generate code in a host language accept executable modules written only in QIEC;
- the three static targets accept them only when every computation lies in the subset above;
- mixed modules must also satisfy the target's support for ordinary probabilistic programs.

A renderer may omit declarations that have no effect at run time, but it cannot drop a computation
body or erase an unsupported effect to make the module fit.

Recursive calls, scoped instances, and authored handler clauses are represented in the checked
core and exercised by the examples above. Their presence in the core does not imply support on
every target, since the capability checks apply to the reachable computation graph. Optimized
handler implementations and larger subsets for static targets are possible extensions, and each
would have to preserve the relevant execution and scoring behavior.

Panproto validates objects persisted by earlier releases against the grammar they were written
under. The migration gate reads the original QVR history without rewriting its content-addressed
object identifiers, so those fixtures test the compatibility contract itself rather than a
regenerated copy of it.
