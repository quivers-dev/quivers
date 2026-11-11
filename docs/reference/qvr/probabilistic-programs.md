# Probabilistic programs

A `program` is domain-specific notation for a checked computation over the
module's canonical `random : Random` and `score : Score` instances. Data,
observations, fibrations, and scalar parameters become value parameters;
sample sites and score terms become effect requests. The checked representation makes these requests explicit.

## Program declaration

In the regression below, `intercept`, `slope`, and `sigma` are unknown
random quantities. Their distributions before observing data are their
**priors**. For each known predictor `x`, the Normal response distribution
supplies the **likelihood** of `y`. [Conditioning](../../guides/inference-foundations.md#conditioning-on-observations) combines prior and
likelihood into a **posterior** over the unknowns; an inference algorithm
is needed to approximate or sample that posterior. Merely executing
`observe` scores data under current latent values. [@gelman2013bayesian]

<!-- compile: qiec -->
```qvr
object Row : FinSet 100

program regression : Row -> Row [effects=[Sample, Score]]
    sample sigma <- HalfNormal(scale=1.0)
    sample intercept <- Normal(loc=0.0, scale=5.0)
    sample slope <- Normal(loc=0.0, scale=2.0)
    let mean = intercept + slope * x
    observe y : Row <- Normal(loc=mean, scale=sigma)
    return y

export regression
```

The declared domain and codomain describe the program's categorical boundary.
The elaborated computation additionally records every runtime input and its
role. `[effects=[...]]` is a source-level summary checked against the body; it
does not replace the QIEC row.

## Step reference

| Step | Checked meaning |
| --- | --- |
| `sample x <- FAMILY(...)` | `Random.sample` at site `x` |
| `sample x : Plate <- FAMILY(...)` | plated draw with one value per object |
| `observe y <- FAMILY(...)` | `Score.add(log_prob(FAMILY(...), y))` |
| `score name = expression` | add an explicit `LogWeight` |
| `let x = expression` | pure binding |
| `let x <- computation(args)` | bind a named computation or program call |
| `let xs <- traverse(values, x -> computation(x))` | call a computation once per value, in order |
| `marginalize z : K <- FAMILY(...)` | enumerate a finite latent and reduce the scope |
| `return expression` | answer the program's value |

A program body is sequential. A name becomes available to later steps after
its binding, and host data may supply free names used by expressions. The
compiler records those names as input parameters rather than treating them as
globals.

## Host data

Each free name is a program input with one checked type, which
`qvr check` and `entry_points()` list and every transpile target declares. The
type is inferred from the name's uses:

| Use of the free name `x` | Checked type of `x` |
| --- | --- |
| filling a family parameter of a plated draw directly, as in `Normal(loc=theta, scale=x)` or `OrderedLogistic(predictor=eta, cutpoints=x)` | one value per row: the plate's first batch axis, then one open extent per axis of the parameter, as `Tensor[Real]([6, x_extent])` for cutpoints |
| subscripted, as in `x[i]` or `x[i, j]`, and read no other way | a table of one open extent per index, plus one per axis of a row when the gathered rows fill a vector-valued parameter; several index vectors in one subscript select one entry per position, `x[i[n], j[n]]` |
| combined elementwise with a value of known shape, as in `x * resid` | that value's shape |
| read by a let that a plated `sample`, `observe`, or `marginalize` step consumes | a vector over the plate's first batch axis |
| used as an index, or gathered from in an index position as in `E[tokens[q], k]` | its elements are `Int` |
| none of the above | a scalar `Real`, or `Int` for an index |

The rule applies to `let` and `score` expressions alike. A free name that a
reduction or a rowwise operation consumes, but that no use fixes the shape of,
as in `score s = sum(weight)`, is reported at the step that reads it: nothing
says how many entries it has. State its extent by combining it elementwise with
a value over the intended object, or by reading it entrywise in a factor, as in
`sum(factor i : Item in weight[i] * resid[i])`. Uses that fix two different
shapes for one name are refused the same way.

A free name that fills a family parameter of a plated draw directly may also be
supplied without the plate axis: a scalar `sigma` for `scale=sigma`, or one
cutpoint vector of shape `(cuts_extent,)` for `cutpoints=cuts`, is one value
every row shares. Each path broadcasts it across the plate: the compiled runtime
reads it as it is, and the reference machine spreads it before validating
([`broadcast_shared_data`](../../api/qiec/program_runtime.md#quivers.qiec.program_runtime.broadcast_shared_data)). Transpile targets declare the per-row
shape, so data for them are spread the same way before they are handed over. A
target whose library cannot read one parameter vector per row refuses the
program under `data:per-row-vector-parameter` rather than score a different
model; NumPyro, Pyro, and Edward2 render it.

A run supplies each input at its checked type. The reference machine validates
every argument, and the compiled runtime (`log_joint`, `trace`, `rsample`, and
the inference built on them) refuses a host datum whose shape or element type
contradicts its checked type, naming the datum and every admissible shape: a
vector for a scalar input, a `(rows, units)` index for a vector over `Row`, a
cutpoint table of neither the per-row nor the shared shape, or a floating-point
tensor for an `Int` index. An open extent takes the length of the
first datum it shapes. The check is what keeps the compiled runtime scoring the
model the checked module and every transpiled program describe.

Pure collection expressions such as `map` stay on the right of `=`. An
effect-aware `traverse` is computation syntax and stays on the right of `<-`;
the called computation's effects join the program row. See
[Finite collection expressions](collection-expressions.md) for the fixed-shape
contract.

## Fixed and conditional families

A morphism with literal family arguments is input-independent:

<!-- compile: false -->
```qvr
morphism prior : A -> R ~ Normal(loc=0.0, scale=1.0)
```

It has no learned lookup table. By contrast, a bare `~ Normal` requests a
parameter source conditional on the morphism's input. Inline distribution
applications in `sample` and `observe` steps may mix literal and variable
arguments, but every argument is named. A call must match one complete schema;
for instance, `Normal(loc=mean, scale=sigma)` may reverse its source order
without changing its meaning. Computation and morphism calls remain positional.

When a family exposes alternative parameterizations, the call selects one by
name. Categorical calls require either `Categorical(probs=weights)` or
`Categorical(logits=scores)`:

<!-- compile: false -->
```qvr
observe choice : Trial <- Categorical(logits=choice_logits)
```

A call cannot supply both `probs` and `logits`. Positional family construction
is rejected uniformly, including calls whose parameterization would otherwise
appear unambiguous. Transpile targets with native logits support preserve the
named parameter; probability-only targets require an explicit `softmax`
followed by `Categorical(probs=softmax(scores))`.

Vector-family values occupy one named parameter:
`Dirichlet(concentration=[1.0, 2.0, 3.0])` supplies an explicit vector. A
single literal, `Dirichlet(concentration=1.0)`, is a symmetric concentration
under a fixed output extent.

## Open input extents

When no declared object fixes the length of a tensor input, elaboration adds a
static `Nat` binder. The invocation reads the extent from data when possible:

<!-- compile: qiec -->
```qvr
object Draw : FinSet 1

program simplex : Draw -> Draw
    sample probs <- Dirichlet(concentration=alpha)
    return probs

export simplex
```

Here `alpha` is an input tensor and its leading extent becomes a static
parameter of the entry. A program call passes an extent it can infer from the
argument. A direct computation call may state one with an integer static
argument, as in `simplex[3](alpha)` when the elaborated entry exposes that
binder.

## Calling computations and programs

A program binds a `define` computation with effectful binding:

<!-- compile: qiec -->
```qvr
define standardize(x : Real, location : Real, scale : Real) : Real !{} =
    return (x - location) / scale

object Row : FinSet 8

program standardized : Row -> Row
    sample location <- Normal(loc=0.0, scale=1.0)
    sample scale <- HalfNormal(scale=1.0)
    let z <- standardize(x, location, scale)
    observe y : Row <- Normal(loc=z, scale=1.0)
    return y

export standardized
```

The callee's residual effects join the program's. Dynamic transpile targets
emit reachable calls through the common runtime ABI. Stan accepts a closed,
monomorphic, effect-free scalar callee as a user-defined function. BUGS and
JAGS refuse program-to-computation calls. Use `qvr check --target TARGET` to
analyze the QIEC call graph, then run the transpiler to exercise the selected
renderer and toolchain.

A program may also draw from another program with `sample x <- sub(...)` or
destructure a pair as `sample (a, b) <- sub(...)`. The callee's local sites are
prefixed by the caller's binding (`x$z` on the reference machine and `x__z` in
rendered targets). Passing only the callee's declared domain arguments carries
its remaining inputs into the caller as parameters.

## Exact marginalization

For a two-component mixture, evaluate the observations under each
component and weight by its prior probability. [Adding these terms](https://pyro.ai/examples/enumeration.html#Mechanics-of-enumeration) [@pyroEnumeration]
accounts for both possible assignments without selecting one. The
`logsumexp` reduction evaluates that sum in log space, where products
of many small likelihoods are more stable. [Grouped observations](../../guides/programs-hierarchical.md#grouped-marginalization-fibred-discrete-latents) share
one assignment, so their likelihoods combine within a component before
the sum over components.

`marginalize` enumerates a finite latent and adds the reduced scope weight to
the enclosing program:

<!-- compile: qiec -->
```qvr
object Item : FinSet 3
object Row : FinSet 6
object Component : FinSet 2

program mixture : Row -> Row
    sample probs <- Dirichlet(concentration=2.0) [over=Component]
    sample mean : Component <- Normal(loc=0.0, scale=3.0)
    marginalize z : Component <- Categorical(probs=probs) [over=Item, reduction=logsumexp]
        observe y : Row <- Normal(loc=mean[z], scale=1.0) [via=item_idx]
    return mean

export mixture
```

The latent's atoms are the elements of its declared index object, here the
two elements of `Component`, numbered from zero; every target enumerates
exactly those atoms, whatever form the probability argument takes (a sampled
simplex, a literal, or a `let`). A latent declared without an index object
takes its atom count from the width of its probability argument, and a
target refuses the block with `marginalize:unknown-cardinality:<family>` when
neither fixes the count. Inside the scope the latent is a number in that
zero-based range, so `real(z + 1)` is 1 at the first atom on every target.

`over=Item` requests one latent mixture assignment per item. `via=item_idx`
maps observation rows into those groups. For product groups write
`[over=[School, Subject]]` and `[via=[school_idx, subject_idx]]`; the product is
flattened row-major, with the last factor varying fastest.

The reductions are:

| Reduction | Aggregate over enumerated atoms |
| --- | --- |
| `logsumexp` | log marginal, the default |
| `sum` | arithmetic sum of scope weights |
| `mean` | arithmetic mean of scope weights |

Nested grouped blocks return one aggregate per group position to the enclosing
block. The inner group must coincide with the enclosing group or refine it by
a product projection. A sample inside a marginal block that reads nothing
bound by the block is hoisted and drawn once; a draw that depends on the
enumerated latent is refused because it would change the stated finite sum.

## Scans

`scan(cell)` in a morphism chain expands to a one-position step program and a
recursive helper over the open sequence extent:

<!-- compile: false -->
```qvr
sample hidden <- token_embedding >> scan(cell)
```

The step program takes one position's entry and the previous state, draws the
chain's elements before the scan at the entry, and applies the cell. The
binding, here `hidden`, is the final state, a value the run computes rather
than a site. `init=learned` adds a learned initial-state input; the default
initial state is zero. The scan returns only the final state. Every scan,
whatever its cell, runs its step as steps of the scanning program, once per
position, and the compiled runtime and the reference machine apply the same
rules to it.

A cell that is a stochastic morphism, `morphism cell : A * H -> H ~ Normal`,
draws the next state at every position, at a site named by the scan's
binding: `hidden@0`, `hidden@1`, and so on. The log joint is the density of
the whole trajectory, `sum_t log p(hidden_t | x_t, hidden_{t-1})`. A cell that
is a deterministic morphism contributes no site. A cell may also be a program,
whose own steps run once per position under the three rules below.

1. **Scope.** A name a program cell reads without binding it is the scanning
   program's binding of that name *before* the scan: a sampled site, a `let`,
   or a program input, read at that binding's type: a scalar, a plated vector,
   or a matrix-valued draw. A name the scanning program does not bind before
   the scan is host data of the scanning program, supplied under that name and
   typed by the cell's uses, as for any free name (a table read as `M[i, j]`
   keeps its open extents); the scanning program cannot bind it afterwards. A
   name the cell binds itself shadows the scanning program's binding inside the
   cell and is not visible after the scan. Gradients flow through the bindings
   the cell reads.
2. **Sites.** Every site the step reaches is reached once per position. The
   `n`-th occurrence of a site labelled `s` has the address `"s@n"`, counted
   from zero over the run, so for a single scan `n` is the position. A trace
   records `s@0`, `s@1`, and so on; a replay, a clamp, or a joint density
   addresses them by these keys. A `let` or `score` in a program cell appears in
   a trace as `name@n`, and each position's `score` joins the log joint. The
   labels the expansion makes derive from the binding the source writes: a
   morphism cell's draw is the binding itself, the chain's elements before the
   scan draw at `"<binding>$x_in"` (and `"<binding>$x_in_chain_<k>"` for a
   longer prefix), and a scan inside a longer chain or a branch of `fan` or `@`
   is named `"<binding>$<intermediate>"`. A source cannot write `$`, so these
   labels never meet a source label. A program cell's own labels are the
   source's; two sites of one program that would share a label, a program
   cell's site and the scanning program's or another scan's, are refused on
   both paths with one diagnostic at the scan.
3. **Observations.** An `observe y` in the cell observes one value per
   position. The scanning program takes the whole sequence as its observation
   input `y`, with the position axis first: the checked elaboration types it
   `Tensor[A]([<input>_extent, ...])` when one position's value is `A[...]`.
   The compiled runtime takes a batch of sequences with the position axis
   after the batch axis, `(batch, seq_len, ...)`; entry `[:, t]` is the
   observation at position `t`, scored at site `y@t`. A single position may be
   given instead as `y@t`, shaped `(batch, ...)`. Giving a position twice, a
   value past the sequence's end, or a value for the scan's final state is
   refused. The scanning program cannot also bind `y`.

The same two layouts address every per-position site: a latent `s` takes
`s@t` for one position or `s` for the whole sequence, `(batch, seq_len,
...)`, which is the shape a lift's `additional_latents` entry declares. A value
under the bare label with any other shape, such as a single final state, is
refused rather than split across positions. On the reference machine, which
replays one sequence at a time, a value under the bare label of a site reached
more than once is refused rather than replayed at every position; give each
occurrence under `s@n`.

The log joint is the sum over positions of every site's density, every
observation's likelihood, and every score. For a state-space model whose cell
draws `s` and observes `o`, it is
`sum_t log p(s_t | u_t, s_{t-1}) + log p(o_t | s_t)`.

<!-- compile: qiec -->
```qvr
object Driver : Real 2
object State : Real 3
object Obs : Real 2

morphism transition : Driver * State -> State ~ Normal
morphism emission : State -> Obs ~ Normal

program step(u_t, s_prev) : Driver * State -> State
    sample s <- transition(u_t, s_prev)
    observe o <- emission(s)
    return s

define run_all = scan(step)

program ssm : Driver -> State
    sample h <- run_all
    return h

export ssm
```

Here `ssm.log_joint(u, {"s@0": ..., "s@1": ..., "o": o})` scores a driver
sequence `u` of shape `(batch, T, 2)` with `T` latent states and the
observations `o` of shape `(batch, T, 2)`.

A program composes with another through `sample y <- p(x)`, or runs at every
position as the cell of `scan(p)`. A program used in `>>`, `repeat`, `stack`,
`fan`, or `@` is refused on both paths, since the chain would treat it as a
host function of its input.

Current transpile targets refuse scans with `scan:no-lowering:<cell>` because
the open sequence extent is not a declared categorical object.

## Network kernels

A morphism with `[param_source=mlp(...)]` elaborates its affine layers and
biases as typed program inputs and applies them under `tanh` before it
constructs the distribution family. An embedding morphism with `[role=embed]`
uses a table of centers and log scales. Plates map these kernels row by row,
and composite morphisms expand to one checked site per stochastic factor.

The reference machine can execute these kernels with the compiled PyTorch
attachments. Host-language transpilers currently refuse model-internal MLP
parameter sources because those weights are neither explicit model sites nor
portable data. The measured [support matrix](../../transpile-support.md)
reports this as `param-source:mlp`.

## Running and scoring

Replay fixes named random choices so a particular execution can be
scored or reproduced. The runtime [handler](effects-and-handlers.md#handler-declarations) draws unspecified choices from
their model distributions. To estimate their distribution given observed
data, pass the model and observations to an
[inference driver](../../guides/inference-foundations.md).

An invocation installs a draw-and-score handler for `random` and an
accumulator for `score`. Given `--site name=value`, that site is replayed and
scored; every unspecified site is drawn and scored. The result includes the
program value, log joint, and [trace](../../guides/inference-foundations.md#trace-and-sample-sites). `run_program` is the stricter replay form
used when every site must be supplied.

See [Execution and tooling](execution-and-tooling.md#run-an-entry) for CLI and
Python forms, and [Transpilation support](../../transpile-support.md) for the
measured backend boundary.
