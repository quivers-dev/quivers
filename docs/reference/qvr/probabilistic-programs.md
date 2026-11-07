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

The reference machine replays the `n`-th occurrence of a step site from
`"site@n"`, which makes a trajectory's addressing stable. `init=learned`
adds a learned initial-state input; the default initial state is zero. Current
transpile targets refuse scans with `scan:no-lowering:<cell>` because the open
sequence extent is not a declared categorical object.

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
