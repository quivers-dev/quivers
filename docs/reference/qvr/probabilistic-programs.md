# Probabilistic programs

A `program` declares a probabilistic model as a sequence of steps, and elaborates to one computation
of the [Quivers Indexed Effect Core](index.md) (QIEC) that performs through the module's two built-in
[effect instances](effects-and-handlers.md#prelude-interfaces), `random : Random` and
`score : Score`. Each draw becomes a `Random.sample` request at a named *sample site*, and each
observation or score term becomes a `Score.add` request that adds a log density to the run's
weight [@wingate2011lightweight]. The program's data inputs and index maps ([Data inputs](#host-data)), observations,
and scalar parameters become value parameters of the computation.

## Program declaration

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

The declared domain and codomain (`Row -> Row`) type the program as a morphism, and the elaborated
computation also records each runtime input and its role. The option `[effects=[...]]` is a
summary checked against the body, in a vocabulary of its own: `Sample` for draws, `Score` for
observations, `Marginal` for `marginalize` blocks, and `Pure` for none of these. The body's effects
must be among those declared, and the summary does not replace the QIEC row. Running the program
draws the latent sites from their priors and scores the observations under the drawn values. A
posterior is computed by an [inference driver](../../guides/inference-foundations.md#conditioning-on-observations)
[@gelman2013bayesian].

## Step reference

| Step | Checked meaning |
| --- | --- |
| `sample x <- FAMILY(...)` | `Random.sample` at site `x` |
| `sample x : Plate <- FAMILY(...)` | one draw per element of the object `Plate`, as a [plate](https://en.wikipedia.org/wiki/Plate_notation) of conditionally independent draws [@koller2009probabilistic] |
| `observe y <- FAMILY(...)` | `Score.add(log_prob(FAMILY(...), y))` |
| `score name = expression` | add the `LogWeight` `expression` |
| `let x = expression` | pure binding |
| `let x <- computation(args)` | bind the result of a computation call or a program call |
| `let xs <- traverse(values, x -> computation(x))` | call a computation once per value, in order |
| `marginalize z : K <- FAMILY(...)` | sum the finite latent `z` out of the weight of the block it scopes |
| `return expression` | the program's value |

A program body is sequential, and a name is available to the steps after its binding. A name that
an expression reads but no step binds is a data input of the program, not a global.

## Data inputs {#host-data}

A *data input* is a name the program reads but does not bind, given at run time with the
observations, as a variable of Stan's `data` block is [@carpenter2017stan]. Each data input has one
checked type, which `qvr check` and `entry_points()` list and every transpile target declares. The
type is inferred from the name's uses:

| Use of the data input `x` | Checked type of `x` |
| --- | --- |
| fills a family parameter of a draw over a plate directly, as in `Normal(loc=theta, scale=x)` or `OrderedLogistic(predictor=eta, cutpoints=x)` | one value per row: the plate's first batch axis, then one open extent per axis of the parameter, as `Tensor[Real]([6, x_extent])` for cutpoints |
| subscripted, as in `x[i]` or `x[i, j]`, and read no other way | a table with one open extent per index, plus one per axis of a row when the gathered rows fill a vector parameter (several index vectors in one subscript select one entry per position, `x[i[n], j[n]]`) |
| combined entry by entry with a value of known shape, as in `x * resid` | that value's shape |
| read by a `let` that a plated `sample`, `observe`, or `marginalize` step consumes | a vector over the plate's first batch axis |
| used as an index, or gathered from in an index position as in `E[tokens[q], k]` | a tensor of `Int` |
| none of the above | a scalar `Real`, or `Int` for an index |

An *open extent* is an axis length that no declared object fixes and that is read from the data at
run time ([open input extents](#open-input-extents)). The *batch axes* of a draw index its
independent copies, and its *event axes* belong to one joint value [@pyroTensorShapes].

The rule applies to `let` and `score` expressions alike. A data input that a reduction or a rowwise
operation consumes, but whose shape no use fixes, as in `score s = sum(weight)`, is refused at the
step that reads it, since nothing gives its number of entries. Its extent is fixed by combining it
entry by entry with a value over the intended object, or by reading it entrywise in a factor, as in
`sum(factor i : Item in weight[i] * resid[i])`. A name whose uses fix two different shapes is
refused the same way.

A data input that fills a family parameter of a draw over a plate directly may also be given without the
plate axis: a scalar `sigma` for `scale=sigma`, or one cutpoint vector of shape `(cuts_extent,)` for
`cutpoints=cuts`, is one value that every row shares. The [compiled runtime](../../guides/dsl-overview.md#compilation-pipeline), the PyTorch implementation of a module, reads such a value as it
is, and the [reference machine](../../developer/qiec.md#reference-evaluator), the library's interpreter for QIEC, broadcasts it across the plate before validating
([`broadcast_shared_data`](../../api/qiec/program_runtime.md#quivers.qiec.program_runtime.broadcast_shared_data)).
Transpile targets declare the shape with one value per row, so the data given to them are broadcast
the same way. A target whose library cannot read one parameter vector per row refuses the program
with `data:per-row-vector-parameter`, and NumPyro, Pyro, and Edward2 render it.

Each data input is given at its checked type. The reference machine validates every argument. The
compiled runtime, which computes gradients and runs stochastic variational inference (SVI) and MCMC
(`log_joint`, `trace`, `rsample`, and the inference built on them), refuses a datum whose shape or
element type contradicts its checked type, and names the datum and every admissible shape. Examples
are a vector for a scalar input, an index of shape `(rows, units)` for a vector over `Row`, a
cutpoint table of neither the shape with one row per draw nor the shared shape, and a floating-point tensor for an `Int`
index. An open extent takes the length of the first datum it shapes. Both runtimes thus score the
model that the checked module and every transpiled program describe.

An *index map* is a function from observation rows to groups, given as an integer array, whose
fiber over a group is the set of rows in that group. The `via=` option of a grouped step names one
([exact marginalization](#exact-marginalization)), and the API gives it the parameter role
`index_map`.

Pure collection expressions such as `map` stay on the right of `=`. An effectful `traverse` is
computation syntax, stays on the right of `<-`, and joins the called computation's effects to the
program's row ([finite collection expressions](collection-expressions.md)).

## Fixed and conditional families

A morphism whose family arguments are all literals does not depend on its input:

<!-- compile: false -->
```qvr
morphism prior : A -> R ~ Normal(loc=0.0, scale=1.0)
```

It has no learned lookup table. A bare `~ Normal`, in contrast, requests a
[parameter source](../../api/continuous/param_source.md), the rule that turns the morphism's input
into the family's parameters. A declaration's family arguments are literals (numbers, vectors, or
matrices) that fill one complete parameterization, or they are absent. An argument that reads a
name, as in `~ Normal(scale=q)`, is refused at that argument on every path, since a declaration
binds no value for it to read.

A distribution application in a `sample` or `observe` step may mix literal and variable arguments,
and every argument is named. A call matches one complete parameterization, so
`Normal(loc=mean, scale=sigma)` and `Normal(scale=sigma, loc=mean)` mean the same. Calls of
computations and morphisms stay positional.

A family with several parameterizations is called with the names of one. `Categorical` takes
either `Categorical(probs=weights)` or `Categorical(logits=scores)`, never both:

<!-- compile: false -->
```qvr
observe choice : Trial <- Categorical(logits=choice_logits)
```

A positional family argument in a program step is refused, even where only one parameterization
could match. A transpile target with native support for logits keeps the named parameter, and a
target that takes only probabilities requires `Categorical(probs=softmax(scores))`.

A family whose value is a vector takes it in one named parameter.
`Dirichlet(concentration=[1.0, 2.0, 3.0])` gives an explicit vector, and a single literal,
`Dirichlet(concentration=1.0)`, is a symmetric concentration over a fixed output extent.

## Open input extents

When no declared object fixes the length of a tensor input, elaboration adds a static `Nat` binder,
and an invocation reads the extent from the data:

<!-- compile: qiec -->
```qvr
object Draw : FinSet 1

program simplex : Draw -> Draw
    sample probs <- Dirichlet(concentration=alpha)
    return probs

export simplex
```

Here `alpha` is a tensor input, and its leading extent becomes a static parameter of the entry. A
program call passes an extent it can infer from the argument. A direct computation call may state
one with an integer static argument, as in `simplex[3](alpha)`, when the elaborated entry exposes
that binder.

## Calling computations and programs

A program binds the result of a `define` computation with `<-`:

<!-- compile: qiec -->
```qvr
define standardize(x : Tensor[Real]([8]), location : Real, scale : Real) : Tensor[Real]([8]) !{} =
    let z_score = v -> (v - location) / scale
    return map(x, z_score)

object Row : FinSet 8

program standardized : Row -> Row
    sample location <- Normal(loc=0.0, scale=1.0)
    sample scale <- HalfNormal(scale=1.0)
    let z <- standardize(x, location, scale)
    observe y : Row <- Normal(loc=z, scale=1.0)
    return y

export standardized
```

The data input `x` takes the type of the parameter it is passed to, a column of eight covariates,
one per `Row`, and `map` standardizes each. The callee's residual effects join the program's row.

A computation keeps the parameter types it declares, and a call does not map a computation with a
scalar parameter over the entries of a tensor. A plated value passed to a scalar parameter is a type
error located at the call. A datum with more axes than the parameter it fills is refused on both
runtimes, located at the call that fixes its type. Declare the parameter as a tensor and `map` over
it inside the computation, as `standardize` does.

A program may also draw from another program with `sample x <- sub(...)`, or destructure a pair
with `sample (a, b) <- sub(...)`. The callee's sites are prefixed by the caller's binding: `x$z` on
the reference machine and `x__z` in rendered targets. A call that passes only the callee's declared
domain arguments carries the callee's remaining inputs into the caller as parameters.

### Which calls are fit

The compiled runtime fits a program (`trace`, `log_joint` and its gradients, SVI, MCMC) by running
a fixed chain of steps, so it fits a call exactly when the callee's body can be stated as such a
chain at the call. These calls are lowered, and the program then compiles to a
[`MonadicProgram`][quivers.continuous.MonadicProgram] whose values agree with the reference
machine's:

| Call | Lowered to |
| --- | --- |
| a pure callee: no effect, no static arguments, pure bindings ending in a return | `let` steps, the callee's bindings under fresh names |
| a pure callee that calls other pure computations | the same, with each nested call inlined in turn |
| a pure callee whose body branches with `if c then e1 else e2` between pure branches | `select(c, e1, e2)`, which evaluates both branches, so a branch undefined where it is not selected (a `log` of a negative number) gives a NaN gradient |
| a callee whose only requests are the program's `Random.sample` and `Score.add` | the callee's steps in place, as a draw from a program runs: the site `label` it draws becomes `y$label` under the binder `y`, and each score becomes a score step |
| `let ys <- traverse(xs, x -> f(x, ...))` with a pure `f` | a comprehension over the items of `xs` |
| the same with an `f` that draws | one draw over a plate per site of `f`, named `ys$label`, over the object that `xs` runs along |

Each run of a traversed callee reads only its own item and names bound before the `traverse`, so
the runs are independent and the draw over the plate is exact. A recurrence, in which a step reads the
previous one, is written with [`scan`](#scans).

The transpile planner shares this analysis of pure calls, so every target inlines nested pure calls
and states a conditional between pure branches in its own form of `select`. A callee that draws is
still called through each dynamic target's runtime, where its site keeps its label as on the
reference machine: a site `noise` reached once is `noise`, and reached repeatedly, by a `traverse`
or by two calls, is `noise@0`, `noise@1`, and so on in run order.

Any other call leaves the program a
[`CheckedProgram`][quivers.effects.checked_program.CheckedProgram], which runs on the reference
machine but cannot be fit. `trace`, MCMC, SVI, and every other fitting entry point raise a
[`CompileError`][quivers.dsl.CompileError] located at the call, and a parametric template that makes
such a call is refused with the same error when it is instantiated:

| Call | Why it cannot be fit | Instead |
| --- | --- | --- |
| a call that reaches a recursive call | its number of steps depends on its arguments | a reduction (`sum`, `fold`) or `map` over a tensor of fixed length, or `scan` over a sequence |
| an `if` whose branches draw or score | its set of sites depends on the branch taken | draw outside the conditional and choose between values with `select` |
| a callee that handles an effect in its body (`with instance`, `handle`) | the compiled runtime does not run handlers or resumptions | a callee with no handler |
| a `case` on a family value | the compiled runtime does not run case analysis | a callee with no `case` |
| a call with static arguments, as `identity[Int](4)` | the compiled runtime does not specialize | a callee with no static binders |
| a program called as a computation, `let y <- sub(...)` | a program is drawn from, not called | `sample y <- sub(...)` |

Dynamic transpile targets emit every reachable call, Stan accepts a callee that is a closed,
monomorphic scalar computation with no effects as a user-defined function, and BUGS and JAGS refuse
calls from a program to a computation ([transpilation](execution-and-tooling.md#transpilation)).

## Exact marginalization

`marginalize` sums a finite latent variable out of the weight of the steps it scopes, by evaluating
each of its values [@pyroEnumeration; @stanFiniteMixtures]:

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

`over=Item` requests one latent assignment per item, and `via=item_idx` is the index map from the
observation rows to items. For an item $g$ with rows $\iota^{-1}(g)$, the block adds

$$
\log \sum_{k} \pi_k \prod_{n \in \iota^{-1}(g)} \mathcal{N}(y_n; \mu_k, 1)
$$

to the log joint, where $\iota$ is the index map, $\pi$ is `probs`, $\mu$ is `mean`, and
$\mathcal{N}(y; m, s)$ is the Normal density at $y$ with mean $m$ and standard deviation $s$. The
rows of one item thus share one assignment, and their likelihoods are multiplied before the sum over
components ([grouped marginalization](../../guides/programs-hierarchical.md)). For product groups,
write `[over=[School, Subject]]` and `[via=[school_idx, subject_idx]]`. The product is flattened in
row-major order, the last factor varying fastest.

The values of the latent are the elements of its declared index object, here the two elements of
`Component`, numbered from zero. Every target enumerates exactly those values, whatever form the
probability argument takes (a sampled simplex, a literal, or a `let`). A latent declared without an
index object takes its number of values from the width of its probability argument, and a target
refuses the block with `marginalize:unknown-cardinality:<family>` when neither fixes the number.
Inside the block the latent is an integer in that range, so `real(z + 1)` is 1 at the first value on
every target.

The reductions are:

| Reduction | Aggregate over the latent's values |
| --- | --- |
| `logsumexp` | the log marginal likelihood, the default |
| `sum` | the arithmetic sum of the block's weights |
| `mean` | the arithmetic mean of the block's weights |

A nested grouped block returns one aggregate per group to the enclosing block. The inner grouping
must coincide with the enclosing one or refine it by a projection of a product. A `sample` inside a
block that reads nothing the block binds is hoisted and drawn once. A draw that depends on the
summed latent is refused, because it would change the finite sum the block states.

## Scans

`scan(cell)` in a morphism chain threads a state along a sequence and records one state per
position. It expands to a step program for one position and a recursive helper over the sequence's
open extent:

<!-- compile: false -->
```qvr
sample hidden <- token_embedding >> scan(cell)
```

The step program takes one position's entry and the previous state, draws the elements of the chain
before the scan at that entry, and applies the cell. The binding, here `hidden`, is the final
state, a value the run computes, not a site. Every scan runs its step as steps of the scanning
program, once per position, and the compiled runtime and the reference machine apply the same
rules to it.

A scan takes its options in any order, each at most once, as in
`scan(cell, init=h0, outputs=all, reverse=true)`:

| Option | Meaning |
| --- | --- |
| `init` | the initial state: zero by default, a learned input with `init=learned`, or with `init=h0` the value of `h0`, a binding the scanning program makes before the scan |
| `outputs` | `last`, the default, binds the final state, and `all` binds the sequence of every state, one per position |
| `reverse` | `true` visits the positions from the last to the first |

With `init=h0`, a binding of the state's type is read as it is, and a scalar binding is the one
entry of a state of width one. A name not bound before the scan, or a binding of another type, is
refused where the scan is written. Gradients reach the binding through the first position.

With `outputs=all`, the checked elaboration types the binding `Tensor[H]([<input>_extent])` for a
state of type `H`, so a state `Tensor[Real]([d])` gives `Tensor[Real]([<input>_extent, d])`. Like the
final state, the sequence is a value, not a site.

With `reverse=true`, the state after position $t$ is computed from the entry at $t$ and the state
after position $t + 1$, the initial state stands before the last position, and the final state is
the one after position 0. The sequence of states is still laid out in input order, entry $t$ being
the state after reading position $t$, so a forward and a reversed sequence over one input pair up
position by position.

A sequence of states is what a later scan in the same chain runs over, so
`scan(lower, outputs=all) >> scan(upper)` stacks two recurrences. An element between them, as in
`scan(lower, outputs=all) >> proj >> scan(upper)`, runs at every position inside the second scan's
step. `fan` and `@` of branches that each end in a scan with `outputs=all` pair the sequences
position by position: the entry at a position joins the branches' entries end to end, as `fan` joins
two vectors, and both sequences must run over the same positions. Three chains are refused where the
draw is written: a scan over another scan's final state (`scan(a) >> scan(b)`), a morphism applied to
a whole sequence of states outside a scan (`scan(a, outputs=all) >> head`), and a `fan` that pairs a
sequence with one value.

### Cells

A cell that is a stochastic morphism, `morphism cell : A * H -> H ~ Normal`, draws the next state at
every position, at a site named by the scan's binding: `hidden@0`, `hidden@1`, and so on. The log
joint is the log density of the whole trajectory,
$\sum_t \log p(h_t \mid x_t, h_{t-1})$. A cell that is a deterministic morphism contributes no site.
A cell may also be a program, whose steps run once per position under three rules.

1. **Scope.** A name that a program cell reads without binding it is the scanning program's binding
   of that name before the scan: a sampled site, a `let`, or a program input, read at that
   binding's type (a scalar, a plated vector, or a draw with a matrix value). A name the scanning
   program does not bind before the scan is a data input of the scanning program, given under that
   name and typed by the cell's uses (a table read as `M[i, j]` keeps its open extents), and the
   scanning program cannot bind it afterwards. A name the cell binds itself shadows the scanning
   program's binding inside the cell and is not visible after the scan. Gradients flow through the
   bindings the cell reads.
2. **Sites.** Every site the step reaches is reached once per position. The $n$-th occurrence of a
   site labeled `s` has the address `"s@n"`, counted from zero over the run, so for a single scan
   $n$ is the $n$-th position the scan visits: position $n$ from the left, and position $T - 1 - n$
   for a reversed scan over $T$ positions. A trace records `s@0`, `s@1`, and so on, and a replay, a
   clamp, or a joint density addresses the sites by these keys. A `let` or `score` in a program cell
   appears in a trace as `name@n`, and each position's `score` joins the log joint. The labels the
   expansion makes derive from the binding the source writes. A morphism cell's draw is the binding
   itself, the chain's elements before the scan draw at `"<binding>$x_in"` (and
   `"<binding>$x_in_chain_<k>"` for a longer prefix), and a scan inside a longer chain or a branch
   of `fan` or `@` is named `"<binding>$<intermediate>"`. These labels can be read in a trace and
   given as keys, but no source binding can be spelled with `$`, so they never collide with a source
   label. A program cell's own labels are the source's. Two sites of one program that would share a
   label, such as a program cell's site and the scanning program's or another scan's, are refused on
   both runtimes with one diagnostic at the scan.
3. **Observations.** An `observe y` in the cell observes one value per position. The scanning
   program takes the whole sequence as its observation input `y`, with the position axis first: the
   checked elaboration types it `Tensor[A]([<input>_extent, ...])` when one position's value is
   `A[...]`. The compiled runtime takes a batch of sequences with the position axis after the batch
   axis, `(batch, seq_len, ...)`. Entry `[:, t]` is the observation at position $t$, scored at site
   `y@t`, or at `y@(T - 1 - t)` in a reversed scan, which reaches position $t$ that many visits
   after its first. A single occurrence may instead be given as `y@n`, of shape `(batch, ...)`.
   Giving a position twice, a value past the end of the sequence, or a value for the scan's final
   state is refused, and the scanning program cannot also bind `y`.

The same two layouts address every site reached once per position: a latent `s` takes `s@t` for one
position or `s` for the whole sequence, of shape `(batch, seq_len, ...)`, which is the shape an
`additional_latents` entry of
[`bayesian_lift_parameters`][quivers.inference.lifts.bayesian_lift_parameters] declares. A value
under the bare label with any other shape, such as a single final state, is refused rather than
split across positions. The reference machine replays one sequence at a time, and on it a value
under the bare label of a site reached more than once is refused rather than replayed at every
position, so each occurrence is given as `s@n`.

The log joint is the sum over positions of every site's log density, every observation's log
likelihood, and every score. For a state space model whose cell draws $s$ and observes $o$, it is
$\sum_t \bigl[\log p(s_t \mid u_t, s_{t-1}) + \log p(o_t \mid s_t)\bigr]$:

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

`ssm.log_joint(u, {"s@0": ..., "s@1": ..., "o": o})` scores a driver sequence `u` of shape
`(batch, T, 2)` with $T$ latent states and the observations `o` of shape `(batch, T, 2)`.

A program composes with another through `sample y <- p(x)`, or runs at every position as the cell
of `scan(p)`. A program used in `>>`, `repeat`, `stack`, `fan`, or `@` is refused on both runtimes,
since the chain would treat it as a deterministic function of its input.

A stationary [autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model) of order
one, AR(1), for log volatility starts its trajectory from the chain's stationary distribution,
$\mathcal{N}\bigl(\mu, \sigma_h / \sqrt{1 - \phi^2}\bigr)$ for $|\phi| < 1$, rather than from zero:

<!-- compile: qiec -->
```qvr
object Val : Real 1

program sv_step(u_t, h_prev) : Val * Val -> Val
    let m = mu + phi * (h_prev - mu)
    sample h <- Normal(loc=m, scale=sigma_h)
    let vol = exp(0.5 * h)
    observe r <- Normal(loc=0.0, scale=vol)
    return h

define run_all = scan(sv_step, init=h0)

program volatility : Val -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample phi <- Uniform(low=-1.0, high=1.0)
    sample sigma_h <- HalfNormal(scale=0.5)
    let s0 = sigma_h / sqrt(1.0 - phi * phi)
    sample h0 <- Normal(loc=mu, scale=s0)
    sample last <- run_all
    return last

export volatility
```

A bidirectional tagger reads, at every position, the state of a forward scan and the state of a
reversed scan over the same input, then scores one tag per position with a third scan:

<!-- compile: qiec -->
```qvr
object In : Real 1
object State : Real 1
object Pair : Real 2

program forward_cell(x_t, h_prev) : In * State -> State
    let m_f = 0.8 * h_prev + x_t
    sample f <- Normal(loc=m_f, scale=0.5)
    return f

program backward_cell(x_t, h_prev) : In * State -> State
    let m_g = 0.6 * h_prev - x_t
    sample g <- Normal(loc=m_g, scale=0.7)
    return g

program tag_cell(p_t, s_prev) : Pair * State -> State
    let m_s = p_t[0] - p_t[1] + 0.3 * s_prev
    sample s <- Normal(loc=m_s, scale=0.6)
    observe y <- Normal(loc=s, scale=0.3)
    return s

define both = fan(scan(forward_cell, outputs=all), scan(backward_cell, outputs=all, reverse=true))
define tagger = both >> scan(tag_cell, outputs=all)

program bidirectional : In -> State
    sample tags <- tagger
    return tags

export bidirectional
```

With $\mathcal{N}(y; m, s)$ the Normal density at $y$ with mean $m$ and standard deviation $s$, and
with the initial states $f_{-1} = g_T = s_{-1} = 0$, its log joint over $T$ positions is

$$
\sum_{t=0}^{T-1} \Bigl[
\log \mathcal{N}(f_t; 0.8 f_{t-1} + x_t, 0.5)
+ \log \mathcal{N}(g_t; 0.6 g_{t+1} - x_t, 0.7)
+ \log \mathcal{N}(s_t; f_t - g_t + 0.3 s_{t-1}, 0.6)
+ \log \mathcal{N}(y_t; s_t, 0.3)
\Bigr],
$$

where $f_t$ is the forward state at site `f@t`, $g_t$ the backward state at site `g@(T-1-t)`, and
$s_t$ the tag state at site `s@t`. The program returns the tag states in input order.

Every transpile target refuses a scan with `scan:no-lowering:<cell>`, because the open extent of the
sequence is not a declared object. A scan with `outputs=all` or `reverse=true` is refused under the
same code.

### Marginalized chains

A scan's state may be a finite latent that the cell marginalizes and returns. Returning the latent of
a `marginalize` block carries it to the next position, so its sum extends over the rest of the chain:
the scan sums out every path of states, which is the likelihood of a
[hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) [@rabiner1989hmm].

<!-- compile: qiec -->
```qvr
object State : FinSet 2
object Symbol : FinSet 4
object StartProbs : Real 2

program hmm_step(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition[z_prev])
        observe symbol <- Categorical(probs=emission[z])
    return z

define chain = scan(hmm_step, init=start)

program hmm : Symbol -> StartProbs
    sample start <- Dirichlet(concentration=1.0) [over=State]
    sample transition : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return start

export hmm
```

Write $z_0$ for the state before the first position, $z_t$ for the state at position $t$ of
$t = 1, \dots, T$, and $o_t$ for the observations of the block at position $t$. The chain is
$z_0 \sim \mathrm{Categorical}(\pi)$ with $\pi$ the `init` vector (`start`), and
$z_t \mid z_{t-1} \sim \mathrm{Categorical}(A_{z_{t-1}})$ with $A$ the `transition` table. Its log
marginal likelihood is computed by the
[forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm) in log space, in
$O(T K^2)$ time for $K$ states:

$$
\begin{aligned}
\tilde\alpha_0(i) &= \log \pi_i, \\
\ell_t(j) &= \log \sum_i \exp\bigl(\tilde\alpha_{t-1}(i) + \log A_{ij}\bigr) + \log p(o_t \mid z_t = j), \\
e_t &= \log \sum_j \exp \ell_t(j), \\
\tilde\alpha_t(j) &= \ell_t(j) - e_t,
\end{aligned}
$$

and the chain adds $\sum_{t=1}^{T} e_t = \log p(o_1, \dots, o_T)$ to the joint, under the name its
`sample` step binds (`path` above). Both runtimes compute it: the compiled runtime as one score per
run, and the reference machine by running the cell's elaborated program with its block enumerated
under a handler that keeps one weight per state. Gradients reach every parameter the cell reads.

1. **The cell.** `program cell(x_t, z_prev) : X * S -> S`, with `S` a declared `FinSet`, consists of
   `let` steps (which may read the entry, the previous state, and the scanning program's bindings),
   then one `marginalize z : S <- Categorical(...)` with `probs=` or `logits=` and the default
   `reduction=logsumexp`, then `return z`. The block holds `let` steps and observations of one value
   per position. An observation named after the entry, as `observe symbol` above, scores the entry
   itself. Any other name `y` is one observation per position, which the scanning program takes as
   the sequence `y`, laid out as for any scan.
2. **The first state.** `init=p` names a probability vector over `S` bound before the scan. It is
   the distribution of $z_0$, which emits nothing, and a vector with zeros starts the chain among
   the other states. A chain without `init=` is refused with `scan:marginal:init:<cell>`.
3. **Order.** `reverse=true` visits the positions from the last to the first, so the chain runs
   right to left, and the observations are still read in input order.
4. **The binding.** The states are summed out, so the name the `sample` step binds holds no state,
   and reading it later or returning it is refused with `scan:marginal:read:<name>`. A trace records
   it as the chain's log marginal likelihood, one value per run, and records no site per position.

Forms outside the forward algorithm are refused where they are written, on both runtimes, with
`scan:marginal:<reason>:<cell>`:

| `<reason>` | Refused form |
| --- | --- |
| `state` | a cell of another type |
| `step` | a step other than a `let` before the block, other than a `let` or an observation inside it, or any step after it |
| `family` | a transition family other than `Categorical` |
| `grouped` | a grouped block |
| `reduction` | a reduction other than `logsumexp` |
| `observation` | an observation in a plate, grouped by an index map, or destructured |
| `chain` | a chain composed with other elements |
| `observe` | a chain observed rather than sampled |
| `scope` | a chain inside a `marginalize` block |
| `outputs` | a chain asked for every state with `outputs=all` |
| `entry` | a draw or call of the cell other than as a scan's cell |

Every transpile target refuses a marginalized chain with `scan:marginal:no-lowering:<cell>`.

## Network kernels

A morphism with `[param_source=mlp(...)]` elaborates the weights and biases of its affine layers as
typed program inputs, applies them under `tanh`, and then builds the distribution family. An
embedding morphism with `[role=embed]` uses a table of centers and log scales. Plates apply these
kernels row by row, and a composite morphism expands to one checked site per stochastic factor.

The reference machine runs these kernels with the compiled PyTorch attachments. Every transpile
target refuses an MLP parameter source inside a model, because its weights are neither model sites
nor data, and the [support matrix](../../transpile-support.md) reports it as `param-source:mlp`.

## Learned functions

A morphism declared `[role=function]` is a learned deterministic map between continuous spaces. A
program calls it in a `let` or `score` expression with one argument per factor of its domain, each
a `Real` for a factor of width one or a `Tensor[Real]([width])`. A data input passed straight to it
has the factor's type. The call elaborates to the function's network: for each hidden layer $k$, an
affine map with weight `f_param_layer<k>_weight` and bias `f_param_layer<k>_bias` under the declared
activation, then the output layer `f_param_weight` and `f_param_bias` under the declared output
nonlinearity. The weights and biases are typed program inputs with the roles `weight` and `bias`,
as a network kernel's are, so the reference machine runs the call with the compiled program's
weights given as data. The value is `Real` for a codomain of width one and
`Tensor[Real]([width])` otherwise, and it is bound before the expression that reads it.

<!-- compile: false -->
```qvr
morphism proj : Feature * Feature -> Score [role=function, param_source=mlp(4), output=tanh]

program p : Feature -> Score
    sample a <- Normal(loc=0.0, scale=1.0) [over=Feature]
    sample b <- Normal(loc=0.0, scale=1.0) [over=Feature]
    let z = 2.0 * proj(a, b) + 1.0
    observe y <- Normal(loc=z, scale=0.5)
    return z
```

A function contributes no site. A `sample` or `observe` that draws from one is refused, as are a
call inside a `factor` or lambda body (bind the call in a `let` of its own) and a call whose
arguments miss a factor of its domain. As a scan's cell, or as a link of a chain, a function is
applied, not drawn from.

Every transpile target refuses a function with weights under `param-source:linear` or
`param-source:mlp`, because the weights are neither sites nor data a target is given. A function
with `param_source=identity` has no weights, being its output nonlinearity applied to its argument,
and every target renders it.

## Sequence functions

A learned function declared with `param_source=self_attention` or `param_source=positions(N)` reads
a whole sequence. A program calls it in a `let` with one argument, a table `Tensor[Real]([T, d])`
whose row width $d$ is the width of its domain and whose length $T$ is any extent, open or fixed. A
data input passed straight to it is a table `Tensor[Real]([<name>_extent, d])`. The value is the
table `Tensor[Real]([T, m])` for a codomain of width $m$.

<!-- compile: false -->
```qvr
morphism attn : Model -> Model [role=function, param_source=self_attention, heads=2, causal=true]
morphism positions : Model -> Model [role=function, param_source=positions(32)]
```

Self-attention [@vaswani2017attention] with $h$ heads, each of width $e = m / h$, elaborates to
comprehensions over the $T$ positions. For an input table $x$ with rows $x_i \in \mathbb{R}^d$:

1. The query, key, and value tables apply one affine map to each row, $q_i = W_q x_i + b_q$, and
   likewise $k_i$ and $v_i$, with weights `attn_param_query_weight` of shape $(m, d)$ and bias
   `attn_param_query_bias` of shape $(m)$, and the same for `key` and `value`.
2. For head $g$ and positions $i$ and $j$, the score is
   $s^{(g)}_{ij} = \frac{1}{\sqrt{e}} \sum_{u=0}^{e-1} q_{i, ge+u}\, k_{j, ge+u}$, and the weights
   $a^{(g)}_{ij}$ are the softmax of $s^{(g)}_{i \cdot}$ over $j$. Under `causal=true`, the softmax
   runs over $j \le i$ only, and $a^{(g)}_{ij} = 0$ for $j > i$. It is computed as
   $\exp\bigl(s^{(g)}_{ij} - \max_{j' \le i} s^{(g)}_{ij'}\bigr)$ for $j \le i$, normalized to sum
   to one.
3. The context joins the heads' weighted values end to end,
   $c_{i, c'} = \sum_j a^{(\lfloor c'/e \rfloor)}_{ij}\, v_{j, c'}$ for each column $c'$.
4. The output is the projection $W_o c_i + b_o$ at every position, with $W_o$ the weight
   `attn_param_weight` of shape $(m, m)$ and $b_o$ the bias `attn_param_bias`.

A position table elaborates to $x_{ik} + P_{ik}$, with $P$ the table
`positions_param_positions` of shape $(N, d)$, a program input of role `weight`. A sequence longer
than $N$ is refused when it is read. The weights are typed program inputs, so the reference machine
runs a call with the compiled program's weights given as data, and the compiled runtime holds the
network as the submodule `_function_<f>` (with parameters `source.query`, `source.key`,
`source.value`, `source.output`, or `source.table`). Each call is bound as `<f>__call<n>`.

A sequence function contributes no site. These uses are refused: drawing from one, a call whose
argument is not one table of rows of width $d$ (a single vector, for instance), a call inside a
`factor` or lambda body, and a sequence function composed in a chain or used as a scan's cell, which
would apply its morphism one position at a time (`chain:sequence-function:<f>`, where the chain is
drawn). The options `heads`, a positive divisor of $m$, and `causal`, `true` or `false` and
required, belong to `self_attention`. The options `output`, `activation`, and `weights=latent` are
refused for both kinds of network.

Every transpile target refuses a sequence function with `param-source:self_attention` or
`param-source:positions`, because its weights are neither sites nor data a target is given.

## Latent network weights

A network kernel or a learned function declared `[weights=latent, prior_scale=s]` draws its
network's weights as sites, which places a prior on every weight, as in a Bayesian neural network
[@mackay1992bayesian]. Before elaboration, each program without value parameters that reaches the
morphism (through a draw, a `let` call, a `define` with a scan, a scan's cell, or a called program)
gains one draw per layer at the head of its body:

<!-- compile: false -->
```qvr
sample net_param_layer0_weight : net__w0_rows <- Normal(loc=0.0, scale=s) [over=net__w0_cols, iid_over=net__w0_rows]
sample net_param_layer0_bias <- Normal(loc=0.0, scale=s) [over=net__w0_rows]
```

Here `net__w<k>_rows` and `net__w<k>_cols` are finite objects of the layer's height and width. The
network's elaboration then reads these bindings where it would otherwise declare `weight` and `bias`
program inputs, so the well-typed computation has no such inputs, and the reference machine draws or
replays the weights as sites. The compiled runtime applies the network at the weights drawn in the
run. A kernel with latent weights is drawn from with its inputs named. A morphism with latent weights
composed in a `define` without a scan, or in a morphism's `~ <expression>`, is refused. Every
transpile target refuses the network as `param-source:linear:latent` or `param-source:mlp:latent`.

## Running and scoring

An invocation installs a handler for `random` that draws or replays each site and scores it, and an
accumulator for `score`. Given `--site name=value`, that site is replayed at the value and scored,
and every site not given is drawn from its distribution in the model and scored. The result holds
the program's value, its log joint, and its
[execution trace](../../guides/inference-foundations.md#trace-and-sample-sites).
[`run_program`][quivers.qiec.program_runtime.run_program] is the stricter replay that requires a
value for every site. A posterior given observed data is computed by an
[inference driver](../../guides/inference-foundations.md), not by an invocation. The command line
and Python forms are under [run an entry](execution-and-tooling.md#run-an-entry), and the support
of each target under [transpilation](execution-and-tooling.md#transpilation) and in the
[transpilation support matrix](../../transpile-support.md).
