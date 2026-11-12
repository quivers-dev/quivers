# DSL Programs and Let-Expressions

This page covers `program` block syntax: program declarations
and their effect signatures, the bind / observe / marginalize / let
steps that make up a program body, the axis-role clause that
configures structured priors and likelihoods, the let-expression
language with its complete set of primitives, and factor expressions.

The grammar summary lives in the
[DSL overview](dsl-overview.md#grammar); declarations of objects,
morphisms, kernels, and combinators are detailed in
[DSL Declarations](dsl-declarations.md).

## Program declarations

Read a program as a sequence of questions. `sample` asks for a random
value; `let` computes from values already available; `observe` scores
a measured value under the current distribution; `return` selects what
the caller receives. `marginalize` asks the model to account for every
value of a finite unknown rather than select one. These steps define
the model; a separate inference procedure estimates its posterior.

A `program` block defines a probabilistic program. The body is a
sequence of *steps* (bind, observe, let, marginalize) followed by
`return`. Sampling steps compose probability kernels over the accumulated variable
context $\Phi$. Observation and score steps additionally reweight executions;
they generally produce an unnormalized measure until normalized. Thus a
purely generative program admits the usual [Kleisli](monadic.md#kleisli-category)
interpretation in [`Kern`](../api/stochastic/categories.md), while a scored
program requires the corresponding measure/weight semantics.

<!-- compile: false -->
```qvr
program my_prog : X -> Y
    sample mu <- LogitNormal(mu=0.0, sigma=1.0)
    sample x <- Normal(loc=mu, scale=1.0)

    return x

program with_params(a, b) : (X * Z) -> Y
    let w = a

    sample x <- f(w)
    sample y <- g(x, b)
    return y
```

The compiled program is a
[`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram);
see the [monadic programs guide](programs.md) for the runtime
contract (`rsample`, `log_joint`, the `observations` dict).

### Effect signatures

A program declaration may carry an `effects = [...]` entry in its
option block listing a subset of `{Sample, Score, Marginal, Pure}`.
The compiler verifies that the body's actual effects are a subset
of the declared set; `effects=[Pure]` rejects any sample, score, or
marginal binds.

<!-- compile: false -->
```qvr
program prior : X -> Y [effects=[Sample]]
    sample mu <- Normal(loc=0.0, scale=1.0)
    return mu

program deterministic : X -> X [effects=[Pure]]
    let y = x
    return y
```

See the [compositional effects guide](effects.md) for the algebraic
basis of the effect system.

### QIEC typed computations

The **typed computation layer** is what a `program` block elaborates to; a
`define` writes such a computation directly. Its header gives a value result
and an instance row, while `perform` names an operation through a lexical
effect instance:

<!-- compile: qiec -->
```qvr
effect Tap
    ping : Int -> Int

instance tap : Tap

define echo(value : Int) : Int !{tap} =
    let answer <- perform tap.ping(value)
    return answer
```

The stable computation terms are `return`, `let ... <- perform ...`, bare
`perform` sequencing, a call of a named computation `let x <- f[statics](args)`,
`handle ... with ... in`, `with instance x : E in` allocating a scoped
instance, and indexed `case` with an explicit motive. Rows contain lexical
instance names, as in `!{tap}`; an open row has the form
`!{tap | rho lacks tap}`. The tail and each `lacks` constraint are checked
rather than treated as comments.

A call is checked against the callee's signature instantiated at the call's
static arguments, and the module's call graph is checked one strongly
connected component at a time, so mutually recursive computations need no
forward declaration; a recursive computation must reach a `case` branch or an
`if` arm that does not call again, which is what lets it stop. A scoped
instance may not survive in its body's row: the checker reports
`qiec-instance-escape` for a body that still performs on it. A [handler](../reference/qvr/effects-and-handlers.md#handler-declarations)'s
clauses may be authored in the source with `=>` bodies, each resuming as often
as its grade admits (`0`, `aff`, `1`, or `omega`), and a foreign handler's
clauses are process-local attachments. A `program` elaborates to such a
computation; a program draws from another with `sample x <- sub(args)` (or
`sample (a, b) <- sub` for a pair), which runs `sub`'s steps in place under
names of its own (`sub`'s local `z` becomes `x$z`), and calls a computation
with `let x <- f(args)`.

Eight dynamic targets execute the full computation graph through generated
target runtimes. Stan lowers closed monomorphic effect-free scalar computations
as user-defined functions and refuses every other feature under
`qiec:capability:<feature>:<name>`; BUGS and JAGS refuse a call under
`call:graph:<name>`. A module with QIEC declarations but no computation the
program reaches remains valid on every target. The complete syntax appears in
[QVR computation reference](../reference/qvr/computations.md) and
[probabilistic-program reference](../reference/qvr/probabilistic-programs.md).

### Kleisli bind syntax

The `sample` keyword introduces a draw, and `<-` separates its
binding pattern from the distribution call:

<!-- compile: false -->
```qvr
sample x <- Normal(loc=0.0, scale=1.0)
```

This step introduces `x` as a random variable distributed according
to the given family. Indexed samples, scored observations, and scoped
marginalizations use the same `<-` separator with their respective
keywords.

### Indexed bind (plate)

A **plate** repeats a conditional distribution across an index set.
If `Subject` contains four people, `sample by_subject : Subject` creates
four subject values. They may be independent conditional on a shared
scale `sigma`, while becoming dependent after uncertainty about that
scale is integrated out. An unusually large effect for one subject can
raise the inferred shared scale and thereby widen the plausible effects
for the others.

`sample v : A <- Family(args)` declares `v` as an $A$-indexed family of
independent $F$-distributed draws. The per-index kernel assigns a distribution on `K` to each index in `A`.
Taking their independent product produces a distribution on `K^A`.
This product factorization specifies conditional independence across
indices. To introduce dependence, pass a shared random parameter or
choose a family with a joint event over the desired axes.

<!-- compile: false -->
```qvr
object Item : FinSet 1000
sample duration_incr : Item <- HalfNormal(scale=1.0)
sample by_subject    : Subject <- Normal(loc=0.0, scale=sigma)
```

### Indexed observe

For independent observations conditional on the model's latents, multiply
their likelihoods, or equivalently add their log likelihoods. A Normal
density may exceed one. For a uniform density on an interval of width
`0.1`, the density is `10` and an observation contributes `log(10)`.
Such scores are likelihood weights; a sub-probability kernel would
additionally require total mass at most one.

`observe r : N <- Family(args)` accumulates a batched
log-likelihood, reweighting the current execution by $\prod_{n \in N} p_F(r_{\mathrm{obs}}(n);
\theta(n, \phi))$. The response buffer `r` is supplied at runtime
via the `observations` dict passed to
[`MonadicProgram.rsample`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.rsample),
[`MonadicProgram.log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint),
or [`ELBO.forward`](../api/inference/elbo.md). Family arguments may
use bracket-indexed sections `theta[N]` to refer to plate variables.

<!-- compile: false -->
```qvr
observe cloze_resp : RespCloze <- Bernoulli(probs=intercept_cloze)
```

### Scoped marginalize

`marginalize c : A <- F(args)` followed by an indented body
introduces a coordinate `c` bound to a kernel `F(args)`,
optionally `A`-indexed, with the indented block as its
integration scope. At the end of the scope the coordinate is
pushed forward through the projection $\pi : \Phi \times C \to
\Phi$, integrating it out by
[log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) for the supported
finite discrete enumeration;
`c` then falls out of scope.

<!-- compile: false -->
```qvr
marginalize class : Item <- Categorical(logits=class_logits)
    observe r : N <- Bernoulli(probs=theta[class[N]])
```

The grouped form with `[over=G]` and per-observe `[via=<idx>]`
options is the fibred [marginalization](programs-hierarchical.md#grouped-marginalization-fibred-discrete-latents) construct; see
[hierarchical programs](programs-hierarchical.md#grouped-marginalization-fibred-discrete-latents).

### Indexed gather in `let`

If `by_verb = [0.2, -0.1, 0.5]` and `verb_of_item = [2, 0, 2]`,
the gather returns `[0.5, 0.2, 0.5]`. Items zero and two share the
same realized verb effect. Sampling again per item would define a
different model. The index map is what preserves the intended sharing.

A `let` expression of the form `arr[idx]` gathers values by their indices.
Once a plate has been sampled, this is deterministic reindexing of its
realized values. For a plate `v : A -> B` and an
index morphism $\iota : N \to A$, the gather $\iota^* v = v \circ
\iota$ is itself a [`Kern`](../api/stochastic/categories.md)-morphism
$N \to B$.

<!-- compile: false -->
```qvr
sample by_verb : Verb <- Normal(loc=0.0, scale=sigma)
let intercept_for_item = by_verb[verb_of_item]
```

A subscript of several indices reads them together. With `row_of` and
`col_of` index vectors over the same rows, `table[row_of, col_of]` is one
entry per row, `table[row_of[n], col_of[n]]` at row `n`, as advanced
indexing reads it in NumPy and as a loop over the rows reads it in Stan. The
result has the index vectors' shape followed by whatever axes of `table` the
subscript leaves, and an `Int` index (a literal or a factor binder) stands for
the same value at every row, so `table[row_of, 1]` is the second column read
at each row. Index vectors of two different shapes in one subscript are
refused at the subscript. A single index vector, `table[row_of]`, gathers whole
rows. A matrix of observations or covariates indexed by (row, column) pairs,
as in matrix factorization or an item-response table, is thus one let:

<!-- compile: false -->
```qvr
sample user : User <- Normal(loc=0.0, scale=1.0)
sample item : Item <- Normal(loc=0.0, scale=1.0)
let affinity = factor u : User, i : Item in user[u] * item[i]
let mu = affinity[user_of, item_of]
observe rating : Rating <- Normal(loc=mu, scale=1.0)
```

### Parametric programs

A `program` declaration whose parameter list contains *typed*
parameters denotes a dependent family of kernels rather than a
single kernel:

$$
\llbracket P \rrbracket \;:\; \prod_{p_1 : P_1} \cdots \prod_{p_k : P_k} \mathbf{Kern}\bigl(\mathrm{dom}(p), \mathrm{cod}(p)\bigr).
$$

Three parameter universes are available:

| Kind | Universe | Quantifies over |
|---|---|---|
| `FinSet`, `Space`, `Object` | object of the relevant subcategory | the carrier of a plate |
| `Real`, `Nat` | hom-object of scalar type | a hyperparameter value |
| `Mor[A, B]` | the hom-set $\mathbf{Kern}(A, B)$ | a kernel passed in by name |

Parametric programs are *not* compiled to runtime `MonadicProgram`s
in isolation; the compiler stores them as templates and inlines
them at each call site:

<!-- compile: false -->
```qvr
sample v <- template(arg1, arg2, ...)
```

At each call site the template's body is substituted (formal
parameters to actual arguments) and α-renamed (internal latents are
prefixed by `v$`, the return variable is renamed to `v` directly).
The renamed step list is inlined into the caller, so distinct call
sites contribute distinct factors to the parent's joint kernel:
fresh latents per use, no inadvertent tying.

The substitution reaches every position a name can occupy in a `let`
or `score` right-hand side and in the arguments of a computation call:
the array and the indices of `arr[idx]`, call arguments, method
receivers, list and tuple items, lambda and `factor` bodies, and an
object parameter named as a `factor` binder's index. It is
capture-avoiding: a lambda parameter or `factor` binder shadows a
parameter or local of the same spelling, and is renamed if a
substituted name would otherwise be captured by it.

<!-- compile: false -->
```qvr
# Parametric random-intercepts template: one HalfNormal scale and
# a per-level Normal(loc=0, scale=sigma) plate, polymorphic over the grouping
# object G and the half-normal hyperparameter scale.
program random_intercepts (G : FinSet, prior_scale : Real) : G -> Real 1
    sample sigma <- HalfNormal(scale=prior_scale)
    sample v : G <- Normal(loc=0.0, scale=sigma)
    return v
```

Worked hierarchical examples (crossed random intercepts, grouped
marginalization over discrete classes) live in the
[hierarchical programs guide](programs-hierarchical.md).

### Posterior blocks

A `program name(latents) : domain -> codomain`
declaration denotes a deterministic post-[conditioning](inference-foundations.md#conditioning-on-observations) kernel. The
`over = model` entry in its option block marks the program as
consuming the named model's latents; the consumed latents appear
as data parameters in the parameter list. The
`effects=[Pure]` signature rejects any sample, score, or
marginal binds; the body must remain pure; a marginalization that performs scoring is
not allowed by this signature. Categorically it is a
[`Kern`](../api/stochastic/categories.md)-morphism $\text{Latents}
\to \tau_{\mathrm{out}}$ that lifts to $\text{Data} \to
\mathcal{G}(\tau_{\mathrm{out}})$ by post-composition with the
model's posterior kernel $q(\theta \mid \mathrm{data})$.

<!-- compile: false -->
```qvr
object Logits4 : Real 4

program scored : Item -> Logits4
    sample raw_logits <- Normal(loc=0.0, scale=1.0)
    return raw_logits

program class_probs(raw_logits) : Item -> Logits4 [effects=[Pure], over=scored]
    let probs = softmax(raw_logits)
    return probs
```

The data parameter `raw_logits` names the model latent the body
consumes; the runtime supplies a per-sample snapshot of the model's
[trace](inference-foundations.md#trace-and-sample-sites).

## Axis-role clause: `over` and `iid_over`

Ask what counts as [one outcome](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape) [@pyroTensorShapes] before choosing the axes. A probability
vector over three topics is one Dirichlet outcome; ten documents can
each have such a vector. The topic axis is an event axis and the
document axis is a batch/plate axis, producing ten jointly constrained
three-component vectors. `over=Topic` records the former; `: Doc` or
`iid_over=Doc` records the latter. This declaration constrains the three
topic probabilities within each document to sum to one and treats the
documents as [conditionally independent draws](https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate)
[@pyroTensorShapes]. Each family’s [event rank](continuous-families.md#event-rank-and-axis-role-syntax)
determines how many axes belong to one outcome.

Every distribution-bearing form (kernel and latent-prior morphism
declarations, `sample` / `observe` / `marginalize` steps) accepts
two option-block keys that configure how the family's axes
decompose: `over` and `iid_over`. Both live inside the bracketed
option block, before any trailing `~ Family(...)` initializer:

```
... [..., over=<axes>, iid_over=<axes>] ~ Family(args)
```

`over=<axes>` names the **event axes**: the axes on which the
family's joint structure lives. On a `morphism` declaration the axis
count must match the family's declared `event_rank`, and on a `sample`
or `observe` step it must be at least that rank, as described below (0 for scalar families like Normal /
Beta / Gamma; 1 for vector families like
[`MultivariateNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMultivariateNormal),
[`Dirichlet`](../api/continuous/families.md#quivers.continuous.families.ConditionalDirichlet),
[`ConditionalGaussianProcess`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess) (named `GP` in QVR),
or
[`ConditionalHorseshoe`](../api/continuous/families.md#quivers.continuous.families.ConditionalHorseshoe);
2 for matrix families like
[`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal),
[`Wishart`](../api/continuous/families.md#quivers.continuous.families.ConditionalWishart),
[`InverseWishart`](../api/continuous/families.md#quivers.continuous.families.ConditionalInverseWishart)).
A single axis is written bare (`over=A`); multiple axes use a
bracketed list (`over=[A, B]`). The positional ordering of `over`
axes corresponds positionally to the family's declared event-axis
ordering (for asymmetric families like `MatrixNormal`, the first
axis is the row axis, the second the column axis). The full
event-rank table lives in
[continuous families](continuous-families.md#event-rank-and-axis-role-syntax).

On a `sample` or `observe` step, `over` may list more axes than the
family's event rank. The leading axes beyond the event rank then index an independent
array of variates, and the trailing ones are the family's event axes:
`sample w : A <- Normal(loc=0.0, scale=1.0) [over=[B, C]]` draws an
`(A, B, C)` array of independent normals, and `sample p : A <-
Dirichlet(concentration=1.0) [over=[B, K]]` draws one `K`-simplex point
per element of `A` and `B`, an `(A, B, K)` array. The site holds every
listed axis in the order written, after the plate of the index
annotation, so a body reads it as `w[a, b, c]`. The independent axes are
batch axes of the site, exactly as if they were written in `iid_over`. A
matrix-valued family (event rank 2) takes no such leading axes, and the
compiler rejects them. A `morphism` declaration's clause keeps the exact
count below.

`iid_over=<axes>` is an optional readability assertion naming the
batch axes (the complement of `over`). Any axis not in `over` is
batched by default, which categorically is a product of independent
distributions on that axis. On a sample step, the `: A` annotation
beside an `over` clause is the batch plate: `sample rows : Doc <-
Dirichlet(concentration=1.0) [over=Topic]` draws one point of the `Topic`-simplex
per document, while an annotation naming one of the `over` axes
restates it. Without an `over`, the annotation on a vector family
names the family's own axis when nothing else fixes it (`sample pi :
K <- Dirichlet(concentration=1.0)` draws one `K`-simplex point) and the batch plate
when the arguments fix the event otherwise (`sample pc : Item <-
Dirichlet(concentration=[1.0, 2.0, 3.0])` draws one three-simplex point per item).

A vector family's single parameter is named like every other family parameter.
Write `Dirichlet(concentration=[1.0, 2.0, 3.0])` for an explicit vector or
`Dirichlet(concentration=1.0)` for a symmetric concentration at the dimension
fixed by the step's annotation, its `over` axis, or the program's declared
codomain; a draw that fixes no simplex dimension is refused at compile time
with its location.

**Axis names.** Names resolve against the named factors of the
surrounding morphism's dom and cod (or the type annotation `: T`
on a sample / observe step). The reserved tokens `dom` and `cod`
are shortcuts when that side is a single unfactored object; for a
product-typed side, every factor must be named explicitly.

**Categorical reading.** The axis-role clause preserves the distinction
between joint-on-a-product-space (the family's event with possibly
non-trivial correlation) and product-of-independents (iid batches
across an axis), and between a flat MVN over $\dim(A) \cdot \dim(B)$
with dense covariance versus a
[`MatrixNormal`](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
with [Kronecker structure](https://en.wikipedia.org/wiki/Kronecker_product)
$V \otimes U$. There is no auto-substitution between families with
different event ranks. Renaming or refactoring a morphism's type
invalidates axis references at type-check time rather than silently
rebinding.

**Compile-time errors.** Two diagnostics fire here:

- *"axis count does not match family event_rank"*: on a `morphism`
  declaration, the family declares event_rank `k` but `over` lists
  `j ≠ k` axes. Common
  when migrating from a flat
  [`MultivariateNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMultivariateNormal)
  (event_rank 1) to a matrix
  [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal)
  (event_rank 2).
- *"unknown axis name(s) `A` in `over`"*: an `over` name is not a
  factor of the morphism's dom or cod. Resolve by renaming, or by
  using `dom` / `cod` if the side is a single unfactored object.

<!-- compile: false -->
```qvr
# Vector prior: 5-dim MVN over the codomain axis.
sample mu : Real 5 <- MultivariateNormal(loc=zeros, scale_tril=L) [over=cod]

# Matrix prior on a morphism: Kronecker MatrixNormal.
morphism W : Real 32 -> Real 64 [role=latent, over=[dom, cod]]
    ~ MatrixNormal(loc=loc, row_covariance=row_scale, col_covariance=col_scale)

# Per-row Dirichlet on a transition kernel: each row is a K-dim
# simplex independently, rows are iid.
morphism T : Real K -> Real K [role=latent, over=cod, iid_over=dom]
    ~ Dirichlet(concentration=alpha)

# MVN response per observation row.
observe y : N <- MultivariateNormal(loc=mu_hat, scale_tril=scale_tril) [over=cod]
```

## Let expressions (arithmetic, primitives, and collections)

Inside a `program` block or a `define` computation, `let` bindings support
arithmetic with standard operator precedence, unary negation, checked tensor
primitives, and finite collection operations:

<!-- compile: false -->
```qvr
# arithmetic: +, -, *, /
let eta = mu + sigma * z_raw + lambda * shared_factor
let adjusted = (1.0 - lapse) * p_raw + 0.5 * lapse
let mean = (x + y + z) / 3.0
let negated = -raw_score

# built-in primitives
let prob = sigmoid(eta)
let positive = softplus(raw)
let log_rate = log(rate)
let magnitude = abs(x - 0.5)
let monotone = cumsum(increments)
let weights = softmax(logits)
let centered = map(xs, x -> x - mean)
```

Each `let`-builtin denotes a deterministic measurable map, lifted
into the [Kleisli category](monadic.md#kleisli-category) as a [Dirac
kernel](https://en.wikipedia.org/wiki/Dirac_delta_function).

### Checked primitive reference

The checked registry defines the portable QIEC expression set. Operators and calls are
resolved by argument type before execution or transpilation. A reduction
collapses the last axis of its argument and a rowwise operation acts along that
axis, keeping the shape. For a vector over a plate, as in `sum(theta)` over
`theta : School`, the last axis is the only one and the result is the plate's
total. For a plate of event vectors, as in `sum(z_row * w_row)` over a
`Resp`-plated value with an `[over=LatentDim]` event axis, the result keeps the
plate and holds one inner product per row. For a table a `factor` builds over
`Row` and `Unit`, the result holds one number per `Row`. `normalize` divides
each row by its sum. Every execution path and transpile target reads the
builtins this way. For a
[contraction](dsl-contractions.md#type-driven-wiring-inference) over a named axis, use a typed
[contraction declaration](dsl-contractions.md).

**Conversions and weights**: `real`, `int`, `weight`, `weight_value`.

**Arithmetic and comparisons**: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`,
`<=`, `>`, `>=`, `&&`, `||`; and the call forms `pow`, `abs`, `min`, and
`max`. The two-argument `min(a, b)` and `max(a, b)` are pointwise, unlike
the one-argument reductions below: two tensors compare entry by entry, a
scalar broadcasts against a tensor, and two `Int` operands give an `Int`,
so `ys[max(t - 1, 0)]` clamps an index arithmetic expression into range
on every target.

**Conditional values and Booleans as numbers**: `select(c, a, b)` takes `a`
where the `Bool` condition `c` holds and `b` elsewhere, entry by entry; a
number among the three broadcasts against the tensors, and the tensors share
one shape, so `select(v > 0.0, v, 0.1 * v)` is a leaky rectifier of a vector.
Both branches are of one type, `Real` or `Int` (write `0.0`, not `0`, beside a
real branch). `real(c)` and `int(c)` read a `Bool` as one where it holds and
zero elsewhere, so a mask needs no integer clamp:

<!-- compile: false -->
```qvr
let causal = factor i : Pos, j : Pos in real(j <= i)
let hinge = select(margin < 1.0, 1.0 - margin, 0.0)
```

`select` evaluates both branches and then keeps one entry of either, as
`torch.where` does, so it is not a guard: a branch that is `NaN` or infinite
where it is not taken still is computed, and its gradient there is the
product of zero and an infinite derivative, which is `NaN`. Where the untaken
branch is finite its gradient is zero and the gradient is the taken branch's.
Keep each branch finite on the whole domain (`log(max(v, 1e-12))` rather than
`log(v)`) when the condition is what keeps it in range. Every transpile target
spells it: `torch.where`, `jnp.where`, `pymc.math.switch`, `tf.where`, Stan's
`c ? a : b`, Julia's `ifelse.(c, a, b)`, JAGS's `ifelse`, and a runtime
function on WebPPL and Church; BUGS has no conditional function and computes
`b + c * (a - b)`, which is the same wherever both branches are finite.

**Pointwise transcendentals**: `exp`, `expm1`, `log`, `log1p`,
`log2`, `log10`, `sqrt`, `rsqrt`, `square`, `abs`, `sign`,
`reciprocal`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
`sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `floor`, `ceil`, `round`,
`trunc`, `erf`, `erfc`, `erfinv`, `lgamma`, `digamma`.

**Activations**: `relu`, `relu6`, `elu`, `selu`, `gelu`, `silu`, `mish`,
`softplus`, `logsigmoid`, `softsign`, `sigmoid`, and `tanh`.

Every transpile target computes each pointwise builtin above with the
reference's meaning: through a function of its own where it has one
(Stan's `inv_logit` for `sigmoid`, JAGS's `loggam` for `lgamma`), and
otherwise through a definition in terms of the others (`gelu(x)` as
`0.5 * x * (1 + erf(x / sqrt(2)))`, JAGS's `erf` through its normal CDF
`phi`). `round` rounds a tie to even on every target. A builtin a target
cannot express is refused by name when the program is transpiled; BUGS
and JAGS refuse `digamma`.

**Tensor reductions**: `sum`, `mean`, `min`, `max`, `prod`, `logsumexp`.

**Last-axis operations**: `softmax`, `log_softmax`, `cumsum`, `sort`,
`normalize`.

**Joining and cutting the last axis**: `concat(a, b, ...)` joins two or more
tensors of one element type along their last axis; their leading axes must
agree, and the last extents add, so `concat(h, c)` of two `Real 4` vectors is a
vector of 8 entries and `concat` of two `(2, 3)` tables is `(2, 6)`.
`slice(v, start, length)` keeps the `length` entries of the last axis from
`start` (counted from zero), so the result's last extent is `length`. Both
bounds are non-negative integer literals, since the checked type states the
slice's extent; a position computed at run time is read with a gather, `v[i]`.
A number joins as the one-entry vector `[x]`, and a slice past the end of the
axis, a zero length, a number where a tensor belongs, or tensors of different
element types or leading axes are refused at the argument. This is how a scan
carries a state of several parts as one vector, an LSTM's hidden and cell state
for instance:

<!-- compile: false -->
```qvr
let h_prev = slice(state, 0, 4)
let c_prev = slice(state, 4, 4)
let c_new = f * c_prev + i * g
let h_new = o * tanh(c_new)
let state_new = concat(h_new, c_new)
```

The checked elaboration writes each call as a tensor of entries selected at
literal positions, under a comprehension per leading axis, so the reference
machine and every transpile target compute it with the selections and tensor
literals they already read; the compiled runtime calls `torch.cat` and a
last-axis slice. Within one run of the compiled runtime the input-batch axis is
never among a let's axes, so a `concat` or `slice` never reads it.

**Products and normalizations**: `matmul(a, b)` sums the last axis of `a`
against the first axis of `b`, which is a vector or a matrix with that axis's
extent: a matrix times a vector is a vector, a vector times a matrix a vector,
a vector times a vector a number, and `a` of any rank is multiplied row by row,
so `matmul(X, W)` with `X : (T, D)` and `W : (D, H)` is `(T, H)`. The two
extents are checked to agree. `layer_norm(v)` and `rms_norm(v)` normalize each
row along the last axis and keep the shape: `layer_norm(v)` is
`(v - mean(v)) / sqrt(mean((v - mean(v))^2) + eps)` (the biased variance, as
`torch.nn.functional.layer_norm` computes it) and `rms_norm(v)` is
`v / sqrt(mean(v^2) + eps)`. An optional second argument, a positive real
literal, sets `eps`; it is `0.00001` and `0.000001` when omitted. Neither takes
a learned gain or bias; multiply and add them as ordinary lets:

<!-- compile: false -->
```qvr
let h = tanh(matmul(W_in, x_t) + matmul(W_rec, h_prev) + b)
let normed = gain * layer_norm(h) + shift
```

The checked elaboration writes `matmul` as a sum of products under one
comprehension per row and column, and the normalizations as row statistics
bound once and a comprehension of quotients, so the reference machine and
every transpile target compute them with forms they already read; the compiled
runtime calls `torch.matmul`, `torch.nn.functional.layer_norm`, and the RMS
formula.

**Finite collections**: `map`, `fold`, `length`, and `logsumexp_over` are pure.
`traverse` sequences a named computation and thus appears on the right
side of `<-`. See the [collection reference](../reference/qvr/collection-expressions.md)
for their typing and shape rules.

The eager PyTorch compiler also exposes the native-only operations `alpha_dropout`,
`amax`, `amin`, `argmax`, `argmin`, `celu`, `clamp`, `cummax`, `cummin`,
`cumprod`, `dropout`, `flip`, `glu`, `hardshrink`, `hardsigmoid`, `hardswish`,
`hardtanh`, `leaky_relu`, `neg`, `norm`, `prelu`,
`rrelu`, `softmin`, `softshrink`, `std`, `swish`, `tanhshrink`, `threshold`,
and `var`. These are an attachment boundary, not portable QIEC primitives. If
a program uses one, `qvr check` reports `qiec-builtin:host-only` rather than
silently omitting the program from the checked module. Use
`qvr check --target TARGET` before relying on a transpiler.

The capability registry lives in `quivers.dsl.pure_builtins`. Each entry records
its eager implementation and arities, purity, shape action, checked QIEC form,
and supported execution routes; eager dispatch and checked name resolution are
derived from it. Calls resolve against this registry before module-scope
callables. User-defined callable arity is checked at compile time.

### Factor expressions: assembling indexed tensors

A factor expression is a typed comprehension: enumerate a finite set
of index tuples and compute a value at every tuple. Indexing reads one
cell of a table; a factor expression constructs the whole table. For
two verbs and three classes, the body runs at six `(verb, class)`
pairs and produces a table with those two leading axes.

The `factor` expression in a `let` body builds a
finite-domain-indexed tensor by evaluating a body once per tuple of
index values. Categorically it is the *left adjoint of indexing*:
while `arr[i, j, ...]` is the elimination rule for `I_1 × ... × I_n
-> body_type` (Kleisli pullback of a plate variable along a finite
fibration), `factor` is the introduction rule.

#### Uniform form

```
factor v_1 : I_1, v_2 : I_2, ..., v_n : I_n in <body>
```

denotes the tensor of shape `(|I_1|, ..., |I_n|, *body_shape)`
whose value at position `(i_1, ..., i_n)` is `<body>` evaluated
with `v_k := i_k`. The binder variables are integer-valued and
visible only inside the body.

<!-- compile: false -->
```qvr
object Verb : FinSet 40
object Class : FinSet 4
# Per-verb, per-class scoring table: shape (40, 4).
let cell = factor v : Verb, cls : Class in coef[v, cls] * weight[cls]
```

#### Pattern-match form (single-axis)

When each cell of a single-axis factor carries a structurally
different expression, the case form lets you state the per-index
expressions side-by-side:

<!-- compile: false -->
```qvr
let class_probs = factor cls : Class in {
    0 -> (1.0 - prob_dur) * (1.0 - prob_telic_nodur),
    1 -> (1.0 - prob_dur) *        prob_telic_nodur,
    2 ->        prob_dur  * (1.0 - prob_telic_dur),
    3 ->        prob_dur  *        prob_telic_dur,
}
```

The case labels must cover `{0, ..., |Index|-1}` exactly; the
compiler rejects gaps, duplicates, or out-of-range labels at
compile time. Braces and comma separators delimit the cases. This
is the natural syntax for structured categorical priors: each
cell of a `Class`-shape probability vector is built from a
different combination of upstream scalar latents.

## Inline distributions

Bind and observe steps support inline distribution construction
with any mix of literal and variable arguments. More than forty
registered families accept literal-or-variable arguments at any position:

<!-- compile: false -->
```qvr
# all-literal (fixed): Unit -> codomain
sample x <- Normal(loc=0.0, scale=1.0)
sample p <- Beta(concentration1=2.0, concentration0=5.0)

# all-variable (direct): variables -> codomain
sample y <- Normal(loc=mu, scale=sigma)
sample b <- Bernoulli(probs=theta)

# mixed literal / variable: any combination works
sample h_cand <- Normal(loc=reset_hidden, scale=0.5)
sample z <- Normal(loc=0.0, scale=learned_scale)
sample r <- TruncatedNormal(mu=mu, sigma=sigma, low=0.0, high=1.0)

# negative literals
sample z <- Normal(loc=-1.5, scale=0.3)
```

A representative subset of the inline-distribution registry (the
full set is documented in
[continuous families](continuous-families.md#family-registry)):

| Family | Parameters | Codomain |
|---|---|---|
| `Normal` | `loc`, `scale` | `Real N` |
| `LogitNormal` | `mu`, `sigma` | `Real N [low=0, high=1]` |
| `Uniform` | `low`, `high` | `Real N [low=..., high=...]` |
| `Bernoulli` | `probs` | `FinSet 2` |
| `Beta` | `concentration1`, `concentration0` | `Real N [low=0, high=1]` |
| `Exponential` | `rate` | `Real N [low=0]` |
| `HalfCauchy` | `scale` | `Real N [low=0]` |
| `HalfNormal` | `scale` | `Real N [low=0]` |
| `LogNormal` | `loc`, `scale` | `Real N [low=0]` |
| `Gamma` | `concentration`, `rate` | `Real N [low=0]` |
| `Dirichlet` | `concentration` | `Simplex K` |
| `TruncatedNormal` | `mu`, `sigma`, `low`, `high` | `Real N [low=..., high=...]` |
| `MultivariateNormal` | `loc`, `scale_tril` | `Real N` (event_rank 1) |
| `MatrixNormal` | `loc`, `row_covariance`, `col_covariance` | `R x C` matrix (event_rank 2) |
| `OrderedProbit`, `OrderedLogistic` | `eta` (`predictor`), `cutpoints` | ordered categories |
| `LKJCholesky` | `concentration` | `CholeskyFactor K` |
| `Wishart`, `InverseWishart` | `df`, `scale_tril` | `Covariance K` |
| `Horseshoe` | `scale` | `Real N` (sparse-shrinkage prior) |

Every parameter position in every family accepts either a literal
value or a previously-bound variable. When all arguments are
literals, a fixed distribution is created; when any argument is a
variable, the family is resolved at runtime against the current
trace's values.

### Host vectors and matrices as family parameters

A name nothing in the program binds is host data, supplied when the
program runs. Filling a vector or matrix parameter of a draw with no
batch axes, it is one tensor over the parameter's own axes, and the
family ties those axes to its sample's: a `MultivariateNormal`'s `loc` is
a vector over the event axis and its `scale_tril` a square matrix over
it, and a `Wishart`'s `scale_tril` a matrix over the sample's two axes.
The draw sizes each tied axis from what it already knows: the event axes
an `[over=...]` clause names, or another argument of known size, such
as a sampled mean. An axis nothing in the draw sizes is one extent,
shared by every datum tied to it, that a later use fixes: an `[over=...]`
event axis of a draw reading the value, or the `Real N` codomain of a
program returning it.

```qvr
object Dk : FinSet 3
object Row : FinSet 4
object D : Real 3

program shrinkage : Row -> D
    # m0 is a 3-vector and L a 3 x 3 factor: the plated observe's
    # event axis Dk fixes the extent the two data share.
    sample mu <- MultivariateNormal(loc=m0, scale_tril=L)
    observe y : Row <- MultivariateNormal(loc=mu, scale_tril=L) [over=Dk]
    return mu

export shrinkage
```

The program then runs with the data at those sizes, on the compiled
runtime and on the checked reference machine alike:

<!-- python: skip -->
```python
import torch
from quivers.dsl import loads

program = loads(source)
data = {"m0": torch.zeros(3), "L": torch.eye(3), "y": torch.randn(4, 3)}
mu = torch.zeros(3)
compiled = program.morphism.log_joint(torch.zeros(1, 1), {**data, "mu": mu})
checked = program.run(
    "shrinkage",
    data={name: value.tolist() for name, value in data.items()},
    sites={"mu": mu.tolist()},
).log_joint
```

A datum of another size is refused on both paths, naming the datum and
the size the checked program reads it at; one datum read at two sizes,
or a tied argument whose size differs from the event axis it spans, is
refused at the draw. A free name filling a vector parameter of a draw
whose plate names event axes is instead one value the plate broadcasts,
as in `Dirichlet(concentration=alpha) [over=K]`; a matrix parameter,
a Cholesky factor, a covariance, or a matrix normal's mean, is never a
broadcast scalar, so it stays a matrix there. The
compiled runtime sizes every parameter when it builds the program, so it
refuses a multivariate draw whose event size nothing in the program
fixes, and the message lists the ways to fix it.

### Ordinal, inverse-Wishart, and matrix-normal draws

An ordinal response under a probit link, a covariance drawn from an
inverse Wishart, and a matrix drawn from a matrix normal are written
inline like their logistic and Wishart siblings:

```qvr
object Row : FinSet 6
object One : FinSet 1
object Dim : FinSet 2
object Rows : FinSet 2
object Cols : FinSet 3

program ordinal : One -> One
    sample eta <- Normal(loc=0.0, scale=1.0)
    observe y : Row <- OrderedProbit(eta=eta, cutpoints=[-1.0, 0.0, 1.5])
    return eta

program covariance : One -> One
    sample sigma : Dim <- InverseWishart(df=5.0, scale_tril=[[1.0, 0.0], [0.3, 0.8]])
    observe x : Row <- MultivariateNormal(loc=[0.0, 0.0], covariance_matrix=sigma) [over=Dim]
    return sigma

program regression : One -> One
    sample B <- MatrixNormal(loc=B0, row_covariance=U, col_covariance=V) [over=[Rows, Cols]]
    return B

export ordinal
export covariance
export regression
```

`OrderedProbit` puts category `k` at `Phi(c_k - eta) - Phi(c_{k-1} -
eta)` for the standard normal distribution function `Phi`, scored in log
space so a category far from the predictor keeps its probability.
`InverseWishart(df, scale_tril=L)` is the law of the inverse of a
`Wishart(df, scale_tril=L)` draw, the conventional
[inverse Wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution)
`IW(df, Psi)` with `Psi = (L L^T)^-1`; a transpile target that takes
`Psi` receives the literal factor converted. `MatrixNormal(loc, U, V)`
draws an `R x C` matrix whose row-major entries have covariance `U (x)
V`, the [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
with row covariance `U` and column covariance `V`. An inverse-Wishart
covariance is a positive-definite latent site, which MCMC and SVI move
through the unconstrained coordinates of its Cholesky factor.

### A Cholesky factor built in the program

The standard hierarchical covariance prior draws an LKJ correlation
factor [@lewandowski2009lkj],
scales its rows by per-coordinate standard deviations, and hands the
product to a multivariate normal as its Cholesky factor:

```qvr
object Dk : FinSet 3
object Row : FinSet 4
object One : FinSet 1

program correlated : One -> One
    sample L : Dk <- LKJCholesky(concentration=2.0)
    sample sigma : Dk <- HalfNormal(scale=1.0)
    sample mu : Dk <- Normal(loc=0.0, scale=1.0)
    let Ls = factor i : Dk, j : Dk in sigma[i] * L[i, j]
    observe y : Row <- MultivariateNormal(loc=mu, scale_tril=Ls) [over=Dk]
    return mu

export correlated
```

Both execution paths score it, and so do the transpile targets whose
libraries take a Cholesky factor known only at run time: Stan through
`multi_normal_cholesky`, NumPyro and Pyro through `scale_tril=`, and
PyMC through `chol=`. A `Wishart` whose `scale_tril` is such a factor
reaches NumPyro and Pyro as `scale_tril=`, and Stan and PyMC multiplied
out into the scale they take. The other targets refuse the factor under
`family:<Family>:scale_tril`; a literal factor renders everywhere,
multiplied out into a covariance.

### Alternative parameterizations

Several families accept more than one complete set of keywords, and the
keywords a draw writes select the parameterization; a draw naming two
of them, such as `Bernoulli(probs=p, logits=l)`, is refused at the draw.

| Family | Keyword sets |
|---|---|
| `MultivariateNormal` | `(loc, scale_tril)`, `(loc, covariance_matrix)`, `(loc, precision_matrix)` |
| `Wishart` | `(df, scale_tril)`, `(df, covariance_matrix)`, `(df, precision_matrix)` |
| `Bernoulli`, `Geometric`, `Categorical` | `(probs)`, `(logits)` |
| `Binomial`, `NegativeBinomial` | `(total_count, probs)`, `(total_count, logits)` |

```qvr
object Row : FinSet 6
object One : FinSet 1
object Dim : FinSet 2

program logistic : One -> One
    sample eta <- Normal(loc=0.0, scale=2.0)
    # The log-odds go straight to the family: no sigmoid in a let.
    observe y : Row <- Bernoulli(logits=eta)
    return eta

program gaussian_precision : One -> One
    sample v <- MultivariateNormal(loc=[0.0, 0.0], precision_matrix=[[2.0, -0.5], [-0.5, 1.0]]) [over=Dim]
    return v

export logistic
export gaussian_precision
```

The compiled runtime and the reference machine score each
parameterization as `torch.distributions` does when built with the same
keywords. Logits are scored in log space, `log p = -softplus(-l)` and
`log (1 - p) = -softplus(l)`, so an observation stays finite where the
probability itself rounds to zero or one: the tests check logits up to
`|l| = 30`, where the probability form scores `-inf` in single precision.

A transpile target receives the parameterization its library has under
its own spelling (Stan's `bernoulli_logit`, `binomial_logit`, and
`multi_normal_prec`, PyMC's `logit_p` and `tau`, Turing's
`BernoulliLogit` and `BinomialLogit`, the `logits=` and
`precision_matrix=` keywords of Pyro, NumPyro, and Edward2). A literal
precision matrix reaches a target without a precision parameterization
inverted into the covariance it is the inverse of; any other alternative
a target has no spelling for is refused under `family:<Family>:<keyword>`.
The [support page](../transpile-support.md) lists every outcome.

For conditional distributions (input-conditional, learned-parameter
form), use a [`morphism f : A -> B ~ Family`](dsl-declarations.md#kernel)
declaration instead.

## Examples

### Simple discrete model

<!-- compile: false -->
```qvr
object X : FinSet 3
object Y : FinSet 4
morphism f : X -> Y [role=latent]
morphism g : Y -> Y [role=latent]

define fg = f >> g

export fg
```

### Continuous conditional model

<!-- compile: false -->
```qvr
object Cond : FinSet 2
object Latent : Real 3
object Obs : Real 5

morphism prior : Cond -> Latent ~ Normal
morphism likelihood : Latent -> Obs ~ Normal
define predictive = prior >> likelihood

export predictive
```

### Probabilistic program with observations

<!-- compile: false -->
```qvr
object Data : FinSet 1
object Y : Real 2

program regression : Data -> Y
    sample theta <- LogitNormal(mu=0.0, sigma=1.0)
    sample y <- Normal(loc=theta, scale=0.5)

    observe _ <- Normal(loc=y, scale=0.1)

    return y
```

### Init recipe end-to-end

<!-- compile: false -->
```qvr
object D : FinSet 1
object In : Real 8
object Out : Real 4

# Default: weights initialized from a small-scale randn.
morphism W_default : In -> Out [role=latent, scale=0.1]

# Algebra-guided: init pulled from Algebra.init_spec, applied
# through the active algebra's bijector.  Under product_fuzzy
# (the default), the raw parameter is set to logit(ln(2) / depth);
# samples land near the algebra's neutral element on draw zero.
morphism W_auto : In -> Out [role=latent, init=auto]

export W_auto
```

For more examples, see the
[Examples Gallery](../examples/index.md). For a formal account of
what `.qvr` programs *mean*, see the
[Denotational Semantics](../semantics/index.md).
