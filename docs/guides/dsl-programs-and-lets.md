# DSL Programs and Let-Expressions

A `program` block is a probabilistic program. Its steps draw values, compute from
them, score observed data, and sum out discrete latent variables, and its last step
returns a value. [DSL Declarations](dsl-declarations.md) covers the objects,
morphisms, and kernels that programs use, and the [Overview](dsl-overview.md#grammar)
summarizes the grammar.

## Program declarations

A program reads as a sequence of steps. `sample` draws a random value, `let` computes
from values already bound, `observe` scores a measured value under a distribution, and
`return` gives the caller the result. `marginalize` sums a finite latent variable out
of the likelihood instead of drawing it. The steps define the model, and a separate
inference procedure ([Inference Foundations](inference-foundations.md)) estimates its
posterior.

```qvr
object X : FinSet 1
object Y : Real 1
object Z : Real 1

morphism f : Y -> Y ~ Normal
morphism g : Y * Z -> Y ~ Normal

program my_prog : X -> Y
    sample mu <- LogitNormal(mu=0.0, sigma=1.0)
    sample x <- Normal(loc=mu, scale=1.0)
    return x

program with_params(a, b) : Y * Z -> Y
    let w = a
    sample x <- f(w)
    sample y <- g(x, b)
    return y
```

A program without `observe` or `score` steps defines a Markov kernel from its input
to its output, the composite of its draws in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category)
of the Giry monad. `observe` and `score` multiply each execution by a nonnegative
weight, so a program with them defines an unnormalized measure on its executions,
which inference normalizes into the posterior. [Semantics](../semantics/programs.md)
states this precisely.

The compiled program is a
[`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram)
([Monadic Programs](programs.md) describes `rsample`, `log_joint`, and the
`observations` dict).

### Effect signatures

A program may declare the effects its body uses with `effects=[...]`, a subset of
`Sample`, `Score`, `Marginal`, and `Pure`. The compiler checks that the body uses no
other effect, so `effects=[Pure]` rejects any `sample`, `observe`, `score`, or
`marginalize` step.

```qvr
object X : FinSet 1
object Y : Real 1

program prior : X -> Y [effects=[Sample]]
    sample mu <- Normal(loc=0.0, scale=1.0)
    return mu

program deterministic : Y -> Y [effects=[Pure]]
    let doubled = 2.0 * x
    return doubled
```

[Compositional Effects](effects.md) covers effects in general.

### QIEC typed computations

Every `program` elaborates to a computation of the Quivers Indexed Effect Core (QIEC),
the typed core language of QVR ([language reference](../reference/qvr/index.md)), and a
typed `define` writes such a computation directly. Its header gives the result type and
the [effect row](../reference/qvr/effects-and-handlers.md#effect-rows), the effect
instances it may use, and `perform` requests an operation of one instance:

<!-- compile: qiec -->
```qvr
effect Tap
    ping : Int -> Int

instance tap : Tap

define echo(value : Int) : Int !{tap} =
    let answer <- perform tap.ping(value)
    return answer
```

A computation body is built from `return`, `let ... <- perform ...`, a call
`let x <- f[statics](args)` of a named computation, `handle ... with ... in`,
`with instance x : E in` (which creates an instance scoped to its body), and `case`
on an indexed value with an explicit motive (the result type as a function of the
indices) [@mcbride2002elimination]. A row such as `!{tap}` lists instance names, and an
open row `!{tap | rho lacks tap}` also admits any other instances `rho` stands for,
which must not include `tap` [@leijen2014koka]. The checker verifies both the tail and
the `lacks` constraint.

Each call is checked against the callee's signature at the call's static arguments.
The checker handles the module's call graph one strongly connected component at a
time, so mutually recursive computations need no forward declarations, and a recursive
computation must reach a `case` branch or an `if` arm that does not call again. A
scoped instance may not remain in its body's row. A program draws from another program
with `sample x <- sub(args)` (or `sample (a, b) <- sub` for a pair), which runs `sub`'s
steps in place under names of its own (`sub`'s local `z` becomes `x$z`), and calls a
computation with `let x <- f(args)`. [Computations](../reference/qvr/computations.md)
and [probabilistic programs](../reference/qvr/probabilistic-programs.md) give the full
syntax, and [Transpiler support](../transpile-support.md) lists which targets accept
which computations.

### Fitting programs that call computations

A program fitted with `trace`, stochastic variational inference (SVI), or Markov chain
Monte Carlo (MCMC) runs as a fixed sequence of compiled steps, so the compiler turns
each call of a computation into steps of the program where the callee's body allows
it. The module below uses each such form: `leaky` branches between two pure values,
`shifted` calls another pure computation, `jitter` draws a site of its own, and
`traverse` runs `jitter` once per group.

```qvr
object Group : FinSet 4

instance random : Random

define leaky(x : Real) : Real !{} =
    if x > 0.0 then
        return x
    else
        return 0.1 * x

define shifted(x : Real) : Real !{} =
    let rectified <- leaky(x)
    return rectified + 1.0

define jitter(x : Real) : Real !{random} =
    let z <- perform random.sample[Real](site("z"), Normal(x, 0.5))
    return z

program grouped : Group -> Group
    sample a <- Normal(loc=0.0, scale=1.0)
    let center <- shifted(a)
    let base = factor k : Group in center
    let effects <- traverse(base, t -> jitter(t))
    observe y : Group <- Normal(loc=effects, scale=1.0)
    return y

export grouped
```

`shifted(a)` becomes `let` steps, with `leaky`'s conditional read as
`select(a > 0.0, a, 0.1 * a)`. The `traverse` becomes one draw of the site
`effects$z` over `Group`, `jitter`'s site `z` repeated over the group and named under
the binder `effects`, as a [draw from a program](#parametric-programs) names the
callee's sites. The compiled program is a
[`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram),
and its log joint density and gradients equal those the reference machine (the
library's interpreter for QIEC) computes:

<!-- python: skip -->
```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel

program = load("grouped.qvr").morphism
result = MCMC(NUTSKernel(), num_warmup=300, num_samples=500, seed=0).run(
    program, torch.zeros(1, 1), {"y": torch.tensor([2.0, 3.0, 2.5, 3.5])}
)
result.samples["effects$z"].shape  # (chains, draws, 4)
```

A call whose steps depend on the values it receives cannot be compiled this way:
recursion, an `if` whose branches draw or score, a callee that handles an effect,
`case`, and static arguments. Such a program still runs on the reference machine
(`Program.run`), and every fitting entry point raises a
[`CompileError`](../api/dsl/compiler.md#quivers.dsl.compiler.CompileError) at the call
saying why and what to write instead
([which calls are fit](../reference/qvr/probabilistic-programs.md#which-calls-are-fit)).

### Draws {#kleisli-bind-syntax}

`sample` introduces a draw, and `<-` separates the bound name from the distribution:

<!-- compile: false -->
```qvr
sample x <- Normal(loc=0.0, scale=1.0)
```

The step binds `x` to a value drawn from the family. `observe` and `marginalize` use
the same `<-`.

### Indexed draws (plates) {#indexed-bind-plate}

A [plate](https://en.wikipedia.org/wiki/Plate_notation) repeats a draw over an index
set [@koller2009probabilistic]. `sample v : A <- F(args)` draws one value of `F` for
each element of `A`, independently given the arguments, so the joint distribution of
`v` is the product of the distributions of its elements. If `Subject` has four elements,
`sample by_subject : Subject <- Normal(loc=0.0, scale=sigma)` draws four subject
effects. They are independent given the shared scale `sigma` and become dependent once
uncertainty about `sigma` is taken into account: one unusually large effect raises the
inferred scale and so widens the plausible range of the others. To make draws
dependent, pass a shared random parameter or choose a family whose event spans the
axes.

```qvr
object Item : FinSet 1000
object Subject : FinSet 4
object Out : Real 1

program p : Out -> Out
    sample sigma <- HalfNormal(scale=1.0)
    sample duration_incr : Item <- HalfNormal(scale=1.0)
    sample by_subject : Subject <- Normal(loc=0.0, scale=sigma)
    return sigma
```

### Indexed observe

`observe r : N <- F(args)` scores observed values `r`, one per element of `N`. It
multiplies the execution's weight by $\prod_{n \in N} p_F(r_n; \theta_n)$, the product
of the family's densities at the observed values, so the log weight grows by the sum
of their log densities. A density can exceed one: a uniform density on an interval of
width `0.1` is `10`, and each observation adds `log(10)`. The observed values are passed
at run time in the [`observations` dict](programs.md#the-observations-dict).

```qvr
object RespCloze : FinSet 10
object Out : Real 1

program p : Out -> Out
    sample intercept_cloze <- Beta(concentration1=1.0, concentration0=1.0)
    observe cloze_resp : RespCloze <- Bernoulli(probs=intercept_cloze)
    return intercept_cloze
```

Family arguments may index plate variables, as `theta[c]` below.

### Scoped marginalize

`marginalize c : K <- F(args)` followed by an indented block sums the finite latent `c`
out of the weight of the steps in the block: for each value of `c` in `K` it computes
the block's likelihood, weights it by `F`'s probability of that value, and adds the
results, in log space by [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp)
[@pyroEnumeration; @stanFiniteMixtures]. The annotation `K` names the latent's values.
Without an `over` option there is one latent for the whole block, shared by every
observation in it:

```qvr
object Row : FinSet 6
object Component : FinSet 2

program mixture1 : Row -> Row
    sample probs <- Dirichlet(concentration=2.0) [over=Component]
    sample mean : Component <- Normal(loc=0.0, scale=3.0)
    marginalize z : Component <- Categorical(probs=probs)
        observe y : Row <- Normal(loc=mean[z], scale=1.0)
    return mean

export mixture1
```

This adds $\log \sum_k \pi_k \prod_{n} \mathcal N(y_n; \mu_k, 1)$ to the log joint
density. One latent per row, or per group of rows, is the grouped form,
`[over=G]` on the block with `[via=idx]` on each observation, described under
[grouped marginalization](programs-hierarchical.md#grouped-marginalization-fibred-discrete-latents).

### Indexed gather in `let`

`arr[idx]` in a `let` reads the entries of an array at the positions an integer array
gives. With `by_verb = [0.2, -0.1, 0.5]` and `verb_of_item = [2, 0, 2]`,
`by_verb[verb_of_item]` is `[0.5, 0.2, 0.5]`: items 0 and 2 share the realized effect of
verb 2. Drawing an effect per item instead would be a different model, and the index
array is what keeps the sharing. Formally, for values $v : A \to B$ of a plate over $A$
and an index map $\iota : N \to A$, the gather is the composite
$v \circ \iota : N \to B$.

```qvr
object Verb : FinSet 3
object Item : FinSet 5
object Out : Real 1

program p : Out -> Out
    sample sigma <- HalfNormal(scale=1.0)
    sample by_verb : Verb <- Normal(loc=0.0, scale=sigma)
    let intercept_for_item = by_verb[verb_of_item]
    observe y : Item <- Normal(loc=intercept_for_item, scale=1.0)
    return sigma
```

Here `verb_of_item` is a [data input](inference-foundations.md#host-data-per-row-covariates-and-index-arrays),
an index array given at run time.

A subscript with several indices reads them together. With index vectors `row_of` and
`col_of` over the same rows, `table[row_of, col_of]` is one entry per row,
`table[row_of[n], col_of[n]]` at row `n`, as NumPy's advanced indexing and a loop over
rows in Stan read it. The result has the index vectors' shape followed by the axes of
`table` the subscript leaves. An `Int` index, a literal or a factor binder, stands for
the same value at every row, so `table[row_of, 1]` reads the second column at each row.
Index vectors of two different shapes in one subscript are refused, and a single index
vector, `table[row_of]`, reads whole rows. A matrix indexed by (row, column) pairs, as
in matrix factorization or an item-response table, is thus one `let`:

```qvr
object User : FinSet 4
object Item : FinSet 5
object Rating : FinSet 12
object Out : Real 1

program p : Out -> Out
    sample user : User <- Normal(loc=0.0, scale=1.0)
    sample item : Item <- Normal(loc=0.0, scale=1.0)
    let affinity = factor u : User, i : Item in user[u] * item[i]
    let mu = affinity[user_of, item_of]
    observe rating : Rating <- Normal(loc=mu, scale=1.0)
    return mu
```

### Parametric programs

A program whose parameter list declares typed parameters is a template, one program
for each value of its parameters. The kinds are:

| Kind | Stands for |
|---|---|
| `FinSet`, `Space`, `Object` | an object, such as the index set of a plate |
| `Real`, `Nat` | a number, such as a hyperparameter |
| `Mor[A, B]` | a morphism `A -> B`, passed by name |

The compiler does not build a `MonadicProgram` for a template by itself. It inlines the
template at each call, `sample v <- template(arg1, arg2, ...)`, substituting the
arguments for the parameters and renaming the template's internal names with the prefix
`v$` and its returned value to `v`, as α-renaming in the
[lambda calculus](https://en.wikipedia.org/wiki/Lambda_calculus) does. Each call thus
adds its own latent variables to the caller's joint, and two calls never share one by
accident.

The substitution reaches every position a name can occupy in a `let` or `score`
expression and in the arguments of a computation call: the array and the indices of
`arr[idx]`, call arguments, method receivers, list and tuple items, lambda and `factor`
bodies, and an object parameter used as a `factor` binder's index set. It avoids
capture: a lambda parameter or `factor` binder shadows a parameter or local with the
same name and is renamed if a substituted name would otherwise be captured.

```qvr
object Subject : FinSet 4
object Out : Real 1

# One HalfNormal scale and a Normal(0, sigma) plate over the grouping
# object G, with the half-normal scale as a parameter.
program random_intercepts (G : FinSet, prior_scale : Real) : G -> Real 1
    sample sigma <- HalfNormal(scale=prior_scale)
    sample v : G <- Normal(loc=0.0, scale=sigma)
    return v

program uses : Out -> Out
    sample a <- random_intercepts(Subject, 1.0)
    return a
```

[Hierarchical Programs](programs-hierarchical.md) works through crossed random
intercepts and grouped marginalization.

### Posterior blocks

A program can compute a derived quantity from the latent variables of another program,
like Stan's generated quantities [@carpenter2017stan]. `over=` in its option block names
the model whose latents it reads, and its parameters name those latents. Its body must
be pure (`effects=[Pure]`). Given data, its output is distributed as the pushforward of
the model's posterior along it: draw the latents from the posterior given the data, then
apply the program. In this one use, `over=` names a program rather than event axes.

```qvr
object Item : FinSet 2
object Out : Real 1

program scored : Item -> Out
    sample logit <- Normal(loc=0.0, scale=1.0)
    return logit

program success_prob(logit : Real) : Item -> Out [effects=[Pure], over=scored]
    let prob = sigmoid(logit)
    return prob
```

The parameter `logit` names the model's latent and gives its type, and at run time each
posterior draw of the model's [trace](inference-foundations.md#trace-and-sample-sites)
gives its value. An untyped parameter is read as an integer by the checked core, so a
real-valued latent needs its `: Real` annotation.

## Axis-role clause: `over` and `iid_over` {#axis-role-clause-over-and-iid_over}

A draw's axes are of two kinds [@pyroTensorShapes]. *Event* axes belong to one outcome
of the family, and *batch* axes index independent outcomes. A probability vector over
three topics is one Dirichlet outcome, so `Topic` is an event axis, and ten documents
each with such a vector make `Doc` a batch axis: ten vectors, each summing to one,
independent given their parameters. `over=Topic` names the event axis, and the plate
annotation `: Doc` or `iid_over=Doc` names the batch axis. A family's number of event
dimensions ([Continuous Families](continuous-families.md#event-rank-and-axis-role-syntax))
fixes how many axes one outcome has. It is 0 for scalar families such as Normal, Beta,
and Gamma, 1 for vector families such as
[`MultivariateNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMultivariateNormal),
[`Dirichlet`](../api/continuous/families.md#quivers.continuous.families.ConditionalDirichlet),
`GP` ([`ConditionalGaussianProcess`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess)),
and [`Horseshoe`](../api/continuous/families.md#quivers.continuous.families.ConditionalHorseshoe),
and 2 for matrix families such as
[`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal),
[`Wishart`](../api/continuous/families.md#quivers.continuous.families.ConditionalWishart),
and [`InverseWishart`](../api/continuous/families.md#quivers.continuous.families.ConditionalInverseWishart).

The clause goes in the option block of every form that carries a family. On a
`morphism` it precedes the `~` initializer, and on a `sample`, `observe`, or
`marginalize` step it follows the family:

```text
morphism NAME : A -> B [..., over=<axes>, iid_over=<axes>] ~ Family(args)
sample NAME : Plate <- Family(args) [over=<axes>, iid_over=<axes>]
```

`over=<axes>` names the event axes, one name bare (`over=A`) and several in a list
(`over=[A, B]`). The order of the axes follows the family's event axes, so for
`MatrixNormal` the first is the row axis and the second the column axis. On a
`morphism` the number of axes must equal the family's number of event dimensions.

On a `sample` or `observe` step, `over` may list more axes than that. The leading
extra axes then index independent outcomes, and the trailing ones are the event axes:
`sample w : A <- Normal(loc=0.0, scale=1.0) [over=[B, C]]` draws an `(A, B, C)` array of
independent normals, and `sample p : A <- Dirichlet(concentration=1.0) [over=[B, K]]`
draws one `K`-simplex point per element of `A` and `B`, an `(A, B, K)` array. The site
holds every listed axis in the order written, after the plate, so a body reads it as
`w[a, b, c]`, and the extra axes are batch axes as if listed in `iid_over`. A matrix
family takes no such extra axes.

`iid_over=<axes>` names the batch axes and is optional: every axis not in `over` is a
batch axis anyway, with independent draws along it. On a sample step, the plate
annotation `: A` beside an `over` clause is the batch plate, so
`sample rows : Doc <- Dirichlet(concentration=1.0) [over=Topic]` draws one point of
the `Topic` simplex per document. Without `over`, the annotation on a vector family
names the family's own axis when nothing else fixes it
(`sample pi : K <- Dirichlet(concentration=1.0)` draws one `K`-simplex point) and the
batch plate when the arguments fix the event
(`sample pc : Item <- Dirichlet(concentration=[1.0, 2.0, 3.0])` draws one point of the
three-simplex per item).

A vector family's parameter is named like every other parameter:
`Dirichlet(concentration=[1.0, 2.0, 3.0])` gives an explicit vector, and
`Dirichlet(concentration=1.0)` a symmetric concentration at the dimension the step's
annotation, its `over` axis, or the program's codomain fixes. A draw that fixes no
simplex dimension is refused at compile time.

**Axis names.** Names resolve against the named factors of the surrounding morphism's
domain and codomain, or against the type annotation of a step. `dom` and `cod` stand
for a side that is a single object, and the factors of a product must be named.

**Joint versus independent axes.** The clause keeps apart a joint distribution on a
product of axes, with possible correlation across them, and independent draws along
an axis. It also keeps apart a multivariate normal over all $|A| \cdot |B|$ entries
with a dense covariance and a [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution),
whose covariance is the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product)
$V \otimes U$ of a column and a row covariance. The compiler never substitutes a family
of a different number of event dimensions, and renaming a morphism's type makes stale
axis names fail at type checking.

**Errors.** Two diagnostics come from this clause:

- *"axis count does not match family event_rank"*: on a `morphism`, `over` lists a
  number of axes different from the family's number of event dimensions, as happens
  when a flat `MultivariateNormal` (one event dimension) is replaced by a
  `MatrixNormal` (two).
- *"unknown axis name(s) `A` in `over`"*: a name in `over` is not a factor of the
  domain or codomain. Rename it, or use `dom` or `cod` for a side that is one object.

```qvr
object Dk : FinSet 5
object N : FinSet 7
object D : FinSet 3
object K : FinSet 2
object Out : Real 1

# A matrix normal prior on a morphism's 3 x 2 tensor
morphism W : D -> K [role=latent, over=[dom, cod]] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0)

program p : Out -> Out
    # A five-dimensional multivariate normal, then one per observation row
    sample mu <- MultivariateNormal(loc=m0, scale_tril=L) [over=Dk]
    observe y : N <- MultivariateNormal(loc=mu, scale_tril=L) [over=Dk]
    return mu
```

Here `m0` and `L` are data inputs.

## Let expressions (arithmetic, primitives, and collections)

In a `program` block or a `define` computation, `let` binds the value of an
expression: arithmetic with the usual precedence, unary negation, the primitives
below, and finite collection operations. Each primitive is a deterministic function, a
[Dirac kernel](https://en.wikipedia.org/wiki/Dirac_measure) in the program's
semantics.

```qvr
object Row : FinSet 3
object Out : Real 1

program p : Out -> Out
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    sample z_raw <- Normal(loc=0.0, scale=1.0)
    sample lapse <- Beta(concentration1=1.0, concentration0=9.0)
    # arithmetic: +, -, *, /
    let eta = mu + sigma * z_raw
    let p_raw = sigmoid(eta)
    let adjusted = (1.0 - lapse) * p_raw + 0.5 * lapse
    let negated = -eta
    # primitives
    let positive = softplus(eta)
    let magnitude = abs(p_raw - 0.5)
    observe y : Row <- Bernoulli(probs=adjusted)
    return mu
```

### Checked primitive reference

The primitives below are those of the checked core, and the compiler resolves each
operator and call by the types of its arguments. A reduction collapses the last axis of
its argument, and a rowwise operation acts along that axis and keeps the shape. For a
vector over a plate, as in `sum(theta)` over `theta : School`, the last axis is the only
one and the result is the plate's total. For a plate of event vectors, as in
`sum(z_row * w_row)` over a `Resp`-plated value with an `[over=LatentDim]` event axis,
the result keeps the plate and holds one inner product per row. For a table a `factor`
builds over `Row` and `Unit`, the result holds one number per `Row`. `normalize` divides
each row by its sum. Every execution path and transpile target reads the primitives in
this way, and a [contraction declaration](dsl-contractions.md) reduces over a named
axis.

**Conversions and weights**: `real`, `int`, `weight`, `weight_value`.

**Arithmetic and comparisons**: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`,
`>=`, `&&`, `||`, and the call forms `pow`, `abs`, `min`, and `max`. The two-argument
`min(a, b)` and `max(a, b)` are pointwise, unlike the one-argument reductions below: two
tensors compare entry by entry, a number broadcasts against a tensor, and two `Int`
operands give an `Int`, so `ys[max(t - 1, 0)]` clamps an index expression into range.

**Conditional values and Booleans as numbers**: `select(c, a, b)` takes `a` where the
`Bool` condition `c` holds and `b` elsewhere, entry by entry. A number among the three
broadcasts against the tensors, and the tensors share one shape, so
`select(v > 0.0, v, 0.1 * v)` is a leaky rectifier of a vector. Both branches have one
type, `Real` or `Int` (write `0.0`, not `0`, beside a real branch). `real(c)` and
`int(c)` read a `Bool` as one where it holds and zero elsewhere, so a mask needs no
integer clamp:

```qvr
object Pos : FinSet 4
object Out : Real 1

program p : Out -> Out
    sample margin <- Normal(loc=0.0, scale=1.0)
    let causal = factor i : Pos, j : Pos in real(j <= i)
    let hinge = select(margin < 1.0, 1.0 - margin, 0.0)
    return hinge
```

`select` evaluates both branches and then keeps one entry of either, as `torch.where`
does, so it does not guard a computation. A branch that is `NaN` or infinite where it
is not taken is still computed, and its gradient there is zero times an infinite
derivative, which is `NaN`. Where the untaken branch is finite its gradient is zero.
When the condition is what keeps a branch in range, write the branch so that it is
finite everywhere, `log(max(v, 1e-12))` rather than `log(v)`.

**Pointwise functions**: `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`,
`rsqrt`, `square`, `abs`, `sign`, `reciprocal`, `sin`, `cos`, `tan`, `asin`, `acos`,
`atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `floor`, `ceil`, `round`, `trunc`,
`erf`, `erfc`, `erfinv`, `lgamma`, `digamma`.

**Activations**: `relu`, `relu6`, `elu`, `selu`, `gelu`, `silu`, `mish`, `softplus`,
`logsigmoid`, `softsign`, `sigmoid`, and `tanh`.

`round` rounds a tie to even. A transpile target computes each pointwise primitive with
the same meaning, through its own function where it has one and otherwise through a
definition in terms of others (`gelu(x)` as `0.5 * x * (1 + erf(x / sqrt(2)))`), and a
primitive that a target cannot express is refused by name at transpilation
([Transpiler support](../transpile-support.md)).

**Reductions over the last axis**: `sum`, `mean`, `min`, `max`, `prod`, `logsumexp`.

**Operations along the last axis**: `softmax`, `log_softmax`, `cumsum`, `sort`,
`normalize`.

**Joining and cutting the last axis**: `concat(a, b, ...)` joins two or more tensors of
one element type along their last axis. Their leading axes must agree and their last
extents add, so `concat(h, c)` of two `Real 4` vectors has 8 entries and `concat` of two
`(2, 3)` tables is `(2, 6)`. `slice(v, start, length)` keeps the `length` entries of
the last axis from `start`, counted from zero. Both bounds are nonnegative integer
literals, since the checked type states the slice's extent, and a position computed at
run time is read with a gather, `v[i]`. A number joins as the one-entry vector `[x]`. A
slice past the end of the axis, a zero length, a number where a tensor belongs, and
tensors of different element types or leading axes are refused. This is how a scan
carries a state of several parts as one vector, such as the hidden and cell states of an
LSTM:

<!-- compile: false -->
```qvr
let h_prev = slice(state, 0, 4)
let c_prev = slice(state, 4, 4)
let c_new = f * c_prev + i * g
let h_new = o * tanh(c_new)
let state_new = concat(h_new, c_new)
```

The compiled runtime computes these with `torch.cat` and a slice of the last axis.
Within one run of the compiled runtime, the batch axis of the input is never among a
`let`'s axes, so `concat` and `slice` never read it.

**Products and normalizations**: `matmul(a, b)` sums the last axis of `a` against the
first axis of `b`, which is a vector or a matrix with that axis's extent. A matrix times
a vector is a vector, a vector times a matrix is a vector, a vector times a vector is a
number, and an `a` of higher rank is multiplied row by row, so `matmul(X, W)` with
`X : (T, D)` and `W : (D, H)` is `(T, H)`. The compiler checks that the two extents
agree. `layer_norm(v)` and `rms_norm(v)` normalize each row along the last axis and keep
the shape: `layer_norm(v)` is
$(v - \bar v) / \sqrt{\overline{(v - \bar v)^2} + \epsilon}$, with the biased variance as
`torch.nn.functional.layer_norm` computes it [@ba2016layer], and `rms_norm(v)` is
$v / \sqrt{\overline{v^2} + \epsilon}$, where a bar is the mean over the last axis. An
optional second argument, a positive real literal, sets $\epsilon$, which is `0.00001`
for `layer_norm` and `0.000001` for `rms_norm` when omitted. Neither has a learned gain
or bias, which are multiplied and added in ordinary `let` steps:

<!-- compile: false -->
```qvr
let h = tanh(matmul(W_in, x_t) + matmul(W_rec, h_prev) + b)
let normed = gain * layer_norm(h) + shift
```

**Finite collections**: `map`, `fold`, `length`, and `logsumexp_over` are pure.
`traverse` calls a named computation once per value and appears on the right of `<-`.
The [collection reference](../reference/qvr/collection-expressions.md) gives their
typing and shape rules.

The compiled runtime also accepts operations that the checked core lacks:
`alpha_dropout`, `amax`, `amin`, `argmax`, `argmin`, `celu`, `clamp`, `cummax`,
`cummin`, `cumprod`, `dropout`, `flip`, `glu`, `hardshrink`, `hardsigmoid`,
`hardswish`, `hardtanh`, `leaky_relu`, `neg`, `norm`, `prelu`, `rrelu`, `softmin`,
`softshrink`, `std`, `swish`, `tanhshrink`, `threshold`, and `var`. A program that uses
one runs in PyTorch, but `qvr check` reports it, the reference machine cannot run it,
and no transpile target accepts it. `qvr check --target TARGET` lists what a target
refuses. Calls resolve to these primitives before module-level declarations, and the
compiler checks the number of arguments of a call to a declared callable.

### Factor expressions: assembling indexed tensors

A `factor` expression is an array comprehension: it evaluates its body at every tuple of
indices from finite sets and collects the results into a tensor
[@vasilache2018tensor]. Indexing reads one entry of a table, and `factor` builds the
whole table. For two verbs and three classes, the body runs at the six
`(verb, class)` pairs and gives a table with those two leading axes.

#### Uniform form

`factor v_1 : I_1, v_2 : I_2, ..., v_n : I_n in <body>` is the tensor of shape
`(|I_1|, ..., |I_n|, *body_shape)` whose entry at `(i_1, ..., i_n)` is `<body>`
evaluated with $v_k = i_k$. The binders are integers visible only inside the body.

```qvr
object Verb : FinSet 40
object Class : FinSet 4
object Out : Real 1

program p : Out -> Out
    sample coef : Verb <- Normal(loc=0.0, scale=1.0) [over=Class]
    sample weight <- Normal(loc=0.0, scale=1.0) [over=Class]
    # One score per verb and class: shape (40, 4)
    let cell = factor v : Verb, cls : Class in coef[v, cls] * weight[cls]
    return weight
```

#### Pattern-match form (single-axis)

When each entry of a one-axis `factor` is a different expression, the case form lists
them side by side:

```qvr
object Class : FinSet 4
object Out : Real 1

program p : Out -> Out
    sample prob_dur <- Beta(concentration1=1.0, concentration0=1.0)
    sample prob_telic_nodur <- Beta(concentration1=1.0, concentration0=1.0)
    sample prob_telic_dur <- Beta(concentration1=1.0, concentration0=1.0)
    let class_probs = factor cls : Class in {
        0 -> (1.0 - prob_dur) * (1.0 - prob_telic_nodur),
        1 -> (1.0 - prob_dur) * prob_telic_nodur,
        2 -> prob_dur * (1.0 - prob_telic_dur),
        3 -> prob_dur * prob_telic_dur,
    }
    return prob_dur
```

The labels must cover `0, ..., |Index| - 1` exactly, and the compiler rejects gaps,
duplicates, and labels out of range. Here each entry of a probability vector over four
classes is a different product of scalar latent variables.

## Inline distributions

`sample` and `observe` steps write their family inline, with literal and bound
arguments in any mix:

```qvr
object Out : Real 1

program p : Out -> Out
    # literal arguments
    sample x <- Normal(loc=0.0, scale=1.0)
    sample prob <- Beta(concentration1=2.0, concentration0=5.0)
    # bound arguments
    sample sigma <- HalfNormal(scale=1.0)
    sample y <- Normal(loc=x, scale=sigma)
    sample b <- Bernoulli(probs=prob)
    # a mix, and a family with differently named parameters
    sample z <- Normal(loc=-1.5, scale=0.3)
    sample r <- TruncatedNormal(mu=x, sigma=sigma, low=0.0, high=1.0)
    return y
```

Some of the registered families ([Continuous Families](continuous-families.md#family-registry)
lists them all):

| Family | Parameters | Codomain |
|---|---|---|
| `Normal` | `loc`, `scale` | `Real N` |
| `LogitNormal` | `mu`, `sigma` | `Real N {low=0, high=1}` |
| `Uniform` | `low`, `high` | `Real N {low=..., high=...}` |
| `Bernoulli` | `probs` or `logits` | `FinSet 2` |
| `Beta` | `concentration1`, `concentration0` | `Real N {low=0, high=1}` |
| `Exponential` | `rate` | `Real N {low=0}` |
| `HalfCauchy` | `scale` | `Real N {low=0}` |
| `HalfNormal` | `scale` | `Real N {low=0}` |
| `LogNormal` | `loc`, `scale` | `Real N {low=0}` |
| `Gamma` | `concentration`, `rate` | `Real N {low=0}` |
| `Dirichlet` | `concentration` | `Simplex K` |
| `TruncatedNormal` | `mu`, `sigma`, `low`, `high` | `Real N {low=..., high=...}` |
| `MultivariateNormal` | `loc`, `scale_tril` | `Real N` (one event dimension) |
| `MatrixNormal` | `loc`, `row_covariance`, `col_covariance` | an `R x C` matrix (two event dimensions) |
| `OrderedProbit`, `OrderedLogistic` | `eta` (`predictor` for `OrderedLogistic`), `cutpoints` | ordered categories |
| `LKJCholesky` | `concentration` | `CholeskyFactor K` |
| `Wishart`, `InverseWishart` | `df`, `scale_tril` | `Covariance K` |
| `Horseshoe` | `scale` | `Real N` |

`LogitNormal` and `TruncatedNormal` name their location and scale `mu` and `sigma`,
where other families use `loc` and `scale`. When every argument is a literal the family
is fixed when the program is built, and otherwise it is built from the current values
of the bound arguments at each run.

### Data vectors and matrices as family parameters

A name that nothing in the program binds is a
[data input](inference-foundations.md#host-data-per-row-covariates-and-index-arrays),
given when the program runs. When a data input fills a vector or matrix parameter of a
draw without batch axes, it is one tensor over the parameter's own axes, and the family
ties those axes to its outcome's: a `MultivariateNormal`'s `loc` is a vector over the
event axis and its `scale_tril` a square matrix over it, and a `Wishart`'s `scale_tril`
is a matrix over the outcome's two axes. The draw sizes each tied axis from what it
already knows: the event axes an `over` clause names, or another argument of known size,
such as a sampled mean. An axis that nothing in the draw sizes has one extent, shared by
every input tied to it and fixed by a later use, such as the `over` event axis of a
draw that reads the value, or the `Real N` codomain of a program that returns it.

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

The program then runs with the data at those sizes, on the compiled runtime and on the
reference machine:

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

A data input of another size is refused on both paths, with a message naming the input
and the size the program reads it at. One input read at two sizes, or a tied argument
whose size differs from the event axis it spans, is refused at the draw. A data input
filling a vector parameter of a draw whose plate names event axes is instead one value
that the plate broadcasts, as in `Dirichlet(concentration=alpha) [over=K]`. A matrix
parameter (a Cholesky factor, a covariance, or a matrix normal's mean) is never a
broadcast scalar and stays a matrix. The compiled runtime sizes every parameter when
it builds the program, so it refuses a multivariate draw whose event size nothing in the
program fixes, and its message lists the ways to fix it.

### Ordinal, inverse-Wishart, and matrix-normal draws

An ordinal response under a probit link [@mccullagh1980regression], a covariance drawn
from an inverse Wishart, and a matrix drawn from a matrix normal are written inline like
their logistic and Wishart counterparts:

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

`OrderedProbit` gives category $k$ probability $\Phi(c_k - \eta) - \Phi(c_{k-1} - \eta)$,
with $\Phi$ the standard normal distribution function and $c$ the cutpoints, computed
in log space so a category far from the predictor keeps its probability.
`InverseWishart(df, scale_tril=L)` is the distribution of the inverse of a
`Wishart(df, scale_tril=L)` draw, which is the
[inverse Wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution)
$\mathcal W^{-1}(\mathrm{df}, \Psi)$ with $\Psi = (L L^\top)^{-1}$.
`MatrixNormal(loc, U, V)` draws an $R \times C$ matrix whose row-major entries have
covariance $U \otimes V$, the
[matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) with row
covariance $U$ and column covariance $V$. An inverse-Wishart covariance is a
positive-definite latent site, which MCMC and SVI move through the unconstrained
coordinates of its Cholesky factor.

### A Cholesky factor built in the program

The usual hierarchical covariance prior draws an LKJ correlation factor
[@lewandowski2009lkj], scales its rows by standard deviations, and passes the product
to a multivariate normal as its Cholesky factor:

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

Both execution paths score it. Which transpile targets accept a Cholesky factor known
only at run time is listed in [Transpiler support](../transpile-support.md).

### Alternative parameterizations

Several families accept more than one complete set of parameters, and the keywords a
draw writes select one. A draw that names two, such as `Bernoulli(probs=p, logits=l)`,
is refused.

| Family | Keyword sets |
|---|---|
| `MultivariateNormal` | `(loc, scale_tril)`, `(loc, covariance_matrix)`, `(loc, precision_matrix)` |
| `Wishart` | `(df, scale_tril)`, `(df, covariance_matrix)`, `(df, precision_matrix)` |
| `Bernoulli`, `ContinuousBernoulli`, `Geometric`, `Categorical` | `(probs)`, `(logits)` |
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

The compiled runtime and the reference machine score each parameterization as
`torch.distributions` does with the same keywords. Logits are scored in log space, with
$\log p = -\mathrm{softplus}(-l)$ and $\log(1 - p) = -\mathrm{softplus}(l)$, so an
observation stays finite where the probability itself rounds to zero or one, which in
single precision happens near $|l| = 30$. A `ContinuousBernoulli` given by its logits
is scored with its normalizer written in the logit, $l / \tanh(l / 2)$, for the same
reason. [Transpiler support](../transpile-support.md) lists which targets accept which
parameterization.

A mixture weight that is the logistic function of a linear predictor is best given on
the same scale. At a `marginalize` head, a `Bernoulli` given by its logits weighs the
values 0 and 1 by $\log \sigma(-l)$ and $\log \sigma(l)$, computed in log space, where
a probability written `sigmoid(eta)` (or `1.0 - sigmoid(eta)`) would be rounded in
single precision first and lose its digits once $|\eta|$ passes about 10. A
zero-inflated Poisson regression [@lambert1992zip] thus gives its gate the linear
predictor itself:

```qvr
object Row : FinSet 4
object One : FinSet 1

program zip : Row -> One
    sample a <- Normal(loc=0.0, scale=5.0)
    sample b <- Normal(loc=0.0, scale=5.0)
    # The gate's log-odds: no sigmoid, and no `1.0 - ...` complement.
    let eta = a + b * x
    marginalize z : Row <- Bernoulli(logits=eta)
        let gated_rate = real(z) * 3.0
        observe y : Row <- Poisson(rate=gated_rate)
    return a

export zip
```

With the covariate `x` a data input over `Row`, the log-odds vary by row, each row has
its own gate, and the block adds
$\sum_n \log\bigl[(1 - \sigma(\eta_n))\, \mathbb 1[y_n = 0] + \sigma(\eta_n)\, \mathrm{Poisson}(y_n; 3)\bigr]$
to the log joint density. With a single scalar log-odds instead, the block has one gate
shared by all rows. A gate uses `Bernoulli`: `ContinuousBernoulli` and
`RelaxedBernoulli` have densities on $[0, 1]$ rather than two atoms, so a sum over the
values 0 and 1 is not their marginal.

A conditional distribution whose parameters a network learns from the input is a
[`morphism f : A -> B ~ Family`](dsl-declarations.md#kernel) declaration instead.

## Examples

### Simple discrete model

```qvr
object X : FinSet 3
object Y : FinSet 4
morphism f : X -> Y [role=latent]
morphism g : Y -> Y [role=latent]

define fg = f >> g

export fg
```

### Continuous conditional model

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

```qvr
object Data : FinSet 1
object Y : Real 1

program regression : Data -> Y
    sample theta <- LogitNormal(mu=0.0, sigma=1.0)
    sample y <- Normal(loc=theta, scale=0.5)
    observe y_obs <- Normal(loc=y, scale=0.1)
    return y

export regression
```

### Init recipe end-to-end

```qvr
object In : FinSet 8
object Out : FinSet 4

# Default: weights initialized from a small-scale randn.
morphism W_default : In -> Out [role=latent, scale=0.1]

# init=auto: the initialization recipe of the module's Algebra,
# applied after the sigmoid under product_fuzzy (the default).
morphism W_auto : In -> Out [role=latent, init=auto]

export W_auto
```

The [Examples Gallery](../examples/index.md) has complete models, and
[Semantics](../semantics/index.md) gives what `.qvr` programs denote.
