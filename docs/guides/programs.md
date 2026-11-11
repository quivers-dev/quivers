# Monadic Programs

## What is a MonadicProgram?

A probabilistic program describes a procedure for generating data. For
instance, draw a latent mean `mu`, draw a response around that mean, and
return the response. The same description can score actual responses by
using their observed values instead of drawing them. Inference is the
additional task of finding plausible latent values given those observations.
The [inference guides](inference-foundations.md) explain how SVI and MCMC
use the program’s scores to fit or sample the latent mean.

A
[`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram)
is a probabilistic program specified as a sequence of sampling, observation,
and deterministic steps. It defines a
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
from a domain to a codomain via [monadic
composition](https://ncatlab.org/nlab/show/Kleisli+category)
(Kleisli bind).

The program syntax mirrors probabilistic programming languages
(Pyro, NumPyro, Stan):

<!-- compile: false -->
```qvr
program name : domain -> codomain
    sample x_1 <- morphism_1
    sample x_2 <- morphism_2(x_1)
    let y = x_1 + x_2
    observe z <- morphism_3(y)
    return y
```

Each `sample ... <- ...` step draws from a conditional distribution
and binds the result. The `observe` keyword conditions the program on
an external observation.

## Program structure

A program is an
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
that, when called, executes forward ancestral sampling:

```python
from quivers.continuous import Draw, MonadicProgram
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
import torch

Unit = FinSet(name="Unit", cardinality=1)
R1 = Euclidean(name="R1", dim=1)

prior      = ConditionalNormal(Unit, R1)  # learned, input-independent parameters
likelihood = ConditionalNormal(R1, R1)    # learned parameters conditional on x

program = MonadicProgram(
    Unit,
    R1,
    steps=[
        Draw(names=("x",), morphism=prior),  # x <- prior
        Draw(names=("y",), morphism=likelihood, args=("x",)),  # y <- likelihood(x)
    ],
    return_vars=("y",),
)

# Forward pass: sampling
samples = program.rsample(
    torch.zeros(4, dtype=torch.long), sample_shape=torch.Size([100])
)  # shape (100, 4, 1)

# Log joint: sum_i log p(z_i | pa(z_i)) given every bound variable,
# one joint per run: the four input rows are four independent runs.
x_val = torch.randn(4, 1)
y_val = torch.randn(4, 1)
log_joint = program.log_joint(
    torch.zeros(4, dtype=torch.long),
    {"x": x_val, "y": y_val},
)  # shape (4,); log_joint.sum() is the joint of the four runs
```

## Bind steps

A *binding* gives a value a name for subsequent steps. [A stochastic bind
first draws the value, then makes it available](https://pyro.ai/examples/intro_long.html#Background:-the-pyro.sample-primitive) [@pyroIntroduction]. Thus [`sample y <- f(x)`](../reference/qvr/probabilistic-programs.md#step-reference)
uses the particular value of `x` drawn earlier in this execution. Repeating
the program produces a new set of draws. The monadic account expresses
this sequencing without treating random computations as ordinary functions
returning fixed values.

A bind step `sample x <- f` or `sample x <- f(y, z)` samples from a morphism,
optionally conditioned on previous variables.

Single bind:

<!-- compile: false -->
```qvr
sample x <- prior_f
```

Conditioned bind:

<!-- compile: false -->
```qvr
sample y <- likelihood_f(x)
```

Destructuring tuple bind (stacked along feature dimension):

<!-- compile: false -->
```qvr
sample (x, y) <- joint_f(z, w)
```

The variable names on the left side are bound in the environment.
An indexed bind `sample v : A <- F(args)` declares `v` as an
$A$-indexed plate of independent draws.

## Let steps

Deterministic binding:

<!-- compile: false -->
```qvr
let x = y + z
let weight = 0.5
```

Supports literals, variable references, arithmetic, and the
let-expression primitives; see
[DSL Programs and Let-Expressions](dsl-programs-and-lets.md#let-expressions-arithmetic-primitives-and-collections)
for the full primitive list.

## Observe keyword

An observation says: "use this measured value here, and score how well
the current latent values predict it." If `mu` changes, the observed
response remains fixed but its likelihood changes. SVI uses the resulting
score to update a guide; MCMC uses it when proposing and accepting changes
to `mu`.

Condition the program on an observation:

<!-- compile: false -->
```qvr
observe y <- likelihood(x)
```

This marks `y` as conditioned. During inference, observations
clamp these variables to external values. An indexed-observe
`observe r : N <- F(args)` accumulates a batched likelihood over
the index set `N`, with the response buffer supplied via the
runtime `observations` dict.

## Return statement

Specify the program output. Single or tuple:

<!-- compile: false -->
```qvr
return x
```

<!-- compile: false -->
```qvr
return (x, y, z)
```

The return value's shape determines the codomain. Tuples are
bare-positional; the resulting product space's components are
ordered by tuple position.

## Domains and codomains

Domains can be:

- A single [`FinSet`](../api/core/objects.md) or
  [`ContinuousSpace`](../api/continuous/spaces.md).
- A product of sets / spaces: `X * Y * Z`.
- Named parameters: the domain is the product, but variables can
  refer to sub-components.

Codomains are determined by the return statement shape.

## rsample and log_joint

Two key operations on a compiled program:

### rsample(x, sample_shape=(), observations=None)

Generate samples by executing the program:

```python
x = torch.zeros(5, dtype=torch.long)
samples = program.rsample(x, sample_shape=torch.Size([1000]))
# shape: (1000, 5, codomain_dim)
```

Sequential ancestral sampling: each draw step samples, previous
draws are available to subsequent steps. `observations` is an
optional `dict[str, torch.Tensor]` clamping observed sites to
runtime data.

The input's rows are independent runs of the whole program: an input
of `B` rows draws every site `B` times, once per run, and a plated
site's rows sit after the run axis. The rows a program observes come
from its data and its plates, not from the input's rows, so a program
whose observations live in a plate is run on one row, such as
`torch.zeros(1, 1)`.

### log_joint(x, intermediates)

A joint score combines all the modeled choices in one execution. For
`mu ~ Normal(0,1)` and `y ~ Normal(mu,1)`, it is
$\log p(\mu,y)=\log p(\mu)+\log p(y\mid\mu)$. The first term
is the prior score; the second is the likelihood. Supplying a candidate
`mu` and an observed `y` lets inference compare candidates. The
normalizing constant of the posterior is not required for this comparison.

Compute $\log p(z_1, \ldots, z_k \mid x) = \sum_i \log p(z_i \mid
\mathrm{pa}(z_i))$ given all bound-variable values:

<!-- python: skip -->
```python
x = torch.randn(5)
intermediates = {"z": z_value, "y": y_value}  # every bound variable

log_pjoint = program.log_joint(x, intermediates)
```

`log_joint` returns one joint per run of the input, shape `(B,)` for
`B` input rows: each entry is the sum of that run's draw / plate-draw /
observe / score terms, and `log_joint(x, values).sum()` is the joint of
all the runs. A value in `intermediates` is either one per run, along a
leading axis of length `B`, or one that every run shares. It is used
inside
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO)
after the guide samples latents.

### The `observations` dict

Indexed-observe steps (`observe r : N <- F(args)`) read their
response buffers from a runtime
`observations: dict[str, torch.Tensor]`, keyed by the
observed-variable name. The dict is passed as the `observations`
kwarg to
[`MonadicProgram.rsample`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.rsample)
and as the final positional argument to
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) /
[`SVI.step`](../api/inference/svi.md#quivers.inference.svi.SVI.step):

<!-- python: skip -->
```python
observations = {
    "cloze_resp": cloze_tensor,    # shape (n_cloze_resp,)
    "prop_resp":  prop_tensor,     # shape (n_prop_resp,)
}

samples = program.rsample(x, observations=observations)
loss = elbo(model, guide, x, observations)
```

There is no `.qvr`-level data block; the tensor sources live in
Python at the call site, and the keys must match the response
identifiers declared in the program body.

## Named parameters

If the domain is a product, name the components via the `params`
argument so steps can reference them by name:

<!-- python: skip -->
```python
A = FinSet(name="A", cardinality=3)
B = FinSet(name="B", cardinality=4)
Z = FinSet(name="Z", cardinality=5)

program = MonadicProgram(
    A * B,
    Z,
    steps=[
        Draw(names=("x",), morphism=f, args=("a", "b")),  # x <- f(a, b)
    ],
    return_vars=("x",),
    params=("a", "b"),
)
```

The program splits the product input along the feature axis at
runtime and binds each slice to the corresponding name in `params`.

## Example: a simple model

```python
import torch
from quivers.continuous import Draw, MonadicProgram
from quivers.continuous.families import (
    ConditionalNormal,
    ConditionalLogitNormal,
)
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet

Unit = FinSet(name="Unit", cardinality=1)
R1 = Euclidean(name="R1", dim=1)
R2 = Euclidean(name="R2", dim=2)

prior_mu    = ConditionalNormal(Unit, R1)
prior_sigma = ConditionalLogitNormal(Unit, R1)
likelihood  = ConditionalNormal(R2, R1)

program = MonadicProgram(
    Unit,
    R1,
    steps=[
        Draw(names=("mu",), morphism=prior_mu),
        Draw(names=("sigma",), morphism=prior_sigma),
        Draw(names=("y",), morphism=likelihood, args=("mu", "sigma")),
    ],
    return_vars=("y",),
)

# Use for inference
optimizer = torch.optim.Adam(program.parameters())
```

## Destructuring binds

Extract multiple values from a tuple-returning sub-program:

<!-- compile: false -->
```qvr
program sub : X -> Y * Y
    sample (a, b) <- some_morphism
    return (a, b)

program main : X -> Z
    sample (u, v) <- sub
    sample w <- g(u, v)
    return w
```

The pattern `sample (u, v) <- sub` destructures the output.

## Observation clamping

During inference, the
[`condition()`](../api/inference/conditioning.md#quivers.inference.conditioning.condition)
function clamps observations:

```python
from quivers.inference import condition

# Condition program on external observations
observed_y = torch.tensor([1.0, -0.5, 2.0])

conditioned = condition(program, {"y": observed_y})

# Trace under the conditioning: observed sites are clamped to the data
x = torch.zeros(3, dtype=torch.long)
tr = conditioned.trace(x)
```

## Product domains and outputs

For multiple domain inputs, stack along the feature dimension:

<!-- compile: false -->
```qvr
program f(x_val, y_val) : (X * Y) -> Z
    sample z <- g(x_val, y_val)
    return z
```

The bare-identifier parameters `x_val`, `y_val` name the
projections of the product domain. Internally, the domain tensor
is reshaped to match.

## Building programs in Python

The step-record interface is useful when a model is assembled by Python
code or generated from another representation. It describes the same
dependencies as QVR source. `Draw` and `Observe` introduce random and
measured values; `Let` computes from values already available. The example
below compares the joint scores of two constructions of the same model,
which is stronger evidence of agreement than merely comparing their shapes.

Every program the DSL compiles is assembled from objects the
[`quivers.continuous`](../api/continuous/programs.md) package exports, so
a program can be built in Python without writing QVR source. The step
records are the contract:

| Record | Source form | Fields |
|---|---|---|
| [`Draw`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Draw) | `sample x <- f(a, b)` | `names`, `morphism`, `args` |
| [`Observe`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Observe) | `observe y <- f(a, b)` | `names`, `morphism`, `args` |
| [`Let`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Let) | `let w = expression` | `name`, `value` |
| [`Score`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Score) | a `marginalize` block | `name`, `score` |

A draw's `args` are binding names, or
[`Indexed`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Indexed)
gathers such as `mu[group]`; `None` reads the program input. A let's
`value` is a constant, the name of an earlier binding, or a function
of the environment, which may declare the names it reads through
[`reading`](../api/continuous/program_steps.md#quivers.continuous.program_steps.reading).
The morphisms are the ones the compiler emits: a fixed-parameter
family comes from a `make_fixed_*` factory such as
[`make_fixed_normal`](../api/continuous/inline.md#quivers.continuous.inline.make_fixed_normal);
a family whose parameters are bindings is a
[`MixedInlineDistribution`](../api/continuous/inline.md#quivers.continuous.inline.MixedInlineDistribution)
over the stacked arguments; an indexed draw `sample v : A <- F(...)` is a
[`PlateDraw`](../api/continuous/plate.md#quivers.continuous.plate.PlateDraw)
of one draw per element of `A`; and an indexed observe is a
[`VectorisedObserve`](../api/continuous/plate.md#quivers.continuous.plate.VectorisedObserve).

The program below is a hierarchical Normal model with one mean per
group, built twice: once from source, and once from the records. The
two score every value identically.

```python
import torch
import torch.distributions as D

from quivers.continuous import (
    Draw,
    Euclidean,
    Indexed,
    MixedInlineDistribution,
    MonadicProgram,
    Observe,
    PlateDraw,
    ProductSpace,
    VectorisedObserve,
    make_fixed_halfnormal,
    make_fixed_normal,
)
from quivers.core import FinSet, Unit
from quivers.dsl import loads

compiled = loads("""
object Group : FinSet 3
object Row : FinSet 6
object Val : Real 1

program model : Row -> Val
    sample mu : Group <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    let m = mu[group]
    observe y : Row <- Normal(loc=m, scale=sigma)
    return y

export model
""").morphism

Row = FinSet(name="Row", cardinality=6)
Val = Euclidean(name="Val", dim=1)

likelihood = MixedInlineDistribution(
    ProductSpace(components=(Val, Val)),
    Val,
    param_spec=[("var", 1), ("var", 1)],
    dist_builder=lambda params: D.Normal(params[0], params[1]),
)
built = MonadicProgram(
    Row,
    Val,
    steps=[
        # sample mu : Group <- Normal(0, 1), one mean per group
        Draw(
            names=("mu",),
            morphism=PlateDraw(3, make_fixed_normal(0.0, 1.0, Val), domain=Unit),
        ),
        # sample sigma <- HalfNormal(1)
        Draw(names=("sigma",), morphism=make_fixed_halfnormal(1.0, Val)),
        # observe y : Row <- Normal(mu[group], sigma)
        Observe(
            names=("y",),
            morphism=VectorisedObserve(likelihood, torch.zeros(6)),
            args=(Indexed(name="mu", indices=("group",)), "sigma"),
        ),
    ],
    return_vars=("y",),
)

values = {
    "mu": torch.randn(3, 1),
    "sigma": torch.rand(1, 1) + 0.5,
    "y": torch.randn(6),
    "group": torch.tensor([0, 0, 1, 1, 2, 2]),
}
x = torch.arange(6)
torch.testing.assert_close(compiled.log_joint(x, values), built.log_joint(x, values))
```

A group-level correlation structure follows the same pattern. A
`PlateDraw` over an LKJ prior on Cholesky factors draws one factor per
group, whether the prior is the inline family
([`make_fixed_lkj_cholesky`](../api/continuous/inline.md#quivers.continuous.inline.make_fixed_lkj_cholesky))
or the [`LKJCorrelationFactor`](../api/continuous/families.md#quivers.continuous.families.LKJCorrelationFactor)
morphism:

```python
from quivers.continuous import CholeskyFactor, make_fixed_lkj_cholesky

factors = PlateDraw(
    3, make_fixed_lkj_cholesky(2.0, CholeskyFactor(name="L", dim=4)), domain=Unit
)
draw = factors.rsample(torch.zeros(1, dtype=torch.long))  # (3, 16): one factor per group
reference = D.LKJCholesky(4, torch.tensor(2.0)).log_prob(draw.reshape(3, 4, 4)).sum()
torch.testing.assert_close(factors.log_prob(torch.zeros(1), draw).squeeze(), reference)
```

## Integration with the DSL

MonadicPrograms are the output of `.qvr` DSL compilation. The DSL
parser translates:

<!-- compile: false -->
```qvr
object X : FinSet 3
object Y : FinSet 4
program my_prog : X -> Y
    sample mu <- LogitNormal(mu=0, sigma=1)
    sample x <- Normal(loc=mu, scale=1)
    return x

export my_prog
```

into a `MonadicProgram` instance that can be trained. The full DSL
syntax for programs is documented in
[DSL Programs and Let-Expressions](dsl-programs-and-lets.md).

## Where to next

- [Hierarchical Programs](programs-hierarchical.md): parametric
  templates for crossed random intercepts, monotone-spline
  coefficients, and the grouped [marginalization](dsl-programs-and-lets.md#scoped-marginalize) construct for
  fibred discrete latents.
- [Variational Inference](inference-foundations.md): how programs
  feed into the variational training loop.
