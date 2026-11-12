# Monadic Programs

## What is a MonadicProgram?

A probabilistic program describes a procedure for generating data: draw a latent mean
`mu`, draw a response around it, and return the response. The same description scores
actual responses when their observed values replace the draws, and inference finds
plausible latent values given those observations
([Inference Foundations](inference-foundations.md)).

A [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram)
is such a program, a sequence of draws, observations, and deterministic steps, and it
is the class that compiling a QVR `program` produces. It is a
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
from its domain to its codomain: a Markov kernel built by composing its steps in the
[Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of a probability
[monad](https://ncatlab.org/nlab/show/monad) [@moggi1991notions], each step drawing
from a distribution that may depend on the values drawn before it. Pyro and NumPyro
write programs in the same sequential style [@bingham2019pyro; @phan2019numpyro]:

<!-- compile: false -->
```qvr
program name : domain -> codomain
    sample x_1 <- morphism_1
    sample x_2 <- morphism_2(x_1)
    let y = x_1 + x_2
    observe z <- morphism_3(y)
    return y
```

Each `sample ... <- ...` step draws from a conditional distribution and binds the
result, and `observe` scores an observed value instead of drawing it.

## Program structure

A program is a PyTorch
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
whose `rsample` runs [ancestral sampling](https://en.wikipedia.org/wiki/Bayesian_network),
drawing each variable after the variables it depends on, in program order:

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

# Log joint density given every bound variable, one value per run:
# the four input rows are four independent runs.
x_val = torch.randn(4, 1)
y_val = torch.randn(4, 1)
log_joint = program.log_joint(
    torch.zeros(4, dtype=torch.long),
    {"x": x_val, "y": y_val},
)  # shape (4,); log_joint.sum() is the joint of the four runs
```

Each row of a program's input is one independent run
([Input rows are independent runs](inference-foundations.md#input-rows-are-independent-runs)).

## Draw steps {#bind-steps}

A draw gives a random value a name for the later steps, as Pyro's `sample` does
[@pyroIntroduction]. In
[`sample y <- f(x)`](../reference/qvr/probabilistic-programs.md#step-reference), `y` is
drawn given the particular value of `x` drawn earlier in the same run, and running the
program again draws new values. A draw takes a morphism with no arguments, a morphism
applied to earlier names, or several names at once from a morphism into a product:

<!-- compile: false -->
```qvr
sample x <- prior_f
sample y <- likelihood_f(x)
sample (u, v) <- joint_f(z, w)
```

In the last form the names are bound to the components of the product in order. A
plated draw, `sample v : A <- F(args)`, draws one value per element of `A`
([plates](dsl-programs-and-lets.md#indexed-bind-plate)).

## Let steps

`let` binds a deterministic value: a literal, an earlier name, arithmetic, or a
primitive ([let expressions](dsl-programs-and-lets.md#let-expressions-arithmetic-primitives-and-collections)).

<!-- compile: false -->
```qvr
let x = y + z
let weight = 0.5
```

## Observe keyword

An observation uses a measured value in place of a draw and scores how well the current
latent values predict it. If `mu` changes, the observed response stays fixed and its
likelihood changes. SVI uses this score to update a guide, and MCMC uses it when
proposing and accepting new values of `mu`.

<!-- compile: false -->
```qvr
observe y <- likelihood(x)
```

The observed value of `y` comes from the [`observations` dict](#the-observations-dict),
and `observe r : N <- F(args)` scores one observed value per element of `N`.

## Return statement

`return` names the program's output, one value or a tuple:

<!-- compile: false -->
```qvr
return x
```

<!-- compile: false -->
```qvr
return (x, y, z)
```

The returned value fixes the codomain, and a tuple's components form a product space
in the order written.

## Domains and codomains

A domain is a [`FinSet`](../api/core/objects.md), a
[`ContinuousSpace`](../api/continuous/spaces.md), or a product of these, `X * Y * Z`,
whose components the program may name ([Named parameters](#named-parameters)). The
codomain is fixed by what the program returns.

## rsample and log_joint

### rsample(x, sample_shape=(), observations=None)

`rsample` runs the program and returns draws of its output:

```python
x = torch.zeros(5, dtype=torch.long)
samples = program.rsample(x, sample_shape=torch.Size([1000]))
# shape: (1000, 5, codomain_dim)
```

Each draw step samples in order, and later steps read earlier draws. The optional
`observations` dict fixes observed sites to data. The draws are reparameterized where
the family allows, so gradients flow through them [@kingma2014autoencoding].

### log_joint(x, intermediates)

`log_joint` scores one run of the program at given values of all its variables. For
$\mu \sim \mathcal N(0, 1)$ and $y \sim \mathcal N(\mu, 1)$ it is
$\log p(\mu, y) = \log p(\mu) + \log p(y \mid \mu)$, the prior term plus the likelihood
term. In general it is $\sum_i \log p(z_i \mid \mathrm{pa}(z_i))$, where
$\mathrm{pa}(z_i)$ are the variables the step for $z_i$ reads, its parents in the
[Bayesian network](https://en.wikipedia.org/wiki/Bayesian_network) of the program
[@koller2009probabilistic]. Inference compares such scores at different latent values,
which does not need the posterior's normalizing constant.

<!-- python: skip -->
```python
x = torch.zeros(5, dtype=torch.long)
intermediates = {"z": z_value, "y": y_value}  # every bound variable

log_pjoint = program.log_joint(x, intermediates)
```

`log_joint` returns one value per run, shape `(B,)` for `B` input rows, each the sum of
that run's draw, plated draw, observe, and score terms, so `log_joint(x, values).sum()`
is the joint of all the runs. A value in `intermediates` either has one entry per run,
along a leading axis of length `B`, or is shared by every run.
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) calls it
after the guide has drawn the latent values.

### The `observations` dict

An observe step reads its observed values from a dict
`observations: dict[str, torch.Tensor]` keyed by the observed name. The dict is the
`observations` argument of
[`MonadicProgram.rsample`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.rsample)
and the last positional argument of
[`ELBO.forward`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) and
[`SVI.step`](../api/inference/svi.md#quivers.inference.svi.SVI.step). The same dict
carries the program's [data inputs](inference-foundations.md#host-data-per-row-covariates-and-index-arrays),
the names it reads without binding.

<!-- python: skip -->
```python
observations = {
    "cloze_resp": cloze_tensor,    # shape (n_cloze_resp,)
    "prop_resp":  prop_tensor,     # shape (n_prop_resp,)
}

samples = program.rsample(x, observations=observations)
loss = elbo(model, guide, x, observations)
```

QVR has no data block: the tensors are passed from Python at the call, and the keys
must match the names the program observes or reads.

## Named parameters

When the domain is a product, `params` names its components so that steps can read
them by name:

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

At run time the program splits the input along its last axis and binds each part to
its name. In QVR the names follow the program's name, `program f(a, b) : A * B -> Z`:

```qvr
object A : Real 1
object B : Real 2
object Z : Real 1

morphism g : A * B -> Z ~ Normal

program f(a_val, b_val) : A * B -> Z
    sample z <- g(a_val, b_val)
    return z

export f
```

## Example: a simple model

A normal model with unknown mean and scale, compiled from QVR:

```python
import torch
from quivers.dsl import loads

program = loads("""
object Unit : FinSet 1
object R1 : Real 1

program model : Unit -> R1
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample sigma <- LogNormal(loc=0.0, scale=1.0)
    sample y <- Normal(loc=mu, scale=sigma)
    return y

export model
""").morphism

draws = program.rsample(torch.zeros(1, 1), sample_shape=torch.Size([3]))  # (3, 1, 1)
score = program.log_joint(
    torch.zeros(1, 1),
    {"mu": torch.tensor([[0.2]]), "sigma": torch.tensor([[1.1]]), "y": torch.tensor([[0.5]])},
)
```

Building the last step in Python as `Draw(names=("y",), morphism=ConditionalNormal(R2, R1), args=("mu", "sigma"))`
would define a different model: `ConditionalNormal` computes a normal's mean and scale
from its input with a learned network, so `y` would not have mean `mu` and scale
`sigma`. [Building programs in Python](#building-programs-in-python) shows the records
that give a family's parameters directly.

## Destructuring draws

A draw from a program that returns a tuple binds each component:

```qvr
object X : FinSet 2
object Y : Real 1
object Z : Real 1

morphism some_morphism : X -> Y ~ Normal
morphism g : Y * Y -> Z ~ Normal

program sub : X -> Y * Y
    sample a <- some_morphism
    sample b <- some_morphism
    return (a, b)

program main : X -> Z
    sample (u, v) <- sub
    sample w <- g(u, v)
    return w

export main
```

## Observation clamping

[`condition()`](../api/inference/conditioning.md#quivers.inference.conditioning.condition)
returns a program whose named sites are fixed to given values
([Conditioning on observations](inference-foundations.md#conditioning-on-observations)):

```python
from quivers.inference import condition

# Condition program on external observations
observed_y = torch.tensor([1.0, -0.5, 2.0])

conditioned = condition(program, {"y": observed_y})

# Trace under the conditioning: observed sites are clamped to the data
x = torch.zeros(3, dtype=torch.long)
tr = conditioned.trace(x)
```

## Building programs in Python

Every program the compiler builds is made of classes that
[`quivers.continuous`](../api/continuous/programs.md) exports, so a model assembled by
Python code, or generated from another representation, can be built without QVR
source. The step records are:

| Record | Source form | Fields |
|---|---|---|
| [`Draw`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Draw) | `sample x <- f(a, b)` | `names`, `morphism`, `args` |
| [`Observe`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Observe) | `observe y <- f(a, b)` | `names`, `morphism`, `args` |
| [`Let`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Let) | `let w = expression` | `name`, `value` |
| [`Score`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Score) | a `marginalize` block | `name`, `score` |

A draw's `args` are names of earlier bindings or
[`Indexed`](../api/continuous/program_steps.md#quivers.continuous.program_steps.Indexed)
gathers such as `mu[group]`, and `None` reads the program input. A `Let`'s `value` is a
constant, the name of an earlier binding, or a function of the environment, which may
declare the names it reads with
[`reading`](../api/continuous/program_steps.md#quivers.continuous.program_steps.reading).
The morphisms are the ones the compiler emits. A family with fixed parameters comes
from a `make_fixed_*` factory such as
[`make_fixed_normal`](../api/continuous/inline.md#quivers.continuous.inline.make_fixed_normal),
a family whose parameters are bindings is a
[`MixedInlineDistribution`](../api/continuous/inline.md#quivers.continuous.inline.MixedInlineDistribution)
over the stacked arguments, a plated draw `sample v : A <- F(...)` is a
[`PlateDraw`](../api/continuous/plate.md#quivers.continuous.plate.PlateDraw) of one
draw per element of `A`, and a plated observe is a
[`VectorisedObserve`](../api/continuous/plate.md#quivers.continuous.plate.VectorisedObserve).

The hierarchical normal model below, with one mean per group, is built once from source
and once from the records, and the two give the same log joint density:

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
x = torch.zeros(1, 1)
torch.testing.assert_close(compiled.log_joint(x, values), built.log_joint(x, values))
```

Here `group` is a data input, the index array assigning each row to its group. A
correlation structure per group follows the same pattern: a `PlateDraw` over an LKJ
prior on Cholesky factors draws one factor per group, whether the prior is the inline
family
([`make_fixed_lkj_cholesky`](../api/continuous/inline.md#quivers.continuous.inline.make_fixed_lkj_cholesky))
or the [`LKJCorrelationFactor`](../api/continuous/families.md#quivers.continuous.families.LKJCorrelationFactor)
morphism [@lewandowski2009lkj]:

```python
from quivers.continuous import CholeskyFactor, make_fixed_lkj_cholesky

factors = PlateDraw(
    3, make_fixed_lkj_cholesky(2.0, CholeskyFactor(name="L", dim=4)), domain=Unit
)
draw = factors.rsample(torch.zeros(1, dtype=torch.long))  # (3, 16): one factor per group
reference = D.LKJCholesky(4, torch.tensor(2.0)).log_prob(draw.reshape(3, 4, 4)).sum()
torch.testing.assert_close(factors.log_prob(torch.zeros(1), draw).squeeze(), reference)
```

## Where to next

- [DSL Programs and Let-Expressions](dsl-programs-and-lets.md): the full QVR syntax of
  programs.
- [Hierarchical Programs](programs-hierarchical.md): templates for crossed random
  intercepts, monotone effects, and marginalizing a discrete latent once per group.
- [Inference Foundations](inference-foundations.md): how programs are fit by
  variational inference and MCMC.
