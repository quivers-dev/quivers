# Tutorial 3: Probabilistic Programs

[`MonadicProgram`](../../api/continuous/programs.md) represents a probabilistic computation as an ordered sequence of typed sampling, observation, marginalization, and deterministic-binding steps. This chapter constructs programs with discrete and continuous random variables, applies conditional distribution families, and samples from the resulting model.

For a Stan/PyMC reader: a [`MonadicProgram`](../../api/continuous/programs.md) is the Python-API analogue of the QVR DSL's `program` block. You construct it as a list of steps (`bind`, `let`, `observe`, `marginalize`), each step typed by the space its result lives in. The runtime provides `rsample` for draws and `log_joint` for trace-based joint scoring. `MonadicProgram.log_prob` is intentionally unavailable because marginalizing arbitrary intermediate latents is intractable in general.

### The two kinds of step in one sentence

- `v <- F(args)` is a *bind*: it draws `v` from `F(args)` and threads it forward. Use this when `v` is a random variable.
- `let v = expr` is a *let*: it computes `v` deterministically from earlier variables. Use this when `v` is computed from existing values without a fresh random draw. It can still vary randomly across executions because its inputs vary.

### Host-data channel

The third source of names is *[host data](../qvr/01-first-model.md#the-observations-dict-contract)*: tensors supplied at runtime through the observations dict. They look like free variables in your program body and resolve at trace time from `observations["name"]`. There's no `<-` and no `let`; you just reference the name. Host data is how you pass design matrices, group indices, or any covariate that varies per fit run without rewriting the model. Every `bind` or `let` step that mentions a name not introduced by an earlier `<-`, `let`, or program input is referencing host data.

Consider `z` as a true temperature and `y` as a noisy measurement. First draw
one temperature, then use that realized temperature as the mean of the
measurement distribution. Sequencing passes a value forward, even though each
step describes a distribution. The word *monadic* refers to the composition
that makes this sequence possible. Milewski explains this pattern through
[functions that return computations](https://bartoszmilewski.com/2014/12/23/kleisli-categories/) [@milewski2014kleisli]; Moggi develops
the underlying semantics of computational effects [@moggi1991notions].

A full trace contains both `z` and `y`, so its [joint density](05-variational-inference.md#tracing-the-model) is
`p(z) * p(y | z)`. If you keep only `y`, scoring that output generally requires
integrating over all temperatures that could have produced it. `log_joint`
evaluates the product at the supplied values. Computing the output marginal
requires a separate integration method, so `MonadicProgram.log_prob` is unavailable.

## Concepts

- **Continuous spaces**: Euclidean, UnitInterval, Simplex, PositiveReals
- **Conditional distribution families**: Parameterized distributions (ConditionalNormal, ConditionalBernoulli, etc.)
- **MonadicProgram**: A probabilistic computation with bind and let steps
- **Bind step**: Sample a random variable from a distribution conditioned on prior variables (`v <- F(args)`)
- **Let step**: Deterministic transformation of prior variables
- **Indexed bind (plate)**: An indexed plate of independent draws (`v : A <- F(args)`)
- **Indexed observe**: A batched-likelihood step reading from a runtime `observations` dict (`observe r : N <- F(args)`)
- **Scoped marginalize**: Log-sum-exp pushforward over a latent coordinate (`marginalize c : A <- F(args)` followed by an indented body)
- **Effect signature**: A program's capability set declared in the option block as `[effects=[Sample, Score, Marginal, Pure]]`; the compiler verifies actual body effects are a subset
- **Parametric program**: A typed-parameter template inlined per call site as fresh latents
- **Log-probability**: The log of the joint density

## Setup

```python
import torch
from quivers.core.objects import FinSet
from quivers.continuous.spaces import (
    Euclidean,
    UnitInterval,
    Simplex,
    PositiveReals,
)
from quivers.continuous.families import ConditionalNormal, ConditionalBernoulli
from quivers.continuous import Draw, Let, MonadicProgram
```

## Defining spaces

Continuous random variables live in [continuous spaces](../../api/continuous/spaces.md). Create a few:

```python
# Euclidean space: R^2
R2 = Euclidean(name="position", dim=2)
print(R2.dim)          # 2
print(R2.event_shape)  # (2,)

# Unit interval: [0, 1]
U = UnitInterval(name="probability")
print(U.dim)  # 1

# Positive reals: (0, inf)
P = PositiveReals(name="variance", dim=1)

# Simplex: probability distributions over k categories
S = Simplex(name="mixture_weights", dim=3)
print(S.dim)  # 3
```

Each space exposes a `contains` predicate, and bounded spaces support uniform sampling:

```python
samples = U.sample_uniform(100)
print(samples.shape)  # [100, 1]
assert (samples >= 0.0).all() and (samples <= 1.0).all()
```

## Conditional distribution families

A [conditional distribution family](../../api/continuous/families.md) maps an input space to a parameterized family of distributions. For instance, [`ConditionalNormal`](../../api/continuous/families.md) learns mean and scale as functions of the input.

Create a conditional normal from a discrete input (finite set) to a continuous output:

```python
Unit = FinSet(name="Unit", cardinality=1)
R = Euclidean(name="latent", dim=1)

prior_family = ConditionalNormal(Unit, R)
```

Sample from it. A discrete input is a 1D LongTensor of indices:

```python
batch = torch.zeros(5, dtype=torch.long)  # batch of 5 unit inputs
samples = prior_family.rsample(batch)
print(samples.shape)        # [5, 1]
log_probs = prior_family.log_prob(batch, samples)
print(log_probs.shape)      # [5]
```

Create a likelihood family from continuous to continuous. Continuous inputs have shape `(batch, dim)`:

```python
likelihood_family = ConditionalNormal(R, R)
z_value = torch.randn(5, 1)
y_samples = likelihood_family.rsample(z_value)
print(y_samples.shape)  # [5, 1]
```

Other families available include [`ConditionalBernoulli`](../../api/continuous/families.md), [`ConditionalBeta`](../../api/continuous/families.md), [`ConditionalLaplace`](../../api/continuous/families.md), and many more. They follow the same `rsample` / `log_prob` interface.

## Building a `MonadicProgram`

A `MonadicProgram` represents a probabilistic computation as a sequence of steps. Each step is either:

1. **Bind**: sample a random variable from a conditional distribution (`v <- F`)
2. **Let**: compute a deterministic transformation

Construct a simple two-stage program: bind z from a prior, then bind y from a likelihood conditioned on z.

```python
Unit = FinSet(name="Unit", cardinality=1)
R = Euclidean(name="real", dim=1)

prior = ConditionalNormal(Unit, R)
likelihood = ConditionalNormal(R, R)

program = MonadicProgram(
    Unit, R,  # input space, output space
    steps=[
        Draw(names=("z",), morphism=prior),  # z <- prior(unit)
        Draw(names=("y",), morphism=likelihood, args=("z",)),  # y <- likelihood(z)
    ],
    return_vars=("y",),
)
```

Each step is a record. A [`Draw`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Draw) has three fields:

- `names`: the variable names the step binds (a tuple, even if there is just one variable)
- `morphism`: the conditional distribution sampled from
- `args`: the earlier bindings the distribution reads, or `None` (the default) to read the program input

[`Observe`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Observe) has the same fields and scores supplied data instead of sampling. [`Let`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Let) and [`Score`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Score) bind one `name` to a deterministic `value` or to a log-density contribution.

Sample from the program. A discrete input is a 1D LongTensor of indices:

```python
batch = torch.zeros(10, dtype=torch.long)
output = program.rsample(batch)
print(output.shape)  # [10, 1]
```

Note that `MonadicProgram.log_prob` is intractable in general because it must
marginalize over intermediate latents, so it raises `NotImplementedError`. To
score a program, use a guide and the SVI implementation covered in
[Tutorial 5](05-variational-inference.md).

## Let bindings

Add a deterministic transformation step. For instance, compute w = z * 2:

```python
Unit = FinSet(name="Unit", cardinality=1)
R = Euclidean(name="real", dim=1)

prior = ConditionalNormal(Unit, R)
likelihood = ConditionalNormal(R, R)

# A let binding: deterministic transformation
def double(env):
    """env is a dict mapping prior variable names to their sampled values."""
    return env["z"] * 2.0

program = MonadicProgram(
    Unit, R,
    steps=[
        Draw(names=("z",), morphism=prior),
        Let(name="w", value=double),  # let w = z * 2
        Draw(names=("y",), morphism=likelihood, args=("w",)),  # y <- likelihood(w)
    ],
    return_vars=("y",),
)

batch = torch.zeros(5, dtype=torch.long)
output = program.rsample(batch)
# internally: z was sampled, w = z*2, then y was sampled from likelihood(w)
```

A let binding's value can be a callable (run on the environment), a string (alias to an existing variable), or a float constant.

## Multi-variable outputs

Return multiple variables. Multi-return programs yield a dict keyed by variable name:

```python
program = MonadicProgram(
    Unit, R * R,  # output is the product space R x R
    steps=[
        Draw(names=("z",), morphism=prior),
        Draw(names=("y",), morphism=likelihood, args=("z",)),
    ],
    return_vars=("z", "y"),
)

batch = torch.zeros(10, dtype=torch.long)
out = program.rsample(batch)
print(out["z"].shape, out["y"].shape)  # both [10, 1]
```

## Conditional Bernoulli

Use discrete outputs. [`ConditionalBernoulli`](../../api/continuous/families.md) requires a FinSet of cardinality 2 (representing {0, 1}) as its codomain:

```python
from quivers.continuous.families import ConditionalBernoulli

R = Euclidean(name="latent", dim=2)
Coin = FinSet(name="Coin", cardinality=2)

bernoulli = ConditionalBernoulli(R, Coin)
```

This learns logits as a function of the continuous input, then samples binary values:

```python
z = torch.randn(5, 2)
samples = bernoulli.rsample(z)
print(samples.shape)               # [5]
print(((samples == 0) | (samples == 1)).all())  # True
```

Note that Bernoulli sampling is not reparameterizable: gradients do not flow through the discrete output.

## Forward and backward

Wrap the program in a [`Program`](../../api/program.md) module so its parameters are visible to a PyTorch optimizer:

```python
from quivers.program import Program

prog = Program(program)
optimizer = torch.optim.Adam(prog.parameters(), lr=0.01)

batch = torch.zeros(32, dtype=torch.long)
target = torch.randn(32, 1)

for epoch in range(10):
    optimizer.zero_grad()
    output = prog.rsample(batch)
    loss = ((output["y"] - target) ** 2).mean()
    loss.backward()
    optimizer.step()
```

This is a toy objective. Proper inference for a generative model is covered in [Tutorial 5](05-variational-inference.md).

## Complex programs

Build larger programs with multiple stages and branches. Here is a model for two observations with a shared latent:

```python
Unit = FinSet(name="Unit", cardinality=1)
R = Euclidean(name="space", dim=1)

# Shared prior
prior = ConditionalNormal(Unit, R)

# Two likelihoods
likelihood1 = ConditionalNormal(R, R)
likelihood2 = ConditionalNormal(R, R)

program = MonadicProgram(
    Unit, R * R,
    steps=[
        Draw(names=("z",), morphism=prior),                  # shared latent
        Draw(names=("y1",), morphism=likelihood1, args=("z",)),         # observation 1
        Draw(names=("y2",), morphism=likelihood2, args=("z",)),         # observation 2
    ],
    return_vars=("y1", "y2"),
)

batch = torch.zeros(20, dtype=torch.long)
out = program.rsample(batch)
print(out["y1"].shape, out["y2"].shape)  # both [20, 1]
```

## Summary

You have:

- Created continuous spaces
- Defined conditional distribution families
- Built `MonadicProgram` instances with bind and let steps
- Sampled from programs
- Used `ConditionalNormal` and `ConditionalBernoulli`
- Integrated programs with PyTorch optimization

Next, see fuzzy logic factorization with algebra-enriched composition in [Tutorial 4](04-fuzzy-factorization.md).
