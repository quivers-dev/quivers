# 3. Probabilistic programs

A [`MonadicProgram`](../../api/continuous/programs.md) is a probabilistic program built in Python: an ordered list of steps, each of which draws a variable from a conditional distribution, observes data, adds to the log weight, or binds a deterministic value. It is what a QVR `program` block compiles to. Each step may read the values bound before it, so the program samples ancestrally, and its joint density is the product of the conditional densities of its steps. Sequencing steps this way is monadic composition [@moggi1991notions]. Build programs over continuous spaces with conditional distribution families, sample them, and score their joint density.

A program provides `rsample` for draws and `log_joint` for the joint log density of given values of all its variables. Its `log_prob`, the density of the output alone, would need an integral over every intermediate variable, so it raises `NotImplementedError`.

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

Continuous variables take values in [continuous spaces](../../api/continuous/spaces.md):

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

Each space has a `contains` predicate, and a bounded space can be sampled uniformly:

```python
samples = U.sample_uniform(100)
print(samples.shape)  # [100, 1]
assert (samples >= 0.0).all() and (samples <= 1.0).all()
```

## Conditional distribution families {#conditional-distribution-families}

A [conditional distribution family](../../api/continuous/families.md) maps each input to a distribution on its output space. [`ConditionalNormal`](../../api/continuous/families.md) learns the mean and scale as functions of the input. From a finite set with one element it is an unconditional Normal with learnable parameters:

```python
Unit = FinSet(name="Unit", cardinality=1)
R = Euclidean(name="latent", dim=1)

prior_family = ConditionalNormal(Unit, R)
```

An input from a finite set is a one-dimensional integer tensor of indices:

```python
batch = torch.zeros(5, dtype=torch.long)  # batch of 5 unit inputs
samples = prior_family.rsample(batch)
print(samples.shape)        # [5, 1]
log_probs = prior_family.log_prob(batch, samples)
print(log_probs.shape)      # [5]
```

An input from a continuous space has shape `(batch, dim)`:

```python
likelihood_family = ConditionalNormal(R, R)
z_value = torch.randn(5, 1)
y_samples = likelihood_family.rsample(z_value)
print(y_samples.shape)  # [5, 1]
```

[`ConditionalBernoulli`](../../api/continuous/families.md), [`ConditionalBeta`](../../api/continuous/families.md), [`ConditionalLaplace`](../../api/continuous/families.md), and the other families have the same `rsample` and `log_prob` interface.

## Building a `MonadicProgram`

Draw `z` from a prior, then `y` from a likelihood whose input is `z`:

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

A [`Draw`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Draw) has three fields:

- `names`: the variables the step binds, as a tuple even for one variable.
- `morphism`: the conditional distribution to draw from.
- `args`: the earlier bindings the distribution reads, or `None` (the default) for the program input.

An [`Observe`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Observe) has the same fields and scores given data instead of drawing. A [`Let`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Let) binds one `name` to a deterministic `value`, and a [`Score`](../../api/continuous/program_steps.md#quivers.continuous.program_steps.Score) binds a `name` to the value of its `score` function and adds that value to the log weight. A name that a step reads and no earlier step binds is a *data input*, resolved at run time from the observations ([data inputs in the guide](../../guides/inference-foundations.md#host-data-per-row-covariates-and-index-arrays)).

Sample the program with an integer input:

```python
batch = torch.zeros(10, dtype=torch.long)
output = program.rsample(batch)
print(output.shape)  # [10, 1]
```

Score given values of `z` and `y` with `log_joint`, which returns $\log p(z) + \log p(y \mid z)$ for each row of the input:

```python
log_joint = program.log_joint(
    torch.zeros(3, dtype=torch.long),
    {"z": torch.zeros(3, 1), "y": torch.ones(3, 1)},
)
print(log_joint.shape)  # [3]
```

[Chapter 5](05-variational-inference.md) fits such a program by stochastic variational inference (SVI).

## Let bindings

Compute `w = 2 z` deterministically and draw `y` given `w`:

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

The value of a `Let` is a callable applied to the bindings so far, a string naming an existing variable, or a float constant. A `Let` draws nothing, though its value varies from run to run when its inputs do.

## Multi-variable outputs

A program that returns several variables yields a dict keyed by name:

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

[`ConditionalBernoulli`](../../api/continuous/families.md) takes a finite set of size 2, standing for $\{0, 1\}$, as its codomain:

```python
from quivers.continuous.families import ConditionalBernoulli

R = Euclidean(name="latent", dim=2)
Coin = FinSet(name="Coin", cardinality=2)

bernoulli = ConditionalBernoulli(R, Coin)
```

It learns the logits as a function of the continuous input and draws binary values:

```python
z = torch.randn(5, 2)
samples = bernoulli.rsample(z)
print(samples.shape)               # [5]
print(((samples == 0) | (samples == 1)).all())  # True
```

A Bernoulli draw has no reparameterization, so no gradient flows through the sampled value.

## Forward and backward

A [`Program`](../../api/program.md) exposes a program's parameters to a PyTorch optimizer:

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

The squared error here only shows the gradient path. [Chapter 5](05-variational-inference.md) fits a generative model by variational inference.

## Shared latent variables

Two observations that share one latent variable:

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

## Next

[Chapter 4](04-fuzzy-factorization.md) factorizes a fuzzy relation into a composite of learnable morphisms.
