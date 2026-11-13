# GRU Language Model

## Overview

A gated recurrent unit (GRU) [@cho2014encoder] is a recurrent cell whose update is controlled by two gates computed from the current input and the previous state: a reset gate that decides how much of the previous state the candidate reads, and an update gate that interpolates between the previous state and the candidate. This source states the GRU exactly, with each gate and the candidate a [learned deterministic function](../guides/dsl-declarations.md#function), threads the cell along the sequence with [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence), and scores the next token at every position. The weights are point-estimated, so fitting the program's log joint is maximum-likelihood training.

## QVR source

```qvr
# A gated recurrent unit language model with learned weights. The gates and
# the candidate state are learned functions of the current token and the
# state; the update interpolates between the previous state and the
# candidate, and each position scores the next token.
object Vocab : Real 6
object Hidden : Real 8
object Logits : Real 6

# z_t = sigmoid(W_z [x_t; h_{t-1}] + b_z), the update gate.
morphism update_gate : Vocab * Hidden -> Hidden [role=function, output=sigmoid]
# r_t = sigmoid(W_r [x_t; h_{t-1}] + b_r), the reset gate.
morphism reset_gate : Vocab * Hidden -> Hidden [role=function, output=sigmoid]
# n_t = tanh(W_n [x_t; r_t * h_{t-1}] + b_n), the candidate state.
morphism candidate : Vocab * Hidden -> Hidden [role=function, output=tanh]
morphism lm_head : Hidden -> Logits [role=function]

program gru_step(x_t, h_prev) : Vocab * Hidden -> Hidden
    let z = update_gate(x_t, h_prev)
    let r = reset_gate(x_t, h_prev)
    let n = candidate(x_t, r * h_prev)
    let h = (1.0 - z) * n + z * h_prev
    let logits = lm_head(h)
    observe next_token <- Categorical(logits=logits)
    return h

define run_all = scan(gru_step)

program gru_lm : Vocab -> Hidden
    sample last <- run_all
    return last

export gru_lm
```

## Understanding the model

With $x_t$ the one-hot current token and $h_0 = 0$, one step is

$$
z_t = \sigma(W_z [x_t; h_{t-1}] + b_z), \quad
r_t = \sigma(W_r [x_t; h_{t-1}] + b_r), \quad
n_t = \tanh(W_n [x_t; r_t \odot h_{t-1}] + b_n),
$$

$$
h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1},
$$

and the token after position $t$ is scored under $\mathrm{softmax}(U h_t + c)$. An update gate near one copies the previous state forward unchanged, which is how a GRU carries information across many positions; a reset gate near zero lets the candidate ignore the past. Each input row is one sequence, its one-hot tokens the input and its next tokens the observation `next_token`.

## Walkthrough

- `update_gate`, `reset_gate`, and `candidate` are `role=function` morphisms: affine maps of their two inputs with a `sigmoid` or `tanh` output, and point-estimated weights. They have no sample sites.
- `let n = candidate(x_t, r * h_prev)` passes the reset-gated state as the second argument, so the reset gate acts inside the candidate exactly as in the GRU equations.
- `let h = (1.0 - z) * n + z * h_prev` is the elementwise interpolation.
- `observe next_token <- Categorical(logits=logits)` scores the next token at every position; `scan(gru_step)` threads the state from zero.

## Try it

The first block checks the compiled runtime and the reference machine against an independent float64 GRU; the second fits the network by maximum likelihood. Tolerances were fixed before the first run.

### The log joint against a hand-written GRU

The compiled runtime must match the oracle to a relative 1e-5 and the reference machine to 1e-9. The block also prints how far an oracle that ignores the reset gate lies from the runtime. That gap measures whether the agreement check could tell the two cells apart: it is 5.4e-3 (relative) on the first run and 3.9e-4 on the second, so both are tens of times the 1e-5 agreement tolerance, but the second is below the 1e-3 margin fixed in advance for this check. At these initial weights the hidden state is small and the reset gate changes the candidate little, so this page does not claim the 1e-3 margin.

```python
import math

import torch
import torch.distributions as D

from quivers.continuous import LearnedFunction, LinearSource
from quivers.dsl import load

SOURCE = "docs/examples/source/gru_lm.qvr"
VOCAB, HIDDEN = 6, 8


def learned_weights(model):
    found = {}
    for path, module in model.named_modules():
        tail = path.split(".")[-1]
        if not tail.startswith("_function_"):
            continue
        assert isinstance(module, LearnedFunction)
        assert isinstance(module.source, LinearSource)
        name = tail.removeprefix("_function_")
        found[f"{name}_param_weight"] = module.source.linear.weight.detach()
        found[f"{name}_param_bias"] = module.source.linear.bias.detach()
    return found


def oracle(weights, tokens, targets, reset=True):
    """One run's log joint: a GRU loop with a softmax readout, in float64."""
    w = {k: v.double() for k, v in weights.items()}

    def affine(name, value):
        return w[f"{name}_param_weight"] @ value + w[f"{name}_param_bias"]

    h = torch.zeros(HIDDEN, dtype=torch.float64)
    total = 0.0
    for t in range(tokens.shape[0]):
        x = torch.nn.functional.one_hot(tokens[t], VOCAB).double()
        z = torch.sigmoid(affine("update_gate", torch.cat([x, h])))
        r = torch.sigmoid(affine("reset_gate", torch.cat([x, h])))
        gated = r * h if reset else h
        n = torch.tanh(affine("candidate", torch.cat([x, gated])))
        h = (1.0 - z) * n + z * h
        total += float(D.Categorical(logits=affine("lm_head", h)).log_prob(targets[t]))
    return total


def host(value):
    return value.item() if value.dim() == 0 else tuple(host(row) for row in value)


torch.manual_seed(0)
program = load(SOURCE)
model = program.morphism
tokens = torch.randint(0, VOCAB, (2, 7))
targets = torch.randint(0, VOCAB, (2, 7))
x = torch.nn.functional.one_hot(tokens, VOCAB).float()
compiled = model.log_joint(x, {"next_token": targets})
weights = learned_weights(model)
for row in range(2):
    expected = oracle(weights, tokens[row], targets[row])
    without_reset = oracle(weights, tokens[row], targets[row], reset=False)
    got = float(compiled[row].detach())
    run = program.run(
        "gru_lm",
        data={
            "vocab": host(x[row].double()),
            "next_token": host(targets[row]),
            **{k: host(v.double()) for k, v in weights.items()},
        },
        sites={},
        seed=0,
    )
    gap = abs(got - without_reset) / abs(without_reset)
    print(f"run {row}: compiled {got:.6f}, reference {run.log_joint:.9f}, oracle {expected:.9f}; "
          f"reset-free oracle {without_reset:.6f} (relative gap {gap:.1e})")
    assert math.isclose(got, expected, rel_tol=1e-5)
    assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
```

### Maximum-likelihood fitting

The same design as the [vanilla RNN](vanilla-rnn-lm.md#maximum-likelihood-fitting): 8 training runs of length 12 that advance by one token (mod 6), 400 Adam steps, and three checks fixed in advance: per-token NLL below 0.05 nats on the training runs and on 8 held-out runs, and above 1.0 on runs that advance by two.

```python
def per_token_nll(model, sequence):
    x = torch.nn.functional.one_hot(sequence[:, :-1], VOCAB).float()
    with torch.no_grad():
        joint = model.log_joint(x, {"next_token": sequence[:, 1:]}).sum()
    return -float(joint) / sequence[:, 1:].numel()


def rule(starts, length, step):
    return (starts + step * torch.arange(length + 1)) % VOCAB


generator = torch.Generator().manual_seed(3)
runs, length = 8, 12
train = rule(torch.randint(0, VOCAB, (runs, 1), generator=generator), length, 1)
held_out = rule(torch.randint(0, VOCAB, (runs, 1), generator=generator), length, 1)
other = rule(torch.randint(0, VOCAB, (runs, 1), generator=generator), length, 2)

torch.manual_seed(4)
model = load(SOURCE).morphism
x = torch.nn.functional.one_hot(train[:, :-1], VOCAB).float()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for _ in range(400):
    optimizer.zero_grad()
    loss = -model.log_joint(x, {"next_token": train[:, 1:]}).sum()
    loss.backward()
    optimizer.step()

fitted, generalized, contrast = (
    per_token_nll(model, train),
    per_token_nll(model, held_out),
    per_token_nll(model, other),
)
print(f"per-token NLL: training {fitted:.4f}, held out {generalized:.4f}, other rule {contrast:.4f}")
assert fitted < 0.05 and generalized < 0.05
assert contrast > 1.0
```

## Categorical perspective

The gates and the candidate are deterministic morphisms, Dirac kernels in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical], and the step composes them pointwise. [`scan(gru_step)`](../guides/dsl-declarations.md#scan-temporal-recurrence) is the left [fold](https://ncatlab.org/nlab/show/fold) of that step along the sequence; the observation at each position is the only factor that weights a run.

## Sources and further reading

- *Learning phrase representations using RNN encoder-decoder for statistical machine translation* [@cho2014encoder].
- *A categorical approach to probability theory* [@giry1982categorical].
