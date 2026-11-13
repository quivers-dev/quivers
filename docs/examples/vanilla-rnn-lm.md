# Vanilla RNN Language Model

## Overview

An Elman recurrent network [@elman1990finding] reads a token sequence one position at a time, keeps a hidden state that summarizes the prefix it has read, and predicts the next token from that state. This source states the network exactly: the hidden-state update is a [learned deterministic function](../guides/dsl-declarations.md#function), `h_t = tanh(W [x_t; h_{t-1}] + b)`, a [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) threads it along the sequence, and every position scores its next token under a Categorical whose logits are an affine readout of the state. The weights are point-estimated, so fitting the program by gradient ascent on its log joint is maximum-likelihood training of the language model.

## QVR source

```qvr
# An Elman recurrent language model with learned weights: the hidden state
# is a deterministic function of the previous state and the current token,
# and each position scores the next token.
object Vocab : Real 6
object Hidden : Real 8
object Logits : Real 6

# h_t = tanh(W [x_t; h_{t-1}] + b), with x_t the one-hot current token, so
# the block of W that reads x_t is the token embedding.
morphism cell : Vocab * Hidden -> Hidden [role=function, output=tanh]
# The next token's logits, an affine readout of the hidden state.
morphism lm_head : Hidden -> Logits [role=function]

program rnn_step(x_t, h_prev) : Vocab * Hidden -> Hidden
    let h = cell(x_t, h_prev)
    let logits = lm_head(h)
    observe next_token <- Categorical(logits=logits)
    return h

define run_all = scan(rnn_step)

program rnn_lm : Vocab -> Hidden
    sample last <- run_all
    return last

export rnn_lm
```

## Understanding the model

With $x_t$ the one-hot encoding of the token at position $t$ and $h_0 = 0$, the program scores

$$
h_t = \tanh\bigl(W [x_t; h_{t-1}] + b\bigr), \qquad
\log p(y_{1:T} \mid x_{1:T}) = \sum_{t=1}^{T} \log \mathrm{softmax}(U h_t + c)_{y_t},
$$

where $y_t$ is the token that follows position $t$. Because $x_t$ is one-hot, the columns of $W$ that multiply it are a token embedding, so the network needs no separate embedding layer. Each input row is one run: a whole sequence, with its one-hot tokens as the input and its next tokens as the observation `next_token`, so a batch of $B$ rows is $B$ independent sequences.

## Walkthrough

- `object Vocab : Real 6` is the space of one-hot tokens and `object Logits : Real 6` the space of next-token logits over the same six-token vocabulary; `object Hidden : Real 8` is the hidden state.
- `morphism cell ... [role=function, output=tanh]` is the recurrence: an affine map of the concatenated input and previous state, followed by `tanh`. `role=function` makes it a deterministic function with point-estimated weights rather than a random kernel, so it has no sample site.
- `morphism lm_head ... [role=function]` is the affine readout to logits.
- `program rnn_step` is one position: it updates the state, observes the next token under `Categorical(logits=logits)`, and returns the new state.
- `define run_all = scan(rnn_step)` runs the step along the sequence from a zero initial state; the observation `next_token` is scored at every position, at the sites `next_token@0`, `next_token@1`, and so on.
- `program rnn_lm` is the exported program: one draw from the scan, which is deterministic apart from its observations.

## Try it

The first block checks the compiled runtime and the reference machine against an independent float64 implementation of the Elman recurrence; the second fits the network by maximum likelihood and checks what it has learned. The tolerances in both were fixed before the first run.

### The log joint against a hand-written recurrence

`oracle` recomputes one run's log joint from the learned functions' weights with a plain torch loop. The compiled runtime (float32) must match it to a relative 1e-5 and the reference machine, given the same weights as data, to 1e-9. An oracle that drops the `tanh` must disagree by more than 1e-3, which shows the first comparison can tell the two recurrences apart.

```python
import math

import torch
import torch.distributions as D

from quivers.continuous import LearnedFunction, LinearSource
from quivers.dsl import load

SOURCE = "docs/examples/source/vanilla_rnn_lm.qvr"
VOCAB, HIDDEN = 6, 8


def learned_weights(model):
    """The learned functions' weights under their checked input names."""
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


def oracle(weights, tokens, targets, squash=True):
    """One run's log joint: an Elman loop with a softmax readout, in float64."""
    w = {k: v.double() for k, v in weights.items()}
    h = torch.zeros(HIDDEN, dtype=torch.float64)
    total = 0.0
    for t in range(tokens.shape[0]):
        x = torch.nn.functional.one_hot(tokens[t], VOCAB).double()
        pre = w["cell_param_weight"] @ torch.cat([x, h]) + w["cell_param_bias"]
        h = torch.tanh(pre) if squash else pre
        logits = w["lm_head_param_weight"] @ h + w["lm_head_param_bias"]
        total += float(D.Categorical(logits=logits).log_prob(targets[t]))
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
    wrong = oracle(weights, tokens[row], targets[row], squash=False)
    run = program.run(
        "rnn_lm",
        data={
            "vocab": host(x[row].double()),
            "next_token": host(targets[row]),
            **{k: host(v.double()) for k, v in weights.items()},
        },
        sites={},
        seed=0,
    )
    print(f"run {row}: compiled {float(compiled[row]):.6f}, reference {run.log_joint:.9f}, "
          f"oracle {expected:.9f}, oracle without tanh {wrong:.6f}")
    assert math.isclose(float(compiled[row]), expected, rel_tol=1e-5)
    assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
    assert not math.isclose(float(compiled[row]), wrong, rel_tol=1e-3)
```

### Maximum-likelihood fitting

The training data follow a deterministic rule, token $t + 1$ = token $t$ + 1 (mod 6), on 8 runs of length 12 from random starts. The next token is a function of the current one, so a network that has learned the rule drives the per-token negative log likelihood toward zero. After 400 Adam steps the fitted mean per-token NLL must be below 0.05 nats on the training runs and on 8 held-out runs (a uniform predictor scores $\log 6 \approx 1.79$), and above 1.0 on runs that advance by two, so the fit has learned the rule it was shown rather than a constant.

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

`cell` and `lm_head` are morphisms of the deterministic subcategory of the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical]: each is a Dirac kernel at its output. [`scan(rnn_step)`](../guides/dsl-declarations.md#scan-temporal-recurrence) is the left [fold](https://ncatlab.org/nlab/show/fold) of the step along the sequence, and the only non-deterministic part of the run is the observation at each position, a sub-probability kernel in $\mathcal{G}_{\le 1}$ that weights the run by the likelihood of its next token.

## Sources and further reading

- *Finding structure in time* [@elman1990finding].
- *A categorical approach to probability theory* [@giry1982categorical].
