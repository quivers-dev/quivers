# LSTM Language Model

## Overview

A long short-term memory (LSTM) network [@hochreiter1997lstm] keeps two states, a hidden state $h_t$ and a cell state $c_t$, and controls them with three gates: the input gate decides how much of a new candidate enters the cell, the forget gate [@gers2000learning] how much of the old cell survives, and the output gate how much of the cell reaches the hidden state. This source states that recurrence exactly. The gates and the candidate are [learned deterministic functions](../guides/dsl-declarations.md#function), the two states travel together through the [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) packed into one vector with `concat` and split again with `slice`, and every position scores the next token. The weights are point-estimated, so fitting the log joint is maximum-likelihood training.

## QVR source

```qvr
# A long short-term memory (LSTM) language model with learned weights. The
# cell carries two states, the hidden state h_t and the cell state c_t,
# packed into one State vector that the scan threads from position to
# position. The gates and the cell candidate are learned functions of the
# current token and h_{t-1}; each position scores the token after it.
object Vocab : Real 6
object Hidden : Real 8
object State : Real 16
object Logits : Real 6

# i_t, f_t, o_t = sigmoid(W [x_t; h_{t-1}] + b), the input, forget and
# output gates.
morphism input_gate : Vocab * Hidden -> Hidden [role=function, output=sigmoid]
morphism forget_gate : Vocab * Hidden -> Hidden [role=function, output=sigmoid]
morphism output_gate : Vocab * Hidden -> Hidden [role=function, output=sigmoid]
# g_t = tanh(W_g [x_t; h_{t-1}] + b_g), the cell candidate.
morphism candidate : Vocab * Hidden -> Hidden [role=function, output=tanh]
morphism lm_head : Hidden -> Logits [role=function]

program lstm_step(x_t, state) : Vocab * State -> State
    let h_prev = slice(state, 0, 8)
    let c_prev = slice(state, 8, 8)
    let i = input_gate(x_t, h_prev)
    let f = forget_gate(x_t, h_prev)
    let o = output_gate(x_t, h_prev)
    let g = candidate(x_t, h_prev)
    let c = f * c_prev + i * g
    let h = o * tanh(c)
    let logits = lm_head(h)
    observe next_token <- Categorical(logits=logits)
    let state_new = concat(h, c)
    return state_new

define run_all = scan(lstm_step)

program lstm_lm : Vocab -> State
    sample last <- run_all
    return last

export lstm_lm
```

## Understanding the model

With $x_t$ the one-hot current token, $h_0 = c_0 = 0$ and $\sigma$ the logistic function, one step is

$$
i_t = \sigma(W_i [x_t; h_{t-1}] + b_i), \quad f_t = \sigma(W_f [x_t; h_{t-1}] + b_f), \quad o_t = \sigma(W_o [x_t; h_{t-1}] + b_o),
$$

$$
g_t = \tanh(W_g [x_t; h_{t-1}] + b_g), \quad c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t),
$$

and the token after position $t$ is scored under $\mathrm{softmax}(U h_t + d)$. A forget gate near one and an input gate near zero carry the cell state forward unchanged, which is how the network holds information across many positions. Each input row is one sequence, its one-hot tokens the input and its next tokens the observation `next_token`.

## Walkthrough

- `object State : Real 16` holds the pair $(h_t, c_t)$: `slice(state, 0, 8)` is the hidden state and `slice(state, 8, 8)` the cell state, and `concat(h, c)` packs the new pair for the next position.
- `input_gate`, `forget_gate`, `output_gate` and `candidate` are `role=function` morphisms with `sigmoid` or `tanh` outputs, all reading the current token and $h_{t-1}$.
- `let c = f * c_prev + i * g` and `let h = o * tanh(c)` are the cell and hidden updates; `tanh` is the checked elementwise builtin.
- `observe next_token <- Categorical(logits=logits)` scores the next token at every position; `scan(lstm_step)` threads the packed state from zero.

## Try it

Both blocks use tolerances fixed before the first run.

### The log joint against a hand-written LSTM

The compiled runtime must match an independent float64 LSTM to a relative 1e-5 and the reference machine to 1e-9. An oracle without the forget gate ($c_t = c_{t-1} + i_t g_t$) must lie more than a relative 1e-3 from the runtime, so the first check can tell the two cells apart; it lies 9.2e-3 and 6.2e-3 away on the two runs.

```python
import math

import torch
import torch.distributions as D

from quivers.continuous import LearnedFunction, LinearSource
from quivers.dsl import load

SOURCE = "docs/examples/source/lstm_lm.qvr"
VOCAB, HIDDEN = 6, 8


def weights_of(model):
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


def oracle(weights, tokens, targets, forget=True):
    w = {k: v.double() for k, v in weights.items()}

    def affine(name, value):
        return w[f"{name}_param_weight"] @ value + w[f"{name}_param_bias"]

    h = torch.zeros(HIDDEN, dtype=torch.float64)
    c = torch.zeros(HIDDEN, dtype=torch.float64)
    total = 0.0
    for t in range(tokens.shape[0]):
        x = torch.nn.functional.one_hot(tokens[t], VOCAB).double()
        xh = torch.cat([x, h])
        i = torch.sigmoid(affine("input_gate", xh))
        f = torch.sigmoid(affine("forget_gate", xh)) if forget else torch.ones(HIDDEN, dtype=torch.float64)
        o = torch.sigmoid(affine("output_gate", xh))
        g = torch.tanh(affine("candidate", xh))
        c = f * c + i * g
        h = o * torch.tanh(c)
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
weights = weights_of(model)
for row in range(2):
    expected = oracle(weights, tokens[row], targets[row])
    without_forget = oracle(weights, tokens[row], targets[row], forget=False)
    got = float(compiled[row].detach())
    run = program.run(
        "lstm_lm",
        data={"vocab": host(x[row].double()), "next_token": host(targets[row]),
              **{k: host(v.double()) for k, v in weights.items()}},
        sites={},
        seed=0,
    )
    gap = abs(got - without_forget) / abs(without_forget)
    print(f"run {row}: compiled {got:.6f}, reference {run.log_joint:.9f}, oracle {expected:.9f}; "
          f"forget-free oracle {without_forget:.6f} (relative gap {gap:.1e})")
    assert math.isclose(got, expected, rel_tol=1e-5)
    assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
    assert gap > 1e-3
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
    (-model.log_joint(x, {"next_token": train[:, 1:]}).sum()).backward()
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

The gates and the candidate are deterministic morphisms, Dirac kernels in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical]. Packing $(h, c)$ with `concat` makes the step an endomorphism of one state space, so [`scan(lstm_step)`](../guides/dsl-declarations.md#scan-temporal-recurrence) is the left [fold](https://ncatlab.org/nlab/show/fold) of a single morphism along the sequence; the observations are the only factors that weight a run.

## Sources and further reading

- *Long short-term memory* [@hochreiter1997lstm].
- *Learning to forget: continual prediction with LSTM* [@gers2000learning].
- *A categorical approach to probability theory* [@giry1982categorical].
