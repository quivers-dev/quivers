# Bidirectional RNN Language Model

## Overview

A bidirectional recurrent network [@schuster1997bidirectional] reads a sequence twice, once from left to right and once from right to left, so the state it keeps at each position has seen the whole sequence: the forward state the tokens up to the position, the backward state the tokens from it to the end. This source is a bidirectional language model in the sense of [@peters2018deep]: the forward recurrence scores, at every position, the token after it, the backward recurrence scores the token before it, and the program returns the pair of states at each position as that token's contextual representation. Both recurrences are Elman cells with `tanh`, the backward one a [`scan` with `reverse=true`](../guides/dsl-declarations.md#scan-temporal-recurrence), and the weights are draws, so the model is Bayesian over them.

## QVR source

```qvr
# Bidirectional RNN Language Model
#
# Two Elman recurrences read the same token sequence, one from left to
# right and one from right to left. The forward recurrence scores, at
# every position, the token after it; the backward recurrence scores the
# token before it. Their states, paired position by position, are the
# contextual representation of each token: the forward state has read
# the tokens up to the position, the backward state the tokens from it
# to the end.
#
# Generative structure (h_t forward, g_t backward, both starting at 0):
#
#   h_t = tanh(Wf h_{t-1} + Uf[x_t])          forward state
#   next_t ~ Categorical(logits=Vf h_t)       the token after x_t
#   g_t = tanh(Wb g_{t+1} + Ub[x_t])          backward state
#   prev_t ~ Categorical(logits=Vb g_t)       the token before x_t
#   context_t = (h_t, g_t)                    returned, one per position
#
# The weights are draws, so the model is Bayesian over them. Each input
# row is one window of tokens; `next_token` and `prev_token` hold, for
# each position of the window, the token after and before it.
#
# Reference: [Schuster and Paliwal 1997](https://doi.org/10.1109/78.650093).

object Token : FinSet 6
object Hid : FinSet 3
object HidVec : Real 3
object Context : Real 6

program forward_step(x_t, h_prev) : Token * HidVec -> HidVec
    let pre_f = factor d : Hid in sum(factor k : Hid in Wf[d, k] * h_prev[k]) + Uf[x_t, d]
    let h_new = tanh(pre_f)
    let logits_f = factor v : Token in sum(factor d : Hid in Vf[v, d] * h_new[d])
    observe next_token <- Categorical(logits=logits_f)
    return h_new

program backward_step(x_t, g_next) : Token * HidVec -> HidVec
    let pre_b = factor d : Hid in sum(factor k : Hid in Wb[d, k] * g_next[k]) + Ub[x_t, d]
    let g_new = tanh(pre_b)
    let logits_b = factor v : Token in sum(factor d : Hid in Vb[v, d] * g_new[d])
    observe prev_token <- Categorical(logits=logits_b)
    return g_new

define forward_states = scan(forward_step, outputs=all)
define backward_states = scan(backward_step, outputs=all, reverse=true)
define contextual = fan(forward_states, backward_states)

program bidirectional_rnn_lm : Token -> Context
    sample Uf : Token <- Normal(loc=0.0, scale=1.0) [over=Hid, iid_over=Token]
    sample Wf : Hid <- Normal(loc=0.0, scale=0.5) [over=Hid, iid_over=Hid]
    sample Vf : Token <- Normal(loc=0.0, scale=1.0) [over=Hid, iid_over=Token]
    sample Ub : Token <- Normal(loc=0.0, scale=1.0) [over=Hid, iid_over=Token]
    sample Wb : Hid <- Normal(loc=0.0, scale=0.5) [over=Hid, iid_over=Hid]
    sample Vb : Token <- Normal(loc=0.0, scale=1.0) [over=Hid, iid_over=Token]
    sample context <- contextual
    return context

export bidirectional_rnn_lm
```

## Understanding the model

With $x_t$ the token at position $t$ of a window of length $T$, $h_0 = g_{T+1} = 0$, and $U[x]$ the row of $U$ for token $x$, the program scores

$$
h_t = \tanh(W_f h_{t-1} + U_f[x_t]), \qquad g_t = \tanh(W_b g_{t+1} + U_b[x_t]),
$$

$$
\log p = \log p(W, U, V) + \sum_{t=1}^{T} \Bigl( \log \mathrm{softmax}(V_f h_t)_{x_{t+1}} + \log \mathrm{softmax}(V_b g_t)_{x_{t-1}} \Bigr),
$$

with independent Normal priors on the six weight matrices, and returns $(h_t, g_t)$ at every position. The forward scan reads left to right and the backward scan right to left, so each direction's likelihood is a proper autoregressive model of the window in its own direction. The two share nothing but the input.

## Walkthrough

- The weights `Uf`, `Wf`, `Vf` (forward) and `Ub`, `Wb`, `Vb` (backward) are drawn by the exported program and read inside the cells, so they are sample sites with Normal priors.
- `forward_step` computes `h_new = tanh(Wf h_prev + Uf[x_t])` with `factor` comprehensions over the hidden axis and observes `next_token` under `Categorical(logits=Vf h_new)`.
- `backward_step` is the same cell with its own weights, observing `prev_token`.
- `scan(forward_step, outputs=all)` returns the state at every position. `scan(backward_step, outputs=all, reverse=true)` visits the positions from last to first and returns its states in input order. The backward cell's sites are numbered in visit order, so `g_new@0` is the last position.
- `fan(forward_states, backward_states)` lays the two trajectories end to end at each position, giving the `Context : Real 6` that the program returns.
- Each input row is one window. `next_token` and `prev_token` hold, for every position of the window, the token after and before it.

## Try it

The blocks below check the program against an independent float64 implementation of the two recurrences and their likelihoods.

### Shared reference calculation

`oracle` computes one window's log joint, priors included, and its per-position context. Its two keyword switches give deliberately wrong models: a backward recurrence that runs left to right, and one that scores the next token instead of the previous one.

```python
import torch
import torch.distributions as D

from quivers.dsl import load
from quivers.inference import trace

V, H, T = 6, 3, 5
SCALES = {"Uf": 1.0, "Wf": 0.5, "Vf": 1.0, "Ub": 1.0, "Wb": 0.5, "Vb": 1.0}



def check(label, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    assert ok, label


def oracle(w, x, nxt, prv, *, backward_left_to_right=False, backward_scores_next=False):
    """The log joint and the per-position context of one window."""
    total = sum(D.Normal(0.0, SCALES[n]).log_prob(w[n]).sum() for n in SCALES)
    h = torch.zeros(H, dtype=w["Wf"].dtype)
    forward = []
    for t in range(T):
        h = torch.tanh(w["Wf"] @ h + w["Uf"][x[t]])
        total = total + torch.log_softmax(w["Vf"] @ h, -1)[nxt[t]]
        forward.append(h)
    g = torch.zeros(H, dtype=w["Wb"].dtype)
    backward = [g] * T
    order = range(T) if backward_left_to_right else range(T - 1, -1, -1)
    target = nxt if backward_scores_next else prv
    for t in order:
        g = torch.tanh(w["Wb"] @ g + w["Ub"][x[t]])
        total = total + torch.log_softmax(w["Vb"] @ g, -1)[target[t]]
        backward[t] = g
    context = torch.stack([torch.cat([forward[t], backward[t]]) for t in range(T)])
    return total, context


program = load("docs/examples/source/bidirectional_rnn_lm.qvr")
model = program.morphism
generator = torch.Generator().manual_seed(0)
windows = torch.randint(0, V, (3, T + 2), generator=generator)
x, nxt, prv = windows[:, 1:-1], windows[:, 2:], windows[:, :-2]
observations = {"next_token": nxt, "prev_token": prv}

```

### The compiled runtime

At a traced draw of the weights, for three windows: the log joint matches the reference calculation to a relative 1e-5, the returned context matches to 1e-5 absolute, and each wrong reference calculation is more than a relative 1e-3 away, so the first check can see both the direction and the target of the backward recurrence. The separation depends on the draw: at the seeded draw the smallest gap is 1.4e-3, but a draw of the same weights made with float64 as the default dtype puts the left-to-right reference calculation 7.8e-4 from the runtime at window 2, under the margin. The margin thus depends on the parameter draw.

```python
# 1, 2, 4: compiled runtime at a traced draw.
torch.manual_seed(1)
recorded = trace(model, x, observations=observations)
for row in range(3):
    weights = {n: recorded.sites[n].value[row].double() for n in SCALES}
    args = (weights, x[row].tolist(), nxt[row].tolist(), prv[row].tolist())
    expected, context = oracle(*args)
    got = float(recorded.log_joint[row])
    check(
        f"1 compiled joint, window {row}",
        abs(got - float(expected)) <= 1e-5 * abs(float(expected)),
        f"{got:.6f} vs oracle {float(expected):.6f}",
    )
    output = recorded.output
    assert isinstance(output, torch.Tensor)
    error = float((output[row].double() - context).abs().max())
    check(f"2 compiled context, window {row}", error <= 1e-5, f"max abs error {error:.2e}")
    for name, kwargs in (
        ("backward run left to right", {"backward_left_to_right": True}),
        ("backward scores the next token", {"backward_scores_next": True}),
    ):
        wrong, _ = oracle(*args, **kwargs)
        gap = abs(got - float(wrong)) / abs(float(wrong))
        check(f"4 wrong oracle ({name}), window {row}", gap > 1e-3, f"relative gap {gap:.2e}")

```

### The reference machine

At fixed weights, the [reference machine](../developer/qiec.md#reference-evaluator)'s log joint and context match the reference calculation to 1e-9.

```python
# 3: reference machine at fixed weights.
weights = {n: SCALES[n] * torch.randn(V if n[0] in "UV" else H, H, generator=generator, dtype=torch.float64) for n in SCALES}
row = 0
run = program.run(
    "bidirectional_rnn_lm",
    data={
        "token": tuple(x[row].tolist()),
        "next_token": tuple(nxt[row].tolist()),
        "prev_token": tuple(prv[row].tolist()),
    },
    sites={n: tuple(tuple(r) for r in w.tolist()) for n, w in weights.items()},
    seed=0,
)
expected, context = oracle(weights, x[row].tolist(), nxt[row].tolist(), prv[row].tolist())
check(
    "3 reference joint",
    abs(run.log_joint - float(expected)) <= 1e-9 * abs(float(expected)),
    f"{run.log_joint:.12f} vs oracle {float(expected):.12f}",
)
returned = torch.tensor(run.value, dtype=torch.float64)
error = float((returned - context).abs().max())
check("3 reference context", error <= 1e-9, f"max abs error {error:.2e}")

```

### Gradients through the reversed recurrence

The gradient of the compiled log joint with respect to the backward input weights `Ub` matches the reference calculation's autograd gradient (relative 1e-4, absolute 1e-6).

```python
# 5: gradients through the reversed recurrence.
clamped = {n: (weights[n].float().unsqueeze(0)) for n in SCALES}
clamped["Ub"] = clamped["Ub"].clone().requires_grad_(True)
joint = model.log_joint(x[:1], {**clamped, **{k: v[:1] for k, v in observations.items()}})
joint.sum().backward()
reference = {n: weights[n].clone() for n in SCALES}
reference["Ub"].requires_grad_(True)
oracle(reference, x[0].tolist(), nxt[0].tolist(), prv[0].tolist())[0].backward()
grad = clamped["Ub"].grad
assert grad is not None and reference["Ub"].grad is not None
ok = torch.allclose(grad[0].double(), reference["Ub"].grad, rtol=1e-4, atol=1e-6)
check("5 gradient wrt Ub", ok, f"max abs diff {float((grad[0].double() - reference['Ub'].grad).abs().max()):.2e}")
```

## Categorical perspective

A deterministic function defines a [Dirac kernel](https://en.wikipedia.org/wiki/Dirac_measure), which puts all probability at its output. A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical].

`scan(..., outputs=all)` repeatedly updates a state and retains every intermediate result. `reverse=true` traverses the sequence from the other end, and `fan` pairs the outputs of two functions with a common input. Each direction scores its own target array conditional on the inputs. Multiplying these scores does not by itself define a normalized joint distribution over a shared window of tokens. The scored transition is a subprobability [kernel](https://en.wikipedia.org/wiki/Markov_kernel) because each token probability is at most one. It belongs to the extension that permits mass below one, rather than to the probability-kernel category itself [@staton2017commutative].
