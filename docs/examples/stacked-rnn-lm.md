# Stacked RNN Language Model

## Overview

A stacked, or deep, recurrent network [@graves2013speech] feeds the whole state sequence of one recurrence to a second recurrence above it, so the upper layer reads a learned representation of each prefix rather than the raw tokens. This source has two Elman layers with `tanh`. The lower [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) returns its state at every position (`outputs=all`), that sequence is what the upper scan runs over, and the upper state at each position scores the token after it. The weights are draws, so the model is Bayesian over them. [@elman1990finding]

## QVR source

```qvr
# Stacked (Two-Layer) RNN Language Model
#
# A deep recurrent language model: a lower Elman recurrence reads the
# tokens, and an upper recurrence reads the lower one's state at every
# position. The lower scan returns every state (`outputs=all`), which is
# the sequence the upper scan runs over; the upper state at each
# position scores the token after it.
#
# Generative structure (both layers start at 0):
#
#   a_t = tanh(W1 a_{t-1} + U1[x_t])            lower state
#   b_t = tanh(W2 b_{t-1} + U2 a_t)             upper state
#   next_t ~ Categorical(logits=V b_t)          the token after x_t
#
# The weights are draws, so the model is Bayesian over them. Each input
# row is one window of tokens; `next_token` holds, for each position of
# the window, the token after it. The program returns the upper state
# after the last position.
#
# Reference: [Graves, Mohamed, and Hinton 2013](https://doi.org/10.1109/ICASSP.2013.6638947).

object Token : FinSet 6
object Hid : FinSet 3
object HidVec : Real 3

program lower_step(x_t, a_prev) : Token * HidVec -> HidVec
    let pre_a = factor d : Hid in sum(factor k : Hid in W1[d, k] * a_prev[k]) + U1[x_t, d]
    let a_new = tanh(pre_a)
    return a_new

program upper_step(a_t, b_prev) : HidVec * HidVec -> HidVec
    let pre_b = factor d : Hid in sum(factor k : Hid in W2[d, k] * b_prev[k] + U2[d, k] * a_t[k])
    let b_new = tanh(pre_b)
    let logits = factor v : Token in sum(factor d : Hid in V[v, d] * b_new[d])
    observe next_token <- Categorical(logits=logits)
    return b_new

define deep = scan(lower_step, outputs=all) >> scan(upper_step)

program stacked_rnn_lm : Token -> HidVec
    sample U1 : Token <- Normal(loc=0.0, scale=1.0) [over=Hid, iid_over=Token]
    sample W1 : Hid <- Normal(loc=0.0, scale=0.5) [over=Hid, iid_over=Hid]
    sample U2 : Hid <- Normal(loc=0.0, scale=0.5) [over=Hid, iid_over=Hid]
    sample W2 : Hid <- Normal(loc=0.0, scale=0.5) [over=Hid, iid_over=Hid]
    sample V : Token <- Normal(loc=0.0, scale=1.0) [over=Hid, iid_over=Token]
    sample top <- deep
    return top

export stacked_rnn_lm
```

## Understanding the model

With $x_t$ the token at position $t$ of a window of length $T$, $a_0 = b_0 = 0$, and $U_1[x]$ the row of $U_1$ for token $x$, the program scores

$$
a_t = \tanh(W_1 a_{t-1} + U_1[x_t]), \qquad b_t = \tanh(W_2 b_{t-1} + U_2 a_t),
$$

$$
\log p = \log p(U_1, W_1, U_2, W_2, V) + \sum_{t=1}^{T} \log \mathrm{softmax}(V b_t)_{x_{t+1}},
$$

with independent Normal priors on the weights, and returns $b_T$. The upper layer at position $t$ reads $a_t$, the lower state at the same position. That is what `outputs=all` provides, since a scan without it returns only its final state.

## Walkthrough

- `lower_step` updates the lower state and observes nothing. `upper_step` reads one lower state, updates the upper state, and observes `next_token`.
- `scan(lower_step, outputs=all)` maps the token sequence to the sequence of lower states, and `>> scan(upper_step)` runs the upper recurrence over it.
- The exported program draws the five weight matrices and runs the composite. Each input row is one window, with `next_token` the token after each position.

## Try it

The blocks check the program against an independent float64 implementation of the two layers.

### Shared reference calculation

`oracle` computes one window's log joint, priors included, and the final upper state. Its keyword switch gives a deliberately wrong model whose upper recurrence reads the lower layer's final state at every position.

```python
import torch
import torch.distributions as D

from quivers.dsl import load
from quivers.inference import trace

V, H, T = 6, 3, 5
SCALES = {"U1": 1.0, "W1": 0.5, "U2": 0.5, "W2": 0.5, "V": 1.0}
SHAPES = {"U1": (V, H), "W1": (H, H), "U2": (H, H), "W2": (H, H), "V": (V, H)}



def check(label, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    assert ok, label


def oracle(w, x, nxt, *, upper_reads_final_lower_state=False):
    """The log joint and the final upper state of one window."""
    total = sum(D.Normal(0.0, SCALES[n]).log_prob(w[n]).sum() for n in SCALES)
    a = torch.zeros(H, dtype=w["W1"].dtype)
    lower = []
    for t in range(T):
        a = torch.tanh(w["W1"] @ a + w["U1"][x[t]])
        lower.append(a)
    b = torch.zeros(H, dtype=w["W2"].dtype)
    for t in range(T):
        read = lower[-1] if upper_reads_final_lower_state else lower[t]
        b = torch.tanh(w["W2"] @ b + w["U2"] @ read)
        total = total + torch.log_softmax(w["V"] @ b, -1)[nxt[t]]
    return total, b


```

### The compiled runtime

At a traced draw of the weights, for three windows, the log joint matches the reference calculation to a relative 1e-5 and the returned upper state to 1e-5 absolute. The block also prints how far the final-lower-state reference calculation lies from the runtime: 1.5e-2 and 2.0e-2 (relative) on windows 0 and 2, and 4.6e-4 on window 1. The specified margin for this comparison was 1e-3, so it fails on window 1: there the prior terms dominate the joint and the likelihood difference between the two models is small. The agreement check on that window still holds at 1e-5, 46 times tighter than the gap, but the 1e-3 margin does not hold on every window.

```python
program = load("docs/examples/source/stacked_rnn_lm.qvr")
model = program.morphism
generator = torch.Generator().manual_seed(0)
windows = torch.randint(0, V, (3, T + 1), generator=generator)
x, nxt = windows[:, :-1], windows[:, 1:]
observations = {"next_token": nxt}

torch.manual_seed(1)
recorded = trace(model, x, observations=observations)
for row in range(3):
    weights = {n: recorded.sites[n].value[row].double() for n in SCALES}
    expected, top = oracle(weights, x[row].tolist(), nxt[row].tolist())
    got = float(recorded.log_joint[row])
    check(
        f"1 compiled joint, window {row}",
        abs(got - float(expected)) <= 1e-5 * abs(float(expected)),
        f"{got:.6f} vs oracle {float(expected):.6f}",
    )
    output = recorded.output
    assert isinstance(output, torch.Tensor)
    error = float((output[row].double() - top).abs().max())
    check(f"2 compiled final state, window {row}", error <= 1e-5, f"max abs error {error:.2e}")
    wrong, _ = oracle(
        weights, x[row].tolist(), nxt[row].tolist(), upper_reads_final_lower_state=True
    )
    gap = abs(got - float(wrong)) / abs(float(wrong))
    # Reported, not asserted: see the prose above this block.
    print(f"window {row}: final-lower-state oracle at relative gap {gap:.2e}")

```

### The reference machine

At fixed weights, the [reference machine](../developer/qiec.md#reference-evaluator)'s log joint and returned state match the reference calculation to 1e-9.

```python
weights = {
    n: SCALES[n] * torch.randn(*SHAPES[n], generator=generator, dtype=torch.float64)
    for n in SCALES
}
run = program.run(
    "stacked_rnn_lm",
    data={"token": tuple(x[0].tolist()), "next_token": tuple(nxt[0].tolist())},
    sites={n: tuple(tuple(r) for r in w.tolist()) for n, w in weights.items()},
    seed=0,
)
expected, top = oracle(weights, x[0].tolist(), nxt[0].tolist())
check(
    "3 reference joint",
    abs(run.log_joint - float(expected)) <= 1e-9 * abs(float(expected)),
    f"{run.log_joint:.12f} vs oracle {float(expected):.12f}",
)
error = float((torch.tensor(run.value, dtype=torch.float64) - top).abs().max())
check("3 reference final state", error <= 1e-9, f"max abs error {error:.2e}")

```

### Gradients through the upper layer

The lower recurrent weights `W1` reach the likelihood only through the upper layer. The gradient of the compiled log joint with respect to them matches the reference calculation's autograd gradient (relative 1e-4, absolute 1e-6).

```python
clamped = {n: weights[n].float().unsqueeze(0) for n in SCALES}
clamped["W1"] = clamped["W1"].clone().requires_grad_(True)
model.log_joint(x[:1], {**clamped, "next_token": nxt[:1]}).sum().backward()
reference = {n: weights[n].clone() for n in SCALES}
reference["W1"].requires_grad_(True)
oracle(reference, x[0].tolist(), nxt[0].tolist())[0].backward()
grad = clamped["W1"].grad
assert grad is not None and reference["W1"].grad is not None
ok = torch.allclose(grad[0].double(), reference["W1"].grad, rtol=1e-4, atol=1e-6)
check("5 gradient wrt W1", ok, f"max abs diff {float((grad[0].double() - reference['W1'].grad).abs().max()):.2e}")
```

## Categorical perspective

A deterministic function defines a [Dirac kernel](https://en.wikipedia.org/wiki/Dirac_measure), which puts all probability at its output. A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical].

`scan(lower_step, outputs=all)` retains the intermediate states of a [fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), so its output is a sequence. The composition operator `>>` passes that sequence to the upper fold. The upper cell scores the observed tokens, multiplying their predicted probabilities into the run weight. The scored transition is a subprobability [kernel](https://en.wikipedia.org/wiki/Markov_kernel) because each token probability is at most one. It belongs to the extension that permits mass below one, rather than to the probability-kernel category itself [@staton2017commutative].
