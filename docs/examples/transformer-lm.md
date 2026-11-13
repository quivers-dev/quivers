# Causal Transformer Language Model

## Overview

A decoder-only Transformer [@vaswani2017attention] scores each next token of a sequence from a representation of the tokens up to it, computed by causal self-attention: every position attends to itself and to the positions before it, never to those after it. This source is one such block, small enough to check by hand. Tokens are embedded and given learned position vectors, two heads of causal scaled dot-product [self-attention](../guides/dsl-declarations.md#sequence-functions-self-attention-and-position-tables) read the whole sequence at once, a position-wise feed-forward network follows, and [layer normalization](https://doi.org/10.48550/arXiv.1607.06450) [@ba2016layer] precedes each sublayer, the pre-LN arrangement [@xiong2020layer], with residual additions around both. Every network is a [learned deterministic function](../guides/dsl-declarations.md#function) with point-estimated weights, so fitting the program's log joint is maximum-likelihood training of the language model.

## QVR source

```qvr
# Causal Transformer Language Model
#
# A one-block, decoder-only Transformer that scores every next token of
# an open-length token sequence. Each position attends to itself and the
# positions before it through two heads of causal scaled dot-product
# self-attention, then passes through a position-wise feed-forward
# block; layer normalization precedes each sublayer and the residual
# stream carries the sum of their outputs.
#
# Generative structure (x_t one-hot, T positions, learned weights):
#
#   e_t = W_e x_t + b_e                         token embedding
#   u_t = e_t + P[t]                            learned position vector
#   h   = u + Attn(LN(u))                       causal self-attention, all t at once
#   z_t = h_t + FF(LN(h_t))                     feed-forward, per position
#   next_t ~ Categorical(logits=W_o LN(z_t) + b_o)
#
# The first and last stages run once per position as scans whose cells
# read no state; attention reads the whole sequence those scans bind.
# Each input row is one sequence, and `next_token` holds, for each of
# its positions, the token after it.
#
# Reference: [Vaswani et al. 2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).

object Vocab : Real 8
object Model : Real 16
object Logits : Real 8

morphism embed : Vocab -> Model [role=function]
morphism positions : Model -> Model [role=function, param_source=positions(32)]
morphism attn : Model -> Model [role=function, param_source=self_attention, heads=2, causal=true]
morphism ff : Model -> Model [role=function, param_source=mlp(32), activation=relu]
morphism lm_head : Model -> Logits [role=function]

program embed_step(x_t, s_prev) : Vocab * Model -> Model
    let e = embed(x_t)
    return e

program head_step(h_t, s_prev) : Model * Model -> Model
    let z = h_t + ff(layer_norm(h_t))
    let logits = lm_head(layer_norm(z))
    observe next_token <- Categorical(logits=logits)
    return z

define embedded = scan(embed_step, outputs=all)
define scored = scan(head_step, outputs=all)

program transformer_lm : Vocab -> Model
    sample e <- embedded
    let u = positions(e)
    let h = u + attn(layer_norm(u))
    sample out <- scored(h)
    return out

export transformer_lm
```

## Understanding the model

With $x_t$ the one-hot token at position $t$ and $\mathrm{LN}$ layer normalization without a learned gain or bias, the block computes

$$
u_t = W_e x_t + b_e + P_t, \qquad
h = u + \mathrm{Attn}(\mathrm{LN}(u)), \qquad
z_t = h_t + \mathrm{FF}(\mathrm{LN}(h_t)),
$$

and scores the token after position $t$ under $\mathrm{softmax}(W_o \mathrm{LN}(z_t) + b_o)$. In each of the two heads, position $i$ forms scores $q_i \cdot k_j / \sqrt{8}$ against the keys of positions $j \le i$, normalizes them by a softmax over those positions only, and takes the weighted sum of their values; the heads' outputs are laid end to end and projected. Because position $i$ reads nothing after it, the logits at position $i$ depend only on the tokens up to $i$, which is what makes it a language model: the next token is scored from its prefix.

Attention is the only stage that mixes positions. The embedding, the feed-forward block, and the readout act on each position separately, and the source runs them as [scans](../guides/dsl-declarations.md#scan-temporal-recurrence) whose cells ignore their state, `scan(..., outputs=all)`, which bind the whole sequence of per-position values. Each input row is one sequence of any length up to 32, the size of the position table.

## Walkthrough

- `embed` and `lm_head` are affine `role=function` morphisms; `ff` is a one-hidden-layer network of width 32 with `relu`, from `param_source=mlp(32)`.
- `positions` adds the `t`-th row of a learned 32-row table to position `t`; `attn` is two-head causal self-attention. Both are sequence functions: they are called in a `let` on the whole table of positions, not one position at a time.
- `embed_step` returns the embedding and ignores `s_prev`, so `define embedded = scan(embed_step, outputs=all)` is the embedding applied at every position.
- `let h = u + attn(layer_norm(u))` is the attention sublayer with its residual; `layer_norm` normalizes each row.
- `head_step` applies the feed-forward sublayer with its residual, the final layer norm, and the readout to one position, and observes `next_token` there; `scored(h)` runs it along `h`.
- Every weight is a point estimate: the program has no sample site, and its log joint is the log likelihood of the next tokens.
- No transpile target lowers a scan, so every target refuses this program with `scan:no-lowering`. The program runs on the PyTorch runtime and the reference machine.

## Try it

The checks below come from a verification script whose tolerances were fixed before it was first run. The first block checks the log joint against an independent implementation, the second checks causality, and the third fits the model on a task that needs attention.

### The log joint against a hand-written Transformer

`oracle` recomputes one sequence's log joint from the compiled weights with plain Python loops over positions, heads, and coordinates, in float64. The weights are first scrambled to standard deviation 0.5, so that every sublayer moves the result. On sequences of lengths 1, 5, 9, and 12 and on a batch of two runs of length 7, the compiled runtime (float32) must match the oracle to a relative 1e-5 and the reference machine, given the same weights as data, to 1e-9. Three wrong oracles must each differ from the right one by more than 1e-5 on every sequence: one without the causal mask, one whose mask lets each position see one position ahead, and one without position vectors. On a single position every mask is the same mask, so there only the position check applies.

```python
import math

import torch
import torch.distributions as D

from quivers.dsl import load, loads
from quivers.inference import trace

SOURCE = "docs/examples/source/transformer_lm.qvr"
VOCAB, MODEL, HEADS = 8, 16, 2


def causal(i, j):
    return j <= i


def unmasked(i, j):
    return True


def one_ahead(i, j):
    return j <= i + 1


def module(model, name):
    found = {m for p, m in model.named_modules() if p.split(".")[-1] == f"_function_{name}"}
    assert len(found) == 1
    return next(iter(found))


def weights(model):
    """Every weight under its checked input name, detached, float64."""
    w = {}
    for name in ("embed", "lm_head"):
        linear = module(model, name).source.linear
        w[f"{name}_param_weight"], w[f"{name}_param_bias"] = linear.weight, linear.bias
    net = module(model, "ff").source.net
    w["ff_param_layer0_weight"], w["ff_param_layer0_bias"] = net[0].weight, net[0].bias
    w["ff_param_weight"], w["ff_param_bias"] = net[2].weight, net[2].bias
    attention = module(model, "attn").source
    for role in ("query", "key", "value"):
        w[f"attn_param_{role}_weight"] = getattr(attention, role).weight
        w[f"attn_param_{role}_bias"] = getattr(attention, role).bias
    w["attn_param_weight"], w["attn_param_bias"] = attention.output.weight, attention.output.bias
    w["positions_param_positions"] = module(model, "positions").source.table
    return {k: v.detach().double() for k, v in w.items()}


def layer_norm(v):
    mean = sum(v) / len(v)
    var = sum((a - mean) ** 2 for a in v) / len(v)
    return [(a - mean) / math.sqrt(var + 1e-5) for a in v]


def affine(weight, bias, v):
    return [sum(float(weight[r, c]) * v[c] for c in range(len(v))) + float(bias[r]) for r in range(weight.shape[0])]


def oracle(w, tokens, targets, mask=causal, use_positions=True):
    """One sequence's log joint, by hand, with loops over positions."""
    length = len(tokens)
    u = []
    for t, token in enumerate(tokens):
        e = affine(w["embed_param_weight"], w["embed_param_bias"], [1.0 if k == token else 0.0 for k in range(VOCAB)])
        if use_positions:
            e = [a + float(w["positions_param_positions"][t, k]) for k, a in enumerate(e)]
        u.append(e)
    normed = [layer_norm(row) for row in u]
    q = [affine(w["attn_param_query_weight"], w["attn_param_query_bias"], r) for r in normed]
    k = [affine(w["attn_param_key_weight"], w["attn_param_key_bias"], r) for r in normed]
    v = [affine(w["attn_param_value_weight"], w["attn_param_value_bias"], r) for r in normed]
    width = MODEL // HEADS
    context = [[0.0] * MODEL for _ in range(length)]
    for g in range(HEADS):
        block = range(g * width, (g + 1) * width)
        for i in range(length):
            seen = [j for j in range(length) if mask(i, j)]
            scores = {j: sum(q[i][c] * k[j][c] for c in block) / math.sqrt(width) for j in seen}
            peak = max(scores.values())
            total = sum(math.exp(s - peak) for s in scores.values())
            for c in block:
                context[i][c] = sum(math.exp(scores[j] - peak) / total * v[j][c] for j in seen)
    joint = 0.0
    for t in range(length):
        attended = affine(w["attn_param_weight"], w["attn_param_bias"], context[t])
        h = [a + b for a, b in zip(u[t], attended)]
        hidden = [max(0.0, a) for a in affine(w["ff_param_layer0_weight"], w["ff_param_layer0_bias"], layer_norm(h))]
        z = [a + b for a, b in zip(h, affine(w["ff_param_weight"], w["ff_param_bias"], hidden))]
        logits = affine(w["lm_head_param_weight"], w["lm_head_param_bias"], layer_norm(z))
        peak = max(logits)
        joint += logits[targets[t]] - (peak + math.log(sum(math.exp(a - peak) for a in logits)))
    return joint


def host(value):
    return float(value) if value.dim() == 0 else tuple(host(row) for row in value)


def onehot(tokens):
    return torch.nn.functional.one_hot(tokens, VOCAB).float()


def scramble(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.5)


program = load(SOURCE)
model = program.morphism
scramble(model, seed=0)
w = weights(model)
generator = torch.Generator().manual_seed(1)
for runs, length in [(1, 1), (1, 5), (1, 9), (1, 12), (2, 7)]:
    tokens = torch.randint(0, VOCAB, (runs, length), generator=generator)
    targets = torch.randint(0, VOCAB, (runs, length), generator=generator)
    compiled = model.log_joint(onehot(tokens), {"next_token": targets}).detach()
    for r in range(runs):
        truth = oracle(w, tokens[r].tolist(), targets[r].tolist())
        run = program.run(
            "transformer_lm",
            data={
                "vocab": host(onehot(tokens[r])),
                "next_token": tuple(int(t) for t in targets[r]),
                **{k: host(v) for k, v in w.items()},
            },
            sites={},
            seed=0,
        )
        gaps = {}
        for label, kwargs in (
            ("no mask", {"mask": unmasked}),
            ("mask one ahead", {"mask": one_ahead}),
            ("no positions", {"use_positions": False}),
        ):
            if length == 1 and label != "no positions":
                continue
            gaps[label] = abs(oracle(w, tokens[r].tolist(), targets[r].tolist(), **kwargs) - truth) / abs(truth)
        print(f"T={length} run {r}: compiled {float(compiled[r]):.6f}, reference {run.log_joint:.9f}, "
              f"oracle {truth:.9f}; wrong-oracle gaps " + ", ".join(f"{k} {v:.1e}" for k, v in gaps.items()))
        assert math.isclose(float(compiled[r]), truth, rel_tol=1e-5)
        assert math.isclose(run.log_joint, truth, rel_tol=1e-9)
        assert all(gap > 1e-5 for gap in gaps.values())
```

### Causality

The mask is checked on the compiled runtime directly. Changing the tokens after position 4 of an eight-token sequence must leave the logits at positions 0 through 4 bit-for-bit unchanged, and must change the logits at positions 5 through 7. The logits at position `t` are recorded at the site `logits@t`.

```python
scramble(model, seed=2)
tokens = torch.tensor([[0, 3, 2, 5, 1, 7, 4, 6]])
later = tokens.clone()
later[0, 5:] = torch.tensor([2, 2, 2])
targets = torch.zeros_like(tokens)
a = trace(model, onehot(tokens), observations={"next_token": targets})
b = trace(model, onehot(later), observations={"next_token": targets})
unchanged = [torch.equal(a.sites[f"logits@{t}"].value, b.sites[f"logits@{t}"].value) for t in range(8)]
print("logits unchanged by a change after position 4:", unchanged)
assert all(unchanged[:5]) and not any(unchanged[5:])
```

### Maximum-likelihood fitting on a task that needs attention

Each training sequence repeats a random three-token pattern, so from position 2 on the next token is the token two positions back. A model that scores one position from that position's own token and place cannot predict it; a model that attends to earlier positions can. The model is fitted by Adam (learning rate 0.01) for 400 steps, each on a fresh batch of 24 sequences of length 9, and the check, fixed in advance, is the mean negative log likelihood over positions 2 and on of 32 held-out sequences: below 0.1 nats for the model, and above 1.0 nats for the same program with the line `let h = u + attn(layer_norm(u))` replaced by `let h = u`, fitted the same way. A uniform predictor scores $\log 8 \approx 2.08$.

The first design of this experiment failed its check, and the experiment was changed before this one was run. It fitted a fixed set of 24 sequences for 300 steps. The model memorized them (training NLL 0.14 nats per token) and did worse than chance on held-out sequences, 2.78 nats against the 0.1 threshold, while the control passed at 5.93. Drawing a fresh batch at every step, so that no sequence is seen twice, is the change; the thresholds are those of the first design. Under it the model scores 0.0028 nats on the held-out positions and the control 2.10. Fitting both took about seventeen minutes on the machine the check was run on, since the runtime evaluates each sequence's scans separately.

```python
def corpus(count, length, generator):
    pattern = torch.randint(0, VOCAB, (count, 3), generator=generator)
    stream = pattern.repeat(1, (length + 1) // 3 + 1)[:, : length + 1]
    return stream[:, :-1], stream[:, 1:]


def fit(model, seed, steps=400):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        tokens, targets = corpus(24, 9, generator)
        optimizer.zero_grad()
        loss = -model.log_joint(onehot(tokens), {"next_token": targets}).sum() / targets.numel()
        loss.backward()
        optimizer.step()
    return model


def held_out_nll(model, data):
    tokens, targets = data
    with torch.no_grad():
        recorded = trace(model, onehot(tokens), observations={"next_token": targets})
    total = 0.0
    for t in range(2, tokens.shape[1]):
        logits = recorded.sites[f"logits@{t}"].value
        total -= float(D.Categorical(logits=logits).log_prob(targets[:, t]).sum())
    return total / (tokens.shape[0] * (tokens.shape[1] - 2))


held_out = corpus(32, 9, torch.Generator().manual_seed(4))
source = open(SOURCE).read()
control_source = source.replace("    let h = u + attn(layer_norm(u))\n", "    let h = u\n")
assert control_source != source

torch.manual_seed(3)
attending = fit(loads(source).morphism, seed=5)
torch.manual_seed(3)
control = fit(loads(control_source).morphism, seed=5)
with_attention, without_attention = held_out_nll(attending, held_out), held_out_nll(control, held_out)
print(f"held-out NLL from position 2: with attention {with_attention:.4f}, without {without_attention:.4f}")
assert with_attention < 0.1
assert without_attention > 1.0
```

## Categorical perspective

Every stage is a deterministic morphism, a Dirac kernel in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical]. The per-position stages are maps of a single position lifted to sequences, which is what a scan whose cell ignores its state computes. Self-attention is not such a lift: it is a map on sequences whose value at position $i$ factors through the prefix ending at $i$, and the causality check above is a direct test of that factorization. The observation at each position is the only factor that weights a run.

## Sources and further reading

- *Attention is all you need* [@vaswani2017attention].
- *On layer normalization in the Transformer architecture* [@xiong2020layer].
- *Layer normalization* [@ba2016layer].
- *A categorical approach to probability theory* [@giry1982categorical].
