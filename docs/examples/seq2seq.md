# Sequence-to-Sequence Transformer

## Overview

A sequence-to-sequence model scores a target sequence given a source sequence [@sutskever2014sequence]. In the encoder-decoder Transformer [@vaswani2017attention] an encoder attends across the whole source, a decoder attends across the target positions up to its own under a causal mask, and cross-attention lets each target position read the encoder's output before a readout scores the target token there. This source states one such layer of each kind over fixed windows, a source of five positions and a target of four, with every weight a sample site under a Normal prior, so the model is Bayesian over its weights. The attention layers are [`factor` comprehensions](../reference/qvr/types-and-indexed-families.md#pure-expressions) over the declared position objects. The page checks the program's joint density against a hand-written Transformer on both execution paths and checks the decoder's causality. It fits nothing.

## QVR source

```qvr
# Sequence-to-Sequence Transformer over Fixed Windows
#
# An encoder-decoder with attention, after Vaswani et al., over a source
# window of SrcPos positions and a target window of TgtPos positions.
# The encoder attends across the whole source; the decoder attends to
# the target positions up to its own under a causal mask, then attends
# across the encoder's output; a readout tied to the token embedding
# scores the target token at every target position. Every weight is a
# sample site with a Normal prior, so the model is Bayesian over them.
#
# Generative structure (one head of width 4 per attention, scores
# scaled by 1 / sqrt(4) = 0.5):
#
#   X = E[source] + P_src                        encoder input
#   H = X + Attn(X, X)                           encoder self-attention
#   Y = E[previous] + P_tgt                      decoder input
#   G = Y + CausalAttn(Y, Y)                     decoder self-attention
#   O = G + Attn(G, H)                           cross-attention
#   target[p] ~ Categorical(logits = O[p] E^T)   tied readout
#
# The tokens arrive as data: `source` holds the source tokens,
# `previous` the target tokens shifted right by one behind a start
# token, and `target` is observed. A causal mask entry is 1 exactly when
# j <= i, from the integer clamp min(1, max(0, i - j + 1)).
#
# Reference: [Vaswani et al. 2017](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).

object Token : FinSet 6
object SrcPos : FinSet 5
object TgtPos : FinSet 4
object Emb : FinSet 4

program seq2seq : TgtPos -> TgtPos
    sample E : Token <- Normal(loc=0.0, scale=1.0) [over=Emb, iid_over=Token]
    sample P_src : SrcPos <- Normal(loc=0.0, scale=1.0) [over=Emb, iid_over=SrcPos]
    sample P_tgt : TgtPos <- Normal(loc=0.0, scale=1.0) [over=Emb, iid_over=TgtPos]
    sample Wq_enc : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wk_enc : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wv_enc : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wq_dec : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wk_dec : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wv_dec : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wq_cross : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wk_cross : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]
    sample Wv_cross : Emb <- Normal(loc=0.0, scale=0.5) [over=Emb, iid_over=Emb]

    let X = factor p : SrcPos, k : Emb in E[int(source[p]), k] + P_src[p, k]
    let Q_enc = factor p : SrcPos, d : Emb in sum(factor k : Emb in X[p, k] * Wq_enc[k, d])
    let K_enc = factor p : SrcPos, d : Emb in sum(factor k : Emb in X[p, k] * Wk_enc[k, d])
    let V_enc = factor p : SrcPos, d : Emb in sum(factor k : Emb in X[p, k] * Wv_enc[k, d])
    let S_enc = factor i : SrcPos, j : SrcPos in 0.5 * sum(factor d : Emb in Q_enc[i, d] * K_enc[j, d])
    let A_enc = softmax(S_enc)
    let H = factor i : SrcPos, d : Emb in X[i, d] + sum(factor j : SrcPos in A_enc[i, j] * V_enc[j, d])

    let Y = factor p : TgtPos, k : Emb in E[int(previous[p]), k] + P_tgt[p, k]
    let Q_dec = factor p : TgtPos, d : Emb in sum(factor k : Emb in Y[p, k] * Wq_dec[k, d])
    let K_dec = factor p : TgtPos, d : Emb in sum(factor k : Emb in Y[p, k] * Wk_dec[k, d])
    let V_dec = factor p : TgtPos, d : Emb in sum(factor k : Emb in Y[p, k] * Wv_dec[k, d])
    let mask = factor i : TgtPos, j : TgtPos in real(min(1, max(0, i - j + 1)))
    let S_dec = factor i : TgtPos, j : TgtPos in 0.5 * sum(factor d : Emb in Q_dec[i, d] * K_dec[j, d]) - 1000000.0 * (1.0 - mask[i, j])
    let A_dec = softmax(S_dec)
    let G = factor i : TgtPos, d : Emb in Y[i, d] + sum(factor j : TgtPos in A_dec[i, j] * V_dec[j, d])

    let Q_cross = factor p : TgtPos, d : Emb in sum(factor k : Emb in G[p, k] * Wq_cross[k, d])
    let K_cross = factor p : SrcPos, d : Emb in sum(factor k : Emb in H[p, k] * Wk_cross[k, d])
    let V_cross = factor p : SrcPos, d : Emb in sum(factor k : Emb in H[p, k] * Wv_cross[k, d])
    let S_cross = factor i : TgtPos, j : SrcPos in 0.5 * sum(factor d : Emb in Q_cross[i, d] * K_cross[j, d])
    let A_cross = softmax(S_cross)
    let O = factor i : TgtPos, d : Emb in G[i, d] + sum(factor j : SrcPos in A_cross[i, j] * V_cross[j, d])

    let logits = factor p : TgtPos, t : Token in sum(factor d : Emb in O[p, d] * E[t, d])
    observe target : TgtPos <- Categorical(logits=logits)
    return target

export seq2seq
```

## Understanding the model

With $E$ the token embedding, $P_{\mathrm{src}}$ and $P_{\mathrm{tgt}}$ learned position vectors, and $\mathrm{Attn}(Q, K, V) = \mathrm{softmax}(Q K^\top / 2) V$ for one head of width 4, the program computes

$$
\begin{aligned}
X &= E[\mathrm{source}] + P_{\mathrm{src}}, & H &= X + \mathrm{Attn}(X W^q_{e}, X W^k_{e}, X W^v_{e}), \\
Y &= E[\mathrm{previous}] + P_{\mathrm{tgt}}, & G &= Y + \mathrm{Attn}_{\le}(Y W^q_{d}, Y W^k_{d}, Y W^v_{d}), \\
O &= G + \mathrm{Attn}(G W^q_{c}, H W^k_{c}, H W^v_{c}), & \mathrm{target}_p &\sim \mathrm{Categorical}(\mathrm{logits} = O_p E^\top),
\end{aligned}
$$

where $\mathrm{Attn}_{\le}$ lets target position $i$ attend only to positions $j \le i$. The decoder reads `previous`, the target shifted right by one behind a start token, so the token it scores at position $p$ is never among the inputs it can attend to: the log probability of the target factors over positions given the source and the target's prefix. The readout reuses the embedding $E$, a tied output layer. The causal mask subtracts $10^6$ from the scores of the positions after $i$, whose softmax weight is then exactly zero in floating point.

Every weight carries a prior, `Normal(0, 1)` for the embeddings and positions and `Normal(0, 0.5)` for the nine projections, so the program's log joint is the priors plus the target's log likelihood, and the data are one (source, target) pair per run. A corpus of pairs would need a plate over pairs so that the pairs share one draw of the weights; this source scores a single pair.

## Walkthrough

- `object Token : FinSet 6`, `object SrcPos : FinSet 5`, `object TgtPos : FinSet 4`, and `object Emb : FinSet 4` are the vocabulary, the two windows, and the embedding width. Because the windows are declared objects, every table is a comprehension over them.
- The twelve `sample` steps draw the embedding, the two position tables, and the query, key, and value projections of the three attention layers.
- `int(source[p])` reads the token at source position `p` from the data and indexes the embedding with it; `previous` and `target` are data in the same way.
- `softmax` acts along the last axis, so `softmax(S_enc)` normalizes each query position's scores over the key positions.
- `mask` is 1 exactly when `j <= i`, from the integer clamp `min(1, max(0, i - j + 1))`.
- `observe target : TgtPos <- Categorical(logits=logits)` scores the target token at every target position.
- The program transpiles to Pyro, NumPyro, PyMC, and Edward2. Stan refuses the row-wise `softmax` of a comprehension, and the other targets refuse `Categorical(logits=...)`. This page does not check that those four emissions score the same density as the program.

## Try it

The checks below come from a verification script whose tolerances were fixed before it was first run. Two of its first runs failed for defects in the script, not in the program, and were fixed without changing a tolerance: the oracle's prior used Python-float scales, which torch turns into float32 tensors, so its log density was off by about 3e-8 per weight matrix and the reference-machine comparison failed at a relative 1.3e-9; and the causality check compared the observed site's log probability, which is the sum over positions, rather than the per-position log probabilities.

### The log joint against a hand-written Transformer

At three traced draws of the weights, each with its own random source and target, the compiled runtime's log joint must equal an oracle written with torch matrix products and an explicit softmax, in float64, to a relative 1e-5, and the reference machine, given the same weights as sites, to 1e-9. Three wrong oracles must each differ by more than a relative 1e-3 at every draw: the decoder without its causal mask, the model without cross-attention, and cross-attention whose queries read the decoder input `Y` rather than its state `G`.

```python
import math

import torch
import torch.distributions as D

from quivers.dsl import load
from quivers.inference import trace

SOURCE = "docs/examples/source/seq2seq.qvr"
TOKENS, SRC, TGT, EMB = 6, 5, 4, 4
WEIGHTS = (
    "E", "P_src", "P_tgt",
    "Wq_enc", "Wk_enc", "Wv_enc",
    "Wq_dec", "Wk_dec", "Wv_dec",
    "Wq_cross", "Wk_cross", "Wv_cross",
)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scores = 0.5 * q @ k.T
    if causal:
        allowed = torch.tril(torch.ones(scores.shape, dtype=torch.bool))
        scores = scores.masked_fill(~allowed, -math.inf)
    weights = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
    return (weights / weights.sum(dim=-1, keepdim=True)) @ v


def per_position(
    w: dict[str, torch.Tensor],
    source: list[int],
    previous: list[int],
    target: list[int],
    variant: str = "",
) -> torch.Tensor:
    """Each target position's log probability, float64."""
    x = w["E"][source] + w["P_src"]
    h = x + attend(x @ w["Wq_enc"], x @ w["Wk_enc"], x @ w["Wv_enc"], causal=False)
    y = w["E"][previous] + w["P_tgt"]
    g = y + attend(y @ w["Wq_dec"], y @ w["Wk_dec"], y @ w["Wv_dec"], causal=variant != "unmasked")
    if variant == "no_cross":
        o = g
    else:
        reader = y if variant == "cross_reads_input" else g
        o = g + attend(reader @ w["Wq_cross"], h @ w["Wk_cross"], h @ w["Wv_cross"], causal=False)
    logits = o @ w["E"].T
    return D.Categorical(logits=logits).log_prob(torch.tensor(target))


def oracle(w: dict[str, torch.Tensor], source: list[int], previous: list[int], target: list[int], variant: str = "") -> float:
    # The scales are float64 tensors: a Python float scale makes a float32
    # one, whose log is off by about 3e-8 per matrix (the first run of this
    # script failed check 2 at rel 1.3e-9 for that reason alone).
    prior = sum(
        float(
            D.Normal(
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.0 if name in ("E", "P_src", "P_tgt") else 0.5, dtype=torch.float64),
            ).log_prob(w[name]).sum()
        )
        for name in WEIGHTS
    )
    return prior + float(per_position(w, source, previous, target, variant).sum())


def host(value: torch.Tensor) -> object:
    return float(value) if value.dim() == 0 else tuple(host(row) for row in value)


def data(source: list[int], previous: list[int], target: list[int]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.tensor(source, dtype=torch.float32),
        "previous": torch.tensor(previous, dtype=torch.float32),
        "target": torch.tensor(target),
    }


program = load(SOURCE)
generator = torch.Generator().manual_seed(0)
for draw in range(3):
    source = torch.randint(1, TOKENS, (SRC,), generator=generator).tolist()
    target = torch.randint(1, TOKENS, (TGT,), generator=generator).tolist()
    previous = [0, *target[:-1]]
    torch.manual_seed(10 + draw)
    recorded = trace(program.morphism, torch.zeros(1, 1), observations=data(source, previous, target))
    w = {name: recorded.sites[name].value.reshape(recorded.sites[name].value.shape[-2:]).double() for name in WEIGHTS}
    expected = oracle(w, source, previous, target)
    got = float(recorded.log_joint)
    run = program.run(
        "seq2seq",
        data={
            "source": tuple(float(t) for t in source),
            "previous": tuple(float(t) for t in previous),
            "target": tuple(target),
        },
        sites={name: host(w[name]) for name in WEIGHTS},
        seed=0,
    )
    wrong = {v: oracle(w, source, previous, target, v) for v in ("unmasked", "no_cross", "cross_reads_input")}
    print(f"draw {draw}: compiled {got:.6f}, reference {run.log_joint:.9f}, oracle {expected:.9f}; "
          + ", ".join(f"{k} {v:.6f}" for k, v in wrong.items()))
    assert math.isclose(got, expected, rel_tol=1e-5)
    assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
    assert all(not math.isclose(got, value, rel_tol=1e-3) for value in wrong.values())
```

### Causality of the decoder

At a fixed draw of the weights, changing the decoder's inputs at positions 2 and 3 must leave the target log probabilities at positions 0 and 1 unchanged to within 1e-6, and must change those at positions 2 and 3. The per-position log probabilities are read from the logits the program records at the site `logits`.

```python
source, target = [1, 3, 2, 5, 4], [4, 5, 2, 3]
previous = [0, *target[:-1]]
changed = [0, 4, 1, 1]
torch.manual_seed(20)
recorded = trace(program.morphism, torch.zeros(1, 1), observations=data(source, previous, target))
sites = {name: recorded.sites[name].value for name in WEIGHTS}
a = trace(program.morphism, torch.zeros(1, 1), observations={**data(source, previous, target), **sites})
b = trace(program.morphism, torch.zeros(1, 1), observations={**data(source, changed, target), **sites})
picked = torch.tensor(target)
lp_a = torch.log_softmax(a.sites["logits"].value.reshape(TGT, TOKENS), -1)[torch.arange(TGT), picked]
lp_b = torch.log_softmax(b.sites["logits"].value.reshape(TGT, TOKENS), -1)[torch.arange(TGT), picked]
print("target log probabilities:", [round(float(v), 6) for v in lp_a])
print("after the change:        ", [round(float(v), 6) for v in lp_b])
assert torch.allclose(lp_a[:2], lp_b[:2], atol=1e-6)
assert not torch.isclose(lp_a[2], lp_b[2], atol=1e-6) and not torch.isclose(lp_a[3], lp_b[3], atol=1e-6)
```

## Categorical perspective

The program is a Kleisli morphism of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical] from the target window to itself: a prior over the weights composed with a likelihood kernel from weights and data to target tokens. Inside that kernel every comprehension is a deterministic map between finite-dimensional real vector spaces indexed by the declared objects, and the three attention layers differ only in which tables they read and in the mask. The causal mask is what makes the target's likelihood factor into a product of conditionals, each a kernel from the source and the target's prefix to one token, and the causality check above tests that factorization at a point.

## Sources and further reading

- *Sequence to sequence learning with neural networks* [@sutskever2014sequence].
- *Attention is all you need* [@vaswani2017attention].
- *A categorical approach to probability theory* [@giry1982categorical].
