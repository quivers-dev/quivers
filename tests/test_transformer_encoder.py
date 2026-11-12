"""`transformer_encoder` computes a transformer encoder over a sequence.

The encoder's output is checked against a transformer layer written out
here from its parameters (Vaswani et al. 2017: multi-head scaled
dot-product self-attention, residual connection and layer
normalization, then a position-wise feed-forward network with the
same), applied to the element embeddings plus sinusoidal position
encodings and mean-pooled. A per-element MLP without attention, the
shape the factory once had, fails the same comparison.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from quivers.structural import Encoder
from quivers.structural.shapes.seq import (
    list_to_term,
    seq_signature,
    transformer_encoder,
)

DIM = 8
HEADS = 2


def _positions(length: int, dim: int) -> torch.Tensor:
    out = torch.zeros(length, dim)
    for p in range(length):
        for i in range(0, dim, 2):
            angle = p / (10000.0 ** (i / dim))
            out[p, i] = math.sin(angle)
            if i + 1 < dim:
                out[p, i + 1] = math.cos(angle)
    return out


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + norm.eps) * norm.weight + norm.bias


def _reference_layer(
    x: torch.Tensor, layer: nn.TransformerEncoderLayer
) -> torch.Tensor:
    n, d = x.shape
    attn = layer.self_attn
    w_q, w_k, w_v = attn.in_proj_weight.split(d)
    b_q, b_k, b_v = attn.in_proj_bias.split(d)
    q, k, v = x @ w_q.T + b_q, x @ w_k.T + b_k, x @ w_v.T + b_v
    hd = d // HEADS
    outs = []
    for h in range(HEADS):
        sl = slice(h * hd, (h + 1) * hd)
        scores = q[:, sl] @ k[:, sl].T / math.sqrt(hd)
        outs.append(torch.softmax(scores, dim=-1) @ v[:, sl])
    attended = torch.cat(outs, dim=-1) @ attn.out_proj.weight.T + attn.out_proj.bias
    x = _layer_norm(x + attended, layer.norm1)
    hidden = nn.functional.gelu(x @ layer.linear1.weight.T + layer.linear1.bias)
    ff = hidden @ layer.linear2.weight.T + layer.linear2.bias
    return _layer_norm(x + ff, layer.norm2)


def _stack(encoder: Encoder) -> nn.TransformerEncoder:
    return next(m for m in encoder.modules() if isinstance(m, nn.TransformerEncoder))


def _reference(encoder: Encoder, tokens: list[str]) -> torch.Tensor:
    embed = encoder.data_embedders["A"]
    x = torch.stack([embed(t) for t in tokens]) + _positions(len(tokens), DIM)
    for layer in _stack(encoder).layers:
        x = _reference_layer(x, layer)
    return x.mean(dim=0)


def test_matches_a_transformer_written_from_its_parameters() -> None:
    torch.manual_seed(0)
    enc = transformer_encoder(seq_signature(dim=DIM), dim=DIM, heads=HEADS, layers=2)
    tokens = ["a", "b", "c", "a"]
    with torch.no_grad():
        got = enc(list_to_term(tokens))
        expected = _reference(enc, tokens)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_the_reference_rejects_an_encoder_without_attention() -> None:
    """A head/tail MLP fold, with the same embeddings, is not the
    transformer the reference computes."""
    torch.manual_seed(0)
    enc = transformer_encoder(seq_signature(dim=DIM), dim=DIM, heads=HEADS)
    tokens = ["a", "b", "c"]
    fuse = nn.Sequential(nn.Linear(2 * DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
    embed = enc.data_embedders["A"]
    with torch.no_grad():
        state = torch.zeros(DIM)
        for t in reversed(tokens):
            state = fuse(torch.cat([embed(t), state]))
        assert not torch.allclose(state, _reference(enc, tokens), atol=1e-3)


def test_every_position_attends_to_the_whole_sequence() -> None:
    """Order matters through the positions, and the first element
    reaches the output."""
    torch.manual_seed(1)
    enc = transformer_encoder(seq_signature(dim=DIM), dim=DIM, heads=HEADS)
    with torch.no_grad():
        ab = enc(list_to_term(["a", "b", "c"]))
        ba = enc(list_to_term(["b", "a", "c"]))
        cb = enc(list_to_term(["c", "b", "c"]))
    assert not torch.allclose(ab, ba)
    assert not torch.allclose(ab, cb)


def test_refuses_a_head_count_that_does_not_divide_the_width() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        transformer_encoder(seq_signature(dim=DIM), dim=DIM, heads=3)
