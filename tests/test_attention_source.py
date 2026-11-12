"""`AttentionSource` computes what its docstring states.

Multi-head self-attention across the sequence positions, mean-pooled,
then a linear head: four heads by default, an output that does not
depend on the order of the positions, and a single feature vector read
as a sequence of length one.
"""

from __future__ import annotations

import torch

from quivers.continuous.param_source import AttentionSource


def test_default_is_four_heads() -> None:
    """The default constructor builds four attention heads."""
    source = AttentionSource(domain_dim=8, param_dim=3)
    assert source.attn.num_heads == 4


def test_output_is_invariant_to_position_order() -> None:
    """Permuting the positions of a sequence leaves the output unchanged."""
    torch.manual_seed(0)
    source = AttentionSource(domain_dim=4, param_dim=2)
    x = torch.randn(2, 5, 4)
    permuted = x[:, torch.tensor([3, 0, 4, 1, 2])]
    assert torch.allclose(source(x), source(permuted), atol=1e-6)


def test_output_depends_on_the_positions_it_attends_to() -> None:
    """Changing one position changes the output (the check above is not vacuous)."""
    torch.manual_seed(0)
    source = AttentionSource(domain_dim=4, param_dim=2)
    x = torch.randn(1, 5, 4)
    changed = x.clone()
    changed[0, 2] += 1.0
    assert not torch.allclose(source(x), source(changed), atol=1e-4)


def test_vector_and_matrix_inputs_are_length_one_sequences() -> None:
    """A ``(domain_dim,)`` or ``(batch, domain_dim)`` input is one position."""
    torch.manual_seed(0)
    source = AttentionSource(domain_dim=4, param_dim=2)
    v = torch.randn(4)
    assert torch.allclose(source(v), source(v.reshape(1, 1, 4)))
    batch = torch.randn(3, 4)
    assert torch.allclose(source(batch), source(batch.unsqueeze(1)))
