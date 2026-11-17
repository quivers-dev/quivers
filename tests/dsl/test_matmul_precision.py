"""Checked Real matrix multiplication preserves common precision and gradients."""

import pytest
import torch

from quivers.dsl.pure_builtins import _tensor_matmul


@pytest.mark.parametrize(
    "left_dtype,right_dtype",
    [
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_real_matmul_uses_common_precision_and_original_head_gradients(
    left_dtype, right_dtype
):
    left = torch.tensor([[0.1, 0.7], [-0.2, 0.4]], dtype=left_dtype, requires_grad=True)
    right = torch.tensor([[0.2], [-0.3]], dtype=right_dtype, requires_grad=True)
    result = _tensor_matmul(left, right)
    dtype = torch.promote_types(left_dtype, right_dtype)
    expected = left.to(dtype) @ right.to(dtype)
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected)
    actual = torch.autograd.grad(
        result.square().sum(), (left, right), retain_graph=True
    )
    oracle = torch.autograd.grad(expected.square().sum(), (left, right))
    for value, reference in zip(actual, oracle, strict=True):
        torch.testing.assert_close(value, reference)
    assert actual[0].dtype == left_dtype
    assert actual[1].dtype == right_dtype


def test_matmul_does_not_coerce_invalid_integer_real_mixtures():
    with pytest.raises(RuntimeError):
        _tensor_matmul(torch.ones(2, 2, dtype=torch.int64), torch.ones(2, 2))
