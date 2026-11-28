"""Check the native typed grouping benchmark's counts and structural oracle."""

import pytest
import torch

from tests.benchmarks.deduction_key_grouping import _fixture, _forward_work, _run


def test_matched_key_hash_and_reduction_counts() -> None:
    fixture = _fixture((6, 2, 3))
    whole = _forward_work(fixture, isolated=False)
    isolated = _forward_work(fixture, isolated=True)
    assert whole["source_projection_calls"] == 6
    assert whole["cold_key_nodes"] == isolated["cold_key_nodes"] == 24
    assert isolated["log_reduction_lanes"] == 8
    assert isolated["forward_saved_logical_elements"] <= 22


def test_structural_oracle_rejects_extra_wrapper_with_unchanged_weights() -> None:
    original = _fixture((4, 2, 1))
    altered = _fixture((4, 2, 2))
    actual = _run(altered.module, "grouped", altered.initial).value
    reference = _run(original.module, "grouped", original.initial).value
    torch.testing.assert_close(altered.weights(actual), original.weights(reference))
    with pytest.raises(AssertionError, match="unexpected key depth"):
        original.weights(actual)
