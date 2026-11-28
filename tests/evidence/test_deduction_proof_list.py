"""Check the measured proof output size and the cap's allocation boundary."""

from tests.benchmarks.deduction_proof_list import _fixture, _refit, _refused, _work


def test_checked_proof_list_size_and_live_refits() -> None:
    fixture = _fixture((4, 3))
    work = _work(fixture)
    assert work["returned_proof_rows"] == 4
    assert work["forest_edge_and_reference_records"] == 18
    assert work["runtime_proof_nodes"] == work["proof_value_nodes"] == 20
    _refit(fixture, 2, verify=True)


def test_structural_cap_refuses_before_building_any_proof_node() -> None:
    counted = _refused(_fixture((5, 4)))
    assert all(value == 0 for value in counted.values())
