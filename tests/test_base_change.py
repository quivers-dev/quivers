"""Tests for change of enrichment."""

import torch

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
from quivers.categorical.base_change import BoolToFuzzy
from quivers.core.algebras import PRODUCT_FUZZY, BOOLEAN


class TestBoolToFuzzy:
    def test_preserves_boolean_tensor(self):
        bc = BoolToFuzzy()
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = bc.apply_to_values(t)
        torch.testing.assert_close(result, t)

    def test_source_target(self):
        bc = BoolToFuzzy()
        assert bc.source is BOOLEAN
        assert bc.target is PRODUCT_FUZZY

    def test_apply_to_morphism(self):
        bc = BoolToFuzzy()
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        f = observed(a, b, data, algebra=BOOLEAN)

        result = bc.apply_to_morphism(f)
        assert result.algebra is PRODUCT_FUZZY
        torch.testing.assert_close(result.tensor, data)


class TestBoolToFuzzyIsLaxMonoidal:
    """The inclusion ``{0, 1} ↪ [0, 1]`` preserves the unit, the tensor
    and the order, so it is a (strict) monoidal monotone map and a base
    change; thresholding at an interior point fails the tensor law,
    which is why it is not offered as a base change."""

    def test_inclusion_preserves_unit_tensor_and_order(self):
        bc = BoolToFuzzy()
        vals = torch.tensor([0.0, 1.0])
        one = bc.apply_to_values(torch.tensor([BOOLEAN.unit]))
        assert one.item() == PRODUCT_FUZZY.unit
        for a in vals:
            for b in vals:
                lhs = PRODUCT_FUZZY.tensor_op(
                    bc.apply_to_values(a.reshape(1)), bc.apply_to_values(b.reshape(1))
                )
                rhs = bc.apply_to_values(BOOLEAN.tensor_op(a, b).reshape(1))
                assert torch.equal(lhs, rhs)

    def test_interior_threshold_is_not_lax_monoidal(self):
        def threshold(x: float) -> float:
            return 1.0 if x >= 0.5 else 0.0

        a = b = 0.6
        assert threshold(a) * threshold(b) > threshold(a * b)
