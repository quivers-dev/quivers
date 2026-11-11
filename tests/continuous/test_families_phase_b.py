"""Unit tests for the Phase B tier 1 conditional families.

Covers construction, support, log-prob finiteness on in-support and
out-of-support inputs, and gradient flow for the reparameterisable
families.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import constraints as _constraints

from quivers import FinSet
from quivers.continuous.families import (
    ConditionalBetaBinomial,
    ConditionalHalfStudentT,
    ConditionalLogistic,
)
from quivers.continuous.ordered import (
    ConditionalOrderedLogistic,
    ConditionalOrderedProbit,
)
from quivers.continuous.spaces import Euclidean


# ---------------------------------------------------------------------------
# BetaBinomial
# ---------------------------------------------------------------------------


class TestConditionalBetaBinomial:
    def test_support_is_integer_interval(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        f = ConditionalBetaBinomial(a, y, total_count=10)
        sup = f.support
        assert isinstance(sup, _constraints._IntegerInterval)
        assert int(sup.lower_bound) == 0
        assert int(sup.upper_bound) == 10

    def test_log_prob_finite_in_support(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        f = ConditionalBetaBinomial(a, y, total_count=10)
        x = torch.tensor([0, 1])
        ys = torch.tensor([[3.0], [7.0]])
        lp = f.log_prob(x, ys)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_log_prob_handles_boundaries(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        f = ConditionalBetaBinomial(a, y, total_count=5)
        x = torch.tensor([0, 1])
        ys = torch.tensor([[0.0], [5.0]])
        lp = f.log_prob(x, ys)
        assert torch.isfinite(lp).all()

    def test_sample_in_range(self) -> None:
        a = FinSet(name="A", cardinality=3)
        y = Euclidean(name="y", dim=2)
        f = ConditionalBetaBinomial(a, y, total_count=4)
        x = torch.tensor([0, 1, 2])
        draws = f.sample(x)
        assert draws.shape == (3, 2)
        assert (draws >= 0).all()
        assert (draws <= 4).all()

    def test_rsample_unsupported(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        f = ConditionalBetaBinomial(a, y, total_count=4)
        with pytest.raises(NotImplementedError, match="rsample is not supported"):
            f.rsample(torch.tensor([0, 1]))

    def test_rejects_nonpositive_total_count(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        with pytest.raises(ValueError, match="total_count must be >= 1"):
            ConditionalBetaBinomial(a, y, total_count=0)


# ---------------------------------------------------------------------------
# Logistic
# ---------------------------------------------------------------------------


class TestConditionalLogistic:
    def test_support_is_real(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        f = ConditionalLogistic(a, y)
        assert f.support is _constraints.real

    def test_log_prob_finite(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=3)
        f = ConditionalLogistic(a, y)
        x = torch.tensor([0, 1])
        ys = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]])
        lp = f.log_prob(x, ys)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_rsample_returns_correct_shape(self) -> None:
        a = FinSet(name="A", cardinality=3)
        y = Euclidean(name="y", dim=4)
        f = ConditionalLogistic(a, y)
        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 4)
        assert torch.isfinite(s).all()

    def test_rsample_gradient_flow(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=2)
        f = ConditionalLogistic(a, y)
        x = torch.tensor([0, 1])
        s = f.rsample(x)
        loss = s.sum()
        loss.backward()
        has_grad = any(
            (p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters())
        )
        assert has_grad

    def test_log_prob_matches_closed_form(self) -> None:
        """At loc=0, scale=1 (default expectations), the logistic
        log-density at zero equals ``-2 * log(2)``."""
        a = FinSet(name="A", cardinality=1)
        y = Euclidean(name="y", dim=1)
        f = ConditionalLogistic(a, y)
        # Force loc=0, scale=1 by zeroing the lookup table.
        with torch.no_grad():
            for p in f.parameters():
                p.zero_()
        x = torch.tensor([0])
        ys = torch.tensor([[0.0]])
        lp = f.log_prob(x, ys)
        # softplus(0) = log(2); scale = softplus(0) + eps ~= log(2);
        # log(scale) ~= log(log(2)). Closed-form: -2 log(2) - log(scale).
        scale = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-7
        expected = -2.0 * math.log(2.0) - scale.log().item()
        assert lp.item() == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# HalfStudentT
# ---------------------------------------------------------------------------


class TestConditionalHalfStudentT:
    def test_support_is_positive(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        f = ConditionalHalfStudentT(a, y, df=3.0)
        sup = f.support
        assert isinstance(sup, _constraints._GreaterThan)
        assert float(sup.lower_bound) == 0.0

    def test_log_prob_finite_in_support(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=2)
        f = ConditionalHalfStudentT(a, y, df=4.0)
        x = torch.tensor([0, 1])
        ys = torch.tensor([[0.5, 1.5], [2.0, 0.1]])
        lp = f.log_prob(x, ys)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_log_prob_neg_inf_outside_support(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        f = ConditionalHalfStudentT(a, y, df=3.0)
        x = torch.tensor([0, 1])
        ys = torch.tensor([[-0.1], [-2.0]])
        lp = f.log_prob(x, ys)
        assert lp.shape == (2,)
        assert (lp == float("-inf")).all()

    def test_rsample_nonnegative(self) -> None:
        a = FinSet(name="A", cardinality=3)
        y = Euclidean(name="y", dim=2)
        f = ConditionalHalfStudentT(a, y, df=5.0)
        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 2)
        assert (s >= 0.0).all()

    def test_rsample_gradient_flow(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=2)
        f = ConditionalHalfStudentT(a, y, df=4.0)
        x = torch.tensor([0, 1])
        s = f.rsample(x)
        loss = s.sum()
        loss.backward()
        has_grad = any(
            (p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters())
        )
        assert has_grad

    def test_rejects_nonpositive_df(self) -> None:
        a = FinSet(name="A", cardinality=2)
        y = Euclidean(name="y", dim=1)
        with pytest.raises(ValueError, match="df must be > 0"):
            ConditionalHalfStudentT(a, y, df=0.0)


# ---------------------------------------------------------------------------
# OrderedLogistic
# ---------------------------------------------------------------------------


class TestConditionalOrderedLogistic:
    def test_support(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=4)
        f = ConditionalOrderedLogistic(a, b, num_categories=4)
        sup = f.support
        assert isinstance(sup, _constraints._IntegerInterval)
        assert int(sup.lower_bound) == 0
        assert int(sup.upper_bound) == 3

    def test_log_prob_finite_with_cutpoints(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=4)
        f = ConditionalOrderedLogistic(a, b, num_categories=4)
        x = torch.tensor([0, 1])
        y = torch.tensor([1, 3])
        cps = torch.tensor([-1.0, 0.0, 1.5])
        lp = f.log_prob(x, y, cutpoints=cps)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_log_probs_sum_to_one(self) -> None:
        a = FinSet(name="A", cardinality=1)
        b = FinSet(name="B", cardinality=5)
        f = ConditionalOrderedLogistic(a, b, num_categories=5)
        x = torch.tensor([0])
        cps = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        ks = torch.arange(5)
        log_probs = torch.stack(
            [f.log_prob(x, k.unsqueeze(0), cutpoints=cps) for k in ks],
            dim=-1,
        ).squeeze(0)
        total = log_probs.exp().sum().item()
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_sample_in_support(self) -> None:
        a = FinSet(name="A", cardinality=3)
        b = FinSet(name="B", cardinality=4)
        f = ConditionalOrderedLogistic(a, b, num_categories=4)
        x = torch.tensor([0, 1, 2])
        cps = torch.tensor([-1.0, 0.0, 1.0])
        draws = f.sample(x, cutpoints=cps)
        assert draws.shape == (3,)
        assert (draws >= 0).all()
        assert (draws <= 3).all()

    def test_rsample_unsupported(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f = ConditionalOrderedLogistic(a, b, num_categories=3)
        with pytest.raises(NotImplementedError, match="rsample is not supported"):
            f.rsample(torch.tensor([0, 1]))

    def test_invalid_num_categories(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=2)
        with pytest.raises(ValueError, match="num_categories >= 2"):
            ConditionalOrderedLogistic(a, b, num_categories=1)

    def test_cutpoints_wrong_length(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=4)
        f = ConditionalOrderedLogistic(a, b, num_categories=4)
        x = torch.tensor([0, 1])
        y = torch.tensor([1, 2])
        with pytest.raises(ValueError, match="cutpoints length"):
            f.log_prob(x, y, cutpoints=torch.tensor([0.0, 1.0]))


# ---------------------------------------------------------------------------
# OrderedProbit
# ---------------------------------------------------------------------------


class TestConditionalOrderedProbit:
    def test_support(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=5)
        f = ConditionalOrderedProbit(a, b, num_categories=5)
        sup = f.support
        assert isinstance(sup, _constraints._IntegerInterval)
        assert int(sup.upper_bound) == 4

    def test_log_prob_finite(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f = ConditionalOrderedProbit(a, b, num_categories=3)
        x = torch.tensor([0, 1])
        y = torch.tensor([0, 2])
        cps = torch.tensor([-0.5, 0.5])
        lp = f.log_prob(x, y, cutpoints=cps)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_log_probs_sum_to_one(self) -> None:
        a = FinSet(name="A", cardinality=1)
        b = FinSet(name="B", cardinality=4)
        f = ConditionalOrderedProbit(a, b, num_categories=4)
        x = torch.tensor([0])
        cps = torch.tensor([-1.0, 0.0, 1.0])
        ks = torch.arange(4)
        log_probs = torch.stack(
            [f.log_prob(x, k.unsqueeze(0), cutpoints=cps) for k in ks],
            dim=-1,
        ).squeeze(0)
        total = log_probs.exp().sum().item()
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_sample_in_support(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=4)
        f = ConditionalOrderedProbit(a, b, num_categories=4)
        x = torch.tensor([0, 1])
        cps = torch.tensor([-1.0, 0.0, 1.0])
        draws = f.sample(x, cutpoints=cps)
        assert draws.shape == (2,)
        assert (draws >= 0).all()
        assert (draws <= 3).all()

    def test_rsample_unsupported(self) -> None:
        a = FinSet(name="A", cardinality=2)
        b = FinSet(name="B", cardinality=3)
        f = ConditionalOrderedProbit(a, b, num_categories=3)
        with pytest.raises(NotImplementedError, match="rsample is not supported"):
            f.rsample(torch.tensor([0, 1]))


# ---------------------------------------------------------------------------
# FAMILY_META wiring
# ---------------------------------------------------------------------------


class TestFamilyMetaWiring:
    def test_all_five_have_quivers_class(self) -> None:
        from quivers.transpile.family_meta import FAMILY_META

        for qvr_name, quivers_cls in [
            ("BetaBinomial", ConditionalBetaBinomial),
            ("OrderedLogistic", ConditionalOrderedLogistic),
            ("OrderedProbit", ConditionalOrderedProbit),
            ("Logistic", ConditionalLogistic),
            ("HalfStudentT", ConditionalHalfStudentT),
        ]:
            meta = FAMILY_META[qvr_name]
            assert meta.quivers_class is quivers_cls
            assert meta.distribution_class.__name__ == qvr_name

    def test_registry_size_unchanged(self) -> None:
        # A registered family is one a target can be asked to emit, so
        # the count is pinned: a family arriving without its per-target
        # names, or leaving, changes what `transpile` accepts.
        from quivers.transpile.family_meta import FAMILY_META

        assert len(FAMILY_META) == 60
