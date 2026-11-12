"""Tests for scoped grouped ``marginalize`` blocks.

Covers (i) the runtime primitive
:func:`quivers.continuous.plate.marginalize_grouped` and
(ii) the DSL surface ``marginalize c : K <- Categorical(probs=probs) over G
via idx in { ... }`` that compiles to it.

The runtime primitive realises the right Kan extension along a finite
index map ``r : Resp → G`` in :math:`\\mathbf{Kern}`, followed by
log-sum-exp over the class axis under a categorical prior, summed
over groups. Concretely::

    Σ_g log_sum_exp_k [ log π(g, k) + Σ_{n: r(n)=g} ll(n, k) ]
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.continuous.plate import (
    marginalize_categorical,
    marginalize_grouped,
)


# A fixed default seed keeps the per-group reference and the
# vectorized primitive at the same realisation.
torch.manual_seed(0)


class TestMarginalizeGroupedPrimitive:
    """Unit tests for the runtime primitive."""

    def test_matches_hand_rolled_reference(self):
        """The vectorized scatter-add + log-sum-exp matches a Python loop."""
        n_groups = 4
        n_classes = 3
        n_rows = 11
        ll = torch.randn(n_rows, n_classes, dtype=torch.float64)
        idx = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3], dtype=torch.long)
        probs = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        log_prior = torch.log(probs)

        # Reference: explicit per-group accumulation in Python.
        per_group_total = torch.zeros(n_groups, dtype=torch.float64)
        for g in range(n_groups):
            ll_g = torch.zeros(n_classes, dtype=torch.float64)
            for n in range(n_rows):
                if int(idx[n]) == g:
                    ll_g = ll_g + ll[n]
            per_group_total[g] = torch.logsumexp(log_prior + ll_g, dim=-1)
        expected = per_group_total.sum()

        out = marginalize_grouped(ll, idx, log_prior, n_groups)
        assert torch.allclose(out, expected, atol=1e-10)

    def test_single_class_collapses_to_sum(self):
        """K = 1: the log-sum-exp is a no-op so the result is
        Σ_g (log π[g, 0] + Σ_n ll[n, 0])."""
        ll = torch.randn(7, 1, dtype=torch.float64)
        idx = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
        probs = torch.tensor([[0.7], [0.6], [0.5]], dtype=torch.float64)
        log_prior = torch.log(probs)

        out = marginalize_grouped(ll, idx, log_prior, 3)

        grouped = torch.zeros(3, 1, dtype=torch.float64).index_add(0, idx, ll)
        expected = (log_prior + grouped).sum()
        assert torch.allclose(out, expected, atol=1e-10)

    def test_identity_index_map_recovers_ungrouped(self):
        """Identity index map (one group per row) reduces to the per-row
        mixture sum: each row is its own group, so the per-group
        accumulator equals the per-row log-likelihood."""
        n_rows = 8
        n_classes = 5
        ll = torch.randn(n_rows, n_classes, dtype=torch.float64)
        idx = torch.arange(n_rows, dtype=torch.long)
        probs = torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float64)
        log_prior = torch.log(probs)

        out = marginalize_grouped(ll, idx, log_prior, n_rows)
        # Ungrouped: each row contributes log_sum_exp_k (log π[k] + ll[n,k]).
        expected = marginalize_categorical(log_prior + ll).sum()
        assert torch.allclose(out, expected, atol=1e-10)

    def test_per_group_prior_broadcasts(self):
        """A ``(G, K)`` prior is honoured per group; a ``(K,)`` prior
        broadcasts to every group."""
        ll = torch.randn(6, 2, dtype=torch.float64)
        idx = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        broadcast_prior = torch.log(torch.tensor([0.4, 0.6], dtype=torch.float64))
        per_group_prior = broadcast_prior.expand(3, 2)

        out_b = marginalize_grouped(ll, idx, broadcast_prior, 3)
        out_g = marginalize_grouped(ll, idx, per_group_prior, 3)
        assert torch.allclose(out_b, out_g, atol=1e-12)

    def test_returns_scalar_and_dtype(self):
        """The primitive returns a 0-d tensor whose dtype matches the
        input log-likelihood tensor."""
        ll = torch.randn(5, 3, dtype=torch.float32)
        idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        log_prior = torch.log(torch.tensor([0.3, 0.4, 0.3], dtype=torch.float32))

        out = marginalize_grouped(ll, idx, log_prior, 2)
        assert out.dim() == 0
        assert out.dtype == torch.float32

    def test_gradients_flow_through_prior_and_likelihood(self):
        """Both the prior probs and the per-class log-likelihood
        receive gradient contributions."""
        probs = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64, requires_grad=True)
        ll = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        idx = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        out = marginalize_grouped(ll, idx, torch.log(probs), 3)
        out.backward()
        assert probs.grad is not None
        assert ll.grad is not None
        assert torch.isfinite(probs.grad).all()
        assert torch.isfinite(ll.grad).all()
        # The likelihood gradient must be non-trivial at every position.
        assert (ll.grad.abs() > 0).all()

    def test_one_d_input_runs_nested_intermediate_path(self):
        """A 1-D ``(K,)`` input is the contribution of an
        already-reduced inner block; the primitive applies the
        prior and reduces over the class axis (no scatter step)."""
        ll = torch.tensor([0.1, 0.2, 0.3])
        log_prior = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        out = marginalize_grouped(
            ll,
            torch.zeros(0, dtype=torch.long),  # no rows to scatter
            log_prior,
            1,
        )
        expected = torch.logsumexp(log_prior + ll, dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_rejects_mismatched_index_length(self):
        with pytest.raises(ValueError, match="leading axis"):
            marginalize_grouped(
                torch.zeros(4, 2),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(2),
                2,
            )

    def test_rejects_nonpositive_num_groups(self):
        with pytest.raises(ValueError, match="num_groups must be positive"):
            marginalize_grouped(
                torch.zeros(3, 2),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(2),
                0,
            )

    def test_rejects_out_of_range_index(self):
        with pytest.raises(ValueError, match="out-of-range"):
            marginalize_grouped(
                torch.zeros(3, 2),
                torch.tensor([0, 1, 3], dtype=torch.long),
                torch.zeros(2),
                2,
            )

    def test_empty_fibre_contributes_logsumexp_of_prior(self):
        """A group with no responses contributes log_sum_exp_k(log π[g,k])
        = log Σ_k π[g,k], i.e. the prior's normalisation constant.
        For a normalized prior this is zero."""
        ll = torch.randn(4, 3, dtype=torch.float64)
        # All rows go to group 0; group 1 is empty.
        idx = torch.zeros(4, dtype=torch.long)
        probs = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        log_prior = torch.log(probs)

        out = marginalize_grouped(ll, idx, log_prior, 2)
        # Group 0: log_sum_exp_k (log π[k] + Σ_n ll[n, k])
        sum_ll = ll.sum(dim=0)
        group0 = torch.logsumexp(log_prior + sum_ll, dim=-1)
        # Group 1: log_sum_exp_k (log π[k]) = log(Σ_k π[k]) = log 1 = 0.
        group1 = torch.logsumexp(log_prior, dim=-1)
        expected = group0 + group1
        assert torch.allclose(out, expected, atol=1e-10)


# DSL-level integration tests over the grouped-marginalize surface.


def _compile(src: str):
    from quivers.dsl.compiler import Compiler
    from quivers.dsl.parser import parse

    m = parse(textwrap.dedent(src))
    c = Compiler(m)
    c.compile()
    return c


class TestGroupedMarginalizeSurface:
    """End-to-end DSL compilation tests for the grouped surface."""

    def test_grouped_block_compiles(self):
        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 4
        object Resp : FinSet 10
        object Class : FinSet 3

        program demo : Item -> Item
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=Item, reduction=logsumexp]
                observe r : Resp <- HalfNormal(scale=1.0) [via=idx]
            return probs

        export demo
        """
        c = _compile(src)
        assert "demo" in c._morphisms

    def test_grouped_missing_via_errors(self):
        from quivers.dsl.compiler import CompileError

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 4
        object Resp : FinSet 10
        object Class : FinSet 3

        program demo : Item -> Item
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=Item]
                observe r : Resp <- HalfNormal(scale=1.0)
            return probs

        export demo
        """
        with pytest.raises(CompileError, match="must carry its own `via"):
            _compile(src)

    def test_grouped_requires_class_annotation(self):
        from quivers.dsl.compiler import CompileError

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 4
        object Resp : FinSet 10
        object Class : FinSet 3

        program demo : Item -> Item
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls <- Categorical(probs=probs) [over=Item]
                sample inner : Resp <- HalfNormal(scale=1.0)
            return probs

        export demo
        """
        with pytest.raises(CompileError, match="explicit class-set"):
            _compile(src)

    def test_grouped_undeclared_over_errors(self):
        from quivers.dsl.compiler import CompileError

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 4
        object Resp : FinSet 10
        object Class : FinSet 3

        program demo : Item -> Item
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=NotAnObject]
                sample inner : Resp <- HalfNormal(scale=1.0)
            return probs

        export demo
        """
        with pytest.raises(CompileError, match="not a declared object"):
            _compile(src)

    def test_ungrouped_still_compiles(self):
        """An ungrouped marginalize (no ``over=`` option) still parses
        and compiles unchanged."""
        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 5
        object Class : FinSet 3

        program demo : Item -> Item
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs)
                sample z : Item <- Normal(loc=0.0, scale=1.0)
            return probs

        export demo
        """
        c = _compile(src)
        assert "demo" in c._morphisms
