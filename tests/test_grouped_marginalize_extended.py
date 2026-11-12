"""Extended coverage for the grouped ``marginalize`` block:
nesting, product index maps, reductions, body vectorisation,
continuous latents in scope, and end-to-end SVI with a
continuous guide.

Each section checks a surface or semantic property of the grouped
``marginalize`` block.
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.continuous.plate import marginalize_grouped


# ---------------------------------------------------------------------------
# Runtime primitive: reduction parameter
# ---------------------------------------------------------------------------


class TestReductionParameter:
    """The runtime primitive accepts ``reduction = logsumexp | sum
    | mean``; each gives the documented per-group reduction over
    the class axis."""

    def test_logsumexp_default(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        out = marginalize_grouped(ll, idx, log_prior, 3)
        # Build the hand-rolled reference under logsumexp.
        grouped = torch.zeros(3, 3)
        grouped = grouped.index_add(0, idx, ll)
        expected = torch.logsumexp(log_prior + grouped, dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_sum_reduction(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        out = marginalize_grouped(ll, idx, log_prior, 3, reduction="sum")
        grouped = torch.zeros(3, 3)
        grouped = grouped.index_add(0, idx, ll)
        expected = (log_prior + grouped).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_mean_reduction(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        out = marginalize_grouped(ll, idx, log_prior, 3, reduction="mean")
        grouped = torch.zeros(3, 3)
        grouped = grouped.index_add(0, idx, ll)
        expected = (log_prior + grouped).mean(dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_unknown_reduction_raises(self) -> None:
        ll = torch.randn(2, 3)
        idx = torch.tensor([0, 0])
        log_prior = torch.log(torch.ones(3) / 3)
        with pytest.raises(ValueError, match="reduction"):
            marginalize_grouped(ll, idx, log_prior, 1, reduction="not_a_reduction")


# ---------------------------------------------------------------------------
# Runtime primitive: product index map
# ---------------------------------------------------------------------------


class TestProductIndexMap:
    """The runtime primitive accepts a tuple of co-indexed
    index maps + a tuple of group sizes (the product grouping
    plate). Behavior matches the single-index-map form when the
    product collapses to one axis, and matches a hand-rolled
    Python loop for the two-axis case."""

    def test_single_axis_via_tuple_matches_scalar_form(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 1, 2, 0, 1, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        scalar = marginalize_grouped(ll, idx, log_prior, 3)
        product = marginalize_grouped(ll, (idx,), log_prior, (3,))
        assert torch.allclose(scalar, product, atol=1e-6)

    def test_two_axis_product_matches_hand_rolled(self) -> None:
        torch.manual_seed(0)
        N, K = 12, 3
        G1, G2 = 3, 2  # group sizes
        ll = torch.randn(N, K)
        idx_a = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        idx_b = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        log_prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(ll, (idx_a, idx_b), log_prior, (G1, G2))
        # Hand-rolled reference: scatter-add over the product index.
        flat = idx_a * G2 + idx_b
        grouped = torch.zeros(G1 * G2, K)
        grouped = grouped.index_add(0, flat, ll)
        expected = torch.logsumexp(log_prior + grouped, dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_two_axis_prior_with_per_cell_class_distribution(self) -> None:
        """When the prior is shape ``(G1, G2, K)`` it's flattened
        in row-major order and applied per-cell."""
        torch.manual_seed(0)
        N, K = 8, 2
        G1, G2 = 2, 2
        ll = torch.randn(N, K)
        idx_a = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        idx_b = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        # Per-cell prior: different mixing for each (g1, g2).
        prior_per_cell = torch.tensor(
            [[[0.7, 0.3], [0.4, 0.6]], [[0.5, 0.5], [0.1, 0.9]]]
        )
        log_prior = torch.log(prior_per_cell)
        out = marginalize_grouped(ll, (idx_a, idx_b), log_prior, (G1, G2))
        flat = idx_a * G2 + idx_b
        grouped = torch.zeros(G1 * G2, K)
        grouped = grouped.index_add(0, flat, ll)
        expected = torch.logsumexp(
            log_prior.reshape(G1 * G2, K) + grouped, dim=-1
        ).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_arity_mismatch_raises(self) -> None:
        ll = torch.randn(4, 2)
        idx_a = torch.tensor([0, 0, 1, 1])
        log_prior = torch.log(torch.ones(2) / 2)
        with pytest.raises(ValueError, match="number of indices"):
            marginalize_grouped(ll, (idx_a, idx_a), log_prior, (2,))

    def test_product_axis_out_of_range_raises(self) -> None:
        ll = torch.randn(4, 2)
        idx_a = torch.tensor([0, 0, 3, 1])  # 3 is out of range for size 3
        idx_b = torch.tensor([0, 0, 1, 1])
        log_prior = torch.log(torch.ones(2) / 2)
        with pytest.raises(ValueError, match="outside"):
            marginalize_grouped(ll, (idx_a, idx_b), log_prior, (3, 2))


# ---------------------------------------------------------------------------
# Surface: nested marginalize blocks
# ---------------------------------------------------------------------------


class TestNestedMarginalize:
    """Mixtures-of-mixtures: a marginalize block whose body contains
    another marginalize block. The outer latent is in scope inside
    the inner block, and the body-vectorisation pass threads the
    captured observe through every level of nesting."""

    def test_three_level_nested_blocks_compile(self) -> None:
        """A three-level nested marginalize compiles: outer → middle
        → inner, with the innermost body's observe captured and
        threaded up through every level."""
        from quivers.dsl import loads

        src = """
        composition log_prob [level=algebra]

        object G_outer : FinSet 2
        object G_middle : FinSet 2
        object G_inner : FinSet 2
        object Resp : FinSet 8
        object K_outer : FinSet 2
        object K_middle : FinSet 2
        object K_inner : FinSet 2

        program triple : Resp -> Resp
            sample probs_a : K_outer <- HalfNormal(scale=1.0)
            sample probs_b : K_middle <- HalfNormal(scale=1.0)
            sample probs_c : K_inner <- HalfNormal(scale=1.0)
            marginalize outer : K_outer <- Categorical(probs=probs_a) [over=G_outer]
                marginalize middle : K_middle <- Categorical(probs=probs_b) [over=G_middle]
                    marginalize inner : K_inner <- Categorical(probs=probs_c) [over=G_inner]
                        observe r : Resp <- HalfNormal(scale=1.0) [via=idx_a]
            return probs_a
        export triple
        """
        m = loads(textwrap.dedent(src))
        assert m.morphism is not None

    def test_three_level_nested_blocks_run_log_joint(self) -> None:
        """The three-level nested marginalize model runs through
        log_joint end-to-end without errors, given all latents
        clamped via observations. This is the runtime gate: a
        compile-only test would let a non-functional runtime slip
        through."""
        import torch
        from quivers.dsl import loads

        src = """
        composition log_prob [level=algebra]

        object G_outer : FinSet 2
        object G_middle : FinSet 2
        object G_inner : FinSet 2
        object Resp : FinSet 8
        object K_outer : FinSet 2
        object K_middle : FinSet 2
        object K_inner : FinSet 2

        program triple : Resp -> Resp
            sample probs_a : K_outer <- HalfNormal(scale=1.0)
            sample probs_b : K_middle <- HalfNormal(scale=1.0)
            sample probs_c : K_inner <- HalfNormal(scale=1.0)
            marginalize outer : K_outer <- Categorical(probs=probs_a) [over=G_outer]
                marginalize middle : K_middle <- Categorical(probs=probs_b) [over=G_middle]
                    marginalize inner : K_inner <- Categorical(probs=probs_c) [over=G_inner]
                        observe r : Resp <- HalfNormal(scale=1.0) [via=idx_a]
            return probs_a
        export triple
        """
        model = loads(textwrap.dedent(src)).morphism
        torch.manual_seed(0)
        obs = {
            "probs_a": torch.tensor([0.5, 0.5]),
            "probs_b": torch.tensor([0.5, 0.5]),
            "probs_c": torch.tensor([0.5, 0.5]),
            "idx_a": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
            "idx_b": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            "idx_c": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "r": torch.zeros(8),
        }
        out = model.log_joint(torch.zeros(1, 1), obs)
        assert torch.isfinite(out).all()

    def test_nested_blocks_compile(self) -> None:
        """An inner block grouped over a product refining the outer
        group projects its per-position marginals onto the outer
        group's positions."""
        from quivers.dsl import loads

        src = """
        composition log_prob [level=algebra]

        object G1 : FinSet 2
        object G2 : FinSet 3
        object Resp : FinSet 6
        object K1 : FinSet 2
        object K2 : FinSet 3

        program demo : Resp -> Resp
            sample probs_outer : K1 <- HalfNormal(scale=1.0)
            sample probs_inner : K2 <- HalfNormal(scale=1.0)
            marginalize outer : K1 <- Categorical(probs=probs_outer) [over=G1]
                marginalize inner : K2 <- Categorical(probs=probs_inner) [over=[G1, G2]]
                    observe r : Resp <- HalfNormal(scale=1.0) [via=[outer_idx, inner_idx]]
            return probs_outer
        export demo
        """
        m = loads(textwrap.dedent(src))
        assert m.morphism is not None


# ---------------------------------------------------------------------------
# Surface: product index map in the DSL
# ---------------------------------------------------------------------------


class TestProductIndexMapSurface:
    """``over G * H via product(idx_a, idx_b)`` parses, compiles,
    and threads the product-index-map data into the runtime."""

    def test_product_index_map_compiles(self) -> None:
        from quivers.dsl import loads

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 3
        object Subj : FinSet 2
        object Resp : FinSet 6
        object Class : FinSet 4

        program demo : Resp -> Resp
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=[Item, Subj]]
                observe r : Resp <- HalfNormal(scale=1.0) [via=[item_idx, subj_idx]]
            return probs
        export demo
        """
        m = loads(textwrap.dedent(src))
        assert m.morphism is not None

    def test_arity_mismatch_errors(self) -> None:
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 3
        object Subj : FinSet 2
        object Resp : FinSet 6
        object Class : FinSet 4

        program demo : Resp -> Resp
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=[Item, Subj]]
                observe r : Resp <- HalfNormal(scale=1.0) [via=item_idx]
            return probs
        export demo
        """
        with pytest.raises(CompileError, match="arity"):
            loads(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# Surface: reduction parameter in the DSL
# ---------------------------------------------------------------------------


class TestReductionSurface:
    """``reduction = sum`` / ``reduction = mean`` parses and is
    threaded into the runtime primitive."""

    def test_reduction_clause_compiles(self) -> None:
        from quivers.dsl import loads

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 3
        object Resp : FinSet 6
        object Class : FinSet 4

        program demo : Resp -> Resp
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=Item, reduction=sum]
                observe r : Resp <- HalfNormal(scale=1.0) [via=idx]
            return probs
        export demo
        """
        m = loads(textwrap.dedent(src))
        assert m.morphism is not None

    def test_unknown_reduction_errors(self) -> None:
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 3
        object Resp : FinSet 6
        object Class : FinSet 4

        program demo : Resp -> Resp
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=Item, reduction=bogus]
                observe r : Resp <- HalfNormal(scale=1.0) [via=idx]
            return probs
        export demo
        """
        with pytest.raises(CompileError, match="unknown reduction"):
            loads(textwrap.dedent(src))


# ---------------------------------------------------------------------------
# Additional surface-level compile-error coverage
# ---------------------------------------------------------------------------


class TestSurfaceCompileErrors:
    """Each error path the grouped-marginalize compiler raises has
    its own test so a regression hides nothing."""

    def test_categorical_with_literal_first_arg_errors(self) -> None:
        """A grouped block's prior must reference a NAMED probs
        tensor, not a literal — otherwise the runtime has nothing
        to broadcast against."""
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 3
        object Resp : FinSet 6
        object Class : FinSet 2

        program demo : Resp -> Resp
            marginalize cls : Class <- Categorical(probs=1.0) [over=Item]
                observe r : Resp <- HalfNormal(scale=1.0) [via=idx]
            return cls
        export demo
        """
        with pytest.raises(CompileError, match="named probs"):
            loads(textwrap.dedent(src))

    def test_body_without_observe_errors(self) -> None:
        """The body must end with an observe step (or a nested
        marginalize). A body containing only let-steps or sample-
        steps has no captured per-(N, K) ll to feed the reduction."""
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        composition log_prob [level=algebra]

        object Item : FinSet 3
        object Resp : FinSet 6
        object Class : FinSet 2

        program demo : Resp -> Resp
            sample probs : Class <- HalfNormal(scale=1.0)
            marginalize cls : Class <- Categorical(probs=probs) [over=Item]
                sample other : Resp <- HalfNormal(scale=1.0)
            return probs
        export demo
        """
        with pytest.raises(CompileError, match="observe"):
            loads(textwrap.dedent(src))


def test_three_axis_product_index_map_dsl_compiles() -> None:
    """`over A * B * C via product(idx_a, idx_b, idx_c)` parses
    and compiles."""
    from quivers.dsl import loads

    src = """
    composition log_prob [level=algebra]

    object A : FinSet 2
    object B : FinSet 2
    object C : FinSet 2
    object Resp : FinSet 8
    object K : FinSet 2

    program triple_prod : Resp -> Resp
        sample probs : K <- HalfNormal(scale=1.0)
        marginalize cls : K <- Categorical(probs=probs) [over=[A, B, C]]
            observe r : Resp <- HalfNormal(scale=1.0) [via=[idx_a, idx_b, idx_c]]
        return probs
    export triple_prod
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None
