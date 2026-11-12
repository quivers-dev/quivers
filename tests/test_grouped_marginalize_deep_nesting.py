"""Deeply nested grouped marginalize: arbitrary-depth stacks.

The grouped ``marginalize`` block is meant to nest arbitrarily;
this module verifies the runtime handles 3 / 4 / 5 / 6 / 7-level
stacks compiled from procedurally generated DSL programs. Each
test checks both compile-time validity and that
``model.log_joint`` produces a finite scalar end-to-end.

The runtime's two operating modes (innermost has a row axis +
index map; outer levels operate on already-reduced contributions
with no row axis) compose multiplicatively across nesting depth,
so a passing 7-level test gives high confidence the compositional
construction is correct.
"""

from __future__ import annotations
import textwrap

from textwrap import dedent

import pytest
import torch

from quivers.continuous.plate import marginalize_grouped


def _build_nested_program(num_levels: int, n_resp: int = 8) -> str:
    """Procedurally generate a marginalize program with
    ``num_levels`` levels of nesting.

    Each level declares its own grouping plate ``G_i``, class set
    ``K_i``, index map ``idx_i``, and categorical prior
    ``probs_i``. The innermost body observes a single Resp-plate
    likelihood; outer levels share the same body.
    """
    decls = [
        "composition log_prob [level=algebra]",
        "",
        f"object Resp : FinSet {n_resp}",
    ]
    for i in range(num_levels):
        decls.append(f"object G_{i} : FinSet 2")
        decls.append(f"object K_{i} : FinSet 2")
    obj_decls = "\n        ".join(decls)
    prog_lines = [f"program nested_{num_levels} : Resp -> Resp"]
    for i in range(num_levels):
        prog_lines.append(f"    sample probs_{i} : K_{i} <- HalfNormal(scale=1.0)")
    # Only the innermost block has an index map into a response
    # plate, host data named by its ``via``; outer levels consume the
    # inner block's already-scattered tensor with identity index map.
    indent = "    "
    nested: list[str] = []
    for i in range(num_levels):
        pad = indent * (i + 1)
        nested.append(
            f"{pad}marginalize lat_{i} : K_{i} <- Categorical(probs=probs_{i}) [over=G_{i}]"
        )
    # Innermost body: a single observe step carrying the
    # index map to its response plate.
    inner_pad = indent * (num_levels + 1)
    nested.append(
        f"{inner_pad}observe r : Resp <- HalfNormal(scale=1.0) [via=idx_inner]"
    )
    prog_lines.extend(nested)
    prog_lines.append("    return probs_0")
    body = "\n        ".join(prog_lines)
    return dedent(
        f"""
        {obj_decls}

        {body}

        export nested_{num_levels}
        """
    )


def _make_obs(num_levels: int, n_resp: int = 8) -> dict[str, torch.Tensor]:
    obs: dict[str, torch.Tensor] = {"r": torch.zeros(n_resp)}
    for i in range(num_levels):
        obs[f"probs_{i}"] = torch.tensor([0.5, 0.5])
    obs["idx_inner"] = torch.tensor([j % 2 for j in range(n_resp)])
    return obs


# ---------------------------------------------------------------------------
# Parametrised arbitrary-depth nesting tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_levels", [3, 4, 5, 6, 7])
def test_deep_nested_marginalize_compiles(num_levels: int) -> None:
    """A ``num_levels``-deep nested marginalize compiles."""
    from quivers.dsl import loads

    src = _build_nested_program(num_levels)
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None


@pytest.mark.parametrize("num_levels", [3, 4, 5, 6, 7])
def test_deep_nested_marginalize_log_joint_finite(num_levels: int) -> None:
    """A ``num_levels``-deep nested marginalize runs through
    ``log_joint`` to a finite scalar end-to-end."""
    from quivers.dsl import loads

    src = _build_nested_program(num_levels)
    model = loads(textwrap.dedent(src)).morphism
    obs = _make_obs(num_levels)
    out = model.log_joint(torch.zeros(1, 1), obs)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Runtime-primitive composability across nested levels
# ---------------------------------------------------------------------------


class TestPrimitiveCompositionAcrossLevels:
    """The primitive in isolation: feed an N-row input through one
    level (producing a ``(K_outer,)`` intermediate) and then feed
    that intermediate through a second level (producing a scalar).
    The composition must agree with a hand-rolled hierarchical
    log-mixture reference."""

    def test_two_level_primitive_composition(self) -> None:
        torch.manual_seed(0)
        N = 6
        K_inner, K_outer = 3, 2
        G_inner = 3
        # Innermost: (N, K_outer, K_inner) per-row per-(K_outer,
        # K_inner) ll.
        ll = torch.randn(N, K_outer, K_inner)
        idx_inner = torch.tensor([0, 1, 2, 0, 1, 2])
        prior_inner = torch.log(torch.ones(K_inner) / K_inner)
        # Inner reduces last axis (K_inner), preserves K_outer.
        inner_out = marginalize_grouped(ll, idx_inner, prior_inner, G_inner)
        # inner_out shape: (K_outer,) — the per-outer-class log-
        # marginal of the inner block.
        assert inner_out.shape == (K_outer,)
        # Outer: feed the (K_outer,) tensor with no row axis; the
        # primitive applies prior_outer + reduces.
        prior_outer = torch.log(torch.ones(K_outer) / K_outer)
        outer_out = marginalize_grouped(
            inner_out,
            torch.zeros(0, dtype=torch.long),
            prior_outer,
            1,
        )
        # Hand-rolled hierarchical reference.
        grouped = torch.zeros(G_inner, K_outer, K_inner)
        grouped = grouped.index_add(0, idx_inner, ll)
        # Reduce inner axis under inner prior.
        per_group = torch.logsumexp(prior_inner + grouped, dim=-1)  # (G_inner, K_outer)
        per_outer = per_group.sum(dim=0)  # (K_outer,)
        outer_expected = torch.logsumexp(prior_outer + per_outer, dim=-1)
        assert torch.allclose(outer_out, outer_expected, atol=1e-6)

    def test_three_level_primitive_composition(self) -> None:
        torch.manual_seed(0)
        N = 8
        K_a, K_b, K_c = 2, 3, 2  # outer-most a, then b, innermost c
        G_c = 4
        # Innermost (level c) sees a (N, K_a, K_b, K_c) ll.
        ll = torch.randn(N, K_a, K_b, K_c)
        idx_c = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        prior_c = torch.log(torch.ones(K_c) / K_c)
        c_out = marginalize_grouped(ll, idx_c, prior_c, G_c)
        # c_out shape: (K_a, K_b)
        assert c_out.shape == (K_a, K_b)
        # Level b consumes c_out. Since b has no row axis (c
        # already integrated it), feed empty idx.
        prior_b = torch.log(torch.ones(K_b) / K_b)
        b_out = marginalize_grouped(
            c_out,
            torch.zeros(0, dtype=torch.long),
            prior_b,
            1,
        )
        # b_out shape: (K_a,)
        assert b_out.shape == (K_a,)
        # Level a.
        prior_a = torch.log(torch.ones(K_a) / K_a)
        a_out = marginalize_grouped(
            b_out,
            torch.zeros(0, dtype=torch.long),
            prior_a,
            1,
        )
        # Hand-rolled three-level reference.
        grouped = torch.zeros(G_c, K_a, K_b, K_c)
        grouped = grouped.index_add(0, idx_c, ll)
        per_group_c = torch.logsumexp(prior_c + grouped, dim=-1)  # (G_c, K_a, K_b)
        c_marg = per_group_c.sum(dim=0)  # (K_a, K_b)
        b_marg = torch.logsumexp(prior_b + c_marg, dim=-1)  # (K_a,)
        a_marg = torch.logsumexp(prior_a + b_marg, dim=-1)  # scalar
        assert torch.allclose(a_out, a_marg, atol=1e-6)


# ---------------------------------------------------------------------------
# Varied index map shapes
# ---------------------------------------------------------------------------


class TestVariedIndexMaps:
    """Cover the different index map shapes the issue calls out:
    identity (per-row mixture), coarser (multi-row group), trivial
    (single group), product (multi-axis)."""

    def test_identity_index_map_recovers_per_row_mixture(self) -> None:
        torch.manual_seed(0)
        N, K = 5, 3
        ll = torch.randn(N, K)
        idx_identity = torch.arange(N)
        prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(ll, idx_identity, prior, N)
        # Each row is its own group: per-row mixture, summed.
        expected = torch.logsumexp(prior + ll, dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_trivial_index_map_single_group(self) -> None:
        torch.manual_seed(0)
        N, K = 6, 4
        ll = torch.randn(N, K)
        idx_trivial = torch.zeros(N, dtype=torch.long)
        prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(ll, idx_trivial, prior, 1)
        # Single group: sum all rows then logsumexp.
        expected = torch.logsumexp(prior + ll.sum(dim=0), dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_coarser_index_map_two_groups(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 0, 0, 1, 1, 1])
        prior = torch.log(torch.ones(3) / 3)
        out = marginalize_grouped(ll, idx, prior, 2)
        # Two groups of 3 rows each: scatter-sum then per-group
        # logsumexp.
        per_group = ll.reshape(2, 3, 3).sum(dim=1)  # (2, 3)
        expected = torch.logsumexp(prior + per_group, dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_three_axis_product_index_map(self) -> None:
        torch.manual_seed(0)
        N, K = 12, 2
        G1, G2, G3 = 2, 3, 2
        ll = torch.randn(N, K)
        idx_a = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        idx_b = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
        idx_c = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(ll, (idx_a, idx_b, idx_c), prior, (G1, G2, G3))
        # Hand-rolled: scatter into a flat (G1*G2*G3,)-indexed
        # accumulator (row-major).
        flat_idx = idx_a * (G2 * G3) + idx_b * G3 + idx_c
        grouped = torch.zeros(G1 * G2 * G3, K)
        grouped = grouped.index_add(0, flat_idx, ll)
        expected = torch.logsumexp(prior + grouped, dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Mixed reduction modes across nesting
# ---------------------------------------------------------------------------


class TestMixedReductionsAcrossLevels:
    """Each nesting level can choose its own reduction. The
    primitive composes them correctly even when reductions
    differ."""

    def test_sum_then_logsumexp(self) -> None:
        torch.manual_seed(0)
        N, K_in, K_out = 6, 3, 2
        ll = torch.randn(N, K_out, K_in)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        prior_in = torch.log(torch.ones(K_in) / K_in)
        prior_out = torch.log(torch.ones(K_out) / K_out)
        # Inner uses sum reduction.
        inner = marginalize_grouped(ll, idx, prior_in, 3, reduction="sum")
        # Outer uses logsumexp reduction.
        out = marginalize_grouped(
            inner,
            torch.zeros(0, dtype=torch.long),
            prior_out,
            1,
            reduction="logsumexp",
        )
        grouped = torch.zeros(3, K_out, K_in)
        grouped = grouped.index_add(0, idx, ll)
        per_group = (prior_in + grouped).sum(dim=-1)  # (G, K_out)
        per_outer = per_group.sum(dim=0)  # (K_out,)
        expected = torch.logsumexp(prior_out + per_outer, dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_mean_at_intermediate_level(self) -> None:
        torch.manual_seed(0)
        N, K = 4, 3
        ll = torch.randn(N, K)
        idx = torch.tensor([0, 0, 1, 1])
        prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(ll, idx, prior, 2, reduction="mean")
        grouped = torch.zeros(2, K)
        grouped = grouped.index_add(0, idx, ll)
        expected = (prior + grouped).mean(dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)
