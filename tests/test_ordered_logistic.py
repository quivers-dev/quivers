"""Tests for the `OrderedLogistic` distribution and its DSL surface.

Three slices are covered:

1. The standalone
   [`OrderedLogistic`][quivers.continuous._ordered.OrderedLogistic]
   distribution: log-prob agrees with the closed-form ordered-logit
   cumulative-link formula, probabilities sum to 1, broadcasting is
   consistent across the shared-cutpoint and per-row-cutpoint shapes.
2. The DSL `observe ... <- OrderedLogistic(eta, cutpoints)` inline
   surface: trace produces a finite log-joint when `cutpoints` is
   supplied as host data.
3. The mixed-model use case: per-participant cutpoints gathered
   through a participant index produce per-row log-densities that
   match a hand-computed reference.
"""

from __future__ import annotations

import pytest
import torch

from quivers.continuous._ordered import OrderedLogistic
from quivers.dsl import loads
from quivers.inference.trace import trace


def _reference_log_prob(
    eta: torch.Tensor, cutpoints: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Hand-rolled ordered-logit log-density for cross-checking the
    `OrderedLogistic` implementation. Computes the cumulative-link
    probabilities from `sigmoid(c_k - eta)` differences and gathers
    the entry at ``y``.
    """
    eta_b = eta.unsqueeze(-1)
    cdf = torch.sigmoid(cutpoints - eta_b)
    zero = torch.zeros_like(cdf[..., :1])
    one = torch.ones_like(cdf[..., :1])
    padded = torch.cat([zero, cdf, one], dim=-1)
    probs = padded[..., 1:] - padded[..., :-1]
    idx = y.long().unsqueeze(-1).expand(*probs.shape[:-1], 1)
    return (
        probs.gather(-1, idx)
        .squeeze(-1)
        .clamp_min(
            torch.finfo(probs.dtype).tiny,
        )
        .log()
    )


def test_log_prob_matches_closed_form_shared_cutpoints() -> None:
    torch.manual_seed(0)
    eta = torch.randn(7)
    cutpoints = torch.tensor([-1.5, 0.0, 1.5])
    y = torch.tensor([0, 1, 2, 3, 2, 1, 0])
    dist = OrderedLogistic(eta, cutpoints)
    assert dist.num_categories == 4
    assert dist.batch_shape == torch.Size([7])
    actual = dist.log_prob(y)
    expected = _reference_log_prob(eta, cutpoints, y)
    torch.testing.assert_close(actual, expected)


def test_log_prob_matches_closed_form_per_row_cutpoints() -> None:
    torch.manual_seed(0)
    eta = torch.randn(5)
    cutpoints = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [-1.0, 0.5, 1.5],
            [-0.5, 0.0, 0.5],
            [-3.0, 0.0, 3.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    y = torch.tensor([0, 1, 2, 3, 1])
    dist = OrderedLogistic(eta, cutpoints)
    assert dist.batch_shape == torch.Size([5])
    actual = dist.log_prob(y)
    expected = _reference_log_prob(eta, cutpoints, y)
    torch.testing.assert_close(actual, expected)


def test_probabilities_sum_to_one() -> None:
    torch.manual_seed(0)
    eta = torch.randn(11)
    cutpoints = torch.tensor([-2.5, -0.5, 0.5, 2.5])
    dist = OrderedLogistic(eta, cutpoints)
    probs = dist._category_probs()
    assert probs.shape == torch.Size([11, 5])
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))


def test_sample_shape_and_support() -> None:
    torch.manual_seed(0)
    eta = torch.randn(8)
    cutpoints = torch.tensor([-1.0, 0.0, 1.0])
    dist = OrderedLogistic(eta, cutpoints)
    one = dist.sample()
    assert one.shape == torch.Size([8])
    assert one.min() >= 0 and one.max() <= 3
    many = dist.sample(torch.Size([4]))
    assert many.shape == torch.Size([4, 8])
    assert many.min() >= 0 and many.max() <= 3


def test_sample_accepts_tuple_and_list_shapes() -> None:
    """`Distribution.sample` accepts any `Sequence[int]` for the
    sample_shape argument. `OrderedLogistic.sample` used to depend on
    `sample_shape.numel()` and raised `AttributeError` on a plain
    tuple. Every shape form must return the documented shape.
    """
    torch.manual_seed(0)
    eta = torch.randn(5)
    cutpoints = torch.tensor([-1.0, 0.0, 1.0])
    dist = OrderedLogistic(eta, cutpoints)
    # tuple
    y_tuple = dist.sample((200,))
    assert y_tuple.shape == torch.Size([200, 5])
    # torch.Size
    y_size = dist.sample(torch.Size((200,)))
    assert y_size.shape == torch.Size([200, 5])
    # empty tuple: no leading axis
    y_empty = dist.sample(())
    assert y_empty.shape == torch.Size([5])
    # list
    y_list = dist.sample([3, 4])
    assert y_list.shape == torch.Size([3, 4, 5])
    # Rank-2 sample shape
    y_2d = dist.sample(torch.Size((3, 4)))
    assert y_2d.shape == torch.Size([3, 4, 5])


def test_rejects_zero_cutpoints() -> None:
    eta = torch.tensor([0.0])
    with pytest.raises(ValueError, match="cutpoints"):
        OrderedLogistic(eta, torch.tensor(0.0))
    with pytest.raises(ValueError, match="cutpoints"):
        OrderedLogistic(eta, torch.empty(0))


def test_dsl_inline_observe_with_per_row_cutpoints() -> None:
    program = loads("""
object Resp : FinSet 6
program ord : Resp -> Resp
    sample eta <- Normal(loc=0.0, scale=1.0)
    observe y : Resp <- OrderedLogistic(predictor=eta, cutpoints=row_cuts)
    return y
export ord
""")
    y = torch.tensor([0, 1, 2, 3, 1, 2]).long()
    row_cuts = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [-1.0, 0.5, 1.5],
            [-0.5, 0.0, 0.5],
            [-3.0, 0.0, 3.0],
            [-1.0, 0.0, 1.0],
            [-2.0, -1.0, 0.0],
        ]
    )
    tr = trace(
        program.morphism,
        torch.zeros(6, 1),
        observations={"y": y, "row_cuts": row_cuts},
    )
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_dsl_inline_observe_with_participant_indexed_cutpoints() -> None:
    """Ordinal mixed model: each participant carries its own
    cutpoints; the program gathers per-row cutpoints through a
    participant index. This is the canonical
    cumulative-link-with-random-thresholds shape.
    """
    program = loads("""
object Resp : FinSet 6
program ord : Resp -> Resp
    sample eta <- Normal(loc=0.0, scale=1.0)
    let row_cuts = cutpoints[participant_idx]
    observe y : Resp <- OrderedLogistic(predictor=eta, cutpoints=row_cuts)
    return y
export ord
""")
    y = torch.tensor([0, 1, 2, 3, 1, 2]).long()
    cutpoints = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [-1.0, 0.5, 1.5],
            [-0.5, 0.0, 0.5],
        ]
    )
    participant_idx = torch.tensor([0, 1, 2, 0, 1, 2]).long()
    tr = trace(
        program.morphism,
        torch.zeros(1, 1),
        observations={
            "y": y,
            "cutpoints": cutpoints,
            "participant_idx": participant_idx,
        },
    )
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()
    # One run, so one joint, which has to equal the sum of the sites it
    # is built from: a joint that repeated a shared site once per row
    # would have the same shape and the wrong value.
    assert tr.log_joint.numel() == 1
    site_total = sum(float(site.log_prob.sum()) for site in tr.sites.values())
    torch.testing.assert_close(
        float(tr.log_joint.sum()), site_total, atol=1e-5, rtol=1e-5
    )


def test_dsl_inline_observe_with_shared_cutpoints_vector() -> None:
    """The inline observe surface routes a shared 1-D cutpoints
    vector as a distribution parameter rather than a per-row input.
    `OrderedLogistic(predictor, cutpoints)` with `predictor: (N,)`
    and `cutpoints: (K-1,)` compiles and traces without stacking
    the two into a single per-row tensor.
    """
    program = loads("""
object Cut  : FinSet 4
object Resp : FinSet 100
program ordinal : Resp -> Resp
    sample eta <- Normal(loc=0.0, scale=1.0)
    observe y : Resp <- OrderedLogistic(predictor=eta, cutpoints=base)
    return y
export ordinal
""")
    N = 100
    torch.manual_seed(0)
    y = torch.randint(0, 5, (N,)).long()
    base = torch.tensor([-1.5, -0.5, 0.5, 1.5])
    # One run of the program: its rows come from the `Resp` plate.
    tr = trace(
        program.morphism,
        torch.zeros(1, 1),
        observations={"y": y, "base": base},
    )
    assert tr.log_joint is not None
    # One run, so one joint, which must equal the sum of its sites; see
    # the participant-indexed case above.
    assert tr.log_joint.numel() == 1
    torch.testing.assert_close(
        float(tr.log_joint.sum()),
        sum(float(site.log_prob.sum()) for site in tr.sites.values()),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(tr.log_joint).all()
    # Numeric equivalence: the DSL log-prob at the sampled eta must
    # match `OrderedLogistic(eta_row, base).log_prob(y).sum()`.
    eta_val = tr.sites["eta"].value.reshape(-1)
    eta_row = eta_val.expand(N) if eta_val.numel() == 1 else eta_val
    ref_lp = OrderedLogistic(eta_row, base).log_prob(y).sum()
    dsl_y_lp = tr.sites["y"].log_prob.sum()
    torch.testing.assert_close(dsl_y_lp, ref_lp, atol=1e-5, rtol=1e-5)


def test_dsl_inline_observe_with_sampled_cutpoints_vector() -> None:
    """A sampled cutpoints vector reaches the inline observe as one
    distribution parameter shared by every row, as a data-supplied one
    does.
    `OrderedLogistic(predictor, cutpoints)` with `predictor: (N,)`
    and `cutpoints: (K-1,)` compiles and traces without stacking
    the two into a single per-row tensor.
    """
    program = loads("""
object Cut  : FinSet 4
object Resp : FinSet 100
program ordinal : Resp -> Resp
    sample eta <- Normal(loc=0.0, scale=1.0)
    sample base <- Normal(loc=0.0, scale=2.0) [over=Cut]
    observe y : Resp <- OrderedLogistic(predictor=eta, cutpoints=base)
    return y
export ordinal
""")
    N = 100
    torch.manual_seed(0)
    y = torch.randint(0, 5, (N,)).long()
    base = torch.tensor([-1.5, -0.5, 0.5, 1.5])
    # One run of the program: its rows come from the `Resp` plate.
    tr = trace(
        program.morphism,
        torch.zeros(1, 1),
        observations={"y": y, "base": base},
    )
    assert tr.log_joint is not None
    # One run, so one joint, which must equal the sum of its sites.
    assert tr.log_joint.numel() == 1
    torch.testing.assert_close(
        float(tr.log_joint.sum()),
        sum(float(site.log_prob.sum()) for site in tr.sites.values()),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(tr.log_joint).all()
    # Numeric equivalence: the DSL log-prob at the sampled eta must
    # match `OrderedLogistic(eta_row, base).log_prob(y).sum()`.
    eta_val = tr.sites["eta"].value.reshape(-1)
    eta_row = eta_val.expand(N) if eta_val.numel() == 1 else eta_val
    ref_lp = OrderedLogistic(eta_row, base).log_prob(y).sum()
    dsl_y_lp = tr.sites["y"].log_prob.sum()
    torch.testing.assert_close(dsl_y_lp, ref_lp, atol=1e-5, rtol=1e-5)
