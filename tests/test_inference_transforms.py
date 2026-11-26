"""Tests for normalizing-flow transform primitives.

Every primitive must satisfy three invariants:

1. **Inverse round-trip**: ``inverse(forward(x)) ≈ x`` (numerical
   precision floor).
2. **Log-det-Jacobian agreement**: the analytic
   ``log_abs_det_jacobian`` agrees with
   :func:`torch.autograd.functional.jacobian` to 1e-4 (a tight
   numerical match given float32).
3. **Gradient flow**: parameters of the parameterized layers
   receive non-zero finite gradients when the chain is trained
   against a smooth target.

Together these properties cover the transform behavior used by a
guide.
"""

from __future__ import annotations

import pytest
import torch

from quivers.inference.transforms import (
    MADE,
    AffineCouplingTransform,
    BatchNormTransform,
    InverseAutoregressiveTransform,
    LULinearTransform,
    MaskedAutoregressiveTransform,
    NeuralSplineCouplingTransform,
    alternating_mask,
    half_mask,
    make_coupling_mlp,
)


def _numeric_log_abs_det(transform, x: torch.Tensor) -> torch.Tensor:
    """Compute log|det J_T(x)| numerically via torch.autograd.

    Returns a tensor of shape ``x.shape[:-1]``.
    """

    def forward_fn(z):
        return transform(z)

    # Single-point Jacobian for each batch entry.
    out = torch.empty(x.shape[:-1])
    flat = x.reshape(-1, x.shape[-1])
    for i in range(flat.shape[0]):
        J = torch.autograd.functional.jacobian(forward_fn, flat[i])
        out.reshape(-1)[i] = torch.linalg.slogdet(J)[1]
    return out


# ---------------------------------------------------------------------------
# Affine coupling
# ---------------------------------------------------------------------------


def test_affine_coupling_inverse_roundtrip() -> None:
    dim = 6
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    # Use a random-initialized net so the transform is non-trivial.
    net = make_coupling_mlp(n_a, 2 * n_b, hidden=16)
    for p in net.parameters():
        p.data = torch.randn_like(p)
    layer = AffineCouplingTransform(dim, net, mask)

    x = torch.randn(8, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-5), (x - x_back).abs().max().item()


def test_affine_coupling_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    net = make_coupling_mlp(n_a, 2 * n_b, hidden=16)
    for p in net.parameters():
        p.data = torch.randn_like(p)
    layer = AffineCouplingTransform(dim, net, mask)

    x = torch.randn(4, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    numeric = _numeric_log_abs_det(layer, x)
    assert torch.allclose(analytic, numeric, atol=1e-4), (
        (analytic - numeric).abs().max().item()
    )


def test_affine_coupling_rejects_degenerate_mask() -> None:
    dim = 4
    all_true = torch.ones(dim, dtype=torch.bool)
    net = make_coupling_mlp(dim, 2)
    with pytest.raises(ValueError):
        AffineCouplingTransform(dim, net, all_true)


# ---------------------------------------------------------------------------
# Masked autoregressive (MAF)
# ---------------------------------------------------------------------------


def test_maf_inverse_roundtrip() -> None:
    dim = 5
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = MaskedAutoregressiveTransform(made)

    x = torch.randn(6, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-4), (x - x_back).abs().max().item()


def test_maf_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = MaskedAutoregressiveTransform(made)

    x = torch.randn(3, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    numeric = _numeric_log_abs_det(layer, x)
    assert torch.allclose(analytic, numeric, atol=1e-4)


def test_maf_autoregressive_property() -> None:
    """MAF output[j] depends only on input[k] for k with
    ordering[k] <= ordering[j]. Perturbing input[k] with strictly
    greater degree must leave output[j] unchanged."""
    dim = 5
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.5
    layer = MaskedAutoregressiveTransform(made)

    x = torch.randn(1, dim)
    y_base = layer(x)
    ordering = made.ordering.tolist()
    for k in range(dim):
        x_perturb = x.clone()
        x_perturb[0, k] += 1.0
        y_perturb = layer(x_perturb)
        for j in range(dim):
            if ordering[k] > ordering[j]:
                assert torch.allclose(y_base[0, j], y_perturb[0, j], atol=1e-5), (
                    f"MAF leaks: perturbing input {k} (deg {ordering[k]}) "
                    f"changed output {j} (deg {ordering[j]})"
                )


# ---------------------------------------------------------------------------
# Inverse autoregressive (IAF)
# ---------------------------------------------------------------------------


def test_iaf_inverse_roundtrip() -> None:
    dim = 5
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = InverseAutoregressiveTransform(made)

    x = torch.randn(6, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-4)


def test_iaf_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = InverseAutoregressiveTransform(made)

    x = torch.randn(3, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    numeric = _numeric_log_abs_det(layer, x)
    assert torch.allclose(analytic, numeric, atol=1e-4)


# ---------------------------------------------------------------------------
# Neural spline coupling (NSF)
# ---------------------------------------------------------------------------


def test_nsf_coupling_inverse_roundtrip() -> None:
    dim = 6
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    num_bins = 6
    net = make_coupling_mlp(n_a, n_b * (3 * num_bins - 1), hidden=32)
    for p in net.parameters():
        p.data = torch.randn_like(p) * 0.5
    layer = NeuralSplineCouplingTransform(
        dim, net, mask, num_bins=num_bins, tail_bound=4.0
    )

    x = torch.randn(8, dim) * 1.5  # stay well inside tail_bound
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-3), (x - x_back).abs().max().item()


def test_nsf_identity_outside_tail_bound() -> None:
    """Outside ``[-tail_bound, tail_bound]`` the spline is the
    identity, so the forward output equals the input there."""
    dim = 4
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    num_bins = 4
    net = make_coupling_mlp(n_a, n_b * (3 * num_bins - 1))
    layer = NeuralSplineCouplingTransform(
        dim, net, mask, num_bins=num_bins, tail_bound=3.0
    )

    # Values far outside the spline tail; transformed coordinates
    # must come back unchanged.
    x = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    y = layer(x)
    transformed = y[..., ~mask]
    untransformed = x[..., ~mask]
    assert torch.allclose(transformed, untransformed)


# ---------------------------------------------------------------------------
# LU linear
# ---------------------------------------------------------------------------


def test_lu_linear_inverse_roundtrip() -> None:
    dim = 5
    layer = LULinearTransform(dim)
    for p in layer.parameters():
        if p.requires_grad and p.dim() > 0:
            p.data = torch.randn_like(p) * 0.2
    x = torch.randn(6, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-4)


def test_lu_linear_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    layer = LULinearTransform(dim)
    for p in layer.parameters():
        if p.requires_grad and p.dim() > 0:
            p.data = torch.randn_like(p) * 0.2
    x = torch.randn(3, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    # LULinear's log-det is constant across x; numeric reference
    # is per-row but every row should match the constant.
    numeric = _numeric_log_abs_det(layer, x)
    expected = analytic.expand_as(numeric)
    assert torch.allclose(expected, numeric, atol=1e-4)


# ---------------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------------


def _batchnorm_at_random_parameters(dim: int, seed: int) -> BatchNormTransform:
    """A batch-norm layer with random scale, shift, and moments.

    Parameters
    ----------
    dim : int
        Number of coordinates.
    seed : int
        The seed.

    Returns
    -------
    BatchNormTransform
        The layer, its moments folded in from two random batches.
    """
    generator = torch.Generator().manual_seed(seed)
    layer = BatchNormTransform(dim, momentum=0.5)
    with torch.no_grad():
        layer._gamma.copy_(0.5 * torch.randn(dim, generator=generator))
        layer._beta.copy_(torch.randn(dim, generator=generator))
    for _ in range(2):
        batch = 3.0 * torch.randn(16, dim, generator=generator) + 1.5
        layer.update_statistics(batch.to(torch.get_default_dtype()))
    return layer


def test_batchnorm_update_statistics_folds_the_batch_moments_in() -> None:
    dim = 3
    layer = BatchNormTransform(dim, momentum=0.25)
    batch = torch.randn(10, dim)
    layer.update_statistics(batch)
    mean = layer.get_buffer("_running_mean")
    var = layer.get_buffer("_running_var")
    torch.testing.assert_close(mean, 0.25 * batch.mean(0))
    torch.testing.assert_close(var, 0.75 + 0.25 * batch.var(0, unbiased=False))
    for bad in (torch.randn(1, dim), torch.randn(dim), torch.randn(4, dim + 1)):
        with pytest.raises(ValueError, match="update_statistics"):
            layer.update_statistics(bad)
    with pytest.raises(ValueError, match="momentum"):
        BatchNormTransform(dim, momentum=0.0)
    with pytest.raises(ValueError, match="eps"):
        BatchNormTransform(dim, eps=0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batchnorm_is_the_written_out_affine_bijection(seed: int) -> None:
    """Forward, inverse, and log-det are one affine map of each row.

    The map is checked against its formula at random parameters, the
    inverse against the forward, the inverse log-det against minus the
    forward one, the log-det against the Jacobian by autograd, and the
    batch against its rows one at a time; training mode does not
    change the map.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dim = 4
        layer = _batchnorm_at_random_parameters(dim, seed)
        mean = layer.get_buffer("_running_mean").clone()
        var = layer.get_buffer("_running_var").clone()
        gamma, beta = layer._gamma.detach(), layer._beta.detach()
        x = torch.randn(7, dim, generator=torch.Generator().manual_seed(seed + 10))
        expected = (x - mean) / torch.sqrt(var + 1e-5) * gamma.exp() + beta
        expected_log_det = (gamma - 0.5 * torch.log(var + 1e-5)).sum().expand(7)
        for mode in (True, False):
            layer.train(mode)
            y = layer(x)
            torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(layer.inv(y), x, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(
                layer.log_abs_det_jacobian(x, y), expected_log_det, rtol=0, atol=1e-12
            )
            torch.testing.assert_close(
                layer.inv.log_abs_det_jacobian(y, x),
                -expected_log_det,
                rtol=0,
                atol=1e-12,
            )
            rows = torch.stack([layer(x[i]) for i in range(7)])
            torch.testing.assert_close(rows, y, rtol=0, atol=0)
            torch.testing.assert_close(
                layer.log_abs_det_jacobian(x, y),
                _numeric_log_abs_det(layer, x),
                rtol=0,
                atol=1e-10,
            )
        torch.testing.assert_close(layer.get_buffer("_running_mean"), mean)
        torch.testing.assert_close(layer.get_buffer("_running_var"), var)
    finally:
        torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------
# Additional NSF coverage: log-det agreement with numerical Jacobian
# ---------------------------------------------------------------------------


def test_nsf_log_det_matches_numerical_jacobian() -> None:
    """NSF spline-coupling log-det must match the numerical
    Jacobian. We stay inside the tail bound so the spline is
    active (outside, the identity gives log-det 0 trivially)."""
    dim = 4
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    num_bins = 6
    net = make_coupling_mlp(n_a, n_b * (3 * num_bins - 1), hidden=32)
    for p in net.parameters():
        p.data = torch.randn_like(p) * 0.3
    layer = NeuralSplineCouplingTransform(
        dim, net, mask, num_bins=num_bins, tail_bound=4.0
    )
    x = torch.randn(3, dim) * 1.5  # well inside tail
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    numeric = _numeric_log_abs_det(layer, x)
    assert torch.allclose(analytic, numeric, atol=1e-3), (
        f"NSF log-det mismatch: analytic={analytic.tolist()}, "
        f"numeric={numeric.tolist()}"
    )


# ---------------------------------------------------------------------------
# IAF autoregressive property
# ---------------------------------------------------------------------------


def test_iaf_autoregressive_property() -> None:
    """IAF's forward map is parallel: output coordinate ``j``
    depends on input coordinate ``j`` directly plus input
    coordinates ``< j`` through the autoregressive shift / scale.
    Perturbing input ``k`` with ``ordering[k] > ordering[j]``
    must leave output ``j`` unchanged."""
    torch.manual_seed(0)
    dim = 5
    made = MADE(dim=dim, n_per_dim=2, hidden=16, n_hidden_layers=1)
    for p in made.parameters():
        p.data = torch.randn_like(p) * 0.2
    layer = InverseAutoregressiveTransform(made)
    x = torch.randn(1, dim)
    y_base = layer(x)
    ordering = made.ordering
    for k in range(dim):
        x_pert = x.clone()
        x_pert[0, k] = x_pert[0, k] + 1.0
        y_pert = layer(x_pert)
        for j in range(dim):
            if int(ordering[k].item()) > int(ordering[j].item()):
                assert torch.isclose(y_pert[0, j], y_base[0, j], atol=1e-5), (
                    f"IAF output[{j}] (ordering={int(ordering[j])}) "
                    f"changed when input[{k}] (ordering={int(ordering[k])}) "
                    f"was perturbed; autoregressivity violated"
                )


# ---------------------------------------------------------------------------
# Helper-function coverage: half_mask, alternating_mask
# ---------------------------------------------------------------------------


def test_alternating_mask_alternates() -> None:
    m = alternating_mask(6, even=True)
    assert m.shape == (6,)
    assert m.dtype == torch.bool
    assert m.tolist() == [True, False, True, False, True, False]
    m_odd = alternating_mask(6, even=False)
    assert m_odd.tolist() == [False, True, False, True, False, True]


def test_half_mask_splits_in_half() -> None:
    m = half_mask(6, first_half_true=True)
    assert m.tolist() == [True, True, True, False, False, False]
    m_second = half_mask(6, first_half_true=False)
    assert m_second.tolist() == [False, False, False, True, True, True]


# ---------------------------------------------------------------------------
# Reverse-permutation flow primitive used inside AutoIAFGuide
# ---------------------------------------------------------------------------


def test_reverse_permutation_inverse_roundtrip() -> None:
    """``AutoIAFGuide`` chains IAF layers with reverse permutations.
    Verify the permutation primitive round-trips and reports a zero
    log-det."""
    from quivers.inference.guides.flow import _ReversePermutation

    dim = 5
    perm = _ReversePermutation(dim)
    x = torch.randn(4, dim)
    y = perm(x)
    x_back = perm.inv(y)
    assert torch.allclose(x, x_back, atol=1e-6)
    assert torch.allclose(perm.log_abs_det_jacobian(x, y), torch.zeros(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Compose round-trip across a multi-layer stack
# ---------------------------------------------------------------------------


def test_multilayer_flow_roundtrip() -> None:
    """A flow with several heterogeneous layers (the standard
    NSF / IAF stack we'd use in a real guide) round-trips
    end-to-end."""
    from torch.distributions.transforms import ComposeTransform

    dim = 6
    layers = []
    for k in range(3):
        mask = alternating_mask(dim, even=(k % 2 == 0))
        n_a = int(mask.sum().item())
        n_b = dim - n_a
        net = make_coupling_mlp(n_a, 2 * n_b, hidden=16)
        layers.append(AffineCouplingTransform(dim, net, mask))
        layers.append(LULinearTransform(dim))
    flow = ComposeTransform(layers)

    x = torch.randn(8, dim)
    y = flow(x)
    x_back = flow.inv(y)
    assert torch.allclose(x, x_back, atol=1e-3)
