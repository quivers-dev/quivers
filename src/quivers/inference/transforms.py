"""Bijector and normalizing-flow primitives.

This module collects the differentiable bijections quivers's
variational guides chain through. Every primitive is a subclass
of `torch.distributions.transforms.Transform`, so they
compose under `torch.distributions.transforms.ComposeTransform`,
plug into `torch.distributions.transforms.biject_to`-style
pipelines, and interop cleanly with
`torch.distributions.transformed_distribution.TransformedDistribution`.

Coverage
========

* `AffineCouplingTransform`: RealNVP coupling layer
  (Dinh, Sohl-Dickstein, Bengio 2017,
  `doi:10.48550/arXiv.1605.08803 <https://doi.org/10.48550/arXiv.1605.08803>`_).
  Splits the input into two halves, scales / shifts one half
  conditional on the other.
* `MaskedAutoregressiveTransform`: MAF
  (Papamakarios, Pavlakou, Murray 2017,
  `doi:10.48550/arXiv.1705.07057 <https://doi.org/10.48550/arXiv.1705.07057>`_).
  Forward pass is parallel (single masked MLP call); inverse is
  sequential (per-coordinate). The mass-density form used by
  variational inference.
* `InverseAutoregressiveTransform`: IAF
  (Kingma, Salimans, Jozefowicz et al. 2016,
  `doi:10.48550/arXiv.1606.04934 <https://doi.org/10.48550/arXiv.1606.04934>`_).
  The dual of MAF: sampling is parallel, density evaluation is
  sequential. Preferred when you sample more often than score
  (which is true for variational guides).
* `NeuralSplineCouplingTransform`: monotone rational-quadratic
  spline coupling (Durkan, Bekasov, Murray, Papamakarios 2019,
  `doi:10.48550/arXiv.1906.04032 <https://doi.org/10.48550/arXiv.1906.04032>`_).
  Allows nonlinear monotone transformations within each coupling
  layer. Expressiveness depends on the spline and network configuration.
* `LULinearTransform`: LU-decomposed linear permutation
  (Kingma, Dhariwal 2018 Glow,
  `doi:10.48550/arXiv.1807.03039 <https://doi.org/10.48550/arXiv.1807.03039>`_).
  Cheap learnable invertible linear layer; the standard inter-
  coupling-layer mixer.
* `BatchNormTransform`: running-statistics-based BN as a
  normalizing-flow transform (Dinh, Sohl-Dickstein, Bengio 2017,
  same paper as RealNVP).

Every primitive is fully implemented (forward, inverse,
``log_abs_det_jacobian``). No primitive raises ``NotImplementedError``
under any input its type signature accepts. The masked-network
factories are designed so callers can pass a callable / module
rather than configuring topology through string flags.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints as _constraints
from torch.distributions.transforms import Transform

_EPS = 1e-6


class TransformModule(Transform, nn.Module):
    """Cooperative base for transforms that also need to register
    `torch.nn.Parameter` and `torch.nn.Module.register_buffer`
    attributes.

    `torch.distributions.transforms.Transform` is not a
    `torch.nn.Module`, so subclassing both is required for
    any flow primitive that holds learnable parameters or buffers
    (every primitive in this module). The cooperative base calls
    both ``__init__``\\ s in MRO order, `nn.Module` first
    so attribute storage is set up before `Transform`
    populates the cache-size machinery.
    """

    def __init__(self, cache_size: int = 0) -> None:
        nn.Module.__init__(self)
        Transform.__init__(self, cache_size=cache_size)

    # ``torch.distributions.transforms.Transform`` overrides
    # ``__eq__`` (it compares by structural identity), which
    # silently strips ``__hash__`` on Python 3. ``nn.Module``
    # iteration (used by ``parameters()``, ``state_dict``, the
    # forward hooks) requires module instances to be hashable.
    # Restore identity-based hashing so the module machinery
    # works.
    __hash__ = object.__hash__  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Affine coupling (RealNVP)
# ---------------------------------------------------------------------------


class AffineCouplingTransform(TransformModule):
    """RealNVP-style affine coupling transform.

    Splits the input vector into two parts: an "identity" part
    ``x_a`` and a "transformed" part ``x_b``. The transformed part
    is shifted and scaled by parameters output from a neural
    network that sees only the identity part. The Jacobian is
    triangular, so its log-determinant is the sum of the log-scales
    on ``x_b``.

    Parameters
    ----------
    dim : int
        Total event dimension.
    net : torch.nn.Module
        Module mapping ``x_a`` of shape ``(..., dim_a)`` to
        ``(log_scale_b, shift_b)`` of shape ``(..., 2 * dim_b)``.
        The factory `make_coupling_mlp` builds a sensible
        default.
    mask : torch.Tensor
        Boolean mask of shape ``(dim,)``: ``True`` entries form
        the identity part, ``False`` entries form the transformed
        part. Permuting the mask between consecutive coupling
        layers is the canonical RealNVP recipe.
    """

    domain = _constraints.real_vector
    codomain = _constraints.real_vector
    bijective = True

    def __init__(self, dim: int, net: nn.Module, mask: torch.Tensor) -> None:
        TransformModule.__init__(self, cache_size=1)
        if mask.dtype != torch.bool:
            raise TypeError(
                f"AffineCouplingTransform: mask must be bool, got {mask.dtype}"
            )
        if mask.shape != (dim,):
            raise ValueError(
                f"AffineCouplingTransform: mask shape {tuple(mask.shape)} "
                f"does not match dim={dim}"
            )
        n_a = int(mask.sum().item())
        n_b = dim - n_a
        if n_a == 0 or n_b == 0:
            raise ValueError(
                f"AffineCouplingTransform: mask must split the input into "
                f"two non-empty parts; got {n_a} identity / {n_b} transformed"
            )
        self._dim = dim
        self._n_a = n_a
        self._n_b = n_b
        self.net = net
        self.register_buffer("_mask", mask, persistent=True)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_a = x[..., self._mask]
        x_b = x[..., ~self._mask]
        return x_a, x_b

    def _join(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            *x_a.shape[:-1], self._dim, device=x_a.device, dtype=x_a.dtype
        )
        out[..., self._mask] = x_a
        out[..., ~self._mask] = x_b
        return out

    def _params(self, x_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x_a)
        if raw.shape[-1] != 2 * self._n_b:
            raise RuntimeError(
                f"AffineCouplingTransform: coupling net produced "
                f"{raw.shape[-1]} outputs but expected 2 * n_b "
                f"= {2 * self._n_b}"
            )
        log_scale = raw[..., : self._n_b]
        shift = raw[..., self._n_b :]
        # Soft-clamp the log-scale so the affine transform stays
        # well-conditioned during early training.
        log_scale = torch.tanh(log_scale) * 5.0
        return log_scale, shift

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        x_a, x_b = self._split(x)
        log_scale, shift = self._params(x_a)
        y_b = x_b * log_scale.exp() + shift
        return self._join(x_a, y_b)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_a, y_b = self._split(y)
        log_scale, shift = self._params(y_a)
        x_b = (y_b - shift) * (-log_scale).exp()
        return self._join(y_a, x_b)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        x_a, _ = self._split(x)
        log_scale, _ = self._params(x_a)
        return log_scale.sum(dim=-1)


def make_coupling_mlp(
    n_in: int,
    n_out: int,
    *,
    hidden: int = 64,
    n_hidden_layers: int = 2,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """Build an MLP for use as the ``net`` argument of
    `AffineCouplingTransform` or
    `NeuralSplineCouplingTransform`.

    The output layer is initialized to zero so the transform
    starts as the identity, which is a standard normalizing-flow
    initialisation (see Glow, Real NVP).
    """
    layers: list[nn.Module] = [nn.Linear(n_in, hidden), activation()]
    for _ in range(n_hidden_layers - 1):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(activation())
    final = nn.Linear(hidden, n_out)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Masked autoregressive (MAF)
# ---------------------------------------------------------------------------


class _MaskedLinear(nn.Linear):
    """Linear layer with a fixed binary mask applied to its weight.

    Used inside `MADE` to enforce the autoregressive
    structure: output unit ``j`` depends only on inputs with
    "degree" strictly less than ``j``'s degree (or ≤ for the
    final output layer).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
    ) -> None:
        super().__init__(in_features, out_features)
        if mask.shape != (out_features, in_features):
            raise ValueError(
                f"_MaskedLinear: mask shape {tuple(mask.shape)} does not "
                f"match (out_features, in_features) = "
                f"({out_features}, {in_features})"
            )
        self.register_buffer("mask", mask.float(), persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation
    (Germain, Gregor, Murray, and Larochelle 2015) [Germain et al. (2015)](<https://proceedings.mlr.press/v37/germain15.html>).

    Outputs ``n_per_dim`` parameters per input coordinate. The
    masking guarantees that output parameters for coordinate ``j``
    depend only on input coordinates ``< j`` in the supplied
    ordering, which is the autoregressive property MAF / IAF
    exploit.
    """

    def __init__(
        self,
        dim: int,
        n_per_dim: int,
        *,
        hidden: int = 64,
        n_hidden_layers: int = 2,
        ordering: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if ordering is None:
            ordering = torch.arange(dim)
        if ordering.shape != (dim,) or ordering.dtype != torch.long:
            raise TypeError("MADE: ordering must be a long tensor of shape (dim,)")
        self._dim = dim
        self._n_per_dim = n_per_dim
        self.register_buffer("_ordering", ordering, persistent=True)
        # Assign integer "degrees" to each hidden unit so that the
        # masking enforces autoregressivity.
        hidden_degrees = [
            torch.randint(1, dim, (hidden,)) for _ in range(n_hidden_layers)
        ]
        # Input degrees: ordering itself; output degrees: ordering
        # repeated n_per_dim times (one block per output parameter).
        in_degrees = ordering
        layers: list[nn.Module] = []
        prev_degrees = in_degrees
        prev_size = dim
        for h_degrees in hidden_degrees:
            mask = (h_degrees.unsqueeze(1) >= prev_degrees.unsqueeze(0)).to(
                torch.float32
            )
            layers.append(_MaskedLinear(prev_size, hidden, mask))
            layers.append(nn.ReLU())
            prev_degrees = h_degrees
            prev_size = hidden
        # Final layer: strict-less-than for the autoregressive
        # property; one output block per parameter.
        out_degrees = ordering.repeat(n_per_dim)
        final_mask = (out_degrees.unsqueeze(1) > prev_degrees.unsqueeze(0)).to(
            torch.float32
        )
        final = _MaskedLinear(prev_size, dim * n_per_dim, final_mask)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_per_dim(self) -> int:
        return self._n_per_dim

    @property
    def ordering(self) -> torch.Tensor:
        return self._ordering  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        # Reshape so trailing axes are (dim, n_per_dim).
        return raw.reshape(*x.shape[:-1], self._n_per_dim, self._dim).transpose(-1, -2)


class MaskedAutoregressiveTransform(TransformModule):
    """Masked Autoregressive Flow (MAF) layer.

    Forward (density) is parallel, one MADE call. Inverse
    (sampling) is sequential in the autoregressive order.

    Parameters
    ----------
    made : MADE
        Masked network producing per-coordinate ``(shift, log_scale)``.
    """

    domain = _constraints.real_vector
    codomain = _constraints.real_vector
    bijective = True

    def __init__(self, made: MADE) -> None:
        super().__init__(cache_size=1)
        if made.n_per_dim != 2:
            raise ValueError(
                f"MaskedAutoregressiveTransform: MADE must have "
                f"n_per_dim == 2 (shift, log_scale), got {made.n_per_dim}"
            )
        self.made = made

    def _params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.made(x)
        shift = raw[..., 0]
        log_scale = raw[..., 1]
        log_scale = torch.tanh(log_scale) * 5.0
        return shift, log_scale

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        shift, log_scale = self._params(x)
        return (x - shift) * (-log_scale).exp()

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        # Sequential inversion: for each coordinate in the
        # autoregressive order, evaluate the network on the
        # partially-reconstructed input.
        ordering = self.made.ordering
        # Order the indices by their degree, lowest first.
        order_idx = torch.argsort(ordering)
        x = torch.zeros_like(y)
        for i in order_idx.tolist():
            shift, log_scale = self._params(x)
            x_i = y[..., i] * log_scale[..., i].exp() + shift[..., i]
            x = x.clone()
            x[..., i] = x_i
        return x

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        _, log_scale = self._params(x)
        return -log_scale.sum(dim=-1)


class InverseAutoregressiveTransform(TransformModule):
    """Inverse Autoregressive Flow (IAF) layer.

    Dual of `MaskedAutoregressiveTransform`: sampling
    (forward) is parallel, density (inverse) is sequential. The
    preferred direction for variational guides because we sample
    more often than we score.

    Parameters
    ----------
    made : MADE
        Masked network producing per-coordinate ``(shift, log_scale)``.
    """

    domain = _constraints.real_vector
    codomain = _constraints.real_vector
    bijective = True

    def __init__(self, made: MADE) -> None:
        super().__init__(cache_size=1)
        if made.n_per_dim != 2:
            raise ValueError(
                f"InverseAutoregressiveTransform: MADE must have "
                f"n_per_dim == 2 (shift, log_scale), got {made.n_per_dim}"
            )
        self.made = made

    def _params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.made(x)
        shift = raw[..., 0]
        log_scale = raw[..., 1]
        log_scale = torch.tanh(log_scale) * 5.0
        return shift, log_scale

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        shift, log_scale = self._params(x)
        return x * log_scale.exp() + shift

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        ordering = self.made.ordering
        order_idx = torch.argsort(ordering)
        x = torch.zeros_like(y)
        for i in order_idx.tolist():
            shift, log_scale = self._params(x)
            x_i = (y[..., i] - shift[..., i]) * (-log_scale[..., i]).exp()
            x = x.clone()
            x[..., i] = x_i
        return x

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        _, log_scale = self._params(x)
        return log_scale.sum(dim=-1)


# ---------------------------------------------------------------------------
# Neural spline coupling (NSF)
# ---------------------------------------------------------------------------


def _searchsorted(bin_locs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Return per-input bin index along the last axis."""
    return torch.sum(inputs.unsqueeze(-1) >= bin_locs, dim=-1) - 1


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    *,
    inverse: bool,
    tail_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monotone rational-quadratic spline (Durkan et al. 2019).

    Inside ``[-tail_bound, tail_bound]``: a monotone rational-
    quadratic spline with the given knot widths / heights /
    derivatives. Outside: the identity (with log-Jacobian zero),
    so the transform stays a valid bijection on the whole real
    line.

    Both directions (forward and inverse) are closed-form; the
    inverse solves a quadratic per input via the quadratic
    formula.
    """
    num_bins = unnormalized_widths.shape[-1]

    in_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    out_mask = ~in_mask

    out_inputs = inputs[in_mask]
    out_widths = unnormalized_widths[in_mask]
    out_heights = unnormalized_heights[in_mask]
    out_derivatives = unnormalized_derivatives[in_mask]

    # Normalize widths / heights to sum to 2 * tail_bound (the
    # spline lives on [-tail_bound, tail_bound]).
    widths = F.softmax(out_widths, dim=-1) * 2 * tail_bound
    heights = F.softmax(out_heights, dim=-1) * 2 * tail_bound
    derivatives = F.softplus(out_derivatives) + _EPS

    cum_widths = torch.cumsum(widths, dim=-1)
    cum_widths = F.pad(cum_widths, (1, 0), value=0.0) - tail_bound
    cum_widths[..., -1] = tail_bound

    cum_heights = torch.cumsum(heights, dim=-1)
    cum_heights = F.pad(cum_heights, (1, 0), value=0.0) - tail_bound
    cum_heights[..., -1] = tail_bound

    # Pad derivatives with the boundary slope (= 1) at both ends
    # so the transform is the identity outside the spline.
    derivatives = F.pad(derivatives, (1, 1), value=1.0)

    if inverse:
        bin_idx = _searchsorted(cum_heights, out_inputs)
    else:
        bin_idx = _searchsorted(cum_widths, out_inputs)
    bin_idx = bin_idx.clamp(0, num_bins - 1)

    input_cum_widths = cum_widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_cum_heights = cum_heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_derivatives = derivatives.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_derivatives_plus_one = derivatives.gather(
        -1, (bin_idx + 1).unsqueeze(-1)
    ).squeeze(-1)

    s = input_bin_heights / input_bin_widths

    if inverse:
        a = (out_inputs - input_cum_heights) * (
            input_derivatives + input_derivatives_plus_one - 2 * s
        ) + input_bin_heights * (s - input_derivatives)
        b = input_bin_heights * input_derivatives - (out_inputs - input_cum_heights) * (
            input_derivatives + input_derivatives_plus_one - 2 * s
        )
        c = -s * (out_inputs - input_cum_heights)
        disc = b**2 - 4 * a * c
        if (disc < 0).any():
            raise RuntimeError(
                "rational-quadratic spline inverse: negative discriminant. "
                "This indicates a numerical issue in the spline parameters."
            )
        root = (2 * c) / (-b - torch.sqrt(disc))
        out_in_outputs = root * input_bin_widths + input_cum_widths
        theta = root
        theta_one_minus = theta * (1 - theta)
        denom = (
            s
            + (input_derivatives + input_derivatives_plus_one - 2 * s) * theta_one_minus
        )
        derivative_numerator = s.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * s * theta_one_minus
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = -(torch.log(derivative_numerator) - 2 * torch.log(denom))
    else:
        theta = (out_inputs - input_cum_widths) / input_bin_widths
        theta_one_minus = theta * (1 - theta)
        numerator = input_bin_heights * (
            s * theta.pow(2) + input_derivatives * theta_one_minus
        )
        denom = (
            s
            + (input_derivatives + input_derivatives_plus_one - 2 * s) * theta_one_minus
        )
        out_in_outputs = input_cum_heights + numerator / denom
        derivative_numerator = s.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * s * theta_one_minus
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denom)

    outputs = inputs.clone()
    log_abs_det = torch.zeros_like(inputs)
    outputs[in_mask] = out_in_outputs
    log_abs_det[in_mask] = logabsdet
    outputs[out_mask] = inputs[out_mask]  # identity outside
    return outputs, log_abs_det


class NeuralSplineCouplingTransform(TransformModule):
    """Monotone rational-quadratic spline coupling
    (Durkan-Bekasov-Murray-Papamakarios 2019).

    Same coupling structure as `AffineCouplingTransform`,
    but the per-coordinate transform on ``x_b`` is a monotone
    rational-quadratic spline rather than an affine map. The
    network outputs ``(unnormalized_widths, unnormalized_heights,
    unnormalized_derivatives)`` per transformed coordinate;
    ``num_bins`` knots per coordinate.

    Parameters
    ----------
    dim : int
        Event dimension.
    net : torch.nn.Module
        Maps ``x_a`` of shape ``(..., n_a)`` to spline parameters
        of shape ``(..., n_b * (3 * num_bins - 1))``.
        `make_coupling_mlp` produces a sensible default with
        ``n_out = n_b * (3 * num_bins - 1)``.
    mask : torch.Tensor
        Boolean mask of shape ``(dim,)``.
    num_bins : int
        Number of spline knots.
    tail_bound : float
        Domain of the spline; outside ``[-tail_bound, tail_bound]``
        the transform is the identity.
    """

    domain = _constraints.real_vector
    codomain = _constraints.real_vector
    bijective = True

    def __init__(
        self,
        dim: int,
        net: nn.Module,
        mask: torch.Tensor,
        *,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ) -> None:
        super().__init__(cache_size=1)
        if mask.dtype != torch.bool or mask.shape != (dim,):
            raise ValueError(
                "NeuralSplineCouplingTransform: mask must be bool of shape "
                f"(dim,) = ({dim},); got {tuple(mask.shape)} {mask.dtype}"
            )
        n_a = int(mask.sum().item())
        n_b = dim - n_a
        if n_a == 0 or n_b == 0:
            raise ValueError(
                "NeuralSplineCouplingTransform: mask must split the input "
                f"into two non-empty parts; got {n_a} / {n_b}"
            )
        self._dim = dim
        self._n_a = n_a
        self._n_b = n_b
        self._num_bins = num_bins
        self._tail_bound = tail_bound
        self.net = net
        self.register_buffer("_mask", mask, persistent=True)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., self._mask], x[..., ~self._mask]

    def _join(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            *x_a.shape[:-1], self._dim, device=x_a.device, dtype=x_a.dtype
        )
        out[..., self._mask] = x_a
        out[..., ~self._mask] = x_b
        return out

    def _spline_params(
        self, x_a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x_a)
        # net's output is (..., n_b * (3 * num_bins - 1))
        n_params = 3 * self._num_bins - 1
        if raw.shape[-1] != self._n_b * n_params:
            raise RuntimeError(
                f"NeuralSplineCouplingTransform: coupling net produced "
                f"{raw.shape[-1]} outputs; expected n_b * (3 * num_bins - 1) "
                f"= {self._n_b * n_params}"
            )
        raw = raw.reshape(*raw.shape[:-1], self._n_b, n_params)
        widths = raw[..., : self._num_bins]
        heights = raw[..., self._num_bins : 2 * self._num_bins]
        derivatives = raw[..., 2 * self._num_bins :]
        return widths, heights, derivatives

    def _apply_spline(
        self,
        x: torch.Tensor,
        *,
        inverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = self._split(x)
        widths, heights, derivatives = self._spline_params(x_a)
        out_b, log_det = _rational_quadratic_spline(
            x_b,
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tail_bound=self._tail_bound,
        )
        return self._join(x_a, out_b), log_det.sum(dim=-1)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self._apply_spline(x, inverse=False)
        return y

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        x, _ = self._apply_spline(y, inverse=True)
        return x

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        _, log_det = self._apply_spline(x, inverse=False)
        return log_det


# ---------------------------------------------------------------------------
# LU-decomposed linear (Glow 1×1 conv generalisation)
# ---------------------------------------------------------------------------


class LULinearTransform(TransformModule):
    """Invertible linear layer parameterized as :math:`A = P L U`
    (Kingma & Dhariwal 2018 Glow).

    The unit-lower-triangular ``L`` and unit-upper-triangular ``U``
    factors are stored along with their respective off-diagonal
    weights; the diagonal of ``U`` is stored as a log-scale so the
    layer remains invertible and the log-determinant is the sum
    of the log-scales. ``P`` is a fixed permutation chosen at
    construction.
    """

    domain = _constraints.real_vector
    codomain = _constraints.real_vector
    bijective = True

    def __init__(self, dim: int, perm: torch.Tensor | None = None) -> None:
        super().__init__(cache_size=1)
        self._dim = dim
        # Lower-triangular off-diagonal (strict)
        self._lower = nn.Parameter(torch.zeros(dim, dim))
        # Upper-triangular off-diagonal (strict)
        self._upper = nn.Parameter(torch.zeros(dim, dim))
        # Log-magnitude of U's diagonal; sign baked into the
        # signed-zero init below so the layer starts as identity.
        self._log_diag = nn.Parameter(torch.zeros(dim))
        if perm is None:
            perm = torch.arange(dim)
        if perm.shape != (dim,):
            raise ValueError(
                f"LULinearTransform: perm shape {tuple(perm.shape)} != ({dim},)"
            )
        self.register_buffer("_perm", perm.long(), persistent=True)
        # The fixed permutation matrix's log-det is 0 (it's a
        # signed permutation; we take +1 here).
        self.register_buffer(
            "_lower_mask",
            torch.tril(torch.ones(dim, dim), diagonal=-1),
            persistent=True,
        )
        self.register_buffer(
            "_upper_mask",
            torch.triu(torch.ones(dim, dim), diagonal=1),
            persistent=True,
        )
        self.register_buffer("_eye", torch.eye(dim), persistent=True)

    def _L(self) -> torch.Tensor:
        return self._lower * self._lower_mask + self._eye

    def _U(self) -> torch.Tensor:
        return self._upper * self._upper_mask + torch.diag(self._log_diag.exp())

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        # y = P L U x
        y = x[..., self._perm]
        y = y @ self._U().T
        y = y @ self._L().T
        return y

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        # x = U^{-1} L^{-1} P^T y
        L = self._L()
        U = self._U()
        z = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False)
        x = torch.linalg.solve_triangular(U, z, upper=True).squeeze(-1)
        inv_perm = torch.empty_like(self._perm)
        inv_perm[self._perm] = torch.arange(self._dim, device=self._perm.device)
        return x[..., inv_perm]

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del x, y
        # det(P L U) = det(P) * det(L) * det(U) = ±1 * 1 * Π exp(log_diag)
        return self._log_diag.sum().expand(())


# ---------------------------------------------------------------------------
# BatchNormTransform
# ---------------------------------------------------------------------------


class BatchNormTransform(TransformModule):
    """Batch-normalisation as a normalizing-flow layer
    (Dinh, Sohl-Dickstein, Bengio 2017).

    In ``training`` mode uses the batch statistics; in
    ``eval`` mode uses the cached running statistics. This module
    is an `torch.nn.Module` to register the running buffers,
    and a `torch.distributions.transforms.Transform` so it
    composes with the rest of the flow stack.
    """

    domain = _constraints.real_vector
    codomain = _constraints.real_vector
    bijective = True

    def __init__(
        self,
        dim: int,
        *,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(cache_size=1)
        self._dim = dim
        self._momentum = momentum
        self._eps = eps
        self._gamma = nn.Parameter(torch.zeros(dim))  # log-scale
        self._beta = nn.Parameter(torch.zeros(dim))
        self.register_buffer("_running_mean", torch.zeros(dim), persistent=True)
        self.register_buffer("_running_var", torch.ones(dim), persistent=True)
        self._training = True

    def train(self, mode: bool = True) -> "BatchNormTransform":
        self._training = mode
        return self

    def eval(self) -> "BatchNormTransform":
        return self.train(False)

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._training:
            # Compute over every axis except the last (event) axis.
            reduce_dims = tuple(range(x.dim() - 1))
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            with torch.no_grad():
                self._running_mean.mul_(1 - self._momentum).add_(mean * self._momentum)
                self._running_var.mul_(1 - self._momentum).add_(var * self._momentum)
            return mean, var
        return self._running_mean, self._running_var

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._stats(x)
        x_hat = (x - mean) / torch.sqrt(var + self._eps)
        return x_hat * self._gamma.exp() + self._beta

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        # Inverse uses the running stats (the batch stats from the
        # forward pass are stale by inverse time).
        mean = self._running_mean
        var = self._running_var
        x_hat = (y - self._beta) * (-self._gamma).exp()
        return x_hat * torch.sqrt(var + self._eps) + mean

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        _, var = self._stats(x)
        # Each coordinate contributes gamma - 0.5 * log(var + eps).
        per_coord = self._gamma - 0.5 * torch.log(var + self._eps)
        return per_coord.sum().expand(x.shape[:-1])


# ---------------------------------------------------------------------------
# Utility: mask factories
# ---------------------------------------------------------------------------


def alternating_mask(dim: int, *, even: bool = True) -> torch.Tensor:
    """Boolean mask alternating ``True`` / ``False``. Even-indexed
    entries are ``True`` when ``even=True``."""
    indices = torch.arange(dim)
    return (indices % 2 == 0) if even else (indices % 2 == 1)


def half_mask(dim: int, *, first_half_true: bool = True) -> torch.Tensor:
    """Boolean mask with the first / second half ``True``."""
    mask = torch.zeros(dim, dtype=torch.bool)
    half = dim // 2
    if first_half_true:
        mask[:half] = True
    else:
        mask[half:] = True
    return mask


__all__ = [
    "TransformModule",
    "AffineCouplingTransform",
    "MaskedAutoregressiveTransform",
    "InverseAutoregressiveTransform",
    "NeuralSplineCouplingTransform",
    "LULinearTransform",
    "BatchNormTransform",
    "MADE",
    "make_coupling_mlp",
    "alternating_mask",
    "half_mask",
]
