"""Scan combinator: temporal recurrence over sequences.

A ScanMorphism wraps a recurrent cell and applies it across a
sequence, threading hidden state from one time step to the next.
This implements the standard RNN pattern:

    h_t = cell(x_t, h_{t-1})

where cell : A * H -> H is a morphism (either a plain
ContinuousMorphism or a MonadicProgram) whose domain is a
product of the per-timestep input space A and the hidden state
space H, and whose codomain is H.

Given a cell : A * H -> H, ``scan(cell)`` produces a morphism
A -> H that, at runtime:

1. Expects a 3D input tensor of shape (batch, seq_len, dim_A).
2. Initializes hidden state h_0 (zeros or a learned parameter).
3. At each step t, concatenates x[:, t, :] with h to form the
   cell input, then calls cell.rsample to produce the new h.
4. Returns the final hidden state h_T of shape (batch, dim_H).

The scan's type in the categorical framework is:

    scan(f : A x H -> H) : A -> H

where the sequence structure is implicit in the tensor's time
dimension, following standard neural network conventions.

Initialization strategies
-------------------------
- ``"zeros"``: h_0 = 0 (default).
- ``"learned"``: h_0 is a learnable nn.Parameter.

Examples
--------
>>> from quivers.continuous.spaces import Euclidean, ProductSpace
>>> from quivers.continuous.families import ConditionalNormal
>>> A = Euclidean(name="input", dim=32)
>>> H = Euclidean(name="hidden", dim=64)
>>> cell = ConditionalNormal(ProductSpace(A, H), H, scale=0.1)
>>> scanned = ScanMorphism(cell, init="zeros")
>>> scanned.domain   # Euclidean(name="input", dim=32)
>>> scanned.codomain # Euclidean(name="hidden", dim=64)
>>> x = torch.randn(8, 10, 32)  # batch=8, seq_len=10, input_dim=32
>>> h = scanned.rsample(x)      # (8, 64)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
    _event_dim,
    dimension_probe,
)
from quivers.continuous.spaces import ProductSpace


def _extract_input_space(cell: ContinuousMorphism) -> AnySpace:
    """Extract the per-timestep input space from a cell's product domain.

    The cell must have a product domain A * H where H matches the
    codomain. Returns A.

    Parameters
    ----------
    cell : ContinuousMorphism
        A recurrent cell with product domain.

    Returns
    -------
    AnySpace
        The input component of the product domain. A product domain
        may pair a discrete per-step input with the hidden state, so
        the component reported here is not always continuous.

    Raises
    ------
    TypeError
        If the domain is not a ProductSpace or the last component
        does not match the codomain.
    """
    domain = cell.domain
    codomain = cell.codomain
    if not isinstance(domain, ProductSpace):
        raise TypeError(
            f"scan cell must have a ProductSpace domain, got {type(domain).__name__}: {domain!r}"
        )
    components = domain.components
    if len(components) < 2:
        raise TypeError(
            f"scan cell product domain must have at least 2 components, got {len(components)}"
        )
    hidden_component = components[-1]
    cod_dim = _event_dim(codomain)
    hid_dim = _event_dim(hidden_component)
    if hid_dim != cod_dim:
        raise TypeError(
            f"scan cell: last domain component dim ({hid_dim}) does not match codomain dim ({cod_dim}); the cell must have type A * H -> H"
        )
    if len(components) == 2:
        return components[0]
    else:
        result = components[0]
        for c in components[1:-1]:
            result = ProductSpace(components=(result, c))
        return result


class ScanMorphism(ContinuousMorphism):
    """Temporal scan: apply a recurrent cell across a sequence.

    Wraps a cell morphism ``f : A * H -> H`` and produces a morphism
    ``A -> H`` that iterates over the time dimension of a 3D input
    tensor, threading hidden state forward.

    This implements standard RNN-style recurrence::

        h_0 = init
        h_t = cell(concat(x_t, h_{t-1}))  for t = 1..T

    The scan returns the final hidden state h_T.

    Parameters
    ----------
    cell : ContinuousMorphism
        The recurrent cell. Must have a product domain ``A * H``
        and codomain ``H``, where ``H`` matches the last component
        of the product domain.
    init : str
        Initialization strategy for h_0. One of ``"zeros"``
        (default) or ``"learned"`` (trainable initial state).
    """

    def __init__(self, cell: ContinuousMorphism, init: str = "zeros") -> None:
        input_space = _extract_input_space(cell)
        hidden_space = cell.codomain
        super().__init__(input_space, hidden_space)
        self._cell = cell
        self._init_strategy = init
        self._input_dim = _event_dim(input_space)
        self._hidden_dim = _event_dim(hidden_space)
        if init == "learned":
            self._h0 = nn.Parameter(torch.zeros(self._hidden_dim))
        elif init != "zeros":
            raise ValueError(
                f"unknown init strategy {init!r}; expected 'zeros' or 'learned'"
            )

    @property
    def cell(self) -> ContinuousMorphism:
        """The recurrent cell the scan applies at every position.

        Returns
        -------
        ContinuousMorphism
            The cell ``A * H -> H``.
        """
        return self._cell

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Run the cell across the time dimension of x.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout
            `_as_sequence`
            reads: ``(batch, seq_len, input_dim)`` or the folded
            ``(batch, seq_len * input_dim)``.
        sample_shape : torch.Size
            Additional leading sample dimensions (applied to the
            cell's rsample at the first time step only).

        Returns
        -------
        torch.Tensor
            Final hidden state. Shape ``(batch, hidden_dim)``,
            or ``(*sample_shape, batch, hidden_dim)`` if
            sample_shape is non-empty.
        """
        x = self._as_sequence(x)
        batch, seq_len, _ = x.shape
        h = self._initial_state(batch, x)
        for t in range(seq_len):
            x_t = x[:, t, :]
            cell_input = torch.cat([x_t, h], dim=-1)
            if t == 0 and len(sample_shape) > 0:
                h = self._cell.rsample(cell_input, sample_shape)
                h = self._flatten_cell_output(h)
                if len(sample_shape) > 0 and h.dim() > 2:
                    x = x.unsqueeze(0).expand(*sample_shape, *x.shape)
            else:
                if h.dim() > 2:
                    x_t = x[..., t, :]
                    cell_input = torch.cat([x_t, h], dim=-1)
                h = self._cell.rsample(cell_input)
                h = self._flatten_cell_output(h)
        return h

    def _initial_state(self, batch: int, x: torch.Tensor) -> torch.Tensor:
        """The hidden state the recurrence starts from, shaped ``(batch, H)``."""
        if self._init_strategy == "learned":
            return self._h0.unsqueeze(0).expand(batch, -1)
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        return torch.zeros(batch, self._hidden_dim, device=x.device, dtype=dtype)

    def _step_dimension(self, seq: torch.Tensor) -> int | None:
        """Base coordinates one application of the cell consumes.

        ``seq`` is already in the ``(batch, seq_len, input_dim)``
        layout `_as_sequence` returns, so the width the cell reads is
        the one the recurrence actually feeds it.
        """
        probe = dimension_probe(seq)
        h = self._initial_state(probe.shape[0], probe)
        cell_input = torch.cat([probe[:, 0, :], h], dim=-1)
        return self._cell.base_dimension(cell_input)

    def base_dimension(self, x: torch.Tensor) -> int | None:
        """One cell's worth of coordinates per time step of the input.

        The recurrence draws once per position, so its coordinate
        budget is the sequence length times the cell's own. The length
        is read off the input rather than declared, because
        ``scan(cell) : A -> H`` says nothing about how many positions a
        given input carries, and it is read through
        `_as_sequence`
        so a folded ``(batch, seq_len * input_dim)`` input reports the
        whole sequence's budget rather than one step's. Under-reporting
        it would hand
        [`push_base`][quivers.continuous.scan.ScanMorphism.push_base]
        a block too short to run the recurrence on, and a chain that
        sliced its coordinates by that count would give every later
        factor the wrong ones.
        """
        seq = self._as_sequence(x)
        step = self._step_dimension(seq)
        if step is None:
            return None
        return int(seq.shape[1]) * step

    def push_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Run the recurrence on supplied coordinates instead of draws.

        Time step ``t`` reads the ``t``-th block of ``base``, so the
        trajectory is a deterministic function of the coordinates and
        the input, and the same coordinates always produce the same
        final state.
        """
        seq = self._as_sequence(x)
        step = self._step_dimension(seq)
        if step is None:
            raise ValueError(
                f"ScanMorphism.push_base: the cell "
                f"{type(self._cell).__name__} declares no "
                f"reparameterization, so the recurrence has none either."
            )
        batch, seq_len, _ = seq.shape
        h = self._initial_state(batch, seq)
        for t in range(seq_len):
            cell_input = torch.cat([seq[:, t, :], h], dim=-1)
            block = base[:, t * step : (t + 1) * step]
            h = self._flatten_cell_output(self._cell.push_base(cell_input, block))
        return h

    @staticmethod
    def _flatten_cell_output(
        result: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Flatten a cell output to a single tensor.

        Monadic programs with tuple returns produce dicts. For scan,
        we concatenate the dict values along the feature dimension
        to reconstruct the full hidden state vector.

        Parameters
        ----------
        result : torch.Tensor or dict
            Cell output (tensor or dict from tuple-returning program).

        Returns
        -------
        torch.Tensor
            Flattened hidden state.
        """
        if isinstance(result, dict):
            return torch.cat(list(result.values()), dim=-1)
        return result

    def _as_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Read ``x`` as ``(batch, seq_len, input_dim)``.

        The recurrence needs a position axis, and the tensor reaching
        it carries that axis in one of two layouts. A caller applying
        the scan directly passes ``(batch, seq_len, input_dim)``, or
        ``(batch, input_dim)`` for a single position. A caller scoring
        the scan as the last factor of a
        [`SampledComposition`][quivers.continuous.morphisms.SampledComposition]
        passes the chain's canonical intermediate, which that class
        folds to one feature axis per row before handing it on, so the
        positions arrive multiplied into the last axis as
        ``(batch, seq_len * input_dim)``. Unfolding by the declared
        per-step width recovers the axis in both cases, and the width
        is what makes the recovery unambiguous rather than a guess.
        """
        if x.dim() >= 3:
            if int(x.shape[-1]) != self._input_dim:
                raise ValueError(
                    f"ScanMorphism: the per-step input axis of "
                    f"{tuple(x.shape)} is {int(x.shape[-1])}, but the "
                    f"cell reads {self._input_dim} coordinates per "
                    f"step."
                )
            return x.reshape(x.shape[0], -1, self._input_dim)
        if x.dim() != 2:
            raise ValueError(
                f"ScanMorphism: expected a sequence input of rank 2 or "
                f"3, got shape {tuple(x.shape)}."
            )
        width = int(x.shape[-1])
        if width % self._input_dim != 0:
            raise ValueError(
                f"ScanMorphism: the input's feature axis of width "
                f"{width} is not a whole number of {self._input_dim}"
                f"-coordinate steps, so it carries no position axis "
                f"the recurrence could read."
            )
        return x.reshape(x.shape[0], width // self._input_dim, self._input_dim)

    def reference_trajectory(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The states :math:`h_1, \\ldots, h_T` this kernel scores at ``y``.

        The recurrence's own deterministic skeleton, re-anchored at the
        observed final state: for :math:`t < T` the state is the cell's
        image of the base measure's origin, and :math:`h_T` is ``y``.

        Prefix states use ``push_base`` at zero coordinates; the final
        state is ``y``. The resulting trajectory is deterministic.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout
            `_as_sequence`
            reads.
        y : torch.Tensor
            Observed final state. Shape ``(batch, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, hidden_dim)``.

        Raises
        ------
        ValueError
            If the cell declares no reparameterization, so the prefix
            has no canonical states to take.
        """
        seq = self._as_sequence(x)
        batch, seq_len, _ = seq.shape
        h = self._initial_state(batch, seq)
        states: list[torch.Tensor] = []
        for t in range(seq_len - 1):
            cell_input = torch.cat([seq[:, t, :], h], dim=-1)
            dimension = self._cell.base_dimension(cell_input)
            if dimension is None:
                raise ValueError(
                    f"ScanMorphism.reference_trajectory: the cell "
                    f"{type(self._cell).__name__} declares no "
                    f"reparameterization, so its base measure has no "
                    f"origin and the recurrence has no canonical "
                    f"prefix to score along."
                )
            base = torch.zeros(batch, dimension, device=seq.device, dtype=h.dtype)
            h = self._flatten_cell_output(self._cell.push_base(cell_input, base))
            states.append(h)
        states.append(y.reshape(batch, self._hidden_dim))
        return torch.stack(states, dim=1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of the scan's trajectory ending at ``y``.

        ``scan(cell)`` denotes a Kleisli morphism
        :math:`\\mathbf{x}_{1:T} \\to \\mathcal{G}(h_T)` whose density at
        :math:`h_T` marginalizes every intermediate state:

        .. math::

            p(h_T \\mid x_{1:T}) = \\int
            p(h_T \\mid x_T, h_{T-1})
            \\prod_{t<T} p(h_t \\mid x_t, h_{t-1})
            \\, dh_{1:T-1}.

        The implementation scores the fixed trajectory returned by
        ``reference_trajectory`` with ``log_joint``:

        .. math::

            \\sum_{t=1}^{T} \\log p(h_t \\mid x_t, h_{t-1}),
            \\qquad h_T = y .

        Floating-point reassociation in the recurrent prefix may be
        amplified by later transitions. Numerical comparisons should
        use the same cell implementation.

        This is the joint density of the fixed trajectory, not the
        marginal density of its endpoint.

        If the cell has no conditional density, this method returns a
        zero contribution.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout
            `_as_sequence`
            reads.
        y : torch.Tensor
            Final hidden state. Shape ``(batch, hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``.
        """
        seq = self._as_sequence(x)
        if not self._cell.has_conditional_density():
            dtype = seq.dtype if seq.is_floating_point() else torch.get_default_dtype()
            return torch.zeros(seq.shape[0], device=seq.device, dtype=dtype)
        return self.log_joint(seq, self.reference_trajectory(seq, y))

    def log_joint(
        self,
        x: torch.Tensor,
        hidden_states: "torch.Tensor | dict[str, torch.Tensor]",
        *,
        state_key: str = "h",
    ) -> torch.Tensor:
        """Joint log-density given all intermediate hidden states.

        Computes:
            log p(h_1, ..., h_T | x_{1:T}) =
                sum_t log p(h_t | x_t, h_{t-1})

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout
            `_as_sequence`
            reads.
        hidden_states : torch.Tensor | dict[str, torch.Tensor]
            All hidden states including final, shape
            ``(batch, seq_len, hidden_dim)``. May be passed
            positionally as a tensor or via a dict keyed by
            ``state_key`` (so the inference layer's standard
            ``log_joint(x, observations: dict)`` contract works
            without an adapter).
        state_key : str
            Dict key under which the hidden-state tensor is
            looked up when ``hidden_states`` is a dict. Defaults
            to ``"h"``.

        Returns
        -------
        torch.Tensor
            Joint log-density. Shape ``(batch,)``.
        """
        if isinstance(hidden_states, dict):
            hidden_states = hidden_states[state_key]
        seq = self._as_sequence(x)
        batch, seq_len, _ = seq.shape
        states = hidden_states.reshape(batch, seq_len, self._hidden_dim)
        h = self._initial_state(batch, seq)
        total = torch.zeros(batch, device=seq.device, dtype=h.dtype)
        for t in range(seq_len):
            h_t = states[:, t, :]
            cell_input = torch.cat([seq[:, t, :], h], dim=-1)
            total = total + self._cell.log_prob(cell_input, h_t)
            h = h_t
        return total

    def __repr__(self) -> str:
        init = f", init={self._init_strategy}" if self._init_strategy != "zeros" else ""
        return f"ScanMorphism({self._cell!r}{init})"


__all__ = [
    "ScanMorphism",
]
