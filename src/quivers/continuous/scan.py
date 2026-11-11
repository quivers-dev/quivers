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
        Initialization strategy for h_0: ``"zeros"`` (default),
        ``"learned"`` (trainable initial state), or the name of a
        binding of an enclosing program (``scan(cell, init=h0)``). A
        scan that starts from a binding runs as a step of that program,
        which supplies the binding; run on its own it has no initial
        state and raises.
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
        elif not init.isidentifier():
            raise ValueError(
                f"unknown init strategy {init!r}; expected 'zeros', 'learned', "
                f"or the name of a binding"
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
        """The hidden state the recurrence starts from, shaped ``(batch, H)``.

        Raises
        ------
        ValueError
            If the scan starts from a binding of an enclosing program,
            which a scan run on its own does not have.
        """
        if self._init_strategy == "learned":
            return self._h0.unsqueeze(0).expand(batch, -1)
        if self._init_strategy != "zeros":
            raise ValueError(
                f"scan(..., init={self._init_strategy}) starts from the binding "
                f"{self._init_strategy!r} of the program it runs in; run it as a "
                f"step of a program that binds {self._init_strategy!r} before the "
                f"scan"
            )
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

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout `_as_sequence` reads.
        base : torch.Tensor
            Standard-normal coordinates, one cell's block per position.

        Returns
        -------
        torch.Tensor
            The final state.
        """
        return self.trajectory_from_base(x, base)[:, -1, :]

    def push_base_scored(
        self, x: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrence on supplied coordinates and score its trajectory.

        The coordinates select every state of the trajectory, not only
        the final one, so the draw's density is the trajectory's joint
        density, as a scan inside a program scores each position's
        site.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout `_as_sequence` reads.
        base : torch.Tensor
            Standard-normal coordinates, one cell's block per position.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The final state, shape ``(batch, hidden_dim)``, and the
            trajectory's log-density, shape ``(batch,)``.

        Raises
        ------
        ValueError
            If the cell declares no reparameterization.
        """
        trajectory = self.trajectory_from_base(x, base)
        return trajectory[:, -1, :], self.log_prob(x, trajectory)

    def trajectory_from_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Every state the recurrence reaches on supplied coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout `_as_sequence` reads.
        base : torch.Tensor
            Standard-normal coordinates, shape ``(batch, dimension)``
            for the dimension `base_dimension` reports at ``x``; time
            step ``t`` reads the ``t``-th block.

        Returns
        -------
        torch.Tensor
            The states :math:`h_1, \\ldots, h_T`, shape
            ``(batch, seq_len, hidden_dim)``.

        Raises
        ------
        ValueError
            If the cell declares no reparameterization.
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
        states: list[torch.Tensor] = []
        for t in range(seq_len):
            cell_input = torch.cat([seq[:, t, :], h], dim=-1)
            block = base[:, t * step : (t + 1) * step]
            h = self._flatten_cell_output(self._cell.push_base(cell_input, block))
            states.append(h)
        return torch.stack(states, dim=1)

    def rsample_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """Draw the recurrence's whole trajectory.

        The trajectory is the value
        [`log_prob`][quivers.continuous.scan.ScanMorphism.log_prob]
        scores; its last state is what
        [`rsample`][quivers.continuous.scan.ScanMorphism.rsample]
        returns.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout `_as_sequence` reads.

        Returns
        -------
        torch.Tensor
            The states :math:`h_1, \\ldots, h_T`, shape
            ``(batch, seq_len, hidden_dim)``.
        """
        seq = self._as_sequence(x)
        batch, seq_len, _ = seq.shape
        h = self._initial_state(batch, seq)
        states: list[torch.Tensor] = []
        for t in range(seq_len):
            cell_input = torch.cat([seq[:, t, :], h], dim=-1)
            h = self._flatten_cell_output(self._cell.rsample(cell_input))
            states.append(h)
        return torch.stack(states, dim=1)

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

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-density of a supplied trajectory.

        ``scan(cell)`` draws one state per position, and every one of
        them is a random variable, as a scan inside a program reaches
        one site per position. The value this method scores is thus the
        whole trajectory :math:`h_{1:T}`:

        .. math::

            \\sum_{t=1}^{T} \\log p(h_t \\mid x_t, h_{t-1}).

        The final state alone has a density only as a marginal over the
        states before it, which the scan does not compute, so a final
        state is refused rather than scored along a trajectory it does
        not determine. A one-position sequence's trajectory is its
        final state.

        Floating-point reassociation in the recurrence may be amplified
        by later transitions, roughly an order of magnitude per step, so
        numerical comparisons should use the same cell implementation.

        A cell with no conditional density contributes zero.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, in either layout
            `_as_sequence`
            reads.
        y : torch.Tensor
            The trajectory, shape ``(batch, seq_len, hidden_dim)`` or
            folded to ``(batch, seq_len * hidden_dim)``, as
            [`rsample_trajectory`][quivers.continuous.scan.ScanMorphism.rsample_trajectory]
            draws it.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``.

        Raises
        ------
        ValueError
            If ``y`` does not hold one state per position of ``x``.
        """
        seq = self._as_sequence(x)
        batch, seq_len, _ = seq.shape
        if y.numel() != batch * seq_len * self._hidden_dim:
            raise ValueError(
                f"ScanMorphism.log_prob: scores the trajectory of "
                f"{seq_len} states of width {self._hidden_dim} per row, "
                f"shape ({batch}, {seq_len}, {self._hidden_dim}), but got "
                f"shape {tuple(y.shape)}; a final state alone has no "
                f"density the scan computes. Pass the trajectory, as "
                f"rsample_trajectory draws it."
            )
        if not self._cell.has_conditional_density():
            dtype = seq.dtype if seq.is_floating_point() else torch.get_default_dtype()
            return torch.zeros(batch, device=seq.device, dtype=dtype)
        return self.log_joint(seq, y)

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
