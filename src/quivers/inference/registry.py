"""Latent-site registry: the canonical introspection surface that
every variational guide and MCMC kernel consumes.

A `LatentRegistry` is built once from a
[`quivers.continuous.programs.MonadicProgram`][quivers.continuous.programs.MonadicProgram] and a set of
observed site names. It walks the model's ``steps`` linear IR
exactly once, classifies each latent into "plate" vs "scalar",
records the prior support and per-site dimensionality, builds the
matching `torch.distributions.transforms.Transform` via
`torch.distributions.constraint_registry.biject_to`, and
caches a *fused* flatten / unflatten pipeline so a single
unconstrained vector can be turned into a fully constrained site
dict (and back) in one pass.

The registry is the only place in the inference layer that knows
about [`PlateDraw`][quivers.continuous.PlateDraw], the layout
of ``MonadicProgram.steps``, or the per-site
constraint-to-bijector mapping. Every downstream component
([`Guide`][quivers.inference.guides.Guide],
[`MCMCKernel`][quivers.inference.mcmc.MCMCKernel], the
flow-based / structured / mixture guides) receives a fully
populated registry and operates against its flat-vector and
per-site dict APIs.

Categorical interpretation
==========================

Each latent site corresponds to a draw in the model's Kleisli
arrow ``Γ → 𝒢(τ)``. The site's prior is a parameterized family
``F : Θ → 𝒢(B)`` with a support constraint
``C ⊆ B``. The bijector ``T : ℝ^d → C`` is the
`biject_to` lift of ``C``; its log-determinant gives the
change-of-variables term in any pushforward density.

For a plate ``v : A → B`` the latent lives in ``C^|A|``; the
registry handles the leading plate axis (size ``|A|``) separately
from the trailing event axis (the support dimension), and the
unconstrained flat vector concatenates plate × event across every
site in declaration order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch
from torch.distributions import constraints as _constraints
from torch.distributions.constraint_registry import biject_to
from torch.distributions.transforms import Transform

from quivers.effects.subsample import refuse_subsampling
from quivers.continuous.plate import PlateDraw
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.program_steps import Call, Draw, Observe, Scan
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import ContinuousSpace
from quivers.effects.trace_types import Trace


def _unconstrained_event_dim(
    support: _constraints.Constraint, constrained_dim: int
) -> int:
    """Return the unconstrained-space event dimension matching a
    ``constrained_dim``-sized support.

    The two cases that change dim:

    * `torch.distributions.constraints.simplex`: the
      stick-breaking bijector maps :math:`(d-1)`-dimensional
      unconstrained space onto the :math:`(d-1)`-simplex embedded
      in :math:`\\mathbb{R}^d`.
    * `torch.distributions.constraints.corr_cholesky`: the
      correlation-Cholesky bijector maps :math:`d(d-1)/2` real
      coordinates onto the lower-triangular Cholesky factor of a
      :math:`d \\times d` correlation matrix.

    Every other supported constraint is dim-preserving (the
    bijector is `torch.distributions.transforms.AffineTransform`,
    `SigmoidTransform`, `ExpTransform`, or a composition
    thereof).
    """
    if support is _constraints.simplex:
        return max(1, constrained_dim - 1)
    if support is _constraints.corr_cholesky:
        return max(1, constrained_dim * (constrained_dim - 1) // 2)
    return constrained_dim


def _apply_transform(transform: Transform, value: torch.Tensor) -> torch.Tensor:
    """Apply a `torch.distributions` transform and return its tensor.

    Parameters
    ----------
    transform : Transform
        The bijector (or its inverse) to apply.
    value : torch.Tensor
        The input point.

    Returns
    -------
    torch.Tensor
        ``transform(value)``.

    Raises
    ------
    TypeError
        If the transform returns something other than a tensor.
    """
    out = transform(value)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"LatentRegistry: transform {transform!r} returned "
            f"{type(out).__name__}, not a tensor"
        )
    return out


def _require_one_run(x: torch.Tensor, consumer: str) -> None:
    """Refuse an input of more than one row for a one-run posterior.

    An input of ``B`` rows is ``B`` independent runs of the whole
    program, each with its own latents. MCMC, the Laplace approximation,
    and annealed importance sampling target the posterior of one run's
    latents, so they take a one-row input; the rows a program observes
    come from its data and its plates, not from the input's rows. They
    target the exact posterior, so they also refuse to run inside a
    [`subsample`][quivers.effects.subsample] block, whose log joint is
    only an unbiased estimate.

    Parameters
    ----------
    x : torch.Tensor
        The program input.
    consumer : str
        The name the error reports.

    Raises
    ------
    ValueError
        If ``x`` has more than one row, or a plate is subsampled.
    """
    refuse_subsampling(consumer)
    rows = int(x.shape[0]) if x.dim() else 1
    if rows != 1:
        raise ValueError(
            f"{consumer}: the input has {rows} rows, and an input of {rows} rows "
            f"is {rows} independent runs of the program, each with its own "
            f"latents; {consumer} targets the posterior of one run. Pass one "
            f"input row (for instance torch.zeros(1, ...)): the rows a program "
            f"observes come from its data and its plates, not from the input"
        )


@dataclass(frozen=True)
class LatentSite:
    """Frozen record describing a single latent site.

    Attributes
    ----------
    name : str
        Variable name (the bound name in the program body).
    morphism : ContinuousMorphism
        The site's runtime morphism (the prior). Held by reference
        so guides / kernels can call ``morphism.log_prob`` directly
        when they need the per-site prior contribution.
    support : torch.distributions.constraints.Constraint
        Support of the prior. Drives the choice of bijector.
    bijector : torch.distributions.transforms.Transform
        ``biject_to(support)``, cached at registry construction.
    constrained_dim : int
        Event dimension on the constrained side (for a scalar
        site, ``1``; for a Dirichlet on a 5-simplex, ``5``).
    unconstrained_dim : int
        Event dimension on the unconstrained side (for a Dirichlet
        on a 5-simplex, ``4``; otherwise equals ``constrained_dim``).
    is_plate : bool
        Whether this site is a plate draw
        ([`PlateDraw`][quivers.continuous.PlateDraw]).
    plate_index_size : int
        ``|A|`` for a plate site, ``0`` otherwise.
    spec_index : int
        Position in ``model.steps``.
    flat_offset : int
        Index of this site's first element in the registry's
        flat unconstrained vector.
    flat_length : int
        Number of elements this site occupies in the flat vector
        (= ``plate_index_size * unconstrained_dim`` for plate
        sites, ``unconstrained_dim`` for scalar sites).
    plate_shape : tuple[int, ...]
        The extents of a plate site's axes, outermost first, whose
        product is ``plate_index_size``: ``(|A|,)`` for a plate over one
        axis, ``(|A|, |B|, |C|)`` for an independent array over
        several; ``()`` otherwise.
    """

    name: str
    morphism: ContinuousMorphism
    support: _constraints.Constraint
    bijector: Transform
    constrained_dim: int
    unconstrained_dim: int
    is_plate: bool
    plate_index_size: int
    spec_index: int
    flat_offset: int
    flat_length: int
    plate_shape: tuple[int, ...] = ()

    @property
    def constrained_shape(self) -> tuple[int, ...]:
        """The shape of one draw of this site on its constrained
        support (after the bijector, with a length-one event axis
        squeezed off).

        Plate sites have shape ``(*plate_shape,)`` or
        ``(*plate_shape, d)``, ``(|A|,)`` or ``(|A|, d)`` for a plate
        over one axis; non-plate
        sites have shape ``()`` for a scalar event and ``(d,)``
        otherwise. Posterior containers index draws by this shape:
        `MCMCResult.samples` holds ``(num_chains, num_samples,
        *constrained_shape)`` and `MCMCResult.r_hat` and
        `MCMCResult.ess` hold ``constrained_shape``.

        The values a program's `MonadicProgram.log_joint`, a trace,
        and a guide exchange carry the run axis of the program's input,
        whose ``B`` rows are ``B`` independent runs: a non-plate site
        is ``(B, *constrained_shape)`` (``(1, *constrained_shape)`` in
        one run), and a plate site is ``constrained_shape`` in one run,
        or one value every run shares, and ``(B, *constrained_shape)``
        given or recorded per run. `LatentRegistry.trace_values` lifts
        one draw to the values runs share.

        A correlation-Cholesky site's draw is a lower-triangular
        ``(d, d)`` factor, so its event shape is ``(d, d)`` rather
        than ``(d,)``.
        """
        event: tuple[int, ...]
        if self.support is _constraints.corr_cholesky:
            event = (self.constrained_dim, self.constrained_dim)
        elif self.constrained_dim == 1:
            event = ()
        else:
            event = (self.constrained_dim,)
        if self.is_plate:
            return (*(self.plate_shape or (self.plate_index_size,)), *event)
        return event

    @property
    def unconstrained_shape(self) -> tuple[int, ...]:
        """The variational-side shape of this site's value in
        unconstrained space (before the bijector)."""
        if self.is_plate:
            return (self.plate_index_size, self.unconstrained_dim)
        return (self.unconstrained_dim,)


def _latent_site(
    name: str,
    morphism: ContinuousMorphism,
    names: int,
    spec_index: int,
    flat_offset: int,
) -> LatentSite:
    """Describe one latent site from the morphism it draws from.

    Parameters
    ----------
    name : str
        The site's name.
    morphism : ContinuousMorphism
        The site's prior.
    names : int
        How many variables the draw binds; a destructuring draw splits
        its codomain evenly among them.
    spec_index : int
        The position, in ``model.steps``, of the step that reaches the
        site.
    flat_offset : int
        The site's first index in the flat unconstrained vector.

    Returns
    -------
    LatentSite
        The site.
    """
    is_plate = isinstance(morphism, PlateDraw)
    if is_plate:
        plate = cast(PlateDraw, morphism)
        inner = plate.family
        inner_cod = inner.codomain
        if isinstance(inner_cod, ContinuousSpace):
            constrained_dim = max(1, math.prod(inner_cod.event_shape))
        else:
            constrained_dim = 1
        support = inner.support
        plate_index_size = plate.index_size
        plate_shape = plate.index_shape
    else:
        cod = morphism.codomain
        if isinstance(cod, ContinuousSpace):
            total_dim = math.prod(cod.event_shape)
            constrained_dim = max(1, total_dim // max(1, names))
        else:
            constrained_dim = 1
        support = morphism.support
        plate_index_size = 0
        plate_shape = ()
    unconstrained_dim = _unconstrained_event_dim(support, constrained_dim)
    flat_length = (
        plate_index_size * unconstrained_dim if is_plate else unconstrained_dim
    )
    return LatentSite(
        name=name,
        morphism=morphism,
        support=support,
        bijector=biject_to(support),
        constrained_dim=constrained_dim,
        unconstrained_dim=unconstrained_dim,
        is_plate=is_plate,
        plate_index_size=plate_index_size,
        spec_index=spec_index,
        flat_offset=flat_offset,
        flat_length=flat_length,
        plate_shape=plate_shape,
    )


def _drawing_programs(program: MonadicProgram) -> bool:
    """Whether a program draws, directly or in a program it calls or scans.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    bool
        ``True`` when any of its steps, or its callees' or scanned
        cells' steps, is a draw or an observation.
    """
    for spec in program.steps:
        if isinstance(spec, Draw | Observe):
            return True
        if isinstance(spec, Call | Scan):
            inner = spec.program if isinstance(spec, Call) else spec.step
            if isinstance(inner, MonadicProgram) and _drawing_programs(inner):
                return True
    return False


def _input_dependent_sites(program: MonadicProgram) -> bool:
    """Whether a program reaches sites that its steps alone do not list.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    bool
        ``True`` when it calls or scans a program that draws.
    """
    for spec in program.steps:
        if isinstance(spec, Call | Scan):
            inner = spec.program if isinstance(spec, Call) else spec.step
            if isinstance(inner, MonadicProgram) and _drawing_programs(inner):
                return True
    return False


def _reaching_steps(program: MonadicProgram) -> dict[str, int]:
    """The step of a program through which each site label is reached.

    Parameters
    ----------
    program : MonadicProgram
        The program.

    Returns
    -------
    dict[str, int]
        Each variable a draw binds, directly or in a program a step
        calls or scans, mapped to that step's position in
        ``program.steps``.
    """

    def drawn(inner: MonadicProgram) -> list[str]:
        """Every variable a program's draws bind, its callees' included.

        Parameters
        ----------
        inner : MonadicProgram
            The program.

        Returns
        -------
        list[str]
            The names, in step order.
        """
        names: list[str] = []
        for spec in inner.steps:
            if isinstance(spec, Draw | Observe):
                names.extend(spec.names)
            elif isinstance(spec, Call | Scan):
                callee = spec.program if isinstance(spec, Call) else spec.step
                if isinstance(callee, MonadicProgram):
                    names.extend(drawn(callee))
        return names

    positions: dict[str, int] = {}
    for index, spec in enumerate(program.steps):
        if isinstance(spec, Draw | Observe):
            for name in spec.names:
                positions.setdefault(name, index)
        elif isinstance(spec, Call | Scan):
            callee = spec.program if isinstance(spec, Call) else spec.step
            if isinstance(callee, MonadicProgram):
                for name in drawn(callee):
                    positions.setdefault(name, index)
    return positions


class LatentRegistry:
    """The complete introspection result for one (model,
    observed-set) pair.

    Constructed once via `from_model`; immutable thereafter.
    Provides:

    * Iteration over latent sites in declaration order
      (`sites`, `plate_sites`, `scalar_sites`).
    * Total unconstrained dimensionality
      (`total_unconstrained_dim`), the dim of the flat
      vector HMC and full-rank Gaussian guides operate on.
    * Round-tripping between a flat unconstrained vector and a
      dict of constrained per-site tensors
      (`unflatten_unconstrained`, `flatten_constrained`).
    * Bijector forward / inverse with Jacobian accumulation
      (`to_constrained`, `to_unconstrained`).
    * Conversion between one draw per site, at
      `LatentSite.constrained_shape`, and the trace-side values a
      program's ``log_joint`` and the guides exchange, which add the
      ``(batch, ...)`` axis to non-plate sites (`trace_values`,
      `trace_row`).

    All operations are vectorized, with no per-site Python loops on
    the hot path beyond the construction-time setup.
    """

    def __init__(
        self,
        sites: dict[str, LatentSite],
        observed_names: frozenset[str],
        model: MonadicProgram,
    ) -> None:
        self._sites: dict[str, LatentSite] = sites
        self._observed_names: frozenset[str] = observed_names
        self._model: MonadicProgram = model
        self._total_unconstrained_dim: int = sum(s.flat_length for s in sites.values())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: MonadicProgram,
        observed_names: set[str] | frozenset[str],
    ) -> "LatentRegistry":
        """Walk ``model.steps`` and assemble the registry.

        Parameters
        ----------
        model : MonadicProgram
            Compiled probabilistic program.
        observed_names : set[str] | frozenset[str]
            Variable names treated as observations. Sample sites
            with names in this set are skipped (their values come
            from the conditioning data dict at trace time, not
            from a guide).

        Returns
        -------
        LatentRegistry
            Fully populated registry with each latent's morphism,
            support, bijector, dims, and flat-vector offsets.
        """
        observed: frozenset[str] = frozenset(observed_names)
        if _input_dependent_sites(model):
            raise ValueError(
                "LatentRegistry.from_model: the program scans or calls a program "
                "that draws, so how many latent sites it has depends on its "
                "input; build the registry from a run with "
                "LatentRegistry.from_trace(model, trace(model, x, observations))"
            )
        sites: dict[str, LatentSite] = {}
        flat_cursor = 0

        for spec_idx, spec in enumerate(model.steps):
            if not isinstance(spec, Draw | Observe):
                continue
            for var in spec.names:
                if var in observed:
                    continue
                morph = model.step_module(spec)
                if morph is None:
                    raise RuntimeError(
                        f"LatentRegistry: morphism module for site "
                        f"{var!r} is unregistered on the model"
                    )
                site = _latent_site(
                    var,
                    cast(ContinuousMorphism, morph),
                    len(spec.names),
                    spec_idx,
                    flat_cursor,
                )
                sites[var] = site
                flat_cursor += site.flat_length

        return cls(sites, observed, model)

    @classmethod
    def from_trace(cls, model: MonadicProgram, recorded: Trace) -> "LatentRegistry":
        """Assemble the registry from a run of the model.

        Every sample site the run reached that is neither observed nor
        intervened on is a latent, in the order the run reached it. A
        site a scan reaches once per position is one latent per
        position, ``"<label>@<n>"``, so a scanned program cell's latent
        trajectory is inferred site by site, as many as the input's
        sequence has positions. Build ``recorded`` with
        ``trace(model, x, observations)`` at the input and data the
        latents are inferred at.

        Parameters
        ----------
        model : MonadicProgram
            Compiled probabilistic program.
        recorded : Trace
            A trace of one run of ``model``.

        Returns
        -------
        LatentRegistry
            Fully populated registry with each latent's morphism,
            support, bijector, dims, and flat-vector offsets.
        """
        reached = _reaching_steps(model)
        sites: dict[str, LatentSite] = {}
        observed: set[str] = set()
        flat_cursor = 0
        for label, record in recorded.sites.items():
            if record.morphism is None or record.is_deterministic:
                continue
            if record.is_observed:
                observed.update(label.split(","))
                continue
            names = label.split(",")
            base = label.rpartition("@")[0] or label
            spec_index = reached.get(base.split(",")[0], -1)
            for var in names:
                site = _latent_site(
                    var if len(names) > 1 else label,
                    record.morphism,
                    len(names),
                    spec_index,
                    flat_cursor,
                )
                sites[site.name] = site
                flat_cursor += site.flat_length
        return cls(sites, frozenset(observed), model)

    # ------------------------------------------------------------------
    # Iteration / lookup
    # ------------------------------------------------------------------

    @property
    def sites(self) -> dict[str, LatentSite]:
        """Every latent site, keyed by name, in declaration order."""
        return self._sites

    @property
    def plate_sites(self) -> dict[str, LatentSite]:
        """The subset of sites that are `PlateDraw`-shaped."""
        return {n: s for n, s in self._sites.items() if s.is_plate}

    @property
    def scalar_sites(self) -> dict[str, LatentSite]:
        """The subset of sites that are not plates."""
        return {n: s for n, s in self._sites.items() if not s.is_plate}

    @property
    def names(self) -> tuple[str, ...]:
        """Latent site names in declaration order."""
        return tuple(self._sites.keys())

    @property
    def observed_names(self) -> frozenset[str]:
        """The observed-name set used to build this registry."""
        return self._observed_names

    @property
    def model(self) -> MonadicProgram:
        """The underlying model."""
        return self._model

    @property
    def total_unconstrained_dim(self) -> int:
        """Sum of every site's flat-vector length.

        This is the dim of the unconstrained vector that full-rank
        Gaussian guides and HMC operate on.
        """
        return self._total_unconstrained_dim

    def __len__(self) -> int:
        return len(self._sites)

    def __iter__(self):
        return iter(self._sites.values())

    def __contains__(self, name: str) -> bool:
        return name in self._sites

    def __getitem__(self, name: str) -> LatentSite:
        return self._sites[name]

    # ------------------------------------------------------------------
    # Flatten / unflatten between dicts and a single vector
    # ------------------------------------------------------------------

    def unflatten_unconstrained(self, vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reshape a flat unconstrained vector into per-site
        unconstrained tensors.

        Parameters
        ----------
        vec : torch.Tensor
            Shape ``(..., total_unconstrained_dim)``. Leading
            axes (batch, particles, chains) are preserved on every
            site's output.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{name: site_tensor}`` where each ``site_tensor`` has
            shape ``vec.shape[:-1] + site.unconstrained_shape``.
        """
        if vec.shape[-1] != self._total_unconstrained_dim:
            raise ValueError(
                f"LatentRegistry.unflatten_unconstrained: expected "
                f"trailing dim {self._total_unconstrained_dim}, got "
                f"{vec.shape[-1]}"
            )
        out: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            chunk = vec[..., site.flat_offset : site.flat_offset + site.flat_length]
            out[site.name] = chunk.reshape(*vec.shape[:-1], *site.unconstrained_shape)
        return out

    def flatten_unconstrained(self, sites: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate per-site unconstrained tensors into a single
        flat vector.

        Inverse of `unflatten_unconstrained` (up to leading
        batch axes). The trailing batch axes of every site tensor
        must agree.
        """
        parts: list[torch.Tensor] = []
        leading_shape: tuple[int, ...] | None = None
        for site in self._sites.values():
            if site.name not in sites:
                raise KeyError(
                    f"LatentRegistry.flatten_unconstrained: missing site {site.name!r}"
                )
            t = sites[site.name]
            trail = len(site.unconstrained_shape)
            site_leading = t.shape[: t.dim() - trail] if trail else t.shape
            if leading_shape is None:
                leading_shape = tuple(site_leading)
            elif tuple(site_leading) != leading_shape:
                raise ValueError(
                    f"LatentRegistry.flatten_unconstrained: site "
                    f"{site.name!r} has leading shape {tuple(site_leading)} "
                    f"which disagrees with previously-seen "
                    f"{leading_shape}"
                )
            parts.append(t.reshape(*site_leading, site.flat_length))
        if not parts:
            device = next(iter(sites.values())).device if sites else torch.device("cpu")
            return torch.zeros((0,), device=device)
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Bijector application with Jacobian accumulation
    # ------------------------------------------------------------------

    def to_constrained(
        self, unconstrained: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Push every site's unconstrained value through its
        bijector.

        Returns
        -------
        constrained : dict[str, torch.Tensor]
            Per-site constrained values shaped according to
            `LatentSite.constrained_shape`. The trailing
            length-1 event axis of a scalar site is squeezed off
            to match the trace-side shape convention.
        log_abs_dets : dict[str, torch.Tensor]
            Per-site ``log |det dT/dz|`` returned by the bijector,
            at the bijector's natural shape (event axes preserved).
            Callers aggregate as the objective requires: for ELBO
            we sum over event axes; for IWAE we keep particles as
            a leading axis; etc.
        """
        constrained: dict[str, torch.Tensor] = {}
        log_abs_dets: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            if site.name not in unconstrained:
                raise KeyError(
                    f"LatentRegistry.to_constrained: missing site {site.name!r}"
                )
            z = unconstrained[site.name]
            v = _apply_transform(site.bijector, z)
            log_abs_dets[site.name] = site.bijector.log_abs_det_jacobian(z, v)
            if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
                v = v.squeeze(-1)
            constrained[site.name] = v
        return constrained, log_abs_dets

    def to_unconstrained(
        self, constrained: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Inverse of `to_constrained`."""
        unconstrained: dict[str, torch.Tensor] = {}
        log_abs_dets: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            if site.name not in constrained:
                raise KeyError(
                    f"LatentRegistry.to_unconstrained: missing site {site.name!r}"
                )
            v = constrained[site.name]
            if site.constrained_dim == 1 and v.dim() == (1 if site.is_plate else 0):
                v = v.unsqueeze(-1)
            z = _apply_transform(site.bijector.inv, v)
            log_abs_dets[site.name] = site.bijector.inv.log_abs_det_jacobian(v, z)
            unconstrained[site.name] = z
        return unconstrained, log_abs_dets

    def trace_values(
        self, constrained: dict[str, torch.Tensor], batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Lift single constrained draws to trace-side values.

        A trace-side value is what `MonadicProgram.log_joint`, a
        trace, and `Guide.rsample` exchange: each non-plate site
        carries the program's leading batch axis, shape
        ``(batch_size, *constrained_shape)``, and each plate site
        keeps ``constrained_shape``. The lift expands the one draw
        across the batch axis, so every input row sees the same
        latent value.

        Parameters
        ----------
        constrained : dict[str, torch.Tensor]
            One draw per latent site, each of shape
            `LatentSite.constrained_shape` (as in one entry of
            `MCMCResult.samples` or the output of `to_constrained`
            on an unbatched unconstrained vector). Keys that name no
            latent site are passed through unchanged.
        batch_size : int
            Size of the program input's leading batch axis.

        Returns
        -------
        dict[str, torch.Tensor]
            The trace-side values, keyed as ``constrained``.

        Raises
        ------
        ValueError
            If ``batch_size`` is less than one, or a latent site's
            value does not have shape `LatentSite.constrained_shape`.
        """
        if batch_size < 1:
            raise ValueError(
                f"LatentRegistry.trace_values: batch_size must be >= 1, "
                f"got {batch_size}"
            )
        out: dict[str, torch.Tensor] = {}
        for name, value in constrained.items():
            site = self._sites.get(name)
            if site is None:
                out[name] = value
                continue
            if tuple(value.shape) != site.constrained_shape:
                raise ValueError(
                    f"LatentRegistry.trace_values: site {name!r} has shape "
                    f"{tuple(value.shape)}; expected its constrained shape "
                    f"{site.constrained_shape}"
                )
            if site.is_plate:
                out[name] = value
            else:
                out[name] = value.unsqueeze(0).expand(batch_size, *value.shape)
        return out

    def trace_row(
        self, trace_side: dict[str, torch.Tensor], row: int = 0
    ) -> dict[str, torch.Tensor]:
        """Read one draw per latent site out of trace-side values.

        The inverse of `trace_values` for one batch row: each
        non-plate site, of shape ``(batch, *constrained_shape)``,
        contributes its row ``row``, and each plate site, of shape
        ``constrained_shape``, is returned as is. A guide's
        `Guide.rsample` output is a valid input.

        Parameters
        ----------
        trace_side : dict[str, torch.Tensor]
            Trace-side values for every latent site. Keys that name
            no latent site are ignored.
        row : int
            Index into the batch axis of the non-plate sites.

        Returns
        -------
        dict[str, torch.Tensor]
            One draw per latent site, each of shape
            `LatentSite.constrained_shape`, in declaration order.

        Raises
        ------
        KeyError
            If a latent site is missing from ``trace_side``.
        ValueError
            If a site's value is not trace-side shaped, or ``row`` is
            outside a non-plate site's batch axis.
        """
        out: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            if site.name not in trace_side:
                raise KeyError(f"LatentRegistry.trace_row: missing site {site.name!r}")
            value = trace_side[site.name]
            if site.is_plate:
                if tuple(value.shape) != site.constrained_shape:
                    raise ValueError(
                        f"LatentRegistry.trace_row: plate site {site.name!r} "
                        f"has shape {tuple(value.shape)}; expected "
                        f"{site.constrained_shape}"
                    )
                out[site.name] = value
                continue
            if value.dim() != len(site.constrained_shape) + 1 or (
                tuple(value.shape[1:]) != site.constrained_shape
            ):
                raise ValueError(
                    f"LatentRegistry.trace_row: site {site.name!r} has shape "
                    f"{tuple(value.shape)}; expected the trace-side shape "
                    f"(batch, *{site.constrained_shape})"
                )
            if not 0 <= row < value.shape[0]:
                raise ValueError(
                    f"LatentRegistry.trace_row: row {row} is outside the "
                    f"batch axis of size {value.shape[0]} on site {site.name!r}"
                )
            out[site.name] = value[row]
        return out

    @staticmethod
    def aggregate_log_abs_det(
        log_abs_dets: dict[str, torch.Tensor],
        leading_shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Sum every per-site ``log_abs_det`` over its event axes
        and produce a single tensor broadcast to ``leading_shape``.

        This is the canonical aggregation an ELBO needs. Objectives
        that want particle-aware or per-site granularity can skip
        this helper and aggregate themselves.
        """
        if not log_abs_dets:
            return torch.zeros(leading_shape) if leading_shape else torch.zeros(())
        total: torch.Tensor | None = None
        n_leading = len(leading_shape)
        for site_lad in log_abs_dets.values():
            # Sum away every axis beyond the leading shape; what
            # remains is broadcast against the leading shape.
            while site_lad.dim() > n_leading:
                site_lad = site_lad.sum(dim=-1)
            total = site_lad if total is None else total + site_lad
        assert total is not None
        return total

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    def zero_unconstrained(
        self,
        *,
        leading_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a dict of zero-initialized unconstrained site
        tensors with the supplied leading batch shape.

        Useful for HMC chain initialisation and AutoLaplace's MAP
        starting point.
        """
        out: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            shape = leading_shape + site.unconstrained_shape
            out[site.name] = torch.zeros(shape, device=device, dtype=dtype)
        return out

    def randn_unconstrained(
        self,
        *,
        leading_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Standard-normal-initialized unconstrained site tensors."""
        out: dict[str, torch.Tensor] = {}
        for site in self._sites.values():
            shape = leading_shape + site.unconstrained_shape
            out[site.name] = scale * torch.randn(shape, device=device, dtype=dtype)
        return out


__all__ = ["LatentRegistry", "LatentSite"]
