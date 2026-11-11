"""Per-site / per-edge structured variational guide.

`AutoStructured` is a compositional guide construction that lets
the user pick, per latent site, the family of the marginal
variational distribution (`"delta"`, `"normal"`, or `"mvn"`) and,
per directed edge between latents, a functional dependence
(`"linear"` for a learnable affine map, or a user-supplied
callable). The resulting joint is a directed graphical model over
the latent sites where each site's conditional is a chosen
one-parameter family whose location is the sum of a learned bias
and every incoming edge's contribution.

This is the Pyro
[`AutoStructured`](https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoStructured)
API, exported into the quivers guide zoo. It subsumes
`AutoNormalGuide` (all sites `"normal"`, no dependencies),
`AutoMultivariateNormalGuide` (a single `"mvn"` block for every
site, no dependencies) and `AutoDeltaGuide` (all sites `"delta"`)
while opening the space between them: a small global block with an
`"mvn"` marginal + `"linear"` dependencies on a large local block
of `"normal"` marginals is the standard sparse-precision guide
that Pyro's tutorial documents.

Sampling ordering
=================

Sites are sampled in the model's declaration order (the same
order as `LatentRegistry.names`). Every dependency reference
must point to a site that appears earlier in declaration order;
`AutoStructured.__init__` checks this and raises
`ValueError` otherwise. This upstream-before-downstream
discipline matches the ancestral-sampling semantics of the
underlying directed graphical model.

Constrained support
===================

Every site's unconstrained-space draw is pushed through the
site's `torch.distributions.transforms.biject_to`-derived
`Transform` before being returned in the sample dict, and the
matching Jacobian correction is folded into
`AutoStructured.log_prob`. Users who want a bespoke bijector
(e.g. `Compose(StickBreaking(), Affine(...))` on a simplex site)
should attach it via the bijector-typed parameter transform
machinery so `LatentRegistry` picks it up.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.registry import LatentSite


type DependencyKind = str | Callable[[torch.Tensor], torch.Tensor]
"""One dependency edge of an `AutoStructured` guide: the string
``"linear"`` for a learned linear shift, or a callable mapping the
parent's unconstrained flat draw to a shift on the child's
unconstrained flat location."""

type ConditionalSpec = str | Mapping[str, str]
"""The per-site marginal families of an `AutoStructured` guide: one
family name (``"delta"``, ``"normal"``, or ``"mvn"``) for every site,
or a mapping from site name to family name."""

type DependencySpec = Mapping[str, Mapping[str, DependencyKind]]
"""The dependency edges of an `AutoStructured` guide, keyed first by
child site and then by parent site."""


_ALLOWED_CONDITIONALS: frozenset[str] = frozenset({"delta", "normal", "mvn"})


class _LinearDependency(nn.Module):
    """A learnable affine map from a parent site's unconstrained
    draw to a child site's location shift.

    Both the parent and the child live in a flat unconstrained
    space (plate sites are flattened to a length-`plate_index_size
    * unconstrained_dim` vector; scalar sites are already flat).
    The map is `W @ z_parent`, with `W` of shape
    `(child_flat, parent_flat)`, initialised to zero so the
    dependency is a no-op at the start of training.
    """

    def __init__(self, parent_flat: int, child_flat: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(child_flat, parent_flat))

    def forward(self, z_parent_flat: torch.Tensor) -> torch.Tensor:
        # z_parent_flat is (parent_flat,); output is (child_flat,).
        return F.linear(z_parent_flat, self.weight)


class _CallableDependency(nn.Module):
    """A user-supplied Python callable wrapped as a `torch.nn.Module`
    so it composes cleanly with the guide's parameter registry."""

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self._fn = fn

    def forward(self, z_parent_flat: torch.Tensor) -> torch.Tensor:
        return self._fn(z_parent_flat)


def _site_flat_length(site: LatentSite) -> int:
    """The unconstrained-space flat dimensionality of a site.

    Plate sites of shape `(|A|, d)` contribute `|A| * d`; scalar
    sites of shape `(d,)` contribute `d`.
    """
    return site.flat_length


def _push_through_bijector(
    site: LatentSite, z_flat: torch.Tensor, batch: int
) -> torch.Tensor:
    """Reshape a flat unconstrained draw into the site's shape,
    apply the constrained-support bijector, then squeeze the
    trailing length-1 event axis if the site is scalar-supported.
    """
    z_shaped = z_flat.reshape(*site.unconstrained_shape)
    if site.is_plate:
        v: torch.Tensor = site.bijector(z_shaped)
    else:
        z_batched = z_shaped.unsqueeze(0).expand(batch, *site.unconstrained_shape)
        v = site.bijector(z_batched)
    if site.constrained_dim == 1 and v.dim() >= 1 and v.shape[-1] == 1:
        v = v.squeeze(-1)
    return v


class AutoStructured(Guide):
    """Per-site / per-edge structured variational guide.

    Parameters
    ----------
    model : MonadicProgram
        Generative model to build a guide for.
    observed_names : set[str]
        Variable names treated as observations.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the latent
        registry from a run, required for a program that scans or calls
        a program that draws, whose latent sites depend on its input.
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The observed values and host data of that run.
    conditionals : str or dict[str, str]
        Per-site marginal-family choice. A single string sets every
        site to the same family. A dict maps site name to family.
        Allowed families:

        * `"delta"`: Dirac mass at a learnable location
          (analogous to `AutoDeltaGuide`).
        * `"normal"`: independent-Normal marginal.
        * `"mvn"`: the site is part of a joint multivariate
          Normal block. Every `"mvn"` site shares one Cholesky
          factor sized to the sum of every `"mvn"` site's flat
          length. This matches Pyro's convention.
    dependencies : dict[str, dict[str, DependencyKind]] or None
        Directed edges. `dependencies[child][parent]` is either
        the string `"linear"` (build a `_LinearDependency` from the
        parent's unconstrained flat draw to a shift on the child's
        unconstrained flat location) or a user-supplied callable
        with the same signature. Every dependency must reference a
        site strictly upstream in declaration order.
    init_scale : float
        Initial scale of every `"normal"` and `"mvn"` marginal.
        Default ``0.1``.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        conditionals: ConditionalSpec,
        dependencies: DependencySpec | None = None,
        init_scale: float = 0.1,
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._registry = self.build_registry(
            model, observed_names, x=x, observations=observations
        )

        # --- Normalise conditionals ---
        site_names = list(self._registry.names)
        if isinstance(conditionals, str):
            if conditionals not in _ALLOWED_CONDITIONALS:
                raise ValueError(
                    f"AutoStructured: conditional {conditionals!r} "
                    f"not in {sorted(_ALLOWED_CONDITIONALS)!r}"
                )
            resolved: dict[str, str] = {n: conditionals for n in site_names}
        else:
            resolved = {}
            for n in site_names:
                if n not in conditionals:
                    raise ValueError(
                        f"AutoStructured: conditionals dict has no "
                        f"entry for latent site {n!r}"
                    )
                choice = conditionals[n]
                if choice not in _ALLOWED_CONDITIONALS:
                    raise ValueError(
                        f"AutoStructured: conditional {choice!r} for "
                        f"site {n!r} not in "
                        f"{sorted(_ALLOWED_CONDITIONALS)!r}"
                    )
                resolved[n] = choice
        self._conditionals: dict[str, str] = resolved

        # --- Register per-site location + scale parameters ---
        init_log_scale = float(torch.tensor(init_scale).log().item())
        for name in site_names:
            site = self._registry.sites[name]
            self.register_parameter(
                f"loc_{name}",
                nn.Parameter(torch.zeros(site.unconstrained_shape)),
            )
            if resolved[name] == "normal":
                self.register_parameter(
                    f"log_scale_{name}",
                    nn.Parameter(torch.full(site.unconstrained_shape, init_log_scale)),
                )
            # "delta" has no scale.  "mvn" scale is shared and set below.

        # --- Set up the shared MVN block, if any ---
        mvn_names: list[str] = [n for n in site_names if resolved[n] == "mvn"]
        self._mvn_names: list[str] = mvn_names
        self._mvn_offsets: dict[str, int] = {}
        cursor = 0
        for n in mvn_names:
            self._mvn_offsets[n] = cursor
            cursor += _site_flat_length(self._registry.sites[n])
        self._mvn_total: int = cursor
        if mvn_names:
            init_diag = torch.full((self._mvn_total,), float(init_scale))
            init_diag_raw = torch.log(torch.expm1(init_diag.clamp(min=1e-6)))
            self.mvn_scale_diag_raw = nn.Parameter(init_diag_raw)
            self.mvn_scale_offdiag = nn.Parameter(
                torch.zeros(self._mvn_total, self._mvn_total)
            )

        # --- Validate + register dependencies ---
        deps = dependencies or {}
        name_order: dict[str, int] = {n: i for i, n in enumerate(site_names)}
        dep_modules: dict[str, nn.ModuleDict] = {}
        for child, parents in deps.items():
            if child not in name_order:
                raise ValueError(
                    f"AutoStructured: dependency child {child!r} is not a latent site"
                )
            child_flat = _site_flat_length(self._registry.sites[child])
            per_parent = nn.ModuleDict()
            for parent, kind in parents.items():
                if parent not in name_order:
                    raise ValueError(
                        f"AutoStructured: dependency parent {parent!r} "
                        f"(for child {child!r}) is not a latent site"
                    )
                if name_order[parent] >= name_order[child]:
                    raise ValueError(
                        f"AutoStructured: dependency {parent!r} -> "
                        f"{child!r} is not strictly upstream in "
                        f"declaration order"
                    )
                parent_flat = _site_flat_length(self._registry.sites[parent])
                if kind == "linear":
                    per_parent[parent] = _LinearDependency(
                        parent_flat=parent_flat, child_flat=child_flat
                    )
                elif callable(kind):
                    per_parent[parent] = _CallableDependency(kind)
                else:
                    raise ValueError(
                        f"AutoStructured: dependency kind {kind!r} "
                        f"for edge {parent!r} -> {child!r} must be "
                        f"either the string 'linear' or a callable"
                    )
            dep_modules[child] = per_parent
        self.deps = nn.ModuleDict(dep_modules)
        self._dep_spec: dict[str, dict[str, DependencyKind]] = {
            child: dict(parents) for child, parents in deps.items()
        }

    # ------------------------------------------------------------------
    # Parameter accessors
    # ------------------------------------------------------------------

    def _loc(self, name: str) -> torch.Tensor:
        return getattr(self, f"loc_{name}")

    def _normal_scale(self, name: str) -> torch.Tensor:
        return getattr(self, f"log_scale_{name}").exp().clamp(min=1e-6)

    def _mvn_scale_tril(self) -> torch.Tensor:
        off = torch.tril(self.mvn_scale_offdiag, diagonal=-1)
        diag = F.softplus(self.mvn_scale_diag_raw) + 1e-6
        return off + torch.diag(diag)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)

    @property
    def conditionals(self) -> dict[str, str]:
        """Per-site marginal family, as resolved at construction."""
        return dict(self._conditionals)

    @property
    def dependencies(self) -> dict[str, dict[str, DependencyKind]]:
        """The declared per-edge dependence structure."""
        return {c: dict(ps) for c, ps in self._dep_spec.items()}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_mvn_block(self) -> dict[str, torch.Tensor]:
        """Return a per-site flat-vector draw for the joint-MVN
        block, keyed by site name."""
        joint_loc = torch.cat(
            [self._loc(n).reshape(-1) for n in self._mvn_names], dim=0
        )
        dist = D.MultivariateNormal(joint_loc, scale_tril=self._mvn_scale_tril())
        joint = dist.rsample()
        result: dict[str, torch.Tensor] = {}
        for n in self._mvn_names:
            off = self._mvn_offsets[n]
            length = _site_flat_length(self._registry.sites[n])
            result[n] = joint[off : off + length]
        return result

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Ancestral reparameterized draw."""
        batch = x.shape[0]
        # First sample the shared MVN block (if any) in flat form.
        if self._mvn_names:
            mvn_flat: dict[str, torch.Tensor] = self._sample_mvn_block()
        else:
            mvn_flat = {}
        # Then walk sites in declaration order, sampling each per its
        # marginal family, offsetting the location by every incoming
        # dependency's contribution.
        result: dict[str, torch.Tensor] = {}
        z_flat_by_site: dict[str, torch.Tensor] = {}
        for site in self._registry.sites.values():
            name = site.name
            kind = self._conditionals[name]
            loc = self._loc(name)
            # Dependency shift, if any.
            shift_flat = torch.zeros(_site_flat_length(site), device=loc.device)
            if name in self.deps:
                for parent, dep_mod in self.deps[name].items():
                    parent_flat = z_flat_by_site[parent]
                    shift_flat = shift_flat + dep_mod(parent_flat)
            shift = shift_flat.reshape(site.unconstrained_shape)
            shifted_loc = loc + shift
            if kind == "delta":
                z_flat = shifted_loc.reshape(-1)
            elif kind == "normal":
                scale = self._normal_scale(name)
                z_shaped = D.Normal(shifted_loc, scale).rsample()
                z_flat = z_shaped.reshape(-1)
            elif kind == "mvn":
                z_flat = mvn_flat[name] + shift_flat
            else:
                raise RuntimeError(
                    f"AutoStructured.rsample: unhandled conditional kind {kind!r}"
                )
            z_flat_by_site[name] = z_flat
            v = _push_through_bijector(site, z_flat, batch)
            result[name] = v
        return result

    # ------------------------------------------------------------------
    # Log-density
    # ------------------------------------------------------------------

    def _v_to_unconstrained(
        self, site: LatentSite, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the bijector and return `(z_shaped, log|det J^{-1}|)`
        at the natural site shape."""
        if site.constrained_dim == 1 and v.dim() == (1 if site.is_plate else 1):
            v_e = v.unsqueeze(-1)
        else:
            v_e = v
        if not site.is_plate and v_e.dim() == len(site.unconstrained_shape) + 1:
            v_e = v_e[0]
        z: torch.Tensor = site.bijector.inv(v_e)
        log_det = site.bijector.inv.log_abs_det_jacobian(v_e, z)
        return z, log_det

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pushforward log-density at the supplied constrained sites."""
        batch = x.shape[0]
        z_flat_by_site: dict[str, torch.Tensor] = {}
        total_scalar = torch.zeros((), device=x.device)
        # First invert every bijector and accumulate the Jacobian terms.
        for site in self._registry.sites.values():
            name = site.name
            if name not in sites:
                raise KeyError(f"AutoStructured.log_prob: missing site {name!r}")
            z_shaped, log_det = self._v_to_unconstrained(site, sites[name])
            z_flat_by_site[name] = z_shaped.reshape(-1)
            total_scalar = total_scalar + log_det.reshape(-1).sum()
        # Next add each site's marginal log-density.
        for site in self._registry.sites.values():
            name = site.name
            kind = self._conditionals[name]
            if kind == "delta":
                continue
            loc = self._loc(name)
            shift_flat = torch.zeros(_site_flat_length(site), device=x.device)
            if name in self.deps:
                for parent, dep_mod in self.deps[name].items():
                    shift_flat = shift_flat + dep_mod(z_flat_by_site[parent])
            shift = shift_flat.reshape(site.unconstrained_shape)
            shifted_loc = loc + shift
            if kind == "normal":
                scale = self._normal_scale(name)
                z_shaped = z_flat_by_site[name].reshape(site.unconstrained_shape)
                log_q = D.Normal(shifted_loc, scale).log_prob(z_shaped)
                total_scalar = total_scalar + log_q.reshape(-1).sum()
            # "mvn" sites are scored jointly below.
        # Finally, the joint MVN block, if any.
        if self._mvn_names:
            joint_loc_parts: list[torch.Tensor] = []
            joint_z_parts: list[torch.Tensor] = []
            for n in self._mvn_names:
                site = self._registry.sites[n]
                loc = self._loc(n)
                shift_flat = torch.zeros(_site_flat_length(site), device=x.device)
                if n in self.deps:
                    for parent, dep_mod in self.deps[n].items():
                        shift_flat = shift_flat + dep_mod(z_flat_by_site[parent])
                shifted_loc_flat = loc.reshape(-1) + shift_flat
                joint_loc_parts.append(shifted_loc_flat)
                joint_z_parts.append(z_flat_by_site[n])
            joint_loc = torch.cat(joint_loc_parts, dim=0)
            joint_z = torch.cat(joint_z_parts, dim=0)
            dist = D.MultivariateNormal(joint_loc, scale_tril=self._mvn_scale_tril())
            total_scalar = total_scalar + dist.log_prob(joint_z)
        return total_scalar.expand(batch)


__all__ = [
    "AutoStructured",
    "ConditionalSpec",
    "DependencyKind",
    "DependencySpec",
]
