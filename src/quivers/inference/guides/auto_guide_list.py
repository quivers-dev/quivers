"""Block-composable variational guide.

`AutoGuideList` concatenates several `Guide`
instances, each covering a disjoint subset of the model's latent
sites, into a single guide that satisfies the
[`quivers.inference.guides.base.Guide`][quivers.inference.guides.base.Guide]
contract. This is the standard Pyro
`AutoGuideList` pattern
(https://docs.pyro.ai/en/stable/infer.autoguide.html#autoguidelist):
a hierarchical model with a small global block and a large local
block can carry `AutoMultivariateNormalGuide` on the global
sites and `AutoNormalGuide` on the locals, buying the strong
posterior correlations where they matter while keeping the local
guide's parameter count linear in the plate size.

Every part's `Guide.registry` is inspected at construction;
the parts' `LatentRegistry.names` must partition (no overlap,
no gap) the union of covered site names. `AutoGuideList` does
not itself build a merged registry, since a partition of
site-space suffices for sampling and log-density: `rsample`
merges per-part dicts, `log_prob` sums per-part scalars.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quivers.inference.guides.base import Guide, GuideDraw


class AutoGuideList(Guide):
    """A block-composed variational guide.

    Parameters
    ----------
    parts : dict[str, Guide]
        Named subguides, each covering a disjoint set of latent
        site names. Keys are labels used for parameter-namespacing
        and error messages; values are any
        `Guide` subclass. Every guide's
        `Guide.latent_names` must be pairwise disjoint from every
        other's.
    """

    def __init__(self, parts: dict[str, Guide]) -> None:
        super().__init__()
        if len(parts) == 0:
            raise ValueError("AutoGuideList: parts must be non-empty")

        seen: dict[str, str] = {}
        for label, part in parts.items():
            for name in part.latent_names:
                if name in seen:
                    raise ValueError(
                        f"AutoGuideList: latent {name!r} appears in "
                        f"both part {seen[name]!r} and part {label!r}; "
                        f"parts must cover disjoint site sets"
                    )
                seen[name] = label
        self._parts_dict: dict[str, Guide] = dict(parts)
        self.parts = nn.ModuleDict(parts)
        self._latent_names: list[str] = list(seen.keys())
        self._site_to_part: dict[str, str] = seen

    @property
    def latent_names(self) -> list[str]:
        return list(self._latent_names)

    @property
    def part_labels(self) -> tuple[str, ...]:
        """Labels of the sub-guides, in declaration order."""
        return tuple(self._parts_dict.keys())

    def part(self, label: str) -> Guide:
        """Look up a sub-guide by label."""
        if label not in self._parts_dict:
            raise KeyError(
                f"AutoGuideList: no part labelled {label!r}; "
                f"available: {tuple(self._parts_dict)!r}"
            )
        return self._parts_dict[label]

    @property
    def registry(self):
        """The list-guide has no single fused registry; downstream
        code that needs one should walk `part_labels` and pull each
        sub-guide's `Guide.registry` directly.
        """
        raise AttributeError(
            "AutoGuideList has no fused registry; access "
            "per-part registries via `guide.part(label).registry`"
        )

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Concatenate per-part reparameterized draws."""
        result: dict[str, torch.Tensor] = {}
        for part in self._parts_dict.values():
            part_sample = part.rsample(x)
            for name, tensor in part_sample.items():
                result[name] = tensor
        return result

    def rsample_strata(self, x: torch.Tensor) -> tuple[GuideDraw, ...]:
        """Product of the parts' strata.

        The parts are independent, so the joint guide's strata are the
        Cartesian product of the parts' strata: each joint draw merges
        one draw per part and carries the sums of the parts'
        log-weights and log-densities. A part that is an
        `AutoMixtureGuide` thus has its components enumerated inside
        the list guide, and an `AutoDAIS` part contributes the density
        of its own trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.

        Returns
        -------
        tuple[GuideDraw, ...]
            ``S`` merged draws, where ``S`` is the product of the parts'
            stratum counts.
        """
        joint: list[GuideDraw] | None = None
        for part in self._parts_dict.values():
            part_draws = part.rsample_strata(x)
            if joint is None:
                joint = list(part_draws)
                continue
            joint = [
                GuideDraw(
                    sites={**prefix.sites, **draw.sites},
                    log_density=prefix.log_density + draw.log_density,
                    log_weight=prefix.log_weight + draw.log_weight,
                )
                for prefix in joint
                for draw in part_draws
            ]
        assert joint is not None
        return tuple(joint)

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sum per-part log-densities.

        Each sub-guide receives only the site keys it owns; a
        missing key raises `KeyError` in the sub-guide.
        """
        total: torch.Tensor | None = None
        for label, part in self._parts_dict.items():
            part_sites = {
                name: sites[name] for name in part.latent_names if name in sites
            }
            missing = [n for n in part.latent_names if n not in sites]
            if missing:
                raise KeyError(
                    f"AutoGuideList.log_prob: part {label!r} needs "
                    f"sites {missing!r} but they are not in the input"
                )
            contribution = part.log_prob(x, part_sites)
            total = contribution if total is None else total + contribution
        assert total is not None
        return total


__all__ = ["AutoGuideList"]
