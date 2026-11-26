"""Reparameterisation strategies.

`Reparam` is the abstract strategy interface; `ReparamOrchestrator`
(returned by the `reparam(strategies)` factory) is the effect handler
that dispatches each covered site to the matching strategy. The concrete
strategies are `LocScaleReparam` (non-centred Normal), `TransformReparam`
(fixed bijector), `NeuTraReparam` (warp geometry through a trained
autoguide), and `ConjugateReparam` (analytic marginalisation for
conjugate pairs).

See the individual module docstrings for citations to the original
literature: [Betancourt and Girolami 2015](https://arxiv.org/abs/1312.0906)
for the non-centred rewrite,
[Hoffman et al. 2019](https://arxiv.org/abs/1903.03704) for NeuTra,
and [Bernardo and Smith 1994](https://doi.org/10.1002/9780470316870)
for the conjugate pairs.
"""

from __future__ import annotations

from quivers.effects.reparam.base import (
    Reparam,
    ReparamOrchestrator,
    SiteRequest,
    reparam,
)
from quivers.effects.reparam.conjugate import ConjugateReparam
from quivers.effects.reparam.loc_scale import LocScaleReparam
from quivers.effects.reparam.neutra import NeuTraDraw, NeuTraReparam
from quivers.effects.reparam.transform import TransformReparam

__all__ = [
    "Reparam",
    "ReparamOrchestrator",
    "SiteRequest",
    "reparam",
    "LocScaleReparam",
    "TransformReparam",
    "NeuTraReparam",
    "NeuTraDraw",
    "ConjugateReparam",
]
