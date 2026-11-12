"""Enriched category structures: ends, coends, Kan extensions, and optics.

This subpackage contains canonical implementations of advanced enriched
category structures, organized thematically:

- `ends_coends`: ends and coends (generalized joins and meets);
- `kan_extensions`: Kan extensions (indexed re-mapping);
- `weighted_limits`: weighted limits and colimits;
- `profunctors`: profunctors (V-valued bimodules);
- `yoneda`: the Yoneda embedding and representable profunctors;
- `day_convolution`: Day convolution on presheaves;
- `optics`: optics (composable bidirectional transformations).
"""

from __future__ import annotations

# ends_coends
from quivers.enriched.ends_coends import coend, end

# kan_extensions
from quivers.enriched.kan_extensions import (
    ObjectMap,
    Projection,
    Inclusion,
    left_kan,
    right_kan,
)

# weighted_limits
from quivers.enriched.weighted_limits import (
    Weight,
    weighted_limit_morphisms,
    weighted_colimit_morphisms,
    representable_weight,
    terminal_weight,
)

# profunctors
from quivers.enriched.profunctors import Profunctor

# yoneda
from quivers.enriched.yoneda import (
    Presheaf,
    representable_profunctor,
    corepresentable_profunctor,
    yoneda_embedding,
    yoneda_lemma,
    yoneda_density,
    verify_yoneda_preserves_composition,
)

# day_convolution
from quivers.enriched.day_convolution import (
    day_convolution,
    day_unit,
    day_convolution_profunctors,
)

# optics
from quivers.enriched.optics import (
    Optic,
    Lens,
    Prism,
    Adapter,
    Grate,
    ComposedOptic,
    compose_optics,
)

__all__ = [
    # ends_coends
    "coend",
    "end",
    # kan_extensions
    "ObjectMap",
    "Projection",
    "Inclusion",
    "left_kan",
    "right_kan",
    # weighted_limits
    "Weight",
    "weighted_limit_morphisms",
    "weighted_colimit_morphisms",
    "representable_weight",
    "terminal_weight",
    # profunctors
    "Profunctor",
    # yoneda
    "Presheaf",
    "representable_profunctor",
    "corepresentable_profunctor",
    "yoneda_embedding",
    "yoneda_lemma",
    "yoneda_density",
    "verify_yoneda_preserves_composition",
    # day_convolution
    "day_convolution",
    "day_unit",
    "day_convolution_profunctors",
    # optics
    "Optic",
    "Lens",
    "Prism",
    "Adapter",
    "Grate",
    "ComposedOptic",
    "compose_optics",
]
