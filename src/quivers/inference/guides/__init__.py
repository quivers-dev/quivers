"""Variational guide families.

Public surface (re-exported by the parent `quivers.inference`
package): one ABC (`Guide`) plus a zoo of concrete
``Auto*Guide`` subclasses spanning the standard variational-family
ladder from mean-field Normal to normalizing-flow stacks and
hierarchical / mixture / structured guides.

Every concrete guide is built against a single
`quivers.inference.registry.LatentRegistry` and obeys the
shape contract documented on `Guide`.

Short-name aliases matching the Pyro `AutoGuide` naming
(`AutoNormal`, `AutoMultivariateNormal`, `AutoLowRankMVN`,
`AutoDelta`, `AutoLaplace`, `AutoIAFNormal`) are re-exported for
users switching over from Pyro. The longer ``*Guide`` names remain
the canonical quivers names.
"""

from __future__ import annotations

from quivers.inference.guides.auto_guide_list import AutoGuideList
from quivers.inference.guides.auto_structured import (
    AutoStructured,
    ConditionalSpec,
    DependencyKind,
    DependencySpec,
)
from quivers.inference.guides.base import Guide, GuideDraw
from quivers.inference.guides.delta import AutoDeltaGuide
from quivers.inference.guides.flow import (
    AutoIAFGuide,
    AutoNeuralSplineGuide,
    AutoNormalizingFlow,
)
from quivers.inference.guides.laplace import AutoLaplaceApproximation
from quivers.inference.guides.mixture import AutoMixtureGuide
from quivers.inference.guides.multivariate_normal import (
    AutoLowRankMultivariateNormalGuide,
    AutoMultivariateNormalGuide,
)
from quivers.inference.guides.normal import AutoNormalGuide

AutoNormal = AutoNormalGuide
"""Pyro-style short name for `AutoNormalGuide`."""

AutoMultivariateNormal = AutoMultivariateNormalGuide
"""Pyro-style short name for `AutoMultivariateNormalGuide`."""

AutoLowRankMVN = AutoLowRankMultivariateNormalGuide
"""Pyro-style short name for `AutoLowRankMultivariateNormalGuide`."""

AutoDelta = AutoDeltaGuide
"""Pyro-style short name for `AutoDeltaGuide`."""

AutoLaplace = AutoLaplaceApproximation
"""Pyro-style short name for `AutoLaplaceApproximation`."""

AutoIAFNormal = AutoIAFGuide
"""Pyro-style short name for `AutoIAFGuide`."""

__all__ = [
    "Guide",
    "GuideDraw",
    "AutoNormalGuide",
    "AutoMultivariateNormalGuide",
    "AutoLowRankMultivariateNormalGuide",
    "AutoDeltaGuide",
    "AutoLaplaceApproximation",
    "AutoNormalizingFlow",
    "AutoIAFGuide",
    "AutoNeuralSplineGuide",
    "AutoMixtureGuide",
    "AutoGuideList",
    "AutoStructured",
    "ConditionalSpec",
    "DependencyKind",
    "DependencySpec",
    "AutoNormal",
    "AutoMultivariateNormal",
    "AutoLowRankMVN",
    "AutoDelta",
    "AutoLaplace",
    "AutoIAFNormal",
]
