"""Categorical structures: functors, natural transformations, adjunctions, monoidal, and traced categories."""

from quivers.categorical.functors import (
    Functor,
    IdentityFunctor,
    ComposedFunctor,
    FreeMonoidFunctor,
    IDENTITY,
)
from quivers.categorical.natural_transformations import (
    NaturalTransformation,
    ComponentwiseNT,
)
from quivers.categorical.adjunctions import (
    Adjunction,
)
from quivers.categorical.monoidal import (
    MonoidalStructure,
    CartesianMonoidal,
    CoproductMonoidal,
    EmptySet,
    EMPTY,
)
from quivers.categorical.base_change import (
    BaseChange,
    BoolToFuzzy,
    FuzzyToBool,
)
from quivers.categorical.traced import (
    TracedMonoidal,
    CartesianTrace,
    IterativeTrace,
    trace,
    partial_trace,
)

__all__ = [
    # functors
    "Functor",
    "IdentityFunctor",
    "ComposedFunctor",
    "FreeMonoidFunctor",
    "IDENTITY",
    # natural_transformations
    "NaturalTransformation",
    "ComponentwiseNT",
    # adjunctions
    "Adjunction",
    # monoidal
    "MonoidalStructure",
    "CartesianMonoidal",
    "CoproductMonoidal",
    "EmptySet",
    "EMPTY",
    # base_change
    "BaseChange",
    "BoolToFuzzy",
    "FuzzyToBool",
    # traced
    "TracedMonoidal",
    "CartesianTrace",
    "IterativeTrace",
    "trace",
    "partial_trace",
]
