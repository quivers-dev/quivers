"""quivers: V-enriched categorical relations as PyTorch tensors.

Provides finite sets as categorical objects, V-enriched relations as
morphisms (tensors with values in an algebra's lattice), and
parameterized composition via algebra enrichment.

Quick start::

    from quivers import FinSet, morphism, observed, identity, Program

    X = FinSet(name="X", cardinality=3)
    Y = FinSet(name="Y", cardinality=4)
    Z = FinSet(name="Z", cardinality=2)

    f = morphism(X, Y)           # latent (learnable)
    g = morphism(Y, Z)           # latent (learnable)
    h = f >> g                   # V-enriched composition X -> Z

    program = Program(h)
    output = program()           # tensor of shape (3, 2)
"""

from importlib.metadata import version as _version

#: Read from the installed distribution's metadata, which the build
#: takes from ``pyproject.toml``. A literal here would be a second
#: place to state the version and a second place to forget it.
__version__ = _version("quivers")

from quivers.core.objects import (
    SetObject,
    FinSet,
    ProductSet,
    CoproductSet,
    FreeMonoid,
    Unit,
)
from quivers.core.algebras import (
    Algebra,
    ProductFuzzyAlgebra,
    BooleanAlgebra,
    PRODUCT_FUZZY,
    BOOLEAN,
)
from quivers.core.morphisms import (
    Morphism,
    ObservedMorphism,
    LatentMorphism,
    ComposedMorphism,
    ProductMorphism,
    MarginalizedMorphism,
    FunctorMorphism,
    morphism,
    observed,
    identity,
)
from quivers.categorical.functors import (
    Functor,
    IdentityFunctor,
    ComposedFunctor,
    FreeMonoidFunctor,
    IDENTITY,
)
from quivers.program import Program
from quivers.core.tensor_ops import (
    noisy_or_contract,
    noisy_or_reduce,
    noisy_and_reduce,
    componentwise_lift,
)
from quivers.enriched.ends_coends import coend, end
from quivers.categorical.natural_transformations import (
    NaturalTransformation,
    ComponentwiseNT,
)
from quivers.monadic.monads import (
    Monad,
    KleisliCategory,
    FuzzyPowersetMonad,
)
from quivers.categorical.adjunctions import (
    Adjunction,
)
from quivers.enriched.kan_extensions import (
    ObjectMap,
    Projection,
    Inclusion,
    left_kan,
    right_kan,
)
from quivers.enriched.profunctors import Profunctor
from quivers.categorical.monoidal import (
    MonoidalStructure,
    CartesianProductMonoidal,
    CoproductMonoidal,
    EmptySet,
    EMPTY,
)
from quivers.categorical.base_change import (
    BaseChange,
    BoolToFuzzy,
)
from quivers.monadic.distributive_laws import (
    DistributiveLaw,
)

# -- new extensions --

from quivers.monadic.comonads import (
    Comonad,
    CoKleisliCategory,
    CofreeComonad,
)
from quivers.monadic.algebras import (
    Algebra as MonadAlgebra,
    FreeAlgebra,
    ObservedAlgebra,
    Coalgebra,
    CofreeCoalgebra,
    ObservedCoalgebra,
    EilenbergMooreCategory,
)
from quivers.enriched.weighted_limits import (
    Weight,
    weighted_limit_morphisms,
    weighted_colimit_morphisms,
    representable_weight,
    terminal_weight,
)
from quivers.core.algebras import (
    StandardMVAlgebra,
    GodelAlgebra,
    TropicalAlgebra,
    MaxPlusAlgebra,
    LogProbAlgebra,
    RealAlgebra,
    BoundedSumProductAlgebra,
    CountingAlgebra,
    LUKASIEWICZ,
    GODEL,
    TROPICAL,
    MAX_PLUS,
    LOG_PROB,
    REAL,
    PROBABILITY,
    COUNTING,
)
from quivers.core.algebra_morphisms import (
    CarrierMap,
    IdentityHom,
    Embedding,
    Expectation,
    LogProb as LogProbHom,
    MaxPlus as MaxPlusHom,
    Threshold,
    MaterialImplication,
    EXPECTATION,
    LOG_PROB_MAP,
    MAX_PLUS_MAP,
    MATERIAL_IMPLICATION,
    threshold,
    embedding,
    CARRIER_MAP_REGISTRY,
    lookup_carrier_map,
)
from quivers.enriched.yoneda import (
    Presheaf,
    representable_profunctor,
    corepresentable_profunctor,
    yoneda_embedding,
    yoneda_lemma,
    yoneda_density,
    verify_yoneda_preserves_composition,
)
from quivers.enriched.day_convolution import (
    day_convolution,
    day_unit,
    day_convolution_profunctors,
)
from quivers.enriched.optics import (
    Optic,
    Lens,
    Prism,
    Adapter,
    Grate,
    compose_optics,
)
from quivers.categorical.traced import (
    TracedMonoidal,
    CompactClosedTrace,
    trace,
    partial_trace,
)

# stochastic (FinStoch / Markov kernels)
from quivers.core.algebras import MarkovAlgebra, MARKOV
from quivers.stochastic import (
    StochasticMorphism,
    CategoricalMorphism,
    DiscretizedNormal,
    DiscretizedLogitNormal,
    DiscretizedBeta,
    DiscretizedTruncatedNormal,
    ConditionedMorphism,
    MixtureMorphism,
    FactoredMorphism,
    NormalizedMorphism,
    condition,
    mix,
    factor,
    normalize,
    prob,
    marginal_prob,
    expectation,
    stochastic,
)
from quivers.giry import (
    GiryMonad,
    FinStoch,
)

# continuous (hybrid discrete-continuous architecture)
from quivers.continuous import (
    ContinuousSpace,
    Euclidean,
    UnitInterval,
    Simplex,
    PositiveReals,
    ProductSpace,
    ContinuousMorphism,
    SampledComposition,
    ProductContinuousMorphism,
    DiscreteAsContinuous,
    # families — original
    ConditionalNormal,
    ConditionalLogitNormal,
    ConditionalBeta,
    ConditionalTruncatedNormal,
    ConditionalDirichlet,
    # families — loc-scale
    ConditionalCauchy,
    ConditionalLaplace,
    ConditionalGumbel,
    ConditionalLogNormal,
    ConditionalStudentT,
    # families — positive-valued
    ConditionalExponential,
    ConditionalGamma,
    ConditionalChi2,
    ConditionalHalfCauchy,
    ConditionalHalfNormal,
    ConditionalInverseGamma,
    ConditionalWeibull,
    ConditionalPareto,
    # families — (0, 1)-valued
    ConditionalKumaraswamy,
    ConditionalLogGamma,
    ConditionalLogitBeta,
    ConditionalContinuousBernoulli,
    # families — two-df
    ConditionalFisherSnedecor,
    # families — special
    ConditionalUniform,
    # families — multivariate
    ConditionalMultivariateNormal,
    ConditionalLowRankMVN,
    # families — relaxed discrete
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
    # families — matrix-valued
    ConditionalWishart,
    ConditionalInverseWishart,
    ConditionalMatrixNormal,
    # families: non-parametric and shrinkage
    ConditionalGaussianProcess,
    ConditionalHorseshoe,
    # families — discrete-valued
    ConditionalBernoulli,
    ConditionalCategorical,
    # families — discrete count
    ConditionalPoisson,
    ConditionalNegativeBinomial,
    ConditionalGeometric,
    ConditionalBinomial,
    # families — circular / simplex / one-hot / correlation
    ConditionalVonMises,
    ConditionalLogisticNormal,
    ConditionalOneHotCategorical,
    ConditionalLKJCholesky,
    # families — distribution wrappers
    ConditionalMixture,
    ConditionalIndependent,
    ConditionalTransformed,
    Truncated,
    LKJCorrelationFactor,
    # families — shim distributions
    ConditionalBetaBinomial,
    ConditionalLogistic,
    ConditionalHalfStudentT,
    ConditionalOrderedLogistic,
    ConditionalOrderedProbit,
    # monadic programs
    MonadicProgram,
    # boundaries & flows
    Discretize,
    Embed,
    AffineCouplingLayer,
    ConditionalFlow,
)

# dsl
from quivers.dsl import (
    parse as dsl_parse,
    loads as dsl_loads,
    load as dsl_load,
    ParseError,
    CompileError,
)

__all__ = [
    # objects
    "SetObject",
    "FinSet",
    "ProductSet",
    "CoproductSet",
    "FreeMonoid",
    "Unit",
    "EmptySet",
    "EMPTY",
    # algebras
    "Algebra",
    "ProductFuzzyAlgebra",
    "BooleanAlgebra",
    "PRODUCT_FUZZY",
    "BOOLEAN",
    # extra algebras
    "StandardMVAlgebra",
    "GodelAlgebra",
    "TropicalAlgebra",
    "LUKASIEWICZ",
    "GODEL",
    "TROPICAL",
    "MaxPlusAlgebra",
    "LogProbAlgebra",
    "RealAlgebra",
    "BoundedSumProductAlgebra",
    "CountingAlgebra",
    "MAX_PLUS",
    "LOG_PROB",
    "REAL",
    "PROBABILITY",
    "COUNTING",
    "CarrierMap",
    "IdentityHom",
    "Embedding",
    "Expectation",
    "LogProbHom",
    "MaxPlusHom",
    "Threshold",
    "MaterialImplication",
    "EXPECTATION",
    "LOG_PROB_MAP",
    "MAX_PLUS_MAP",
    "MATERIAL_IMPLICATION",
    "threshold",
    "embedding",
    "CARRIER_MAP_REGISTRY",
    "lookup_carrier_map",
    # morphisms
    "Morphism",
    "ObservedMorphism",
    "LatentMorphism",
    "ComposedMorphism",
    "ProductMorphism",
    "MarginalizedMorphism",
    "FunctorMorphism",
    "morphism",
    "observed",
    "identity",
    # functors
    "Functor",
    "IdentityFunctor",
    "ComposedFunctor",
    "FreeMonoidFunctor",
    "IDENTITY",
    # program
    "Program",
    # tensor ops
    "noisy_or_contract",
    "noisy_or_reduce",
    "noisy_and_reduce",
    "componentwise_lift",
    # ends and coends
    "coend",
    "end",
    # natural transformations
    "NaturalTransformation",
    "ComponentwiseNT",
    # monads
    "Monad",
    "KleisliCategory",
    "FuzzyPowersetMonad",
    # comonads
    "Comonad",
    "CoKleisliCategory",
    "CofreeComonad",
    # algebras and coalgebras
    "MonadAlgebra",
    "FreeAlgebra",
    "ObservedAlgebra",
    "Coalgebra",
    "CofreeCoalgebra",
    "ObservedCoalgebra",
    "EilenbergMooreCategory",
    # adjunctions
    "Adjunction",
    # kan extensions
    "ObjectMap",
    "Projection",
    "Inclusion",
    "left_kan",
    "right_kan",
    # profunctors
    "Profunctor",
    # monoidal
    "MonoidalStructure",
    "CartesianProductMonoidal",
    "CoproductMonoidal",
    # base change
    "BaseChange",
    "BoolToFuzzy",
    # distributive laws
    "DistributiveLaw",
    # weighted limits
    "Weight",
    "weighted_limit_morphisms",
    "weighted_colimit_morphisms",
    "representable_weight",
    "terminal_weight",
    # yoneda
    "Presheaf",
    "representable_profunctor",
    "corepresentable_profunctor",
    "yoneda_embedding",
    "yoneda_lemma",
    "yoneda_density",
    "verify_yoneda_preserves_composition",
    # day convolution
    "day_convolution",
    "day_unit",
    "day_convolution_profunctors",
    # optics
    "Optic",
    "Lens",
    "Prism",
    "Adapter",
    "Grate",
    "compose_optics",
    # traced monoidal
    "TracedMonoidal",
    "CompactClosedTrace",
    "trace",
    "partial_trace",
    # stochastic
    "MarkovAlgebra",
    "MARKOV",
    "StochasticMorphism",
    "CategoricalMorphism",
    "DiscretizedNormal",
    "DiscretizedLogitNormal",
    "DiscretizedBeta",
    "DiscretizedTruncatedNormal",
    "ConditionedMorphism",
    "MixtureMorphism",
    "FactoredMorphism",
    "NormalizedMorphism",
    "condition",
    "mix",
    "factor",
    "normalize",
    "prob",
    "marginal_prob",
    "expectation",
    "stochastic",
    # giry monad
    "GiryMonad",
    "FinStoch",
    # continuous — spaces
    "ContinuousSpace",
    "Euclidean",
    "UnitInterval",
    "Simplex",
    "PositiveReals",
    "ProductSpace",
    # continuous — morphisms
    "ContinuousMorphism",
    "SampledComposition",
    "ProductContinuousMorphism",
    "DiscreteAsContinuous",
    # continuous — families (original)
    "ConditionalNormal",
    "ConditionalLogitNormal",
    "ConditionalBeta",
    "ConditionalTruncatedNormal",
    "ConditionalDirichlet",
    # continuous — families (loc-scale)
    "ConditionalCauchy",
    "ConditionalLaplace",
    "ConditionalGumbel",
    "ConditionalLogNormal",
    "ConditionalStudentT",
    # continuous — families (positive-valued)
    "ConditionalExponential",
    "ConditionalGamma",
    "ConditionalChi2",
    "ConditionalHalfCauchy",
    "ConditionalHalfNormal",
    "ConditionalInverseGamma",
    "ConditionalWeibull",
    "ConditionalPareto",
    # continuous — families ((0,1)-valued)
    "ConditionalKumaraswamy",
    "ConditionalLogGamma",
    "ConditionalLogitBeta",
    "ConditionalContinuousBernoulli",
    # continuous — families (two-df)
    "ConditionalFisherSnedecor",
    # continuous — families (special)
    "ConditionalUniform",
    # continuous — families (multivariate)
    "ConditionalMultivariateNormal",
    "ConditionalLowRankMVN",
    # continuous — families (relaxed discrete)
    "ConditionalRelaxedBernoulli",
    "ConditionalRelaxedOneHotCategorical",
    # continuous — families (matrix-valued)
    "ConditionalWishart",
    "ConditionalInverseWishart",
    "ConditionalMatrixNormal",
    # continuous: families (non-parametric and shrinkage)
    "ConditionalGaussianProcess",
    "ConditionalHorseshoe",
    # continuous — families (discrete-valued)
    "ConditionalBernoulli",
    "ConditionalCategorical",
    # continuous — families (discrete count)
    "ConditionalPoisson",
    "ConditionalNegativeBinomial",
    "ConditionalGeometric",
    "ConditionalBinomial",
    # continuous — families (circular / simplex / one-hot / correlation)
    "ConditionalVonMises",
    "ConditionalLogisticNormal",
    "ConditionalOneHotCategorical",
    "ConditionalLKJCholesky",
    # continuous — families (distribution wrappers)
    "ConditionalMixture",
    "ConditionalIndependent",
    "ConditionalTransformed",
    "Truncated",
    "LKJCorrelationFactor",
    # continuous — families (shim distributions)
    "ConditionalBetaBinomial",
    "ConditionalLogistic",
    "ConditionalHalfStudentT",
    "ConditionalOrderedLogistic",
    "ConditionalOrderedProbit",
    # continuous — monadic programs
    "MonadicProgram",
    "Discretize",
    "Embed",
    "AffineCouplingLayer",
    "ConditionalFlow",
    # dsl
    "dsl_parse",
    "dsl_loads",
    "dsl_load",
    "ParseError",
    "CompileError",
]
