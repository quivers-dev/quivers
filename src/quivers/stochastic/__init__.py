"""Stochastic morphisms: Markov kernels over finite sets.

This module provides the probabilistic layer for quivers, implementing
the category **FinStoch** of finite sets and stochastic maps (Markov
kernels). This is the Kleisli category of the Giry monad restricted
to finite sets, where:

- Objects are finite sets (same as the base category).
- Morphisms A -> B are stochastic matrices: tensors of shape
  (|A|, |B|) whose rows sum to 1.
- Composition is standard matrix multiplication.
- Identity is the Kronecker delta.

The package also re-exports `MarkovAlgebra` and its singleton
`MARKOV` from `quivers.core.algebras`, the algebra these morphisms
compose under.

Submodules
----------
morphisms : StochasticMorphism, CategoricalMorphism
families : Discretized distribution families
transforms : condition, mix, factor, normalize
queries : prob, marginal_prob, expectation
giry : GiryMonad, FinStoch
categories : Category types including atoms, slashes, products, unit, modals
semiring : Chart semiring abstractions (LogProb, Viterbi, Boolean, Counting)
schema : Composable rule schemas (functors CategorySystem -> RuleSystem)
rules : Rule systems and the CCG / Lambek presets
span : Span-based CKY components (LexicalAxiom, BinarySpanDeduction, etc.)
parsers, ccg, lambek : Chart parsers
inside : The inside algorithm over a rule system
agenda : The agenda-driven weighted-deduction engine
productions : A grammar's productions grouped by left-hand side
effect_lifts : Effect-lifted rule schemas for deduction systems
deduction : Abstract weighted deductive system framework and its operations
stdlib : Pre-registered deduction systems
"""

from __future__ import annotations

from quivers.core.algebras import MARKOV, MarkovAlgebra
from quivers.stochastic.morphisms import (
    StochasticMorphism,
    CategoricalMorphism,
    stochastic,
)
from quivers.stochastic.families import (
    DiscretizedNormal,
    DiscretizedLogitNormal,
    DiscretizedBeta,
    DiscretizedTruncatedNormal,
)
from quivers.stochastic.transforms import (
    ConditionedMorphism,
    condition,
    MixtureMorphism,
    mix,
    FactoredMorphism,
    factor,
    NormalizedMorphism,
    normalize,
)
from quivers.stochastic.queries import (
    prob,
    marginal_prob,
    expectation,
)
from quivers.stochastic.giry import (
    GiryMonad,
    FinStoch,
)
from quivers.stochastic.categories import (
    Category,
    AtomicCategory,
    SlashCategory,
    ProductCategory,
    UnitCategory,
    ModalCategory,
    BUILTIN_CONSTRUCTOR_NAMES,
    CategorySystem,
)
from quivers.stochastic.semiring import (
    ChartSemiring,
    LogProbSemiring,
    ViterbiSemiring,
    BooleanSemiring,
    CountingSemiring,
    LOG_PROB,
    VITERBI,
    BOOLEAN,
    COUNTING,
)
from quivers.stochastic.schema import (
    RuleSchema,
    UnionSchema,
    WeightedSchema,
    BinaryRuleSchema,
    UnaryRuleSchema,
    ForwardApplication,
    BackwardApplication,
    ForwardComposition,
    BackwardComposition,
    ForwardCrossedComposition,
    BackwardCrossedComposition,
    CommutativeForwardApplication,
    CommutativeBackwardApplication,
    TensorIntroduction,
    LeftUnitElimination,
    RightUnitElimination,
    ModalApplication,
    RightLifting,
    LeftLifting,
    LeftProjection,
    RightProjection,
    UnitCoercion,
    ModalInjection,
    ModalProjection,
    GeneralizedComposition,
    PatternBinarySchema,
    PatternUnarySchema,
    EVALUATION,
    HARMONIC_COMPOSITION,
    CROSSED_COMPOSITION,
    COMMUTATIVE_EVALUATION,
    ADJUNCTION_UNITS,
    TENSOR_INTRODUCTION,
    TENSOR_PROJECTION,
    UNIT_INTRODUCTION,
    UNIT_ELIMINATION,
    MODAL_INTRODUCTION,
    MODAL_ELIMINATION,
    MODAL_APPLICATION,
    generalized_composition,
    CCG,
    LAMBEK,
    NL,
    LP,
    SCHEMA_REGISTRY,
)
from quivers.stochastic.rules import (
    RuleSystem,
    ccg_rules,
    lambek_rules,
    custom_rules,
)
from quivers.stochastic.span import (
    LexicalAxiom,
    SpanChart,
    BinarySpanDeduction,
    UnarySpanDeduction,
    SpanGoal,
    CKYSchedule,
)
from quivers.stochastic.parsers import (
    ChartParser,
)
from quivers.stochastic.ccg import (
    CCGParser,
)
from quivers.stochastic.lambek import (
    LambekParser,
)
from quivers.stochastic.inside import (
    InsideAlgorithm,
)
from quivers.stochastic.agenda import (
    Item,
    Wildcard,
    make_wildcard,
    Pattern,
    Bindings,
    InferenceRule,
    instantiate,
    match,
    Chart,
    HashChart,
    ChartView,
    Agenda,
    FIFOAgenda,
    LIFOAgenda,
    PriorityQueueAgenda,
    AgendaResult,
    run_agenda,
    DeductionSystem,
    cky_agenda,
    earley_agenda,
    viterbi_agenda,
    astar_agenda,
    knuth_agenda,
    depth_first_agenda,
    semi_naive_agenda,
)
from quivers.stochastic.productions import (
    Production,
    ProductionGroup,
    ProductionKey,
    ProductionWeights,
)
from quivers.stochastic.effect_lifts import (
    class_directed_lifts,
    make_swap_schema,
    swap_rule_set,
    lift_rule_set,
)

__all__ = [
    # core.algebras
    "MarkovAlgebra",
    "MARKOV",
    # morphisms
    "StochasticMorphism",
    "CategoricalMorphism",
    "stochastic",
    # families
    "DiscretizedNormal",
    "DiscretizedLogitNormal",
    "DiscretizedBeta",
    "DiscretizedTruncatedNormal",
    # transforms
    "ConditionedMorphism",
    "condition",
    "MixtureMorphism",
    "mix",
    "FactoredMorphism",
    "factor",
    "NormalizedMorphism",
    "normalize",
    # queries
    "prob",
    "marginal_prob",
    "expectation",
    # giry
    "GiryMonad",
    "FinStoch",
    # categories
    "Category",
    "AtomicCategory",
    "SlashCategory",
    "ProductCategory",
    "UnitCategory",
    "ModalCategory",
    "BUILTIN_CONSTRUCTOR_NAMES",
    "CategorySystem",
    # semiring
    "ChartSemiring",
    "LogProbSemiring",
    "ViterbiSemiring",
    "BooleanSemiring",
    "CountingSemiring",
    "LOG_PROB",
    "VITERBI",
    "BOOLEAN",
    "COUNTING",
    # schema
    "RuleSchema",
    "UnionSchema",
    "WeightedSchema",
    "BinaryRuleSchema",
    "UnaryRuleSchema",
    "ForwardApplication",
    "BackwardApplication",
    "ForwardComposition",
    "BackwardComposition",
    "ForwardCrossedComposition",
    "BackwardCrossedComposition",
    "CommutativeForwardApplication",
    "CommutativeBackwardApplication",
    "TensorIntroduction",
    "LeftUnitElimination",
    "RightUnitElimination",
    "ModalApplication",
    "RightLifting",
    "LeftLifting",
    "LeftProjection",
    "RightProjection",
    "UnitCoercion",
    "ModalInjection",
    "ModalProjection",
    "GeneralizedComposition",
    "PatternBinarySchema",
    "PatternUnarySchema",
    "EVALUATION",
    "HARMONIC_COMPOSITION",
    "CROSSED_COMPOSITION",
    "COMMUTATIVE_EVALUATION",
    "ADJUNCTION_UNITS",
    "TENSOR_INTRODUCTION",
    "TENSOR_PROJECTION",
    "UNIT_INTRODUCTION",
    "UNIT_ELIMINATION",
    "MODAL_INTRODUCTION",
    "MODAL_ELIMINATION",
    "MODAL_APPLICATION",
    "generalized_composition",
    "CCG",
    "LAMBEK",
    "NL",
    "LP",
    "SCHEMA_REGISTRY",
    # rules
    "RuleSystem",
    "ccg_rules",
    "lambek_rules",
    "custom_rules",
    # span
    "LexicalAxiom",
    "SpanChart",
    "BinarySpanDeduction",
    "UnarySpanDeduction",
    "SpanGoal",
    "CKYSchedule",
    # parsers
    "ChartParser",
    # ccg
    "CCGParser",
    # lambek
    "LambekParser",
    # inside
    "InsideAlgorithm",
    # agenda
    "Item",
    "Wildcard",
    "make_wildcard",
    "Pattern",
    "Bindings",
    "InferenceRule",
    "instantiate",
    "match",
    "Chart",
    "HashChart",
    "ChartView",
    "Agenda",
    "FIFOAgenda",
    "LIFOAgenda",
    "PriorityQueueAgenda",
    "AgendaResult",
    "run_agenda",
    "DeductionSystem",
    "cky_agenda",
    "earley_agenda",
    "viterbi_agenda",
    "astar_agenda",
    "knuth_agenda",
    "depth_first_agenda",
    "semi_naive_agenda",
    # productions
    "Production",
    "ProductionGroup",
    "ProductionKey",
    "ProductionWeights",
    # effect_lifts
    "class_directed_lifts",
    "make_swap_schema",
    "swap_rule_set",
    "lift_rule_set",
]
