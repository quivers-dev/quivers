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

import importlib
from types import FunctionType, GenericAlias
from typing import TYPE_CHECKING, TypeAliasType

if TYPE_CHECKING:
    from quivers.core.algebras import MARKOV, MarkovAlgebra
    from quivers.stochastic.agenda import (
        Agenda,
        AgendaResult,
        Bindings,
        Chart,
        ChartView,
        CorpusChart,
        DeductionSystem,
        FIFOAgenda,
        HashChart,
        InferenceRule,
        Item,
        LIFOAgenda,
        Pattern,
        PriorityQueueAgenda,
        Wildcard,
        astar_agenda,
        cky_agenda,
        depth_first_agenda,
        earley_agenda,
        instantiate,
        knuth_agenda,
        make_wildcard,
        match,
        run_agenda,
        semi_naive_agenda,
        viterbi_agenda,
    )
    from quivers.stochastic.categories import (
        BUILTIN_CONSTRUCTOR_NAMES,
        AtomicCategory,
        Category,
        CategorySystem,
        ModalCategory,
        ProductCategory,
        SlashCategory,
        UnitCategory,
    )
    from quivers.stochastic.ccg import (
        CCGParser,
    )
    from quivers.stochastic.effect_lifts import (
        class_directed_lifts,
        lift_rule_set,
        make_swap_schema,
        swap_rule_set,
    )
    from quivers.stochastic.families import (
        DiscretizedBeta,
        DiscretizedLogitNormal,
        DiscretizedNormal,
        DiscretizedTruncatedNormal,
    )
    from quivers.stochastic.giry import (
        FinStoch,
        GiryMonad,
    )
    from quivers.stochastic.inside import (
        InsideAlgorithm,
    )
    from quivers.stochastic.lambek import (
        LambekParser,
    )
    from quivers.stochastic.morphisms import (
        CategoricalMorphism,
        StochasticMorphism,
        stochastic,
    )
    from quivers.stochastic.parsers import (
        ChartParser,
    )
    from quivers.stochastic.productions import (
        Production,
        ProductionGroup,
        ProductionKey,
        ProductionWeights,
    )
    from quivers.stochastic.queries import (
        expectation,
        marginal_prob,
        prob,
    )
    from quivers.stochastic.rules import (
        RuleSystem,
        ccg_rules,
        custom_rules,
        lambek_rules,
    )
    from quivers.stochastic.schema import (
        ADJUNCTION_UNITS,
        CCG,
        COMMUTATIVE_EVALUATION,
        CROSSED_COMPOSITION,
        EVALUATION,
        HARMONIC_COMPOSITION,
        LAMBEK,
        LP,
        MODAL_APPLICATION,
        MODAL_ELIMINATION,
        MODAL_INTRODUCTION,
        NL,
        SCHEMA_REGISTRY,
        TENSOR_INTRODUCTION,
        TENSOR_PROJECTION,
        UNIT_ELIMINATION,
        UNIT_INTRODUCTION,
        BackwardApplication,
        BackwardComposition,
        BackwardCrossedComposition,
        BinaryRuleSchema,
        CommutativeBackwardApplication,
        CommutativeForwardApplication,
        ForwardApplication,
        ForwardComposition,
        ForwardCrossedComposition,
        GeneralizedComposition,
        LeftLifting,
        LeftProjection,
        LeftUnitElimination,
        ModalApplication,
        ModalInjection,
        ModalProjection,
        PatternBinarySchema,
        PatternUnarySchema,
        RightLifting,
        RightProjection,
        RightUnitElimination,
        RuleSchema,
        TensorIntroduction,
        UnaryRuleSchema,
        UnionSchema,
        UnitCoercion,
        WeightedSchema,
        generalized_composition,
    )
    from quivers.stochastic.semiring import (
        BOOLEAN,
        COUNTING,
        LOG_PROB,
        VITERBI,
        BooleanSemiring,
        ChartSemiring,
        CountingSemiring,
        LogProbSemiring,
        ViterbiSemiring,
    )
    from quivers.stochastic.span import (
        BinarySpanDeduction,
        CKYSchedule,
        LexicalAxiom,
        SpanChart,
        SpanGoal,
        UnarySpanDeduction,
    )
    from quivers.stochastic.transforms import (
        ConditionedMorphism,
        FactoredMorphism,
        MixtureMorphism,
        NormalizedMorphism,
        condition,
        factor,
        mix,
        normalize,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "MarkovAlgebra": "quivers.core.algebras",
    "MARKOV": "quivers.core.algebras",
    "StochasticMorphism": "quivers.stochastic.morphisms",
    "CategoricalMorphism": "quivers.stochastic.morphisms",
    "stochastic": "quivers.stochastic.morphisms",
    "DiscretizedNormal": "quivers.stochastic.families",
    "DiscretizedLogitNormal": "quivers.stochastic.families",
    "DiscretizedBeta": "quivers.stochastic.families",
    "DiscretizedTruncatedNormal": "quivers.stochastic.families",
    "ConditionedMorphism": "quivers.stochastic.transforms",
    "condition": "quivers.stochastic.transforms",
    "MixtureMorphism": "quivers.stochastic.transforms",
    "mix": "quivers.stochastic.transforms",
    "FactoredMorphism": "quivers.stochastic.transforms",
    "factor": "quivers.stochastic.transforms",
    "NormalizedMorphism": "quivers.stochastic.transforms",
    "normalize": "quivers.stochastic.transforms",
    "prob": "quivers.stochastic.queries",
    "marginal_prob": "quivers.stochastic.queries",
    "expectation": "quivers.stochastic.queries",
    "GiryMonad": "quivers.stochastic.giry",
    "FinStoch": "quivers.stochastic.giry",
    "Category": "quivers.stochastic.categories",
    "AtomicCategory": "quivers.stochastic.categories",
    "SlashCategory": "quivers.stochastic.categories",
    "ProductCategory": "quivers.stochastic.categories",
    "UnitCategory": "quivers.stochastic.categories",
    "ModalCategory": "quivers.stochastic.categories",
    "BUILTIN_CONSTRUCTOR_NAMES": "quivers.stochastic.categories",
    "CategorySystem": "quivers.stochastic.categories",
    "ChartSemiring": "quivers.stochastic.semiring",
    "LogProbSemiring": "quivers.stochastic.semiring",
    "ViterbiSemiring": "quivers.stochastic.semiring",
    "BooleanSemiring": "quivers.stochastic.semiring",
    "CountingSemiring": "quivers.stochastic.semiring",
    "LOG_PROB": "quivers.stochastic.semiring",
    "VITERBI": "quivers.stochastic.semiring",
    "BOOLEAN": "quivers.stochastic.semiring",
    "COUNTING": "quivers.stochastic.semiring",
    "RuleSchema": "quivers.stochastic.schema",
    "UnionSchema": "quivers.stochastic.schema",
    "WeightedSchema": "quivers.stochastic.schema",
    "BinaryRuleSchema": "quivers.stochastic.schema",
    "UnaryRuleSchema": "quivers.stochastic.schema",
    "ForwardApplication": "quivers.stochastic.schema",
    "BackwardApplication": "quivers.stochastic.schema",
    "ForwardComposition": "quivers.stochastic.schema",
    "BackwardComposition": "quivers.stochastic.schema",
    "ForwardCrossedComposition": "quivers.stochastic.schema",
    "BackwardCrossedComposition": "quivers.stochastic.schema",
    "CommutativeForwardApplication": "quivers.stochastic.schema",
    "CommutativeBackwardApplication": "quivers.stochastic.schema",
    "TensorIntroduction": "quivers.stochastic.schema",
    "LeftUnitElimination": "quivers.stochastic.schema",
    "RightUnitElimination": "quivers.stochastic.schema",
    "ModalApplication": "quivers.stochastic.schema",
    "RightLifting": "quivers.stochastic.schema",
    "LeftLifting": "quivers.stochastic.schema",
    "LeftProjection": "quivers.stochastic.schema",
    "RightProjection": "quivers.stochastic.schema",
    "UnitCoercion": "quivers.stochastic.schema",
    "ModalInjection": "quivers.stochastic.schema",
    "ModalProjection": "quivers.stochastic.schema",
    "GeneralizedComposition": "quivers.stochastic.schema",
    "PatternBinarySchema": "quivers.stochastic.schema",
    "PatternUnarySchema": "quivers.stochastic.schema",
    "EVALUATION": "quivers.stochastic.schema",
    "HARMONIC_COMPOSITION": "quivers.stochastic.schema",
    "CROSSED_COMPOSITION": "quivers.stochastic.schema",
    "COMMUTATIVE_EVALUATION": "quivers.stochastic.schema",
    "ADJUNCTION_UNITS": "quivers.stochastic.schema",
    "TENSOR_INTRODUCTION": "quivers.stochastic.schema",
    "TENSOR_PROJECTION": "quivers.stochastic.schema",
    "UNIT_INTRODUCTION": "quivers.stochastic.schema",
    "UNIT_ELIMINATION": "quivers.stochastic.schema",
    "MODAL_INTRODUCTION": "quivers.stochastic.schema",
    "MODAL_ELIMINATION": "quivers.stochastic.schema",
    "MODAL_APPLICATION": "quivers.stochastic.schema",
    "generalized_composition": "quivers.stochastic.schema",
    "CCG": "quivers.stochastic.schema",
    "LAMBEK": "quivers.stochastic.schema",
    "NL": "quivers.stochastic.schema",
    "LP": "quivers.stochastic.schema",
    "SCHEMA_REGISTRY": "quivers.stochastic.schema",
    "RuleSystem": "quivers.stochastic.rules",
    "ccg_rules": "quivers.stochastic.rules",
    "lambek_rules": "quivers.stochastic.rules",
    "custom_rules": "quivers.stochastic.rules",
    "LexicalAxiom": "quivers.stochastic.span",
    "SpanChart": "quivers.stochastic.span",
    "BinarySpanDeduction": "quivers.stochastic.span",
    "UnarySpanDeduction": "quivers.stochastic.span",
    "SpanGoal": "quivers.stochastic.span",
    "CKYSchedule": "quivers.stochastic.span",
    "ChartParser": "quivers.stochastic.parsers",
    "CCGParser": "quivers.stochastic.ccg",
    "LambekParser": "quivers.stochastic.lambek",
    "InsideAlgorithm": "quivers.stochastic.inside",
    "Item": "quivers.stochastic.agenda",
    "Wildcard": "quivers.stochastic.agenda",
    "make_wildcard": "quivers.stochastic.agenda",
    "Pattern": "quivers.stochastic.agenda",
    "Bindings": "quivers.stochastic.agenda",
    "InferenceRule": "quivers.stochastic.agenda",
    "instantiate": "quivers.stochastic.agenda",
    "match": "quivers.stochastic.agenda",
    "Chart": "quivers.stochastic.agenda",
    "HashChart": "quivers.stochastic.agenda",
    "ChartView": "quivers.stochastic.agenda",
    "CorpusChart": "quivers.stochastic.agenda",
    "Agenda": "quivers.stochastic.agenda",
    "FIFOAgenda": "quivers.stochastic.agenda",
    "LIFOAgenda": "quivers.stochastic.agenda",
    "PriorityQueueAgenda": "quivers.stochastic.agenda",
    "AgendaResult": "quivers.stochastic.agenda",
    "run_agenda": "quivers.stochastic.agenda",
    "DeductionSystem": "quivers.stochastic.agenda",
    "cky_agenda": "quivers.stochastic.agenda",
    "earley_agenda": "quivers.stochastic.agenda",
    "viterbi_agenda": "quivers.stochastic.agenda",
    "astar_agenda": "quivers.stochastic.agenda",
    "knuth_agenda": "quivers.stochastic.agenda",
    "depth_first_agenda": "quivers.stochastic.agenda",
    "semi_naive_agenda": "quivers.stochastic.agenda",
    "Production": "quivers.stochastic.productions",
    "ProductionGroup": "quivers.stochastic.productions",
    "ProductionKey": "quivers.stochastic.productions",
    "ProductionWeights": "quivers.stochastic.productions",
    "class_directed_lifts": "quivers.stochastic.effect_lifts",
    "make_swap_schema": "quivers.stochastic.effect_lifts",
    "swap_rule_set": "quivers.stochastic.effect_lifts",
    "lift_rule_set": "quivers.stochastic.effect_lifts",
}

type _LazyExport = (
    type
    | FunctionType
    | GenericAlias
    | TypeAliasType
    | MarkovAlgebra
    | ChartSemiring
    | RuleSchema
    | frozenset[str]
    | dict[str, RuleSchema]
)


def __getattr__(name: str) -> _LazyExport:
    """Resolve a public export from its defining module.

    Category definitions can be imported by core objects without loading
    inference, compilers, or chart parsers through this package facade.

    Parameters
    ----------
    name : str
        The requested public name.

    Returns
    -------
    _LazyExport
        The defining module's class, function, constant, or type alias.

    Raises
    ------
    AttributeError
        If the name is not exported by this package.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: _LazyExport = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include deferred public exports in module introspection.

    Returns
    -------
    list[str]
        Sorted module attributes and public names.
    """
    return sorted(set(globals()) | set(__all__))


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
    "CorpusChart",
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
