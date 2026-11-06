# Conceptual Guides

These guides explain the mathematical and computational foundations of
Quivers by feature area. Familiarity with Python values, arrays, and functions
is enough to begin; categorical and probabilistic terms are introduced where
they are used. For complete worked
examples, see the [tutorials](../tutorials/index.md); for formal semantics, see
[Semantics](../semantics/index.md).

For declaration-by-declaration syntax and operational behavior, use the
[QVR language reference](../reference/qvr/index.md). The guides explain why a
feature exists and how it relates to the rest of Quivers; the reference states
what source forms are accepted and how they run.

The sections below group guides by task and mathematical topic.

## Foundations

If these subjects are new, start with [Concepts](../getting-started/concepts.md),
then read Core Types and Morphisms before the more advanced categorical
pages. If your immediate goal is a statistical analysis, start with
[Data and Formulas](analysis-data-and-formulas.md) and
[Inference Foundations](inference-foundations.md). Readers already familiar
with the mathematics can use the formal definitions and API references
on each page, with the [semantics](../semantics/index.md) for assumptions
and derivations.


These guides introduce the categorical primitives used throughout the library.

1. [Core Types & Algebras](core.md). Finite sets (the SetObject
   hierarchy), algebras as enrichment algebras, and the algebraic
   primitives that underpin all morphism composition.
2. [Transformations & Composition Rules](transformations.md).
   First-class change-of-base transformations, the `>>>`
   composition operator, and the
   `CompositionRule → Semigroupoid → Algebra` hierarchy.
3. [Morphisms & Composition](morphisms.md). What a morphism is as
   a tensor in $\mathcal{V}^{|A| \times |B|}$, the morphism
   hierarchy, composition, tensor product, [marginalization](morphisms.md#marginalization), and
   the compact-closed operations (`dagger`, `trace`, `cup`, `cap`).

## DSL

Writing programs in the typed `.qvr` language.

4. [DSL Overview](dsl-overview.md). File format, grammar, doc
   comments, compilation pipeline, error handling.
5. [DSL Declarations](dsl-declarations.md). `composition`,
   `object`, `morphism` (with the `[role=...]` option block
   selecting `latent` / `observed` / `kernel` / `embed` /
   `discretize`), `bundle`, the fan / repeat / stack / scan /
   curry combinators, and `export`.
6. [DSL Programs and Let-Expressions](dsl-programs-and-lets.md).
   `program` blocks, bind / observe / marginalize / let steps, the
   axis-role clause, factor expressions, the let-expression
   primitive operators, and inline distributions.
7. [DSL Contractions](dsl-contractions.md). Operadic n-ary
   contractions, type-driven wiring inference, `share` clause,
   explicit `wiring` escape hatch.

## Probabilistic Programming

The runtime semantics of programs and distributions.

8. [Monadic Programs](programs.md). Probabilistic programming via
   sequential bind, `let`, `observe`, and `return` steps; ancestral
   sampling; log-joint computation.
9. [Hierarchical Programs](programs-hierarchical.md). Parametric
   templates for crossed random intercepts, monotone-spline
   coefficients, and grouped marginalization over fibred discrete
   latents.
10. [Continuous Spaces and Morphisms](continuous-spaces.md). The
    `ContinuousSpace` hierarchy, the
    [`ContinuousMorphism`](../api/continuous/morphisms.md)
    interface, sampled composition, the discrete / continuous
    boundary, normalizing flows.
11. [Continuous Families](continuous-families.md). The registry of more than forty
    parameterized distributions, event ranks, and the
    structured priors (MatrixNormal, InverseWishart, GP,
    Horseshoe, LKJ) that use the axis-role syntax.
12. [Stochastic Morphisms](stochastic.md). The FinStoch category:
    Markov kernels, [conditioning](inference-foundations.md#conditioning-on-observations), queries, the Giry monad.

## Inference

Fitting models to data.

13. [Inference Foundations](inference-foundations.md). The
    six-layer inference stack, trace and sample-site interface,
    conditioning.
14. [Variational Inference: Guides, Objectives, and
    SVI](inference-svi.md). The `Auto*Guide` family, ELBO / IWAEBound /
    Rényi / VR-IWAE objectives, gradient estimators, SVI loop,
    predictive sampling.
15. [MCMC and Hybrid
    Samplers](inference-mcmc.md). HMC and NUTS kernels, `AutoDAIS`,
    `WarmupThenHMC`, predictive sampling from MCMC chains.

## Categorical Structures

These guides cover higher-order categorical constructions used elsewhere in
the library.

16. [Categorical Structures](categorical.md). [Functors](categorical.md#functors), natural
    transformations, [adjunctions](categorical.md#adjunctions), [monoidal structures](categorical.md#monoidal-structures), base change.
17. [Monads & Comonads](monadic.md). Monadic abstractions, Kleisli
    / coKleisli categories, algebras, coalgebras, distributive
    laws.
18. [Enriched Category Theory](enriched.md). Ends, coends, Kan
    extensions, weighted limits, [profunctors](enriched.md#profunctors), [Yoneda](enriched.md#yoneda), Day
    convolution, optics.
19. [Compositional Effects](effects.md). Algebraic-effects
    framework over the residuated category universe.

## Structured Prediction

These guides cover weighted parsing and the structural-compression framework.

20. [Weighted Deduction Systems](deduction.md). Agenda-engine
    runtime, semirings, charts as differentiable values, the seven
    canonical parameters.
21. [Structural Compression: Signatures and
    Encoders](structural-compression-signatures-and-encoders.md).
    `signature` and `encoder` blocks; the F-algebra interface for
    compressing structured objects; factory form and
    sequence / graph sugar.
22. [Structural Compression: Decoders and
    Losses](structural-compression-decoders-and-losses.md).
    Kleisli-coalgebra `decoder` blocks, `loss` declarations,
    deduction and Bayesian integration.

## Analysis

These guides cover data preparation, model fitting, and diagnostics.

23. [Analysis: Data and Formulas](analysis-data-and-formulas.md).
    `DatasetSchema`, brms-style formulas, the formula-to-QVR
    compile lens, family registry.
24. [Analysis: Fitting and
    Diagnostics](analysis-fitting-and-diagnostics.md). One-line
    `fit`, ArviZ-based diagnostics, algebra-guided training tooling
    (`ChainShape`, `recommend_init`, `saturation_warnings`),
    autograd-safe morphism transforms.

## Quick navigation

- **Probabilistic programming.** [Core](core.md), [Stochastic](stochastic.md),
  [Monadic Programs](programs.md),
  [Inference Foundations](inference-foundations.md), [SVI](inference-svi.md).
- **Hybrid discrete-continuous models.**
  [Continuous Spaces](continuous-spaces.md),
  [Continuous Families](continuous-families.md),
  [Monadic Programs](programs.md),
  [Inference](inference-foundations.md).
- **Building models declaratively.** [DSL Overview](dsl-overview.md),
  [DSL Declarations](dsl-declarations.md),
  [DSL Programs](dsl-programs-and-lets.md),
  [QVR Language Reference](../reference/qvr/index.md),
  [Transformations](transformations.md),
  [Inference](inference-foundations.md).
- **Category-theoretic extension.**
  [Categorical Structures](categorical.md),
  [Monads & Comonads](monadic.md),
  [Enriched Category Theory](enriched.md).
- **Structured prediction and parsing.**
  [Weighted Deduction Systems](deduction.md),
  [Structural Compression: Signatures](structural-compression-signatures-and-encoders.md),
  [Structural Compression: Decoders](structural-compression-decoders-and-losses.md).
- **Working with fitted models.**
  [Inference Foundations](inference-foundations.md),
  [SVI](inference-svi.md), [MCMC](inference-mcmc.md),
  [Analysis: Data](analysis-data-and-formulas.md),
  [Analysis: Fitting](analysis-fitting-and-diagnostics.md).
