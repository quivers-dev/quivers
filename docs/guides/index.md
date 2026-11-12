# Conceptual Guides

Guides to each feature area of quivers, with the mathematics each feature relies on.
The [QVR language reference](../reference/qvr/index.md) specifies each source form,
[Semantics](../semantics/index.md) gives the formal denotations, and the
[tutorials](../tutorials/index.md) build complete models.

## Foundations

1. [Core Types & Algebras](core.md): finite objects, and the `Algebra` that fixes how
   scores combine in composition.
2. [Transformations & Composition Rules](transformations.md): converting a morphism's
   scores between algebras, the levels of composition rules, and contractions of
   several morphisms.
3. [Morphisms & Composition](morphisms.md): morphisms as score tables, composition,
   parallel product, marginalization, transpose, and trace.

## The QVR language

4. [DSL Overview](dsl-overview.md): the file format, the compilation pipeline, the
   grammar, and comments.
5. [DSL Declarations](dsl-declarations.md): objects, morphisms and their roles,
   kernels, learned functions, deductions, and the combinators `fan`, `repeat`,
   `stack`, and `scan`.
6. [DSL Programs and Let-Expressions](dsl-programs-and-lets.md): program steps, the
   `over` clause, primitives, factor expressions, and inline distributions.
7. [DSL Contractions](dsl-contractions.md): contractions of several morphisms, with the
   wiring inferred from their types or given as an einsum string.

## Probabilistic programming

8. [Probabilistic Programs](programs.md): running, sampling, and scoring a compiled program,
   and building one in Python.
9. [Hierarchical Programs](programs-hierarchical.md): templates for crossed random
   effects, and marginalizing a discrete latent once per group.
10. [Continuous Spaces and Morphisms](continuous-spaces.md): continuous spaces,
    conditional distributions between them, and normalizing flows.
11. [Continuous Families](continuous-families.md): the registered distribution families,
    their event dimensions, and priors on matrices.
12. [Stochastic Morphisms](stochastic.md): Markov kernels between finite sets,
    conditioning, Bayesian inversion, and the Giry monad.

## Inference

13. [Inference Foundations](inference-foundations.md): traces, sample sites, the batch
    axis of the input, data inputs, and conditioning.
14. [Variational Inference: Guides, Objectives, and SVI](inference-svi.md): guides,
    objectives, gradient estimators, the training loop, and predictive sampling.
15. [MCMC and Hybrid Samplers](inference-mcmc.md): Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS), composed kernels,
    tempered sequential Monte Carlo, and convergence diagnostics.

## Categorical structures

16. [Categorical Structures](categorical.md): functors, natural transformations,
    adjunctions, monoidal products, and traces as library classes.
17. [Monads & Comonads](monadic.md): the monad and comonad classes, Kleisli categories,
    and algebras for a monad.
18. [Enriched Category Theory](enriched.md): ends and coends, the Yoneda lemma, Kan
    extensions, weighted limits, profunctors, Day convolution, and optics on finite
    tensors.
19. [Compositional Effects](effects.md): type classes for effect-driven
    interpretation, lifting grammar rules through effects, and effect handlers.

## Structured prediction

20. [Weighted Deduction Systems](deduction.md): parsing as weighted deduction,
    differentiable charts, and probabilistic grammars with priors.
21. [Tree and Graph Autoencoders: Signatures and Encoders](structural-compression-signatures-and-encoders.md):
    signatures of terms and encoders that fold a term into a vector.
22. [Tree and Graph Autoencoders: Decoders and Losses](structural-compression-decoders-and-losses.md):
    decoders that score and sample terms, and training losses.

## Analysis

23. [Analysis: Data and Formulas](analysis-data-and-formulas.md): dataframes and
    regression formulas compiled to QVR.
24. [Analysis: Fitting and Diagnostics](analysis-fitting-and-diagnostics.md): one-call
    fitting, ArviZ diagnostics, convergence checks, simulation-based calibration, and
    initialization tools.

## Tools

25. [Interactive: REPL and Language Server](repl-and-lsp.md): the REPL, the Jupyter
    kernel, and the editor language server.
26. [Migrating `.qvr` Source Between Grammar Revisions](migration.md): `qvr migrate`.
