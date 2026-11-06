# Tutorials

Quivers provides two ways to work with a model: a `.qvr` language for writing probabilistic programs and a typed Python API for building on the category-theoretic core. The tutorials follow these two tracks. Choose the one that matches your task.

For a short route, read the [conceptual introduction](../getting-started/concepts.md),
QVR chapters 1–3, and the diagnostics chapter after fitting a model. Read
Python chapters 1–2 when you want to understand composition as table operations.
Indexed families, handlers, and structural models can wait until your task
needs them; their tutorials introduce their own vocabulary.

## QVR DSL track

Start here to write a statistical model, whether or not you have used another probabilistic programming language. QVR is the primary user-facing language: you declare types, write a `program` block whose steps look like `v <- Normal(loc=0, scale=1)` or `observe y <- Bernoulli(probs=p)`, and [fit it with SVI or NUTS](qvr/06-inference-zoo.md#step-1-vi-mcmc-or-hybrid). The categorical structure informs the implementation but need not appear in the model.

The early chapters include comparisons with Pyro, NumPyro, Stan, or PyMC. Those tabs are optional: the QVR explanation stands on its own.

1. **[Your first model](qvr/01-first-model.md)**. Linear regression. Side-by-side with PyMC. SVI fit, posterior predictive check, point summaries.
2. **[Generalized linear models](qvr/02-glms.md)**. Logistic and Poisson regression. Link functions. Posterior calibration plots.
3. **[Hierarchical models](qvr/03-hierarchical.md)**. Random intercepts; the eight-schools model; centered vs non-centered parameterizations; running NUTS with diagnostics.
4. **[Mixtures and discrete latents](qvr/04-marginalize.md)**. Finite mixtures and HMM-shaped models via the `marginalize` block: QVR's typed-scope marginalization, the feature that distinguishes it most from Pyro/NumPyro.
5. **[Sequence models](qvr/05-time-series.md)**. Plates, `scan`, and weighted deduction for chart-shaped models. State-space models and HMMs.
6. **[Choosing an inference algorithm](qvr/06-inference-zoo.md)**. A decision tree over eleven concrete guide classes, seven objectives, two MCMC kernels, and two hybrid approaches. Which combination matches which model shape.
7. **[Categorical foundations](qvr/07-categorical.md)**. *(Optional reading.)* Algebras as enrichment algebras, `>>` as enriched composition, and change-of-base as a functor. This chapter supplies the background needed to extend the library or interpret its algebra type errors.
8. **[Diagnostics and model comparison](qvr/08-diagnostics-and-comparison.md)**. ArviZ end-to-end: `to_datatree`, convergence diagnostics, posterior-predictive checks, PSIS-LOO model comparison.
9. **[Indexed data and total functions](qvr/09-indexed-data.md)**. Closed index sorts, indexed families, constructor refinement, recursive case analysis, and direct entry execution.
10. **[Effects and handlers](qvr/10-effects-and-handlers.md)**. Lexical instances, open rows, authored handlers, resumption grades, and stable execution traces.
11. **[Parsing as an effectful computation](qvr/11-parsing-and-search.md)**. Schema-backed parsers, generated search and weight effects, reference inside scores, and parser calls from probabilistic programs.
12. **[Structural autoencoders](qvr/12-structural-autoencoders.md)**. Signatures, encoder and decoder attachments, checked loss invocation, autograd, and checkpoint registration.
13. **[Check, inspect, and transpile](qvr/13-release-workflow.md)**. Target-aware checking, entry inspection, runtime traces, LSP configuration, grammar provenance, and release automation.
14. **[Exact Bayes with finite collection expressions](qvr/14-collection-programs.md)**. Pure `map`, effect-aware `traverse`, finite folds, and stable log-space normalization in a grid posterior.

You can read the first six chapters without touching category theory. Chapter 7 is the bridge to the Python API track; chapter 8 covers the Bayesian-analysis workflow once a fit is in hand. Chapters 9–14 cover indexed types, effects, generated search and neural computation graphs, the release boundary, and finite collection programs.

## Python API track

The Python API gives you direct access to the typed categorical layer: `FinSet`, `Morphism`, `Algebra`, `MonadicProgram`, inference primitives, and structural-compression components. Use this track to build tooling on top of Quivers, extend its categorical implementation, or inspect what the DSL compiles into.

1. **[Your first quiver](python/01-first-quiver.md)**. `FinSet` objects, observed and latent morphisms, the `>>` composition operator, `Program`.
2. **[Stochastic relations](python/02-stochastic-relations.md)**. Markov kernels and the FinStoch category. Conditioning, marginalization, expectation queries.
3. **[Probabilistic programs](python/03-probabilistic-programs.md)**. `MonadicProgram` by hand: continuous spaces, conditional families, bind (`<-`) / `let` / `observe` steps, sampling, and joint trace scoring.
4. **[Fuzzy logic factorization](python/04-fuzzy-factorization.md)**. Factorizing an observed fuzzy relation into a composition of latents, training under product-fuzzy noisy-OR composition.
5. **[Variational inference](python/05-variational-inference.md)**. `Guide` + `Objective` + `SVI` + `Predictive`. Setting up the full inference loop end-to-end.
6. **[First-class transformations](python/06-first-class-trans.md)**. `MorphismTransformation` and `AlgebraHomomorphism` as values: let-binding, the `>>>` composition operator, change-of-base pipelines.
7. **[Composition rules beyond algebras](python/07-composition-rules.md)**. The `CompositionRule → Semigroupoid → Algebra` hierarchy, `BilinearForm`, and the operadic `EinsumWiring` API for n-ary contractions.
8. **[Analysis pipelines](python/08-analysis-pipelines.md)**. Formula → fit → diagnostics: brms-style `fit("y ~ x + (1|g)", data=df, ...)`, the emitted `.qvr` source, SVI on a hierarchical model, NUTS + PSIS-LOO model comparison, and ArviZ `DataTree` posterior-predictive checks.
9. **[Debugging quivers programs](python/09-debugging.md)**. Reading `CompileError`, inspecting a compiled `Program`, tracing intermediate values, watching SVI gradients, and using NUTS diagnostics to find the root cause of a misbehaving fit.

## Prerequisites

For the QVR track:

- Python 3.14+, PyTorch 2.0+, quivers installed ([Installation](../getting-started/installation.md)).
- Basic Python and familiarity with arrays. The [conceptual introduction](../getting-started/concepts.md) explains prior, likelihood, posterior, sampling, and composition; no previous PPL or category theory is required.
- For chapters 9–13, install Quivers from one source and keep the package, QVR grammar, reference machine, and language server at the same release.

For the Python API track:

- Python and PyTorch as above.
- Familiarity with tensor shapes and functions. Chapter 1 introduces objects, morphisms, and composition through tables. The [conceptual introduction](../getting-started/concepts.md) gives additional preparation; the [semantics](../semantics/index.md) retains the formal account for readers who want its assumptions and proofs.

## How to read

Each chapter:

- States the model or feature in plain English first.
- Shows complete runnable code.
- Calls out what changed from the previous chapter.
- Ends with a "try this" exercise and pointers to the next chapter.

The chapters are independent enough that you can skim or skip, but each builds on the previous one's vocabulary, so out-of-order reading is easier in the second half.
