# Tutorials

Two tracks. The QVR track writes and fits statistical models in [QVR](../reference/qvr/index.md), the modeling language of quivers. The Python track builds the same objects from the Python classes that QVR compiles to: finite sets, morphisms, algebras, and probabilistic programs. Prerequisites: Python 3.14 or later, PyTorch, and quivers ([installation](../getting-started/installation.md)).

## QVR track

1. **[Your first model](qvr/01-first-model.md)**. Bayesian linear regression beside PyMC and NumPyro, effect signatures, data inputs, an SVI fit, and posterior draws.
2. **[Generalized linear models](qvr/02-glms.md)**. Logistic and Poisson regression, inverse link functions, and the calibration of the posterior mean probability.
3. **[Hierarchical models](qvr/03-hierarchical.md)**. Eight schools, plates, centered and non-centered parameterizations, and NUTS.
4. **[Mixtures and discrete latents](qvr/04-marginalize.md)**. Summing a discrete latent variable out with `marginalize`, for each row or for each group of rows.
5. **[Sequence models](qvr/05-time-series.md)**. `scan`, recurrent networks, hidden Markov models summed by the forward algorithm, and state space models.
6. **[Choosing an inference algorithm](qvr/06-inference-zoo.md)**. Guides, variational objectives, gradient estimators, MCMC kernels, and hybrids, with the literature for each.
7. **[Categorical foundations](qvr/07-categorical.md)**. Morphisms as matrices over a semiring or t-norm, the laws of each composition rule, change of base, and programs as Markov kernels.
8. **[Diagnostics and model comparison](qvr/08-diagnostics-and-comparison.md)**. Convergence diagnostics, posterior predictive checks, and PSIS-LOO comparison through ArviZ.
9. **[Indexed data and total functions](qvr/09-indexed-data.md)**. Inductive families, parameters and indices, and case analysis with a motive.
10. **[Effects and handlers](qvr/10-effects-and-handlers.md)**. Effect instances, effect rows, authored and foreign handlers, and resumption grades.
11. **[Parsing as an effectful computation](qvr/11-parsing-and-search.md)**. A categorial grammar compiled to a chart parser and to a search computation, and parsing inside a probabilistic program.
12. **[Structural autoencoders](qvr/12-structural-autoencoders.md)**. Tree autoencoders over typed terms, attachments, and gradients through the reference machine.
13. **[Check, inspect, and transpile](qvr/13-release-workflow.md)**. Target checks, entry points, traces, the language server, and transpiling before a release.
14. **[Exact Bayes with finite collection expressions](qvr/14-collection-programs.md)**. `traverse`, `map`, and `logsumexp_over` in a posterior computed by enumeration.

Chapters 1 to 6 and 8 need no category theory.

## Python track

1. **[Your first quiver](python/01-first-quiver.md)**. `FinSet`, learnable and observed morphisms, composition with `>>`, and `Program`.
2. **[Stochastic relations](python/02-stochastic-relations.md)**. Markov kernels on finite sets, discretized families, conditioning, and queries.
3. **[Probabilistic programs](python/03-probabilistic-programs.md)**. `MonadicProgram` built from `Draw` and `Let` steps, sampling, and `log_joint`.
4. **[Fuzzy logic factorization](python/04-fuzzy-factorization.md)**. Factorizing a relation under the product t-norm and noisy-OR.
5. **[Variational inference](python/05-variational-inference.md)**. Traces, conditioning, `Guide`, `ELBO`, `SVI`, and `Predictive`.
6. **[First-class transformations](python/06-first-class-trans.md)**. Entrywise maps and morphism transformations as values, `compose_trans`, and Bayesian inversion.
7. **[Composition rules beyond algebras](python/07-composition-rules.md)**. `Algebra`, `Semigroupoid`, and `BilinearForm` by the laws they claim, and `EinsumWiring` for contractions of several tensors.
8. **[Analysis pipelines](python/08-analysis-pipelines.md)**. Regression formulas compiled to QVR, hierarchical, ordinal, categorical, and mixture regressions, PSIS-LOO, and posterior predictive checks.
9. **[Debugging quivers programs](python/09-debugging.md)**. `CompileError`, compiled steps, traces, the SVI loss, and NUTS diagnostics.
