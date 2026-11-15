# Model interface requirements and capability audit

This audit distinguishes three questions: (i) whether QVR can state a model;
(ii) whether an execution path can fit it; and (iii) whether the Python interface
provides the corresponding data preparation and analysis workflow. Expressibility
alone does not establish parity with another modeling system.

The implementation supplies checked input schemas, explicit external declarations,
finite extent inference, and reusable inference methods. The tables below record
remaining work against brms and the open enhancement issues as of November 15, 2026.

## brms coverage

The comparison uses the published brms 2.23.2
[reference index](https://paulbuerkner.com/brms/reference/index.html),
[formula specification](https://paulbuerkner.com/brms/reference/brmsformula.html),
[family catalog](https://paulbuerkner.com/brms/reference/brmsfamily.html), and
[fitting interface](https://paulbuerkner.com/brms/reference/brm.html).
These sources establish the comparison's scope; the QVR assessments below come
from the implementation and its tests. Full brms coverage is not established.

| Capability | QVR evidence and remaining requirements |
| --- | --- |
| Data dimensions and external inputs | `Model.schema` retains checked types and roles; `Model.bind` specializes finite extents and validates supplied values. Tested in `tests/modeling/test_model.py`. |
| Common regression families | `quivers.formulas.families` contains 14 entries. The larger QVR family registry is a separate interface. Test every brms family, link, support, and parameterization before claiming family parity. |
| Group effects | Formula compilation provides independent varying coefficients. Correlated coefficient blocks, shared correlation IDs, and covariance-aware prediction still require implementation and numerical checks. |
| Distributional and nonlinear formulas | QVR can express predictors directly. The formula interface needs separate predictors and links for auxiliary parameters, a nonlinear parameter dependency graph, and prior discovery. |
| Multivariate responses | QVR supports multiple observation sites. Joint formula construction, residual correlations, response selection, and aligned output coordinates need a common interface. |
| Splines, Gaussian processes, and special predictor terms | Some corresponding kernels exist. Smooth construction, monotonic effects, category-specific effects, multi-membership, and group-level term semantics require formula and prediction tests. |
| Dependence structures | QVR has scans and state-space examples. brms's temporal, spatial, and supplied covariance structures require individually verified parameterizations and prediction behavior. |
| Additional response information | Some QVR programs express weighting, censoring, truncation, or trials directly. A uniform formula and data contract must cover every supported response addition and distinguish likelihood weights from replication. |
| Missing data and measurement error | Explicit latent-variable models are possible. Automatic imputation formulas, measurement-error preparation, multiple datasets, and their prediction semantics remain work. |
| Priors | QVR has explicit sample sites and formula prior overrides. Class-based selectors, prior inspection, regularized horseshoe and R2D2 workflows, and prior-only fitting need matching acceptance tests. |
| Sampling and variational fitting | Existing MCMC and variational components remain configurable through reusable methods. Every algorithm's objective, stopping condition, initialization, and checkpoint semantics stay visible. This does not establish equivalence to Stan's samplers. |
| Prediction | `ModelFit.predict` releases observations at the fitted layout. The brms [prediction contract](https://paulbuerkner.com/brms/reference/posterior_predict.brmsfit.html) also requires new-data validation, retained factor coding, new group levels, selected effects and responses, and several predictive quantities. |
| Parameter summaries and diagnostics | The wrapper reuses QVR's MCMC summaries. Regression coefficient reconstruction, varying-effect extraction, covariance summaries, and fitted-value summaries require formula-aware metadata. |
| Model comparison | QVR provides ArviZ adapters and comparison functions. The model wrapper still needs pointwise likelihoods, observation coordinates, LOO and WAIC integration, exact refits, K-fold validation, and model weighting. A whole-program joint score is insufficient. |
| Other post-processing | Hypothesis tests, Bayes factors, marginal likelihoods, R-squared, conditional effects, model updates, and persistent fitted artifacts require their own implementations and tests. |
| Custom families and backends | Existing family and target registries are extension points. Custom scoring, sampling, prediction, and emitted-target support must each be checked; a registered name proves none of these by itself. |

Full coverage requires an executable fixture for every supported feature and
interaction, an independent density or prediction oracle, and a stated tolerance.
For instance, comparing correlated effects requires checking the covariance
model; matching a formula's printed text would not establish the same prior.

## Open enhancement issues

The table covers every open enhancement returned by the repository issue listing
on the audit date. It records requirements on the interface without asserting
that the proposed executors or distributions have been implemented.

| Issue | Requirement on model and fitting interfaces |
| --- | --- |
| [#232](https://github.com/quivers-dev/quivers/issues/232) | Preserve a checked entry independently of its tensor executor. The built-in methods refuse unsupported `CheckedProgram` fitting. Future tensor execution must guard changes in a run's site structure. |
| [#333](https://github.com/quivers-dev/quivers/issues/333) | Retain typed carriers, ordered rules, contexts, substitutions, and parameter ownership. `ModelSchema` preserves QIEC type objects; a rendered type string is only a display. Rule presentations and their adapters remain separate work. |
| [#334](https://github.com/quivers-dev/quivers/issues/334) | Keep computed structured inputs and results in QIEC. `InferenceMethod[Result]` returns the method's own type, avoiding an obligatory tensor or posterior-draw conversion. Providers and typed composition still need integration tests. |
| [#335](https://github.com/quivers-dev/quivers/issues/335) | Keep sampling, scoring, enumeration, normalization, and semantic outcome aggregation as distinct capabilities. Binding retains live values and gradients; it does not claim structured distribution adapters or exact semantic marginalization. |
| [#336](https://github.com/quivers-dev/quivers/issues/336) | A solver must check its algebraic laws, observer requirements, recursion contract, and resource limits. A generic `run(bound)` entry point supplies no exactness or termination guarantee. |
| [#337](https://github.com/quivers-dev/quivers/issues/337) | Sparse finite-state data and lazy relations need typed provider-owned representations. The binding maps retain object identity, though current validation supports tensor and string inputs. Provider validators must be integrated before sparse structured inputs can be bound. |
| [#338](https://github.com/quivers-dev/quivers/issues/338) | Preserve differentiable scores, topology ownership, and parameter identity. Path sums, decoding, expected statistics, and derivatives of a solver need separate result contracts and enumeration oracles. |
| [#339](https://github.com/quivers-dev/quivers/issues/339) | Permit message results with explicit family, normalization scale, approximation status, schedule, and stopping reason. Such results can be returned by a method without pretending they are samples. |
| [#340](https://github.com/quivers-dev/quivers/issues/340) | MAP results must identify their objective, assignment, primal and dual bounds, agreement, and certification conditions. A custom method's certificate is retained unchanged; this interface does not certify it. |
| [#341](https://github.com/quivers-dev/quivers/issues/341) | EP methods need their projection direction, message family, penalty, damping, feature complexity, and local and outer convergence outcomes. Convergence of messages must not be reported as exact marginals. |
| [#342](https://github.com/quivers-dev/quivers/issues/342) | Preserve the roadmap's separation of model implementation, executor capability, approximation, and experimental reproduction. The four proposed model deliverables need independent executable acceptance programs. |

## Sequencing further work

1. Finish typed structured binding, explicit executor capability checks, and
   regression artifacts with persistent design and group coding.
2. Add distributional and multivariate formula graphs, correlated varying effects,
   and the remaining families and response modifiers, each with numerical oracles.
3. Add prediction on new data, pointwise likelihoods, comparisons, and persistence.
4. Connect the structured executors as their issues land, retaining each solver's
   objective, approximation, termination, and differentiation contract.

The useful open questions are concrete: which observer must a structured result
support, which dimensions may change at prediction time, and which cached objects
remain valid after parameters change? Each answer should be expressed in a typed
contract and tested before a new executor or formula term is advertised.
