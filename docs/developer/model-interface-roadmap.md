# Model interface requirements and capability audit

This audit distinguishes three questions: (i) whether QVR can state a model;
(ii) whether an execution path can fit it; and (iii) whether the Python interface
provides the corresponding data preparation and analysis workflow. Expressibility
alone does not establish parity with another modeling system.

The implementation supplies checked input schemas, explicit runtime parameters,
finite extent inference, reusable inference methods, new-data prediction, and raw
pointwise observation likelihoods. The tables below record
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
| Common regression families | `quivers.formulas.families` contains 26 entries, with 15 inverse links and mean/precision conversions for binomial, beta-binomial, gamma, Weibull, and other families. Asymmetric Laplace quantile regression supports a fixed or regressed quantile, with independent SciPy density/CDF/quantile and sampling tests. `tests/test_formula_distributional.py` checks independent likelihood and derivative oracles. ExGaussian uses the response mean and Gaussian/exponential components; inverse Gaussian uses response mean/shape with implicit sample gradients. These have independent density, CDF, derivative, and sampling oracles. Zero-inflated Beta regression reuses the native zero/one mixture with its one-atom probability fixed at zero; atom/interior densities, endpoint derivatives, sampling moments, and response means have independent oracles. Hurdle Gamma and LogNormal responses use a shared exact atom/positive-component implementation, with independent density, derivative, CDF, moment, and sampling oracles and SVI/HMC/NUTS formula checks. The larger native family registry remains a separate interface; the remaining brms families and their interactions need implementation and tests. |
| Group effects | Single-bar blocks use native LKJ Cholesky correlations; double bars request independence. Shared IDs couple coefficients across location and auxiliary formulas. Scalar responses support centered and noncentered coordinates; categorical blocks couple all non-reference-category coefficients in noncentered coordinates. `tests/test_formula_correlations.py` checks Gaussian covariance densities, derivatives, and SVI/HMC/NUTS execution. Supplied group covariance, multi-membership, repeated coefficients across blocks, and covariance-aware prediction remain work. |
| Distributional and nonlinear formulas | `distributional={"sigma": "~ z + (1 | g)"}` supplies evaluated auxiliary predictors and parameter-specific links in the formula interface. Design columns and group coding are retained, and shared correlation IDs can span predictors. A nonlinear parameter dependency graph remains work. |
| Multivariate responses | QVR supports multiple observation sites. Joint formula construction, residual correlations, response selection, and aligned output coordinates need a common interface. |
| Splines, Gaussian processes, and special predictor terms | Some corresponding kernels exist. Smooth construction, monotonic effects, category-specific effects, multi-membership, and group-level term semantics require formula and prediction tests. |
| Dependence structures | QVR has scans and state-space examples. brms's temporal, spatial, and supplied covariance structures require individually verified parameterizations and prediction behavior. |
| Additional response information | Formula `offset(expr)` and `exposure(expr)` terms enter location and auxiliary predictors with coefficient one, preserve evaluated arrays, and agree with independent density, gradient, and posterior-prediction oracles in `tests/test_formula_offsets.py`. Binomial and beta-binomial trials support constants or per-row columns. Formula `weights="column"` retains typed row weights and scales only observation scores; independent prior/likelihood gradient, exposure-interaction, and fixed-coefficient prediction oracles are in `tests/test_formula_weights.py`. Censoring, truncation, and their interactions remain work; likelihood weights are distinct from replication. |
| Missing data and measurement error | Explicit latent-variable models are possible. Automatic imputation formulas, measurement-error preparation, multiple datasets, and their prediction semantics remain work. |
| Priors | `get_prior` inspects the exact overrideable priors in the formula AST, including column autoscaling, auxiliary coefficients, and shared correlations. Unused overrides fail before fitting; `fixed_parameters` removes known auxiliary parameters from the latent and prior layouts. Class-based selectors, regularized horseshoe and R2D2 workflows, and prior-only fitting need implementations and acceptance tests. |
| Sampling and variational fitting | Existing MCMC and variational components remain configurable through reusable methods. Every algorithm's objective, stopping condition, initialization, and checkpoint semantics stay visible. This does not establish equivalence to Stan's samplers. |
| Prediction | `BayesianFit.posterior_linpred` and `posterior_epred` replay native predictors and response distributions on training rows, with explicit class labels and binomial count expectations. `posterior_predict` samples native responses with independent row noise, retained chain axes, and reproducible separate latent and response random streams. `ModelFit.predict` validates new covariates and can infer a new row count while retaining the fitted latent layout. The brms [prediction contract](https://paulbuerkner.com/brms/reference/posterior_predict.brmsfit.html) additionally requires retained factor coding, new group levels, selected effects and responses, and several predictive quantities. |
| Parameter summaries and diagnostics | `BayesianFit.draws` preserves chain/draw axes; `fixed_effects`, `random_effects`, and `coefficients` reconstruct native population/group coefficients with original labels and retained level coding. `tests/test_formula_posterior.py` checks centered, noncentered, shared, and categorical effects. Covariance summaries and a common summary/diagnostic interface remain work. |
| Model comparison | `ModelFit.log_likelihood` extracts raw pointwise observation densities, and QVR provides ArviZ adapters and comparison functions. Marginal-likelihood observers, observation coordinates, integrated LOO and WAIC, exact refits, K-fold validation, and model weighting require further work. A whole-program joint score is insufficient. |
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
2. Extend the distributional formula interface with nonlinear and multivariate
   dependency graphs, supplied group covariances, and the remaining families and
   response modifiers, each with numerical oracles.
3. Extend new-data prediction and raw pointwise likelihoods with retained encodings, marginal-likelihood observers, comparisons, and persistence.
4. Connect the structured executors as their issues land, retaining each solver's
   objective, approximation, termination, and differentiation contract.

The useful open questions are concrete: which observer must a structured result
support, which dimensions may change at prediction time, and which cached objects
remain valid after parameters change? Each answer should be expressed in a typed
contract and tested before a new executor or formula term is advertised.
