# Model interface requirements and capability audit

This audit distinguishes three questions: (i) whether QVR can state a model;
(ii) whether an execution path can fit it; and (iii) whether the Python interface
provides the corresponding data preparation and analysis workflow. Expressibility
alone does not establish parity with another modeling system.

The implementation supplies checked input schemas, explicit runtime parameters,
finite extent inference, reusable inference methods, new-data prediction, and raw
pointwise observation likelihoods. The tables below record
remaining work against brms and the open enhancement issues as of November 16, 2026.

## brms coverage

The released comparison baseline is
[brms 2.23.0 on CRAN](https://CRAN.R-project.org/package=brms), published
September 9, 2025, with its
[source metadata pinned at `b124949c`](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/DESCRIPTION).
The audit also checks the official development snapshot
[`f4f906dc`, dated November 12, 2026](https://github.com/paul-buerkner/brms/commit/f4f906dc0db56fd42b58cdd7d57e76b40f650331),
whose [metadata identifies version 2.23.2](https://github.com/paul-buerkner/brms/blob/f4f906dc0db56fd42b58cdd7d57e76b40f650331/DESCRIPTION).
That development version is distinct from the CRAN release. The website's
[reference index](https://paulbuerkner.com/brms/reference/index.html),
[formula specification](https://paulbuerkner.com/brms/reference/brmsformula.html),
[family catalog](https://paulbuerkner.com/brms/reference/brmsfamily.html), and
[fitting interface](https://paulbuerkner.com/brms/reference/brm.html)
describe the development version.
These sources establish the comparison's scope; the QVR assessments below come
from the implementation and its tests. Full brms coverage is not established.

| Capability | QVR evidence and remaining requirements |
| --- | --- |
| Data dimensions and external inputs | `Model.schema` retains checked types and roles; `Model.bind` specializes finite extents and validates supplied values. Tested in `tests/modeling/test_model.py`. |
| Common regression families | `quivers.formulas.families` contains 50 entries, with 17 inverse links and mean/precision conversions for binomial, beta-binomial, gamma, Weibull, and other families. Asymmetric Laplace quantile regression supports a fixed or regressed quantile, with independent SciPy density/CDF/quantile and sampling tests. `tests/test_formula_distributional.py` checks independent likelihood and derivative oracles. ExGaussian uses the response mean and Gaussian/exponential components; inverse Gaussian uses response mean/shape with implicit sample gradients. These have independent density, CDF, derivative, and sampling oracles. Zero-inflated Beta regression reuses the native zero/one mixture with its one-atom probability fixed at zero; atom/interior densities, endpoint derivatives, sampling moments, and response means have independent oracles. Hurdle Gamma and LogNormal responses use a shared exact atom/positive-component implementation, with independent density, derivative, CDF, moment, and sampling oracles and SVI/HMC/NUTS formula checks. The larger native family registry remains a separate interface; the remaining brms families and their interactions need implementation and tests. |
| Zero-inflated and hurdle counts | Native and formula interfaces add zero-inflated negative-binomial, binomial, and beta-binomial responses plus hurdle negative-binomial responses. `tests/test_formula_count_mixtures.py` checks component-to-regression conversions, likelihood and gradient oracles, fitting, and predictions. Shared `Family.trials_parameter` metadata validates trial counts in `0..2**53` at fitting and new-data prediction; zero trials give a zero response. Native Binomial/BetaBinomial/NB/Poisson and the new mixtures use float64 count parameters, scores, and moments while preserving gradients to original learned heads. The hurdle sampler uses compound-Poisson/logarithmic-count sampling rather than rejection from a vanishing positive event. Native/QIEC reference execution supports all four; all 11 external renderers explicitly refuse them, checked in 44 family/backend cells. The current `hurdle_poisson` auxiliary remains named `zi`; a brms-compatible `hu` naming change is separate work. |
| Group effects | Single-bar blocks use native LKJ Cholesky correlations; double bars request independence. Shared IDs couple coefficients across location and auxiliary formulas. Scalar responses support centered and noncentered coordinates; categorical blocks couple all non-reference-category coefficients in noncentered coordinates. `tests/test_formula_correlations.py` checks Gaussian covariance densities, derivatives, and SVI/HMC/NUTS execution. Mixed scalar/transition widths and shared auxiliary blocks retain their joint noncentered covariance, checked against native/reference densities and every parameter gradient. New-group prediction supports joint donor and fitted Gaussian/LKJ policies. Supplied group covariance, multi-membership, repeated coefficients across blocks, and prediction with supplied covariance remain work. |
| Distributional and nonlinear formulas | `distributional={"sigma": "~ z + (1 | g)"}` supplies evaluated auxiliary predictors and parameter-specific links in the formula interface. Design columns and group coding are retained, and shared correlation IDs can span predictors. A nonlinear parameter dependency graph remains work. |
| Multivariate responses | QVR supports multiple observation sites. Joint formula construction, residual correlations, response selection, and aligned output coordinates need a common interface. |
| Splines, Gaussian processes, and special predictor terms | Some corresponding kernels exist. Ordinal `cs(expr)` effects retain a transition axis, including noncentered shared correlations. Cumulative/hurdle effects use training-domain bounds and proper conditional slack priors to ensure ordered effective cutpoints. Complete grouped profiles reuse exactly one varying location, and support noncentered category-specific slopes. Prediction retains fitted cutpoints, exposes `posterior_thresholds`, and offers explicit support-conditioned new-group policies. Centered category-specific groups, smooth construction, monotonic effects, and multi-membership remain unsupported or require further work. |
| Dependence structures | QVR has scans and state-space examples. brms's temporal, spatial, and supplied covariance structures require individually verified parameterizations and prediction behavior. |
| Additional response information | Formula `offset(expr)` and `exposure(expr)` terms enter location and auxiliary predictors with coefficient one, preserve evaluated arrays, and agree with independent density, gradient, and posterior-prediction oracles in `tests/test_formula_offsets.py`. Binomial and beta-binomial trials support constants or per-row columns. Formula `weights="column"` retains typed row weights and scales only observation scores; independent prior/likelihood gradient, exposure-interaction, and fixed-coefficient prediction oracles are in `tests/test_formula_weights.py`. Censoring, truncation, and their interactions remain work; likelihood weights are distinct from replication. |
| Missing data and measurement error | Explicit latent-variable models are possible. Automatic imputation formulas, measurement-error preparation, multiple datasets, and their prediction semantics remain work. |
| Priors | `get_prior` inspects the exact overrideable priors in the formula AST, including column autoscaling, auxiliary coefficients, and shared correlations. Unused overrides fail before fitting; `fixed_parameters` removes known auxiliary parameters from the latent and prior layouts. Class-based selectors, regularized horseshoe and R2D2 workflows, and prior-only fitting need implementations and acceptance tests. |
| Sampling and variational fitting | Existing MCMC and variational components remain configurable through reusable methods. Every algorithm's objective, stopping condition, initialization, and checkpoint semantics stay visible. This does not establish equivalence to Stan's samplers. |
| Prediction | `BayesianFit.posterior_linpred`, `posterior_epred`, and `posterior_predict` accept `new_data`, retaining training transforms, contrasts, category labels, and group coding. They preserve chain/draw axes and validate new covariates, offsets, exposures, auxiliary predictors, and per-row trials without response or likelihood-weight columns. Known groups may be reordered or repeated. New or missing group IDs require `allow_new_levels=True` and an explicit donor/Gaussian policy; ordinary categorical predictor levels remain strict. `re_formula` selects fitted varying effects, with unknown selectors refused. Neural fits require explicit new `predictor_data`. `tests/test_formula_new_data.py` checks independent prediction and derivative oracles and retained layouts. `ModelFit.predict` accepts checked tensor bindings with a new row count when the latent layout remains compatible. Tests additionally check joint cross-auxiliary donors, compact prediction axes, categorical group-major layouts, and group-effect/response RNG independence. Selected responses and further predictive quantities from the brms [prediction contract](https://paulbuerkner.com/brms/reference/posterior_predict.brmsfit.html) remain separate work. |
| Parameter summaries and diagnostics | `BayesianFit.draws` preserves chain/draw axes; `fixed_effects`, `random_effects`, and `coefficients` reconstruct native population/group coefficients with original labels and retained level coding. `tests/test_formula_posterior.py` checks centered, noncentered, shared, and categorical effects. Both source and formula fits expose retained `PosteriorAnalysis` uncertainty summaries; recorded MCMC chains supply convergence diagnostics, while variational pseudo-chains do not. Covariance summaries remain work. |
| Model comparison | `ModelFit.log_likelihood` and `BayesianFit.log_likelihood` extract raw pointwise observation densities. Their shared `PosteriorAnalysis` reuses scores for torch-only empirical WAIC, optional ArviZ PSIS-LOO, and ranked model comparison with stacking weights. Compared analyses check response values and predictive score units. Retained analyses accept explicit observation/event coordinates through the shared DataTree converter. Marginal-likelihood observers, exact refits, and K-fold validation require further work. A whole-program joint score is insufficient. |
| Other post-processing | Hypothesis tests, Bayes factors, marginal likelihoods, R-squared, conditional effects, model updates, and persistent fitted artifacts require their own implementations and tests. |
| Custom families and backends | Existing family and target registries are extension points. Custom scoring, sampling, prediction, and emitted-target support must each be checked; a registered name proves none of these by itself. |

Full coverage requires an executable fixture for every supported feature and
interaction, an independent density or prediction oracle, and a stated tolerance.
For instance, comparing correlated effects requires checking the covariance
model; matching a formula's printed text would not establish the same prior.

The pinned [released family registry](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/family-lists.R)
and [development family registry](https://github.com/paul-buerkner/brms/blob/f4f906dc0db56fd42b58cdd7d57e76b40f650331/R/family-lists.R)
each contain 49 named entries plus `custom`; their definitions agree apart
from trailing whitespace. QVR's 50 formula entries cover all 49 of those names
after normalizing `student`/`student_t`, `negbinomial`/`negative_binomial`,
and `inverse.gaussian`/`inverse_gaussian`. Its remaining entry is generic
mixture composition. This name comparison does not establish equivalent
parameterizations, links, or fitting behavior.

The `cox` and `xbeta` formula frontends complete this name inventory.
`cox` retains an identified normalized spline baseline and a proper
positive-hazard tail through the generic `BernsteinHazard` law; `xbeta`
reuses an affine Beta law and censoring atoms. Neither name establishes
all baseline-group, response-modifier, prior, or workflow behavior.
The native kernel registry is a separate capability: a native distribution
does not imply its corresponding formula workflow. `dirichlet2` is an
alternate parameterization, rather than a new distribution law.

The vector frontends share the categorical compiler's component
coordinates and the ordinary posterior replay machinery. Multinomial
and Dirichlet-multinomial responses retain integer row totals;
Dirichlet responses retain mean/precision or component-concentration
parameterizations. Additive-log-ratio Normal responses retain an
explicit reference component, covariance coordinates, and the simplex
Jacobian. Their optional Monte Carlo response expectations require an
explicit sample budget. Independent tests compare complete joints,
every fitted coefficient gradient, mixed-width group correlations, and
new-data predictions. These paths do not establish joint multivariate
formula construction or residual-correlation support.

Wiener formulas retain unsigned reaction times and decisions as separate
typed inputs and encode their joint law inside QVR. The native and
reference kernels share one certified rejection sampler. Density and
moment tests use independent series and differential-equation oracles;
formula tests cover SVI, HMC, NUTS, all distributional-parameter
derivatives, and new-data predictions. Low-level predictive draws retain
the signed joint encoding, while structured response draws decode
reaction time and decision without repeating sampling. This does not
establish support for every response modifier.

COM-Poisson and all five ordinal constructions now have formula
frontends. COM uses the pinned compiled rate law consistently for fitting
and prediction, while the released R helpers use a mode parameterization.
The [COM guide](../guides/analysis-data-and-formulas.md#com-poisson-rate-and-response-mean)
states this discrepancy and the resource limits. The
[ordinal guide](../guides/analysis-data-and-formulas.md#ordinal-families-thresholds-and-category-specific-effects)
states all 28 family/link combinations, discrimination checks, threshold
support, and differences between QVR's identified-coordinate priors and
the pinned brms defaults. These implementations do not establish full
prior, response-modifier, or workflow parity.

The released registry also includes `squareplus` and `softit` links,
which the website's abbreviated link lists do not enumerate. QVR now
implements both with stable staged expressions; value and first/second
derivative tests include the origin and large finite predictors. Neither
the family count nor QVR's 17 inverse links establishes full coverage.
Links, thresholds, response modifiers, interactions, and the other
workflows in the table need separate acceptance tests against each pinned
baseline. Future audits should compare released and development sources
independently, retaining their commit identities and support status.

Skew-normal, shifted-lognormal, Fréchet, and GEV frontends distinguish
native location/scale parameters from response means, retain support and
moment domains in prediction, and state their GEV shape-prior policy.
`tests/test_formula_shape_families.py` checks those conversions, priors,
and predictions; `tests/continuous/test_shape_families.py` checks the
native numerical, derivative, and sampling contracts. All four support
native and checked-reference execution; all 11 external targets refuse
them explicitly. The native `Truncated` wrapper now uses deterministic
interval masses for independent scalar coordinates, while coupled events
without a joint rectangle-mass operation remain refused. Formula response
truncation, censoring, and their interactions remain separate work.

Discrete Weibull and zero-inflated asymmetric Laplace now have native
and checked-reference kernels, with independent source-density and
formula-prediction checks. Their 22 family/backend cells all refuse
external rendering explicitly. The formula-only `negbinomial2` frontend
adds reciprocal dispersion to the existing negative-binomial law, with
a direct `Gamma(0.4, 0.3)` prior on `sigma`. Formula density, coefficient
gradient, SVI/HMC/NUTS, and new-data prediction checks are in
`tests/test_formula_reparameterized_families.py`. Discrete Weibull
expected responses retain explicit tail-certification and work limits;
its main transformed predictor is a probability rather than a mean.

## Shared implementation paths

The consolidation audit checks both older repository implementations and additions
on this branch. A new interface should extend the following paths before adding a
second parser, compiler, fitting loop, or prediction collector.

| Concern | Shared implementation and preserved behavior |
| --- | --- |
| Static and runtime declarations | One ordered parameter telescope supplies source elaboration, typed schemas, specialization, and tooling. Display strings do not replace retained QIEC types. |
| Dependent dimensions | Family signatures collect equations into the existing exact Nat solver. Source and kernel checking share sample-carrier inference; the older independent event-offset and pure-result classifiers have been removed. Conflicting input uses retain both extents in their diagnostic. |
| Vector and ordinal regressions | Component layouts extend the existing categorical emitter. Scalar, auxiliary, category, and threshold effects use the same correlation blocks and posterior replay machinery. |
| Formula expressions | Cox and ordinary regression compilation import one set of AST builders. Spline priors use the existing parsed auxiliary-prior path. |
| Hazard baselines | An immutable M-spline basis contracts into normalized Bernstein controls. Native and reference Cox adapters use the generic hazard law rather than separate density and sampling implementations. |
| Censored Beta laws | Affine coordinates retain both endpoint distances; censoring supplies the boundary atoms. Beta tails and the existing Dirichlet log-Gamma sampler are shared with the other probability laws. |
| Density and mass queries | `LogDensity` and `LogMass` use the same checked sampleable values and recursive raw constructors. Native total mass delegates to measure normalizers. A density at a selected point is not used as a mass surrogate. |
| Restrictions and mixtures | One endpoint-aware event operation handles atoms, nested restrictions, mixtures, and normalization. Mixture sampling selects components by weighted raw mass; conditioning retains restricted components and normalizes their sum once. |
| Numerical specializations | Closed count laws preserve stable large-count scores and boundary derivatives. They remain numerical specializations of the same measures; replacing them with generic log-mixture arithmetic would lose tested endpoint derivatives. |
| Observed values | Lexical computed observations use the existing observation handler and latent registry. They add scores without allocating a second latent coordinate. |
| Editor vocabularies | One generator supplies Tree-sitter and TextMate family/builtin sets. LSP, REPL, and Pygments read the same runtime registries. |

The reference machine retains scalar standard-library arithmetic, while native
kernels retain Torch arrays and automatic differentiation. Standalone emitted
runtimes likewise require their host's array operations. These execution boundaries
share semantic records and numerical recurrences where possible; independent
oracles check their agreement. Shared names alone do not establish agreement.

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
3. Extend the tested new-group and effect-selection policies with supplied covariance and selected responses, and raw pointwise likelihoods with marginal-likelihood observers, exact refits, and persistence. Retained formula evaluators currently remain live process state, with an explicit serialization refusal.
4. Connect the structured executors as their issues land, retaining each solver's
   objective, approximation, termination, and differentiation contract.

The useful open questions are concrete: which observer must a structured result
support, which dimensions may change at prediction time, and which cached objects
remain valid after parameters change? Each answer should be expressed in a typed
contract and tested before a new executor or formula term is advertised.
