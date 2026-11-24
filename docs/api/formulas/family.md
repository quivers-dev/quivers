# `quivers.formulas.family`

A [response family](../../guides/analysis-data-and-formulas.md#family-registry) is the distribution of
the observations, and its [link function](https://en.wikipedia.org/wiki/Generalized_linear_model)
connects the linear predictor to a parameter of that distribution [@nelder1972glm]. In
[Bernoulli regression](../../tutorials/qvr/02-glms.md), the inverse logit maps an unconstrained
predictor to a probability. The module gives `Family`, `Link`, `AuxParam`, `OrdinalSpec`, and `ResponseSpec`, and the `families`
and `links` registries. `Family.with_links(mu="probit")` selects a location link,
while `Family.with_links(sigma="softplus")` changes a distributional scale link;
[link selection](../../guides/analysis-data-and-formulas.md#distributional-regression-and-link-selection)
gives the supported names and parameterizations.

`squareplus` supplies the positive inverse link
`(eta + sqrt(eta**2 + 4)) / 2`; `softit` supplies the probability inverse
link `softplus(eta) / (1 + softplus(eta))`. The compiler uses stable
equivalent forms and stages shared intermediates. The
[link guide](../../guides/analysis-data-and-formulas.md#distributional-regression-and-link-selection)
distinguishes a distributional predictor's link from a scalar auxiliary
prior, and the [formal definition](../../semantics/programs.md#inverse-links-as-value-maps)
states their value-map semantics.

## What a family declares

A `Family` states everything the compiler needs to emit it, so a new
family is a new value and not a new branch in the compiler.

- **Native parameterization.** `Family.observe_lets` binds quantities
  derived from `mu` and the auxiliary parameters, and
  `Family.observe_args` lists the native constructor arguments in order.
  `families["gamma"]`, for instance, binds `gamma_rate = shape / mu` and
  passes `(shape, gamma_rate)`.
- **Links.** A `Link` carries its inverse as a QVR expression in
  `Link.inverse_expr`, with any shared `Link.intermediates`; the emitted
  program contains exactly that expression. `Link.image` is the interval
  the inverse maps onto.
- **Admissible links.** `Family.location_links` and `AuxParam.links` list
  the links `with_links` admits. An auxiliary parameter on the positive
  half-line or the unit interval takes brms's links for that interval when
  it lists none. `Family.admissible_links` reads either list, and a
  parameter without one refuses `with_links`.
- **Parameter ranges.** Each `AuxParam` has a `Domain`. A fixed value
  outside it is refused. A lower endpoint above zero also restricts the
  prior, and the link must map into it: the `nu` of `frechet`, for
  instance, must exceed one, so its link is `logm1`.
- **Shifts.** `AuxParam.below_response` marks a shift every response
  exceeds, such as the `ndt` of `shifted_lognormal`. A fixed value is
  checked against the responses, a prior may name the smallest one as
  `response_min`, and inference starts below it.
- **Response support.** `Family.response_interval` is the interval a
  continuous scalar response must lie in. It comes from the support and
  atoms of the native distribution, reached through `Family.native`,
  unless `Family.response_domain` states it. Every such family checks its
  response before inference.
- **Exposure.** `Family.exposure` gives the distribution of a count over a
  `rate()` denominator. A family without one refuses `rate()`.
- **Log-scale location.** `Family.log_location_argument` names a native
  parameter that takes the location on the log scale; under a log link
  the linear predictor is passed to it without exponentiation.
- **Known errors.** `Family.error_scale` names the residual scale that the
  errors of a `se()` term replace, or join in quadrature under
  `sigma = TRUE`. A family without one refuses `se()`.
- **Correlated residuals.** `Family.elliptical` is an `Elliptical` value
  naming the scale and the degrees of freedom of a Normal or Student-t
  residual law. Autocorrelation terms and `set_rescor` read the
  multivariate law from it. Responses are correlated only under one common
  value.
- **Recorded events.** `Family.recorded_events` admits `cens()` and
  `trunc()`. A mixture admits them when all its components do.
- **Missing responses.** A family admits `mi()` when a missing response can
  be a latent coordinate of its law: a continuous scalar law without point
  masses, decision, baseline or shift below the responses.
  `Family.latent_response_obstacle` gives the property that rules it out.
- **Factor responses.** A factor is coded `0, ..., n - 1` when the native
  support is that set, or when `Family.factor_levels` states the number of
  levels (`Family.coded_levels`).
- **Explained variance.** `Family.r_squared` selects the variance
  `bayes_r2` reads by default.
- **Mixture law.** `Family.mixture_law` names a native law for a mixture of
  copies of the family, such as `MixtureNormal` for `gaussian`.
- **Component scales.** `AuxParam.per_component` gives a parameter one value
  per free component of the response, as the `sigma` of `logistic_normal`.

Every declaration is open to a new family. A family built with `Family`, or
with [`custom_family`](index.md#quivers.formulas.custom_family) and then
`with_`, takes part in a feature by declaring its property:

```python
from quivers.formulas import Elliptical, families

heavy = families["student_t"]
heavy.error_scale   # "sigma"
heavy.elliptical    # Elliptical(scale="sigma", df="nu")
heavy.with_(error_scale=None)   # refuses se(), naming Family.error_scale
```

::: quivers.formulas.family.Elliptical

## Ordinal configuration

`Family.ordinal` retains an `OrdinalSpec` for cumulative, adjacent-category,
stopping-ratio, continuation-ratio, or hurdle-cumulative probabilities.
`families["acat"].with_links(mu="probit").with_thresholds("equidistant")`
returns a configured value. Discrimination defaults to the constant one;
explicit scalar or distributional estimates must be finite and positive.
`cs(expr)` supplies transition-specific effects. Cumulative and
hurdle-cumulative effects use training-domain spacing bounds plus positive
slack to keep effective row cutpoints ordered. Their proper conditional
slack priors differ from independent brms threshold priors.
`thresholds_by` combines centered group shape with exactly one reused or
automatic varying location; centered and noncentered category-specific slopes may
coexist with it. A fixed `cs(1)` with free cumulative/hurdle thresholds
and a proportional intercept is redundant and fails explicitly, as do
duplicate same-factor category-specific intercept/profile coordinates. The
[ordinal guide](../../guides/analysis-data-and-formulas.md#ordinal-families-thresholds-and-category-specific-effects)
gives links, threshold support, generated priors, and prediction shapes.

## Joint response coordinates

`Family.response` retains a `ResponseSpec` for scalar categorical responses
and joint vector families. `with_reference(index_or_label)` selects a retained
component and returns a configured family value. The default reference is the
first component; `dirichlet2` instead estimates every concentration and permits
only `None`. Additive log-ratio `logistic_normal` requires a reference and uses
identity-linked Normal coordinates, with coordinate-specific scales and an
LKJ response correlation. The native `AdditiveLogRatioNormal` constructor
retains its own final-component default.

`multinomial` and `dirichlet_multinomial` use count events; `dirichlet`,
`dirichlet2`, and `logistic_normal` use open-simplex events. `ResponseLayout`
retains the distinction between factor levels and named vector components.
The [vector guide](../../guides/analysis-data-and-formulas.md#factor-and-vector-responses)
gives component labels, inferred training totals, new-row trial counts,
and explicit Monte Carlo expectations for additive log-ratio responses.

## Finite mixtures

`Family.mixture` retains a `MixtureSpec` for families built by
[`mixture`](index.md#quivers.formulas.mixture): the component families
after `nmix` repetition, the `order` identification, and equated
parameters. The mixture's auxiliary parameters are the component
parameters with the component number appended (`mu1`, `sigma1`, `nu2`)
and the proportions `theta1, ..., thetaK`. Links belong to the components,
so `with_links` refuses a mixture. The
[mixture guide](../../guides/formula-mixtures.md) gives the brms
correspondence and the identification by ordered intercepts.

## Retained proportional-hazards baselines

`Family.baseline` retains a `BaselineSpec` for `cox` regressions.
`families["cox"].with_baseline(df=5, degree=3)` selects knots from training
responses; `with_baseline(basis)` accepts an immutable `MSplineBasis`.
The specification stores typed knot tuples and dimensions and preserves
its value through link selection and serialization. The generated program
retains the actual coefficient table for prediction on fresh rows.

`with_baseline(..., by="site")` retains separate normalized simplex profiles
for a grouping column on the same basis. Fitting retains its level coding
without inserting a varying intercept. Direct lens users supply the column
through `formula_from_data(..., groups=("site",))`. New-data group policies
share donor identities across baseline and coefficient sites; `gaussian`
draws a new baseline from the actual Dirichlet law. Baseline conditioning
remains active under `re_formula="~0"`.

The location parameter is a hazard multiplier, with default log link;
it is not response mean. Baseline weights have a unit-sum simplex prior
`sbhaz ~ Dirichlet(1)`, separating shape from rate. A finite positive
constant tail makes response moments finite at each fixed parameter draw,
but no closed mean is exposed.
Use an explicit `posterior_epred(mc_samples=...)` budget. The
[regression example](../../guides/analysis-data-and-formulas.md#spline-proportional-hazards)
distinguishes rate, event draws, and expected time.

::: quivers.formulas._survival.BaselineSpec

## COM-Poisson parameters

`com_poisson` uses log-linked rate `mu=lambda` and log-linked dispersion
`shape=nu`, with scalar default `Gamma(0.01, 0.01)`. Rate is not generally
response mean. The compiler passes `log_rate` directly to the native
kernel; transformed predictors return rate, while response expectations
use its certified normalized moment. The
[rate guide](../../guides/analysis-data-and-formulas.md#com-poisson-rate-and-response-mean)
distinguishes the pinned released Stan law from its R helper's mode
parameterization and states resource and capability limits.

## Count and zero-atom parameterizations

`discrete_weibull` uses a logit-linked survival probability, called `mu`
in the formula interface, and a log-linked positive `shape`. Its scalar
shape default is `HalfCauchy(2)`. The transformed predictor gives a
probability; the expected response requires a bounded numerical tail sum.

`negbinomial2` reuses native `NegativeBinomial` with reciprocal shape
`1/sigma`, mean `mu`, and variance `mu + sigma*mu**2`. Its positive
`sigma` has scalar default `Gamma(0.4, 0.3)` and a log link when
regressed. It is a dispersion coefficient, rather than a standard
deviation or the original `negative_binomial` family's `disp`.

`zero_inflated_asym_laplace` uses identity-linked component quantile
location, log-linked scale `sigma`, and logit-linked `quantile` and
`zi`. Their scalar defaults are `HalfCauchy(2)`, `Beta(2, 2)`, and
`Beta(2, 2)`, respectively. The response mean and whole-response
quantiles include the zero atom. The
[regression guide](../../guides/analysis-data-and-formulas.md#count-dispersion-and-a-zero-atom-at-a-quantile)
gives scoring conventions, parameter maps, and fixed or distributional
parameter examples.

## Skew, shifted, and extreme-value parameters

`skew_normal` uses response mean and standard deviation, converting them
to native skew-normal location and scale. `shifted_lognormal` predicts
the log-location of the component above `ndt`; its response expectation
adds `ndt` after exponentiating the lognormal mean correction.
`frechet` predicts response mean and restricts its `nu` to exceed one.
`gen_extreme_value` predicts location and uses a direct real shape `xi`.
The [parameter guide](../../guides/analysis-data-and-formulas.md#skew-shifted-and-extreme-value-regressions)
gives conversions, moment domains, and the GEV shape-policy distinction
from the pinned brms release.

A regressed Fréchet `nu` requires `logm1`. Its scalar default conditions
`Gamma(2, 0.1)` on `nu > 1`; custom scalar priors retain that restriction.
The shifted-lognormal scalar `ndt` default is bounded by the smallest
observed response, while a regressed `ndt` uses its declared link without
hidden response-dependent rescaling. GEV uses identity-linked `xi` with
scalar `Normal(0, 0.3)`, rather than the pinned brms shape-prior policy.

## Other response parameters

`xbeta` uses logit-linked underlying Beta mean `mu`, log-linked precision
`phi`, and log-linked extension `kappa`. Its native `ExtendedBeta` law
creates boundary atoms by censoring an affine Beta component; it does not
add independent inflation parameters. Fixed `kappa=0` recovers ordinary
Beta responses. Expected responses use the censored law's full mean,
whereas transformed linear predictions retain the component mean.
Scalar auxiliary defaults are `HalfCauchy(2)` for precision and
`Gamma(0.01, 0.01)` for extension; distributional coefficient priors use
the shared QVR coefficient policy.

`zero_inflated_beta` uses logit mean and zero-probability links plus a log
precision link. It lowers to native `ZeroOneInflatedBeta` with `coi=0`, retaining
only the `phi` and `zi` auxiliaries. Its response mean includes the zero-atom
weight; see the [regression example](../../guides/analysis-data-and-formulas.md#zero-inflated-beta-regression).

`hurdle_gamma` uses positive-component mean/shape parameters;
`hurdle_lognormal` uses log-location/log-scale parameters. Both have a `hu`
auxiliary with a logit link and exact zero-atom semantics. Their response
expectations include the positive-component probability, as shown in the
[hurdle guide](../../guides/analysis-data-and-formulas.md#positive-hurdle-regressions).

The count variants `zero_inflated_negbinomial`, `zero_inflated_binomial`,
`zero_inflated_beta_binomial`, and `hurdle_negbinomial` retain the native
component's count support. `zi` denotes a structural-zero probability;
`hu` denotes the hurdle's full zero probability. The negative-binomial
variants use `shape`, and the beta-binomial variant uses precision `phi`.
The [count guide](../../guides/analysis-data-and-formulas.md#zero-inflated-and-hurdle-counts)
gives their component and response means.

`Family.trials_parameter` names the native parameter for a known trial
count. Fitting and prediction use this metadata for binomial, beta-binomial,
and their zero-inflated variants. `binomial_trials` accepts a common count
or a column of nonnegative integer counts no greater than `2**53`;
zero trials imply a zero response. The bound preserves exact integer
representation in native float64 count arithmetic. The language's `Int`
carrier remains signed 64-bit. The trial count is data, not a fitted
auxiliary parameter.

::: quivers.formulas.family
