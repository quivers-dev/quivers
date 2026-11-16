# Continuous Families

The module gives the families of continuous distributions
([family registry](../../guides/continuous-families.md#family-registry)), each chosen by the values
a response can take: a Normal for real values, and other families for positive or bounded
supports. A [conditional family](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families)
computes its parameters from its input through a learned parameter map, while a fixed call such as
`Normal(loc=0, scale=1)` has constant parameters.

::: quivers.continuous.families

## Asymmetric Laplace quantiles

`AsymmetricLaplace(loc, scale, quantile)` models a selected response quantile;
`ConditionalAsymmetricLaplace` learns all three parameters from its input. The
native distribution supplies differentiable sampling, density, CDF, inverse CDF,
mean, variance, and entropy. Its scale follows the check-loss convention used in
[brms quantile regression](https://paulbuerkner.com/brms/articles/brms_families.html#quantile-regression):
at quantile `0.5`, it equals `Laplace(loc, 2 * scale)`.

::: quivers.continuous._asymmetric_laplace

## COM-Poisson counts

`COMPoisson` accepts exactly one of `rate` and `log_rate`, together with
positive `dispersion`. `ConditionalCOMPoisson` learns native rate and
dispersion. The family has mass proportional to
`rate**k / factorial(k)**dispersion`; zero rate is a point mass at zero.
Parameters, log scores, and moments use float64; samples use int64 and
supply no pathwise gradient.

Centered sums certify omitted mass and weighted moments with default
`normalization_rtol=1e-12` and `max_terms=262144` candidate cells.
Failure to certify within the budget raises `NotImplementedError`.
The exact log-concave rejection sampler has its own
`max_proposals=10000` round budget and raises `RuntimeError` on proposal
or count-capacity failure. It does not sample a truncated normalization
table. Native and checked-reference execution are supported; all 11
external targets refuse this family. CDF and inverse-CDF operations,
thus censoring and interval truncation, remain unavailable.

::: quivers.continuous._com_poisson

## Vector count events

`Multinomial(total_count, probs=...)` and
`Multinomial(total_count, logits=...)` choose exclusive parameterizations;
`DirichletMultinomial(total_count, concentration)` adds a positive concentration
vector. Their trailing category axis is one event, independent of batch axes.
A batch may have heterogeneous trial counts. Every event has nonnegative
integer coordinates summing to its trial count; zero trials give an all-zero
vector, including for a single category.

Trial counts are checked before floating conversion and lie in `0..2**53`.
Parameters, scores, means, and variances use float64; discrete samples use
int64 without pathwise reparameterization. Dirichlet-multinomial concentration
sums must remain finite. `ConditionalMultinomial` learns category logits and
`ConditionalDirichletMultinomial` learns positive concentrations, with fixed
trials supplied to each conditional constructor.

Native and checked-reference execution support both kernels. The
[source example](../../guides/continuous-families.md#vector-count-events)
keeps row dimensions separate from the category event axis. External rendering
is a separate capability from these native interfaces.

::: quivers.continuous._multinomial

## Bernstein and spline baseline hazards

`BernsteinHazard(breaks, controls, rate=...)` defines a positive-time
proportional-hazards law. `breaks` has intrinsic shape `[s + 1]` and
`controls` has shape `[s, k]`; the event itself is scalar. The dependent
family signature checks this relation independently of observation batch
axes. Boundaries start at zero and increase strictly. Positive Bernstein
coefficients integrate to one over the retained interval, within numerical
simplex tolerance, fixing baseline scale separately from the rate.

After the final boundary, the endpoint hazard continues constantly and
positively. Survival thus tends to zero; the kernel retains all tail mass.
Supply exactly one of `rate` and `log_rate`. The latter avoids forming an
unnecessary exponential. The native implementation evaluates parameters and
scores in float64, preserving gradients to original learned tensors.

`CoxPH(basis, weights, ...)` contracts an immutable `MSplineBasis` with
positive simplex weights and delegates to this same kernel. It retains
knots, degree, and tail policy. `cox_spline_basis` places interior knots
using event times and sets the upper boundary from all follow-up times;
new observations reuse the resulting basis. Explicit knots are required
when the requested distinct event quantiles are unavailable.

`ConditionalBernsteinHazard` learns only the log rate over explicitly
supplied fixed baseline tables. It validates and copies those tables;
neural heads do not construct unconstrained knots or coefficients. Source
programs can pass differentiable controls explicitly instead.

Both kernels provide density, CDF, log tails, inverse CDF, and pathwise
sampling. The inverse uses a bounded safeguarded solve within the interval
and an analytic constant-hazard tail; numerical capacity or convergence
failures raise explicitly. Analytic response moments are unavailable.
Native and checked-reference execution support the typed source family;
external family rendering remains an explicit capability check.

::: quivers.continuous._bernstein_hazard

::: quivers.continuous._cox

::: quivers._splines.MSplineBasis

::: quivers._spline_design.cox_spline_basis

## Censored extended Beta responses

`ExtendedBeta(mu, phi, kappa)` starts from
`X ~ Beta(mu*phi, (1-mu)*phi)` and returns
`clamp((1+2*kappa)*X-kappa, 0, 1)`. The parameter `mu` is the
underlying Beta mean; the response mean generally differs. Precision
`phi` is positive, `0 < mu < 1`, and the extension `kappa` is nonnegative.
At zero extension the law reduces to an ordinary Beta, with no endpoint
atoms. Positive extension places the exact censored tail masses at zero
and one. Scoring an endpoint evaluates its atom mass; interior observations
use the affine Beta density.

Parameters, scores, moments, and draws use float64. Complementary Beta
coordinates retain small endpoint distances before subtraction or sampling
rounding. `log_cdf` includes boundary atoms, while `log_cdf_left` excludes
them; the two survival methods use complementary strict/inclusive events.
Pathwise sampling differentiates the affine map and censoring pushforward.
It requires both Beta concentrations to be at least the square root of
float64's smallest normal value, keeping shape-gradient denominators
representable. Ordinary sampling retains the shared log-Gamma underflow
fallback for smaller shapes.
A mathematically interior draw that cannot be represented strictly between
zero and one raises `RuntimeError`, rather than adding numerical atom mass.

Response means and variances use positive adaptive quadrature, bounded by
`moment_atol=1e-12`, `moment_rtol=1e-10`, and
`moment_max_panels=4096` per batch row, processed in chunks of at most 64
rows. Tiny extensions use partial-Beta moment corrections; zero extension
retains the exact Beta moments and their one-sided first extension
derivatives. These derivatives do not imply that every second extension
derivative exists at zero. A failed convergence or interval-resolution
check raises explicitly. An inverse CDF is unavailable because the shared
Beta base does not implement one. Native and checked-reference execution
support this law; external renderers currently refuse it by family name.

::: quivers.continuous._extended_beta

## Additive log-ratio Normal events

`AdditiveLogRatioNormal(loc, scale_tril, reference=None)` maps a correlated
Normal vector of dimension `d` to a simplex of dimension `d + 1`. By default,
the final component is the reference: `log(y[:-1] / y[-1])` has that Normal
distribution. An integer `reference` selects another component without
reordering the response; Normal coordinates follow the remaining components
in their original order. Batched reference indices may vary by observation.
`scale_tril` is a lower-triangular Cholesky factor with a positive diagonal.
Batch axes broadcast separately from the coordinate and matrix event axes.
`ConditionalAdditiveLogRatioNormal` learns both Normal heads.

The density includes the change-of-variables term `-sum(log(y))`.
Every component is strictly positive, and components must sum to one
within the native simplex tolerance. Native sampling is pathwise;
native and checked-reference scores retain the entire simplex as one event.
This family differs from `LogisticNormal`, which retains its existing
stick-breaking transform of independent Normal coordinates. Neither the
Normal location nor its softmax is generally the simplex expectation.

The checked response extent is `nat_add(d, 1)`, expressing the same
dimension relationship used by native construction. All external targets
currently report this family's missing rendering capability explicitly.

::: quivers.continuous._additive_log_ratio

## Discrete Weibull counts

`DiscreteWeibull(probs, concentration)` has nonnegative integer support,
with survival `P(Y >= k) = probs**(k**concentration)`. The probability
lies in `[0, 1)` and the concentration is positive; zero probability
gives a point mass at zero. `ConditionalDiscreteWeibull` learns these
native parameters, which are distinct from a response mean.

Parameters, scores, and moments use float64; samples use int64 and
are non-reparameterized. Finite samples and finite requested quantiles
above `2**53` are refused before integer conversion. Moment calculation
uses adaptive sums with survival-integral remainder bounds. Native
`moment_rtol=1e-10` and `moment_max_terms=262144` control its relative
tolerance and count-term budget, retained by `expand`. A failure to
certify the requested tolerance raises explicitly; mathematical moments
still exist throughout the admitted parameter domain.

::: quivers.continuous._discrete_weibull

## Zero-inflated asymmetric Laplace responses

`ZeroInflatedAsymmetricLaplace(zero_prob, loc, scale, quantile)` has an
exact atom at zero and a continuous asymmetric Laplace component.
Its score at zero is `log(zero_prob)`, independently of the component's
density there. `loc` is the component's selected quantile, with positive
scale and a quantile strictly between zero and one. Its CDF includes
the atom at zero; strict survival excludes it there. The
[formal definition](../../semantics/programs.md#a-zero-atom-with-asymmetric-laplace-responses)
specifies the mixed dominating measure and response mean.

`ConditionalZeroInflatedAsymmetricLaplace` learns the native parameters.
The mixture sampler selects its branch before drawing a continuous
response and supplies no pathwise sampling gradient. Native and
checked-reference execution support this family and DiscreteWeibull;
all 11 external renderers explicitly refuse both.

::: quivers.continuous._zero_inflated_asymmetric_laplace

## Response-time distributions

`ExGaussian(loc, scale, rate)` is the sum of independent Normal and Exponential
variables, with response mean `loc + 1/rate`. `InverseGaussian(mu, shape)` uses
the response mean and positive shape directly. Both provide native densities,
CDFs, moments, broadcasting, and reparameterized sampling. Inverse Gaussian
samples use an exact vectorized transformation and implicit CDF derivatives;
its discrete sampler branch is not differentiated as though it were fixed.

Both laws expose `log_cdf` and `log_survival_function`. These methods use
Float64 arithmetic and retain the smaller tail before complementing it;
`cdf` projects the result to the promoted parameter/query dtype. Interval
normalization uses the paired tails once per endpoint. A positive local
density rule handles narrow intervals whose endpoint tails round to the
same number, including adjacent representable bounds.

The Gaussian divided-difference calculation checks agreement between fixed
eight- and 32-node rules for short separations. Nonconvergence, a scaled argument magnitude above
`1e150`, or a separation above `1e150` raises `RuntimeError`; the scalar
reference backend raises `DistributionError`. This is a bounded numerical algorithm,
with no uniform accuracy guarantee across all parameter regimes. Finite
log-tail and local-interval calculations support first and second parameter
derivatives and forward automatic differentiation.

::: quivers.continuous._response_time

## Skew, shifted, and extreme-value distributions

`SkewNormal(loc, scale, skewness)` uses the native Azzalini
location/scale parameters. `ShiftedLogNormal(loc, scale, shift)` shifts
a lognormal component by a nonnegative amount.
`Frechet(scale, concentration)` has a positive concentration;
`GeneralizedExtremeValue(loc, scale, concentration)` has a real one,
with the Gumbel limit at zero. These scalar response laws have different
moment domains and support endpoints; see the
[native-family guide](../../guides/continuous-families.md#skew-shifted-and-extreme-value-responses).
The formula interface performs its own response-parameter conversions.

The four native distributions provide `cdf`, `log_cdf`,
`log_survival_function`, `icdf`, and direct vectorized pathwise samples.
Their `ConditionalSkewNormal`, `ConditionalShiftedLogNormal`,
`ConditionalFrechet`, and `ConditionalGeneralizedExtremeValue` wrappers
learn the native constructor parameters. Checked-reference execution is
supported; all 11 external renderers explicitly refuse these four laws.
Skew-normal CDF and quantile evaluation uses float64 intermediate
arithmetic and deterministic quadrature, with no uniform error guarantee.
Those two methods require a device with float64 support.

::: quivers.continuous._skew_normal

::: quivers.continuous._shape_families

## Truncating conditional families

`Truncated` denotes a normalized restriction of the base conditional
distribution. For independent scalar coordinates, it sums each coordinate's
log interval mass; a coupled event requires a joint rectangle-mass
operation. The [formal definition](../../semantics/programs.md#truncation-as-normalized-restriction)
distinguishes this conditioning from interval censoring.

The wrapper reads the base distribution through `_get_dist(x)` and
uses deterministic CDF or log-tail interval masses. Normal tails have a
stable specialized calculation; Gamma and Poisson use incomplete-gamma
probabilities. Unsupported base/event contracts and zero or nonfinite
computed interval masses raise explicit errors. Normal sampling uses a
stable inverse CDF; other bases use bounded rejection controlled by
`max_rejection_iterations`. The rejection branch does not establish a
general pathwise derivative through the acceptance event.

Gamma interval masses use float64 lower-tail series, upper-tail continued
fractions, and small-shape complement calculations. Shape and rate
derivatives propagate through converged recurrences; nonconvergence
within 512 iterations raises explicitly. These numerical and derivative
contracts do not assert a uniform accuracy or convergence guarantee.

## Generalized Pareto tails

`ConditionalGeneralizedPareto` learns `loc`, positive `scale`, and real
`concentration`, the tail index. The registered `GeneralizedPareto`
family supports native and QIEC reference execution, with density, CDF,
inverse CDF, and reparameterized draws. Its finite endpoint for negative
indices and its exponential limit at zero are described in the
[tail-family guide](../../guides/continuous-families.md#generalized-pareto-tails).
The native distribution implementation is private; the public API is
the conditional wrapper and its registered
[`FamilySpec`](family_spec.md#quivers.continuous.family_spec.FamilySpec).

## Positive hurdle distributions

`HurdleGamma(zero_prob, concentration, rate)` and
`HurdleLogNormal(zero_prob, loc, scale)` combine an exact zero atom with a
strictly positive component. `ConditionalHurdleGamma` and
`ConditionalHurdleLogNormal` learn those parameters from their input.
Both native distributions provide log mass at zero, positive-response log
density, CDF, response mean/variance, broadcasting, and sampling. The structural
zero gate makes sampling non-reparameterized. Their scoring and moment
parameters remain differentiable, including finite permitted-event gradients
at zero probabilities of zero and one. The Gamma CDF inherits PyTorch's
incomplete-gamma derivative limitations for its concentration parameter.

::: quivers.continuous._positive_hurdle

## Zero-inflated and hurdle counts

`ZeroInflatedNegativeBinomial(zero_prob, total_count, probs)`,
`ZeroInflatedBinomial(zero_prob, total_count, probs)`, and
`ZeroInflatedBetaBinomial(zero_prob, total_count, concentration1, concentration0)`
add a structural zero to their count component. Their total zero probability
includes both that atom and the component's own zero mass.
`HurdleNegativeBinomial(zero_prob, total_count, probs)` instead assigns the
whole zero probability to the atom and normalizes its component over
positive counts. Negative-binomial `probs` follows PyTorch's convention;
its component mean is `total_count * probs / (1-probs)`.

These scalar count distributions provide differentiable log masses and
moments, batch broadcasting, and non-reparameterized sampling. Binomial
trial counts may be zero, giving a point mass at zero. At negative-binomial
`probs=0`, the hurdle's positive component uses its limiting point mass at
one. Its sampler uses a compound-Poisson representation with logarithmic
counts, avoiding rejection from an almost-always-zero count distribution.
The [count guide](../../guides/analysis-data-and-formulas.md#zero-inflated-and-hurdle-counts)
relates these native parameters to regression means, precision, and links.
Native and QIEC reference execution support all four. Each of the 11
external renderers currently refuses them with a `family:<Family>`
diagnostic; native registration alone does not provide a target mapping.

Native Binomial, BetaBinomial, NegativeBinomial, Poisson, and these four count mixtures
compute parameters, log scores, and mean/variance in float64. Promotion
preserves gradients to learned float32 parameter heads, whose gradients
retain their original dtype. Finite trial counts must satisfy
`0 <= total_count <= 2**53` before conversion, preserving exact integer
representation; the general `Int` carrier remains signed 64-bit. Large
counts should arrive as integer arrays or integer literals, since promotion
cannot recover a value already rounded in float32. Discrete samples remain
integer-valued and non-reparameterized.
Arithmetic count observations are checked before conversion as well: a finite
original value outside `[-2**53, 2**53]` is refused. With argument validation
disabled, fractional, nonfinite, or other off-support values return negative
infinity rather than being truncated. Finite category indices retain their
separate support contract without imposing the arithmetic-count bound.
The count density kernels use stable deviance, Stirling, and
gamma-increment formulas where applicable; this precision policy does
not assert a uniform numerical error bound over all parameter regimes.

For Poisson, negative-binomial, and their inflated or hurdle samplers,
nonfinite generated intensities or counts, or values above `2**53`,
raise `RuntimeError` natively and `DistributionError` in the reference
backend. No truncation, renormalization, or overflow-hiding redraw is
performed. Structural-zero outcomes skip the unreachable count branch.
See the [count guide](../../guides/continuous-families.md#discrete-categorical-families)
for the distinction between sampling limits and density evaluation.

::: quivers.continuous._zero_inflated_count
