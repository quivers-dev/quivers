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

## Response-time distributions

`ExGaussian(loc, scale, rate)` is the sum of independent Normal and Exponential
variables, with response mean `loc + 1/rate`. `InverseGaussian(mu, shape)` uses
the response mean and positive shape directly. Both provide native densities,
CDFs, moments, broadcasting, and reparameterized sampling. Inverse Gaussian
samples use an exact vectorized transformation and implicit CDF derivatives;
its discrete sampler branch is not differentiated as though it were fixed.

::: quivers.continuous._response_time

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
