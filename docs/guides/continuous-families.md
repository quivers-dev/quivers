# Continuous Distribution Families

A conditional distribution family in quivers is a
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
whose distribution's parameters a parameter source, a lookup table or a network,
computes from each input. More than fifty families are registered. QVR writes
supported families inline in program steps with checked parameters
([Inline distributions](dsl-programs-and-lets.md#inline-distributions)).
[Continuous Spaces](continuous-spaces.md) describes the spaces they live on.

## Family registry

A family is chosen by what one outcome represents and which values it may take: a
Normal for an unconstrained measurement, a Gamma for a positive duration, a Bernoulli
for a binary outcome, a Dirichlet for a whole probability vector. Among families with
the same support, the tails decide: a Student-t puts more probability on extreme
deviations than a Normal.

`ConditionalNormal(domain, codomain)` learns a map from inputs to the normal's
parameters, including its mean. A mean given exactly is written in QVR as
`Normal(loc=mu, scale=sigma)`, which uses the values bound to `mu` and `sigma` at that
step.

Registry membership, checked source construction, and external rendering are
separate capabilities. In particular, native `Independent` requires a batched
base distribution, which the current checked scalar-family argument contract
cannot express as an inline source call. Use its native Python construction;
registration does not establish a QVR source or export form.

### Vector count events

A Multinomial response is a vector of category counts for one observation.
The row axis indexes observations; the category axis belongs to each event.
A typed runtime header states both axes, while a static telescope can leave
the row count to be inferred at binding:

```qvr
object Category : FinSet 3
program count_vectors[n : Nat](
    trials : Tensor[Int]([n]),
    probabilities : Tensor[Real]([n, Category]),
    y : Tensor[Int]([n, Category]),
) : n -> n
    observe y : n <- Multinomial(total_count=trials, probs=probabilities) [over=Category]
    return y
export count_vectors
```

`[over=Category]` marks a single vector event. Its log probability reduces
the category coordinates together, leaving one score per observation row.
This distinction also holds when row and category extents happen to be equal.
The supplied trials may vary by row; responses must sum to those trials.
Zero trials require an all-zero response. Replacing the kernel with
`DirichletMultinomial(total_count=trials, concentration=concentrations)`
uses a typed concentration tensor with the same axes.

Both native kernels check finite trial counts in `0..2**53` before conversion,
use float64 scores and moments, and draw int64 vectors without pathwise
sampling gradients. The static `[n : Nat]` determines a shape; the runtime
`trials` tensor supplies counts to the probability law. Native and
checked-reference execution support these kernels; the
[native API](../api/continuous/families.md#vector-count-events) gives the
constructors and conditional learned-parameter variants.

### Simplex events with correlated log ratios

An additive log-ratio Normal models a whole composition using correlated
Normal log ratios. Its `d` coordinates determine `d + 1` response components;
the typed interface expresses that relationship with static Nat addition:

```qvr
program compositions[n : Nat, d : Nat](
    location : Tensor[Real]([n, d]),
    factor : Tensor[Real]([n, d, d]),
    y : Tensor[Real]([n, nat_add(d, 1)]),
) : n -> n
    observe y : n <- AdditiveLogRatioNormal(loc=location, scale_tril=factor) [iid_over=n]
    return y
export compositions
```

Binding location shape `(12, 2)`, factor shape `(12, 2, 2)`, and response
shape `(12, 3)` infers `n=12` and `d=2`. An inconsistent response width
fails before execution. `nat_add` combines static natural indices; its
parentheses are index application syntax, rather than a runtime pure call.

The last component is the reference, so the Normal coordinates are
`log(y[..., :-1] / y[..., -1:])`. The density includes their Jacobian,
and each row contributes one joint score. This differs from the existing
`LogisticNormal` stick-breaking transform. The
[native API](../api/continuous/families.md#additive-log-ratio-normal-events)
describes the constructor and learned conditional heads.

### COM-Poisson rates

`COMPoisson(rate=..., dispersion=...)` uses a nonnegative rate and positive
dispersion, or accepts `log_rate=...` instead of rate. Its Poisson limit
has dispersion one; elsewhere rate is not generally the response mean.
Float64 centered sums certify normalization and moments within explicit
budgets, and an independent exact rejection envelope generates integer
samples. Both paths fail explicitly when their resource or exact-count
capacity is exceeded. The
[native API](../api/continuous/families.md#com-poisson-counts)
gives defaults and exception types; the
[formula guide](analysis-data-and-formulas.md#com-poisson-rate-and-response-mean)
gives regression parameters and the released brms helper distinction.

```qvr
object Out : Real 1
program count_model[n : Nat](
    log_rate : Real,
    shape : Real,
    y : Tensor[Int]([n]),
) : n -> Out
    observe y : n <- COMPoisson(log_rate=log_rate, dispersion=shape)
    return log_rate
export count_model
```

Here `n` specializes the response extent, while `shape` is a runtime
dispersion parameter. The family argument's name does not make it a
static shape index, and the typed response declaration alone contributes
no likelihood; the `observe` step supplies it.

### Core families {#hand-written-core-families}

```python
from quivers.continuous.families import (
    ConditionalNormal,
    ConditionalLogitNormal,
    ConditionalBeta,
    ConditionalTruncatedNormal,
    ConditionalDirichlet,
)
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
import torch

# Input: finite set X
X = FinSet(name="X", cardinality=5)
domain = Euclidean(name="X", dim=5)        # or FinSet
codomain = Euclidean(name="Y", dim=3)

# Conditional normal: learns maps x -> mu(x) and x -> log sigma(x)
normal = ConditionalNormal(domain, codomain)

# Sample from p(y | x)
x = torch.randn(8, 5)                              # batch of 8 inputs
samples = normal.rsample(x, sample_shape=torch.Size((100,)))  # (100, 8, 3)

# Log probability
y = torch.randn(8, 3)
log_p = normal.log_prob(x, y)  # shape (8,)
```

### Loc-scale families

Families with a learned location and scale, all with reparameterized draws:

```python
from quivers.continuous.families import (
    ConditionalCauchy,
    ConditionalLaplace,
    ConditionalAsymmetricLaplace,
    ConditionalExGaussian,
    ConditionalInverseGaussian,
    ConditionalGumbel,
    ConditionalLogNormal,
    ConditionalStudentT,
)
```

### Families on the positive reals {#positive-valued-families}

For positive outputs, $(0, \infty)$:

```python
import torch
from quivers.continuous.families import (
    ConditionalExponential,
    ConditionalGamma,
    ConditionalWeibull,
    ConditionalPareto,
    ConditionalInverseGamma,
    ConditionalHalfCauchy,
    ConditionalHalfNormal,
)
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
codomain = Euclidean(name="Y", dim=3)
x = torch.randn(8, 5)

gamma = ConditionalGamma(domain, codomain)
samples = gamma.rsample(x)  # positive
```

### Skew, shifted, and extreme-value responses

The native constructor parameters describe a distribution directly.
A formula frontend may instead parameterize its response mean or
standard deviation; the [regression guide](analysis-data-and-formulas.md#skew-shifted-and-extreme-value-regressions)
gives those conversions.

| Native family | Parameters and support | Response mean |
| --- | --- | --- |
| `SkewNormal(loc, scale, skewness)` | real location and skewness, positive scale; real responses | `loc + scale * delta * sqrt(2/pi)`, where `delta = skewness / sqrt(1 + skewness**2)` |
| `ShiftedLogNormal(loc, scale, shift)` | Normal log-location and positive log-scale, nonnegative shift; responses greater than `shift` | `shift + exp(loc + scale**2/2)` |
| `Frechet(scale, concentration)` | positive scale and concentration; positive responses | `scale * Gamma(1 - 1/concentration)` when concentration exceeds one |
| `GeneralizedExtremeValue(loc, scale, concentration)` | real location and concentration, positive scale; `1 + concentration * (y-loc)/scale > 0`, with the Gumbel limit at zero | location plus the tail correction in the [formal definition](../semantics/programs.md#scalar-response-kernels-and-regression-parameters), when concentration is below one |

Fréchet variance requires concentration above two; GEV variance requires
concentration below one half. A supported density can thus have no
finite response mean or variance. GEV's `concentration` has the opposite
sign to the shape argument of
[SciPy's `genextreme`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html).

Static response extents and runtime distribution parameters remain
separate. In the following program, `n` fixes a shared row axis;
`sigma`, `nu`, and the other real inputs supply distribution values.
Their declarations add no draws or scores.

Native and checked-reference execution support all four families. Each of
the 11 external renderers currently refuses them explicitly; family
registration does not supply an external implementation.

```qvr
object Out : Real 1
program scalar_responses[n : Nat](sigma : Real, skewness : Real, ndt : Real, nu : Real, xi : Real) : n -> Out
    observe skewed : n <- SkewNormal(loc=0.0, scale=sigma, skewness=skewness)
    observe shifted : n <- ShiftedLogNormal(loc=0.0, scale=sigma, shift=ndt)
    observe extreme : n <- Frechet(scale=1.0, concentration=nu)
    observe maxima : n <- GeneralizedExtremeValue(loc=0.0, scale=sigma, concentration=xi)
    return sigma
export scalar_responses
```

All four native distributions provide `cdf`, `log_cdf`,
`log_survival_function`, `icdf`, and pathwise `rsample`. Skew-normal
CDFs and quantiles use deterministic quadrature and float64 intermediate
arithmetic. Their tail and derivative checks
do not establish a uniform quadrature-error bound. Both methods require
a device with float64 support. Fréchet and
GEV return infinite moments outside the finite-moment domains above.

### Generalized Pareto tails

`GeneralizedPareto(loc, scale, concentration)` implements the
[generalized Pareto law](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html)
above `loc`,
with positive `scale` and real tail index `concentration`. A negative
tail index gives the finite upper endpoint `loc - scale/concentration`;
a nonnegative index gives an unbounded upper tail. At index zero, the
law is `loc + Exponential(rate=1/scale)`.

```qvr
object Row : FinSet 4
program tail(scale: Real) : Row -> Row
    observe y : Row <- GeneralizedPareto(loc=0.0, scale=scale, concentration=-0.2)
    return y
export tail
```

The native family supplies density, CDF, inverse CDF, broadcasting, and
reparameterized sampling, including differentiable zero-index limits.
Its mean exists for `concentration < 1` and its variance for
`concentration < 0.5`; the implementation returns `NaN` outside those
moment domains. The public learned wrapper is
[`ConditionalGeneralizedPareto`](../api/continuous/families.md#quivers.continuous.families.ConditionalGeneralizedPareto).
The underlying distribution class is an implementation detail, rather
than a `quivers.continuous.GeneralizedPareto` export. Native and QIEC
reference execution support this source family; the current external
renderers refuse it.

### Zero atoms with positive responses

`HurdleGamma(zero_prob, concentration, rate)` combines a structural zero with a
Gamma response. `HurdleLogNormal(zero_prob, loc, scale)` combines it with a
LogNormal response. Both constructors are exported by `quivers.continuous`;
their `ConditionalHurdleGamma` and `ConditionalHurdleLogNormal` wrappers compute
parameters from a learned input map. The positive component contributes no mass
to the zero atom, including when its continuous density diverges at zero.

The native distributions expose density/mass, CDF, mean, variance, expansion,
and sampling. Sampling includes a discrete gate and has no pathwise derivative;
likelihood and response-moment parameters remain differentiable. The
[formula interface](analysis-data-and-formulas.md#positive-hurdle-regressions)
uses `hu` for the structural zero probability and retains the component family's
usual mean/shape or log-location/log-scale conventions.

### Families on the unit interval {#unit-interval-families}

For outputs in $(0, 1)$:

```python
from quivers.continuous.families import (
    ConditionalBeta,
    ConditionalKumaraswamy,
    ConditionalContinuousBernoulli,
)
```

### Log-space families for small shapes

A [Beta](https://en.wikipedia.org/wiki/Beta_distribution) or
[Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) latent reaches
the samplers and the guides on the real line: its unconstrained value is
mapped into the support by a sigmoid or an exponential and scored there,
with the map's Jacobian. When the shape is small, as it tends to be in
sparse and stick-breaking priors, that round trip loses the tails. The
sigmoid of `-800` is already the smallest normal float, so
`Beta(0.001, 1)` scores the logit `-800` at about `-99` where its exact
log density is about `-7.71`, and a Gamma sampler at shape `0.001` returns the smallest
normal float for roughly half its draws.

[`LogitBeta`][quivers.continuous.log_space.LogitBeta] and
[`LogGamma`][quivers.continuous.log_space.LogGamma] put the variable on
the real line from the start. `LogitBeta(concentration1=a,
concentration0=b)` is the law of $x = \operatorname{logit}(p)$ for
$p \sim \mathrm{Beta}(a, b)$, and `LogGamma(concentration=a, rate=r)` is
the law of $y = \log g$ for $g \sim \mathrm{Gamma}(a, r)$:

$$
\log p(x) = a \log \sigma(x) + b \log \sigma(-x) - \log \mathrm{B}(a, b),
\qquad
\log p(y) = a (\log r + y) - r e^{y} - \log \Gamma(a).
$$

Both are exact for every finite argument, since no quantity is formed on
the constrained scale. Sampling stays in log space through the shape
augmentation of Marsaglia and Tsang [@marsaglia2000gamma]: for independent
$G' \sim \mathrm{Gamma}(a + 1)$ and $U \sim \mathrm{Uniform}(0, 1)$,
$\log G' + \log(U) / a$ has the law of $\log G$ with
$G \sim \mathrm{Gamma}(a)$, and a logit-Beta draw is the difference of two
such draws. $G'$ has shape at least one, so it never comes near the
underflow threshold, and its implicit reparameterization gradient, which
differentiates the distribution function instead of the sampler, carries through
[@figurnov2018implicit]. Draws are finite for every shape at or above 64 times the
smallest normal number of the dtype, and at shape `1e-8` the median draw of `LogGamma`
is near `-6.9e7`.

Because the support is the real line, a site drawn from either family
takes the identity as its bijector, so Hamiltonian Monte Carlo (HMC), the No-U-Turn Sampler (NUTS), and the guides see the exact
density, and prior draws, `rsample`, and initialization from the prior
all sample the declared prior. A downstream quantity on the constrained
scale is a `let`:

<!-- compile: standalone -->
```qvr
object Item : FinSet 20

program sticks : Item -> Item
    sample weight_logit : Item <- LogitBeta(concentration1=1.0, concentration0=0.01)
    sample log_rate <- LogGamma(concentration=0.001, rate=1.0)
    let weight = sigmoid(weight_logit)
    let rate = exp(log_rate)
    observe y : Item <- Normal(loc=weight, scale=1.0)
    return weight

export sticks
```

From Python, the distributions are ordinary `torch.distributions`
classes, and
[`make_fixed_logit_beta`][quivers.continuous.inline.make_fixed_logit_beta]
and [`make_fixed_log_gamma`][quivers.continuous.inline.make_fixed_log_gamma]
build the fixed program steps the compiler emits for literal parameters. The inline
paths pass small literal shapes through unchanged, and a shape that is not positive
raises.

```python
import torch
from quivers.continuous import LogGamma, LogitBeta

shape = torch.tensor(1e-3, dtype=torch.float64)
prior = LogitBeta(shape, torch.tensor(1.0, dtype=torch.float64))
prior.log_prob(torch.tensor(-800.0, dtype=torch.float64))  # about -7.71

draws = LogGamma(shape, torch.tensor(1.0, dtype=torch.float64)).rsample((5,))
assert torch.isfinite(draws).all()  # near -1000 on average, never clamped
```

The conditional forms
[`ConditionalLogitBeta`][quivers.continuous.families.ConditionalLogitBeta]
and [`ConditionalLogGamma`][quivers.continuous.families.ConditionalLogGamma]
learn their shapes as the softplus of a network output. The conditional Beta and Gamma
families add a positive shift to that softplus (0.1 for `ConditionalBeta`), so their
learned shapes cannot fall below it. [Transpiler support](../transpile-support.md)
lists the targets that accept the two log-space families.

### Multivariate families

```python
from quivers.continuous.families import (
    ConditionalMultivariateNormal,
    ConditionalLowRankMVN,
    ConditionalDirichlet,
    ConditionalWishart,
    ConditionalInverseWishart,
    ConditionalMatrixNormal,
    ConditionalLKJCholesky,
    ConditionalGaussianProcess,
    ConditionalHorseshoe,
)
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)

# Multivariate normal with learned mean and cov
mvn = ConditionalMultivariateNormal(domain, Euclidean(name="Y", dim=5))
```

### Discrete / categorical families

Count mixtures combine an exact zero with a discrete response. The registered
native constructors are:

| Family | Parameters |
| --- | --- |
| `ZeroInflatedNegativeBinomial` | `zero_prob`, `total_count`, `probs` |
| `ZeroInflatedBinomial` | `zero_prob`, `total_count`, `probs` |
| `ZeroInflatedBetaBinomial` | `zero_prob`, `total_count`, `concentration1`, `concentration0` |
| `HurdleNegativeBinomial` | `zero_prob`, `total_count`, `probs` |

For zero inflation, `zero_prob` selects a structural zero in addition to
zeros from the count component. For the hurdle family, it is the complete
zero probability; the other branch is a positive-truncated negative
binomial. Its component mean is `total_count * probs / (1-probs)` before
truncation. At `probs=0`, the positive branch uses its limiting count of one.
Binomial trial counts may be zero, in which case the response is always zero.
These scalar count samples are not reparameterized. Their log masses and
moments retain gradients with respect to the distribution parameters.

```qvr
object Row : FinSet 3
program counts : Row -> Row
    observe nb : Row <- ZeroInflatedNegativeBinomial(zero_prob=0.3, total_count=2.5, probs=0.4)
    observe bin : Row <- ZeroInflatedBinomial(zero_prob=0.3, total_count=8, probs=0.4)
    observe bb : Row <- ZeroInflatedBetaBinomial(zero_prob=0.3, total_count=8, concentration1=0.7, concentration0=1.1)
    observe hurdle : Row <- HurdleNegativeBinomial(zero_prob=0.3, total_count=2.5, probs=0.4)
    return nb
export counts
```

The [formula interface](analysis-data-and-formulas.md#zero-inflated-and-hurdle-counts)
provides mean/precision parameterizations and known trial counts. The
[native API](../api/continuous/families.md#zero-inflated-and-hurdle-counts)
gives the constructors and sampling behavior. Native and reference execution
support these families. All 11 external renderers currently refuse each
with a `family:<Family>` diagnostic; the 44 family/backend cells are checked
by `tests/transpile/test_family_matrix.py`. These family fixtures are separate
from the gallery and construct rows of the general support matrix.

Native Binomial, BetaBinomial, NegativeBinomial, Poisson, and the four count mixtures
use float64 parameters, log scores, and moments to avoid float32
cancellation at large counts. Gradients still reach the original learned
heads, including float32 heads. Finite trial counts are checked before
conversion and must lie between zero and `2**53`, inclusive. This bound
preserves their exact integer representation in float64; `Int` remains a
signed 64-bit carrier. Supply large counts as integer tensors or integer
source literals. Converting an already-rounded float32 count to float64
does not restore the original integer.
A source integer literal also retains its integer value through parsing,
static substitution, and family argument binding. Stable deviance,
Stirling, and gamma-increment formulas avoid subtracting nearly equal
large log-gamma values in the count kernels. Float64 remains an
approximation; these choices do not establish a uniform error bound for
every parameter regime.
A runtime header can declare `trials : Tensor[Int]([Row])` and pass it as
`Binomial(total_count=trials, probs=p)`; the checked integer input reaches
the count kernel before floating-point conversion.
A shared scalar count can instead use `(trials : Nat)`. A static
`[trials : Nat]` is an index extent; it cannot directly supply the
family's runtime `total_count` argument. The source checker reports this
stage mismatch explicitly.

Poisson, negative-binomial, and their inflated or hurdle samplers refuse
a generated intensity or count that is nonfinite or exceeds `2**53`.
Native execution raises `RuntimeError`; reference execution raises
`DistributionError`, a `ValueError` subtype. These failures do not
truncate or renormalize the law, or redraw to conceal overflow.
A structural-zero outcome bypasses its unreachable count branch, so a
zero-atom probability of one does not attempt to sample a huge base
count. The sampling ceiling is separate from the general `Int` carrier;
density and moment evaluation can still accept larger real-valued
negative-binomial shapes or Poisson rates.

```python
from quivers.continuous.families import (
    ConditionalBernoulli,
    ConditionalCategorical,
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
)
```

## Discrete Weibull counts and zero-atom quantiles

`DiscreteWeibull(probs=q, concentration=b)` models nonnegative integer
counts with survival probability `P(Y >= k) = q**(k**b)`. It requires
`0 <= q < 1` and positive `b`; `q=0` gives a point mass at zero.
The probability parameter is not a response mean. Its mean is the
sum of `q**(k**b)` over positive integers, as in the
[formal count definition](../semantics/programs.md#discrete-weibull-counts).

`ZeroInflatedAsymmetricLaplace(zero_prob, loc, scale, quantile)` combines
an exact zero atom with a continuous asymmetric Laplace response. The
atom probability may range from zero to one; `scale` is positive and
`quantile` lies strictly between zero and one. `loc` is that quantile
of the continuous component. It need not be the same quantile of the
whole response distribution after the zero atom is added.

At an exact zero, scoring returns `log(zero_prob)`. At a nonzero
response it returns `log(1-zero_prob)` plus the component log density.
The continuous density at zero is not added to the atom probability.
Its CDF includes the atom at zero; its strict survival probability
`P(Y > y)` excludes it at `y=0`. These conventions follow the
[mixed-kernel semantics](../semantics/programs.md#a-zero-atom-with-asymmetric-laplace-responses).

Both declarations below use ordinary runtime values for distribution
parameters and a static `n` for their common row extent. The typed
inputs distinguish count from real responses; the observation steps
give them likelihood roles.

```qvr
object Out : Real 1
program count_and_atom[n : Nat](
    q : Real,
    shape : Real,
    zi : Real,
    location : Real,
    scale : Real,
    quantile : Real,
    counts : Tensor[Int]([n]),
    residuals : Tensor[Real]([n]),
) : n -> Out
    observe counts : n <- DiscreteWeibull(probs=q, concentration=shape)
    observe residuals : n <- ZeroInflatedAsymmetricLaplace(zero_prob=zi, loc=location, scale=scale, quantile=quantile)
    return location
export count_and_atom
```

Native and checked-reference execution support both families; all 11
external renderers explicitly refuse them. Both samplers are
non-reparameterized. Discrete Weibull parameters, scores, and moments
use float64, while its draws are int64. Finite draws or requested finite
quantiles above `2**53` are refused; the law is not clipped or redrawn
to hide an unrepresentable result.

Discrete Weibull moments use adaptive sums and analytic omitted-tail
bounds. Native `moment_rtol` and `moment_max_terms` control tolerance
and work, defaulting to `1e-10` and `262144` count terms. If the requested
remainder bound cannot be established within that budget, moment
evaluation refuses explicitly. A mathematical finite mean does not
guarantee that a chosen numerical budget suffices.

## Truncating an independent response

`Truncated(base, lower=..., upper=...)` restricts a base family to an
interval and divides its density by the probability of that interval.
This normalizer depends on the base parameters. Censoring instead
records that a value falls in an interval; it contributes the interval
probability without changing the full response law.

For independent scalar coordinates, interval probabilities multiply,
and the corresponding log normalizers add. A coupled multivariate
distribution needs a joint rectangle probability, which cannot be
recovered from separate marginal CDFs. Missing CDF or rectangle-mass
capabilities must produce an explicit error.

The native wrapper reads the base's scalar-coordinate distributions and
computes deterministic interval masses. It uses log-CDF or log-survival
methods where supplied, stable Normal tails, and incomplete-gamma
calculations for Gamma and Poisson. Other scalar bases need a working
CDF; unsupported bases and coupled events raise explicit errors. An
interval with zero or nonfinite computed probability is rejected.

Normal sampling uses a stable inverse CDF, including far-tail intervals.
Other bases use bounded rejection, with `max_rejection_iterations`
controlling failure. That rejection path does not establish a general
pathwise derivative through a parameter-dependent acceptance event.

```python
import torch
from quivers.continuous.families import ConditionalNormal, Truncated
from quivers.continuous.spaces import Euclidean

base = ConditionalNormal(
    Euclidean(name="Input", dim=2), Euclidean(name="Response", dim=1)
)
restricted = Truncated(base, lower=0.0, upper=1.0)
inputs = torch.zeros(3, 2)
responses = restricted.rsample(inputs)
log_density = restricted.log_prob(inputs, responses)
assert responses.shape == (3, 1)
assert torch.isfinite(log_density).all()
```

The Gamma interval normalizer supports shape and rate derivatives through
converged float64 tail recurrences. Its lower-tail series, upper-tail
continued fraction, and small-shape complement calculations have a
512-iteration convergence limit; nonconvergence raises explicitly. This
policy supplies no uniform accuracy or convergence guarantee across all
parameter regimes. Derivative support remains a separate question for
each other base family.

## Event and batch axes {#event-rank-and-axis-role-syntax}

Three kinds of axis appear in a drawn tensor [@pyroTensorShapes]. A *sample* axis counts
repeated draws, a *batch* axis counts separate distributions, and an *event* axis
holds coordinates scored jointly as one outcome. In the example above, `(100, 8, 3)` is
100 draws for eight inputs with three output coordinates, and `log_prob` sums over the
output coordinates and keeps one score per input. The
[`over` clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over) assigns
these roles to named QVR axes.

A family's number of event dimensions, `event_rank` in the code, counts its event axes,
not their length: a 100-dimensional multivariate normal has one and a matrix family has
two. Scalar normal draws over three coordinates are independent even when stored
together, while a multivariate normal can correlate them, and the `over` clause makes
that choice explicit:

| Family | Event dimensions | Interpretation |
|---|---|---|
| `Normal`, `Beta`, `Gamma`, `Exponential`, `LogNormal`, `LogitNormal`, `Cauchy`, `Laplace`, `AsymmetricLaplace`, `ExGaussian`, `InverseGaussian`, `HurdleGamma`, `HurdleLogNormal`, `Gumbel`, `StudentT`, `Weibull`, `Pareto`, `Kumaraswamy`, `ContinuousBernoulli`, `HalfNormal`, `HalfCauchy`, `InverseGamma`, `TruncatedNormal`, `Uniform`, `Bernoulli`, `Horseshoe` | 0 | Scalar; every codomain axis is iid by default |
| `SkewNormal`, `ShiftedLogNormal`, `Frechet`, `GeneralizedExtremeValue` | 0 | Scalar real response; extent and runtime distribution shape are separate |
| `DiscreteWeibull` | 0 | Nonnegative integer count; probability and concentration are runtime parameters |
| `ZeroInflatedAsymmetricLaplace` | 0 | Real response with an exact zero atom and a continuous component |
| `ZeroInflatedNegativeBinomial`, `ZeroInflatedBinomial`, `ZeroInflatedBetaBinomial`, `HurdleNegativeBinomial` | 0 | Integer count; every codomain axis is iid by default |
| `MultivariateNormal`, `LowRankMVN`, `Dirichlet`, `OneHotCategorical`, `RelaxedOneHotCategorical`, `LogisticNormal`, `GP` | 1 | Vector; one named event axis carries the joint distribution |
| `Wishart`, `InverseWishart`, `MatrixNormal`, `LKJCholesky` | 2 | Matrix; two named event axes carry the joint distribution |

On a `morphism` declaration, `[over=<axes>] ~ Family` must list exactly as many axes as
the family has event dimensions, and a mismatch is a compile error. A `sample` or
`observe` step may list more, the extra leading axes indexing independent draws
([`over` clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over)). A
multivariate normal over all $|A| \cdot |B|$ entries, with a dense covariance and one
event axis whose length is the product, is a different model from a `MatrixNormal` over
`(A, B)`, whose covariance is the Kronecker product $V \otimes U$ and which has two
event axes, so the declaration records both the event axes and the covariance
structure.

## Structured priors over weight matrices

### MatrixNormal: matrix prior with Kronecker covariance {#matrixnormal-kronecker-covariance-matrix-prior}

The rows of a weight matrix may covary because their inputs are related, and its
columns because their outputs are. A
[matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution)
$\mathcal{MN}(M, U, V)$ has a row covariance $U$ and a column covariance $V$, and the
covariance of its flattened entries is their
[Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product). This needs far
fewer parameters than a covariance over every entry and also allows fewer
correlations.

```python
from quivers.continuous.families import ConditionalMatrixNormal
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)

# Matrix-valued kernel: domain -> R^(rows*cols), with samples
# reshaped to (rows, cols) and Kronecker covariance Sigma = V (x) U.
mn = ConditionalMatrixNormal(
    domain, Euclidean(name="W", dim=4 * 8), rows=4, cols=8
)
```

As a prior on a learnable table in QVR,
`morphism W : D -> K [role=parameter, over=[dom, cod]] ~ MatrixNormal(loc=0.0, row_covariance=1.0, col_covariance=1.0)`,
the first axis in `over` takes the row covariance and the second the column
covariance ([Latent morphism prior](dsl-declarations.md#latent-morphism-prior)).

### InverseWishart: conjugate covariance prior

```python
from quivers.continuous.families import ConditionalInverseWishart
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)

# Conjugate prior on a d-dim covariance matrix.  Realized as the
# inversion of a Wishart sample with the correct symmetric-matrix
# change-of-variables Jacobian.
d = 4
iw = ConditionalInverseWishart(domain, Euclidean(name="Sigma", dim=d))
```

The inverse Wishart is the [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior)
for the covariance of a multivariate normal, so with normal data its posterior is again
an inverse Wishart [@gelman2013bayesian]. The change-of-variables Jacobian for inverting a positive-definite
symmetric matrix contributes a `-(d + 1) log det(Sigma)` term to
the log-density.

### LKJCholesky: prior on correlation matrices {#lkjcholesky-correlation-matrix-prior}

A correlation matrix describes association apart from each variable's scale, and its
[Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition) is the
lower-triangular $L$ with $L L^\top$ equal to it. Placing the prior on the factor means
every draw is a valid correlation matrix, and separate scale parameters then give the
marginal variances.

```python
from quivers.continuous.families import ConditionalLKJCholesky
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
d = 4

lkj = ConditionalLKJCholesky(
    domain, Euclidean(name="L", dim=d),
)
```

The family draws Cholesky factors of correlation matrices from the
[LKJ distribution](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution)
[@lewandowski2009lkj], with density proportional to $\det(R)^{\eta - 1}$ for the
correlation matrix $R$ and concentration $\eta$, so $\eta = 1$ is uniform over
correlation matrices. The matrix dimension is `codomain.dim`, and the parameter source
learns $\eta$ from the input.

Draws use the onion method of section 3.2 of
[@lewandowski2009lkj]: row
$k$ of the factor (counting from zero) is
$(\sqrt{y}\, u, \sqrt{1 - y})$ with
$y \sim \mathrm{Beta}(k/2, \eta + (d - 1 - k)/2)$ and $u$ uniform on
the unit sphere in $\mathbb{R}^k$. Every off-diagonal correlation $r$
of a $d \times d$ draw thus has
$(r + 1)/2 \sim \mathrm{Beta}(\eta - 1 + d/2, \eta - 1 + d/2)$. The
[compiled runtime](dsl-overview.md#compilation-pipeline), every factory over the family (the fixed and
forms with a variable concentration, a plate over it, and
[`LKJCorrelationFactor`][quivers.continuous.families.LKJCorrelationFactor]),
and the [reference machine](../developer/qiec.md#reference-evaluator), the library's interpreter for its core language, all draw through one implementation of
this construction, so a program's prior draws follow the same law in
both runtimes. `rsample` is differentiable in the concentration
through implicitly reparameterized Beta draws, and the directions $u$ do not depend
on it. The log density is torch's `LKJCholesky.log_prob`.

### Wishart

```python
from quivers.continuous.families import ConditionalWishart
from quivers.continuous.spaces import Euclidean

domain = Euclidean(name="X", dim=5)
d = 4

wishart = ConditionalWishart(
    domain, Euclidean(name="Sigma", dim=d),
)
```

## Non-parametric and structured families

### ConditionalGaussianProcess

A [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) on $\mathbb R^D$
gives the function values at any finite set of inputs $x_1, \dots, x_N \in \mathbb R^D$
a joint multivariate normal distribution with covariance $K(x_i, x_j)$
[@rasmussen2006gp]. As a family it is a kernel from the $N$ input locations to
distributions on $\mathbb R^N$, and each evaluation builds the covariance matrix from
all the locations at once, so moving one location changes its covariance with the
others. With an RBF covariance kernel, nearby inputs get more strongly correlated
values, and the length scale sets how fast the correlation decays.
[`ConditionalGaussianProcess`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess)
exposes three covariance kernels (RBF, Matern 5/2, linear) with
learnable length scale and amplitude.

```python
from quivers.continuous.families import ConditionalGaussianProcess
from quivers.continuous.spaces import Euclidean
import torch

D, N = 2, 8
gp = ConditionalGaussianProcess(
    Euclidean(name="X", dim=D),
    Euclidean(name="Y", dim=N),
    kernel="rbf",
    length_scale=0.5,
    amplitude=1.0,
)
x = torch.randn(N, D)            # N input locations in R^D
f = gp.rsample(x)                # one GP draw at those inputs, shape (N,)
log_p = gp.log_prob(x, f)        # MVN(0, K(x, x) + jitter * I) density
```

QVR registers the family as `GP`, with one event dimension: the axis of the $N$ input
locations carries the joint distribution.

### ConditionalHorseshoe

The horseshoe prior [@carvalho2010horseshoe] prefers small coefficients and still
allows some large ones. In its hierarchical form below, the global scale $\tau$ sets
the overall shrinkage, and each local scale $\lambda_d$ can let its coefficient escape
it. The heavy tails of the scale priors are what distinguish it from a small normal
variance for every coefficient:

$$
\begin{aligned}
\tau &\sim \mathrm{HalfCauchy}(\text{scale}) \\
\lambda_d &\sim \mathrm{HalfCauchy}(1) \quad \text{for } d = 1, \dots, D \\
\beta_d \mid \tau, \lambda_d &\sim \mathcal{N}\!\bigl(0, (\tau \lambda_d)^2\bigr).
\end{aligned}
$$

The marginal density of $\beta_d$ after integrating the local scale
$\lambda_d$ has no elementary closed form.
[`ConditionalHorseshoe`](../api/continuous/families.md#quivers.continuous.families.ConditionalHorseshoe)
approximates the marginal with 16-point
[Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature)
after the change of variables $\lambda = \tan(\pi t / 2)$ which
maps $(0, \infty)$ to $(0, 1)$ with Jacobian
$(\pi / 2) \sec^2(\pi t / 2)$.

```python
import torch
from quivers.continuous.families import ConditionalHorseshoe
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet

hs = ConditionalHorseshoe(
    FinSet(name="A", cardinality=4),
    Euclidean(name="beta", dim=10),
    scale=0.1,
)
x = torch.tensor([0, 1, 2, 3])
beta = hs.rsample(x)             # shape (4, 10)
log_p = hs.log_prob(x, beta)     # shape (4,), summed over coordinates
```

QVR registers the family as `Horseshoe` with no event dimensions: each coefficient is
one draw, scored by its marginal density, and draws over several coefficients are
independent.

### Choosing among multivariate priors

| Goal | Family | Why |
|---|---|---|
| Multivariate prior with a dense covariance | `ConditionalMultivariateNormal` | Standard; learnable Cholesky factor. |
| Weight matrix with separable row and column correlations | `ConditionalMatrixNormal` | The Kronecker covariance $V \otimes U$ has on the order of $r^2 + c^2$ parameters instead of $(rc)^2$. |
| Covariance prior for a downstream MVN | `ConditionalInverseWishart` | Places a distribution on positive-definite covariance matrices. |
| Correlation matrix in a hierarchical model | `ConditionalLKJCholesky` | Prior on correlations alone, leaving scales free. |
| Non-parametric regression with smoothness control | `ConditionalGaussianProcess` | Prior over functions; bandwidth from kernel length-scale. |
| Sparse signals, such as variable selection in regression | `ConditionalHorseshoe` | Strong shrinkage toward zero with heavy tails for large signals. |

## See also

- [Continuous Spaces and Morphisms](continuous-spaces.md): the
  surrounding space hierarchy and composition operations.
- [DSL Programs and Let-Expressions](dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over):
  the `over` and `iid_over` clause, which depends on the number of event dimensions.
- [Examples Gallery: Horseshoe Regression](../examples/horseshoe-regression.md):
  end-to-end use of `ConditionalHorseshoe` in a regression.
