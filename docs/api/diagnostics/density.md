# Density ordinates and point-null evidence

`log_density_ordinate` estimates a marginal density at a specified value from
retained floating contrast draws. `density_ratio` compares posterior and
joint-prior ordinates at the same null. Both use PyTorch, retain event labels,
and leave inference to the caller. They do not estimate a posterior model
probability.

## Direct Gaussian estimates

For each scalar event coordinate, the estimator is

$$
\widehat p_h(c_0)=\frac{1}{Nh}\sum_{s=1}^N
\phi\!\left(\frac{c_0-c_s}{h}\right),
$$

where $\phi$ is the standard Normal density. Input shape is
`(chain, draw, *event)`: chain and draw axes are pooled, while each event
coordinate receives its own marginal estimate. These marginal ordinates do
not supply a joint density at a vector null.

The default `bandwidth="nrd0"` uses
$0.9\min(\operatorname{SD},\operatorname{IQR}/1.34)N^{-1/5}$, with SD replacing
a zero IQR for nonconstant draws. This is the
[R bandwidth rule](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/bandwidth.html)
used by the default Gaussian estimator in the
[pinned brms density-ratio implementation](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/hypothesis.R#L446-L473).
QVR evaluates the Gaussian sum directly. It does not reproduce brms's
[FFT-grid interpolation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/density.html),
cubic spline evaluation, or negative-value clipping.

`bandwidth="scott"` instead uses $\operatorname{SD}N^{-1/5}$, following the
one-dimensional covariance-factor convention of
[SciPy's `gaussian_kde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).
A positive scalar or event-shaped tensor supplies an absolute bandwidth.
`adjust` multiplies the selected bandwidth. The same rule is applied
separately to posterior and prior draws, using each distribution's own scale
and count; `prior_bandwidth` can override the prior rule or absolute value.
All draw weights are equal. The automatic rules use retained draw counts,
without replacing them by an estimated MCMC effective sample size.

<!-- python: run -->
```python
import math
import torch
from quivers.diagnostics import (
    PointNullAssumptions, density_ratio, log_density_ordinate,
)

generator = torch.Generator().manual_seed(19)
# theta ~ Normal(0, 1); y=0.6 with known Normal noise SD=0.5.
# The posterior is Normal(0.48, sqrt(0.2)); there are no nuisance parameters.
prior = torch.randn(1, 4096, generator=generator, dtype=torch.float64)
posterior = 0.48 + math.sqrt(0.2) * torch.randn(
    1, 4096, generator=generator, dtype=torch.float64,
)
ordinate = log_density_ordinate(posterior, at=0., bandwidth="scott")
evidence = density_ratio(
    posterior, prior, null=0., bandwidth="scott",
    assumptions=PointNullAssumptions(
        proper_continuous_prior=True, compatible_null_prior=True,
    ),
)
assert ordinate.num_samples == 4096
assert torch.isfinite(evidence.log_bayes_factor_01)
assert evidence.posterior.log_bandwidth.shape == ()
```

This example declares the assumptions from its stated conjugate model;
the arrays alone could not establish them. The result remains a KDE estimate.
Bandwidth sensitivity and chain diagnostics matter, especially in tails.

## Density ratios and their assumptions

`DensityRatio.log_ratio` always means posterior log density minus prior log
density. Its separate `log_bayes_factor_01` field is supplied only when both
`PointNullAssumptions` declarations are true:

1. The encompassing prior is proper and induces a continuous contrast with a
   finite, positive density at the null; the posterior also has a density
   there, and neither law has an atom at the null.
2. The null model uses the encompassing likelihood restricted to that null,
   and its nuisance prior equals the encompassing prior conditional on the
   null, under the same dominating measures.

Under these assumptions, the Savage–Dickey comparison is

$$
\log\mathrm{BF}_{01}=\log p(c_0\mid y)-\log p(c_0).
$$

The declarations are retained, without mechanical verification. A proper
parameter prior alone does not establish a compatible nuisance prior or a
finite contrast density. A singular nonlinear map, an atom at the null,
or a null outside the prior's support can invalidate the argument.
Gaussian smoothing across a support boundary does not establish positive
underlying prior density there. Without both declarations, the empirical
`log_ratio` remains available and `log_bayes_factor_01` is `None`.

Posterior and prior contrast tensors may have different sample counts,
but must have identical event shapes and devices. Construct a derived prior
contrast jointly within each draw, preserving hierarchical dependence.
No parameter draws are permuted. Exact equalities and integrated region odds
are separate quantities; see the [contrast summary](hypothesis.md).
Neither an equality frequency nor a one-sided region probability is a
point-density ordinate.

A common smooth one-to-one change of scalar coordinates cancels its nonzero Jacobian
from the population density ratio. KDE estimates under an affine change
retain that identity when bandwidths change by the same scale. For a
nonlinear change, finite-bandwidth smoothing is coordinate-dependent;
transforming and re-estimating need not give exactly the same result.

## Diagnostics, capacity, and storage

Each `DensityOrdinate` retains `log_density`, `log_bandwidth`, the absolute
`bandwidth`, the selected rule and adjustment, and original sample counts.
`name`, `dims`, and `coords` use the shared
[event-coordinate policy](posterior.md#joint-covariance-and-correlation).
`kernel_effective_count` reports
$(\sum_s w_s)^2/\sum_s w_s^2$ for local Gaussian weights. It describes
local weight concentration, without measuring MCMC mixing or approximation
error. None of these values supplies a universal KDE accuracy guarantee.

Inputs must be finite floating tensors with at least two pooled draws and
nonzero spread on every coordinate. Known Boolean/integer samples, missing
values, constant coordinates, and invalid bandwidths fail explicitly.
No missing draws are deleted and no jitter is added. Floating storage or
occasional repeated values does not establish continuity; that is part of
the caller's distributional argument. Two distinct draws are accepted, but
the interface does not assert that this is enough for reliable evidence.

The estimator reuses shared scaled summary arithmetic for automatic
bandwidths. Kernel evaluation tiles both sample and event axes, with
`max_working_elements` bounding each temporary kernel tile. The budget is a
positive integer, excluding Boolean values. It does not bound input/output
storage or the shared bandwidth workspace. The `nrd0` quartiles require a
sort; Scott bandwidth selection avoids that sort.

Direct sums retain Gaussian factors around a nearby sampled value. Ratio
calculation factors the quadratic difference before subtraction, so
recomputing it from two rounded displayed log densities may lose tail
precision. Bandwidths remain authoritative in log space: exponentiation
may underflow to zero or overflow to infinity even when the log estimate
is representable. A log score beyond Float64 capacity causes an explicit
error, rather than a false zero-density estimate.

All numerical outputs are detached Float64 tensors on the original device.
There is no inference replay or density-gradient promise. Returned tensor
storage remains caller-owned; copying and serialization retain the nested
ordinate records and their opaque tensors. CUDA behavior is conditional
on a Float64-capable device; this API does not claim MPS Float64 support.


## Requests from checked hypotheses

`PointNullSpec` packages the existing bandwidth, assumption, and workspace
options for `hypothesis_summary`, `PosteriorAnalysis.hypothesis`, and
`BayesianFit.hypothesis`. The result places this comparison in
`HypothesisSummary.point_evidence`, alongside unchanged region and equality
summaries. The checked-expression methods require a Real result; the formula
method also requires `prior=True`. All methods reuse the same joint contrast
streams for both summaries, preserving hierarchical dependence.

<!-- python: run -->
```python
import math
import torch
from quivers.diagnostics import PointNullSpec, PosteriorAnalysis

generator = torch.Generator().manual_seed(29)
prior = PosteriorAnalysis(
    samples={"theta": torch.randn(1, 1024, generator=generator)},
    log_likelihood={}, kind="prior",
)
posterior = PosteriorAnalysis(
    samples={"theta": 0.48 + math.sqrt(0.2) * torch.randn(
        1, 1024, generator=generator,
    )},
    log_likelihood={}, kind="mcmc",
)
comparison = posterior.hypothesis(
    "theta", prior=prior,
    point=PointNullSpec(bandwidth="scott", max_working_elements=128),
)
assert torch.isfinite(comparison.point_evidence.log_ratio)
assert comparison.point_evidence.log_bayes_factor_01 is None
```

This request returns a diagnostic density ratio. To request a point-null Bayes
factor, supply `assumptions=PointNullAssumptions(...)` in the configuration,
with both declarations justified by the model described above. Each event
coordinate remains a separate scalar comparison. The returned `log_ratio`
retains the factored tail calculation; subtracting the displayed ordinate
fields again may lose precision.

::: quivers.diagnostics.density
