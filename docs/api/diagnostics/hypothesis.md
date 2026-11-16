# Contrast summaries and region evidence

`hypothesis_summary` analyzes a retained contrast tensor with shape
`(chain, draw, *event_shape)`. Construct derived contrasts within each draw:
subtracting two retained coefficient tensors preserves their joint dependence.
The function does not sample a guide, replay a model, parse an expression, or
estimate a density at a point.

<!-- python: run -->
```python
import math
import torch
from quivers.diagnostics import hypothesis_summary

contrast = torch.tensor([[-2., -1., 0., 0., 1., 2., 3., 4.]])
prior_contrast = torch.tensor([[-3., -2., -1., 0., 0., 0., 0., 1.]])
result = hypothesis_summary(contrast, prior=prior_contrast, kind="mcmc")
assert result.above.probability.item() == 0.5
assert result.equal_probability.item() == 0.25
assert result.above.log_odds.item() == 0.0
assert abs(result.above.log_bayes_factor.item() - math.log(7)) < 1e-12
```

## Uncertainty and coordinates

The default `estimate` and `est_error` are the pooled mean and sample standard
deviation, with divisor `S - 1`. Set `robust=True` for the median and
normal-consistent median absolute deviation, that is, MAD divided by the
standard normal's 0.75 quantile. `probs` selects linearly interpolated type-7
quantiles in the requested order; an empty sequence omits them. Summaries
retain the contrast's original units: `null` sets the comparison boundary and
is not subtracted from the summarized values. A single draw has undefined
sample SD, reported as NaN; a constant multi-draw coordinate has zero SD.

Event axes remain separate. `name`, `dims`, and `coords` follow the shared
[retained-analysis coordinate policy](posterior.md#joint-covariance-and-correlation).
The result retains event dimension names and labels, chain and draw labels,
and original sample counts. Prior draws may have different chain and draw
counts, but must have exactly the same event shape and device. Construct the
prior contrast jointly as well; independently permuting its parameter draws
would change a dependent prior. The caller supplies `kind`; this records
provenance without adding convergence diagnostics or correcting an approximate
posterior.

Inputs are detached, finite real tensors. Boolean and integer draws use the
shared Float64 conversion policy, including refusal of original integers
outside the exact range before conversion. Missing values are not deleted.
Scaled arithmetic preserves finite extreme locations and small spreads;
dispersion overflow causes an explicit error, while values below Float64
resolution may underflow to zero. Returned tensors remain
on the original device. A tensor null must share that device and broadcast
only over event axes, never across chain or draw axes.

## Region odds and Bayes factors

`below` describes the strict region $R_- = \{c < c_0\}$ and `above` the strict
region $R_+ = \{c > c_0\}$. Their complements include exact ties. Each record
returns an empirical posterior probability and its natural-log odds. Thus
these two odds are not generally reciprocals when ties occur.
`equal_probability` reports the frequency of $c=c_0$ separately.

With aligned prior contrasts, a region record also returns prior probability,
prior log odds, and their difference:

$$
\log \widehat{\mathrm{BF}}_{R:R^c}
= \log\frac{\widehat P(R\mid y)}{1-\widehat P(R\mid y)}
- \log\frac{\widehat P(R)}{1-\widehat P(R)}.
$$

This is an empirical encompassing-prior comparison. Its Bayes-factor
interpretation requires a common proper prior restricted and renormalized
to $R$ and $R^c$, with the same likelihood under both models. Both regions
must have positive prior mass. Supplied draws must represent that prior and
its corresponding posterior, including their hierarchical dependence.
A variational posterior may introduce approximation error into the estimate.
Posterior region odds alone are not a prior-corrected Bayes factor.

No pseudocounts are added. Empirical posterior probabilities zero and one
produce negative and positive infinite log odds; these finite-sample results
do not establish exact population probabilities. If the empirical prior
probability is zero or one, `log_bayes_factor` is NaN. Such draws cannot
estimate both restricted priors, even when the true prior assigns positive
mass to both regions. Without prior draws, prior fields are `None`.

Exact equality is descriptive. Neither a continuous point null nor a
comparison between an atom and a continuous alternative is inferred from
an equality count. This API supplies no point-null density estimate,
Savage–Dickey ratio, or posterior model probability; those require additional
null-model and prior-compatibility assumptions.

## Checked expressions over retained draws

`PosteriorAnalysis.hypothesis` evaluates a QVR value expression over its
retained sites, then calls the same summary function. `ContrastExpression`
maps expression aliases to original labels when a label is not a QVR
identifier. The expression is checked once with QIEC and evaluated with
chain and draw axes protected: `sum(a)` reduces an event axis rather than
pooling samples. Named input event axes remain nominal; equal lengths alone
do not make two axes interchangeable.

<!-- python: run -->
```python
from quivers.diagnostics import ContrastExpression, PosteriorAnalysis

posterior = PosteriorAnalysis(
    samples={"condition[red]": torch.tensor([[0.2, 0.4, -0.1, 0.3]]),
             "baseline": torch.tensor([[0.1, 0.2, 0.0, 0.1]])},
    log_likelihood={},
    kind="variational",
)
expression = ContrastExpression(
    expression="exp(red)-baseline", bindings={"red": "condition[red]"}
)
contrast = posterior.hypothesis(expression, name="red effect")
assert contrast.sample_shape == (1, 4)
assert contrast.dimension_names == ()
```

Use `dims={"a": ("Item",), "b": ("Item",)}` to declare a shared event
axis explicitly. Omitting `dims` assigns distinct site-specific axes;
subtracting vectors on distinct axes fails at checking. `coords` supplies
their original labels and optional chain/draw labels. A corresponding
`prior=other_analysis` evaluates the same checked expression on its joint
draw stream, with no parameter permutation. Different sample counts are
allowed; event shapes and devices must match. These operations reuse
retained tensors and do not refit a model.

::: quivers.diagnostics.contrast

::: quivers.diagnostics.hypothesis
