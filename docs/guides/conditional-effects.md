# Conditional effects and empirical marginalization

A conditional effect evaluates a fitted model while varying specified
predictors and holding the remaining predictors at explicit values. This
workflow has two parts: (i) construct a bounded prediction grid; and (ii) retain
the joint predictions before summarizing their uncertainty. The
[brms conditional-effects interface](https://search.r-project.org/CRAN/refmans/brms/html/conditional_effects.brmsfit.html)
likewise supplies conditioning rows for predictor effects. QVR makes those rows
explicit, so selecting a reference category or a numeric mean remains visible.

## Condition on selected predictor values

`prediction_grid` crosses named effect coordinates with each row of
`conditions`. The insertion order of the named axes determines point order;
the conditioning rows retain their original order within each point. Grid
coordinates replace matching columns in the conditioning frame.

Core source workflows can supply `conditions={"z": [-2., 2.]}` instead of a
dataframe; `conditions={}` specifies one empty row for a single-predictor grid.
This mapping interface and `grid.to_mapping()` need only the core dependencies.
Dataframe inputs use the selected Narwhals backend without conversion to pandas.
The formula adapter's `grid.to_frame()` uses pandas from the formula frontend's
dependencies.

<!-- python: run -->
```python
import pandas as pd
from quivers.diagnostics import prediction_grid
from quivers.formulas import fit

data = pd.DataFrame({
    "y": [0., 1., 0., 1.],
    "x": [-1., 0., 1., 2.],
    "z": [-2., 2., -2., 2.],
})
fitted = fit("y ~ x + z", data=data, family="bernoulli",
             method="svi", num_steps=2, seed=7)
grid = prediction_grid(
    {"x": [0., 1., 2.]}, conditions=pd.DataFrame({"z": [-2., 2.]}),
)
effects = fitted.conditional_effects(grid, num_samples=5, seed=11)
assert effects.draws.shape == (1, 5, 3, 2)
uncertainty = effects.summary(probs=(0.05, 0.5, 0.95))
assert uncertainty.mean.shape == (3, 2)
```

The axes are `(chain, draw, point, condition, *event)`. The singleton leading
axis of a variational fit is a layout axis; it does not supply MCMC convergence
diagnostics. Categorical or ordinal expected responses retain a final category
axis and its `effects.labels`.

The default `method="epred"` evaluates expected responses. Use
`method="linpred"` for linear predictors, with optional `parameter` and
`transform`, or `method="predict"` for response replications. Replications
include response noise, so their uncertainty differs from uncertainty about
the expected response. The existing prediction methods retain their response,
trial-count, and finite-moment checks.

## Average over an empirical reference population

Empirical marginalization evaluates each point at every supplied reference
row, then averages those predictions within each posterior draw. It thus
retains the joint uncertainty of the average. This generally differs from
conditioning on mean predictor values when the model includes a nonlinear link
or interaction.

<!-- python: run -->
```python
reference = prediction_grid(
    {"x": [0., 1., 2.]}, conditions=data[["z"]], marginalize=True,
)
marginal = fitted.conditional_effects(reference, num_samples=5, seed=11)
assert marginal.draws.shape == (1, 5, 3)
paired_change = marginal.draws[:, :, 2] - marginal.draws[:, :, 0]
assert paired_change.shape == (1, 5)
joint = marginal.analysis()
covariance = joint.covariance()
assert covariance.covariance.shape == (3, 3)
```

The reference-row axis disappears after averaging. Optional
`weights=[...]` in `prediction_grid` supplies finite, nonnegative empirical
weights with positive total; the grid normalizes them for this average. These
weights describe the reference population and do not alter the fitted
likelihood. Averaging interval endpoints or independently resampling each grid
point would discard the dependence retained by this workflow.

## Retain prediction policies and bound allocation

Each effects request calls the selected posterior prediction method once for
all grid rows. Fitted transforms, contrasts, offsets, auxiliary formulas,
trial counts, and grouping encodings follow the ordinary new-data contract.
`re_formula`, `allow_new_levels`, and `sample_new_levels` have their existing
meanings. Repeated group identities across points share each joint group draw.
Setting `re_formula="~0"` excludes varying effects; it does not integrate over
their distribution. Empirical averaging integrates only over the reference
rows that were explicitly supplied.

Supply any retained sampling-law columns needed by the prediction, including
exposures, truncation bounds, measurement errors, or trial columns. Neural
predictors require `predictor_data` aligned with the entire generated grid.
Fresh groups follow the fitted prediction policy, including its order-support
checks for constrained ordinal models.

Before allocating the Cartesian product, `prediction_grid` checks
`number_of_points * number_of_conditions <= max_rows`; the default limit is
10,000. A larger request requires an explicit larger limit. This check bounds
the row expansion, while the requested draw count and response event dimensions
still determine prediction storage. Grid coordinates are finite scalar
numbers, strings, Boolean values, or explicit missing-group values `None`;
structured inputs can use the underlying prediction interface directly.

The retained result reuses
[posterior analysis](../api/diagnostics/posterior.md) for covariance and checked
contrasts. No additional fitting, convergence estimator, or contrast parser is
introduced. See the [conditional-effects API](../api/diagnostics/conditional.md)
for the grid and result records.
