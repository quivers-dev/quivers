# Updating and refitting models

A retained refit recipe records how to fit a fresh posterior. It shares the
existing fitting operation and its configuration; it does not change the
posterior already stored in a fit. Prediction continues to use the fitted
parameter layout, while a refit may learn a different layout from replacement
training data.

## Source models

`Model.fit` and `BoundModel.infer` attach a recipe when an inference method
returns `ModelFit`. `update` retains the original source, entry, static
arguments, domain input, compilation seed, and prior-scoring policy. It reuses
the checked source preparation and creates fresh native compiler state.

<!-- python: run -->
```python
import torch
from quivers.modeling import Model, VariationalMethod

source = """object Row : FinSet _
object One : FinSet 1
program regression(x:Tensor[Real]([Row]), y:Tensor[Real]([Row])) : One -> One
    sample slope <- Normal(loc=0.0,scale=1.0)
    let mu = slope * x
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return slope
export regression
"""
model = Model(source)
fitted = model.fit(
    {"x": torch.tensor([1., 2., 3., 4.]), "y": torch.tensor([.5, 1., 1.5, 2.])},
    method=VariationalMethod(num_steps=20, seed=3),
)
updated = fitted.update(
    {"x": torch.tensor([1., 2.]), "y": torch.tensor([.4, .9])},
    method=VariationalMethod(num_steps=30, seed=5),
)
assert updated is not fitted
assert updated.model.data["y"].shape == (2,)
assert fitted.model.data["y"].shape == (4,)
```

The source `seed` argument controls compilation. An inference method owns its
sampling or optimization seed. Updating only the compilation seed leaves the
method's seed unchanged. Methods with a custom result type continue to return
that result; they are not converted into posterior fits or given a recipe.

## Data ownership

The default `snapshot_policy="clone"` retains an independent copy of training
data. Tensor snapshots are detached, so subsequent caller mutations and
autograd graphs do not enter a later refit. Reading `refit_recipe.data` returns
another isolated copy. This costs one retained data copy per fit, independent
of its draw count.

Use `snapshot_policy="reference"` explicitly when the caller owns and freezes
the data, or intentionally wants later mutations to affect a refit. Inference
configurations and callbacks remain live objects held by identity. A recipe is
not a durable fitted artifact: serializing its live fitting operation is
refused. Existing MCMC checkpoints retain their separate exact-resume contract.

## Explicit refit partitions

`refit_callback` adapts a retained recipe to the existing
[`cross_validate`](../api/diagnostics/cross_validation.md#quivers.diagnostics.cross_validation.cross_validate)
runner. A source convenience adapter partitions inputs using a checked nominal
row axis. An unrelated tensor with the same extent stays unchanged. A fixed
finite extent remains fixed; declare an inferred extent when refits need fewer
training rows.

Each fold requires an explicit inference configuration policy. The callback
passes its split seed to that policy and to compilation; it does not silently
replace a seed inside an opaque custom method.

<!-- python: run -->
```python
from quivers.diagnostics.cross_validation import FoldPlan, cross_validate

callback = fitted.refit_callback(
    row_axis="Row",
    method_factory=lambda original, seed: original.with_(seed=seed),
    num_samples=32,
)
comparison = cross_validate(
    FoldPlan(assignments=(0, 0, 1, 1), seed=11), callback,
)
assert comparison.complete
assert comparison.log_predictive_density.shape == (4,)
```

For another data backend or partition geometry, supply an explicit nonmutating
partitioner to the source adapter, or use the shared
[`refit_callback`](../api/modeling.md#quivers.modeling.refit.refit_callback)
with a recipe and raw held-out likelihood accessor. The callback snapshots its
original data once, then builds the requested training and test partitions.
It delegates uncertainty and joint predictive scoring to the existing runner.
`save_fits=True` retains actual fold fits and their data; its additional storage
is explicit. Refit accuracy still depends on the chosen inference method and
its budget.

## Refitting influential observations

`PosteriorAnalysis.reloo` connects the same callback to ArviZ's exact-refit
interface. It first computes PSIS-LOO from retained raw likelihoods, then refits
observations whose Pareto shape exceeds `k_threshold`, including undefined
tail fits. With the default
`None`, ArviZ chooses a threshold from the retained sample count. Explicit
`-float("inf")` requests a refit for every row. This removes the importance
sampling approximation for those rows; the inference method and its finite
draw budget still determine refit accuracy.

<!-- python: run -->
```python
analysis = fitted.analysis(num_samples=32, seed=23)
criterion = analysis.reloo(callback, k_threshold=-float("inf"), seed=11)
assert criterion.elpd_i.shape == (4,)
assert (criterion.pareto_k == 0).all()
```

Each split excludes exactly one original included row. Named observation
coordinates stay attached to the ArviZ result, while the callback receives
positional row identities. A response event must already be reduced to one
score per row; with multiple responses, select one explicitly using `var_name`.
Whole-group omission or joint response scoring uses `cross_validate` with an
explicit fold plan, since it changes the prediction question. Neither adapter
multiplies held-out densities by training likelihood weights.

## Formula models

`BayesianFit.update` accepts replacement raw data and overrides of the existing
`fit` arguments, including the formula, family, priors, method, and budgets.
The recipe captures every argument through that fitting signature. Unknown
overrides fail before model preparation. A method change clears inherited
budgets belonging to the previous method; explicitly supplied incompatible
budgets still raise the usual fitting error.

<!-- python: run -->
```python
import pandas as pd
from quivers.formulas import fit

frame = pd.DataFrame({"y": [.3, .6, .2, .8], "x": [-1., 0., 1., 2.]})
regression = fit("y ~ x", data=frame, method="svi", num_steps=2, seed=7)
intercept_only = regression.update(formula="y ~ 1", num_steps=3)
assert len(intercept_only.formula.fixed_columns) == 1
assert len(regression.formula.fixed_columns) == 2
```

Fresh fitting evaluates transforms, contrasts, group coding, default prior
scales, and training-domain constraints on its own training rows. Prediction
continues to use its fitted encodings and retained cutpoints. Neural recipes
retain the original predictor state and predictor inputs before training or
subset selection. Every refit copies that initial state, so it can train a new
predictor without changing an earlier fit.

`BayesianFit.refit_callback` partitions the retained raw frame and evaluates
the existing held-out likelihood accessor. Its split coordinates refer to the
rows included in the original fit, in their fitted order:

<!-- python: run -->
```python
formula_callback = regression.refit_callback(
    method_factory=lambda settings, seed: {**settings, "seed": seed},
    num_samples=8,
)
formula_comparison = cross_validate(
    FoldPlan(assignments=(0, 0, 1, 1), seed=13), formula_callback,
)
assert formula_comparison.complete
```

With `subset(keep)`, the callback maps each included-row position back to its
original raw row before partitioning. It preserves dataframe backends and
selects by position, independent of dataframe index labels. Retained neural
`predictor_data` is an explicit row attachment: the same original-row selection
supplies fitting inputs and held-out inputs. Neural parameters remain part of
the retained initial predictor state, even when a parameter dimension happens
to equal the row count. The method factory chooses the fitting seed and budget;
the split seed separately controls held-out posterior replay.

The callback validates row alignment and finite neural inputs before fitting.
Pass a nonmutating `partition` function for another frame-selection policy.
Response roles remain on the selected raw rows, so held-out censoring and
truncation retain their recorded-event meaning. Prediction keywords such as
`re_formula`, `allow_new_levels`, and `sample_new_levels` pass to the existing
raw likelihood accessor; new grouping identities still require an explicit
policy. The adapter never substitutes posterior predictive samples for held-out
log probabilities.
