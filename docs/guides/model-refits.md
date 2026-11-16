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
