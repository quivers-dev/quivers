# Refit cross-validation

A **fold plan** records which original rows a refit may use and which rows it
must predict. `fold_plan` constructs reusable random, grouped, or stratified
partitions; `cross_validate` applies an existing fitting procedure to those
partitions and scores its raw held-out likelihoods. Both need only PyTorch.

The result retains (i) the complete partition and selected folds, (ii) the original
rows in each predictive unit, and (iii) optional actual fitted results. This page
first specifies partition and scoring semantics, then applies the same runner to
QVR source and formula fits.

## Partitions and predictive units

`FoldPlan.assignments` contains one zero-based fold per original included row.
`fold_plan(n, folds=k, groups=labels)` keeps every group in one fold;
`strata=labels` balances strata across folds. A grouped stratum must be constant
within each indivisible group. Random tie breaking uses a local seeded stream.
Construction takes linear storage and uses a heap to balance groups, avoiding
work proportional to the number of folds for every row.

For leave-one-out prediction, use one fold per row. To leave out the rest of that
row's group as well, set `FoldPlan.omission_groups`: `split.test` still names the
one designated row, while `split.omitted` includes its whole group. Thus the
training set and prediction unit can express different questions.

A refit returns `FoldPrediction(log_likelihood=...)`, with unweighted response
scores of shape `(chain, draw, test_row)` in exactly `split.test` order. The
callback owns preprocessing, priors, basis construction, and inference quality.
In particular, fit learned preprocessing on training rows and apply the retained
transformation to held-out rows; splitting after fitting preprocessing may leak
information. An excluded group can leave no training rows, so that callback must
then support a prior fit.

`joint="observation"` scores each row separately. `joint="fold"` scores the whole
held-out fold together; `joint="group"` and explicit `groups` score each group
within a fold. Multiple responses require an explicit `responses` selection and
aligned joint posterior draws. The runner sums all selected response and row log
likelihoods **within each draw**, then computes log-mean-exp across draws. Summing
marginal log predictive densities instead would describe a different prediction.
The [brms K-fold reference](https://search.r-project.org/CRAN/refmans/brms/html/kfold.brmsfit.html)
distinguishes these observation, fold, and group prediction units. The
[loo ELPD reference](https://mc-stan.org/loo/reference/elpd.html) describes
pointwise log-likelihood arrays with separate iteration, chain, and data axes.

`fold_ids` evaluates a subset without extrapolating missing folds. `result.elpd`
sums only evaluated units; `result.complete` reports whether all folds ran.
`result.se` is the descriptive square root of the number of units times their
sample variance, rather than a Monte Carlo error estimate or a correction for
overlapping training sets. Zero predictive densities remain `-inf`; an undefined
SE is `NaN`. Compare models using the same rows, response selection, partition,
and predictive units. These policies also correspond to the partition, selected
fold, and joint-scoring controls in the
[brms K-fold interface](https://search.r-project.org/CRAN/refmans/brms/html/kfold.brmsfit.html).

## QVR source refits

Each callback fits a fresh binding of the same specification. Held-out scoring
uses `ModelFit.log_likelihood`, including its checked response semantics and
retained parameters. The small budgets below demonstrate the interface; assess
optimization or MCMC diagnostics before interpreting predictive comparisons.

<!-- python: run -->
```python
import torch
from quivers.diagnostics import FoldPrediction, cross_validate, fold_plan
from quivers.modeling import Model, VariationalMethod

source = """object Row : FinSet _
object Out : Real 1
program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""
specification = Model(source)
data = {"x": torch.tensor([-1., 0., 1., 2.]),
        "y": torch.tensor([-0.7, 0.1, 0.9, 1.8])}
plan = fold_plan(4, folds=2, seed=11)

def refit(split):
    training = {name: value[list(split.train)] for name, value in data.items()}
    testing = {name: value[list(split.test)] for name, value in data.items()}
    fitted = specification.fit(
        training, method=VariationalMethod(num_steps=2, seed=split.seed),
    )
    scores = fitted.log_likelihood(testing, num_samples=4, seed=split.seed)
    return FoldPrediction(log_likelihood=scores, fit=fitted)

result = cross_validate(plan, refit, save_fits=True)
assert result.complete and result.log_predictive_density.shape == (4,)
assert set(result.fits) == {0, 1}
```

Prediction reuses the fitted latent layout. Row-specific latent variables whose
layout changes with held-out data require an explicit marginal or new-level
prediction policy; a refitter must provide those raw scores rather than silently
rebinding them to unrelated latent coordinates.

## Formula refits

The callback changes, while the partition and scorer remain the same. Formula
fits use their existing `log_likelihood(new_data=...)` path, which retains response
transforms, categories, and declared group prediction policies.

<!-- python: run -->
```python
import pandas as pd
from quivers.diagnostics import FoldPrediction, cross_validate, fold_plan
from quivers.formulas import fit

frame = pd.DataFrame({"x": [-1., 0., 1., 2.], "y": [-0.7, 0.1, 0.9, 1.8]})
plan = fold_plan(len(frame), folds=2, seed=11)

def refit_formula(split):
    fitted = fit(
        "y ~ x", data=frame.iloc[list(split.train)],
        method="svi", num_steps=2, seed=split.seed,
    )
    scores = fitted.log_likelihood(
        new_data=frame.iloc[list(split.test)], num_samples=4, seed=split.seed,
    )
    return FoldPrediction(log_likelihood=scores)

result = cross_validate(plan, refit_formula, joint="fold")
assert result.complete and result.log_predictive_density.shape == (2,)
```

A new grouping level requires the fitted predictor's explicit new-level policy.
The runner does not infer that policy from fold labels. Fold-specific streams are
stable under fold selection and evaluation order, and participating caller RNG
streams are restored after each callback. Actual fits are retained only when
`save_fits=True`; otherwise they can be released before the next fold.

::: quivers.diagnostics.cross_validation
