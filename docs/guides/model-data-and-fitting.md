# Bind data and fit a QVR model

A model has one input contract: the external symbols and types inferred by the
Quivers Indexed Effect Core (QIEC). The Python interface uses this contract to
(i) infer finite dimensions from supplied values; (ii) validate the binding; and
(iii) distinguish observations from covariates when predicting.

The following sections declare a model, inspect and bind its inputs, fit it with
reusable components, and explain the limits of dimension inference.

## Declare external values when useful

An external symbol can remain implicit when its uses determine its type. A
typed runtime parameter fixes that type explicitly, including a tensor's dimensions.
The compiler checks every use against the declaration.

These are ordinary parameters of the program, written in its header. They are
not steps that execute. This uses the same distinction as a `define`
computation: static parameters in `[...]` determine types and structure, and
values in `(...)` inhabit those types. The
[language reference](../reference/qvr/index.md#parameter-telescopes) explains how
the convention applies to programs, computations, constructors, effects, and handlers.

<!-- compile: qiec -->
```qvr
object Row : FinSet _
object Out : Real 1

program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b

export regression
```

`Row` is the finite index space of observation rows. Its underscore asks binding
to determine its cardinality; the number of observations in `y` and the length
of `x` must agree. `Out` and the prior specify the model independently of that
cardinality.

The parameter annotation on `y` gives its type, while `observe y` gives its role. An
observation is thus identified by the program itself. No Python list of observed
names is required.

A reusable program can bind its row object in a static telescope instead of
using a module-level object, as in
`program regression[Row : FinSet](x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out`.
The dependency of both tensor types on `Row` states that their lengths must
agree. Inference of an omitted extent uses the supplied values as evidence;
it does not turn the extent into a latent variable or a runtime array.

If a count is instead a value read by the model, declare it in the runtime
header, as `(count : Nat)`, and include it in the data mapping. Binding accepts
nonnegative integers and retains the restriction in the checked contract.
Static extents such as `[n : Nat]` belong to specialization rather than this
runtime data mapping.

## Inspect and bind

Use the QVR block above as `source`, or read the same model from a file:

```python
from quivers.modeling import Model

model = Model(source)
print(model.schema.describe())
```

The schema reports `x` as data and `y` as an observation, both at
`Tensor[Real]([Row])`. It retains the checked type objects in `schema.types`;
`describe()` renders them for display.

`Model` selects the first exported program by default. Helper programs, such as
a scan's step, need no separate selection. Use `entry="name"` to select another
program; several programs without an export require an explicit choice.

```python
import torch

bound = model.bind({
    "x": torch.tensor([1.0, 2.0, 3.0]),
    "y": torch.tensor([0.5, 1.0, 1.5]),
})
print(bound.schema.describe())
```

The bound schema has extent three. Binding another dataset produces an independent
compiled instance. Missing inputs, extra names, incompatible shapes, and wrong
element dtypes raise errors before fitting. Integer arrays are not silently
converted into real arrays, and tensors retain their autograd connections.

A real-valued domain that the program reads needs an explicit one-run `input`.
Observation rows belong to the QVR plates; an input with several rows would mean
several independent executions of the entire model and is refused here.

## Weight observation likelihoods

Add `[weights=w]` to an observation to give its rows power-likelihood weights.
Declare `w : Tensor[Real]([Row])` in the runtime header when one weight belongs
to each row, and include `"w"` with `x` and `y` in the data passed to
`Model.bind`. A declared scalar `w : Real` or a literal such as `[weights=0.5]`
uses the same power for every row. The
[language reference](../reference/qvr/probabilistic-programs.md#observation-weights)
gives a complete weighted regression.

Weights must be finite and nonnegative, and tensor weights must match the
observation batch shape. Event components of one multivariate response share
one weight. A zero-weight row contributes zero score and zero likelihood
parameter gradient, even if its raw log density is negative infinity. Weights
are applied before row or group reductions; subsampling selects the matching
weights and applies its sampling correction once. Priors keep their original
scores, and weights are not normalized automatically.

These weights change the fitting target. At fixed parameters, the response
sampler and predictive distribution retain their original behavior. A fitted
posterior can change, so its posterior predictive results can change as well.
Keep weights with the other checked inputs when rebinding; declaring a weight
input alone performs no scoring.

## Choose reusable fitting components

The default `MCMCMethod()` runs four seeded NUTS chains, with 500 warmup
transitions and 1000 retained draws per chain. A configured driver exposes the
existing kernels, budgets, callbacks, and checkpoints:

```python
from quivers.inference import MCMC, NUTSKernel
from quivers.modeling import MCMCMethod

method = MCMCMethod(sampler=MCMC(
    NUTSKernel(), num_warmup=500, num_samples=1000, num_chains=4, seed=7,
))
fit = bound.infer(method)
print(fit.summary())
predictive = fit.predict(num_samples=100)
```

Prediction releases `y` and retains `x`. `fit.samples` preserves separate chain
and draw axes. These methods reuse the package's existing samplers, predictive
execution, and convergence estimators.

For variational inference, choose a step budget and optionally supply a guide
factory, objective, optimizer factory, or callback:

```python
from quivers.modeling import VariationalMethod

variational = bound.infer(VariationalMethod(num_steps=1000, seed=7))
posterior_draws = variational.draws(num_samples=100)
predictive = variational.predict(num_samples=100)
losses = variational.losses
```

The default guide is built from a trace, including a scan's individual sites.
Variational draws have one pseudo-chain; they do not supply MCMC convergence
evidence. `summary()` thus requires an MCMC result.
For formula-generated fits, the [posterior workflow](analysis-data-and-formulas.md#posterior-coefficients-and-predictions)
describes labeled coefficients, retained group levels, linear predictors, and
expected responses on the fitted training rows.


`model.fit(data, method=method)` combines binding and inference. Keep the bound
instance when several methods should use the same initialized model. A method's
`run(bound)` can also return its own result type: messages, optimization bounds,
and certificates need not be converted into posterior draws.

Scanned observations use their checked sequence shape in `bound.data`; fitting
adds the one-run axis expected by the tensor executor through a view. Custom
tensor methods and guide factories can obtain these values with
`bound.tensor_data()`. `bound.trace` also accepts a scanned latent's address,
such as `s@2`, to condition one position without marking it as an external input.

### Predict at new inputs

Pass a new covariate binding to predict on different rows. Responses may be
omitted, since prediction generates them:

```python
new_data = {"x": torch.tensor([0.5, 1.5])}
predicted = fit.predict(new_data, num_samples=100, seed=11)
assert predicted["y"].shape == (100, 2)
```

The row extent is inferred again from the new covariates. The model keeps its
fitted parameters and checks that the names and shapes of its latent variables
are compatible with the posterior. A regression with one global slope can thus
predict on a different number of rows. A model with one latent variable per row
needs a refit when that latent layout changes. Prediction does not mutate the
original binding, and an explicit seed preserves the caller's random state.

This operation accepts values in the model's checked representation. It does
not reconstruct dataframe contrasts or assign meanings to unseen participant
labels; those operations require the corresponding retained data encoding.
Formula fits retain that encoding: use
`formula_fit.posterior_predict(new_data=frame)` for a `BayesianFit`, with
fitted transforms and known group levels. The
[formula prediction guide](analysis-data-and-formulas.md#predictions-on-new-rows)
also covers linear predictors, expected responses, and rejected new levels.

### Extract pointwise log likelihoods

`log_likelihood` returns the response density at each posterior draw, retaining
the observation rows. A new binding must include the responses to score:

```python
pointwise = fit.log_likelihood(num_samples=100)
new_pointwise = fit.log_likelihood({
    "x": torch.tensor([0.5, 1.5]),
    "y": torch.tensor([0.25, 0.75]),
}, num_samples=100)
assert new_pointwise["y"].shape[-1] == 2
```

MCMC output has chain and draw axes followed by the response's observation
axes; variational output has one pseudo-chain. These are raw, unweighted
likelihood terms, without the prior or posterior normalization, including when
the fitting target uses `[weights=w]`. For an unweighted observation, summing
the observation axes recovers that response's contribution to the joint
density. To reconstruct a weighted contribution, apply the checked weights to
the raw terms with the zero-weight convention before summing. Event coordinates
of one multivariate response are reduced together, while independent response
rows remain separate.

An observation hidden inside exact marginalization requires a corresponding
marginal-likelihood observer. The current method refuses that case rather
than assigning the whole marginalized joint to individual observations.

## Infer only dimensions determined by data

A supplied tensor's checked shape provides exact extent evidence. If `x` and `y`
share `Row`, they must fix it at the same length. A shared scalar parameter of a
plated family does not determine the plate's size.

A directly read external gather index can provide a lower bound. For an inferred
finite space, binding chooses the smallest zero-based space containing all its
indices: its cardinality is `max(index) + 1`. Gaps remain positions in the space;
this operation does not recode participant labels. An explicit cardinality or a
full table's shape can retain participants absent from the observations.

For the [LDA model](../examples/lda.md), replace `Doc` and `Token` cardinalities
with underscores. The words and token maps determine `Token`; gathering document
topic weights through `doc_idx` determines the smallest document space. Keep
`Topic` explicit: the observed words do not identify the intended number of topics.
Pass scalar hyperparameters with the other runtime inputs, as in
`model.bind({**observations, "alpha": 0.5, "beta": 0.1})`.
Their annotations belong in the runtime telescope, as in
`program lda(alpha : Real, beta : Real, ...) : Word -> Word`.
Object and morphism parameters belong in the static telescope instead.
The Python `parameters=` mapping supplies static arguments, such as a known
extent; runtime scalar values belong in `data`. Rebinding for prediction must
retain these scalar inputs along with the other covariates.

An empty index array cannot identify the size of its target space. A transformed
index, such as `group_idx + 1`, does not establish the raw index's target extent.
When the available evidence does not determine a size, binding requests an
explicit cardinality. It never fits a different number of latent groups by
silently selecting a fallback size.

## Execution limits

`qvr check` checks the symbolic QIEC program before data are available; binding
then specializes its dimensions and runs the tensor compiler's checks. A
`CheckedProgram` retains its checked schema and can be passed to a custom method,
but the tensor fitting components refuse it pending
[enhancement #232](https://github.com/quivers-dev/quivers/issues/232).

Persistent factor coding, prediction for new group levels, and the wider
regression interface require additional contracts, listed in the
[capability audit](../developer/model-interface-roadmap.md).
