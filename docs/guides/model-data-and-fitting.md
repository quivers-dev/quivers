# Bind data and fit a QVR model

A model has one input contract: the external symbols and types inferred by the
Quivers Indexed Effect Core (QIEC). The Python interface uses this contract to
(i) infer finite dimensions from supplied values; (ii) validate the binding; and
(iii) distinguish observations from covariates when predicting.

The following sections declare a model, inspect and bind its inputs, fit it with
reusable components, and explain the limits of dimension inference.

## Declare external values when useful

An external symbol can remain implicit when its uses determine its type. A
`data` declaration fixes that type explicitly, including a tensor's dimensions.
The compiler checks every use against the declaration.

<!-- compile: qiec -->
```qvr
object Row : FinSet _
object Out : Real 1

program regression : Row -> Out
    data x : Tensor[Real]([Row])
    data y : Tensor[Real]([Row])
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

The declaration of `y` gives its type, while `observe y` gives its role. An
observation is thus identified by the program itself. No Python list of observed
names is required.

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

`model.fit(data, method=method)` combines binding and inference. Keep the bound
instance when several methods should use the same initialized model. A method's
`run(bound)` can also return its own result type: messages, optimization bounds,
and certificates need not be converted into posterior draws.

Scanned observations use their checked sequence shape in `bound.data`; fitting
adds the one-run axis expected by the tensor executor through a view. Custom
tensor methods and guide factories can obtain these values with
`bound.tensor_data()`. `bound.trace` also accepts a scanned latent's address,
such as `s@2`, to condition one position without marking it as an external input.

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
Pass the template arguments through `parameters={"alpha": 0.5, "beta": 0.1}`.

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

The fitted wrapper currently predicts at the fitted data layout. Prediction on
new rows, persistent factor coding, and the wider regression interface remain
separate work, listed in the [capability audit](../developer/model-interface-roadmap.md).
