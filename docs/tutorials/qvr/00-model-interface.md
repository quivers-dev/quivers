# Checked model fitting

The model interface uses the program's checked data contract to bind a
dataset and select an inference method. This tutorial (i) lets data determine
the row count; (ii) inspects the external types and observation roles; and
(iii) fits and predicts without constructing a guide or an optimizer.

## Declare, bind, and fit

The example below is executable as one Python block. The short fitting budget
checks the workflow; it does not establish posterior convergence.

```python
import torch
from quivers.modeling import Model, VariationalMethod

source = """
object Row : FinSet _
object Out : Real 1
program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""

model = Model(source)
assert model.schema.observations == ("y",)
print(model.schema.describe())
data = {
    "x": torch.tensor([1.0, 2.0, 3.0]),
    "y": torch.tensor([0.5, 1.0, 1.5]),
}
bound = model.bind(data)
print(bound.schema.describe())
fit = bound.infer(VariationalMethod(num_steps=30, seed=7))
draws = fit.draws(num_samples=10)
replicated = fit.predict(num_samples=10)
assert draws["b"].shape[:2] == (1, 10)
assert replicated["y"].shape[0] == 10
assert torch.isfinite(torch.tensor(fit.losses)).all()
```

`Row` becomes three at binding. The parameter `y : Tensor[Real]([Row])` declares the response type;
`observe y` gives it the observation role. Prediction thus releases `y` and
retains `x`. A new binding can change the number of rows when the fitted latent
layout remains the same:

```python
new_inputs = {"x": torch.tensor([0.5, 1.5])}
new_predictions = fit.predict(new_inputs, num_samples=10, seed=11)
assert new_predictions["y"].shape == (10, 2)
pointwise = fit.log_likelihood(num_samples=10)
assert pointwise["y"].shape == (1, 10, 3)
```

The pointwise array retains one likelihood term per response and posterior draw;
the first axis is the variational result's single pseudo-chain. Scoring new
responses additionally requires their `y` values. The
[fitting guide](../../guides/model-data-and-fitting.md#extract-pointwise-log-likelihoods)
specifies event reductions and the current restriction on marginalized responses.

## Report invalid inputs before inference

```python
from quivers.dsl import CompileError

try:
    model.bind({"x": torch.ones(2), "y": torch.ones(3)})
except CompileError:
    pass
else:
    raise AssertionError("conflicting row counts must be refused")

try:
    model.bind({"x": torch.ones(3, dtype=torch.long), "y": data["y"]})
except TypeError:
    pass
else:
    raise AssertionError("integer values do not inhabit Tensor[Real]")
```

Declarations are optional when uses determine the type. A declaration is needed
when uses leave an ambiguous input. Dimensions also need evidence: a shared
scalar does not determine the number of participants, and an empty index array
does not determine the number of groups. The [binding guide](../../guides/model-data-and-fitting.md)
specifies exact-shape and index-bound inference, templates, scans, and custom methods.

## Choose another method

Replace `VariationalMethod(...)` with `MCMCMethod(sampler=MCMC(...))` to reuse a
configured sampler and obtain MCMC summaries. The [fitting guide](../../guides/model-data-and-fitting.md#choose-reusable-fitting-components)
gives the complete call. A custom `InferenceMethod[Result]` can instead return
its own messages, bounds, or certificates. Its result type does not imply that
its algorithm is exact or that its certificate has been verified.

The [interactive tools guide](../../guides/repl-and-lsp.md#checked-parameters-in-interactive-tools)
shows the same symbolic model in the editor, REPL, and Jupyter kernel. The
[capability audit](../../developer/model-interface-roadmap.md) records the remaining
brms and structured-executor requirements. [Chapter 1](01-first-model.md)
then opens the fitting components to explain what inference computes.
