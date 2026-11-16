# Model binding and fitting

The [model interface](../guides/model-data-and-fitting.md) binds external values
using the types and roles checked from QVR, then applies a reusable inference method.

Declare explicit inputs in the runtime telescope of a program header, as in
`program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out`.
Static objects, morphisms, and extents use a preceding bracketed telescope.
Annotations constrain values; `observe` determines which inputs are responses.
The [parameter reference](../reference/qvr/index.md#parameter-telescopes) explains
the convention across language constructs.

`MCMCMethod` and `VariationalMethod` return `ModelFit`. `SMCMethod` instead
returns the existing `SMCResult` with its particles, normalizing-constant
estimate, and tempering diagnostics. All three consume the same checked
`BoundModel`; their result contracts remain specific to the inference method.

An existing `ModelFit` exposes `fit.evidence(sampler)`. This applies the same
SMC driver to the retained checked program and data, including a fit initialized
with a variational guide. The guide and retained posterior draws do not enter
the evidence calculation. Observation weights define a powered target whose
normalizing constant need not be an ordinary data probability.

`ModelFit.posterior_epred` collects native conditional response expectations
with retained chain/draw axes. `ModelFit.conditional_effects` applies this
collector once to a bounded predictor grid, using the checked nominal row axis
to distinguish row inputs from unrelated fitted tensors. Both reuse the fitted
latent layout and deterministic parameters; neither fits a new posterior.
See [source conditional effects](../guides/conditional-effects.md#expected-responses-from-qvr-source)
for conditioning, Scan addressing, and finite-moment approximation semantics.

<!-- python: run -->
```python
import torch
from quivers.inference import SMC
from quivers.inference.mcmc import MetropolisKernel, RandomWalkProposal
from quivers.modeling import Model, SMCMethod

source = """object One:FinSet 1
object Out:Real 1
program model(y:Real):One->Out
    sample mu <- Normal(loc=0.0,scale=1.0)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export model
"""
result = Model(source).fit(
    {"y": torch.tensor(0.7)},
    method=SMCMethod(sampler=SMC(
        MetropolisKernel(RandomWalkProposal(0.5)),
        num_particles=16, num_mcmc_steps=1, num_pilot_steps=0, seed=7,
    )),
)
assert result.num_particles == 16
assert result.temperatures[0] == 0 and result.temperatures[-1] == 1
```

::: quivers.modeling.model

::: quivers.modeling.methods
