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

## Retained fitting recipes

`ModelFit.update` runs a fresh fit through its retained `RefitRecipe`, with
explicit data ownership and inference overrides. Its fold adapter delegates to
the shared cross-validation runner. See [updating and refitting](../guides/model-refits.md)
for nominal row axes, data snapshots, and method seeds.

::: quivers.modeling.refit

## Durable fitted artifacts

`save_fit` records a source `ModelFit` as a versioned JSON specification and
safe tensor checkpoint. `load_fit` recompiles the original QVR, restores its
module and posterior state, and rebuilds the retained fitting recipe. It does
not optimize a model or draw replacement posterior values. For a scanned guide,
strict conditioned replay recovers every reached latent address from saved
values; a missing value is an error.

<!-- python: run -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from quivers.modeling import Model, VariationalMethod, load_fit, save_fit

artifact_source = """object Row:FinSet _
program regression(y:Tensor[Real]([Row])):Row->Row
    sample mu <- Normal(loc=0.0,scale=1.0)
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return mu
export regression
"""
artifact_fit = Model(artifact_source).fit(
    {"y": torch.tensor([-0.5, 0.25, 1.0])},
    method=VariationalMethod(num_steps=0, seed=7),
)
with TemporaryDirectory() as artifact_directory:
    artifact_path = Path(artifact_directory) / "regression.qvrfit"
    save_fit(artifact_fit, artifact_path)
    restored_fit = load_fit(artifact_path)
    for name, value in artifact_fit.posterior.state_dict().items():
        assert torch.equal(value, restored_fit.posterior.state_dict()[name])
```

The checkpoint retains static arguments, typed data, compilation and method
seeds, parameter and buffer dtypes, tied parameters, module modes, prior policy,
and recorded chain statistics. Loaded recipes reuse `update` and the existing
fold adapter. Saving captures values at write time. A loaded reference-owned
source recipe shares the loaded binding's values. A cloned recipe retains an
independent training snapshot. Neither shares memory with the original process.

The default Normal guide, recorded MCMC results, and retained prior draws have
built-in contracts. Custom guides and external Python modules require an
explicit named `ArtifactFactory` at both write and read time. Its decoder
receives an `ArtifactContext` when reconstructing a guide. Factories are trusted
caller code and must reconstruct without fitting or sampling. An artifact never
chooses an import path or falls back to arbitrary object serialization.

Analysis exports retain posterior tensors and diagnostics. A fitted artifact
additionally retains the checked model needed for prediction and refitting.
Unsupported artifact versions, missing factories, changed latent geometry, and
failed integrity checks are refused before a fit is returned.

::: quivers.modeling.artifact

::: quivers.modeling.ArtifactContext

::: quivers.modeling.ArtifactFactory
