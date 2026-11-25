# Custom response families

A **custom family** binds a checked distribution contract to a native
distribution factory. The contract states parameter types, intrinsic dimensions,
sample support, and sampling capabilities; the factory supplies the corresponding
PyTorch distribution. The same law then supplies fitting densities, raw held-out
likelihoods, response expectations, and predictive draws through the existing
formula workflow.

`FamilyCatalog` is immutable and local to a compilation. Extending one catalog
does not change another model's laws or the built-in registry. Thus two models
may use the same custom source name with different implementations, while their
checked caches and fitted recipes retain the appropriate catalog identity.

## Constructor parameters and predictor parameters

`custom_family` maps native constructor parameters onto existing formula roles:
(i) `location` receives the linked main predictor `mu`; (ii) `auxiliary` supplies
ordinary scalar priors or distributional regressions; and (iii) `data` supplies
deterministic parameters from typed numeric columns. An auxiliary mapping can
separate the constructor name from the regression name. For instance, native
`scale` can receive formula parameter `sigma`.

The family takes part in further formula features by
[declaration](family.md#what-a-family-declares). For instance,
`family.with_(error_scale="sigma")` admits `se()`, and
`family.with_(elliptical=Elliptical(scale="sigma"))` admits autocorrelation
terms and `set_rescor` for a law that is Normal in its location and scale.

The following factory returns a symmetric two-component Normal mixture. Its
component locations differ by two, and a known per-row shift enters both
components. The example uses a small optimization budget to exercise the API.

<!-- python: run -->
```python
import pandas as pd
import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

from quivers.family_catalog import FamilyCatalog
from quivers.formulas import AuxParam, custom_family, fit, links
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

contract = DistributionFamily(
    name="TwoMode",
    parameters=(
        FamilyParameter(name="center", constraint="real", rank=0),
        FamilyParameter(name="scale", constraint="positive", rank=0),
        FamilyParameter(name="bias", constraint="real", rank=0),
    ),
    support="real", element=REAL, event_rank=0, event_sources=(),
    discrete=False, reparameterizable=False, compositional=False,
)

def two_mode(parameters):
    center = parameters["center"] + parameters["bias"]
    locations = torch.stack((center - 1, center + 1), dim=-1)
    scales = parameters["scale"].unsqueeze(-1).expand_as(locations)
    weights = Categorical(probs=torch.full_like(locations, 0.5))
    return MixtureSameFamily(weights, Normal(locations, scales))

catalog = FamilyCatalog().with_family(contract, two_mode)
response_family = custom_family(
    "TwoMode", catalog=catalog, location="center",
    auxiliary={
        "scale": AuxParam(
            name="sigma", prior="LogNormal(0.0, 0.3)", link=links["log"],
        ),
    },
    data={"bias": "known_shift"},
)
frame = pd.DataFrame({
    "y": [-0.7, 0.8, -0.4, 1.2],
    "x": [-1.0, 0.0, 0.5, 1.0],
    "known_shift": [0.1, -0.2, 0.3, 0.0],
})
result = fit(
    "y ~ x", data=frame, family=response_family,
    distributional={"sigma": "~x"},
    method="svi", num_steps=2, seed=7,
)
fresh = frame.iloc[[3, 1]].copy()
expected = result.posterior_epred(
    new_data=fresh.drop(columns="y"), num_samples=5, seed=11,
)
scores = result.log_likelihood(new_data=fresh, num_samples=5, seed=11)["y"]
assert expected.shape == scores.shape == (1, 5, 2)
assert result.posterior_predict(
    new_data=fresh.drop(columns="y"), num_samples=5, seed=11,
).shape == expected.shape
```

Deterministic data parameters introduce no latent coefficient or prior. They
enter the existing checked predictor graph, preserving their fitted numeric
carriers and row identities during new-data prediction, subsets, and refits.
The native and formula parameter names need not coincide. Location and auxiliary
predictors retain ordinary group effects, nonlinear expressions, priors, and
likelihood weights; raw posterior likelihoods remain unweighted.

## Capabilities and ownership

The factory must return the normalized law described by its contract. QVR
checks constructor argument names, intrinsic dimensions, batch broadcasting,
and declared reparameterization at the native boundary. These checks cannot
prove that a user-written `log_prob` integrates to one or that its sampler has
the claimed law; those mathematical properties require independent tests.

A custom `torch.distributions.Distribution` subclass can supply an arbitrary
`log_prob`, `sample` or `rsample`, and `mean`. Existing response helpers use those
operations directly. Censoring and truncation additionally require the existing
interval-mass or CDF capabilities and a truthful atomicity contract. The
[catalog API](../family-catalog.md) describes typed native inputs and execution
capacity. Host implementations do not acquire external target implementations
through a name alone; unsupported source targets report their capabilities.

Catalogs retain live native factories by identity. Updates and cross-validation
reuse that binding, while [durable fitted artifacts](../modeling.md#durable-fitted-artifacts)
require an explicit factory key and configuration when restoring it. The stored
semantic contract is checked against the supplied implementation catalog.

::: quivers.formulas._custom_family.custom_family

::: quivers.formulas._custom_types.CustomFamilySpec
