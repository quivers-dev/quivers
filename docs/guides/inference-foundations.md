# Inference Foundations

This page introduces the inference stack's architecture, the trace
and sample site interface, and the conditioning API used to
clamp observations. Variational inference (guides,
objectives, SVI) is covered in
[Variational Inference: SVI](inference-svi.md); gradient-based MCMC is covered in
[MCMC and Hybrid Samplers](inference-mcmc.md).

## Architecture

Inference asks which unobserved values are plausible after seeing data.
Write `z` for latent values, `y` for observations, and `x` for known
inputs. A model supplies the prior and likelihood, hence
$p(z\mid x,y) \propto p(y,z\mid x)$. Evaluating a candidate `z` gives its joint density
$p(y,z\mid x)$. Inference uses these evaluations to approximate how
posterior mass is distributed across possible values of `z`.

For a first model, list its named sample sites, identify which are
observed, and choose an inference driver.
[SVI](inference-svi.md) fits an adjustable approximation to the posterior;
[MCMC](inference-mcmc.md) constructs a sequence of posterior draws.
The registry records the remaining latent sites; the driver updates or
samples their values.


The inference subpackage is a six-layer stack, each layer consumable
independently and re-exported from
[`quivers.inference`](../api/index.md#inference-package):

```mermaid
flowchart TB
    L6["Drivers<br/><small>SVI · MCMC · Predictive</small>"]
    L5["Hybrid algorithms<br/><small>AutoDAIS · WarmupThenHMC</small>"]
    L4["Posterior approximations<br/><small>Auto*Guide · HMC · NUTS</small>"]
    L3["Objectives × estimators<br/><small>ELBO · IWAE · Rényi × Reparam · StL · DReG · Score</small>"]
    L2["Transforms<br/><small>affine coupling · MAF · IAF · NSF · BN · LU</small>"]
    L1["LatentRegistry<br/><small>support · dimensions · plates · parents</small>"]
    L6 -->|orchestrate| L5 -->|combine| L4 -->|optimize with| L3 -->|parameterize through| L2 -->|read structure from| L1
    class L6,L5 qv-output
    class L4,L3 qv-checked
    class L2,L1 qv-foundation
```

Every guide and MCMC kernel consumes a single
`LatentRegistry.from_model(model, observed_names)`,
which flattens / unflattens between site-keyed dicts and a single
unconstrained vector and routes every per-site bijector through
[`torch.distributions.constraint_registry.biject_to`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.constraint_registry.biject_to).

## The variational pipeline

The latent registry gives algorithms a consistent numerical representation.
A positive scale and a real-valued mean have different supports, but an
optimizer or Hamiltonian sampler can operate on one unconstrained real
vector. Per-site bijections map its pieces back to valid values. Density
calculations include Jacobian corrections for this coordinate change;
otherwise the algorithm would target the wrong distribution.

Stan’s [change-of-variables example](https://mc-stan.org/docs/stan-users-guide/reparameterization.html#changes-of-variables) derives the density correction for a log transform. [@stanReparameterization] The same correction appears in [continuous flows](continuous-spaces.md#normalizing-flows) and [constrained guides](inference-svi.md#autonormalguide).


```mermaid
flowchart TB
    M["Model<br/><small>MonadicProgram</small>"] -->|execute once| T["Trace<br/><small>record sample sites</small>"]
    T -->|clamp observed sites| C["Conditioned model"]
    C -->|inspect latent sites| LR["LatentRegistry"]
    LR -->|choose family| G["Guide<br/><small>Auto*Guide subclass</small>"]
    G -->|estimate bound| O["Objective + estimator<br/><small>ELBO · IWAE · Rényi · VR-IWAE</small>"]
    O -->|differentiate| S["SVI<br/><small>stochastic optimization</small>"]
    S -->|fitted posterior| P["Predictive draws"]
    class M qv-input
    class T,C,LR,G,O qv-checked
    class S,P qv-output
```

## Trace and sample sites

A trace is a record of one execution. For a model with latent `mu` and
observed `y`, it records a candidate `mu`, the supplied `y`, and both
local log densities. A *site* is the named point where a value is drawn
or scored. Repeated executions give that same site different values. The joint log density adds the sites' contributions.
A deterministic `let` records a computed value with zero additional
log density, because it introduces no independent random choice.

Pyro’s [first effect handler example](https://pyro.ai/examples/effect_handlers.html#A-first-look-at-Poutine:-Pyro%E2%80%99s-library-of-algorithmic-building-blocks) builds a log joint evaluator by combining tracing with conditioning. [@pyroEffects] In QVR, [running and scoring](../reference/qvr/probabilistic-programs.md#running-and-scoring) expose these execution records.


A trace records all stochastic operations in a program. Each sample
point is a
[`SampleSite`](../api/inference/trace.md#quivers.inference.trace.SampleSite).

<!-- python: skip -->
```python
from quivers.inference import trace, Trace, SampleSite

model = ...  # MonadicProgram

# Execute model with tracing
tr = trace(model, x)

# Access sites
sites = tr.sites  # dict[site_name -> SampleSite]

for name, site in sites.items():
    print(f"{name}: {site.log_prob}")
```

A `SampleSite` records:

- `name`: identifier of the sample
- `morphism`: the generating distribution (`None` for `let` bindings)
- `value`: sampled or observed value
- `log_prob`: log-density of the value under the morphism, shape `(batch,)` (zero for `let` bindings)
- `is_observed`: whether the site was clamped to an observed value
- `is_deterministic`: whether the site is a deterministic `let` binding

## Conditioning on observations

To *clamp* a site is to reuse a supplied value whenever execution reaches
that site. This makes its likelihood available to inference. It does
not replace prior draws at other sites with posterior draws: that
requires the inference algorithm. Also distinguish a random latent
site from an `nn.Parameter`; the latter is an optimized model or guide
parameter unless the model explicitly puts a prior on it.


The
[`condition()`](../api/inference/conditioning.md#quivers.inference.conditioning.condition)
function clamps observations, fixing certain variables:

<!-- python: skip -->
```python
from quivers.inference import condition, Conditioned

model = ...  # MonadicProgram

# Observed values (e.g., from an experiment)
observations = {
    "y_1": torch.tensor(1.5),
    "y_2": torch.tensor(-0.3),
}

# Create conditioned model
conditioned = condition(model, observations)

# Trace the conditioned model: observed sites are clamped to the data
tr = conditioned.trace(x)
```

The conditioned model is a
[`Conditioned`](../api/inference/conditioning.md#quivers.inference.conditioning.Conditioned)
instance that wraps the original model and enforces observation
constraints.

### Host data: per-row covariates and index arrays

Keys in the `condition` data dict that don't match any declared
sample / observe site are exposed to the program's runtime
environment as deterministic values, visible to `let`-expression
evaluation. Supply per-row covariates or group indices this way when
fitting a hierarchical regression:

<!-- python: skip -->
```python
import torch
from quivers.dsl import loads
from quivers.inference import condition

model = loads('''
object Subj : FinSet 4
object Resp : FinSet 12

program p : Resp -> Resp
    sample by_subj : Subj <- Normal(0.0, 1.0)
    let mu = by_subj[subj_idx]
    observe r : Resp <- Normal(mu, 1.0)
    return r
export p
''').morphism

subj_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
r_obs    = torch.zeros(12)

cond = condition(model, {"subj_idx": subj_idx, "r": r_obs})
tr   = cond.trace(torch.zeros(12, 1))
```

`r` matches the observed sample site `r : Resp <- Normal(mu, 1.0)`
and is clamped as usual. `subj_idx` doesn't match any site; it
lands in the runtime environment, and `let mu = by_subj[subj_idx]`
advance-indexes into the per-subject draw. Free variables in `let`
expressions (names not bound by any sample / observe / let / lambda
step) resolve against the data dict at trace time; if the value is
missing the runtime raises a clear `KeyError`.

## Debugging

Enable tracing to inspect sites and log probabilities:

<!-- python: skip -->
```python
from quivers.inference import trace

tr = trace(model, x)

for name, site in tr.sites.items():
    print(f"{name}: log_prob={site.log_prob.item():.4f}")
```

Monitor the [ELBO](inference-svi.md#elbo-evidence-lower-bound) during training to detect divergence or poor guide
fit (see [SVI](inference-svi.md#svi-stochastic-variational-inference)).

## Where to next

- [Variational Inference: SVI](inference-svi.md): guides,
  objectives, gradient estimators, the SVI training loop, and
  predictive sampling.
- [MCMC and Hybrid Samplers](inference-mcmc.md): [HMC](inference-mcmc.md#mcmc-hmc-and-nuts), NUTS,
  hybrid samplers, and predictive sampling from MCMC chains.
- [Analysis Pipelines: Fitting and Diagnostics](analysis-fitting-and-diagnostics.md):
  the higher-level fit / compare / posterior-predictive-check
  interface built on top of these primitives.
