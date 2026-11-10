# 6. Choosing an inference algorithm

Quivers exports eleven concrete [guide](../../api/inference/guide.md) classes, seven [objectives](../../api/inference/elbo.md), two MCMC kernels (HMC, [@neal2011mcmc]; NUTS, [@hoffman2014nuts]), two hybrid approaches, and four gradient [estimators](../../api/inference/estimators.md). The appropriate combination depends on [posterior geometry](03-hierarchical.md#centered-or-non-centered), computational budget, and the quantities the analysis needs.

First choose VI, MCMC, or a hybrid. For VI, then choose a guide, objective,
and gradient estimator. The sections below explain how these choices affect
the approximation and its computational cost.

Start with two ways of representing an answer. Variational inference keeps
an adjustable distribution, such as a Normal with a learned mean and scale.
MCMC keeps a sequence of parameter draws whose long-run distribution targets
the posterior. The first has approximation restrictions from its chosen family;
the second has [dependence between draws](08-diagnostics-and-comparison.md#convergence-diagnostics) and finite-run sampling error.

Within VI, the guide determines which distribution shapes are expressible,
the objective determines what is optimized, and the gradient estimator
determines how a noisy derivative is computed. A richer guide may represent
correlation that a diagonal guide cannot, but it also has more parameters to
fit. These choices affect different parts of the approximation.

## Step 1: VI, MCMC, or hybrid?

| Posterior shape | First-pass choice |
|---|---|
| Smooth, broadly unimodal, < 1000 latents | Variational inference. |
| Strongly correlated or curved | Try NUTS and inspect diagnostics; reparameterization may still be necessary. |
| Well-separated modes | Run several initializations and compare the regions reached. Separate modes may require a mixture guide or a sampling method designed to move between modes. |
| Hierarchical with funnel-shaped posteriors | NUTS, or VI on the non-centered reparameterization. |
| High-dim (>1000 latents) and continuous | VI, unless you have a lot of compute. |
| Mixed discrete-continuous with small discrete support | VI with a `marginalize` block (chapter 4); HMC over the continuous remainder. |
| You don't know yet | Start with `AutoNormalGuide` + ELBO. It's cheap; the failure modes are diagnostic. |

VI is optimization-based and can scale to large latent spaces. HMC and NUTS are asymptotically exact under their regularity conditions, but finite runs still have Monte Carlo error and require convergence diagnostics. Runtime differences are model- and hardware-dependent, so benchmark the alternatives on the model at hand.

## Step 2: Which variational guide?

```mermaid
flowchart TD
    start(["Posterior shape"]) --> id{"Identifiable,<br/>weak correlation?"}
    id -- yes --> normal["AutoNormalGuide<br/>default"]
    id -- no  --> corr{"Strong correlations?"}
    corr -- yes --> dim{"Cholesky fits in memory?"}
    dim -- yes --> mvn["AutoMultivariateNormalGuide"]
    dim -- no  --> lr["AutoLowRankMultivariateNormalGuide"]
    corr -- no  --> multi{"Multimodal?"}
    multi -- yes --> mix["AutoMixtureGuide<br/>or AutoNormalizingFlow<br/>or AutoIAFGuide"]
    multi -- no  --> tails{"Non-Gaussian tails?"}
    tails -- yes --> flow["AutoIAFGuide<br/>or AutoNeuralSplineGuide"]
    tails -- no  --> normal2["back to AutoNormalGuide"]
    class start qv-input
    class id,corr,dim,multi,tails qv-decision
    class normal,mvn,lr,mix,flow,normal2 qv-output
```

The shipped guides:

| Guide | Family | When |
|---|---|---|
| `AutoNormalGuide` | Diagonal Normal | Default. Cheap, scales, often good enough. |
| `AutoDeltaGuide` | Dirac at MAP | Quick MAP; no posterior uncertainty. |
| `AutoMultivariateNormalGuide` | Full-rank Normal (Cholesky) | Strong posterior correlations and total latent dimension small enough for a dense Cholesky to fit in memory. Memory and per-step cost both scale as $O(D^2)$. |
| `AutoLowRankMultivariateNormalGuide` | Low-rank + diagonal | Hierarchical models with localized correlations; specify `rank` (typically 5-20). |
| `AutoLaplaceApproximation` | Gaussian centered at MAP with Hessian inverse | Post-hoc quadratic-around-MAP; cheap upgrade from `AutoDeltaGuide`. |
| `AutoNormalizingFlow` | Composed bijector over Normal base | Multimodal or heavy-tailed posteriors; you supply the flow stack. |
| `AutoIAFGuide` | Inverse autoregressive flow | Flagship NF default; good general-purpose non-Gaussian guide. |
| `AutoNeuralSplineGuide` | Rational-quadratic spline coupling | Sharper than IAF on bounded support. |
| `AutoMixtureGuide` | K-component mixture of guides | Multimodal posteriors; pass a list of at least two component guides. The objectives enumerate the components exactly. |
| `AutoGuideList` | Composition of guides | Assign different guide families to different latent subsets. |
| `AutoStructured` | Structured Gaussian approximation | Preserve selected posterior dependencies. |

The benchmark grid in `tests/benchmarks/` walks these against synthetic posteriors with known answers; the report at `docs/developer/inference-benchmarks.md` reports recovery checks for the algorithms, budgets, and problems actually listed.

## Step 3: Which objective?

| Objective | Bound | When |
|---|---|---|
| `ELBO(num_particles=1)` | Reparameterized ELBO | Default. |
| `ELBO(num_particles=K)` | Monte Carlo ELBO | More particles can reduce estimator variance; the bound itself is unchanged. |
| `IWAEBound(K)` | Importance-weighted bound | Tighter than ELBO for `K > 1`. |
| `RenyiBound(alpha, K)` | Rényi divergence bound | Interpolates ELBO (α=1) and IWAE (α=0). |
| `VRIWAEBound(alpha, K)` | Variational Rényi-IWAE | Mass-covering vs mode-seeking knob via α; tight via K. |
| `ChiVI(num_particles=K)` | $\chi$-divergence objective | Alternative mass-covering variational objective. |
| `RWS(num_particles=K)` | Reweighted wake-sleep (wake phase) | Separate model and guide updates from importance weights, with the draws held fixed; takes no estimator. |
| `DReGsBound(num_particles=K)` | Doubly reparameterized bound | Importance-weighted objective with DReG guide gradients. |

For most workflows, the default `ELBO()` is the right call. `IWAEBound` matters when the posterior is multimodal and the guide is flow-shaped (the importance weighting tightens the bound; coupling layers exploit it). `RenyiBound` is useful when you specifically want a mass-covering posterior (e.g. for Bayesian-optimization acquisition functions).

## Step 4: Which gradient estimator?

| Estimator | What it does | When |
|---|---|---|
| `Reparameterized` | Standard pathwise gradient | Default. |
| `StickingTheLanding` | Detaches variational params in `log q` | When `q → p*` and you're getting noisy gradients at convergence. Valid for `ELBO` (and single-draw bounds); the multi-draw bounds reject it. |
| `DoublyReparameterized` | DReG (generalized to the Rényi, VR-IWAE, and χ bounds) | With `IWAEBound(K)` at large K; not with a mixture guide on a multi-draw bound. |
| `ScoreFunction` | REINFORCE with the draws held fixed | Guides whose draws are not differentiable. |

`IWAEBound(num_particles=16, estimator=DoublyReparameterized())` combines importance weighting with DReG guide gradients. `K` trades computation and memory for a tighter importance-weighted bound; choose it by profiling rather than relying on a fixed threshold.

## Calling SVI

<!-- python: skip -->
```python
from quivers.inference import (
    AutoIAFGuide, IWAEBound, DoublyReparameterized, SVI,
)

guide = AutoIAFGuide(model, observed_names={"y"}, num_flows=4)
objective = IWAEBound(num_particles=16, estimator=DoublyReparameterized())
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-3,
)
svi = SVI(model, guide, optimizer, objective)

for step in range(5000):
    loss = svi.step(x_data, {"y": y_data})
```

## Calling NUTS

For sampling-based posterior inference:

<!-- python: skip -->
```python
from quivers.inference import NUTSKernel, MCMC

kernel = NUTSKernel(
    target_accept=0.85,
    max_tree_depth=10,
    mass_matrix="diagonal",
)
mcmc = MCMC(
    kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    init_strategy="prior",
)
result = mcmc.run(model, x_data, {"y": y_data})
print(result.r_hat)                       # per-site split-Rhat
print(result.ess)                         # per-site effective sample size
print("divergences:", result.total_divergences)
```

Diagnostics to check before trusting the run:

- Rhat < 1.01 for every site (otherwise: chains haven't mixed; run longer warmup).
- ESS large enough for the precision your estimand needs. Low ESS calls for longer runs, better parameterization, or sampler tuning.
- Divergences near zero (otherwise: posterior has curvature the integrator missed; bump `target_accept` to 0.95 or reparameterize).

## Hybrid samplers

Two configurations sit between VI and pure MCMC:

- `WarmupThenHMC(guide, kernel, svi_steps, mcmc_warmup, mcmc_samples)` runs `svi_steps` of variational warm-up and then launches MCMC chains initialized from the trained guide. Useful when HMC's warmup wastes a lot of compute exploring tails the guide already knows are empty.
- `AutoDAIS(base, model, observations, num_steps=..., init_step_size=...)` wraps a base guide with `num_steps` uncorrected Hamiltonian transitions along an annealing path. Each draw carries its own trajectory's importance weight, so `ELBO()` estimates a valid lower bound on the log evidence. It can represent posteriors that a simple mean-field guide misses, at additional cost; use the default `Reparameterized` estimator, since the guide has no density at arbitrary points.

<!-- python: skip -->
```python
from quivers.inference import AutoDAIS, ELBO

base   = AutoNormalGuide(model, observed_names={"y"})
guide  = AutoDAIS(
    base, model, observations={"y": y_data},
    num_steps=8, init_step_size=0.05,
)
elbo   = ELBO()
svi    = SVI(model, guide, optimizer, elbo)
```

## Predictive

`Predictive` is the same regardless of which inference you ran:

<!-- python: skip -->
```python
from quivers.inference import Predictive

# `posterior` accepts either a trained Guide or an MCMCResult
pred = Predictive(model, posterior=guide,       num_samples=1000)
pred = Predictive(model, posterior=mcmc_result, num_samples=1000)

posterior_predictive = pred(x_test)           # dict of (num_samples, ...) tensors
```

## Reading the benchmark grid

Whenever you're unsure which guide handles a model shape, look at
`docs/developer/inference-benchmarks.md`. The benchmark grid runs its listed algorithms against synthetic posteriors with reference answers (conjugate models, hierarchical structures, pathological geometries, multimodal mixtures, latent-variable models, constrained supports), reports the metrics, and gives you an empirical answer. If a guide is missing for your model shape, the failing-cell pattern often suggests the right reparameterization or a different guide.

## Try this

- Take the eight-schools model from chapter 3 and run every guide on it, comparing posterior `tau` recovery against the NUTS reference.
- Take the GMM from chapter 4 and compare an `AutoMixtureGuide` over four `AutoNormalGuide` components with a single `AutoNormalGuide`, accounting for label switching when you assess mode recovery.
- Run `AutoDAIS` on the donut posterior (a non-simply-connected support); compare to `AutoIAFGuide`.

## Next

[Chapter 7](07-categorical.md) describes the categorical structure behind
algebras, change-of-base, and enriched composition. Chapters 9–12 then build
indexed families, authored handlers, checked search, and structural
attachments; chapter 13 collects the release checks for a QVR module.


## Sources and further reading

- *MCMC Using Hamiltonian Dynamics* [@neal2011mcmc].
