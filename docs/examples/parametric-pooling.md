# Parametric Partial Pooling

## Overview

A one-way random-effects model [@rubin1981parallel] whose group-level effects come from a [parametric program template](../guides/dsl-programs-and-lets.md#parametric-programs). The template `school_effects` declares the [non-centered parametrization](https://doi.org/10.1214/088342307000000014) of a set of partially pooled effects once, typed over a scalar `spread : Real` and a plate `K : FinSet`; the two host programs instantiate it at a tight (`0.6`) and a loose (`2.5`) between-group spread. A `score` step adds a soft sum-to-zero factor on the effects [@morris2019spatial], and a labeled return tuple names the program's outputs. [@papaspiliopoulos2007parametrization]

## QVR source

```qvr
# Parametric Partial Pooling
#
# A one-way random-effects model whose group-level effects come
# from a parametric program template. The template fixes the
# non-centered pooling structure once; each host program
# instantiates it at a concrete between-group spread, so a single
# source expresses tight and loose partial pooling side by side.
#
# Generative structure:
#
#   z_j      ~ Normal(0, 1)              standardized group effect
#   theta_j  = spread * z_j              non-centered scaling
#   sigma    ~ LogNormal(0, 0.5)         observation noise scale
#   y_j      ~ Normal(theta_j, sigma)    per-group observed summary
#
# The score step multiplies the joint density by a soft
# sum-to-zero factor on the group effects, the standard
# identification device for random effects, and the labeled
# return tuple names the program's outputs. The export selects
# which host program the compiled module runs.
#
# School is the plate the group summaries are observed over.
# Effect and Scale are the value spaces of the two returned
# quantities, both real scalars, so each host program's codomain
# is the product Effect * Scale that its labeled return tuple
# inhabits.
#
# References: [Rubin 1981](https://doi.org/10.3102/10769986006004377),
# [Papaspiliopoulos et al. 2007](https://doi.org/10.1214/088342307000000014),
# [Morris et al. 2019](https://doi.org/10.1016/j.sste.2019.100301).

object School : FinSet 8
object Effect : Real 1
object Scale : Real 1

program school_effects(spread : Real, K : FinSet) : K -> Effect
    sample z : K <- Normal(loc=0.0, scale=1.0)
    let effect = spread * z
    return effect

program pooled_tight : School -> Effect * Scale
    sample theta <- school_effects(0.6, School)
    sample sigma <- LogNormal(loc=0.0, scale=0.5)
    let total_effect = sum(theta)
    score centering = -50.0 * total_effect * total_effect
    observe y : School <- Normal(loc=theta, scale=sigma)
    return (effects: theta, scale: sigma)

program pooled_loose : School -> Effect * Scale
    sample theta <- school_effects(2.5, School)
    sample sigma <- LogNormal(loc=0.0, scale=0.5)
    let total_effect = sum(theta)
    score centering = -50.0 * total_effect * total_effect
    observe y : School <- Normal(loc=theta, scale=sigma)
    return (effects: theta, scale: sigma)

export pooled_tight
```

## Understanding the model

A program template is a reusable local model. Calling it twice creates two
sets of random draws with distinct [site names](../tutorials/python/05-variational-inference.md#tracing-the-model), just as calling a function
twice creates separate local bindings. The `spread` argument determines how
far each group's effect can lie from zero; a smaller spread imposes stronger
[pooling](../tutorials/qvr/03-hierarchical.md#eight-schools-centered) toward the common baseline.

The [centering score](https://mc-stan.org/docs/stan-users-guide/regression.html#soft-centering)
adds a finite penalty when the group effects sum to a value away from zero
[@stanRegression]. Nonzero sums remain possible; the penalty controls how
strongly they are discouraged. The returned labels identify group effects
and residual scales even when their internal site names differ between
template instantiations.

## Walkthrough

`school_effects` is a parametric program: its parameter list carries typed parameters rather than data parameters, so it is a computation over the extent of `K` and the real `spread`, runnable on its own with those given. The draw `sample theta <- school_effects(0.6, School)` substitutes `0.6` for `spread` and `School` for `K`, then runs an alpha-renamed copy of the body in the host program: the local `z` becomes `theta$z` and the return variable `effect` becomes `theta` itself, on the reference machine, the torch runtime, and every transpile target alike (which spell the site `theta__z`). Each draw contributes fresh latents, which is why `pooled_tight` and `pooled_loose` share structure but not parameters.

The non-centered form routes `spread` through the `let` arithmetic (`effect = spread * z`) rather than through a plate-draw scale, so the bound value shapes both forward simulation and the joint density. `score centering = -50.0 * total_effect * total_effect` adds $-\tfrac{1}{2}(\sum_j \theta_j / 0.1)^2$ to the program's log-joint: a soft sum-to-zero constraint that pins the effects' grand mean near zero without a hard reparametrization.

`object School : FinSet 8` is the plate the group summaries are observed over, while `object Effect : Real 1` and `object Scale : Real 1` are the value spaces of the two returned quantities. Each host program is thus declared `School -> Effect * Scale`: its domain is the group plate and its codomain is the product space its labeled return tuple inhabits. The template's own signature `K -> Effect` follows the same reading, with the plate left as the type-parameter `K` the call site fixes.

`return (effects: theta, scale: sigma)` labels the output tuple. The compiled program's [`rsample`](../api/continuous/programs.md) returns a dict keyed by the labels (`effects`, `scale`) rather than by the internal variable names, and [`log_joint`](../api/continuous/programs.md) accepts intermediates keyed either way. `export pooled_tight` selects which of the two host programs the compiled module wraps; swapping the export line for `export pooled_loose` selects the weakly pooled variant with no other change.

## DSL features

- **Typed program parameters**: `(spread : Real, K : FinSet)` declares a parametric template; admissible parameter kinds are `Real`, `Nat`, `FinSet`, `Space`, `Object`, and `Mor[A, B]`.
- **Template instantiation**: `sample v <- template(args)` inlines the substituted, alpha-renamed body at the call site.
- **Labeled return tuple**: `return (label: var, ...)` names the program's outputs on the compiled output path.
- **`score`**: adds an arbitrary tensor expression to the program's log-joint as an extra density factor.
- **`export`**: selects the module's exported morphism among the declared candidates.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/parametric_pooling.qvr")
model = prog.morphism

n_school = 8
true_spread = 0.6
true_sigma = 0.4
theta_true = true_spread * torch.randn(n_school)
theta_true = theta_true - theta_true.mean()
true_theta_z = theta_true / true_spread
y = torch.distributions.Normal(theta_true, true_sigma).sample()

observations = {"y": y}
x_in = torch.zeros(n_school, 1)

draws = model.rsample(x_in)
print(sorted(draws.keys()))
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Normal(theta_true, true_sigma).log_prob(y).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = []
for _ in range(200):
    losses.append(svi.step(x_in, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
print(f"oracle NLL:   {oracle_nll:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

A parametric program denotes a dependent kernel $\Pi (p_1 : P_1) \ldots \Pi (p_n : P_n).\ \mathbf{Kern}(\mathrm{dom}(p), \mathrm{cod}(p))$ in the indexed family of [Kleisli](https://ncatlab.org/nlab/show/Kleisli+category) arrows of the [Giry monad](https://doi.org/10.1007/BFb0092872) over the parameter category; each call site realizes a section of the family at the supplied arguments. The `score` step multiplies the weighted measure by a density factor. This particular nonpositive score gives a factor at most one, but arbitrary scoring does not generally preserve a subprobability bound; and the labeled return tuple names the projections out of the program's product codomain. [@giry1982categorical]

## Sources and further reading

- *A categorical approach to probability theory* [@giry1982categorical].
- *A General Framework for the Parametrization of Hierarchical Models* [@papaspiliopoulos2007parametrization].
- *Estimation in Parallel Randomized Experiments* [@rubin1981parallel].
- *Bayesian hierarchical spatial models: Implementing the Besag York Mollié model in stan* [@morris2019spatial].
