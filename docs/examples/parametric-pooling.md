# Parametric Partial Pooling

## Overview

Rubin's parallel-experiments model [@rubin1981parallel], best known from the eight-schools data, combines estimates from several experiments. Each school $j$ reports an estimated effect $y_j$ with a known standard error $s_j$, and the true effects $\mu + \theta_j$ are drawn around a common mean with a between-school spread $\tau$:

$$
\mu \sim \mathcal{N}(0, 10^2), \quad \tau \sim \mathrm{HalfCauchy}(5), \quad
\theta_j = \tau z_j,\ z_j \sim \mathcal{N}(0, 1), \quad
y_j \sim \mathcal{N}(\mu + \theta_j, s_j^2).
$$

The posterior mean of each school's effect lies between its own estimate $y_j$ and the common mean, and how far it moves depends on $\tau$ relative to $s_j$: this is partial pooling. When $\tau$ is inferred, the data decide how much to pool. When $\tau$ is fixed, the analyst does.

The example states this model with a [parametric program template](../guides/dsl-programs-and-lets.md#parametric-programs). The template `school_effects` declares the [non-centered parametrization](https://doi.org/10.1214/088342307000000014) of a set of effects once [@papaspiliopoulos2007parametrization], typed over a scalar `spread : Real` and a [plate](https://en.wikipedia.org/wiki/Plate_notation) `K : FinSet`. The host program `pooled` instantiates it at a sampled spread `tau`, and the hosts `pooled_tight` and `pooled_loose` instantiate it at the fixed spreads 2 and 20. A labeled return tuple names each host's outputs.

## QVR source

```qvr
# Parametric Partial Pooling
#
# The eight-schools model of Rubin (1981): each school j reports an
# estimated treatment effect y_j with a known standard error se_j,
# and the true effects are drawn around a common mean. A parametric
# program template fixes the non-centered form of the school effects
# once; each host program instantiates it at a different between-
# school spread, so one source states the fully adaptive model and
# two fixed-spread variants side by side.
#
# Generative structure:
#
#   mu       ~ Normal(0, 10)           common mean effect
#   tau      ~ HalfCauchy(5)           between-school spread
#   z_j      ~ Normal(0, 1)            standardized school effect
#   theta_j  = tau * z_j               non-centered scaling
#   y_j      ~ Normal(mu + theta_j, se_j)
#
# The host program `pooled` samples tau, so the data decide how much
# the school effects are pooled toward mu. The hosts `pooled_tight`
# and `pooled_loose` instantiate the same template at a fixed spread
# of 2 and of 20, which shrinks the effects strongly or barely. The
# standard errors se are host data, as in Rubin's analysis. Each host
# returns a labeled tuple naming the school deviations and the mean.
#
# School is the plate the estimates are observed over. Effect and
# Mean are the value spaces of the two returned quantities, so each
# host program's codomain is the product Effect * Mean.
#
# References: [Rubin 1981](https://doi.org/10.3102/10769986006004377),
# [Papaspiliopoulos et al. 2007](https://doi.org/10.1214/088342307000000014).

object School : FinSet 40
object Effect : Real 1
object Mean : Real 1

program school_effects(spread : Real, K : FinSet) : K -> Effect
    sample z : K <- Normal(loc=0.0, scale=1.0)
    let effect = spread * z
    return effect

program pooled : School -> Effect * Mean
    sample mu <- Normal(loc=0.0, scale=10.0)
    sample tau <- HalfCauchy(scale=5.0)
    sample theta <- school_effects(tau, School)
    let school_mean = mu + theta
    observe y : School <- Normal(loc=school_mean, scale=se)
    return (effects: theta, mean: mu)

program pooled_tight : School -> Effect * Mean
    sample mu <- Normal(loc=0.0, scale=10.0)
    sample theta <- school_effects(2.0, School)
    let school_mean = mu + theta
    observe y : School <- Normal(loc=school_mean, scale=se)
    return (effects: theta, mean: mu)

program pooled_loose : School -> Effect * Mean
    sample mu <- Normal(loc=0.0, scale=10.0)
    sample theta <- school_effects(20.0, School)
    let school_mean = mu + theta
    observe y : School <- Normal(loc=school_mean, scale=se)
    return (effects: theta, mean: mu)

export pooled
```

## Understanding the model

A program template is a reusable local model. Calling it creates a set of random draws with [site names](../tutorials/python/05-variational-inference.md#tracing-the-model) prefixed by the call site, just as calling a function creates separate local bindings. The `spread` argument determines how far each school's effect can lie from the common mean. A smaller spread imposes stronger [pooling](../tutorials/qvr/03-hierarchical.md#eight-schools-centered) toward it.

The three enclosing programs differ in their treatment of the spread. In `pooled` it is the latent `tau`, so its posterior, and with it the amount of pooling, is learned from how much the estimates vary beyond what their standard errors explain. In `pooled_tight` it is fixed at 2, smaller than every standard error in the example, so the school effects are pulled almost all the way to the common mean. In `pooled_loose` it is fixed at 20, larger than every standard error, so each school's effect stays close to its own estimate. These two bracket the adaptive model. Their posterior estimates answer the same school-effect questions under different pooling assumptions.

The standard errors `se` are input data, as in Rubin's analysis, where each school's estimate comes with a standard error computed from its own experiment. They are not inferred, and that is what lets the model separate within-school noise from between-school variation.

## Walkthrough

`school_effects` is a parametric program: its parameter list carries typed parameters rather than data parameters, so it is a computation over the extent of `K` and the real `spread`. The draw `sample theta <- school_effects(tau, School)` substitutes `tau` for `spread` and `School` for `K`, then runs an alpha-renamed copy of the body in the host program: the local `z` becomes `theta$z` and the return variable `effect` becomes `theta` itself, on the [reference machine](../developer/qiec.md#reference-evaluator), the torch runtime, and every transpile target alike. An argument may be a literal, as in the two fixed-spread hosts, or a value sampled earlier in the host, as in `pooled`.

The non-centered form routes `spread` through the `let` arithmetic (`effect = spread * z`) rather than through the scale of a plate draw, so the sampled latent is the standardized `z` and the effect is a deterministic function of it. With few or noisy schools, the posterior of $\tau$ often reaches near zero, where the centered form $\theta_j \sim \mathcal{N}(0, \tau^2)$ produces a funnel that is hard to sample. The non-centered form can improve this geometry when the data weakly constrain the effects [@papaspiliopoulos2007parametrization].

`let school_mean = mu + theta` adds the common mean, and `observe y : School <- Normal(loc=school_mean, scale=se)` scores each school's estimate against its own known standard error.

`object School : FinSet 40` is the plate the estimates are observed over, while `object Effect : Real 1` and `object Mean : Real 1` are the value spaces of the two returned quantities. Each host program is thus declared `School -> Effect * Mean`: its domain is the school plate and its codomain is the product space its labeled return tuple inhabits. The template's own signature `K -> Effect` follows the same reading, with the plate left as the type-parameter `K` the call site fixes.

`return (effects: theta, mean: mu)` labels the output tuple. The compiled program's [`rsample`](../api/continuous/programs.md) returns a dict keyed by the labels (`effects`, `mean`) rather than by the internal variable names, and [`log_joint`](../api/continuous/programs.md) accepts intermediates keyed either way. `export pooled` selects which host the compiled module wraps. The [`Compiler`](../api/dsl/compiler.md#quivers.dsl.compiler.Compiler) environment holds all three.

## DSL features

- **Typed program parameters**: `(spread : Real, K : FinSet)` declares a parametric template. Admissible parameter kinds are `Real`, `Nat`, `FinSet`, `Space`, `Object`, and `Mor[A, B]`.
- **Template instantiation**: `sample v <- template(args)` inlines the substituted, alpha-renamed body at the call site, with literal or sampled arguments.
- **Labeled return tuple**: `return (label: var, ...)` names the program's outputs on the compiled output path.
- **`export`**: selects the module's exported morphism among the declared candidates.

## Try it

The blocks below simulate 40 schools from the adaptive model, check its compiled joint against a hand-written density, fit all three hosts by [SVI](../guides/inference-svi.md) (stochastic variational inference), check that the adaptive fit recovers the common mean and the spread and predicts the true effects better than the raw estimates, check that the fixed spreads bracket its shrinkage, and run a short [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler) chain on the adaptive model. The model's scalar sites (`mu`, `tau`) combine with `School`-shaped sites, so the program input is a single row, `torch.zeros(1, 1)`. The plate's extent comes from the observed estimates.

### Generating synthetic data

```python
import torch
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse_file

D = torch.distributions

compiler = Compiler(parse_file("docs/examples/source/parametric_pooling.qvr"))
prog = compiler.compile()
model = prog.morphism
hosts = {name: compiler.morphisms[name] for name in ("pooled", "pooled_tight", "pooled_loose")}

torch.manual_seed(0)
n_school = 40
se = 5.0 + 10.0 * torch.rand(n_school)
true_mu, true_tau = 8.0, 6.0
true_theta_z = torch.randn(n_school)
true_effect = true_mu + true_tau * true_theta_z
y = D.Normal(true_effect, se).sample()

observations = {"y": y, "se": se}
x_in = torch.zeros(1, 1)
```

The standard errors range from 5 to 15, so they straddle the true spread of 6: some schools are measured precisely enough to stand mostly on their own, and others are mostly noise.

### Checking the joint density

At the true values, [`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) of the adaptive model should equal the three prior terms plus the Normal likelihood of each estimate under its own standard error, written out by hand. The template's draw appears under its call-site name `theta$z`.

```python
truth = {
    "mu": torch.tensor(true_mu),
    "tau": torch.tensor(true_tau),
    "theta$z": true_theta_z,
    **observations,
}
log_joint = model.log_joint(x_in, truth)[0]

by_hand = (
    D.Normal(0.0, 10.0).log_prob(torch.tensor(true_mu))
    + D.HalfCauchy(5.0).log_prob(torch.tensor(true_tau))
    + D.Normal(0.0, 1.0).log_prob(true_theta_z).sum()
    + D.Normal(true_effect, se).log_prob(y).sum()
)
assert torch.allclose(log_joint, by_hand, rtol=1e-5)
print(f"log joint at the truth: {float(log_joint):.2f}")
```

### SVI fits

Fit an [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) to each host by [`SVI`](../api/inference/svi.md#svi) and summarize each school's effect $\mu + \text{spread} \cdot z_j$ by its mean over 400 guide draws. The raw estimates $y_j$ are the no-pooling baseline.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

FIXED_SPREAD = {"pooled_tight": 2.0, "pooled_loose": 20.0}


def fit_host(name, steps=1000):
    host = hosts[name]
    torch.manual_seed(1)
    guide = AutoNormalGuide(host, observed_names={"y", "se"})
    optim = torch.optim.Adam(
        list(host.parameters()) + list(guide.parameters()), lr=5e-2,
    )
    svi = SVI(host, guide, optim, ELBO(num_particles=4))
    for _ in range(steps):
        svi.step(x_in, observations)
    with torch.no_grad():
        draws = [guide.rsample(x_in) for _ in range(400)]
    mu = torch.stack([d["mu"].reshape(-1)[0] for d in draws])
    if name == "pooled":
        spread = torch.stack([d["tau"].reshape(-1)[0] for d in draws])
    else:
        spread = torch.full_like(mu, FIXED_SPREAD[name])
    z = torch.stack([d["theta$z"].reshape(-1) for d in draws])
    effects = (mu[:, None] + spread[:, None] * z).mean(0)
    return {"mu": float(mu.mean()), "spread": float(spread.mean()), "effects": effects}


fits = {name: fit_host(name) for name in hosts}


def mse(estimate):
    return float((estimate - true_effect).pow(2).mean())


print(f"{'':13s} {'mu':>6s} {'spread':>7s} {'sd of effects':>14s} {'MSE':>7s}")
print(f"{'raw y':13s} {'':6s} {'':7s} {float(y.std()):14.2f} {mse(y):7.1f}")
for name, fit in fits.items():
    print(
        f"{name:13s} {fit['mu']:6.2f} {fit['spread']:7.2f} "
        f"{float(fit['effects'].std()):14.2f} {mse(fit['effects']):7.1f}"
    )

adaptive = fits["pooled"]
assert abs(adaptive["mu"] - true_mu) < 4.0
assert abs(adaptive["spread"] - true_tau) < 4.0
assert mse(adaptive["effects"]) < 0.5 * mse(y)

dispersion = {name: float(fit["effects"].std()) for name, fit in fits.items()}
assert dispersion["pooled_tight"] < dispersion["pooled"] < dispersion["pooled_loose"]
```

The tolerances on $\mu$ and $\tau$ are about two standard errors for 40 schools of this precision (for $\mu$, the precision-weighted standard error $\bigl(\sum_j 1/(\tau^2 + s_j^2)\bigr)^{-1/2}$ is about 1.7). The prediction check is the classical case for partial pooling: the adaptive posterior means should be much closer to the true effects than the raw estimates, whose expected mean squared error is the average squared standard error (about 100 for these schools). The last check is the template's point: the tight spread shrinks the school effects toward their mean more than the adaptive fit does, and the loose spread less.

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) runs on the adaptive model, with $\mu$, $\tau$, and the standardized effects latent. The chain below is short. A real analysis would run several chains and check their diagnostics, especially for $\tau$.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in, observations)

assert torch.isfinite(result.log_densities).all()
print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. A parameterized program gives a family $\{K_p : \mathrm{dom}(p) \to \mathcal{G}(\mathrm{cod}(p))\}_{p \in P}$ of sampling kernels, where $P$ is the set of admissible arguments and $\mathcal{G}(X)$ is the space of probability measures on $X$. Multiple arguments range over the corresponding product of parameter sets. A literal argument selects one member of the family. Drawing `tau` from its prior $\pi$ and then sampling from $K_\tau$ gives the mixture $\int K_\tau\,\pi(d\tau)$.

Observations weight this joint and determine a posterior over `tau`. Averaging posterior predictions uses that posterior, whereas the generative composition above uses the prior [@staton2017commutative]. The labeled return tuple names the projections from the product of returned values.
