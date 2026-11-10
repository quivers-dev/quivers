# Bayesian Gaussian Mixture Model

## Overview

A finite [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) treats each observed value as a draw from one of $K$ [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) components. The components share three per-component vector parameters: the mixing weights on the simplex, the component locations, and the component scales. A [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior governs the mixing weights, and independent priors govern the locations and scales.

Rather than sample a discrete per-row component label and integrate it out by hand, this example scores each row directly against the mixture through the [`MixtureNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMixtureNormal) likelihood. The per-row marginal is the closed-form convex combination

$$
p(r_n) \;=\; \sum_{k=1}^{K} \mathrm{probs}[k] \; \mathcal{N}\!\bigl(r_n;\, \mu[k],\, \sigma[k]\bigr),
$$

so the model carries no discrete latent: `MixtureNormal` integrates the component assignment out analytically, evaluating the marginal by [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) over the $K$ components.

## QVR source

```qvr
# Bayesian Gaussian Mixture Model
#
# A finite Gaussian mixture with K components. Per-component means,
# scales, and mixing weights are latents on the Component plate; each
# observed row is drawn from the resulting mixture. The per-row
# component assignment is integrated out in closed form by the
# MixtureNormal likelihood, so the model carries no discrete latent.
#
# Generative structure:
#
#   probs    ~ Dirichlet(alpha)                    Component-simplex mixing weights
#   mu_k     ~ Normal(0, 5)                        per-component mean
#   sigma_k  ~ HalfNormal(1)                       per-component scale
#   r_n      ~ MixtureNormal(probs, mu, sigma)     observed row
#
# Per-row marginal likelihood (closed form):
#
#   p(r_n) = sum_k probs[k] * Normal(r_n; mu[k], sigma[k]).
#
# MixtureNormal takes three per-component vector parameters (the
# mixing weights, the locations, and the scales), each shared across
# every row of the Resp plate, and scores each row against the
# K-component mixture they define.
#
# Resp is the plate the rows are observed over; Weights is the
# value space of the returned mixing weights, a point of the
# Component simplex embedded in R^K.

composition log_prob [level=algebra]

object Component : FinSet 3
object Resp : FinSet 500
object Weights : Real 3

program gmm(alpha : Real) : Resp -> Weights
    sample probs <- Dirichlet(concentration=alpha) [over=Component]
    sample mu : Component <- Normal(loc=0.0, scale=5.0)
    sample sigma : Component <- HalfNormal(scale=1.0)

    observe r : Resp <- MixtureNormal(weights=probs, loc=mu, scale=sigma)

    return probs

export gmm
```

## Understanding the model

For one observed value, each component contributes its density weighted by
its probability. If two components have weights `0.4` and `0.6` and give
that value densities `0.1` and `0.5`, its [mixture density](../tutorials/qvr/04-marginalize.md#a-two-component-gaussian-mixture)
is `0.4 * 0.1 + 0.6 * 0.5 = 0.34`. `MixtureNormal` [sums over component
assignments](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#summing-out-the-responsibility-parameter)
[@stanFiniteMixtures], eliminating the need to sample a label when scoring
an observation. Inference still estimates the uncertain component means and scales.

Component names do not identify physical groups: swapping their labels leaves
the mixture density unchanged. Compare predicted densities or use a stated
labeling convention before averaging component-specific draws. The fit below
infers the weights, locations, and scales jointly and aligns labels by sorting
the fitted locations.

## Walkthrough

`composition log_prob [level=algebra]` selects the log-probability semiring so the program's `Score` effect accumulates log-densities additively. `object Component : FinSet 3` and `object Resp : FinSet 500` declare the two discrete plates: $K = 3$ mixture components and $N = 500$ observed rows, and `object Weights : Real 3` is the value space of the returned mixing weights, a point of the component simplex embedded in $\mathbb{R}^K$. `program gmm(alpha : Real) : Resp -> Weights` parameterizes the program by the Dirichlet concentration and declares that what it returns is a mixing-weight vector rather than a row index.

The three `sample` steps draw the shared per-component parameters:

- `sample probs <- Dirichlet(alpha) [over=Component]` draws the mixing weights as a single point on the `Component` simplex; `over=Component` names the family's event axis (Dirichlet event-rank 1).
- `sample mu : Component <- Normal(0.0, 5.0)` draws the $K$ component locations as plate-bound continuous latents, one per component.
- `sample sigma : Component <- HalfNormal(1.0)` draws the $K$ positive component scales the same way.

`observe r : Resp <- MixtureNormal(probs, mu, sigma)` scores each observed row against the $K$-component mixture the three shared vectors define. `MixtureNormal` takes the mixing weights, the locations, and the scales as three per-component vectors, each broadcast across every row of the `Resp` plate, and returns the per-row marginal $\sum_k \mathrm{probs}[k]\,\mathcal{N}(r_n;\,\mu[k],\,\sigma[k])$ in closed form. No component-assignment latent is sampled: the mixture likelihood integrates it out.

`return probs` projects the program's joint kernel onto the mixing-weight site.

## Try it

The blocks below simulate a corpus from the model, check the compiled joint against a hand-written density, fit all three parameter vectors by SVI, and run a short NUTS chain on the same model.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/mixture_model.qvr")
fit = prog.gmm(alpha=1.0)
model = fit.morphism

K, N = 3, 500
true_probs = torch.tensor([0.4, 0.35, 0.25])
true_mu    = torch.tensor([-3.0, 0.0, 3.0])
true_sigma = torch.tensor([0.5, 0.7, 0.4])

comps = torch.distributions.Categorical(true_probs).sample((N,))
r     = torch.distributions.Normal(true_mu[comps], true_sigma[comps]).sample()

observations = {"r": r}
x_in = torch.zeros(N, 1)
```

The rows are drawn by sampling a component per row and then a value from that component's Gaussian. Only the rows `r` enter `observations`: the component labels, the mixing weights `probs`, the locations `mu`, and the scales `sigma` are all left for inference.

### Checking the joint density

At the true parameters, [`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) should equal the Dirichlet, Normal, and half-Normal prior terms plus the closed-form mixture log-likelihood, written here with torch's own [`MixtureSameFamily`](https://docs.pytorch.org/docs/stable/distributions.html#mixturesamefamily).

```python
D = torch.distributions

truth = {"r": r, "probs": true_probs, "mu": true_mu, "sigma": true_sigma}
log_joint = model.log_joint(x_in, truth)[0]

mixture = D.MixtureSameFamily(D.Categorical(true_probs), D.Normal(true_mu, true_sigma))
by_hand = (
    D.Dirichlet(torch.ones(K)).log_prob(true_probs)
    + D.Normal(0.0, 5.0).log_prob(true_mu).sum()
    + D.HalfNormal(1.0).log_prob(true_sigma).sum()
    + mixture.log_prob(r).sum()
)
assert torch.allclose(log_joint, by_hand, rtol=1e-5)
print(f"log joint at the truth: {float(log_joint):.2f}")
```

### SVI fit

The mixture likelihood is multimodal: a fit can settle where two components share one cluster, and every relabeling of the components is an equally good optimum. The block below thus runs three restarts from different seeds, keeps the one with the best average ELBO over its last 100 steps (a criterion that never looks at the truth), and summarizes each site by the mean of 200 draws from the fitted [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide). Components are aligned to the truth by sorting on the fitted locations [@stephens2000labels].

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI


def fit_restart(seed, steps=1000):
    torch.manual_seed(seed)
    guide = AutoNormalGuide(model, observed_names={"r"})
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=5e-2,
    )
    svi = SVI(model, guide, optim, ELBO(num_particles=1))
    losses = [svi.step(x_in, observations) for _ in range(steps)]
    with torch.no_grad():
        draws = [guide.rsample(x_in) for _ in range(200)]
    summary = {
        name: torch.stack([d[name] for d in draws]).mean(0).reshape(-1)
        for name in ("probs", "mu", "sigma")
    }
    return sum(losses[-100:]) / 100, summary


restarts = [fit_restart(seed) for seed in (1, 2, 3)]
for seed, (loss, _) in zip((1, 2, 3), restarts):
    print(f"seed {seed}: final ELBO loss {loss:.1f}")
best_loss, best = min(restarts, key=lambda pair: pair[0])

order = best["mu"].argsort()
fitted = {name: value[order] for name, value in best.items()}
for name, true_value in (("probs", true_probs), ("mu", true_mu), ("sigma", true_sigma)):
    print(f"{name:5s} fitted {fitted[name].numpy().round(2)}  true {true_value.numpy()}")

assert (fitted["probs"] - true_probs).abs().max() < 0.06
assert (fitted["mu"] - true_mu).abs().max() < 0.15
assert (fitted["sigma"] - true_sigma).abs().max() < 0.15
```

On this corpus two of the three restarts end with a loss about 230 nats worse than the third, at a configuration where the components overlap instead of separating; a single-seed fit could thus report a poor optimum as if it were the posterior. The tolerances are about three sampling standard errors at $N = 500$ (for instance, $\sqrt{0.25 \cdot 0.75 / 500} \approx 0.02$ for the smallest weight), so they leave room for finite-sample error but not for a collapsed or merged component.

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) targets the same joint, with all three parameter vectors latent. The chain below is short; a real analysis would run several chains and check their diagnostics.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

assert torch.isfinite(result.log_densities).all()
print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The per-row likelihood `MixtureNormal(probs, mu, sigma)` is a conditional [Kleisli arrow](https://en.wikipedia.org/wiki/Kleisli_category) into the real-valued response space in the [Giry monad](https://doi.org/10.1007/BFb0092872). It is a finite convex combination of the $K$ Gaussian component measures, weighted by the categorical measure $\mathrm{probs}$ on `Component`: the Giry-monad mixture operation that draws a component from $\mathrm{Categorical}(\mathrm{probs})$, then a value from the chosen Gaussian, and keeps the marginal on the value. Equivalently, `MixtureNormal` is the [pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) of the joint component-and-value measure along the projection $\mathsf{Component} \times \mathbb{R} \to \mathbb{R}$, which is exactly the closed-form marginal $\sum_k \mathrm{probs}[k]\,\mathcal{N}(\cdot;\,\mu[k],\,\sigma[k])$. Because the component index is integrated out inside the likelihood rather than sampled, the program carries no discrete latent to marginalize and every site it holds is continuous. [@giry1982categorical]

## See also

- [Latent Dirichlet Allocation](lda.md), a topic model in which every token draws its own topic from per-document weights and a `marginalize` block integrates each token's topic out exactly.
