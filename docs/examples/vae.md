# Latent decoder with encoder and decoder paths

## Overview

A [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) [@kingma2014autoencoding] trains a decoder with an observation-dependent encoder used as the variational guide. This source declares both an encoder-decoder composition and a prior-decoder composition, but exports only `vae_program`, whose latent `z` is sampled from `prior`. The runnable SVI block uses `AutoNormalGuide`; it does not use the declared `encoder` as an amortized guide. Thus this page demonstrates the two paths needed for a VAE without implementing joint VAE training.

## QVR source

```qvr
# Latent Decoder with Encoder and Decoder Paths
#
# The two network paths needed by a variational autoencoder,
# expressed with stack and explicit Kleisli composition. The
# exported program samples z from the prior and runs only the
# decoder. The declared encoder is not connected to the
# AutoNormalGuide used on the page, so this source does not train
# an amortized VAE end to end.
#
# Structural form:
#
#   encoder     = pixel_embed >> stack(enc_deep, 1) >> enc_to_latent
#   decoder     = dec_1       >> stack(dec_deep, 1) >> dec_to_obs
#   generative  = prior   >> decoder              ancestral sampling
#   reconstruct = encoder >> decoder              posterior predictive
#
# The encoder maps observations to a distribution over latent
# codes; the decoder maps latent codes back to observation space.
# A complete VAE must use the encoder as an observation-dependent
# guide and add the corresponding KL term.
#
# Reference: [Kingma and Welling 2014](https://arxiv.org/abs/1312.6114).

object Pixel : FinSet 8
object Latent : Real 4
object EncoderHidden, DecoderHidden : Real 16
object ObsSpace : Real 8
object UnitSpace : Real 1

morphism pixel_embed : Pixel -> EncoderHidden [role=embed]
morphism enc_deep : EncoderHidden -> EncoderHidden [param_source=mlp] ~ Normal
morphism enc_to_latent : EncoderHidden -> Latent ~ Normal

define encoder = pixel_embed >> stack(enc_deep, 1) >> enc_to_latent

morphism prior : UnitSpace -> Latent ~ Normal

morphism dec_1 : Latent -> DecoderHidden ~ Normal
morphism dec_deep : DecoderHidden -> DecoderHidden [param_source=mlp] ~ Normal
morphism dec_to_obs : DecoderHidden -> ObsSpace ~ Normal

define decoder = dec_1 >> stack(dec_deep, 1) >> dec_to_obs

define generative = prior >> decoder
define reconstruct = encoder >> decoder

# Probabilistic model for the generative branch: sample the
# latent code under the standard-Normal prior, then push it
# through the decoder Kleisli morphism to score the observation
# Y. The decoder's per-layer weights carry the kernel-prior
# Normals declared above; the program traces both the latent and
# the observation sites for inference.
program vae_program : UnitSpace -> ObsSpace
    sample z <- prior
    observe Y <- decoder(z)
    return Y

export vae_program
```

## Understanding the model

Generation starts from a prior draw of a latent code and asks what observation
the decoder would produce. Reconstruction starts from an observation and asks
which code its encoder proposes before decoding. A trained VAE uses that
observation-dependent encoder as its guide, sharing the inference mapping
across observations; this is [amortized inference](../tutorials/python/05-variational-inference.md#creating-a-guide).

The runnable fit below optimizes an `AutoNormalGuide` for `z`. Training the
declared encoder as a [VAE guide](https://pyro.ai/examples/vae.html#VAE-in-Pyro)
requires calling it on each observation and passing its parameters to the
optimizer alongside the decoder's [@pyroVAE]. The encoder then learns to
predict a distribution over latent values for each new observation.

## Walkthrough

The encoder begins with `morphism pixel_embed : Pixel -> EncoderHidden [role=embed]`, a deterministic embedding lookup mapping the discrete `Pixel` object into the continuous `EncoderHidden` space. The `stack(enc_deep, 1)` combinator inserts one independently-parameterized stochastic Normal hidden layer, distinct from `repeat(enc_deep, 1)` which would weight-tie. The final `enc_to_latent` projects to the latent space at small init scale.

The decoder mirrors the encoder: an initial `dec_1` lifts the latent code into the decoder hidden width, one stacked deep layer `stack(dec_deep, 1)` adds depth, and `dec_to_obs` projects to the observation space at tight init scale (the reconstruction should be more precise than the encoding).

The two top-level compositions

<!-- compile: false -->
```qvr
define generative = prior >> decoder
define reconstruct = encoder >> decoder
```

express generative and reconstruction-shaped execution paths. The exported `vae_program` traverses `prior` and `decoder`; the current ELBO does not traverse `reconstruct`. A complete VAE would connect `encoder` to a guide conditioned on the observation.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

```python
import torch
from quivers.dsl import load
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/vae.qvr")
model = prog.morphism

N = 32
unit = torch.zeros(N, 1)
x_in = unit

# Run one forward trace under the program's own parameters so the
# captured latent and observation are jointly consistent under the
# generative kernel.
with torch.no_grad():
    forward = trace(model, unit)
true_z = forward.sites["z"].value.detach()
Y = forward.sites["Y"].value.detach()

observations = {"Y": Y, "z": true_z}
print("Y shape:", Y.shape)
```

The exported `vae_program` is a [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) morphism `UnitSpace -> ObsSpace` whose body samples the latent `z` and scores the observation `Y` under `decoder(z)`; the forward trace captures a jointly-consistent `(z, Y)` pair, which the observation dict then clamps while [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts the entire parameter vector into a Bayesian model for SVI and NUTS.

### SVI fit

```python
from quivers.inference import (
    AutoNormalGuide, ELBO, SVI, bayesian_lift_parameters,
)

model, x_in, observations = bayesian_lift_parameters(
    prog.morphism, x_in, observations, prior_scale=1.0,
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names=set())
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(50)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
# The lifted parameter vector is high-dimensional, so a small
# step size and shallow tree keep one full chain inside a
# documentation-friendly budget.
kernel = NUTSKernel(step_size=0.005, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=5, num_samples=5, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

The two compositions `prior >> decoder` and `encoder >> decoder` share the decoder but differ in which morphism produces the latent code. `stack(f, N)` creates independently parameterized copies, unlike a weight-sharing repetition. Only the prior-decoder program participates in the runnable inference block on this page.

For a complete VAE with guide $q(z \mid y)$, the ELBO is the expected decoder log likelihood minus $\operatorname{KL}(q(z \mid y) \| p(z))$. The first term rewards reconstruction and the second compares the guide with the latent prior. That interpretation requires connecting and training an observation-dependent guide, which the declared `encoder` does not do in the runnable fit here.

## See also

- [Probabilistic PCA](ppca.md) for a linear-Gaussian latent-variable model.
- [DSL Guide](../guides/dsl-overview.md) for morphism composition (`>>`, `stack`, `embed`).

## Sources and further reading

- *Auto-Encoding Variational Bayes* [@kingma2014autoencoding].
