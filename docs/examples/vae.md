# Variational Autoencoder

## Overview

A [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) (VAE) [@kingma2014autoencoding] is a latent-variable model whose likelihood is a neural network, fitted together with a second network that approximates its posterior. Each observation $y_n$ has a code $z_n$:

$$
z_n \sim \mathcal{N}(0, I),
\qquad
y_n \mid z_n \sim \mathcal{N}\bigl(\mu_\theta(z_n), \operatorname{diag} \sigma_\theta^2(z_n)\bigr),
$$

where the decoder $(\mu_\theta, \sigma_\theta)$ is a one-hidden-layer tanh network. The encoder $q_\phi(z \mid y) = \mathcal{N}(m_\phi(y), \operatorname{diag} v_\phi^2(y))$ is a second network of the same shape. Training maximizes the evidence lower bound (ELBO)

$$
\mathcal{L}(\theta, \phi) = \sum_n \mathbb{E}_{q_\phi(z_n \mid y_n)}\bigl[\log p_\theta(y_n \mid z_n)\bigr] - \operatorname{KL}\bigl(q_\phi(z_n \mid y_n) \,\|\, p(z_n)\bigr)
$$

jointly in the decoder weights $\theta$ and the encoder weights $\phi$, with the reparameterized gradient. Because one encoder serves every observation, its posterior approximation is *amortized*: a new observation gets its code distribution from a single forward pass.

## QVR source

```qvr
# Variational Autoencoder
#
# The Gaussian variational autoencoder of Kingma and Welling. Each
# observation y_n has a latent code z_n under a standard-Normal
# prior, and a decoder network maps the code to a diagonal Normal
# over observation space. An encoder network maps an observation to
# a diagonal Normal over codes, the amortized variational posterior
# q(z | y) used to train both networks by the ELBO.
#
# Generative structure:
#
#   z_n            ~ Normal(0, I)                    latent code
#   mu(z), s(z)    = decoder MLP(z)                  one tanh layer
#   y_n            ~ Normal(mu(z_n), diag s(z_n))    observation
#
# Variational posterior:
#
#   m(y), v(y)     = encoder MLP(y)                  one tanh layer
#   q(z_n | y_n)   = Normal(m(y_n), diag v(y_n))
#
# Each network is a single conditional-Normal kernel whose mean and
# log-scale come from a tanh MLP parameter source, so the hidden
# layer is deterministic and the only random quantities are z and y.
# Row is the plate of observations. `reconstruct` composes the
# encoder with the decoder, mapping an observation to a distribution
# over its reconstruction.
#
# Reference: [Kingma and Welling 2014](https://doi.org/10.48550/arXiv.1312.6114).

object Row : FinSet 512
object Latent : Real 2
object ObsSpace : Real 5

morphism encoder : ObsSpace -> Latent [param_source=mlp(32)] ~ Normal
morphism decoder : Latent -> ObsSpace [param_source=mlp(32)] ~ Normal

define reconstruct = encoder >> decoder

program vae : Row -> ObsSpace
    sample z : Row <- Normal(loc=0.0, scale=1.0) [over=Latent]
    observe y : Row <- decoder(z)
    return y

export vae
export encoder
export decoder
export reconstruct
```

## Understanding the model

Generation runs left to right: draw a code from the prior, decode it to a Normal over observations, draw an observation. Inference runs the other way: from an observation, the encoder proposes a Normal over the codes that could have produced it. The ELBO rewards the decoder for explaining each observation from codes the encoder proposes for it, and penalizes the encoder for proposing codes the prior finds implausible. A VAE fitted this way can do three things the runnable blocks check: score new data, reconstruct an observation through `encoder >> decoder`, and capture curved structure that a linear latent model cannot.

The network weights are point estimates, not random variables. Only $z$ and $y$ are random, which is what distinguishes a VAE from a [Bayesian neural network](bnn.md).

## Walkthrough

`object Row : FinSet 512` is the plate of observations; `Latent : Real 2` and `ObsSpace : Real 5` are the code and observation spaces.

`morphism decoder : Latent -> ObsSpace [param_source=mlp(32)] ~ Normal` declares the decoder as a single conditional-Normal kernel. The [`MLPSource`](../api/continuous/param_source.md#quivers.continuous.param_source.MLPSource) behind it maps a code through one tanh layer of width 32 to a mean and a log-scale for each of the five observation coordinates, so the hidden layer is deterministic and the only randomness is the Normal draw at the output. `encoder` is the same construction in the other direction, from an observation to a mean and log-scale for each code coordinate.

`program vae : Row -> ObsSpace` is the generative model. `sample z : Row <- Normal(loc=0.0, scale=1.0) [over=Latent]` draws one standard-Normal code per row, and `observe y : Row <- decoder(z)` scores each row's observation under the decoder at that row's code. `define reconstruct = encoder >> decoder` is the composite kernel from an observation to the distribution of its reconstruction. The four exports make `prog.encoder`, `prog.decoder` and `prog.reconstruct` available next to the program, and all of them share one set of network weights.

The declared `encoder` is the variational posterior. [`AmortizedGuide`](../api/inference/guide.md#quivers.inference.guides.AmortizedGuide) takes the program, its observed sites, and a map from each latent site to the morphism that proposes it and the datum it reads, here `{"z": (encoder, "y")}`: it draws `z` from `encoder(y)` row by row and scores the draw under the same kernel, so one set of encoder weights serves every observation, $q(z_n \mid y_n) = \mathrm{encoder}(y_n)$. The guide's parameters are the encoder's own, and [`SVI`](../api/inference/svi.md#quivers.inference.svi.SVI) with the [`ELBO`](../api/inference/elbo.md) objective optimizes them together with the decoder's weights (the program's parameters), which is Kingma and Welling's training procedure. `guide.observe` points the same guide at new observations, which is how the held-out checks below get the posterior of data the networks never saw.

## Try it

### Generating synthetic data

The data lie near a two-dimensional *twisted strip* in five dimensions: the first code coordinate winds around a circle and the second sets the width of a twist about it. The generator is a member of the model's own family, $z \sim \mathcal{N}(0, I)$ and $y \sim \mathcal{N}(g(z), 0.1^2 I)$ for a fixed smooth map $g$, which the decoder network has to learn. A second sample of 512 rows is held out. The block also checks the program's log joint at the true codes against a hand-written prior plus decoder density.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/vae.qvr")
model = prog.morphism
encoder, decoder = prog.encoder, prog.decoder

N, latent_dim, obs_dim = 512, 2, 5
# One run of the program: its N rows come from the `Row` plate.
x_in = torch.zeros(1, 1)


def twisted_strip(z):
    """The first code coordinate winds around a circle; the second
    sets the radius of a twist around it."""
    angle = 1.5 * z[:, 0]
    return torch.stack(
        [
            torch.cos(angle),
            torch.sin(angle),
            z[:, 1],
            0.5 * z[:, 1] * torch.cos(angle),
            0.5 * z[:, 1] * torch.sin(angle),
        ],
        dim=-1,
    )


noise = 0.1
true_z = torch.randn(N, latent_dim)
y = twisted_strip(true_z) + noise * torch.randn(N, obs_dim)
z_test = torch.randn(N, latent_dim)
y_test = twisted_strip(z_test) + noise * torch.randn(N, obs_dim)
observations = {"y": y}

# Independent oracle: the program's log joint at the true codes is the
# standard-Normal prior plus the decoder's Normal density, computed
# here from the decoder's own MLP weights.
first, second = decoder.param_source.net[0], decoder.param_source.net[2]
raw = torch.tanh(true_z @ first.weight.T + first.bias) @ second.weight.T + second.bias
loc, scale = raw[:, :obs_dim], raw[:, obs_dim:].exp()
Normal = torch.distributions.Normal
hand = Normal(0.0, 1.0).log_prob(true_z).sum() + Normal(loc, scale).log_prob(y).sum()
log_joint_true = model.log_joint(x_in, {"y": y, "z": true_z})
assert torch.allclose(log_joint_true, hand.reshape(1), rtol=1e-5)
print(f"log joint at the true codes: {log_joint_true[0].item():.1f} (hand-written {float(hand):.1f})")
```

### Training the encoder and decoder by the ELBO

The fit conditions on `y` alone; the codes stay latent and are drawn from the encoder at every step. Before training, the block checks that the guide scores a draw exactly as the declared encoder does: its log density must equal `encoder.log_prob` summed over the rows, to a relative 1e-6.

```python
from quivers.inference import ELBO, SVI, AmortizedGuide

torch.manual_seed(1)
guide = AmortizedGuide(model, {"y"}, {"z": (encoder, "y")}, x=x_in, observations=observations)
draw = guide.rsample(x_in)
assert draw["z"].shape == (N, latent_dim)
assert torch.allclose(guide.log_prob(x_in, draw)[0], encoder.log_prob(y, draw["z"]).sum(), rtol=1e-6)
assert {id(p) for p in guide.parameters()} == {id(p) for p in encoder.parameters()}

optim = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=3e-3)
svi = SVI(model, guide, optim, ELBO())
losses = [svi.step(x_in, observations) for _ in range(3000)]
print(f"training ELBO per row: first step {-losses[0] / N:.2f}, last 100 steps {-sum(losses[-100:]) / 100 / N:.2f}")
```

### Held-out evidence and reconstruction

Two baselines, both fitted to the same training rows. [Probabilistic PCA](ppca.md) (PPCA) [@tipping1999ppca] is the linear-Gaussian latent model with the same two latent dimensions, and its maximum-likelihood fit has a closed form and an exact marginal likelihood. Since the ELBO lower-bounds the VAE's own log likelihood, a held-out ELBO above the exact PPCA log likelihood shows that the VAE models the curved data better than any linear latent model can. The reconstruction check runs held-out rows through `reconstruct`, averages 20 draws, and compares the error with predicting the training mean. Both assertions were fixed before the run.

```python
def heldout_elbo(y_eval, particles=20):
    """Monte Carlo ELBO per row on rows the networks never saw."""
    q = AmortizedGuide(model, {"y"}, {"z": (encoder, "y")}, x=x_in, observations={"y": y_eval})
    with torch.no_grad():
        terms = []
        for _ in range(particles):
            z = q.rsample(x_in)
            log_p = model.log_joint(x_in, {"y": y_eval, **z})[0]
            terms.append(log_p - q.log_prob(x_in, z)[0])
    return float(torch.stack(terms).mean()) / y_eval.shape[0]


def ppca_loglik(y_fit, y_eval, k):
    """Exact held-out log likelihood per row of maximum-likelihood PPCA."""
    mean = y_fit.mean(0)
    eigvals, eigvecs = torch.linalg.eigh(torch.cov((y_fit - mean).T, correction=0))
    eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
    noise_var = eigvals[k:].mean()
    loading = eigvecs[:, :k] * (eigvals[:k] - noise_var).sqrt()
    cov = loading @ loading.T + noise_var * torch.eye(y_fit.shape[1])
    return float(torch.distributions.MultivariateNormal(mean, cov).log_prob(y_eval).mean())


elbo_test = heldout_elbo(y_test)
ppca_test = ppca_loglik(y, y_test, latent_dim)

with torch.no_grad():
    reconstruction = torch.stack([prog.reconstruct.rsample(y_test) for _ in range(20)]).mean(0)
recon_mse = float(((reconstruction - y_test) ** 2).mean())
baseline_mse = float(((y.mean(0) - y_test) ** 2).mean())

print(f"held-out ELBO per row:          {elbo_test:.2f}")
print(f"held-out PPCA log lik per row:  {ppca_test:.2f}")
print(f"reconstruction MSE: {recon_mse:.3f} (training-mean baseline {baseline_mse:.3f})")

assert elbo_test > ppca_test
assert recon_mse < 0.5 * baseline_mse
```

### An oracle on linear data

On data from a linear-Gaussian generator the true marginal likelihood is available in closed form, and a VAE whose decoder can represent the linear map and whose encoder can represent the Gaussian posterior should reach it from below. The block refits a fresh copy of the model to such data and requires the held-out ELBO per row to come within 0.25 nats of the exact log likelihood, and to exceed it by no more than 0.05 nats of sampling noise.

```python
torch.manual_seed(2)
linear = load("docs/examples/source/vae.qvr")
loading = torch.randn(obs_dim, latent_dim)
offset = torch.randn(obs_dim)
linear_noise = 0.3


def linear_gaussian(n):
    return torch.randn(n, latent_dim) @ loading.T + offset + linear_noise * torch.randn(n, obs_dim)


y_lin, y_lin_test = linear_gaussian(N), linear_gaussian(N)
exact = torch.distributions.MultivariateNormal(
    offset, loading @ loading.T + linear_noise**2 * torch.eye(obs_dim)
)
exact_test = float(exact.log_prob(y_lin_test).mean())

model, encoder = linear.morphism, linear.encoder
guide = AmortizedGuide(model, {"y"}, {"z": (encoder, "y")}, x=x_in, observations={"y": y_lin})
optim = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=3e-3)
svi = SVI(model, guide, optim, ELBO())
for _ in range(3000):
    svi.step(x_in, {"y": y_lin})
elbo_lin = heldout_elbo(y_lin_test)

print(f"held-out ELBO per row:                {elbo_lin:.3f}")
print(f"held-out exact log lik per row:       {exact_test:.3f}")
gap = exact_test - elbo_lin
assert -0.05 <= gap <= 0.25
```

### NUTS over the codes

With the networks fixed at their trained weights, the program's only latent site is `z`, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples the 1,024 code coordinates directly. `init_strategy="guide"` starts the chain from an encoder draw. Comparing the chain's mean codes with the encoder's means gives a direct look at the amortization gap, the error the encoder makes by predicting each posterior instead of computing it. Thirty warmup and thirty retained draws keep the block short; a longer run is needed before reading the difference as a precise measurement.

```python
from quivers.inference import MCMC, NUTSKernel

model, encoder = prog.morphism, prog.encoder
guide = AmortizedGuide(model, {"y"}, {"z": (encoder, "y")}, x=x_in, observations=observations)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=30, num_samples=30, num_chains=1, init_strategy="guide", seed=3)
result = mc.run(model, x_in, observations, guide=guide)

with torch.no_grad():
    encoder_means = encoder.param_source(y)[:, :latent_dim]
nuts_means = result.samples["z"][0].mean(0)
print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
print(f"mean |NUTS mean - encoder mean| per code coordinate: {float((nuts_means - encoder_means).abs().mean()):.3f}")
```

## Categorical perspective

`encoder : ObsSpace -> Latent` and `decoder : Latent -> ObsSpace` are [Kleisli morphisms](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical]: each sends a point to a Gaussian measure on the other space. The generative program composes the prior $1 \to \mathcal{G}(\mathsf{Latent})$ with the decoder, giving the model's marginal over observations, and `reconstruct` composes the encoder with the decoder, giving an endomorphism of $\mathsf{ObsSpace}$. Training by the ELBO asks the encoder to approximate the Bayesian inverse of the decoder with respect to the prior, the kernel $\mathsf{ObsSpace} \to \mathcal{G}(\mathsf{Latent})$ that disintegrates the joint distribution of $(z, y)$ along $y$. Under that reading, `reconstruct` is the composite of a kernel with its approximate inverse, and its distance from the identity is what the reconstruction check measures.

## See also

- [Probabilistic PCA](ppca.md) for the linear-Gaussian latent model that serves as the baseline and the oracle here.
- [Bayesian Neural Network](bnn.md) for a network whose weights, rather than its inputs, are the latent variables.

## Sources and further reading

- *Auto-Encoding Variational Bayes* [@kingma2014autoencoding].
- *Probabilistic Principal Component Analysis* [@tipping1999ppca].
