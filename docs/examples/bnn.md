# Bayesian Neural Network

## Overview

A [Bayesian neural network](https://en.wikipedia.org/wiki/Bayesian_neural_network) [@mackay1992bayesian] places priors over network weights. The QVR source below declares a conditional MLP kernel without those priors; the later `lift_from_log_prob` block adds them. The response is [Normal](https://en.wikipedia.org/wiki/Normal_distribution) about a learned function of the input,

$$
\mu(x),\, \sigma(x) \;=\; \mathrm{MLP}(x), \qquad y_n \;\sim\; \mathcal{N}\!\bigl(\mu(x_n),\, \sigma(x_n)\bigr).
$$

The network emits both the mean and the log-scale, so the model is [heteroscedastic](https://en.wikipedia.org/wiki/Heteroscedasticity): the predictive spread varies with the input rather than being pinned to a single global noise level.

The nonlinearity lives in the morphism's parameter network. A morphism declared `~ Normal` over continuous spaces is a [Kleisli arrow](https://en.wikipedia.org/wiki/Kleisli_category) whose distribution parameters are produced from its input by a [`ParamSource`](../api/continuous/param_source.md#quivers.continuous.param_source.ParamSource), and `[param_source=mlp, hidden_dim=64]` selects an [`MLPSource`](../api/continuous/param_source.md#quivers.continuous.param_source.MLPSource): two hidden layers of width 64 with tanh activations between them. That is where the model departs from a linear map, and it is the reason this example can fit a curve that no linear model can.

## QVR source

```qvr
# Conditional MLP Kernel for Bayesian Nonlinear Regression
#
# A multi-layer perceptron kernel. The response is Normal about a
# nonlinear function of the input, and that function is a
# two-hidden-layer tanh MLP carried inside the morphism's parameter
# network. The network emits both the mean and the log-scale, so the
# model is heteroscedastic: the predictive spread varies with the
# input.
#
# Generative structure:
#
#   mu(x), sigma(x) = MLP(x)                   two hidden layers, tanh
#   y_n             ~ Normal(mu(x_n), sigma(x_n))
#
# The per-row input x is a free variable, supplied as host data
# through the observations dict; the Resp plate indexes the rows.
#
# What makes the fit Bayesian is a prior over the network's weights.
# The weights are ordinary learnable parameters here, so the prior is
# supplied at the inference layer by lift_from_log_prob rather than
# declared in the source: a Normal prior on every weight turns a
# maximum-likelihood fit into a posterior over networks.
#
# Reference: [MacKay (1992)](https://doi.org/10.1162/neco.1992.4.3.448).

object Feature : Real 1
object Target : Real 1
object Resp : FinSet 200

morphism net : Feature -> Target [param_source=mlp, hidden_dim=64] ~ Normal

program bnn : Resp -> Target
    observe y : Resp <- net(x)
    return y

export bnn
```

## Understanding the model

There are two sources of uncertainty in a Bayesian neural network. Even if
the weights were known, observations would vary around the network's mean;
that is the Normal kernel's noise. With finite data, the weights themselves
are also uncertain. A [prior and posterior over weights](https://pyro.ai/examples/bayesian_regression.html#Model)
represent this second source [@pyroBayesianRegression]. The code below first
fits one weight vector, then adds priors so inference can estimate uncertainty
in those weights.

A [posterior prediction](../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks) draws weights, computes a mean and scale for the new
input, then draws an outcome. The initial maximum-likelihood block instead
holds one fitted weight vector. Compare their predictions on inputs outside
the training region, where uncertainty about the function can matter more
than the fitted residual noise.

## Walkthrough

`object Feature : Real 1` and `object Target : Real 1` declare the input and response as one-dimensional continuous spaces; `object Resp : FinSet 200` is the discrete plate indexing the rows. The plate is what the `observe` step reduces over, so it is the program's domain, and `Target` is the value space of the returned response, so it is the codomain.

`morphism net : Feature -> Target [param_source=mlp, hidden_dim=64] ~ Normal` declares the network. The `~ Normal` clause makes `net` a conditional-Normal kernel rather than a tensor: applied to an input it returns a distribution over `Target`, parameterized by a mean and a log-scale. Those two numbers are what the parameter network emits, so the concrete module behind `net` is

```
Linear(1, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 2)
```

Its 4418 weights are the network's parameters. `hidden_dim` sets the width and `param_source` the architecture; `linear` is the default, and `identity` and `attention` are the other choices. Without `param_source=mlp` this kernel would map its input to the Normal's parameters through a single matrix, and the model would be a linear regression with a learned noise scale.

`program bnn : Resp -> Target` then does the only thing left. `observe y : Resp <- net(x)` applies the kernel to the per-row input and scores the observed response under the resulting Normal, accumulating over the `Resp` plate. `x` is a free variable: it never appears in a `sample` or `let`, so it is supplied as host data through the observations dict alongside `y`, exactly as a covariate would be in a regression.

The checked elaboration makes the network computation explicit: each affine
weight and bias is a typed program input, and the QIEC body applies the layers
under `tanh` before constructing the Normal family. The reference machine runs
that graph with the compiled PyTorch modules. Transpile targets refuse
`param-source:mlp`, however, because these model-internal weights are neither
declared sample sites nor portable data from which a host program could
reconstruct the network. The [measured support matrix](../transpile-support.md)
records the resulting capability diagnostic for each target.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/bnn.qvr")
model = prog.morphism

N = 200
x = torch.linspace(-3.0, 3.0, N).unsqueeze(-1)
y = torch.sin(3.0 * x) + 0.1 * torch.randn(N, 1)

x_in = torch.zeros(N, 1)
observations = {"y": y, "x": x}
```

The target is a sine wave in noise, so its mean bends across the input range. `y` carries the `Target` event axis, so it is shaped `(N, 1)` rather than `(N,)`. The dummy `x_in` supplies the program's batch axis; the real covariate enters through `observations["x"]`.

### Fitting the network

```python
optim = torch.optim.Adam(model.parameters(), lr=5e-3)

for step in range(2000):
    optim.zero_grad()
    loss = -model.log_joint(x_in, observations)[0]
    loss.backward()
    optim.step()

print(f"final negative log-likelihood: {float(loss.detach()):.1f}")
```

[`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) returns the total log-joint broadcast across the batch axis, so every entry is the same number and `[0]` reads it. Summing instead would count the same scalar once per row.

### Does the nonlinearity earn its keep?

The honest comparison is against the best a linear model can do on the same data, scored the same way. Ordinary least squares gives the optimal line, and its residual spread gives the matching Gaussian noise level.

```python
design = torch.cat([torch.ones(N, 1), x], dim=-1)
beta   = torch.linalg.lstsq(design, y).solution
resid  = y - design @ beta
linear_nll = -torch.distributions.Normal(design @ beta, resid.std()).log_prob(y).sum()

print(f"MLP    negative log-likelihood: {float(loss.detach()):.1f}")
print(f"linear negative log-likelihood: {float(linear_nll):.1f}")
print(f"residual sd of the best line:   {float(resid.std()):.3f}")
```

The line's residual spread is larger than the data-generating noise because a straight line cannot represent the sine target. Compare the printed likelihoods from the current run; their difference reflects both the nonlinear mean and the input-dependent scale available to the MLP.

### NUTS posterior over the weights

Nothing so far is Bayesian: the fit above is maximum likelihood over the weights. `bnn` declares no `sample` sites, so there is no prior in the source to sample. [`lift_from_log_prob`](../api/inference/lifts.md#quivers.inference.lifts.lift_from_log_prob) supplies one, putting a Normal prior on every weight and scoring the data through a log-density function of your choosing. The result is a [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) whose sites are the weights themselves, which is exactly the Bayesian neural network the overview describes.

```python
from quivers.inference import MCMC, NUTSKernel, lift_from_log_prob

def log_prob_fn(x_unused, y_obs):
    return model.log_joint(x_in, {"y": y_obs, "x": x})[0].reshape(1)

lifted, lift_x, lift_obs = lift_from_log_prob(
    model,
    log_prob_fn=log_prob_fn,
    parameter_prior_scale=1.0,
    target_key="y",
    observations={"y": y},
)

kernel = NUTSKernel(step_size=0.01, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(lifted, lift_x, lift_obs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

`log_prob_fn` evaluates the likelihood with the sampled weights installed in
the network. The resulting posterior has 4,418 weight coordinates. Ten warmup
steps and ten draws from one chain exercise the call, but provide little
information about exploration of that space. For a posterior analysis, increase
warmup and draws, run multiple chains, and compare predictive functions across
them alongside divergence and effective sample size diagnostics.

## Categorical perspective

`net : Feature -> Target` is a Kleisli arrow $\mathbb{R} \to \mathcal{G}(\mathbb{R})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872), sending an input to a Gaussian measure over the response. The MLP is not part of that arrow's type: it is the map from the domain into the family's parameter space, and composing it with the Gaussian's parameterization is what makes the kernel's mean and scale depend nonlinearly on the input. [@giry1982categorical]

Under a strictly linear algebra, a chain
`W_1 >> W_2 >> W_3` of tensor morphisms collapses to a single linear map; the
intermediate objects impose only a rank bound. Placing the nonlinearity inside
the kernel's parameter source avoids that collapse while retaining the V-Cat
representation, because the kernel need not be linear in its input. The
[`ParamSource`](../api/continuous/param_source.md#quivers.continuous.param_source.ParamSource)
interface supplies the common boundary: `mlp`, `linear`, `attention`, and
`identity` differ only in what they compute.

Putting a prior on the weights is then a second, independent move. It lifts the deterministic parameter $\theta$ into a sample site, so the model becomes a mixture of networks $\int p(y \mid x, \theta)\, p(\theta)\, d\theta$ rather than a single one.

## See also

- [Deep Markov Model](deep-markov.md) for the same conditional-Normal kernels threaded through `scan` as a nonlinear state-space model.
- [Bayesian Regression](bayesian-regression.md) for the linear counterpart, where the coefficients carry declared priors in the source.
