# Bayesian Neural Network

## Overview

A [Bayesian neural network](https://en.wikipedia.org/wiki/Bayesian_neural_network) [@mackay1992bayesian] treats the weights of a network as random variables. A prior over the weights and a likelihood for the data give a posterior over networks, and predictions average the network's output over that posterior. This example is a one-hidden-layer tanh regression network with four hidden units, with every weight declared as a sample site in the source:

$$
\mu(x) = b_{\text{out}} + \sum_{h=1}^{4} w^{\text{out}}_h \tanh\bigl(w^{\text{in}}_h x + b^{\text{in}}_h\bigr),
\qquad
y_n \sim \mathcal{N}\bigl(\mu(x_n), \sigma\bigr),
$$

with $w^{\text{in}}_h, b^{\text{in}}_h \sim \mathcal{N}(0, 2)$, $w^{\text{out}}_h \sim \mathcal{N}(0, 1)$, $b_{\text{out}} \sim \mathcal{N}(0, 1)$, and $\sigma \sim \text{HalfNormal}(0.5)$.

What the posterior buys over a single fitted network is *epistemic* uncertainty, uncertainty about the function itself. Inside the range of the training inputs the data pin the function down and the predictive spread is close to the noise level; outside it many networks fit the data equally well and disagree, so the predictive spread should grow. The runnable blocks fit the posterior with MacKay's [Laplace approximation](https://en.wikipedia.org/wiki/Laplace%27s_approximation) and check both halves of that claim against held-out data from the true network.

## QVR source

```qvr
# Bayesian Neural Network Regression
#
# A one-hidden-layer tanh network with four hidden units whose
# weights are random variables. Every weight and bias carries a
# Normal prior and the observation noise a half-Normal prior, so
# conditioning on data gives a posterior over networks rather than a
# single fitted one.
#
# Generative structure:
#
#   w_in_h, b_in_h  ~ Normal(0, 2)        input-to-hidden weights
#   w_out_h         ~ Normal(0, 1)        hidden-to-output weights
#   b_out           ~ Normal(0, 1)        output bias
#   sigma           ~ HalfNormal(0.5)     observation noise
#   h_h(x)          = tanh(w_in_h x + b_in_h)
#   mu(x)           = b_out + sum_h w_out_h h_h(x)
#   y_n             ~ Normal(mu(x_n), sigma)
#
# Resp indexes the observed rows and Hidden the hidden units; the
# per-row input x is host data. Each hidden unit's activation is one
# let over the Resp plate, and the output layer sums them.
#
# Reference: [MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448).

object Resp : FinSet 128
object Hidden : FinSet 4
object Target : Real 1

program bnn : Resp -> Target
    sample w_in : Hidden <- Normal(loc=0.0, scale=2.0)
    sample b_in : Hidden <- Normal(loc=0.0, scale=2.0)
    sample w_out : Hidden <- Normal(loc=0.0, scale=1.0)
    sample b_out <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=0.5)

    let h0 = tanh(w_in[0] * x + b_in[0])
    let h1 = tanh(w_in[1] * x + b_in[1])
    let h2 = tanh(w_in[2] * x + b_in[2])
    let h3 = tanh(w_in[3] * x + b_in[3])
    let mu = b_out + w_out[0] * h0 + w_out[1] * h1 + w_out[2] * h2 + w_out[3] * h3

    observe y : Resp <- Normal(loc=mu, scale=sigma)
    return y

export bnn
```

## Understanding the model

There are two sources of uncertainty in a prediction. Even with the weights known, a new response varies around the network's mean by the observation noise $\sigma$. With finite data the weights are also uncertain, and that second source is what the weight prior and posterior represent [@pyroBayesianRegression]. A single fitted network, whether by maximum likelihood or at the posterior mode (MAP), keeps only the first source: its predictive interval has the same width at every input.

A [posterior prediction](../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks) draws a weight vector from the posterior, computes the network's mean at the new input, and draws a response around it. Averaging over weight draws widens the interval exactly where the posterior networks disagree, which tends to be away from the training inputs.

## Walkthrough

`object Resp : FinSet 128` is the plate of observed rows and `object Hidden : FinSet 4` indexes the hidden units; `object Target : Real 1` is the value space of the response, so `program bnn : Resp -> Target` scores a plate of real responses.

The five `sample` lines are the priors. `sample w_in : Hidden <- Normal(loc=0.0, scale=2.0)` draws one input weight per hidden unit, and likewise for the hidden biases `b_in` and the output weights `w_out`; `b_out` and `sigma` are scalars. Because every weight is a sample site, the [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) this source compiles to has 14 latent coordinates, and every inference driver sees them directly. No Python-side lift is needed to make the network Bayesian.

The `let` lines are the network. `x` is the per-row input supplied as host data. `let h0 = tanh(w_in[0] * x + b_in[0])` is the first hidden unit's activation at every row: `w_in[0]` and `b_in[0]` are scalars, so the expression stays on the `Resp` plate. The output layer `mu` weights the four activations by `w_out` and adds `b_out`. One line per unit keeps every product between a scalar and a plated vector; the checked program language has no outer product of the `Resp` and `Hidden` axes, so the hidden layer cannot be written as a single `(row, unit)` matrix.

`observe y : Resp <- Normal(loc=mu, scale=sigma)` scores the observed responses. A network whose weights are declared sites is also an ordinary program for the transpilers: there is no model-internal parameter network, so the [measured support matrix](../transpile-support.md) reports this source like any other plated regression.

## Try it

### Generating synthetic data

The true network is a draw from the source's own priors. Tracing the program with only `sigma` clamped samples the weights from their priors and the responses from the likelihood, so the data come from exactly the model being fit. The block then checks the program's log joint at the true weights against a hand-written density.

```python
import torch
from quivers.dsl import load
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/bnn.qvr")
model = prog.morphism

N = 128
x_in = torch.zeros(1, 1)


# Draw the true network from the source's own priors, with the noise
# scale held at 0.1, and simulate the training responses from it.
x = torch.linspace(-2.0, 2.0, N)
true_sigma = torch.tensor(0.1)
with torch.no_grad():
    forward = trace(model, x_in, {"x": x, "sigma": true_sigma})
true_w_in = forward.sites["w_in"].value
true_b_in = forward.sites["b_in"].value
true_w_out = forward.sites["w_out"].value
true_b_out = forward.sites["b_out"].value.reshape(())
y = forward.sites["y"].value.reshape(N)
truth = {"w_in": true_w_in, "b_in": true_b_in, "w_out": true_w_out, "b_out": true_b_out}

observations = {"x": x, "y": y}

# Independent oracle: the program's log joint at the true weights is
# the hand-written prior plus likelihood.
Normal, HalfNormal = torch.distributions.Normal, torch.distributions.HalfNormal
mu_true = (torch.tanh(x[:, None] * true_w_in + true_b_in) * true_w_out).sum(-1) + true_b_out.reshape(())
hand = (
    Normal(0.0, 2.0).log_prob(true_w_in).sum()
    + Normal(0.0, 2.0).log_prob(true_b_in).sum()
    + Normal(0.0, 1.0).log_prob(true_w_out).sum()
    + Normal(0.0, 1.0).log_prob(true_b_out).sum()
    + HalfNormal(0.5).log_prob(true_sigma)
    + Normal(mu_true, true_sigma).log_prob(y.reshape(-1)).sum()
)
log_joint_true = model.log_joint(x_in, {**observations, **truth, "sigma": true_sigma})
assert torch.allclose(log_joint_true, hand.reshape(1), rtol=1e-5)
print(f"log joint at the truth: {float(log_joint_true):.2f} (hand-written {float(hand):.2f})")
```

### Fitting the posterior mode

[`AutoLaplaceApproximation`](../api/inference/guide.md#quivers.inference.guides.AutoLaplaceApproximation) is a two-phase guide. Before its Hessian is fitted it is a point mass, and maximizing the ELBO under it maximizes the log joint, so the [`SVI`](../api/inference/svi.md#quivers.inference.svi.SVI) loop below finds the posterior mode. An L-BFGS polish then takes the mode to a stationary point, which the curvature step needs.

```python
from quivers.inference import AutoLaplaceApproximation, ELBO, SVI

torch.manual_seed(1)
guide = AutoLaplaceApproximation(model, observed_names=set(observations))
with torch.no_grad():
    guide.map_z.normal_()
svi = SVI(model, guide, torch.optim.Adam(guide.parameters(), lr=2e-2), ELBO())
for _ in range(1000):
    svi.step(x_in, observations)

# Polish the mode with L-BFGS so the Hessian below is taken at a
# stationary point rather than wherever Adam stopped.
objective = ELBO()
lbfgs = torch.optim.LBFGS(
    guide.parameters(), max_iter=500, tolerance_grad=1e-7,
    tolerance_change=1e-12, line_search_fn="strong_wolfe",
)


def closure():
    lbfgs.zero_grad()
    loss = objective(model, guide, x_in, observations)
    loss.backward()
    return loss


lbfgs.step(closure)
with torch.no_grad():
    map_point = {k: v.detach() for k, v in guide.rsample(x_in).items()}
log_joint_map = model.log_joint(x_in, {**observations, **map_point})
print(f"log joint at the MAP:   {float(log_joint_map):.2f}")
print(f"MAP noise scale: {float(map_point['sigma']):.3f} (true {float(true_sigma):.3f})")
```

The log joint at the mode exceeds the log joint at the true weights, as it should: the mode maximizes this quantity on the training data.

### Laplace posterior and the predictive check

`fit_hessian` replaces the point mass by a Gaussian whose covariance is the inverse Hessian of the negative log joint at the mode, the approximation MacKay used [@mackay1992bayesian; @tierney1986laplace]. Following MacKay, the predictive pushes Gaussian weight draws through the network *linearized* at the mode, computed here with a [Jacobian-vector product](https://pytorch.org/docs/stable/generated/torch.func.jvp.html) of the program's own `mu`. Linearization keeps the predictive Gaussian in the function and avoids the large excursions that raw Gaussian draws produce along weakly determined directions of a saturating network.

The check uses 128 fresh responses from the true network inside the training range $[-2, 2]$ and 128 outside it, in $[-4, -3] \cup [3, 4]$. Three assertions, fixed before the run: the 90% predictive interval covers between 80% and 97% of the inside responses; the posterior standard deviation of the mean function is at least three times larger outside than inside; and the posterior intervals cover more of the outside responses than the MAP plug-in intervals do.

```python
from torch.func import jvp

WEIGHTS = ("w_in", "b_in", "w_out", "b_out")


def network_mean(x_new, weights):
    """The program's own `mu` let at inputs `x_new` under `weights`."""
    clamp = {"x": x_new, **weights, "sigma": map_point["sigma"], "y": torch.zeros(1, N)}
    return trace(model, x_in, clamp).sites["mu"].value.reshape(-1)


guide.fit_hessian(model, x_in, observations)

# MacKay's predictive: weights from the Gaussian posterior at the mode,
# pushed through the network linearized at the mode.
base = {k: map_point[k] for k in WEIGHTS}


def linearized_means(x_new, draws):
    rows = []
    for draw in draws:
        tangent = {k: draw[k] - base[k] for k in WEIGHTS}
        mean, delta = jvp(lambda w: network_mean(x_new, w), (base,), (tangent,))
        rows.append(mean + delta)
    return torch.stack(rows)


torch.manual_seed(3)
with torch.no_grad():
    draws = [{k: v.detach() for k, v in guide.rsample(x_in).items()} for _ in range(300)]
sigma_draws = torch.stack([d["sigma"].reshape(()) for d in draws])

# Fresh responses from the true network, inside and outside the
# training range [-2, 2].
x_inside = 4.0 * torch.rand(N) - 2.0
x_outside = torch.cat([torch.rand(N // 2) - 4.0, torch.rand(N // 2) + 3.0])
with torch.no_grad():
    y_inside = network_mean(x_inside, truth) + true_sigma * torch.randn(N)
    y_outside = network_mean(x_outside, truth) + true_sigma * torch.randn(N)


def summarize(x_new, y_new):
    means = linearized_means(x_new, draws)
    predictive = means + sigma_draws[:, None] * torch.randn_like(means)
    low, high = predictive.quantile(0.05, dim=0), predictive.quantile(0.95, dim=0)
    coverage = ((y_new >= low) & (y_new <= high)).float().mean()
    return coverage, means.std(dim=0).mean()


coverage_in, spread_in = summarize(x_inside, y_inside)
coverage_out, spread_out = summarize(x_outside, y_outside)

# The plug-in predictive of the MAP network: one function, noise only.
half_width = 1.645 * map_point["sigma"].reshape(())
map_coverage_in = ((y_inside - network_mean(x_inside, base)).abs() <= half_width).float().mean()
map_coverage_out = ((y_outside - network_mean(x_outside, base)).abs() <= half_width).float().mean()

print(f"90% interval coverage inside:  BNN {float(coverage_in):.2f}  MAP {float(map_coverage_in):.2f}")
print(f"90% interval coverage outside: BNN {float(coverage_out):.2f}  MAP {float(map_coverage_out):.2f}")
print(f"posterior sd of the mean function: inside {float(spread_in):.3f}  outside {float(spread_out):.3f}")

assert 0.80 <= float(coverage_in) <= 0.97
assert float(spread_out) >= 3.0 * float(spread_in)
assert float(coverage_out) > float(map_coverage_out)
```

Inside the range both predictives are close to nominal, since the noise dominates there. Outside it the MAP network extrapolates one function with noise-width intervals and misses most responses, while the posterior networks disagree and the interval widens to cover them.

### NUTS from the mode

The same program runs under [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) with no change: the sampler moves over the 14 declared coordinates. `init_strategy="guide"` starts the chain at the mode by handing [`MCMC`](../api/inference/mcmc.md#quivers.inference.mcmc.MCMC) a point-mass guide placed there. Thirty warmup and thirty retained draws keep the block short; they explore the neighborhood of the mode and do not replace a multi-chain run with convergence diagnostics.

```python
from quivers.inference import MCMC, NUTSKernel

mode = AutoLaplaceApproximation(model, observed_names=set(observations))
with torch.no_grad():
    mode.map_z.copy_(guide.map_z)
kernel = NUTSKernel(step_size=0.01, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=30, num_samples=30, num_chains=1, init_strategy="guide", seed=4)
result = mc.run(model, x_in, observations, guide=mode)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
print(f"posterior mean noise scale: {float(result.samples['sigma'].mean()):.3f}")
```

## Categorical perspective

`bnn : Resp -> Target` is a [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical]. Its sample steps compose a prior over the parameter space $\mathbb{R}^{13} \times \mathbb{R}_{>0}$ with a likelihood kernel from weights to responses, and the network is the deterministic map inside that kernel from a weight vector and an input to a Normal mean. The posterior predictive at a new input is then the Kleisli composite of the posterior over weights with the same kernel, which is why its spread inherits the posterior's disagreement about the function.

Declaring the weights as sites is what separates this source from a neural kernel such as `morphism net : A -> B [param_source=mlp] ~ Normal`. There the network's weights belong to the morphism's [`ParamSource`](../api/continuous/param_source.md#quivers.continuous.param_source.ParamSource) and are learned as point estimates; no source-level syntax places a prior on them. Here they are random variables of the program, so a single source denotes the whole mixture $\int p(y \mid x, \theta)\, p(\theta \mid \mathcal{D})\, d\theta$.

## See also

- [Bayesian Regression](bayesian-regression.md) for the linear special case, a network with no hidden layer.
- [Variational Autoencoder](vae.md) for neural kernels whose weights are point estimates, with an amortized posterior over latent codes instead of over weights.

## Sources and further reading

- *A Practical Bayesian Framework for Backpropagation Networks* [@mackay1992bayesian].
- *Accurate Approximations for Posterior Moments and Marginal Densities* [@tierney1986laplace].
