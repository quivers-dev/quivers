# Bayesian Neural Network

## Overview

A [Bayesian neural network](https://en.wikipedia.org/wiki/Bayesian_neural_network) [@mackay1992bayesian] treats the weights of a network as random variables. A prior over the weights and a likelihood for the data give a posterior over networks, and posterior prediction averages their response distributions. This example is a one-hidden-layer tanh regression network with four hidden units, with every weight declared as a [sample site](../guides/inference-foundations.md#trace-and-sample-sites) in the source:

$$
\mu(x) = b_{\text{out}} + \sum_{h=1}^{4} w^{\text{out}}_h \tanh\bigl(w^{\text{in}}_h x + b^{\text{in}}_h\bigr),
\qquad
y_n \sim \mathcal{N}\bigl(\mu(x_n), \sigma^2\bigr),
$$

with $w^{\text{in}}_h, b^{\text{in}}_h \sim \mathcal{N}(0, 2^2)$, $w^{\text{out}}_h \sim \mathcal{N}(0, 1)$, $b_{\text{out}} \sim \mathcal{N}(0, 1)$, and $\sigma \sim \text{HalfNormal}(0.5)$.

What the posterior buys over a single fitted network is *epistemic* uncertainty, uncertainty about the function itself. Inside the range of the training inputs the data pin the function down and the predictive spread is close to the noise level. Outside it many networks fit the data equally well and disagree, so the predictive spread should grow. The runnable blocks fit the posterior with MacKay's [Laplace approximation](https://en.wikipedia.org/wiki/Laplace%27s_approximation) and check both halves of that claim against held-out data from the true network.

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

There are two sources of uncertainty in a prediction. Even with the weights known, a new response varies around the network's mean by the observation noise $\sigma$. With finite data the weights are also uncertain, and that second source is what the weight prior and posterior represent [@pyroBayesianRegression]. A single fitted network, whether by maximum likelihood or at the posterior mode ([MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (maximum a posteriori)), keeps only the first source: its predictive interval has the same width at every input.

A [posterior prediction](../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks) draws a weight vector from the posterior, computes the network's mean at the new input, and draws a response around it. Averaging over weight draws widens the interval exactly where the posterior networks disagree, which tends to be away from the training inputs.

## Walkthrough

`object Resp : FinSet 128` is the [plate](https://en.wikipedia.org/wiki/Plate_notation) of observed rows and `object Hidden : FinSet 4` indexes the hidden units. `object Target : Real 1` is the value space of the response, so `program bnn : Resp -> Target` scores a plate of real responses.

The five `sample` lines are the priors. `sample w_in : Hidden <- Normal(loc=0.0, scale=2.0)` draws one input weight per hidden unit, and likewise for the hidden biases `b_in` and the output weights `w_out`. `b_out` and `sigma` are scalars. Because every weight is a sample site, the [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) this source compiles to has 14 latent coordinates, and every inference driver sees them directly. No Python-side lift is needed to make the network Bayesian.

The `let` lines are the network. `x` is the per-row input provided as input data. `let h0 = tanh(w_in[0] * x + b_in[0])` is the first hidden unit's activation at every row: `w_in[0]` and `b_in[0]` are scalars, so the expression stays on the `Resp` plate. The output layer `mu` weights the four activations by `w_out` and adds `b_out`. One line per unit keeps every product between a scalar and a plated vector. The checked program language has no outer product of the `Resp` and `Hidden` axes, so the hidden layer cannot be written as a single `(row, unit)` matrix.

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

[`AutoLaplaceApproximation`](../api/inference/guide.md#quivers.inference.guides.AutoLaplaceApproximation) is a two-phase guide. Before its Hessian is fitted it is a point mass, and maximizing the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) (evidence lower bound) under it maximizes the log joint, so the [`SVI`](../api/inference/svi.md#quivers.inference.svi.SVI) (stochastic variational inference) loop below finds the posterior mode. An L-BFGS polish then takes the mode to a stationary point, which the curvature step needs.

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

The check uses 128 fresh responses from the true network inside the training range $[-2, 2]$ and 128 outside it, in $[-4, -3] \cup [3, 4]$. The criteria require (i) coverage of 80% to 97% of the inside responses by the 90% predictive interval, (ii) posterior standard deviation of the mean function at least three times larger outside than inside, and (iii) greater coverage of outside responses by the posterior intervals than by the MAP plug-in intervals.

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

The same program runs under [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) with no change: the sampler moves over the 14 declared coordinates. `init_strategy="guide"` starts the chain at the mode by handing [`MCMC`](../api/inference/mcmc.md#quivers.inference.mcmc.MCMC) a point-mass guide placed there. Thirty warmup and thirty retained draws keep the block short. They explore the neighborhood of the mode and do not replace a multi-chain run with convergence diagnostics.

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

### A network with latent weights {#the-same-network-declared-with-latent-network-weights}

A neural [kernel](https://en.wikipedia.org/wiki/Markov_kernel) can carry weight priors directly. This variant has eight hidden units and predicts both mean and scale, so its architecture and likelihood differ from the four-unit model above. Declaring `net : Feature -> Target [param_source=mlp(8), weights=latent, prior_scale=1.0] ~ Normal` makes the weights and biases of its one-hidden-layer `tanh` network [sample sites](../reference/qvr/probabilistic-programs.md#latent-network-weights) of every program that reaches it, each with a `Normal(0, prior_scale)` prior, at the names `net_param_layer0_weight`, `net_param_layer0_bias`, `net_param_weight`, and `net_param_bias`. The network has two output heads: the first is the response mean and the exponential of the second the response scale, so the noise level is a function of the weights rather than a separate site. The [support matrix](../transpile-support.md) records backend coverage for latent network weights.

At a traced draw of the weights, the compiled runtime must match a hand-written float64 density to a relative 1e-5 and the [reference machine](../developer/qiec.md#reference-evaluator) to 1e-9, and a reference calculation with prior scale 2 must disagree by more than 1e-3. Then [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler) (one chain, 300 warmup and 300 kept draws, tree depth 7) is run on 20 points of $y = \sin(2x) + \mathcal{N}(0, 0.1^2)$, $x$ uniform on $[-1.5, 1.5]$. The posterior-mean network output at 50 held-out inputs must lie within an RMSE of 0.25 of $\sin(2x)$, while the prior-mean output, the same network averaged over 300 prior draws, must be further than 0.5 from it. The noise is 0.1, and the best straight line has an RMSE of about 0.45 on this range.

```python
import math

import torch.distributions as D

from quivers.dsl import loads
from quivers.inference import trace

LATENT_SOURCE = """
# A Bayesian neural network regression: a one-hidden-layer network kernel
# whose weights and biases are latent sites with Normal(0, 1) priors. The
# network's first output head is the response mean and the exponential of
# its second is the response scale, so the noise level is learned with the
# weights.
object Feature : Real 1
object Target : Real 1
object Row : FinSet 20

morphism net : Feature -> Target [param_source=mlp(8), weights=latent, prior_scale=1.0] ~ Normal

program bnn : Row -> Target
    observe y : Row <- net(x)
    return y

export bnn
"""
HIDDEN = 8
SHAPES = {
    "net_param_layer0_weight": (HIDDEN, 1),
    "net_param_layer0_bias": (HIDDEN,),
    "net_param_weight": (2, HIDDEN),
    "net_param_bias": (2,),
}


def heads(w, inputs):
    """The network's raw output heads at inputs (n, 1), float64."""
    hidden = torch.tanh(inputs @ w["net_param_layer0_weight"].T + w["net_param_layer0_bias"])
    return hidden @ w["net_param_weight"].T + w["net_param_bias"]


def latent_oracle(w, inputs, responses, scale=1.0):
    prior = D.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64))
    total = sum(float(prior.log_prob(v).sum()) for v in w.values())
    raw = heads(w, inputs)
    return total + float(D.Normal(raw[:, 0], raw[:, 1].exp()).log_prob(responses).sum())


def host(value):
    return float(value) if value.dim() == 0 else tuple(host(row) for row in value)


def sine_data(seed, n):
    generator = torch.Generator().manual_seed(seed)
    inputs = (torch.rand(n, 1, generator=generator) * 3.0 - 1.5).double()
    return inputs, torch.sin(2.0 * inputs[:, 0]) + 0.1 * torch.randn(n, generator=generator, dtype=torch.float64)


torch.manual_seed(0)
latent_program = loads(LATENT_SOURCE)
inputs, responses = sine_data(seed=1, n=20)
recorded = trace(latent_program.morphism, torch.zeros(1, 1), observations={"x": inputs.float(), "y": responses.float()})
w = {name: recorded.sites[name].value.reshape(shape).double() for name, shape in SHAPES.items()}
expected = latent_oracle(w, inputs.float().double(), responses.float().double())
wrong = latent_oracle(w, inputs.float().double(), responses.float().double(), scale=2.0)
got = float(recorded.log_joint[0])
run = latent_program.run(
    "bnn",
    data={"x": host(inputs.float().double()), "y": host(responses.float().double()[:, None])},
    sites={name: host(value) for name, value in w.items()},
    seed=0,
)
print(f"compiled {got:.6f}, reference {run.log_joint:.9f}, oracle {expected:.9f}; prior scale 2 {wrong:.6f}")
assert math.isclose(got, expected, rel_tol=1e-5)
assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
assert not math.isclose(got, wrong, rel_tol=1e-3)

inputs, responses = sine_data(seed=2, n=20)
result = MCMC(NUTSKernel(max_tree_depth=7), num_warmup=300, num_samples=300, num_chains=1, seed=0).run(
    loads(LATENT_SOURCE).morphism, torch.zeros(1, 1), {"x": inputs.float(), "y": responses.float()}
)
grid = torch.linspace(-1.5, 1.5, 50, dtype=torch.float64)[:, None]
target = torch.sin(2.0 * grid[:, 0])
draws = {name: result.samples[name].reshape(-1, *shape).double() for name, shape in SHAPES.items()}
count = next(iter(draws.values())).shape[0]
posterior_mean = torch.stack([heads({k: v[i] for k, v in draws.items()}, grid)[:, 0] for i in range(count)]).mean(0)
generator = torch.Generator().manual_seed(3)
prior_mean = torch.stack(
    [heads({k: torch.randn(*s, generator=generator, dtype=torch.float64) for k, s in SHAPES.items()}, grid)[:, 0]
     for _ in range(count)]
).mean(0)
posterior_rmse = float(((posterior_mean - target) ** 2).mean().sqrt())
prior_rmse = float(((prior_mean - target) ** 2).mean().sqrt())
print(f"held-out RMSE: posterior mean {posterior_rmse:.4f}, prior mean {prior_rmse:.4f}; "
      f"acceptance {float(result.acceptance_rates.mean()):.2f}")
assert posterior_rmse < 0.25
assert prior_rmse > 0.5
```

## Categorical perspective

A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. Prior draws over $\mathbb{R}^{13}\times\mathbb{R}_{>0}$ compose with the response kernel to define the generative model. The deterministic network maps a weight vector and an input to the Normal mean. Observing data weights this sampling distribution by a likelihood. The resulting measure need not have total mass one, so the scored program uses weighted-measure semantics [@staton2017commutative].

At a new input, composing the inferred parameter posterior with the response kernel gives the posterior predictive $\int p(y\mid x,\theta)p(\theta\mid\mathcal{D})\,d\theta$. Its spread reflects both observation noise and disagreement among posterior functions. A neural kernel with point-estimated weights instead stores those weights in its [`ParamSource`](../api/continuous/param_source.md#quivers.continuous.param_source.ParamSource). Declared sample sites and `weights=latent` make them random variables.

## See also

- [Bayesian Regression](bayesian-regression.md) for the linear special case, a network with no hidden layer.
- [Variational Autoencoder](vae.md) for neural kernels whose weights are point estimates, with an amortized posterior over latent codes instead of over weights.
