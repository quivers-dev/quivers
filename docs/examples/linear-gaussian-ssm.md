# Linear-Gaussian State-Space Model

## Overview

A linear-Gaussian [state-space model](https://en.wikipedia.org/wiki/State-space_representation) explains a sequence of noisy observations by a hidden state that evolves linearly under Gaussian noise. Given the model's matrices and noise scales, the posterior over the state path is Gaussian, and the [Kalman filter](https://doi.org/10.1115/1.3662552) [@kalman1960filtering] followed by the [Rauch-Tung-Striebel smoother](https://doi.org/10.2514/3.3166) [@rauch1965smoothing] computes its means and variances exactly. This source states the model with a [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) whose cell draws the next state given the previous one, so the program's latent is the whole coupled state path. The transition and emission maps are [learned deterministic functions](../guides/dsl-declarations.md#function), affine maps with point-estimated weights, and the two noise scales are sites of the program. The examples compare the program's joint density against a hand-written one and its posterior against the smoother.

## QVR source

```qvr
# A linear-Gaussian state-space model with a learned linear transition and
# emission and constant noise scales:
#   s_t = A s_{t-1} + B u_t + b + q e_t,    o_t = C s_t + c + r f_t,
# with e_t, f_t standard normal and s_{-1} = 0. The transition and emission
# maps are learned functions, so their weights are point estimates; the
# noise scales are sites of the scanning program, read by every position.
object Driver : Real 1
object State : Real 2
object Obs : Real 2

morphism transition : Driver * State -> State [role=function]
morphism emission : State -> Obs [role=function]

program step(u_t, s_prev) : Driver * State -> State
    let mean = transition(u_t, s_prev)
    sample s <- Normal(loc=mean, scale=q)
    let center = emission(s)
    observe o <- Normal(loc=center, scale=r)
    return s

define run_all = scan(step)

program ssm : Driver -> State
    sample q <- HalfNormal(scale=1.0)
    sample r <- HalfNormal(scale=1.0)
    sample last <- run_all
    return last

export ssm
```

## Understanding the model

With $u_t$ an observed driver and $s_{-1} = 0$, the program scores

$$
\begin{aligned}
s_t &= A s_{t-1} + B u_t + b + q\, e_t, & e_t &\sim \mathcal{N}(0, I), \\
o_t &= C s_t + c + r\, f_t, & f_t &\sim \mathcal{N}(0, I),
\end{aligned}
$$

with half-Normal priors on the scales $q$ and $r$. The transition reads the concatenation $[u_t; s_{t-1}]$, so the learned weight of `transition` is the block matrix $[B\ A]$ and its bias is $b$. The weight and bias of `emission` are $C$ and $c$. Each input row is one run: a sequence of drivers, with the state path drawn at the sites `s@0`, `s@1`, and so on, and the observations `o` scored at every position.

The process noise is isotropic, $Q = q^2 I$, and so is the observation noise, $R = r^2 I$. With $A$, $B$, $b$, $C$, $c$, $q$, and $r$ fixed, the state path's posterior given the observations is Gaussian, and the smoother gives its means exactly. That is the check below: the program's posterior, sampled by [NUTS](../guides/inference-mcmc.md) (No-U-Turn Sampler), against the smoother.

## Walkthrough

- `object Driver : Real 1`, `object State : Real 2`, and `object Obs : Real 2` are the driver, state, and observation spaces.
- `morphism transition : Driver * State -> State [role=function]` is the affine map $[u_t; s_{t-1}] \mapsto B u_t + A s_{t-1} + b$, and `emission` is $s_t \mapsto C s_t + c$. Both are deterministic with point-estimated weights.
- `program step` is one position: it draws the next state around the transition's mean at scale `q`, and observes `o` around the emission's mean at scale `r`. The scales `q` and `r` are not parameters of `step`. The cell reads them from the scanning program, so every position shares them.
- `define run_all = scan(step)` threads the state from zero along the drivers. `program ssm` draws the two scales and then the path.
- The program runs on the PyTorch runtime and the [reference evaluator](../developer/qiec.md#reference-evaluator). Backend coverage is recorded in the [support matrix](../transpile-support.md).

## Try it

Both blocks set the learned functions to known matrices.

### The log joint against a hand-written recursion

With the weights at the known matrices, the compiled runtime's log joint at fixed values of `q`, `r`, the state path, and the observations must equal a float64 loop over positions to a relative 1e-5, and the reference machine, given the weights as data, to 1e-9. A reference calculation with the two noise scales swapped must disagree by more than a relative 1e-3, which shows the first comparison can tell them apart.

```python
import math

import torch
import torch.distributions as D

from quivers.continuous import LearnedFunction, LinearSource
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel

SOURCE = "docs/examples/source/linear_gaussian_ssm.qvr"
LENGTH = 12

A = torch.tensor([[0.8, 0.1], [-0.2, 0.7]], dtype=torch.float64)
B = torch.tensor([[0.5], [-0.3]], dtype=torch.float64)
B_SHIFT = torch.tensor([0.1, 0.0], dtype=torch.float64)
C = torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64)
C_SHIFT = torch.tensor([0.0, 0.2], dtype=torch.float64)
Q, R = 0.4, 0.3


def learned(model, name):
    found = [m for p, m in model.named_modules() if p.split(".")[-1] == f"_function_{name}"]
    assert len(found) == 1 and isinstance(found[0], LearnedFunction)
    assert isinstance(found[0].source, LinearSource)
    return found[0].source


def set_known_weights(model):
    """Set the learned functions to the known matrices; return them as data."""
    transition, emission = learned(model, "transition"), learned(model, "emission")
    with torch.no_grad():
        # The transition reads [u_t; s_{t-1}], the domain's factors in order.
        transition.linear.weight.copy_(torch.cat([B, A], dim=1).float())
        transition.linear.bias.copy_(B_SHIFT.float())
        emission.linear.weight.copy_(C.float())
        emission.linear.bias.copy_(C_SHIFT.float())
    return {
        "transition_param_weight": torch.cat([B, A], dim=1),
        "transition_param_bias": B_SHIFT,
        "emission_param_weight": C,
        "emission_param_bias": C_SHIFT,
    }


def oracle(q, r, u, s, o, swap_scales=False):
    """The joint density by a loop over positions, float64."""
    q_t, r_t = (r, q) if swap_scales else (q, r)
    total = float(D.HalfNormal(1.0).log_prob(torch.tensor(q, dtype=torch.float64)))
    total += float(D.HalfNormal(1.0).log_prob(torch.tensor(r, dtype=torch.float64)))
    previous = torch.zeros(2, dtype=torch.float64)
    for t in range(u.shape[0]):
        total += float(D.Normal(A @ previous + B @ u[t] + B_SHIFT, q_t).log_prob(s[t]).sum())
        total += float(D.Normal(C @ s[t] + C_SHIFT, r_t).log_prob(o[t]).sum())
        previous = s[t]
    return total


def simulate(seed):
    """Drivers, a state path, and observations from the known model."""
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(LENGTH, 1, generator=generator, dtype=torch.float64)
    s = torch.empty(LENGTH, 2, dtype=torch.float64)
    o = torch.empty(LENGTH, 2, dtype=torch.float64)
    previous = torch.zeros(2, dtype=torch.float64)
    for t in range(LENGTH):
        s[t] = A @ previous + B @ u[t] + B_SHIFT + Q * torch.randn(2, generator=generator, dtype=torch.float64)
        o[t] = C @ s[t] + C_SHIFT + R * torch.randn(2, generator=generator, dtype=torch.float64)
        previous = s[t]
    return u, s, o


def host(value):
    return float(value) if value.dim() == 0 else tuple(host(row) for row in value)


torch.manual_seed(0)
program = load(SOURCE)
model = program.morphism
weights = set_known_weights(model)
u, s, o = simulate(seed=1)
given = {
    "q": torch.tensor([[Q]]),
    "r": torch.tensor([[R]]),
    "o": o.float()[None],
    **{f"s@{t}": s[t].float()[None] for t in range(LENGTH)},
}
got = float(model.log_joint(u.float()[None], given).detach()[0])
expected = oracle(Q, R, u, s, o)
wrong = oracle(Q, R, u, s, o, swap_scales=True)
run = program.run(
    "ssm",
    data={"driver": host(u), "o": host(o), **{k: host(v) for k, v in weights.items()}},
    sites={"q": Q, "r": R, **{f"s@{t}": host(s[t]) for t in range(LENGTH)}},
    seed=0,
)
print(f"compiled {got:.6f}, reference {run.log_joint:.9f}, oracle {expected:.9f}; scales swapped {wrong:.6f}")
assert math.isclose(got, expected, rel_tol=1e-5)
assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
assert not math.isclose(got, wrong, rel_tol=1e-3)
```

### The posterior against the Rauch-Tung-Striebel smoother

With the weights at the known matrices and `q` and `r` observed at their true values, the only latents are the 24 coordinates of the state path, and their posterior means are the smoother's. NUTS (one chain, 200 warmup and 400 kept draws) must put every coordinate's posterior mean within four Monte Carlo standard errors of the smoother's mean, the standard error being the smoother's standard deviation over the square root of the coordinate's effective sample size. A smoother with the transition matrix transposed must be rejected at the same tolerance at one coordinate or more, which shows the check can tell the two dynamics apart.

```python
def smoother(u, o, transition):
    """Rauch-Tung-Striebel smoothed means and marginal variances, float64."""
    eye = torch.eye(2, dtype=torch.float64)
    mean, cov = torch.zeros(2, dtype=torch.float64), torch.zeros(2, 2, dtype=torch.float64)
    filtered_means, filtered_covs, predicted_means, predicted_covs = [], [], [], []
    for t in range(u.shape[0]):
        prior_mean = transition @ mean + B @ u[t] + B_SHIFT
        prior_cov = transition @ cov @ transition.T + Q**2 * eye
        gain = prior_cov @ C.T @ torch.linalg.inv(C @ prior_cov @ C.T + R**2 * eye)
        mean = prior_mean + gain @ (o[t] - C @ prior_mean - C_SHIFT)
        cov = (eye - gain @ C) @ prior_cov
        predicted_means.append(prior_mean)
        predicted_covs.append(prior_cov)
        filtered_means.append(mean)
        filtered_covs.append(cov)
    means, covs = list(filtered_means), list(filtered_covs)
    for t in range(u.shape[0] - 2, -1, -1):
        gain = filtered_covs[t] @ transition.T @ torch.linalg.inv(predicted_covs[t + 1])
        means[t] = filtered_means[t] + gain @ (means[t + 1] - predicted_means[t + 1])
        covs[t] = filtered_covs[t] + gain @ (covs[t + 1] - predicted_covs[t + 1]) @ gain.T
    return torch.stack(means), torch.stack([torch.diagonal(c) for c in covs])


torch.manual_seed(2)
model = load(SOURCE).morphism
set_known_weights(model)
u, _, o = simulate(seed=3)
observations = {"q": torch.tensor([[Q]]), "r": torch.tensor([[R]]), "o": o.float()[None]}
result = MCMC(NUTSKernel(max_tree_depth=6), num_warmup=200, num_samples=400, num_chains=1, seed=0).run(
    model, u.float()[None], observations
)
means, variances = smoother(u, o, A)
wrong_means, _ = smoother(u, o, A.T)
worst, rejected = 0.0, 0
for t in range(LENGTH):
    draws = result.samples[f"s@{t}"].reshape(-1, 2).double()
    effective = result.ess[f"s@{t}"].reshape(-1).double()
    for k in range(2):
        tolerance = 4.0 * math.sqrt(float(variances[t, k])) / math.sqrt(float(effective[k]))
        deviation = abs(float(draws[:, k].mean()) - float(means[t, k]))
        worst = max(worst, deviation / tolerance)
        assert deviation <= tolerance, (t, k, deviation, tolerance)
        if abs(float(draws[:, k].mean()) - float(wrong_means[t, k])) > tolerance:
            rejected += 1
print(f"largest deviation {worst:.2f} of the tolerance; transposed-transition smoother rejected at {rejected} coordinates")
assert rejected >= 1
```

What this demonstrates is narrow and exact: at known matrices and scales, the program's posterior over the coupled state path is the Kalman smoother's. The page does not fit the matrices or the scales.

## Categorical perspective

A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. The state transition and observation distributions are probability kernels. With the driver held fixed, composing affine Gaussian transitions and conditioning on Gaussian observations gives a Gaussian distribution over the state path, which the smoother computes. The sampling cell takes a driver and a state, $\mathrm{Driver}\times\mathrm{State}$, and returns a state. `scan(step)` repeats this transition along the driver sequence.

Observing data weights this sampling distribution by a likelihood. The resulting measure need not have total mass one, so the scored program uses weighted-measure semantics [@staton2017commutative].
