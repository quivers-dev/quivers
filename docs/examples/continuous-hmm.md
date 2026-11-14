# Continuous State-Space Model

## Overview

A continuous state-space model is a hidden Markov model whose hidden state is a real vector: the state evolves by a Gaussian Markov chain and each observation is Gaussian given the state at its position. This source writes the two conditionals as Normal kernels with the default linear parameter source, so each maps its input affinely to both the mean and the log-scale of a diagonal Normal, and the noise of each step can depend on the state. A [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) whose cell draws the next state and scores that position's observation makes the whole state path the program's latent, one site per position. The examples compare the program's joint density on both execution paths and, at weights that make the model linear-Gaussian, compare the posterior mode of the state path against the Rauch-Tung-Striebel smoother [@rauch1965smoothing]. Two variational checks of the posterior failed and are reported below.

## QVR source

```qvr
# Continuous State-Space Model
#
# A hidden Markov model whose state is a real vector: the state follows a
# Gaussian Markov chain and each position's observation is Gaussian given
# that position's state. Both kernels take the default linear parameter
# source, so each maps its input affinely to the mean and the log-scale
# of a diagonal Normal; the noise thus varies with the state. The
# scan's entries only set the length of the sequence.
#
# Generative structure (s_{-1} = 0):
# Normal arguments below are mean and standard deviation.
#
#   s_t  ~ Normal(A s_{t-1} + a, diag exp(B s_{t-1} + b))
#   o_t  ~ Normal(C s_t + c, diag exp(D s_t + d))
#
# The weights are point estimates; the states s@0, s@1, ... are the
# latent sites, one per position.

object Tick : Real 1
object State : Real 2
object Obs : Real 3

morphism transition : State -> State ~ Normal
morphism emission : State -> Obs ~ Normal

program step(u_t, s_prev) : Tick * State -> State
    sample s <- transition(s_prev)
    observe o <- emission(s)
    return s

define run_all = scan(step)

program continuous_hmm : Tick -> State
    sample s_last <- run_all
    return s_last

export continuous_hmm
```

## Understanding the model

With $s_{-1} = 0$, the program scores

$$
\begin{aligned}
s_t &\sim \mathcal{N}\bigl(A s_{t-1} + a, \operatorname{diag} \exp(2(B s_{t-1} + b))\bigr), \\
o_t &\sim \mathcal{N}\bigl(C s_t + c, \operatorname{diag} \exp(2(D s_t + d))\bigr),
\end{aligned}
$$

Here the second argument of $\mathcal{N}$ is the covariance. The networks output log standard deviations, so doubling them before exponentiation gives the diagonal variances.

The learned weight of `transition` is the stacked matrix $[A; B]$ with bias $[a; b]$, and likewise $[C; D]$ and $[c; d]$ for `emission`. When $B$ and $D$ are zero the noise is constant and the model is the linear-Gaussian state-space model of the [Kalman filter](linear-gaussian-ssm.md) [@kalman1960filtering], whose state posterior is Gaussian with means the smoother computes. With $B$ or $D$ nonzero the model is heteroscedastic, and its posterior has no closed form. The program has no driver. The scan's entries, `Tick`, only set the length of the sequence. Each input row is one run, with the states drawn at the sites `s@0`, `s@1`, and so on and the observations `o` scored at every position.

## Walkthrough

- `morphism transition : State -> State ~ Normal` and `morphism emission : State -> Obs ~ Normal` have no parameter source declared, so each takes the default linear one: one affine map from the input to the concatenated mean and log-scale.
- `program step(u_t, s_prev)` ignores `u_t`, draws `s` from the transition at the previous state, observes `o` under the emission at `s`, and returns `s`.
- `define run_all = scan(step)` threads the state from zero. `program continuous_hmm` draws the path and returns its last state.
- The program runs on the PyTorch runtime and the [reference evaluator](../developer/qiec.md#reference-evaluator). Backend coverage is recorded in the [support matrix](../transpile-support.md).

## Try it

### The log joint against a hand-written recursion

At a traced draw of the state path on two runs of length 7, the compiled runtime's log joint must equal a float64 loop over positions to a relative 1e-5, and the reference machine to 1e-9. A reference calculation whose emission reads the previous state and a reference calculation that drops the transition term must each differ by more than a relative 1e-3.

```python
import math

import torch
import torch.distributions as D

from quivers.dsl import load
from quivers.inference import trace

SOURCE = "docs/examples/source/continuous_hmm.qvr"
STATE, OBS = 2, 3


def networks(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    found: dict[str, torch.nn.Linear] = {}
    for path, module in model.named_modules():
        if path.endswith("_step_s.param_source.linear"):
            found["transition"] = module
        elif path.endswith("_step_o.param_source.linear"):
            found["emission"] = module
    assert set(found) == {"transition", "emission"}, found
    return found


def weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The checked input names of both kernels' weights, float64."""
    w: dict[str, torch.Tensor] = {}
    for name, linear in networks(model).items():
        w[f"{name}_param_weight"] = linear.weight
        w[f"{name}_param_bias"] = linear.bias
    return {k: v.detach().double() for k, v in w.items()}


def head(w: dict[str, torch.Tensor], name: str, value: torch.Tensor, dim: int) -> D.Normal:
    raw = w[f"{name}_param_weight"] @ value + w[f"{name}_param_bias"]
    return D.Normal(raw[:dim], raw[dim:].exp())


def oracle(
    w: dict[str, torch.Tensor],
    length: int,
    states: torch.Tensor,
    observations: torch.Tensor,
    variant: str = "",
) -> float:
    """One run's log joint by a loop over positions, float64."""
    total = 0.0
    previous = torch.zeros(STATE, dtype=torch.float64)
    for t in range(length):
        transition = head(w, "transition", previous, STATE)
        if variant != "drop_transition":
            total += float(transition.log_prob(states[t]).sum())
        read = previous if variant == "emission_reads_previous" else states[t]
        total += float(head(w, "emission", read, OBS).log_prob(observations[t]).sum())
        previous = states[t]
    return total


def host(value: torch.Tensor) -> object:
    return float(value) if value.dim() == 0 else tuple(host(row) for row in value)


torch.manual_seed(0)
program = load(SOURCE)
model = program.morphism
runs, length = 2, 7
generator = torch.Generator().manual_seed(1)
ticks = torch.zeros(runs, length, 1)
observations = torch.randn(runs, length, OBS, generator=generator)
recorded = trace(model, ticks, observations={"o": observations})
w = weights(model)
for r in range(runs):
    states = torch.stack([recorded.sites[f"s@{t}"].value[r] for t in range(length)]).double().detach()
    expected = oracle(w, length, states, observations[r].double())
    got = float(recorded.log_joint[r].detach())
    wrong = {
        label: oracle(w, length, states, observations[r].double(), variant=label)
        for label in ("emission_reads_previous", "drop_transition")
    }
    run = program.run(
        "continuous_hmm",
        data={
            "tick": host(ticks[r].double()),
            "o": host(observations[r].double()),
            **{k: host(v) for k, v in w.items()},
        },
        sites={f"s@{t}": host(states[t]) for t in range(length)},
        seed=0,
    )
    print(f"run {r}: compiled {got:.6f}, reference {run.log_joint:.9f}, oracle {expected:.9f}; "
          + ", ".join(f"{k} {v:.6f}" for k, v in wrong.items()))
    assert math.isclose(got, expected, rel_tol=1e-5)
    assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
    assert all(not math.isclose(got, value, rel_tol=1e-3) for value in wrong.values())
```

### The posterior mode against the smoother

The kernels are set to a known generator: the seeded initial weights, with the log-scale rows zeroed and their biases set so that the transition's scale is 0.3 and the emission's 0.1. The model is then linear-Gaussian, a path of 10 states and its observations are simulated from it, and with the weights frozen L-BFGS maximizes the log joint over the state path. For a Gaussian posterior the mode is the mean, so the result must match the smoother's means within 1e-3 at every coordinate, while a smoother with the transition matrix transposed must differ from it by more than 1e-3 somewhere.

The fitted mode differs from the smoother by at most 8.8e-5, below the 1e-3 comparison tolerance. This agreement at a fixed linear-Gaussian setting is a density diagnostic. It does not establish that the variational fit below approximates the posterior accurately.

```python
def set_generator(model: torch.nn.Module) -> None:
    """Fix both networks' noise scales: zero the log-scale rows, set the biases."""
    with torch.no_grad():
        for name, dim, scale in (("transition", STATE, 0.3), ("emission", OBS, 0.1)):
            last = networks(model)[name]
            last.weight[dim:].zero_()
            last.bias[dim:].fill_(math.log(scale))


def simulate(
    w: dict[str, torch.Tensor], length: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    states, observations = [], []
    previous = torch.zeros(STATE, dtype=torch.float64)
    for t in range(length):
        transition = head(w, "transition", previous, STATE)
        state = transition.loc + transition.scale * torch.randn(STATE, generator=generator, dtype=torch.float64)
        emission = head(w, "emission", state, OBS)
        observations.append(emission.loc + emission.scale * torch.randn(OBS, generator=generator, dtype=torch.float64))
        states.append(state)
        previous = state
    return torch.stack(states), torch.stack(observations)


def smoother(
    observations: torch.Tensor, transition: torch.Tensor, shift: torch.Tensor,
    emission: torch.Tensor, offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rauch-Tung-Striebel smoothed means and marginal variances, float64."""
    q, r = 0.3, 0.1
    eye_s, eye_o = torch.eye(STATE, dtype=torch.float64), torch.eye(OBS, dtype=torch.float64)
    mean, cov = torch.zeros(STATE, dtype=torch.float64), torch.zeros(STATE, STATE, dtype=torch.float64)
    filtered_means, filtered_covs, predicted_means, predicted_covs = [], [], [], []
    for t in range(observations.shape[0]):
        prior_mean = transition @ mean + shift
        prior_cov = transition @ cov @ transition.T + q**2 * eye_s
        gain = prior_cov @ emission.T @ torch.linalg.inv(emission @ prior_cov @ emission.T + r**2 * eye_o)
        mean = prior_mean + gain @ (observations[t] - emission @ prior_mean - offset)
        cov = (eye_s - gain @ emission) @ prior_cov
        predicted_means.append(prior_mean)
        predicted_covs.append(prior_cov)
        filtered_means.append(mean)
        filtered_covs.append(cov)
    means, covs = list(filtered_means), list(filtered_covs)
    for t in range(observations.shape[0] - 2, -1, -1):
        gain = filtered_covs[t] @ transition.T @ torch.linalg.inv(predicted_covs[t + 1])
        means[t] = filtered_means[t] + gain @ (means[t + 1] - predicted_means[t + 1])
        covs[t] = filtered_covs[t] + gain @ (covs[t + 1] - predicted_covs[t + 1]) @ gain.T
    return torch.stack(means), torch.stack([torch.diagonal(c) for c in covs])


torch.manual_seed(2)
model = load(SOURCE).morphism
set_generator(model)
for p in model.parameters():
    p.requires_grad_(False)
w = weights(model)
length = 10
generator = torch.Generator().manual_seed(3)
_, observations = simulate(w, length, generator)
x = torch.zeros(1, length, 1)
data = {"o": observations.float()[None]}
path = torch.zeros(length, STATE, requires_grad=True)
optimizer = torch.optim.LBFGS([path], max_iter=200, line_search_fn="strong_wolfe")


def closure():
    optimizer.zero_grad()
    loss = -model.log_joint(x, {**data, **{f"s@{t}": path[t][None] for t in range(length)}}).sum()
    loss.backward()
    return loss


optimizer.step(closure)
A, a = w["transition_param_weight"][:STATE], w["transition_param_bias"][:STATE]
C, c = w["emission_param_weight"][:OBS], w["emission_param_bias"][:OBS]
means, _ = smoother(observations, A, a, C, c)
wrong_means, _ = smoother(observations, A.T, a, C, c)
mode = path.detach().double()
deviation, separation = float((mode - means).abs().max()), float((mode - wrong_means).abs().max())
print(f"largest |mode - smoother mean| {deviation:.2e}; transposed smoother largest gap {separation:.3f}")
assert deviation < 1e-3
assert separation > 1e-3
```

### Variational checks that failed

Two checks of the posterior by [SVI](../guides/inference-svi.md) (stochastic variational inference) with a [mean-field](../guides/inference-svi.md) Normal guide (1500 Adam steps at learning rate 0.02, 400 posterior draws) used the same generator and data, and both failed.

1. The first required the posterior-mean path to be closer to the true path than the prior-mean path, by an RMSE ratio below 0.5. The ratio was 0.69 (RMSE 0.169 against 0.245). The prior-mean path is already close to the true one, and the threshold was not calibrated against what the observations can add, so the ratio alone does not diagnose an error in the density.
2. The second required every coordinate of the SVI posterior mean to lie within four Monte Carlo standard errors plus 0.02 of the smoother's mean. A mean-field Gaussian guide has the exact posterior means at its optimum when the posterior is Gaussian, so this checks that the fit reached that optimum. The largest deviation was 0.108, 1.58 times its tolerance. The posterior mode above agrees with the smoother to 8.8e-5, but this agreement does not identify the cause of the variational error. The reported SVI fit fails its posterior-mean criterion.

## Categorical perspective

A [probability kernel](https://en.wikipedia.org/wiki/Markov_kernel) assigns a probability distribution to each input. Such kernels compose by integration in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) [@giry1982categorical]. `transition : State -> State` and `emission : State -> Obs` assign Gaussian distributions to their inputs. `scan(step)` composes the state transitions along the sequence and scores each observation at its state. When $B=D=0$, the log-scales are constant and the means are affine. Conditioning then gives a Gaussian state-path posterior, and the smoother computes its mean and mode.

Observing data weights this sampling distribution by a likelihood. The resulting measure need not have total mass one, so the scored program uses weighted-measure semantics [@staton2017commutative].
