# Deep Markov Model

## Overview

The deep Markov model of Krishnan, Shalit, and Sontag [@krishnan2017structured] is a state-space model whose transition and emission are Normal distributions with neural-network parameters: the next state is drawn around a nonlinear function of the previous state and an observed driver, and each observation around a nonlinear function of its state. This source states the model with a [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) whose cell draws the next state and scores that position's observation, so the program's latent is the whole state path, one site per position. Each kernel is a [network kernel](../guides/dsl-declarations.md#kernel) whose one-hidden-layer `tanh` network maps its input to the mean and log-scale of a diagonal Normal, with point-estimated weights. The page checks the program's joint density against a hand-written one on both execution paths. It does not demonstrate inference of the state path or fitting of the networks; the section on inference below reports a check that failed.

## QVR source

```qvr
# Deep Markov Model
#
# A state-space model whose transition and emission are Normal kernels
# with neural-network parameters, after Krishnan, Shalit, and Sontag.
# Each kernel maps its input through a one-hidden-layer tanh network to
# the mean and log-scale of a diagonal Normal. A scan threads the state
# along the sequence of drivers: at every position the cell draws the
# next state given the driver and the previous state, then scores that
# position's observation under the emission.
#
# Generative structure (s_{-1} = 0):
#
#   s_t  ~ Normal(m_T(u_t, s_{t-1}), diag exp(l_T(u_t, s_{t-1})))
#   o_t  ~ Normal(m_E(s_t), diag exp(l_E(s_t)))
#
# The network weights are point estimates; the states s@0, s@1, ...
# are the latent sites, one per position.
#
# Reference: [Krishnan, Shalit, and Sontag 2017](https://doi.org/10.1609/aaai.v31i1.10779).

object Driver : Real 2
object State : Real 3
object Obs : Real 4

morphism transition : Driver * State -> State [param_source=mlp(32)] ~ Normal
morphism emission : State -> Obs [param_source=mlp(32)] ~ Normal

program step(u_t, s_prev) : Driver * State -> State
    sample s <- transition(u_t, s_prev)
    observe o <- emission(s)
    return s

define run_all = scan(step)

program deep_markov : Driver -> State
    sample s_last <- run_all
    return s_last

export deep_markov
```

## Understanding the model

With $u_t$ the driver at position $t$ and $s_{-1} = 0$, the program scores

$$
\begin{aligned}
s_t &\sim \mathcal{N}\bigl(m_T(u_t, s_{t-1}), \operatorname{diag} \exp(\ell_T(u_t, s_{t-1}))\bigr), \\
o_t &\sim \mathcal{N}\bigl(m_E(s_t), \operatorname{diag} \exp(\ell_E(s_t))\bigr),
\end{aligned}
$$

where $(m_T, \ell_T)$ and $(m_E, \ell_E)$ are the two halves of the output of a network $W_2 \tanh(W_1 v + b_1) + b_2$ applied to the kernel's input $v$. Without the hidden layer each kernel's parameters would be affine in its input, as in the [continuous state-space model](continuous-hmm.md); the `tanh` layer is what makes the dynamics and the emission nonlinear. Each input row is one run: a sequence of drivers, with the state path drawn at the sites `s@0`, `s@1`, and so on and the observations `o` scored at every position.

Krishnan, Shalit, and Sontag fit such a model by maximizing an ELBO whose variational posterior over the path is itself a recurrent network run over the observations [@krishnan2017structured]. This page states the generative model and checks its density; it does not include that inference network.

## Walkthrough

- `morphism transition : Driver * State -> State [param_source=mlp(32)] ~ Normal` is a Normal kernel whose mean and log-scale come from a `tanh` network with 32 hidden units reading the concatenation $[u_t; s_{t-1}]$. `emission` is the same construction from a state to an observation.
- `program step` is one position: it draws `s` from the transition at the current driver and previous state, observes `o` under the emission at `s`, and returns `s`.
- `define run_all = scan(step)` threads the state from zero along the drivers. `program deep_markov` draws the path and returns its last state.
- No transpile target lowers a scan, so every target refuses this program with `scan:no-lowering`. The program runs on the PyTorch runtime and on the reference machine, which takes the networks' weights as data.

## Try it

The checks below come from a verification script whose tolerances were fixed before it was first run.

### The log joint against a hand-written recursion

At a traced draw of the state path on two runs of length 7, the compiled runtime's log joint must equal a float64 loop over positions computed from the networks' weights to a relative 1e-5, and the reference machine, given the same weights as data and the same states, to 1e-9. Two wrong oracles must each differ by more than a relative 1e-3: one whose emission reads the previous state rather than the one just drawn, an off-by-one in the scan's wiring that leaves every shape valid, and one that drops the transition term.

```python
import math

import torch
import torch.distributions as D

from quivers.dsl import load
from quivers.inference import trace

SOURCE = "docs/examples/source/deep_markov.qvr"
DRIVER, STATE, OBS = 2, 3, 4


def networks(model: torch.nn.Module) -> dict[str, torch.nn.Sequential]:
    found: dict[str, torch.nn.Sequential] = {}
    for path, module in model.named_modules():
        if path.endswith("_step_s.param_source.net"):
            found["transition"] = module
        elif path.endswith("_step_o.param_source.net"):
            found["emission"] = module
    assert set(found) == {"transition", "emission"}, found
    return found


def weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The checked input names of both networks' weights, float64."""
    w: dict[str, torch.Tensor] = {}
    for name, net in networks(model).items():
        w[f"{name}_param_layer0_weight"] = net[0].weight
        w[f"{name}_param_layer0_bias"] = net[0].bias
        w[f"{name}_param_weight"] = net[2].weight
        w[f"{name}_param_bias"] = net[2].bias
    return {k: v.detach().double() for k, v in w.items()}


def head(w: dict[str, torch.Tensor], name: str, value: torch.Tensor, dim: int) -> D.Normal:
    hidden = torch.tanh(w[f"{name}_param_layer0_weight"] @ value + w[f"{name}_param_layer0_bias"])
    raw = w[f"{name}_param_weight"] @ hidden + w[f"{name}_param_bias"]
    return D.Normal(raw[:dim], raw[dim:].exp())


def oracle(
    w: dict[str, torch.Tensor],
    drivers: torch.Tensor,
    states: torch.Tensor,
    observations: torch.Tensor,
    variant: str = "",
) -> float:
    """One run's log joint by a loop over positions, float64."""
    total = 0.0
    previous = torch.zeros(STATE, dtype=torch.float64)
    for t in range(drivers.shape[0]):
        transition = head(w, "transition", torch.cat([drivers[t], previous]), STATE)
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
drivers = torch.randn(runs, length, DRIVER, generator=generator)
observations = torch.randn(runs, length, OBS, generator=generator)
recorded = trace(model, drivers, observations={"o": observations})
w = weights(model)
for r in range(runs):
    states = torch.stack([recorded.sites[f"s@{t}"].value[r] for t in range(length)]).double().detach()
    expected = oracle(w, drivers[r].double(), states, observations[r].double())
    got = float(recorded.log_joint[r].detach())
    wrong = {
        label: oracle(w, drivers[r].double(), states, observations[r].double(), variant=label)
        for label in ("emission_reads_previous", "drop_transition")
    }
    run = program.run(
        "deep_markov",
        data={
            "driver": host(drivers[r].double()),
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

### Inference of the state path: a check that failed

The verification script also ran an inference check. It set the networks to a known generator (the seeded initial networks, with the transition's scale fixed at 0.3 and the emission's at 0.1), simulated a path of 10 states and its observations, and inferred the path with the networks frozen by SVI with a mean-field Normal guide (1500 Adam steps, 400 posterior draws). The threshold, fixed in advance, required the posterior-mean path to be closer to the true path than the prior-mean path by an RMSE ratio below 0.5. The ratio was 0.84 (RMSE 0.289 against 0.345), so the check failed.

The failure does not separate a weak inference from a weak design. The prior-mean path is already within 0.35 of the true path, close to the transition noise, so the observations had little room to improve on it, and the threshold was not calibrated against how much information they carry. On the [continuous state-space model](continuous-hmm.md), whose posterior is Gaussian, the same design failed in the same way while the exact posterior mode was recovered. For a nonlinear model no exact posterior is available to calibrate against, and this page makes no inference claim.

## Categorical perspective

The cell `step` is a Kleisli morphism $\mathrm{Driver} \otimes \mathrm{State} \to \mathrm{State}$ of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical], weighted by the likelihood of the observation it scores, and `scan(step)` is its composite along the drivers in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category). The networks enter only through the kernels' parameter maps, so the categorical structure is that of the linear-Gaussian model; what changes is that the composite's posterior is no longer Gaussian, which is why it has no closed form.

## Sources and further reading

- *Structured inference networks for nonlinear state space models* [@krishnan2017structured].
- *A categorical approach to probability theory* [@giry1982categorical].
