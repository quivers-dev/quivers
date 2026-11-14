# Hidden Markov Model

## Overview

A discrete [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM) [@rabiner1989hmm] explains a sequence of observed symbols by a chain of hidden states: the chain starts from an initial distribution, moves by a row-stochastic transition matrix, and each state it moves to emits one symbol from its row of an emission matrix. The likelihood of the symbols sums over every path of hidden states. This source writes the model as a [marginalized chain](../reference/qvr/probabilistic-programs.md#marginalized-chains): a scan whose cell marginalizes the next state and returns it, which the library sums out with the forward algorithm in log space. The three parameter matrices carry symmetric Dirichlet priors, so the program's log joint is those priors plus the exact log marginal likelihood of the sequence.

## QVR source

```qvr
# Hidden Markov Model
#
# A K-state hidden Markov model over a discrete alphabet. The hidden
# chain z_0, z_1, ..., z_T starts from initial_row and moves by
# transition_rows; each z_t (t >= 1) emits the symbol o_t from its row of
# emission_rows. The cell marginalizes the next state and returns it, so
# the scan sums every path of states out: the library computes the
# likelihood, a sum over K^(T+1) paths, by the forward algorithm in
# O(T K^2), in log space, and adds it to the joint under `path`.
#
# Generative structure:
#
#   initial_row          ~ Dirichlet(1)            p(z_0)
#   transition_rows[i]   ~ Dirichlet(1)            p(z_t | z_{t-1} = i)
#   emission_rows[k]     ~ Dirichlet(1)            p(o_t | z_t = k)
#   z_0 ~ initial_row,  z_t ~ transition_rows[z_{t-1}],  o_t ~ emission_rows[z_t]
#
# The scanned sequence is the observed symbols: the block observes the
# cell's entry `symbol`, which scores the symbol at the state the chain
# moves to. The state before the first symbol, z_0, emits nothing.
#
# Reference: [Rabiner 1989](https://doi.org/10.1109/5.18626).

object State : FinSet 2
object Symbol : FinSet 4
object StateProbs : Real 2

program hmm_step(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition_rows[z_prev])
        observe symbol <- Categorical(probs=emission_rows[z])
    return z

define chain = scan(hmm_step, init=initial_row)

program hmm : Symbol -> StateProbs
    sample initial_row <- Dirichlet(concentration=1.0) [over=State]
    sample transition_rows : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission_rows : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return initial_row

export hmm
```

## Understanding the model

With $\pi$ the initial row, $A$ the transition matrix, and $B$ the emission matrix, a sequence $o_1, \ldots, o_T$ has likelihood

$$
p(o_{1:T} \mid \pi, A, B)
  = \sum_{z_0, \ldots, z_T} \pi_{z_0} \prod_{t=1}^{T} A_{z_{t-1} z_t} B_{z_t o_t}.
$$

The chain has $T + 1$ states for $T$ symbols. The state $z_0$ is drawn from $\pi$ and emits nothing. The first symbol is emitted by $z_1$, the state the chain moves to from $z_0$. This differs from the convention in which the first state is drawn from $\pi$ and emits $o_1$ directly [@rabiner1989hmm]. This program's likelihood at $\pi$ equals that convention's likelihood at $\pi A$, the distribution of $z_1$. At a given $\pi$ the two score different numbers, and the checks below show the program computes this one and not the other.

The sum has $K^{T+1}$ terms. The forward algorithm computes it in $O(T K^2)$ by carrying $\alpha_t(k) = p(o_{1:t}, z_t = k)$ along the sequence,

$$
\alpha_0 = \pi, \qquad
\alpha_t(k) = B_{k o_t} \sum_{j} \alpha_{t-1}(j) A_{j k}, \qquad
p(o_{1:T}) = \sum_k \alpha_T(k),
$$

and the library runs that recursion in log space, so a long sequence whose path probabilities underflow in floating point still has a finite log likelihood. Each input row is one run: one sequence, and its own draw of the three parameter matrices.

## Walkthrough

- `object State : FinSet 2` is the hidden state space and `object Symbol : FinSet 4` the alphabet. `object StateProbs : Real 2` is the value space of what the program returns, the initial row as a point of the two-state simplex.
- `program hmm_step(symbol, z_prev)` is one position. The `marginalize z : State` block draws the next state from the transition row of `z_prev`, and its body observes the cell's entry `symbol` under the emission row of `z`. Because the cell returns `z`, the marginalized state is carried to the next position rather than summed out locally.
- `define chain = scan(hmm_step, init=initial_row)` runs the cell along the input sequence. `init=initial_row` makes the initial state a draw from `initial_row`. A scan whose cell marginalizes the state it returns is summed out by the forward algorithm.
- `program hmm` draws `initial_row`, one `transition_rows` row per source state, and one `emission_rows` row per state from `Dirichlet(1)`, then scores the sequence under `path`. On the simplex `Dirichlet(1)` is flat, so the posterior mode is the maximum-likelihood estimate.
- The program runs on the PyTorch runtime and the [reference evaluator](../developer/qiec.md#reference-evaluator). Backend coverage is recorded in the [support matrix](../transpile-support.md).

## Try it

### Simulating sequences

A two-state, four-symbol HMM in which state 0 favours the first two symbols, state 1 the last two, and each state tends to persist. Four sequences of 200 symbols are simulated from it. Each row of `x_in` is one observed sequence, the input the scan runs over.

```python
import torch

from quivers.dsl import load

SOURCE = "docs/examples/source/hmm.qvr"
K, V = 2, 4

true_initial_row = torch.tensor([0.5, 0.5])
true_transition_rows = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
true_emission_rows = torch.tensor([[0.6, 0.3, 0.05, 0.05], [0.05, 0.05, 0.3, 0.6]])


def simulate(count, steps, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(count):
        state = int(torch.multinomial(true_initial_row, 1, generator=generator))
        row = []
        for _ in range(steps):
            state = int(torch.multinomial(true_transition_rows[state], 1, generator=generator))
            row.append(int(torch.multinomial(true_emission_rows[state], 1, generator=generator)))
        rows.append(row)
    return torch.tensor(rows)


x_in = simulate(4, 200, seed=1)
observations = {}
program = load(SOURCE)
model = program.morphism
print(x_in.shape, x_in[0, :20].tolist())
```

### The log joint against enumeration and the forward algorithm

At a traced draw of the parameters, the compiled runtime's log joint on a seven-symbol sequence must equal the Dirichlet priors plus two independent computations of the log marginal likelihood: brute-force enumeration of all $2^8$ state paths, and a forward recursion written in torch (float32 runtime, absolute 1e-4). Two wrong likelihoods must each disagree with the runtime by more than that tolerance: one with the transition matrix transposed, and the convention in which the first state emits the first symbol. The reference machine, given fixed parameters, must equal the enumeration to a relative 1e-9.

```python
import itertools
import math

import torch.distributions as D

from quivers.inference import trace


def priors(initial, transition, emission):
    return (
        D.Dirichlet(torch.ones(K, dtype=initial.dtype)).log_prob(initial)
        + D.Dirichlet(torch.ones(K, dtype=initial.dtype)).log_prob(transition).sum()
        + D.Dirichlet(torch.ones(V, dtype=initial.dtype)).log_prob(emission).sum()
    )


def enumerated(initial, transition, emission, symbols):
    """log p(o_1..o_T) by summing over every path z_0..z_T."""
    terms = []
    for path in itertools.product(range(K), repeat=len(symbols) + 1):
        term = math.log(float(initial[path[0]]))
        for t, symbol in enumerate(symbols):
            term += math.log(float(transition[path[t], path[t + 1]]))
            term += math.log(float(emission[path[t + 1], symbol]))
        terms.append(term)
    peak = max(terms)
    return peak + math.log(math.fsum(math.exp(term - peak) for term in terms))


def forward(initial, transition, emission, symbols):
    """log p(o_1..o_T) by the forward algorithm in log space."""
    log_alpha = torch.log(initial)
    for symbol in symbols:
        log_alpha = torch.logsumexp(log_alpha[:, None] + torch.log(transition), dim=0)
        log_alpha = log_alpha + torch.log(emission[:, symbol])
    return torch.logsumexp(log_alpha, dim=0)


def first_state_emits(initial, transition, emission, symbols):
    """The other convention: z_1 ~ initial emits o_1, with no state before it."""
    log_alpha = torch.log(initial) + torch.log(emission[:, symbols[0]])
    for symbol in symbols[1:]:
        log_alpha = torch.logsumexp(log_alpha[:, None] + torch.log(transition), dim=0)
        log_alpha = log_alpha + torch.log(emission[:, symbol])
    return torch.logsumexp(log_alpha, dim=0)


symbols = [0, 3, 2, 1, 1, 3, 0]
torch.manual_seed(0)
recorded = trace(model, torch.tensor([symbols]))
initial = recorded.sites["initial_row"].value.reshape(K).double()
transition = recorded.sites["transition_rows"].value.reshape(K, K).double()
emission = recorded.sites["emission_rows"].value.reshape(K, V).double()
prior = float(priors(initial, transition, emission))
by_paths = prior + enumerated(initial, transition, emission, symbols)
by_forward = prior + float(forward(initial, transition, emission, symbols))
got = float(recorded.log_joint)
print(f"compiled runtime {got:.6f}, enumeration {by_paths:.6f}, forward {by_forward:.6f}")
assert abs(got - by_paths) < 1e-4
assert abs(got - by_forward) < 1e-4

wrong = {
    "transposed transitions": prior + float(forward(initial, transition.T.contiguous(), emission, symbols)),
    "first state emits": prior + float(first_state_emits(initial, transition, emission, symbols)),
}
for label, value in wrong.items():
    print(f"{label}: {value:.6f}")
    assert abs(got - value) > 1e-4

fixed = {
    "initial_row": torch.tensor([0.3, 0.7], dtype=torch.float64),
    "transition_rows": torch.tensor([[0.8, 0.2], [0.35, 0.65]], dtype=torch.float64),
    "emission_rows": torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.3, 0.5]], dtype=torch.float64),
}
run = program.run(
    "hmm",
    data={"symbol": tuple(symbols)},
    sites={
        "initial_row": tuple(fixed["initial_row"].tolist()),
        "transition_rows": tuple(tuple(row) for row in fixed["transition_rows"].tolist()),
        "emission_rows": tuple(tuple(row) for row in fixed["emission_rows"].tolist()),
    },
    seed=0,
)
expected = float(priors(*fixed.values())) + enumerated(*fixed.values(), symbols)
print(f"reference machine {run.log_joint:.9f}, enumeration {expected:.9f}")
assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
```

### A long sequence in log space

On a sequence of 2000 symbols the forward recursion in probability space underflows to exactly zero in float64. The compiled runtime's log joint must still be finite and equal the float64 log-space forward recursion to a relative 1e-5.

```python
long_symbols = torch.randint(0, V, (2000,), generator=torch.Generator().manual_seed(3)).tolist()
alpha = fixed["initial_row"].clone()
for symbol in long_symbols:
    alpha = (alpha @ fixed["transition_rows"]) * fixed["emission_rows"][:, symbol]
assert float(alpha.sum()) == 0.0

expected = float(priors(*fixed.values()) + forward(*fixed.values(), long_symbols))
got = float(model.log_joint(
    torch.tensor([long_symbols]),
    {name: value.float() for name, value in fixed.items()},
)[0])
print(f"probability space: {float(alpha.sum())}; log space: runtime {got:.4f}, oracle {expected:.4f}")
assert math.isfinite(got) and math.isclose(got, expected, rel_tol=1e-5)
```

### Maximum-likelihood recovery

The four simulated sequences are fitted by L-BFGS on unconstrained softmax parameters of the three simplices. The `Dirichlet(1)` priors are constant on the simplex, so this is the maximum-likelihood fit. The same parameter values are given for every row, so the loss is the pooled log likelihood of the four sequences. The fit starts from a seeded perturbation of the label-symmetric point, which is a stationary point of the likelihood.

The tolerances follow from the design. The chain's stationary distribution is $(2/3, 1/3)$, so state 1 is visited about 267 times in 800 steps. If the states were observed, the largest sampling standard deviation of an emission probability would be $\sqrt{0.6 \cdot 0.4 / 267} \approx 0.030$ and of a transition probability $\sqrt{0.8 \cdot 0.2 / 267} \approx 0.024$. With the states summed out the data carry less information, and each tolerance is four of those standard deviations: every emission probability within 0.12 of the truth and every transition probability within 0.10, after matching the two fitted states to the truth's labels (the likelihood is invariant to relabelling them). The initial row is weakly identified by four sequences and is not compared.

```python
logits = {
    "initial_row": torch.zeros(K),
    "transition_rows": torch.zeros(K, K),
    "emission_rows": 0.5 * torch.randn(K, V, generator=torch.Generator().manual_seed(4)),
}
for value in logits.values():
    value.requires_grad_(True)
optimizer = torch.optim.LBFGS(list(logits.values()), max_iter=80, line_search_fn="strong_wolfe")


def closure():
    optimizer.zero_grad()
    values = {name: torch.softmax(value, dim=-1) for name, value in logits.items()}
    loss = -model.log_joint(x_in, values).sum()
    loss.backward()
    return loss


optimizer.step(closure)
with torch.no_grad():
    fitted_transition = torch.softmax(logits["transition_rows"], dim=-1)
    fitted_emission = torch.softmax(logits["emission_rows"], dim=-1)
order = [0, 1] if float(fitted_emission[0, 0]) > float(fitted_emission[1, 0]) else [1, 0]
fitted_transition = fitted_transition[order][:, order]
fitted_emission = fitted_emission[order]
print("transition:", [[round(v, 3) for v in row] for row in fitted_transition.tolist()])
print("emission:  ", [[round(v, 3) for v in row] for row in fitted_emission.tolist()])
assert float((fitted_emission - true_emission_rows).abs().max()) < 0.12
assert float((fitted_transition - true_transition_rows).abs().max()) < 0.10
```

## Categorical perspective

Before scoring a symbol, the transition is a probability [kernel](https://en.wikipedia.org/wiki/Markov_kernel) from `State` to `State`. Multiplying its entries by the observed symbol's probability produces a weighted transition matrix, which need not have row sums one. With $\pi$ a row vector, the forward calculation is $\pi\prod_t A\,\mathrm{diag}(B_{\cdot o_t})$, followed by a sum over the final state.

This is matrix composition in the sum-product [semiring](https://ncatlab.org/nlab/show/rig), whose two operations combine successive transitions and alternative paths. The marginalized chain expresses this calculation as a program. In log space, replacing log-sum-exp over alternative paths by maximum gives the Viterbi recursion [@rabiner1989hmm].
