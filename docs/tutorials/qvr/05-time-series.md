# 5. Sequence models

QVR has three constructs for sequences:

- **Plates** for values that are conditionally independent along an index, as in [chapter 3](03-hierarchical.md).
- **`scan`** for recurrences: a cell that maps an input and the previous state to the next state is applied along the sequence axis, as Haskell's `mapAccumL` applies a function and threads an accumulator.
- **Weighted deduction** for chart parsing, where the computation is not a left-to-right pass, as in the [CKY algorithm](https://en.wikipedia.org/wiki/CYK_algorithm). [Chapter 11](11-parsing-and-search.md) covers it.

The first two express recurrent networks, [hidden Markov models](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMMs) [@rabiner1989hmm], and linear Gaussian [state space models](https://en.wikipedia.org/wiki/State-space_representation). A plate repeats a distribution across positions given its parameters, whereas a recurrence passes the state at one position into the transition at the next.

## Plates revisited

A plate binds one value for each element of a finite set:

<!-- compile: false -->
```qvr
object School : FinSet 8
sample theta : School <- Normal(loc=mu, scale=tau)
```

This is NumPyro's `with plate("schools", 8): theta = sample("theta", dist.Normal(mu, tau))`. The result has shape `(8,)`, and later `let` arithmetic broadcasts over it. A vectorized observe statement over a plate has the same shape:

<!-- compile: false -->
```qvr
observe y : School <- Normal(loc=theta, scale=sigma_j)
```

## `scan` for left-to-right evaluation

`scan` takes a *cell* with signature `Input * Hidden -> Hidden`, a map from a pair (the input at one position and the previous state) to the next state, and applies it along the sequence axis of its input. The cell may be a morphism with a family, such as `~ Normal` for a Gaussian transition, or a `program` with its own draws.

```qvr
object Token : FinSet 256
object Embedded : Real 64
object Hidden : Real 128
object Output : Real 64

morphism tok_embed : Token -> Embedded [role=embed]

morphism cell : Embedded * Hidden -> Hidden ~ Normal
morphism output_proj : Hidden -> Output ~ Normal
define rnn = tok_embed >> scan(cell) >> output_proj
export rnn
```

`tok_embed` (role `embed`) maps each token to a vector in $\mathbb R^{64}$. `scan(cell)` applies `cell` along the sequence, starting from a zero or learned initial state, and returns the final state, which `output_proj` maps to $\mathbb R^{64}$. The sequence axis is the second axis of the input tensor: an integer input of shape `(3, 5)` (three sequences of five tokens) gives an output of shape `(3, 64)`.

## Hidden Markov models

An HMM has an initial distribution over states, a row-stochastic transition matrix, and a row-stochastic emission matrix. A program cell can `marginalize` the next state and return it. Returning the latent of a `marginalize` block carries it to the next position, so its sum extends over the rest of the chain, and `scan(cell, init=p)` sums out every path of states with the [forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm):

```qvr
object State : FinSet 2
object Symbol : FinSet 4
object StartProbs : Real 2

program hmm_step(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition[z_prev])
        observe symbol <- Categorical(probs=emission[z])
    return z

define chain = scan(hmm_step, init=start)

program hmm : Symbol -> StartProbs
    sample start <- Dirichlet(concentration=1.0) [over=State]
    sample transition : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return start

export hmm
```

The chain starts from $z_0 \sim$ `start`, moves by `transition`, and emits each symbol from the state it moves to. The block observes the cell's input `symbol`, so the scanned sequence is the observed symbols. In `[over=State, iid_over=State]` on `transition`, `over` names the event axis of one Dirichlet draw (one row on the simplex over `State`) and `iid_over` the axis along which rows are drawn independently, so `transition` is a row-stochastic matrix with a Dirichlet prior on each row.

The program's log joint density is the sum of the three Dirichlet log priors and the log marginal likelihood. A forward recursion written in PyTorch reproduces it:

```python
import torch
import torch.distributions as D
from quivers.dsl import loads

source = """
object State : FinSet 2
object Symbol : FinSet 4
object StartProbs : Real 2

program hmm_step(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition[z_prev])
        observe symbol <- Categorical(probs=emission[z])
    return z

define chain = scan(hmm_step, init=start)

program hmm : Symbol -> StartProbs
    sample start <- Dirichlet(concentration=1.0) [over=State]
    sample transition : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return start

export hmm
"""
model = loads(source).morphism

start = torch.tensor([0.5, 0.5])
transition = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
emission = torch.tensor([[0.6, 0.3, 0.05, 0.05], [0.05, 0.05, 0.3, 0.6]])
symbols = torch.tensor([[0, 1, 0, 3, 3, 2, 3, 0]])

log_joint = model.log_joint(
    symbols, {"start": start, "transition": transition, "emission": emission}
)

log_alpha = torch.log(start)
for symbol in symbols[0]:
    log_alpha = torch.logsumexp(
        log_alpha[:, None] + torch.log(transition) + torch.log(emission[:, symbol]),
        dim=0,
    )
priors = (
    D.Dirichlet(torch.ones(2)).log_prob(start)
    + D.Dirichlet(torch.ones(2)).log_prob(transition).sum()
    + D.Dirichlet(torch.ones(4)).log_prob(emission).sum()
)
assert torch.allclose(log_joint, priors + torch.logsumexp(log_alpha, dim=0), atol=1e-5)
```

The recursion stays in log space, so a chain of thousands of positions, whose path probabilities underflow, still has a finite log marginal. A trace records the marginal under `path` and no state for each position. The cell's `let` steps may read its input, so transitions can depend on a covariate, and an observation with a name other than the input is one observation per position. The [marginalized chains reference](../../reference/qvr/probabilistic-programs.md#marginalized-chains) gives the rules.

## Finite-state path composition

The gallery file [`hmm.qvr`](../../examples/source/hmm.qvr) is not an HMM likelihood. Its export `hmm` composes three morphisms under the `product_fuzzy` composition rule, `initial >> repeat(transition) >> emission`. Under that rule a learnable table (`role=latent`) holds entries in $[0, 1]$, its rows are not normalized, and the composite is neither a forward probability nor a Viterbi score. Its second export, `hmm_program`, marginalizes a single state shared by the whole emitted sequence, and its `transition_rows` draw does not enter the likelihood. The `hmm` program of the previous section is the HMM.

On a `morphism` declaration the axes of a prior are set by the options `over` (event axes) and `iid` (axes of independent replication), with `cod` and `dom` naming the codomain and domain:

<!-- compile: false -->
```qvr
morphism transition : State -> State [role=latent, over=cod, iid=dom] ~ Dirichlet(concentration=1.0)
```

declares a row-stochastic matrix with an independent Dirichlet prior on each row, as `[over=State, iid_over=State]` does on a `sample` step. The [axis clause guide](../../guides/dsl-programs-and-lets.md#axis-role-clause-over-and-iid_over) and the [event shapes of each family](../../guides/continuous-families.md#event-rank-and-axis-role-syntax) give the rules.

## State space models

A linear Gaussian state space model has Gaussian transitions and Gaussian emissions. The gallery file [`linear_gaussian_ssm.qvr`](../../examples/source/linear_gaussian_ssm.qvr) declares:

```qvr
object Driver : Real 2
object State : Real 4
object Obs : Real 2

morphism transition_cell : Driver * State -> State ~ Normal
morphism emission : State -> Obs ~ Normal
morphism filter_cell : Obs * State -> State ~ Normal
define generate = scan(transition_cell) >> emission
define filter = scan(filter_cell)

export filter
```

`scan(transition_cell)` returns the final state, so `generate` emits one observation from that state, not one for each position. `filter` applies a separately learned Gaussian cell along the observations and is not the closed-form [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) [@kalman1960filtering]. [`ScanMorphism.log_prob`](../../api/continuous/scan.md) scores a full state trajectory of the kind `rsample_trajectory` draws, and refuses a final state alone.

To score the whole observation sequence, put the emission inside a program cell:

<!-- compile: qiec -->
```qvr
object Driver : Real 2
object State : Real 3
object Obs : Real 2

morphism transition : Driver * State -> State ~ Normal
morphism emission : State -> Obs ~ Normal

program step(u_t, s_prev) : Driver * State -> State
    sample s <- transition(u_t, s_prev)
    observe o <- emission(s)
    return s

define run_all = scan(step)

program ssm : Driver -> State
    sample h <- run_all
    return h

export ssm
```

Each position draws a latent state, addressed `s@0`, `s@1`, and so on, and scores that position's observation. For driver sequences of shape `(batch, T, 2)` the program takes the observations as one sequence `o` of shape `(batch, T, 2)`, and its log joint density sums the transition and emission log densities over the positions. A trace records the state and observation at position $t$ under `s@t` and `o@t`. The cell may also read bindings that `ssm` makes before the scan, such as a sampled noise scale. The [scans reference](../../reference/qvr/probabilistic-programs.md#scans) gives the scoping and addressing rules. The gallery files [`continuous_hmm.qvr`](../../examples/source/continuous_hmm.qvr) and [`deep_markov.qvr`](../../examples/source/deep_markov.qvr) replace the linear Gaussian cells with neural ones.

## Try this

- Give each of several sequences its own transition scale, drawn around a shared population scale as in chapter 3.
- Replace the linear `transition` in the state space program with a neural Gaussian cell from `deep_markov.qvr` and compare the ELBO after the same number of stochastic variational inference (SVI) steps.
- Feed the HMM a sequence of 2000 symbols and confirm that `log_joint` stays finite.

## Next

[Chapter 6](06-inference-zoo.md) chooses among the variational guides, objectives, and samplers.
