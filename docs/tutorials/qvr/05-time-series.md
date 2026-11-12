# 5. Sequence models

QVR has three constructs for sequence-shaped problems such as time series, text, recurrent networks, hidden Markov models, and state-space models:

- **Plate-draws** for IID-along-an-index data (chapter 3 used this for `sample theta : School <- Normal(...)`).
- **`scan`** for sequential evaluation: a per-step cell function fold-applied along the sequence dimension.
- **The deduction layer** for chart-shaped problems whose computation is not a
  simple left-to-right scan, such as CKY, Earley, or another bounded weighted
  search. [Chapter 11](11-parsing-and-search.md) introduces that layer, and the
  deduction guide describes its complete API.

This chapter covers the first two, then distinguishes the finite-state path
demonstration in the gallery from a normalized hidden Markov model (HMM),
which a `scan` over a cell that marginalizes its state expresses.

A repeated measurement axis and a recurrent state do different jobs. A plate
repeats a distribution across indexed positions, conditionally independently
given its parameters. A recurrence passes the realized state at one position
into the next transition. A state might store the current temperature, while
an emission describes the noisy sensor reading from that temperature.

Read every sequence example by asking what is threaded, what is observed,
and what is returned. In the current `scan` interface, the returned value is
the terminal state. Following it with an emission scores or generates from
that state only. To score an observation at every position, put the
`observe` inside a program cell: the cell runs once per position, and the
program that draws the scan takes the whole observation sequence.

## Plates revisited

A *plate-draw* binds one value per index of a finite-set object:

<!-- compile: false -->
```qvr
object School : FinSet 8
sample theta : School <- Normal(loc=mu, scale=tau)
```

is the QVR analogue of NumPyro's `with plate("schools", 8): theta = sample("theta", dist.Normal(mu, tau))`. The result has shape `(8,)`; subsequent `let` arithmetic broadcasts over it. A *vectorized observe* over a plate has the same shape:

<!-- compile: false -->
```qvr
observe y : School <- Normal(loc=theta, scale=sigma_j)
```

A plate represents [conditionally independent draws](03-hierarchical.md#eight-schools-centered) across an index. Use `scan` when step `j+1` takes the state produced at step `j` as an input.

## `scan` for left-to-right evaluation

`scan` takes a *cell* whose signature is `Input * Hidden -> Hidden` and lifts it to operate along the sequence dimension of an input tensor. The `*` in `Input * Hidden -> Hidden` is "the cell takes two inputs in parallel (an input vector and the previous hidden state) and returns one". That's all you need to read it; the categorical reading (binary product in a monoidal category) is in chapter 7. The cell may be a `kernel ... ~ Family` morphism (one-step state update under a Gaussian transition) or a `program` block (one-step update with its own random draws).

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

The pipeline reads as: embed each token to a 64-dim vector, fold the cell over the sequence dimension (with zero or learned initial state), project the final hidden state to a 64-dim output. The compiler infers the sequence axis from the input tensor's second dimension at evaluation time.

If you've used Haskell's `mapAccumL` or NumPy's `np.cumsum`, this is the same idea generalized to a learnable cell.

## Finite-state path composition

An HMM [@rabiner1989hmm] factors into an
initial distribution, a row-stochastic transition kernel, and a row-stochastic
emission kernel. The similarly named gallery file is deliberately narrower.
Its first export demonstrates repeated composition of fuzzy relations:

```qvr
composition product_fuzzy [level=algebra]
object State : FinSet 8
object Obs : FinSet 16
morphism initial : State -> State [role=latent]
morphism transition : State -> State [role=latent]
morphism emission : State -> Obs [role=latent]
define n_step = repeat(transition) >> emission
define hmm    = initial >> n_step

export hmm
```

Two semantic facts keep this export from being an HMM likelihood:

- The three declarations have no `Dirichlet` families, and the module uses `product_fuzzy`. Their learned tensors are sigmoid-constrained fuzzy relations, not row-stochastic kernels.
- `repeat(transition)` composes that relation a runtime-selected number of times before the final emission. Under `product_fuzzy`, its result is neither a sum-product forward probability nor a Viterbi score.

### Axis-role syntax in 60 seconds

The clause `~ Family(args) over <axis> [iid over <axis>]` attaches a prior to a kernel `A -> B`. The axes name *which* dimensions of the kernel tensor the family describes and *which* dimensions are replicated independently.

- `over cod` puts the family on the codomain axis. For `transition : State -> State` declared as `~ Dirichlet(concentration=1.0) over cod`, each draw is a simplex over the 8 codomain states.
- `iid over dom` says the prior is replicated independently across the domain axis. So `over cod iid over dom` gives 8 independent simplex draws, one per domain row, which is exactly a row-stochastic matrix with a row-wise Dirichlet prior.
- Order matters because the family is built outward: first you say what one row looks like (`over cod`), then you say how rows replicate (`iid over dom`).

Other useful combinations:

| Pattern | Meaning |
|---|---|
| `~ Normal(loc=0, scale=1) over cod` | one Gaussian vector over the codomain (no replication) |
| `~ Normal(loc=0, scale=1) over cod iid over dom` | per-row independent Gaussian vectors |
| `~ Normal(loc=0, scale=1) iid over cod iid over dom` | full IID matrix of scalars |
| `~ Dirichlet(concentration=0.1) over cod iid over dom` | sparse row-stochastic prior (concentration < 1) |

If neither `over` nor `iid over` is given, the family is broadcast scalar-wise. The compiler synthesizes the appropriate `PlateDraw` internally; you don't need to think about plates.

The separate `hmm_program` demonstrates the parameter shapes a probabilistic
version would use: one Dirichlet initial row, one transition row per source
state, and one emission row per state. Its checked likelihood marginalizes a
single state shared by the observed emission sequence. `transition_rows` is a
prior-only draw. Consequently, the observation likelihood depends on one
shared hidden state, and the data leave `transition_rows` at its prior.
This program implements a finite-state marginal mixture.

Use the file to study relation powering, axis roles, and one-state finite
marginalization. A recurrent HMM carries a categorical state from position to
position, which the next section writes with `scan`.

## Hidden Markov models

A program cell may marginalize the next state and return it. Returning the
latent of a `marginalize` block carries it to the next position, so its sum
extends over the rest of the chain, and `scan(cell, init=p)` sums every path
of states out with the forward algorithm:

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

The chain starts from `z_0 ~ start`, moves by `transition`, and emits each
symbol from the state it moves to. The block observes the cell's entry, so
the scanned sequence is the observed symbols. The program's log joint is the
three Dirichlet priors plus the log marginal likelihood, which a forward
recursion written in torch reproduces:

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

The recursion stays in log space, so a chain of thousands of positions, whose
path probabilities underflow, still has a finite log marginal. A trace
records the marginal under `path` and no per-position state. The cell's lets
may read the entry, so transitions can depend on a covariate, and an
observation named other than the entry is one observation per position. The
[marginalized chains reference](../../reference/qvr/probabilistic-programs.md#marginalized-chains)
states the rules.

## State-space models

For continuous-state sequences, the per-step transition and emission can be Gaussian kernels. The following state-space demonstration appears in `docs/examples/source/linear_gaussian_ssm.qvr`:

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

`scan(transition_cell)` folds the per-step Gaussian transition along the
sequence dimension and returns the final state. Composing with `emission`
thus emits from that final state; it does not return the full latent
trajectory. The export named `filter` similarly folds a separately learned
Gaussian kernel. It is not the closed-form Kalman filter
[@kalman1960filtering], and `ScanMorphism.log_prob`
does not supply the marginal density of the final state: it scores a supplied
full trajectory, as `rsample_trajectory` draws it, and refuses a final state
alone.

A state-space model that scores the whole observation sequence writes the
emission inside a program cell:

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

Each position draws a latent state, addressed `s@0`, `s@1`, and so on, and
scores that position's observation. The program takes the observations as
one sequence `o`, shaped `(batch, T, 2)` for a batch of driver sequences of
shape `(batch, T, 2)`, and its log joint sums the transition and emission
densities over the positions. A trace records each position's state and
observation under `s@t` and `o@t`. The cell may also read bindings that
`ssm` makes before the scan, such as a sampled noise scale. The
[scans reference](../../reference/qvr/probabilistic-programs.md#scans) states
the scoping and addressing rules.

For a fully nonlinear (deep) variant where transition and emission are neural Gaussians, see `docs/examples/source/continuous_hmm.qvr` and `docs/examples/source/deep_markov.qvr`.

## Try this

- Change the fuzzy composition algebra and compare the resulting path score.
  This exercise changes relation semantics; it still does not create a
  normalized HMM likelihood.
- Make the linear-Gaussian SSM hierarchical: each sequence has its own transition cell drawn from a hyperprior. (Chapter 3's plate-draw applies.)
- Swap the linear `transition_cell` for the deep-Markov nonlinear kernel and compare ELBO convergence.

## Next

[Chapter 6](06-inference-zoo.md) surveys the inference algorithms: eleven concrete guide classes, seven objectives, HMC and NUTS, and two hybrid approaches.


## Sources and further reading

- *A tutorial on hidden Markov models and selected applications in speech recognition* [@rabiner1989hmm].
- *A New Approach to Linear Filtering and Prediction Problems* [@kalman1960filtering].
