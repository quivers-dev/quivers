# Weighted Combinatory Categorial Grammar

## QVR source

```qvr
# Weighted Combinatory Categorial Grammar
#
# A learnable weighted CCG parser declared as an agenda-based
# weighted deduction over chart items. Categories carry slash
# constructors Fwd(X, Y) = X/Y and Bwd(X, Y) = X\Y; chart items
# are span(I, J, X) triples; the semiring is LogProb for
# differentiable inside scores.
#
# Deduction:
#
#   fwd_app    : X/Y, Y       |- X           forward application
#   bwd_app    : Y,   X\Y     |- X           backward application
#   fwd_comp   : X/Y, Y/Z     |- X/Z         forward composition
#   bwd_comp   : Y\Z, X\Y     |- X\Z         backward composition
#   fwd_xcomp  : X/Y, Y\Z     |- X\Z         forward crossed composition
#   bwd_xcomp  : Y/Z, X\Y     |- X/Z         backward crossed composition
#
# CCG is the internal language of a closed monoidal category:
# the slashes are internal homs, application is the counit of
# the hom-tensor adjunction, and composition is composition of
# internal hom morphisms.

object Term : FinSet 16

object Rule : FinSet 16

object Weight : Real 1

# Probabilistic model for transpilation: each learnable rule weight
# carries an independent Normal(0, 1) prior, and a treebank reports
# how often each rule fired. Exponentiating a weight gives that
# rule's firing rate, so the counts are Poisson in the rate; the
# chart parser downstream consumes the same weights as its per-rule
# log-probabilities. Rule indexes the weight vector, so it is the
# plate extent; the codomain Weight is the value space of the one
# real number a single weight is.
program ccg_prior : Rule -> Weight
    sample rule_weights : Rule <- Normal(loc=0.0, scale=1.0)
    let rule_rate = exp(rule_weights)
    observe rule_counts : Rule <- Poisson(rate=rule_rate)
    return rule_weights

export ccg_prior

deduction CCG : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms NP, S, N, VP, PP, Fwd, Bwd, span, the, cat, sleeps, barks
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    rule fwd_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z)) #[learnable]
    rule bwd_comp : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Bwd(X, Z)) #[learnable]
    rule fwd_xcomp : span(I, K, Fwd(X, Y)), span(K, J, Bwd(Y, Z)) |- span(I, J, Bwd(X, Z)) #[learnable]
    rule bwd_xcomp : span(I, K, Fwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Fwd(X, Z)) #[learnable]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "cat" : N = cat #[learnable]
        "sleeps" : Bwd(S, NP) = sleeps #[learnable]
        "barks" : Bwd(S, NP) = barks #[learnable]
```

## Overview

Combinatory Categorial Grammar (CCG) is expressed as an agenda-based weighted deduction whose items are chart spans `span(I, J, X)` (token range `[I, J)` carrying category `X`). The structural combinators of CCG, forward and backward application, harmonic composition, and crossed composition, each become one sequent rule. The semiring is `LogProb`, so inside scores flow as differentiable tensors back to whatever axiom / rule weights the user marks `learnable`.

## Understanding the model

A category describes what an expression can combine with. Read `NP/N` as
an expression that takes a noun on its right and returns a noun phrase:
`the` with that category can combine with `dog : N` to form `the dog : NP`.
A verb seeking a noun phrase on its left can then complete the sentence.
The [chart](pcfg.md#understanding-the-model) stores these partial results by their token intervals.

Rule weights score derivations; alternatives leading to the same goal are
combined by the chosen [semiring](../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine). Under `LogProb`, weights add along a
single derivation and `logsumexp` combines alternatives. Learnable log weights
are not automatically normalized production or sentence probabilities.
The separate Poisson rule-count program illustrates a Bayesian model of
counts and should not be confused with the chart's sentence-scoring path.

## Walkthrough

`object Term : FinSet 16` declares a finite carrier for chart items; the concrete cardinality is irrelevant because the deduction reasons symbolically over constructor-tagged tuples, not over enumerated elements of `Term`. `object Rule : FinSet 16` plays a different part: it indexes the rule-weight vector, and the `ccg_prior` program draws one `Normal(0.0, 1.0)` coordinate per slot of that index. Here `Rule` indexes the coordinates; the `Normal` family makes each coordinate real-valued. The program returns one real-valued weight, so its codomain is `object Weight : Real 1`. The count model uses each exponentiated weight as a Poisson rate and observes one `rule_counts` value per rule.

`atoms NAME, NAME, ...` lists every identifier the rules may match literally, category atoms (`NP`, `S`, `N`, `VP`, `PP`), slash constructors (`Fwd`, `Bwd`), and the chart-item constructor (`span`). Identifiers not listed here that appear in a rule pattern are bound as wildcards; the convention is single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`).

Each `rule` is a sequent: premises on the left of `|-`, conclusion on the right. `Fwd(X, Y)` constructs the forward-slash category `X/Y`; `Bwd(X, Y)` constructs the backward-slash category `X\Y`. Adjacent spans whose end / start indices agree fire whichever rule's pattern matches their categories.

The header's option block sets the remaining knobs: the `semiring=LogProb` option selects log-space inside scores, the `start=S` option declares the goal category for a successful parse, and the `depth=6` option bounds derivation depth to keep the agenda finite.

## DSL features

- **`deduction Name : Dom -> Cod [options]` header plus indented body**: declares the agenda-based weighted deduction in a single construct. The header's option block sets the semiring, the start symbol, and the depth bound; the body supplies the item algebra (via `atoms`), the rule set, and the lexicon that serves as the axiom source.
- **`atoms NAME, NAME, ...`**: closes the constructor universe. Identifiers listed here match literally in rule patterns; any identifier not listed is bound as a wildcard, with single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`) as the convention.
- **Sequent rules**: arbitrary-arity premises on the left of `|-`, single conclusion on the right; rules with one premise are unary chart rules, with two are binary, and so on.
- **Slash constructors**: `Fwd(X, Y)` and `Bwd(X, Y)` are user-declared atoms, not built-in syntax. The combinators are theorems in this presentation.

## Try it

Every `#[learnable]` lexicon entry and every `#[learnable]` rule
exposes a real `nn.Parameter` on the compiled
[`DeductionSystem`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem). The
system is callable: `ded(sentence)` returns a
[`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose
`goal_weight()` is the differentiable log-marginal
$\log Z(s; \mathbf{w}) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$
summed over every derivation $d$ that the start symbol licenses for
the input. Fitting the lexicon and rule weights together is then a
regression-style problem: minimise $-\sum_n \log Z(s_n)$ over a
corpus of sentences. The
[`quivers.stochastic.deduction`](../api/stochastic/deduction.md) module provides
the fitting utilities used below.

### Generating synthetic data

The `ccg_prior` program is a standalone Bayesian model over the same
rule weights. Each rule draws one log-weight from a unit Normal;
exponentiating that weight gives the rate at which the rule fires, and
a treebank reports the count. Drawing the weights from their own prior
and the counts from those weights keeps the synthetic point
self-consistent, so a fit has a ground truth to recover.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ccg.qvr")
model = prog.morphism

N_RULES = 16
true_rule_weights = torch.randn(N_RULES)
rule_counts = torch.poisson(torch.exp(true_rule_weights))

observations = {"rule_counts": rule_counts}
x_in = torch.zeros(N_RULES, 1)
```

### MAP fit (Adam on rule & lexicon weights)

```python
from collections import Counter

import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction, sample_corpus

torch.manual_seed(0)
prog = load("docs/examples/source/ccg.qvr")
ded  = prog.deductions["CCG"]

corpus = [["the", "cat", "sleeps"], ["the", "cat", "barks"]]

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

# Forward-sample under the fitted parameters and check the
# dominant length-3 yield recovers the training corpus.
draws = sample_corpus(ded, length=3, n_samples=32, seed=0)
print("dominant yield:", Counter(map(tuple, draws)).most_common(1)[0][0])
# → ('the', 'cat', 'barks')
```

`adam_fit_deduction` maximizes the corpus log-marginal under an
optional Normal prior on the parameters; `prior_scale=1.0` gives MAP
under a unit Normal. `sample_corpus` enumerates yields of the chosen
length and draws from the categorical defined by their chart weights; exact forward sampling because the chart marginalizes the
derivation forest.

### NUTS (full Bayesian posterior)

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import nuts_program_from_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/ccg.qvr")
ded  = prog.deductions["CCG"]

corpus = [["the", "cat", "sleeps"], ["the", "cat", "barks"]]

model, x, observations = nuts_program_from_deduction(
    ded, corpus, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.1, max_tree_depth=4, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
posterior_means = {
    name: float(samples.mean()) for name, samples in result.samples.items()
}
print("posterior mean log-weights:", posterior_means)
```

`nuts_program_from_deduction` lifts every learnable parameter of the
deduction into a [`Normal(0, σ)`](../api/continuous/families.md) sample
site and adds the corpus log-marginal $\log Z$ to the joint via a
[`score`](../api/program.md) step. The standard
[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) drives the
posterior $p(\mathbf{w} \mid s_1, \ldots, s_N) \propto p(\mathbf{w})
\cdot \prod_n Z(s_n; \mathbf{w})$. The same Bayesian object
[`bayesian_regression`](bayesian-regression.md) fits, with the chart
total in place of the Gaussian likelihood.

## Categorical perspective

The slash categories support an internal-hom reading for the application and harmonic-composition rules. Crossed composition is an additional CCG combinator; describing it as a braiding requires a specified categorical model that this deduction does not provide. Operationally, the page defines the six sequent rules shown in the source and lets the chart combine matching adjacent spans.

## Semiring selection

The choice of semiring affects the parser's behavior: `LogProb` accumulates inside log-probabilities (numerically stable, differentiable); `Viterbi` returns the highest-weight derivation; `Counting` counts distinct derivations; `Boolean` checks membership without weights. The same deduction block serves all four objectives via the `semiring` option.
