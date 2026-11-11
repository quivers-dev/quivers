# Probabilistic Context-Free Grammar

## Overview

A [probabilistic context-free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar) as an agenda-based weighted deduction over [CKY](https://en.wikipedia.org/wiki/CYK_algorithm) chart items `span(I, J, N)` under the `LogProb` semiring.

## QVR source

```qvr
# Probabilistic Context-Free Grammar
#
# A PCFG as an agenda-based weighted deduction over CKY chart
# items ``span(I, J, N)`` (token range ``[I, J)`` with
# nonterminal ``N``) under the LogProb semiring. Each production
# is a specific ``rule`` (no free wildcards on the conclusion):
# the chart fires only the productions the grammar licenses, each
# ``#[learnable]`` weight is keyed by the rule's category bindings, and the
# chart's goal weight at ``span(0, n, S)`` is the corpus inside
# log-probability.

object Term : FinSet 16
object Rule : FinSet 9

object Weight : Real 1

# Probabilistic model for transpilation: each learnable rule weight
# carries an independent Normal(0, 1) prior, and a treebank reports
# how often each rule fired. Exponentiating a weight gives that
# rule's firing rate, so the counts are Poisson in the rate; the
# chart parser downstream consumes the same weights as its per-rule
# log-probabilities. Rule indexes the weight vector, so it is the
# plate extent; the codomain Weight is the value space of the one
# real number a single weight is.
program pcfg_prior : Rule -> Weight
    sample rule_weights : Rule <- Normal(loc=0.0, scale=1.0)
    let rule_rate = exp(rule_weights)
    observe rule_counts : Rule <- Poisson(rate=rule_rate)
    return rule_weights

export pcfg_prior

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4]
    atoms S, NP, VP, Det, N, V, span, the, a, cat, dog, sleeps, runs
    # Concrete binary productions. Each ``#[learnable]`` allocates
    # one log-weight per binding of its category variables; these
    # productions bind only span positions, so each has one weight.
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule vp_v : span(I, J, V) |- span(I, J, VP) #[learnable]
    lexicon
        "the"    : Det = the     #[learnable]
        "a"      : Det = a       #[learnable]
        "cat"    : N   = cat     #[learnable]
        "dog"    : N   = dog     #[learnable]
        "sleeps" : V   = sleeps  #[learnable]
        "runs"   : V   = runs    #[learnable]
```

## Understanding the model

For `the dog sleeps`, a parser can combine `the` and `dog` into a noun
phrase, turn `sleeps` into a verb phrase, then combine those two phrases into
a sentence. A [chart](../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores)
saves each partial result at the corresponding [substring interval](https://www.nltk.org/book/ch08.html#well-formed-substring-tables)
[@nltkSyntax2019]. If several trees produce the sentence, the inside
computation sums their weights.

The source learns unconstrained log weights. A normalized PCFG would require
production probabilities that sum to one for each left-hand side, plus conditions
ensuring total probability one over finite derivations. The separate rule-count
program fits [Poisson observations](../tutorials/qvr/02-glms.md#poisson-regression) on rule rates.

## Walkthrough

`object Term : FinSet 16` declares the index set the deduction's domain and codomain range over; the cardinality is incidental, since the chart reasons symbolically over constructor-tagged tuples rather than over enumerated elements. `object Rule : FinSet 9` indexes the rule-weight vector instead: one coordinate per learnable weight, three rules plus six lexicon entries. The `pcfg_prior` program draws that vector from an independent `Normal(0.0, 1.0)` per coordinate, so `Rule` names the axis the weights live on, not the values they take; those come from the family. The program returns one real-valued weight, so its codomain is `object Weight : Real 1`. The count model uses each exponentiated weight as a Poisson rate and observes one `rule_counts` value per rule.

The `atoms` block enumerates the constructor universe: nonterminals (`S`, `NP`, `VP`), preterminals (`Det`, `N`, `V`), the chart-item constructor `span`, and the closed-class terminal vocabulary (`the`, `a`, `cat`, `dog`, `sleeps`, `runs`).

The three concrete branching rules declare the grammar's productions one by one. `s_np_vp` combines an `NP` span and an adjacent `VP` span into an `S` span; `np_det_n` combines a `Det` and an `N` into an `NP`; `vp_v` lifts a `V` span to a `VP`. Because each conclusion has no free wildcard, `#[learnable]` allocates one log-weight per rule firing, giving exactly one production probability per rule. Pattern variables `I`, `J`, `K` are single-uppercase identifiers ranging over token positions.

The `lexicon` block ships one entry per closed-class terminal; each `"word" : Cat = lf #[learnable]` line becomes a unit-width `span(I, I+1, Cat)` axiom with the lexical entry's logical form whenever the input token at position `I` matches `"word"`. The semiring is `LogProb`, so inside probabilities accumulate in log space; gradients flow back through the agenda's semiring operations to the learnable weights.

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/pcfg.qvr")
```

To fit per-rule probabilities, wrap the deduction in a [`program`](../guides/dsl-overview.md) that draws lexicon weights from a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior per preterminal and observes the chart's `chart.goal_weight()` against a corpus of sentences. The `#[learnable]` markers on each lexicon entry expose `nn.Parameter`s the optimizer adjusts; an [`AutoNormalGuide`](../api/inference/guide.md) plus [`SVI`](../api/inference/svi.md) over an [`ELBO`](../api/inference/elbo.md) objective drives the fit. For Viterbi or counting semirings the chart returns a single max-derivation score or a derivation count, which can be conditioned in the same way.

### Generating synthetic data

The `pcfg_prior` program is a standalone Bayesian model over the same
rule weights. Each rule draws one log-weight from a unit Normal;
exponentiating that weight gives the rate at which the rule fires, and
a treebank reports the count. Drawing the weights from their own prior
and the counts from those weights keeps the synthetic point
self-consistent, so a fit has a ground truth to recover.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/pcfg.qvr")
model = prog.morphism

N_RULES = 9
true_rule_weights = torch.randn(N_RULES)
rule_counts = torch.poisson(torch.exp(true_rule_weights))

observations = {"rule_counts": rule_counts}
x_in = torch.zeros(1, 1)
```

## Categorical perspective

A chart assigns a weight to each item, here a `span(I, J, N)` triple saying that nonterminal `N` covers a substring. A rule combines the weights of its premises; alternative derivations of the same item are combined by the semiring's addition. In `LogProb`, this means adding log weights along a derivation and using log-sum-exp across alternatives. Thus an inside weight accounts for all derivations, whereas a best-derivation weight requires a different semiring [@goodman1999semiring].

Rules can be drawn as directed hyperedges: several premise items lead to one conclusion item. This representation separates the grammar from the algorithm that evaluates it. Equivalent evaluation schedules must still satisfy their algorithms' assumptions, including termination or convergence and correct treatment of repeated contributions. CKY, A\*, and Knuth's algorithm are not interchangeable for every semiring and rule system. The compiler selects a default strategy using the rule arities and semiring properties.

## Inside computation

The classic runtime computes chart values through the agenda and supports
autodiff through those operations. It exposes no analytic outside pass;
gradients come from autodiff over the inside computation.

The same declaration also generates a QIEC item family, recursive derivation,
`Search`, `Weight[LogWeight]`, and `Param` instances, collecting handlers, and
a `PCFG__run` entry. `run_deduction(program.qiec, "PCFG", tokens=...)`
evaluates that checked search on the reference machine. Given the same
parameter store and depth bound, its goal weight equals the agenda chart's
inside weight. The [parsing tutorial](../tutorials/qvr/11-parsing-and-search.md)
shows the comparison and the target-capability boundary.

## Connections to Language Modeling

When the grammar licenses a sentence, `chart.goal_weight()` is its inside score under the chosen semiring. Changing `semiring` selects a different aggregation strategy without changing the rules.
