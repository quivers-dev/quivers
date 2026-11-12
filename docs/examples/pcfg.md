# Probabilistic Context-Free Grammar

## Overview

A [probabilistic context-free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar) (PCFG) assigns each production a probability, so that the productions sharing a left-hand side form a distribution, and scores a sentence by the total probability of its derivations. This source states a small PCFG as a [weighted deduction](../guides/deduction.md) over [CKY](https://en.wikipedia.org/wiki/CYK_algorithm) chart items `span(I, J, N)` under the `LogProb` semiring. `normalize=lhs` passes each left-hand side's learned log-weights through a softmax, so the grammar is a proper PCFG and the chart's goal weight is the sentence's log probability. The page checks the chart against a CKY recursion written by hand, fits the production probabilities by maximum likelihood, and, with a Dirichlet prior on each left-hand side, checks the Bayesian program's joint on both execution paths and its NUTS posterior against the exact conjugate posterior.

## QVR source

```qvr
# Probabilistic Context-Free Grammar
#
# A PCFG as a weighted deduction over CKY chart items ``span(I, J, N)``
# (token range ``[I, J)`` with nonterminal ``N``) under the LogProb
# semiring. ``normalize=lhs`` makes the productions sharing a left-hand
# side a probability distribution: each group's learned log-weights pass
# through a softmax, so the grammar is a proper PCFG, maximum likelihood
# is bounded, and the chart's goal weight at ``span(0, n, S)`` is the
# sentence's log probability.

object Term : FinSet 16

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, Det, N, V, span, the, a, cat, dog, sleeps, runs
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

For `the dog sleeps`, the chart combines `the` and `dog` into a noun phrase, lifts `sleeps` to a verb phrase, and combines the two into a sentence. A [chart](../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) saves each partial result at its [substring interval](https://www.nltk.org/book/ch08.html#well-formed-substring-tables) [@nltkSyntax2019], and when several derivations produce the same item the semiring's addition, here log-sum-exp, combines them [@goodman1999semiring]. The goal weight at `span(0, n, S)` is thus the inside log probability $\log P(S \Rightarrow^* w_1 \cdots w_n)$.

With `normalize=lhs`, the productions are grouped by left-hand side: `S`, `NP`, and `VP` have one production each, which therefore has probability one, while `Det`, `N`, and `V` each choose between two words. The probabilities within a group are the softmax of that group's learned log-weights, so each group sums to one by construction, and the likelihood of a corpus is bounded above, which is what makes maximum-likelihood fitting well posed. Normalizing every group gives a distribution over derivations only when the grammar is consistent, its derivations terminating with probability one [@booth1973applying]; this grammar has no recursive production, so every derivation terminates.

## Walkthrough

- `object Term : FinSet 16` is the index set the deduction's domain and codomain range over; the chart reasons over constructor-tagged items, not over its elements.
- The `atoms` block lists the constructors: nonterminals (`S`, `NP`, `VP`), preterminals (`Det`, `N`, `V`), the chart-item constructor `span`, and the six words.
- `s_np_vp`, `np_det_n`, and `vp_v` are the three non-lexical productions. Each conclusion has no free category variable, so `#[learnable]` allocates one log-weight per production. `I`, `J`, and `K` range over token positions.
- Each `lexicon` line `"word" : Cat = lf #[learnable]` is a unit-width axiom `span(I, I+1, Cat)` wherever the token at position `I` is `word`, with one learned log-weight.
- `normalize=lhs` turns the log-weights into per-left-hand-side log probabilities, which [`DeductionSystem.production_log_probs`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem.production_log_probs) reports.

## Try it

The checks below come from verification scripts whose tolerances were fixed before they were first run.

### An independent grammar and a CKY oracle

The generating grammar is written out by hand and shares no code with quivers: `sample_corpus` draws sentences top-down and counts the productions each derivation uses, and `cky_log_inside` computes $\log P(S \Rightarrow^* w)$ by the CKY recursion with unary closure, in float64.

```python
import math
import random
from collections import Counter

#: The generating grammar: each left-hand side's productions as
#: (production label, right-hand side, probability). Labels follow
#: ``DeductionSystem.production_log_probs``: rules by name, words quoted.
TRUE_GRAMMAR: dict[str, list[tuple[str, tuple[str, ...], float]]] = {
    "S": [("s_np_vp", ("NP", "VP"), 1.0)],
    "NP": [("np_det_n", ("Det", "N"), 1.0)],
    "VP": [("vp_v", ("V",), 1.0)],
    "Det": [('"the"', ("the",), 0.7), ('"a"', ("a",), 0.3)],
    "N": [('"cat"', ("cat",), 0.4), ('"dog"', ("dog",), 0.6)],
    "V": [('"sleeps"', ("sleeps",), 0.25), ('"runs"', ("runs",), 0.75)],
}


def _expand(rng: random.Random, symbol: str, counts: Counter) -> list[str]:
    productions = TRUE_GRAMMAR[symbol]
    label, rhs, _ = rng.choices(productions, weights=[p for *_, p in productions])[0]
    counts[(symbol, label)] += 1
    if label.startswith('"'):
        return list(rhs)
    return [word for child in rhs for word in _expand(rng, child, counts)]


def sample_corpus(n: int, seed: int) -> tuple[list[list[str]], Counter]:
    """Draw ``n`` sentences and the production counts of their derivations."""
    rng = random.Random(seed)
    counts: Counter = Counter()
    return [_expand(rng, "S", counts) for _ in range(n)], counts


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def cky_log_inside(
    tokens: list[str], probs: dict[str, dict[str, float]], start: str = "S"
) -> float:
    """log P(start =>* tokens) under per-left-hand-side probabilities."""
    binary = []
    unary = []
    lexical = []
    for lhs, productions in TRUE_GRAMMAR.items():
        for label, rhs, _ in productions:
            logp = math.log(probs[lhs][label])
            if label.startswith('"'):
                lexical.append((lhs, rhs[0], logp))
            elif len(rhs) == 2:
                binary.append((lhs, rhs[0], rhs[1], logp))
            else:
                unary.append((lhs, rhs[0], logp))

    def close(cell: dict[str, float]) -> dict[str, float]:
        closed = dict(cell)
        for _ in range(len(unary) + 1):
            updated = dict(cell)
            for lhs, child, logp in unary:
                if child in closed:
                    updated[lhs] = _logaddexp(updated.get(lhs, -math.inf), logp + closed[child])
            closed = updated
        return closed

    n = len(tokens)
    chart: dict[tuple[int, int], dict[str, float]] = {}
    for i, token in enumerate(tokens):
        cell: dict[str, float] = {}
        for lhs, word, logp in lexical:
            if word == token:
                cell[lhs] = _logaddexp(cell.get(lhs, -math.inf), logp)
        chart[(i, i + 1)] = close(cell)
    for width in range(2, n + 1):
        for i in range(n - width + 1):
            j = i + width
            cell = {}
            for k in range(i + 1, j):
                for lhs, left, right, logp in binary:
                    a = chart[(i, k)].get(left)
                    b = chart[(k, j)].get(right)
                    if a is not None and b is not None:
                        cell[lhs] = _logaddexp(cell.get(lhs, -math.inf), logp + a + b)
            chart[(i, j)] = close(cell)
    return chart[(0, n)].get(start, -math.inf)
```

### Maximum-likelihood fitting

A corpus of 300 sentences is sampled from the generating grammar and the grammar is fitted by [`adam_fit_deduction`](../api/stochastic/deduction/fit.md#quivers.stochastic.deduction.fit.adam_fit_deduction) (300 steps, learning rate 0.1). Every sentence of this grammar has one derivation, so the maximum-likelihood estimate of each production probability is its relative frequency among the derivations' uses of its left-hand side, and the corpus negative log likelihood at those frequencies is a lower bound the loss can reach. The checks: the final loss is within 0.02 nats of that bound; each left-hand side's probabilities sum to one within 1e-6; each fitted probability is within 0.005 of its relative frequency and within four binomial standard errors of the generating probability; and the chart's goal weight at the fitted probabilities equals the CKY oracle within 1e-4 on ten sentences.

```python
from collections import Counter

from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction

corpus, counts = sample_corpus(300, seed=2026)
uses = Counter()
for (lhs, _), c in counts.items():
    uses[lhs] += c

grammar = load("docs/examples/source/pcfg.qvr").deductions["PCFG"]
history = adam_fit_deduction(grammar, corpus, steps=300, lr=0.1)
relative = {
    lhs: {label: counts[(lhs, label)] / uses[lhs] for label, _, _ in prods}
    for lhs, prods in TRUE_GRAMMAR.items()
}
bound = -sum(c * math.log(relative[lhs][label]) for (lhs, label), c in counts.items())
print(f"final loss {history[-1]:.4f}, bound at the relative frequencies {bound:.4f}")
assert abs(history[-1] - bound) < 2e-2

table = {
    lhs: {label: math.exp(float(w.detach())) for label, w in group.items()}
    for lhs, group in grammar.production_log_probs().items()
}
for lhs, group in table.items():
    assert abs(sum(group.values()) - 1.0) < 1e-6, lhs
for lhs, prods in TRUE_GRAMMAR.items():
    for label, _, truth in prods:
        fitted = table[lhs][label]
        tolerance = 4 * math.sqrt(truth * (1 - truth) / uses[lhs]) if 0 < truth < 1 else 1e-6
        print(f"{lhs:>3} -> {label:<10} truth {truth:.3f} relative {relative[lhs][label]:.3f} fitted {fitted:.3f}")
        assert abs(fitted - relative[lhs][label]) < 5e-3
        assert abs(fitted - truth) <= tolerance

for tokens in corpus[:10]:
    chart = float(grammar(tokens).goal_weight().detach())
    assert abs(chart - cky_log_inside(tokens, table)) < 1e-4, tokens
```

### A Bayesian PCFG

A program can place a [prior on the production probabilities](../guides/deduction.md#priors-on-production-probabilities) of a normalized grammar and infer their posterior, the Bayesian PCFG of [@johnson2007bayesian]. The source below is the same grammar without learned weights; the program draws each two-production left-hand side's probabilities from `Dirichlet(1, 1)` and binds them in the `parse` call. The corpus is one input of one run, a table with one sentence per object of `Sentence`, and `chart.goal_weight()` is the sum of the sentences' log inside weights.

Because every sentence has one derivation, each left-hand side's posterior is the conjugate $\mathrm{Dirichlet}(1 + n)$, with $n$ the production counts. On 60 sampled sentences, the program's log joint at a fixed point must equal the CKY oracle (the `Dirichlet(1, 1)` density is one) to 1e-9 on the reference machine and to 1e-3 on the compiled runtime. NUTS (two chains, 200 warmup and 300 kept draws each) must then reach an effective sample size of at least 100 for every site and put every posterior mean within $4\,\mathrm{sd} / \sqrt{100}$ of the exact posterior mean, and the exact posterior mean must lie within four posterior standard deviations of the generating probabilities.

```python
import torch

from quivers.dsl import loads
from quivers.inference import MCMC, NUTSKernel
from quivers.inference.trace import trace

BAYESIAN_SOURCE = """
# Bayesian Probabilistic Context-Free Grammar
#
# The pcfg page's grammar with a Dirichlet prior on each left-hand
# side's production probabilities. ``normalize=lhs`` makes the
# grammar a proper PCFG; the program binds each left-hand side of
# several productions (Det, N, V) to a simplex-valued sample site, and
# the corpus is one input of one run: a table with one sentence per
# object of ``Sentence``. The program's log-joint is the Dirichlet log
# densities plus the corpus log-likelihood, the density a sampler of the
# posterior over the grammar targets.

object Term : FinSet 16
object Weight : Real 1
object Sentence : FinSet 60

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, Det, N, V, span, the, a, cat, dog, sleeps, runs
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S)
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP)
    rule vp_v : span(I, J, V) |- span(I, J, VP)
    lexicon
        "the"    : Det = the
        "a"      : Det = a
        "cat"    : N   = cat
        "dog"    : N   = dog
        "sleeps" : V   = sleeps
        "runs"   : V   = runs

program bayesian_pcfg : Term -> Weight
    sample p_det <- Dirichlet(concentration=[1.0, 1.0])
    sample p_n <- Dirichlet(concentration=[1.0, 1.0])
    sample p_v <- Dirichlet(concentration=[1.0, 1.0])
    let chart = parse(PCFG, corpus, Sentence, ("Det", p_det), ("N", p_n), ("V", p_v))
    score evidence = chart.goal_weight()
    return evidence

export bayesian_pcfg
"""

corpus, counts = sample_corpus(60, seed=7)
program = loads(BAYESIAN_SOURCE)

point = {"p_det": [0.6, 0.4], "p_n": [0.3, 0.7], "p_v": [0.5, 0.5]}
probs = {
    "Det": {'"the"': 0.6, '"a"': 0.4}, "N": {'"cat"': 0.3, '"dog"': 0.7},
    "V": {'"sleeps"': 0.5, '"runs"': 0.5}, "S": {"s_np_vp": 1.0},
    "NP": {"np_det_n": 1.0}, "VP": {"vp_v": 1.0},
}
expected = sum(cky_log_inside(s, probs) for s in corpus)
checked = program.run(
    "bayesian_pcfg",
    data={"corpus": tuple(map(tuple, corpus))},
    sites={k: tuple(v) for k, v in point.items()},
    seed=0,
).log_joint
compiled = float(trace(
    program.morphism,
    torch.zeros(1, 1),
    observations={"corpus": corpus, **{k: torch.tensor(v) for k, v in point.items()}},
).log_joint)
print(f"log joint: oracle {expected:.6f}, reference {checked:.6f}, compiled {compiled:.6f}")
assert abs(checked - expected) < 1e-9 and abs(compiled - expected) < 1e-3

torch.manual_seed(0)
result = MCMC(NUTSKernel(step_size=0.3, max_tree_depth=4), num_warmup=200, num_samples=300, num_chains=2).run(
    program.morphism, torch.zeros(1, 1), {"corpus": corpus}
)
for site, lhs in (("p_det", "Det"), ("p_n", "N"), ("p_v", "V")):
    labels = [label for label, _, _ in TRUE_GRAMMAR[lhs]]
    alpha = torch.tensor([1.0 + counts[(lhs, label)] for label in labels], dtype=torch.float64)
    mean = alpha / alpha.sum()
    sd = torch.sqrt(mean * (1 - mean) / (alpha.sum() + 1))
    got = result.samples[site].reshape(-1, len(labels)).double().mean(0)
    ess = float(result.ess[site].min())
    truth = torch.tensor([p for *_, p in TRUE_GRAMMAR[lhs]], dtype=torch.float64)
    print(f"{lhs}: truth {truth.tolist()}, exact posterior mean {[round(x, 3) for x in mean.tolist()]}, "
          f"NUTS {[round(x, 3) for x in got.tolist()]}, ess {ess:.0f}")
    assert ess >= 100
    assert torch.all((got - mean).abs() <= 4 * sd / math.sqrt(100))
    assert torch.all((mean - truth).abs() <= 4 * sd)
```

## Categorical perspective

A chart assigns a weight to each item, here a `span(I, J, N)` triple saying that nonterminal `N` covers a substring. A rule combines the weights of its premises by the semiring's multiplication, and alternative derivations of the same item are combined by its addition; in `LogProb` this is adding log weights along a derivation and taking log-sum-exp across alternatives, so the inside weight accounts for every derivation, whereas a best-derivation weight needs a different semiring [@goodman1999semiring]. Rules are directed hyperedges from premise items to a conclusion item, which separates the grammar from the schedule that evaluates it.

## Inside computation

The compiled runtime computes chart values through the agenda and differentiates through its semiring operations; gradients reach the learned log-weights, or, in the Bayesian program, the sampled probabilities. The same declaration also generates a checked search on the reference machine, and given the same weights and depth bound its goal weight equals the agenda chart's inside weight; the [parsing tutorial](../tutorials/qvr/11-parsing-and-search.md) shows the comparison.

## Sources and further reading

- *Semiring parsing* [@goodman1999semiring].
- *Applying probability measures to abstract languages* [@booth1973applying].
- *Bayesian inference for PCFGs via Markov chain Monte Carlo* [@johnson2007bayesian].
- *Analyzing sentence structure*, chapter 8 of *Natural Language Processing with Python* [@nltkSyntax2019].
