# Weighted Combinatory Categorial Grammar

## QVR source

```qvr
# Weighted Combinatory Categorial Grammar
#
# A learnable weighted CCG parser declared as an agenda-based
# weighted deduction over chart items. Categories carry slash
# constructors Fwd(X, Y) = X/Y and Bwd(X, Y) = X\Y (result first,
# slash direction marking where the argument is sought); chart
# items are span(I, J, X) triples; the semiring is LogProb for
# differentiable inside scores.
#
# Deduction (Steedman's combinators):
#
#   fwd_app    : X/Y, Y       |- X           forward application
#   bwd_app    : Y,   X\Y     |- X           backward application
#   fwd_comp   : X/Y, Y/Z     |- X/Z         forward composition
#   bwd_comp   : Y\Z, X\Y     |- X\Z         backward composition
#   fwd_xcomp  : X/Y, Y\Z     |- X\Z         forward crossed composition
#   bwd_xcomp  : Y/Z, X\Y     |- X/Z         backward crossed composition
#
# Each combinator is needed by one sentence of the lexicon below:
#
#   the cat sleeps                  application only
#   the dog whom Mary saw sleeps    fwd_comp: Mary : S/(S\NP) with
#                                   saw : (S\NP)/NP gives S/NP
#   the cat that sleeps today purrs bwd_comp: sleeps : S\NP with
#                                   today : S\S gives S\NP
#   Mary probably sleeps            fwd_xcomp: probably : S/S with
#                                   sleeps : S\NP gives S\NP
#   Mary saw yesterday the cat      bwd_xcomp: saw : (S\NP)/NP with
#                                   yesterday : (S\NP)\(S\NP)
#                                   gives (S\NP)/NP
#
# Application and harmonic composition are theorems of the
# Lambek calculus, whose models are biclosed monoidal categories:
# the slashes are the two internal homs, application is
# evaluation, and harmonic composition is composition of internal
# homs. The crossed rules are not Lambek theorems; they hold only
# once the tensor is symmetric, as in the commutative calculus LP.
# The full rule set is thus an extension of the Lambek fragment
# rather than the internal language of a biclosed monoidal
# category.

object Term : FinSet 16

object LogWeight : Real 1

deduction CCG : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms NP, S, N, Fwd, Bwd, span, the, cat, dog, mary, saw, sleeps, purrs, whom, that, probably, today, yesterday
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    rule fwd_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z)) #[learnable]
    rule bwd_comp : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Bwd(X, Z)) #[learnable]
    rule fwd_xcomp : span(I, K, Fwd(X, Y)), span(K, J, Bwd(Y, Z)) |- span(I, J, Bwd(X, Z)) #[learnable]
    rule bwd_xcomp : span(I, K, Fwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Fwd(X, Z)) #[learnable]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "cat" : N = cat #[learnable]
        "dog" : N = dog #[learnable]
        "Mary" : NP = mary #[learnable]
        "Mary" : Fwd(S, Bwd(S, NP)) = mary #[learnable]
        "saw" : Fwd(Bwd(S, NP), NP) = saw #[learnable]
        "sleeps" : Bwd(S, NP) = sleeps #[learnable]
        "purrs" : Bwd(S, NP) = purrs #[learnable]
        "whom" : Fwd(Bwd(N, N), Fwd(S, NP)) = whom #[learnable]
        "that" : Fwd(Bwd(N, N), Bwd(S, NP)) = that #[learnable]
        "probably" : Fwd(S, S) = probably #[learnable]
        "today" : Bwd(S, S) = today #[learnable]
        "yesterday" : Bwd(Bwd(S, NP), Bwd(S, NP)) = yesterday #[learnable]

# The grammar's contribution to a probabilistic program: parsing an
# observed sentence and scoring the chart's goal weight adds the
# inside log-weight log Z(s), summed over every derivation of S, to
# the program's log-joint. The program returns that one real number,
# so its codomain is LogWeight : Real 1.
program score_sentence : Term -> LogWeight
    let chart = parse(CCG, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export score_sentence
```

## Overview

Combinatory Categorial Grammar (CCG) is expressed as an agenda-based weighted deduction whose items are chart spans `span(I, J, X)` (token range `[I, J)` carrying category `X`). Steedman's combinators, forward and backward application, harmonic composition, and crossed composition [@steedman2000syntactic], each become one sequent rule. The lexicon is chosen so that each of the six rules is needed by at least one sentence, and the page checks this by deleting each rule in turn. The semiring is `LogProb`, so inside scores flow as differentiable tensors back to the learnable rule and lexicon weights.

## Understanding the model

A category describes what an expression can combine with. Read `NP/N` as
an expression that takes a noun on its right and returns a noun phrase:
`the` with that category can combine with `dog : N` to form `the dog : NP`.
A verb seeking a noun phrase on its left can then complete the sentence.
The [chart](pcfg.md#understanding-the-model) stores these partial results by their token intervals.

Composition combines two functors before either has its argument. In
`the dog whom Mary saw sleeps`, the type-raised subject `Mary : S/(S\NP)`
composes with `saw : (S\NP)/NP` into `Mary saw : S/NP`, a sentence missing
its object, which is what the relative pronoun `whom : (N\N)/(S/NP)` asks
for. Backward composition does the same leftward: in
`the cat that sleeps today purrs`, `sleeps : S\NP` and `today : S\S` compose
into `S\NP`, the argument of `that : (N\N)/(S\NP)`. Crossed composition
combines functors whose slashes point in opposite directions. In
`Mary probably sleeps`, the sentence adverb `probably : S/S` composes with
`sleeps : S\NP` into `S\NP`; in the heavy-NP-shift order
`Mary saw yesterday the cat`, `saw : (S\NP)/NP` composes with
`yesterday : (S\NP)\(S\NP)` into `(S\NP)/NP`, which then takes the shifted
object.

Unrestricted crossed composition also permits orders English does not: with
the same lexicon, `the sleeps cat` parses, because backward crossed
composition of `the : NP/N` with `sleeps : S\NP` yields `S/N`. Steedman
restricts the crossed rules to particular categories for this reason; the
deduction here keeps them unrestricted, and the enumeration below lists such
strings among the grammatical ones.

Rule weights score derivations; alternatives leading to the same goal are
combined by the chosen [semiring](../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine). Under `LogProb`, weights add along a
single derivation and `logsumexp` combines alternatives. The goal weight
$Z(s)$ is thus a sum of exponentiated derivation weights, not a probability.
Over the finitely many strings of a fixed length $n$ it normalizes into the
*length-conditional distribution*
$P(s \mid n) = Z(s) / \sum_{|s'| = n} Z(s')$, which is the model the fit below
recovers.

## Walkthrough

`object Term : FinSet 16` declares a finite carrier for chart items; the concrete cardinality is irrelevant because the deduction reasons symbolically over constructor-tagged tuples, not over enumerated elements of `Term`.

`atoms NAME, NAME, ...` lists every identifier the rules may match literally: category atoms (`NP`, `S`, `N`), slash constructors (`Fwd`, `Bwd`), the chart-item constructor (`span`), and the lexical logical-form constants. Identifiers not listed here that appear in a rule pattern are bound as wildcards; the convention is single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`).

Each `rule` is a sequent: premises on the left of `|-`, conclusion on the right. `Fwd(X, Y)` constructs the forward-slash category `X/Y`; `Bwd(X, Y)` constructs the backward-slash category `X\Y`, with the result first in both. Adjacent spans whose end / start indices agree fire whichever rule's pattern matches their categories. `Mary` has two lexical entries, the plain `NP` and the type-raised `S/(S\NP)` that forward composition needs; the chart sums over both.

Every rule and lexical entry carries `#[learnable]`. A learnable rule's log-weight is keyed by the full binding of its pattern variables, the span positions as well as the categories ([Weighted Deduction Fragment §2.1](../semantics/grammar.md#21-learnable-rule-weights-learnable)), and is allocated the first time that binding fires.

The header's option block sets the remaining knobs: the `semiring=LogProb` option selects log-space inside scores, the `start=S` option declares the goal category for a successful parse, and the `depth=6` option bounds the constructor depth of derived categories to keep the agenda finite.

The `score_sentence` program connects the grammar to a probabilistic program: `parse(CCG, sentence)` runs the deduction on an observed sentence, and `score log_Z = chart.goal_weight()` adds $\log Z(s)$ to the program's log-joint. That is an unnormalized factor; the fit below divides by the length-$n$ total explicitly.

## DSL features

- **`deduction Name : Dom -> Cod [options]` header plus indented body**: declares the agenda-based weighted deduction in a single construct. The header's option block sets the semiring, the start symbol, and the depth bound; the body supplies the item algebra (via `atoms`), the rule set, and the lexicon that serves as the axiom source.
- **`atoms NAME, NAME, ...`**: closes the constructor universe. Identifiers listed here match literally in rule patterns; any identifier not listed is bound as a wildcard, with single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`) as the convention.
- **Sequent rules**: arbitrary-arity premises on the left of `|-`, single conclusion on the right; rules with one premise are unary chart rules, with two are binary, and so on.
- **Slash constructors**: `Fwd(X, Y)` and `Bwd(X, Y)` are user-declared atoms, not built-in syntax. The combinators are ordinary rules over them.
- **`parse(D, sentence)` in a program**: a program body can run a deduction and score its goal weight.

## Try it

Every `#[learnable]` lexicon entry and every `#[learnable]` rule
exposes a real `nn.Parameter` on the compiled
[`DeductionSystem`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem). The
system is callable: `ded(sentence)` returns a
[`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose
`goal_weight()` is the differentiable log-weight
$\log Z(s; \mathbf{w}) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$
summed over every derivation $d$ that the start symbol licenses for
the input.

### Each combinator is needed

Removing a rule from the compiled system and reparsing shows which sentences
depend on it. Each sentence below parses with all six rules and fails
without the rule it is meant to exercise.

```python
import dataclasses

import torch
from quivers.dsl import load

prog = load("docs/examples/source/ccg.qvr")
ded = prog.deductions["CCG"]


def parses(system, sentence):
    return bool(torch.isfinite(system(sentence.split()).goal_weight()))


def without(system, rule_name):
    rules = tuple(rule for rule in system.rules if rule.name != rule_name)
    assert len(rules) == len(system.rules) - 1
    return dataclasses.replace(system, rules=rules)


needs = {
    "fwd_app": "the cat sleeps",
    "bwd_app": "the cat sleeps",
    "fwd_comp": "the dog whom Mary saw sleeps",
    "bwd_comp": "the cat that sleeps today purrs",
    "fwd_xcomp": "Mary probably sleeps",
    "bwd_xcomp": "Mary saw yesterday the cat",
}
for rule_name, sentence in needs.items():
    assert parses(ded, sentence), sentence
    assert not parses(without(ded, rule_name), sentence), (rule_name, sentence)
    print(f"{rule_name:9s} is needed for {sentence!r}")

# Unrestricted backward crossed composition scrambles the noun phrase.
assert parses(ded, "the sleeps cat")
assert not parses(without(ded, "bwd_xcomp"), "the sleeps cat")
```

### Recovering a length-conditional distribution

A copy of the grammar with every weight drawn from a unit Normal serves as the
ground truth. [`sample_corpus`](../api/stochastic/deduction/sample.md#quivers.stochastic.deduction.sample.sample_corpus)
draws 2,000 three-word sentences from its length-conditional distribution,
enumerating every string over the vocabulary and sampling in proportion to
$Z(s)$. A freshly loaded copy, with every weight at zero, is then fit by
maximizing the same length-conditional log-likelihood. The denominator needs
only the strings that parse at all, and whether a string parses does not
depend on the weights, so the grammatical strings are enumerated once.

```python
import itertools

from quivers.stochastic.deduction import sample_corpus

torch.manual_seed(0)
VOCAB = [
    "the", "cat", "dog", "Mary", "saw", "sleeps", "purrs",
    "whom", "that", "probably", "today", "yesterday",
]
truth = load("docs/examples/source/ccg.qvr").deductions["CCG"]
model = load("docs/examples/source/ccg.qvr").deductions["CCG"]

strings = [
    list(words)
    for words in itertools.product(VOCAB, repeat=3)
    if parses(truth, " ".join(words))
]
print(len(strings), "grammatical three-word strings")
assert ["the", "sleeps", "cat"] in strings


def log_conditional(system):
    """log P(s | n = 3) for every grammatical three-word string."""
    scores = torch.stack([system(s).goal_weight() for s in strings])
    return scores - torch.logsumexp(scores, dim=0)


def kl(log_p, log_q):
    return float((log_p.exp() * (log_p - log_q)).sum())


# Parsing every string once allocates every weight either system uses.
for s in strings:
    truth(s).goal_weight()
    model(s).goal_weight()
with torch.no_grad():
    for p in truth.parameters():
        p.normal_(0.0, 1.0)
    true_log_p = log_conditional(truth)

corpus = sample_corpus(truth, length=3, n_samples=2000, seed=0)
index = {tuple(s): i for i, s in enumerate(strings)}
counts = torch.bincount(
    torch.tensor([index[tuple(s)] for s in corpus]), minlength=len(strings)
).float()

with torch.no_grad():
    kl_initial = kl(true_log_p, log_conditional(model))
opt = torch.optim.Adam(list(model.parameters()), lr=5e-2)
for _ in range(200):
    opt.zero_grad()
    loss = -(counts * log_conditional(model)).sum() / counts.sum()
    loss.backward()
    opt.step()
with torch.no_grad():
    kl_fitted = kl(true_log_p, log_conditional(model))

print(f"KL(true || initial) = {kl_initial:.4f}")
print(f"KL(true || fitted)  = {kl_fitted:.4f}")
assert kl_fitted < 0.05 and kl_fitted < 0.25 * kl_initial
```

Twenty-one three-word strings parse. Besides plain application, they include
`Mary saw Mary` and `probably Mary sleeps`, with derivations through forward
composition of the raised subject; `Mary sleeps today`, through backward
composition; `Mary probably sleeps`, through forward crossed composition; and
the scrambled `the sleeps cat` and `Mary today sleeps`, through backward
crossed composition. With 2,000 draws over 21 outcomes, the sampling error of
the maximum-likelihood fit is near $20 / (2 \cdot 2000) = 0.005$ nats, so a
divergence below 0.05 indicates that the fit has recovered the distribution.
The weights themselves are not identifiable, since several bindings can trade
mass along one derivation, so the check is on the distribution over strings.

## Categorical perspective

Application and harmonic composition are theorems of the [Lambek calculus](https://doi.org/10.2307/2310058) [@lambek1958mathematics], whose models are [biclosed monoidal categories](https://ncatlab.org/nlab/show/closed+monoidal+category): `X/Y` and `X\Y` are the two internal homs, application is the evaluation counit of each hom-tensor adjunction, and harmonic composition is composition of internal homs. Crossed composition is not a Lambek theorem. It becomes derivable once the tensor is symmetric, as in the commutative calculus LP, which is also why it licenses `the sleeps cat`: a symmetric tensor forgets word order. The six rules thus extend the Lambek fragment rather than present the internal language of a biclosed monoidal category, and operationally the page defines exactly the six sequent rules shown in the source.

## Semiring selection

The choice of semiring affects the parser's behavior: `LogProb` accumulates inside log-weights (numerically stable, differentiable); `Viterbi` returns the highest-weight derivation; `Counting` counts distinct derivations; `Boolean` checks membership without weights. The same deduction block serves all four objectives via the `semiring` option.
