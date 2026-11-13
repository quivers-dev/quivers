# Custom Sequent Rules

## QVR source

```qvr
# Custom Rules
#
# An AB grammar (Ajdukiewicz-Bar-Hillel) plus harmonic
# composition, illustrating that every rule in a deduction block
# is a sequent declared in the DSL itself; there are no built-in
# named rule schemas.
#
# Deduction:
#
#   fwd_app   : X/Y, Y    |- X        forward application
#   bwd_app   : Y,   X\Y  |- X        backward application
#   fwd_comp  : X/Y, Y/Z  |- X/Z      forward composition
#   bwd_comp  : Y\Z, X\Y  |- X\Z      backward composition
#
# Fwd(X, Y) is X/Y and Bwd(X, Y) is X\Y: the result comes first
# and the slash direction says where the argument is sought.
#
# The lexicon is chosen so that each rule is needed by some
# sentence. Application alone derives "the dog runs". The object
# relative "the dog whom Mary saw runs" needs forward composition:
# the type-raised subject Mary : S/(S\NP) composes with
# saw : (S\NP)/NP into S/NP, the argument whom : (N\N)/(S/NP)
# seeks. The subject relative "the dog that runs today barks"
# needs backward composition: runs : S\NP composes with the
# sentence modifier today : S\S into S\NP, the argument
# that : (N\N)/(S\NP) seeks.
#
# Pattern variables are single-uppercase identifiers (X, Y, Z,
# I, J, K); every other identifier appearing in a rule pattern
# must be listed in the atoms block.

object Term : FinSet 16

object LogWeight : Real 1

deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, cat, mary, saw, runs, barks, whom, that, today
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    rule fwd_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z)) #[learnable]
    rule bwd_comp : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Bwd(X, Z)) #[learnable]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "dog" : N = dog #[learnable]
        "cat" : N = cat #[learnable]
        "Mary" : NP = mary #[learnable]
        "Mary" : Fwd(S, Bwd(S, NP)) = mary #[learnable]
        "saw" : Fwd(Bwd(S, NP), NP) = saw #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]
        "barks" : Bwd(S, NP) = barks #[learnable]
        "whom" : Fwd(Bwd(N, N), Fwd(S, NP)) = whom #[learnable]
        "that" : Fwd(Bwd(N, N), Bwd(S, NP)) = that #[learnable]
        "today" : Bwd(S, S) = today #[learnable]

# The grammar's contribution to a probabilistic program: parsing an
# observed sentence and scoring the chart's goal weight adds the
# inside log-weight log Z(s), summed over every derivation of S, to
# the program's log-joint. The program returns that one real number,
# so its codomain is LogWeight : Real 1.
program score_sentence : Term -> LogWeight
    let chart = parse(AB, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export score_sentence
```

## Overview

Every rule in a `deduction { … }` block is a sequent declared in the DSL itself; there are no built-in named rule schemas to import. This example defines an AB grammar (Ajdukiewicz-Bar-Hillel) plus harmonic composition over chart-spans `span(I, J, X)`. `Fwd(X, Y)` is the forward-slash constructor `X/Y`; `Bwd(X, Y)` is the backward-slash constructor `X\Y`. The lexicon is chosen so that each of the four rules is needed by at least one sentence, and the page checks this by deleting each rule in turn.

## Understanding the model

A rule is a template for combining known facts. Suppose one chart entry is
`span(0, 1, Fwd(NP, N))` and the next is `span(1, 2, N)`. The repeated position
and category variables in forward application must agree, giving
`span(0, 2, NP)`. This agreement is unification: different occurrences of one
variable receive one value within that rule firing.

Application suffices for `the dog runs`. Composition earns its place when a
constituent is missing an argument at its edge. In the object relative
`the dog whom Mary saw runs`, the clause `Mary saw` lacks its object, and
`whom : (N\N)/(S/NP)` asks for exactly such a clause. Forward composition
builds it: the type-raised subject `Mary : S/(S\NP)` composes with
`saw : (S\NP)/NP` into `S/NP`. In the subject relative
`the dog that runs today barks`, `that : (N\N)/(S\NP)` asks for a verb phrase,
and backward composition of `runs : S\NP` with the sentence modifier
`today : S\S` supplies the verb phrase `runs today : S\NP`.

An [agenda](../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) records newly derived entries and tries rules that can use them.
Weights say how to combine evidence along a derivation and across alternatives.
Changing the rule changes what can be derived; changing the [semiring](../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine) changes
what is accumulated about those derivations. Keep those two choices separate
when interpreting the returned goal weight.

The goal weight $Z(s)$ is a sum of exponentiated derivation weights, not a
probability. Over the finitely many strings of a fixed length $n$ it
normalizes into the *length-conditional distribution*
$P(s \mid n) = Z(s) / \sum_{|s'| = n} Z(s')$, which is the model the fit below
recovers.

## Walkthrough

`object Term : FinSet 16` is the index the deduction's domain and codomain range over; its cardinality is incidental, since rules match constructor-tagged tuples symbolically.

`atoms NAME, NAME, …` lists every identifier the rules may match literally. Category atoms (`S`, `NP`, `N`), slash constructors (`Fwd`, `Bwd`), the chart-item constructor (`span`), and the lexical logical-form constants are atoms. Identifiers that appear in a rule pattern but are *not* listed in `atoms` are pattern variables; the convention is single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`). A pattern variable must not reuse an atom's spelling: had a rule written a position as `N`, it would match the noun category literally.

Each rule's body is a sequent: comma-separated premises on the left of `|-`, a single conclusion on the right. The premise multiplicity determines whether the rule fires on a single chart cell (unary) or on several cells (binary and higher).

A variable appearing multiple times in the same rule unifies across occurrences: in `fwd_app`, the `Y` in the first premise must match the `Y` in the second premise. Different rules instantiate independently.

Every rule and lexical entry carries `#[learnable]`. A learnable rule's log-weight is keyed by the binding of its category variables `X`, `Y`, `Z`, not by the span positions `I`, `J`, `K` ([Weighted Deduction Fragment §2.1](../semantics/grammar.md#21-learnable-rule-weights-learnable)), and is allocated the first time that binding fires. Two firings of `fwd_app` on the same categories at different positions thus read one weight.

The `score_sentence` program connects the grammar to a probabilistic program: `parse(AB, sentence)` runs the deduction on an observed sentence, and `score log_Z = chart.goal_weight()` adds $\log Z(s)$ to the program's log-joint. That is an unnormalized factor; the fit below divides by the length-$n$ total explicitly.

## DSL features

- **`rule NAME : premises |- conclusion`**: a sequent rule. Arbitrary-arity premise lists are supported; the compiler dispatches to the appropriate chart-cell shape.
- **Pattern variables vs atoms**: single-uppercase identifiers bind as wildcards; every other identifier in a rule pattern must appear in the `atoms` list.
- **Constructor applications in patterns**: `Fwd(X, Y)`, `Bwd(X, Y)`, `span(I, J, X)` are patterns whose head is an atom and whose arguments are nested patterns. Matching is structural and unification-based.
- **Several entries per word**: `Mary` is listed twice, as `NP` and as the type-raised `S/(S\NP)`; the lexicon injects one axiom per entry, and the chart sums over both.
- **`parse(D, sentence)` in a program**: a program body can run a deduction and score its goal weight.

## Pattern Matching and Unification

The compiler uses first-order pattern matching with occurs-checking-free unification to fire rules:

- **Variables** (`X`, `Y`, `Z`): metavariables that match any concrete subterm. Repeated occurrences within a rule must unify.
- **Constructor patterns** (`Fwd(X, Y)`): match constructor-application items whose head is the same atom and whose children unify.

The agenda fires each rule at every chart cell (unary) or every combination of cells (binary and higher) whose items unify with the rule's premises; the conclusion item is then inserted into the chart under the rule's weight.

## Extending the Rule Set

Additional combinators are spelled out as more sequent rules in the same block. Every variable in a conclusion must be bound by a premise, because the chart holds only ground items. General type-raising, `span(I, J, X) |- span(I, J, Fwd(Y, Bwd(Y, X)))`, leaves the result category `Y` free, and the compiler rejects it with `rule 'type_raise' concludes with 'Y', which no premise binds`. Fixing the raised-over category makes the rule well formed. Subject type-raising to `S/(S\NP)`, the category `Mary` carries lexically above, compiles and fires:

```qvr
object Term : FinSet 16

deduction Raised : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X)
    rule subject_raise : span(I, J, NP) |- span(I, J, Fwd(S, Bwd(S, NP)))
    lexicon
        "the" : Fwd(NP, N) = the
        "dog" : N = dog
        "runs" : Bwd(S, NP) = runs
```

With `subject_raise`, `the dog runs` has two derivations, one by backward application and one by forward application of the raised subject, so its goal weight is $\log 2$ with every weight at zero. Literal atoms in a pattern restrict a rule in the same way: `span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, NP)) |- span(I, J, Fwd(X, NP))` is forward composition limited to functors whose argument is `NP`.

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

### Each rule is needed

Removing a rule from the compiled system and reparsing shows which sentences
depend on it. Each sentence below parses with the full rule set and fails
without the rule it is meant to exercise.

```python
import dataclasses

import torch
from quivers.dsl import load

prog = load("docs/examples/source/custom_rules.qvr")
ded = prog.deductions["AB"]


def parses(system, sentence):
    return bool(torch.isfinite(system(sentence.split()).goal_weight()))


def without(system, rule_name):
    rules = tuple(rule for rule in system.rules if rule.name != rule_name)
    assert len(rules) == len(system.rules) - 1
    return dataclasses.replace(system, rules=rules)


needs = {
    "fwd_app": "the dog runs",
    "bwd_app": "the dog runs",
    "fwd_comp": "the dog whom Mary saw runs",
    "bwd_comp": "the dog that runs today barks",
}
for rule_name, sentence in needs.items():
    assert parses(ded, sentence), sentence
    assert not parses(without(ded, rule_name), sentence), (rule_name, sentence)
    print(f"{rule_name:9s} is needed for {sentence!r}")
```

### Recovering a length-conditional distribution

A copy of the grammar with every weight drawn from a unit Normal serves as the
ground truth. [`sample_corpus`](../api/stochastic/deduction/sample.md#quivers.stochastic.deduction.sample.sample_corpus) draws 2,000
three-word sentences from its length-conditional distribution, enumerating
every string over the vocabulary and sampling in proportion to $Z(s)$. A
freshly loaded copy, with every weight at zero, is then fit by maximizing the
same length-conditional log-likelihood. The denominator needs only the
strings that parse at all, and whether a string parses does not depend on the
weights, so the grammatical strings are enumerated once.

```python
import itertools

from quivers.stochastic.deduction import sample_corpus

torch.manual_seed(0)
VOCAB = ["the", "dog", "cat", "Mary", "saw", "runs", "barks", "whom", "that", "today"]
truth = load("docs/examples/source/custom_rules.qvr").deductions["AB"]
model = load("docs/examples/source/custom_rules.qvr").deductions["AB"]

strings = [
    list(words)
    for words in itertools.product(VOCAB, repeat=3)
    if parses(truth, " ".join(words))
]
print(len(strings), "grammatical three-word strings")


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

Seven three-word strings parse, and two of them use composition:
`Mary saw Mary` has one derivation through forward composition of the raised
subject with `saw`, and `Mary runs today` two through backward composition of
`runs` with `today`. With 2,000 draws over seven outcomes, the sampling error
of the maximum-likelihood fit is near $6 / (2 \cdot 2000) = 0.0015$ nats, so a
divergence below 0.05 indicates that the fit has recovered the distribution.
The weights themselves are not identifiable: several bindings can trade mass
along one derivation, so the check is on the distribution over strings.

## Categorical perspective

Sequent rules are hyperedges in the rule-system multicategory. A binary rule is a 2-input / 1-output hyperedge whose endpoints are pattern templates; firing the rule against the chart is the substitution along a pattern morphism into the category of concrete chart items. Variable unification across premises is exactly the pullback in the category of variable assignments: two premise patterns sharing a variable `Y` constrain the two assignments to agree on `Y`'s value. The chart accumulates the weights of derivations, combining premises and alternatives with the semiring operations [@goodman1999semiring]. Agreement between evaluation orders requires an appropriate algorithm: repeated contributions must be handled correctly, and recursive systems need termination or convergence conditions. The runtime selects a default strategy from rule arities and semiring properties; an arbitrary firing order need not preserve the result.
