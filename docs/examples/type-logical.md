# Type-Logical Grammar (Lambek calculus)

## Overview

This example is a weighted chart parser whose every rule is a theorem of the associative Lambek calculus L [@lambek1958mathematics]. Categories are built from the atoms `S`, `NP`, `N` with the two slashes and the product, and a chart item `span(I, J, X)` records that the words in `[I, J)` derive `X`. Because each rule is an L-theorem and the chart only combines adjacent spans in order, everything the chart derives is L-derivable: the parser is a sound fragment of L. It is not complete, since L also proves sequents that need hypothetical reasoning, and a sequent requiring that reasoning is given below.

## QVR source

```qvr
# Weighted Type-Logical Grammar: a sound fragment of the Lambek calculus
#
# Categories are built from the atoms S, NP, N with the two slashes
# and the product of the associative Lambek calculus L (Lambek 1958),
# written in Lambek's argument-under convention throughout:
#
#   Fwd(A, B) = A/B    an A that seeks a B on its right
#   Bwd(B, A) = B\A    an A that seeks a B on its left
#   Tns(A, B) = A*B    an A followed by a B, kept together
#
# so that residuation reads  A*B |- C  iff  A |- C/B  iff  B |- A\C.
# Chart items are span(I, J, X): the words in [I, J) derive X.
#
# Every rule below is a theorem of L. A rule with premises
# X_1, ..., X_n and conclusion Y licenses the sequent
# X_1, ..., X_n |- Y, and each such sequent has a cut-free L proof:
#
#   fwd_app    : A/B, B      |- A        (/L on the axioms)
#   bwd_app    : B, B\A      |- A        (\L on the axioms)
#   fwd_comp   : A/B, B/C    |- A/C      (/L twice, then /R)
#   bwd_comp   : C\B, B\A    |- C\A      (\L twice, then \R)
#   raise_subj : NP          |- S/(NP\S) (\L, then /R; the instance
#                                         A |- B/(A\B) at A = NP, B = S)
#   prod_intro : A, B        |- A*B      (*R)
#   curry      : C/(A*B)     |- (C/B)/A  (*R under /L, then /R twice)
#
# Every rule uses each premise exactly once and keeps the premises in
# order, so the chart has no weakening, contraction, or permutation.
# A finite rule set over adjacent spans has no hypothetical reasoning,
# so the chart derives a proper subset of the L-theorems: it is a sound
# fragment of L, not L itself.

object Term : FinSet 16

object LogWeight : Real 1

deduction Lambek : Term -> Term [semiring=LogProb, start=S, depth=4]
    atoms S, NP, N, Fwd, Bwd, Tns, span, kim, lee, books, every, linguist, book, sleeps, admires, gives, that, today
    rule fwd_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule bwd_app : span(I, K, B), span(K, J, Bwd(B, A)) |- span(I, J, A) #[learnable]
    rule fwd_comp : span(I, K, Fwd(A, B)), span(K, J, Fwd(B, C)) |- span(I, J, Fwd(A, C)) #[learnable]
    rule bwd_comp : span(I, K, Bwd(C, B)), span(K, J, Bwd(B, A)) |- span(I, J, Bwd(C, A)) #[learnable]
    rule raise_subj : span(I, J, NP) |- span(I, J, Fwd(S, Bwd(NP, S))) #[learnable]
    rule prod_intro : span(I, K, A), span(K, J, B) |- span(I, J, Tns(A, B)) #[learnable]
    rule curry : span(I, J, Fwd(C, Tns(A, B))) |- span(I, J, Fwd(Fwd(C, B), A)) #[learnable]
    lexicon
        "Kim"      : NP                          = kim      #[learnable]
        "Lee"      : NP                          = lee      #[learnable]
        "books"    : NP                          = books    #[learnable]
        "every"    : Fwd(NP, N)                  = every    #[learnable]
        "linguist" : N                           = linguist #[learnable]
        "book"     : N                           = book     #[learnable]
        "sleeps"   : Bwd(NP, S)                  = sleeps   #[learnable]
        "admires"  : Fwd(Bwd(NP, S), NP)         = admires  #[learnable]
        "gives"    : Fwd(Bwd(NP, S), Tns(NP, NP)) = gives   #[learnable]
        "that"     : Fwd(Bwd(N, N), Fwd(S, NP))  = that     #[learnable]
        "today"    : Bwd(S, S)                   = today    #[learnable]

# The program scores a sentence by the chart's goal weight: the inside
# log-sum, over every derivation of S that spans the whole input, of
# the learnable lexicon and rule log-weights the derivation uses. Its
# value is one real number, so the codomain is LogWeight : Real 1.
# The token sequence enters as the data parameter ``sentence``.
program fit_grammar : Term -> LogWeight
    let chart = parse(Lambek, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit_grammar
```

## Understanding the model

L is a logic of ordered resources. A sequent `X_1, ..., X_n |- Y` holds when the words typed `X_1` through `X_n`, in that order and each used exactly once, form an expression of type `Y`. L has no structural rules: it cannot reorder premises (exchange), drop one (weakening), or reuse one (contraction). A chart over spans enforces two of those restrictions by construction, since a binary rule combines two adjacent spans and its conclusion covers exactly their union. What the chart cannot enforce by itself is that the *category* bookkeeping respects resources: a rule `Tns(A, B) |- A` would discard the `B` resource while keeping its words in the span, which is weakening in disguise. Every rule here passes a stricter test, namely that its premises entail its conclusion in L.

The lexicon uses this to do linguistic work. `that` has type `(N\N)/(S/NP)`, so a relative clause must be a sentence missing an `NP` at its right edge. "Kim admires" is such a clause: type raising lifts `Kim` to `S/(NP\S)`, and composition with `admires : (NP\S)/NP` yields `S/NP`. "Kim gives Lee" is another, once currying turns `gives : (NP\S)/(NP*NP)` into `((NP\S)/NP)/NP`. A clause with a medial gap such as "Kim admires _ today" has no such analysis, which is correct for L: extraction from a non-peripheral position needs a structural rule that L lacks (the [multimodal example](multimodal-tlg.md) adds a controlled one).

## Walkthrough

The source uses Lambek's argument-under convention throughout. `Fwd(A, B)` is `A/B`, a functor that seeks a `B` on its right. `Bwd(B, A)` is `B\A`, a functor that seeks a `B` on its left. `Tns(A, B)` is the product `A*B`. In that notation the residuation laws read `A*B |- C` iff `A |- C/B` iff `B |- A\C`.

| Rule | Sequent | Cut-free L derivation |
|---|---|---|
| `fwd_app` | `A/B, B \|- A` | `/L` on the axioms `B \|- B` and `A \|- A` |
| `bwd_app` | `B, B\A \|- A` | `\L` on the axioms |
| `fwd_comp` | `A/B, B/C \|- A/C` | `A/B, B/C, C \|- A` by `/L` twice, then `/R` |
| `bwd_comp` | `C\B, B\A \|- C\A` | `C, C\B, B\A \|- A` by `\L` twice, then `\R` |
| `raise_subj` | `NP \|- S/(NP\S)` | `NP, NP\S \|- S` by `\L`, then `/R`. The instance of type raising `A \|- B/(A\B)` at `A = NP`, `B = S` |
| `prod_intro` | `A, B \|- A*B` | `*R` |
| `curry` | `C/(A*B) \|- (C/B)/A` | `A, B \|- A*B` by `*R`, so `C/(A*B), A, B \|- C` by `/L`. Then `/R` twice |

Composition and type raising are the standard derived rules of L [@lambek1958mathematics; @moortgat1997categorial]. The order in `curry` matters: the argument adjacent to the functor, `A`, becomes the outer argument, so `C/(A*B)` becomes `(C/B)/A` and not `(C/A)/B`. The converse `(C/B)/A |- C/(A*B)` is also an L-theorem, but adding it as a rule would close a cycle between the two forms, so the fragment keeps one direction.

Pattern variables (single uppercase letters such as `A`, `B`, `I`, `J`) bind at firing time. Type raising is stated at one instance because a rule's conclusion may only mention variables its premises bind: the general schema `A |- B/(A\B)` leaves `B` free. The `depth=4` option bounds the constructor depth of a rule's conclusion category. It is what keeps `prod_intro` from building ever-deeper products over a span, and the lexicon's deepest derived category, the curried `((NP\S)/NP)/NP`, sits at that bound. No rule closes a cycle, so the chart needs no `#[bounded]` rule and no convergence tolerance.

`fit_grammar` is the probabilistic program over the grammar: it parses the data parameter `sentence` and scores the chart's goal weight, $\log Z(s; \mathbf{w}) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$, the log-sum over every derivation $d$ of `S` over the whole input of the learnable lexicon and rule log-weights it uses. A learnable rule's log-weight is allocated per binding of its category variables ([§2.1](../semantics/grammar.md#21-learnable-rule-weights-learnable)). Firings with the same category bindings at different span positions share that weight.

## Try it

The first block types each word, as in the source, and implements an independent decision procedure for L: cut-free backward proof search, which terminates because every sequent rule removes a connective and is complete by Lambek's cut-elimination theorem. It then checks that each chart rule's sequent is an L-theorem and that the two projections a weakening rule would use are not.

```python
import itertools
from functools import lru_cache

import torch
from quivers.dsl import load, loads

SOURCE = "docs/examples/source/type_logical.qvr"
prog = load(SOURCE)
ded = prog.deductions["Lambek"]

atom = lambda name: ("atom", name)
S, NP, N = atom("S"), atom("NP"), atom("N")
over = lambda a, b: ("Fwd", a, b)    # a/b
under = lambda b, a: ("Bwd", b, a)   # b\a
prod = lambda a, b: ("Tns", a, b)    # a*b

# The lexical type of each word, as the source assigns it; the loop
# confirms that each one is an item of the word's one-word chart.
LEX = {
    "Kim": NP, "Lee": NP, "books": NP,
    "every": over(NP, N), "linguist": N, "book": N,
    "sleeps": under(NP, S), "admires": over(under(NP, S), NP),
    "gives": over(under(NP, S), prod(NP, NP)),
    "that": over(under(N, N), over(S, NP)), "today": under(S, S),
}
for word, category in LEX.items():
    cats = {item[3] for item, _ in ded([word]).chart.items()}
    assert category in cats, word


@lru_cache(maxsize=None)
def lambek_proves(gamma, goal):
    """Decide gamma |- goal in the Lambek calculus L by cut-free
    backward search. Every rule removes one connective, so the search
    terminates, and cut elimination makes it complete."""
    if len(gamma) == 1 and gamma[0] == goal and goal[0] == "atom":
        return True
    if goal[0] == "Fwd" and lambek_proves(gamma + (goal[2],), goal[1]):
        return True                                            # /R
    if goal[0] == "Bwd" and lambek_proves((goal[1],) + gamma, goal[2]):
        return True                                            # \R
    if goal[0] == "Tns" and any(                               # *R
        lambek_proves(gamma[:k], goal[1]) and lambek_proves(gamma[k:], goal[2])
        for k in range(1, len(gamma))
    ):
        return True
    for i, f in enumerate(gamma):
        left, right = gamma[:i], gamma[i + 1:]
        if f[0] == "Tns" and lambek_proves(left + (f[1], f[2]) + right, goal):
            return True                                        # *L
        if f[0] == "Fwd" and any(                              # /L
            lambek_proves(gamma[i + 1:j], f[2])
            and lambek_proves(left + (f[1],) + gamma[j:], goal)
            for j in range(i + 2, len(gamma) + 1)
        ):
            return True
        if f[0] == "Bwd" and any(                              # \L
            lambek_proves(gamma[k:i], f[1])
            and lambek_proves(gamma[:k] + (f[2],) + right, goal)
            for k in range(i)
        ):
            return True
    return False


def types(words):
    return tuple(LEX[w] for w in words)


def parses(sentence):
    return bool(ded(sentence.split()).goal_items)


# Each rule's sequent is an L-theorem.
A, B, C = atom("A"), atom("B"), atom("C")
rule_sequents = {
    "fwd_app": ((over(A, B), B), A),
    "bwd_app": ((B, under(B, A)), A),
    "fwd_comp": ((over(A, B), over(B, C)), over(A, C)),
    "bwd_comp": ((under(C, B), under(B, A)), under(C, A)),
    "raise_subj": ((NP,), over(S, under(NP, S))),
    "prod_intro": ((A, B), prod(A, B)),
    "curry": ((over(C, prod(A, B)),), over(over(C, B), A)),
}
for name, (gamma, goal) in rule_sequents.items():
    assert lambek_proves(gamma, goal), name
# The two projections a "tensor elimination" by weakening would need
# are not L-theorems.
assert not lambek_proves((prod(A, B),), A)
assert not lambek_proves((prod(A, B),), B)
```

### Soundness and the fragment's limits

Each positive sentence parses, is L-derivable, and contains a chart item that only the named rule can produce. Each negative sentence would need a structural rule L lacks, and neither the chart nor the decision procedure accepts it. The soundness check then runs the decision procedure on every item of every chart: each must be an L-theorem over the words of its span. The last lines exhibit the incompleteness: `NP |- (S/NP)\S` is an L-theorem that no chart rule derives.

```python
# Positive sentences, each with the chart item that only a given rule
# can produce.
positives = {
    "Kim sleeps": ("bwd_app", None),
    "every linguist sleeps": ("fwd_app", (0, 2, NP)),
    "Kim sleeps today": ("bwd_comp", (1, 3, under(NP, S))),
    "Kim gives Lee books": ("prod_intro", (2, 4, prod(NP, NP))),
    "every linguist that Kim admires sleeps": ("fwd_comp", (3, 5, over(S, NP))),
    "every book that Kim gives Lee sleeps": ("curry", (4, 6, over(under(NP, S), NP))),
}
for sentence, (rule, witness) in positives.items():
    words = sentence.split()
    chart = ded(words)
    assert chart.goal_items, sentence
    assert lambek_proves(types(words), S), sentence
    if witness is not None:
        i, j, category = witness
        assert ("span", i, j, category) in dict(chart.chart.items()), (sentence, rule)
    print(f"parses   {sentence!r:44} (uses {rule})")

# Non-theorems. A permuted order needs exchange; "Kim sleeps Lee" needs
# weakening (Lee is never consumed); "Kim admires" as a sentence needs
# contraction (Kim as both subject and object).
negatives = ["Lee Kim admires", "admires Kim Lee", "Kim sleeps Lee",
             "Kim admires", "every linguist that admires Kim sleeps"]
for sentence in negatives:
    assert not parses(sentence), sentence
    assert not lambek_proves(types(sentence.split()), S), sentence
    print(f"rejected {sentence!r}")

# Soundness: every item in every chart is an L-theorem over its span.
checked = 0
for sentence in [*positives, *negatives]:
    words = sentence.split()
    for item, _ in ded(words).chart.items():
        _, i, j, category = item
        assert lambek_proves(types(words[i:j]), category), (sentence, item)
        checked += 1
print(f"{checked} chart items, every one an L-theorem over its span")

# Incompleteness: object type raising NP |- (S/NP)\S is an L-theorem,
# but no rule of the fragment derives it.
object_raised = under(over(S, NP), S)
assert lambek_proves((NP,), object_raised)
assert ("span", 0, 1, object_raised) not in dict(ded(["Kim"]).chart.items())
```

### Derivations versus readings

L has spurious ambiguity: distinct proofs of one sequent that a semantics would not distinguish. The goal weight sums over proofs, so a sentence with more proofs carries more mass at equal parameter values. The [Counting semiring](../semantics/grammar.md#3-semirings) makes the proof count visible. "Kim sleeps" has two proofs (`bwd_app` directly, or `raise_subj` then `fwd_app`) and "Kim gives Lee books" has seven.

```python
# Derivations, not readings: one sentence can have several L proofs.
# Re-scoring the bare deduction in the Counting semiring, with every
# weight left at the semiring's one, counts them.
import re

grammar_only = open(SOURCE).read().split("# The program scores")[0]
unweighted = re.sub(r"\s*#\[learnable\]", "", grammar_only)
counting = loads(unweighted.replace("semiring=LogProb", "semiring=Counting"))
count = counting.deductions["Lambek"]
for sentence, expected in [("Kim sleeps", 2), ("Kim admires Lee", 3), ("Kim gives Lee books", 7)]:
    n = int(count(sentence.split()).goal_weight())
    print(f"{sentence!r}: {n} derivations")
    assert n == expected, (sentence, n)
```

### Recovering a distribution over sentences

The goal weight $Z(s; \mathbf{w})$ is unnormalized. Conditioning on sentence length makes it a distribution: $p(s \mid |s| = n, \mathbf{w}) = Z(s; \mathbf{w}) / \sum_{|s'| = n} Z(s'; \mathbf{w})$, the distribution that [`sample_corpus`](../api/stochastic/deduction/sample.md#quivers.stochastic.deduction.sample.sample_corpus) draws from exactly by enumerating yields. The block draws every learnable log-weight from a Normal(0, 1) prior, samples 2,000 three-word sentences, refits the weights from zero by [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (maximum a posteriori) under the same prior, and requires the fitted distribution over the grammatical yields to lie within total-variation distance 0.05 of the generating one. Individual weights are not identified (a sentence's probability depends only on sums of weights along its derivations), so the assertion is on the distribution, which is.

```python
from quivers.stochastic.deduction import sample_corpus

torch.manual_seed(0)
vocabulary = list(LEX)
yields = [
    list(words) for words in itertools.product(vocabulary, repeat=3)
    if ded(list(words)).goal_items
]
print(f"{len(yields)} grammatical yields of length 3")


def log_probs():
    """log p(s) = log Z(s) - log sum_s' Z(s') over the length-3 yields."""
    log_z = torch.stack([ded(words).goal_weight() for words in yields])
    return log_z - torch.logsumexp(log_z, dim=0)


# Ground truth: every learnable weight drawn from its Normal(0, 1) prior.
params = list(ded.parameters())
with torch.no_grad():
    for p in params:
        p.copy_(torch.randn(()))
    true_probs = log_probs().exp()

N_SENTENCES = 2000
corpus = sample_corpus(ded, length=3, n_samples=N_SENTENCES, seed=0)
index = {tuple(words): k for k, words in enumerate(yields)}
counts = torch.bincount(
    torch.tensor([index[tuple(s)] for s in corpus]), minlength=len(yields)
).to(torch.get_default_dtype())

# MAP fit from a neutral start, under the same Normal(0, 1) prior.
with torch.no_grad():
    for p in params:
        p.zero_()
optim = torch.optim.Adam(params, lr=0.1)
for step in range(150):
    optim.zero_grad()
    loss = -(counts * log_probs()).sum() + 0.5 * sum((p ** 2).sum() for p in params)
    loss.backward()
    optim.step()

with torch.no_grad():
    fitted_probs = log_probs().exp()
empirical = counts / counts.sum()
tv = lambda p, q: 0.5 * float((p - q).abs().sum())
uniform = torch.full_like(true_probs, 1.0 / len(yields))
print(f"TV(fitted, true) = {tv(fitted_probs, true_probs):.3f}; "
      f"TV(empirical, true) = {tv(empirical, true_probs):.3f}; "
      f"TV(uniform, true) = {tv(uniform, true_probs):.3f}")
assert tv(fitted_probs, true_probs) < 0.05
```

## Categorical perspective

L is the internal language of a [residuated monoid](https://ncatlab.org/nlab/show/residuated+lattice) or, proof-relevantly, of a [biclosed monoidal category](https://ncatlab.org/nlab/show/closed+monoidal+category): the product `*` is the monoidal tensor and the two slashes are its right adjoints, `- * B -| -/B` and `A * - -| A\-`. Application is the [counit](https://ncatlab.org/nlab/show/adjunction) of these adjunctions, composition is composition of internal homs, and `curry` is one direction of the isomorphism `C/(A*B) ≅ (C/B)/A`. The chart computes, for each span, a finite set of objects reachable from the lexical types by these maps, weighted in the [LogProb semiring](../semantics/algebras.md). Because the category has no symmetry, diagonal, or projections, no chart item can permute, copy, or discard a word.

## Connections to other formalisms

The application rules alone give AB grammar ([custom rules](custom-rules.md)). Adding composition and type raising gives the core of combinatory categorial grammar ([CCG](ccg.md)), whose crossed composition rules are not L-theorems. The [multimodal example](multimodal-tlg.md) extends this fragment with unary modalities whose structural rule licenses the medial extraction that L rejects here, and the [quantifier-scope example](quantifier-scope.md) adds logical forms.
