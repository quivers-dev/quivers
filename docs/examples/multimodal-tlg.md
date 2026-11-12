# Multimodal Type-Logical Grammar

## QVR source

```qvr
# Multimodal Type-Logical Grammar: controlled extraction
#
# The associative Lambek calculus L extended with the unary pair
# Dia and Box (Moortgat's diamond and box-down), written in the
# argument-under convention of the type-logical example:
#
#   Fwd(A, B) = A/B      Bwd(B, A) = B\A
#   Dia(A)    = <>A      Box(A)    = []A
#
# Dia and Box are residuated:  <>A |- B  iff  A |- []B.  Two theorems
# follow at once: the counit <>[]A |- A and the unit A |- []<>A.
# Structural control adds one postulate that only Dia-marked
# structure may use. Writing <D> for the structure a Dia formula
# unpacks to, controlled permutation P lets a marked structure move
# rightward past its neighbor:
#
#   G, <D>, X, H |- C
#   ----------------- P
#   G, X, <D>, H |- C
#
# Unmarked structure never permutes, so the logic stays as
# order-sensitive as L everywhere outside the modal marking.
#
# Chart items are span(I, J, X). Every rule is a theorem of L with
# Dia, Box, and P:
#
#   fwd_app    : A/B, B      |- A          (/L)
#   bwd_app    : B, B\A      |- A          (\L)
#   fwd_comp   : A/B, B/C    |- A/C        (/L twice, /R)
#   raise_subj : NP          |- S/(NP\S)   (\L, /R)
#   counit_arg : A/B         |- A/<>[]B    (the counit <>[]B |- B under /L, then /R)
#   perm_arg   : B\A         |- A/<>[]B    (<[]B>, B\A |- A by the counit and \L;
#                                           P moves <[]B> to the right; <>L, /R)
#   x_comp     : A/B, A\C    |- C/<>[]B    (A/B, <[]B>, A\C |- C by the counit,
#                                           /L, \L; P moves <[]B> to the right;
#                                           <>L, /R)
#
# The relative pronoun is the one decorated word: "that" seeks a
# sentence missing a <>[]NP, so its gap is a marked hypothesis that P
# may move. perm_arg and x_comp are the chart's image of P: they let
# the gap sit at the left edge (subject relatives) or before a
# modifier (medial object relatives). Neither applies to an undecorated
# NP argument, since NP does not derive <>[]NP.

object Term : FinSet 16

object LogWeight : Real 1

deduction MMTLG : Term -> Term [semiring=LogProb, start=S, depth=3]
    atoms S, NP, N, Fwd, Bwd, Dia, Box, span, kim, lee, every, linguist, sleeps, admires, today, that
    rule fwd_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule bwd_app : span(I, K, B), span(K, J, Bwd(B, A)) |- span(I, J, A) #[learnable]
    rule fwd_comp : span(I, K, Fwd(A, B)), span(K, J, Fwd(B, C)) |- span(I, J, Fwd(A, C)) #[learnable]
    rule raise_subj : span(I, J, NP) |- span(I, J, Fwd(S, Bwd(NP, S))) #[learnable]
    rule counit_arg : span(I, J, Fwd(A, B)) |- span(I, J, Fwd(A, Dia(Box(B)))) #[learnable]
    rule perm_arg : span(I, J, Bwd(B, A)) |- span(I, J, Fwd(A, Dia(Box(B)))) #[learnable]
    rule x_comp : span(I, K, Fwd(A, B)), span(K, J, Bwd(A, C)) |- span(I, J, Fwd(C, Dia(Box(B)))) #[learnable]
    lexicon
        "Kim"      : NP                                  = kim      #[learnable]
        "Lee"      : NP                                  = lee      #[learnable]
        "every"    : Fwd(NP, N)                          = every    #[learnable]
        "linguist" : N                                   = linguist #[learnable]
        "sleeps"   : Bwd(NP, S)                          = sleeps   #[learnable]
        "admires"  : Fwd(Bwd(NP, S), NP)                 = admires  #[learnable]
        "today"    : Bwd(Bwd(NP, S), Bwd(NP, S))         = today    #[learnable]
        "that"     : Fwd(Bwd(N, N), Fwd(S, Dia(Box(NP)))) = that    #[learnable]

# The program scores a sentence by the chart's goal weight: the inside
# log-sum, over every derivation of S that spans the whole input, of
# the learnable lexicon and rule log-weights the derivation uses. Its
# value is one real number, so the codomain is LogWeight : Real 1.
# The token sequence enters as the data parameter ``sentence``.
program fit_grammar : Term -> LogWeight
    let chart = parse(MMTLG, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit_grammar
```

## Overview

Multimodal type-logical grammar adds unary modalities to the Lambek calculus and lets structural rules apply only to material those modalities mark [@moortgat1997categorial; @kurtonina1997structural]. This example implements a sound fragment of that system: the associative Lambek calculus L, the residuated pair `Dia` and `Box` (written `<>` and `[]`), and one structural postulate, controlled permutation, that only `Dia`-marked structure may use. The relative pronoun `that` is the one decorated word. Its gap is typed `<>[]NP`, and that decoration is what lets the gap sit at the left edge of the clause or before a modifier, positions plain L cannot reach. Removing the decoration removes exactly those analyses.

## Understanding the model

The [type-logical example](type-logical.md) shows that L handles extraction only at the right edge: "linguist that Kim admires" is fine because "Kim admires" derives `S/NP`, but "linguist that Kim admires today" is out, because the gap sits before the modifier and L cannot reorder. Adding a global permutation rule would fix that and wreck word order everywhere, since every sentence would then be acceptable in every order.

Structural control takes a narrower route. The modalities obey residuation, `<>A |- B` iff `A |- []B`, which yields the counit `<>[]A |- A` and the unit `A |- []<>A`. A `Dia` formula in a proof unpacks to a bracketed structure `<D>`, and the postulate P lets a bracketed structure, and nothing else, move rightward past its neighbor. A hypothesis typed `<>[]NP` can thus start wherever an `NP` is needed, move to the right edge, and be discharged there by `/R`; the counit turns it back into an `NP` at its original position. Unmarked material never moves, so "Kim admires today Lee" stays ungrammatical.

## Walkthrough

The chart has no hypotheses of its own: a rule combines adjacent spans. Each rule is therefore a derived sequent of L with `Dia`, `Box`, and P, chosen so that the chart can use the postulate without hypothetical reasoning.

| Rule | Sequent | Derivation |
|---|---|---|
| `fwd_app` | `A/B, B \|- A` | `/L` |
| `bwd_app` | `B, B\A \|- A` | `\L` |
| `fwd_comp` | `A/B, B/C \|- A/C` | `/L` twice, then `/R` |
| `raise_subj` | `NP \|- S/(NP\S)` | `\L`, then `/R` |
| `counit_arg` | `A/B \|- A/<>[]B` | the counit `<>[]B \|- B` under `/L`, then `/R` |
| `perm_arg` | `B\A \|- A/<>[]B` | `<[]B>, B\A \|- A` by `[]L` and `\L`; P moves `<[]B>` to the right; `<>L`, then `/R` |
| `x_comp` | `A/B, A\C \|- C/<>[]B` | `A/B, <[]B>, A\C \|- C` by `[]L`, `/L`, `\L`; P moves `<[]B>` to the right; `<>L`, then `/R` |

`counit_arg` is a residuation consequence: an argument slot that accepts `B` also accepts the weaker `<>[]B`. `perm_arg` and `x_comp` are the chart's image of P. With `perm_arg`, "admires Kim" (`NP\S`) also derives `S/<>[]NP`, a sentence whose missing subject is a marked gap, which gives the subject relative "linguist that admires Kim". With `x_comp`, "admires today" derives `(NP\S)/<>[]NP`, and composition with the raised subject gives "Kim admires today" the type `S/<>[]NP` that `that` seeks. All three conclusions end in `<>[]B`, and because `B |- <>[]B` is not a theorem, no undecorated argument can fill the slot they create: the reordering stays confined to marked gaps.

The `depth=3` option bounds the constructor depth of rule conclusions. It admits every category these rules need and stops `counit_arg` from wrapping an argument twice (`A/<>[]<>[]B`), which would add derivations but no new strings. No rule closes a cycle (each unary rule strictly enlarges its category), so no rule is `#[bounded]` and the chart needs no convergence tolerance. `fit_grammar` parses the data parameter `sentence` and scores the chart's goal weight, the inside log-sum over every derivation of `S` of the learnable log-weights it uses.

## Try it

The first block implements an independent decision procedure for L with `Dia`, `Box`, and P (cut-free backward search) and checks residuation, every chart rule's sequent, and the failure of the unmarked counterparts of `perm_arg` and `x_comp`.

```python
import itertools
from functools import lru_cache

import torch
from quivers.dsl import load, loads

SOURCE = "docs/examples/source/multimodal_tlg.qvr"
text = open(SOURCE).read()
ded = load(SOURCE).deductions["MMTLG"]

atom = lambda name: ("atom", name)
S, NP, N = atom("S"), atom("NP"), atom("N")
over = lambda a, b: ("Fwd", a, b)    # a/b
under = lambda b, a: ("Bwd", b, a)   # b\a
dia_box = lambda a: ("Dia", ("Box", a))
GAP = dia_box(NP)

LEX = {
    "Kim": NP, "Lee": NP, "every": over(NP, N), "linguist": N,
    "sleeps": under(NP, S), "admires": over(under(NP, S), NP),
    "today": under(under(NP, S), under(NP, S)),
    "that": over(under(N, N), over(S, GAP)),
}
for word, category in LEX.items():
    assert category in {item[3] for item, _ in ded([word]).chart.items()}, word

BRACKET = "<>"   # the structure <D> that a Dia formula unpacks to


@lru_cache(maxsize=None)
def proves(gamma, goal):
    """Decide gamma |- goal in L with the residuated pair Dia, Box and
    the controlled permutation P, by cut-free backward search. Antecedent
    elements are formulas or bracketed structures (BRACKET, elements).
    With this lexicon an antecedent holds at most one bracket, around a
    single Box formula, so the left rules act at the top level and P
    moves that bracket past formulas only."""
    if len(gamma) == 1 and gamma[0] == goal and goal[0] == "atom":
        return True
    head = goal[0]
    if head == "Fwd" and proves(gamma + (goal[2],), goal[1]):
        return True                                                    # /R
    if head == "Bwd" and proves((goal[1],) + gamma, goal[2]):
        return True                                                    # \R
    if head == "Dia" and len(gamma) == 1 and gamma[0][0] == BRACKET \
            and proves(gamma[0][1], goal[1]):
        return True                                                    # <>R
    if head == "Box" and proves(((BRACKET, gamma),), goal[1]):
        return True                                                    # []R
    for i, f in enumerate(gamma):
        left, right = gamma[:i], gamma[i + 1:]
        if f[0] == "Fwd" and any(                                      # /L
            proves(gamma[i + 1:j], f[2]) and proves(left + (f[1],) + gamma[j:], goal)
            for j in range(i + 2, len(gamma) + 1)
        ):
            return True
        if f[0] == "Bwd" and any(                                      # \L
            proves(gamma[k:i], f[1]) and proves(gamma[:k] + (f[2],) + right, goal)
            for k in range(i)
        ):
            return True
        if f[0] == "Dia" and proves(left + ((BRACKET, (f[1],)),) + right, goal):
            return True                                                # <>L
        if f[0] == BRACKET and len(f[1]) == 1 and f[1][0][0] == "Box" \
                and proves(left + (f[1][0][1],) + right, goal):
            return True                                                # []L
        # P, read upward: G, X, <D>, H follows from G, <D>, X, H. The
        # bracket only moves leftward in the search, which terminates.
        if f[0] == BRACKET and i > 0 and gamma[i - 1][0] != BRACKET \
                and proves(gamma[:i - 1] + (f, gamma[i - 1]) + right, goal):
            return True
    return False


# Residuation and its consequences.
A, B, C = atom("A"), atom("B"), atom("C")
assert proves((dia_box(A),), A)                    # counit <>[]A |- A
assert proves((A,), ("Box", ("Dia", A)))           # unit   A |- []<>A
assert not proves((A,), dia_box(A))                # the counit does not invert
rule_sequents = {
    "fwd_app": ((over(A, B), B), A),
    "bwd_app": ((B, under(B, A)), A),
    "fwd_comp": ((over(A, B), over(B, C)), over(A, C)),
    "raise_subj": ((NP,), over(S, under(NP, S))),
    "counit_arg": ((over(A, B),), over(A, dia_box(B))),
    "perm_arg": ((under(B, A),), over(A, dia_box(B))),
    "x_comp": ((over(A, B), under(A, C)), over(C, dia_box(B))),
}
for name, (gamma, goal) in rule_sequents.items():
    assert proves(gamma, goal), name
# Without the marking, the permuted forms are not theorems: P is the
# only source of reordering.
assert not proves((under(B, A),), over(A, B))
assert not proves((over(A, B), under(A, C)), over(C, B))
```

### What the modal decoration licenses

Two relative clauses parse only because `that` is decorated: a medial object gap ("Kim admires _ today") and a subject gap ("_ admires Kim"). Replacing `that : (N\N)/(S/<>[]NP)` with the undecorated `(N\N)/(S/NP)` makes both fail, in the chart and in the logic, while the right-peripheral gap still parses. The controls show that P does not leak into unmarked material, and the soundness check runs the decision procedure on every chart item. The last lines show a theorem the chart misses: P lets a gap cross any number of modifiers, but `x_comp` crosses one.

```python
def parses(d, sentence):
    return bool(d(sentence.split()).goal_items)


def derivable(lexicon, sentence):
    return proves(tuple(lexicon[w] for w in sentence.split()), S)


# The same grammar with the decoration removed from "that".
undecorated = "Fwd(Bwd(N, N), Fwd(S, NP))"
plain_text = text.replace("Fwd(Bwd(N, N), Fwd(S, Dia(Box(NP))))", undecorated)
assert plain_text != text
plain = loads(plain_text).deductions["MMTLG"]
PLAIN_LEX = dict(LEX, that=over(under(N, N), over(S, NP)))

licensed = [
    "every linguist that Kim admires today sleeps",   # medial gap, x_comp
    "every linguist that admires Kim sleeps",         # subject gap, perm_arg
]
for sentence in licensed:
    assert parses(ded, sentence) and derivable(LEX, sentence), sentence
    assert not parses(plain, sentence), sentence
    assert not derivable(PLAIN_LEX, sentence), sentence
    print(f"parses only with the modal license: {sentence!r}")

# A right-peripheral gap needs no reordering, so it parses either way:
# with the decoration through counit_arg, without it through plain L.
peripheral = "every linguist that Kim admires sleeps"
assert parses(ded, peripheral) and parses(plain, peripheral)

# Control: P moves only marked structure. With no gap in play nothing
# reorders, so a misplaced object or subject is still rejected, and a
# relative clause with its gap filled has an unused hypothesis.
for sentence in ["Kim admires today Lee", "Lee Kim admires",
                 "every linguist that Kim admires Lee sleeps",
                 "every linguist that admires sleeps"]:
    assert not parses(ded, sentence) and not derivable(LEX, sentence), sentence
    print(f"rejected: {sentence!r}")
# "admires today" does derive (NP\S)/<>[]NP, but an undecorated NP
# cannot fill its argument slot.
assert ("span", 1, 3, over(under(NP, S), GAP)) in dict(ded("Kim admires today Lee".split()).chart.items())
assert not proves((NP,), GAP)

# Soundness: every chart item is a theorem over the words of its span.
checked = 0
for sentence in [*licensed, peripheral, "Kim admires today Lee", "Kim admires Lee today"]:
    words = sentence.split()
    for (_, i, j, category), _ in ded(words).chart.items():
        assert proves(tuple(LEX[w] for w in words[i:j]), category), (sentence, i, j, category)
        checked += 1
print(f"{checked} chart items, every one a theorem over its span")

# Incompleteness: P lets a gap cross any number of modifiers, but the
# chart's x_comp crosses one, so this theorem has no chart derivation.
two_modifiers = "every linguist that Kim admires today today sleeps"
assert derivable(LEX, two_modifiers) and not parses(ded, two_modifiers)
```

### Recovering a distribution over sentences

As in the type-logical example, conditioning on length turns the goal weight into a distribution, $p(s \mid |s| = n, \mathbf{w}) = Z(s; \mathbf{w}) / \sum_{|s'| = n} Z(s'; \mathbf{w})$, which [`sample_corpus`](../api/stochastic/deduction/sample.md#quivers.stochastic.deduction.sample.sample_corpus) samples exactly. The block draws the learnable log-weights from a Normal(0, 1) prior, samples 2,000 three-word sentences, refits by MAP from zero, and requires the fitted distribution over grammatical yields to lie within total-variation distance 0.05 of the generating one.

```python
from quivers.stochastic.deduction import sample_corpus

torch.manual_seed(0)
yields = [
    list(words) for words in itertools.product(list(LEX), repeat=3)
    if ded(list(words)).goal_items
]
print(f"{len(yields)} grammatical yields of length 3")


def log_probs():
    """log p(s) = log Z(s) - log sum_s' Z(s') over the length-3 yields."""
    log_z = torch.stack([ded(words).goal_weight() for words in yields])
    return log_z - torch.logsumexp(log_z, dim=0)


params = list(ded.parameters())
with torch.no_grad():
    for p in params:
        p.copy_(torch.randn(()))
    true_probs = log_probs().exp()

corpus = sample_corpus(ded, length=3, n_samples=2000, seed=0)
index = {tuple(words): k for k, words in enumerate(yields)}
counts = torch.bincount(
    torch.tensor([index[tuple(s)] for s in corpus]), minlength=len(yields)
).to(torch.get_default_dtype())

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
tv = 0.5 * float((fitted_probs - true_probs).abs().sum())
print(f"TV(fitted, true) = {tv:.3f}")
assert tv < 0.05
```

## Categorical perspective

In a residuated monoid with a unary residuated pair, `<>` is left adjoint to `[]` (`<>A |- B` iff `A |- []B`), so `<>[]A |- A` is the [counit](https://ncatlab.org/nlab/show/adjunction) and `A |- []<>A` the unit of the adjunction. P is an additional arrow `B * <>A |- <>A * B`, an exchange map whose second factor must lie in the image of `<>`. No other arrow reorders a product, which is the algebraic content of "only marked structure permutes"; the chart's rules are composites of these arrows with the biclosed structure of L.

## Linguistic applications

- **Medial extraction.** "every linguist that Kim admires today sleeps": the gap precedes the modifier, which L rejects and the decorated relative pronoun licenses.
- **Subject relatives.** "every linguist that admires Kim sleeps": one relative-pronoun type serves subject and object gaps, because P lets the left-edge gap move to the right edge.
- **Word order outside the gap.** "Kim admires today Lee" and "Lee Kim admires" stay ungrammatical, since no unmarked material reorders.

The postulate lets a gap appear anywhere inside the relative clause, so the fragment models no island constraints; Kurtonina and Moortgat discuss postulate packages that restrict where marked material may travel.
