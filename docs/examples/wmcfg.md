# Weighted Multiple Context-Free Grammar

## Overview

A Multiple Context-Free Grammar (MCFG) [@seki1991mcfg] generalizes a context-free grammar by letting a nonterminal derive a *tuple* of strings rather than one string. Each nonterminal $A$ has a fixed rank $k(A) \ge 1$, the number of components it derives, and a production combines its premises' components by a *linear* yield function: every component is used exactly once and in a fixed order. The rank-1 case is CFG. Ranks $\ge 2$ represent discontinuous constituents. This example attaches a learnable log-weight to every production, giving a weighted MCFG whose chart computes the inside weight $\log Z(s) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$.

The grammar covers Swiss German cross-serial verb clusters, Shieber's evidence that natural language is not context-free [@shieber1985evidence]. In

> mer d'chind em Hans es huus lönd hälfe aastriiche
> ("we let the children help Hans paint the house")

the $i$-th object is governed by the $i$-th verb: *lönd* ("let") takes the accusative *d'chind*, *hälfe* ("help") the dative *em Hans*, and *aastriiche* ("paint") the accusative *es huus*. The dependencies cross instead of nesting, and case agreement makes the crossing visible in the string. Shieber shows that intersecting Swiss German with a regular language yields the pattern $w\,a^m b^n x\,c^m d^n y$, which is not context-free. A CFG can thus at best approximate this construction, while an MCFG generates it directly.

## QVR source

```qvr
# Weighted Multiple Context-Free Grammar
#
# A multiple context-free grammar (MCFG) whose productions carry
# learnable log-weights. A rank-k nonterminal of an MCFG derives a
# k-tuple of strings, and a production combines its premises'
# components by a linear yield function: every component is used
# exactly once, in a fixed order. The rank-1 case is CFG.
#
# The grammar covers Swiss German cross-serial verb clusters
# (Shieber 1985), the standard evidence that natural language is
# not context-free. In
#
#     mer d'chind em Hans es huus lönd hälfe aastriiche
#     we  the children(ACC) Hans(DAT) the house(ACC) let help paint
#     "we let the children help Hans paint the house"
#
# the i-th noun phrase is the object of the i-th verb, and each
# verb fixes its object's case: lönd ("let") and aastriiche
# ("paint") take accusatives, hälfe ("help") a dative. The
# dependencies cross rather than nest, and the case agreement makes
# them visible in the string.
#
# The rank-2 nonterminal vc pairs a run of objects with the run of
# verbs that take them:
#
#     vc(I1, J1, I2, J2)   objects span [I1, J1), verbs span [I2, J2)
#
# vc_acc and vc_dat realize the yield function
#
#     VC(np x1, v x2) :- NP_c(np), V_c(v), VC(x1, x2)
#
# where the object np and the verb v agree in case c. The inner
# item's two components are separated by the verb v, which this
# rule places between them, so every vc item below the top of the
# cluster is discontinuous: in the sentence above,
# vc(2, 4, 7, 8) covers "em Hans" and "hälfe" with "es huus lönd"
# in between. fin_acc and fin_dat close the cluster with its finite
# first verb, making the two components adjacent. A rule's weight
# is keyed by its category bindings; span positions are not part of
# the key.

object Term : FinSet 16

object LogWeight : Real 1

deduction WMCFG : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, VC, Nom, Dat, Acc, DetDat, DetAcc, Noun, VDat, VAcc, VDatFin, VAccFin, vc, span, mer, chind, em, es, hans, huus, loend, laa, haelfed, haelfe, aastriiche
    # Noun phrases: the article carries the case.
    rule dat_np : span(I, K, DetDat), span(K, J, Noun) |- span(I, J, Dat) #[learnable]
    rule acc_np : span(I, K, DetAcc), span(K, J, Noun) |- span(I, J, Acc) #[learnable]
    # Innermost object-verb pair: VC(np, v) :- NP_c(np), V_c(v).
    rule vc_acc_base : span(I1, J1, Acc), span(I2, J2, VAcc) |- vc(I1, J1, I2, J2) #[learnable]
    rule vc_dat_base : span(I1, J1, Dat), span(I2, J2, VDat) |- vc(I1, J1, I2, J2) #[learnable]
    # VC(np x1, v x2) :- NP_c(np), V_c(v), VC(x1, x2).
    rule vc_acc : span(I, J, Acc), vc(J, K, M, P), span(L, M, VAcc) |- vc(I, K, L, P) #[learnable]
    rule vc_dat : span(I, J, Dat), vc(J, K, M, P), span(L, M, VDat) |- vc(I, K, L, P) #[learnable]
    # The finite verb opens the cluster: VC(np x1 v x2) :- NP_c(np),
    # Vfin_c(v), VC(x1, x2), a rank-1 constituent.
    rule fin_acc : span(I, J, Acc), vc(J, K, M, P), span(K, M, VAccFin) |- span(I, P, VC) #[learnable]
    rule fin_dat : span(I, J, Dat), vc(J, K, M, P), span(K, M, VDatFin) |- span(I, P, VC) #[learnable]
    rule clause : span(I, J, Nom), span(J, K, VC) |- span(I, K, S) #[learnable]
    lexicon
        "mer"        : Nom     = mer        #[learnable]
        "d'chind"    : Acc     = chind      #[learnable]
        "em"         : DetDat  = em         #[learnable]
        "es"         : DetAcc  = es         #[learnable]
        "Hans"       : Noun    = hans       #[learnable]
        "huus"       : Noun    = huus       #[learnable]
        "lönd"       : VAccFin = loend      #[learnable]
        "laa"        : VAcc    = laa        #[learnable]
        "hälfed"     : VDatFin = haelfed    #[learnable]
        "hälfe"      : VDat    = haelfe     #[learnable]
        "aastriiche" : VAcc    = aastriiche #[learnable]

# The grammar's contribution to a probabilistic program: parsing an
# observed sentence and scoring the chart's goal weight adds the
# inside log-weight log Z(s), summed over every derivation of S, to
# the program's log-joint. The program returns that one real number,
# so its codomain is LogWeight : Real 1.
program score_sentence : Term -> LogWeight
    let chart = parse(WMCFG, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export score_sentence
```

## Understanding the model

A [context-free chart item](pcfg.md#understanding-the-model) covers one
continuous interval. An MCFG item of rank 2 holds two intervals that belong to
one constituent. Here the rank-2 nonterminal `vc` pairs a run of objects with
the run of verbs that take them: `vc(I1, J1, I2, J2)` says that the objects
span `[I1, J1)` and their verbs span `[I2, J2)`.

The recursive productions realize the yield function

$$
\mathrm{VC}(np\,x_1,\ v\,x_2) \;\leftarrow\; \mathrm{NP}_c(np),\ \mathrm{V}_c(v),\ \mathrm{VC}(x_1, x_2),
$$

which prefixes a new object to the first component and its verb to the second,
with $np$ and $v$ agreeing in case $c$. The verb $v$ lands between the inner
item's two components, so every `vc` item below the top of the cluster is
discontinuous. In the example sentence the chart holds `vc(2, 4, 7, 8)`:
`em Hans` and `hälfe`, with `es huus lönd` between them. The closing
productions `fin_acc` and `fin_dat` add the finite verb that opens the cluster
and make the two components adjacent, yielding an ordinary rank-1 `VC` span.

The weighted chart adds log weights along each derivation and uses
[log-sum-exp](../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine) to combine alternative derivations of the same item. The goal
weight $Z(s)$ is thus not a probability. Over the finitely many strings of a
fixed length $n$ it normalizes into the *length-conditional distribution*
$P(s \mid n) = Z(s) / \sum_{|s'| = n} Z(s')$, which is the model the fit below
recovers.

## Walkthrough

`object Term : FinSet 16` indexes the deduction's domain and codomain. The chart reasons symbolically over constructor-tagged tuples, so its cardinality is incidental.

`dat_np` and `acc_np` build case-marked noun phrases from an article and a noun, as in *em Hans* (dative) and *es huus* (accusative). *d'chind* is an accusative noun phrase in one word. The noun category is spelled `Noun` rather than `N` because an atom's spelling cannot double as a pattern variable.

`vc_acc_base` and `vc_dat_base` pair the innermost object with its verb. Their two premises are unconstrained in position, so the chart holds a `vc` item for every case-matched object-verb pair in the input. Only pairs that the later productions can place survive into a derivation of `S`. `vc_acc` and `vc_dat` are the recursive productions above: the shared variables `J`, `K`, `M`, `P` say that the new object ends where the inner objects begin, the inner objects end where the outer pair's first component ends, and the new verb ends where the inner verbs begin.

`fin_acc` and `fin_dat` require the finite verb to sit exactly between the objects and the infinitives (`span(K, M, ...)`), which closes the discontinuity, and `clause` attaches the subject *mer*. Every production is linear: each premise interval appears exactly once in the conclusion, in input order.

Every production and lexical entry carries `#[learnable]`. A learnable rule's log-weight is keyed by the binding of its category variables, the variables that do not fill a chart-position slot ([Weighted Deduction Fragment §2.1](../semantics/grammar.md#21-learnable-rule-weights-learnable)), and is allocated the first time that binding fires. Span positions are not part of the key, so one production firing at different positions reads one weight.

The `score_sentence` program connects the grammar to a probabilistic program: `parse(WMCFG, sentence)` runs the deduction on an observed sentence, and `score log_Z = chart.goal_weight()` adds $\log Z(s)$ to the program's log-joint. That is an unnormalized factor. The fit below divides by the length-$n$ total explicitly.

## DSL features

- **Tuple-valued chart items** for nonterminals of rank $\ge 2$. The `vc(I1, J1, I2, J2)` item is a four-position structural pattern. The chart engine pattern-matches it the same way it does the rank-1 `span(I, J, X)`.
- **Linear yield functions** as ordinary sequent rules. Concatenation and interleaving of components are expressed by where each position variable appears in the premises and the conclusion.
- **Three-premise rules**: `vc_acc`, `vc_dat`, `fin_acc`, and `fin_dat` each combine three chart items.
- **`#[learnable]` weights on every production**, lexicon entries and rules alike.

## Try it

The deduction system is callable: `ded(sentence)` returns a [`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose `goal_weight()` is the differentiable log-weight $\log Z(s; \mathbf{w})$ summed over every derivation the start symbol licenses for the input.

### Cross-serial, not nested

The grammar accepts the cross-serial order with case-matched object-verb
pairs, and rejects the nested order of Standard German as well as a case
mismatch. The discontinuous item of the example sentence appears in its chart.

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/wmcfg.qvr")
ded = prog.deductions["WMCFG"]


def parses(sentence):
    return bool(torch.isfinite(ded(sentence.split()).goal_weight()))


accepted = [
    "mer em Hans es huus hälfed aastriiche",
    "mer d'chind em Hans es huus lönd hälfe aastriiche",
]
rejected = [
    # nested order: the verbs mirror the objects
    "mer em Hans es huus aastriiche hälfed",
    # case mismatch: hälfed takes a dative, es huus is accusative
    "mer es huus em Hans hälfed aastriiche",
    # one object too many for the verbs
    "mer d'chind em Hans es huus hälfed aastriiche",
]
for sentence in accepted:
    assert parses(sentence), sentence
for sentence in rejected:
    assert not parses(sentence), sentence

chart = ded("mer d'chind em Hans es huus lönd hälfe aastriiche".split())
vc_items = sorted(item for item, _ in chart.chart.items() if item[0] == "vc")
print(vc_items)
assert ("vc", 2, 4, 7, 8) in vc_items  # em Hans ... hälfe
```

### Recovering a length-conditional distribution

Brute-force enumeration of seven-word strings over the eleven-word vocabulary
would take some twenty million parses. The cross-serial language has a direct
description instead: *mer*, then $k \ge 2$ objects, then $k$ verbs, the first
finite and the rest infinitives, with the $i$-th verb matching the $i$-th
object's case. `cross_serial` enumerates it independently of the deduction,
and every string it produces must parse.

A copy of the grammar with every weight drawn from a unit Normal serves as the
ground truth. Its length-conditional distribution over the 28 seven-word
strings generates a corpus of 2,000 sentences, and a freshly loaded copy, with
every weight at zero, is fit by maximizing the same length-conditional
log-likelihood.

```python
import itertools

OBJECTS = {
    "acc": [["d'chind"], ["es", "Hans"], ["es", "huus"]],
    "dat": [["em", "Hans"], ["em", "huus"]],
}
FINITE = {"acc": ["lönd"], "dat": ["hälfed"]}
INFINITIVE = {"acc": ["laa", "aastriiche"], "dat": ["hälfe"]}


def cross_serial(length):
    """Every sentence of the cross-serial language with `length` words."""
    out = []
    for k in range(2, length):
        for cases in itertools.product(["acc", "dat"], repeat=k):
            for objects in itertools.product(*(OBJECTS[c] for c in cases)):
                if 1 + sum(map(len, objects)) + k != length:
                    continue
                verb_choices = [FINITE[cases[0]]] + [INFINITIVE[c] for c in cases[1:]]
                for verbs in itertools.product(*verb_choices):
                    words = [w for obj in objects for w in obj]
                    out.append(["mer"] + words + list(verbs))
    return out


strings = cross_serial(7)
print(len(strings), "seven-word sentences")
assert all(parses(" ".join(s)) for s in strings)

torch.manual_seed(0)
truth = load("docs/examples/source/wmcfg.qvr").deductions["WMCFG"]
model = load("docs/examples/source/wmcfg.qvr").deductions["WMCFG"]


def log_conditional(system):
    """log P(s | n = 7) for every seven-word sentence."""
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

draws = torch.multinomial(true_log_p.exp(), 2000, replacement=True)
counts = torch.bincount(draws, minlength=len(strings)).float()

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

`cross_serial` agrees with brute-force enumeration over the vocabulary at
length five, where the 161,051 candidate strings are still cheap enough to
parse. With 2,000 draws over
28 outcomes, the sampling error of the maximum-likelihood fit is near
$27 / (2 \cdot 2000) \approx 0.007$ nats, so a divergence below 0.05 indicates
that the fit has recovered the distribution. The weights themselves are not
identifiable, so the check is on the distribution over strings.

## Categorical perspective

An MCFG production with $k$-component nonterminals is a hyperedge in a directed hypergraph whose nodes are tuple-positioned chart items. When the derivation sum converges and the evaluation computes it correctly, the chart represents the total weight of derivations [@goodman1999semiring]. Each derivation contributes the sum of the log-weights of its rule firings. Cross-serial dependencies are recovered by the linear yield function that interleaves an outer object-verb pair with the two components of the inner `vc` item.

The framework imposes no built-in commitment to rank-1 (CFG) items. Weighted MCFG, LCFRS, and their probabilistic variants share the same chart implementation: only the rule patterns and their conclusion arities change.
