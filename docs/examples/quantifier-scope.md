# Quantifier Scope

## Overview

"Every linguist admires some model" has two readings: for every linguist there is some model they admire (surface scope, $\forall\exists$), and there is one model every linguist admires (inverse scope, $\exists\forall$). This example derives both from a single lexicon. Generalized quantifiers denote computations in the [continuation monad](https://ncatlab.org/nlab/show/continuation+monad), and the clause combines its subject and object computations by the monad's applicative structure, which has two evaluation orders: left to right yields surface scope and right to left yields inverse scope [@barker2002continuations; @shan2006explaining]. The weighted chart derives exactly two sentence items for the doubly quantified sentence, carrying the two readings as logical forms, and the Try-it fits a reading preference from truth-value judgments by marginalizing the reading through the chart.

## QVR source

```qvr
# Quantifier Scope from Evaluation Order in the Continuation Monad
#
# A generalised quantifier has type (e -> t) -> t, which is Cont(e),
# the continuation monad Cont(A) = (A -> t) -> t at A = e. A
# determiner maps a noun, of type e -> t, to such a computation, so
# "every" has type (e -> t) -> Cont(e). Composing a clause whose
# subject and object both denote Cont(e) computations needs the
# monad's applicative structure, and that structure has two lawful
# evaluation orders. Evaluating the subject first gives surface scope;
# evaluating the object first gives inverse scope (Barker 2002;
# Shan and Barker 2006). The ambiguity is thus a choice between two
# derivations, not between two lexical entries.
#
# Chart items are span(I, J, Cat, LF). Categories use the argument-
# under convention (Fwd(A, B) = A/B, Bwd(B, A) = B\A) and Cont(A)
# marks a span that denotes a Cont(A) computation. The deduction
# runtime substitutes into rule conclusions and never beta-reduces,
# so every rule builds its logical form directly in beta-normal form:
#
#   fwd_app  : A/B : F,  B : X           |- A : F(X)
#   bwd_app  : B : X,    B\A : F         |- A : F(X)
#   map_obj  : A/B : F,  Cont(B) : G     |- Cont(A) : Fmap(F, G)
#   lower_s  : Cont(B) : G,  B\S : F     |- S : G(F)
#   lower_o  : B : X,  Cont(B\S) : Fmap(F, G)
#                                        |- S : G(\y. F(y)(X))
#   surface  : Cont(B) : H,  Cont(B\S) : Fmap(F, G)
#                                        |- S : H(\x. G(\y. F(y)(x)))
#   inverse  : Cont(B) : H,  Cont(B\S) : Fmap(F, G)
#                                        |- S : G(\y. H(\x. F(y)(x)))
#
# Fmap(F, G) is the normal form of map F G = \k. G(\y. k(F(y))): the
# pure function F mapped over the object's computation G. lower_s and
# lower_o are map followed by lowering (running the computation with
# the identity continuation). surface and inverse are the two
# applicative applications, left-to-right and right-to-left, followed
# by lowering. A sentence with one quantifier has one derivation and
# one reading; a transitive sentence with two has exactly two.
#
# The determiners denote GQ constants: App(App(every_q, P), Q) is
# every(P)(Q), true when every P is a Q, and some_q likewise. Relations
# are curried object first, App(App(admire_p, y), x) for "x admires y".
# A rule's conclusion may not introduce a fresh name, so the clause
# rules bind the reserved names x (subject) and y (object), declared
# as atoms. Every term substituted under those binders is a closed
# constant term from the lexicon, so no capture can occur.

object Term : FinSet 16

object LogWeight : Real 1

deduction QScope : Term -> Term [semiring=LogProb, start=S, depth=4]
    atoms S, NP, N, Fwd, Bwd, Cont, span, App, Var, Lam, Fmap, x, y, every_q, some_q, linguist_p, model_p, sleep_p, admire_p, kim, lee
    rule fwd_app : span(I, K, Fwd(A, B), F), span(K, J, B, X) |- span(I, J, A, App(F, X)) #[learnable]
    rule bwd_app : span(I, K, B, X), span(K, J, Bwd(B, A), F) |- span(I, J, A, App(F, X)) #[learnable]
    rule map_obj : span(I, K, Fwd(A, B), F), span(K, J, Cont(B), G) |- span(I, J, Cont(A), Fmap(F, G)) #[learnable]
    rule lower_s : span(I, K, Cont(B), G), span(K, J, Bwd(B, S), F) |- span(I, J, S, App(G, F)) #[learnable]
    rule lower_o : span(I, K, B, X), span(K, J, Cont(Bwd(B, S)), Fmap(F, G)) |- span(I, J, S, App(G, Lam(y, App(App(F, Var(y)), X)))) #[learnable]
    rule surface : span(I, K, Cont(B), H), span(K, J, Cont(Bwd(B, S)), Fmap(F, G)) |- span(I, J, S, App(H, Lam(x, App(G, Lam(y, App(App(F, Var(y)), Var(x))))))) #[learnable]
    rule inverse : span(I, K, Cont(B), H), span(K, J, Cont(Bwd(B, S)), Fmap(F, G)) |- span(I, J, S, App(G, Lam(y, App(H, Lam(x, App(App(F, Var(y)), Var(x))))))) #[learnable]
    lexicon
        "every"    : Fwd(Cont(NP), N)    = every_q    #[learnable]
        "some"     : Fwd(Cont(NP), N)    = some_q     #[learnable]
        "linguist" : N                   = linguist_p #[learnable]
        "model"    : N                   = model_p    #[learnable]
        "sleeps"   : Bwd(NP, S)          = sleep_p    #[learnable]
        "admires"  : Fwd(Bwd(NP, S), NP) = admire_p   #[learnable]
        "Kim"      : NP                  = kim        #[learnable]
        "Lee"      : NP                  = lee        #[learnable]

# The program scores a sentence by the chart's goal weight: the log-sum
# over every derivation of S, and thus over both scope readings of a
# doubly quantified sentence, of the learnable log-weights each
# derivation uses. Its value is one real number, so the codomain is
# LogWeight : Real 1. The token sequence enters as the data parameter
# ``sentence``.
program fit_scope : Term -> LogWeight
    let chart = parse(QScope, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit_scope
```

## Understanding the model

A generalized quantifier such as "every linguist" denotes a function from properties to truth values, type $(e \to t) \to t$ [@barwise1981quantifiers]. That is $\mathrm{Cont}(e)$, where $\mathrm{Cont}(A) = (A \to t) \to t$ is the continuation monad at answer type $t$: a quantified noun phrase is a computation that will produce an individual once it is handed the rest of the sentence. A determiner maps a noun, type $e \to t$, to such a computation, so its type is $(e \to t) \to \mathrm{Cont}(e)$, which unfolds to $(e \to t) \to ((e \to t) \to t)$.

When only one constituent is a computation, composition is the monad's map: apply the pure function under the computation, then lower by running it with the identity continuation. "Every linguist sleeps" comes out as $\mathsf{every}(\mathsf{linguist})(\mathsf{sleep})$ and nothing else. When both the subject and the object are computations, the clause must sequence them, and the applicative application of a function computation to an argument computation can run either one first. Running the subject first gives $\mathsf{every}(\mathsf{linguist})(\lambda x.\, \mathsf{some}(\mathsf{model})(\lambda y.\, \mathsf{admire}(y)(x)))$. Running the object first gives $\mathsf{some}(\mathsf{model})(\lambda y.\, \mathsf{every}(\mathsf{linguist})(\lambda x.\, \mathsf{admire}(y)(x)))$. The scope ambiguity is a choice between two derivations over the same lexical entries, which is what a weighted chart can score.

## Walkthrough

The deduction runtime builds a rule's conclusion by substitution and never beta-reduces (the [Montague NLI example](montague-nli.md#why-the-sentence-lf-is-built-in-normal-form) explains why). Each rule thus states the beta-normal form of its result directly, and each corresponds to one piece of the monadic structure:

| Rule | Premises and conclusion | Monadic reading |
|---|---|---|
| `fwd_app`, `bwd_app` | `A/B : F, B : X \|- A : F(X)` and its mirror image | ordinary application of pure values. The determiner applied to its noun yields `Cont(NP)` |
| `map_obj` | `A/B : F, Cont(B) : G \|- Cont(A) : Fmap(F, G)` | map: `Fmap(F, G)` is the normal form of $\lambda k.\, G(\lambda y.\, k(F\,y))$ |
| `lower_s` | `Cont(B) : G, B\S : F \|- S : G(F)` | map then lower, with the computation as subject |
| `lower_o` | `B : X, Cont(B\S) : Fmap(F, G) \|- S : G(λy. F(y)(X))` | map then lower, with the computation inside the verb phrase |
| `surface` | `Cont(B) : H, Cont(B\S) : Fmap(F, G) \|- S : H(λx. G(λy. F(y)(x)))` | left-to-right applicative application, then lower |
| `inverse` | `Cont(B) : H, Cont(B\S) : Fmap(F, G) \|- S : G(λy. H(λx. F(y)(x)))` | right-to-left applicative application, then lower |

To check the last two rows, write the subject computation as $s$ and the verb phrase computation as $f = \lambda k.\, G(\lambda y.\, k(F\,y))$. Left-to-right application is $\lambda k.\, s(\lambda x.\, f(\lambda g.\, k(g\,x)))$, which reduces to $\lambda k.\, H(\lambda x.\, G(\lambda y.\, k(F\,y\,x)))$. Lowering with $k = \mathrm{id}$ gives the `surface` conclusion. Right-to-left application is $\lambda k.\, f(\lambda g.\, s(\lambda x.\, k(g\,x)))$, which lowers to the `inverse` conclusion.

There is no explicit unit rule. The applicative law $\mathrm{pure}\,f \circledast x = \mathrm{map}\,f\,x$ means that lifting a pure verb phrase with a unit rule and then applying would rederive exactly what `lower_s` derives, so the chart would hold two derivations of one reading and double its weight. The fragment keeps only the map form, and the Try-it checks with the Counting [semiring](https://ncatlab.org/nlab/show/rig) that every reading has exactly one derivation. A sentence with one quantifier thus has one reading, and a sentence with two has exactly two.

Categories use Lambek notation [@lambek1958mathematics], as in the [type-logical example](type-logical.md): `Fwd(A, B)` is `A/B` and `Bwd(B, A)` is `B\A`. Logical forms use the determiner constants `every_q` and `some_q` applied to a restrictor and a scope, and curried relations with the object first, so `App(App(admire_p, y), x)` says that `x` admires `y`. A rule conclusion cannot mint a fresh variable, so the two clause rules bind the reserved names `x` and `y`, declared as atoms. Every term substituted under those binders is a closed constant from the lexicon, so no capture can occur, and the Try-it checks that both readings are closed. No rule closes a cycle and the `depth=4` bound admits every category the lexicon needs.

`fit_scope` parses the data parameter `sentence` and scores the chart's goal weight, which for a doubly quantified sentence is the log-sum over both readings. That is the unnormalized string weight with the reading summed out. The judgment model below reads the per-reading weights from the same chart.

## Try it

The first block reads the logical forms of the full-sentence `S` items and checks them against hand-built terms: the two readings for the doubly quantified sentence and a single reading for each sentence with at most one quantifier. Both readings are in beta-normal form and closed.

```python
import itertools
import re

import torch
from quivers.dsl import load, loads

SOURCE = "docs/examples/source/quantifier_scope.qvr"
ded = load(SOURCE).deductions["QScope"]

# Logical-form constructors, mirroring the runtime's term encoding.
const = lambda name: ("atom", name)
app = lambda f, a: ("App", f, a)
lam = lambda v, body: ("Lam", const(v), body)
var = lambda v: ("Var", const(v))


def readings(sentence):
    """The logical forms of the S items spanning the whole sentence."""
    return [item[4] for item, _ in ded(sentence.split()).goal_items]


every_linguist = app(const("every_q"), const("linguist_p"))
some_model = app(const("some_q"), const("model_p"))
admires = lambda obj, subj: app(app(const("admire_p"), obj), subj)

# forall x. linguist(x) -> exists y. model(y) & admire(x, y)
SURFACE = app(every_linguist, lam("x", app(some_model, lam("y", admires(var("y"), var("x"))))))
# exists y. model(y) & forall x. linguist(x) -> admire(x, y)
INVERSE = app(some_model, lam("y", app(every_linguist, lam("x", admires(var("y"), var("x"))))))

two = readings("every linguist admires some model")
assert len(two) == 2 and set(two) == {SURFACE, INVERSE}
for sentence, expected in [
    ("every linguist sleeps", app(every_linguist, const("sleep_p"))),
    ("every linguist admires Lee", app(every_linguist, app(const("admire_p"), const("lee")))),
    ("Kim admires some model", app(some_model, lam("y", admires(var("y"), const("kim"))))),
    ("Kim admires Lee", admires(const("lee"), const("kim"))),
]:
    assert readings(sentence) == [expected], sentence
print("two readings for the doubly quantified sentence, one for each other sentence")


def no_redex(term):
    """True when no App node has a Lam in function position."""
    if term[0] == "App":
        return term[1][0] != "Lam" and no_redex(term[1]) and no_redex(term[2])
    if term[0] == "Lam":
        return no_redex(term[2])
    return True


def free_vars(term, bound=frozenset()):
    if term[0] == "Var":
        return set() if term[1] in bound else {term[1]}
    if term[0] == "Lam":
        return free_vars(term[2], bound | {term[1]})
    if term[0] == "App":
        return free_vars(term[1], bound) | free_vars(term[2], bound)
    return set()


for lf in two:
    assert no_redex(lf) and not free_vars(lf)
```

The second block counts derivations in the Counting semiring and interprets the readings model-theoretically. Over every admiration relation between two linguists and two models, the inverse reading entails the surface reading and some relations make the surface reading true and the inverse reading false, as the logic requires.

```python
# Count derivations with every weight at the semiring's one: each
# reading must have exactly one, so no reading's weight is inflated by
# a spurious duplicate derivation.
grammar_only = open(SOURCE).read().split("# The program scores")[0]
unweighted = re.sub(r"\s*#\[learnable\]", "", grammar_only)
count = loads(unweighted.replace("semiring=LogProb", "semiring=Counting")).deductions["QScope"]
for sentence, n_readings in [
    ("every linguist admires some model", 2),
    ("some model admires every linguist", 2),
    ("every linguist sleeps", 1),
    ("Kim admires some model", 1),
]:
    per_reading = [int(w) for _, w in count(sentence.split()).goal_items]
    assert per_reading == [1] * n_readings, (sentence, per_reading)


def denote(term, model, env=None):
    """Interpret a logical form in a finite model."""
    env = env or {}
    tag = term[0]
    if tag == "atom":
        return model[term[1]]
    if tag == "Var":
        return env[term[1]]
    if tag == "Lam":
        return lambda d: denote(term[2], model, {**env, term[1]: d})
    return denote(term[1], model, env)(denote(term[2], model, env))


def make_model(linguists, models, admire):
    domain = linguists | models | {"kim", "lee"}
    return {
        "every_q": lambda p: lambda q: all(q(d) for d in domain if p(d)),
        "some_q": lambda p: lambda q: any(q(d) for d in domain if p(d)),
        "linguist_p": lambda d: d in linguists,
        "model_p": lambda d: d in models,
        "sleep_p": lambda d: False,
        "admire_p": lambda obj: lambda subj: (subj, obj) in admire,
        "kim": "kim", "lee": "lee",
    }


# The inverse reading entails the surface reading whenever there is a
# linguist, and not conversely: check every admiration relation between
# two linguists and two models.
L, M = {"l1", "l2"}, {"m1", "m2"}
pairs = list(itertools.product(sorted(L), sorted(M)))
separating = 0
for bits in itertools.product([False, True], repeat=len(pairs)):
    model = make_model(L, M, {p for p, b in zip(pairs, bits) if b})
    surface, inverse = denote(SURFACE, model), denote(INVERSE, model)
    assert surface or not inverse
    separating += surface and not inverse
assert separating > 0
print(f"inverse entails surface; {separating} of {2 ** len(pairs)} models separate them")
```

### A judgment model over readings

A listener hears the sentence in a scenario, resolves its scope, and reports whether the sentence is true. The report is wrong with probability $\epsilon$. The reading distribution is the chart's: $p(r \mid s) = \alpha[\mathrm{S}, \phi_r] / \sum_{r'} \alpha[\mathrm{S}, \phi_{r'}]$, the weight of the item carrying logical form $\phi_r$ over the goal weight. The truth value comes from evaluating $\phi_r$ in the scenario. In scenario A each linguist admires a different model, so only the surface reading is true. In scenario B every linguist admires the same model, so both are. The block sets the scope rules' weights so that the chart puts 0.7 on the surface reading and simulates 800 trials from that model.

```python
# Scenario A: each linguist admires a different model, so only the
# surface reading is true. Scenario B: everyone admires m1, so both are.
SCENARIOS = {
    "A": make_model({"l1", "l2", "l3"}, {"m1", "m2", "m3"},
                    {("l1", "m1"), ("l2", "m2"), ("l3", "m3")}),
    "B": make_model({"l1", "l2", "l3"}, {"m1", "m2", "m3"},
                    {("l1", "m1"), ("l2", "m1"), ("l3", "m1")}),
}
SENTENCE = "every linguist admires some model".split()
EPSILON = 0.1            # judgment noise: p(report the wrong truth value)
TRUE_PREFERENCE = 0.7    # p(surface reading)

items = [item for item, _ in ded(SENTENCE).goal_items]
truth = torch.tensor([[float(denote(item[4], SCENARIOS[s])) for item in items]
                      for s in ("A", "B")])                  # (scenario, reading)
surface_index = [item[4] for item in items].index(SURFACE)


def reading_log_probs():
    """log p(reading | sentence): each reading's chart weight, normalized
    over the sentence's readings by the chart's own goal weight."""
    chart = ded(SENTENCE)
    return torch.stack([w for _, w in chart.goal_items]) - chart.goal_weight()


def preference_parameters():
    named = dict(ded.named_parameters())
    surface = next(p for n, p in named.items() if n.startswith("_rule_module.rule_surface."))
    inverse = next(p for n, p in named.items() if n.startswith("_rule_module.rule_inverse."))
    return surface, inverse


# Ground truth: set the scope rules' weights so the chart puts
# TRUE_PREFERENCE on the surface reading.
surface_w, inverse_w = preference_parameters()
with torch.no_grad():
    for p in ded.parameters():
        p.zero_()
    surface_w.fill_(float(torch.logit(torch.tensor(TRUE_PREFERENCE))))
true_probs = reading_log_probs().exp().detach()
assert abs(float(true_probs[surface_index]) - TRUE_PREFERENCE) < 1e-6

# Simulate: each trial draws a scenario, a reading from the chart's
# distribution, and a judgment from that reading's truth value.
torch.manual_seed(0)
N_TRIALS = 800
scenario = torch.randint(0, 2, (N_TRIALS,))
reading = torch.multinomial(true_probs, N_TRIALS, replacement=True)
p_true = torch.where(truth[scenario, reading] > 0, 1 - EPSILON, EPSILON)
judgment = torch.bernoulli(p_true)
```

The fit starts from zero weights and maximizes $\sum_n \log \sum_r p(r \mid s)\, p(y_n \mid r, \text{scenario}_n)$, with the reading summed out through the chart. Scenario B trials carry no information about scope, and the marginal makes them drop out of the estimate: the fitted preference matches the closed-form maximum likelihood estimate from the scenario A trials alone, and it lies within 0.1 of the generating preference (the standard error at 400 scenario A trials is about 0.03).

```python
# Fit: the likelihood of a judgment marginalizes the reading through
# the chart, p(y | s, scenario) = sum_r p(r | s) p(y | r, scenario).
with torch.no_grad():
    for p in ded.parameters():
        p.zero_()
judge = torch.where(truth > 0, 1 - EPSILON, EPSILON)          # p(y = 1 | scenario, r)
p_y_given_r = torch.where(judgment[:, None] > 0, judge[scenario], 1 - judge[scenario])
optim = torch.optim.Adam(list(ded.parameters()), lr=0.05)
for step in range(400):
    optim.zero_grad()
    log_marginal = torch.logsumexp(reading_log_probs() + p_y_given_r.log(), dim=1)
    (-log_marginal.sum()).backward()
    optim.step()
fitted = float(reading_log_probs().exp()[surface_index])

# Only scenario A separates the readings; its closed-form maximum
# likelihood estimate is (mean judgment - EPSILON) / (1 - 2 EPSILON).
in_a = scenario == 0
closed_form = (float(judgment[in_a].mean()) - EPSILON) / (1 - 2 * EPSILON)
print(f"p(surface): true {TRUE_PREFERENCE}, fitted {fitted:.3f}, closed form {closed_form:.3f}")
assert abs(fitted - closed_form) < 0.01
assert abs(fitted - TRUE_PREFERENCE) < 0.1
```

## Categorical perspective

$\mathrm{Cont}$ is a [strong monad](https://ncatlab.org/nlab/show/strong+monad), and a strong monad is the structure [@moggi1991notions] uses to model computational effects. On a non-commutative monad such as $\mathrm{Cont}$, the two ways of using the strength to combine two computations give two applicative structures that differ in evaluation order. Quantification is such an effect: a quantified noun phrase acts on its continuation. The chart's `surface` and `inverse` rules are the two applicative applications followed by lowering at the answer type, and the LogProb chart sums over them as alternative derivations, so the goal weight sums over readings. Dividing each item weight by their total gives the conditional reading distribution. The [effect-driven interpretation](https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4) program of [@bumford2026effect] develops this functorial view of composition in general.

## Connections

The fragment reuses the slash categories of the [type-logical example](type-logical.md) and the normal-form discipline of the [Montague NLI example](montague-nli.md), whose prover could take these readings as premises.
