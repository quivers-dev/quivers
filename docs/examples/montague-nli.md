# Montague NLI

## Overview

A two-stage natural-language-inference architecture built entirely in QVR's
weighted-deduction fragment: a Montague-style grammar derives a logical form
for every token span, then an entailment prover closes those logical forms
under three syllogistic rules. The model declares everything through `atoms`,
`binders`, and `rule` sequents. QVR does not build in either the grammar
formalism or the proof system.

The two halves share one term language, so the prover's items are literally the terms the grammar builds. That is what lets a single Adam step move the lexicon's log-weights in response to an entailment error.

## QVR source

```qvr
# Montague Grammar + Syllogistic Entailment
#
# Two-stage natural-language-inference architecture composed
# entirely in QVR's weighted-deduction fragment:
#
#   1. A Montague-style grammar derives a logical form for every
#      token span. Common nouns and intransitive verbs denote
#      honest lambda terms over a user-declared free term algebra;
#      ``Lam`` is listed in the ``binders`` block, so the compiler
#      treats its first argument as a bound variable and
#      alpha-renames it to a fresh canonical symbol per term
#      construction. Structural equality on the chart is thus
#      alpha-equivalence in the source language.
#
#   2. An entailment prover closes the resulting claims under
#      three syllogistic rules. The prover's items are the
#      grammar's own logical forms wrapped in ``Claim``, so the
#      two deductions share one term language.
#
# Why the sentence LF is built in normal form
# -------------------------------------------
# A deduction rule's conclusion is a constructor tree over the
# rule's pattern variables, and the runtime instantiates it by
# structural substitution alone. There is no beta-rule: nothing
# reduces a redex, and the chart never normalizes an item. So a
# determiner cannot denote a continuation-form lambda term
# ``Lam(P, Lam(Q, ...))`` that later beta-reduces against its
# arguments; the chart would simply carry the unreduced redex
# forever, and no prover rule could see through it.
#
# The grammar thus builds the sentence LF in normal form
# directly, in the generalised-quantifier style of
# [Barwise and Cooper (1981)](https://doi.org/10.1007/BF00350139):
# a determiner denotes a relation between two sets, written here
# as the binary constructors ``Every(P, Q)`` and ``Some(P, Q)``.
# A determiner contributes a nullary LF constant (``every_q`` or
# ``some_q``); sentence rules inspect that constant to build the
# matching constructor. Restrictor and scope stay genuine lambda
# terms, so the ``binders`` declaration is doing real work at the
# predicate level.
#
# Reference: [Montague (1973)](https://doi.org/10.1007/978-94-010-2506-5_10).

object Term : FinSet 8

object LogWeight : Real 1

# ---------------------------------------------------------------------------
# Grammar deduction
# ---------------------------------------------------------------------------

deduction Montague : Term -> Term [semiring=LogProb, start=S, depth=12]
    # Categories, the chart-item constructor ``span``, the LF
    # constructors (``App``, ``Var``, ``Every``, ``Some``), and the
    # predicate constants all live in ``atoms``.
    atoms S, N, VP, Nom, Art, Cop, Det, QNP, span, App, Var, Every, Some, dog_p, cat_p, animal_p, bark_p, walk_p, every_q, some_q, an_q, is_q
    # ``Lam`` is the binder constructor: ``Lam(x, body)``. The
    # compiler alpha-renames ``x`` to a fresh canonical symbol per
    # lexicon entry, so the predicate a word denotes is one
    # canonical term wherever that word occurs.
    binders Lam
    # Determiner + common noun. One syntactic rule packages the
    # determiner's LF constant together with the restrictor.
    rule det_np : span(I, K, Det, D), span(K, J, N, P) |- span(I, J, QNP, App(D, P)) #[learnable]
    # Quantified NP + verb phrase. Matching ``every_q`` and
    # ``some_q`` inside the LF selects the force of the resulting
    # generalised quantifier, already in normal form.
    rule every_s : span(I, K, QNP, App(every_q, P)), span(K, J, VP, Q) |- span(I, J, S, Every(P, Q)) #[learnable]
    rule some_s : span(I, K, QNP, App(some_q, P)), span(K, J, VP, Q) |- span(I, J, S, Some(P, Q)) #[learnable]
    # Predicative "is an N": the article and the copula are
    # semantically vacuous, so both rules pass the noun's predicate
    # through unchanged.
    rule art_n : span(I, K, Art, D), span(K, J, N, P) |- span(I, J, Nom, P) #[learnable]
    rule cop_nom : span(I, K, Cop, D), span(K, J, Nom, P) |- span(I, J, VP, P) #[learnable]
    lexicon
        # Common nouns are unary predicates: ``λx. dog(x)``.
        "dog"    : N  = Lam(x, App(dog_p, Var(x)))      #[learnable]
        "cat"    : N  = Lam(x, App(cat_p, Var(x)))      #[learnable]
        "animal" : N  = Lam(x, App(animal_p, Var(x)))   #[learnable]
        # Intransitive verbs are unary predicates over individuals.
        "barks"  : VP = Lam(x, App(bark_p, Var(x)))     #[learnable]
        "walks"  : VP = Lam(x, App(walk_p, Var(x)))     #[learnable]
        # Determiners contribute their force as an LF constant;
        # the semantically vacuous article and copula are consumed
        # by rules that ignore their constants.
        "every"  : Det = every_q #[learnable]
        "some"   : Det = some_q  #[learnable]
        "an"     : Art      = an_q    #[learnable]
        "is"     : Cop      = is_q    #[learnable]

# ---------------------------------------------------------------------------
# Prover deduction
# ---------------------------------------------------------------------------
#
# The prover's items are ``Claim(phi)`` for ``phi`` a sentence LF
# the grammar built, plus ``Nonempty(P)`` for an explicit
# existence premise. All three rules are classically valid on the
# generalised-quantifier reading
#
#     Every(P, Q)  =  forall x. P(x) -> Q(x)
#     Some(P, Q)   =  exists x. P(x) & Q(x)
#     Nonempty(P)  =  exists x. P(x)
#
# ``ex_import`` is the one that needs care. ``Every(P, Q)`` alone
# does *not* entail ``Some(P, Q)``: on the standard first-order
# reading a universal over an empty restrictor is vacuously true,
# while the existential is false. The inference is valid only
# under existential import, so the rule takes ``Nonempty(P)`` as a
# second premise rather than smuggling the assumption in. A caller
# that wants the inference must provide that premise.

deduction Prover : Term -> Term [semiring=LogProb, depth=12]
    atoms Claim, Every, Some, Nonempty, App, Var
    binders Lam
    # Barbara: every P is Q, every Q is R, thus every P is R.
    # Valid with no existence assumption.
    rule barbara : Claim(Every(P, Q)), Claim(Every(Q, R)) |- Claim(Every(P, R)) #[learnable]
    # Darii: some P is Q, every Q is R, thus some P is R.
    # Valid; the existential premise carries the witness.
    rule darii : Claim(Some(P, Q)), Claim(Every(Q, R)) |- Claim(Some(P, R)) #[learnable]
    # Subalternation, licensed only by an explicit non-emptiness
    # premise on the restrictor.
    rule ex_import : Claim(Every(P, Q)), Claim(Nonempty(P)) |- Claim(Some(P, Q)) #[learnable]

# ---------------------------------------------------------------------------
# Fit program: the chart's goal weight is the inside log-marginal
# of the sentence at the start symbol under the current parameters.
# The program returns that log-marginal, one real number, so its
# codomain is LogWeight : Real 1 rather than the Term index the
# deduction ranges over.
# ---------------------------------------------------------------------------

program fit_grammar : Term -> LogWeight
    let chart = parse(Montague, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit_grammar
```

## Understanding the model

Start with two sentences: “every dog is an animal” and “every animal barks.”
The grammar builds `Every(DOG, ANIMAL)` and `Every(ANIMAL, BARK)`. The prover
can then derive `Every(DOG, BARK)`. Parsing provides the [logical forms](term-autoencoder.md#understanding-the-model) that
the prover uses as premises.

The grammar's weight follows its logical form into the proof, so a derived
claim can depend differentiably on lexical parameters. Derivability depends
on the provided rules and premises: deriving “some dog barks,” for instance,
requires evidence that dogs exist as well as a universal statement about them.
An absent derivation can reflect a missing premise or rule.

## Walkthrough

The grammar half declares the categories (`S`, `N`, `VP`, `Nom`, `QNP`, and the closed-class categories `Art`, `Cop`, `Det`), the chart-item constructor `span(I, J, X, F)` that packages a derivation covering tokens `[I, J)` of category `X` with logical form `F`, the logical-form constructors `App`, `Var`, `Every`, and `Some`, and the predicate constants `dog_p`, `cat_p`, `animal_p`, `bark_p`, `walk_p`. The `binders Lam` block tells the compiler that `Lam`'s first argument is a binding site, so every bound variable is alpha-renamed to a fresh canonical symbol per lexicon entry and structural equality on the chart uses [alpha-equivalence](https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence). This matters for the prover: because "dog" is compiled once, the predicate it denotes is one canonical term, and the prover's pattern variables bind it by structural equality wherever it occurs.

### Why the sentence LF is built in normal form

One constraint in the deduction language drives the design. A rule's conclusion
is a constructor tree over the rule's pattern variables, and the runtime
instantiates it with
[`instantiate`](../api/stochastic/agenda.md#quivers.stochastic.agenda.instantiate),
which performs structural substitution and nothing else. There is no beta
rule: a conclusion cannot call a function, no chart item is normalized on
entry, and a `binders` declaration only alpha-renames lexical logical forms at
compile time.

Thus a determiner cannot denote a continuation-form lambda term such as `Lam(P, Lam(Q, App(forall_t, ...)))` and rely on later beta-reduction against its arguments. Nothing would reduce the redex. The chart would carry `App(App(Lam(...), dog_LF), bark_LF)` forever, and no prover rule could pattern-match through the unreduced application to find the quantifier.

The grammar thus builds the sentence logical form in normal form directly, in the [generalized quantifier](https://en.wikipedia.org/wiki/Generalized_quantifier) style of Barwise and Cooper [@barwise1981quantifiers]: a determiner denotes a relation between two sets, written here as the binary constructors `Every(P, Q)` and `Some(P, Q)`. Both determiners have category `Det`. Their lexical constants distinguish their force. The shared `det_np` rule retains that constant in `App(D, P)`, and the `every_s` and `some_s` rules select the appropriate constructor by matching `every_q` or `some_q` inside that logical form. The restrictor and scope remain lambda terms. The binder declaration gives `Lam(x, App(dog_p, Var(x)))` a canonical representation shared by terms that differ only in the bound variable's name. See [@montague1973quantification] for the type-driven compositionality this fragment instantiates.

Two consequences follow, and both are visible in the source. First, `every` and `some` make a semantic contribution through `every_q` and `some_q`. Unlike a category split, this factorization keeps their shared syntax in one rule. Second, the article and the copula are semantically vacuous, so `art_n` and `cop_nom` pass the noun's predicate through unchanged, which is what makes `every dog is an animal` and `every animal barks` come out as `Every(DOG, ANIMAL)` and `Every(ANIMAL, BARK)` over the *same* `ANIMAL` term.

### The prover

The prover's items are `Claim(phi)` for `phi` a sentence logical form the grammar built, plus `Nonempty(P)` for an explicit existence premise. Read the constructors as

$$\mathtt{Every}(P, Q) \;=\; \forall x.\, P(x) \to Q(x), \qquad \mathtt{Some}(P, Q) \;=\; \exists x.\, P(x) \wedge Q(x), \qquad \mathtt{Nonempty}(P) \;=\; \exists x.\, P(x).$$

All three rules are classically valid on that reading. `barbara` (every $P$ is $Q$, every $Q$ is $R$, thus every $P$ is $R$) needs no existence assumption. `darii` (some $P$ is $Q$, every $Q$ is $R$, thus some $P$ is $R$) is valid because the existential premise carries its own witness.

The `ex_import` rule takes both `Every(P, Q)` and `Nonempty(P)` as premises to derive `Some(P, Q)`. On the standard first-order reading, “every dog barks” is true even if there are no dogs, while “some dog barks” requires a dog. The second premise provides this [existential import](https://en.wikipedia.org/wiki/Square_of_opposition#Existential_import). The Python example provides it for `DOG`.

### The fit program, and what `Term` and `LogWeight` each name

`Term : FinSet 8` declares the deduction carrier, but the chart matches constructor-tagged items symbolically. `fit_grammar` returns the real inside log-weight from [`goal_weight`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView.goal_weight), so its codomain is `LogWeight : Real 1`. The caller passes `sentence` as a data parameter containing the token sequence.

## Try it

Every `#[learnable]` lexicon entry and every `#[learnable]` rule exposes a real `nn.Parameter` on the compiled [`DeductionSystem`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem). The system is callable: `ded(sentence)` returns a [`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose [`goal_weight`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView.goal_weight) is the differentiable inside log-marginal at the start symbol.

### MAP fit on the grammar

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/montague_nli.qvr")
grammar = prog.deductions["Montague"]
prover = prog.deductions["Prover"]

corpus = [
    ["every", "dog", "barks"],
    ["some", "dog", "barks"],
    ["every", "cat", "walks"],
    ["every", "dog", "is", "an", "animal"],
    ["some", "dog", "is", "an", "animal"],
    ["every", "animal", "barks"],
]

# Touch every sentence once so the deduction's lazily allocated
# lexicon- and rule-weight ParameterDicts materialise every weight,
# one per binding of a rule's category variables, the fit will see.
for sentence in corpus:
    grammar(sentence).goal_weight()

history = adam_fit_deduction(
    grammar, corpus, steps=200, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} -> {history[-1]:.2f}")


def sentence_lf(sentence):
    """The logical form the grammar assigns to the whole sentence,
    paired with the chart's goal weight for that derivation."""
    chart = grammar(sentence)
    target = ("span", 0, len(sentence), ("atom", "S"))
    for item, _ in chart.chart.items():
        if isinstance(item, tuple) and item[:4] == target:
            return item[4], chart.goal_weight()
    raise ValueError(f"no S-derivation for {' '.join(sentence)!r}")


for sentence in corpus:
    lf, _ = sentence_lf(sentence)
    print(f"LF({' '.join(sentence)}) = {lf}")
```

Every logical form printed there is in normal form: its head is `Every` or `Some`, never an `App` whose function is a `Lam`. That is the property the prover depends on.

### Entailment via the prover

The prover takes an axiom list of `(item, log_weight)` pairs. Seeding `Claim(premise_lf)` at the *grammar's* goal weight for that premise, rather than at a constant, is what puts the lexicon's parameters into the prover's chart: the weight of every derived claim is a sum that includes it, so a gradient at `Claim(hypothesis_lf)` reaches back through the grammar.

The NLI corpus below is three entailments and one non-entailment. The third pair is the interesting one. It is licensed only by the explicit `Nonempty(DOG)` premise, which asserts that the dog predicate has a witness. Without it, `every dog barks` would not entail `some dog barks`, and the prover would (correctly) derive nothing.

```python
def noun_pred(noun):
    """The one-place predicate a common noun denotes."""
    chart = grammar([noun])
    target = ("span", 0, 1, ("atom", "N"))
    for item, _ in chart.chart.items():
        if isinstance(item, tuple) and item[:4] == target:
            return item[4]
    raise ValueError(f"{noun!r} is not a common noun in the lexicon")


# (premises, nouns-asserted-non-empty, hypothesis, label)
nli = [
    # Barbara. Valid with no existence assumption.
    ([["every", "dog", "is", "an", "animal"], ["every", "animal", "barks"]],
     [], ["every", "dog", "barks"], 1),
    # Darii. The existential premise carries its own witness.
    ([["some", "dog", "is", "an", "animal"], ["every", "animal", "barks"]],
     [], ["some", "dog", "barks"], 1),
    # Subalternation. Valid ONLY given the existence premise, which
    # is why "dog" is listed as non-empty here and nowhere else.
    ([["every", "dog", "barks"]], ["dog"], ["some", "dog", "barks"], 1),
    # Not entailed: nothing in the premises is about cats.
    ([["every", "dog", "is", "an", "animal"], ["every", "animal", "barks"]],
     [], ["every", "cat", "walks"], 0),
]


def entailment_score(premises, nonempty_nouns, hypothesis):
    """Score Claim(hypothesis_lf) against the prover's chart.

    Returns the log-weight at the hypothesis claim together with the
    log-normalizer over every claim in the prover's closure (premises,
    existence premises, and derived claims alike). Their difference is
    the log-probability that a draw from the closure, with probability
    proportional to weight, is the hypothesis. The log-weight is the
    LogProb semiring's zero (-inf) when the prover derives no such
    claim."""
    axioms = []
    for premise in premises:
        lf, log_z = sentence_lf(premise)
        axioms.append((("Claim", lf), log_z))
    for noun in nonempty_nouns:
        axioms.append((("Claim", ("Nonempty", noun_pred(noun))), torch.zeros(())))
    h_lf, _ = sentence_lf(hypothesis)
    chart = prover(axioms)
    claims = torch.stack(
        [w for item, w in chart.chart.items() if item[0] == "Claim"]
    )
    return chart.try_weight(("Claim", h_lf)), torch.logsumexp(claims, dim=0)


# Touch every prover chart once so its rule-weight ParameterDicts
# allocate every category-binding weight before the optimizer reads them.
for premises, nonempty, hypothesis, label in nli:
    entailment_score(premises, nonempty, hypothesis)

params = list(grammar.parameters()) + list(prover.parameters())
optim = torch.optim.Adam(params, lr=5e-2)

history, grad_at_start = [], []
for step in range(150):
    optim.zero_grad()
    loss = torch.zeros(())
    for premises, nonempty, hypothesis, label in nli:
        score, log_norm = entailment_score(premises, nonempty, hypothesis)
        if label == 1:
            assert torch.isfinite(score), (
                f"{' '.join(hypothesis)!r} is labelled entailed but the "
                f"prover cannot derive it"
            )
            loss = loss - (score - log_norm)
        else:
            # Every prover rule is valid, so a claim the prover derives
            # is entailed. The converse check is that this pair, which
            # is not entailed, has no derivation. Its share of the
            # closure is then exactly zero for every parameter value,
            # so it contributes no loss and no gradient.
            assert not torch.isfinite(score), (
                f"{' '.join(hypothesis)!r} is labelled not entailed but the "
                f"prover derived it"
            )
    loss.backward()
    if step == 0:
        grad_at_start = [
            p for p in params if p.grad is not None and float(p.grad.abs().max()) > 0
        ]
    history.append(float(loss.detach()))
    optim.step()

print(f"NLI loss: {history[0]:.3f} -> {history[-1]:.3f}")
assert torch.isfinite(loss), "loss must be finite"
assert history[-1] < history[0], "the fit must reduce the NLI loss"

# Gradients are read at the first step, before the fit drives them
# toward zero by converging.
print(f"parameters with a non-zero gradient: {len(grad_at_start)} / {len(params)}")
assert grad_at_start, "no parameter received a gradient"

for premises, nonempty, hypothesis, label in nli:
    score, log_norm = entailment_score(premises, nonempty, hypothesis)
    share = float(torch.exp(score - log_norm).detach())
    print(f"share of closure on {' '.join(hypothesis)!r} = {share:.3f}  (label {label})")
```

The final loop reports, per pair, the hypothesis's share of the prover's closure: the probability that a claim drawn from the closure in proportion to its weight is the hypothesis. That is not a probability of entailment. Entailment here is decided symbolically: the prover's rules are valid, so a finite score means the hypothesis follows from the premises, and the share only says how the learned weights distribute mass over the claims that follow (premises included). The loss raises the hypothesis's share for the three entailed pairs, which is the whole of the training signal. The non-entailed pair stays at `0.000` for every parameter value, because no rule derives its claim. It serves as a soundness check. A contradiction or neutral training signal would need a component that can put weight on a hypothesis the premises do not entail, such as a learned classifier over premise and hypothesis, and a sound prover is not such a component.

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples the grammar's log-weights. [`nuts_program_from_deduction`](../api/stochastic/deduction.md#quivers.stochastic.deduction.nuts_program_from_deduction) lifts every learnable parameter into a [`Normal`](../api/continuous/families.md) [sample site](../guides/inference-foundations.md#trace-and-sample-sites) and adds the corpus score $\sum_n \log Z(s_n; \mathbf{w})$ to the joint via a `score` step, so the target density is proportional to the prior times $\prod_n Z(s_n; \mathbf{w})$. This target multiplies the prior by unnormalized sentence weights. A normalized generative likelihood would also divide each weight by a normalizer that depends on $\mathbf{w}$.

```python
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import nuts_program_from_deduction

torch.manual_seed(0)
model, x, observations = nuts_program_from_deduction(
    grammar, corpus, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=30, num_samples=30, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```

## Categorical perspective

Each deduction assigns weights to derivable chart items using the [LogProb semiring](../semantics/algebras.md): addition of log weights combines premises, and log-sum-exp combines alternative derivations. The chart is a weighted collection of items. Its symbolic constructors do not by themselves define a dense tensor arrow $\mathrm{Term}\to\mathrm{Term}$.

`compose(D1, D2)` passes `D1`'s goal items to `D2` as axioms. The grammar produces `span(0, n, S, phi)` items, but the prover's rules match `Claim(phi)`, so the helper wraps each logical form in `Claim` and retains its grammar weight. This conversion lets the prover's rules apply and preserves differentiation through the grammar weights: an entailment loss can train the lexicon.

## Limitations

The fragment is deliberately small, and one limit is structural rather than
incidental. The deduction system does not perform beta reduction, so the
grammar cannot use continuation-form determiner denotations and recover a
readable logical form. It builds the normal form directly instead. A grammar
that needed higher-order denotations, such as quantifier raising or the scope
ambiguities encoded by the
[quantifier-scope example](quantifier-scope.md), would have to encode reduction
in its rules or categories rather than rely on the runtime to normalize.
