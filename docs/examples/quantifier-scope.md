# Quantifier Scope

## Overview

Quantifier-scope ambiguity, treated as the choice of which scope-taking lift to apply during composition. Generalized quantifiers are typed as continuations `Cont(X) = S/(S/X)`. A reference for the broader approach is Bumford and Charlow's forthcoming [*Effect-Driven Interpretation*](https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4). [@bumford2026effect]

## QVR source

```qvr
# Quantifier Scope via the Continuation Monad
#
# Generalised quantifiers as Cont-typed denotations whose lifts
# through the residuation calculus give surface vs. inverse
# scope readings. Categories carry slash (Fwd, Bwd) constructors
# and a unary ``Cont(X)`` constructor for the continuation-typed
# lift ``S/(S/X)``; chart items are ``span(I, J, X)`` triples.
#
# Deduction:
#
#   fwd_app      : X/Y, Y                 |- X        base application
#   bwd_app      : Y,   X\Y               |- X        base application
#   fwd_app_cont : Cont(X/Y), Cont(Y)     |- Cont(X)  applicative lift
#   pure_cont    : A                      |- Cont(A)  pure
#   scope_take   : Cont(A), B\A           |- Cont(B)  scope bind
#   cont_elim    : Cont(S)                |- S        lower at the answer type
#
# A determiner is a generalised quantifier in continuation form:
# ``every : Cont(NP/N)``. Applying it to a common-noun-typed
# argument lifted into ``Cont(N)`` via ``pure_cont`` yields
# ``Cont(NP)``, which then binds an inhomogeneous-typed VP under
# ``scope_take`` to give a saturated ``Cont(S)`` reading; the
# final ``cont_elim`` lowers to a flat ``S``.

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
program quantifier_scope_prior : Rule -> Weight
    sample rule_weights : Rule <- Normal(loc=0.0, scale=1.0)
    let rule_rate = exp(rule_weights)
    observe rule_counts : Rule <- Poisson(rate=rule_rate)
    return rule_weights

export quantifier_scope_prior

deduction QScope : Term -> Term [semiring=LogProb, start=S, depth=4, tolerance=1e-5]
    atoms S, NP, N, Fwd, Bwd, Cont, span, every, dog, barks
    rule fwd_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule bwd_app : span(I, K, B), span(K, J, Bwd(A, B)) |- span(I, J, A) #[learnable]
    rule fwd_app_cont : span(I, K, Cont(Fwd(A, B))), span(K, J, Cont(B)) |- span(I, J, Cont(A)) #[learnable]
    rule pure_cont : span(I, J, A) |- span(I, J, Cont(A)) #[learnable, bounded]
    rule scope_take : span(I, K, Cont(A)), span(K, J, Bwd(B, A)) |- span(I, J, Cont(B)) #[learnable]
    rule cont_elim : span(I, J, Cont(S)) |- span(I, J, S) #[learnable, bounded]
    lexicon
        "every" : Cont(Fwd(NP, N)) = every #[learnable]
        "dog"   : N                = dog   #[learnable]
        "barks" : Bwd(S, NP)       = barks #[learnable]
```

## Understanding the model

Scope determines which quantifier's choice depends on which. “Every student
read a book” can describe one possibly different book per student, or one
book shared by all students. A [continuation-style representation](../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler) delays using
a phrase's meaning until the surrounding computation is available, making
some choices of composition order explicit.

The runnable vocabulary contains `every`, `dog`, and `barks`, so its sentences
have one quantifier. Comparing the two book readings would require adding a
[second quantified noun phrase](montague-nli.md#understanding-the-model) and assigning distinct meanings to the resulting
composition orders.

## Walkthrough

`object Term : FinSet 16` is the index set the deduction's domain and codomain range over; the chart itself reasons symbolically over constructor-tagged tuples. `object Rule : FinSet 16` indexes the rule-weight vector that the `quantifier_scope_prior` program draws from an independent `Normal(0.0, 1.0)` per coordinate. The program returns one real-valued weight, so its codomain is `object Weight : Real 1`. The count model uses each exponentiated weight as a Poisson rate and observes one `rule_counts` value per rule.

The `atoms` block declares category atoms (`S`, `NP`, `N`), slash constructors (`Fwd`, `Bwd`), the unary continuation constructor (`Cont`), the chart-item constructor (`span`), and the closed-class terminal vocabulary (`every`, `dog`, `barks`). Pattern variables (single-uppercase identifiers) are bound at firing time.

Six sequent rules realize the scope fragment:

- **`fwd_app` / `bwd_app`**: base forward and backward application from the underlying [Lambek calculus](type-logical.md).
- **`fwd_app_cont`**: the applicative lift of forward application under `Cont`. Two scope-taking expressions compose by pulling both continuations to the outside.
- **`pure_cont`**: the unit `A |- Cont(A)` of the continuation monad. Promotes any non-scope-taking constituent to a trivial scope-taker. Marked `#[bounded]` so the agenda's `depth=4` bound terminates the otherwise unbounded `Cont`-tower.
- **`scope_take`**: the scope-extruding bind. A continuation-typed expression of type `Cont(A)` adjacent to a `Bwd(B, A)` (a functor expecting an `A` argument) absorbs the surrounding context and yields a `Cont(B)`. Surface vs inverse scope corresponds to the order in which two scope-takers apply `scope_take` against the surrounding sentence-internal functors.
- **`cont_elim`**: the lower-at-answer-type closing step `Cont(S) |- S`, which collapses a saturated continuation reading to a flat `S` so the grammar's `start=S` goal applies. It closes a cycle with `pure_cont` (`S |- Cont(S) |- S`), so it is `#[bounded]` as well: with both rules bounded, the joint sub-stochastic cap of [§2.3](../semantics/grammar.md#23-bounded-rule-weights-bounded) keeps the cycle's Kleene-star series convergent at every parameter value, where a bound on `pure_cont` alone would let a fitted `cont_elim` weight push the cycle's mass past one.

A determiner is a generalized quantifier in continuation form: `"every" : Cont(Fwd(NP, N)) = every` lifts the determiner type `NP/N` into the continuation monad, so applying it to a common-noun-typed argument lifted into `Cont(N)` via `pure_cont` yields `Cont(NP)`, which then binds an inhomogeneous-typed VP under `scope_take` to give a saturated `Cont(S)`; the final `cont_elim` lowers to a flat `S`. The deduction's semiring is `LogProb`, so every distinct derivation of the start category `S` contributes a differentiable inside score; summing over derivations gives the marginal log-probability under the grammar, while taking max gives the most-likely single reading.

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/quantifier_scope.qvr")
```

To use the deduction inside a probabilistic program, wrap the chart's per-derivation weights in an outer [`program`](../guides/dsl-overview.md) that draws a categorical reading variable and observes a downstream comprehension judgment; conditioning on the judgment and marginalizing over the scoping yields a posterior over which reading speakers commit to. The [`MonadicProgram.marginalize`](../api/program.md) step is the categorical handle for the scoping marginal: the projection $\pi : \Phi \times R \to \Phi$ integrates out the reading coordinate $R$ via log-sum-exp on the joint log-likelihood.

### Generating synthetic data

The `quantifier_scope_prior` program is a standalone Bayesian model over the same
rule weights. Each rule draws one log-weight from a unit Normal;
exponentiating that weight gives the rate at which the rule fires, and
a treebank reports the count. Drawing the weights from their own prior
and the counts from those weights keeps the synthetic point
self-consistent, so a fit has a ground truth to recover.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/quantifier_scope.qvr")
model = prog.morphism

N_RULES = 16
true_rule_weights = torch.randn(N_RULES)
rule_counts = torch.poisson(torch.exp(true_rule_weights))

observations = {"rule_counts": rule_counts}
x_in = torch.zeros(1, 1)
```

## Categorical perspective

`Cont` is the continuation monad on the category of formulas: $\mathrm{Cont}(X) = S/(S/X)$ realizes double negation $\neg\neg X$ in the answer-type $S$. The `pure_cont` rule is its unit $\eta : X \to \mathrm{Cont}(X)$; `fwd_app_cont` realizes its applicative-functor structure; `scope_take` realizes the [monadic bind](https://doi.org/10.1016/0890-5401(91)90052-4) $\mu \circ \mathrm{Cont}(f)$ in the form licensed by the [Lambek calculus](https://doi.org/10.2307/2310058) residuation laws. Surface vs inverse scope is then a choice of derivation in the rule-system multicategory; the agenda's `LogProb`-enriched chart records every choice, and downstream `marginalize` integrates them away into a single scope-marginal likelihood. [@moggi1991notions] [@lambek1958mathematics]

## Connections

The fragment composes directly with [type-logical](type-logical.md) and [multimodal](multimodal-tlg.md) grammars: replacing `Cont` with a controlled modality `Dia` yields a modal scope fragment in which structural rules are licensed only inside the scope-taker.


## Further reading

- Dylan Bumford and Simon Charlow. Forthcoming. *Effect-Driven Interpretation: Functors for Natural Language Composition*. Cambridge Elements in Semantics. Cambridge University Press.
