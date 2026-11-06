# Lambek-inspired weighted deduction

## QVR source

```qvr
# Weighted Type-Logical Grammar
#
# The non-commutative Lambek calculus L expressed as an
# agenda-based weighted deduction. Categories carry slash (Fwd,
# Bwd) and tensor (Tns) constructors; chart items are span(I,
# J, X) triples.
#
# Deduction:
#
#   right_app    : X/Y, Y    |- X              forward application
#   left_app     : Y,   X\Y  |- X              backward application
#   tensor_intro : A,   B    |- Tns(A, B)      product introduction
#   tensor_left  : Tns(A, B) |- A              left projection
#   tensor_right : Tns(A, B) |- B              right projection
#
# The Lambek calculus is resource-sensitive (every hypothesis is
# used exactly once) and order-preserving (no permutation). It
# is the internal logic of a biclosed monoidal category; the
# residuation law A (x) B |- C iff A |- C/B iff B |- A\C makes
# L the free residuated monoid over its atoms.

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
program type_logical_prior : Rule -> Weight
    sample rule_weights : Rule <- Normal(loc=0.0, scale=1.0)
    let rule_rate = exp(rule_weights)
    observe rule_counts : Rule <- Poisson(rate=rule_rate)
    return rule_weights

export type_logical_prior

deduction Lambek : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, VP, PP, Fwd, Bwd, Tns, span, every, dog, barks
    rule right_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule left_app : span(I, K, B), span(K, J, Bwd(A, B)) |- span(I, J, A) #[learnable]
    rule tensor_intro : span(I, K, A), span(K, J, B) |- span(I, J, Tns(A, B)) #[learnable]
    rule tensor_left : span(I, J, Tns(A, B)) |- span(I, J, A) #[learnable]
    rule tensor_right : span(I, J, Tns(A, B)) |- span(I, J, B) #[learnable]
    lexicon
        "every" : Fwd(NP, N) = every #[learnable]
        "dog" : N = dog #[learnable]
        "barks" : Bwd(S, NP) = barks #[learnable]
```

## Overview

This deduction is inspired by the non-commutative Lambek calculus and preserves span order. It also includes `tensor_left` and `tensor_right`, which discard one component of a tensor item. Those projection rules amount to weakening and are not product elimination rules of the strict resource-sensitive Lambek calculus.

## Understanding the model

Read a slash category as an instruction to look for an argument on one
side. `A/B` needs a `B` on the right to produce an `A`; in this source's
convention `A\B` needs it on the left. The [span indices](custom-rules.md#understanding-the-model) ensure that the
pieces are adjacent and occur in that order.

A tensor category keeps two resources together. The two projection rules here
can discard either resource, permitting weakening. The strict Lambek calculus
requires both resources to be used, so these projections enlarge the set of
derivations beyond that calculus.

## Walkthrough

`object Term : FinSet 16` indexes the deduction's domain and codomain; the chart reasons symbolically over constructor-tagged tuples, so the cardinality is incidental. `object Rule : FinSet 16` indexes the rule-weight vector, which the `type_logical_prior` program draws from an independent `Normal(0.0, 1.0)` per coordinate. The program returns one real-valued weight, so its codomain is `object Weight : Real 1`. The count model uses each exponentiated weight as a Poisson rate and observes one `rule_counts` value per rule.

`atoms NAME, NAME, ...` declares the constructor vocabulary. Category atoms are `S`, `NP`, `N`, `VP`, `PP`; structural constructors are `Fwd(A, B) ≡ A/B`, `Bwd(A, B) ≡ A\B`, `Tns(A, B) ≡ A⊗B`. The chart-item constructor `span(I, J, X)` packages a derivation covering tokens `[I, J)` carrying category `X`. Single-uppercase identifiers (`A`, `B`, `I`, `J`, `K`) appearing in rule patterns bind as wildcards.

The rules realize the four logical core operations of the Lambek calculus:

- **`right_app`**: modus ponens for forward slash: `A/B, B ⊢ A`.
- **`left_app`**: modus ponens for backward slash: `B, A\B ⊢ A`.
- **`tensor_intro`**: product introduction: adjacent derivations of `A` and `B` combine into a derivation of `A⊗B`.
- **`tensor_left` / `tensor_right`**: extra projection rules that retain one component and discard the other. They are useful operationally but relax Lambek resource sensitivity.

Together these rules define the weighted fragment implemented on this page. The agenda runs to depth 6; the `LogProb` semiring accumulates inside scores that remain differentiable with respect to learnable weights.

## DSL features

- **Sequent rules with arbitrary arity**: rule bodies declare premises on the left of `|-` and a single conclusion on the right; the compiler routes unary patterns to unary chart cells and binary patterns to binary chart cells.
- **Ordered span composition**: binary premises occupy adjacent chart cells. The projection rules nevertheless introduce weakening at the category level.
- **Order preservation**: pattern variables appear in textual order; the parser enforces left-to-right span composition.
- **Tensor and slash as user atoms**: there is no special syntax, `Tns`, `Fwd`, `Bwd` are atoms declared in the `atoms NAME, NAME, ...` block and may be replaced or extended by the user.

## Try it

Every `#[learnable]` lexicon entry and every `#[learnable]` rule
exposes a real `nn.Parameter` on the compiled
[`DeductionSystem`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem). The
system is callable: `ded(sentence)` returns a
[`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose
`goal_weight()` is the differentiable log-marginal
$\log Z(s; \mathbf{w}) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$
summed over every derivation $d$ that the start symbol licenses for
the input. Fitting the lexicon and rule weights together is then a
regression-style problem: minimise $-\sum_n \log Z(s_n)$ over a
corpus of sentences. The
[`quivers.stochastic.deduction`](../api/stochastic/deduction.md) module provides
the fitting utilities used below.

### Generating synthetic data

The `type_logical_prior` program is a standalone Bayesian model over the same
rule weights. Each rule draws one log-weight from a unit Normal;
exponentiating that weight gives the rate at which the rule fires, and
a treebank reports the count. Drawing the weights from their own prior
and the counts from those weights keeps the synthetic point
self-consistent, so a fit has a ground truth to recover.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/type_logical.qvr")
model = prog.morphism

N_RULES = 16
true_rule_weights = torch.randn(N_RULES)
rule_counts = torch.poisson(torch.exp(true_rule_weights))

observations = {"rule_counts": rule_counts}
x_in = torch.zeros(N_RULES, 1)
```

### MAP fit (Adam on rule & lexicon weights)

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction, sample_corpus

torch.manual_seed(0)
prog = load("docs/examples/source/type_logical.qvr")
ded  = prog.deductions["Lambek"]

corpus = [["every", "dog", "barks"]] * 4

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

# Forward-sample under the fitted parameters and check the
# dominant length-3 yield recovers the training corpus.
draws = sample_corpus(ded, length=3, n_samples=32, seed=0)
print("dominant yield:", max(set(map(tuple, draws)), key=draws.count))
# → ("every dog barks",)
```

`adam_fit_deduction` maximizes the corpus log-marginal under an
optional Normal prior on the parameters; `prior_scale=1.0` gives MAP
under a unit Normal. `sample_corpus` enumerates yields of the chosen
length and draws from the categorical defined by their chart weights; exact forward sampling because the chart marginalizes the
derivation forest.

### NUTS (full Bayesian posterior)

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import nuts_program_from_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/type_logical.qvr")
ded  = prog.deductions["Lambek"]

corpus = [["every", "dog", "barks"]] * 4

model, x, observations = nuts_program_from_deduction(
    ded, corpus, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.1, max_tree_depth=4, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
posterior_means = {
    name: float(samples.mean()) for name, samples in result.samples.items()
}
print("posterior mean log-weights:", posterior_means)
```

`nuts_program_from_deduction` lifts every learnable parameter of the
deduction into a [`Normal(0, σ)`](../api/continuous/families.md) sample
site and adds the corpus log-marginal $\log Z$ to the joint via a
[`score`](../api/program.md) step. The standard
[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) drives the
posterior $p(\mathbf{w} \mid s_1, \ldots, s_N) \propto p(\mathbf{w})
\cdot \prod_n Z(s_n; \mathbf{w})$. The same Bayesian object
[`bayesian_regression`](bayesian-regression.md) fits, with the chart
total in place of the Gaussian likelihood.

## Categorical perspective

The Lambek calculus is the internal language of a residuated monoidal category (biclosed monoidal category). The tensor `⊗` is the monoidal product; the two slashes are its left and right adjoints. The residuation laws

```
A ⊗ B  ⊢  C   iff   A  ⊢  C/B   iff   B  ⊢  A\C
```

These are the usual residuation laws. The application and tensor-introduction rules are compatible with that reading. The two projection rules are additional and prevent this particular deduction from being a faithful presentation of the strict calculus.

## Connections to Other Formalisms

Lambek grammars and their extensions can encode useful word-order and resource constraints. Their precise weak generative capacity depends on the chosen calculus and structural rules, so extraction or gapping does not follow from these five rules alone.
