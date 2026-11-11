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
# Pattern variables are single-uppercase identifiers (X, Y, Z,
# I, J, K); every other identifier appearing in a rule pattern
# must be listed in the atoms block.

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
program custom_rules_prior : Rule -> Weight
    sample rule_weights : Rule <- Normal(loc=0.0, scale=1.0)
    let rule_rate = exp(rule_weights)
    observe rule_counts : Rule <- Poisson(rate=rule_rate)
    return rule_weights

export custom_rules_prior

deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, VP, PP, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    rule fwd_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z)) #[learnable]
    rule bwd_comp : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Bwd(X, Z)) #[learnable]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "dog" : N = dog #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]
```

## Overview

Every rule in a `deduction { … }` block is a sequent declared in the DSL itself; there are no built-in named rule schemas to import. This example defines an AB grammar (Ajdukiewicz-Bar-Hillel) plus harmonic composition over chart-spans `span(I, J, X)`. `Fwd(X, Y)` is the forward-slash constructor `X/Y`; `Bwd(X, Y)` is the backward-slash constructor `X\Y`.

## Understanding the model

A rule is a template for combining known facts. Suppose one chart entry is
`span(0, 1, Fwd(NP, N))` and the next is `span(1, 2, N)`. The repeated position
and category variables in forward application must agree, giving
`span(0, 2, NP)`. This agreement is unification: different occurrences of one
variable receive one value within that rule firing.

An [agenda](../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) records newly derived entries and tries rules that can use them.
Weights say how to combine evidence along a derivation and across alternatives.
Changing the rule changes what can be derived; changing the [semiring](../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine) changes
what is accumulated about those derivations. Keep those two choices separate
when interpreting the returned goal weight.

## Walkthrough

`object Term : FinSet 16` is the index the deduction's domain and codomain range over; its cardinality is incidental, since rules match constructor-tagged tuples symbolically. `object Rule : FinSet 16` indexes the rule-weight vector instead, and the `custom_rules_prior` program puts an independent `Normal(0.0, 1.0)` on each of its coordinates. Here the `Normal` family makes each indexed coordinate real-valued. The program returns one real-valued weight, so its codomain is `object Weight : Real 1`. The count model uses each exponentiated weight as a Poisson rate and observes one `rule_counts` value per rule.

`atoms NAME, NAME, …` lists every identifier the rules may match literally. Category atoms (`S`, `NP`, `N`, `VP`, `PP`), slash constructors (`Fwd`, `Bwd`), and the chart-item constructor (`span`) are atoms. Identifiers that appear in a rule pattern but are *not* listed in `atoms` are pattern variables; the convention is single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`).

Each rule's body is a sequent: comma-separated premises on the left of `|-`, a single conclusion on the right. The premise multiplicity determines whether the rule fires on a single chart cell (unary) or on a pair of adjacent cells (binary).

A variable appearing multiple times in the same rule unifies across occurrences: in `fwd_app`, the `Y` in the first premise must match the `Y` in the second premise. Different rules instantiate independently.

## DSL features

- **`rule NAME : premises |- conclusion`**: a sequent rule. Arbitrary-arity premise lists are supported; the compiler dispatches to the appropriate chart-cell shape.
- **Pattern variables vs atoms**: single-uppercase identifiers bind as wildcards; every other identifier in a rule pattern must appear in the `atoms` list.
- **Constructor applications in patterns**: `Fwd(X, Y)`, `Bwd(X, Y)`, `span(I, J, X)` are patterns whose head is an atom and whose arguments are nested patterns. Matching is structural and unification-based.

## Pattern Matching and Unification

The compiler uses first-order pattern matching with occurs-checking-free unification to fire rules:

- **Variables** (`X`, `Y`, `Z`): metavariables that match any concrete subterm. Repeated occurrences within a rule must unify.
- **Constructor patterns** (`Fwd(X, Y)`): match constructor-application items whose head is the same atom and whose children unify.

The agenda fires each rule at every chart cell (unary) or every pair of adjacent cells (binary) whose categories unify with the rule's premises; the conclusion item is then inserted into the chart under the rule's weight.

## Extending the Rule Set

Additional combinators are spelled out as more sequent rules in the same block:

- **Type-raising** (unary): `rule type_raise : span(I, J, X) |- span(I, J, Fwd(Y, Bwd(Y, X)))`, note that introducing a fresh wildcard like `Y` in the conclusion requires either an `axioms = …` source or a downstream rule that pins it down.
- **Restricted composition**: `rule restricted_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, NP)) |- span(I, J, Fwd(X, NP))`, by replacing the third wildcard with the literal `NP` atom we constrain when the rule fires.

Rule premise multiplicity is unary or binary; combinators that need three or more premises are expressed as a chain of binary rules sharing intermediate categories.

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

The `custom_rules_prior` program is a standalone Bayesian model over the same
rule weights. Each rule draws one log-weight from a unit Normal;
exponentiating that weight gives the rate at which the rule fires, and
a treebank reports the count. Drawing the weights from their own prior
and the counts from those weights keeps the synthetic point
self-consistent, so a fit has a ground truth to recover.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/custom_rules.qvr")
model = prog.morphism

N_RULES = 16
true_rule_weights = torch.randn(N_RULES)
rule_counts = torch.poisson(torch.exp(true_rule_weights))

observations = {"rule_counts": rule_counts}
x_in = torch.zeros(1, 1)
```

### MAP fit (Adam on rule & lexicon weights)

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction, sample_corpus

torch.manual_seed(0)
prog = load("docs/examples/source/custom_rules.qvr")
ded  = prog.deductions["AB"]

corpus = [["the", "dog", "runs"]] * 4

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

# Forward-sample under the fitted parameters and check the
# dominant length-3 yield recovers the training corpus.
draws = sample_corpus(ded, length=3, n_samples=32, seed=0)
print("dominant yield:", max(set(map(tuple, draws)), key=draws.count))
# → ("the dog runs",)
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
prog = load("docs/examples/source/custom_rules.qvr")
ded  = prog.deductions["AB"]

corpus = [["the", "dog", "runs"]] * 4

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

Sequent rules are hyperedges in the rule-system multicategory. A binary rule is a 2-input / 1-output hyperedge whose endpoints are pattern templates; firing the rule against the chart is the substitution along a pattern morphism into the category of concrete chart items. Variable unification across premises is exactly the pullback in the category of variable assignments: two premise patterns sharing a variable `Y` constrain the two assignments to agree on `Y`'s value. The chart accumulates the weights of derivations, combining premises and alternatives with the semiring operations [@goodman1999semiring]. Agreement between evaluation orders requires an appropriate algorithm: repeated contributions must be handled correctly, and recursive systems need termination or convergence conditions. The runtime selects a default strategy from rule arities and semiring properties; an arbitrary firing order need not preserve the result.
