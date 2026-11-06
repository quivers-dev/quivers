# Probabilistic Multiple Context-Free Grammar

## QVR source

```qvr
# Probabilistic Multiple Context-Free Grammar
#
# An MCFG with learnable per-production log-weights. PMCFG is to
# MCFG what PCFG is to CFG: each production carries a probability
# (or arbitrary semiring weight) and parsing recovers the inside
# weight of the input under those probabilities.
#
# This grammar handles WH-movement in English relative clauses, a
# textbook motivation for MCFG over CFG. In a relative clause like
#
#     "the man who Mary saw"
#
# the WH-word ``who`` appears at the left edge of the clause but
# its grammatical role is the *object* of ``saw``: there is a gap
# (a missing NP) immediately after the verb that ``who``
# discontinuously fills. CFGs cannot express this directly because
# the filler and the gap are separated by a constituent boundary;
# MCFGs handle it by giving the relevant non-terminal a *tuple*
# of yields, one component per disjoint substring of the input.
#
# We use a rank-2 non-terminal ``sg`` (an S with an NP gap). An
# item
#
#     sg(I1, J1, I2, J2)
#
# means: the first component of the sg spans ``[I1, J1)``, the
# second component spans ``[I2, J2)``, and the NP gap sits
# between the two components. For ``"Mary saw"`` parsed with a
# trailing gap we get
#
#     sg(3, 5, 5, 5)
#
# over the input ``[the, man, who, Mary, saw]``: component 1 =
# "Mary saw" = positions [3, 5); component 2 = "" = positions
# [5, 5); the gap would have sat between them.
#
# The relative-clause production then concatenates a WH-word with
# both components of the sg in their input order, which is the
# linear yield function of the rank-2-to-rank-1 reduction, producing
# a rank-1 RC item that spans the whole "who Mary saw" substring.
# Modifying ``N`` with that RC and then combining with the
# determiner gives the full NP yield.
#
# Every production carries ``#[learnable]``; the bindings-keyed
# parameter dictionary allocates one log-weight per distinct
# binding tuple at run time. ``inside_NP`` is thus the
# log-marginal of the corpus under a learnable MCFG, fittable
# with the standard regression-style SVI and NUTS workflows.

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
program pmcfg_prior : Rule -> Weight
    sample rule_weights : Rule <- Normal(loc=0.0, scale=1.0)
    let rule_rate = exp(rule_weights)
    observe rule_counts : Rule <- Poisson(rate=rule_rate)
    return rule_weights

export pmcfg_prior

deduction PMCFG : Term -> Term [semiring=LogProb, start=NP, depth=4]
    atoms NP, N, Det, V, who, S, RC, sg, span, the, man, dog, mary, john, saw, met
    # Lexical entries are span(I, J, X) for rank-1 categories.
    # The transitive-with-object-gap rule pairs a subject NP and
    # a transitive verb V into a rank-2 sg item whose first
    # component covers the subject and the verb, and whose second
    # component is empty (positioned at the gap site).
    rule transitive_obj_gap : span(I, K, NP), span(K, J, V) |- sg(I, J, J, J) #[learnable]
    # Linear yield function ``RC(w x y) :- who(w) sg(x, y)``:
    # concatenates the WH-filler with both components of the sg.
    # The pattern enforces ``P' == I_sg`` (the WH-word and the
    # subject of the gapped clause are adjacent in the input).
    rule relative_clause : span(P, Pp, who), sg(Pp, J1, J1, J2) |- span(P, J2, RC) #[learnable]
    # N modified by RC remains N: N -> N RC.
    rule modify_n : span(I, J, N), span(J, K, RC) |- span(I, K, N) #[learnable]
    # Standard NP -> Det N.
    rule np_det_n : span(I, J, Det), span(J, K, N) |- span(I, K, NP) #[learnable]
    lexicon
        "the"  : Det = the   #[learnable]
        "man"  : N   = man   #[learnable]
        "dog"  : N   = dog   #[learnable]
        "who"  : who = who   #[learnable]
        "Mary" : NP  = mary  #[learnable]
        "John" : NP  = john  #[learnable]
        "saw"  : V   = saw   #[learnable]
        "met"  : V   = met   #[learnable]
```

## Overview

A Probabilistic Multiple Context-Free Grammar (PMCFG) is a Multiple Context-Free Grammar (MCFG) [@seki1991mcfg] with a probability (or, more generally, a semiring weight) attached to every production. PMCFG is to MCFG what PCFG is to CFG: the same rule set, decorated with learnable weights, fitted to data via the chart's inside marginal $\log Z(s) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$.

The defining feature of MCFG, and thus PMCFG, is that each non-terminal $A$ has a fixed rank $k(A) \ge 1$ and generates a *tuple of strings* with $k(A)$ components. A production rewrites a non-terminal as a linear combination of its premises' tuple components. The rank-1 case is CFG; ranks $\ge 2$ provide a direct representation of discontinuous constituents.

This example uses MCFG to model English **WH-movement in relative clauses**, a textbook motivation for the formalism. In a noun phrase like

> "the man who Mary saw"

the relative pronoun *who* appears at the left edge of the embedded clause, but its grammatical role is the *object* of *saw*. A CFG can generate the surface string, but a flat single-yield non-terminal does not directly represent the two separated pieces of the gapped constituent. MCFG represents that structure with a **rank-2 non-terminal** whose components straddle the gap site.

## Understanding the model

A [context-free chart item](pcfg.md#understanding-the-model) normally covers one continuous interval. An MCFG
item can hold several intervals that belong to one constituent with a gap.
For `Mary saw _`, the two pieces lie before and after the missing object;
a relative-clause rule can combine those pieces with `who` while preserving
their required order.

Rank counts the pieces belonging to one constituent. A rank-2 item thus
stores two spans, even if a rule producing it has three premises. The weighted
chart adds log weights along each derivation and uses [log-sum-exp](../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine) to combine
alternative derivations of the same item.

## Walkthrough

`object Term : FinSet 16` indexes the deduction's domain and codomain, and `object Rule : FinSet 16` indexes the rule-weight vector that the `pmcfg_prior` program draws from an independent `Normal(0.0, 1.0)` per coordinate. The `Normal` family makes each coordinate of the indexed weight vector real-valued. The program returns one real-valued weight, so its codomain is `object Weight : Real 1`. The count model uses each exponentiated weight as a Poisson rate and observes one `rule_counts` value per rule.

`sg(I1, J1, I2, J2)` is the rank-2 item for an *S with an NP gap*. Its first yield component spans `[I1, J1)` (the prefix before the gap) and its second component spans `[I2, J2)` (the suffix after the gap). For the input `[the, man, who, Mary, saw]` parsed as `the man who Mary saw`, the gapped clause is `Mary saw _` with the gap at the very end, giving

```
sg(3, 5, 5, 5)         # component 1 = "Mary saw"; component 2 = ""
```

The relative-clause production

```
rule relative_clause : span(P, Pp, who), sg(Pp, J1, J1, J2) |- span(P, J2, RC)
```

implements the linear yield function `RC(w x y) :- who(w) sg(x, y)`: the WH-word, the prefix component of the sg, and the suffix component are concatenated *in input order* into a single rank-1 RC item. The variable `Pp` appearing in both the `who` span and the start of the sg's first component enforces input-adjacency between the WH-filler and the subject of the gapped clause. Likewise `J1` appearing both at the end of the first sg component and at the start of the second pins the gap site.

`modify_n` and `np_det_n` are the standard CFG productions for N-modification and the determiner-noun NP. The compiler allocates one `nn.Parameter` per distinct binding tuple it observes at run time, so each production becomes a weighted edge in the chart and the goal weight at `span(0, 5, NP)` is the inside log-probability of the full NP under the grammar.

## DSL features

- **Tuple-valued chart items** for non-terminals of rank $\ge 2$. The `sg(I1, J1, I2, J2)` item is a four-position structural pattern; the chart engine pattern-matches it the same way it does the rank-1 `span(I, J, X)`.
- **Linear yield functions** as ordinary sequent rules. Concatenation and component permutation across premises are expressed by where each variable appears in the conclusion.
- **`#[learnable]` weights on every production**, lexicon entries and rules alike. The bindings-keyed parameter dictionary stores one log-weight per distinct binding tuple, giving the same partial-application parameterization as a per-production-instantiation PCFG.

## Try it

The deduction system is callable: `ded(sentence)` returns a [`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose `goal_weight()` is the differentiable log-marginal $\log Z(s; \mathbf{w})$ summed over every derivation the start symbol licenses for the input. Fitting the lexicon and rule weights together is then a regression-style problem; the [`quivers.stochastic.deduction`](../api/stochastic/deduction.md) module provides the fitting utilities used below.

### Generating synthetic data

The `pmcfg_prior` program is a standalone Bayesian model over the same
rule weights. Each rule draws one log-weight from a unit Normal;
exponentiating that weight gives the rate at which the rule fires, and
a treebank reports the count. Drawing the weights from their own prior
and the counts from those weights keeps the synthetic point
self-consistent, so a fit has a ground truth to recover.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/pmcfg.qvr")
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
from quivers.stochastic.deduction import adam_fit_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/pmcfg.qvr")
ded  = prog.deductions["PMCFG"]

corpus = [
    ["the", "man", "who", "Mary", "saw"],
    ["the", "dog", "who", "John", "met"],
]

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

for sentence in corpus:
    log_z = float(ded(sentence).goal_weight().detach())
    print(f"  log Z({' '.join(sentence)}) = {log_z:.2f}")
```

### NUTS (full Bayesian posterior)

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import nuts_program_from_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/pmcfg.qvr")
ded  = prog.deductions["PMCFG"]

corpus = [
    ["the", "man", "who", "Mary", "saw"],
    ["the", "dog", "who", "John", "met"],
]

model, x, observations = nuts_program_from_deduction(
    ded, corpus, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.1, max_tree_depth=4, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```

## Categorical perspective

An MCFG production with $k$-component non-terminals is a hyperedge in a multi-colored directed hypergraph whose nodes are tuple-positioned chart items. When the derivation sum converges and the evaluation computes it correctly, the chart represents the total weight of derivations [@goodman1999semiring]; each derivation contributes the sum of the log-weights of its rule firings, lifted through the bindings-keyed parameter dictionary. The PMCFG inside algorithm is the agenda-driven evaluation of that fixed point; WH-movement is recovered as the linear yield function that interleaves a filler with the components of a higher-rank non-terminal.

The framework imposes no built-in commitment to rank-1 (CFG) items. Higher-rank PMCFG, MCFG, LCFRS, and PLCFRS all share the same chart implementation: only the rule patterns and their conclusion arities change.
