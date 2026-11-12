# Weighted Deduction Systems

A [weighted deduction system](https://en.wikipedia.org/wiki/Chart_parser) derives
weighted facts, called *items*, by applying inference rules to items already derived
[@shieber1995principles; @goodman1999semiring; @nederhof2003weighted]. In a parser one
item might say that words 0 through 1 form a noun phrase and another that words 1
through 3 form a verb phrase, and a rule combines them into a sentence spanning 0
through 3. A [*chart*](#charts-as-first-class-differentiable-values) stores the items
derived so far with their accumulated weights, and an *agenda* holds the items still
to be processed [@eisner2005compiling]. Separating the two keeps what counts as a
derivation apart from the order in which derivations are found.

QVR's `deduction` declaration defines such a system, and one agenda-based engine
evaluates it. Choosing the agenda gives the usual parsing and search strategies: CKY
(bottom-up over spans), Earley, Viterbi, A* [@klein2003astar], Knuth's best-first
algorithm [@knuth1977generalization], semi-naive Datalog evaluation
[@mcallester2002complexity], and depth-first proof search.

## The framework

A weighted deduction system is a tuple

$$
\mathcal D = (I, R, K, \mathrm{ax}, \mathrm{goal})
$$

of a set $I$ of items, terms built from the declared constructor symbols; a set $R$
of inference rules, each with premises, a conclusion, and a weight, and each standing
for all instances of its pattern variables; a [semiring](core.md#algebras-enrichment-algebras)
$K$ of weights; the axioms $\mathrm{ax}$, the weighted items given by the input; and
the goal items $\mathrm{goal} \subseteq I$, whose weights are the answer
[@goodman1999semiring]. A rule with several premises is a hyperedge of a
[hypergraph](https://en.wikipedia.org/wiki/Hypergraph), an edge that joins several
inputs to one output [@klein2001parsing]. The library's `DeductionSystem` also takes
an agenda factory and a chart factory, which choose how the system is evaluated and
not what it derives.

A derivation's weight is the $\otimes$-product of its rule and axiom weights, and the
weights of different derivations of one item combine with $\oplus$. The same rules
thus answer different questions under different semirings: Boolean weights ask
whether any derivation exists, `LogProb` (the log semiring) sums probabilities in log
space, and `Viterbi` (the max-plus semiring) keeps the best derivation.

The chart assigns each item the total weight of its derivations, its *inside weight*.
When the semiring is ordered completely and the rules are monotone, this is the least
fixed point of the rules. Whether that fixed point exists, whether iteration reaches
it, and whether the result is the same for every processing order depend on the
semiring, on cycles among the rules, and on the agenda
([Strategy independence](#strategy-independence)).

## QVR syntax

```qvr
object Atom : FinSet 4

deduction CG : Atom -> Atom [semiring=LogProb, start=S, depth=4]
    atoms NP, S, VP, Fwd, Bwd, span
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X)
    rule harm_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z))
```

The option block sets the semiring (`LogProb`, `Boolean`, `Viterbi`, or `Counting`),
the start category, a bound `depth` on the constructor depth of categories, an
optional `tolerance` for cycles under `LogProb`, and an optional cap
`max_iterations`. The body lists the constructor symbols in `atoms` and then the
rules. In a rule pattern, an identifier that is neither a declared atom nor a numeral
is a pattern variable, whatever its spelling, so `I`, `K`, `X`, and `Y` above are
variables and `span`, `Fwd`, and `NP` are constants.

Slashes are not built-in syntax but constructors the grammar declares. The rules here
write them in Steedman notation, with the result leftmost [@steedman2000syntactic]:
`Fwd(X, Y)` is `X/Y`, which takes a `Y` on its right to give `X`, and `Bwd(X, Y)` is
`X\Y`, which takes a `Y` on its left. `harm_comp` is forward harmonic composition,
`X/Y, Y/Z ⊢ X/Z`. The declaration compiles to a `DeductionSystem` registered under
its name, here `CG`.

## Charts as first-class differentiable values

Running a deduction returns a `ChartView` that maps items to weights. Its `weight`
method reads one item, `enumerate` lists the items matching a pattern, and
`goal_weight` reads the goal. The weights are PyTorch tensors, and the agenda
combines them with the semiring's operations, so gradients flow from any chart weight
back to every rule or axiom weight with `requires_grad=True`.

With log weights `-0.3` and `-0.4`, a binary rule of weight 0 gives its conclusion
`-0.7`. A second derivation of the same item would combine with the first by
log-sum-exp, not by adding the two log weights. Differentiating the inside weight of
the goal trains the summed score of all derivations, while the `Viterbi` semiring
trains the score of the best derivation alone.

<!-- python: skip -->
```python
import torch
from quivers.dsl import loads
from quivers.stochastic import Wildcard

CG = loads(source).deductions["CG"]          # source: the QVR block above
NP, S = ("atom", "NP"), ("atom", "S")        # constants as compiled item terms

np_w = torch.tensor(-0.3, requires_grad=True)
vp_w = torch.tensor(-0.4, requires_grad=True)
view = CG([
    (("span", 0, 1, NP), np_w),
    (("span", 1, 3, ("Bwd", S, NP)), vp_w),
])

log_weight = view.weight(("span", 0, 3, S))                       # -0.7
all_S = view.enumerate(("span", Wildcard("i"), Wildcard("j"), S))

(-log_weight).backward()   # gradients reach np_w and vp_w
```

A compiled item lists its chart positions before its category, and declared
constants appear as `("atom", name)` terms.

## Strategy independence

`Boolean` OR and `Viterbi` max are idempotent: adding the same contribution twice
changes nothing. `LogProb` addition is log-sum-exp and is not idempotent. Two copies
of a derivation with log weight $p$ combine to $\log(e^{p} + e^{p}) = p + \log 2$, so
an agenda that derived the same item twice by the same route would double its
probability. A non-idempotent semiring thus gives the same chart under every agenda
only if each derivation is counted exactly once, and a system with cycles also needs
the iteration to converge [@goodman1999semiring]. For an acyclic grammar under
`LogProb`, the CKY, semi-naive, depth-first, and Earley agendas give the same goal
weights to within `1e-5`. When one item matches two premises of a rule, as a single
`NP` matches both premises of `NP, NP ⊢ S`, every agenda fires the rule once for
each premise position, so under `LogProb` that one derivation contributes $\log 2$
instead of $0$.

## Pre-registered stdlib {#pre-registered-stdlib}

`quivers.stochastic.stdlib` provides ready-made `DeductionSystem` instances:

| System | Items | Semiring | Agenda |
|----------------|---------------------------|------------|----------------------|
| `CCG` | categories built with `("slash", dir, X, Y)`, without positions | `LogProb` | CKY |
| `Lambek` | categories, as for `CCG` | `LogProb` | CKY |
| `STLC` | checking and synthesis judgments of the simply typed lambda calculus | `Boolean` | depth-first |
| `MLTT` | typing judgments of dependent type theory, with bounded term depth | `Boolean` | depth-first |
| `Datalog` | ground facts of transitive closure | `Boolean` | semi-naive |
| `Dijkstra` | `("dist", node)` | `Viterbi` | Knuth |
| `HMM` | `("alpha", t, state)`, the forward algorithm | `LogProb` | CKY |
| `ViterbiHMM` | as for `HMM` | `Viterbi` | CKY |
| `EditDistance` | `("dist", i, j)` | `Viterbi` | CKY |

Each is callable: `view = Datalog(axiom_list)` returns a `ChartView`.

## Strategy factory functions

`quivers.stochastic.agenda` provides the agenda factories that a `DeductionSystem`
takes as `agenda_factory=`:

- `cky_agenda()` and `earley_agenda()`, first in first out;
- `viterbi_agenda(priority_fn)`, a priority queue;
- `astar_agenda(g_plus_h)`, a priority queue ordered by cost plus an admissible
  heuristic [@klein2003astar];
- `knuth_agenda()`, best-first search over hyperpaths
  [@knuth1977generalization; @nederhof2003weighted];
- `depth_first_agenda()`, last in first out, for proof search;
- `semi_naive_agenda()`, first in first out, for systems in the style of Datalog
  [@mcallester2002complexity].

## Production probabilities per left-hand side

A `#[learnable]` rule or lexicon entry carries a free log weight, and the inside
weight of a sentence grows with every one of them. Maximum likelihood over free
weights thus has no optimum: fitting drives the weights up without bound. A
[probabilistic context-free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar)
instead makes the productions that rewrite one category a probability distribution,
$\sum_{A \to \beta} P(A \to \beta) = 1$ for every category $A$ [@booth1973applying].
Its likelihood is bounded, and on a corpus whose sentences each have one derivation,
the maximum-likelihood estimate of a production's probability is its relative
frequency among the uses of its left-hand side [@chi1998estimation].

The option `normalize=lhs` gives this reading. Each rule and each expanded lexicon
entry is a production whose free log weight is its learned parameter, or zero when it
is not `#[learnable]`. The productions that share a left-hand side form a group, and
each group's free log weights pass through `log_softmax`. The chart, the search in the
checked core, fitting, and sampling all read the normalized weights. `normalize=none`,
the default, keeps the free weights.

<!-- compile: qiec -->
```qvr
object Term : FinSet 16

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, Det, N, V, span, the, a, cat, dog, sleeps, runs
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule vp_v : span(I, J, V) |- span(I, J, VP) #[learnable]
    lexicon
        "the"    : Det = the     #[learnable]
        "a"      : Det = a       #[learnable]
        "cat"    : N   = cat     #[learnable]
        "dog"    : N   = dog     #[learnable]
        "sleeps" : V   = sleeps  #[learnable]
        "runs"   : V   = runs    #[learnable]
```

The left-hand side of a production depends on its form:

| Production | Left-hand side |
| --- | --- |
| a rule concluding a span, `... \|- span(I, J, NP)`, unary or binary | the category, `NP` |
| a rule concluding a structured category, `... \|- span(I, J, Fwd(NP, N))` | the category term, `Fwd(NP, N)` |
| a lexicon entry `"the" : Det` | `Det`, the category of the `span(i, i + 1, Det)` axiom it adds |
| a lexicon entry with several categories, `"saw" : {N, V}` | each listed category, one production per category |
| a rule concluding another item, `... \|- p(I1, J1, I2, J2)` | the item with its chart positions erased, `p(_, _, _, _)` |

A group is fixed only if no production's left-hand side or weight depends on how its
variables are bound. The compiler thus refuses, and locates, each of the following:

- a rule that binds a variable outside the chart positions, such as
  `span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)`, which stands for one
  production per binding of `X` and `Y`, with groups that depend on which categories
  the lexicon makes reachable;
- a system whose spans carry logical forms;
- a rule with `bounded` or `parent`, each of which reparameterizes the rule's weight;
- a lexicon category that is not ground, or a word listed twice under one category;
- a system that takes its axioms as input instead of from a lexicon;
- a semiring other than `LogProb` or `Viterbi`.

Both evaluators read one analysis of the declaration, so they normalize over the same
groups.

[`DeductionSystem.production_groups`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem.production_groups)
lists the groups, and
[`DeductionSystem.production_log_probs`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem.production_log_probs)
returns each production's log probability. After
[`adam_fit_deduction`](../api/stochastic/deduction/fit.md#quivers.stochastic.deduction.fit.adam_fit_deduction)
on the three sentences below, `the` has probability two thirds among the
determiners, its relative frequency:

<!-- python: skip -->
```python
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction

grammar = load("pcfg.qvr").deductions["PCFG"]
corpus = [["the", "cat", "sleeps"], ["a", "dog", "runs"], ["the", "dog", "sleeps"]]
adam_fit_deduction(grammar, corpus, steps=400, lr=0.1)
probabilities = {
    lhs: {production: round(float(weight.exp()), 3) for production, weight in group.items()}
    for lhs, group in grammar.production_log_probs().items()
}
# probabilities["Det"] == {'"the"': 0.667, '"a"': 0.333}
```

Normalizing per left-hand side gives a distribution over derivations only for a
consistent grammar, one whose derivations terminate with probability one. A grammar
whose recursive productions are too probable puts mass on infinite derivations, and
its sentence probabilities sum to less than one [@booth1973applying]. The option does
not check consistency.

## Priors on production probabilities

A program can place a prior on a normalized grammar's production probabilities and
infer their posterior, the Bayesian PCFG of Johnson et al. [@johnson2007bayesian].
The call `parse(...)` binds each left-hand side with more than one production to a
probability vector over its productions, in the order `production_groups` lists them,
as a pair of the left-hand side's label and a program value. A production alone in
its group has probability one and takes no binding. The corpus is one input of one
run. With a plate object after the input, `parse` reads a table with one sentence per
element of that object, all of one length, and `chart.goal_weight()` is the sum of
the sentences' log inside weights.

<!-- compile: qiec -->
```qvr
object Term : FinSet 16
object Weight : Real 1
object Sentence : FinSet 3

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, Det, N, V, span, the, a, cat, dog, sleeps, runs
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S)
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP)
    rule vp_v : span(I, J, V) |- span(I, J, VP)
    lexicon
        "the"    : Det = the
        "a"      : Det = a
        "cat"    : N   = cat
        "dog"    : N   = dog
        "sleeps" : V   = sleeps
        "runs"   : V   = runs

program bayesian_pcfg : Term -> Weight
    sample p_det <- Dirichlet(concentration=[1.0, 1.0])
    sample p_n <- Dirichlet(concentration=[1.0, 1.0])
    sample p_v <- Dirichlet(concentration=[1.0, 1.0])
    let chart = parse(PCFG, corpus, Sentence, ("Det", p_det), ("N", p_n), ("V", p_v))
    score evidence = chart.goal_weight()
    return evidence

export bayesian_pcfg
```

The program's log joint density is the sum of the Dirichlet log densities and the
corpus's log likelihood. Both the [compiled runtime](dsl-overview.md#compilation-pipeline) and the
[reference machine](../developer/qiec.md#reference-evaluator) compute it. The compiled runtime passes
the drawn values to the chart and gradients reach them, so
[`MCMC`](../api/inference/mcmc.md#quivers.inference.mcmc.MCMC) with
[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples the
posterior. The checked core elaborates the call to a variant of the deduction's
computations that reads its weights from the bound values.

<!-- python: skip -->
```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel

program = load("bayesian_pcfg.qvr")
corpus = [["the", "cat", "runs"], ["a", "dog", "sleeps"], ["the", "dog", "runs"]]
result = MCMC(NUTSKernel(step_size=0.3), num_warmup=200, num_samples=300).run(
    program.morphism, torch.zeros(1, 1), {"corpus": corpus}
)
posterior_mean = result.samples["p_det"].mean(dim=(0, 1))
```

Each sentence of this grammar has one derivation, so the posterior for each
left-hand side is the [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior)
$\mathrm{Dirichlet}(\alpha + n)$, with $n$ the production counts of the corpus's
derivations. The compiler checks the bindings when the program is lowered: each bound
label is a left-hand side of several productions, every such left-hand side is bound
once, each value is a vector of its group's size, a value drawn at a sample site comes
from a family whose support is the probability simplex (a Dirichlet draw, or a
`softmax` of an unconstrained draw for a logistic-normal prior), and the grammar
declares `normalize=lhs`. Without bindings, the call reads the learned weights. In
Python,
[`DeductionSystem.__call__`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem.__call__)
and
[`DeductionSystem.corpus`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem.corpus)
take the same probabilities as a mapping from left-hand side to vector. No transpile
target supports a program that calls a deduction
([Transpiler support](../transpile-support.md)).

## The deduction as a QIEC computation {#the-deduction-as-a-kernel-computation}

A deduction declaration also elaborates to computations of the Quivers Indexed Effect
Core (QIEC), the typed core language of QVR ([language reference](../reference/qvr/index.md)).
This is how a `program` calls a deduction and how the reference machine, the
library's interpreter for QIEC, scores one. The items become a closed family of
terms. The rules become a recursive derivation that chooses rules and split points
through a [`Search`](../developer/qiec.md#qvr-surface) effect and accumulates
weights through a `Weight[K]` effect, and the entry returns the inside weight of the
goal in the declared semiring, the number that
[`goal_weight`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView.goal_weight)
reads from the chart for the whole sentence. A program writes
`let chart = parse(D, sentence)` and scores `chart.goal_weight()`.
[`run_deduction`](../api/qiec/program_runtime.md#quivers.qiec.program_runtime.run_deduction)
runs a deduction alone on the reference machine, and
[`deduction_item`](../api/qiec/program_runtime.md#quivers.qiec.program_runtime.deduction_item)
builds the axioms of a system without a lexicon.

```python
from quivers.dsl import Compiler
from quivers.dsl.parser import parse_file
from quivers.qiec.program_runtime import run_deduction

compiler = Compiler(parse_file("docs/examples/source/pcfg.qvr"))
compiler.compile()
module = compiler.qiec_module
run = run_deduction(module, "PCFG", tokens=["the", "dog", "runs"])
# run.weight is the log inside weight of S over the sentence.
```

The chart computes each shared subproblem once and reuses it. The reference machine
instead enumerates derivations, choosing rules and split positions and combining the
alternatives with the semiring's addition, so it expresses rule choices and weights
as effects. In both, `depth` bounds the constructor depth of the categories in rule
conclusions, never the length of a derivation. The search recurs on smaller spans,
or on smaller yields for items of a
multiple context-free grammar [@seki1991mcfg], and a cycle of unary rules runs until its weights change by less
than `tolerance`, as the chart's iteration does. Learned weights reach the
computation through the module's `params` instance under the names the chart engine
gives its parameters, so a run can use the parameters of a fitted system.

[Parsing as an effectful computation](../tutorials/qvr/11-parsing-and-search.md)
compares the chart with the generated search on one grammar, and
[generated computations](../reference/qvr/generated-computations.md#deduction-declarations)
lists the families, effects,
[handlers](../reference/qvr/effects-and-handlers.md#handler-declarations), and entries
a deduction or a parser built from rule schemata adds.

## Schema export

A compiled `DeductionDecl` can also be exported as a
[panproto](https://github.com/panproto/panproto) schema under the protocol
`QVR_DEDUCTION_PROTOCOL`, whose vertex kinds are `deduction_system`,
`deduction_rule`, `deduction_atom`, `deduction_premise`, and `deduction_conclusion`.
panproto's diff and lens tools then apply to grammars.

```python
from quivers.dsl import Compiler
from quivers.dsl.parser import parse_file
from quivers.dsl.program_theory import extract_deduction_schema

compiler = Compiler(parse_file("docs/examples/source/ccg.qvr"))
compiler.compile()
schema = extract_deduction_schema(compiler)
# schema is a panproto.Schema, usable with diff / lens-generation tooling.
```
