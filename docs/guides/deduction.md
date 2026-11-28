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
inputs to one output [@klein2001parsing]. The library's
[`DeductionSystem`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem)
also takes an agenda factory and a chart factory, which choose how the system is
evaluated and not what it derives. The chart factory is called with the system's
`tolerance`, so a custom [`Chart`](../api/stochastic/agenda.md#quivers.stochastic.agenda.Chart)
accepts it, and its `lookup(pattern, bindings)` receives the values the run has
already bound. A system whose items cover stretches of the input declares
`goal_extent`, the positions an item covers, and its goal items must then cover the
whole input. A compiled deduction with a lexicon declares it for its span items;
without the declaration the goal is the goal predicate alone.

The default
[`HashChart`](../api/stochastic/agenda.md#quivers.stochastic.agenda.HashChart) keys
its lookups on the positions the bindings fix, each item that arrives is matched
only against premises of its own form, and the other premises of a rule are found by
[Generic Join](../api/stochastic/joins.md), so over a run a rule's comparisons stay
within the [AGM bound](../api/stochastic/agenda.md#finding-rule-instances) of its body
and no item is compared against every premise of every rule.
[`AgendaResult.work`](../api/stochastic/agenda.md#quivers.stochastic.agenda.AgendaResult)
reports the pops, matches, sums, products, and chart and queue operations of a run,
and `AgendaResult.applications` lists every rule application as
`(rule_index, rule_name, antecedents, conclusion)`.

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
([Strategy independence](#strategy-independence)). A
[plan](../api/stochastic/planning.md#monotone-polynomial-plans) under the
[`MonotonePolynomial`](../api/qiec/contracts.md#quivers.qiec.contracts.MonotonePolynomial)
contract computes it without iterating to a residual: in `LogProb` and `Markov` by Newton's method
with a certified interval around each value, in an idempotent semiring with a closure such as
`Viterbi` (max-plus) or `Tropical` exactly, after as many Newton steps as a cycle has items, and a
system whose least fixed point is infinite is refused
([Cyclic systems](../api/stochastic/recursion.md#monotone-polynomial-systems)).

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

Running a deduction returns a
[`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) that maps
items to weights. Its `weight` method reads one item, `enumerate` lists the items
matching a pattern, `derivations` lists the rule applications that derive an item as
`(rule_index, rule_name, antecedents)` triples, and `goal_weight` reads the goal. The weights are PyTorch tensors, and the agenda
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
all_S = view.enumerate(("span", Wildcard(name="i"), Wildcard(name="j"), S))

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
weights to within `1e-5`. Every agenda counts each derivation once, as in the agenda
algorithm of Eisner, Goldlust, and Smith [@eisner2005compiling]: when one item fills two premises
of a rule, as a single `NP` of weight $w$ fills both premises of `NP, NP ⊢ S`, the rule fires once
for that pair, and `S` receives $w \otimes w$. A rule's `weight_fn` must distribute over the
semiring sum in each premise weight for this count to hold.

The agenda reaches a cyclic system's value only by iterating, and a `tolerance` stops
that iteration once no weight changes by more than it. A cyclic system whose cycles
are linear, each strongly connected group of items deriving one another through at
most one premise of each rule, can be solved without iterating.
[`plan`](../api/stochastic/planning.md#quivers.stochastic.planning.plan) with the
recursion contract
[`LinearCyclic`](../api/qiec/contracts.md#quivers.qiec.contracts.LinearCyclic) and an
algebra with the
[`Star`](../api/core/algebras.md#quivers.core.algebras.Star) capability computes its
least solution through
[`solve`](../api/stochastic/recursion.md#quivers.stochastic.recursion.solve), which
visits the groups so that every premise from outside a group is final before the
group, and solves each group through the algebra's closure [@lehmann1977algebraic].
The plan takes the rule presentation of the system,
[`DeductionSystem.presentation`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem),
and refuses a group whose rules put it on both sides of a product that does not
commute, or whose closure diverges or cannot be decided.

For a rule presentation in span form, `plan` can choose a
[span executor](../api/stochastic/planning.md) for a weight, support, or marginal
query that keeps no derivation records. It sums over the split points for each
premise pair before applying the rules that use the pair. Empty spans are admitted
when their dependencies and the within-cell closure satisfy the recursion
contract. Unreachable items have the algebra's zero weight.

## Pre-registered stdlib {#pre-registered-stdlib}

`quivers.stochastic.stdlib` provides ready-made `DeductionSystem` instances:

| System | Items | Semiring | Agenda |
|----------------|---------------------------|------------|----------------------|
| `CCG` | `("span", i, j, category)`, with categories built by `("slash", dir, X, Y)`; application and harmonic composition [@steedman2000syntactic] | `LogProb` | CKY |
| `AB` | as for `CCG`; forward and backward application only, the AB (Ajdukiewicz-Bar-Hillel) grammar | `LogProb` | CKY |
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
[`DeductionSystem.__call__`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem)
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
terms. With a lexicon, the rules become a recursive derivation that chooses rules and split points
through a [`Search`](../developer/qiec.md#qvr-surface) effect and accumulates
weights through a `Weight[K]` effect, and the entry returns the inside weight of the
goal in the declared semiring, the number that
[`goal_weight`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView.goal_weight)
reads from the chart for the whole sentence. A program writes
`let chart = parse(D, sentence)` and scores `chart.goal_weight()`.
The single-sentence form also accepts a program-local rank-one
`Tensor[String]`. A parse with a plate reads its corpus from program data.
[`run_deduction`](../api/qiec/program_runtime.md#quivers.qiec.program_runtime.run_deduction)
runs a deduction alone on the reference machine, and
[`deduction_item`](../api/qiec/program_runtime.md#quivers.qiec.program_runtime.deduction_item)
builds the axioms of a system without a lexicon. Such a system computes a weighted chart by
repeatedly adding the axiom weights to rule contributions from the preceding chart; so does a
lexical system whose derivations no search bounds, with the axioms its lexicon reads off the
sentence. Beside each weight the chart counts the item's derivations, and it stops when two rounds
agree in every count, or in every weight, exactly or within `tolerance`. Equal counts mean that no
derivation is left, so the weight is exact up to rounding. Equal weights prove the same only when
the arithmetic is exact, as it is for `Boolean` and `Counting`, since a fixed point reached from
below is the least one; in rounded arithmetic they stop the chart only once a cycle is shown. A `LogProb` or `Viterbi` cycle whose weights stop changing while its derivations
keep growing has not shown its value, and neither has a stop within `tolerance`. Such a run raises
[`ContractNotMet`](../api/qiec/contracts.md#quivers.qiec.contracts.ContractNotMet) by default.
With `on_failure=FailurePolicy.RETURN_UNESTABLISHED`, `run_deduction` instead returns the last
iterate as an
[`Unestablished`](../api/qiec/contracts.md#quivers.qiec.contracts.Unestablished) value,
whose report names the stop and its residual. A nonconvergent system raises an execution error
when the caller's `fuel` budget is exhausted.

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
multiple context-free grammar [@seki1991mcfg]. A lexical deduction whose derivations no such
search bounds, such as one with an item that holds no position or a rule that is not linear,
iterates the chart instead, reading its axioms off the sentence through its lexicon. In rounded
arithmetic the chart stops on agreeing weights only once a cycle is shown, so an acyclic
deduction always settles with equal counts. A cycle of unary rules raises the number of
times a chain may repeat an item until two weights agree or the number of derivations stops
growing, and its stop is judged as the chart's is. A deduction `D` declares three handlers,
`D__search`, `D__collect`, and `D__settle`, whose
[`HandlerDef.foreign`](../api/qiec/effects.md#quivers.qiec.effects.HandlerDef) field states what
each computes: the semiring sum over the search's choices, the semiring product of the
weights, and the settlement of an iteration, each in the deduction's semiring given as
data, a
[`KernelSemiring`](../api/qiec/foreign.md#quivers.qiec.foreign.KernelSemiring). The runtime
builds each handler from that statement, never from its name, and requires the laws it relies on.
Learned weights reach the
computation through the module's `params` instance under the names the chart engine
gives its parameters, so a run can use the parameters of a fitted system.

[Parsing as an effectful computation](../tutorials/qvr/11-parsing-and-search.md)
compares the chart with the generated search on one grammar, and
[generated computations](../reference/qvr/generated-computations.md#deduction-declarations)
lists the families, effects,
[handlers](../reference/qvr/effects-and-handlers.md#handler-declarations), and entries
a deduction or a parser built from rule schemata adds.

### Rule weights computed from matching fields

On a checked `LogProb` deduction, `#[weight=operation]` supplies the log
weight of one rule application. The named `define` takes the variables of
the rule's instantiated telescope in presentation order, which starts with
their first occurrence in the written premises. Position fields have type
`Int`. Term fields have the deduction's own nominal item type. The operation
returns `LogWeight`, has an empty effect row, and takes no static arguments.
Its reachable calls cannot install a handler, since an empty row can hide
effects handled inside the operation.

<!-- compile: qiec -->
```qvr
object Token : FinSet 2
object Result : Real 1

deduction Agreement : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, D, S, lex
    rule join : span(I, K, Left, lex), span(K, J, D, lex) |- span(I, J, S, lex) #[weight=mass]
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]
        "y" : D = lex

define mass(i : Int, k : Int, left : Agreement__Item, j : Int) : LogWeight !{} =
    let matches <- Agreement__eq(left, construct Agreement__A() as Agreement__Item)
    if matches then
        return weight(log(0.2))
    else
        return weight(log(0.6))

program response : Token -> Result
    let chart = parse(Agreement, sentence)
    score result = weight_value(chart)
    return result

export response
```

For the sentence `["x", "y"]`, the two derivations use the same split
and differ in the category assigned to `"x"`. If those lexical choices
have log weights $a$ and $b$, the inside mass is
$0.2\exp(a) + 0.6\exp(b)$. Each successful application adds the computed
log weight to the log weights of its premises. In the checked chart, all
proof tuples with the same matching fields use the same factor, so
distributivity gives the same sum when those premises are already
aggregated. Negative infinity denotes zero mass. NaN and positive infinity
are rejected at the application.

Computed weights contribute to raw inside mass. They cannot be combined
with `normalize=lhs`, inherited through `parent`, or attached to a
`learnable` or `bounded` rule. The checked reference machine can execute
them through `parse` or `run_deduction`. The classic agenda, packed
outcome and proof queries, and the exact linear planner require weight
policies they can evaluate independently and refuse these operations.

### Guards on matched coordinates

A rule can name a Boolean computation with `#[guard=operation]`. The
computation receives the rule's matched position variables as `Int` values
in their telescope order. The example tests one split of a sentence.

<!-- compile: qiec -->
```qvr
object Token : FinSet 4
object Result : Real 1

deduction Guarded : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, S, word
    rule join : span(I, K, A), span(K, J, B) |- span(I, J, S) #[guard=first_split]
    lexicon
        "x" : A = word
        "y" : B = word

define first_split(i : Int, k : Int, j : Int) : Bool !{} =
    return k == 1

program response : Token -> Result
    let chart = parse(Guarded, sentence)
    score result = chart.goal_weight()
    return result

export response
```

The guard has an empty effect row and no static parameters. Its body uses
scalar `Int` and `Bool` operations. A true decision admits the matched
application. A false decision removes that application before weighting or
recording its edge. Distinct admitted applications still contribute
separately. A runtime error in a guard fails any query that reaches it.
The indexed query can check matches outside the goal. The equality
of the inside sums assumes that the guard evaluations succeed.

#### Guards on matched terms

A guard can also take matched constructor terms at the deduction's
generated `D__Item` type. That signature contains the entire matched
telescope in order, with each position variable at `Int` and each term
variable at `D__Item`. It can use exhaustive `case` on those terms and their
matched children, alongside the scalar operations. The deduction's own
constructor declarations determine the type and field order. A guard still
has no static parameters, named calls, or effects. Rule patterns and
closed lexical forms share one constructor vocabulary, with one arity
for each symbol. A constructor can occur only in the lexicon. The
grounded route reconstructs each matched term once per argument
occurrence, checking its constructors in field order.

Checked scalar `parse`, `run_deduction`, and packed lexical outcome queries
use the same checked guard body. The packed query evaluates the guard once
per completed match before adding an edge. Removing edges preserves the
span order certified from the unfiltered rule patterns. The production
probabilities of a locally normalized grammar still use their original groups.

For a successful packed outcome query, `deduction.outcomes.plan` records
the count from construction, including rejected matches. Its `guard_bindings`
payload contains the checked rule identities, their side conditions and
ordered binders, and the checked bodies referenced by those conditions.
Two rules referring to one condition share its body in the payload. The
plan describes the built-in item, category, or form projection and states
whether the positive rows share a common normalizer.

With $D$ decisions, scalar guard execution adds $O(D(1+Q)+L)$ construction
work, where $Q$ bounds the size of one guard body, including branch search
and field bindings, and $L$ counts all argument occurrences. Terms add
$O(PX+SU)$ work, where $P$ bounds a telescope scan, $X$ counts term argument
occurrences, $S$ bounds the work of one vocabulary scan, and $U$ counts
all reified constructor occurrences. A guard body has no calls or effects
and visits each executed syntax occurrence at most once. Guards run
sequentially, so their temporary space depends on the largest body and
the arguments and reified nodes of a single call. These bounds use the
word RAM model with expected hashing. The trace measures $D$ directly.
Refits reconstruct the chart and repeat the guards. Their decisions fix
support for the weight derivative and add no reverse tape.

The legacy agenda and external targets refuse guarded executions. Proof
values, packed Boolean proof interpretations, and the linear-cycle
executor require unconditional rules.

### Cyclic inside weights and finite-height proofs

For the one-token sentence `seed`, let $a$ be the lexical mass and $r$
the rule mass. The unary rule in `Linear` gives $x=a+rx$. The rule in
`Branching` uses the same item in two premise positions and gives
$x=a+rx^2$. Each premise independently chooses a proof. The examples share
the `LogProb` carrier and supply $\log a$ and $\log r$ as learned weights.
[Checked cyclic deductions](../examples/qiec/checked-cyclic-deduction.qvr)

The checked `run_deduction` entry iterates from zero. With
$(a,r)=(0.3,0.4)$ and the declared absolute log tolerance $0.15$, it stops at
$0.468$ for the unary rule and $0.336$ for the binary rule. The default
failure policy refuses those stops. The explicit return policy exposes
each finite iterate in an `Unestablished` wrapper with its residual.
Differentiating that iterate in the host gives the derivative of the
executed rounds. The settlement report makes no derivative claim.

The Python `plan` interface accepts the same compiled rule presentations.
`LinearCyclic` selects the span executor with unary closure. For $r<1$ it
sums the geometric series to $a/(1-r)$, with log derivatives $1$ and
$r/(1-r)$ with respect to $\log a$ and $\log r$. For $r=1$ and $a>0$ the
closure refuses divergence even though the checked log residual can become
small as the mass grows.

In a checked program, `let mass = parse_linear(Linear, sentence, 16)` selects
the same span executor and returns a scalar `LogWeight`. With the weights
above it gives $\log 0.5$. Its sentence is a rank-one token tensor that can
come from program data or a checked local binding. If the deduction names
a start category, the result is the mass of that category over the full
sentence. With no start category, it sums all full-span categories.

This entry admits `LogProb` lexical systems with atomic categories, unary
rules that preserve both endpoints, and binary rules that concatenate
adjacent spans. Each lexical occurrence has width one, so both premises
of a binary rule have smaller width than its conclusion. Thus by induction
on width, the lexical terms and binary contributions to a cell give a
known vector $b$ before that cell's unary rules run. The unary closure
computes the least sum $\sum_{k\geq 0} bU^k$, where $U$ contains the
total weight of the unary steps between each pair of categories.
Duplicate lexical alternatives and rule applications are added in
`LogProb`. An absent full-span goal has zero mass.

The final argument is a positive `Int` bound on the order of the unary
closure matrix. An extra parallel step occupies an auxiliary state, so
the limit includes those states. The call refuses a larger matrix before
allocating it. This bound covers the matrix order and leaves the number
of chart cells and the rest of the run's work unconstrained. The run records
the cap as a control alongside the selected algorithm and arithmetic.

Each rule uses its fixed weight or one learned parameter independent
of the rule's bindings. The entry refuses nullable axioms, logical forms,
binding-keyed and inherited rule weights, and left-hand-side production
normalization. Native execution uses `float64` or `float32` log weights.
The closure checks that its unary series converges for the supplied
weights whenever it evaluates a chart with lexical occurrences. It
distinguishes divergence from a case where the available arithmetic
cannot establish convergence or evaluate a usable pivot. For positive
goal mass in the open convergence domain, differentiating the finite
elimination computes the
derivative of the least solution in the run's precision. Indeed, for
$v=b+vU$, the differential is $dv=(db+v\,dU)U^*$. Logarithmic differentiation
then divides the relevant mass differential by the mass. Structural zero
cells use the `LogProb` zero-gradient convention. The ordinary `parse`
and `run_deduction` entries retain their iteration and settlement
contracts.

`MonotonePolynomial` selects the certified Newton plan for the branching
rule. For $a,r>0$ and $4ar<1$, the smaller quadratic root is
$x=2a/(1+\sqrt{1-4ar})$. The iterates from zero stay below that root by
monotonicity. Their limit is a fixed point by continuity, so this root is
the least solution. The plan encloses it and reports an implicit derivative
when it certifies $2rx<1$. In that domain,
$(1-2rx)\,dx=da+x^2\,dr$. At $a=r=1/2$ the root is $1$ and its Jacobian
is $1$. The plan refuses the requested bound and implicit derivative. The
checked iteration stops at $89/128$, with a log residual below $0.15$ and
a log error above $0.15$.

A forest plan with `Bounded(measure="height", limit=2)` asks for a different
sum. A lexical leaf has height one, so the unary example counts precisely
the lexical proof and its one-step extension, with mass $a(1+r)$. On
these same rules, a checked source `parse_proofs` query requires acyclic
spans and refuses the cycle. Its row budget cannot change the recurrence
into a finite-height model.

### Weighted outcomes of a lexical deduction

In a `program`, write `let readings = parse_outcomes(D, sentence, "form")` to get
the full-span logical forms and their log inside weights. The selectors
`"category"` and `"item"` return the category or the entire span item,
respectively. All three produce the closed family `D__Outcomes`, with
`D__outcome_end` for the empty list and `D__outcome_entry(key, weight, tail)`
for each entry. The key has type `D__Item` and the weight has type `Real`.
The query sums the weights of all derivations whose goal has the same
structural key, including derivations of distinct goal items that map to that
key. When the deduction declares a start category, the query first restricts
the full-span items to that category.

`parse_normalized_outcomes` subtracts the log of the total goal mass from each
log weight. A key with zero mass has no entry. A raw query with no positive
goal mass returns an empty list, and a normalized query raises an error in that
case. For a response with likelihood $L(k)$ under key $k$, a QVR computation
can traverse the normalized entries and sum $P(k\mid\text{sentence}) L(k)$,
then pass the logarithm of that sum to `score`.

The sentence can come from program data or from an earlier binding of type
`Tensor[String]([n])`. For instance, `let sentence <- tokenize(input)` can
call a checked computation before the query. The query takes the already bound
tensor and specializes its own extent argument to `n`. Its input type thus
matches the local's type exactly. Given the same tokens and weights, this call
computes the same masses as a query on program data. The extent may be a
program index specialized when the program runs. A computation with a
data-dependent output length needs a checked extent to feed this query.

These calls use the checked runtime and accept one sentence. The current
fragment has one lexical span judgment and unit-width lexical entries.
A rule combines contiguous spans in premise order or preserves the span of
one premise. In the latter case its two category heads must be fixed, and
the graph from unary premise heads to conclusion heads must be acyclic.
The pair of span width and category rank thus grows along each rule edge.
The rules use unit weights or weights from `normalize=lhs` groups.
Learned lexical log weights remain
available. The normalized groups also accept the same labeled probability
bindings as `parse`, with components in $[0,1]$ and positive total mass in
each bound group. The fragment excludes category depth bounds, open
hypotheses, and rule weights computed from bindings. A ground rule can
also supply an ordered interval, including an empty span. For instance,
`rule left_boundary : |- span(0, 0, E, empty)` supplies an empty span at
the first token boundary. A zero-premise rule's endpoints are natural
number literals and its conclusion is ground. The checker computes which
category heads may derive empty spans. A premise can preserve the
conclusion's width when the other
premises are empty. The checker requires an acyclic graph of fixed
category heads for all such transitions, including transitions from
rules with several premises.
Lexical categories and forms must inhabit the vocabulary of the deduction's
closed rule presentation. The query supports derivatives on fixed positive
support, with zero masses omitted from its output. When every span is
positive, a derivation for $n > 0$ tokens has at most $(2n - 1)(h + 1)$
nodes, where $h$ is the longest unary path. With nullable heads, $h$ instead
bounds every chain of equal-width transitions. A zero-width subtree has
at most $1+a+\cdots+a^h$ nodes for maximum rule arity $a$. A positive-width
derivation retains the bound on its positive-width nodes, with at most
$a$ zero-width subtrees attached to each. These bounds also account for
an empty input. The runtime uses them to check whether finite log weights
leave room for their sum and normalization in the run's number type.

The runtime grounds a packed forest and computes one inside value per useful
item, shared across all requested goals. For each incoming rule instance, a
proof independently chooses a proof for each premise occurrence. Distributivity
thus lets the fold multiply the premise sums and the rule weight, then sum
across incoming instances. After grounding, the fold uses one visit per useful
edge and one multiplication per premise occurrence. On scalar machine
numbers the fold and its reverse pass require arithmetic linear in those
counts, plus the cost of constructing and grouping the structural keys.
Grounding uses the indexed agenda to find the rule instances. The runtime
grounds the forest again on each invocation.

In a native checked trace, each completed built-in outcome fold emits
`deduction.outcomes`. The event names the selector, the raw or normalized
mass mode, and the indexed packed-forest schedule. It reports the number
of full-span goal items before grouping and positive-mass rows after
grouping. Construction work counts indexed lookups, candidates examined,
and the stored forest edges plus their premise references. Those records
are live during the query. The result contains the weighted rows.

The event states the derivative contract for the log weights on fixed
support. Its `unmeasured` field names fold and backward arithmetic, tensors
saved for backward, and peak bytes. The report also covers the item query
that precedes a user-defined key. The subsequent calls to that key are
outside its construction counters. When trace recording and the observer
are both disabled, the runtime skips constructing the event and counting
the forest records for it.

A successful `parse_outcomes` or `parse_normalized_outcomes` query with
an unguarded deduction and the `"item"`, `"category"`, or `"form"` key
also emits `deduction.outcomes.plan`. Decode the `plan` field with
`quivers.qiec.loads(event.detail["plan"])` to read its `PlanRecord`. The
record identifies the checked presentation, the log weight algebra, the
acyclic schedule, the selected rule joins, and the assumptions and cost
bounds of the fold. The exactness claim concerns which finite proof sum
the fold computes. Arithmetic uses the reported number type, with no
certified rounding error bound or host compute or memory budget.

The plan's `NoRetention` policy applies to returned proof provenance.
The query still builds a temporary packed forest, and the returned
weights can keep an autograd tape over finite log weights at fixed
support. The reverse bound includes dense adjoints for indexed reads
from the weight tensor. The ordinary `deduction.outcomes` event continues
to report its measured construction counters. For a category or form,
`ProjectedWeightRows` identifies the actual judgment, field position,
and declared index binder. The fold groups all full-span items that
share that field's value. A normalized plan wraps the selected weight
observable in `NormalizedWeightRows` and requires division. The
runtime checks that the common log normalizer is finite, which gives
positive total mass, before emitting the plan. Computing and differentiating that normalizer
scales linearly in the number of positive rows. When a custom key
consumes raw item rows, its grouping is a separate operation. Guarded
folds emit their ordinary event.

#### Ground rules and empty spans

This deduction supplies one empty left boundary and uses it to complete
the parse of `word`. The first rule applies only at boundary zero.

<!-- compile: qiec -->
```qvr
object Token : FinSet 1
object Result : Real 1

deduction WithBoundary : Token -> Token [semiring=LogProb, start=S]
    atoms span, E, A, S, empty, word, complete
    rule boundary : |- span(0, 0, E, empty)
    rule finish : span(I, K, E, empty), span(K, J, A, word) |- span(I, J, S, complete)
    lexicon
        "word" : A = word

define total(rows : WithBoundary__Outcomes) : Real !{} =
    case rows motive => Real
        WithBoundary__outcome_end =>
            return 0.0
        WithBoundary__outcome_entry(key, weight, tail) =>
            let rest <- total(tail)
            return exp(weight) + rest

program parse_with_boundary : Token -> Result
    let rows = parse_outcomes(WithBoundary, sentence, "form")
    let answer <- total(rows)
    return answer

export parse_with_boundary
```

#### Choosing a typed key

Instead of a string selector, pass the name of a checked computation. For
instance, `parse_outcomes(D, sentence, classify)` calls `classify` on each
positive-mass full-span goal. If `classify` has type `D__Item -> K` and an
empty effect row, the result has family `D__classify__Outcomes`. Its
constructors are `D__classify__outcome_end` and
`D__classify__outcome_entry(key, weight, tail)`, where `key : K`.

The key type may be `Bool`, `Int`, `String`, `Unit`, or a closed algebraic
family without parameters or indices whose fields have these key types.
Constructor identity and ordered field
values determine equality, recursively. The query refuses key types whose
equality it has not established, including `Real`, functions, and indexed
families. The key computation must terminate on the goals it receives.
It and the computations it calls may not install effect handlers.

If $g$ is a goal item, the chart has already summed the weights of all its
proofs into $m(g)$. The mass of a key $k$ is thus
$\sum_{g:\,\mathrm{classify}(g)=k} m(g)$. Each proof contributes through
its goal exactly once, and the caller can normalize by $\sum_g m(g)$.
The query runs the projection once per supported goal, then groups keys
in a hash table. Its extra work depends on the number of goal items and
the cost of computing and comparing their keys, even when the proof count
is much larger. The masses and their normalizer retain their gradients.
The native trace emits a separate `deduction.grouped_outcomes` event after
this stage. It records the key type, number of input rows, distinct keys,
positive output rows, and whether this stage normalizes the result. The
underlying packed query reports its own raw item or proof masses.
The trace lists the key calculation, comparisons, and reductions as
unmeasured. Run `python -m tests.benchmarks.deduction_key_grouping` to
measure the checked query and its isolated reducer as item count, key
count, and key depth vary. The benchmark checks the masses and gradients
at each parameter point in the refit.

When a query maps unguarded lexical item rows with a named key,
`deduction.grouped_outcomes.plan` identifies the checked mapper, the key
computation it calls, and the request that groups the results. Its
`Outcome` uses that same computation identity. With normalization,
`NormalizedWeightRows` wraps this outcome and records the division
requirement and the common normalizer's cost. Chart construction has its
own event and still computes raw item weights.

The conversion makes one key call per row and forwards its log weight.
The plan counts key evaluation separately from hash-table work. With
expected table access, the grouping takes time linear in the number of
rows and the total size of their keys. Collisions can require repeated
comparisons. The reverse bound covers the grouping's scalar log sums at
fixed keys and support, including one additional pass over the positive
rows when normalizing. Proof keys and calls that supply already mapped
rows emit the ordinary grouping event.

#### Returning weighted proofs

Use `parse_proofs(D, sentence, limit)` when a later computation needs to
inspect the rule applications of each proof. `limit` is a positive `Int`.
The runtime counts the structural proofs of all selected goals against
this limit before constructing any proof values. The count includes
zero-weight proofs. An exceeded limit raises an error, and a successful
query returns each positive-weight proof with its log weight.

A deduction can also declare `proof_nodes=512` in its option block.
This positive integer below $2^{53}$ limits the total rule and axiom occurrences in all
selected structural proofs. A shared subtree contributes once per premise
position that uses it. The node limit applies before filtering zero-weight
proofs, and it excludes result list entries and the packed chart itself.
Omitting the option leaves the proof-count limit alone.

Run `python -m tests.benchmarks.deduction_proof_list` to measure the
checked listing as the number of alternatives and the proof length
vary. The benchmark counts proof constructors and checks that a
refused cap stops before any proof node is built. The chart and its
saturating count still incur work before that decision.

`parse_normalized_proofs` normalizes over these same proofs and requires
positive total mass. A raw query returns an empty list when no proof has
positive mass. Two proofs of the same goal keep separate entries. The
rule and lexical weight bindings of `parse_outcomes` are available here
as well.

The result family is `D__Proofs`. Its `D__proof_end` constructor is empty.
`D__proof_entry[i, j, c, f](proof, weight, tail)` binds the endpoints,
category, and form indices of a proof of type `D__SpanProof(i, j, c, f)`.
For a deduction without forms, the entry has three indices. The category
and form indices inhabit `D__Term`. The proof's constructors include
`D__rule_<name>` for each rule and `D__axiom_span` for a lexical occurrence.
A QVR `case` can bind these indices and pass the proof to another checked
computation. Enumeration makes no requests on behalf of that computation.
Reifying positive-weight proofs takes work proportional to their explicit
proof sizes.

The native trace emits `deduction.proofs` after a successful enumeration.
It records the budget, the number of structural proofs before filtering,
the number of typed result rows, and the packed-forest construction
counters. Reification work and backward storage are unmeasured. If the
budget is exceeded, `deduction.proofs.refused` records the budget and the
reason. With only a proof-count limit, the check saturates counts at the
limit plus one. A refusal can report a lower bound on the total.

A successful run also emits `deduction.proofs.plan`. As with the item
query, decode its `plan` field with `quivers.qiec.loads`. Its
`WeightedProofRows` observable records whether the actual result was
normalized. Its controls give the structural proof cap and, when
present, the unfolded-node cap. These limits bound which lists may be
returned. They leave the chart size and the tape's peak memory open.

The proof rows are the result values. The provider keeps no additional
provenance after returning them, though it builds a temporary packed
forest and shares intermediate derivations during enumeration. Each
useful subderivation can be placed at a fixed position in a complete
goal proof. Distinct subderivations then map to distinct pairs of a
complete proof and a position in it. This bounds the size of the shared
list by the unfolded structural output. The report also accounts for
prefix copying when a rule enumerates all combinations of its premises,
and for checking and reifying the typed proof values.

With a node limit, the same packed traversal counts proofs and their
total unfolded size. For a partial combination with proof count $p$ and
node sum $q$, adding a premise occurrence with count $c$ and node sum $s$
gives $(pc, ps+qc)$. Each choice of that premise repeats the earlier nodes,
and each earlier combination repeats the chosen premise's nodes. An edge adds
$pc$ root nodes after the last premise. Summing alternatives and distinct
goals gives the structural output size. A missing premise contributes
$(0,0)$, so an impossible edge adds no nodes even when an earlier premise
already exceeded a limit.

The traversal saturates both counts at one above the larger limit. Every
formula uses nonnegative integer sums and products, which preserve this
truncation. After selecting the goal ancestors, the check makes one pass
over them. This uses $O(V+E+R)$ arithmetic operations for $V$ items,
$E$ edges and $R$ premise references in that part of the forest. Selecting
the ancestors also scans the original chart's item lists. The count
tables use $O(V)$ integers whose bit length is logarithmic in the larger
limit. This work follows chart construction and precedes proof
enumeration, proof weights and constructor allocation.
The trace includes `proof_node_budget` when configured. A
`too-many-proof-nodes` refusal also reports
`minimum_structural_proof_nodes`.

For instance, two rules can prove the same full span. This program uses
the packed chart to return their total weight by summing the two proof rows.

<!-- compile: qiec -->
```qvr
object Token : FinSet 2
object Mass : Real 1

deduction Reading : Token -> Token [semiring=LogProb, start=S, proof_nodes=6]
    atoms span, A, B, C, S, meaning, word
    rule first : span(I, K, A, word), span(K, J, C, word) |- span(I, J, S, meaning)
    rule second : span(I, K, B, word), span(K, J, C, word) |- span(I, J, S, meaning)
    lexicon
        "left" : A = word
        "left" : B = word
        "right" : C = word

define total(rows : Reading__Proofs) : Real !{} =
    case rows motive => Real
        Reading__proof_end =>
            return 0.0
        Reading__proof_entry[i, j, c, f](proof, weight, tail) =>
            let rest <- total(tail)
            return exp(weight) + rest

program proof_mass : Token -> Mass
    let proofs = parse_proofs(Reading, sentence, 2)
    let weight <- total(proofs)
    return weight

export proof_mass
```

#### Interpreting a proof

An `interpret` declaration assigns one checked computation to each rule and
lexical axiom slot. Its result is a named computation over the indexed proof
family. The computation interprets the premise proofs in their declared order,
then passes their results and the constructor's value fields to the selected
operation. For `axiom span`, the value field is the lexical word.

<!-- compile: qiec -->
```qvr
object Token : FinSet 2
object Result : Real 1

deduction Words : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, S, word
    rule join : span(I, K, A), span(K, J, B) |- span(I, J, S)
    lexicon
        "left" : A = word
        "right" : B = word

interpret difference : Words -> Real
    rule join = subtract
    axiom span = word_value

define subtract(left : Real, right : Real) : Real !{} =
    return left - right

define word_value(word : String) : Real !{} =
    if word == "left" then
        return 3.0
    else
        return 1.0

define read_example() : Real !{} =
    let left = construct Words__axiom_span[0, 1, A]("left") as Words__SpanProof(0, 1, A)
    let right = construct Words__axiom_span[1, 2, B]("right") as Words__SpanProof(1, 2, B)
    let proof = construct Words__rule_join[0, 1, 2](left, right) as Words__SpanProof(0, 2, S)
    difference[0, 2, S](proof)

program difference_example : Token -> Result
    let value <- read_example()
    return value

export difference_example
```

Each operation takes one carrier value per premise, then its constructor's
ordinary fields, and returns the carrier at the rule's conclusion. For a
lexical span the carrier may use the span judgment's indices. `x0` and
`x1` are its left and right endpoints. The category is `x2`, and the
form, when present, is `x3`. For instance, a declared family
`Measurement(position : Nat)` can serve as the carrier
`Measurement(x1)`. A rule joining spans `(I, K)` and `(K, J)`
then takes `Measurement(K)` and `Measurement(J)` in that order and
returns `Measurement(J)`. An operation that takes static arguments
must take the full constructor telescope in its declared order and
at its declared sorts. A closed operation can omit that telescope
when its input and result types agree at every constructor instantiation.

A rule operation may use a closed effect row independent of its static
arguments. The interpretation's row is the union of the reachable operation
rows, and callers handle those instances as for any checked computation.

The result type template binds only the indices of the judgment. Lowering
gives the fold's static parameters names distinct from the case motive's
index binders. In each constructor branch it substitutes the premise
indices into their carrier templates and the conclusion indices into the
result template. By induction on the input proof, the recursive calls
return the types required by the rule operation. At runtime the selected
constructor supplies the branch indices by identity, including indices
hidden from the constructor's result type. This preserves the distinction
between, for instance, the right endpoint of the first premise and the right
endpoint of the conclusion.

The checked case and calls follow the initial algebra for the presentation's
finite ordered proof trees. They visit each constructor occurrence once and
make one rule or axiom call there. The traversal therefore makes as many
operation calls as the proof has nodes. The operations' own work, static
checking, and effect trace storage are additional costs. Reusing a premise
value as part of a proof preserves both occurrences. Its interpretation
executes once for each, with separate dynamic addresses. A finite proof can
be interpreted even if the grammar admits infinitely many other proofs.
Finding all proofs in a chart is a separate query with its own finiteness check.

#### Grouping proof meanings

Use `parse_proof_outcomes(D, sentence, limit, meaning)` if the outcome
depends on the proof, including when different proofs give the same chart
item. For each supported proof the query calls the named pure
`meaning` computation at the proof's indices. A pure `interpret` declaration
can supply that computation. Its result is a closed outcome key of the same
kind supported by the named keys of `parse_outcomes`.

The query adds the weights of proofs with equal keys. It returns a
`D__meaning__Outcomes` list with one entry per key. An entry contains the
meaning and its unnormalized log weight. The normalized form,
`parse_normalized_proof_outcomes`, divides these masses by their total and
requires positive total mass. Each key receives the sum of the mass
assigned to its proofs, so grouping preserves the total mass and its
derivative with respect to the grammar weights.

This query performs the same structural budget check as `parse_proofs`
before rendering any proofs or calling the key computation. It then calls
the key once per returned proof and groups the result. The cost includes
the explicit proof sizes, the key computations, and hashing the resulting
key values. When the key depends only on the conclusion item, use
`parse_outcomes` with an item key to sum the proofs first in the packed chart.

The `deduction.grouped_outcomes.plan` trace describes the grouping stage
when the checked mapper uses one declared proof row owner. It records
the call site of the named key. The mapper opens each row's indices and
passes its proof and weight through in their original roles. The upstream
`deduction.proofs.plan` records the structural budgets and proof enumeration.
For $N$ supplied rows, map construction takes $O(N+T)$ work, where $T$ counts
static specialization and key execution. Grouping takes expected $O(N+S)$
work under the word RAM model with expected hashing, where $S$ counts the
words in all produced keys. At fixed support and keys, one reverse pass
takes $O(N+D)$ additional work for $D$ distinct keys. Normalization adds a
pass over at most $D$ positive rows. The forward space bound also counts
the supplied proofs, the peak workspace of one specialization and key call,
and the static graphs and indices retained in the specialization caches.

#### Packing Boolean proof meanings

Use `parse_packed_boolean_outcomes(D, sentence, meaning)` for an
`interpret meaning : D -> Bool` whose operations can be tabulated. The
[parity example](../examples/qiec/packed-boolean-proof-outcomes.qvr) assigns
each word one of two readings and computes the parity of a proof's readings.
Its checked likelihood sums the probabilities of every odd-parity proof.

The result is a `D__BooleanOutcomes` list. A
`D__boolean_outcome_entry` contains a Boolean key, its log mass, and
the remaining rows. The list ends with `D__boolean_outcome_end`.
It contains only positive-mass rows, in false-then-true order.
`parse_normalized_packed_boolean_outcomes` divides by the total goal mass and
refuses an empty or zero-mass result. Either list can serve as input to
`outcome_law` when the total mass is positive.

A `normalize=lhs` grammar can take program-supplied production probabilities
in either packed query. Append the same `("left-hand side", probabilities)`
pairs used by [ordinary parsing](#priors-on-production-probabilities)
after the interpretation. These probabilities replace the grammar's local
weights for that call. The choice to normalize the returned meaning masses
still divides by the total mass of the current sentence.

This query uses the finite acyclic lexical fragment of `parse_outcomes`.
Rules have at most two ordered premises and carry no additional value
fields. The interpretation's rule and axiom operations have empty effect
rows, take no static parameters, and install no handlers in any reachable
call. QIEC calls the actual operations once for each declared lexical
word and each ordered tuple of premise truth values. Thus those calls must
succeed at all entries, including tuples and words the current sentence
cannot reach. If a call fails, the packed query fails. Use the budgeted
proof query for computations that need a proof's indices or are defined
only on the reachable proofs.

For an item $v$, let $m_v(b)$ be the mass of proofs with meaning $b$.
An edge $e$ with ordered premises $u_1,\ldots,u_k$ has weight $w_e$ and
an operation $f_e$. In probability scale the fold computes

$$
m_v(b) = \sum_{e 	o v} w_e
    \sum_{\substack{(b_1,\ldots,b_k) \\ f_e(b_1,\ldots,b_k)=b}}
       \prod_{j=1}^{k} m_{u_j}(b_j).
$$

An axiom contributes its own weight at the value of its lexical operation.
For each rule, the Cartesian product of its premise proof occurrences
partitions by the tuple of their meanings. Distributivity factors the
sum within each tuple into the product of its premise masses. Induction
in topological order gives the same mass as interpreting and grouping
the individual proofs. Repeated premises contribute one factor per
premise position, including when both positions refer to the same item.

The `deduction.boolean_outcomes` trace records the indexed
forest construction separately from the folded portion that reaches a
goal. Each rule edge in that portion contributes at most four meaning
tuples, each with at most two LogProb products. Thus after grounding, the
fold has linear scalar arithmetic and graph storage in the number of
items and edges. The retained differentiation graph has the same scalar
size bound. The key table has one entry per distinct declared lexical
word and at most four entries per rule. Each run builds the table,
grounds the forest, and differentiates the current numerical weights
at fixed truth tables. Those costs include the authored operation
evaluations, and the trace leaves actual allocator and backward
measurements unspecified.

For $n\ge 1$ copies of the ambiguous word in the example, there are $2^n$
proofs and $10n-2$ meaning tuples in the fold over goal ancestors.
The forest construction still records edges for shorter spans that
the goal never uses, so its provenance grows quadratically on this
grammar. QIEC's step budget applies to the interpreted computation
and its truth calls. The packed host fold uses its separate finite
grounding contract.

The opt-in CPU benchmark checks the likelihood and gradient at every
refit against a closed form. It reports forward and reverse tensor
dispatch, logical saved elements, the tensor allocator peak, and the
traced Python peak separately. For this fixture it checks the derived
autograd tape bound of $28n+20$ saved elements. Run it from the
repository root.

```bash
python -m tests.benchmarks.packed_boolean_outcomes
```

#### Encoding two proofs of one logical form

The [shared proof encoder](../examples/qiec/shared-proof-encoder.qvr)
uses one weighted deduction and two interpretations. `meaning` gives a
nominal logical form, and `embed_proof` folds the ordered rule applications
into a two-dimensional code. For the sentence `x y`, both proofs conclude
the same item and return the logical form `approved`. The two rule
operations give codes $(7,2)$ and $(11,-2)$, respectively. Both
interpretations consume proofs supplied by the same packed parse.

The model evaluates a neural response at each code and weights it by that
proof’s normalized mass. If $q_r$ is a proof’s probability, $c_r$ its code,
and $p(c_r)$ the response probability for the common form, the observation
has probability $\sum_r q_r p(c_r)$. The model preserves proof occurrences
even though their logical forms are identical. Replacing the codes by one
item-level code changes the response and removes the lexical-weight
gradient in this example. Scoring an average code also changes the
response because the neural operation is nonlinear. The authored proof
query has a bound of two derivations and checks that bound before folding
the requested proofs.

#### Scoring a response with a prover

The example computes the tokens of “item reviewed” inside a QVR program. Its lexical rules
yield two derivations of the form `approved` and one of `unresolved`. The
form selector adds the two masses for `approved` before normalizing. The
checked `to_claim` computation uses the parser's structural equality to
construct an item of the prover's own family. For each such item, the call
`Prover__run[1]([claim], [true])` asks whether the one supplied axiom proves
the goal in the Boolean semiring.

<!-- compile: qiec -->
```qvr
object Word : FinSet 2
object Weight : Real 1

deduction Reading : Word -> Word [semiring=LogProb, start=S]
    atoms span, A, B, C, D, S, lexical, approved, unresolved
    rule first : span(I, K, A, lexical), span(K, J, D, lexical) |- span(I, J, S, approved)
    rule second : span(I, K, B, lexical), span(K, J, D, lexical) |- span(I, J, S, approved)
    rule third : span(I, K, C, lexical), span(K, J, D, lexical) |- span(I, J, S, unresolved)
    lexicon
        "item" : A = lexical #[learnable]
        "item" : B = lexical #[learnable]
        "item" : C = lexical #[learnable]
        "reviewed" : D = lexical

deduction Prover : Word -> Word [semiring=Boolean, start=Goal]
    atoms Accepted, Pending, Licensed, Goal
    rule license : Accepted |- Licensed
    rule discharge : Licensed |- Goal

define pack_tokens(first : String, last : String) : Tensor[String]([2]) !{} =
    return [first, last]

define to_claim(form : Reading__Item) : Prover__Item !{} =
    let accepted = construct Reading__approved() as Reading__Item
    let matches : Bool <- Reading__eq(form, accepted)
    if matches then
        return construct Prover__Accepted() as Prover__Item
    else
        return construct Prover__Pending() as Prover__Item

define response_mass(rows : Reading__Outcomes, yes : Real, no : Real) : Real !{} =
    case rows motive => Real
        Reading__outcome_end =>
            return 0.0
        Reading__outcome_entry(form, log_weight, tail) =>
            let claim : Prover__Item <- to_claim(form)
            let proved : Bool <- Prover__run[1]([claim], [true])
            let rest : Real <- response_mass(tail, yes, no)
            if proved then
                return rest + exp(log_weight) * yes
            else
                return rest + exp(log_weight) * no

program assess(first : String, last : String, yes : Real, no : Real) : Word -> Weight
    let sentence <- pack_tokens(first, last)
    let readings = parse_normalized_outcomes(Reading, sentence, "form")
    let likelihood <- response_mass(readings, yes, no)
    score response = log(likelihood)
    return response

define read_network() : Tensor[Real]([4]) !{params} =
    perform params.get[Tensor[Real]([4])]("response.network")

define neural_score(feature : Real, theta : Tensor[Real]([4])) : Real !{} =
    return sigmoid(theta[0] + theta[1] * tanh(theta[2] + theta[3] * feature))

define neural_response(proved : Bool, theta : Tensor[Real]([4])) : Real !{} =
    if proved then
        neural_score(1.0, theta)
    else
        neural_score(-1.0, theta)

define neural_mass(rows : Reading__Outcomes, theta : Tensor[Real]([4])) : Real !{} =
    case rows motive => Real
        Reading__outcome_end =>
            return 0.0
        Reading__outcome_entry(form, log_weight, tail) =>
            let claim <- to_claim(form)
            let proved <- Prover__run[1]([claim], [true])
            let probability <- neural_response(proved, theta)
            let remaining <- neural_mass(tail, theta)
            return remaining + exp(log_weight) * probability

program assess_neural(first : String, last : String, response : Bool) : Word -> Weight
    let sentence <- pack_tokens(first, last)
    let readings = parse_normalized_outcomes(Reading, sentence, "form")
    let theta <- read_network()
    let probability <- neural_mass(readings, theta)
    observe response <- Bernoulli(probs=probability)
    return probability

instance random : Random

define sample_response(law : Sampleable[Reading__Item], yes : Real, no : Real) : Real !{random} =
    let form <- perform random.sample[Reading__Item](site("reading"), law)
    let claim <- to_claim(form)
    let proved <- Prover__run[1]([claim], [true])
    if proved then
        return yes
    else
        return no

program assess_drawn(first : String, last : String, response : Bool, yes : Real, no : Real) : Word -> Weight
    let sentence <- pack_tokens(first, last)
    let readings = parse_outcomes(Reading, sentence, "form")
    let law = outcome_law(readings)
    let probability <- sample_response(law, yes, no)
    observe response <- Bernoulli(probs=probability)
    return probability

export assess
export assess_neural
export assess_drawn
```

The likelihood sums the reading probabilities weighted by `yes` when the
prover succeeds and `no` otherwise. For lexical log weights $w_0,w_1,w_2$,
the likelihood is

$$
\frac{(e^{w_0}+e^{w_1})\,\mathrm{yes}+e^{w_2}\,\mathrm{no}}
     {e^{w_0}+e^{w_1}+e^{w_2}}.
$$

Save the source as `weighted_parser_prover.qvr` and run it with the default
lexical log weights and different response probabilities.

```bash
qvr run weighted_parser_prover.qvr assess \
    --data 'first="item"' --data 'last="reviewed"' --data yes=0.8 --data no=0.3
```

This example uses a pure typed adapter and Boolean deduction. Typed rule
interpretations with effects require a separate account of what each rule
does and which uses of its result may share an effect.

`assess_neural` uses the same parser and prover to compute the response
probability. The prover's result determines the input to a one-hidden-unit
network with the four parameters named `response.network`. The program
reads them once through `Param` and observes a Boolean response under the
mixture of the network's predictions over readings. Differentiating its
log joint reaches both the lexical weights and the network parameters.
Supply the parameter values by name to
[`run_program`][quivers.qiec.program_runtime.run_program].

`assess_drawn` uses the same weighted readings as a latent distribution.
`outcome_law` preserves their nominal type and normalizes the log masses.
It accepts weighted item rows and rows grouped by a pure typed outcome
key. Construction is deterministic and requires positive total mass.
The program's `Random` operation draws one form and replay scores the
same form under the current grammar weights. Summing the joint scores
of all supported forms recovers the exact response likelihood above.

## Schema export

A compiled `DeductionDecl` can also be exported as a
[panproto](https://github.com/panproto/panproto) schema under the protocol
`QVR_DEDUCTION_PROTOCOL`, whose vertex kinds are `deduction_system`,
`deduction_rule`, `deduction_atom`, `deduction_premise`, and `deduction_conclusion`.
panproto's diff and lens tools then apply to grammars. The export reads the deduction's
[rule presentation](../api/qiec/rules.md#deductions-and-rule-schemata), the one reading of its
rules that the chart engine and the reference machine both derive theirs from. Each rule keeps
its identity and its weight parameter, each pattern its judgment, and the system the status of
every law its semiring claims.

```python
from quivers.dsl import Compiler
from quivers.dsl.parser import parse_file
from quivers.dsl.program_theory import extract_deduction_schema

compiler = Compiler(parse_file("docs/examples/source/ccg.qvr"))
compiler.compile()
schema = extract_deduction_schema(compiler)
# schema is a panproto.Schema, usable with diff / lens-generation tooling.
```
