# Weighted Deduction Systems

A deduction system repeatedly applies rules to facts already established.
For a parser, one fact might say that words 0 through 1 form a noun phrase,
and another that words 1 through 3 form a verb phrase. A rule combines
them into a sentence spanning 0 through 3. A **[chart](#charts-as-first-class-differentiable-values)** stores facts and
their accumulated weights; an **agenda** schedules facts still to process.
This separates what a valid derivation is from the order used to find it.


QVR's `deduction` block declares a weighted deductive system: a
hypergraph of inference rules over an item algebra, paired with a
[semiring for weight aggregation](core.md#algebras-enrichment-algebras). A single agenda-driven runtime
evaluates the system under strategies for CKY, Earley, Viterbi, A*,
Knuth's algorithm, semi-naive Datalog, and proof search.

## The framework

A rule with several premises is a hyperedge: one connection consumes
several inputs to produce one conclusion. Its weight combines with the
premise weights. If two different derivations reach the same conclusion,
the semiring specifies how their contributions combine. Boolean weights
ask whether any derivation exists; LogProb adds probabilities in log space;
Viterbi retains the best-scoring derivation. The same rule shapes can
thus answer different queries. [@goodman1999semiring]


Categorically, a weighted deduction system is a tuple

$$\mathcal{D} = (I, R, K, \mathrm{ax}, \mathrm{goal})$$

with $I$ an *item algebra* (an abstract data type of judgments / facts / sequents), $R$ a *hypergraph* of inference rules (each a premise multiset, a conclusion, and a weight in $K$, universally quantified over pattern variables), $K$ an *algebra* / semiring for weight aggregation, $\mathrm{ax}$ an *axiom injector* producing the input's lexical / boundary items, and $\mathrm{goal} \subseteq I$ a *termination predicate*.

The chart assigns an accumulated weight to each item. In settings with an
appropriate complete order and monotone rule operator, a least-fixed-point
account describes closure under the rules. The existence of such a fixed
point, convergence to it, and independence from the processing order require
hypotheses on the algebra, cycles, and evaluation strategy.

See [Further reading](#further-reading) below.

## QVR syntax

```text
object Atom : FinSet 4
deduction CG : Atom -> Atom [semiring=LogProb, start=S, depth=4]
    atoms NP, S, VP

    rule fwd_app   : X/Y, Y     |- X
    rule bwd_app   : Y, Y\X     |- X
    rule harm_comp : X/Y, Y/Z   |- X/Z
```

The unified option block carries the deduction-level configuration (semiring, start symbol, depth bound, optional `tolerance` for convergent-cycle LogProb, optional `max_iterations` cap). The indented body declares the atom set and rules; single-uppercase-letter identifiers in rule patterns (`X`, `Y`, `Z`) bind as wildcards, and non-wildcard names match literally. The declaration compiles to a `DeductionSystem` registered under the declared name (here `CG`).

## Charts as first-class differentiable values

For log weights `-0.3` and `-0.4`, a unit-weight binary rule contributes
`-0.7` to its conclusion. A second derivation contributes through
log-sum-exp, not ordinary addition of the two log weights. Differentiating
the final score can then update the lexical or rule parameters that
supported those derivations. A score for the best derivation and a score
summed over all derivations train different objectives.


Running a deduction yields a `ChartView`, a $K$-presheaf value with query methods:

<!-- python: skip -->
```python
view = CG([
    (("span", "NP", 0, 1), torch.tensor(-0.3, requires_grad=True)),
    (("span", "VP", 1, 3), torch.tensor(-0.4, requires_grad=True)),
])

log_weight = view.weight(("span", "S", 0, 3))
all_S = view.enumerate(("span", "S", Wildcard("i"), Wildcard("j")))
total = view.goal_weight()

loss = -log_weight
loss.backward()  # gradients flow back to the axiom weights
```

The weights are `torch.Tensor` values; autograd flows through the agenda's semiring operations (`semiring.times`, `semiring.plus`) so any `requires_grad=True` rule weight or axiom weight feeding into the deduction is differentiable. This enables grammatical-inference / weighted-deduction loss objectives end-to-end.

## Strategy independence

Boolean OR and Viterbi max are idempotent: adding the same contribution
twice changes nothing. LogProb addition is log-sum-exp and is not
idempotent: two copies of a derivation of weight `p` contribute `2*p`.
Strategy-independent accumulation must thus account correctly for
duplicate work and cycles, as well as satisfy termination or convergence
conditions. Semiring parsing makes these algebraic distinctions explicit
[@goodman1999semiring], §3. The `test_strategy_independence_idempotent` test runs the same deduction under CKY-sweep, semi-naive FIFO, depth-first LIFO, and Earley-FIFO agendas and asserts the chart's goal-item weights agree to within `1e-5`.

## Pre-registered stdlib

The `quivers.stochastic.stdlib` module ships ready-to-use `DeductionSystem` instances:

| System         | Item algebra              | Semiring   | Strategy             |
|----------------|---------------------------|------------|----------------------|
| `CCG`          | `("span", cat, i, j)`     | LogProb    | CKY sweep            |
| `Lambek`       | sequents over patterns    | LogProb    | CKY sweep            |
| `STLC`         | check / synth judgments   | Boolean    | depth-first          |
| `MLTT`         | typing judgments          | Boolean    | depth-first (bounded depth) |
| `Datalog`      | ground atoms              | Boolean    | semi-naive FIFO      |
| `Dijkstra`     | `("dist", node)`          | Viterbi    | Knuth's              |
| `HMM`          | `("alpha", t, state)`     | LogProb    | CKY sweep            |
| `ViterbiHMM`   | same                      | Viterbi    | CKY sweep            |
| `EditDistance` | `("dist", i, j)`          | Viterbi    | CKY sweep            |

Each is a callable: `view = Datalog(axiom_list)` returns a `ChartView`.

## Strategy factory functions

The agenda module exposes canonical strategy factories that may be passed to the `agenda_factory=` parameter of a `DeductionSystem`:

- `cky_agenda()`, `earley_agenda()`, FIFO (semi-naive).
- `viterbi_agenda(priority_fn)`: priority-queue.
- `astar_agenda(g_plus_h)`: priority-queue with an admissible heuristic.
- `knuth_agenda()`: best-first hyperpath search [@knuth1977generalization; @nederhof2003weighted].
- `depth_first_agenda()`: LIFO (proof search).
- `semi_naive_agenda()`: FIFO for Datalog-shaped systems [@mcallester2002complexity].

## panproto integration

A `DeductionDecl` compiles to a panproto schema via `QVR_DEDUCTION_PROTOCOL`. Vertex kinds: `deduction_system`, `deduction_rule`, `deduction_atom`, `deduction_premise`, `deduction_conclusion`. Schema morphisms over the protocol correspond to deduction-system specializations.

```python
from quivers.dsl import Compiler
from quivers.dsl.parser import parse_file
from quivers.dsl.program_theory import extract_deduction_schema

compiler = Compiler(parse_file("docs/examples/source/ccg.qvr"))
compiler.compile()
schema = extract_deduction_schema(compiler)
# schema is a panproto.Schema, usable with diff / lens-generation tooling.
```

## The deduction as a kernel computation

The [generated computation reference](../reference/qvr/generated-computations.md#deduction-declarations) traces this expansion into choices, weights, and collecting handlers. The [effect interface reference](../reference/qvr/effects-and-handlers.md#prelude-interfaces) defines the requests those computations make.

The chart engine reuses stored subresults. The generated computation
expresses the same bounded problem by choosing rules and recursively
deriving their premises. Thus the chart computes shared subproblems once,
while the generated form expresses rule choices and accumulated weights
as effects. Both calculations stop at the declared search bound; increasing
that bound can add derivations.


The same declaration elaborates to QIEC computations, which is how a
`program` calls a deduction and how the reference machine scores one. The
items become a closed family, the rules become a recursive derivation
that searches through a [`Search`](../developer/qiec.md#qvr-surface)
effect and adds weights through `Weight[K]`, and the entry answers the
inside weight of the goal in the declared semiring, the number
[`goal_weight`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView.goal_weight)
reads off the chart for the whole sentence. A program writes
`let chart = parse(D, sentence)` and scores `chart.goal_weight()`;
[`run_deduction`](../api/qiec/program_runtime.md#quivers.qiec.program_runtime.run_deduction)
runs the deduction alone on the reference machine, and
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

The reference machine enumerates derivations, choosing split positions
and rules through the search handler and combining the shots by the
semiring's addition, while the agenda engine tabulates them; the two
agree on every system whose derivations of the goal fit the declared
`depth`. Learned weights reach the computation through the module's
`params` instance under the names the engine keys its parameters by, so
a run may be given a parameter store built from a fitted system.

For a worked comparison between the batched chart and the generated search
computation, see [Parsing as an effectful
computation](../tutorials/qvr/11-parsing-and-search.md). The
[generated-computation reference](../reference/qvr/generated-computations.md)
lists the families, effects, [handlers](../reference/qvr/effects-and-handlers.md#handler-declarations), and entries a deduction or
schema-backed parser adds.

## Further reading

- [@klein2001parsing]
- [@mcallester2002complexity]
- [@eisner2007transformations]
- [@pereira1983parsing]
- [@goodman1999semiring]
- [@knuth1977generalization]
- [@huang2005kbest]
- [@nederhof2003weighted]
- [@shieber1995principles]
