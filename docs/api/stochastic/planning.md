# Plans

[`plan`][quivers.stochastic.planning.plan] takes a query, a
[`RulePresentation`][quivers.qiec.rules.RulePresentation], a weight algebra, a recursion contract,
an accuracy contract and a retention, and either returns a
[`Plan`][quivers.stochastic.planning.Plan] or raises
[`PlanRefused`][quivers.stochastic.planning.PlanRefused] with the reason. The plan's
[`PlanRecord`][quivers.stochastic.planning.PlanRecord] states what is computed (the query, with its
observable and the inputs it differentiates by, and the accuracy contract, with its numerics,
derivative semantics and failure policy), the law claims of the algebra it relies on, the analysis
of the presentation, the selected algorithm, its assumptions, the alternatives considered, a static
proof of acyclicity when one is found, and the cost claim of the selected backward pass.

```python
from quivers.stochastic import plan

current = plan(query, presentation, LOG_PROB, Acyclic(), contract, NoRetention())
evaluation = current.run(occurrences, weights, like=torch.zeros((), dtype=torch.float64))
evaluation.value(goal)
current.gradient(occurrences, weights, goal, like=...)
```

## Two-pass evaluation

The executor is the two-pass evaluation of Eisner (2023, sections 5 and 6) on an
[`IndexedChart`][quivers.stochastic.indexed.IndexedChart]. Write $\mathcal I(x)$ for the
grounded rule instances concluding the item $x$, each a rule $r$ with weight $w_r$ and premise
items $p_1, \dots, p_n$ in premise order, and $a(x)$ for the sum of the input occurrences of $x$.
The value of $x$ is

$$
v(x) = a(x) \oplus \bigoplus_{e \in \mathcal I(x)} v(p_1) \otimes \cdots \otimes v(p_n) \otimes w_r.
$$

The first pass discovers every item and counts $|\mathcal I(x)|$ for each, storing no instance.
The second pass keeps, for every item, its count of instances that have not yet contributed; an
item whose count reaches zero is final, and its new instances are found and added to their
conclusions. A rule without premises is an instance with $n = 0$ whose conclusion validation makes
ground; it is counted and added before the first item is final.

**Lemma (each instance once).** Admit items one at a time, and let each admitted item $x$ trigger
every premise position whose form it has, enumerating at position $t$ the instances in which $x$
fills $t$, every premise is admitted, and no premise before $t$ is $x$. Then each instance is
enumerated exactly once.

*Proof.* Let an instance have premise items $p_1, \dots, p_n$ and let $x$ be the last of them to be
admitted. Before $x$ is admitted the instance has a premise that is not, so it is not enumerated.
When $x$ is admitted, let $t$ be the least position with $p_t = x$. At $t$ every premise is
admitted and none before $t$ is $x$, so the instance is enumerated there; at a later position $u$
with $p_u = x$ the premise $p_t$ before $u$ is $x$, so it is not. Items admitted after $x$ do not
fill this instance. The keyed lookups return each item of a form once, and matching a premise
pattern against an item is deterministic, so one choice of premise items gives one enumeration.
QED

How the other premises are found, by Generic Join from each trigger, and the bound that gives on
the lookups and candidates $J$ of each rule, are on [Indexed charts](indexed.md); the record
states both for every rule.

One item filling several premises, as in $c \leftarrow b, b, b$, thus contributes
$v(b) \otimes v(b) \otimes v(b) \otimes w$ once.

**Theorem.** If the derived hypergraph is acyclic, the second pass makes every discovered item
final, and $v(x)$ is the sum, over the derivations of $x$, of the product of their weights in
premise order. If it has a cycle, the items on or above it are never final, and the run is refused
with [`PlanRefusal.CYCLE`][quivers.stochastic.planning.PlanRefusal].

*Proof.* By the lemma, an item's count reaches zero exactly when every instance concluding it has
contributed, and an instance contributes once all its premises are final. If the hypergraph is
acyclic, induction on the longest path from the inputs shows that every item becomes final (this
is Kahn's (1962) argument for topological order). Conversely an item on a cycle waits on an
instance whose premise waits on it, so neither is ever final, and nor is any item above them. When
an item is final, each term of its sum is a product of final values, so by induction $v(x)$ is the
displayed sum with every $v(p_i)$ the sum over the derivations of $p_i$. Expanding the products of
sums by distributivity, which keeps the order of factors, gives one product per derivation in
premise order. Only associativity and commutativity of $\oplus$ and the two distributive laws are
used, so the product need not commute. QED

The products are taken in premise order with the rule weight last, as a derivation's factors are
read (children first, then the rule), so the algebra may be a
[`MatrixSemiring`][quivers.core.algebras.MatrixSemiring] whose product does not commute. Space is
the values, the counts and the chart's indexes, $O(V)$ for a fixed presentation; parent counts
are the only provenance kept, which is the [`ParentCounts`][quivers.qiec.contracts.ParentCounts]
retention.

## Requested evidence

With [`RequestedEvidence`][quivers.qiec.contracts.RequestedEvidence] as the retention, a run
keeps, for every derived item of a requested form, the edges that conclude it: its input
occurrences, each on its own, and its grounded rule instances with their premise items. They are
in [`Evaluation.evidence`][quivers.stochastic.planning.Evaluation], and nothing else of the
forest is kept.

**Theorem.** For every item $x$ of a requested form, the kept evidence of $x$ is the list of edges
of $x$ in the [packed forest](forest.md) of the same inputs, in the same order; no other item has
evidence.

*Proof.* The first pass and the forest's construction admit the inputs, then the rules without
premises, then items in the same queue order, and enumerate the same instances from each admitted
item. By the lemma above each instance is enumerated once, and the forest stores it as one edge
of its conclusion; the first pass keeps it exactly when its conclusion's form is requested. Inputs
are kept one edge per occurrence in both, and only items of requested forms are kept at all. So
the two lists of $x$ are equal, element by element. QED

The evidence of $x$ is one level of its derivations: each alternative's rule instance or input
occurrence, with the premise items it uses. A derivation of $x$ is recovered from it together with
the evidence of the premises, so requesting the forms of every item below $x$ recovers all of them,
and requesting fewer keeps less. The memory is $R$, the edges and premise references of the
requested items, beside the parent counts
([`with_requested_evidence`][quivers.stochastic.planning.with_requested_evidence]).

## Gradients

With autograd as the backward pass, the run is recorded on the autograd tape, which holds every
product and sum: $P E$ elements beside the $V$ values. A contract with a memory budget selects the
recomputing backward pass instead, which keeps only the values and the order in which they became
final, and visits items in reverse.

**Theorem.** Visiting items in the reverse of the order they became final, rebuilding each item's
sum from the final values of its premises and propagating its cotangent through that sum alone,
gives the gradient of the goal's value in every input weight.

*Proof.* $v$ is a composition of the item sums along the order of the second pass, so the chain
rule gives the gradient as a sum over the uses of each value. Every use of $v(y)$ is in the sum of
an item that became final after $y$, so when $y$ is visited its cotangent has received every
term. The partial derivatives of an item's sum in its premises' values and in its weights are those
of the rebuilt sum, since it is the same function of the same arguments. QED

The pass enumerates each item's incoming instances again with the chart's backward join plan, so
its time is that of the forward pass, and its space is $O(V)$ beside the $P D$ elements of the
largest sum of one item. Neither pass drops a term because its value is zero at the current
weights: support is decided by the instances, not by numbers, so a derivative at an exact zero is
kept.

## Acyclicity by a measure {#proof-measure}

A presentation whose form graph has a cycle, like the span rules
$\mathrm{span}(A, i, j) \leftarrow \mathrm{span}(B, i, k), \mathrm{span}(C, k, j)$, may still
derive an acyclic hypergraph on every input it is meant for.
[`acyclicity_certificate`][quivers.stochastic.planning.acyclicity_certificate] proves this with a
[ranking function](https://en.wikipedia.org/wiki/Ranking_function): for each form $f$ of a cyclic
component $C$, a measure $\mu_f(x) = \sum_q a_{f,q} x_q$ over its natural-number arguments.

**Theorem.** Suppose that for every rule concluding a form of $C$ with premises of $C$ at
$\pi_1, \dots, \pi_m$ ($m \ge 1$), every assignment of natural numbers to its variables with
$\mu(\pi_l) \ge 1$ for all $l$ satisfies $\mu(\text{conclusion}) \ge 1$ and
$\mu(\text{conclusion}) \ge \mu(\pi_l) + 1$ for all $l$. Then in a run where every item of the
forms of $C$ has measure at least one, no item of $C$ lies on a cycle.

*Proof.* A cycle of the derived hypergraph runs through items whose forms lie on a cycle of the
form graph, so all in one component $C$. Each step of the cycle is an instance whose conclusion
and one of whose premises are consecutive items of the cycle, both in $C$, and every premise of
$C$ has measure at least one, so the conclusion's measure exceeds that premise's. Around the
cycle the measure strictly increases and returns to its start, a contradiction. QED

The hypothesis is checked by [Fourier-Motzkin
elimination](https://en.wikipedia.org/wiki/Fourier%E2%80%93Motzkin_elimination): each obligation
$g \ge 1$ under the facts $\mu(\pi_l) - 1 \ge 0$ and $x \ge 0$ for every variable is proved by
showing that adding $g \le 0$ leaves no real solution. A system with no real solution has no
natural one, so the check is sound; it ignores side conditions, which only remove instances. The
search tries coefficients in $\{-1, 0, 1\}$ on natural-number arguments, at most 4096
combinations per component, and keeps every measure it proves, as the alternatives of a
[`MeasureComponent`][quivers.stochastic.planning.MeasureComponent]. A component with none is not
thereby cyclic; its run still refuses a cycle by the counts. Unary rules among the forms of a
component, $\mathrm{span}(A, i, j) \leftarrow \mathrm{span}(B, i, j)$, make the conclusion's
measure equal a premise's, so no measure exists, which is right: such rules may form a cycle.

**Corollary (spans).** For the binary span rules the measure $j - i$ is proved: from
$k - i \ge 1$ and $j - k \ge 1$ follow $j - i \ge 2$, $j - i \ge (k - i) + 1$ and
$j - i \ge (j - k) + 1$. Every item has measure at least one when every input spans a nonempty
interval, $i < j$, since the conclusions of binary rules then do too; each
[`Evaluation`][quivers.stochastic.planning.Evaluation] reports in `certified` whether some proved
measure holds on all of its items. An input spanning an empty interval breaks the condition of
$j - i$, and a grammar with empty spans needs its own argument.

## Span plans

A presentation is in *span form* when it has one judgment family whose indices are a category, of
closed sorts of nullary constructors, and two natural numbers $i \le j$, and every rule is binary,
$A(i, j) \leftarrow B(i, k), C(k, j)$, unary, $A(i, j) \leftarrow B(i, j)$, or without premises;
a category argument may be a variable of its sort, which stands for each constructor.
[`span_grammar`][quivers.stochastic.span.span_grammar] finds the form or says why there is none.
When the query keeps no provenance and asks for a value, a support or a marginal, `plan` selects
the span executor, [`SPAN_ALGORITHM`][quivers.stochastic.planning.SPAN_ALGORITHM], and records
the two-pass executor as the alternative; with parent counts or a forest it selects the two-pass
or forest executor as before.

The schedule comes from the presentation: a binary rule's premises are strictly shorter than its
conclusion, so the cells of length $\ell$ are final once those of the shorter lengths are, and
[`span_inside`][quivers.stochastic.span.span_inside] computes all cells of one length at once.
For a cell $(i, i + \ell)$ and a premise pair $(B, C)$ it first sums over the split points,

$$
M_{BC}(i) = \bigoplus_{k} x(B, i, i + k) \otimes x(C, i + k, i + \ell),
$$

then each rule $r = (A \leftarrow B\,C)$ contributes $M_{BC}(i) \otimes w_r$ to $A$, and the
cell is closed under the unary rules, $x \otimes U^*$. Summing over $k$ before the rule is right
distributivity, $\bigoplus_k (a_k \otimes b_k) \otimes w = (\bigoplus_k a_k \otimes b_k) \otimes
w$, so the premises keep their order and the weight comes last, and the plan needs only an
[`OrderPreservingSemiring`][quivers.core.algebras.OrderPreservingSemiring]. No tensor over
categories, rules and split points together is formed: the work is $O(P n^3 + R n^2)$ for $P$
distinct premise pairs and $R$ rule instances, against $O(R n^3)$ for one instance at a time.
A side condition that reads the split point makes its rule a pair of its own; one that reads only
the span masks the rule's contribution. The unary closure is the algebra's star when the unary
rules have a cycle, which needs the linear cyclic contract, and otherwise each row
$U^*[b] = e_b \oplus U[b]\,U^*$ in an order that puts a category after those it derives, which
needs no star. Zeros are the algebra's own zeros, so a cell no derivation reaches is
$-\infty$ in the log semiring, and its derivative is zero. The cells reported are those some
derivation reaches when every input and rule is present, so a cell whose derivations all weigh zero
is reported, with the zero, as the two-pass executor reports it. The cost claim is
[`SPAN_WORK`][quivers.stochastic.planning.SPAN_WORK]; its optimality is unresolved for the reasons
of complexity claim Q3, and the run measures the counters of
[`SPAN_MEASURES`][quivers.stochastic.planning.SPAN_MEASURES].

Inputs over empty spans $(i, i)$ make categories *nullable*. Over each empty span the weights
solve $e = s \oplus e\,U \oplus \bigoplus_r e(B) \otimes e(C) \otimes w_r$, a polynomial system in the
categories; its strongly connected components are solved in order, a component without a cycle
by its constant and a linear one as $b M^*$ by the algebra's star, which needs the linear cyclic
contract. A nonlinear cycle of nullable categories is a monotone polynomial system and is refused
here. A binary rule with an empty premise then joins the unary rules of its cell: $A(i, j)
\leftarrow B(i, j), C(j, j)$ is the step from $B$ to $A$ with weight $e_j(C) \otimes w$, and $A(i, j)
\leftarrow B(i, i), C(i, j)$ the step from $C$ with weight $e_i(B) \otimes w$, which moves $e_i(B)$
past the cell's value and so needs a commutative product. A cell with such steps is closed on its
own, and a cycle among its steps needs the linear cyclic contract and a star.

The support of a value is a separate observer plan,
[`SPAN_RECOGNITION_ALGORITHM`][quivers.stochastic.planning.SPAN_RECOGNITION_ALGORITHM], selected
for a support query when the algebra establishes zero-sum-freeness and the absence of zero
divisors, so that the support of a sum is the union of the supports and that of a product their
intersection. [`span_recognize`][quivers.stochastic.span.span_recognize] computes $M_{BC}(i)$ for
all pairs at once as the batched product of two 0-1 matrices over the split points, read as
$> 0$. Valiant (1975) recognizes in $O(\mathrm{BM}(n))$ for a fixed grammar, below this cubic
schedule, so its claim [`SPAN_RECOGNITION_WORK`][quivers.stochastic.planning.SPAN_RECOGNITION_WORK]
is not called optimal.

## Monotone polynomial plans

Under [`MonotonePolynomial`][quivers.qiec.contracts.MonotonePolynomial] the plan selects
[`NEWTON_ALGORITHM`][quivers.stochastic.planning.NEWTON_ALGORITHM]: the components of the derived
hypergraph are solved by decomposed Newton with certified intervals, as
[Cyclic systems](recursion.md#monotone-polynomial-systems) describes. The contract needs the
algebra's claim of a [`RealEmbedding`][quivers.core.algebras.RealEmbedding] and of a commutative
product, and its guarantee is at best certified, since no finite method computes the solution
exactly. Each run carries an [`AccuracyReport`][quivers.qiec.contracts.AccuracyReport]: the
certified bound of its widest interval in the contract's metric with that interval as the
certificate, or no guarantee with the conditioning where an upper end did not certify, why it
stopped (completed, or the iteration budget spent), and the implicit derivative when the spectral
radius at the solution is certified below one. [`Plan.settled`][quivers.stochastic.planning.Plan.settled]
and [`Plan.settled_gradient`][quivers.stochastic.planning.Plan.settled_gradient] settle the report
by the contract: they return the result when it meets the contract, refuse otherwise, or return it
as [`Unestablished`][quivers.qiec.contracts.Unestablished] when the contract says so.
[`Plan.run`][quivers.stochastic.planning.Plan.run] and
[`Plan.gradient`][quivers.stochastic.planning.Plan.gradient] return only established results, so
under a contract that returns unestablished values they refuse and name the settled methods. The cost claim
is [`NEWTON_WORK`][quivers.stochastic.planning.NEWTON_WORK].

## Refusals

| [`PlanRefusal`][quivers.stochastic.planning.PlanRefusal] | when |
|---|---|
| `inadmissible` | the contracts are inadmissible for the algebra's law claims, or the query names another presentation or retention |
| `recursion-without-executor` | the recursion contract is not acyclic |
| `open-premises` | a premise has hypotheses, which a chart of closed items does not derive |
| `observable-without-executor` | the observable does not survive aggregation, such as a fold that does not commute with sums: for alternatives $-1$ and $1$ the square of the sum is $0$ and the sum of the squares $2$ |
| `retention-without-executor` | the retention asks for more than parent counts |
| `algebra` | a best derivation is asked of an algebra in which no term attains an item's value |
| `divergent` | a closure diverges, or a monotone polynomial component's least fixed point is certified infinite |
| `cycle` | the inputs make the derived hypergraph cyclic, or the unary rules of a span plan have a cycle under the acyclic contract |

## References

- Eisner, J. (2023). Time-and-space-efficient weighted deduction. *Transactions of the Association
  for Computational Linguistics* 11, 960-973.
  [doi:10.1162/tacl_a_00588](https://doi.org/10.1162/tacl_a_00588)
- Valiant, L. G. (1975). General context-free recognition in less than cubic time. *Journal of
  Computer and System Sciences* 10(2), 308-315.
  [doi:10.1016/S0022-0000(75)80046-8](https://doi.org/10.1016/S0022-0000(75)80046-8)
- Kahn, A. B. (1962). Topological sorting of large networks. *Communications of the ACM* 5(11),
  558-562. [doi:10.1145/368996.369025](https://doi.org/10.1145/368996.369025)

::: quivers.stochastic.planning
