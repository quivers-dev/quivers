# Morphisms

A finite morphism stores one weight for each pair of an input and an output, as a tensor.
[Composing](../../guides/morphisms.md#composition-the-operator) a table from users to features with a
table from features to items sums out the feature axis under the chosen `Algebra` and gives
weights from users to items. An `ObservedMorphism` holds given values, and a `LatentMorphism` holds
learnable parameters. The module defines the base morphism classes, their composition, and shared
morphism utilities.

::: quivers.core.morphisms

## Sparse string relations

A `RationalRelation` assigns a weight to each tuple of words on its ordered tapes. Its topology
identifies arcs by position in the arc table, so parallel arcs contribute separately, even when
they read the same symbols and use the same weight slot. Initial and final states have explicit
weights. Every slot must belong to the declared semiring carrier. The relation checks the
current weight vector each time it reads the vector for evaluation.
The [QIEC codec](../qiec/serialization.md#sparse-relations) records the topology and
weight identities for later runtime binding.

For a fixed tuple of $k$ words, a query state consists of a machine state and $k$ positions in
the observed words. An arc labeled with a symbol on a tape advances that tape only if the symbol
matches. An epsilon label leaves the position unchanged. Every accepted original walk has a
unique lift to this graph because its arc sequence determines all positions. Every accepting
walk in the graph likewise projects to an original walk that realizes the observed tuple.

The plan keeps only states on a walk from a weighted initial state to a weighted final state
that has consumed every word. The level of a state is the length of the longest path to it
in this graph. All predecessors of a state lie in earlier levels. In each level, the weight
of reaching a state is the sum of its initial weight, if present, and one product per
incoming arc. Each product multiplies the weight of reaching the source by the arc weight.
By induction on the levels and distributivity, these terms account for every accepting walk
when the final sum multiplies each completed state's weight by its final weight. A cycle in
this graph would have to leave every position unchanged. By default, a cycle in the
surviving graph raises `ProductiveEpsilonCycleError`.

An index assigns each arc to its first non-epsilon label or to the all-epsilon group for its
source state. With $E$ arcs on $k$ tapes, validating and sorting the indexed arc labels
takes at most $O(k E \log(E+1))$ operations, in addition to reading the boundary records and
weight-slot names. For $S$ reached configurations and $A$ candidate arcs inspected, building a
query takes $O(k(S + A))$ operations after reading the words and the initial states, with
$O(kS + A)$ space. Within a level, incoming rows are grouped by their length rounded up to
a power of two. Each padded row has fewer than twice as many entries as its unpadded row.
The evaluator forms one vector of outgoing products per level and splits it into chunks by
destination level. Thus the forward pass uses time and space linear in the active state and
arc counts and in the number of initial and final marks. It uses one semiring join for each
width greater than one present in a level. These bounds count scalar semiring operations and
entries.
Computing the weight vector has a separate cost, and validation tests membership once per
weight slot. A bounded tensor restriction reads and validates the vector once for the whole
table. It repeats construction for each entry, subject to the explicit tensor size limit and
optional state and arc budgets.

### Productive cyclic components

`RationalRelation.query(words, max_closure_size=n)` admits productive cyclic components.
Every accepting walk visits a sequence of strongly connected components in topological
order. For each visit, its first state, its last state and the walk between them determine
its contribution within that component. A local star sums those internal walks for each
entry and exit pair, including the empty walk at each state. The packed pass multiplies
these weights with the weights of edges between components and of the boundary marks.

Each component uses a matrix with one direct arc per ordered state pair. Each additional
parallel occurrence passes through its own auxiliary vertex, with the original weight
on the first edge and the unit on the second. This preserves the individual weights
through the solver's convergence check. `max_closure_size` bounds the order of every
matrix, including its auxiliary vertices, and `QueryWork.closure_sizes` reports the
orders the plan will use. For component matrix orders $m_i$, splitting and packing
adds at most $O(\sum_i m_i^2)$ work and storage to the observation graph. The shipped
closures take $O(\sum_i m_i^3)$ algebra operations, plus any exact or enclosed
numerical decisions made by `Star`.

Scoring uses floating-point tensors and requires established semiring laws and the
least-solution law of `Star`. Divergence and numerical uncertainty in a local matrix
raise `DivergentClosureError` and `UndecidedClosureError`, respectively. An unusable
closure weight also raises `UndecidedClosureError`. The structural plan retains
zero-weight occurrences and rechecks the current weights each time it scores a query.

### Binary composition at fixed words

`ComposedRelation(left, right)` joins the output tape of the left relation to the input
tape of the right relation. Each operand has one input and one output tape, and the joined
tape types and scalar semiring must agree. For outer words $x$ and $z$, the score is
$\bigoplus_y R(x,y)\otimes S(y,z)$. The shared semiring must establish commutative
multiplication so that interleaved products preserve the product of the two path weights.

The query aligns the corresponding non-epsilon occurrences on the joined tape. Between
two such matches, including the leading and trailing intervals, the filter pairs epsilon
occurrences until one side has finished its run. The remaining occurrences advance their
own side. A match of a non-epsilon symbol resets the filter. This gives exactly one
filtered path per compatible pair of original paths, including parallel paths and paths
with no arc. Current weights leave this correspondence unchanged.

The constructor indexes arcs by state, outer label, and shared label in time and space
linear in the total number of operand arcs. A query explores only configurations reachable
at the supplied words. For $P$ reached configurations, $H$ shared-label probes and $A$
candidate arc occurrences, planning after word validation takes $O(P+H+A)$ time and
$O(P+A)$ space. At each configuration a probe examines one label from the smaller of two
allowed shared-label groups. The arc count includes every admitted occurrence of a matched
arc pair or one-sided epsilon arc. `max_states` bounds $P$ and `max_work` bounds $H+A$.
Evaluation reads both original weight sources and uses the packed acyclic plan for the
productive part. The weight vectors must have a common dtype and device. Evaluation work
is linear in the operand slot counts and the productive graph, apart from computing the
operand weights. A productive cycle on the fixed outer words requires closure, even
when it emits symbols on the shared tape or its current weight is zero.

### Reusable topologies and tape operations

`ComposedRelation.materialize(max_states=..., max_work=...)` builds a sparse product
topology with the same epsilon filter. The result is a `RationalRelation` that can be
used in a cascade, queried with bounded cyclic closure, or projected onto selected
tapes. The limits cover reachable product and intermediate states, shared-label
probes and constructed arc occurrences.

Each matched pair becomes two consecutive arcs. The first writes the two outer
labels and takes the left weight. The other advances with epsilon labels and takes
the right weight. The intermediate state has that one continuation, so expansion
preserves each path and its weight. With cyclic composition the local star receives
the two primitive factors as separate matrix entries, preserving the information
needed to check convergence. Both operands' original parameter sources are live.

`invert` exchanges input and output tapes. `project` retains the supplied original
tape positions in increasing order, summing over the words on erased tapes.
`trim` keeps the arcs and boundary marks on structurally accepting walks, including
zero-weight occurrences. These operations reuse the relation's weight source.

`condition` fixes one tape to a word and erases that tape. After indexing the
source arcs by their selected label, it visits the reachable pairs of source
state and consumed word length. The remaining relation assigns the joint
weight at the fixed word, so it can be used to compute a normalizer.
Building the label index takes time and space linear in the source arc count.
`max_states` bounds the visited pairs and `max_work` bounds the matching arc
occurrences. Each output arc copies the labels on the retained tapes.

`join` synchronizes a single pair of tapes of the same type, one from
each operand. The remaining input tapes precede the remaining output
tapes. Each path pair with the same word on the two selected tapes
contributes once by the epsilon filter. The joined tape can be kept
once. For one-tape operands with the same tape orientation this gives
`intersect`, the product of their weighted languages.
A joined transition writes at most one label per retained tape. The
construction limits apply to reached states, shared-label probes and
constructed arcs after indexing the source arcs.

`total_weight` sums all accepting walks by erasing their labels before
running the same fixed-word query. Each path realizes one word tuple,
so this is the sum over the weighted relation's entire support. If
erasing the labels exposes a productive cycle, the caller supplies a
closure limit and the algebra checks convergence in its domain.

`union` adds relations with identical tape types by taking the disjoint
union of their topologies. `concatenate` uses a connecting state, so
each accepting path contains one path from each operand. Its two
connecting arcs carry the original boundary weights. It adds all ways
to split a tuple of words into such a pair.

`closure` uses a new initial and final state to mark the end of each
repetition. Every visit to this state after leaving it ends one
accepting path of the operand. The empty path contributes one. If the
operand accepts the empty word tuple, a query may have to sum
infinitely many paths. The local star checks decide whether the sum
can be used. Construction of these three topologies uses linear space
in the original records and introduces one arc per connecting boundary.


A fixed-word query with an acyclic productive graph supports `best_path`
and `k_best_paths` for Markov and LogProb weights. Ranking compares the exact
values of the stored binary weights and reports path occurrence IDs in the
original topology. Several paths can realize the same strings. To sample
from these paths, `sample_path` chooses an initial state by its weighted
suffix mass, then chooses an arc or a stop with probability proportional to
the weight of an accepting continuation. The observations enter through the
query, and a positive normalizer is required. Each sampled walk terminates.

For expected counts, `expected_arc_counts` uses one packed pass in each
direction and gathers the corresponding prefix and suffix at every active arc
occurrence. An occurrence contributes its prefix times its arc weight times
its suffix, divided by the normalizer. Summing these contributions gives the
expected number of traversals of each original arc. This takes linear
arithmetic work and storage in the active graph, in addition to the source
weight evaluation and construction of the graph. Queries reuse the graph
structure across evaluations.

::: quivers.core.relations
    options:
      members:
        - RationalRelation
        - RelationTopology
        - Arc
        - ArcCounts
        - Boundary
        - EPSILON
        - Label
        - Words
        - FixedStringQuery
        - ComposedRelation
        - ComposedStringQuery
        - QueryWork
        - FiniteRelationRestriction
        - ProductiveEpsilonCycleError
        - RelationLimitError
        - RelationPath
        - closure
        - concatenate
        - condition
        - intersect
        - invert
        - join
        - project
        - trim
        - union
