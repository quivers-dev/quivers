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
`intersect` multiplies two weighted languages on a common one-sided tape
through the same epsilon-filtered product. Each pair of accepting paths with
the same word contributes exactly once to this product.

::: quivers.core.relations
    options:
      members:
        - RationalRelation
        - RelationTopology
        - Arc
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
        - condition
        - intersect
        - invert
        - project
        - trim
