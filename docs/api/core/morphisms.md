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
that has consumed every word. On this graph, a topological pass adds each incoming product
to the weight of reaching its target. The final sum multiplies each completed state's weight
by its final weight. A cycle in this graph would have to leave every position unchanged, so
the query raises `ProductiveEpsilonCycleError` when the surviving graph contains a cycle.

An index assigns each arc to its first non-epsilon label or to the all-epsilon group for its
source state. With $E$ arcs on $k$ tapes, validating and sorting the indexed arc labels
takes at most $O(k E \log(E+1))$ operations, in addition to reading the boundary records and
weight-slot names. For $S$ reached configurations and $A$ candidate arcs inspected, building a
query takes $O(k(S + A))$ operations after reading the words and the initial states, with
$O(kS + A)$ space. The pass over the trimmed graph takes a number of semiring operations
linear in its arc count and its initial and final state counts. Computing the weight vector
has a separate cost, and validation tests membership once per weight slot. A bounded tensor
restriction reads and validates the vector once for the whole table. It repeats construction
for each entry, subject to the explicit tensor size limit and optional state and arc budgets.

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
        - QueryWork
        - FiniteRelationRestriction
        - ProductiveEpsilonCycleError
        - RelationLimitError
