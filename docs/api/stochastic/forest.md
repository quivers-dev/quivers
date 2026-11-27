# Packed forests

A plan whose retention is [`PackedForest`][quivers.qiec.contracts.PackedForest] keeps the derived
hypergraph of a run. [`build_forest`][quivers.stochastic.forest.build_forest] discovers the items
from the input occurrences on an [`IndexedChart`][quivers.stochastic.indexed.IndexedChart] and
stores, for each item, one [`ForestEdge`][quivers.stochastic.forest.ForestEdge] per grounded rule
instance concluding it and one per input occurrence of it. The result is a
[`DerivationForest`][quivers.stochastic.forest.DerivationForest].

**Lemma (unfoldings are derivations).** The tree unfoldings of an item in the forest correspond
one to one to its derivations, occurrence identities included.

*Proof.* A derivation of $x$ is an input occurrence of $x$, or a rule instance concluding $x$
together with a derivation of each premise. The forest has exactly one edge per occurrence of
$x$, since occurrences are added as their own edges and never merged even when they share a slot
and a binding, and exactly one edge per instance concluding $x$, by the lemma that each instance
is produced once ([Plans](planning.md)). Unfolding chooses an edge at each node and recurses into
its premises, so the two definitions agree by induction on the derivation. QED

Without the occurrence identities two inputs of one word at one position would be one leaf, and
every count and statistic above them would be halved; the checks compare the unfoldings with the
derivation oracle on exactly such an input.

## Linear folds

[`forest_fold`][quivers.stochastic.forest.forest_fold] computes
$\bigoplus_d \bigotimes_{e \in d} \mathrm{value}(e)$, each derivation's product read as its factors
are, the premises' values in order and then the edge's own. By distributivity, as for two-pass
evaluation, the sum over the derivations of an item is the sum over its edges of the edge's
product with its premises' sums, so dynamic programming over the forest in order computes it in
time linear in the forest's size. A carrier such as the
[`ExpectationSemiring`][quivers.core.carriers.ExpectationSemiring] makes the fold a statistic.
[`forest_folds`][quivers.stochastic.forest.forest_folds] computes the folds of several goals
in one pass, sharing the values of any premises used by more than one goal.

## Nonlinear folds

[`nonlinear_fold`][quivers.stochastic.forest.nonlinear_fold] computes $\bigoplus_d h(d, w(d))$ for
any $h$. No aggregate summarizes it: for two derivations of weights $-1$ and $1$ and $h$ the square,
it is $2$, while the square of the aggregate is $0$. So every derivation is built, by
[`fold_derivations`][quivers.stochastic.forest.fold_derivations], within a limit the caller
declares.

The goal's derivations are first counted exactly, in integers, by the same dynamic programming;
if there are more than the limit, the forest is refused with
[`ForestRefusal.TOO_MANY`][quivers.stochastic.forest.ForestRefusal] and no derivation is
returned, since a part of them is not the fold. Otherwise derivations are built bottom up, a
derivation of a premise once for all the derivations above it.

**Lemma (the limit bounds every list).** Call a pair of an item and a height bound useful when it
is the goal's, or a premise of an edge of a useful pair whose premises all have derivations within
the bound. Every useful pair has at most as many derivations as the goal.

*Proof.* Let $(y, h)$ be a premise of a live edge of a useful pair $(x, h + 1)$. Fix a derivation
within the bound for each other premise of the edge, which exists since the edge is live, and, by
induction, an extension of the derivations of $x$ to derivations of the goal. Substituting a
derivation of $y$ into that context gives a derivation of the goal, and distinct derivations of
$y$ give distinct ones. QED

Only useful pairs are listed, so each list holds at most the limit and the work is at most the
limit times the number of useful pairs, which is the cost in
[`FOREST_WORK`][quivers.stochastic.planning.FOREST_WORK].

## Heights and cycles

A forest with a cycle has items with infinitely many derivations. It is counted, folded and
unfolded only within a bound on height, under the recursion contract
[`Bounded`][quivers.qiec.contracts.Bounded] with measure `"height"`: an input or a rule without
premises has height one, and a rule instance one more than its highest premise. Asked without a
bound, the forest is refused with `ForestRefusal.UNBOUNDED`. A plan under bounded recursion
refuses the two-pass run, whose meaning counts every derivation, and folds on the forest within
[`Plan.height`][quivers.stochastic.planning.Plan.height].

## Expected proof embeddings

[`expected_embedding`][quivers.stochastic.forest.expected_embedding] computes
$\sum_d p(d)\, \phi(d) / \sum_d p(d)$ for a fold $\phi$ of derivations into vectors given as a
[`HostInterpretation`][quivers.qiec.rules.HostInterpretation], with $p(d)$ the derivation's
weight as a nonnegative real. A neural fold is not linear, so the derivations are enumerated
within the limit; each shared sub-derivation is folded once.
[`encoder_interpretation`][quivers.stochastic.forest.encoder_interpretation] makes an
[`Encoder`][quivers.structural.encoder.Encoder] such a fold over the derivations of its
signature's [presentation][quivers.qiec.rules.presentation_of], one
[`Encoder.step`][quivers.structural.encoder.Encoder.step] per node, which equals the encoder
applied to each derivation's term.

::: quivers.stochastic.forest
