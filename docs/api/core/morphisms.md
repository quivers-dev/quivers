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

Each composed relation retains its latest materialized topology and
weight-slot mapping. Repeating the same two construction limits reuses
that structure. Every result reads the original live operand weights.

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

`auto_intersect` keeps paths whose words agree on two tapes of the same
type. After trimming, it checks whether any accepting path can encounter
both a positive and a negative cycle in the signed tape-length difference.
A state-potential check recognizes zero-drift strongly connected
components in linear time. For other components, signed relaxations
detect positive and negative cycles. Their signs propagate along the
component graph. Opposite signs on separate accepting branches are
allowed.

Consider a prefix of an accepting path whose tape lengths agree at
the end. Remove cycles independently from that prefix and its suffix.
Their simple remainders each have length at most the number of
productive states minus one. Since all removed cycles have one
common sign or zero delay, each remainder bounds the original
delay from one side. The prefix and suffix delays sum to zero,
so the prefix's absolute delay is bounded by that same state count
minus one.

The checked graph records the unmatched suffix on the tape ahead.
Each transition appends the original labels, cancels their common
prefix, and rejects a mismatch or delay beyond the certified bound.
An accepting path has no unmatched suffix. Projection to the
original arcs is bijective on paths with identical selected words,
preserving shared weights and derivatives. The work limit includes
signed relaxations and examined product arcs. The state-bound
certificate is independent of the current weights.

`RationalRelation.conditional` fixes all input tapes and compiles a
normalizer over every output word tuple. `ConditionalStringQuery.log_prob`
compares the score of a supplied output to that normalizer. The
denominator includes every alignment and output permitted by the
relation. A productive insertion cycle uses the usual closure check
and size bound. Each call uses the same live weights in both sums.

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
original topology. Several paths can realize the same strings. Each
outgoing arc gives one ordered stream of suffixes, since multiplying
by a positive weight preserves their order. The query merges the
stream heads. With $d_q$ streams at state $q$ and $K_q$ retained suffixes,
the work is $O(d_q + K_q \log(d_q+1))$ arithmetic operations per
state, and an analogous merge chooses the initial marks. Reading
weight slots, the bit lengths of the exact rational scores, and
materializing the selected arc sequences contribute additional
costs.

To sample from these paths, `sample_path` chooses an initial state by its weighted
suffix mass, then chooses an arc or a stop with probability proportional to
the weight of an accepting continuation, evaluated in float64 log space. The
observations enter through the query, and a positive normalizer is required.
In an acyclic observation every path has finitely many steps.

For expected counts, `expected_arc_counts` uses one packed pass in each
direction and gathers the corresponding prefix and suffix at every active arc
occurrence. An occurrence contributes its prefix times its arc weight times
its suffix, divided by the normalizer. Summing these contributions gives the
expected number of traversals of each original arc. In acyclic graphs
this takes linear arithmetic work and storage in the active graph, in
addition to the source
weight evaluation and construction of the graph. Queries reuse the graph
structure across evaluations.
For a differentiable weight, `weight(recompute_backward=True)` lets PyTorch
discard the intermediate tensors of the path pass and compute them again
during backward. The source weight evaluation and its saved tensors are
outside the path checkpoint.


## Exact string factor trees

A `StringVariable` names a string type and may specify an observed word.
A `StringFactor` assigns a distinct variable to each tape of a weighted
relation. A `StringFactorGraph` checks these assignments and supports
exact tree inference with the Markov and LogProb algebras.

To compile a marginal, the tree schedule sends one message toward the
root of each connected component on every edge. A variable multiplies
the incoming unary messages and its observation indicator, if present.
A factor synchronizes each incoming unary message with its assigned
tape and sums over the shared word. If that variable has no incoming
weight, projection sums directly over its tape. Subtrees below a factor
share only the variables on these separating tapes. Induction from the
leaves gives the product of the factors summed over all variables other
than the root. An independent component contributes its own total
weight to the partition.

Messages retain their unnormalized weight. `TreeStringMarginal.partition`
returns the total weight across components. `log_prob` divides the
weight at a word by the mass of its component and requires every
component to have positive finite mass. It evaluates each original
weight source once across the numerator and denominator within a call.
A cyclic factor graph is refused by the tree schedule. Cycles within a
single relation use the ordinary productive closure checks.


## Certified finite string MAP

A `FiniteStringScore` lists the feasible tuples of words for a factor and
assigns each tuple a rational additive score. `dual_string_map` uses the
same typed variables and observations as a string factor graph. Bindings
are full words in the declared finite supports.

Each variable occurrence gets a word-indexed potential. The potentials
sum to zero across occurrences for each variable and word. Thus they
cancel on a consistent assignment, and the sum of the adjusted local
maxima is an upper bound for the original objective. Agreement on full
words attains this bound. The certificate gives each local maximizer,
its adjusted score, and the potentials needed to check the claim.
When the iteration limit is reached before agreement, the result gives
the best bound and any feasible score encountered. An empty local
support certifies infeasibility.

A sweep scans the explicitly supplied local supports. Centering the
update takes one pass over variable occurrences and uses rational
arithmetic. The costs of exact rational operations depend on the bit
length of the adjusted scores and potentials.


## Fixed n-gram projection

An `NgramFamily` defines log-linear distributions over a finite support
supplied by the caller. It counts contiguous n-grams up to the given
order. Exact elimination on integer feature differences retains an
affine basis that represents the same distributions. The base measure
may be set to the cavity weight of a local EP update.

`NgramFamily.project` minimizes inclusive KL from the supplied tilted
distribution to this family. Its convex objective is the log
normalizer minus the dot product of the coefficients with the tilted
feature means. The gradient is the difference in feature means and
the Hessian is their covariance under the fitted distribution. Both
input measures require finite log weights throughout the declared
support. This ensures that the projected moments lie in the relative
interior of the family's moment set.

The result reports the fitted probabilities, coefficients, divergence,
moment residual, and solver status. The tolerance bounds the
infinity norm of the feature residual at convergence. It does not
bound a graph posterior error. For a support of size $m$ and basis
of size $r$, each Newton direction takes $O(mr^2+r^3)$ arithmetic
work. Each line-search trial takes $O(mr)$. Gradients follow the
accepted sequence of operations.


## Finite string belief propagation

A `FiniteStringBeliefPlan` uses an explicit list of admissible words for
each variable. It compiles the factor scores on their finite supports,
then reads the live sources once each per run. Construction counts the
factor entries and the states and candidate arcs of their queries.
Observed variables retain just their supplied word.

One synchronous sweep forms variable cavities from the previous
messages, then sums the factor products over every other variable.
The cavities and outgoing messages are normalized in log space.
Prefix and suffix products allow the update to handle excluded words
without subtracting an infinite log value. Damping mixes the old and
proposed message probabilities.

The result reports the maximum undamped total variation between the
returned messages and their one-sweep updates. This measures a
fixed-point residual. The sweep limit can give a result with larger
residual, and the residual gives no bound on the posterior error of
a loopy graph. A fixed sweep count is available for differentiating a
declared schedule.


## Fixed-family string expectation propagation

`FiniteStringBeliefPlan.expectation_propagation` uses the same factor
tables and cavities as finite string belief propagation. At each site
it combines the cavity with the factor's outgoing log weights to form
the tilted distribution and records its log normalizer. The fitted
marginal is the inclusive-KL projection of that tilt onto the
fixed-order n-gram family with the cavity as its base measure.
Dividing the fitted marginal by the cavity gives the new site message.
Damping interpolates log message weights to stay in the same family.

The schedule reports the outer message residual and the separate fit
for each evaluated site. A local fitting limit, numerical stall, or
tilt without positive weight on the full support leaves the previous
accepted sweep in place. A fixed sweep count differentiates the
executed updates. These finite message estimates use only the
supplied word supports.


A positive `projection_penalty` adds the sum of Euclidean coefficient
norms over the n-gram prefix groups, scaled by that penalty. Each
group contains every feature with that prefix, including the feature
equal to the prefix when present. The empty prefix covers all
coefficients. This objective uses the full n-gram coordinates
because an affine change of basis can change the penalty.

`NgramFamily.project_penalized` takes a step bounded by the maximum
squared centered feature norm, then shrinks each prefix group from its
descendants to the root. Two groups are disjoint or nested. A parent
shrink preserves the directions of its descendants' subgradients,
so the sum of the shrink displacements is a subgradient of the
complete penalty at the final point. This gives the proximal update.
The reported local convergence criterion is the infinity norm of its
gradient mapping. The finite group implementation scans the declared
feature vocabulary. Each shrink copies the coefficient vector, so a
proximal update takes $O(rg)$ work for $r$ coefficients and $g$ groups.

## Candidate string messages

`StringCandidatePlan` scores a finite list of words in a unary LogProb
relation. Its complement is a deterministic acceptor that recognizes
every other word. Its unique accepting path for each omitted word gives
exactly one matching path for every original occurrence of that word.
The candidate queries and the complement thus partition the original
path sum. In particular, the omitted mass can remain visible when it is
too small to recover by subtracting the retained mass from the total.

An optional beam retains the highest-scoring candidates, breaking ties
lexicographically and reporting a tie across the boundary. The full
message scale and the weights of the selected words stay differentiable
on the selected support. Construction limits raise a distinct error.

## Batched contextual edits

`ContextualEditChannel` sums the alignments of observed input-output
pairs with a single packed semiring plan. At input position $i$ and
consecutive insertion count $j$, the scorer supplies logits for all
legal edits using the complete input word. It assigns local
probabilities before any restriction to the observed output, so
competitors that emit other symbols still enter the denominator.

A consuming edit advances $i$ and resets $j$. An insertion advances $j$
at fixed $i$, with a finite upper bound. A stop is available only at
the end of the input. Thus the generation graph is acyclic and has
total mass one by reverse induction on $(i,j)$. For input length $L$
and insertion bound $I$, every emitted word has length at most
$L+(L+1)I$. The plan for a pair adds the observed output position to
this state and includes only matching arcs. It groups the accepted
mass by example after a shared packed forward pass. The original
scorer supplies new logits on every evaluation, with gradients
through both compatible paths and local normalization.

The scorer's rectangular action table scales with input positions,
the insertion bound, and output alphabet size. Its cells and the
compatible observation arcs count toward the explicit work budget.
The bound on consecutive insertions is part of the channel model, and the scorer
can use the whole input as context for each edit.

### Packed training on a contextual channel

A two-layer scorer reads each word's remaining suffix and the current
insertion count. The training fixture observes paired binary strings and
fits their conditional log likelihood. Both execution modes reuse their
observation plans and start with the same neural parameters. The benchmark
checks the loss at all four updates and compares the final parameters.
The singleton mode concatenates the individual scores before each backward
pass so both modes optimize the same batch mean.

| Batch size | Mode | Update ms | Peak Python KiB | Saved tape KiB | Final loss |
| --- | --- | --- | --- | --- | --- |
| 4 | Packed | 2.54 | 20.0 | 25.5 | 4.152484 |
| 4 | Singletons | 5.45 | 36.3 | 22.7 | 4.152484 |
| 16 | Packed | 2.32 | 35.8 | 104.2 | 4.248501 |
| 16 | Singletons | 22.92 | 130.7 | 95.2 | 4.248501 |
| 64 | Packed | 5.69 | 133.9 | 418.8 | 4.332192 |
| 64 | Singletons | 114.46 | 509.0 | 384.1 | 4.332192 |

Times are medians per update across three warmed runs on one arm64 CPU
thread with PyTorch 2.15. They include scoring, backward, and parameter
updates. The two memory columns measure peak Python allocations and
unique tensor storage saved for backward in separate runs. At batch size
64, packing reduced Python allocations while retaining more saved tensor
storage. To reproduce the comparison, run
`python tools/benchmark_contextual_edits.py` from the repository checkout.

### Path entropy and additive risk

`FixedStringQuery.path_statistics` uses packed prefix and suffix
passes on the productive observation graph. Let $\alpha(q)$ and
$\beta(q)$ be its prefix and suffix log masses and $Z$ its total
mass. The posterior log mass for a traversal of arc $q \to t$ is
$\alpha(q) + \log w(q,t) + \beta(t) - \log Z$. Its conditional
log probability given the visit to $q$ is
$\log w(q,t) + \beta(t) - \beta(q)$. Initial and final choices
have the corresponding one-sided prefix or suffix terms.

Entropy is the sum of the initial-choice entropy and the expected
entropy of each subsequent choice, including stop decisions. This
follows by factoring any complete path's posterior probability into
its successive choice probabilities. The same event probabilities
weight the supplied arc and boundary cost tensors. Thus the method
computes the expectation of any additive path cost, which can also
serve as a differentiable risk objective. Original occurrence IDs
select costs even when arcs share weight parameters or an original
arc appears at several observed offsets.

The log normalizer, entropy, and expected costs have derivatives
on the current positive support. Zero-probability choices contribute
zero. An entropy of paths can differ from an entropy of the output
words when several paths emit the same word.

### Alignment-marginal output ranking

`best_marginal_output` compares the Markov path sum for each supplied
output at a fixed input. Each observation graph must be productively
acyclic. Summing over its aligned paths can favor a word whose best
individual path is weaker than another word's. Whenever a fixed
input has a positive finite conditional normalizer, that same
normalizer applies to all candidates.

Each candidate is scored by a sparse topological pass over the
reachable product of states and tape positions. Input coefficients
are interpreted as exact binary rationals for comparison. State and
arc limits apply cumulatively across candidates. The selected result
also provides a packed log-space score whose derivatives flow to the
original weights on the current positive support. Local derivatives
hold the selected word fixed. This computation requires the
candidate set to be supplied explicitly.

### Adaptive joint alignment MAP

`dual_alignment_map` maximizes the sum of log weights of one path per
factor subject to agreement on the complete shared strings. Each
factor comes from the same typed graph used for sum-product messages.
The score includes the initial and final weights. LogProb entries
are interpreted as exact binary rationals for decoding. In a model
where several paths emit the same strings, a different objective would
sum their probabilities before comparing strings.

Each factor must have acyclic productive positive support before
observation restrictions. Its exact decoder tracks the original
state, positions in observed strings, and one finite suffix state
per tape. The suffix state counts occurrences of active n-grams,
including beginning and end markers. Two partial paths at the same
configuration have the same continuations and future adjustments.
Thus only the better adjusted partial score and its original
trace need be retained. The decoder counts state visits and
examined arcs against its configured limits.

For each shared variable and n-gram, the solver's multipliers sum
to zero across copies. Thus the adjusted score sum of any feasible
joint assignment equals its unadjusted score. Summing exact local
maxima gives an upper bound. If the selected local paths agree on
their full words, they are jointly feasible and attain that bound.
If their words differ, the solver can activate a distinguishing
n-gram within the order and feature limits and update centered
multipliers. It can return an uncertified feasible assignment and an
upper bound at the iteration limit. Feature-count agreement alone
does not establish a certificate.

The returned score and certificate concern the evaluated parameter
snapshot. Local paths are discrete decisions. The solver reports
the peak number of reached local configurations and total local
arc examinations, including primal feasibility checks.

### Compact regular n-gram messages

`NgramMessagePlan` compiles the product of a one-tape LogProb
relation and a deterministic suffix automaton over supplied n-grams.
The automaton moves on emitted symbols and retains its state across
epsilon arcs. It charges every matching feature at its ending symbol,
including overlapping occurrences. An original path has exactly one
product path and receives the same feature score as any other path
with the same output word.

The compiled message has log weight $\log M(w) + \theta^\mathsf{T}
\phi(w)$, where $M$ is the original weighted language and
$\phi$ counts feature occurrences. It reuses the original weight
and coefficient sources. Each weight evaluation gathers the
required coefficient rows and shares identical pairs of original
slots and feature emissions.

The automaton uses prefixes of the supplied feature set and failure
links to proper suffixes. Only reached product configurations are
created, and no iteration over the alphabet is required. The work
budget includes trie construction, failure links, examined product
arcs, and feature emissions. Ordinary relation limits and closure
checks apply when summing the message. In particular, tilting a
cyclic language can change whether its normalizer is finite.

### Penalized projection on a regular language

`penalized_regular_ngram_projection` fits a regular n-gram message
relative to a finite acyclic base language $M$. The normalized
target is the product of $M$ and a supplied likelihood, so the
target's support is included in the base. The product uses sparse
epsilon-filtered intersection. The fitted distribution is
proportional to $M(w)\exp(\theta^\mathsf{T}\phi(w))$.

Let $Z_\theta$ be its normalizer and let $\mu$ be the target
feature mean. The change in inclusive KL relative to the base is
$\log Z_\theta-\log Z_0-\theta^\mathsf{T}\mu$. The projection
adds the sum of Euclidean norms of the coefficient groups for
each feature prefix, scaled by the specified penalty. It obtains
both sets of expected counts by differentiating sparse path sums.
If the base's longest word has length $L$, a feature for a gram $g$
counts at most $r_g=\max(0,L-|g|+1)$ occurrences. Its variance is at
most $r_g^2/4$. The Hessian of $\log Z_\theta$ is the feature covariance,
so its operator norm is at most the trace bound
$B=\sum_g r_g^2/4$. The optimizer tries a larger proximal step up to
one and checks the quadratic upper bound on the smooth objective.
Halving stops by the globally valid step $1/\max(1,B)$. Thus the
backtracking work per accepted update is logarithmic in the bound,
and an accepted step gives descent in exact arithmetic. The
implementation checks trial steps in float64 with a rounding allowance.

The projection reports the proximal gradient residual separately
from its iteration status. The fitted message compiles only
nonzero features, so the penalty can change the number of reached
states. The result also retains the target's log mass and the site
scale that matches it. The fit uses a fixed weight snapshot and
reports its numerical result without differentiating the optimizer.

### Expectation propagation with regular messages

`regular_string_ep` takes a typed string factor graph and a finite
acyclic reference language for each variable. Its model weights
multiply the graph factors and the supplied reference weights.
A unit-weight reference restricts a variable to its declared
regular support. An observed variable is further restricted
to its observed word.

A site has a log scale and a linear score over its active n-grams.
A cavity reweights the reference language using the coefficients
from all other incident sites. The factor multiplies its
incoming cavities by joining on their shared tapes, and marginalizes
the other variables. Local projection fits the resulting tilted
distribution in the cavity-based n-gram family. The site then
retains the fitted coefficients and a scale that matches the
tilted mass under normalized cavities.

Each sweep proposes all sites from the preceding state. Damping
interpolates log scores and scales. The residual takes the largest
undamped change in log scale plus the maximum accepted word length
times the coefficient-change one-norm. This bounds the change in
a site's log score uniformly over its reference support. A failed
local optimizer and an exhausted outer sweep retain the last
accepted messages with distinct statuses. Relation construction
or closure limits raise their usual errors. The result describes
an approximate inference run over a fixed weight snapshot.

#### Two-variable comparison

The benchmark gives each variable all four two-symbol words. One binary
factor links them and each variable has a unary factor. Exact sparse tree
marginals agree with enumeration over the 16 assignments. The beam keeps
three words at each directed message and approximates the intermediate sum
using the retained incoming support. The n-gram vocabulary includes both
unigrams and all four bigrams.

| Method | Status | Mean TV error | Site features | Site states | Marginal states | Median ms | Peak Python KiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact sparse tree | exact | 0.000000 | — | — | 452 | 10.1 | 229.3 |
| Beam 3 | truncated | 0.399886 | — | 24 | 8 | 0.1 | 2.6 |
| Regular EP (0) | converged | 0.000000 | 24 | 28 | 14 | 895.6 | 197.1 |
| Regular EP (0.025) | converged | 0.048646 | 12 | 24 | 14 | 3185.6 | 205.1 |
| Regular EP (0.1) | converged | 0.112319 | 7 | 19 | 13 | 114.5 | 196.0 |
| Regular EP (0.3) | converged | 0.153415 | 0 | 12 | 6 | 38.1 | 175.0 |
| Regular EP (1) | converged | 0.153415 | 0 | 12 | 6 | 33.5 | 174.6 |

The site and marginal counts sum states across the returned representations.
Regular site counts use the automaton compiled on each site's active features.
Beam counts use tries of the surviving word prefixes. Times are medians of
five runs on one arm64 CPU thread with PyTorch 2.15. Peak memory counts Python
allocations traced during a separate run. To reproduce the comparisons, run
`python tools/benchmark_string_messages.py --repeats 5` from the checkout.
The penalty reduced site and marginal sizes in this graph. The smaller first
penalty increased optimization time as compared with the unpenalized fit.

### Posteriors on closed productive components

For an explicitly permitted finite star, the prefix weight at a
state is the weight of every walk ending there, including walks
inside its component. Within a cyclic component the chart reads the
prefix at the component exit introduced for that state. The suffix
chart reverses the arcs and exchanges boundary roles, using the
same component closure. This gives the prefix and suffix factors
needed for expected traversal counts and path entropy. A primitive
arc's traversal can recur along one path, and each occurrence enters
its expected count.

Sampling first selects an initial mark in proportion to its weight
times its suffix mass. At each state it chooses a stop or primitive
arc in proportion to the stop weight or the arc weight times the
successor suffix mass. In exact arithmetic the suffix ratios
telescope to the normalized weight of each complete path. Their
weights sum to one under the finite closure contract, giving
almost-sure termination. Runtime sampling uses float64 and a
cyclic query requires an arc-step budget. A draw beyond that budget
raises a resource error. The reported path weight uses the
original algebra and must fit its input precision.

## Sparse backoff queries

A `FailureArc` applies when a one-tape deterministic state lacks a
transition for the requested symbol. An explicit zero-weight
transition still takes precedence. `backoff_query` carries the
unconsumed symbol through the failure chain, then consumes it at its
first explicit match. It multiplies the failure weights in traversal
order before the explicit weight. Completion uses the state's own
final weight. Every failure chain must be acyclic.

After indexing the declared transitions and checking the failure
links, query construction follows only the selected path. Its work is
linear in the number of examined symbol probes, and its storage is
linear in the selected path length. In particular, the number of
symbols in the alphabet does not determine either cost. The ordinary
fixed-string evaluator reads the original weights when it scores
the resulting path.

`backoff_total_weight` sums over every word when the reachable union of
ordinary and failure edges is acyclic. In reverse topological order,
the suffix total at a state is the sum of its own final weight and its
first-symbol contributions. Each explicit symbol contributes its arc
weight times the target's suffix total. A missing symbol can inherit its
contribution from the failure target, left-multiplied by the failure
weight. This recurrence follows from partitioning nonempty words by
their first symbol. Its empty-word contribution uses the state's own
final weight.

A persistent sum tree stores the contributions across the distinct
symbols present on reachable arcs. A failure edge scales a shared tree,
and an explicit symbol overrides a single leaf, even at zero weight.
This uses distributivity and requires only semiring operations.
Following a linear scan of the declared topology and weight source,
each reached fallback needs at most one tree update, and each reached
ordinary edge needs logarithmically many in the number of distinct
reached symbols. State and work limits apply to the reachable graph and
tree updates. The calculation keeps derivatives through the original
weights for differentiable semirings.

### Static deterministic minimization

`minimize_static_acceptor` snapshots a one-tape Markov acceptor, removes
zero-weight arcs and marks, and trims states outside its positive
support. The remaining graph must be acyclic. Each state must have at
most one outgoing arc per symbol, with all arcs consuming one symbol.

For a state $q$, let $R_q$ be its weighted suffix series and $g_q$ the
rational greatest common divisor of its nonzero coefficients. Outgoing
labels have disjoint suffix supports. Thus $g_q$ is the gcd of the final
weight and the products $w(q,a,t)g_t$ over outgoing arcs. A pass from the
leaves computes $g_q$ and hashes a signature containing the final weight
divided by $g_q$, the scaled outgoing weights, and the successor classes.
Equality of signatures is equivalent to equality of $R_q/g_q$ by induction
on the maximum suffix length. Multiplying the original initial weight
by $g_q$ preserves the weighted language. Any epsilon-free deterministic
acceptor with one initial state must distinguish positive prefixes whose
residual series are not proportional, so this construction uses the
minimum number of states.

The pass performs one rational gcd per retained arc and sorts each
state's outgoing labels. Its work also includes validating the source
weight slots and the input topology. Rational arithmetic costs depend
on coefficient bit lengths. Coefficients in the result must be exactly
representable in the input floating-point dtype. Trainable weights
are refused because changing them can change the equivalence classes.

### Ranked constraints for a fixed input

`ranked_output_candidates` restricts a candidate generator to one
input word. Each original arc occurrence carries a vector of nonnegative
integer violations. Each path sums its vectors and the highest-ranked
coordinate on which two candidates differ determines their order.
The returned acceptor keeps every globally optimal path and retains
the original sources of its semiring weights. Ranking considers
structural paths even when their current semiring weight is zero.

For each state $q$ in the fixed-input graph, a reverse Dijkstra pass
computes its least lexicographic suffix rank $d_q$. Lexicographic order
is translation-invariant and every arc rank is nonnegative in that
order. An arc $q \to t$ with rank $v$ belongs to an optimal suffix
exactly when $d_q = v + d_t$. The construction keeps these tight arcs
and the least-ranked initial marks, then trims inaccessible arcs.
Summing the equality along any retained accepting path gives the
global optimum. Conversely, a globally optimal path has no positive
rank slack at any step. A retained cycle can contain only zero
violation vectors. Finite-state weight queries use their ordinary
closure contract if such a cycle emits no output.

For $Q$ reached configurations, $E$ examined compatible transitions,
and $r$ coordinates, the ranking pass takes $O(Q + E + (Q + E)r
\log(Q + E + 1))$ operations counting each integer operation as one.
Building the fixed-input graph obeys its explicit state and work
budgets and also scans the original topology and rank table. The
acceptor describes candidates for this input alone. Compiling one
transducer for arbitrary ranked inputs requires additional
regularity hypotheses.

### Uniform compilation for capped ranked constraints

`compile_ranked_ot` compiles a single transducer for all inputs
when each violation count has an explicit finite cap. A cap of
one records whether a constraint has been violated. The count
update adds the arc's violations, then clips each coordinate
at its cap.

Each weighted product state records the current candidate and
its capped vector. It also records the set of all candidate
configurations reachable by consuming the same input prefix.
This set is closed under input-epsilon transitions. A consuming
transition updates the set for its input symbol and takes the
epsilon closure again. The construction retains all reachable
vectors, since saturation can reverse the ordering of two prefix
vectors after they receive the same further violations.

An original path uniquely determines its weighted product path.
That path accepts exactly when the original path ends in a final
state and its rank equals the minimum among all final
configurations in the competitor set. The competitor set filters
the original paths without contributing any weight or alignment
multiplicity. All tied original paths keep their weights, and
sharing through the original parameters remains live.

The compiler visits reachable weighted product states and caches
competitor transitions by state set and observed label. It
does not iterate over the entire alphabet or the entire cap
product. Nevertheless the number of reachable sets can be
exponential. State and work limits give explicit failure when
exploration exceeds the supplied budgets. The caps define the
model, and the limits bound construction.

## Beam-pruned finite string beliefs

`FiniteStringBeliefPlan.run_pruned` applies synchronous sum-product
over the declared typed candidate supports. It ranks each proposed
factor-to-variable message by raw marginal weight, breaks ties by
lexicographic word order, and normalizes the retained positive
weights. The full and omitted masses use the same gauge, with
incoming variable cavities normalized before the factor sum. These
masses refer to the declared finite candidates.

Damping first restricts the old message to the positive support of
the new pruned proposal. When the old message has positive retained
mass, the update mixes the two normalized vectors on this support.
When their positive supports are disjoint, the update uses the new
proposal. Every transmitted message after the first sweep thus has
at most the requested positive support size.

The reported total-variation residual compares the current messages
with the undamped pruned proposal. `converged` describes this
approximate update. The result also records positive omitted mass
across completed proposal evaluations and per-message masses for
the latest complete proposal. A hard support conflict reports
`empty_support` with no normalized beliefs. Plan construction still
reports resource exhaustion through the configured state, work, and
factor-entry limits.

The declared factor tables are built within those limits. The beam
caps positive message support and leaves the factor-table allocation
bound in place. Derivatives follow the executed updates with beam
choices held fixed.

::: quivers.core.relations
    options:
      members:
        - RationalRelation
        - RegularNgramProjection
        - RegularStringBelief
        - RegularStringEPResult
        - RegularStringSite
        - RegularStringSiteUpdate
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
        - ConditionalStringQuery
        - ContextualEditChannel
        - ContextualEditPlan
        - QueryWork
        - FailureArc
        - FiniteRelationRestriction
        - FiniteStringBeliefPlan
        - FiniteStringScore
        - MarginalOutputChoice
        - NgramFamily
        - NgramMessagePlan
        - NgramProjection
        - PathStatistics
        - PenalizedNgramProjection
        - ProductiveEpsilonCycleError
        - PrunedStringBeliefResult
        - RelationLimitError
        - RelationPath
        - StringAlignment
        - StringAlignmentCertificate
        - StringAlignmentMapResult
        - StringBeliefResult
        - StringEPResult
        - StringSiteProjection
        - StringFactor
        - StringFactorGraph
        - StringMapCertificate
        - StringMapResult
        - StringCandidateMessage
        - StringCandidatePlan
        - StringVariable
        - TreeStringMarginal
        - auto_intersect
        - backoff_query
        - backoff_total_weight
        - best_marginal_output
        - closure
        - compile_ranked_ot
        - concatenate
        - condition
        - dual_alignment_map
        - dual_string_map
        - intersect
        - invert
        - join
        - penalized_regular_ngram_projection
        - minimize_static_acceptor
        - project
        - regular_string_ep
        - ranked_output_candidates
        - trim
        - union
