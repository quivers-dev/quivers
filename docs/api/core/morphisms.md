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
A finite LogProb score whose normalized log ratio overflows the
evaluation dtype raises `OverflowError`. An impossible output retains
log zero. Exact tree marginals apply the same precision check.

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
costs. A [checked native path rank request](../qiec/serialization.md#checked-deterministic-path-ranking)
exposes the same ranked traces and separates the observation, rank,
and result storage limits.

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


### Certifying numerical accuracy of Markov path weights

`certify_markov_path_weight` checks an absolute or relative forward-error
tolerance for the ordinary packed score. Its mathematical input is the
evaluated binary64 Markov weight vector from the same call. The verifier
reads that vector as exact binary rationals. This scopes the guarantee to
the path evaluation, including for a vector produced by a rounded source
weight expression. The returned tensor keeps its ordinary derivative.
The certificate concerns only its scalar value.

The verifier computes rational suffix intervals on the productive graph.
At an acyclic state it sums the final weight and weighted successor
intervals. Within a cyclic component, let $A$ be the matrix of internal
arc weights, with a contribution from every original occurrence. The
right-hand side $b$ includes final and outgoing weights and has a known
interval $[b_-,b_+]$. The ordinary local star proposes a suffix vector
$y$ and a positive direction $v$. The verifier checks
$d=(I-A)v>0$ over the original sparse rows. The resulting inequality
$Av<v$ gives $\rho(A)<1$, and $(I-A)^{-1}$ is entrywise nonnegative.

For either endpoint of the right-hand side, the residual is
$r_\pm=b_\pm-(I-A)y$. Set $c=\max_i\max(|r_{-,i}|,|r_{+,i}|)/d_i$.
Every admitted residual lies between $-cd$ and $cd$. Applying the
nonnegative inverse bounds the suffix by $y-cv$ and $y+cv$. The lower
endpoint may be clipped at zero. Thus a small residual with very small
$d$ can still give a wide interval. The verifier propagates the suffix
intervals to the initial marks and tests the returned scalar against
both endpoints. The relative check requires a positive lower endpoint
or a zero-width interval and a score both equal to zero.

Observation construction has the same state and arc budgets as a fixed
query. Each declared local star order $m_i$ includes auxiliary states
for parallel transitions. The compiler checks the aggregate
$\sum_i m_i^2$ before allocating the associated packed expansions or
dense star matrices. Ordinary closure uses $O(\sum_i m_i^3)$ scalar work.
The matrix-vector proposals add $O(\sum_i m_i^2)$ floating-point work,
and the verifier adds $O(V+E+W)$ exact operations for active states $V$,
arc occurrences $E$, and original weight slots $W$. Its bit budget is
checked conservatively before producing exact integer intermediates.
The arithmetic cost per rational operation includes bounded integer
products and gcds. Saved tensors for differentiating the packed score
follow the usual evaluation contract and can retain quadratic arrays
at each of the $m_i$ elimination pivots. The certificate verifier keeps
one pair of rational endpoints per active state and one exact value per
weight slot, in addition to the sparse observation and closed matrices.

When the reachable graph, star order, closure cells, or exact arithmetic
exceeds a limit, the call raises the corresponding resource error.
Unsupported algebra or precision, divergence, an unverified contraction,
and an interval that misses the requested tolerance have distinct
diagnostics. The acyclic LogProb guarantee uses directed
transcendental enclosures.

### Certifying numerical accuracy of acyclic LogProb path weights

`certify_logprob_path_weight` checks an absolute tolerance in log units
for a fixed-word observation whose productive graph is acyclic. The
ordinary packed score and the verifier read the same evaluated binary64
slot snapshot. The returned tensor keeps its ordinary derivative, and
the certificate concerns its value.

The verifier converts each used binary64 log weight to Decimal exactly.
It treats negative infinity as exact zero mass and a zero log weight as
exact unit mass. For each other weight, it encloses the exponential by
the two neighbors of the correctly rounded nearest-even result.
[Python's Decimal operations](https://docs.python.org/3.14/library/decimal.html#decimal.Decimal.exp)

Let $B_q$ be the final weight at configuration $q$. In reverse
topological order the verifier bounds $S_q=B_q+\sum_{e:q\to r}w_e S_r$.
All weights and suffix masses are nonnegative, so each lower-bound
operation may use the lower endpoints of its operands and round down.
The upper-bound operations use upper endpoints and round up. Each
sparse update uses a directed fused multiply-add with one final rounding.
Summing over the initial marks in the same way encloses the exact
mass $Z$. [General Decimal Arithmetic Specification](https://speleotrove.com/decimal/daops.html)

Monotonicity of the logarithm and the neighboring decimals around its
nearest-even result give rational bounds $L\leq\log Z\leq U$ when
the mass interval is positive. The verifier tests the larger distance
from the packed score to $L$ and $U$ in exact rational arithmetic. Both
mass endpoints equal to zero and a packed score of negative infinity
give a verified zero mass. Its log endpoints are absent and its error
is zero by convention.

The declared adjusted exponent limit $E$ is symmetric. Each exponentiated
input must satisfy $|\ell|\leq 2E$, which keeps its exact mass in range
because $e^2<10$. Later arithmetic, including the two logarithms and their
needed neighbors, must also stay in normal Decimal range. A path with
a finite log weight of $-1000$ can pass these checks even though its mass
would underflow in binary64. Exponent exhaustion and subnormal or
unavailable endpoints yield resource refusals. A productive epsilon
cycle is outside the certified domain. An unsupported dtype, a nonfinite
packed score on positive support, and an interval too wide for the
requested tolerance have distinct diagnostics.

The verifier uses $O(V+E_a+W)$ graph work for $V$ active configurations,
$E_a$ arc occurrences, and $W$ weight slots. The Decimal work budget
counts at most four operations per exponentiated slot, two per used arc
or initial mark, and four for the terminal logarithm bounds. This
counts calls to the arithmetic library. Decimal precision and exponent
range bound the stored operand sizes. The verifier checks the integer
bit bound before converting a Decimal endpoint to a rational and
charges the exact operations in the final error test separately.

### Certifying numerical accuracy of cyclic LogProb path weights

`certify_cyclic_logprob_path_weight` admits productive epsilon cycles
under explicit local star order and aggregate cell limits. The
ordinary score and verifier read one evaluated float64 snapshot.
The result remains the ordinary LogProb tensor and the certificate
covers its value in log units.

For each productive strongly connected component, let
$A_{ij}=\sum_{e:i\to j} \exp(\ell_e)$ sum the weights of its internal
arc occurrences. The right side $b$ contains final weights and
weighted suffixes from downstream components. Directed exponential
endpoints bound every arc weight, so the lower and upper systems
preserve the ordering of each finite partial path sum. Within each
local system the endpoint coefficients are exact rationals, with
every parallel occurrence included.

Eliminating state $k$ replaces its contribution by

$$
x_k=\frac{b_k+\sum_{j>k} A_{kj}x_j}{1-A_{kk}}.
$$

At this step $A_{kk}$ sums the nonempty return walks through
previously eliminated states. Every segment through $k$ consists of
an entrance, zero or more such return walks, and an exit. Thus a
strictly positive pivot $1-A_{kk}$ permits the exact geometric
sum and the corresponding nonnegative substitution into the
remaining equations. Positive pivots at all upper-system steps
establish a finite upper star for the component because these checks
apply to every unit right side. Back substitution then gives its upper
suffix sums. Coefficientwise comparison of finite partial walk sums
and their increasing limits justifies the lower and upper results.

The verifier transfers the two exact rational solutions outward into
Decimal. Directed sparse updates connect the components and the
terminal logarithm yields the log endpoints. The verifier preserves
masses such as $\exp(-1000)$ when they fall below the binary64 range.
If an upper pivot is nonpositive, the verifier checks whether the
lower endpoint system proves divergence. A nonpositive lower pivot
gives return mass at least one, so its geometric series diverges.
If all lower pivots remain positive, the available bounds give an
unverified accuracy outcome. This may occur when the true mass is
finite. The verifier checks every structurally productive component,
including those with zero current boundary mass, so a divergence
diagnostic concerns that local star. The requested score accuracy and
the numerical star evaluator must also pass their respective checks.

The rational elimination uses $O(n^2)$ cells and $O(n^3)$ exact
operations per component of $n$ original observation states, in
addition to processing its arc occurrences. The order and aggregate
cell limits include auxiliary states used by the ordinary closure for
parallel arcs. Lower and upper rational systems are solved in sequence.
The exact integer bit and operation budgets apply throughout the proof,
and the Decimal range and work budgets also cover the directed
transfers. The acyclic core entry point and checked native certificate
retain their acyclic domains.

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

### Centered acyclic string likelihoods

`ConditionalStringQuery.centered_query` and `TreeStringMarginal.centered_query`
compile an opt-in binary64 LogProb likelihood. The fixed-word numerator
and its all-word denominator use the same evaluated weight vector. Both
productive observation graphs must be acyclic. Disconnected tree
components retain their existing normalizer requirements.

For each graph, let $h_q$ be the maximum log weight of a positive suffix
from state $q$, including its final mark. Compute these maxima by a reverse
topological pass over the exact dyadic input weights. For an arc $q\to r$,
replace its log weight $w$ by $w+h_r-h_q$. Subtract $h_q$ from a final
mark at $q$. An initial mark at $q$ receives $h_q-M$, where $M$
is the maximum log weight of a full path. The total shift is $-M$ on
every accepting path, for any path length. At each productive positive
state, at least one suffix attains its maximum and thus has reduced
log weight zero. A reverse packed sum operates on these centered suffixes.

The evaluator combines the exact difference of the two maxima with
the two rounded packed log sums before rounding the scalar result.
Every included occurrence retains the derivative of its original live
log weight, with the chosen gauge held fixed. The gauge cancels from
each full path, even when distinct arcs use the same parameter.
A zero-weight prefix or suffix contributes no positive path. A zero
numerator returns log zero. A zero denominator is undefined. A reduced
factor or resulting log probability outside binary64 precision raises
`OverflowError`. The packed sums and their derivatives still use
binary64 arithmetic. This method gives no requested-error certificate.

The state and candidate-arc budgets apply to each observation as usual.
The combined productive charts contain $V$ states and $A$ occurrences,
counting both boundary marks and arcs. With $W$ weight slots, the exact
pass preflights $W+2V+5A$ visits before building the lifted charts. Together the
observations must fit within the combined occurrence cap. The
reported work counts visits actually made. A pass uses a linear number of
dyadic operations. Once scaled by $2^{1074}$, its exact integers have
$O(2098+\log(V+1))$ bits. The packed charts and their backward tape have
linear storage in the two active graphs, apart from the source weights
and observation construction.

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
accepted sequence of operations. The line search evaluates the change in the objective
from the current probabilities. For a coefficient displacement $d$ and
current mean $\hat{\mu}$, that change is

$$
(\hat{\mu}-\mu)^\mathsf{T}d
+\log\!\left(1+\mathbb{E}\left[e^U-1-U\right]\right),
\qquad U=(\phi(w)-\hat{\mu})^\mathsf{T}d.
$$

The centered change has zero expectation under the current probabilities.
Evaluating the nonnegative exponential remainder preserves small changes
near the optimum. Convergence still uses the requested moment tolerance.


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
on the selected support. The normalized message uses centered log
weights, while proposal reports retain the raw scale. A finite
normalized candidate that overflows raises `OverflowError`. Construction
limits raise a distinct error.

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

### Packed nonnegative feature moments

`FixedStringQuery.log_feature_moments` runs the packed acyclic executor
over an occurrence-indexed plan with `ExpectationSemiring` over
`FreeSemimodule(LogProb, d)`.
Each initial mark, arc, and final mark has an optional nonnegative
feature increment $f$ represented by its logarithm. An occurrence
with mass $p$ starts with the pair $(p,pf)$. Multiplying two pairs adds
the increments for the extended path because their second component
is $p_1(p_2f_2) + p_2(p_1f_1)$. Adding pairs sums the weighted
increments across distinct paths. [Li and Eisner (2009), §3](https://aclanthology.org/D09-1005/)

The query returns $\log Z$ and the vector of unnormalized log moments.
Subtract $\log Z$ from the vector for conditional log expectations
when $Z$ is positive. Operations stay in log space, including for a
path with a log weight of $-1000$. Feature rows follow original
occurrences even when weights are tied or an arc is traversed at
multiple offsets. `path_statistics` computes entropy and signed
additive costs, including on observations with a supported cyclic
closure. The packed expectation pass uses a fixed-width carrier at each
active entry of an acyclic observation chart.

For a productive cyclic observation, pass `allow_cyclic=True` to a
query with a scalar closure budget. For each arc occurrence $e:s\to t$,
the unnormalized moment adds $\alpha(s)w(e)f(e)\beta(t)$, with
corresponding terms for initial and final marks. Choosing one marked
visit gives a unique prefix, occurrence, and suffix. Since the summands
are nonnegative, their order can change under a certified finite scalar
closure. Inside one component this computes $A^* B_f A^*$, where $B_f$
sums the feature-weighted original arcs of that component. Repeated visits
and distinct parallel occurrences both contribute. Divergence and an
uncertifiable scalar closure produce the corresponding scalar errors.

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

Before returning a bound, the solver checks the saved local witnesses
under the multipliers that produced it. First it reconstructs each
original walk, verifies its observed words, and recomputes its two
scores from the boundaries, arc weights, and direct n-gram counts.
Then it derives the exact suffix optimum in each reachable local
configuration. In reverse topological order, each scalar suffix value
is the maximum over its available final score and every arc score
plus its successor's suffix value. This pass does not use the decoder's
prefix maxima or backpointers. The claimed adjusted scores must attain
these independently calculated maxima, and the multipliers must still
sum to zero. Feasible lower witnesses are checked against their
complete word assignment. An infeasibility claim requires a local
subproblem with no accepting path under its observations.

Reachability and suffix verification each examine the same reachable
configurations and candidate arcs as a local forward decode. They
reuse its state limits. With `max_work = B`, each scan examines at
most $B$ arcs, for at most $2B$ candidate arc probes per checked local
witness, plus visits to the witness's original arcs. The verifier stores one exact
score per reachable configuration and no materialized transition table.
Trace checking additionally visits the selected primitive arcs and
computes n-gram counts for the selected words. Exact rational arithmetic
and feature work depend on coefficient bit lengths, tape widths, word
lengths and active feature orders. `max_verification_states` and
`verification_arc_examinations` record the additional graph work.

The solver may also stop at a requested absolute or relative objective
gap. With an exact feasible score $L$ and dual upper bound $U$, the
unknown optimum $z$ satisfies $L\leq z\leq U$. Thus $U-L$ bounds the
absolute suboptimality of the returned assignment. For $z>0$, the
relative error is $1-L/z$. For $z<0$, it is $L/z-1$. On a positive
interval with $L\geq0$, or a strictly negative interval, the relevant
function increases with $z$. The maximum relative error is thus
$(U-L)/|U|$. If $L<0\leq U$, these bounds alone give no finite relative
guarantee. When $L=U$, the relative gap is defined to be zero, including
at the exact zero optimum.

Supply `absolute_gap` or `relative_gap` as a nonnegative exact rational.
All supplied targets must be established. An early stop based only on
these bounds has status `gap`, with `gap_target_met=True`. The separate
status `optimal` and its agreement certificate require exact local
maximizers to agree on their complete words. When there is no feasible
result, neither gap can be established.

The returned score and certificate concern the evaluated parameter
snapshot. Local paths are discrete decisions. The solver reports
the peak number of reached local configurations and total local
arc examinations, including primal feasibility checks. The [joint alignment
MAP benchmark](../../developer/string-alignment-map-benchmark.md)
compares these counts with a materialized intersection and verifies
both against independent path enumeration.

`paired_dual_alignment_map` accepts a partition of original factor
positions into singletons and pairs. Each pair must share exactly one
nominal tape variable. Its filtered sparse join keeps the shared tape
once, retains the other tapes in join order, and represents every
compatible original path pair once. The solver applies one multiplier
copy per group for each shared variable. Equality between groups thus
imposes the same original feasible assignments as the singleton solver.

The grouped decoder reads the original binary-rational log coefficients.
Its arc macros retain primitive weight slots, and it adds the two
boundary coefficients directly as rationals. This preserves the
original objective even when adding the boundary in a float would
lose a smaller coefficient. A feasible grouped path yields one
`StringAlignmentTrace` per original factor. Each expansion checks its
original endpoints, ordered arcs, emitted words, and exact base score.
The grouped result's certificate has one local maximum per group.

Each paired product uses its own state and work limits. If its two
operands have $V_l$ and $V_r$ states and the builder emits $M$ matched
moves, it has at most $3 V_l V_r+M$ states. The extra states keep
matched primitive weights separate. Label probes and emitted arcs are
charged to `max_group_work`, and products remain available between
dual updates. Local graph sizes and work are then bounded and reported
as for singleton decoding. This groups only the explicitly requested
factor pairs. A pair with zero or multiple shared variables is refused.

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
$B=\sum_g r_g^2/4$. The fixed-Lipschitz optimizer uses a step at most
$1/\max(1,B)$, rounded downward in float64 if needed. It evaluates the
target and base moments once each, then makes one moment evaluation per
accepted step. Adaptive backtracking tries a step up to one and checks the
quadratic upper bound on the smooth objective with a float64 rounding
allowance. Halving ends at the globally valid step. The number of trials
per accepted update grows at most logarithmically with $\max(1,B)$.
Both policies report actual moment evaluations, including rejected
trials. They reuse the same sparse moment plan across proposals.
On a uniform base over the empty word and an eight-symbol repetition,
with tilted mass ratio $7:1$ in favor of the repetition, both policies
choose the first coefficient step $3/16$. Adaptive backtracking takes
seven moment evaluations including four rejected trials. The fixed
policy takes three.

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

Unit sites initialize with zero coefficients and log scales. A supplied
initializer may instead give finite coefficients and a finite log scale
for every site in factor, tape, and feature order. This sets the initial
iterate and carries no convergence claim. The solver still measures the
undamped defect at the last accepted iterate, including at a zero sweep
budget. Total moment evaluations include every attempted local fit of
this synchronous schedule.

#### Two-variable comparison

All methods use the same graph with one binary factor, two unary
factors, and the same four two-symbol words on each variable.
An independent dense sum over the 16 assignments supplies the exact
marginals. The exact sparse tree and full-width production beam agree
with these marginals. The width-three row runs
`FiniteStringBeliefPlan.run_pruned` on the same graph, including its
final proposal pass. Regular EP uses both unigrams and all four
bigrams as candidate features.

| Method | Status | Mean TV error | Site features | Site states/arcs | Marginal states/arcs |
| --- | --- | --- | --- | --- | --- |
| Exact sparse tree | exact | 0.000000 | — | — | 452/480 |
| Production beam 3 | converged (truncated) | 0.399886 | — | 24/20 | 8/6 |
| Regular EP (0) | converged | 0.000000 | 24 | 28/24 | 14/12 |
| Regular EP (0.025) | converged | 0.048646 | 12 | 24/22 | 14/12 |
| Regular EP (0.1) | converged | 0.112319 | 7 | 19/20 | 13/12 |
| Regular EP (0.3) | converged | 0.153415 | 0 | 12/16 | 6/8 |
| Regular EP (1) | converged | 0.153415 | 0 | 12/16 | 6/8 |

The representation counts sum states and arcs across the individual
sites or queried marginals. For regular sites the benchmark compiles
the automaton induced by the fitted active features. The beam counts
derive from the tries of the surviving word prefixes. The production
beam holds dense vectors over the declared finite axes.

Let $A$ count the factor-argument sites, $S$ count accepted sweeps, and
$L$ count fits recorded on the final pass. Each accepted sweep visits
all $A$ sites, and the solver evaluates another pass at the returned
state. Thus it attempts $AS+L$ local fits, including any failing fit
on a final partial pass. A moment evaluation is one evaluation of
the sparse normalizer and feature counts. The solver counts both
target and base calls and the calls made by rejected optimizer
trials. The benchmark test observes the actual site and moment
invocations and detects a false reported sweep.

On this graph a complete beam pass has four proposals and charges
$2\cdot16+4+4=40$ factor-table cells across the outgoing axes. The
three executed passes include the final check. These counts
describe the input cells for the proposal reductions. Beam sorting,
cavity normalization, and table construction have separate costs.
A test derives the expected width-three support from the sums over
retained incoming candidates and detects a wider beam substituted for it.

| Method | Sweeps | Site fits | Moment evaluations | Beam input cells | Median ms | Peak Python KiB |
| --- | --- | --- | --- | --- | --- | --- |
| Exact sparse tree | — | — | — | — | 9.0 | 229.0 |
| Production beam 3 | 2 | — | — | 120 | 4.4 | 30.5 |
| Regular EP (0) | 4 | 20 | 1709 | — | 630.2 | 194.7 |
| Regular EP (0.025) | 9 | 40 | 4139 | — | 2171.0 | 203.7 |
| Regular EP (0.1) | 2 | 12 | 141 | — | 93.3 | 196.1 |
| Regular EP (0.3) | 1 | 8 | 16 | — | 25.0 | 175.2 |
| Regular EP (1) | 1 | 8 | 16 | — | 23.9 | 175.1 |

Run `python tools/benchmark_string_messages.py --repeats 5` from the
checkout. Times are medians of five calls after one warmup per row,
on macOS 15.7.4 arm64 with one CPU thread and PyTorch 2.15.0. Calls
include solver preparation, marginal scoring, and representation
counts. Every invocation starts from the same initial sites. The
memory column measures peak allocations traced by Python's
`tracemalloc` during a separate call. It describes Python allocations
and omits native tensor and allocator storage. The smallest positive
penalty halved the retained feature count in this graph and increased
moment evaluations from 1,709 to 4,139.

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

`minimize_static_acceptor_with_work` also accepts cyclic positive
support. For each productive state $q$, let $u_q$ be its least accepted
suffix in length-then-symbol order and let $p_q=R_q(u_q)>0$. A reverse
breadth-first traversal finds the distance to a final mark. A pass in
increasing distance uses the least first symbol that decreases the
distance, multiplying its weight by the already computed scale of its
target. This is finite-suffix weight pushing over exact rationals.
[Eisner (2003), §6](https://www.cs.jhu.edu/~jason/papers/eisner.hlt03.pdf).

The quotient construction scales an initial coefficient at $q$ by
$p_q$, divides a final coefficient at $q$ by $p_q$, and replaces an arc
coefficient $w(q,a,t)$ by $w(q,a,t)p_t/p_q$. These factors telescope on
every finite word. The argument applies even if the sum of the weights
of all words diverges. Two nonzero residuals are positively proportional
if and only if their normalized series agree, since proportional series
have the same support and hence the same least suffix. Equality of
normalized series forces equal final coefficients and equal support
after every symbol. The normalized target series assign weight one to
the same least suffix, so evaluating this suffix also forces the two
pushed arc weights to agree. The resulting partition is thus exactly
the stable partition of the deterministic partial automaton with final
weights as state observations and `(symbol, pushed weight)` as labels.
Each class represents a nonzero residual of a reachable prefix, up to
its positive prefix scale. An equivalent deterministic epsilon-free
one-initial acceptor must reach a different state for each such class,
since two prefixes reaching one state have proportional residuals. This
gives the state lower bound for the class in which minimization is claimed.

Refinement initially queues all final-weight blocks. Each popped block
only examines its own states and their incoming arcs, grouped by label.
If two states remain in the same source block but a label takes them to
distinct target blocks, at least one of the defined target blocks remains
queued. Processing that block separates the sources. Splitting a queued
block retains both children in the queue. Splitting any other block adds
its smaller child, which suffices for this invariant when targets first
separate. A state returns to the queue after processing only in a block of
at most half its previous processed size. With $V$ productive states and
$E$ productive arcs, the measured state and predecessor visits are thus
bounded by $V(1+\lfloor\log_2 V\rfloor)$ and $E(1+\lfloor\log_2 V\rfloor)$
respectively for $V>0$. Grouping and splitting use expected constant-time
hash operations per record at fixed key size. Preprocessing is linear in
the input records, and rational arithmetic and coefficient storage depend
separately on bit length. This uses sparse partial-automaton refinement.
[Hopcroft (1971)](https://www.cs.cmu.edu/~cdm/resources/Hopcroft1971.pdf),
[Valmari and Lehtinen (2008), §§2–5](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.STACS.2008.1328).

`StaticMinimizationWork` reports input records, productive states and
arcs, and the two refinement counts. `max_states` bounds the productive
states, and `max_work` bounds the input records plus the two refinement
counts. The returned snapshot uses the input dtype and device. The
shortest-suffix gauge can produce coefficients outside the original
floating-point dtype, even from exactly represented source weights. The
routine raises in that case. The acyclic gcd entry point retains its
own gauge and coefficient ordering.

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
        - CenteredStringLogScore
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
        - PathWeightCertificate
        - LogPathWeightCertificate
        - PenalizedNgramProjection
        - ProductiveEpsilonCycleError
        - PrunedStringBeliefResult
        - RelationLimitError
        - RelationAccuracyError
        - RelationPath
        - StringAlignment
        - StringAlignmentCertificate
        - StringAlignmentMapResult
        - StringAlignmentTrace
        - PairedAlignmentMapResult
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
        - certify_markov_path_weight
        - certify_cyclic_logprob_path_weight
        - certify_logprob_path_weight
        - closure
        - compile_ranked_ot
        - concatenate
        - condition
        - dual_alignment_map
        - paired_dual_alignment_map
        - dual_string_map
        - intersect
        - invert
        - join
        - penalized_regular_ngram_projection
        - StaticMinimizationWork
        - minimize_static_acceptor
        - minimize_static_acceptor_with_work
        - project
        - regular_string_ep
        - ranked_output_candidates
        - trim
        - union
