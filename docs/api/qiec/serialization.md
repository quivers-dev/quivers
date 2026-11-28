# Serialization

`dumps` and `loads` encode a stable [Quivers Indexed Effect Core](index.md) (QIEC) graph as JSON text and decode it again, so that a module
can be stored and rebuilt. A decoded module is checked again, and running it needs the
[runtime providers](../../reference/qvr/execution-and-tooling.md#runtime-providers) that its
[entry computations](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary)
require. The [QIEC developer note](../../developer/qiec.md#stable-serialization) states the format.

::: quivers.qiec.serialization

## Sparse relations

`RationalRelationSpec.of` records the topology and semiring of a sparse relation. The wire
record preserves each tape's alphabet name and cardinality, the domain and codomain split,
and every arc and boundary occurrence in order. Two equal arc records still have different
positions. The slot at each position in `weight_names` refers to the `ParameterId` at the
same position in `weight_parameters`. Repeating an identity shares a weight across slots.

A decoded specification uses `bind` to resolve those identities in an explicit runtime
mapping. Each trainable value is the caller's original scalar `torch.nn.Parameter`. A constant
may be a scalar leaf tensor. All supplied values that the relation uses must share a dtype
and device and satisfy the semiring's carrier. The bind checks the current values, and each
score reads and checks them again. The wire record contains the identities and topology.
Tensor values and runtime providers belong to the binding.

::: quivers.qiec.relations.RationalRelationSpec

A `RelationWeightComplement` binds a Markov weight slot to one minus
an original probability owner. The owner is a floating-point scalar
leaf in the same binding mapping. A composed or derived relation
computes the complement from that owner each time it evaluates its
weights, so both uses contribute to the original gradient. Decoding
the relation restores the weight identities. The complement record
can also be serialized with its original owner identity. A new
process supplies the leaf tensors when attaching the checked request.

::: quivers.qiec.relations.RelationWeightComplement

::: quivers.qiec.relations.RelationWeightBindings


## Fixed-word relation requests

A `FixedRelationRequest` pairs a sparse relation specification with
observed words in tape order, a scoring operation, and
explicit query limits. `pair_weight` returns the scalar weight of
all accepting paths at those words. `conditional_log_prob` uses the
declared input-tape split and returns the log probability of the
output under the closure-checked conditional normalizer.
`path_entropy` returns the conditional entropy of the accepting
paths in nats as a `Real`, under the same finite closure contract.
Existing pair and conditional requests keep the same wire fields.

A `FixedRelationRuntimeProvider` checks the declared foreign
`Compute[Unit, A]` handler before attaching to a reference-machine
invocation. Its scalar result type follows the request. The
provider binds the stable weight identities to the original live
tensors and preserves derivatives through the score. The wire
record carries the structural request. A run supplies the provider
and parameter objects as host attachments.

The Stan, BUGS, and JAGS projections report the checked foreign
handler as an obstacle. Hosted targets that preserve the handler
declaration require an implementation in the target runtime.

A `FixedRelationCostRequest` uses a separate wire record for an
expected additive cost. Each supplied cost vector follows the original
occurrence order of the corresponding boundary table or arc list.
An omitted group contributes zero. The provider returns a signed
`Real` and preserves derivatives to the original relation weights.
Costs are fixed data in the serialized request.

### Checked best-path weights

A `FixedRelationRequest` with operation `best_path_weight` selects
the highest-weight positive alignment for the observed word on each
tape. The exact ranking compares the rational values represented by
the input tensors. The returned scalar uses the original Markov or
LogProb carrier. A tie follows the fixed topology order and carries
the derivative of that selected path's original occurrences.

An empty positive support returns the semiring zero. This objective
requires an acyclic productive graph. In the QIEC reference machine
a foreign linear `Compute[Unit, Real]` or
`Compute[Unit, LogWeight]` handler can bind the request. Native score
targets report that handler as an unsupported capability.

### Checked traversal counts

A `FixedRelationArcCountsRequest` returns one real expected traversal
count for each requested original arc ID, in the requested order. An
ID may occur repeatedly in the request. Distinct parallel arcs keep
distinct counts, including arcs with the same weight owner. Visits
at multiple observed offsets or repeated visits within a productive
cycle all contribute to the same original arc's expectation. A
requested arc outside the observation has count zero.

The result is a checked `Tensor[Real]` whose single extent equals the
length of the request's ID tuple. The linear Compute provider reads
the current weights on every invocation. Cyclic observations require
the existing finite closure certificate and an explicit component
bound. The conditional count requires a positive finite observed
path sum.

::: quivers.qiec.relations.FixedRelationArcCountsRequest

### Checked alignment-marginal output ranking

`FixedRelationMarginalMapRequest` maximizes the total alignment
weight across an explicitly ordered set of output candidates. It
holds all input tapes fixed. The `choose` method returns the selected
words, their exact rational mass, and the differentiable log pair
weight. The reference provider exposes that log weight through a
linear `Compute[Unit, LogWeight]` attachment.

Candidates require acyclic productive Markov observations and share
the declared state and arc limits. The first exact maximum wins a tie.
The derivative holds that output fixed. If every candidate has zero
mass, the scalar score is negative infinity. Native targets identify
the foreign handler that would require this capability.

::: quivers.qiec.relation_marginal_map.FixedRelationMarginalMapRequest

### Checked static minimization

A `FixedMinimizedRelationRequest` wraps a `pair_weight` observation of
an authored deterministic one-tape Markov acceptor. It uses the score's
existing checked identity for the original named topology. The provider
requires that exact authored identity and reads the original nontrainable
weight tensors to produce a detached quotient. It checks the quotient's
representability in the input dtype and applies separate minimization
and observation limits.
The serialized request stores the original topology and limits. The
caller supplies the weights for each new binding. An ordinary
`FixedRelationRequest` continues to score the same declaration with its
live weights.

::: quivers.qiec.relations.FixedRelationRequest

::: quivers.qiec.relations.FixedRelationCostRequest

::: quivers.qiec.relations.FixedMinimizedRelationRequest

::: quivers.qiec.relation_runtime.FixedRelationRuntimeProvider

## Compact backoff relation requests

A `FixedBackoffRelationRequest` scores a word against an authored
deterministic one-tape Markov or LogProb topology with explicit failure
links. The checked foreign handler name identifies the original
topology, ordered failure links and their weight identities, and the
observation. State and work limits are invocation settings. An ordinary
pair-weight request cannot bind this handler.

For a missing symbol, a chain of failures leaves the symbol unconsumed
until the first ordinary match. Its weight is the ordered product of
the failure weights followed by the matching ordinary weight. This is
the same transition a full deterministic expansion would supply for
that state and symbol. Each ordinary transition is present structurally,
so a zero ordinary coefficient takes precedence over any fallback. At
the end of the word, only the current state's final mark can contribute.
The compact plan follows this sequence for the observed word and keeps
the original tensors, including shared and failure-only identities.

In QVR, a named computation can install a `foreign` `Compute[Unit, A]`
handler using the request's `handler_name`, with `apply resumes 1`.
Here `A` is `Real` or `LogWeight` according to the request. A native run
supplies `FixedRelationRuntimeProvider` and the original weight tensors.
The budget counts visited state configurations and ordinary symbol
probes after validation of the sparse graph. The exhaustive alphabet
expansion is only an equivalence description.

::: quivers.qiec.relation_backoff.FixedBackoffRelationRequest

## Capped ranked relation requests

A `FixedRankedOTRelationRequest` supplies one ordered violation profile
per original generator arc and one finite cap per rank. For each input,
the compiler retains all original accepting paths whose capped profiles
are lexicographically minimal. The pair score then sums the weights of
the retained paths at the observed output. Tied parallel paths remain
distinct, and the weights may change without altering their ranked
preferences. The result is an unnormalized pair mass. A conditional
output distribution would additionally require its own output
normalizer and positive finite mass.

A nonnegative cap is a part of the model. For instance, a cap of one
makes a rank binary, and a cap of zero ignores it. The competitor
tracker charges its work separately from the subsequent fixed-word
query. Its intermediate subset count can grow exponentially. Exhausted
limits raise without returning a truncated ranked result. This request
uses the finite capped ranking of the sparse calculus. [Eisner (2002)](https://www.cs.jhu.edu/~jason/papers/eisner.acl02-ot.pdf).

A named QVR computation installs the request's `foreign`
`Compute[Unit, A]` handler as with compact backoff. Its checked handler
identity includes the original typed generator, original occurrence
profiles, caps, and the observed pair. The reference provider validates
that identity and binds the original weight tensors. The Stan, BUGS and
JAGS projections name the foreign handler when refusing this provider.

::: quivers.qiec.relation_ranked.FixedRankedOTRelationRequest

## Packed contextual edit requests

A `FixedContextualEditRequest` records paired observations in batch
order, their nominal alphabets, the consecutive insertion bound,
construction limits, and a stable `AttachmentId` for the channel. Its
result is a `Tensor[LogWeight]` with one entry per observed pair. Each
entry is the conditional log probability of the observed output and
sums over the compatible alignments. The local normalization still
includes every legal edit at each input position and insertion count.

A native `ContextualEditRuntimeProvider` binds the original channel
and builds its packed plan once. The scorer reads the complete fixed
input batch on every execution, so later optimizer steps reach the
same original parameters. The request can travel on the wire, and the
caller supplies the original network as a runtime attachment. A QVR
computation can install the request's named foreign `Compute`
handler, as shown in the [contextual edit example](../../examples/contextual-edit-channel.md).
The Stan, BUGS and JAGS projections report that foreign handler when
refusing the computation.

::: quivers.qiec.contextual_edit_runtime.FixedContextualEditRequest

::: quivers.qiec.contextual_edit_runtime.ContextualEditRuntimeProvider

## Sparse relation composition requests

A `ComposedRelationSpec` records the two original operands of a
binary composition. Each has one input and one output tape. The
contract checks the nominal shared tape and the common scalar
algebra. The operands' parameter identities are resolved from one
runtime map, so tied weights retain their original gradient owner.

A `FixedComposedRelationRequest` scores the pair of outer words.
With lazy execution it builds only the reachable fixed-word product.
With materialized execution it builds a reusable sparse product
and can close an explicitly permitted productive component. The
three-state epsilon filter gives each compatible pair of original
paths exactly one product contribution. The selected execution
mode and its limits are part of the wire record.

The existing pair-weight provider serves a composed request for
an unbound source score. A score bound to a specific authored
primitive relation checks that declaration's topology instead.

::: quivers.qiec.relation_composition.ComposedRelationSpec

::: quivers.qiec.relation_composition.FixedComposedRelationRequest

## Typed sparse path posteriors

A `FixedRelationPathSampleRequest` binds a fixed word tuple and a
reproducible CPU generator to the original Markov or LogProb weights.
The result is a checked `Sampleable` whose values contain the initial
state, final state, and a fixed-size vector of original arc IDs.
Trailing positions use `-1` padding. Parallel arcs remain distinct
even when their weights share one parameter.

The productive fixed-word graph must be acyclic. Before binding a
law, a reverse topological pass verifies that every structurally
accepting path fits the trace capacity, including paths whose current
weights are zero. This preserves the full support after changes to the
weights. Scoring validates each original trace and subtracts the
observed log partition from its log weight. Draws use the existing
float64 conditional suffix sampler. Multiplying the successive
choice probabilities cancels the intermediate suffix weights and
leaves the same normalized path probability.

Pass the law to a checked `Random.sample` operation or use it in
`log_prob` and `log_mass`. A trace that violates the path topology,
does not match the words, or has zero weight has log probability
negative infinity. The law reads live parameter values and scoring
differentiates on the current positive support. The reference machine
can replay a trace by site name with the same checked result type.

::: quivers.qiec.relation_path_law.FixedRelationPathSampleRequest

::: quivers.qiec.relation_path_law.RelationPathLaw

## Checked sparse relation transformations

A `UnaryRelationTransformSpec` stores the original typed topology,
parameter identities, and one supported operation. Projection sums
over erased tapes. Conditioning fixes one tape word. Inversion
exchanges the input and output groups, trim retains only structurally
productive records, and closure permits all finite repetitions.

A `BinaryRelationTransformSpec` keeps both original operands in order.
Union adds their weights. Concatenation sums over the possible splits
of each observed word. A join synchronizes one common typed tape and
can retain it once or sum over it. Intersection is the shared-tape
case for two languages. These products use the bounded epsilon filter
to count each compatible pair of paths once.

A `FixedTransformedRelationRequest` provides the retained word tuple
and explicit query limits. Conditioning also uses those limits for
its intermediate topology. A binary sparse product uses the same
limits for its own construction. The reference provider builds and reuses
the transformed topology and query. Each evaluation reads the
original live parameters. A productive cycle needs a declared
closure bound up to the limit supported by the scalar algebra.

This request supports a linear Compute pair weight, including an
unbound QVR `relation_score` declaration. Native targets identify
the foreign handler as a required runtime capability.

::: quivers.qiec.relation_transforms.UnaryRelationTransformSpec

::: quivers.qiec.relation_transforms.BinaryRelationTransformSpec

A `RelationCascadeSpec` extends this recipe to a dependency-ordered
table of named declarations. A reference to an earlier step can occur
in either operand position or in both. Sharing construction and weight
reads preserves the path multiplicity of each operation. Cycles between
declarations, ill-typed layouts, and unused steps are refused before
binding. The provider evaluates the table in order and keeps a fresh
weight snapshot for each call, so parameter mutations and gradients use
the original owners.

In a cascade, `max_states` limits the state count of each intermediate
topology. `max_work` limits the number of distinct steps, the number of
retained topology records at each step, and the native sparse construction
work at each conditioned or product step. The storage count includes tapes,
arcs, per-tape labels, initial and final marks, and weight slots.
Linear operations that expand the topology check their output size
before allocation. Native sparse products can temporarily produce
labels for up to the allowed arcs at the inferred tape width before
checking the retained record count. The final fixed observation and
any productive component closure have their own query limits.
For $s$ steps and limits $S$ and $W$, the retained intermediate tables
have at most $sS$ states and $sW$ records. Each evaluation reads each
distinct weight source once. Both references to a shared operand still
contribute to the enclosing operation's path sum and gradient.

::: quivers.qiec.relation_cascades.RelationCascadeStep

::: quivers.qiec.relation_cascades.RelationCascadeSpec

::: quivers.qiec.relation_transforms.FixedTransformedRelationRequest

## Checked log feature moments

A `FixedRelationFeatureObservation` records fixed words, the original
typed relation, a positive feature extent $d$, and optional boundary
features. Each table has one row per original occurrence and $d$
columns. Entries contain finite log increments or `None` for log zero.
The codec preserves `None` as JSON null. The provider restores it as
negative infinity on the original weight dtype and device.

The observation's `handler_name` identifies its exact topology, words,
and ordered features. A checked foreign handler at that name declares
`Compute[Unit, Tensor[LogWeight]([d+1])]`. Its result places the log
total first and the unnormalized log moments in feature order. Use
`FixedRelationRuntimeProvider` with the request and its `handler_name`
to attach the evaluator. The provider reads the original parameter
objects each time the handler runs. An observation with no accepting
mass returns log zero in every position.

`FixedRelationFeatureMomentRequest` sets independent state, examined-arc,
and feature-cell limits. The feature-cell count is $(d+1)(N+V+E)$,
where $N$ counts original arc and boundary occurrences, and $V$ and $E$
count productive states and arcs in the observation. Budget changes keep
the observation's handler name. A productive epsilon cycle requires a
different carrier capability. Native targets that require foreign calls
to be discharged report the observation's handler as the obstacle.

::: quivers.qiec.relation_feature_moments.FixedRelationFeatureObservation

::: quivers.qiec.relation_feature_moments.FixedRelationFeatureMomentRequest

`FixedCyclicRelationFeatureMomentRequest` adds an explicit per-component
closure order to the same observation and chart budgets. It uses the
same handler identity and returns the same unnormalized log-space
observable. Its marked-occurrence calculation combines certified scalar
prefix and suffix closures, including repeated visits. The component
order includes auxiliary states needed to preserve parallel occurrences.
Insufficient order, divergence, and an uncertifiable closure are reported
by the scalar evaluator.

::: quivers.qiec.relation_feature_moments.FixedCyclicRelationFeatureMomentRequest

`FixedComposedArcFeatureMomentRequest` uses a materialized composition
and one log feature row per original arc of each operand. A product arc
keeps the supplying operand and occurrence ID, including when two
arcs share a weight. The epsilon filter contributes one accepting product
path per compatible original path pair. A matched step has one edge for
each primitive factor, so their two feature rows both enter the moment
even when the shared word is erased. Boundary marks still contribute
their weights. The request assigns increments only to operand arcs.

The feature-cell count is the feature extent plus one, times the sum of
operand arc rows, materialized state, arc and boundary records, and
active observation states and arcs. Limits on the product and any
productive scalar closure also apply. The handler binds the ordered
operands, outer observation, and original feature rows. Its result is
the log total followed by unnormalized log feature moments.

::: quivers.qiec.relation_feature_moments.FixedComposedArcFeatureMomentRequest

## String factor graphs

`StringFactorGraphSpec` records the nominal alphabet and optional
observation of each variable, the sparse relations by name, and the
ordered variable scope of each factor occurrence. A factor's scope must
match the corresponding relation's tape order and types. Bind the record
to a single parameter store to share a relation across repeated factor
occurrences and to share stable weight identities across distinct relations.
An entry can hold a primitive topology or a named `RelationCascadeSpec`.
The graph checks each factor against the resulting cascade tape layout.
It constructs each referenced derived relation once and shares that
machine across all occurrences of its name. Tree, pruned, MAP, and EP
requests use their state and work bounds for each cascade intermediate.
Direct graph binding requires both limits when a used factor is derived.

::: quivers.qiec.string_graphs.StringFactorGraphSpec

`TreeStringGraphRequest` stores a partition or a normalized word
query with explicit construction limits. The checked handler includes
the operation in its stable identity. A runtime provider compiles the
query once and uses the live graph parameters on every invocation.

::: quivers.qiec.string_graphs.TreeStringGraphRequest

::: quivers.qiec.string_graph_runtime.tree_string_graph_handler

::: quivers.qiec.string_graph_runtime.TreeStringGraphRuntimeProvider

QVR can name these queries as ordinary scalar computations.

<!-- compile: qiec -->
```qvr
string_graph_score log_evidence : LogWeight [operation=tree_partition]
string_graph_score reading : LogWeight [operation=tree_log_prob]
```

Bind each declaration with `TreeStringGraphRuntimeProvider.for_score`.
A partition uses `Real` for a Markov graph or `LogWeight` for a LogProb
graph. A word query has type `LogWeight` and sums out the other
variables before normalizing. Attachments serve these computations on
the checked reference machine, with foreign-handler diagnostics for
targets that cannot lower them.

### Checked pruned string beliefs

`PrunedStringGraphRequest` uses that same graph with a
`PrunedStringBPConfig` and selects one variable to report. The output
coordinates are the variable's declared candidate words in order. An
observed variable has one coordinate for its fixed word. The checked
`pruned_string_graph_handler` binds the graph, the ordered candidate pools,
and the variable. The same handler can run with a different beam,
damping schedule, or resource limits. The provider retains the compiled
factor queries and reads the original weights on every invocation.

The foreign computation returns a checked product with these fields
in order.

1. Status and diagnostic text. Status is `converged`, `limit` for exhausted
   sweeps, `empty_support` for a support conflict, or `resource_limit` for
   a factor table or observation construction limit.
2. Completed sweep count, message residual, and a Boolean truncation flag.
   The residual is $-1$ when undefined.
3. The selected variable's log belief tensor. An unavailable belief has
   log zero in every coordinate.
4. A Boolean indicating whether a complete proposal table is available,
   and the table's log masses. Each row gives full, retained, and omitted
   mass before renormalization. Rows follow the original factor and tape
   order. Unavailable rows contain log zero.

Convergence concerns the pruned update. The belief remains an approximation
over the declared finite pool. Native targets that require foreign calls
to be discharged report the checked pruned handler as an obstacle.

::: quivers.qiec.pruned_string_graph_runtime.PrunedStringGraphRequest

::: quivers.qiec.pruned_string_graph_runtime.PrunedStringGraphRuntimeProvider

::: quivers.qiec.pruned_string_graph_runtime.pruned_string_graph_handler

## String solver configurations

Encode a `StringFactorGraphSpec` and its solver settings separately. A
`JointAlignmentMapConfig` reconstructs the exact acyclic local decoder
with the saved iteration, feature, state, and work limits. Its rational
step is stored as two integers. Local maximization adds the original path
scores, including boundary marks. The dual bound applies to the best
joint alignment and words, and a certificate requires agreement on the
complete words. An iteration limit may return a feasible assignment and
bound without a certificate. General joint string MAP can be undecidable
for cyclic multi-tape factors, as the reduction from Post's correspondence
problem shows. See §2.5 of [Peng, Cotterell, and Eisner (2015)](https://aclanthology.org/D15-1108/).

`JointAlignmentMapGapConfig` pairs the ordinary solver settings with
one or both rational gap targets. The executor checks them against the
feasible score and dual bound computed from the original live model.
An unsupported relative guarantee or a missing feasible assignment
cannot satisfy the target. A `gap` stop reports that the objective
bound was established. The `optimal` status and agreement certificate
retain their stricter condition. The original settings record keeps
its wire representation.

A `RegularStringEPConfig` uses the same factor declaration for approximate
sum-product. Its settings record finite regular reference languages,
ordered candidate n-grams, penalty, synchronous scheduling limits, and
projection tolerances. Reference weights enter the model for this run.
A reference that gives unit weight to each supported word only restricts
support. The result distinguishes local projection failure from an outer
sweep limit. Relation resource failures are reported by the underlying
operation. Message stabilization alone does not certify exact marginals.

Both solvers bind the original weight identities at invocation. The wire
records carry neither a previous fitted belief nor an optimum certificate.

A `PrunedStringBPConfig` records ordered finite word supports, a
positive beam width, damping, and construction and sweep limits. Bind
it to the same graph specification used for exact tree inference. The
executor derives convergence, support failures, and the pre-normalization
message masses from the live weights. These observations belong to the
returned approximate result.

::: quivers.qiec.string_solvers.JointAlignmentMapConfig

::: quivers.qiec.string_solvers.JointAlignmentMapGapConfig

::: quivers.qiec.string_solvers.RegularStringEPConfig

::: quivers.qiec.string_solvers.PrunedStringBPConfig
