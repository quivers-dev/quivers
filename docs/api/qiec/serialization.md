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


## Fixed-word relation requests

A `FixedRelationRequest` pairs a sparse relation specification with
observed words in tape order, one of two scoring operations, and
explicit query limits. `pair_weight` returns the scalar weight of
all accepting paths at those words. `conditional_log_prob` uses the
declared input-tape split and returns the log probability of the
output under the closure-checked conditional normalizer.

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

::: quivers.qiec.relations.FixedRelationRequest

::: quivers.qiec.relation_runtime.FixedRelationRuntimeProvider

## String factor graphs

`StringFactorGraphSpec` records the nominal alphabet and optional
observation of each variable, the sparse relations by name, and the
ordered variable scope of each factor occurrence. A factor's scope must
match the corresponding relation's tape order and types. Bind the record
to a single parameter store to share a relation across repeated factor
occurrences and to share stable weight identities across distinct relations.

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
