# The program-shape protocol

[`quivers.dsl.program_theory`](../api/dsl/program_theory.md) extracts from a compiled QVR module a
[panproto](https://github.com/panproto/panproto) schema, a typed graph that describes the module's
shape without running it. panproto is the schema and migration toolkit that quivers uses to version
its grammar. In the extracted graph an object is a vertex, a morphism has edges to the vertices of its
domain and codomain, and a call joins the caller to the callee. A *protocol* fixes which kinds of
vertices, edges, and constraints such a graph may contain, and the library calls this one the
program-shape protocol.

The schema records shape, not values. Two transition tables with the same domain and codomain and
different learned weights give the same schema, though they assign different probabilities.

## 1. `QVR_PROGRAM_PROTOCOL`

`QVR_PROGRAM_PROTOCOL` builds on panproto's shape-graph theories `ThBratSchema` and `ThBratInstance`,
with kinds, constraints, and edge rules for QVR. Its vertices fall into these groups:

| Group | Kinds |
|---|---|
| Finite sets | `finset`, `product_set`, `coproduct_set`, `free_monoid`, `empty_set`, `enum_set`, `free_residuated` |
| Spaces | `euclidean`, `simplex`, `positive_reals`, `product_space` |
| Declarations | `object_decl`, `space_decl`, a kind for each morphism role, `output_decl`, `schema_decl` |
| QIEC declarations | `effect_decl`, `operation`, `instance_decl`, `handler_decl`, `handler_clause`, `computation_decl`, `computation_parameter` |
| Entry points | `program_entry`, `program_parameter`, `program_site` |
| Root | `program` |

Edges record membership in the module, the binding of declarations to values, the components of
products, generators, the domains and codomains of morphisms, and the selected output. Constraints
record names, cardinalities, dimensions, families, roles, and bounds.

The QIEC group mirrors the checked [`QiecModule`](../api/qiec/module.md) that the compiler keeps as
[`Compiler.qiec_module`](../api/dsl/compiler.md), with one vertex for each effect interface and each
of its operations, each effect instance, each handler and each of its clauses, and each computation
and each of its parameters. Edges record the interface an instance applies (`instance_of`), the
interface a handler handles (`handles`), the instances a computation's row names (`row`), and what a
body does: the computations it calls (`calls`), the instances it sends requests to (`performs`), and
the handlers it installs (`installs`).

A `program` elaborates to a computation and a `program_entry` vertex, whose `body` edge names the
computation. Its `parameter` children carry the role of each parameter (`domain`, `scalar`, `data`,
`observation`, `fibration` for an index map, `weight`, `bias`, `table`, or `kernel-input`), its
`site` children the kind, family, and plate axes of each probabilistic step, and its `random` and
`score` edges the built-in instances that its draws and scores address. When the elaboration of a
module's programs fails, the root carries the diagnostic as a `gap` constraint.

A program and a computation that calls it are vertices of one kind, joined by the same `calls` edge,
so a module that mixes them has one call graph.

## 2. Extraction

`extract_program_schema(compiler)` reads a compiled `Compiler` and returns a `panproto.Schema` with the
protocol name `qvr_program`. `extract_deduction_schema(compiler)` does the same for deductions.

The extraction caches emitted runtime objects by Python identity, so that two equal components that
occur separately become two vertices. Otherwise panproto, whose edges form a set, would merge their
edges.

Extraction is deterministic: compiling one module twice gives schemas with the same structure, and every
extracted schema validates against `QVR_PROGRAM_PROTOCOL`.

## 3. What a schema records {#3-what-a-schema-records}

The schema records the static shape of the bindings in the [environment](setting.md#5-environments),
selected metadata of the declarations, and the call, request, and handler structure of each checked
body. Learned tensors, observations, optimizer state, distribution objects, and executable bodies are
not in it: they belong to the compiled model and its serialization.

`panproto.diff_schemas(a, b)` finds structural changes between two schemas. Changing a Normal site to a
Gamma site changes the family recorded in the schema. Changing only the learned location of a Normal
site leaves the schema the same, while it changes the model's predictions. Equal schemas thus show equal
shape, not equal denotations.

## 4. Migration

A schema change does not say how to change a model. Source files are migrated with the grammar
migration tooling, which updates declarations and their references
([what `qvr migrate` does](../guides/migration.md#what-qvr-migrate-does)). Learned parameters need a
stated map from old tensors to new declarations, including any change of axis order or dimension
([what survives migration](../guides/migration.md#what-survives-migration)).
