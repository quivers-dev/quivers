# The Program-Shape Protocol

`quivers.dsl.program_theory` provides a schema-level view of a compiled QVR environment. The protocol specifies the vertices, edges, and constraints used to describe that environment.

Think of the extracted schema as an inspectable graph of a module. An object
is a vertex, a morphism's input and output are edges to object vertices, and
a call links the caller to the callee. A protocol specifies which kinds of
vertices, edges, and metadata this graph may contain. This supports
structural inspection without executing the model.

For instance, two transition tables may have identical input and output
objects but different learned weights. Their extracted shape can agree even
though they assign different probabilities. The schema records their common domain, codomain, and declaration structure.

## 1. `QVR_PROGRAM_PROTOCOL`

`QVR_PROGRAM_PROTOCOL` reuses panproto's `ThBratSchema` and `ThBratInstance` shape-graph theories with QVR-specific kinds, constraints, and edge rules.

The principal vertex groups are:

| Group | Representative kinds |
|---|---|
| Discrete values | `finset`, `product_set`, `coproduct_set`, `free_monoid`, `empty_set`, `enum_set`, `free_residuated` |
| Continuous values | `euclidean`, `simplex`, `positive_reals`, `product_space` |
| Declarations | `object_decl`, `space_decl`, morphism-role declarations, `output_decl`, `schema_decl` |
| Kernel declarations | `effect_decl`, `operation`, `instance_decl`, `handler_decl`, `handler_clause`, `computation_decl`, `computation_parameter` |
| Entry points | `program_entry`, `program_parameter`, `program_site` |
| Root | `program` |

Edges record membership in the module, declaration-to-value bindings, product components, generators, morphism domains and codomains, and the selected output. Constraints record metadata such as names, cardinalities, dimensions, families, roles, and bounds.

The kernel group mirrors the checked [`QiecModule`](../api/qiec/module.md) the compiler holds as [`Compiler.qiec_module`](../api/dsl/compiler.md): one vertex per effect interface and its operations, per lexical instance, per handler and its clauses, and per named computation with its parameters. Edges record which interface an instance applies (`instance_of`), which interface a handler handles (`handles`), which instances a computation's declared row names (`row`), and what a body does: the computations it calls (`calls`), the instances it performs requests on (`performs`), and the handlers it installs (`installs`). A `program` declaration elaborates to a computation plus a `program_entry` vertex whose `body` edge names it, whose `parameter` children carry each parameter's role (domain, scalar, data, observation, fibration, weight, bias, kernel-input), whose `site` children carry each probabilistic step's kind, family, and plate axes, and whose `random` and `score` edges name the canonical instances its samples and scores address. A module whose programs the elaboration cannot yet lower carries the diagnostic on the root as a `gap` constraint.

Because a program and a computation that calls it are vertices of the same kind joined by the same `calls` edge, a mixed module yields one call graph rather than a morphism graph beside a computation graph.

## 2. Extraction

`extract_program_schema(compiler)` reads a populated `Compiler` environment and returns a `panproto.Schema` with protocol name `qvr_program`. `extract_deduction_schema(compiler)` performs the analogous extraction for deduction structures.

The writer caches emitted runtime objects by Python identity. This avoids collapsing two equal-looking component occurrences into one vertex when repeated edges would otherwise be lost under panproto's edge-set semantics.

Extraction is deterministic for the cases covered by `tests/test_program_theory.py`, and every example schema in that test validates against `QVR_PROGRAM_PROTOCOL`.

## 3. What a schema records

[Semantic environments](setting.md#5-environments) describe the compiled bindings being inspected. The [typing implementation map](typing.md#9-implementation-correspondence) locates the checker structures that supply the extracted metadata.

The schema records static shape, selected declaration metadata, and the call, request, and handler structure of each checked body. Learned tensors, observations, optimizer state, distribution objects, and executable bodies belong to the compiled model and its serialization.

Use `panproto.diff_schemas(a, b)` to locate structural changes. Changing a Normal site to a Gamma site changes the extracted family metadata. Updating only the learned location of a Normal site preserves the schema while changing the model's predictions.

## 4. Migration scope

The migration guide distinguishes [the source transformations performed by `qvr migrate`](../guides/migration.md#what-qvr-migrate-does) from [state that survives migration](../guides/migration.md#what-survives-migration). Those concrete operations determine how an extracted structural change reaches a usable model.

Panproto schema operations can consume the extracted shape. To migrate `.qvr` source, use the grammar migration tooling so declaration changes are propagated to their references. To migrate learned parameters, specify how old tensors map to the new declarations, including any changes in axis order or dimension.

## 5. Evidence

`tests/test_program_theory.py` checks that:

1. every current example produces a validating `qvr_program` schema;
2. extracted schemas contain a `program` root;
3. selected object and output metadata are recorded;
4. structurally distinct examples produce a nonempty diff;
5. recompiling the same example produces schemas with the same recorded structure;
6. a module holding only a `program` yields its computation, entry point, sites, and canonical instances;
7. computations and programs calling one another form a single call graph;
8. changing one site's family changes the extracted schema.
