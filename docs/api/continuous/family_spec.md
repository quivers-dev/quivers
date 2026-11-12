# Family Registry

Every distribution family is described once by a
[`FamilySpec`](#quivers.continuous.family_spec.FamilySpec), which records its parameter names,
support, output representation, and constructors, and is registered in
[`FAMILY_REGISTRY`](#quivers.continuous.family_spec.FAMILY_REGISTRY). The inline builders and
conditional families dispatch on the registered specification, so registering a family with
[`register_family`](#quivers.continuous.family_spec.register_family), and refreshing the inline
tables with [`reload_inline_registry`](inline.md#quivers.continuous.inline.reload_inline_registry),
makes it available to programs. A specification separates
[batch and event shapes](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape)
[@pyroTensorShapes]: batch axes index independent distributions, and event axes belong to one
jointly scored outcome. QVR states these roles with the
[`over` and `iid over` clauses](../../guides/continuous-families.md#event-rank-and-axis-role-syntax).

::: quivers.continuous.family_spec
