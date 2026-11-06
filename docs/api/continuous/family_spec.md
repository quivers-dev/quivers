# Family Registry

A family specification records parameter names, support, shapes, and implementation
choices for constructors and compiler code. Extend this registry when adding a
distribution family. For modeling with an existing family, use its public constructor.

Pyro's [batch and event shape explanation](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape) supplies concrete scalar and multivariate examples [@pyroTensorShapes].

See [event rank and axis roles](../../guides/continuous-families.md#event-rank-and-axis-role-syntax) and
[constructing conditional families in Python](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families).

Every distribution family is described once by a
[`FamilySpec`](#quivers.continuous.family_spec.FamilySpec) and registered
in [`FAMILY_REGISTRY`](#quivers.continuous.family_spec.FAMILY_REGISTRY).
The inline builders and conditional families dispatch on the registered
specification, so registering a family through
[`register_family`](#quivers.continuous.family_spec.register_family)
(and refreshing the inline tables with
[`reload_inline_registry`](inline.md#quivers.continuous.inline.reload_inline_registry))
makes it available to programs.

::: quivers.continuous.family_spec
