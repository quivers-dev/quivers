# Family Registry

A family specification records parameter names, support, output representation,
and implementation choices for constructors and compiler code. When extending
the registry, distinguish [batch and event shapes](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape)
[@pyroTensorShapes]: batch axes index distributions, while event axes belong
to one jointly scored outcome. QVR expresses these roles through [event rank
and axis declarations](../../guides/continuous-families.md#event-rank-and-axis-role-syntax).
For modeling with an existing family, use its [public constructor](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families).

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
