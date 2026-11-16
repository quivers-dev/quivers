# Families

The distribution family registry fixes, for each family, its
[parameters](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families), its
support, and its density, which every runtime and transpiler implements in its own
[parameterization](../../semantics/transpile-correctness/index.md#4-distribution-parameterization).
The [support table](../../transpile-support.md) lists which families each target accepts.

`FamilyShapeSignature` binds local Nat indices and relates each parameter's
intrinsic dimensions to the sample event shape. Leading plate dimensions
remain separate. Source elaboration and kernel checking instantiate this
same contract; the concrete-dimension helper uses the shared exact Nat solver.
Legacy event-dimension metadata is translated into this representation once.
It does not introduce another checking path.

For instance, the additive log-ratio normal binds a coordinate extent `d`,
takes location shape `[d]` and Cholesky shape `[d, d]`, and samples a
simplex of shape `[nat_add(d, 1)]`. Structural dimensions can also relate
parameters when the sample is scalar. `FamilyParameter.scalar_broadcast`
distinguishes ordinary plated broadcasting from a structural table that
requires its own dimensions.

::: quivers.qiec.families

::: quivers.qiec.family_shapes
