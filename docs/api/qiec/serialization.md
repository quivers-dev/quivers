# Serialization

`dumps` and `loads` encode a stable [Quivers Indexed Effect Core](index.md) (QIEC) graph as JSON text and decode it again, so that a module
can be stored and rebuilt. A decoded module is checked again, and running it needs the
[runtime providers](../../reference/qvr/execution-and-tooling.md#runtime-providers) that its
[entry computations](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary)
require. The [QIEC developer note](../../developer/qiec.md#stable-serialization) states the format.

::: quivers.qiec.serialization
