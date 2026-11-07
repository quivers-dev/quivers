# Serialization

Serialization stores a core graph’s declarations and references so it can be
reconstructed later. After decoding, check the module and [supply the runtime providers](../../reference/qvr/execution-and-tooling.md#runtime-providers)
required by its [entry computations](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary).

The canonical wire codec for stable kernel graphs.

::: quivers.qiec.serialization
