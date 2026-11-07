# Chain shapes

[Shape inspection](../../guides/analysis-fitting-and-diagnostics.md#chainshape) reports the inputs and outputs of each step in a composition. For a
long chain, locate the first step whose input dimensions disagree with the preceding
output. This narrows a [tensor shape error](../../tutorials/python/09-debugging.md#inspecting-a-compiled-program) to the expression that introduced it.

::: quivers.analysis.chain_shape
