# Chain shapes

Shape inspection reports the inputs and outputs of each step in a composition. For a
long chain, locate the first step whose input dimensions disagree with the preceding
output. This narrows a tensor shape error to the expression that introduced it.

See [inspecting intermediate shapes](../../guides/analysis-fitting-and-diagnostics.md#chainshape) and
[inspecting the stages of a compiled program](../../tutorials/python/09-debugging.md#inspecting-a-compiled-program).

::: quivers.analysis.chain_shape
