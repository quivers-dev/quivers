# Chain shapes

`ChainShape` records, for each step of a program, its source location, its `Algebra`, its depth in
the chain of stochastic steps, and the dimensions of its inputs and outputs
([ChainShape](../../guides/analysis-fitting-and-diagnostics.md#chainshape)). In a long chain, the first
step whose input dimensions disagree with the preceding output locates a
[tensor shape error](../../tutorials/python/09-debugging.md#inspecting-a-compiled-program).

::: quivers.analysis.chain_shape
