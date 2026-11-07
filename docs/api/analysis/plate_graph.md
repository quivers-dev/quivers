# Plate graphs

A [plate graph](../../guides/repl-and-lsp.md#program-exploration) summarizes which variables repeat over index sets and how sites depend on
one another. It can reveal whether an item effect is shared or redrawn for every
observation. Inspect the graph together with the model’s [grouping and lookup expressions](../../guides/dsl-programs-and-lets.md#indexed-bind-plate)
to understand its [conditional independence assumptions](https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate) [@pyroTensorShapes].

::: quivers.analysis.plate_graph
