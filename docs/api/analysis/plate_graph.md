# Plate graphs

A plate graph summarizes which variables repeat over index sets and how sites depend on
one another. It can reveal whether an item effect is shared or redrawn for every
observation. Inspect the graph together with the model’s grouping and lookup expressions
to understand its conditional independence assumptions.

Pyro illustrates the independence assumptions in [its plate tutorial](https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate) [@pyroTensorShapes].

See [inspecting a program's plate graph](../../guides/repl-and-lsp.md#program-exploration) and
[the repeated draws shown in a plate graph](../../guides/dsl-programs-and-lets.md#indexed-bind-plate).

::: quivers.analysis.plate_graph
