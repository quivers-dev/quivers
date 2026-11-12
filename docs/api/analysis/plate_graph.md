# Plate graphs

A [plate graph](../../guides/repl-and-lsp.md#program-exploration) shows which sites repeat over which
index sets and how they depend on one another, as in
[plate notation](https://en.wikipedia.org/wiki/Plate_notation) [@koller2009probabilistic]. It shows,
for instance, whether an item effect is shared across observations or drawn again for each, which
together with the model's [indexing](../../guides/dsl-programs-and-lets.md#indexed-bind-plate) fixes its
conditional independence assumptions.

::: quivers.analysis.plate_graph
