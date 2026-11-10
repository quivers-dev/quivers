# Plates

A [plate repeats a draw or observation over an index set](../../guides/dsl-programs-and-lets.md#indexed-bind-plate), such as one response per
participant. The repeated values are [conditionally independent](https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate) [@pyroTensorShapes] given the parameters
supplied to the plate. They can still be dependent after shared uncertain parameters are
integrated out. [Plate dimensions](../../reference/qvr/probabilistic-programs.md#step-reference) describe repeated draws; event dimensions describe the
components of one draw.

Indexed draws and observations. A
[`PlateDraw`](#quivers.continuous.plate.PlateDraw) draws one value per
element of a finite set, as `sample v : A <- F(...)` does in source; a
[`VectorisedObserve`](#quivers.continuous.plate.VectorisedObserve)
scores a column of responses at once, as an indexed `observe` does. The
marginalization helpers integrate a finite latent out of a plate of
scores.

::: quivers.continuous.plate
