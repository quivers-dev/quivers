# Plates

A [plate](https://en.wikipedia.org/wiki/Plate_notation) repeats a draw or observation over an index
set, such as one response per participant, as conditionally independent copies given the
parameters passed to it [@koller2009probabilistic; @pyroTensorShapes]
([indexed draws](../../guides/dsl-programs-and-lets.md#indexed-bind-plate)). The copies may still be
dependent once shared uncertain parameters are integrated out. A
[`PlateDraw`](#quivers.continuous.plate.PlateDraw) draws one value per element of a finite set, as
`sample v : A <- F(...)` does in source, and a
[`VectorisedObserve`](#quivers.continuous.plate.VectorisedObserve) scores a column of responses at
once, as an indexed `observe` does. Plate axes index repeated draws, while
[event axes](../../reference/qvr/probabilistic-programs.md#step-reference) index the components of
one draw. The marginalization helpers sum a finite latent out of a plate of scores.

::: quivers.continuous.plate
