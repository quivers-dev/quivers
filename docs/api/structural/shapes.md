# Shapes

These factories give ready-made signatures, encoders, and decoders for the three common
[shapes](../../guides/structural-compression-signatures-and-encoders.md#stdlib-shapes): sequences,
trees, and graphs. Each builds on the generic `Encoder` and `Decoder`, so the objects it returns can
be inspected, composed, and extended with new constructors or fields like any hand-built one. A
shape is chosen to match the [relations in the data](../../tutorials/qvr/12-structural-autoencoders.md#2-construct-typed-host-data).

::: quivers.structural.shapes.seq

::: quivers.structural.shapes.tree

::: quivers.structural.shapes.graph
