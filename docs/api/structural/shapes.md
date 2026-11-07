# Shapes

These factories provide common [sequence, tree, and graph structures](../../guides/structural-compression-signatures-and-encoders.md#stdlib-shapes) with encoder or
decoder components. Choose a shape that matches the [relationships in the data](../../tutorials/qvr/12-structural-autoencoders.md#2-construct-typed-host-data), then
inspect its generated signature before extending it with new constructors or fields.

Ready-made signatures, encoders, and decoders for the three principal
compressible shapes: sequences, trees, and graphs. Each factory builds
on the generic `Encoder` and `Decoder` runtimes, so the objects it
returns can be inspected and composed like any hand-built one.

::: quivers.structural.shapes.seq

::: quivers.structural.shapes.tree

::: quivers.structural.shapes.graph
