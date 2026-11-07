# Continuous Spaces

A [space specifies admissible values](../../guides/continuous-spaces.md#continuousspace-hierarchy) and the shape of one value. A positive scalar can
represent a standard deviation; a simplex contains nonnegative vectors whose entries sum
to one. Repeated observations add [batch dimensions](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape) [@pyroTensorShapes] around this value shape.

Typed carriers for continuous values, including Euclidean, simplex,
positive, matrix, sphere, ball, and product spaces.

::: quivers.continuous.spaces
