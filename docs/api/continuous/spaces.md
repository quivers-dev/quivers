# Continuous Spaces

A [space](../../guides/continuous-spaces.md#continuousspace-hierarchy) gives the admissible values
of a continuous variable and the shape of one value. The module gives Euclidean, simplex, positive,
Cholesky factor, covariance, correlation, orthogonal, Stiefel, sphere, ball, triangular, diagonal,
and product spaces. A positive scalar can hold a standard deviation, and a simplex holds
nonnegative vectors whose entries sum to one. Repeated observations add
[batch dimensions](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape)
[@pyroTensorShapes] around this value shape.

::: quivers.continuous.spaces
