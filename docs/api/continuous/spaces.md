# Continuous Spaces

A space specifies admissible values and the shape of one value. A positive scalar can
represent a standard deviation; a simplex contains nonnegative vectors whose entries sum
to one. Repeated observations add batch dimensions around this value shape.

Pyro's [batch and event shape explanation](https://pyro.ai/examples/tensor_shapes.html#Distributions-shapes:-batch_shape-and-event_shape) works through the shape of samples and log probabilities [@pyroTensorShapes].

See [supports and continuous spaces](../../guides/continuous-spaces.md#continuousspace-hierarchy) and
[the distinction between mass, density, and a sampled value](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample).

The [PyTorch distribution reference](https://docs.pytorch.org/docs/stable/distributions.html) explains support constraints and distinguishes sample, batch, and event shapes.

Typed carriers for continuous values, including Euclidean, simplex,
positive, matrix, sphere, ball, and product spaces.

::: quivers.continuous.spaces
