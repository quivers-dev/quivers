# Tensor Operations

These helpers apply fuzzy reductions directly to tensors along specified axes. Noisy-OR
combines scores 0.2 and 0.3 as 1 − (1 − 0.2)(1 − 0.3) = 0.44. Probability
marginalization uses addition, which would give 0.5 for those two contributions; use the
reduction appropriate to the model.

Pyro's [distribution shape examples](https://pyro.ai/examples/tensor_shapes.html#Examples) distinguish the axes retained by `log_prob` from those reduced within an event [@pyroTensorShapes].

See [contracting several tensors](../../guides/transformations.md#operadic-contractions) and
[a worked bilinear contraction](../../guides/dsl-contractions.md#worked-example-bilinear-scoring).

For tensor shapes and broadcasting, consult the [PyTorch broadcasting notes](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).

Noisy-OR contraction and reduction, noisy-AND reduction, and
componentwise tensor lifting.

::: quivers.core.tensor_ops
