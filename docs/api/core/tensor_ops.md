# Tensor Operations

These helpers apply [fuzzy reductions directly to tensors](../../guides/transformations.md#operadic-contractions) along [specified axes](https://docs.pytorch.org/docs/stable/notes/broadcasting.html). Noisy-OR
combines scores 0.2 and 0.3 as 1 − (1 − 0.2)(1 − 0.3) = 0.44. Probability
marginalization uses addition, which would give 0.5 for those two contributions; use the
reduction appropriate to the model.

Noisy-OR contraction and reduction, noisy-AND reduction, and
componentwise tensor lifting.

::: quivers.core.tensor_ops
