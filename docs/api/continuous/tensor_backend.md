# Tensor Distribution Backend

[`TensorBackend`](#quivers.continuous.tensor_backend.TensorBackend) samples and scores the
[runtime distributions](../qiec/distributions.md) of the
[reference machine](../../developer/qiec.md#reference-evaluator) by torch. It builds each family's
distribution as the compiled runtime builds the value of a `Distribution` term, so the two score a
family by the same code, and every density keeps its gradient.

::: quivers.continuous.tensor_backend
