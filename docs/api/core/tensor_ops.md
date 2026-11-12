# Tensor Operations

These functions apply fuzzy reductions to tensors along given axes. Noisy-OR combines independent
probabilities $p_i$ as $1 - \prod_i (1 - p_i)$ [@pearl1988probabilistic], so 0.2 and 0.3 combine to
0.44 where a sum gives 0.5. The module gives noisy-OR contraction and reduction, noisy-AND
reduction, and componentwise lifting of tensors. Composition under the default `Algebra`, the
product t-norm with noisy-OR, matches `noisy_or_contract`.

::: quivers.core.tensor_ops
