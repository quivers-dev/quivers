# Structural runtime

This bridge connects checked [operations on structured values](../../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders) to registered host
implementations of [encoders, decoders, and losses](../../tutorials/qvr/12-structural-autoencoders.md#4-differentiate-through-the-entry-boundary). Execution requires a provider for
each requested operation. A transpilation target must also support those operations to
run the translated computation.

Host-backed execution of checked encoder, decoder, and structural-loss
computations.

::: quivers.qiec.structural_runtime
