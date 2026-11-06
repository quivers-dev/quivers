# Structural runtime

This bridge connects checked operations on structured values to registered host
implementations of encoders, decoders, and losses. Execution requires a provider for
each requested operation. A transpilation target must also support those operations to
run the translated computation.

See [generated structural operations](../../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders) and
[calling and differentiating a structural entry](../../tutorials/qvr/12-structural-autoencoders.md#4-differentiate-through-the-entry-boundary).

Host-backed execution of checked encoder, decoder, and structural-loss
computations.

::: quivers.qiec.structural_runtime
