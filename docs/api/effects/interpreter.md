# Interpreter

The interpreter executes program operations under the active handlers [@plotkin2009handlers; @bauer2015programming]. A sample request
receives a value from its handler; an [observation contributes its score](../../guides/inference-foundations.md#conditioning-on-observations) according to the
handler’s interpretation. Inference procedures can install handlers that replay,
condition, or otherwise control [these executions](../../tutorials/python/05-variational-inference.md#tracing-the-model).

Running a program on the reference machine under a handler stack.

::: quivers.effects.interpreter
