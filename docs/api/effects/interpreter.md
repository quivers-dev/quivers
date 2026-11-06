# Interpreter

The interpreter executes program operations under the active handlers. A sample request
receives a value from its handler; an observation contributes its score according to the
handler’s interpretation. Inference procedures can install handlers that replay,
condition, or otherwise control these executions.

See [executing draws and observations](../../guides/inference-foundations.md#conditioning-on-observations) and
[inspecting a concrete trace](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

Running a program on the reference machine under a handler stack.

::: quivers.effects.interpreter
