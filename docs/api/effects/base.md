# Handler base

An `EffectHandler` describes a handler of a program's `random` or `score` instance
[@plotkin2009handlers; @bauer2015programming]. Entering it as a context manager pushes it on a
thread-local handler stack, and a run installs every handler on the stack around the program, so
that each [sample request](../../guides/inference-foundations.md#trace-and-sample-sites) passes
through the active handlers before it receives a value. The module gives the handler stack, the
run context, and the installation record. Conditioning,
[tracing](../../tutorials/python/05-variational-inference.md#tracing-the-model), and the other common
behaviors are concrete [handlers](handlers.md).

::: quivers.effects.base
