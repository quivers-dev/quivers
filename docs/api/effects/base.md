# Handler base

The handler stack records which handlers are active and the context of one execution.
Think of a sample request passing through the applicable handlers before receiving its
value. This module supplies the infrastructure for installing those handlers; use a
concrete handler when you only need conditioning, tracing, or another existing behavior.

See [requests at sample sites](../../guides/inference-foundations.md#trace-and-sample-sites) and
[inspecting a concrete trace](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

The handler stack, the run context, and the installation record.

::: quivers.effects.base
