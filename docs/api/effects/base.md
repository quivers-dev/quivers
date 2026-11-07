# Handler base

The handler stack records which handlers are active and the context of one execution.
Think of a [sample request](../../guides/inference-foundations.md#trace-and-sample-sites) passing through the applicable handlers before receiving its
value [@plotkin2009handlers; @bauer2015programming]. This module supplies the infrastructure for installing those handlers; use a
concrete handler when you only need conditioning, [tracing](../../tutorials/python/05-variational-inference.md#tracing-the-model), or another existing behavior.

The handler stack, the run context, and the installation record.

::: quivers.effects.base
