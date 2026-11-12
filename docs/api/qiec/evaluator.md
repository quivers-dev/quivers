# Evaluator

The [reference machine](../../developer/qiec.md#reference-evaluator) is the library's small-step
interpreter for QIEC, a [CEK machine](https://en.wikipedia.org/wiki/CEK_Machine) that applies one
rule at a time. Each step shows how a request is handled, how a value is returned, and
[which continuation resumes](../../reference/qvr/effects-and-handlers.md#handler-declarations), so
the machine fixes how a [checked computation](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler)
runs [@reynolds1972definitional].

::: quivers.qiec.evaluator
