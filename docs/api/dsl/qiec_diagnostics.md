# QIEC diagnostics

When lowering rejects a source construct, [a diagnostic](../../reference/qvr/execution-and-tooling.md#stable-diagnostic-families) should identify both the
unsupported operation and its location. This module defines that error information. Use
the diagnostic to distinguish a source error from a construct that is valid in principle
but unavailable on the [requested path](../../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing).

The diagnostic the QIEC elaboration raises for a rejected source.

::: quivers.dsl.qiec_diagnostics
