# QIEC diagnostics

`QiecDiagnosticError` is the error the QIEC elaboration raises for a rejected source. It carries a
[stable code](../../reference/qvr/execution-and-tooling.md#stable-diagnostic-families) and the
source location, which separate a source error from a construct that is valid but unavailable on
the [requested path](../../tutorials/qvr/13-release-workflow.md#4-use-target-diagnostics-while-editing).

::: quivers.dsl.qiec_diagnostics
