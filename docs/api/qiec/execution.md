# Execution

Whole-module execution looks up a named checked entry and supplies its arguments and
runtime services. This separates finding a computation from evaluating its body. Inspect
the entry signature and available providers when an otherwise well-typed module cannot
execute in a particular environment.

See [invoking a checked entry](../../reference/qvr/execution-and-tooling.md#run-an-entry) and
[inspecting an entry signature](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary).

Named, checked execution for whole modules.

::: quivers.qiec.execution
