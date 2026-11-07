# Execution

[Whole-module execution](../../reference/qvr/execution-and-tooling.md#run-an-entry) looks up a named checked entry and supplies its arguments and
runtime services. This separates finding a computation from evaluating its body. Inspect
the [entry signature](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary) and available providers when an otherwise well-typed module cannot
execute in a particular environment.

Named, checked execution for whole modules.

::: quivers.qiec.execution
