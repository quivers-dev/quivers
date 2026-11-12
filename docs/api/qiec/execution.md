# Execution

[Running a module](../../reference/qvr/execution-and-tooling.md#run-an-entry) looks up a named,
checked entry point and gives it its arguments and runtime services, separately from evaluating
its body. A *provider* (`RuntimeProvider`) binds the handlers and type validators that a run needs
and that live only in the running Python process. When a well-typed module cannot run in some
environment, the [entry signature](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary)
and the available providers show what is missing.

::: quivers.qiec.execution
