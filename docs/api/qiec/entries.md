# Entry points

[Entry information](../../reference/qvr/execution-and-tooling.md#list-entry-points) identifies a computation exported by a checked module and [describes
its inputs](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary). The command line, REPL, and Python interface use it to invoke the same
entry. For posterior inference, the caller additionally selects an inference algorithm
and its settings.

The executable entry points of a checked module, and one invocation the command line, the REPL, and Python share.

::: quivers.qiec.entries
