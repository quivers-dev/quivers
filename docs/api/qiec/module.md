# Module

A `QiecModule` holds a module's declarations, definitions, and entry points as one serializable
value. A module is [checked](../../reference/qvr/execution-and-tooling.md#check-before-you-run)
before its computations run, and a run needs the runtime services that the
[chosen entry](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary) requires.

::: quivers.qiec.module
