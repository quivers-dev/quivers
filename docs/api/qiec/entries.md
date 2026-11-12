# Entry points

An *entry point* is a named computation or program that a checked module exposes
([list entry points](../../reference/qvr/execution-and-tooling.md#list-entry-points)), with a
description of [its inputs](../../tutorials/qvr/13-release-workflow.md#2-inspect-the-public-boundary).
`entry_points` lists them, and `invoke_entry` runs one in the same way for the command line, the
REPL, and Python. Posterior inference also needs an algorithm and its settings, which the caller
chooses.

::: quivers.qiec.entries
