# Scope inspection

These functions resolve [scope paths](../../guides/repl-and-lsp.md#scope-paths), `name::name::...`, in a compiled module, which the REPL
commands that take a binding name (`:info`, `:type`, `:doc`, `:browse`, `:where`) use. A path
separates two sites of the same name that belong to
[different nested computations](../../semantics/setting.md#5-environments).

::: quivers.analysis.scope
