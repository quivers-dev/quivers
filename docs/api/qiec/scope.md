# Scopes

The [reference machine](../../developer/qiec.md#reference-evaluator) keeps the values of the locals
in scope in a [`Scope`](#quivers.qiec.scope.Scope), a persistent map. Binding a local returns a new
scope that shares the old one's nodes, so a pending frame keeps its environment without a copy.
Binding and lookup walk one path of the trie and scan only the entries that share the key's full
hash. For well-distributed hashes, the path length grows logarithmically with the number of locals.

::: quivers.qiec.scope
