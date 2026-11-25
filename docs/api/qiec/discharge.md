# Discharge

Discharge rewrites a computation that installs
[authored handlers](../../reference/qvr/effects-and-handlers.md#discharged-handlers) to an equal one that installs
none, with the answer of each handler clause in place of the request it answers. The
[compiled runtime](../../reference/qvr/probabilistic-programs.md#which-calls-are-fit) and the transpile targets that
render no handler read the result. The rewriting and its correctness are stated in the
[semantics of effects](../../semantics/effects.md#82-static-discharge).

::: quivers.qiec.discharge
