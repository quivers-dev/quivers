# Scan Morphism

A [scan](../../reference/qvr/probabilistic-programs.md#scans) carries a state from one position of a
sequence to the next. `ScanMorphism(cell)` composes a cell, a kernel from an input and the previous
state to the next state, once per position, as a
[time series](../../tutorials/qvr/05-time-series.md#scan-for-left-to-right-evaluation) transition
draws the next state given the previous one. `ScanMorphism.rsample(x)` threads the hidden state
through the sequence and returns the final state. `ScanMorphism(cell, outputs="all")` returns every
state, of shape `(batch, seq_len, hidden_dim)`, and `ScanMorphism(cell, reverse=True)` visits the
positions from last to first while its trajectory, returned or scored, stays in input order.

`ScanMorphism.rsample_trajectory(x)` draws every state, and `ScanMorphism.log_prob(x, trajectory)`
scores a trajectory as the sum of the log densities of its steps. A final state alone is refused,
since its density is a marginal over the earlier states.
`ScanMorphism.log_joint(x, hidden_states)` sums the same log densities and accepts the trajectory
of hidden states either as a positional tensor or as a `{state_key: tensor}` dictionary, with
`state_key` defaulting to `"h"`.

`ScanMorphism` is the scan as a standalone morphism. Inside a
[`MonadicProgram`][quivers.continuous.MonadicProgram], the compiler binds
every scan with a [`Scan`][quivers.continuous.Scan] step instead, whose
step's sites are sites of the program addressed `"<label>@<n>"`. A
stochastic morphism cell draws at the scan's binding, and a program cell's
sites keep their own labels. A program refuses a draw from a
`ScanMorphism`.

::: quivers.continuous.scan
