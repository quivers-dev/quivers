# Scan Morphism

A [scan carries a state](../../reference/qvr/probabilistic-programs.md#scans) from one sequence step to the next. For a [time series](../../tutorials/qvr/05-time-series.md#scan-for-left-to-right-evaluation), the
transition draws the next state conditional on the previous one. The sampling interface
returns the final state. Scoring a supplied trajectory evaluates its joint density;
obtaining the final state’s marginal density requires integrating over earlier states.

`ScanMorphism(cell)` realizes the iterated Kleisli composition of a
per-step cell over a sequence. `ScanMorphism.rsample(x)` applies the
kernel at each step while threading the hidden state.
`ScanMorphism(cell, outputs="all")` returns every state, `(batch,
seq_len, hidden_dim)`, rather than the last, and
`ScanMorphism(cell, reverse=True)` visits the positions from the last
to the first; its trajectory, returned or scored, stays in input order.
`ScanMorphism.rsample_trajectory(x)` draws every state, and
`ScanMorphism.log_prob(x, trajectory)` scores a trajectory as the sum of
its per-step log densities; a final state alone is refused, since its
density is a marginal over the earlier states.
`ScanMorphism.log_joint(x, hidden_states)` sums the same per-step log
density and accepts the hidden-state trajectory either as a positional
tensor or as a `{state_key: tensor}` dictionary. The default
`state_key` is `"h"`.

`ScanMorphism` is the scan as a standalone morphism. Inside a
[`MonadicProgram`][quivers.continuous.MonadicProgram], the compiler binds
every scan with a [`Scan`][quivers.continuous.Scan] step instead, whose
step's sites are sites of the program addressed `"<label>@<n>"`: a
stochastic morphism cell's draw at the scan's binding, a program cell's
sites under their own labels. A program refuses a draw from a
`ScanMorphism`.

::: quivers.continuous.scan
