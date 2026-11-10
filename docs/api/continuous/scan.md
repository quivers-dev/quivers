# Scan Morphism

A [scan carries a state](../../reference/qvr/probabilistic-programs.md#scans) from one sequence step to the next. For a [time series](../../tutorials/qvr/05-time-series.md#scan-for-left-to-right-evaluation), the
transition draws the next state conditional on the previous one. The sampling interface
returns the final state. Scoring a supplied trajectory evaluates its joint density;
obtaining the final state’s marginal density requires integrating over earlier states.

`ScanMorphism(cell)` realizes the iterated Kleisli composition of a
per-step cell over a sequence. `ScanMorphism.rsample(x)` applies the
kernel at each step while threading the hidden state.
`ScanMorphism.log_joint(x, hidden_states)` sums the per-step log
density. It accepts the hidden-state trajectory either as a positional
tensor or as a `{state_key: tensor}` dictionary. The default
`state_key` is `"h"`.

Inside a [`MonadicProgram`][quivers.continuous.MonadicProgram], a scan
whose cell is a program is not drawn as one `ScanMorphism` sample; the
compiler binds it with a [`Scan`][quivers.continuous.Scan] step, whose
cell's sites are sites of the program addressed `"<label>@<n>"`. A
program refuses a draw from a `ScanMorphism` over a program cell.

::: quivers.continuous.scan
