# Scan Morphism

A scan carries a state from one sequence step to the next. For a time series, the
transition draws the next state conditional on the previous one. The sampling interface
returns the final state. Scoring a supplied trajectory evaluates its joint density;
obtaining the final state’s marginal density requires integrating over earlier states.

See [scan syntax and state passing](../../reference/qvr/probabilistic-programs.md#scans) and
[a temporal scan example](../../tutorials/qvr/05-time-series.md#scan-for-left-to-right-evaluation).

`ScanMorphism(cell)` realizes the iterated Kleisli composition of a
per-step cell over a sequence. `ScanMorphism.rsample(x)` applies the
kernel at each step while threading the hidden state.
`ScanMorphism.log_joint(x, hidden_states)` sums the per-step log
density. It accepts the hidden-state trajectory either as a positional
tensor or as a `{state_key: tensor}` dictionary. The default
`state_key` is `"h"`.

::: quivers.continuous.scan
