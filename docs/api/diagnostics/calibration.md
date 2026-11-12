# `quivers.diagnostics.calibration`

[Simulation-based calibration](../../guides/analysis-fitting-and-diagnostics.md#simulation-based-calibration)
checks an inference procedure against the program it targets [@talts2018sbc]. Each simulation
draws parameters from the prior and data from the program, fits the simulated data, and ranks each
true quantity among its $L$ posterior draws. An exact procedure gives ranks uniform on
`{0, ..., L}`. Ties are broken at random so that discrete quantities qualify. Uniformity is tested
with the simultaneous ECDF statistic [@sailynoja2022graphical] calibrated under the exact discrete
null, and coverage of the central rank band is judged against its exact null mass with
Clopper-Pearson intervals [@clopper1934binomial]. Test quantities that read the data can expose
miscalibration the parameters alone miss [@modrak2025sbc].

::: quivers.diagnostics.calibration
