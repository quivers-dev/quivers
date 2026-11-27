# Numbers

The [reference machine](../../developer/qiec.md#reference-evaluator) holds every number as a
tensor. A [`NumberDomain`](#quivers.qiec.numbers.NumberDomain) names the real dtype and the device
of a run; [`machine_value`](#quivers.qiec.numbers.machine_value) converts a host value to that form
where it enters the machine, and [`host_value`](#quivers.qiec.numbers.host_value) converts a
machine value back to plain Python data where it leaves.

::: quivers.qiec.numbers
