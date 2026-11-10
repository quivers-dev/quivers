# Program Steps

[Each step](../../guides/programs.md#program-structure) answers one local question: draw a value, score an observation, compute a
deterministic result, or combine an indexed collection. The program supplies the
surrounding environment of named values. Inspect these records when [constructing
programs in Python](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram) or understanding how a QVR statement becomes an executable operation.

The records a [`MonadicProgram`](programs.md#quivers.continuous.programs.MonadicProgram)
is built from: [`Draw`](#quivers.continuous.program_steps.Draw),
[`Observe`](#quivers.continuous.program_steps.Observe),
[`Let`](#quivers.continuous.program_steps.Let),
[`Score`](#quivers.continuous.program_steps.Score),
[`Call`](#quivers.continuous.program_steps.Call), which runs a
sub-program's steps in place, and
[`Scan`](#quivers.continuous.program_steps.Scan), which runs a step
program once per position of a sequence, together with the
[`Indexed`](#quivers.continuous.program_steps.Indexed) argument form and
[`reading`](#quivers.continuous.program_steps.reading), through which a
step function declares the names it reads. The DSL compiler emits these
records, so a program built from them in Python is the program the
compiler would build from the equivalent source; the
[programs guide](../../guides/programs.md#building-programs-in-python)
works through one.

::: quivers.continuous.program_steps
