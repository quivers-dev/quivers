# Program Steps

A [`MonadicProgram`](programs.md#quivers.continuous.programs.MonadicProgram) is built from these
records, each of which does one thing in the environment of named values
([program structure](../../guides/programs.md#program-structure)):
[`Draw`](#quivers.continuous.program_steps.Draw) samples a value,
[`Observe`](#quivers.continuous.program_steps.Observe) scores an observation,
[`Let`](#quivers.continuous.program_steps.Let) computes a deterministic value,
[`Score`](#quivers.continuous.program_steps.Score) adds a log weight,
[`Call`](#quivers.continuous.program_steps.Call) runs a subprogram's steps in place, and
[`Scan`](#quivers.continuous.program_steps.Scan) runs a step program once per position of a
sequence. The [`Indexed`](#quivers.continuous.program_steps.Indexed) argument form indexes a
collection, and [`reading`](#quivers.continuous.program_steps.reading) declares the names a step
function reads. The DSL compiler emits these records, so a program built from them in Python is the
one the compiler would build from the equivalent source, as the
[programs guide](../../guides/programs.md#building-programs-in-python) shows.

::: quivers.continuous.program_steps
