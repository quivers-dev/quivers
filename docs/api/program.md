# Program

The top-level `Program` contains named objects, morphisms, and exports [compiled from a
module](dsl/index.md#quivers.dsl.load). Use its [lookup methods](../tutorials/python/09-debugging.md#inspecting-a-compiled-program) to select the particular model or other exported
computation you want to execute, then pass that computation to an execution or inference
procedure.

Top-level probabilistic program definitions and execution.

::: quivers.program
