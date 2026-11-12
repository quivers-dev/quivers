# CLI: `qvr`

The `qvr` console script ships with the package as a thin wrapper around the parser, the
constraint solver, and the compiler. `check`
[validates source files](../tutorials/qvr/13-release-workflow.md#1-check-the-source-and-one-target),
`run` [runs one entry point](../reference/qvr/execution-and-tooling.md#run-an-entry), `transpile`
writes target source, and `repl` and `lsp` give interactive and editor access. Fitting a posterior
with stochastic variational inference (SVI) or Markov chain Monte Carlo (MCMC) goes through the [inference API](inference/svi.md).

## `qvr check FILES...`

Parse, run the constraint solver on, and compile every `.qvr` file given.
The command emits structured diagnostics and exits with 0 on full success
and 1 on any error.

Flags:

- `--json`: emit a single JSON document on stdout containing the
  full diagnostic list, as expected by CI and pre-commit hooks.
- `--target TARGET`: compare every checked [Quivers Indexed Effect Core](qiec/index.md) (QIEC) computation with the selected
  transpiler's advertised capabilities. Unsupported features are errors, so
  this mode serves as an early release gate for one target. It does not
  render the target program, so `qvr transpile --to TARGET FILE` remains the
  check of the whole lowering and rendering path.

Diagnostic codes:

- `parse`: tree-sitter rejected the source.
- `compile`: the compiler raised `CompileError`.
- `residuated_constraint`: a slash category appears outside a
  residuated context, that is, outside a schema parameterized by a
  declared `FreeResiduated` set of categories.
- `effect_constraint`: an applied effect has a name that neither starts
  with an uppercase letter nor contains an underscore.
- `bundle_unknown_member`: a `bundle` declaration names a member that is
  not a declared rule, schema, bundle, or built-in schema.
- `family-arg-shape`: a distribution call has the wrong number of
  arguments or incompatible argument shapes.
- `implicit-family-defaults`: a distribution call relies on deprecated
  implicit defaults. This diagnostic is a warning.
- `qiec-route`: the didactic lowering route from the QVR source protocol
  to the QIEC core ABI is unavailable or does not match.
- `qiec-kind`: a static argument or binder has the wrong kind.
- `qiec-index`: an index expression has the wrong sort or constructor shape.
- `qiec-row`: an effect row violates instance identity, unification, or a
  `lacks` constraint.
- `qiec-coverage`: indexed case branches are incomplete, duplicated, or not
  justified by refinement.
- `qiec-skolem-escape`: static evidence local to a branch escapes its scope.
- `qiec-handler`: a handler signature violates coverage, forwarding, or
  resumption requirements.
- `qiec-unhandled-effect`: a computation performs an instance absent from its
  declared row.
- `qiec-backend`: a requested backend boundary cannot preserve a QIEC form.
- `qiec:capability:FEATURE:COMPUTATION`: `--target` selected a backend that
  cannot preserve `FEATURE` in the named computation.
- `io`: a file-system error, such as a missing or unreadable file.

## `qvr migrate PATHS...`

Lower `.qvr` source forward along the QVR grammar release chain.
The composer chains the migrators between adjacent grammar releases defined in
[`quivers.cli.migrations`](#quivers.cli.migrations), so the intermediate releases need not be
named. `--from` and `--to` select part of the chain.

Flags:

- `--from VERSION`: required source revision in the chain. QVR files do not
  encode an unambiguous grammar version.
- `--to VERSION`: target revision (defaults to `HEAD`).
- `--dry-run`: report which files would change without writing.
- `--output DIR`: write migrated copies under `DIR` instead
  of overwriting the originals.
- `--check`: compare each adjacent grammar pair with its migrator and
  fail if a removed rule has no converter. This mode does not migrate files.

Directory arguments are walked recursively, and individual `.qvr`
files may also be given. The migration tooling is built on the
in-tree panproto VCS at `grammars/qvr/vcs/`. A new release adds a
migration without changing the earlier migration steps.

## `qvr repl [FILE] [--plain] [--target TARGET]`

Start the interactive REPL. Without a file, the REPL opens an empty
session. With a file, it loads and elaborates the file before showing
the prompt. `--plain` uses the plain front end that reads one line at a time instead of the
TUI, and `--target TARGET` makes every load report the capability
diagnostics of that transpile target beside the module's own, as
`qvr check --target` does. See
[REPL and Language Server](../guides/repl-and-lsp.md).

## `qvr run FILE [ENTRY [ARGUMENTS...]]`

Execute one entry point of a checked module: a `define` computation or a
`program`. With no entry name, or with `--list`, the command prints every
entry point with its signature. A computation's signature gives its value
parameters, static parameters, performed effects, and result type. A program's
gives its [data inputs](../reference/qvr/probabilistic-programs.md#host-data), its result type, and the sample sites a run may
condition on.

Each positional argument is a JSON literal and corresponds to one value
parameter in declaration order. A computation runs under the configured
runtime providers, validating the arguments and result against the
specialized QIEC types with a fresh attachment table.

A program runs forward on the [reference machine](../developer/qiec.md#reference-evaluator), the
library's interpreter for QIEC. Its `random` instance replays the sites given with `--site` and
draws and scores every other site, and its `score` instance accumulates. The run reports the
program's value with the log joint density of the drawn and given sites, its observations, and its
scores. A program's parameters may also be given by name with `--data`. Its extents are read from
the data where the data fix them, so `--static` names only an extent that no data fixes, such as the
object parameter of a program template (`qvr run pooling.qvr school_effects 0.6 --static K=8`). The
run fixes a program's handlers, so `--runtime` applies to computations only.

Flags:

- `--list`: print the file's entry points and exit.
- `--static NAME=TERM`: specialize one static telescope binder with a closed
  type, index, or effect term. Repeat the flag once for every binder.
- `--data NAME=JSON`: give one program parameter by name (a data input, an
  observation, an [index map](../reference/qvr/probabilistic-programs.md#host-data) from rows to
  groups, or a scalar). Repeatable.
- `--site NAME=JSON`: condition one sample site of a program on a value,
  drawing every other site. Repeatable. A site the run never reaches is an
  error.
- `--seed N`: seed the reference generator a program run draws with, so two
  runs seeded alike draw alike.
- `--runtime FILE.json`: select runtime providers and their options through a
  non-executable JSON configuration.
- `--fuel STEPS`: bound the number of evaluation steps.
- `--trace`: write stable execution events to stderr in human-readable mode.
- `--json`: emit the result, runtime label, and complete trace as JSON, with
  the entry's kind and a program's log joint. A failure instead emits a JSON
  diagnostic and exits 1.

For instance, this command specializes `A` to `Int` and passes the value
argument `7`:

```bash
qvr run identity.qvr identity 7 --static A=Int --json
```

and this one runs a program on four observations with its site `a` fixed:

```bash
qvr run calls.qvr prog '[0.1, 0.2, 0.3, 0.4]' --site a=0.5 --seed 0
```

The same invocation is available from Python through
[`Program.run`][quivers.program.Program.run] on a loaded module, from
[`Compiler.entry`][quivers.dsl.compiler.Compiler.entry] and
[`invoke_entry`][quivers.qiec.entries.invoke_entry] on its checked module,
and from the REPL's `:run`, with the same validation, providers, trace, and
diagnostic codes.

The default runtime configuration selects the built-in `core` provider. A
configuration can select that provider with handler behavior or name an
installed `quivers.qiec_runtime` entry point:

```json
{
  "providers": [
    {
      "name": "core",
      "options": {
        "handlers": {
          "pass": {"kind": "passthrough"}
        }
      }
    }
  ]
}
```

The core provider accepts `passthrough`, `scripted`, and `state` handler
configurations, and the prelude kinds `draw`, `draw-scoring`, `enumerate`,
`replay`, `collect`, `score`, `search`, and `param`. These options give the
clause behavior, which lives only in the running Python process. The source
handler still fixes the checked effect, coverage, forwarding policy, clause
set, and resumption grades.
Runtime plugins are discoverable by name but are loaded only when the
configuration explicitly selects them.

Execution failures use stable codes including `qiec-run-computation`,
`qiec-run-config`, `qiec-run-arity`, `qiec-run-static`, `qiec-run-provider`,
`qiec-run-validator`, `qiec-run-argument`, `qiec-run-evaluation`,
`qiec-run-fuel`, and `qiec-run-result`.

## `qvr lsp`

Run the Language Server over stdio, or pass `--tcp PORT` to bind a
TCP port. `--target TARGET` enables live `qiec:capability:*` diagnostics for
the selected transpiler. Editor extensions invoke this command over the
Language Server Protocol. The server accepts later target changes through
`workspace/didChangeConfiguration`, with either
`{"qvr": {"transpileTarget": "TARGET"}}` or
`{"transpileTarget": "TARGET"}`, and an empty target turns these diagnostics off.
See [REPL and Language Server](../guides/repl-and-lsp.md).

## `qvr transpile FILE`

Transpile a `.qvr` file to a registered probabilistic programming
backend. Use `--to BACKEND` for one target, `--to-all` for every
registered target, or `--list-targets` to inspect the registry. The
`--output` flag selects the destination of a run for one target, and
`--out-dir` that of a run for several.

## `qvr kernel`

Install or run the quivers Jupyter kernel. `qvr kernel install`
registers the kernelspec. `--user` selects the user kernel directory,
and `--prefix PREFIX` selects an explicit Jupyter prefix.

## Module reference

The `quivers.cli` package exports the `qvr` entry point together with
the tooling its subcommands share: checking of each file, the REPL session
and its front ends, completion, and highlighting.
`quivers.cli.migrations` exports the grammar migration chain, the
language server is built by `quivers.lsp`, and the Jupyter kernel
lives in `quivers.kernel`.

::: quivers.cli

::: quivers.cli.migrations

::: quivers.lsp

::: quivers.kernel
