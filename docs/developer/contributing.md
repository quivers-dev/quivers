# Contributing to Quivers

This guide covers setting up a development environment, understanding the project structure, and contributing code to the quivers library.

A contribution usually changes one of three things: what source programs can
express, what those programs compute, or how their results reach users.
For syntax work, follow the parser and compiler tests. For distribution or
inference work, compare numerical scores and [posterior behavior](../tutorials/qvr/08-diagnostics-and-comparison.md#convergence-diagnostics). For a
transpiler, test both emitted syntax and agreement with the source model's
[joint score](../tutorials/python/05-variational-inference.md#tracing-the-model) at the same supplied values, since valid target syntax can still
encode the wrong distribution.

The [architecture guide](../getting-started/architecture.md) maps these tasks
to packages. The [conceptual introduction](../getting-started/concepts.md)
explains the modeling and categorical vocabulary used in those packages.

## Development Environment Setup

### Prerequisites

- Python 3.14 or later
- pip or conda
- git
- A C toolchain (`cc`, Clang, GCC, or MSVC) for editable parser builds

### Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/quivers-dev/quivers
cd quivers
pip install -e ".[dev]"
```

This installs the package in editable mode along with development dependencies.

### Running Tests

Run the full test suite:

```bash
python -m pytest tests/ -x
```

The `-x` flag stops at the first failure, which is useful for iterative development. For more options:

```bash
python -m pytest tests/ -v                  # verbose
python -m pytest tests/ -k test_name        # run specific tests
python -m pytest tests/path/to/test_file.py # run specific file
```

### Test tiers

Two independent markers govern runtime. Tests marked `probe` drive a real
probabilistic programming language inside a Docker container to compare its
log density against the reference. Recovery and benchmark tests are marked
`slow` and are deselected by the default `pyproject.toml` configuration.

Use these commands for the common tiers:

```bash
python -m pytest tests/ -m "not slow and not probe"  # fast, no Docker
python -m pytest tests/ -m "not slow and probe"      # container probes
python -m pytest tests/ -m slow                       # recovery + benchmarks
python -m pytest tests/ --runslow                     # the complete suite
```

The marker is applied by module, from the registry in
`tests/transpile/conftest.py`, and audited against the source: a module
that calls `run_probe` without appearing there fails
`test_probe_marker_registry.py`, so a new probe module cannot land in
the cheap tier by omission.

Probe cells need Docker. The session fixture starts the daemon and
builds any missing images so the requested checks run. Set `QUIVERS_SKIP_DOCKER=1` to opt out, which is
appropriate when running only `-m "not slow and not probe"`.

### Running tests in parallel

Both tiers are parallel-safe: probe scratch directories are unique per
call, containers are anonymous, and the image build is serialized
across workers by a file lock.

```bash
python -m pytest tests/ -m "not slow and not probe" -n auto --dist loadfile
python -m pytest tests/ -m "not slow and probe" -n auto
```

`--dist loadfile` is the right mode for the cheap tier. Its tests are
individually far shorter than the cost of handing one to a worker, so
distributing whole files wins where distributing single tests loses.

To divide the probe tier across machines, `pytest-split` takes a share:

```bash
python -m pytest tests/ -m "not slow and probe" --splits 4 --group 1
```

### Caching probe measurements

Loading Turing into a fresh Julia session costs seconds, while the container
starts in a fraction of one. Several cells measure the same density before
checking different properties of it. Point `QUIVERS_PROBE_CACHE` at a directory
to memoize those measurements and avoid repeating the runtime startup:

```bash
QUIVERS_PROBE_CACHE=.probe-cache python -m pytest tests/ -m "not slow and probe"
```

The key covers the image ID, the emitted source, the points, the shape
and dtype tables, and the probe script together with its reshape
helpers, all hashed by content. Rebuilding an image changes its ID and
invalidates the cached result even when its tag stays the same.

Caching is off unless the variable is set. A memoised measurement is
only as sound as its key, so a run that gates a release should execute
every container for real.

## Project Structure

```
quivers/
├── docs/                          # Documentation
│   ├── api/                       # API reference
│   ├── developer/                 # Developer documentation
│   ├── getting-started/           # User guides
│   ├── guides/                    # Long-form guides
│   └── tutorials/                 # Tutorials
├── grammars/qvr/                  # Tree-sitter grammar for the QVR DSL
│   ├── grammar.js                 # Grammar source of truth
│   ├── grammar.json               # Generated from grammar.js
│   ├── src/                       # Generated parser.c + node-types.json
│   ├── test/corpus/               # Tree-sitter fixtures
│   └── queries/                   # Editor highlight queries
├── src/quivers/                   # Main package
│   ├── __init__.py
│   ├── analysis/                  # Static program and plate analysis
│   ├── arrows/                    # Hughes-style arrow hierarchy
│   ├── categorical/               # Categorical algebra
│   ├── continuous/                # Continuous distributions (more than 40 families)
│   ├── core/                      # Core types (didactic Models)
│   ├── data/                      # Data-frame schemas and encoders
│   ├── diagnostics/               # ArviZ-compatible diagnostics
│   ├── dsl/                       # QVR DSL pipeline
│   │   ├── parser/                # panproto-driven parser walker
│   │   ├── ast_nodes/             # didactic Model AST nodes
│   │   ├── compiler/              # AST -> Program lowering and resolution
│   │   ├── emit.py                # canonical AST -> QVR source printer
│   │   ├── program_theory.py      # QVR_PROGRAM_PROTOCOL + Schema extractor
│   │   ├── pygments_lexer.py      # Pygments lexer for docs highlighting
│   ├── effects/                   # Algebraic effect handlers
│   ├── enriched/                  # Enriched categories
│   ├── formulas/                  # Formula parsing and model construction
│   ├── inference/                 # Variational inference
│   ├── kernel/                    # Jupyter kernel
│   ├── lsp/                       # Language Server Protocol implementation
│   ├── monadic/                   # Monadic programs (draw, observe, return)
│   ├── qiec/                      # Typed indexed-effect core and evaluator
│   ├── stochastic/                # Stochastic morphisms
│   ├── structural/                # Structural encoders, decoders, and losses
│   └── transpile/                 # Cross-language PPL emitters
├── tests/                         # Test suite (mirrors src structure)
├── pyproject.toml                 # Package metadata
└── mkdocs.yml                     # Documentation config
```

## Code Style Conventions

### Type Hints

Include type hints in all function signatures. Use modern Python 3.14+ syntax:

- Use `dict[K, V]` not `Dict[K, V]`
- Use `list[T]` not `List[T]`
- Use `X | None` not `Optional[X]`
- Use `tuple[T, ...]` for variable-length tuples

```python
def process_data(values: list[float], multiplier: float = 1.0) -> dict[str, float]:
    """Process numeric data."""
    return {str(i): v * multiplier for i, v in enumerate(values)}
```

Do not use type hints in function bodies or variable assignments unless necessary for clarity in complex code sections.

### Docstrings

Use numpy-style docstrings for all public modules, classes, and functions:

```python
def calculate_entropy(probabilities: list[float]) -> float:
    """Calculate Shannon entropy of a probability distribution.

    Parameters
    ----------
    probabilities : list[float]
        Probabilities that sum to 1.0.

    Returns
    -------
    float
        Shannon entropy in nats.

    Raises
    ------
    ValueError
        If probabilities do not sum to approximately 1.0.

    Examples
    --------
    >>> entropy([0.5, 0.5])
    0.6931471805599453
    """
```

### Comments

Use lowercase inline comments to clarify non-obvious logic:

```python
# compute sufficient statistics for exponential family
sufficient_stats = compute_stats(data)

# handle edge case where prior is uniform
if prior_strength == 0:
    posterior = likelihood
```

Avoid stating the obvious. Comments should explain "why," not "what."

### Python Version and Modern Features

Target Python 3.14 and later. Use modern features:

- Type union syntax: `X | None` instead of `Union[X, None]`
- Positional-only parameters: `def func(a, /, b)`
- `Literal[...]` discriminators for `dx.TaggedUnion` variants

### Value types are didactic Models

Every record-shaped value (AST nodes, `FinSet`, `ProductSet`, `CoproductSet`, `ContinuousSpace` variants, `Category` variants, `RuleSystem`) subclasses `didactic.api.Model`. Recursive sums are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field. Use `dx.field(..., converter=...)` for normalization (e.g., flattening nested `ProductSet` components), `@dx.derived` for computed fields, and `__axioms__` for cross-field invariants.

Tensor-bearing accumulators (`Presheaf`, `Weight`, `SampleSite`, `Trace`) remain `@dataclass` because they hold mutable `torch.Tensor` fields.

### Exports

What a module exports, and where users import it from, follows the
[export policy](public-api.md): every public module declares `__all__`, each
package re-exports its modules' public names, public signatures name
only public types, and every public name renders in the API reference.
`tests/test_public_api.py` and `tools/check_api_reference.py` enforce it.

## The DSL Pipeline

The QVR DSL processes `.qvr` files through these stages:

Read the stages as progressively answering more specific questions. Parsing
asks whether the text has a valid grammatical structure and builds an abstract
syntax tree (AST), a record of that structure. Resolution asks what each name
refers to. Checking asks whether the resolved pieces have compatible types and
effects. Elaboration makes implicit source operations explicit in the core,
and compilation attaches the runtime objects needed to execute them.

For instance, `observe y <- Normal(mu, sigma)` mentions a distribution and an
observation in source. Its core computation explicitly evaluates that density
and requests that its log weight be accumulated. This makes the same scoring
operation inspectable by the runtime and by target-language generation.

### 1. Parsing

`quivers.dsl.parser.parse(source)` and `parse_file(path)` register Quivers'
generated QVR grammar and its source-bound native parser with panproto's
tree-sitter-driven `AstParserRegistry`. Platform wheels load the verified native
pair directly; editable checkouts may compile a content-addressed development
parser. The parser walker then converts the parse tree into a tree of `dx.Model`
AST nodes. Lexical and syntactic errors both raise `ParseError`.

### 2. AST Nodes

Each syntax construct is a `dx.Model`. Recursive sums (`ObjectExpr`,
`Expr`, `LetExprNode`, `ProgramStep`, `Statement`) are
`dx.TaggedUnion` roots:

<!-- python: skip -->
```python
from typing import Literal
from quivers.dsl.ast_nodes import ObjectExpr, OptionEntry, ProgramParam, ProgramStep, Statement

class ProgramDecl(Statement):
    name: str
    params: tuple[str, ...] | None = None
    type_params: tuple[ProgramParam, ...] | None = None
    domain: ObjectExpr
    codomain: ObjectExpr
    options: tuple[OptionEntry, ...] = ()
    draws: tuple[ProgramStep, ...] = ()
    return_vars: tuple[str, ...] = ()
    return_labels: tuple[str, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["program_decl"] = "program_decl"
```

### 3. Resolution

`quivers.dsl.compiler.resolution._ResolutionMixin` resolves an
`ObjectExpr` to either a discrete `SetObject` or a
`ContinuousSpace`. `_resolve_any_space` performs the shared dispatch;
the narrower helpers reject a result in the wrong category.

### 4. Compilation

`quivers.dsl.compiler.Compiler(ast).compile()` walks the AST, calls the resolution lenses, validates domain/codomain compatibility, builds the morphism DAG, and wraps the result in a `quivers.Program`.

### 5. Schema Extraction

`extract_program_schema(compiler)` walks the resolved environment and emits a `panproto.Schema` over `QVR_PROGRAM_PROTOCOL`. Use this to compare two programs with `panproto schema diff` or generate migration lenses between them.

## Adding a New Distribution Family

Most independent families use the single registry in
`src/quivers/continuous/family_spec.py`. If PyTorch already provides
the distribution, add an `_make_family(...)` call in
`src/quivers/continuous/families.py` with the parameter names and
bijectors that map unconstrained parameters onto their supports.
That call creates the `Conditional*` class and registers both inline
DSL paths.

<!-- python: skip -->
```python
ConditionalMyFamily = _make_family(
    "ConditionalMyFamily",
    MyTorchDistribution,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional MyFamily(loc(x), scale(x)).",
)
```

Families with vector or matrix events, custom constructors, or
nonstandard sampling behavior need a hand-written
`ContinuousMorphism` plus a `FamilySpec` whose override fields point
to that implementation. An entirely new underlying distribution may
also require a `torch.distributions.Distribution` implementation.

Add registry, fixed-inline, mixed-inline, conditional sampling, and
`log_prob` coverage. `tests/test_family_registry.py` contains the
shared registry checks; specialized behavior belongs under
`tests/continuous/`. Document the family in
`docs/guides/continuous-families.md` and
`docs/api/continuous/families.md`.

## Testing Philosophy

- Write tests for all public APIs
- Test both happy paths and edge cases
- Use pytest fixtures for common setup
- Organize tests to mirror the source tree structure
- Aim for clear, descriptive test names: `test_<function>_<condition>_<expected>`

## Git Workflow

1. Create a feature branch: `git checkout -b feature/description`
2. Make focused commits with clear messages
3. Push to your fork and open a pull request
4. Ensure all tests pass before requesting review
5. Respond to feedback and update as needed

## Questions and Issues

Open an issue on the repository for bugs, feature requests, or questions. For development-specific questions, use discussions.
