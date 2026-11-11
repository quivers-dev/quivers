"""Load synthetic data from the gallery documentation.

Each example may contain a Python block under ``Generating synthetic
data``. [`load_gallery_data`][tests.transpile._gallery_data.load_gallery_data]
executes that block in an isolated namespace, de-aliases structural
subscripts from row order, and returns observations, latent values,
and scalar template parameters. Missing or failing blocks return
``None``.

The module also creates per-run container scratch directories and
records probe-script digests so concurrent or mid-session changes do
not mix probe inputs.
"""

from __future__ import annotations

import ast
import atexit
import dataclasses
import hashlib
import math
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
from torch.distributions import constraints

from quivers.continuous import ContinuousMorphism, Draw, MonadicProgram, Observe
from quivers.dsl import loads
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.types import IndexLiteral
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe


_GALLERY_DOCS = Path(__file__).resolve().parents[2] / "docs" / "examples"
_GALLERY_SOURCE = _GALLERY_DOCS / "source"

PROBE_SCRIPT_DIR = Path(__file__).resolve().parent / "probes" / "_scripts"
"""Directory of the per-backend probe entrypoints and their shared
reshape helpers.

[`run_probe`][tests.transpile._docker.run_probe] copies the selected
entrypoint plus `_reshape.py` / `_reshape.jl` out of here and into the
bind-mounted scratch at launch time, so the container executes whatever
these files hold *at that instant* rather than whatever they held when
the session started."""


PROBE_SCRATCH_PARENT = Path("/tmp")
"""Parent of the per-process probe-scratch root.

`/tmp` rather than [`tempfile.gettempdir`][tempfile.gettempdir]:
`run_probe` bind-mounts the scratch into a container, and `/tmp`
(`/private/tmp` once resolved) is on the host-sharing list every Docker
installation the matrix runs against exposes by default, while the
per-user `$TMPDIR` on macOS is not guaranteed to be."""


_PROBE_SCRATCH_ROOT: Path | None = None
"""Lazily-created root holding every scratch this *process* hands out.

One per interpreter, named after the pid and a
[`tempfile`][tempfile.mkdtemp] suffix, and removed at interpreter exit.
Two concurrent pytest processes on one machine therefore never see each
other's probe inputs."""


_PROBE_LABEL_RE = re.compile(r"[^0-9A-Za-z._-]+")

_PROBE_ROOT_RE = re.compile(r"^quivers-probe-(\d+)-[0-9A-Za-z_]+$")
"""Shape of a probe-scratch root name, capturing the owning pid.

The pid is what makes an abandoned root identifiable. A root is swept
at interpreter exit, which a killed run never reaches, so the leftovers
have to be distinguishable from the roots of processes still using
them, and only the owner's pid does that."""


def sweep_abandoned_probe_roots(parent: Path | None = None) -> list[Path]:
    """Remove probe-scratch roots whose owning interpreter is gone.

    A root belonging to a live pid is in use right now, possibly by
    another session on this machine, and is never touched. A root whose
    pid names no process belongs to nobody: the run that owned it was
    killed before its exit hook could sweep it, and what it holds (a
    compiled model, a point set, a container's result) is of no use to
    anyone.

    Parameters
    ----------
    parent
        Directory to sweep, defaulting to
        [`PROBE_SCRATCH_PARENT`][tests.transpile._gallery_data.PROBE_SCRATCH_PARENT].
        A test names its own so that exercising the sweep cannot reach
        the roots of concurrent sessions, which is the one thing this
        function must never do.

    Returns
    -------
    list[Path]
        The roots removed, in sorted order, so a caller can report what
        a run reclaimed.
    """
    root_parent = PROBE_SCRATCH_PARENT if parent is None else parent
    swept: list[Path] = []
    for entry in sorted(root_parent.glob("quivers-probe-*")):
        if not entry.is_dir():
            continue
        match = _PROBE_ROOT_RE.match(entry.name)
        if match is None:
            continue
        pid = int(match.group(1))
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # No such process: the owner is gone and cannot come back
            # under this pid without the OS having recycled it, which
            # would make the directory its own anyway.
            shutil.rmtree(entry, ignore_errors=True)
            swept.append(entry)
        except OverflowError, PermissionError:
            # Out of pid range, or owned by another user whose live
            # process this one may not signal. Either way the root is
            # not established as abandoned, so it stays.
            continue
    return swept


def probe_scratch(label: str) -> Path:
    """Create an empty, process-private directory for one probe run.

    Callers must request a new directory for every
    [`run_probe`][tests.transpile._docker.run_probe] invocation. A
    reused path may retain compiled artifacts or allow concurrent
    processes to overwrite ``points.json`` and ``result.json``.
    ``tempfile.mkdtemp`` supplies uniqueness; `label` is descriptive
    only.

    Parameters
    ----------
    label
        Human-readable tag for the run, conventionally
        `"<tier>-<model>-<backend>"`. Characters outside
        `[0-9A-Za-z._-]` are folded to `_`.

    Returns
    -------
    Path
        A directory that exists, is empty, and is returned exactly once.

    Raises
    ------
    ValueError
        When `label` folds to the empty string, which would leave the
        run untraceable in a directory listing.
    """
    global _PROBE_SCRATCH_ROOT
    if _PROBE_SCRATCH_ROOT is None:
        sweep_abandoned_probe_roots()
        root = Path(
            tempfile.mkdtemp(
                prefix=f"quivers-probe-{os.getpid()}-",
                dir=PROBE_SCRATCH_PARENT,
            )
        )
        # A container may leave artefacts the host user cannot unlink
        # (a `__pycache__` written under a different uid mapping).
        # Interpreter shutdown is not the place to raise about that, so
        # the sweep is best-effort; what matters for correctness is that
        # nothing is ever *read* from a reused directory, which
        # `mkdtemp` already guarantees.
        atexit.register(shutil.rmtree, root, True)
        _PROBE_SCRATCH_ROOT = root
    safe = _PROBE_LABEL_RE.sub("_", label).strip("_")
    if not safe:
        raise ValueError(
            f"probe_scratch label {label!r} folds to the empty string, "
            f"so the run would be anonymous in a directory listing. "
            f"Pass a tag with at least one of [0-9A-Za-z._-]."
        )
    return Path(tempfile.mkdtemp(prefix=f"{safe}-", dir=_PROBE_SCRATCH_ROOT))


def probe_scratch_root() -> Path | None:
    """The per-process scratch root, or None before the first
    [`probe_scratch`][tests.transpile._gallery_data.probe_scratch] call.

    Tests use this value to check that processes do not share probe
    inputs.
    """
    return _PROBE_SCRATCH_ROOT


def probe_script_digests() -> dict[str, str]:
    """SHA-256 of every file in
    [`PROBE_SCRIPT_DIR`][tests.transpile._gallery_data.PROBE_SCRIPT_DIR],
    keyed by file name.

    `run_probe` copies these files at launch. Comparing a baseline with
    a later call detects probe changes during a test session; see
    [`assert_probe_scripts_unchanged`][tests.transpile._gallery_data.assert_probe_scripts_unchanged].
    """
    return {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in sorted(PROBE_SCRIPT_DIR.iterdir())
        if entry.is_file()
    }


def assert_probe_scripts_unchanged(
    baseline: dict[str, str],
    names: frozenset[str] | None = None,
) -> None:
    """Fail when the probe sources moved since `baseline` was taken.

    Parameters
    ----------
    baseline
        A [`probe_script_digests`][tests.transpile._gallery_data.probe_script_digests]
        mapping captured earlier, conventionally at module import.
    names
        Restrict the comparison to these file names. A cell copies only
        its backend entrypoint and the two reshape helpers. Passing
        None compares the whole directory.

    Raises
    ------
    RuntimeError
        When any file in scope was added, removed, or rewritten. The
        message names the changed files.
    """
    current = probe_script_digests()
    candidates = set(baseline) | set(current)
    if names is not None:
        candidates &= names
    moved = sorted(
        name for name in candidates if baseline.get(name) != current.get(name)
    )
    if not moved:
        return
    raise RuntimeError(
        f"the probe sources under {PROBE_SCRIPT_DIR} changed while this "
        f"session was measuring: {moved!r}. `run_probe` copies them into "
        f"the container at launch, so a cell measured before the edit "
        f"and a cell measured after it ran against different harnesses, "
        f"and the numbers this cell reported are of unknown provenance. "
        f"This is a fault in the session, not in any model: re-run the "
        f"tier against a tree nothing else is writing to. A change to "
        f"`_reshape.py` in particular moves the data every Python-side "
        f"backend scores, so it surfaces as a whole row of one model's "
        f"cells failing together, which is indistinguishable by eye "
        f"from a finding about that model."
    )


# The synthetic-data block is delimited by `### Generating synthetic
# data` on the start and the next `### ` or `## ` heading on the
# end (capturing the fenced ```python block within).
_DATA_BLOCK_RE = re.compile(
    r"###\s+Generating synthetic data\b.*?```python\n(.*?)```",
    re.DOTALL,
)


@dataclasses.dataclass(frozen=True)
class GalleryDataset:
    """One example's synthetic data + ground-truth parameters."""

    observations: dict[str, torch.Tensor]
    params: dict[str, torch.Tensor]
    """One point in latent space at which to evaluate the joint
    log-density, keyed by the program's post-inline sample-site names.

    Two channels fill it. The first is the snippet's captured ground
    truth, bound as `<site>`, `true_<site>`, or `<site>_true`. The
    second is the observations dict itself: a state-space example ships
    its latent trajectory there (`sites = {"s_new": ..., "o": ...}`)
    because that dict is what its inference demos clamp, and the site
    is a latent all the same. Either way the value reaches a backend
    through the parameter channel, never through a model input the
    emitted program does not declare."""

    scalar_params: dict[str, float]
    """The exported program's scalar type-parameters, at the concrete
    values the synthetic-data block instantiated the template with.

    A parametric program header (`program lda(alpha : Real, beta :
    Real) : Token -> Mix`) renders as a required *input* of the emitted
    program on every backend: numpyro emits `def model(alpha, beta,
    word_idx, w=None)`, Stan declares `real alpha; real beta;` in its
    `data` block, JAGS reads `alpha` as an unknown variable. These
    names are neither `observe` binders nor `sample` sites, so they
    reach the container through neither
    [`observations`][tests.transpile._gallery_data.GalleryDataset] nor
    [`params`][tests.transpile._gallery_data.GalleryDataset]; the point
    builder emits them into `Point.data`, which is the section every
    probe script hands to the model as an input."""

    observe_names: frozenset[str]
    """Every `observe <name>` binder the QVR source declares.

    Distinguishes an observation the program scores from a plain
    covariate the program only reads (a plate subscript such as
    `word_idx` or `out_idx`). The distinction decides whether an
    integer-dtyped entry of the observations dict may be perturbed:
    stepping an observed count to a neighbouring in-support value is a
    real data perturbation, whereas stepping a subscript would gather a
    different parameter row rather than move the data."""

    x_input: torch.Tensor | None
    """The program-input tensor the synthetic-data block prepared.
    Recognised under any of the names `x_in`, `x`, or `state_prev`;
    the first one bound in the namespace wins. Programs that declare
    `(alpha : Real, ...)` scalar parameters consume this directly;
    state-space and sequence-model programs consume the per-step
    input slice from it."""

    param_wires: dict[str, torch.Tensor]
    """The compiled parameter maps the emitted program reads as inputs.

    Keyed by wire name, which the lowering derives from the morphism a
    site draws from rather than from the site itself. They ride in
    `Point.data` beside the scalar type-parameters, and never in
    `observations`: they are inputs of the emitted program, not sites
    the in-process trace clamps. See
    [`_compiled_param_wires`][tests.transpile._gallery_data._compiled_param_wires]."""

    single_row: bool
    """Whether the exported program declares no plate.

    Such a program denotes one row and every backend emits one, so the
    fixture carries one. The tensors keep a length-one batch axis for
    the in-process trace, which reads a bare `(16,)` as a 16-row batch
    of scalars; the container wants that axis gone, because the
    emitted program declares `vector[16]`. The flat list travelling to
    the probe is the same either way, so only the shape it is rebuilt
    against differs, and that is where the axis is dropped."""

    monadic: MonadicProgram | None
    """The compiled [`MonadicProgram`][quivers.continuous.programs.MonadicProgram]
    the synthetic-data block bound to `model` (or, for state-space
    examples, `inner`). Parametric program templates are exported as
    a `Program` with `_morphism=None`; the block invokes the template
    at concrete arguments (e.g. `prog.gmm(alpha=1.0)`) and binds the
    instantiated MonadicProgram. The QVR probe consumes this directly
    rather than re-parsing the source, so it sees the same template
    instance the data was generated under."""

    shared_plates: dict[str, int] = dataclasses.field(default_factory=dict)
    """The plate extent of each datum supplied as one value every row of
    that plate shares. See
    [`_shared_plates`][tests.transpile._gallery_data._shared_plates]."""


def md_path_for(source_qvr: Path) -> Path:
    """The docs `.md` corresponding to a `docs/examples/source/<stem>.qvr`."""
    md_stem = source_qvr.stem.replace("_", "-")
    return _GALLERY_DOCS / f"{md_stem}.md"


_OBSERVE_NAME_RE = re.compile(
    r"^\s*observe\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    re.MULTILINE,
)

_SAMPLE_NAME_RE = re.compile(
    r"^\s*sample\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    re.MULTILINE,
)

_EXPORT_NAME_RE = re.compile(
    r"^\s*export\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    re.MULTILINE,
)

_PROGRAM_HEADER_RE = re.compile(
    r"^\s*program\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(([^)]*)\)",
    re.MULTILINE,
)

_TYPE_PARAM_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*:\s*([A-Za-z_][A-Za-z_0-9]*)\s*$",
)

_SCALAR_PARAM_TYPE = "Real"
"""The only declared type-parameter kind that reaches a backend as a
numeric input. A `FinSet`-typed parameter names an *object* the
compiler resolves at instantiation time (`school_effects(spread :
Real, K : FinSet)`), so it has no runtime value to send."""


def _qvr_observe_names(source_qvr: Path) -> list[str]:
    """Extract every `observe <name>` binder from the QVR source.
    The conservative regex matches the ``observe IDENT`` prefix and
    returns no match for an unrecognized form.
    """
    try:
        text = source_qvr.read_text(encoding="utf-8")
    except OSError:
        return []
    return _OBSERVE_NAME_RE.findall(text)


def _qvr_sample_names(source_qvr: Path) -> list[str]:
    """Extract every `sample <name>` binder from the QVR source.

    Used to pair `Point.params` with the program's latent-variable
    names: a `.md` snippet may bind the ground-truth value to either
    ``<name>``, ``<name>_true``, or ``true_<name>``, and the loader
    needs the list of latent names to recognise any of those three
    spellings as the same parameter.
    """
    try:
        text = source_qvr.read_text(encoding="utf-8")
    except OSError:
        return []
    return _SAMPLE_NAME_RE.findall(text)


def _exported_type_parameters(
    source_qvr: Path,
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Declared type-parameters of every program the source exports.

    Keyed by program name, each value is the header's parameter list in
    declaration order as `(name, type)` pairs. Only a header whose
    parameters are *typed* contributes: `program gru_cell(x_t, h_prev)`
    binds per-step values rather than type-parameters, and its
    untyped entries are dropped.

    Restricting to exported programs is what keeps an inner parametric
    program out of the table. `parametric_pooling` declares
    `program school_effects(spread : Real, K : FinSet)` and samples it
    from within `pooled_tight`; only `pooled_tight` is exported, so the
    emitted program takes no scalar input and none is sent.
    """
    try:
        text = source_qvr.read_text(encoding="utf-8")
    except OSError:
        return {}
    exported = frozenset(_EXPORT_NAME_RE.findall(text))
    out: dict[str, tuple[tuple[str, str], ...]] = {}
    for name, raw_params in _PROGRAM_HEADER_RE.findall(text):
        if name not in exported:
            continue
        declared: list[tuple[str, str]] = []
        for chunk in raw_params.split(","):
            typed = _TYPE_PARAM_RE.match(chunk)
            if typed is not None:
                declared.append((typed.group(1), typed.group(2)))
        if declared:
            out[name] = tuple(declared)
    return out


def _literal_argument(
    node: ast.expr,
    program: str,
    param: str,
    md: Path,
) -> float:
    """Value of one literal argument of a template invocation.

    The synthetic-data block instantiates a parametric program at
    constants (`prog.lda(alpha=1.0, beta=0.5)`), and those constants
    are the values every emitted program expects as inputs. A
    non-literal argument would make the harness guess at a value the
    backend must be driven with, so it raises instead.
    """
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise RuntimeError(
            f"{md.name}: the synthetic-data block invokes "
            f"{program!r} with a non-literal value for the scalar "
            f"type-parameter {param!r}, so the harness cannot send "
            f"the emitted program the input it declares. Bind the "
            f"argument to a numeric literal in the snippet."
        ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"{md.name}: scalar type-parameter {param!r} of "
            f"{program!r} was given {value!r}, which is not a real "
            f"number; the emitted program declares it as a real "
            f"input."
        )
    return float(value)


def _scalar_program_arguments(
    snippet: str,
    md: Path,
    declared: dict[str, tuple[tuple[str, str], ...]],
) -> dict[str, float]:
    """Concrete values the snippet instantiated the exported
    parametric program at, keyed by type-parameter name.

    Template instantiation bakes the arguments into the resulting
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram]'s
    families and leaves no record of them (`model._params` is None for
    an instantiated template), so the invocation site is the only place
    the snippet is the remaining source of the declared names.
    """
    if not declared:
        return {}
    tree = ast.parse(snippet, str(md))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        params = declared.get(func.attr)
        if params is None:
            continue
        scalar = {name for name, kind in params if kind == _SCALAR_PARAM_TYPE}
        bound: dict[str, float] = {}
        for (name, _kind), positional in zip(params, node.args):
            if name in scalar:
                bound[name] = _literal_argument(
                    positional,
                    func.attr,
                    name,
                    md,
                )
        for keyword in node.keywords:
            if keyword.arg in scalar:
                bound[keyword.arg] = _literal_argument(
                    keyword.value,
                    func.attr,
                    keyword.arg,
                    md,
                )
        missing = sorted(scalar - set(bound))
        if missing:
            raise RuntimeError(
                f"{md.name}: the synthetic-data block invokes "
                f"{func.attr!r} without a value for the scalar "
                f"type-parameter(s) {missing!r}, which every emitted "
                f"program declares as required inputs."
            )
        return bound
    return {}


def _sample_site_names(
    source_qvr: Path,
    monadic: MonadicProgram | None,
) -> list[str]:
    """Return the program's actual latent sample-site names.

    Observed steps (`observe <name>`) are excluded: their values come
    from the observations dict, not the captured ground truth. When
    the compiled
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram] is
    available it is the authority: template inlining alpha-renames an
    inner draw (``sample z`` inside a ``sample theta <- template(...)``
    step) to ``theta$z``, and folds the outer name into a
    deterministic ``let``. The compiled step specs carry the
    post-inline names the trace actually clamps, whereas the raw
    ``sample <name>`` regex reports the pre-inline source names and so
    would key the ground truth to a name no site answers to. Falls
    back to the source-level regex only when no compiled program was
    captured (the snippet bound neither ``model`` nor ``inner``).
    """
    if monadic is not None:
        names: list[str] = []
        for spec in monadic.steps:
            if isinstance(spec, Draw):
                names.extend(spec.names)
        return names
    return _qvr_sample_names(source_qvr)


def _is_tensor_like(value: object) -> bool:
    """True iff `value` is a torch.Tensor or a list of numerics
    convertible to one (the loader's two accepted shapes)."""
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(x, (int, float, list, tuple)) for x in value)
    return False


_DEALIAS_ATTEMPTS = 8
"""Row permutations drawn per plate before the de-aliasing pass gives
up. A uniform permutation of a plate of more than a handful of rows
leaves the design untouched with vanishing probability, so a draw is
rejected only in the degenerate case, and eight independent draws
exhaust that case rather than papering over it."""


def _row_permutation(
    stem: str,
    length: int,
    attempt: int,
) -> torch.Tensor:
    """A deterministic permutation of `length` row positions.

    Seeded from the example's stem, the plate width, and the attempt
    index, so the de-aliased design is reproducible run to run and
    independent of both the global RNG and whatever the example's
    synthetic-data snippet seeded.
    """
    digest = hashlib.sha256(
        f"{stem}:{length}:{attempt}".encode("utf-8"),
    ).digest()[:8]
    generator = torch.Generator()
    generator.manual_seed(int.from_bytes(digest, "big") % (2**63))
    return torch.randperm(length, generator=generator)


def _structural_subscript_names(
    observations: dict[str, torch.Tensor],
    observe_names: frozenset[str],
) -> frozenset[str]:
    """Names in `observations` that gather a parameter row rather than
    carry a coordinate of the support.

    A subscript is an integer-dtyped vector the program only reads:
    `out_idx`, `word_idx`, `item_idx`. An `observe` binder is data
    however it is typed, and a floating-point covariate is a real
    coordinate however integral its values, so both are excluded.
    """
    return frozenset(
        name
        for name, value in observations.items()
        if name not in observe_names
        and value.dim() == 1
        and not value.dtype.is_floating_point
    )


def _dealias_row_order(
    observations: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    x_input: torch.Tensor | None,
    observe_names: frozenset[str],
    stem: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor | None]:
    """Relabel plates whose structural subscript matches row order.

    Structural subscripts remain fixed across evaluation points. A
    renderer could thus replace ``arange(D).repeat(N)`` with ``i % D``
    without changing the measured density. A deterministic row
    permutation separates the supplied subscript from the row counter
    while preserving the joint, which is a product over rows.

    All arrays with the plate width move together, including program
    inputs. Arrays of other widths and plate-level latent parameters
    are unchanged.
    """
    subscripts = _structural_subscript_names(observations, observe_names)
    if not subscripts:
        return observations, params, x_input
    dealiased = dict(observations)
    moved_params = dict(params)
    moved_input = x_input
    for length in sorted({int(observations[name].shape[0]) for name in subscripts}):
        rows = sorted(
            name
            for name, value in observations.items()
            if value.dim() == 1 and int(value.shape[0]) == length
        )
        design = sorted(subscripts & frozenset(rows))
        for attempt in range(_DEALIAS_ATTEMPTS):
            order = _row_permutation(stem, length, attempt)
            if all(
                not torch.equal(dealiased[name], dealiased[name][order])
                for name in design
            ):
                break
        else:
            raise RuntimeError(
                f"{stem!r}: {_DEALIAS_ATTEMPTS} row permutations of the "
                f"{length}-row plate all left {design!r} unchanged, so "
                f"the design cannot be separated from the row counter. "
                f"A plate whose subscript is invariant under every "
                f"permutation is constant, and a constant subscript "
                f"gathers one parameter row for the whole plate."
            )
        for name in rows:
            dealiased[name] = dealiased[name][order]
            if name in moved_params:
                moved_params[name] = moved_params[name][order]
        if (
            moved_input is not None
            and moved_input.dim() > 0
            and int(moved_input.shape[0]) == length
        ):
            moved_input = moved_input[order]
    return dealiased, moved_params, moved_input


def _observations_from_namespace(
    source_qvr: Path,
    ns: dict[str, object],
) -> dict[str, object]:
    """Build an observations dict by matching the QVR program's
    `observe <name>` binders against tensors in `ns` by name.

    Strategy, in order:

    1. Direct hit: `ns[<observe_name>]` is a tensor.
    2. Prefixed hit: `ns["obs_" + <observe_name>]` is a tensor.
    3. Common alias: a single namespace tensor whose name matches a
       conventional observation alias (`counts`, `targets`, `Y`, `y`,
       `data`). Used only when the program has exactly one observe
       step; ambiguous otherwise.

    Returns an empty dict when no match is found; the caller treats
    that as "no observation data was generated."
    """
    observe_names = _qvr_observe_names(source_qvr)
    if not observe_names:
        return {}
    out: dict[str, object] = {}
    for name in observe_names:
        if name in ns and _is_tensor_like(ns[name]):
            out[name] = ns[name]
            continue
        prefixed = f"obs_{name}"
        if prefixed in ns and _is_tensor_like(ns[prefixed]):
            out[name] = ns[prefixed]
            continue
    if out:
        return out
    if len(observe_names) == 1:
        name = observe_names[0]
        for alias in ("counts", "targets", "Y", "y", "data", "obs"):
            if alias in ns and _is_tensor_like(ns[alias]):
                return {name: ns[alias]}
    return out


def load_gallery_data(source_qvr: Path) -> GalleryDataset | None:
    """Run the example's `### Generating synthetic data` block and
    return its observations + captured ground-truth `true_*` params.

    Returns None when the doc lacks the data block, the snippet
    raises, or the resulting `observations` is not a dict of
    tensors."""
    md = md_path_for(source_qvr)
    if not md.exists():
        return None
    text = md.read_text(encoding="utf-8")
    match = _DATA_BLOCK_RE.search(text)
    if match is None:
        return None
    snippet = match.group(1)

    # Build a controlled namespace. Pre-bind `torch` so the snippet
    # does not need to import it again; pre-import `load` and seed.
    ns: dict[str, object] = {
        "__name__": "__gallery_data__",
        "torch": torch,
    }
    try:
        exec(compile(snippet, str(md), "exec"), ns)
    except Exception:
        return None

    observations = ns.get("observations")
    if not isinstance(observations, dict):
        # `sites` is the alternative idiomatic name in state-space
        # examples (continuous_hmm threads observed step values via
        # `sites = {"s_new": ..., "o": ...}` rather than an
        # `observations` dict).
        sites = ns.get("sites")
        if isinstance(sites, dict):
            observations = sites
    if not isinstance(observations, dict):
        # Fall back: pull observations from the namespace by name
        # match against the QVR program's `observe <name> : ...`
        # binders. This lets a doc's synthetic-data block bind its
        # generated tensors to natural variable names (e.g.
        # `obs_counts`, `targets`, `y`) without also assembling a
        # separate `observations` dict.
        observations = _observations_from_namespace(source_qvr, ns)
        if not observations:
            return None
    obs_tensors: dict[str, torch.Tensor] = {}
    for k, v in observations.items():
        if isinstance(v, torch.Tensor):
            obs_tensors[k] = v
        else:
            try:
                obs_tensors[k] = torch.as_tensor(v, dtype=torch.float64)
            except TypeError, ValueError:
                return None

    # Compiled MonadicProgram. Templates compile to `Program(None)`
    # with `program.templates[<name>]` invokers; the synthetic-data
    # block instantiates the template at concrete arguments and binds
    # the result to `model` (or, for examples that wire the bare
    # morphism, `inner`). Capturing either lets the QVR probe walk
    # the instantiated program directly, and pins the ground-truth
    # capture below to the program's actual (post-inline) site names.
    monadic: MonadicProgram | None = None
    for monad_name in ("model", "inner"):
        candidate = ns.get(monad_name)
        if isinstance(candidate, MonadicProgram):
            monadic = candidate
            break

    # Match every sample site against the namespace under three
    # spellings: bare `<site>`, suffixed `<site>_true`, and prefixed
    # `true_<site>`, keying the captured value under the site's real
    # name so the trace clamps it. A template-inlined site carries a
    # `$` (`theta$z`); `$` is not a legal Python identifier char, so
    # the snippet spells it with `_` (`true_theta_z`) and the matcher
    # accepts that normalized base too. The bare spelling is accepted
    # only for a real sample site, so intermediate snippet bindings
    # (`T = 64`, `model = ...`, the let-step `mu_true` / `alpha_true`
    # intermediates) never get mis-captured as ground truth.
    sample_sites = _sample_site_names(source_qvr, monadic)
    params: dict[str, torch.Tensor] = {}

    def _coerce(value: object) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float64)
        if isinstance(value, (int, float)):
            return torch.tensor(float(value), dtype=torch.float64)
        if isinstance(value, (list, tuple)):
            try:
                return torch.as_tensor(value, dtype=torch.float64)
            except TypeError, ValueError:
                return None
        return None

    for site in sample_sites:
        if site in params:
            continue
        bases = [site]
        normalized = site.replace("$", "_")
        if normalized != site:
            bases.append(normalized)
        for base in bases:
            hit: torch.Tensor | None = None
            for spelling in (f"true_{base}", f"{base}_true", base):
                if spelling not in ns:
                    continue
                hit = _coerce(ns[spelling])
                if hit is not None:
                    break
            if hit is not None:
                params[site] = hit
                break

    # A sample site the snippet ships inside its observations dict is
    # still a LATENT of the program: the dict is what the `.md`'s SVI
    # and MCMC demos clamp, not a declaration that the value is data.
    # Filing it under `params` is what routes it to the container's
    # parameter channel (Stan's constrained_params, numpyro's
    # substitution dict, WebPPL's clampedParams) instead of trying to
    # hand a latent's value in through a model input the emitted
    # program never declares. `observations` is left intact, so the
    # in-process QVR trace keeps clamping the site at full rank.
    for site in sample_sites:
        if site in params or site not in obs_tensors:
            continue
        params[site] = obs_tensors[site].detach().to(dtype=torch.float64)

    scalar_params = _scalar_program_arguments(
        snippet,
        md,
        _exported_type_parameters(source_qvr),
    )
    collisions = sorted(
        frozenset(scalar_params) & (frozenset(params) | frozenset(obs_tensors))
    )
    if collisions:
        raise RuntimeError(
            f"{md.name}: scalar type-parameter(s) {collisions!r} share "
            f"a name with a sample site or an observation, so the "
            f"point's data section cannot carry both. Rename the "
            f"program's type-parameter in the `.qvr` source."
        )

    # Program-input tensor: the snippet may bind any of the canonical
    # names below. Try each in order; the first tensor-typed binding
    # wins. State-space examples conventionally use `state_prev`;
    # sequence and template examples use `x_in`; transformer-style
    # examples use `x`.
    x_input: torch.Tensor | None = None
    for x_name in ("x_in", "x", "state_prev", "x_lift"):
        candidate = ns.get(x_name)
        if isinstance(candidate, torch.Tensor):
            x_input = candidate
            break

    observe_names = frozenset(_qvr_observe_names(source_qvr))
    obs_tensors, params, x_input = _dealias_row_order(
        obs_tensors,
        params,
        x_input,
        observe_names,
        source_qvr.stem,
    )

    shared_plates = _shared_plates(source_qvr, obs_tensors)

    # The program's input rows are independent runs of the whole
    # program, and the rows a gallery program observes come from its
    # data and its plates, so the fixture scores one run: the first row
    # of the input the snippet built.
    if x_input is not None:
        x_input = _first_row(x_input)

    single_row = _program_is_plateless(source_qvr)
    if single_row:
        # The program denotes one row and every backend emits one, so
        # the fixture presents one. Broadcasting it over the rows the
        # snippet generated would score a joint of that many iid
        # copies, which is a different quantity from the one the
        # emitted program computes and cannot be reconciled by any
        # payload: a `(32, 16)` matrix against a declared `(16)` is a
        # rank mismatch. Sliced here rather than in the snippet, whose
        # SVI and MCMC demos want the full batch.
        obs_tensors = {k: _first_row(v) for k, v in obs_tensors.items()}
        params = {k: _first_row(v) for k, v in params.items()}

    return GalleryDataset(
        observations=obs_tensors,
        params=params,
        scalar_params=scalar_params,
        observe_names=observe_names,
        x_input=x_input,
        param_wires=(
            _compiled_param_wires(source_qvr, monadic)
            | (_domain_wires(source_qvr, x_input) if single_row else {})
        ),
        single_row=single_row,
        monadic=monadic,
        shared_plates=shared_plates,
    )


def _shared_plates(
    source_qvr: Path, observations: dict[str, torch.Tensor]
) -> dict[str, int]:
    """The data supplied as one value every row of a plate shares.

    A datum that fills a family parameter of a plated draw directly is
    checked as one value per row of that plate; the snippet may supply
    it without the row axis, as one value every row shares. The
    reference machine and the compiled runtime read it as given, and
    every transpile target declares the per-row shape, so
    [`target_point`][tests.transpile._gallery_data.target_point] hands
    each such datum to a target spread across the plate, as
    [`broadcast_shared_data`][quivers.qiec.broadcast_shared_data] does.

    Parameters
    ----------
    source_qvr : Path
        The example's QVR source.
    observations : dict[str, torch.Tensor]
        The snippet's observations, by name.

    Returns
    -------
    dict[str, int]
        The plate's extent for each datum supplied without its axis.
    """
    module = loads(source_qvr.read_text()).qiec
    if module is None:
        return {}
    shared: dict[str, int] = {}
    for entry in module.entries:
        for parameter in entry.parameters:
            value = observations.get(parameter.name)
            if not parameter.per_row or value is None:
                continue
            shape = tensor_shape(parameter.type)
            if shape is None or not shape[1]:
                continue
            plate = shape[1][0]
            if isinstance(plate, IndexLiteral) and value.dim() == len(shape[1]) - 1:
                shared[parameter.name] = int(plate.value)
    return shared


type PointSection = dict[str, float | int | list[float] | list[int]]
"""One section of a point as it crosses the probe boundary."""


def target_point(dataset: GalleryDataset, point: Point) -> dict[str, PointSection]:
    """A point as a transpile target reads it, shared data spread.

    Parameters
    ----------
    dataset : GalleryDataset
        The example's data, naming the shared data.
    point : Point
        The point.

    Returns
    -------
    dict[str, PointSection]
        The point's ``params`` and ``data``, each shared datum repeated
        once per row of its plate in row-major order.
    """
    data: PointSection = dict(point.data)
    for name, rows in dataset.shared_plates.items():
        if name not in data:
            continue
        value = data[name]
        if isinstance(value, list):
            data[name] = cast(list[float], value) * rows
        else:
            data[name] = [float(value)] * rows
    return {"params": point.params, "data": data}


def target_shapes(
    dataset: GalleryDataset, shapes: dict[str, list[int]]
) -> dict[str, list[int]]:
    """A shape table with each shared datum's plate axis restored.

    Parameters
    ----------
    dataset : GalleryDataset
        The example's data, naming the shared data.
    shapes : dict[str, list[int]]
        The shape of each name as the dataset holds it.

    Returns
    -------
    dict[str, list[int]]
        The shapes, each shared datum led by its plate's extent.
    """
    spread = dict(shapes)
    for name, rows in dataset.shared_plates.items():
        if name in spread:
            spread[name] = [rows, *spread[name]]
    return spread


def _program_is_plateless(source_qvr: Path) -> bool:
    """True when the exported program declares no plate.

    A `sample` / `observe` step carries its plate in an index
    annotation (`observe y : Resp <- ...`). A program whose steps
    carry none denotes one row, and that is what every backend emits:
    a single `vector[16] state`, not an array of them. The in-process
    trace will happily broadcast such a program over as many rows as
    the snippet generated, so without this the two sides score
    different things and no payload can bridge the rank.
    """
    text = source_qvr.read_text(encoding="utf-8")
    steps = re.findall(
        r"^\s*(?:sample|observe)\s+[\w$, ]+?(\s*:\s*\w+)?\s*<-",
        text,
        re.M,
    )
    return bool(steps) and not any(index for index in steps)


def _first_row(value: torch.Tensor) -> torch.Tensor:
    """The leading row of a batched fixture tensor, rank preserved.

    The batch axis is kept at length one rather than dropped. The
    in-process trace reads it: a program's morphisms are applied to a
    batch of rows and a bare `(16,)` would be taken for a single
    16-row batch of scalars. The container wants the axis gone, since
    the emitted program declares `vector[16]`, and
    `_shapes_from_dataset` is where that happens: the flat row-major
    list is identical either way, so only the shape it is rebuilt
    against differs.
    """
    if value.dim() == 0 or value.shape[0] <= 1:
        return value
    return value[:1]


def _compiled_param_wires(
    source_qvr: Path,
    monadic: MonadicProgram | None,
) -> dict[str, torch.Tensor]:
    """The linear parameter maps the emitted program reads as inputs.

    A Kleisli morphism declared `~ Family` with no `[param_source=...]`
    takes the default linear source, whose affine weights are drawn
    when the module compiles. The lowering names them
    `<morphism>_param_weight` / `<morphism>_param_bias`
    ([`_param_map_weight_name`][quivers.transpile.lower._param_map_weight_name])
    and emits them as *inputs*, because they are named nowhere in the
    `.qvr` text and are not sample sites. Nothing else supplies them:
    the snippet does not know their names and the trace does not clamp
    them, so without this a container is handed a program whose data
    block it cannot fill.

    The compiled program files the tensors under the *site* that draws
    from the morphism (`_step_<site>.param_source.linear.*`), so the
    source is what bridges the two, saying which morphism each step
    draws from. A site whose morphism carries an explicit
    `[param_source=...]` network is absent: those are refused rather
    than wired, since a network's forward pass is not reconstructible
    from a weight table.
    """
    if monadic is None:
        return {}
    text = source_qvr.read_text(encoding="utf-8")
    draws = re.findall(
        r"^\s*(?:sample|observe)\s+(\w+)[^<]*<-\s*(\w+)",
        text,
        re.M,
    )
    named = dict(monadic.named_parameters())
    wires: dict[str, torch.Tensor] = {}
    for site, morphism in draws:
        for kind in ("weight", "bias"):
            tensor = named.get(f"_step_{site}.param_source.linear.{kind}")
            if tensor is not None:
                wires[f"{morphism}_param_{kind}"] = tensor.detach().to(
                    dtype=torch.float64,
                )
    return wires


def _domain_wires(
    source_qvr: Path,
    x_input: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """The program's own domain, split into the inputs it declares.

    `program generative_step : Driver * State -> State` reaches a
    backend as two inputs, `driver` and `state`, named for the domain
    objects in declaration order and sized by their widths. The
    snippet ships them as one concatenated matrix, which only the
    in-process trace consumes, so without this split a container is
    handed a program whose arguments nothing fills.

    Split by the declared widths rather than guessed: each factor of
    the domain product is looked up in the module's `object ... : Real
    N` declarations, and the slices must account for the input's last
    axis exactly, or the fixture and the signature disagree about what
    the program reads and this raises rather than mis-wire it.
    """
    if x_input is None:
        return {}
    text = source_qvr.read_text(encoding="utf-8")
    widths: dict[str, int] = {}
    for names, width in re.findall(
        r"^object\s+([\w, ]+?)\s*:\s*Real\s+(\d+)",
        text,
        re.M,
    ):
        for name in names.split(","):
            widths[name.strip()] = int(width)
    exported = set(re.findall(r"^export\s+(\w+)", text, re.M))
    signature = None
    for name, domain in re.findall(
        r"^program\s+(\w+)[^:\n]*:\s*([^\n]+?)\s*->",
        text,
        re.M,
    ):
        if signature is None or name in exported:
            signature = (name, domain)
        if name in exported:
            break
    if signature is None:
        return {}
    factors = [factor.strip() for factor in signature[1].split("*")]
    if not all(factor in widths for factor in factors):
        # A domain factor over a `FinSet` contributes no data wire:
        # it is an index space, not a value the program reads.
        return {}
    total = sum(widths[factor] for factor in factors)
    if x_input.shape[-1] != total:
        raise AssertionError(
            f"{source_qvr.stem}: the program's domain "
            f"{' * '.join(factors)} declares {total} columns and the "
            f"snippet's input carries {x_input.shape[-1]}. One of the "
            f"two is wrong; splitting on a mismatch would wire a "
            f"backend to columns the program does not read."
        )
    wires: dict[str, torch.Tensor] = {}
    offset = 0
    for factor in factors:
        width = widths[factor]
        wires[factor.lower()] = x_input[..., offset : offset + width]
        offset += width
    return wires


def gallery_examples_with_data() -> list[Path]:
    """Return every `docs/examples/source/*.qvr` whose `.md` carries
    a `### Generating synthetic data` block."""
    out: list[Path] = []
    for qvr in sorted(_GALLERY_SOURCE.glob("*.qvr")):
        md = md_path_for(qvr)
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        if _DATA_BLOCK_RE.search(text) is not None:
            out.append(qvr)
    return out


def point_from_dataset(dataset: GalleryDataset) -> Point:
    """Build a single `Point` from a GalleryDataset.

    The Point's `params` map carries the captured `true_*` ground-
    truth parameter values (one tuple-typed entry per latent name);
    `data` carries every observation tensor as a flat list.

    When a name appears in both `dataset.params` (captured as a
    ground-truth latent) and `dataset.observations` (clamped in the
    `.md` snippet's observations dict for the SVI demo), the params
    spelling wins: the latent ground truth is the canonical value to
    score the joint at, and the entry is dropped from the data section
    so a backend that declares the name as a parameter (Stan's
    `parameters {}` block, a PyMC unobserved RV) does not also receive
    it as a data input. `dataset.observations` itself is left intact,
    so the in-process QVR trace still clamps every site.

    The program's scalar type-parameters ride in the same `data`
    section: they are inputs of the emitted program on every backend,
    and `data` is the section each probe script binds to the model's
    formal arguments."""
    return _point_from_tensors(
        dataset.params,
        dataset.observations,
        dataset.scalar_params,
        dataset.param_wires,
    )


def _point_from_tensors(
    params: dict[str, torch.Tensor],
    observations: dict[str, torch.Tensor],
    scalar_params: dict[str, float],
    param_wires: dict[str, torch.Tensor] | None = None,
) -> Point:
    """Flatten a (latent, observed) tensor pair into a wire
    [`Point`][tests.transpile.probes._protocol.Point].

    Every value becomes a row-major float list; a length-1 list
    collapses to a bare float so the probe's dict-to-Tensor casting
    picks the scalar shape. A name bound in both sections is emitted
    only under ``params`` (see
    [`point_from_dataset`][tests.transpile._gallery_data.point_from_dataset]),
    and each entry of ``scalar_params`` joins the data section as a
    bare float. ``param_wires`` joins it for the same reason: a
    compiled parameter map is an input of the emitted program, so the
    section that binds model arguments is where it belongs.
    """

    def _flatten(t: torch.Tensor) -> list[float]:
        return t.detach().to(dtype=torch.float64).flatten().tolist()

    flat_params = {k: _flatten(v) for k, v in params.items()}
    flat_data = {k: _flatten(v) for k, v in observations.items() if k not in params}
    squeezed_params: dict[str, float | int | list[float] | list[int]] = {
        k: (v[0] if len(v) == 1 else v) for k, v in flat_params.items()
    }
    squeezed_data: dict[str, float | int | list[float] | list[int]] = {
        k: (v[0] if len(v) == 1 else v) for k, v in flat_data.items()
    }
    squeezed_data.update(scalar_params)
    for name, tensor in (param_wires or {}).items():
        flat = _flatten(tensor)
        squeezed_data[name] = flat[0] if len(flat) == 1 else flat
    return Point(params=squeezed_params, data=squeezed_data)


# ---------------------------------------------------------------------------
# Multi-point evaluation.
#
# Theorem 4.1 of `docs/semantics/transpile-correctness.md` fixes the
# transpile contract as CONSTANT-spread equivalence: the pointwise
# difference `log p_QVR - log p_backend` must be the same constant `c`
# at every (theta, y) in the support. A one-point comparison cannot
# test that -- the spread of a single difference around its own mean is
# identically zero -- so the check is vacuous until the point set has
# real variation in BOTH the latents and the observed data. Latent-only
# variation is not enough either: a backend that drops a data-dependent
# term (Stan's `~` sampling statement discards data-only summands) keeps
# a constant offset as the latents move and only breaks constancy when
# the data moves.
#
# Two parameters of the point set decide what a passing check is
# evidence FOR, and neither is cosmetic.
#
# The SEED fixes which finite sample of an infinite support the check
# runs on. A defect confined to a region the sample never visits is
# invisible, and no statistic computed from that sample can say so, so a
# sensitivity claim proved at one seed is a claim about that draw rather
# than about the perturbation design. `GALLERY_SEEDS` and
# `points_across_seeds` exist to make the claim about the design; the
# default seed stays fixed so a regression cannot present as a flake.
#
# The SCALE fixes how far the sample travels. A defect whose per-point
# discrepancy is nonlinear in the point contributes a spread that grows
# superlinearly in the scale, so one quadratic in the displacement is
# roughly nine times louder at three times the excursion and can sit
# under tolerance at one excursion while being rejected outright at
# another. `WIDE_PERTURBATION_SCALE` and `point_displacements` exist so
# that gap can be measured rather than assumed away.
# ---------------------------------------------------------------------------


PERTURB_GROUND_TRUTH = "ground-truth"
"""Label of point 0: the captured ground truth, unperturbed."""

PERTURB_LATENTS = "latents"
"""Label of a point whose latent sites moved and whose data is at
ground truth. Isolates a backend that mis-scores the prior."""

PERTURB_DATA = "data"
"""Label of a point whose observed data moved and whose latents are at
ground truth. Isolates a backend that drops a data-dependent term."""

PERTURB_BOTH = "latents+data"
"""Label of a point where both sections moved. Catches an offset that
cancels when only one section varies."""

_PERTURBATION_CYCLE: tuple[str, ...] = (
    PERTURB_LATENTS,
    PERTURB_DATA,
    PERTURB_BOTH,
)
"""Mode schedule for points 1..n-1, cycled by index. The cycle puts a
latents-only and a data-only point before the joint one, so a broken
constancy localises to the section that moved."""

_PERTURBATION_SCALE = 0.2
"""Base perturbation magnitude, in the natural unconstrained
coordinate of each value's support (log scale for positive values,
logit scale for bounded ones, additive nats for reals). Large enough
that a dropped data term or a truncated latent shifts the difference
by orders of magnitude more than float round-off; small enough that
the joint stays well inside the region where every backend's
log-density is numerically well behaved."""

PERTURBATION_SCALE = _PERTURBATION_SCALE
"""Published base scale for point-set perturbations.

A quadratic discrepancy grows by a factor of nine when the scale is
tripled. The validation tests therefore compare this scale with
[`WIDE_PERTURBATION_SCALE`][tests.transpile._gallery_data.WIDE_PERTURBATION_SCALE]
to measure sensitivity to excursion size."""

WIDE_PERTURBATION_SCALE = 0.6
"""A deliberately wider excursion, three times
[`PERTURBATION_SCALE`][tests.transpile._gallery_data.PERTURBATION_SCALE].

At the base scale, current gallery cells remain in support without a
redraw. At three times that scale, one bounded-response cell needs one
halving. Because a redraw reduces the realized excursion,
[`point_excursion`][tests.transpile._gallery_data.point_excursion] and
the validation tests measure the resulting point set directly."""

GALLERY_SEEDS: tuple[int, ...] = (0, 1, 2, 3)
"""Seeds the strength sweep draws independent point sets at.

The fixed tuple makes sensitivity checks reproducible while covering
more than one draw. Each additional seed re-scores every example and
point."""

_INTEGER_STEP_FRACTION = 0.25
"""Fraction of a count vector's mean magnitude used as the standard
deviation of its integer perturbation, floored at one count so a
small-count fixture still moves."""

_BOOLEAN_FLIP_PROBABILITY = 0.25
"""Per-entry flip probability for a Boolean-supported value."""

_SUPPORT_EPS = 1e-12
"""Floor applied before taking a log / logit, so a ground-truth value
sitting numerically on a support boundary perturbs to a finite
interior point instead of producing a non-finite coordinate."""

_MAX_REDRAWS = 6
"""Attempts per point before giving up. Each retry halves the
perturbation scale, so the last attempt is 1/32 of the base scale."""

_MAX_FAMILY_DEPTH = 8
"""Depth bound on the wrapper-unwrapping walk in
[`_resolve_support`][tests.transpile._gallery_data._resolve_support]."""


def perturbation_labels(n_points: int = 6) -> list[str]:
    """Per-index perturbation label for a
    [`points_from_dataset`][tests.transpile._gallery_data.points_from_dataset]
    point list of the same length.

    Index 0 is always
    [`PERTURB_GROUND_TRUTH`][tests.transpile._gallery_data.PERTURB_GROUND_TRUTH];
    every later index cycles through `_PERTURBATION_CYCLE`. The
    schedule is a pure function of the index, so a failure message can
    name the perturbation that broke constancy without threading the
    labels through the point set.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points!r}")
    return [PERTURB_GROUND_TRUTH] + [
        _PERTURBATION_CYCLE[(i - 1) % len(_PERTURBATION_CYCLE)]
        for i in range(1, n_points)
    ]


def _resolve_support(
    morphism: ContinuousMorphism,
) -> constraints.Constraint | None:
    """The support constraint of the distribution a draw step samples.

    A plate draw / vectorised observe wraps a per-row family and
    inherits its support, but the generic
    [`ContinuousMorphism.support`][quivers.continuous.morphisms.ContinuousMorphism.support]
    default is `real`, so a wrapper that does not forward the inner
    constraint reports `real` for a positively-supported family. The
    walk thus descends into `.family` whenever the current level
    reports the `real` default and an inner family exists, and returns
    the first non-default constraint it finds.
    """
    current: ContinuousMorphism = morphism
    for _ in range(_MAX_FAMILY_DEPTH):
        support = getattr(current, "support", None)
        inner = getattr(current, "family", None)
        if (
            inner is not None
            and isinstance(inner, ContinuousMorphism)
            and isinstance(support, type(constraints.real))
        ):
            current = inner
            continue
        return support if isinstance(support, constraints.Constraint) else None
    support = getattr(current, "support", None)
    return support if isinstance(support, constraints.Constraint) else None


def site_supports(
    dataset: GalleryDataset,
) -> dict[str, constraints.Constraint]:
    """Declared support constraint per stochastic site of the example's
    compiled program, keyed by the site's post-inline name.

    This is the authority on how a value may be perturbed and on
    whether it is integer- or real-valued: it comes from the family the
    QVR source declared for the site, not from the ground-truth value
    the synthetic-data block happened to draw. Returns an empty map
    when the example's `.md` bound no compiled
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram].
    """
    monadic = dataset.monadic
    if monadic is None:
        return {}
    out: dict[str, constraints.Constraint] = {}
    for spec in monadic.steps:
        if not isinstance(spec, Draw | Observe):
            continue
        morphism = monadic.step_module(spec)
        if not isinstance(morphism, ContinuousMorphism):
            continue
        support = _resolve_support(morphism)
        if support is None:
            continue
        for name in spec.names:
            out[name] = support
    return out


def _base_constraint(
    support: constraints.Constraint,
) -> constraints.Constraint:
    """Strip the event-axis wrapper an
    [`independent`][torch.distributions.constraints.independent]
    constraint puts around a per-coordinate one (`real_vector` is
    `independent(real, 1)`), so the per-coordinate constraint is what
    decides the dtype tag and the perturbation coordinate."""
    current = support
    for _ in range(_MAX_FAMILY_DEPTH):
        inner = getattr(current, "base_constraint", None)
        if not isinstance(inner, constraints.Constraint):
            return current
        current = inner
    return current


def is_discrete_support(support: constraints.Constraint) -> bool:
    """True iff `support` admits only integer-valued points.

    Used to decide the wire dtype tag for a site: an integer-supported
    site must reach a backend that separates `int` from `real` (Stan,
    JAGS, BUGS) as an integer, and a continuous-supported site must
    reach it as a float even when its ground-truth value happens to
    have no fractional part.
    """
    return isinstance(
        _base_constraint(support),
        (
            constraints.integer_interval,
            type(constraints.nonnegative_integer),
            type(constraints.boolean),
        ),
    )


def _perturb_integer(
    work: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    *,
    declared: bool,
) -> torch.Tensor:
    """Move an integer tensor within its inferred admissible window.

    A tensor with multiple values uses the intersection of its
    attested range and the declared bounds. A constant tensor uses
    finite declared bounds. If neither case provides a finite range,
    the tensor remains fixed. A lone count whose window is a site's
    declared support, such as a category index, is redrawn across that
    window rather than stepped by its own magnitude.

    Parameters
    ----------
    work : torch.Tensor
        The integer values, as floats.
    noise : torch.Tensor
        Standard-normal noise of the same shape.
    lower : float
        The support's least value.
    upper : float
        The support's greatest value, possibly infinite.
    declared : bool
        Whether the bounds are a site's declared support rather than a
        window inferred from host data.

    Returns
    -------
    torch.Tensor
        The moved values.
    """
    if work.numel() == 0:
        return work
    attested_low = float(work.min().item())
    attested_high = float(work.max().item())
    if attested_low < attested_high:
        low = max(lower, attested_low)
        high = min(upper, attested_high)
    elif math.isfinite(lower) and math.isfinite(upper):
        low, high = lower, upper
        if declared and work.numel() == 1 and high > low:
            return _redraw_single_count(work, noise, low, high)
    else:
        return work
    magnitude = max(
        1.0,
        _INTEGER_STEP_FRACTION * float(work.abs().mean().item()),
    )
    moved = torch.round(work + torch.round(noise * magnitude))
    if low > high:
        return work
    clamped = moved.clamp(min=low, max=high)
    if torch.equal(clamped, work):
        return _nudge_frozen_integer(work, noise, low, high)
    return clamped


def _redraw_single_count(
    work: torch.Tensor,
    noise: torch.Tensor,
    low: float,
    high: float,
) -> torch.Tensor:
    """Move a single count to another value of its declared window.

    A lone count has no spread to scale a step by, and a step scaled
    by its own magnitude sends independent seeds to the same neighbour.
    The noise is mapped through the standard-normal CDF to a uniform
    position among the window's other values, so every other value is
    equally likely and the count always moves.

    Parameters
    ----------
    work : torch.Tensor
        The count, one entry.
    noise : torch.Tensor
        Standard-normal noise of the same shape.
    low : float
        The window's least admissible value.
    high : float
        The window's greatest admissible value, above ``low``.

    Returns
    -------
    torch.Tensor
        The moved count, with ``work``'s shape and dtype.
    """
    others = int(high - low)
    uniform = 0.5 * (1.0 + math.erf(float(noise.flatten()[0].item()) / math.sqrt(2.0)))
    offset = 1 + min(others - 1, int(uniform * others))
    current = int(work.flatten()[0].item()) - int(low)
    moved = int(low) + (current + offset) % (others + 1)
    return torch.full_like(work, moved)


def _nudge_frozen_integer(
    work: torch.Tensor,
    noise: torch.Tensor,
    low: float,
    high: float,
) -> torch.Tensor:
    """Move one entry when rounding and clamping leave a draw unchanged.

    The selected entry has the largest absolute noise. It moves one
    count in the noise direction, then the opposite direction if the
    first step is outside ``[low, high]``. The function consumes no
    additional randomness and returns `work` if no entry can move.
    """
    flat = work.flatten().clone()
    flat_noise = noise.flatten()
    order = torch.argsort(flat_noise.abs(), descending=True)
    for position in order.tolist():
        current = float(flat[position].item())
        direction = 1.0 if float(flat_noise[position].item()) >= 0.0 else -1.0
        for step in (direction, -direction):
            candidate = current + step
            if low <= candidate <= high:
                flat[position] = candidate
                return flat.reshape(work.shape)
    return work


def _perturb_lower_cholesky(
    work: torch.Tensor,
    noise: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Perturb a lower-triangular factor with positive diagonal.

    The diagonal moves multiplicatively (staying strictly positive)
    and the strict lower triangle moves additively, so the result is
    still a valid Cholesky factor and the covariance it induces stays
    positive definite.
    """
    tril = torch.tril(work)
    diag = torch.diagonal(tril, dim1=-2, dim2=-1)
    diag_noise = torch.diagonal(noise, dim1=-2, dim2=-1)
    moved = torch.tril(tril + scale * noise, diagonal=-1)
    new_diag = diag.clamp_min(_SUPPORT_EPS) * torch.exp(scale * diag_noise)
    return moved + torch.diag_embed(new_diag)


def _perturb_by_support(
    value: torch.Tensor,
    support: constraints.Constraint,
    generator: torch.Generator,
    scale: float,
    *,
    declared: bool,
) -> torch.Tensor | None:
    """Move `value` inside `support` by roughly `scale` in the
    support's natural unconstrained coordinate.

    ``declared`` says whether `support` is a site's declared support,
    rather than a window inferred from host data.

    Returns None for a constraint whose interior this helper cannot
    parameterise; the caller then leaves the value at ground truth
    rather than risk stepping outside the support and comparing two
    `-inf` joints (which would be a vacuous match of a different
    kind).
    """
    support = _base_constraint(support)
    work = value.detach().to(dtype=torch.float64)
    noise = torch.randn(
        work.shape,
        generator=generator,
        dtype=torch.float64,
    )
    moved: torch.Tensor
    if isinstance(support, type(constraints.simplex)):
        logits = torch.log(work.clamp_min(_SUPPORT_EPS)) + scale * noise
        moved = torch.softmax(logits, dim=-1)
    elif isinstance(support, type(constraints.boolean)):
        uniform = torch.rand(
            work.shape,
            generator=generator,
            dtype=torch.float64,
        )
        moved = torch.where(
            uniform < _BOOLEAN_FLIP_PROBABILITY,
            1.0 - work,
            work,
        )
    elif isinstance(support, constraints.integer_interval):
        moved = _perturb_integer(
            work,
            noise,
            float(support.lower_bound),
            float(support.upper_bound),
            declared=declared,
        )
    elif isinstance(support, type(constraints.nonnegative_integer)):
        moved = _perturb_integer(
            work,
            noise,
            float(support.lower_bound),
            math.inf,
            declared=declared,
        )
    elif isinstance(
        support,
        (constraints.interval, constraints.half_open_interval),
    ):
        lower = float(support.lower_bound)
        upper = float(support.upper_bound)
        width = upper - lower
        unit = ((work - lower) / width).clamp(
            _SUPPORT_EPS,
            1.0 - _SUPPORT_EPS,
        )
        logit = torch.log(unit) - torch.log1p(-unit)
        moved = lower + width * torch.sigmoid(logit + scale * noise)
    elif isinstance(
        support,
        (constraints.greater_than, constraints.greater_than_eq),
    ):
        lower = float(support.lower_bound)
        moved = lower + (work - lower).clamp_min(
            _SUPPORT_EPS,
        ) * torch.exp(scale * noise)
    elif isinstance(support, constraints.less_than):
        upper = float(support.upper_bound)
        moved = upper - (upper - work).clamp_min(
            _SUPPORT_EPS,
        ) * torch.exp(scale * noise)
    elif isinstance(support, type(constraints.lower_cholesky)):
        moved = _perturb_lower_cholesky(work, noise, scale)
    elif isinstance(support, type(constraints.corr_cholesky)):
        rows = torch.tril(work + scale * torch.tril(noise, diagonal=-1))
        norms = rows.norm(dim=-1, keepdim=True).clamp_min(_SUPPORT_EPS)
        moved = rows / norms
    elif isinstance(support, type(constraints.positive_definite)):
        factor, info = torch.linalg.cholesky_ex(work)
        if int(info.max().item()) != 0:
            return None
        perturbed = _perturb_lower_cholesky(factor, noise, scale)
        moved = perturbed @ perturbed.transpose(-2, -1)
    elif isinstance(support, type(constraints.real)):
        moved = work + scale * noise
    else:
        return None
    return moved.to(dtype=value.dtype)


def _observed_count_floor(
    section: dict[str, torch.Tensor],
    observe_names: frozenset[str],
) -> float:
    """Smallest value a scalar count parameter of `section` may take.

    A scalar integer covariate parameterises a bounded count family:
    it is the `total_count` of a Binomial or Beta-Binomial, the size
    of a Multinomial draw, the trial budget of a Negative-Binomial
    framing. Every one of those families supports the counts in
    `[0, total]`, so the largest count the fixture actually scores is
    the point below which the parameter cannot move without dropping
    an observed response out of the support.

    The floor reads the integer-valued `observe` binders only. A
    fractional binder is a real observation that no count parameter
    bounds, and a covariate is not scored at all.
    """
    floor = 0.0
    for name, value in section.items():
        if name not in observe_names or value.numel() == 0:
            continue
        if value.dtype.is_floating_point and not torch.equal(
            value,
            value.round(),
        ):
            continue
        floor = max(floor, float(value.max().item()))
    return floor


def _data_section_support(
    name: str,
    value: torch.Tensor,
    observe_names: frozenset[str],
    count_floor: float,
) -> constraints.Constraint | None:
    """Infer a constraint for data without a compiled stochastic site.

    Such entries are either covariates or ``observe`` binders folded
    into a grouped-marginalize closure. Floating covariates and
    fractional observations use real support. Integer vectors used as
    covariates are structural subscripts and remain fixed. Integer
    scalar covariates vary within ``[count_floor, 2 * value -
    count_floor]``. Integer observations vary within their attested
    ``[min, max]`` range. A singleton or empty range has no inferred
    perturbation.
    """
    if value.numel() == 0:
        return None
    if name not in observe_names:
        if value.dtype.is_floating_point:
            return constraints.real
        if value.dim() != 0:
            return None
        ground_truth = float(value.item())
        ceiling = 2.0 * ground_truth - count_floor
        if ceiling <= count_floor:
            return None
        return constraints.integer_interval(
            int(count_floor),
            int(ceiling),
        )
    if value.dtype.is_floating_point and not torch.equal(
        value,
        value.round(),
    ):
        return constraints.real
    lower = int(value.min().item())
    upper = int(value.max().item())
    if lower == upper:
        return None
    return constraints.integer_interval(lower, upper)


def _perturb_section(
    section: dict[str, torch.Tensor],
    supports: dict[str, constraints.Constraint],
    generator: torch.Generator,
    scale: float,
    *,
    infer_from_value: bool,
    observe_names: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(),
) -> dict[str, torch.Tensor]:
    """Perturb every value in `section` whose constraint is known.

    A name with a declared site support moves under that constraint. A
    name without one falls back to
    [`_data_section_support`][tests.transpile._gallery_data._data_section_support]
    when `infer_from_value` is set (the observations dict, which mixes
    observe sites with plain covariates), and otherwise stays at ground
    truth. Names in `exclude` are copied through untouched.
    """
    out: dict[str, torch.Tensor] = {}
    count_floor = (
        _observed_count_floor(section, observe_names) if infer_from_value else 0.0
    )
    for name, value in section.items():
        if name in exclude:
            out[name] = value
            continue
        support = supports.get(name)
        declared = support is not None
        if support is None and infer_from_value:
            support = _data_section_support(
                name,
                value,
                observe_names,
                count_floor,
            )
        if support is None:
            out[name] = value
            continue
        moved = _perturb_by_support(value, support, generator, scale, declared=declared)
        out[name] = value if moved is None else moved
    return out


def observations_for_point(
    dataset: GalleryDataset,
    point: Point,
) -> dict[str, torch.Tensor]:
    """Rebuild the pre-shaped observation dict the in-process
    [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] clamps with, for
    one point of a
    [`points_from_dataset`][tests.transpile._gallery_data.points_from_dataset]
    list.

    The probe's `observations` keyword preserves multi-axis shapes and
    takes precedence over the flat payload. Each entry is rebuilt from
    `data`, or from `params` when the point carries it as a latent, with
    the reference tensor's shape and dtype.
    """
    out: dict[str, torch.Tensor] = {}
    for name, reference in dataset.observations.items():
        if name in point.data:
            flat = point.data[name]
        elif name in point.params:
            flat = point.params[name]
        else:
            out[name] = reference
            continue
        values = (
            [float(flat)]
            if isinstance(flat, (int, float))
            else [float(v) for v in flat]
        )
        out[name] = (
            torch.tensor(values, dtype=torch.float64)
            .reshape(tuple(reference.shape))
            .to(dtype=reference.dtype)
        )
    return out


def observed_data_names(dataset: GalleryDataset) -> frozenset[str]:
    """Names a point carries as *observed* data.

    Three kinds of entry live in a point's `data` section, and only one
    of them is data the model conditions on:

    1. An `observe` binder or a covariate the program reads. These are
       the observed data.
    2. A sample site the `.md` snippet also clamps for its inference
       demo. The point files it under `params` (the latent spelling is
       canonical), so it is not observed data and is excluded here.
    3. A scalar type-parameter of the program. It is an *input* of the
       emitted program, fixed by the template instantiation, so it
       never varies and is excluded here too.
    """
    return frozenset(dataset.observations) - frozenset(dataset.params)


def structural_subscript_names(dataset: GalleryDataset) -> frozenset[str]:
    """Names of `dataset`'s observations that gather a parameter row
    rather than carry a coordinate of the support.

    The same classifier the perturber consults, published so a caller
    that wants to *exempt* a frozen coordinate has to justify the
    exemption against the harness's own reading of the entry rather
    than against a written claim. A subscript is an integer-dtyped
    vector the program only reads: it has one entry per plate row and
    each entry selects a parameter row, so stepping it gathers
    different parameters instead of moving the data. An `observe`
    binder is data however it is typed, a floating-point covariate is
    a real coordinate however integral its values, and a scalar
    integer covariate is a count parameter rather than a subscript,
    so none of the three is reported here.
    """
    return _structural_subscript_names(
        dataset.observations,
        dataset.observe_names,
    )


def varying_observation_names(
    dataset: GalleryDataset,
    points: list[Point],
) -> frozenset[str]:
    """The
    [`observed_data_names`][tests.transpile._gallery_data.observed_data_names]
    whose value actually differs somewhere in `points`.

    This distinguishes data variation from a joint that moved only
    because its latent values changed.
    """
    moved: set[str] = set()
    for name in observed_data_names(dataset):
        seen = {_wire_key(point.data[name]) for point in points if name in point.data}
        if len(seen) > 1:
            moved.add(name)
    return frozenset(moved)


def _wire_key(
    value: float | int | list[float] | list[int],
) -> tuple[float, ...]:
    """Hashable identity of one point-section entry, for equality
    comparison across the point set."""
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


def _qvr_log_density(
    dataset: GalleryDataset,
    point: Point,
    fixture: str,
) -> float:
    """Joint log-density of the compiled program at one point."""
    monadic = dataset.monadic
    if monadic is None:
        raise RuntimeError(f"{fixture!r}: no compiled MonadicProgram to score")
    result = QvrProbe().evaluate(
        b"",
        fixture,
        [point],
        # The in-process probe writes nothing; the path is required by
        # the probe Protocol and discarded on entry.
        scratch=_GALLERY_DOCS,
        monadic=monadic,
        x_input=dataset.x_input,
        observations=observations_for_point(dataset, point),
    )
    return result.log_densities[0]


def points_from_dataset(
    dataset: GalleryDataset,
    n_points: int = 6,
    seed: int = 0,
    scale: float = _PERTURBATION_SCALE,
) -> list[Point]:
    """Build a deterministic multi-point evaluation set for `dataset`.

    Point 0 is the captured ground truth. Later points perturb the
    latents, observed data, or both, following the schedule
    [`perturbation_labels`][tests.transpile._gallery_data.perturbation_labels]
    reports for the same length. Each value moves inside its own
    support. Positive scales move multiplicatively, bounded values in
    logit space, simplex rows are renormalized, and structured matrices
    retain their constraints. Integer values remain in their inferred
    ranges.
    A structural subscript, the one entry kind that names a parameter
    row rather than a point of the support, stays at ground truth; its
    row order is de-aliased at load time instead, by
    [`_dealias_row_order`][tests.transpile._gallery_data._dealias_row_order].

    Parameters
    ----------
    dataset
        The example's synthetic data and captured ground truth.
    n_points
        Total points to return, ground truth included. The default of
        6 gives two latents-only, two data-only, and one joint
        perturbation.
    seed
        Seed of a local [`torch.Generator`][torch.Generator]. The
        global RNG is never touched, so the point set is reproducible
        run to run and independent of whatever the example's
        synthetic-data snippet seeded.

        The default is 0. Sweeping
        [`GALLERY_SEEDS`][tests.transpile._gallery_data.GALLERY_SEEDS]
        through
        [`points_across_seeds`][tests.transpile._gallery_data.points_across_seeds]
        checks the perturbation design across independent draws.
    scale
        Excursion magnitude, in the natural unconstrained coordinate of
        each value's support, before the redraw ladder halves it. The
        default is
        [`PERTURBATION_SCALE`][tests.transpile._gallery_data.PERTURBATION_SCALE];
        [`WIDE_PERTURBATION_SCALE`][tests.transpile._gallery_data.WIDE_PERTURBATION_SCALE]
        is the wider excursion used by the strength tests.

    Returns
    -------
    list[Point]
        Exactly `n_points` points, in schedule order.

    Raises
    ------
    AssertionError
        When a perturbed point still scores a non-finite QVR joint
        after `_MAX_REDRAWS` attempts at successively halved scales.
    """
    labels = perturbation_labels(n_points)
    ground_truth = point_from_dataset(dataset)
    points: list[Point] = [ground_truth]
    if n_points == 1:
        return points

    fixture = "points_from_dataset"
    # Validate against the QVR joint only when the ground-truth point
    # itself scores finitely. An example whose oracle cannot score the
    # program at all (a non-deterministic composition marginalisation,
    # a free latent the snippet never bound) gives no baseline to
    # measure a perturbation against, so its points are emitted
    # unvalidated rather than measured against a broken reference.
    validate = False
    if dataset.monadic is not None:
        try:
            baseline = _qvr_log_density(dataset, ground_truth, fixture)
        except RuntimeError:
            validate = False
        else:
            validate = math.isfinite(baseline)

    generator = torch.Generator()
    generator.manual_seed(seed)
    supports = site_supports(dataset)
    # A name bound in both sections is a latent whose value the
    # snippet also clamps for its SVI demo. The latent spelling is
    # canonical, so the data pass leaves it alone and the latent pass
    # owns it.
    shared_names = frozenset(dataset.params) & frozenset(dataset.observations)

    for index in range(1, n_points):
        mode = labels[index]
        for attempt in range(_MAX_REDRAWS):
            attempt_scale = scale * (0.5**attempt)
            params = (
                _perturb_section(
                    dataset.params,
                    supports,
                    generator,
                    attempt_scale,
                    infer_from_value=False,
                )
                if mode in (PERTURB_LATENTS, PERTURB_BOTH)
                else dict(dataset.params)
            )
            observations = (
                _perturb_section(
                    dataset.observations,
                    supports,
                    generator,
                    attempt_scale,
                    infer_from_value=True,
                    observe_names=dataset.observe_names,
                    exclude=shared_names,
                )
                if mode in (PERTURB_DATA, PERTURB_BOTH)
                else dict(dataset.observations)
            )
            candidate = _point_from_tensors(
                params,
                observations,
                dataset.scalar_params,
                dataset.param_wires,
            )
            if not validate:
                points.append(candidate)
                break
            if math.isfinite(_qvr_log_density(dataset, candidate, fixture)):
                points.append(candidate)
                break
        else:
            raise AssertionError(
                f"point {index} ({mode}) at seed {seed}: every "
                f"perturbation attempt from scale {scale:.4g} down to "
                f"{scale * 0.5 ** (_MAX_REDRAWS - 1):.4g} left the QVR "
                f"joint non-finite, so no in-support point exists under "
                f"the constraints this module derived. Sites: "
                f"{sorted(supports)}."
            )
    return points


def points_across_seeds(
    dataset: GalleryDataset,
    n_points: int = 6,
    seeds: Sequence[int] = GALLERY_SEEDS,
    scale: float = _PERTURBATION_SCALE,
) -> dict[int, list[Point]]:
    """Independent point sets for `dataset`, one per entry of `seeds`.

    Each seed draws from the same perturbation design: every set uses
    the same schedule
    [`perturbation_labels`][tests.transpile._gallery_data.perturbation_labels]
    reports, at the same scale, under the same support constraints,
    and differs only in the sampled support points.

    Parameters
    ----------
    dataset
        The example's synthetic data and captured ground truth.
    n_points
        Points per set, ground truth included.
    seeds
        Seeds to draw at, defaulting to
        [`GALLERY_SEEDS`][tests.transpile._gallery_data.GALLERY_SEEDS].
    scale
        Excursion magnitude, shared by every set so the sets differ in
        draw alone.

    Returns
    -------
    dict[int, list[Point]]
        One point set per seed, keyed by the seed.

    Raises
    ------
    ValueError
        When `seeds` is empty or contains a duplicate.
    """
    ordered = list(seeds)
    if not ordered:
        raise ValueError(
            "points_across_seeds was given no seeds, so it would "
            "return an empty sweep that vacuously satisfies every "
            "claim made over it. Pass at least one seed."
        )
    duplicates = sorted({seed for seed in ordered if ordered.count(seed) > 1})
    if duplicates:
        raise ValueError(
            f"points_across_seeds was given repeated seed(s) "
            f"{duplicates!r}. A repeated seed yields a byte-identical "
            f"point set, so a sweep over it reports one draw as "
            f"several and overstates the coverage it achieved."
        )
    return {
        seed: points_from_dataset(
            dataset,
            n_points=n_points,
            seed=seed,
            scale=scale,
        )
        for seed in ordered
    }


def point_displacements(points: Sequence[Point]) -> list[float]:
    """Euclidean distance of each point from point 0, measured in the
    flat wire coordinate both evaluators are driven with.

    This measures the realized excursion after the redraw ladder may
    have reduced the requested scale. Both parameter and data sections
    contribute.

    Every coordinate the point carries under a name point 0 also
    carries is included; a name present at only one of the two is
    skipped, since a coordinate that appears and disappears has no
    displacement to measure.
    """
    if not points:
        return []
    origin = points[0]
    out: list[float] = []
    for point in points:
        total = 0.0
        for section, base in (
            (point.params, origin.params),
            (point.data, origin.data),
        ):
            for name, value in section.items():
                if name not in base:
                    continue
                here = _wire_key(value)
                there = _wire_key(base[name])
                if len(here) != len(there):
                    continue
                total += sum((a - b) ** 2 for a, b in zip(here, there))
        out.append(math.sqrt(total))
    return out


def point_excursion(points: Sequence[Point]) -> float:
    """Largest distance any point of the set travels from the ground
    truth, in the flat wire coordinate.

    Returns 0.0 for an empty or single-point set.
    """
    displacements = point_displacements(points)
    return max(displacements) if displacements else 0.0


__all__ = [
    "GALLERY_SEEDS",
    "GalleryDataset",
    "PERTURB_BOTH",
    "PERTURB_DATA",
    "PERTURB_GROUND_TRUTH",
    "PERTURB_LATENTS",
    "PERTURBATION_SCALE",
    "PROBE_SCRATCH_PARENT",
    "PROBE_SCRIPT_DIR",
    "WIDE_PERTURBATION_SCALE",
    "assert_probe_scripts_unchanged",
    "gallery_examples_with_data",
    "is_discrete_support",
    "load_gallery_data",
    "md_path_for",
    "observations_for_point",
    "observed_data_names",
    "perturbation_labels",
    "point_displacements",
    "point_excursion",
    "point_from_dataset",
    "points_across_seeds",
    "points_from_dataset",
    "probe_scratch",
    "probe_scratch_root",
    "probe_script_digests",
    "structural_subscript_names",
    "sweep_abandoned_probe_roots",
    "site_supports",
    "varying_observation_names",
]
