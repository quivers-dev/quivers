"""Docker container driver for out-of-process probes.

Each Tier-3 probe runs in its own pinned image, invoked once per
``(fixture, point_set)``. The driver here centralises the bind-mount
+ argv shape so per-backend probes only need to specify the image
tag and their entrypoint script.

The host's `docker` CLI is the dependency. When `docker` is not on
PATH or the host daemon is not running, [`docker_available`][.]
returns False and the calling test should `pytest.skip`.

Each image carries a corresponding script under
[`tests/transpile/probes/_scripts/<backend>.py`][tests.transpile.probes._scripts]
that reads `/io/source.<ext>`, `/io/points.json`, and emits
`/io/result.json` (a JSON object with one ``log_densities`` key).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Mapping

from quivers.transpile.plan import target_name


#: Whether the daemon has answered once this process; a daemon that has
#: answered is not asked again, so a slow `docker info` under the load of
#: the probe containers cannot read as an absent daemon mid-session.
_DAEMON_ANSWERED = False

#: The image tags found present this process; an image does not vanish
#: mid-session, so a slow `docker images` under load cannot read as an
#: absent image once the image has been seen.
_IMAGES_SEEN: set[str] = set()


def docker_available() -> bool:
    """True iff the `docker` CLI is on PATH and the daemon answers `docker info`.

    Returns
    -------
    bool
        Whether the daemon is reachable; once it has answered, True for
        the rest of the process.
    """
    global _DAEMON_ANSWERED
    if _DAEMON_ANSWERED:
        return True
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired, OSError:
        return False
    _DAEMON_ANSWERED = completed.returncode == 0
    return _DAEMON_ANSWERED


def image_available(tag: str) -> bool:
    """True iff a Docker image matching ``tag`` is built or pulled locally.

    Uses ``docker images --filter reference=<tag>`` rather than
    ``docker image inspect`` because Docker Desktop 28+ has a daemon
    bug where ``image inspect <name>`` and ``image inspect
    <name>:latest`` both raise ``No such image`` even when ``docker
    images`` lists the image. The filter-form query goes through a
    different daemon path and reliably returns the image when it
    exists.

    Parameters
    ----------
    tag : str
        The image tag.

    Returns
    -------
    bool
        Whether the image is present; once seen, True for the rest of
        the process.
    """
    if tag in _IMAGES_SEEN:
        return True
    if not docker_available():
        return False
    try:
        completed = subprocess.run(
            [
                "docker",
                "images",
                "--filter",
                f"reference={tag}",
                "--format",
                "{{.ID}}",
            ],
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired, OSError:
        return False
    if completed.returncode != 0:
        return False
    present = bool(completed.stdout.strip())
    if present:
        _IMAGES_SEEN.add(tag)
    return present


#: Environment variable naming the directory that holds memoised probe
#: results. Unset, nothing is cached and every call runs its container.
PROBE_CACHE_ENV = "QUIVERS_PROBE_CACHE"


def probe_cache_dir() -> pathlib.Path | None:
    """The probe cache directory, or None when caching is off.

    Caching is opt-in rather than the default. A memoised measurement
    is only as trustworthy as the key that selects it, and a suite that
    gates a release should be able to run every container for real; the
    variable is set on the tiers where turnaround matters and left
    unset where it does not.
    """
    configured = os.environ.get(PROBE_CACHE_ENV)
    if not configured:
        return None
    path = pathlib.Path(configured).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_id(tag: str) -> str | None:
    """The local image ID for `tag`, or None when it cannot be read.

    The cache keys on this rather than on the tag. A tag is a mutable
    pointer: rebuilding `panproto-test-julia` with a changed Dockerfile
    leaves the name identical and every cached measurement taken under
    the old image wrong.
    """
    completed = subprocess.run(
        ["docker", "images", "--filter", f"reference={tag}", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    ids = completed.stdout.split()
    return ids[0] if ids else None


def _probe_cache_key(
    *,
    image: str,
    resolved_image_id: str,
    source: bytes,
    source_ext: str,
    points: list[dict],
    shapes: dict[str, list[int]] | None,
    dtypes: dict[str, str] | None,
    script: pathlib.Path,
    side_files: Mapping[str, bytes],
) -> str:
    """Digest every input that can move the measurement.

    That is the image the code runs under, the emitted source and its
    extension, the points, the shape and dtype side-tables, the probe
    script together with the reshape helpers copied in beside it, and
    every other file the caller placed in the mounted directory. A
    probe reads its instructions from that directory, so a file there
    that the key ignored (the list of names to export, for one) would
    let a measurement taken without it answer a call made with it.
    Anything omitted here is something that can change while the cache
    keeps answering with a stale number, so files are hashed by content
    as well as by name.

    Parameters
    ----------
    image : str
        The image tag.
    resolved_image_id : str
        The image identifier the tag resolves to.
    source : bytes
        The emitted program.
    source_ext : str
        The program's file extension.
    points : list[dict]
        The points the probe scores.
    shapes : dict[str, list[int]] | None
        The shape side-table.
    dtypes : dict[str, str] | None
        The dtype side-table.
    script : pathlib.Path
        The probe script.
    side_files : Mapping[str, bytes]
        The other files in the mounted directory, by relative path.

    Returns
    -------
    str
        The hexadecimal digest.
    """
    digest = hashlib.sha256()
    parts: list[bytes] = [
        image.encode(),
        resolved_image_id.encode(),
        source,
        source_ext.encode(),
        json.dumps(points, sort_keys=True).encode(),
        json.dumps(shapes, sort_keys=True).encode(),
        json.dumps(dtypes, sort_keys=True).encode(),
        script.name.encode(),
        script.read_bytes(),
    ]
    for reshape_name in ("_reshape.py", "_reshape.jl"):
        reshape_path = script.parent / reshape_name
        parts.append(reshape_name.encode())
        parts.append(reshape_path.read_bytes() if reshape_path.exists() else b"")
    for name in sorted(side_files):
        parts.append(name.encode())
        parts.append(side_files[name])
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def spelled_keys[ValueT](table: dict[str, ValueT]) -> dict[str, ValueT]:
    """A name-keyed table under the targets' spelling of each name.

    Parameters
    ----------
    table : dict[str, ValueT]
        The table, keyed by names as the reference machine labels them.

    Returns
    -------
    dict[str, ValueT]
        The same table keyed by
        [`target_name`][quivers.transpile.plan.target_name] of each
        name, so a site a program draw renamed (``theta$z``) reaches
        the emitted program under the identifier it declares.
    """
    return {target_name(name): value for name, value in table.items()}


def spelled_points(points: list[dict]) -> list[dict]:
    """Points with their parameter and data names spelled for the targets.

    Parameters
    ----------
    points : list[dict]
        The points, each a ``params`` and a ``data`` table.

    Returns
    -------
    list[dict]
        The points with every name under the targets' spelling.
    """
    return [
        {
            key: spelled_keys(value) if key in ("params", "data") else value
            for key, value in point.items()
        }
        for point in points
    ]


def _side_files(scratch: pathlib.Path, source_ext: str) -> dict[str, bytes]:
    """The files in a probe's mounted directory that the caller put there.

    Parameters
    ----------
    scratch : pathlib.Path
        The directory mounted into the container.
    source_ext : str
        The extension of the emitted program, which names the source
        file `run_probe` writes.

    Returns
    -------
    dict[str, bytes]
        Each file's content by its path relative to `scratch`, leaving
        out the files `run_probe` writes from its own arguments and the
        result the probe writes back.
    """
    own = {
        f"source.{source_ext}",
        "points.json",
        "shapes.json",
        "dtypes.json",
        "probe.py",
        "_reshape.py",
        "_reshape.jl",
        "result.json",
    }
    return {
        path.relative_to(scratch).as_posix(): path.read_bytes()
        for path in sorted(scratch.rglob("*"))
        if path.is_file() and path.relative_to(scratch).as_posix() not in own
    }


def run_probe(
    *,
    image: str,
    script: pathlib.Path,
    source: bytes,
    source_ext: str,
    points: list[dict],
    scratch: pathlib.Path,
    timeout: float = 120.0,
    shapes: dict[str, list[int]] | None = None,
    dtypes: dict[str, str] | None = None,
) -> dict:
    """Invoke a probe image against ``source`` + ``points``.

    Layout under the bind-mounted ``scratch`` directory:

    ```
    /io/source.<ext>   # transpiled source bytes
    /io/points.json    # list[Point.dict()]
    /io/shapes.json    # {name -> [dim, ...]} (omitted when shapes is None)
    /io/dtypes.json    # {name -> "int" | "float"} (omitted when dtypes is None)
    /io/probe.py       # the entrypoint script for this backend
    /io/_reshape.py    # the shared reshape helper (Python probes import it)
    /io/_reshape.jl    # the shared reshape helper (Julia probes include it)
    /io/result.json    # probe writes this
    ```

    Probes that need to reshape flat-list `Point` values back into
    multi-dim arrays read `shapes.json` (and optionally `dtypes.json`).
    When the caller omits either file, the probe falls through to its
    legacy scalar / list contract.

    The container is launched read-only against /io except for the
    result file; this lets the host harness reason about side effects.

    Returns the decoded JSON object the script wrote to result.json.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    source_path = scratch / f"source.{source_ext}"
    source_path.write_bytes(source)
    (scratch / "points.json").write_text(json.dumps(spelled_points(points)))
    if shapes is not None:
        (scratch / "shapes.json").write_text(json.dumps(spelled_keys(shapes)))
    if dtypes is not None:
        (scratch / "dtypes.json").write_text(json.dumps(spelled_keys(dtypes)))
    (scratch / "probe.py").write_bytes(script.read_bytes())
    # The per-backend probe scripts import a shared reshape helper that
    # sits beside them in `_scripts/`: Python probes do `from _reshape
    # import ...` and Julia probes `include("/io/_reshape.jl")`. Copy
    # both reshape modules from the script's own directory into `/io/`
    # so the container-side import resolves whatever the probe language.
    for reshape_name in ("_reshape.py", "_reshape.jl"):
        reshape_path = script.parent / reshape_name
        if reshape_path.exists():
            (scratch / reshape_name).write_bytes(reshape_path.read_bytes())
    result_path = scratch / "result.json"
    if result_path.exists():
        result_path.unlink()

    # Each per-backend image carries `ENTRYPOINT ["python"]` (Python
    # backends) or its target-language equivalent (`julia`, `node`,
    # `jags`); the container takes the script path as the single
    # post-entrypoint argument. Do NOT also pass `python` here -- the
    # entrypoint already provides it, so prefixing would run
    # `python python /io/probe.py` and the container fails with
    # "no such file: /io/python".
    argv = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{scratch.resolve()}:/io",
        "-w",
        "/io",
        "-e",
        f"FIXTURE_EXT={source_ext}",
        image,
        "/io/probe.py",
    ]
    cache_dir = probe_cache_dir()
    cache_path: pathlib.Path | None = None
    if cache_dir is not None:
        resolved = image_id(image)
        if resolved is not None:
            cache_path = cache_dir / (
                _probe_cache_key(
                    image=image,
                    resolved_image_id=resolved,
                    source=source,
                    source_ext=source_ext,
                    points=points,
                    shapes=shapes,
                    dtypes=dtypes,
                    script=script,
                    side_files=_side_files(scratch, source_ext),
                )
                + ".json"
            )
            if cache_path.exists():
                # The inputs above are already on disk, so a caller that
                # inspects the scratch sees what this call wrote either
                # way. What the hit skips is the container, and the key
                # says the container ran on exactly these bytes under
                # exactly this image.
                cached = json.loads(cache_path.read_text())
                result_path.write_text(json.dumps(cached))
                return cached

    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"probe container {image!r} exited {completed.returncode}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )
    if not result_path.exists():
        raise RuntimeError(
            f"probe container {image!r} did not write /io/result.json\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )
    result = json.loads(result_path.read_text())
    if cache_path is not None:
        # Write through a unique temporary name and rename, so two
        # xdist workers finishing the same cell cannot leave a reader
        # holding half a file.
        staged = cache_path.with_suffix(f".{os.getpid()}.tmp")
        staged.write_text(json.dumps(result))
        staged.replace(cache_path)
    return result


__all__ = [
    "PROBE_CACHE_ENV",
    "docker_available",
    "image_available",
    "image_id",
    "probe_cache_dir",
    "run_probe",
]


def run_probe_script(
    *,
    image: str,
    script: pathlib.Path,
    scratch: pathlib.Path,
    timeout: float = 900.0,
) -> dict | list:
    """Run one probe script against whatever the caller placed in ``scratch``.

    The script is copied to ``/io/<name>`` beside the caller's inputs and
    launched through the image's entrypoint; it writes ``/io/result.json``.

    Parameters
    ----------
    image : str
        The probe image tag.
    script : pathlib.Path
        The probe script; its language must match the image's entrypoint.
    scratch : pathlib.Path
        The bind-mounted directory holding the script's inputs.
    timeout : float
        Seconds before the container is killed.

    Returns
    -------
    dict | list
        The decoded ``result.json``.

    Raises
    ------
    AssertionError
        If the container exits nonzero, with the tail of its stderr.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / script.name).write_bytes(script.read_bytes())
    result_path = scratch / "result.json"
    if result_path.exists():
        result_path.unlink()
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{scratch.resolve()}:/io",
            "-w",
            "/io",
            image,
            f"/io/{script.name}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(result_path.read_text())
