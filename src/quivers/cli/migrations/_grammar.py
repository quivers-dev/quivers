"""Load immutable QVR parser snapshots from wheels or source checkouts."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path

import panproto

from quivers.cli.migrations._assets import candidate_vcs_roots, is_packaged_vcs_path
from quivers.cli.migrations._manifest import (
    GRAMMAR_ASSETS,
    REVISION_SNAPSHOTS,
    GrammarAsset,
    compiled_inputs_sha256,
    snapshot_for,
    verify_snapshot,
)
from quivers.dsl._grammar_build import _validate_native_manifest


_LIB_KEEPALIVE: dict[str, ctypes.CDLL] = {}
_REGISTRIES: dict[str, object] = {}


class HistoricalGrammarUnavailable(RuntimeError):
    """An immutable grammar snapshot cannot be loaded or compiled."""


def _shared_lib_extension() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _snapshot_dir(revision: str) -> Path:
    try:
        snapshot = snapshot_for(revision)
    except ValueError as error:
        raise HistoricalGrammarUnavailable(str(error)) from error
    searched: list[Path] = []
    for root in candidate_vcs_roots():
        candidate = root / "parsers" / snapshot
        searched.append(candidate)
        src = candidate / "source" / "src"
        if (src / "parser.c").is_file():
            try:
                verify_snapshot(snapshot, src)
            except (OSError, ValueError) as error:
                raise HistoricalGrammarUnavailable(str(error)) from error
            return candidate
    raise HistoricalGrammarUnavailable(
        f"immutable QVR parser source for {revision} is unavailable; searched "
        + ", ".join(str(path) for path in searched),
    )


def available_revisions() -> tuple[str, ...]:
    """Return revisions backed by an immutable parser source snapshot."""
    revisions: set[str] = set()
    for root in candidate_vcs_roots():
        parsers = root / "parsers"
        if not parsers.is_dir():
            continue
        for child in parsers.iterdir():
            if (child / "source" / "src" / "parser.c").is_file():
                revisions.add(child.name)
    for alias, target in REVISION_SNAPSHOTS.items():
        if target in revisions:
            revisions.add(alias)
    return tuple(sorted(revisions))


def _source_fingerprint(src_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(compiled_inputs_sha256(src_dir).encode("ascii"))
    for name in ("grammar.json", "node-types.json"):
        path = src_dir / name
        digest.update(name.encode("ascii"))
        digest.update(path.read_bytes())
    digest.update(platform.machine().encode("utf-8"))
    digest.update(sys.platform.encode("ascii"))
    return digest.hexdigest()[:24]


def grammar_identity(revision: str) -> GrammarAsset:
    """Return a verified parser-source identity for ``revision``."""
    _snapshot_dir(revision)
    snapshot = snapshot_for(revision)
    # `_snapshot_dir` verified the bytes before returning. The immutable
    # manifest value is the identity used by release-edge checks.
    return GRAMMAR_ASSETS[snapshot]


def _compiler_for_platform(platform_name: str | None = None) -> str | None:
    platform_name = platform_name or sys.platform
    candidates = (
        ("cl", "clang", "gcc", "cc")
        if platform_name == "win32"
        else ("cc", "clang", "gcc")
    )
    return next((path for name in candidates if (path := shutil.which(name))), None)


def _compile_command(
    compiler: str,
    src_dir: Path,
    sources: tuple[Path, ...],
    output: Path,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> list[str]:
    """Build the platform/compiler-specific shared-library command.

    On macOS the system compiler is a universal binary that builds for
    the architecture of the process that runs it, which need not be the
    interpreter's own (a build started under Rosetta, say); the command
    names the interpreter's architecture so the library always loads.

    Parameters
    ----------
    compiler : str
        The compiler executable.
    src_dir : Path
        The directory of the parser's headers.
    sources : tuple[Path, ...]
        The C sources.
    output : Path
        The shared library to write.
    platform_name : str | None
        The platform, ``sys.platform`` by default.
    machine : str | None
        The architecture the library is loaded on,
        ``platform.machine()`` by default.

    Returns
    -------
    list[str]
        The command.
    """
    platform_name = platform_name or sys.platform
    machine = machine or platform.machine()
    compiler_name = Path(compiler).name.lower()
    if platform_name == "win32" and compiler_name in {"cl", "cl.exe"}:
        return [
            compiler,
            "/nologo",
            "/O2",
            "/LD",
            f"/I{src_dir}",
            *(str(path) for path in sources),
            "/link",
            f"/OUT:{output}",
        ]
    flags = ["-O2", "-shared"]
    if platform_name != "win32":
        flags.append("-fPIC")
    if platform_name == "darwin":
        flags.extend(["-arch", machine])
    return [
        compiler,
        *flags,
        "-I",
        str(src_dir),
        *(str(path) for path in sources),
        "-o",
        str(output),
    ]


@contextmanager
def _build_lock(cache_dir: Path) -> Iterator[None]:
    """Serialize cross-process builds of one content-addressed parser."""
    lock_path = cache_dir / "build.lock"
    with lock_path.open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _compile_parser(snapshot_dir: Path, revision: str) -> Path:
    src_dir = snapshot_dir / "source" / "src"
    compiler = _compiler_for_platform()
    if compiler is None:
        raise HistoricalGrammarUnavailable(
            f"QVR parser {revision} has no prebuilt library for {sys.platform}; "
            "install a supported C compiler (MSVC cl, clang, gcc, or cc) to "
            "compile the packaged immutable source snapshot",
        )

    # The library is specific to the architecture it is built for, so
    # each architecture has its own build of a source snapshot.
    cache_dir = (
        Path(tempfile.gettempdir())
        / "quivers-qvr-parsers"
        / f"{_source_fingerprint(src_dir)}-{platform.machine()}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"qvr{_shared_lib_extension()}"
    if output.is_file():
        return output

    sources = [src_dir / "parser.c"]
    scanner = src_dir / "scanner.c"
    if scanner.is_file():
        sources.append(scanner)
    with _build_lock(cache_dir):
        if output.is_file():
            return output
        build_dir = Path(tempfile.mkdtemp(prefix="build-", dir=cache_dir))
        temporary = build_dir / f"qvr{_shared_lib_extension()}"
        command = _compile_command(
            compiler,
            src_dir,
            tuple(sources),
            temporary,
        )
        try:
            subprocess.run(
                command,
                cwd=build_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            if not temporary.is_file():
                raise OSError(f"compiler did not create {temporary}")
            os.replace(temporary, output)
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = (
                exc.stderr.strip()
                if isinstance(exc, subprocess.CalledProcessError)
                else str(exc)
            )
            raise HistoricalGrammarUnavailable(
                f"failed to compile immutable QVR parser {revision} with "
                f"{compiler}: {detail}",
            ) from exc
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)
    return output


def _library_for(snapshot_dir: Path, revision: str) -> Path:
    bundled = snapshot_dir / f"qvr{_shared_lib_extension()}"
    manifest = snapshot_dir / "native-manifest.json"
    packaged = is_packaged_vcs_path(snapshot_dir)
    if bundled.is_file() and manifest.is_file():
        try:
            _validate_native_manifest(snapshot_dir / "source", bundled, manifest)
        except (OSError, ValueError) as error:
            raise HistoricalGrammarUnavailable(
                f"bundled QVR parser {revision} failed integrity validation: {error}"
            ) from error
        return bundled
    if packaged:
        raise HistoricalGrammarUnavailable(
            f"this quivers installation has incomplete native parser assets for "
            f"{revision}; install a matching platform wheel"
        )
    return _compile_parser(snapshot_dir, revision)


def registry_for(revision: str) -> object:
    """Return a panproto registry for an immutable grammar revision."""
    if revision in _REGISTRIES:
        return _REGISTRIES[revision]

    snapshot_dir = _snapshot_dir(revision)
    library_path = _library_for(snapshot_dir, revision)
    library = ctypes.CDLL(str(library_path))
    library.tree_sitter_qvr.argtypes = []
    library.tree_sitter_qvr.restype = ctypes.c_void_p
    language_ptr = library.tree_sitter_qvr()
    if not language_ptr:
        raise HistoricalGrammarUnavailable(
            f"tree_sitter_qvr() returned NULL for {revision}",
        )

    src_dir = snapshot_dir / "source" / "src"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        registry = panproto.AstParserRegistry()
    registry.override_grammar(
        name="qvr",
        extensions=["qvr"],
        language_ptr=language_ptr,
        node_types=(src_dir / "node-types.json").read_bytes(),
        grammar_json=(src_dir / "grammar.json").read_bytes(),
    )
    _LIB_KEEPALIVE[revision] = library
    _REGISTRIES[revision] = registry
    return registry


def parse(revision: str, source: bytes) -> object:
    """Parse bytes through an immutable historical grammar."""
    return registry_for(revision).lens("qvr").parse(source)


__all__ = [
    "HistoricalGrammarUnavailable",
    "available_revisions",
    "grammar_identity",
    "parse",
    "registry_for",
]
