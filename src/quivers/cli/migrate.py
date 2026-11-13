"""``qvr migrate`` command.

Lowers ``.qvr`` source forward along the QVR grammar release chain
declared in [`quivers.cli.migrations`][quivers.cli.migrations].

Surface::

    qvr migrate path/to/file.qvr [paths...]
    qvr migrate --from v0.10.0 --to HEAD --dry-run docs/examples/source/
    qvr migrate --output OUT_DIR --to HEAD path/to/file.qvr

``--from`` is required because existing QVR source does not carry an
unambiguous grammar-version marker; ``--to`` defaults to ``HEAD``. Both must be members of
`quivers.cli.migrations.CHAIN`; the CLI composes the
intermediate adjacent-pair migrators automatically, so adding a new
release is purely additive to the migrations package.

``--dry-run`` reports which files would change without writing
output. ``--output DIR`` writes migrated copies under ``DIR``
instead of overwriting the originals.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import didactic.api as dx

from quivers.cli.migrations import (
    MigrationError,
    compose_migration,
    vcs_coverage_report,
)
from quivers.cli.migrations._grammar import HistoricalGrammarUnavailable


class MigrateError(Exception):
    """Raised by the migrate CLI on a recoverable user-facing error."""


class _MigrationInput(dx.Model):
    """One source path and its relative migration destination.

    Parameters
    ----------
    source : Path
        Input source file.
    relative_output : Path
        Destination relative to the requested output directory.
    """

    source: Path
    relative_output: Path


def _atomic_write(
    target: Path,
    data: bytes,
    *,
    source_mode: int | None = None,
) -> None:
    """Validate and atomically replace ``target`` with ``data``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != data:
            raise OSError(f"staged migration write was incomplete: {temporary}")
        if target.exists():
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
        elif source_mode is not None:
            os.chmod(temporary, stat.S_IMODE(source_mode))
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                # Directory fsync is unavailable on some platforms. The
                # sibling-file fsync and atomic replacement have completed.
                pass
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _walk_inputs(
    paths: Iterable[str],
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> list[_MigrationInput]:
    """Expand inputs while retaining each file's relative topology."""
    out: list[_MigrationInput] = []
    excluded = tuple(root.resolve(strict=False) for root in excluded_roots)

    def is_excluded(path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return any(
            resolved == root or resolved.is_relative_to(root) for root in excluded
        )

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for current, directories, files in os.walk(p):
                current_path = Path(current)
                directories[:] = sorted(
                    directory
                    for directory in directories
                    if not is_excluded(current_path / directory)
                )
                for filename in sorted(files):
                    child = current_path / filename
                    if child.suffix == ".qvr" and not is_excluded(child):
                        out.append(
                            _MigrationInput(
                                source=child,
                                relative_output=child.relative_to(p),
                            )
                        )
        elif p.suffix == ".qvr":
            if not p.is_file():
                raise MigrateError(f"input file does not exist: {raw!r}")
            out.append(_MigrationInput(source=p, relative_output=Path(p.name)))
        else:
            raise MigrateError(
                f"unsupported input {raw!r}: not a directory or ``.qvr`` file",
            )
    return out


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by the top-level ``qvr`` dispatcher.

    Returns 0 when every file migrated cleanly; 2 on invalid input.
    """
    # ``--check`` runs the panproto-VCS coverage check across the
    # migration chain. For each adjacent (from, to) pair, computes
    # the schema diff and reports any source rule removed at the
    # target whose hop migrator has no converter. Non-zero exit
    # if any pair has uncovered removed rules.
    if getattr(args, "check", False):
        try:
            reports = vcs_coverage_report()
        except (MigrationError, HistoricalGrammarUnavailable) as exc:
            print(f"qvr migrate: {exc}", file=sys.stderr)
            return 2
        any_uncovered = False
        for r in reports:
            print(r.format())
            if r.uncovered_removed:
                any_uncovered = True
        return 1 if any_uncovered else 0

    if args.from_ref is None:
        print(
            "qvr migrate: --from is required because QVR source files do not "
            "encode an unambiguous grammar version",
            file=sys.stderr,
        )
        return 2
    from_ref = args.from_ref
    to_ref = args.to_ref or "HEAD"
    out_root = Path(args.output) if args.output is not None else None

    if out_root is not None:
        resolved_output = out_root.resolve(strict=False)
        for raw in args.paths:
            input_path = Path(raw)
            if (
                input_path.is_dir()
                and input_path.resolve(strict=False) == resolved_output
            ):
                print(
                    "qvr migrate: --output cannot be the same directory as an input",
                    file=sys.stderr,
                )
                return 2

    try:
        migrate_fn = compose_migration(from_ref, to_ref)
        inputs = _walk_inputs(
            args.paths,
            excluded_roots=(out_root,) if out_root is not None else (),
        )
    except (MigrateError, MigrationError) as exc:
        print(f"qvr migrate: {exc}", file=sys.stderr)
        return 2

    if not inputs:
        print("qvr migrate: no .qvr files to migrate", file=sys.stderr)
        return 2

    if out_root is not None:
        source_paths = {
            item.source.resolve(strict=False): item.source for item in inputs
        }
        destinations: dict[Path, Path] = {}
        for item in inputs:
            destination = (out_root / item.relative_output).resolve(strict=False)
            aliased_source = source_paths.get(destination)
            if aliased_source is not None:
                print(
                    f"qvr migrate: output {out_root / item.relative_output} "
                    f"aliases input {aliased_source}",
                    file=sys.stderr,
                )
                return 2
            prior = destinations.get(destination)
            if prior is not None and prior != item.source:
                print(
                    f"qvr migrate: output collision: {prior} and {item.source} "
                    f"both map to {out_root / item.relative_output}",
                    file=sys.stderr,
                )
                return 2
            destinations[destination] = item.source

    changed = 0
    copied = 0
    for item in inputs:
        src_path = item.source
        try:
            source = src_path.read_bytes()
            source_mode = stat.S_IMODE(src_path.stat().st_mode)
        except OSError as exc:
            print(f"qvr migrate: {src_path}: {exc}", file=sys.stderr)
            return 2
        try:
            migrated = migrate_fn(source)
        except (
            HistoricalGrammarUnavailable,
            MigrateError,
            MigrationError,
        ) as exc:
            print(f"qvr migrate: {src_path}: {exc}", file=sys.stderr)
            return 2
        did_change = migrated != source
        if did_change:
            changed += 1
            prefix = "would migrate" if args.dry_run else "migrated"
            print(f"{prefix} {src_path}")
        if args.dry_run:
            continue
        if out_root is None and not did_change:
            continue
        target = out_root / item.relative_output if out_root is not None else src_path
        try:
            _atomic_write(target, migrated, source_mode=source_mode)
        except OSError as exc:
            print(f"qvr migrate: {target}: {exc}", file=sys.stderr)
            return 2
        if out_root is not None:
            copied += 1
            if not did_change:
                print(f"copied {src_path}")

    if changed == 0 and copied == 0:
        print(
            f"qvr migrate: {len(inputs)} file(s) already at {to_ref}",
        )
    return 0


__all__ = []
