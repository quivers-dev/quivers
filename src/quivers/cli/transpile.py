"""``qvr transpile`` subcommand: emit a parsed QVR file as source for
another PPL.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from didactic.codegen._emitter import lookup_emitter

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, available_targets, transpile

if TYPE_CHECKING:
    import argparse


def main(args: argparse.Namespace) -> int:

    if getattr(args, "list_targets", False):
        for target in available_targets():
            print(target)
        return 0

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"qvr transpile: {file_path}: no such file", flush=True)
        return 2

    try:
        module = parse(file_path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"qvr transpile: parse failed: {e}", flush=True)
        return 1

    targets: list[str]
    if args.to_all:
        targets = available_targets()
    elif args.to is not None:
        targets = [args.to]
    else:
        print(
            "qvr transpile: pass --to <target> or --to-all "
            f"(available: {', '.join(available_targets())})",
            flush=True,
        )
        return 2

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for target in targets:
        try:
            bytes_out = transpile(module, target=target)
        except UnsupportedConstruct as e:
            print(f"qvr transpile [{target}]: {e}", flush=True)
            failures += 1
            continue
        except LookupError as e:
            print(f"qvr transpile [{target}]: {e}", flush=True)
            failures += 1
            continue

        if out_dir is not None:
            ext = _extension_for(target)
            # With --to-all, multiple backends share an extension (Python:
            # numpyro/pyro/pymc/edward2). Suffix the stem with the backend
            # name so they don't collide.
            stem = f"{file_path.stem}.{target}" if args.to_all else file_path.stem
            out_path = out_dir / f"{stem}.{ext}"
            out_path.write_bytes(bytes_out)
            print(f"qvr transpile [{target}]: wrote {out_path}", flush=True)
        elif args.output is not None:
            Path(args.output).write_bytes(bytes_out)
            print(
                f"qvr transpile [{target}]: wrote {args.output}",
                flush=True,
            )
        else:
            sys.stdout.buffer.write(bytes_out)
            sys.stdout.buffer.write(b"\n")

    return 1 if failures else 0


def _extension_for(target: str) -> str:
    """Look up the registered backend's `file_extension`, falling back
    to the target name."""

    emitter = lookup_emitter(f"qvr-{target}")
    if emitter is None:
        return target
    return getattr(emitter, "file_extension", target)


__all__ = []
