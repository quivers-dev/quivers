"""Atomic sibling-file replacement shared by source and analysis artifacts."""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import tempfile


@contextmanager
def atomic_output(target: Path, *, source_mode: int | None = None) -> Iterator[Path]:
    """Yield a sibling staging path, then sync and atomically replace its target.

    Parameters
    ----------
    target : Path
        Final destination; parent directories are created as needed.
    source_mode : int or None
        Permission bits for a new file. Existing destination permissions prevail.

    Yields
    ------
    Path
        Unique temporary sibling, removed if writing or replacement fails.

    Raises
    ------
    OSError
        If staging, file synchronization, or replacement fails. Directory
        synchronization is best effort where the platform does not support it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        if target.exists():
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
        elif source_mode is not None:
            os.chmod(temporary, stat.S_IMODE(source_mode))
        os.replace(temporary, target)
        try:
            directory = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(directory)
            except OSError:
                pass
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
