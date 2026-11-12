"""Checkpoints of an MCMC run, written atomically and read without code.

A checkpoint is one file, ``mcmc_checkpoint.pt`` in the directory an
`MCMCCheckpoint` names. It holds a header, an `_CheckpointHeader` (a
didactic model, stored as JSON) that identifies the file's format, the
model and data, the kernel's configuration, and the run's settings,
and the state of every chain: its phase and iteration, its
`KernelState`, its kernel's `MCMCKernel.state_dict`, its generator's
state, and the draws it has collected. The file is written with
`torch.save` and read with ``torch.load(..., weights_only=True)``,
which reconstructs tensors and plain containers only, so reading a
checkpoint executes no code.

Every write goes to a temporary file in the same directory, is flushed
to disk, and then atomically replaces the previous checkpoint
(`os.replace`), so an interruption mid-write leaves the previous
checkpoint intact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import didactic.api as dx
import torch

from quivers.inference.mcmc.kernel import StateValue


CHECKPOINT_FORMAT_VERSION = 1
"""The version of the checkpoint layout this library writes and reads.
`MCMC.resume` refuses a checkpoint of another version."""

CHECKPOINT_FILENAME = "mcmc_checkpoint.pt"
"""The name of the checkpoint file inside an `MCMCCheckpoint` directory."""


class MCMCCheckpoint(dx.Model):
    """Where and how often `MCMC` checkpoints its chains.

    A checkpoint is written after every ``every`` transitions of a chain
    (warmup and sampling alike), after ``every_seconds`` of wall-clock
    time since the last write, whichever comes first, and always when a
    chain finishes or a budget stops it. `MCMC.resume` continues a run
    from the checkpoint, bit for bit as if it had not been interrupted.

    Attributes
    ----------
    directory : str
        The directory the checkpoint file is written to; created if
        absent.
    every : int | None
        Transitions between writes; ``None`` writes only on the time
        criterion and at chain ends. Default ``100``.
    every_seconds : float | None
        Seconds between writes; ``None`` disables the time criterion.

    Raises
    ------
    didactic.api.ValidationError
        If ``directory`` is empty, ``every`` is less than one, or
        ``every_seconds`` is not positive.
    """

    directory: str
    every: int | None = 100
    every_seconds: float | None = None

    @dx.model_validator()
    def _check(self) -> None:
        """Refuse an empty directory and non-positive intervals.

        Raises
        ------
        ValueError
            If a field is out of range.
        """
        if not self.directory:
            raise ValueError("MCMCCheckpoint: directory must not be empty")
        if self.every is not None and self.every < 1:
            raise ValueError(f"MCMCCheckpoint: every must be >= 1, got {self.every}")
        if self.every_seconds is not None and not self.every_seconds > 0:
            raise ValueError(
                f"MCMCCheckpoint: every_seconds must be > 0, got {self.every_seconds}"
            )

    @property
    def path(self) -> Path:
        """The checkpoint file."""
        return Path(self.directory) / CHECKPOINT_FILENAME


class _CheckpointHeader(dx.Model):
    """What a checkpoint was written for.

    Attributes
    ----------
    format_version : int
        `CHECKPOINT_FORMAT_VERSION` of the writer.
    library : str
        ``"quivers"``.
    model_fingerprint : str
        Digest of the registry's layout, the Gibbs sites, the input, the
        data, and the potential at two probe positions.
    kernel_fingerprint : str
        Digest of `MCMCKernel.configuration`.
    run_fingerprint : str
        Digest of the warmup and sample counts, the chain count, and the
        initialization strategy.
    run_seed : int
        The base seed every chain's generator derives from.
    """

    format_version: int
    library: str
    model_fingerprint: str
    kernel_fingerprint: str
    run_fingerprint: str
    run_seed: int


def _digest(value: StateValue) -> str:
    """A SHA-256 digest of a configuration value.

    Parameters
    ----------
    value : StateValue
        The value; tensors are hashed by dtype, shape, and bytes.

    Returns
    -------
    str
        The hexadecimal digest.
    """
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True).encode()
    ).hexdigest()


def _canonical(value: StateValue) -> StateValue:
    """A JSON-serializable rendering of a configuration value.

    Parameters
    ----------
    value : StateValue
        The value.

    Returns
    -------
    StateValue
        The value with every tensor replaced by its digest and every
        float by its exact hexadecimal spelling.
    """
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    return value


def tensor_digest(value: torch.Tensor) -> str:
    """A SHA-256 digest of a tensor's dtype, shape, and contents.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    str
        The hexadecimal digest.
    """
    flat = value.detach().cpu().contiguous().reshape(-1)
    raw = bytes(flat.view(torch.uint8).tolist()) if flat.numel() else b""
    header = f"{flat.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(header + raw).hexdigest()


@runtime_checkable
class _ArrayLike(Protocol):
    """A value that exposes an array view, such as a NumPy array."""

    def __array__(self) -> _ArrayLike:
        """The value as an array.

        Returns
        -------
        _ArrayLike
            The array.
        """
        ...


type ObservedDatum = (
    torch.Tensor
    | _ArrayLike
    | bool
    | int
    | float
    | str
    | Sequence[ObservedDatum]
    | Mapping[str, ObservedDatum]
)
"""A value a run conditions on: an observed site's value or a host datum,
such as a tensor, a number, an array, a token, or a nested sequence or
mapping of these (a corpus of tokenized sentences, for instance)."""


def _numeric_shape(value: ObservedDatum) -> tuple[int, ...] | None:
    """The shape of a rectangular nest of Python numbers.

    Parameters
    ----------
    value : ObservedDatum
        The value.

    Returns
    -------
    tuple[int, ...] | None
        The lengths of its nesting levels (empty for one number), or
        ``None`` when a leaf is not a Python number or the sequences at
        one level differ in length.
    """
    if isinstance(value, bool | int | float):
        return ()
    if not isinstance(value, list | tuple):
        return None
    shapes = {_numeric_shape(item) for item in value}
    if None in shapes or len(shapes) > 1:
        return None
    inner = shapes.pop() if shapes else ()
    return None if inner is None else (len(value), *inner)


def observation_digest(name: str, value: ObservedDatum) -> StateValue:
    """A deterministic, content-complete rendering of an observed datum.

    Numeric data are rendered by `tensor_digest` of their tensor form,
    so a tensor, an array, and a rectangular nest of numbers holding the
    same values at the same dtype render identically, as the program
    reads them identically. Text is rendered verbatim, and every other
    sequence and mapping is rendered item by item, so a corpus of
    tokenized sentences, ragged or not, is covered token for token.

    Parameters
    ----------
    name : str
        The datum's name, for the diagnostic.
    value : ObservedDatum
        The datum.

    Returns
    -------
    StateValue
        A JSON-serializable rendering that two data share exactly when
        they hold the same content.

    Raises
    ------
    TypeError
        If the datum, or an item nested in it, is none of the kinds an
        `ObservedDatum` admits, or a mapping in it has a key that is not
        a string.
    """
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    if isinstance(value, str):
        return {"text": value}
    if isinstance(value, bool | int | float):
        return tensor_digest(torch.as_tensor(value))
    if isinstance(value, list | tuple):
        if _numeric_shape(value) is not None:
            return tensor_digest(torch.as_tensor(value))
        return {"sequence": [observation_digest(name, item) for item in value]}
    if isinstance(value, Mapping):
        rendered: dict[str, StateValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"observed datum {name!r} holds a mapping with the key "
                    f"{key!r} of type {type(key).__name__}; keys must be strings"
                )
            rendered[key] = observation_digest(name, item)
        return {"mapping": rendered}
    if isinstance(value, _ArrayLike):
        return tensor_digest(torch.as_tensor(value))
    raise TypeError(
        f"observed datum {name!r} holds a value of type {type(value).__name__}, "
        "which is not a tensor, an array, a number, a string, or a sequence or "
        "mapping of these"
    )


def write_checkpoint(
    path: Path, header: _CheckpointHeader, chains: list[StateValue]
) -> None:
    """Write a checkpoint atomically.

    Parameters
    ----------
    path : Path
        The checkpoint file; its directory is created if absent.
    header : _CheckpointHeader
        The header.
    chains : list[StateValue]
        Each chain's state dictionary, or ``None`` for a chain not yet
        started.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload: dict[str, StateValue] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "header": json.dumps(header.model_dump(), sort_keys=True),
        "chains": chains,
    }
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_checkpoint(path: Path) -> tuple[_CheckpointHeader, list[StateValue]]:
    """Read a checkpoint without executing code.

    Parameters
    ----------
    path : Path
        The checkpoint file.

    Returns
    -------
    tuple[_CheckpointHeader, list[StateValue]]
        The header and each chain's state dictionary (``None`` for a
        chain not yet started).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a checkpoint this library can read, or was
        written in another format version.
    """
    if not path.is_file():
        raise FileNotFoundError(f"MCMC.resume: no checkpoint at {path}")
    try:
        payload = torch.load(path, weights_only=True)
    except Exception as error:
        raise ValueError(
            f"MCMC.resume: {path} is not a readable MCMC checkpoint ({error})"
        ) from error
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(f"MCMC.resume: {path} is not an MCMC checkpoint")
    version = payload["format_version"]
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"MCMC.resume: {path} was written in checkpoint format {version}; this "
            f"library reads format {CHECKPOINT_FORMAT_VERSION}"
        )
    header_text, chains = payload.get("header"), payload.get("chains")
    if not isinstance(header_text, str) or not isinstance(chains, list):
        raise ValueError(f"MCMC.resume: {path} has a malformed header or chain list")
    header = _CheckpointHeader(**json.loads(header_text))
    if header.library != "quivers":
        raise ValueError(f"MCMC.resume: {path} was not written by quivers")
    return header, chains


__all__ = ["MCMCCheckpoint", "CHECKPOINT_FORMAT_VERSION", "CHECKPOINT_FILENAME"]
