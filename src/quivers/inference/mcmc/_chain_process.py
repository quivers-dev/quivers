"""Start chain workers in fresh interpreters that never import the caller's script.

Python's ``spawn`` start method prepares a child process by importing the
parent's main module again. A script that calls a fit with several cores at
its top level, outside an ``if __name__ == "__main__":`` block, is then
executed once more in every child, which starts the fit again there. The
workers here are started as ``python -m quivers.inference.mcmc._chain_entry``
instead: the child's main module is that entry module, so the caller's script
is never executed a second time and needs no guard.

A worker recipe is sent to the child as a pickle and the chain records come
back as a pickle, each in a file of a private temporary directory. Pickle
stores a function or class by the name of its module, and the child does
not import the caller's main module, so a recipe that refers to an object
defined there is refused before any process starts.

Cost: one interpreter start and one rebuild of the run per worker.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import pickle
import pickletools
import subprocess
import sys
import tempfile
import traceback
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from quivers.inference.mcmc.parallel import ChainStates, ChainWorker

_MAIN = "__main__"


def _main_references(payload: bytes) -> bool:
    """Report whether a pickle refers to an object of the main module.

    Parameters
    ----------
    payload : bytes
        A pickle stream.

    Returns
    -------
    bool
        True when the stream names the module ``__main__``, either in a
        ``GLOBAL`` opcode or as the module string of a ``STACK_GLOBAL``.
    """
    for opcode, argument, _ in pickletools.genops(payload):
        if not isinstance(argument, str):
            continue
        if opcode.name == "GLOBAL" and argument.split(" ", 1)[0] == _MAIN:
            return True
        if opcode.name.endswith("UNICODE") and argument == _MAIN:
            return True
    return False


def worker_payload(worker: ChainWorker, chains: tuple[int, ...]) -> bytes:
    """Pickle one worker's recipe and its assigned chains.

    Parameters
    ----------
    worker : ChainWorker
        The recipe.
    chains : tuple[int, ...]
        Chain indices assigned to the worker.

    Returns
    -------
    bytes
        The pickle a worker process reads.

    Raises
    ------
    ValueError
        If the recipe cannot be pickled, or refers to a function, class, or
        value defined in the caller's main module.
    """
    try:
        payload = pickle.dumps((worker, chains))
    except (pickle.PicklingError, TypeError, AttributeError) as error:
        raise ValueError(
            "parallel chains rebuild the run in worker processes, so every fit "
            f"argument must be picklable ({error}); use cores=1 or give "
            "module-level functions and values"
        ) from error
    if _main_references(payload):
        raise ValueError(
            "parallel chains rebuild the run in worker processes that do not "
            "import the calling script, so that the script is never executed "
            "again; a fit argument refers to a function or class defined in "
            "the main module (__main__). Define it in an importable module, "
            "or use cores=1"
        )
    return payload


def run_worker_processes(
    worker: ChainWorker, assignments: Sequence[tuple[int, ...]], threads: int
) -> list[ChainStates]:
    """Run one worker process per chain assignment and collect their records.

    Parameters
    ----------
    worker : ChainWorker
        Picklable recipe rebuilding the run.
    assignments : Sequence[tuple[int, ...]]
        Chain indices of each worker.
    threads : int
        Intra-op thread count of every worker, the parent's.

    Returns
    -------
    list[ChainStates]
        The records of each assignment, in assignment order.

    Raises
    ------
    ValueError
        If the recipe cannot cross a process boundary.
    RuntimeError
        If a worker fails; the message carries the worker's own error and
        traceback. Every other worker is stopped before the error is raised,
        also when the caller is interrupted.
    """
    payloads = [worker_payload(worker, chains) for chains in assignments]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(entry for entry in sys.path if entry)
    processes: list[subprocess.Popen[bytes]] = []
    with tempfile.TemporaryDirectory(prefix="quivers-chains-") as directory:
        root = Path(directory)
        try:
            for index, payload in enumerate(payloads):
                (root / f"recipe-{index}.pkl").write_bytes(payload)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "quivers.inference.mcmc._chain_entry",
                            str(root / f"recipe-{index}.pkl"),
                            str(root / f"records-{index}.pkl"),
                            str(threads),
                        ],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                    )
                )
            parts: list[ChainStates] = []
            for index, process in enumerate(processes):
                code = process.wait()
                records = root / f"records-{index}.pkl"
                if not records.exists():
                    raise RuntimeError(
                        f"parallel chain worker {index} (chains "
                        f"{list(assignments[index])}) exited with status {code} "
                        "before returning its chains"
                    )
                outcome = pickle.loads(records.read_bytes())
                if outcome[0] != "ok":
                    raise RuntimeError(
                        f"parallel chain worker {index} (chains "
                        f"{list(assignments[index])}) failed with "
                        f"{outcome[1]}: {outcome[2]}\n{outcome[3]}"
                    )
                parts.append(outcome[1])
            return parts
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()


def main(arguments: Sequence[str]) -> int:
    """Run one worker's chains in this process and write their records.

    Parameters
    ----------
    arguments : Sequence[str]
        The recipe file, the records file, and the thread count.

    Returns
    -------
    int
        Zero when the chains ran, one when the worker raised. Either way the
        records file holds the outcome: ``("ok", records)`` or ``("error",
        type name, message, traceback)``.
    """
    recipe, records, threads = arguments
    torch.set_num_threads(int(threads))
    try:
        worker, chains = pickle.loads(Path(recipe).read_bytes())
        outcome: tuple[str, ...] | tuple[str, ChainStates] = ("ok", worker(chains))
    except Exception as error:
        outcome = ("error", type(error).__name__, str(error), traceback.format_exc())
    Path(records).write_bytes(pickle.dumps(outcome))
    return 0 if outcome[0] == "ok" else 1


__all__: list[str] = []
