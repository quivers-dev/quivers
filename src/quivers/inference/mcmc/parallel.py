"""Run the chains of one MCMC run in separate processes.

A compiled program holds compilation-local family catalogs and prepared
evaluation plans that do not cross a process boundary, and a kernel holds
the adaptation state of the chain it runs. Each worker process therefore
rebuilds the program from a picklable recipe (a `ChainWorker`), builds its
own driver and kernel, and runs only the chains assigned to it with the
streams `MCMC` derives from the run seed and the chain index. The parent
reads every chain's record back into the driver's result, so the draws,
statistics, and evaluation counts are those of the serial run with the same
seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc._chain_process import run_worker_processes
from quivers.inference.mcmc.driver import MCMC, MCMCResult, _ChainRecord
from quivers.inference.mcmc.kernel import StateValue


class ChainStates(ValueModel):
    """The records of the chains one worker ran.

    Parameters
    ----------
    fingerprint : str
        Model fingerprint of the worker's rebuilt program and data, which
        must equal the parent's.
    run_seed : int
        The run seed the worker derived its chain streams from.
    states : Mapping[int, dict[str, StateValue]]
        Each chain's record, as a checkpoint stores it, keyed by chain index.
    """

    fingerprint: str
    run_seed: int
    states: Mapping[int, dict[str, StateValue]] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class ChainWorker(Protocol):
    """A picklable recipe that rebuilds a run and runs some of its chains.

    A worker is called in a fresh process with the chain indices assigned to
    it. It rebuilds the program, data, and driver exactly as the parent built
    them and returns `run_chains` of the assigned chains.
    """

    def __call__(self, chains: tuple[int, ...]) -> ChainStates:
        """Run the assigned chains.

        Parameters
        ----------
        chains : tuple[int, ...]
            Chain indices.

        Returns
        -------
        ChainStates
            Their records.
        """
        ...


def run_chains(
    mcmc: MCMC,
    model: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
    chains: Sequence[int],
    guide: Guide | None = None,
) -> ChainStates:
    """Run some chains of a run as `MCMC.run` runs them.

    Chain ``c`` starts from its own stream ``MCMC._chain_generator(seed, c)``
    with fresh kernel adaptation, so its record does not depend on which
    other chains run in the same process.

    Parameters
    ----------
    mcmc : MCMC
        The driver, with a seed and no checkpoint or callback.
    model : MonadicProgram
        The program.
    x : torch.Tensor
        One-row input.
    observations : dict[str, torch.Tensor]
        Data.
    chains : Sequence[int]
        Chain indices to run.
    guide : Guide or None
        Guide for ``init_strategy="guide"``.

    Returns
    -------
    ChainStates
        The chains' records and the model fingerprint.

    Raises
    ------
    ValueError
        If a chain index is out of range.
    """
    if any(not 0 <= chain < mcmc.num_chains for chain in chains):
        raise ValueError(
            f"run_chains: chain indices {list(chains)} are outside "
            f"0..{mcmc.num_chains - 1}"
        )
    run_seed = mcmc._run_seed()
    setup = mcmc._setup(model, x, observations, "MCMC.run", run_seed)
    header = mcmc._header(setup, run_seed)
    records: list[_ChainRecord | None] = [None] * mcmc.num_chains
    states: dict[int, dict[str, StateValue]] = {}
    for chain in chains:
        generator = mcmc._chain_generator(run_seed, chain)
        record = mcmc._start_chain(setup, chain, generator, guide)
        records[chain] = record
        mcmc._advance(setup, header, records, chain, record, generator)
        states[chain] = record.state_dict()
    return ChainStates(
        fingerprint=header.model_fingerprint, run_seed=run_seed, states=states
    )


def assemble_chains(
    mcmc: MCMC,
    model: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
    parts: Sequence[ChainStates],
) -> MCMCResult:
    """Assemble the result of a run from its chains' records.

    Parameters
    ----------
    mcmc : MCMC
        The parent's driver.
    model : MonadicProgram
        The parent's program.
    x : torch.Tensor
        One-row input.
    observations : dict[str, torch.Tensor]
        Data.
    parts : Sequence[ChainStates]
        Every worker's records.

    Returns
    -------
    MCMCResult
        The result `MCMC.run` returns for the same seed.

    Raises
    ------
    ValueError
        If a worker rebuilt a different program or data, used another run
        seed, or the parts do not cover every chain exactly once.
    """
    run_seed = mcmc._run_seed()
    setup = mcmc._setup(model, x, observations, "MCMC.run", run_seed)
    fingerprint = mcmc._header(setup, run_seed).model_fingerprint
    records: dict[int, _ChainRecord] = {}
    for part in parts:
        if part.fingerprint != fingerprint or part.run_seed != run_seed:
            raise ValueError(
                "parallel chains: a worker rebuilt a different program, data, or "
                "seed than the parent; the run recipe is not reproducible"
            )
        for chain, state in part.states.items():
            if chain in records:
                raise ValueError(f"parallel chains: chain {chain} ran twice")
            records[chain] = _ChainRecord.from_state_dict(state, setup.site_names)
    if sorted(records) != list(range(mcmc.num_chains)):
        raise ValueError(
            f"parallel chains: chains {sorted(records)} ran, expected "
            f"0..{mcmc.num_chains - 1}"
        )
    return mcmc._result(setup, [records[chain] for chain in range(mcmc.num_chains)])


def parallel_chains(
    mcmc: MCMC,
    model: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
    worker: ChainWorker,
    cores: int,
) -> MCMCResult:
    """Run a driver's chains in up to ``cores`` worker processes.

    Chains are dealt to workers in turn (chain ``c`` to worker
    ``c mod cores``). Each worker is a fresh interpreter running
    `quivers.inference.mcmc._chain_entry`, so none inherits the parent's
    state or imports the calling script, and each uses the parent's thread
    count. The result equals ``mcmc.run(model, x, observations)``.

    Parameters
    ----------
    mcmc : MCMC
        The parent's driver: it must have an explicit seed, which every
        worker shares, and no checkpoint or callback, which cannot be
        shared across processes.
    model : MonadicProgram
        The parent's program.
    x : torch.Tensor
        One-row input.
    observations : dict[str, torch.Tensor]
        Data.
    worker : ChainWorker
        Picklable recipe rebuilding the run in a worker.
    cores : int
        Maximum worker processes (brms ``cores``).

    Returns
    -------
    MCMCResult
        The run's result.

    Raises
    ------
    ValueError
        If ``cores`` is not a positive integer, the driver has no seed or
        has a checkpoint, callback, or generator, or the worker cannot be
        pickled or refers to an object defined in the caller's main module.
    RuntimeError
        If a worker process fails; the remaining workers are stopped.
    """
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ValueError("cores must be a positive integer")
    if mcmc.seed is None or mcmc.generator is not None:
        raise ValueError(
            "parallel chains need an explicit integer seed shared by every worker"
        )
    if mcmc.checkpoint is not None or mcmc.callback is not None:
        raise ValueError(
            "parallel chains cannot checkpoint or call back from worker "
            "processes; run with cores=1"
        )
    count = min(cores, mcmc.num_chains)
    assignments = [
        tuple(range(start, mcmc.num_chains, count)) for start in range(count)
    ]
    # Workers start in fresh interpreters that never import the caller's
    # script, so a script needs no ``if __name__ == "__main__":`` guard.
    parts = run_worker_processes(worker, assignments, torch.get_num_threads())
    return assemble_chains(mcmc, model, x, observations, parts)


__all__ = [
    "ChainStates",
    "ChainWorker",
    "assemble_chains",
    "parallel_chains",
    "run_chains",
]
