"""A program morphism that runs its checked computation.

A ``program`` that calls a named computation has no step table the
runtime compiler can build: the call's body is a term of the checked
module, and only the reference machine runs it. Its compiled form is a
[`CheckedProgram`][quivers.effects.checked_program.CheckedProgram], a
morphism whose forward samples and joint densities are the module's own
[`sample_program`][quivers.qiec.program_runtime.sample_program] and
[`run_program`][quivers.qiec.program_runtime.run_program] runs, so the
program's Python surface is the one entry point the command line and the
REPL invoke.

The reference machine carries numbers, not tensors, so a checked program
cannot be fit: it has no step table, and reading one raises the refusal
the compiler recorded, which says which call could not be lowered to
compiled runtime steps and why.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch

from quivers.continuous.morphisms import AnySpace, ContinuousMorphism
from quivers.qiec.entries import HostValue
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import (
    program_entry,
    run_program,
    sample_program,
)
from quivers.qiec.programs import ProgramEntry


def _nested(value: object) -> HostValue:
    """Nested lists as nested tuples.

    Parameters
    ----------
    value : object
        A number, or lists of numbers however nested.

    Returns
    -------
    HostValue
        The same with every list a tuple.
    """
    if isinstance(value, list):
        return tuple(_nested(item) for item in value)
    assert isinstance(value, bool | int | float)
    return value


def _host(value: torch.Tensor) -> HostValue:
    """A tensor as the machine takes it.

    The reference machine carries numbers and tuples of numbers, not
    tensors, so a run keeps no gradient path to the values it is given.

    Parameters
    ----------
    value : torch.Tensor
        The value.

    Returns
    -------
    HostValue
        A scalar tensor as a number; any other tensor as nested tuples.
    """
    if value.dim() == 1 and value.shape[0] == 1:
        value = value.reshape(())
    return _nested(value.detach().cpu().tolist())


def _tensor(value: HostValue) -> torch.Tensor:
    """A returned host value as a tensor.

    Parameters
    ----------
    value : HostValue
        The value the run returned.

    Returns
    -------
    torch.Tensor
        The value as a tensor; a scalar becomes a one-element vector.

    Raises
    ------
    TypeError
        If the value is a record, which has no tensor form.
    """
    if isinstance(value, torch.Tensor):
        return value.reshape(1) if value.dim() == 0 else value
    if isinstance(value, Mapping):
        raise TypeError("a program returning a record has no tensor form")
    tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
    return tensor.reshape(1) if tensor.dim() == 0 else tensor


class CheckedProgram(ContinuousMorphism):
    """A program run as its elaborated computation on the reference machine.

    Parameters
    ----------
    domain : AnySpace
        The program's declared domain.
    codomain : AnySpace
        The program's declared codomain.
    module : QiecModule
        The checked module holding the program's entry point.
    name : str
        The program's name.
    refusal : Exception | None
        Why the program cannot be fit, raised by every fitting entry
        point (`trace`, MCMC, SVI, and the rest read the program's
        steps); ``None`` gives a refusal naming the program.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        module: QiecModule,
        name: str,
        refusal: Exception | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._module = module
        self._name = name
        self._entry: ProgramEntry = program_entry(module, name)
        self._refusal = refusal

    @property
    def refusal(self) -> Exception:
        """Why the program cannot be fit.

        Returns
        -------
        Exception
            The located refusal the compiler recorded for the call it
            could not lower, or a refusal naming the program.
        """
        if self._refusal is not None:
            return self._refusal
        return TypeError(
            f"program {self._name!r} runs on the reference machine only and "
            f"has no compiled steps to fit"
        )

    @property
    def steps(self) -> tuple[()]:
        """The compiled step table, which a checked program does not have.

        Every fitting entry point reads a program's steps, so reading
        them here raises the refusal that says why the program cannot
        be fit.

        Returns
        -------
        tuple[()]
            Never returns.

        Raises
        ------
        Exception
            [`refusal`][quivers.effects.checked_program.CheckedProgram.refusal].
        """
        raise self.refusal

    @property
    def module(self) -> QiecModule:
        """The checked module the program runs in."""
        return self._module

    @property
    def name(self) -> str:
        """The program's name."""
        return self._name

    @property
    def entry(self) -> ProgramEntry:
        """The program's elaborated entry point."""
        return self._entry

    def observed_names(self) -> set[str]:
        """The names the program observes.

        Returns
        -------
        set[str]
            The labels of its ``observe`` sites.
        """
        return {site.name for site in self._entry.sites if site.kind == "observe"}

    def _split(
        self, x: torch.Tensor, row: int, observations: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, HostValue], dict[str, HostValue]]:
        """Divide the given values into the entry's data and its sites.

        Parameters
        ----------
        x : torch.Tensor
            The program input, one row per run.
        row : int
            The run's row of the input.
        observations : Mapping[str, torch.Tensor]
            The given values, by name.

        Returns
        -------
        tuple[dict[str, HostValue], dict[str, HostValue]]
            The entry's parameters by name, the input filling its
            ``domain`` parameters, and the sites the run conditions on:
            every given name that is not a parameter.
        """
        parameters = {parameter.name for parameter in self._entry.parameters}
        data: dict[str, HostValue] = {}
        sites: dict[str, HostValue] = {}
        domain_parameters = [
            parameter.name
            for parameter in self._entry.parameters
            if parameter.role == "domain"
        ]
        if domain_parameters:
            value = x[row]
            if len(domain_parameters) == 1:
                data[domain_parameters[0]] = _host(value)
            else:
                for index, parameter_name in enumerate(domain_parameters):
                    data[parameter_name] = _host(value[..., index])
        for given, value in observations.items():
            if given in parameters and given not in data:
                data[given] = _host(value)
            else:
                # A site of the program or of a computation it calls;
                # the run rejects a site it never reaches.
                sites[given] = _host(value)
        return data, sites

    def rsample(  # type: ignore[override]
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
        observations: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the program forward, once per input row and sample.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.
        sample_shape : torch.Size
            Additional leading sample dimensions; each element is one
            independent run.
        observations : Mapping[str, torch.Tensor] or None
            The program's data by parameter name, and values for the
            sample sites the runs condition on; every other site is
            drawn.

        Returns
        -------
        torch.Tensor
            The returned values, shape ``(*sample_shape, batch, ...)``.
        """
        given = dict(observations or {})
        batch = int(x.shape[0]) if x.dim() >= 1 else 1
        runs = math.prod(sample_shape)
        rows: list[torch.Tensor] = []
        for _ in range(max(runs, 1)):
            values: list[torch.Tensor] = []
            for row in range(batch):
                data, sites = self._split(x, row, given)
                run = sample_program(self._module, self._name, data=data, sites=sites)
                values.append(_tensor(run.value))
            rows.append(torch.stack(values))
        stacked = torch.stack(rows)
        if not sample_shape:
            return stacked[0]
        return stacked.reshape((*sample_shape, *stacked.shape[1:]))

    def has_conditional_density(self) -> bool:
        """No: a program's density at its output marginalizes its draws.

        Returns
        -------
        bool
            Always ``False``;
            [`log_joint`][quivers.effects.checked_program.CheckedProgram.log_joint]
            scores the draws themselves.
        """
        return False

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability is not defined for a program's output alone.

        Parameters
        ----------
        x : torch.Tensor
            Program input.
        y : torch.Tensor
            Program output.

        Raises
        ------
        NotImplementedError
            Always: the density at the output marginalizes every draw.
        """
        del x, y
        raise NotImplementedError(
            "log_prob is not supported for programs; computing p(y | x) "
            "requires marginalizing over all intermediate draws. Use "
            "log_joint with a value for every site."
        )

    def log_joint(
        self, x: torch.Tensor, intermediates: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """The joint log-density at given values for every site.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.
        intermediates : Mapping[str, torch.Tensor]
            The program's data by parameter name and a value for every
            sample site, by label.

        Returns
        -------
        torch.Tensor
            The log joint of each input row's run. Shape ``(batch,)``.
            The machine scores numbers, so the result carries no
            gradient path to the given tensors.

        Raises
        ------
        ExecutionFailure
            If a site the program reaches is given no value, or a given
            site is never reached.
        """
        batch = int(x.shape[0]) if x.dim() >= 1 else 1
        totals: list[torch.Tensor] = []
        for row in range(batch):
            data, sites = self._split(x, row, intermediates)
            run = run_program(self._module, self._name, data=data, sites=sites)
            totals.append(
                torch.as_tensor(run.log_joint, dtype=torch.get_default_dtype())
            )
        return torch.stack(totals)

    def __repr__(self) -> str:
        return f"CheckedProgram({self._name!r}: {self.domain!r} -> {self.codomain!r})"


__all__ = ["CheckedProgram"]
