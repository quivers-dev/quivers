"""Typed entry-point metadata for programs elaborated into computations.

A ``program`` declaration elaborates to an ordinary named computation whose
parameters are the program's data, observations, and fibrations, and whose
body performs the canonical ``Random.sample`` and ``Score.add`` requests.
What the computation alone does not say is which parameter plays which
role, which locals are the program's sites, and what the return tuple's
labels are. The records here carry that, beside the computation, so a
runtime, a transpiler, or a tool can drive the entry point without
re-reading the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quivers.qiec.identifiers import ComputationId, EffectInstanceId
from quivers.qiec.kinds import Telescope
from quivers.qiec.terms import PlateAxis
from quivers.qiec.types import TypeExpr

type ParameterRole = Literal[
    "domain",
    "scalar",
    "data",
    "observation",
    "fibration",
    "weight",
    "bias",
    "table",
    "kernel-input",
]
"""How a program parameter is supplied.

``domain`` parameters carry the value the program is applied to, one per
real factor of its domain; ``scalar`` ones are its declared parameters;
``data`` ones are free names its steps read; ``observation`` ones are the
values ``observe`` steps score; ``fibration`` ones map observation rows to
groups; ``weight`` and ``bias`` are the numbers of a declared morphism's
affine parameter map and ``table`` those of a morphism over a finite
domain, one row per element; ``kernel-input`` holds a kernel's input
locations.
"""

type SiteKind = Literal["sample", "observe", "marginal"]
"""What a program step does at its site."""


@dataclass(frozen=True, slots=True)
class ProgramParameter:
    """One parameter of an elaborated program.

    Parameters
    ----------
    name
        The parameter's name, which is the computation parameter's.
    role
        How the parameter is supplied.
    type
        Its type.
    axes
        The name of each dimension of a tensor-typed parameter, the
        object or extent it ranges over; empty for a scalar.
    per_row
        Whether the parameter fills a family parameter of a plated draw
        directly, so that its leading axis is the draw's plate: a value
        without that axis is one value shared by every row, which a run
        broadcasts across the plate.
    tag
        The serialization discriminator; always ``"program_parameter"``.
    """

    name: str
    role: ParameterRole
    type: TypeExpr
    axes: tuple[str, ...] = ()
    per_row: bool = False
    tag: Literal["program_parameter"] = "program_parameter"


@dataclass(frozen=True, slots=True)
class ProgramSite:
    """One probabilistic step of an elaborated program.

    Parameters
    ----------
    name
        The site's label, the name the step binds or scores.
    kind
        Whether the step samples, observes, or marginalizes.
    family
        The distribution family the step draws from.
    batch
        The plate's batch axes.
    event
        The plate's event axes.
    tag
        The serialization discriminator; always ``"program_site"``.
    """

    name: str
    kind: SiteKind
    family: str
    batch: tuple[PlateAxis, ...] = ()
    event: tuple[PlateAxis, ...] = ()
    tag: Literal["program_site"] = "program_site"


@dataclass(frozen=True, slots=True)
class ProgramEntry:
    """The typed entry point a program declaration elaborates to.

    Parameters
    ----------
    name
        The program's source name.
    computation
        The identity of the named computation holding its body.
    parameters
        The computation's parameters with their roles, in order.
    return_names
        The names the program returns, in order.
    return_labels
        The labels of a labelled return tuple, or ``None``.
    sites
        The program's probabilistic steps, in source order, those of
        marginalization scopes included.
    random_instance
        The lexical ``Random`` instance the program's samples address.
    score_instance
        The lexical ``Score`` instance the program's scores address.
    telescope
        The static binders the computation takes: one index binder per
        extent of an input the program's steps leave open, which a run
        reads off the data it is given.
    tag
        The serialization discriminator; always ``"program_entry"``.
    """

    name: str
    computation: ComputationId
    parameters: tuple[ProgramParameter, ...]
    return_names: tuple[str, ...]
    return_labels: tuple[str, ...] | None
    sites: tuple[ProgramSite, ...]
    random_instance: EffectInstanceId
    score_instance: EffectInstanceId
    telescope: Telescope = ()
    tag: Literal["program_entry"] = "program_entry"


__all__ = [
    "ParameterRole",
    "ProgramEntry",
    "ProgramParameter",
    "ProgramSite",
    "SiteKind",
]
