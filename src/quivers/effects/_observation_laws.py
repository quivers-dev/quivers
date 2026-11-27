"""The observations of a checked program as sites with laws.

A program's ``observe y <- Family(...)`` elaborates to a score of the
family's log density at the datum ``y``, a parameter of the program's
computation whose role is ``"observation"``. The elaboration states both
halves of the observation: the program's entry lists ``y`` as a site of
kind ``"observe"``, and the score names the law ``y`` is scored under. An
observation scored at the top level of the program's body, and read
nowhere else, therefore has a generative law: the program may draw ``y``
from it rather than score a value it is given.

`observation_variant` rewrites a program's computation accordingly. Each
such observation is either drawn (the score becomes a ``Random.sample``
of the same law at the site ``y``, so a run conditions it when it is
given and draws it when it is not) or dropped (the score is removed, which
is the program's prior). A datum read anywhere else, or scored inside a
computation the program calls, keeps its role as a datum the program must
be given, and `ObservationLaws.kept` says why for each.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import site_type
from quivers.qiec.effects import ComputationType, EffectRequest, RowEntry
from quivers.qiec.identifiers import ComputationId, EffectInstanceId
from quivers.qiec.module import NamedComputation, NamedEffectInstance, QiecModule
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.substitution import instantiate_operation, instantiate_telescope
from quivers.qiec.terms import (
    Bind,
    Call,
    Computation,
    DistributionValue,
    Local,
    LogDensity,
    Perform,
    SiteValue,
    Var,
    WeightSum,
)


class ObservationLaws(QiecModel):
    """A program's computation with its observations given their laws.

    Parameters
    ----------
    computation : NamedComputation
        The rewritten computation; its parameters are the original ones
        less the observations it draws or drops.
    observations : tuple[str, ...]
        The observations rewritten, each a site of the computation when
        drawn.
    kept : Mapping[str, str]
        Each observation that stays a datum, with why it has no law the
        program can draw it from.
    unremoved : tuple[str, ...]
        The dotted paths of the observation densities a variant without
        likelihoods still adds: they are scored inside a computation the
        program calls, or under a branch, where no bind of the program's
        own body removes them.
    """

    computation: NamedComputation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    observations: tuple[str, ...] = ()
    kept: Mapping[str, str] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    unremoved: tuple[str, ...] = ()


def _mentions(node: object, local: Local) -> int:
    """How often a term reads a local.

    Parameters
    ----------
    node : object
        A term, a value, or a tuple of them.
    local : Local
        The local.

    Returns
    -------
    int
        The number of ``Var`` nodes naming it.
    """
    count = 0
    for item in _nodes(node):
        if isinstance(item, Var):
            count += item.local == local
    return count


def _nodes(node: object) -> Iterator[object]:
    """Every node of a term, the term first.

    Parameters
    ----------
    node : object
        A term, a value, or a tuple of them.

    Yields
    ------
    object
        Each node, field and tuple item, at any depth.
    """
    pending: list[object] = [node]
    while pending:
        item = pending.pop()
        yield item
        if isinstance(item, tuple):
            pending.extend(item)
        elif isinstance(item, QiecModel):
            pending.extend(getattr(item, name) for name in type(item).__field_specs__)


class ScoreRoles:
    """Which scores of a module's runs are observations' densities.

    A score adds an observation's density when it adds the log density of
    the law of an ``observe`` site of a program of the module, or a value
    computed from such densities: the total of a marginalization block,
    which sums its observations' densities over the marginalized values.
    Any other score is a potential of the prior.

    Parameters
    ----------
    module : QiecModule
        The module.
    """

    def __init__(self, module: QiecModule) -> None:
        self._computations = {item.id: item for item in module.computations}
        self._observed = frozenset(
            (entry.name, site.name)
            for entry in module.entries
            for site in entry.sites
            if site.kind == "observe"
        )
        self._densities: dict[ComputationId, bool] = {}

    def _is_observed_law(self, node: object) -> bool:
        """Whether a node is the log density of an ``observe`` site's law.

        Parameters
        ----------
        node : object
            A node of a term.

        Returns
        -------
        bool
            Whether it is such a density.
        """
        if not isinstance(node, LogDensity) or not isinstance(
            node.sampleable, DistributionValue
        ):
            return False
        path = node.sampleable.origin.structural_path
        return (
            len(path) == 5
            and path[0] == "programs"
            and path[2] == "sites"
            and (path[1], path[3]) in self._observed
        )

    def holds_density(self, node: object) -> bool:
        """Whether a term computes an observation's density, at any depth.

        Parameters
        ----------
        node : object
            A term, followed into the computations it calls.

        Returns
        -------
        bool
            Whether an ``observe`` site's log density is among its nodes.
        """
        pending: list[object] = [node]
        while pending:
            item = pending.pop()
            if self._is_observed_law(item):
                return True
            if isinstance(item, Call):
                if self._callee_holds_density(item.callee):
                    return True
            if isinstance(item, tuple):
                pending.extend(item)
            elif isinstance(item, QiecModel):
                pending.extend(
                    getattr(item, name) for name in type(item).__field_specs__
                )
        return False

    def _callee_holds_density(self, callee: ComputationId) -> bool:
        """Whether a called computation computes an observation's density.

        Parameters
        ----------
        callee : ComputationId
            The computation.

        Returns
        -------
        bool
            The answer, found once per computation; a recursive call
            reached while its own answer is pending adds nothing.
        """
        known = self._densities.get(callee)
        if known is not None:
            return known
        self._densities[callee] = False
        computation = self._computations.get(callee)
        found = computation is not None and self.holds_density(computation.body)
        self._densities[callee] = found
        return found

    def is_likelihood(
        self, request: EffectRequest, bound: Mapping[Local, Computation]
    ) -> bool:
        """Whether a score request adds an observation's density.

        Parameters
        ----------
        request : EffectRequest
            A request to a score instance.
        bound : Mapping[Local, Computation]
            The computation each local of the request's body is bound to.

        Returns
        -------
        bool
            Whether its weight is an ``observe`` site's log density, or
            reads a local bound to a computation of one.
        """
        if self.holds_density(request.arguments):
            return True
        return any(
            isinstance(node, Var)
            and node.local in bound
            and self.holds_density(bound[node.local])
            for node in _all_nodes(request.arguments)
        )

    def scores(
        self, body: Computation, instance: EffectInstanceId
    ) -> list[tuple[str, bool]]:
        """Every score a run adds to an instance, with its role.

        The body is walked, and so is every computation it calls,
        transitively.

        Parameters
        ----------
        body : Computation
            The body run.
        instance : EffectInstanceId
            The score instance.

        Returns
        -------
        list[tuple[str, bool]]
            The dotted structural path of every request to the instance,
            in order, and whether it adds an observation's density.
        """
        found: list[tuple[str, bool]] = []
        visited: set[ComputationId] = set()
        bodies: list[Computation] = [body]
        while bodies:
            current = bodies.pop()
            bound = _bindings(current)
            for item in _all_nodes(current):
                if isinstance(item, EffectRequest) and item.instance == instance:
                    found.append(
                        (
                            ".".join(
                                str(part) for part in item.origin.origin.structural_path
                            ),
                            self.is_likelihood(item, bound),
                        )
                    )
                if isinstance(item, Call) and item.callee not in visited:
                    visited.add(item.callee)
                    callee = self._computations.get(item.callee)
                    if callee is not None:
                        bodies.append(callee.body)
        return found


def _bindings(body: Computation) -> dict[Local, Computation]:
    """The computation each bind of a body binds its local to.

    Parameters
    ----------
    body : Computation
        The body.

    Returns
    -------
    dict[Local, Computation]
        The first computation of every ``Bind``, by its binder.
    """
    return {
        item.binder: item.first for item in _all_nodes(body) if isinstance(item, Bind)
    }


def _all_nodes(node: object) -> list[object]:
    """Every node of a term.

    Parameters
    ----------
    node : object
        A term, a value, or a tuple of them.

    Returns
    -------
    list[object]
        The nodes, the term first.
    """
    found: list[object] = []
    pending: list[object] = [node]
    while pending:
        item = pending.pop()
        found.append(item)
        if isinstance(item, tuple):
            pending.extend(reversed(item))
        elif isinstance(item, QiecModel):
            pending.extend(
                reversed([getattr(item, name) for name in type(item).__field_specs__])
            )
    return found


def _observation(
    term: Computation, score: EffectInstanceId, names: frozenset[str]
) -> tuple[Local, DistributionValue, EffectRequest] | None:
    """The observation a term scores, if it is the score of one.

    Parameters
    ----------
    term : Computation
        A bind's first computation.
    score : EffectInstanceId
        The program's canonical score instance.
    names : frozenset[str]
        The program's observations.

    Returns
    -------
    tuple[Local, DistributionValue, EffectRequest] or None
        The observed datum, its law and the score request, when the term
        adds the log density of an observation at its datum to the score.
    """
    if not isinstance(term, Perform) or term.request.instance != score:
        return None
    if len(term.request.arguments) != 1:
        return None
    (weight,) = term.request.arguments
    if not isinstance(weight, WeightSum) or not isinstance(weight.value, LogDensity):
        return None
    density = weight.value
    if (
        not isinstance(density.value, Var)
        or density.value.local.name not in names
        or not isinstance(density.sampleable, DistributionValue)
    ):
        return None
    return density.value.local, density.sampleable, term.request


def _sample(
    module: QiecModule,
    random: NamedEffectInstance,
    datum: Local,
    law: DistributionValue,
    scored: EffectRequest,
    program: str,
) -> Perform:
    """A draw of an observation from its law, at its site.

    Parameters
    ----------
    module : QiecModule
        The module.
    random : NamedEffectInstance
        The program's canonical random instance.
    datum : Local
        The observation.
    law : DistributionValue
        Its law.
    scored : EffectRequest
        The score request it replaces, whose source position it keeps.
    program : str
        The program's name.

    Returns
    -------
    Perform
        The ``Random.sample`` request of the site named by the datum.

    Raises
    ------
    KeyError
        If the module declares no ``Random`` interface with a ``sample``
        operation.
    """
    declaration = next(
        (item for item in module.effects if item.ref.id == random.entry.effect.id), None
    )
    if declaration is None:
        raise KeyError(f"module {module.module!r} declares no Random interface")
    operation = next(item for item in declaration.operations if item.name == "sample")
    outer = instantiate_telescope(declaration.telescope, random.entry.effect.arguments)
    _, result_type = instantiate_operation(operation, (datum.type,), outer)
    origin = scored.origin.origin.with_(
        structural_path=("programs", program, "sites", datum.name),
        role="effect-request",
    )
    return Perform(
        request=EffectRequest(
            instance=random.entry.instance,
            effect=random.entry.effect,
            operation=operation.id,
            static_arguments=(datum.type,),
            arguments=(
                SiteValue(label=datum.name, result_type=site_type(datum.type)),
                law,
            ),
            result_type=result_type,
            origin=scored.origin.with_(origin=origin),
        )
    )


def observation_variant(
    module: QiecModule,
    computation: NamedComputation,
    entry: ProgramEntry,
    random: NamedEffectInstance,
    score: EffectInstanceId,
    *,
    drop: bool,
    likelihoods: bool,
) -> ObservationLaws:
    """A program's computation with its observations drawn or dropped.

    The body's spine of binds is walked from the end. A bind whose first
    computation scores an observation at its datum, read nowhere else in
    the body, is replaced: with ``drop``, by what follows it; otherwise by
    a draw of the datum from the law it was scored under, bound to the
    datum for what follows. Without ``likelihoods``, every other bind that
    scores an observation's density (`ScoreRoles`) is removed as well, with
    the binds that computed its weight and nothing else.

    Parameters
    ----------
    module : QiecModule
        The module.
    computation : NamedComputation
        The program's computation.
    entry : ProgramEntry
        The program's entry.
    random : NamedEffectInstance
        The canonical random instance.
    score : EffectInstanceId
        The canonical score instance.
    drop : bool
        Whether to drop the observations' scores rather than draw them.
    likelihoods : bool
        Whether to keep the scores of observations that stay data.

    Returns
    -------
    ObservationLaws
        The rewritten computation, the observations it rewrote, and, without
        ``likelihoods``, the observation densities it could not remove.
    """
    sites = {site.name for site in entry.sites if site.kind == "observe"}
    names = frozenset(
        parameter.name
        for parameter in entry.parameters
        if parameter.role == "observation" and parameter.name in sites
    )
    roles = ScoreRoles(module)
    bound = _bindings(computation.body)
    spine: list[Bind] = []
    term: Computation = computation.body
    while isinstance(term, Bind):
        spine.append(term)
        term = term.then
    rebuilt: Computation = term
    rewritten: list[str] = []
    removed: set[Local] = set()
    weights: set[Local] = set()
    for bind in reversed(spine):
        found = _observation(bind.first, score, names)
        unused = _mentions(rebuilt, bind.binder) == 0
        if found is not None and _mentions(computation.body, found[0]) == 1 and unused:
            datum, law, request = found
            rewritten.append(datum.name)
            removed.add(bind.binder)
            if not drop:
                rebuilt = Bind(
                    binder=datum,
                    first=_sample(module, random, datum, law, request, entry.name),
                    then=rebuilt,
                )
            continue
        if (
            not likelihoods
            and unused
            and isinstance(bind.first, Perform)
            and bind.first.request.instance == score
            and roles.is_likelihood(bind.first.request, bound)
        ):
            removed.add(bind.binder)
            weights.update(
                node.local
                for node in _all_nodes(bind.first.request.arguments)
                if isinstance(node, Var)
            )
            continue
        if (
            bind.binder in weights
            and unused
            and not any(
                item in (score, random.entry.instance) for item in _nodes(bind.first)
            )
        ):
            # A weight only a removed density read, which draws and
            # scores nothing of the program's own.
            removed.add(bind.binder)
            continue
        rebuilt = bind.with_(then=rebuilt) if rebuilt is not bind.then else bind
    kept = {
        name: (
            f"observation {name!r} is read elsewhere in the program, or scored "
            "inside a computation it calls, so it is a datum the program is given"
        )
        for name in sorted(names - set(rewritten))
    }
    remaining = (
        ()
        if likelihoods
        else tuple(
            path for path, likelihood in roles.scores(rebuilt, score) if likelihood
        )
    )
    effects = computation.type.effects
    if not any(item == score for item in _nodes(rebuilt)):
        # No score is left: the computation performs none.
        effects = effects.with_(
            entries=tuple(item for item in effects.entries if item.instance != score)
        )
    if (
        not drop
        and rewritten
        and all(item.instance != random.entry.instance for item in effects.entries)
    ):
        effects = effects.with_(
            entries=(
                *effects.entries,
                RowEntry(instance=random.entry.instance, effect=random.entry.effect),
            )
        )
    role = ("prior" if drop else "generative") + ("" if likelihoods else ".unscored")
    variant = computation.with_(
        id=ComputationId.derive(module.module, "computation", computation.name, role),
        name=f"{computation.name}.{role}",
        parameters=tuple(
            parameter
            for parameter in computation.parameters
            if parameter.name not in rewritten
            and not (parameter.name in names and _mentions(rebuilt, parameter) == 0)
        ),
        body=rebuilt,
        type=ComputationType(effects=effects, result=computation.type.result),
    )
    return ObservationLaws(
        computation=variant,
        observations=tuple(reversed(rewritten)),
        kept=kept,
        unremoved=remaining,
    )


__all__ = ["ObservationLaws", "ScoreRoles", "observation_variant"]
