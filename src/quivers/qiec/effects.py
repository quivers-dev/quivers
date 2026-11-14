"""Typed effect interfaces, lexical rows, and handler signatures for QIEC."""

from __future__ import annotations

from collections.abc import Hashable
from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._models import QiecModel
import didactic.api as dx

from collections.abc import Mapping
from enum import Enum
from typing import Literal

from quivers.qiec.identifiers import (
    EffectInstanceId,
    HandlerId,
    OperationId,
    RowVariableId,
    SiteProvenance,
)
from quivers.qiec.kinds import Telescope, validate_telescope
from quivers.qiec.types import (
    EffectRef,
    StaticArgument,
    TypeExpr,
    check_static_arguments,
    render_static,
)


class RowVariable(QiecModel):
    """The open tail of an effect row, with the instances it excludes.

    Parameters
    ----------
    name : str
        Display name, used in diagnostics only. Two variables with the
        same identity and different names are the same variable.
    identity : RowVariableId
        Stable identity. Equality and hashing are taken from this rather
        than from `name`, so renaming for readability cannot silently
        split one variable into two.
    lacks : tuple[EffectInstanceId, ...]
        Instances the tail is proven not to contain. Sorted at
        construction, so equality is insensitive to the order given.
    """

    name: str
    identity: RowVariableId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    lacks: tuple[EffectInstanceId, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Sort the lacks constraints and reject a malformed variable.

        Raises
        ------
        ValueError
            If `name` is empty, or `lacks` names one instance twice.
        """
        if not self.name:
            raise ValueError("row variable name cannot be empty")
        ordered = tuple(sorted(self.lacks, key=str))
        if len(set(ordered)) != len(ordered):
            raise ValueError("a row lacks constraint cannot repeat an instance")
        self._normalize_field("lacks", ordered)

    def proves_lacks(self, instance: EffectInstanceId) -> bool:
        """Whether this tail is known not to contain `instance`.

        Parameters
        ----------
        instance : EffectInstanceId
            The lexical instance to test.

        Returns
        -------
        bool
            True when the constraint is carried, which is what permits an
            explicit entry for `instance` to sit beside this tail.
        """
        return instance in self.lacks

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Key the fields that define this kernel object's identity.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            Its nominal fields and recursively compared children.
        """
        return self._comparison_key(self.identity, self.lacks)

    def merge(self, other: RowVariable) -> RowVariable:
        """Unify two occurrences, accumulating their lacks constraints.

        Parameters
        ----------
        other : RowVariable
            Another occurrence of the same variable.

        Returns
        -------
        RowVariable
            One variable carrying the union of both constraint sets, which
            is sound because each occurrence's constraints hold of the
            single variable they both name.

        Raises
        ------
        ValueError
            If the two variables have different identities, and so are not
            occurrences of one variable at all.
        """
        if self.identity != other.identity:
            raise ValueError("cannot unify distinct row variables")
        merged = tuple(sorted(set((*self.lacks, *other.lacks)), key=str))
        return RowVariable(name=self.name, identity=self.identity, lacks=merged)

    def with_lacks(self, instances: tuple[EffectInstanceId, ...]) -> RowVariable:
        """Return this variable with further instances excluded.

        Parameters
        ----------
        instances : tuple[EffectInstanceId, ...]
            Instances to add to the constraint set. Instances already
            excluded are absorbed rather than repeated.

        Returns
        -------
        RowVariable
            A variable of the same identity carrying the wider constraint.
        """
        merged = tuple(sorted(set((*self.lacks, *instances)), key=str))
        return RowVariable(name=self.name, identity=self.identity, lacks=merged)


class RowEntry(QiecModel):
    """One lexical effect instance and the interface it implements.

    Parameters
    ----------
    instance : EffectInstanceId
        The lexical instance. Two instances of one interface stay
        distinct, so this rather than `effect` is the row's key.
    effect : EffectRef
        The saturated interface application the instance implements.
    """

    instance: EffectInstanceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    effect: EffectRef = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class EffectRow(QiecModel):
    """A finite map from lexical instance IDs to interfaces plus a tail.

    Entries are sorted by stable identifier at construction time, so row
    equality is insensitive to source order.  Two instances of the same
    interface remain distinct entries.

    Parameters
    ----------
    entries
        The instances the row names, each with its interface; any order is
        accepted and normalized.
    tail
        A row variable standing for further entries, or ``None`` for a
        closed row. An open row's tail must prove it lacks every explicit
        entry.
    """

    entries: tuple[RowEntry, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    tail: RowVariable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Sort the entries and reject a row that cannot be well formed.

        Raises
        ------
        ValueError
            If one lexical instance appears twice, or if the row is open
            and its tail does not prove it lacks every explicit entry.
            Without that proof the tail could later be instantiated to a
            row repeating an instance already named here.
        """
        ordered = tuple(sorted(self.entries, key=lambda entry: str(entry.instance)))
        instances = [entry.instance for entry in ordered]
        if len(set(instances)) != len(instances):
            raise ValueError("an effect row cannot contain a lexical instance twice")
        if self.tail is not None:
            unconstrained = [
                instance
                for instance in instances
                if not self.tail.proves_lacks(instance)
            ]
            if unconstrained:
                raise ValueError(
                    "an open row must prove its tail lacks every explicit instance"
                )
        self._normalize_field("entries", ordered)

    def lookup(self, instance: EffectInstanceId) -> EffectRef | None:
        """The interface an instance implements in this row.

        Parameters
        ----------
        instance : EffectInstanceId
            The lexical instance to resolve.

        Returns
        -------
        EffectRef or None
            The interface when the row carries the instance explicitly,
            and None otherwise. An open tail is not consulted, so None
            means absent from the explicit entries rather than absent
            from every instantiation of the row.
        """
        return next(
            (entry.effect for entry in self.entries if entry.instance == instance),
            None,
        )

    def contains(self, instance: EffectInstanceId) -> bool:
        """Whether the row carries an explicit entry for an instance.

        Parameters
        ----------
        instance : EffectInstanceId
            The lexical instance to test.

        Returns
        -------
        bool
            True when an explicit entry names the instance.
        """
        return self.lookup(instance) is not None

    def add(self, entry: RowEntry) -> EffectRow:
        """Return this row extended with one entry.

        Parameters
        ----------
        entry : RowEntry
            The instance and interface to add. Re-adding an entry the row
            already carries returns the row unchanged, so the operation is
            idempotent.

        Returns
        -------
        EffectRow
            The extended row. When the row is open, its tail gains a lacks
            constraint for the new instance, which keeps the invariant
            model validation enforces.

        Raises
        ------
        ValueError
            If the row already binds this instance to a different
            interface, since one lexical instance implements one
            interface.
        """
        existing = self.lookup(entry.instance)
        if existing is not None and existing != entry.effect:
            raise ValueError("one lexical instance cannot implement two interfaces")
        if existing is not None:
            return self
        tail = self.tail
        if tail is not None and not tail.proves_lacks(entry.instance):
            tail = tail.with_lacks((entry.instance,))
        return EffectRow(entries=(*self.entries, entry), tail=tail)

    def without(self, instance: EffectInstanceId) -> EffectRow:
        """Return this row with one instance discharged.

        Parameters
        ----------
        instance : EffectInstanceId
            The lexical instance to remove, as a handler does when it
            handles every operation of that instance.

        Returns
        -------
        EffectRow
            The row without that entry, keeping the tail as it was.

        Raises
        ------
        KeyError
            If the row carries no explicit entry for the instance. This
            is an error rather than a no-op because discharging an effect
            the row never had indicates the caller has the wrong row.
        """
        if not self.contains(instance):
            raise KeyError(str(instance))
        return EffectRow(
            entries=tuple(
                entry for entry in self.entries if entry.instance != instance
            ),
            tail=self.tail,
        )

    def union(self, other: EffectRow) -> EffectRow:
        """Union two rows, preserving a compatible open tail.

        Parameters
        ----------
        other : EffectRow
            The row to merge in.

        Returns
        -------
        EffectRow
            A row carrying every entry of both. When both are open their
            tails are merged, and the surviving tail is constrained to
            lack every explicit instance on either side.

        Raises
        ------
        ValueError
            If the two rows bind one instance to different interfaces, or
            if their tails are distinct variables and so cannot merge.
        """
        if self.tail is not None and other.tail is not None:
            tail = self.tail.merge(other.tail)
        else:
            tail = self.tail or other.tail
        if tail is not None:
            tail = tail.with_lacks(
                tuple(entry.instance for entry in (*self.entries, *other.entries))
            )
        result = EffectRow(entries=self.entries, tail=tail)
        for entry in other.entries:
            result = result.add(entry)
        return result


EMPTY_ROW = EffectRow()
"""The row of a computation that performs no effects."""


class RowSubstitution(QiecModel):
    """A finite, occurs-checked substitution for stable row variables.

    Parameters
    ----------
    bindings : tuple[tuple[RowVariableId, EffectRow], ...]
        What each row variable stands for. At most one binding per
        variable, and no binding whose row is tailed by the variable it
        replaces.
    """

    bindings: tuple[tuple[RowVariableId, EffectRow], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Reject a substitution that is ambiguous or immediately cyclic.

        Raises
        ------
        ValueError
            If one variable is bound twice, or if a variable is bound to
            a row whose own tail is that same variable. The second is the
            occurs check at depth one; deeper cycles are caught while
            applying the substitution.
        """
        identities = [identity for identity, _ in self.bindings]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate row substitution binding")
        for identity, row in self.bindings:
            if row.tail is not None and row.tail.identity == identity:
                raise ValueError("recursive row substitution")

    def lookup(self, variable: RowVariable) -> EffectRow | None:
        """The row a variable stands for under this substitution.

        Parameters
        ----------
        variable : RowVariable
            The variable to resolve, matched on its identity rather than
            its display name.

        Returns
        -------
        EffectRow or None
            The bound row, or None when the variable is unbound and so
            remains an open tail.
        """
        return next(
            (row for identity, row in self.bindings if identity == variable.identity),
            None,
        )

    def apply(self, row: EffectRow) -> EffectRow:
        """Replace this row's tail, and any tail it resolves to.

        Parameters
        ----------
        row : EffectRow
            The row to substitute into. A closed row is returned as is.

        Returns
        -------
        EffectRow
            The row with its tail resolved as far as the bindings reach.

        Raises
        ------
        ValueError
            If the bindings are cyclic, or if resolving a tail would
            introduce an instance that tail was proven to lack.
        """
        return self._apply(row, frozenset())

    def _apply(
        self,
        row: EffectRow,
        visiting: frozenset[RowVariableId],
    ) -> EffectRow:
        """Resolve one tail, tracking the variables already entered.

        Parameters
        ----------
        row : EffectRow
            The row whose tail is being resolved.
        visiting : frozenset[RowVariableId]
            Variables on the current resolution path. This is the occurs
            check: re-entering one of these means the bindings are
            cyclic.

        Returns
        -------
        EffectRow
            The row with its tail resolved.

        Raises
        ------
        ValueError
            If resolution re-enters a variable already on the path, or if
            the resolved row carries an instance the tail lacks.
        """
        if row.tail is None:
            return row
        replacement = self.lookup(row.tail)
        if replacement is None:
            return row
        if row.tail.identity in visiting:
            raise ValueError("recursive row substitution")
        resolved = self._apply(replacement, visiting | {row.tail.identity})
        forbidden = set(row.tail.lacks)
        if any(entry.instance in forbidden for entry in resolved.entries):
            raise ValueError("row substitution violates a lacks constraint")
        if resolved.tail is not None:
            resolved = EffectRow(
                entries=resolved.entries,
                tail=resolved.tail.with_lacks(row.tail.lacks),
            )
        return EffectRow(entries=row.entries).union(resolved)


class RowUnification(QiecModel):
    """The common row two rows unify to, and how their tails were bound.

    Parameters
    ----------
    row : EffectRow
        The unified row.
    substitution : RowSubstitution
        The bindings that carry each input row to `row`. Empty when the
        rows already agreed.
    """

    row: EffectRow = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    substitution: RowSubstitution = dx.field(
        default=RowSubstitution(), opaque=True, extras=OPAQUE_BY_VALUE
    )


def unify_effect_rows(left: EffectRow, right: EffectRow) -> RowUnification:
    """Unify two rows, returning their common row and tail substitution.

    Unification is symmetric and driven by the tails. Two closed rows
    unify only when already equal. One open row unifies with a closed one
    by binding its tail to the entries it lacks. Two open rows with
    distinct tails unify by binding both to a fresh tail carrying every
    instance either side names.

    Parameters
    ----------
    left : EffectRow
        One row to unify.
    right : EffectRow
        The other row. The result does not depend on the order of the two
        arguments.

    Returns
    -------
    RowUnification
        The common row and the substitution taking both inputs to it.

    Raises
    ------
    ValueError
        If the rows bind one instance to different interfaces, if two
        closed rows differ, if an open row carries an entry the closed
        row lacks, if one side needs an instance the other's tail is
        proven to lack, or if the computed substitution fails to carry
        both inputs to the unified row.
    """
    left_entries = {entry.instance: entry.effect for entry in left.entries}
    right_entries = {entry.instance: entry.effect for entry in right.entries}
    for instance in left_entries.keys() & right_entries.keys():
        if left_entries[instance] != right_entries[instance]:
            raise ValueError("row entries disagree on an effect interface")
    left_only = tuple(
        RowEntry(instance=instance, effect=effect)
        for instance, effect in left_entries.items()
        if instance not in right_entries
    )
    right_only = tuple(
        RowEntry(instance=instance, effect=effect)
        for instance, effect in right_entries.items()
        if instance not in left_entries
    )

    if left.tail is None and right.tail is None:
        if left_only or right_only:
            raise ValueError("closed effect rows do not unify")
        return RowUnification(row=left)

    if left.tail is not None and right.tail is None:
        if left_only:
            raise ValueError("an open row has entries absent from the closed row")
        replacement = EffectRow(entries=right_only)
        substitution = RowSubstitution(bindings=((left.tail.identity, replacement),))
        if substitution.apply(left) != right:
            raise ValueError("row substitution violates a lacks constraint")
        return RowUnification(row=right, substitution=substitution)

    if left.tail is None and right.tail is not None:
        result = unify_effect_rows(right, left)
        return RowUnification(row=result.row, substitution=result.substitution)

    assert left.tail is not None and right.tail is not None
    if left.tail.identity == right.tail.identity:
        if left_only or right_only:
            raise ValueError(
                "two occurrences of one row variable have incompatible entries"
            )
        tail = left.tail.merge(right.tail)
        return RowUnification(row=EffectRow(entries=left.entries, tail=tail))

    if any(entry.instance in left.tail.lacks for entry in right_only):
        raise ValueError("left row lacks an entry required by the right row")
    if any(entry.instance in right.tail.lacks for entry in left_only):
        raise ValueError("right row lacks an entry required by the left row")
    all_instances = tuple((*left_entries.keys(), *right_entries.keys()))
    fresh_identity = RowVariableId.derive(
        "unify",
        *sorted((left.tail.identity, right.tail.identity), key=str),
        *sorted(all_instances, key=str),
    )
    fresh = RowVariable(
        name="unified_tail",
        identity=fresh_identity,
        lacks=tuple(
            sorted(
                set((*left.tail.lacks, *right.tail.lacks, *all_instances)),
                key=str,
            )
        ),
    )
    left_replacement = EffectRow(entries=right_only, tail=fresh)
    right_replacement = EffectRow(entries=left_only, tail=fresh)
    substitution = RowSubstitution(
        bindings=(
            (left.tail.identity, left_replacement),
            (right.tail.identity, right_replacement),
        )
    )
    unified_entries = tuple(
        RowEntry(instance=instance, effect=effect)
        for instance, effect in {**left_entries, **right_entries}.items()
    )
    unified = EffectRow(entries=unified_entries, tail=fresh)
    if substitution.apply(left) != unified or substitution.apply(right) != unified:
        raise ValueError("row unification produced an inconsistent substitution")
    return RowUnification(row=unified, substitution=substitution)


class ComputationType(QiecModel):
    """The type of a computation: its effect row and result type.

    Parameters
    ----------
    effects
        The row of effect instances the computation may perform.
    result
        The type of the value it returns.
    """

    effects: EffectRow = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class ArgumentDef(QiecModel):
    """One named, typed value argument of an operation.

    Parameters
    ----------
    name
        The argument's display name.
    type
        The argument's type, which may mention the operation's telescope.
    """

    name: str
    type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class OperationDef(QiecModel):
    """One result-indexed request constructor in an effect interface.

    Parameters
    ----------
    id : OperationId
        Stable identity, derived from the owning effect and this
        operation's name.
    name : str
        Source name, as written in the effect declaration.
    telescope : Telescope
        Static binders this operation introduces on top of the
        interface's own. They may not shadow the interface binders.
    arguments : tuple[ArgumentDef, ...]
        Value arguments the request carries, named uniquely.
    result_type : TypeExpr
        What resuming the request supplies, possibly mentioning the
        binders above.
    """

    id: OperationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    telescope: Telescope = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arguments: tuple[ArgumentDef, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Validate the telescope and the argument names.

        Raises
        ------
        ValueError
            If the telescope is malformed, or two arguments share a name.
        """
        validate_telescope(self.telescope)
        names = [argument.name for argument in self.arguments]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate argument in operation {self.name!r}")


class EffectDef(QiecModel):
    """An operation interface.

    Identity is nominal: the declaration is fixed by its module and its
    own name, carried in `ref`. Static arguments distinguish applications
    of the interface without changing which declaration they apply.

    Parameters
    ----------
    ref : EffectRef
        The declaration's own reference, which must be unsaturated: a
        declaration names the interface, and applying it is what supplies
        arguments.
    telescope : Telescope
        Static binders the interface takes.
    operations : tuple[OperationDef, ...]
        The declared operations, unique by both identity and name.
    """

    ref: EffectRef = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    telescope: Telescope = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operations: tuple[OperationDef, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Validate the declaration against its operations.

        Raises
        ------
        ValueError
            If the telescope is malformed, if `ref` is saturated, if two
            operations share an identity or a name, or if an operation's
            telescope shadows an interface binder. Shadowing is rejected
            rather than resolved so that a binder in an operation
            signature always denotes the one thing.
        """
        validate_telescope(self.telescope)
        if self.ref.arguments:
            raise ValueError("an effect declaration ref must be unsaturated")
        ids = [operation.id for operation in self.operations]
        names = [operation.name for operation in self.operations]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate operation ID in effect {self.ref.name!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate operation name in effect {self.ref.name!r}")
        interface_names = {binder.name for binder in self.telescope}
        for operation in self.operations:
            overlap = interface_names & {binder.name for binder in operation.telescope}
            if overlap:
                raise ValueError(
                    f"operation {operation.name!r} shadows interface binders: {overlap!r}"
                )

    def apply(self, arguments: tuple[StaticArgument, ...]) -> EffectRef:
        """Instantiate the interface telescope as a concrete effect.

        Parameters
        ----------
        arguments : tuple[StaticArgument, ...]
            Static arguments to saturate the telescope with.

        Returns
        -------
        EffectRef
            The saturated application, carrying this declaration's
            identity together with the arguments given.

        Raises
        ------
        TypeError
            If an argument is of the wrong static class for its binder.
        ValueError
            If the argument count does not match the telescope, or an
            argument is ill-kinded.
        """
        check_static_arguments(self.telescope, arguments)
        return EffectRef(
            id=self.ref.id,
            name=self.ref.name,
            arguments=arguments,
        )

    def matches(self, effect: EffectRef) -> bool:
        """Whether a concrete effect is a well-kinded application here.

        Parameters
        ----------
        effect : EffectRef
            The saturated application to test.

        Returns
        -------
        bool
            True when `effect` names this declaration and its arguments
            saturate the telescope. False rather than raising, so callers
            can test candidates; use `apply` when the reason matters.
        """
        if effect.id != self.ref.id:
            return False
        try:
            self.apply(effect.arguments)
        except TypeError, ValueError:
            return False
        return True

    def operation(self, operation: OperationId) -> OperationDef:
        """The declared operation with a given identity.

        Parameters
        ----------
        operation : OperationId
            The identity to resolve.

        Returns
        -------
        OperationDef
            The matching declaration.

        Raises
        ------
        KeyError
            If this interface declares no such operation, which is how a
            request naming an operation another effect owns is caught.
        """
        for candidate in self.operations:
            if candidate.id == operation:
                return candidate
        raise KeyError(str(operation))


class ResumptionGrade(str, Enum):
    """Quantitative use allowed for a handler-clause resumption."""

    ZERO = "0"
    AFFINE = "aff"
    LINEAR = "1"
    UNRESTRICTED = "omega"


class HandlerClauseDef(QiecModel):
    """One operation a handler covers, and how it may resume.

    Parameters
    ----------
    operation : OperationId
        The operation this clause handles.
    grade : ResumptionGrade
        How often the clause may invoke its continuation. The grade is
        part of the clause contract and is checked against the body.
    parameters : tuple[Local, ...]
        Binders for the operation's value arguments, in declaration
        order. The body refers to the request's arguments through these.
    body : Computation or None
        What the clause does. ``None`` is a signature rather than an
        implementation, and is legal only for a handler whose
        implementation is supplied at runtime. An absent body never means
        "look up a callback": the declaration has to say so.
    """

    operation: OperationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    grade: ResumptionGrade = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameters: tuple[Local, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    body: Computation | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


class HandlerReturnClauseDef(QiecModel):
    """What a handler does with a value its computation returns.

    Separate from an operation clause because it answers a different
    thing: an operation clause responds to a request and may resume,
    while this responds to the computation finishing and has no
    continuation to invoke.

    Parameters
    ----------
    binder : Local
        Names the returned value inside the body.
    body : Computation
        What the handler produces from it. There is no signature-only
        form: a handler that answers returns at all has to say how.
    """

    binder: Local = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    body: Computation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class HandlerDef(QiecModel):
    """Serializable type-and-coverage signature for a handler.

    An authored handler carries its clause bodies in the stable core; a
    foreign handler's bodies are runtime attachments keyed by ``id``.

    Parameters
    ----------
    id
        The handler's stable identity.
    name
        The handler's display name.
    effect
        The interface application the handler handles.
    clauses
        One clause per operation the handler covers.
    input_type
        The result type of the computation the handler accepts.
    output_type
        The type of the handler's answer.
    introduced
        The effects the handler's own clauses perform, which the residual
        row of a handled computation gains.
    total
        Whether the clauses cover every operation of ``effect``.
    forwards_unknown
        Whether an operation no clause covers is forwarded outward rather
        than rejected; incompatible with ``total``.
    telescope
        The static parameters the handler abstracts over, in order.
    return_clause
        How the handler answers the computation's return, or ``None`` to
        return the value unchanged.
    implementation
        ``"authored"`` when every clause carries a body, ``"foreign"`` when
        none does and a runtime provider supplies the behavior.
    """

    id: HandlerId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str
    effect: EffectRef = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    clauses: tuple[HandlerClauseDef, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    input_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    output_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    introduced: EffectRow = dx.field(
        default=EMPTY_ROW, opaque=True, extras=OPAQUE_BY_VALUE
    )
    total: bool = True
    forwards_unknown: bool = False
    telescope: Telescope = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)
    return_clause: HandlerReturnClauseDef | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    implementation: Literal["authored", "foreign"] = "foreign"

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Validate the clause set against the coverage claim.

        Raises
        ------
        ValueError
            If the telescope is malformed, if two clauses cover the same
            operation, if the handler claims to be both total and
            forwarding, or if its bodies disagree with its declared
            implementation. A total handler covering every operation has
            nothing left to forward. An authored handler missing a body
            has nowhere for that operation's behavior to live, and a body
            on a foreign handler would be ignored in favour of its
            runtime provider; both are worse than being rejected.
        """
        validate_telescope(self.telescope)
        operations = [clause.operation for clause in self.clauses]
        if len(set(operations)) != len(operations):
            raise ValueError(f"duplicate clause in handler {self.name!r}")
        if self.total and self.forwards_unknown:
            raise ValueError("a handler cannot be both total and forwarding")
        authored = self.implementation == "authored"
        missing = [clause.operation for clause in self.clauses if clause.body is None]
        if authored and missing:
            raise ValueError(
                f"authored handler {self.name!r} has clauses without a body: "
                f"{missing!r}; write the bodies, or declare the handler "
                f"foreign if its behavior comes from a runtime provider"
            )
        supplied = [
            clause.operation for clause in self.clauses if clause.body is not None
        ]
        if not authored and supplied:
            raise ValueError(
                f"foreign handler {self.name!r} carries authored clause bodies: "
                f"{supplied!r}; a foreign handler's behavior comes from its "
                f"runtime provider, so a body here would be ignored"
            )
        if not authored and self.return_clause is not None:
            raise ValueError(
                f"foreign handler {self.name!r} carries a return clause; its "
                f"answer comes from its runtime provider"
            )

    def clause(self, operation: OperationId) -> HandlerClauseDef | None:
        """The clause covering an operation, if this handler has one.

        Parameters
        ----------
        operation : OperationId
            The operation to look for.

        Returns
        -------
        HandlerClauseDef or None
            The clause, or None when the operation is uncovered. For a
            forwarding handler None means the request passes through
            rather than that it is an error.
        """
        return next(
            (clause for clause in self.clauses if clause.operation == operation),
            None,
        )


class EffectRequest(QiecModel):
    """A typed request allocated before filtering or handler dispatch.

    Parameters
    ----------
    instance
        The effect instance the request is addressed to.
    effect
        The instance's interface application.
    operation
        The operation requested.
    static_arguments
        The operation's telescope instantiation, in binder order.
    arguments
        The value arguments, one per operation argument.
    result_type
        The type of the value the operation returns under the instantiation.
    origin
        Where the request was written and how it was reached.
    tag
        The serialization discriminator; always ``"effect_request"``.
    """

    instance: EffectInstanceId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    effect: EffectRef = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operation: OperationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    static_arguments: tuple[StaticArgument, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    arguments: tuple[Value, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result_type: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    origin: SiteProvenance = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["effect_request"] = "effect_request"


def instantiate_effect(
    effect: EffectRef,
    *,
    module: str,
    lexical_path: tuple[str | int, ...],
) -> RowEntry:
    """Allocate a deterministic lexical instance of an effect interface.

    The instance identity is derived rather than generated, so the same
    interface allocated at the same place in the same module is the same
    instance on every run. That is what lets a serialized row refer to an
    instance across processes.

    Parameters
    ----------
    effect : EffectRef
        The saturated interface application to allocate.
    module : str
        Module the allocation occurs in. Part of the identity, so two
        modules allocating at the same lexical path stay distinct.
    lexical_path : tuple[str | int, ...]
        Position of the allocation within the module. Two allocations of
        one interface at different paths are different instances, which
        is what makes two instances of the same effect distinguishable.

    Returns
    -------
    RowEntry
        The fresh instance paired with the interface it implements, ready
        to add to a row.
    """
    instance = EffectInstanceId.derive(
        str(effect.id),
        effect.arguments,
        module,
        lexical_path,
    )
    return RowEntry(instance=instance, effect=effect)


def render_instance(
    identity: EffectInstanceId, names: Mapping[EffectInstanceId, str] | None = None
) -> str:
    """Render one effect instance in the surface spelling.

    Parameters
    ----------
    identity : EffectInstanceId
        The instance.
    names : Mapping[EffectInstanceId, str] | None
        The source name of each lexical instance.

    Returns
    -------
    str
        Its source name, or ``instance:`` and the first eight characters
        of its digest when it has none.
    """
    table = names or {}
    return table.get(identity, f"instance:{identity.digest[:8]}")


def render_row(
    row: EffectRow, names: Mapping[EffectInstanceId, str] | None = None
) -> str:
    """Render an effect row in the surface spelling.

    Parameters
    ----------
    row : EffectRow
        The row.
    names : Mapping[EffectInstanceId, str] | None
        The source name of each lexical instance; an instance with no
        name renders as the first eight characters of its digest.

    Returns
    -------
    str
        ``!{a : Effect, b : Other[Int] | rho}``: each entry as its
        instance's name and interface, in name order, and an open row's
        tail after a bar, with the instances it lacks after a backslash.
    """
    entries = ", ".join(
        sorted(
            f"{render_instance(entry.instance, names)} : {render_static(entry.effect)}"
            for entry in row.entries
        )
    )
    if row.tail is None:
        return "!{" + entries + "}"
    tail = row.tail.name
    if row.tail.lacks:
        tail += " \\ " + ", ".join(
            render_instance(item, names) for item in row.tail.lacks
        )
    return "!{" + (entries + " | " if entries else "") + tail + "}"


__all__ = [
    "ArgumentDef",
    "ComputationType",
    "EMPTY_ROW",
    "EffectDef",
    "EffectRequest",
    "EffectRow",
    "HandlerClauseDef",
    "HandlerDef",
    "HandlerReturnClauseDef",
    "OperationDef",
    "ResumptionGrade",
    "RowEntry",
    "RowSubstitution",
    "RowUnification",
    "RowVariable",
    "instantiate_effect",
    "render_instance",
    "render_row",
    "unify_effect_rows",
]

# Handler clauses carry computations and requests carry values, and the
# serializer and `typing.get_type_hints` resolve a record's annotations at
# runtime, so `Computation`, `Local`, and `Value` have to be real names in
# this module rather than `TYPE_CHECKING` ones. The
# import sits at the foot of the file because `terms` imports the effect
# records defined above it: by the time control reaches here they exist,
# so the cycle closes rather than deadlocking.
from quivers.qiec.terms import Computation, Local, Value  # noqa: E402
