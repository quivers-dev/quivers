"""Response blocks: the namespace of one emitted program inside a model.

A model joins several emitted programs: the responses of a joint model, the
components and mixing-proportion predictors of a mixture. Each is emitted
alone under local names and placed in the model by a `Namespace`, the one
renaming rule of bindings, nominal objects, and compilation-local laws.
`rename_program` applies it to an emitted module.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import TypeVar, cast

import didactic.api as dx
import numpy as np

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    QiecIndexName,
    QiecValueParameter,
    TypeFromExpr,
    TypeName,
)
from quivers.dsl.program_templates import (
    TemplateBindings,
    _bound_names,
    _rename_step,
    substitute_program_object,
)
from quivers.dsl.step_resolution import _step_value_names
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._modifiers import response_columns
from quivers.formulas._residual_law import ResidualLaw
from quivers.formulas.formula import FixedColumn, Formula

__all__: list[str] = []

_Value = TypeVar("_Value")


class Namespace(ValueModel):
    """The renaming of one emitted program's names into a model.

    A binding is renamed by the first rule that applies: a shared name is
    kept; a name in ``renamed`` takes its entry; a name starting with a stem
    in ``stems`` has the stem replaced; any other name takes ``prefix``.
    A nominal object takes its entry in ``shared_objects`` and is otherwise
    renamed by the same rules without the shared names. Compilation-local
    laws take ``prefix``.

    Parameters
    ----------
    prefix : str
        Prefix of every name no other rule renames.
    renamed : dict[str, str]
        Names with a replacement of their own.
    stems : dict[str, str]
        Prefixes of local names and their replacements, tried in order.
    shared : frozenset[str]
        Bindings and inputs the model owns, which keep their names.
    shared_objects : dict[str, str]
        Nominal objects the model owns, with their names in the model.
    """

    prefix: str
    renamed: dict[str, str] = dx.field(default_factory=dict)
    stems: dict[str, str] = dx.field(default_factory=dict)
    shared: frozenset[str] = frozenset()
    shared_objects: dict[str, str] = dx.field(default_factory=dict)

    def local(self, value: str) -> str:
        """Rename a name by the rules other than the shared names.

        Parameters
        ----------
        value : str
            Local name.

        Returns
        -------
        str
            The name in the model.
        """
        if value in self.renamed:
            return self.renamed[value]
        for stem, replacement in self.stems.items():
            if value.startswith(stem):
                return replacement + value[len(stem) :]
        return self.prefix + value

    def name(self, value: str) -> str:
        """Rename one binding or input.

        Parameters
        ----------
        value : str
            Local name.

        Returns
        -------
        str
            The name in the model; a shared name is unchanged.
        """
        if value in self.shared:
            return value
        return self.local(value)

    def object(self, value: str) -> str:
        """Rename one nominal object.

        Parameters
        ----------
        value : str
            Local object name.

        Returns
        -------
        str
            The object's name in the model.
        """
        if value in self.shared_objects:
            return self.shared_objects[value]
        return self.local(value)

    def predictor(self, value: str) -> str:
        """Rename the coefficient prefix of a predictor.

        Parameters
        ----------
        value : str
            Local prefix: empty for the location.

        Returns
        -------
        str
            The prefix in the model.
        """
        return self.prefix if not value else self.name(value)


class ResponseColumn(ValueModel):
    """One column of a response term that a response program reads.

    Parameters
    ----------
    role : str
        Role of the column (``weights``, ``censoring``, ``trials``, ...).
    column : FixedColumn
        Checked values and input name.
    """

    role: str
    column: FixedColumn = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)


class ResponseData(ValueModel):
    """The rows and recorded values of one response, without any predictor.

    Parameters
    ----------
    name : str
        Response column.
    values : np.ndarray
        Response values, NaN where a value is not recorded.
    recorded : np.ndarray
        Boolean, one entry per row: whether the row's value is recorded.
    columns : tuple[ResponseColumn, ...]
        Columns of the response terms the program reads, in input order.
    """

    name: str
    values: np.ndarray = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    recorded: np.ndarray = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    columns: tuple[ResponseColumn, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    def column_items(self) -> tuple[tuple[str, FixedColumn], ...]:
        """Return the role and column of each input.

        Returns
        -------
        tuple[tuple[str, FixedColumn], ...]
            The columns in input order.
        """
        return tuple((item.role, item.column) for item in self.columns)


def response_data(formula: Formula, residual: ResidualLaw | None) -> ResponseData:
    """Read the response data of a formula.

    Parameters
    ----------
    formula : Formula
        Evaluated formula with the response and its terms.
    residual : ResidualLaw or None
        Residual law correlated across the rows, which may take response
        columns into its covariance.

    Returns
    -------
    ResponseData
        Values, recorded rows, and the columns the program reads.
    """
    values = np.asarray(formula.response_values)
    recorded: np.ndarray = (
        np.asarray(~np.isnan(values).reshape(len(values), -1).any(axis=1))
        if values.dtype.kind == "f"
        else np.ones(len(values), dtype=bool)
    )
    return ResponseData(
        name=formula.response_name,
        values=values,
        recorded=recorded,
        columns=tuple(
            ResponseColumn(role=role, column=column)
            for role, column in response_columns(formula.modifiers, residual)
        ),
    )


class RenamedProgram(ValueModel):
    """One emitted program in a model's names.

    Parameters
    ----------
    objects : tuple[ObjectDecl, ...]
        Renamed object declarations.
    program : ProgramDecl
        The renamed program.
    names : dict[str, str]
        Local bindings and inputs and their names in the model; shared names
        are left out.
    object_names : dict[str, str]
        Local object names and their names in the model.
    morphisms : dict[str, str]
        Compilation-local laws of the terms and their names in the model.
    families : FamilyCatalog or None
        Those laws under their names in the model.
    """

    objects: tuple[ObjectDecl, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    program: ProgramDecl = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    names: dict[str, str] = dx.field(default_factory=dict)
    object_names: dict[str, str] = dx.field(default_factory=dict)
    morphisms: dict[str, str] = dx.field(default_factory=dict)
    families: FamilyCatalog | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def renamed_indices(value: _Value, objects: Mapping[str, str]) -> _Value:
    """Rename nominal indices throughout a runtime parameter type.

    Parameters
    ----------
    value : _Value
        A type expression, a tuple of them, or a leaf field value.
    objects : Mapping[str, str]
        Local object names and their names in the model.

    Returns
    -------
    _Value
        The value with every nominal index renamed, of the same type.
    """
    if isinstance(value, QiecIndexName):
        return cast(_Value, value.with_(name=objects.get(value.name, value.name)))
    if isinstance(value, tuple):
        return cast(_Value, tuple(renamed_indices(item, objects) for item in value))
    if isinstance(value, dx.Model):
        return cast(
            _Value,
            value.with_(
                **{
                    key: renamed_indices(getattr(value, key), objects)
                    for key in type(value).__field_specs__
                }
            ),
        )
    return value


def rename_program(
    module: Module,
    namespace: Namespace,
    *,
    sites: Iterable[str] = (),
    inputs: Iterable[str] = (),
    omitted: Collection[str] = (),
    laws: FamilyCatalog | None = None,
) -> RenamedProgram:
    """Place one emitted single-program module in a model's names.

    Parameters
    ----------
    module : Module
        Emitted module: finite object declarations and one program.
    namespace : Namespace
        Renaming of the program.
    sites : Iterable[str]
        Sites the program's records name besides its own bindings, such as
        the support sites of its group blocks.
    inputs : Iterable[str]
        Inputs renamed by the namespace's rules even when they are shared;
        a name already renamed keeps its name.
    omitted : Collection[str]
        Runtime parameters left out of the renamed program, under local
        names.
    laws : FamilyCatalog or None
        Compilation-local laws of the program's terms.

    Returns
    -------
    RenamedProgram
        The renamed objects, program, and maps.

    Raises
    ------
    TypeError
        If an object lacks its finite declaration.
    """
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    object_names = {
        value: namespace.object(value)
        for item in module.statements
        if isinstance(item, ObjectDecl)
        for value in item.names
    }
    variables = set(_bound_names(program.draws)) | {
        parameter.name for parameter in program.value_parameters
    }
    variables.update(
        value for step in program.draws for value in _step_value_names(step)
    )
    variables.update(sites)
    names = {
        value: namespace.name(value)
        for value in variables
        if value not in namespace.shared
    }
    for value in inputs:
        names.setdefault(value, namespace.local(value))
    morphisms = (
        {}
        if laws is None
        else {
            record.name: namespace.prefix + record.name for record in laws.definitions
        }
    )
    bindings = TemplateBindings(objects=object_names, values={}, morphisms=morphisms)
    types = {key: TypeName(name=value) for key, value in object_names.items()}
    objects: list[ObjectDecl] = []
    for item in module.statements:
        if not isinstance(item, ObjectDecl):
            continue
        if not isinstance(item.init, TypeFromExpr):
            raise TypeError("formula objects require their retained finite declaration")
        objects.append(
            item.with_(
                names=tuple(object_names[value] for value in item.names),
                init=item.init.with_(
                    expr=substitute_program_object(item.init.expr, types)
                ),
            )
        )
    parameters: tuple[QiecValueParameter, ...] = tuple(
        parameter.with_(
            name=names.get(parameter.name, parameter.name),
            type_expr=renamed_indices(parameter.type_expr, object_names),
        )
        for parameter in program.value_parameters
        if parameter.name not in omitted
    )
    steps: tuple[ProgramStep, ...] = tuple(
        _rename_step(step, bindings, names) for step in program.draws
    )
    renamed = program.with_(
        name=namespace.prefix + program.name,
        value_parameters=parameters,
        domain=substitute_program_object(program.domain, types),
        codomain=substitute_program_object(program.codomain, types),
        draws=steps,
        return_vars=tuple(names.get(value, value) for value in program.return_vars),
    )
    return RenamedProgram(
        objects=tuple(objects),
        program=renamed,
        names=names,
        object_names=object_names,
        morphisms=morphisms,
        families=None if laws is None else laws.renamed(morphisms),
    )
