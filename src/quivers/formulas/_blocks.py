"""Response blocks: programs a model emits on their own and places by name.

A model joins several emitted programs: the responses of a joint model, the
components and mixing-proportion predictors of a mixture. Each is emitted
alone under local names and placed in the model by a `Namespace`, the one
renaming rule of bindings, nominal objects, and compilation-local laws.
`emit_block` emits one program and `place_block` renames it into the model.

A `ResponseBlock` is a response law a kernel emits as a nested program: its
family, location design, auxiliary formulas and fixed values, and the
namespace it takes in the model. `NestedBlocks` are the blocks of one kernel
and state which predictors they own. A predictor belongs to the block whose
kernel reads it, so the model plans only the predictors its own kernel reads
and the correlations that cross the blocks' predictors; each block plans the
rest. `emit_nested_block` emits one block and places it.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import TYPE_CHECKING, TypeVar, cast

import didactic.api as dx
import numpy as np

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprNode,
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    QiecIndexName,
    QiecValueParameter,
    SampleStep,
    Statement,
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
from quivers.formulas._group_law import GroupBlockLaw, level_names, renamed_law
from quivers.formulas._modifiers import response_columns
from quivers.formulas._observation import Observation
from quivers.formulas._priors import PriorTarget
from quivers.formulas._residual_law import ResidualLaw
from quivers.formulas._term import ColumnName, ProgramNames
from quivers.formulas.family import Family
from quivers.formulas.formula import FixedColumn, Formula, RandomTerm, _qvr_name

if TYPE_CHECKING:
    from quivers.formulas._row_source import RowSource
    from quivers.formulas.compile import FormulaData, FormulaToQVRModule

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


def emit_block(
    lens: FormulaToQVRModule,
    formula: Formula,
    *,
    supplied_groups: Mapping[tuple[str, str, str], LetExprNode] | None,
    supplied_sites: Collection[str],
    rows: RowSource | None,
) -> tuple[Module, FormulaData]:
    """Emit one block of a model under its local names.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler of the block.
    formula : Formula
        Location design of the block with its response.
    supplied_groups : Mapping or None
        Group coefficients the model emits, keyed by the block's predictor
        prefix, group, and slope.
    supplied_sites : Collection[str]
        Level hyperparameters the model draws and the block reads.
    rows : RowSource or None
        Rows the block covers; None for its training rows.

    Returns
    -------
    tuple[Module, FormulaData]
        The emitted single-program module and its formula complement.
    """
    return lens._forward(
        formula,
        supplied_groups=supplied_groups,
        supplied_sites=supplied_sites,
        check=False,
        rows=rows,
    )


def place_block(
    lens: FormulaToQVRModule,
    module: Module,
    namespace: Namespace,
    *,
    inputs: Iterable[str] = (),
    omitted: Collection[str] = (),
) -> RenamedProgram:
    """Place an emitted block in a model's names.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler that emitted the block, whose group blocks and term laws
        are renamed with it.
    module : Module
        The block's emitted module.
    namespace : Namespace
        Renaming of the block.
    inputs : Iterable[str]
        Inputs renamed even when they are shared.
    omitted : Collection[str]
        Runtime parameters left out, under local names.

    Returns
    -------
    RenamedProgram
        The renamed objects, program, and maps.
    """
    return rename_program(
        module,
        namespace,
        sites=(
            law.support_site for law in lens.group_laws if law.support_site is not None
        ),
        inputs=inputs,
        omitted=omitted,
        # Two blocks may draw from one law over different coordinates.
        laws=lens.term_laws,
    )


class ResponseBlock(ValueModel):
    """A response law that a kernel emits as a program of its own.

    The block is emitted by the ordinary compiler of its family and placed
    in the model under the names of the model parameter it computes: its
    location ``<location><index>``, each auxiliary ``<aux><index>``, and
    every other name in the namespace ``dpar_<location><index>_``.

    Parameters
    ----------
    location : str
        Parameter class of the block's location, such as ``mu`` for a
        mixture component and ``theta`` for a predicted mixing proportion.
    index : int
        One-based number of the block among the blocks of its class.
    auxiliaries : tuple[str, ...]
        Auxiliary parameters of the block's family, by local name.
    family : Family
        Family the block is emitted with.
    formula : Formula or None
        Location design over the shared response; None when the model gives
        the location no formula, which the block refuses when emitted.
    distributional : dict[str, Formula]
        Auxiliary formulas, by local name.
    fixed : dict[str, float]
        Fixed auxiliary values, by local name.
    trials : int or str or None
        Trial count or trial column of a family with finite trials.
    """

    location: str
    index: int
    auxiliaries: tuple[str, ...] = ()
    family: Family = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    formula: Formula | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    distributional: dict[str, Formula] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    fixed: dict[str, float] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    trials: int | str | None = None

    @property
    def label(self) -> str:
        """The model parameter of the block's location.

        Returns
        -------
        str
            ``<location><index>``.
        """
        return f"{self.location}{self.index}"

    @property
    def parameters(self) -> frozenset[str]:
        """The model parameters the block computes.

        Returns
        -------
        frozenset[str]
            The location and every auxiliary, under model names.
        """
        return frozenset(
            (self.label, *(f"{aux}{self.index}" for aux in self.auxiliaries))
        )

    def namespace(self, shared: frozenset[str]) -> Namespace:
        """Return the renaming of the block into the model.

        The location becomes ``<location><k>``, an auxiliary ``<aux><k>``, a
        name of an auxiliary's predictor ``dpar_<aux><k>_...``, and any other
        name ``dpar_<location><k>_...``.

        Parameters
        ----------
        shared : frozenset[str]
            Inputs, bindings and objects the model owns.

        Returns
        -------
        Namespace
            The renaming.
        """
        return Namespace(
            prefix=f"dpar_{self.label}_",
            renamed={
                "mu": self.label,
                **{aux: f"{aux}{self.index}" for aux in self.auxiliaries},
            },
            stems={
                f"dpar_{aux}_": f"dpar_{aux}{self.index}_" for aux in self.auxiliaries
            },
            shared=shared,
            shared_objects={name: name for name in shared},
        )

    def dpar(self, value: str | None) -> str | None:
        """Rename a predictor scope into the model.

        Parameters
        ----------
        value : str or None
            Local distributional parameter.

        Returns
        -------
        str or None
            The model parameter.
        """
        if value is None:
            return None
        if value == "mu":
            return self.label
        return f"{value}{self.index}"

    def prefixes(self) -> dict[str, str]:
        """Map the model's predictor prefixes of the block to local ones.

        Returns
        -------
        dict[str, str]
            Prefix in the model and the block's own prefix, for the location
            and every auxiliary.
        """
        return {f"dpar_{self.label}_": ""} | {
            f"dpar_{aux}{self.index}_": f"dpar_{aux}_" for aux in self.auxiliaries
        }

    def admits_category_specific(self, name: str) -> bool:
        """Report whether the block's own compiler checks a ``cs()`` design.

        Parameters
        ----------
        name : str
            Model parameter with a formula.

        Returns
        -------
        bool
            True for the location of a block whose family has ordinal
            thresholds, which reads category-specific terms.
        """
        return name == self.label and self.family.ordinal is not None


class NestedBlocks(ValueModel):
    """The blocks a response kernel emits as nested programs.

    A predictor belongs to the block whose kernel reads it. The model plans
    the predictors no block reads and the group blocks of correlations that
    cross the predictors of its blocks; each block plans its own.

    Parameters
    ----------
    blocks : tuple[ResponseBlock, ...]
        The blocks, in emission order.
    replaces_location : bool
        Whether the blocks' location designs take the place of the main
        design, which the model then plans no predictor for.
    """

    blocks: tuple[ResponseBlock, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    replaces_location: bool = False

    @property
    def parameters(self) -> frozenset[str]:
        """The model parameters the blocks compute.

        Returns
        -------
        frozenset[str]
            Every block's location and auxiliaries.
        """
        return frozenset(name for block in self.blocks for name in block.parameters)

    def reads(self, name: str) -> bool:
        """Report whether a block, not the model, reads a parameter's design.

        Parameters
        ----------
        name : str
            Model parameter, ``mu`` for the main formula.

        Returns
        -------
        bool
            True when a block plans the parameter's predictor.
        """
        return name in self.parameters or (name == "mu" and self.replaces_location)

    def admits_category_specific(self, name: str) -> bool:
        """Report whether a block checks the ``cs()`` design of a parameter.

        Parameters
        ----------
        name : str
            Model parameter with a formula.

        Returns
        -------
        bool
            True when a block reads category-specific terms for it.
        """
        return any(block.admits_category_specific(name) for block in self.blocks)

    def own(self, distributional: Mapping[str, Formula]) -> dict[str, Formula]:
        """Keep the distributional formulas the model plans.

        Parameters
        ----------
        distributional : Mapping[str, Formula]
            Every distributional formula, by model parameter.

        Returns
        -------
        dict[str, Formula]
            The formulas no block reads.
        """
        return {
            name: value
            for name, value in distributional.items()
            if not self.reads(name)
        }

    def grouping(
        self,
        main: Formula,
        distributional: Mapping[str, Formula],
        supplied: Collection[tuple[str, str, str]],
    ) -> tuple[Formula, dict[str, Formula]]:
        """Return the designs whose group blocks the model plans.

        A design no block reads is the model's in full. A design a block
        reads keeps only its group terms whose identifier, ``(1 | p | g)``,
        another such design shares: brms correlates them across every
        parameter, so their group block spans several designs, and the model
        emits it and gives its coefficients to the blocks. A correlation
        another program emits is not planned again, and a replaced main
        design keeps no term.

        Parameters
        ----------
        main : Formula
            Main formula.
        distributional : Mapping[str, Formula]
            Distributional formulas, by model parameter.
        supplied : Collection[tuple[str, str, str]]
            Coefficients, as (prefix, group, slope), another program emits.

        Returns
        -------
        tuple[Formula, dict[str, Formula]]
            The main design and the distributional designs.
        """
        if not self.blocks:
            return main, dict(distributional)

        def crossing(name: str, term: RandomTerm) -> bool:
            return (
                term.shared
                and term.correlation_id is not None
                and (f"dpar_{name}_", term.group, term.slope) not in supplied
            )

        owners: dict[str, set[str]] = {}
        for name, predictor in distributional.items():
            if not self.reads(name):
                continue
            for term in predictor.random_terms:
                if crossing(name, term) and term.correlation_id is not None:
                    owners.setdefault(term.correlation_id, set()).add(name)
        cross = {identifier for identifier, names in owners.items() if len(names) > 1}
        designs = {
            name: predictor
            if not self.reads(name)
            else _group_terms(
                predictor,
                tuple(
                    term
                    for term in predictor.random_terms
                    if crossing(name, term) and term.correlation_id in cross
                ),
            )
            for name, predictor in distributional.items()
        }
        return (_group_terms(main, ()) if self.reads("mu") else main), designs


def _group_terms(formula: Formula, terms: tuple[RandomTerm, ...]) -> Formula:
    """Keep only some group terms of a design.

    Parameters
    ----------
    formula : Formula
        A design.
    terms : tuple[RandomTerm, ...]
        Group terms kept.

    Returns
    -------
    Formula
        The design with its response and groupings, no population-level,
        smooth, offset, latent or monotonic terms, and only ``terms``.
    """
    return formula.with_(
        fixed_columns=(),
        design_terms=(),
        offsets=(),
        latent_terms=(),
        monotonic_terms=(),
        random_terms=terms,
    )


class PlacedBlock(ValueModel):
    """One nested block emitted and placed in a model's names.

    Parameters
    ----------
    objects : tuple[Statement, ...]
        Renamed object declarations.
    steps : tuple[ProgramStep, ...]
        Renamed program steps without the observation.
    observation : Observation
        The renamed observation, without the response events and weights of
        the model, which the model's observation takes.
    parameters : tuple[QiecValueParameter, ...]
        Renamed runtime parameters.
    prior_targets : tuple[PriorTarget, ...]
        Renamed prior catalogue entries with model parameter scopes.
    group_laws : tuple[GroupBlockLaw, ...]
        Renamed group-block laws.
    families : FamilyCatalog or None
        Laws its terms draw from, under model names.
    names : ProgramNames
        Names of its columns and bindings, under model names.
    """

    objects: tuple[Statement, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    steps: tuple[ProgramStep, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observation: Observation = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameters: tuple[QiecValueParameter, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    prior_targets: tuple[PriorTarget, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    group_laws: tuple[GroupBlockLaw, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    families: FamilyCatalog | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    names: ProgramNames = dx.field(
        default_factory=ProgramNames, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _sample_sites(module: Module) -> tuple[str, ...]:
    """Read every sampled latent site of an emitted module.

    Parameters
    ----------
    module : Module
        Emitted single-program module.

    Returns
    -------
    tuple[str, ...]
        Sampled site names.
    """
    return tuple(
        step.vars[0]
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, SampleStep)
    )


def _namespaced_owner(namespace: Namespace, owner: str) -> str:
    """Rename the owner of a private binding into the model.

    Parameters
    ----------
    namespace : Namespace
        Names of the block.
    owner : str
        Predictor namespace, followed by the parts of a group coefficient.

    Returns
    -------
    str
        The owner in the model.
    """
    prefix, separator, rest = owner.partition("|")
    return namespace.predictor(prefix) + separator + rest


def emit_nested_block(
    lens: FormulaToQVRModule,
    block: ResponseBlock,
    *,
    shared: frozenset[str],
    supplied: Mapping[tuple[str, str, str], LetExprNode],
    supplied_sites: Collection[str],
) -> PlacedBlock:
    """Emit a nested block through its family's compiler and place it.

    The block's compiler inherits the model's options. The levels of a
    grouping factor are the response's, so the block reads the model's level
    names, and site priors the model's configuration gives under model names
    are given to the block under its local names.

    Parameters
    ----------
    lens : FormulaToQVRModule
        The model's compiler, whose options and priors the block inherits.
    block : ResponseBlock
        The block.
    shared : frozenset[str]
        Inputs, bindings and objects the model owns.
    supplied : Mapping[tuple[str, str, str], LetExprNode]
        Group coefficients the model emits, keyed by the model's predictor
        prefix, group, and slope.
    supplied_sites : Collection[str]
        Level hyperparameters the model draws.

    Returns
    -------
    PlacedBlock
        The renamed program parts.

    Raises
    ------
    ValueError
        If the block's location has no formula, or the block's compiler
        rejects its formula, modifiers or options.
    """
    formula = block.formula
    if formula is None:
        raise ValueError(f"the response block {block.label!r} has no location formula")
    groups = {
        group
        for predictor in (formula, *block.distributional.values())
        for group in predictor.group_levels
    }
    covariances = {
        group: covariance
        for group, covariance in lens.group_covariances.items()
        if group in groups
    }
    prefixes = block.prefixes()
    gathered = {
        (prefixes[prefix], group, slope): value
        for (prefix, group, slope), value in supplied.items()
        if prefix in prefixes
    }

    def build(
        priors: Mapping[str, str], namespace: Namespace
    ) -> tuple[FormulaToQVRModule, Module]:
        sub = lens.block_compiler(block, priors, covariances)
        module, _ = emit_block(
            sub,
            formula,
            supplied_groups=gathered,
            supplied_sites=supplied_sites,
            rows=lens.rows.within(namespace.predictor)
            if lens.rows is not None
            else None,
        )
        return sub, module

    namespace = block.namespace(shared)
    sub, module = build({}, namespace)
    # The blocks of a response share the levels of a grouping factor.
    namespace = block.namespace(
        shared | level_names(sub.group_laws) | set(supplied_sites)
    )
    configured = lens.site_priors
    priors = {
        site: configured[namespace.name(site)]
        for site in _sample_sites(module)
        if namespace.name(site) in configured
    }
    if priors:
        sub, module = build(priors, namespace)
    group_names = {_qvr_name(group) for group in groups}
    namespace = namespace.with_(
        shared=namespace.shared | {f"{name}_idx" for name in group_names},
        shared_objects=namespace.shared_objects | {name: name for name in group_names},
    )
    renamed = place_block(sub, module, namespace)
    names, objects, morphisms = (
        renamed.names,
        renamed.object_names,
        renamed.morphisms,
    )
    steps = list(renamed.program.draws)
    observed = sub.observation
    if observed is None:
        raise ValueError("a nested response block emitted no observation")
    observation = observed.renamed(names, morphisms, objects)
    # The block's own observation is the last step it writes; the model
    # scores the block's law instead.
    written = steps.pop()
    if written != observation.step():
        raise ValueError("a nested response block wrote a step after its observation")
    return PlacedBlock(
        objects=renamed.objects,
        steps=tuple(steps),
        observation=observation.with_events(()).with_(weights=None),
        parameters=renamed.program.value_parameters,
        prior_targets=tuple(
            target.with_(
                site=names.get(target.site, namespace.name(target.site)),
                axes=tuple(objects.get(axis, axis) for axis in target.axes),
                class_=f"{target.class_}{block.index}"
                if target.class_ in block.auxiliaries
                else target.class_,
                dpar=block.dpar(target.dpar),
            )
            for target in sub.prior_targets
        ),
        group_laws=tuple(
            renamed_law(
                law,
                names,
                objects,
                namespace.predictor,
                namespace.prefix + law.identity,
            )
            for law in sub.group_laws
        ),
        families=renamed.families,
        names=ProgramNames(
            inputs={
                ColumnName(
                    prefix=namespace.predictor(key.partition(":")[0]),
                    column=key.partition(":")[2],
                ).key: namespace.name(value)
                for key, value in sub.names.inputs.items()
            },
            bindings={
                f"{_namespaced_owner(namespace, owner)}:{name}": namespace.name(value)
                for key, value in sub.names.bindings.items()
                for owner, _, name in (key.rpartition(":"),)
            },
        ),
    )
