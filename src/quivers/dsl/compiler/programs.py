"""Compiler mixin: program / contraction / define compilation.

Handles bind-step expansion, effect verification, template inlining,
program bodies, contractions, define bindings, and let-expression
compilation.
"""

from __future__ import annotations
import dataclasses
import inspect
from abc import abstractmethod
import math
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from dataclasses import replace as _dc_replace
from itertools import product as _cartesian_product
from typing import Literal, cast
import didactic.api as dx
import torch
from torch.distributions import constraints as _constraints
from quivers.continuous.deterministic import (
    LearnedFunction,
    learned_function_draw_message,
)
from quivers.continuous.family_spec import (
    FAMILY_REGISTRY,
    FamilySpec,
    build_torch_distribution,
    clamp_param,
)
from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
    MarginalizedFactor,
    _event_dim,
)
from quivers.core.algebras import CompositionRule
from quivers.core.morphisms import Morphism, ObservedMorphism

from quivers.continuous.inline import (
    UnsizedParameterError,
    get_inline_param_names,
    make_inline_distribution,
)
from quivers.continuous.plate import PlateDraw, VectorisedObserve, marginalize_grouped
from quivers.continuous.program_steps import (
    Call,
    Draw,
    Indexed,
    Let,
    Observe,
    Scan,
    Score,
    Step,
    StepArgument,
    reading,
    reads_of,
)
from quivers.continuous.programs import MonadicProgram, _lookup_arg
from quivers.effects.checked_program import CheckedProgram
from quivers.continuous.param_source import (
    AttentionSource,
    IdentitySource,
    ParamSource,
    attention_draw_message,
)
from quivers.continuous.spaces import (
    CholeskyFactor,
    ContinuousSpace,
    Covariance,
    Euclidean,
    LowerTriangular,
    PositiveReals,
    ProductSpace,
    Simplex,
    UnitInterval,
)
from quivers.core.objects import FinSet, ProductSet, SetObject
from quivers.core.wiring import EinsumWiring
from quivers.dsl.ast_nodes import (
    AxisSpec,
    BindStep,
    ContractionDecl,
    ContractionInput,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    DrawStep,
    Expr,
    ExprIdent,
    ExprMorphismCall,
    ExprTransCompose,
    DefineDecl,
    GroupedBodyObserveStep,
    GroupedLatentInitStep,
    GroupedObserveEntry,
    LetExprBinOp,
    LetExprBool,
    LetExprCall,
    LetExprIndex,
    LetExprLambda,
    LetExprFactor,
    LetExprList,
    LetFactorCase,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprTuple,
    LetExprUnaryOp,
    LetExprUnit,
    LetExprVar,
    LetStep,
    CallStep,
    ScoreStep,
    MarginalizeStep,
    Module,
    GroupedMarginalizeStep,
    MorphismParam,
    ObjectParam,
    ObserveStep,
    PlateDrawStep,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    ScalarParam,
    ObjectExpr,
    TypeName,
    ObjectProduct,
    VectorisedObserveStep,
)
from quivers.dsl.draw_args import bind_family_arguments
from quivers.dsl.compiler._options import (
    check_option_keys,
    find_option,
    get_option_name,
    get_option_name_list,
    get_option_string,
    get_program_effects,
    get_program_over_model,
)
from quivers.dsl.compiler._deduction_terms import (
    LET_BOUND_VARIABLES_KEY,
    LET_CONSTRUCTORS_KEY,
    LET_INTERNAL_KEYS,
    atom_term,
    variable_term,
)
from quivers.dsl.compiler._call_lowering import lower_calls
from quivers.transpile._pure_calls import LOG_PROB_CALL
from quivers.dsl.compiler.sugar import desugar_step
from quivers.dsl.composite_lets import (
    expand_program_scans,
    ScanCall,
    is_learned_scan_init,
    positions_let_call,
    scan_let_call,
)
from quivers.dsl.let_expr_traversal import (
    free_let_callees,
    free_let_names,
    map_let_expr,
    substitute_let_expr,
    walk_let_expr,
)
from quivers.dsl.markov_chains import chain_cells, markov_cell
from quivers.dsl.step_resolution import StepResolutionError, _expr_ident_names
from quivers.dsl.pure_builtins import EAGER_BUILTINS
from quivers.program import Program
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import QiecModule
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.terms import Computation, Local
from quivers.qiec.families import FAMILIES
from quivers.qiec.types import REAL, IndexLiteral, TypeExpr
from quivers.stochastic.agenda import DeductionSystem
from quivers.dsl.compiler._prelude import (
    CompileError,
    _CompiledContraction,
    _ALGEBRA_REGISTRY,
    _get_family_registry,
    _numel_shape,
)


def _arg_names(args: tuple) -> tuple[str, ...]:
    """The environment names a tuple of draw arguments reads.

    Parameters
    ----------
    args : tuple[DrawArgName | DrawArgIndex | Indexed | str, ...]
        The arguments as the compiler carries them, in source form or
        as step arguments.

    Returns
    -------
    tuple[str, ...]
        The referenced names, an indexed reference's indices included;
        a scalar literal reads nothing.
    """
    names: list[str] = []
    for arg in args:
        if isinstance(arg, DrawArgNamed):
            names.extend(_arg_names((arg.value,)))
        elif isinstance(arg, DrawArgDist):
            names.extend(_arg_names(arg.args))
        elif isinstance(arg, DrawArgList):
            names.extend(_arg_names(arg.items))
        elif isinstance(arg, str):
            names.append(arg)
        elif isinstance(arg, DrawArgIndex | Indexed):
            names.append(arg.name)
            names.extend(arg.indices)
        elif isinstance(arg, DrawArgName):
            names.append(arg.text)
    return tuple(names)


def _per_row_parameters(
    family: str,
    args: tuple,
    parameter_names: tuple[str, ...] | list[str],
    value_extents: Mapping[str, tuple[int, ...]],
    rows: int,
) -> frozenset[str]:
    """The bindings a plated observe reads one entry of per row.

    ``observe y : Row <- Normal(loc=z, ...)`` with ``z`` one vector over
    ``Row`` scores row ``i`` at ``z[i]``: a binding of exactly the
    plate's rows filling a rank-0 parameter holds one value per row,
    not a vector every row reads whole.

    Parameters
    ----------
    family : str
        The inline family's name.
    args : tuple
        The draw's arguments, positional or named.
    parameter_names : tuple[str, ...] | list[str]
        The family's parameters in positional order.
    value_extents : Mapping[str, tuple[int, ...]]
        The checked extents of the program's bindings.
    rows : int
        The plate's rows.

    Returns
    -------
    frozenset[str]
        The names of the bindings read at each row.
    """
    record = FAMILIES.get(family)
    if record is None:
        return frozenset()
    found: set[str] = set()
    for position, arg in enumerate(args):
        if isinstance(arg, DrawArgNamed):
            parameter: str | None = arg.parameter
            value = arg.value
        else:
            parameter = (
                parameter_names[position] if position < len(parameter_names) else None
            )
            value = arg
        if (
            parameter is None
            or parameter not in record.parameter_names
            or record.parameter(parameter).rank != 0
            or not isinstance(value, DrawArgName)
            or tuple(value_extents.get(value.text, ())) != (rows,)
        ):
            continue
        found.add(value.text)
    return frozenset(found)


def _let_expr_reads(node: LetExprNode) -> frozenset[str]:
    """The names a let expression reads from its environment.

    Parameters
    ----------
    node : LetExprNode
        The expression.

    Returns
    -------
    frozenset[str]
        Every variable the expression references outside a binder of
        its own (a lambda's parameter, a factor's index variables). A
        name that resolves elsewhere (a deduction, a built-in) may be
        included; the encoding passes only the names bound in the
        program.
    """
    return frozenset(free_let_names(node))


def _checked_real_space(name: str, type_: TypeExpr, per_row: bool) -> Euclidean | None:
    """The Euclidean space a checked real-valued datum occupies.

    Parameters
    ----------
    name : str
        The datum's name.
    type_ : TypeExpr
        Its checked type.
    per_row : bool
        Whether the datum holds one value per row of a plate, along its
        leading extent.

    Returns
    -------
    Euclidean | None
        One coordinate for ``Real``; for ``Tensor[Real]`` of fixed
        extents its entries, as rows of its leading extent when it
        ranges over a plate. ``None`` for any other element type or an
        extent no literal fixes.
    """
    split = tensor_shape(type_)
    element, dimensions = (type_, ()) if split is None else split
    if element != REAL:
        return None
    extents: list[int] = []
    for dimension in dimensions:
        if not isinstance(dimension, IndexLiteral) or not isinstance(
            dimension.value, int
        ):
            return None
        extents.append(dimension.value)
    if not extents:
        return Euclidean(name=f"_{name}", dim=1)
    total = math.prod(extents)
    return Euclidean(
        name=f"_{name}", dim=total, plate_rows=extents[0] if per_row else None
    )


def _static_extents(type_: TypeExpr) -> tuple[int, ...] | None:
    """The extents of a checked value whose every extent a literal fixes.

    Parameters
    ----------
    type_ : TypeExpr
        The value's checked type.

    Returns
    -------
    tuple[int, ...] | None
        ``()`` for a value that is not a tensor, a tensor's extents
        outermost first, or ``None`` when an extent is not a literal.
    """
    split = tensor_shape(type_)
    if split is None:
        return ()
    extents: list[int] = []
    for dimension in split[1]:
        if not isinstance(dimension, IndexLiteral) or not isinstance(
            dimension.value, int
        ):
            return None
        extents.append(dimension.value)
    return tuple(extents)


def _checked_local_extents(
    parameters: tuple[Local, ...], body: Computation
) -> dict[str, tuple[int, ...]]:
    """The extents of every value a checked computation binds by name.

    The computation's parameters and every local its body binds (each
    site's draw and each let) are read off their checked types. A name
    bound at two different types, or at an extent no literal fixes, is
    left out, so no width is read for it.

    Parameters
    ----------
    parameters : tuple[Local, ...]
        The computation's parameters.
    body : Computation
        The computation's body, a tree of checked terms.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Each name's extents, outermost first.
    """
    types: dict[str, set[TypeExpr]] = {}
    seen: set[int] = set()
    pending: list[Hashable] = [*parameters, body]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Local):
            types.setdefault(node.name, set()).add(node.type)
        elif isinstance(node, tuple | list):
            pending.extend(node)
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            pending.extend(
                getattr(node, field.name) for field in dataclasses.fields(node)
            )
    extents: dict[str, tuple[int, ...]] = {}
    for name, bound in types.items():
        if len(bound) != 1:
            continue
        static = _static_extents(next(iter(bound)))
        if static is not None:
            extents[name] = static
    return extents


def _space_rows(space: AnySpace | None) -> tuple[int | None, int] | None:
    """A parameter space as plate rows and per-row width.

    Parameters
    ----------
    space : AnySpace | None
        The space.

    Returns
    -------
    tuple[int | None, int] | None
        The plate rows (``None`` for a value shared by every row) and
        the width of one row; ``None`` for a space with no Euclidean
        reading.
    """
    if isinstance(space, Euclidean):
        return (space.plate_rows, space.row_width)
    if isinstance(space, ContinuousSpace) and not isinstance(space, ProductSpace):
        dim = getattr(space, "dim", None)
        if isinstance(dim, int) and len(space.event_shape) <= 1:
            return (None, dim)
    return None


#: Builtins that apply entrywise, so their result has the shape of
#: their argument.
_ENTRYWISE_BUILTINS = frozenset(
    {
        "abs",
        "exp",
        "expm1",
        "int",
        "log",
        "log1p",
        "neg",
        "real",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "softplus",
        "sqrt",
        "square",
        "tanh",
    }
)


def _let_space(
    expr: LetExprNode,
    bound: Mapping[str, AnySpace | None],
    known: Mapping[str, AnySpace],
    functions: Mapping[str, int],
) -> Euclidean | None:
    """The space an entrywise let expression's value occupies.

    A variable, a number, and entrywise arithmetic and builtins over
    them have the broadcast shape of their operands: a value shared by
    every row broadcasts against one plated over rows, and a single
    coordinate against several.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    bound : Mapping[str, AnySpace | None]
        The spaces of the names bound in the body.
    known : Mapping[str, AnySpace]
        The spaces of host data and of earlier lets.
    functions : Mapping[str, int]
        The codomain width of each learned function by name; a call of
        one has that width per row of its arguments.

    Returns
    -------
    Euclidean | None
        The broadcast space; ``None`` for any other expression or an
        operand whose space is unknown or whose shape does not
        broadcast.
    """

    def shape(node: LetExprNode) -> tuple[int | None, int] | None:
        """The plate rows and per-row width of a let expression's value.

        Parameters
        ----------
        node : LetExprNode
            The expression.

        Returns
        -------
        tuple[int | None, int] | None
            The rows (``None`` for a value shared by every row) and the
            width of one row, or ``None`` when they cannot be read.
        """
        if isinstance(node, LetExprLiteral):
            return (None, 1)
        if isinstance(node, LetExprVar):
            space = bound.get(node.name)
            return _space_rows(space if space is not None else known.get(node.name))
        if isinstance(node, LetExprUnaryOp) and node.op == "-":
            return shape(node.operand)
        if isinstance(node, LetExprBinOp) and node.op in ("+", "-", "*", "/"):
            left, right = shape(node.left), shape(node.right)
            if left is None or right is None:
                return None
            if left[0] is not None and right[0] is not None and left[0] != right[0]:
                return None
            if left[1] != right[1] and 1 not in (left[1], right[1]):
                return None
            return (
                left[0] if left[0] is not None else right[0],
                max(left[1], right[1]),
            )
        if (
            isinstance(node, LetExprCall)
            and node.func in _ENTRYWISE_BUILTINS
            and len(node.args) == 1
        ):
            return shape(node.args[0])
        if isinstance(node, LetExprCall) and node.func in functions:
            # A network maps each row of its arguments: its value has
            # their plate rows, if any, and its codomain's width.
            rows: int | None = None
            for argument in node.args:
                found = shape(argument)
                if found is not None and found[0] is not None:
                    rows = found[0]
            return (rows, functions[node.func])
        return None

    found = shape(expr)
    if found is None:
        return None
    rows, width = found
    return Euclidean(
        name="_let",
        dim=width if rows is None else rows * width,
        plate_rows=rows,
    )


def _first_call_step(steps: tuple[ProgramStep, ...]) -> CallStep | None:
    """The first step of a body that calls a named computation.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The body, whose marginalization scopes are searched too.

    Returns
    -------
    CallStep | None
        The first `CallStep` at any depth, or ``None``.
    """
    for step in steps:
        if isinstance(step, CallStep):
            return step
        if isinstance(step, MarginalizeStep):
            found = _first_call_step(step.scope)
            if found is not None:
                return found
    return None


def _names_in(steps: tuple[ProgramStep, ...]) -> set[str]:
    """Every name a program's steps bind or read.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps.

    Returns
    -------
    set[str]
        The names: every bound variable and every variable an
        expression reads, host data included.
    """
    found: set[str] = set()

    def visit(node: dx.Model | tuple[dx.Model | str, ...]) -> None:
        if isinstance(node, LetExprVar):
            found.add(node.name)
        if isinstance(node, tuple):
            for item in node:
                if not isinstance(item, str):
                    visit(item)
            return
        specs = getattr(type(node), "__field_specs__", None)
        if specs is None:
            return
        for field_name in specs:
            value = getattr(node, field_name, None)
            if field_name in ("name", "var") and isinstance(value, str):
                found.add(value)
            elif field_name == "vars" and isinstance(value, tuple):
                found.update(item for item in value if isinstance(item, str))
            elif isinstance(value, tuple) or hasattr(type(value), "__field_specs__"):
                visit(value)

    visit(steps)
    return found


def _fresh_name(base: str, occupied: set[str]) -> str:
    """A name not in ``occupied``, which it joins.

    Parameters
    ----------
    base : str
        The preferred name.
    occupied : set[str]
        The names in use.

    Returns
    -------
    str
        ``base``, or ``base`` with a numeric suffix.
    """
    candidate = base.lstrip("_") or "arg"
    suffix = 0
    while candidate in occupied:
        suffix += 1
        candidate = f"{base.lstrip('_') or 'arg'}_{suffix}"
    occupied.add(candidate)
    return candidate


def _calls_a_computation(steps: tuple[ProgramStep, ...]) -> bool:
    """Whether a program body calls a named computation anywhere.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The body, whose marginalization scopes are searched too.

    Returns
    -------
    bool
        ``True`` if a `CallStep` appears at any depth.
    """
    for step in steps:
        if isinstance(step, CallStep):
            return True
        if isinstance(step, MarginalizeStep) and _calls_a_computation(step.scope):
            return True
    return False


def _scanned_step_programs(steps: tuple[ProgramStep, ...]) -> tuple[str, ...]:
    """The step programs a body's scans run.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps, searched through marginalization scopes.

    Returns
    -------
    tuple[str, ...]
        The step program of every ``let h = scan(step, xs)``.
    """
    found: list[str] = []
    for step in steps:
        if isinstance(step, MarginalizeStep):
            found.extend(_scanned_step_programs(step.scope))
        elif isinstance(step, LetStep):
            scan = scan_let_call(step.value)
            if scan is not None and not scan.marginal:
                found.append(scan.step)
    return tuple(found)


def _marginal_chains(steps: tuple[ProgramStep, ...]) -> bool:
    """Whether a body draws a marginalized chain.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The expanded steps.

    Returns
    -------
    bool
        ``True`` when a ``let h = scan(cell, xs, p, "marginal")`` is
        among them.
    """
    for step in steps:
        if isinstance(step, LetStep):
            scan = scan_let_call(step.value)
            if scan is not None and scan.marginal:
                return True
    return False


def _observation_parameter(
    env: Mapping[str, torch.Tensor],
    arguments: tuple[StepArgument, ...] | None,
    shared_vector: bool,
) -> torch.Tensor:
    """The parameter an observation family is scored at, at one atom.

    Parameters
    ----------
    env : Mapping[str, torch.Tensor]
        The bindings at the atom.
    arguments : tuple[StepArgument, ...] | None
        The observation's arguments; ``None`` reads the program input.
    shared_vector : bool
        Whether the family takes one vector-valued parameter, which a
        single shared value presents with a leading axis of one.

    Returns
    -------
    torch.Tensor
        The parameter: one argument as it is (a number as one column),
        or several stacked along a last axis.
    """
    if arguments is None:
        return env["_x_input"]
    parts = [_lookup_arg(env, argument) for argument in arguments]
    if len(parts) == 1:
        theta = parts[0]
        if theta.dim() == 0:
            # One value at this atom, shared by every row: a single
            # column, as the stacking of several parameters gives each.
            theta = theta.reshape(1)
        if shared_vector and theta.dim() == 1:
            theta = theta.unsqueeze(0)
        return theta
    common = torch.broadcast_shapes(*(part.shape for part in parts))
    theta = torch.stack([part.expand(common) for part in parts], dim=-1)
    if theta.dim() == 1:
        # Values shared by every row at this atom stack to one row of
        # their columns.
        theta = theta.unsqueeze(0)
    return theta


def _binary_log_prior(
    weight: torch.Tensor, parameterization: str | None
) -> torch.Tensor:
    """The log weights of the atoms 0 and 1 of a Bernoulli-family prior.

    From ``logits`` the weights are ``log sigmoid(-l)`` and
    ``log sigmoid(l)``, computed in log space, so both stay finite and
    accurate where the probability ``sigmoid(l)`` rounds to zero or one
    in the runtime's precision. From ``probs`` they are ``log1p(-p)`` and
    ``log p``, with ``p`` held inside the open unit interval.

    Parameters
    ----------
    weight : torch.Tensor
        The prior's probability, or its logits.
    parameterization : str | None
        ``"logits"`` when `weight` holds logits; anything else reads it
        as a probability.

    Returns
    -------
    torch.Tensor
        The log weights, stacked on a new trailing axis in atom order.
    """
    if parameterization == "logits":
        return torch.stack(
            [
                torch.nn.functional.logsigmoid(-weight),
                torch.nn.functional.logsigmoid(weight),
            ],
            dim=-1,
        )
    p = weight.clamp(1e-38, 1.0 - 1e-7)
    return torch.stack([torch.log1p(-p), torch.log(p)], dim=-1)


def _shared_vector_parameter(family: ContinuousMorphism | None) -> bool:
    """Whether an observation family takes one vector-valued parameter.

    Parameters
    ----------
    family : ContinuousMorphism | None
        The family.

    Returns
    -------
    bool
        ``True`` when its single variable parameter slot is wider than
        one, as a ``Categorical`` row is.
    """
    spec = getattr(family, "_param_spec", None)
    widths = (
        [1 if width is None else int(width) for kind, width in spec if kind == "var"]
        if spec is not None
        else []
    )
    return len(widths) == 1 and widths[0] > 1


def _draw_bundle(
    value: LetExprNode, bound_vars: dict[str, AnySpace | None]
) -> tuple[str, ...] | None:
    """The draws a list literal bundles, when it bundles only draws.

    The expansion of ``fan`` and ``@`` in a chain gathers the branches'
    tails, each a per-row vector a draw or a scan bound, into a list
    the next element of the chain reads as its input. That input is the
    branches' product, the vectors laid end to end per row, as the
    checked elaboration flattens it.

    Parameters
    ----------
    value : LetExprNode
        A let's value.
    bound_vars : dict[str, AnySpace | None]
        The names bound so far and their spaces.

    Returns
    -------
    tuple[str, ...] | None
        The bundled names, when ``value`` is a list of names each bound
        to a continuous space; ``None`` otherwise.
    """
    if not isinstance(value, LetExprList) or not value.items:
        return None
    names: list[str] = []
    for item in value.items:
        if not isinstance(item, LetExprVar) or not isinstance(
            bound_vars.get(item.name), ContinuousSpace
        ):
            return None
        names.append(item.name)
    return tuple(names)


def _concatenation(names: tuple[str, ...]) -> Callable[[dict], torch.Tensor]:
    """A let function laying bound per-row vectors end to end.

    Parameters
    ----------
    names : tuple[str, ...]
        The bound names, in order.

    Returns
    -------
    Callable[[dict], torch.Tensor]
        The function, reading exactly ``names``, returning their
        concatenation along the feature axis.
    """

    def concatenate(env: dict) -> torch.Tensor:
        """Concatenate the bundled values per row.

        Parameters
        ----------
        env : dict
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(batch, sum of widths)``.
        """
        return torch.cat([env[name] for name in names], dim=-1)

    return reading(concatenate, names)


def _input_name(decl: ProgramDecl) -> str:
    """The name a program without named parameters reads its input under.

    Parameters
    ----------
    decl : ProgramDecl
        The program.

    Returns
    -------
    str
        Its domain object's name in lowercase, which the expansion of a
        scan over the input names the sequence by; empty for a product
        domain.
    """
    domain = decl.domain
    return domain.name.lower() if isinstance(domain, TypeName) else ""


def _fibration_index(env: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Resolve a grouped-marginalize ``via`` fibration index from the
    runtime environment.

    The index is a per-row map from the response plate into the
    grouping plate, supplied as host data through the observations
    dict (like any free covariate). A name that resolves to nothing
    is a user error, so it fails with a clear message rather than a
    bare ``KeyError``.
    """
    idx = env.get(name)
    if idx is None:
        raise CompileError(
            f"grouped marginalize: the ``via`` fibration index {name!r} "
            f"is neither a bound program variable nor supplied at runtime; "
            f"pass it in the observations dict as a long tensor of "
            f"row-to-group indices",
            0,
            0,
        )
    return idx


def _project_nested_aggregates(
    aggregates: torch.Tensor,
    inner_sizes: tuple[int, ...],
    kept_axes: tuple[int, ...],
    slot: str,
) -> torch.Tensor:
    """Project an inner grouped block's per-position aggregates onto
    the enclosing block's grouping plate.

    The inner plate is a product whose factors include every factor
    of the enclosing plate; the aggregates are summed along the
    factors the enclosing plate lacks and laid out in the enclosing
    plate's factor order, flat in row-major order.

    Parameters
    ----------
    aggregates : torch.Tensor
        The inner block's aggregates, flat over its product plate.
    inner_sizes : tuple[int, ...]
        The cardinality of each factor of the inner plate.
    kept_axes : tuple[int, ...]
        The position in ``inner_sizes`` of each factor of the
        enclosing plate, in the enclosing plate's order.
    slot : str
        The environment slot the aggregates were read from, named in
        the error.

    Returns
    -------
    torch.Tensor
        The aggregates over the enclosing plate, flat.

    Raises
    ------
    ValueError
        If the aggregates do not have one entry per inner position.
    """
    total = 1
    for size in inner_sizes:
        total *= size
    if tuple(aggregates.shape) != (total,):
        raise ValueError(
            f"grouped marginalize: the nested block at slot {slot!r} must "
            f"contribute one aggregate per position of its plate "
            f"{inner_sizes}; got shape {tuple(aggregates.shape)}"
        )
    shaped = aggregates.reshape(inner_sizes)
    summed_axes = tuple(
        axis for axis in range(len(inner_sizes)) if axis not in kept_axes
    )
    if summed_axes:
        shaped = shaped.sum(dim=summed_axes)
        remaining = [axis for axis in range(len(inner_sizes)) if axis in kept_axes]
        order = tuple(remaining.index(axis) for axis in kept_axes)
    else:
        order = kept_axes
    return shaped.permute(order).reshape(-1)


# Value carried by a let-binding at compile time.  The let
# sublanguage is a small typed lambda calculus over heterogeneous
# values: numeric literals lift to torch tensors; identifier
# references resolve to whatever is bound (morphisms, programs,
# chart views, structured tuples); composite expressions return
# tensors of derived shape; lambdas wrap functions.  The union
# below enumerates the kinds the program-theory check admits at a
# let position.
type LetValue = (
    torch.Tensor
    | int
    | float
    | bool
    | str
    | tuple["LetValue", ...]
    | list["LetValue"]
    | Morphism
    | ContinuousMorphism
    | Callable[[dict[str, "LetValue"]], "LetValue"]
)


def _collection_value(items: list[LetValue]) -> torch.Tensor | list[LetValue]:
    """The eager value of a list literal or a ``map``.

    The checked surface types a collection of numbers as a ``Tensor``
    whose leading axis runs over the items, and indexes, reduces, and
    folds it as one. A collection whose items are all tensors is thus
    stacked along a new leading axis, after broadcasting the items to
    one shape. A collection of other host values, such as the tokens a
    ``parse`` call reads, has no tensor form and stays a Python list.

    Parameters
    ----------
    items : list[LetValue]
        The evaluated items, in order.

    Returns
    -------
    torch.Tensor | list[LetValue]
        The stacked tensor, or the items themselves.
    """
    if items and all(isinstance(item, torch.Tensor) for item in items):
        tensors = cast(list[torch.Tensor], items)
        return torch.stack(torch.broadcast_tensors(*tensors))
    return items


# ---------------------------------------------------------------------------
# Built-in tensor primitives visible to let-expression bodies.
# ---------------------------------------------------------------------------
#
# Each entry maps a function name (as users write it in source) to a
# callable that the let-expression compiler dispatches through.
# Categorically these are the deterministic generators of the
# ``Smooth(R^n, R^m)`` strata embedded into our V-Cat surface as
# pointwise / shape-preserving operations on tensors.  User-defined
# neural-network programs compose these with ``latent`` morphisms
# (the learnable weights) inside the existing ``program`` declaration
# — no separate ``module`` construct is needed.
#
# Coverage: every standard ``torch.nn.functional`` activation and a
# broad set of element-wise + reduction primitives from
# ``torch``.  When the underlying torch function takes a ``dim``
# keyword we default to ``dim=-1`` (the natural choice for per-row
# operations in V-Cat morphisms); users wanting a different axis
# write the contraction via the typed ``contraction`` declaration.
_LET_EXPR_BUILTINS: Mapping[str, Callable] = EAGER_BUILTINS

#: The calls the let evaluator implements itself rather than through
#: the builtin table: the collection combinators, which read a lambda,
#: and the deduction and term operations.
_LET_SPECIAL_CALLS: frozenset[str] = frozenset(
    {
        "length",
        "map",
        "filter",
        "fold",
        "logsumexp_over",
        "parse",
        "subst",
        "compose",
        "cholesky_quad_form",
        LOG_PROB_CALL,
    }
)


def _family_log_prob(arguments: list[LetValue]) -> torch.Tensor:
    """Evaluate ``$log_prob("Family", value, "parameter", argument, ...)``.

    A called computation's ``score.add(log_prob(D, v))`` lowers to this
    internal let call: the family's log density at the value, entrywise,
    under the parameters given by name, each clamped as an inline draw
    from the family clamps it.

    Parameters
    ----------
    arguments : list[LetValue]
        The family's name, the value, then each parameter's name and
        value.

    Returns
    -------
    torch.Tensor
        The log density, broadcast over the value and the parameters.

    Raises
    ------
    CompileError
        If the family is not registered or a parameter is not one of
        its parameters.
    """
    family, value, *pairs = arguments
    spec = FAMILY_REGISTRY.get(str(family))
    if spec is None:
        raise CompileError(f"no registered family {family!r} to score under")
    parameters: dict[str, torch.Tensor] = {}
    for position in range(0, len(pairs), 2):
        name = str(pairs[position])
        tensor = torch.as_tensor(pairs[position + 1], dtype=torch.get_default_dtype())
        try:
            parameters[name] = clamp_param(spec, name, tensor)
        except KeyError as error:
            raise CompileError(
                f"family {spec.name!r} has no parameter {name!r} to score under"
            ) from error
    observed = torch.as_tensor(value, dtype=torch.get_default_dtype())
    return build_torch_distribution(spec, parameters).log_prob(observed)


def _checked_layout(
    function: Callable[[dict[str, LetValue]], LetValue],
    reads: frozenset[str],
    extents: Mapping[str, tuple[int, ...]],
    result: str,
) -> Callable[[dict[str, LetValue]], LetValue]:
    """Evaluate a let body on its operands' checked shapes.

    Within one run, the runtime holds a number drawn from a continuous
    family, or read from a one-coordinate input, as a vector of one
    entry. Elementwise arithmetic cannot tell the difference, but a
    body that lays numbers out along a new axis (a list literal or a
    ``factor``) or reduces one would read that stray axis as data:
    ``[h, 1.0]`` would be ``(2, 1)`` rather than the checked ``(2,)``.
    Each operand the checked program types as a number is thus handed
    to the body as a number, and a result the checked program types as
    a number is given back the one-entry layout its operands had.

    Parameters
    ----------
    function : Callable[[dict[str, LetValue]], LetValue]
        The compiled body.
    reads : frozenset[str]
        The names the body reads.
    extents : Mapping[str, tuple[int, ...]]
        The checked extents of the program's bindings, ``()`` for a
        number.
    result : str
        The name the body's value is bound to.

    Returns
    -------
    Callable[[dict[str, LetValue]], LetValue]
        The body, reading and returning values at their checked layout.
    """
    numbers = frozenset(name for name in reads if extents.get(name) == ())
    if not numbers:
        return function
    returns_number = extents.get(result) == ()

    def _run(env: dict[str, LetValue]) -> LetValue:
        view = dict(env)
        unwrapped = False
        for name in numbers:
            value = env.get(name)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == (1,):
                view[name] = value.reshape(())
                unwrapped = True
        value = function(view)
        if (
            unwrapped
            and returns_number
            and isinstance(value, torch.Tensor)
            and value.dim() == 0
        ):
            return value.reshape(1)
        return value

    return _run


def _unknown_let_call(
    node: LetExprNode,
    globals_: Mapping[str, LetValue | frozenset[str]],
    bound: Collection[str],
) -> LetExprCall | None:
    """The first call in a let expression to a name the evaluator lacks.

    The let evaluator resolves a call against its builtin table, its own
    combinators, the static globals it is compiled with (compiled
    deductions and injected callables), and the declared constructors.
    A call naming none of them fails whenever it runs, so it is reported
    when the program compiles instead.

    Parameters
    ----------
    node : LetExprNode
        The let's value.
    globals_ : Mapping[str, LetValue | frozenset[str]]
        The static globals the expression is compiled with.
    bound : Collection[str]
        The names the program binds before the let, which a call may
        name when one of them holds a function.

    Returns
    -------
    LetExprCall | None
        The first such call in preorder, or ``None``.
    """
    constructors = globals_.get(LET_CONSTRUCTORS_KEY, frozenset())
    assert isinstance(constructors, frozenset)
    for candidate in walk_let_expr(node):
        if not isinstance(candidate, LetExprCall):
            continue
        name = candidate.func
        if (
            name in _LET_EXPR_BUILTINS
            or name in _LET_SPECIAL_CALLS
            or (name in globals_ and name not in LET_INTERNAL_KEYS)
            or name in constructors
            or name in bound
        ):
            continue
        return candidate
    return None


def _drawn_weights(
    function: LearnedFunction, env: Mapping[str, "LetValue"], name: str
) -> dict[str, torch.Tensor] | None:
    """The drawn values of a latent network's weights, read from a run.

    Parameters
    ----------
    function : LearnedFunction
        The network.
    env : Mapping[str, LetValue]
        The run's bindings.
    name : str
        The name the network is called by, for the error.

    Returns
    -------
    dict[str, torch.Tensor] | None
        Each weight site's value; ``None`` for point weights.

    Raises
    ------
    CompileError
        If a weight site is not bound in the run.
    """
    sites = function.latent_sites
    if not sites:
        return None
    drawn: dict[str, torch.Tensor] = {}
    for site in sites.values():
        value = env.get(site)
        if not isinstance(value, torch.Tensor):
            raise CompileError(
                f"the latent network {name!r} reads its weight {site!r}, which "
                f"the run does not bind; a program without value parameters "
                f"that uses the network draws it"
            )
        drawn[site] = value
    return drawn


def _let_callees(steps: Sequence[ProgramStep]) -> tuple[str, ...]:
    """The callees a body's lets and scores call, in first-use order.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        The body, a marginalization's scope included.

    Returns
    -------
    tuple[str, ...]
        Each called name once.
    """
    found: dict[str, None] = {}
    for step in steps:
        if isinstance(step, LetStep | ScoreStep):
            for callee in free_let_callees(step.value):
                found.setdefault(callee, None)
        elif isinstance(step, MarginalizeStep):
            for callee in _let_callees(step.scope):
                found.setdefault(callee, None)
    return tuple(found)


def _expected_call_arity(target: object) -> int | None:
    """Return the expected number of positional arguments for ``target``
    when invoked from a let-expression call site, or ``None`` when the
    arity cannot be determined statically.

    Resolution rules:

    * `MonadicProgram` with named ``params``: arity is
      ``len(params)`` (each param becomes one positional argument).
    * `MonadicProgram` without ``params``: arity is ``1`` (the
      packed program input).
    * `Morphism`: arity is ``1`` (the domain tensor).
    * `LearnedFunction`: one argument per factor of its domain.
    * Any other callable: use `inspect.signature` and count
      positional parameters without defaults; ``*args`` makes arity
      unknowable, return ``None``.
    """
    if isinstance(target, MonadicProgram):
        params = getattr(target, "_params", None)
        return len(params) if params else 1
    if isinstance(target, Morphism):
        return 1
    if isinstance(target, LearnedFunction):
        return len(target.factor_widths)
    try:
        sig = inspect.signature(target)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    count = 0
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if p.kind in positional_kinds and p.default is inspect.Parameter.empty:
            count += 1
    return count


def _flatten_type_axes(expr: ObjectExpr) -> tuple[str, ...]:
    """Flatten a `ObjectExpr` into the ordered sequence of axis
    type names it denotes.

    ``A`` is one axis ``A``.  ``(A * B)`` is two axes ``(A, B)``;
    nested products flatten left-to-right.  Coproducts and other
    non-product TypeExprs are not supported here; the inferred-
    wiring path raises a `ValueError` and points the user at
    the explicit ``wiring`` clause.
    """
    if isinstance(expr, TypeName):
        return (expr.name,)
    if isinstance(expr, ObjectProduct):
        out: list[str] = []
        for component in expr.components:
            out.extend(_flatten_type_axes(component))
        return tuple(out)
    raise ValueError(
        f"contraction signature contains a non-product / non-named axis "
        f"type {type(expr).__name__}; the type-driven wiring inference "
        "only handles named axes and products of named axes. Pass an "
        'explicit ``wiring "..."`` clause.'
    )


def _infer_wiring_from_signature(
    *,
    inputs: tuple[ContractionInput, ...],
    output_domain: ObjectExpr,
    output_codomain: ObjectExpr,
    shared_axes: tuple[str, ...],
) -> str:
    """Build an einsum-style wiring spec from the contraction's
    typed signature.

    Each input's axis sequence is the flattening of its
    ``input_domain -> input_codomain``.  The output's axis sequence
    is the flattening of ``output_domain -> output_codomain``.

    The inference rule:

    * Every axis name that appears in the output sequence is a
      *kept* axis (propagates from inputs to output).
    * Every axis name that appears in two or more input sequences
      but not in the output is a *contracted* axis (joined via the
      rule).
    * Every axis name that appears in exactly one input sequence
      and not in the output is *anomalous*: it would need to be
      summed out by the rule but no other input shares it.  We
      report that as an inference failure and direct the user to
      the explicit ``wiring`` clause.
    * Axes named in ``shared_axes`` are forced to propagate (kept)
      even if the default rule would have contracted them; this is
      the disambiguator for element-wise contractions where the
      same axis appears in two inputs and the output.

    The output spec is the standard numpy / torch einsum form:
    one letter per *distinct* axis name, joined with commas across
    inputs, then ``->``, then the kept axes' letters in the
    output's declared order.
    """
    input_axes = [
        _flatten_type_axes(inp.input_domain) + _flatten_type_axes(inp.input_codomain)
        for inp in inputs
    ]
    output_seq = _flatten_type_axes(output_domain) + _flatten_type_axes(output_codomain)
    shared = set(shared_axes)
    # An axis is kept iff it appears in the output OR was named in
    # the ``share`` clause.  Anything else that appears in ≥ 2 inputs
    # is contracted.
    kept: set[str] = set(output_seq) | shared
    appearances: dict[str, int] = {}
    for axes in input_axes:
        seen_in_this = set()
        for axis in axes:
            if axis in seen_in_this:
                continue
            seen_in_this.add(axis)
            appearances[axis] = appearances.get(axis, 0) + 1
    # An anomalous axis appears in exactly one input and is not in
    # the output / share list: there is nothing to contract it
    # against and nothing to project it to.
    anomalous = [axis for axis, n in appearances.items() if n == 1 and axis not in kept]
    if anomalous:
        raise ValueError(
            f"axes {sorted(anomalous)} appear in exactly one input and not in the "
            "output; the type-driven rule cannot decide what to do with them. "
            "Add them to the contraction's output type, list them in a "
            "``share`` clause, or pass an explicit ``wiring`` spec"
        )
    # Assign single letters to axes.  Standard einsum lower-case
    # alphabet; we keep insertion order so the generated string is
    # deterministic given the signature.
    letters_pool = "abcdefghijklmnopqrstuvwxyz"
    axis_to_letter: dict[str, str] = {}
    for axes in input_axes:
        for axis in axes:
            if axis in axis_to_letter:
                continue
            if len(axis_to_letter) >= len(letters_pool):
                raise ValueError(
                    "contraction has more distinct axis names than einsum "
                    "letters available; pass an explicit ``wiring`` spec"
                )
            axis_to_letter[axis] = letters_pool[len(axis_to_letter)]
    # Output axes preserve the declared order on the output type
    # (which is the order the user wrote them in source).
    output_letters = "".join(axis_to_letter[axis] for axis in output_seq)
    input_letter_groups = [
        "".join(axis_to_letter[axis] for axis in axes) for axes in input_axes
    ]
    return f"{', '.join(input_letter_groups)} -> {output_letters}"


# Closed option-key sets for the program surface. Each set lives next
# to the code that consumes it; `check_option_keys` rejects
# anything outside the set with a did-you-mean diagnostic at the
# offending entry's own line/col.
#
# * ``sample`` steps read the axis-role clause (``over`` / ``iid_over``),
#   which the parser lifts into the step's `AxisSpec`.
# * ``observe`` steps additionally read ``via`` (the per-observe
#   fibration inside grouped marginalize blocks).
# * ``marginalize`` steps read the grouping plate (``over``) and the
#   pushforward ``reduction``.
# * ``program`` declarations read the ``effects`` capability row and
#   the posterior-block ``over`` model reference.
# * ``contraction`` declarations read the composition ``rule``, the
#   explicit einsum ``wiring`` escape hatch, and the ``share`` list.
_MULTIVARIATE_REAL_FAMILIES: frozenset[str] = frozenset(
    {"MultivariateNormal", "LowRankMVN"}
)
"""Families whose sample is one real vector over the event axis."""

_SAMPLE_STEP_OPTION_KEYS: frozenset[str] = frozenset({"over", "iid_over"})
_OBSERVE_STEP_OPTION_KEYS: frozenset[str] = frozenset({"via", "over", "iid_over"})
_MARGINALIZE_STEP_OPTION_KEYS: frozenset[str] = frozenset({"over", "reduction"})
_PROGRAM_OPTION_KEYS: frozenset[str] = frozenset({"effects", "over"})
_CONTRACTION_OPTION_KEYS: frozenset[str] = frozenset({"rule", "wiring", "share"})


def _compile_parse_call(
    node: LetExprCall, arg_fns: list[Callable[[dict], object]]
) -> Callable[[dict], object]:
    """Compile ``parse(D, input[, Plate][, (lhs, probabilities)...])``.

    The first argument names a compiled deduction and the second is its
    input: a sentence, or with a plate object as the third argument, a
    corpus of sentences, one per row, whose goal weight is the semiring
    product of the sentences'. The remaining arguments bind production
    probabilities to a grammar normalized per left-hand side, each a pair
    of a left-hand side's label and a probability vector over its
    productions; the deduction checks that they cover its groups.

    Parameters
    ----------
    node : LetExprCall
        The call.
    arg_fns : list[Callable[[dict], object]]
        Its compiled arguments.

    Returns
    -------
    Callable[[dict], object]
        The evaluator, answering a `ChartView` for a sentence and a
        `CorpusChart` for a corpus.

    Raises
    ------
    CompileError
        If the call has fewer than two arguments, a binding is not a
        pair of a string label and a value, a label is bound twice, or
        (when evaluated) the first argument is no deduction.
    """
    if len(node.args) < 2:
        raise CompileError(
            "parse() takes a deduction and its input, then optionally a plate "
            "object and (left-hand side, probabilities) pairs"
        )
    rest = list(node.args[2:])
    corpus = bool(rest) and isinstance(rest[0], LetExprVar)
    if corpus:
        rest = rest[1:]
    labels: list[str] = []
    for argument in rest:
        if not (
            isinstance(argument, LetExprTuple)
            and len(argument.items) == 2
            and isinstance(argument.items[0], LetExprString)
        ):
            raise CompileError(
                'parse() binds production probabilities as ("<left-hand side>", '
                "<probabilities>) pairs after its input and optional plate",
                argument.line,
                argument.col,
            )
        label = argument.items[0].value
        if label in labels:
            raise CompileError(
                f"parse() binds the left-hand side {label!r} twice",
                argument.line,
                argument.col,
            )
        labels.append(label)
    pair_fns = arg_fns[len(node.args) - len(rest) :]
    target_fn, input_fn = arg_fns[0], arg_fns[1]

    def _parse(env: dict) -> object:
        system = target_fn(env)
        if not isinstance(system, DeductionSystem):
            raise CompileError(
                f"parse() first arg must be a DeductionSystem, "
                f"got {type(system).__name__}"
            )
        probabilities: dict[str, torch.Tensor] | None = None
        if labels:
            probabilities = {}
            for label, pair_fn in zip(labels, pair_fns, strict=True):
                pair = pair_fn(env)
                assert isinstance(pair, tuple) and len(pair) == 2
                value = pair[1]
                tensor = (
                    value if isinstance(value, torch.Tensor) else torch.tensor(value)
                )
                probabilities[label] = tensor.reshape(tensor.shape[-1:])
        source = input_fn(env)
        if corpus:
            assert isinstance(source, Sequence)
            return system.corpus(list(source), probabilities)
        return system(source, probabilities)

    return _parse


class _ProgramsMixin:
    """Mixin: program / contraction / define compilation methods.

    The compiler base supplies every environment slot below; the
    annotations let the type checker verify each access from a
    mixin method.
    """

    _algebra: CompositionRule
    _morphisms: dict
    _objects: dict[str, SetObject]
    _spaces: dict[str, ContinuousSpace]
    _output_expr: Expr | None
    _groups: dict[str, list[str]]
    _program_templates: dict[str, ProgramDecl]
    _call_refusals: dict[str, CompileError]
    _parameter_space_stack: list[dict[str, AnySpace]]
    _value_extents_stack: list[dict[str, tuple[int, ...]]]
    _contractions: dict
    _transformations: dict
    _trans_singletons: dict
    _trans_constructors: dict
    _module: Module
    _latent_networks: dict[str, LearnedFunction]
    _scan_step_declarations: dict[str, ProgramDecl]
    _scan_step_programs: dict[str, MonadicProgram]
    _scan_initial_states: dict[str, torch.nn.Parameter]

    # ``_resolve_type``, ``_resolve_any_space``, ``_resolve_index_size``
    # come from `_ResolutionMixin` via the ``Compiler`` MRO.

    @property
    @abstractmethod
    def qiec_module(self) -> QiecModule | None:
        """The checked QIEC module, which the compiler lowers on first access.

        Returns
        -------
        QiecModule | None
            The module, or ``None`` when the source has no QIEC surface.
        """

    @property
    def _ungrouped_marg_bodies(
        self,
    ) -> dict[
        int,
        tuple[
            str,
            str,
            tuple[DrawArgName | DrawArgIndex | str, ...],
            tuple[ProgramStep, ...],
        ],
    ]:
        """Per-marker record of an ungrouped ``marginalize`` block's body.

        Keyed by ``id`` of the placeholder `GroupedMarginalizeStep`
        emitted for an ungrouped block, each entry carries the
        ``(latent_name, prior_family, prior_args, body_steps)`` the
        compile pass needs to build the block's integrated score
        closure. Populated lazily so the base compiler's ``__init__``
        stays agnostic to this mixin's state.
        """
        cache = self.__dict__.get("_ungrouped_marg_cache")
        if cache is None:
            cache = {}
            self.__dict__["_ungrouped_marg_cache"] = cache
        return cache

    def _surface_to_bind(self, step: ProgramStep) -> ProgramStep:
        """Normalize surface `SampleStep` / `ObserveStep`
        / `MarginalizeStep` nodes into `BindStep` IR.

        The surface grammar emits one node per step keyword for clarity;
        the compiler is structured around the unified `BindStep`
        with a ``mode`` discriminator. This translator is the seam: it
        is the only place the surface step kinds appear inside the
        compiler.

        Before normalising, sugar families are desugared to their
        canonical operator-algebra form (`TruncatedNormal` ->
        `Restrict(Normal, low, high)`, etc.) via
        [`desugar_step`][quivers.dsl.compiler.sugar.desugar_step], so
        downstream IR walks the single operator vocabulary.
        """
        if isinstance(step, CallStep):
            # A call reaching the step compiler is a scan's step program
            # calling its cell, which compiles to a call run in place.
            return step
        if isinstance(step, (SampleStep, ObserveStep)):
            step = desugar_step(step)
        if isinstance(step, SampleStep):
            check_option_keys(
                step.options,
                _SAMPLE_STEP_OPTION_KEYS,
                owner="sample step",
                line=step.line,
                col=step.col,
            )
            return BindStep(
                vars=step.vars,
                morphism=step.morphism,
                args=step.args,
                index=step.index,
                mode="sample",
                axes=step.axes,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, ObserveStep):
            check_option_keys(
                step.options,
                _OBSERVE_STEP_OPTION_KEYS,
                owner="observe step",
                line=step.line,
                col=step.col,
            )
            if len(step.vars) != 1:
                # The runtime observe path (response clamping via the
                # observations dict, VectorisedObserve, grouped
                # marginalize capture) is single-response throughout;
                # a tuple pattern on observe has no runtime meaning.
                raise CompileError(
                    f"observe takes a single variable; got ({', '.join(step.vars)})",
                    step.line,
                    step.col,
                )
            # An observe's axis-role clause arrives as options; read it
            # as the axes a sample's clause gives.
            over = get_option_name_list(
                step.options, "over", line=step.line, col=step.col
            )
            iid_over = get_option_name_list(
                step.options, "iid_over", line=step.line, col=step.col
            )
            axes = step.axes
            if axes is None and (over or iid_over):
                axes = AxisSpec(
                    over=over, iid_over=iid_over, line=step.line, col=step.col
                )
            return BindStep(
                vars=step.vars,
                morphism=step.morphism,
                args=step.args,
                index=step.index,
                mode="score",
                axes=axes,
                via=step.via,
                via_axes=step.via_axes,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, MarginalizeStep):
            check_option_keys(
                step.options,
                _MARGINALIZE_STEP_OPTION_KEYS,
                owner="marginalize step",
                line=step.line,
                col=step.col,
            )
            return BindStep(
                vars=(step.var,),
                morphism=step.morphism,
                args=step.args,
                index=step.index,
                mode="marginal",
                scope=tuple(self._surface_to_bind(s) for s in step.scope),
                over=step.over,
                over_objs=step.over_objs,
                reduction=step.reduction,
                line=step.line,
                col=step.col,
            )
        return step

    def _expand_bind_steps(
        self, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Translate the surface `BindStep` IR into the
        compiler's internal step-IR (`DrawStep`,
        `PlateDrawStep`, `VectorisedObserveStep`,
        `GroupedMarginalizeStep`).

        The expansion is purely a syntactic refinement: each
        BindStep dispatches on its ``mode`` and ``index`` fields
        to one of the four internal step shapes. Marginalize binds
        additionally inline a synthesized sample step for the
        coordinate, followed by the scope's recursively-expanded
        steps, followed by a `GroupedMarginalizeStep` reduction.

        ``LetStep`` passes through unchanged. The expansion
        preserves the Kleisli-arrow denotation of the program body
        — it is a reorganisation of the surface IR, not a change
        of semantics.
        """
        out: list[ProgramStep] = []
        for raw in steps:
            step = self._surface_to_bind(raw)
            if isinstance(step, LetStep):
                out.append(step)
                continue
            if not isinstance(step, BindStep):
                # Pass-through for any internal-IR step that has
                # already been expanded (e.g., template-inlined
                # bodies that synthesized internal steps directly).
                out.append(step)
                continue
            if step.mode == "sample":
                if step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=False,
                            axes=step.axes,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    if len(step.vars) != 1:
                        raise CompileError(
                            "indexed sample bind must bind a single name",
                            step.line,
                            step.col,
                        )
                    # The per-row codomain for a plate-draw is taken
                    # from the family's natural codomain at compile
                    # time; the IR carries a `codomain` field that
                    # the compiler's PlateDrawStep handler resolves
                    # via the family's domain/codomain dimensions.
                    # Under the unified surface, the index annotation
                    # `: A` declares the index set; the per-row codomain
                    # is implicit (taken from the family). We supply a
                    # placeholder `TypeName("1")` which the family
                    # resolver interprets as "scalar per-row codomain"
                    # (Euclidean(1)); families that declare richer
                    # codomains override this.
                    out.append(
                        PlateDrawStep(
                            name=step.vars[0],
                            index=step.index,
                            codomain=TypeName(name="1", line=step.line, col=step.col),
                            morphism=step.morphism,
                            args=step.args,
                            axes=step.axes,
                            line=step.line,
                            col=step.col,
                        )
                    )
            elif step.mode == "score":
                if step.index is None:
                    out.append(
                        DrawStep(
                            vars=step.vars,
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=True,
                            line=step.line,
                            col=step.col,
                        )
                    )
                else:
                    if len(step.vars) != 1:
                        raise CompileError(
                            "indexed observe bind must bind a single name",
                            step.line,
                            step.col,
                        )
                    out.append(
                        VectorisedObserveStep(
                            index_var=step.vars[0],
                            index_set=step.index,
                            morphism=step.morphism,
                            args=step.args,
                            response_var=step.vars[0],
                            fibration_var=step.via,
                            fibration_axes=step.via_axes,
                            axes=step.axes,
                            line=step.line,
                            col=step.col,
                        )
                    )
            elif step.mode == "marginal":
                if len(step.vars) != 1:
                    raise CompileError(
                        "marginalize bind must bind a single name",
                        step.line,
                        step.col,
                    )
                # Grouped form: `over G` (single plate) or
                # `over G * H` (product grouping plate). The
                # fibration data lives on each observe inside the
                # body via that observe's `via <idx>` (or
                # `via product(...)`) clause; the header carries
                # only the grouping plate.
                has_over = step.over is not None or step.over_objs is not None
                has_grouping = has_over
                # Normalize the grouping plate into tuple form so
                # the downstream code handles single + product
                # uniformly.
                over_names: tuple[str, ...] | None = None
                if step.over_objs is not None:
                    over_names = step.over_objs
                elif step.over is not None:
                    over_names = (step.over,)
                # Resolve the per-class size from the latent's index
                # annotation when present. The grouped form requires
                # an explicit `: K` annotation so the prior's class
                # axis can be sized at compile time.
                class_size = 0
                if step.index is not None:
                    if isinstance(step.index, TypeName):
                        nm = step.index.name
                        if nm.isdigit():
                            class_size = int(nm)
                        elif nm in self._objects:
                            class_size = int(self._objects[nm].size)
                if has_over and class_size == 0:
                    raise CompileError(
                        "grouped marginalize requires an explicit class-set "
                        "annotation (e.g. `marginalize c : K <- ...`) so the "
                        "class axis is sized at compile time",
                        step.line,
                        step.col,
                    )
                if over_names is not None:
                    for nm in over_names:
                        if nm not in self._objects:
                            raise CompileError(
                                f"grouped marginalize: `over` object "
                                f"{nm!r} is not a declared object",
                                step.line,
                                step.col,
                            )
                # Reduction defaults to logsumexp when unspecified.
                reduction = step.reduction or "logsumexp"
                if reduction not in ("logsumexp", "sum", "mean"):
                    raise CompileError(
                        f"grouped marginalize: unknown reduction "
                        f"{reduction!r}; expected one of logsumexp, "
                        f"sum, mean",
                        step.line,
                        step.col,
                    )
                # Extract the categorical prior's probability or logits tensor.
                probs_var: str | None = None
                probs_indices: tuple[str, ...] = ()
                parameterization: Literal["probs", "logits"] = "probs"
                if has_over:
                    if not step.args:
                        raise CompileError(
                            "grouped marginalize requires the latent's "
                            "categorical family to carry an explicit argument "
                            "(e.g. `Categorical(probs=weights)`)",
                            step.line,
                            step.col,
                        )
                    first = step.args[0]
                    if not isinstance(first, DrawArgNamed) or (
                        len(step.args) != 1
                        or first.parameter not in ("probs", "logits")
                    ):
                        raise CompileError(
                            "Categorical requires exactly one explicit "
                            "parameterization; use probs=... or logits=...",
                            step.line,
                            step.col,
                        )
                    parameterization = first.parameter
                    first = first.value
                    # `first` is a `DrawArg` tagged variant on the
                    # widened AST. A `DrawArgName` carries the
                    # identifier text and a `DrawArgIndex` a probs
                    # tensor gathered by enclosing latents; other
                    # variants (literal, nested distribution call,
                    # list literal) are not admissible as the probs
                    # argument.
                    if isinstance(first, DrawArgName):
                        probs_var = first.text
                    elif isinstance(first, DrawArgIndex):
                        probs_var = first.name
                        probs_indices = tuple(first.indices)
                    elif isinstance(first, str):
                        probs_var = first
                    else:
                        raise CompileError(
                            "grouped marginalize: the categorical family's "
                            "first argument must be a named probs tensor, "
                            f"indexed or not (got literal {first!r})",
                            step.line,
                            step.col,
                        )
                # Introduce the coordinate. A *grouped* marginalize
                # block seeds the latent with the implicit
                # class-index vector ``torch.arange(K)`` so any
                # downstream let-step that references the latent
                # broadcasts across the class axis; the terminal
                # observe in the body (rewritten below) overwrites
                # that slot with the per-(N, K) log-likelihood the
                # marginalize step consumes. An *ungrouped* block
                # integrates the latent out analytically: the body
                # is re-scored at each value in the prior's support
                # and the per-class log-likelihoods are combined with
                # the log-prior under the reduction, so the block
                # emits a single integrated score step rather than a
                # live latent draw and a separately-scored observe.
                latent_name = step.vars[0]
                if not has_grouping:
                    ungrouped_scope = self._expand_bind_steps(
                        step.scope if step.scope is not None else ()
                    )
                    # Partition the body: deterministic lets and the
                    # terminal observe drive the marginal score (they
                    # are re-evaluated at each value of the latent's
                    # support), while any inner latent samples that do
                    # not participate in the marginal are emitted into
                    # the main IR and scored as ordinary latents.
                    marg_body: list[ProgramStep] = []
                    for scoped in ungrouped_scope:
                        if (
                            isinstance(scoped, LetStep)
                            or (isinstance(scoped, DrawStep) and scoped.is_observed)
                            or isinstance(scoped, VectorisedObserveStep)
                        ):
                            marg_body.append(scoped)
                        else:
                            out.append(scoped)
                    marker = GroupedMarginalizeStep(
                        var_name=latent_name,
                        class_size=0,
                        probs_var=None,
                        parameterization=parameterization,
                        over_obj=None,
                        over_objs=None,
                        body_ll_var=latent_name,
                        body_observes=None,
                        reduction=step.reduction,
                        line=step.line,
                        col=step.col,
                    )
                    self._ungrouped_marg_bodies[id(marker)] = (
                        latent_name,
                        step.morphism,
                        tuple(step.args or ()),
                        tuple(marg_body),
                    )
                    out.append(marker)
                    continue
                out.append(
                    GroupedLatentInitStep(
                        latent_name=latent_name,
                        class_size=class_size,
                        line=step.line,
                        col=step.col,
                    )
                )
                # Scope's steps. For a grouped block, rewrite every
                # VectorisedObserveStep in the expanded body as a
                # GroupedBodyObserveStep that writes its per-row
                # per-class log-likelihood to a dedicated env slot.
                # The surrounding GroupedMarginalizeStep's runtime callable
                # collects each slot, pairs it with that observe's
                # `via <idx>` fibration, and scatter-sums each
                # contribution into the shared `(|G|, K)`
                # accumulator before the reduction.
                scope_steps = step.scope if step.scope is not None else ()
                expanded_scope = list(self._expand_bind_steps(scope_steps))
                body_observes: list[GroupedObserveEntry] = []
                if has_grouping:
                    # Walk the body forward; rewrite each observe
                    # into a GroupedBodyObserveStep with a unique
                    # ll_slot and capture its per-observe fibration.
                    for j, scoped_step in enumerate(expanded_scope):
                        if isinstance(scoped_step, VectorisedObserveStep):
                            if (
                                scoped_step.fibration_var is None
                                and scoped_step.fibration_axes is None
                            ):
                                raise CompileError(
                                    "grouped marginalize: every observe "
                                    "inside the body must carry its own "
                                    "`via <idx>` clause "
                                    "(e.g. `observe r : N via idx <- ...`); "
                                    f"observe {scoped_step.response_var!r} "
                                    "has no `via`",
                                    scoped_step.line,
                                    scoped_step.col,
                                )
                            ll_slot = f"_grouped_ll_{latent_name}_{len(body_observes)}"
                            expanded_scope[j] = GroupedBodyObserveStep(
                                response_var=scoped_step.response_var,
                                morphism=scoped_step.morphism,
                                args=scoped_step.args,
                                index_set=scoped_step.index_set,
                                index_var=scoped_step.index_var,
                                latent_name=latent_name,
                                class_size=class_size,
                                fibration_var=scoped_step.fibration_var,
                                fibration_axes=scoped_step.fibration_axes,
                                ll_slot=ll_slot,
                                line=scoped_step.line,
                                col=scoped_step.col,
                            )
                            body_observes.append(
                                GroupedObserveEntry(
                                    ll_slot=ll_slot,
                                    fibration_var=scoped_step.fibration_var,
                                    fibration_axes=scoped_step.fibration_axes,
                                )
                            )
                    if not body_observes:
                        # Nested case: a grouped marginalize block whose
                        # body's only contribution to the per-group
                        # accumulator is an inner grouped block.  The
                        # inner block's GroupedMarginalizeStep produces the
                        # (N_outer, K_outer) tensor the outer block
                        # consumes.  Re-point the inner's
                        # ``body_ll_var`` at the outer latent so the
                        # outer codegen finds the tensor at the
                        # expected slot, and record a single
                        # body_observes entry whose ll_slot is the
                        # outer latent's name (no per-observe
                        # fibration, since the inner block already
                        # performed its own scatter-add).
                        nested_marg_idx: int | None = None
                        for j in range(len(expanded_scope) - 1, -1, -1):
                            if isinstance(expanded_scope[j], GroupedMarginalizeStep):
                                nested_marg_idx = j
                                break
                        if nested_marg_idx is None:
                            raise CompileError(
                                "grouped marginalize: the body must "
                                "contain at least one `observe` step "
                                "(or an inner grouped marginalize) "
                                "whose per-row log-likelihood produces "
                                "the per-group accumulator's "
                                "contribution",
                                step.line,
                                step.col,
                            )
                        inner_marg = expanded_scope[nested_marg_idx]
                        assert isinstance(inner_marg, GroupedMarginalizeStep)
                        expanded_scope[nested_marg_idx] = GroupedMarginalizeStep(
                            var_name=inner_marg.var_name,
                            class_size=inner_marg.class_size,
                            probs_var=inner_marg.probs_var,
                            probs_indices=inner_marg.probs_indices,
                            parameterization=inner_marg.parameterization,
                            over_obj=inner_marg.over_obj,
                            over_objs=inner_marg.over_objs,
                            body_ll_var=latent_name,
                            body_observes=inner_marg.body_observes,
                            reduction=inner_marg.reduction,
                            line=inner_marg.line,
                            col=inner_marg.col,
                        )
                        inner_groups = (
                            inner_marg.over_objs
                            if inner_marg.over_objs is not None
                            else (
                                (inner_marg.over_obj,)
                                if inner_marg.over_obj is not None
                                else None
                            )
                        )
                        if inner_groups is None:
                            raise CompileError(
                                "grouped marginalize: an inner marginalize "
                                "block nested in a grouped block must carry "
                                "its own grouping plate",
                                inner_marg.line,
                                inner_marg.col,
                            )
                        body_observes.append(
                            GroupedObserveEntry(
                                ll_slot=latent_name, inner_groups=inner_groups
                            )
                        )
                out.extend(expanded_scope)
                # Pushforward reduction. When grouped, the
                # GroupedMarginalizeStep carries the list of per-observe
                # (ll_slot, fibration) entries the runtime callable
                # consumes.
                single_over = (
                    over_names[0]
                    if over_names is not None and len(over_names) == 1
                    else None
                )
                product_overs = (
                    over_names
                    if over_names is not None and len(over_names) > 1
                    else None
                )
                out.append(
                    GroupedMarginalizeStep(
                        var_name=step.vars[0],
                        class_size=class_size,
                        probs_var=probs_var,
                        probs_indices=probs_indices,
                        parameterization=parameterization,
                        over_obj=single_over,
                        over_objs=product_overs,
                        body_ll_var=step.vars[0],
                        body_observes=(tuple(body_observes) if has_grouping else None),
                        reduction=step.reduction,
                        line=step.line,
                        col=step.col,
                    )
                )
            else:
                raise CompileError(
                    f"unknown bind mode {step.mode!r}",
                    step.line,
                    step.col,
                )
        return tuple(out)

    @staticmethod
    def _infer_effects(steps: tuple[ProgramStep, ...]) -> set[str]:
        """Compute the effect set actually used by ``steps``.

        Each program step contributes:

        * `DrawStep` / `PlateDrawStep` (sample shape) -> ``Sample``
        * `DrawStep` (observed) / `VectorisedObserveStep` -> ``Score``
        * `GroupedMarginalizeStep` -> ``Marginal``
        * `LetStep` -> nothing (deterministic)
        """
        actual: set[str] = set()
        for step in steps:
            if isinstance(step, (DrawStep, PlateDrawStep)):
                if isinstance(step, DrawStep) and step.is_observed:
                    actual.add("Score")
                else:
                    actual.add("Sample")
            elif isinstance(step, VectorisedObserveStep):
                actual.add("Score")
            elif isinstance(step, GroupedMarginalizeStep):
                actual.add("Marginal")
        return actual

    def _verify_effects(
        self,
        decl: ProgramDecl,
        steps: tuple[ProgramStep, ...],
        scanned: set[str],
    ) -> None:
        """Verify the program body's effects against its declaration.

        When the declaration omits the ``effects=[...]`` option, the
        effect set is inferred from the body; no check is needed.
        When it is given, the compiler enforces that the body's
        actual effects are a subset of the declared set; a declared
        ``Pure`` rejects any of ``{Sample, Score, Marginal}``.

        Parameters
        ----------
        decl : ProgramDecl
            The program.
        steps : tuple[ProgramStep, ...]
            Its expanded steps.
        scanned : set[str]
            The effects the step programs its scans run perform.

        Raises
        ------
        CompileError
            If the body performs an effect the declaration omits.
        """
        declared = get_program_effects(
            decl.options,
            line=decl.line,
            col=decl.col,
        )
        if declared is None:
            return
        actual = self._infer_effects(steps) | scanned
        if "Pure" in declared and actual:
            raise CompileError(
                f"program {decl.name!r} is declared as Pure but body "
                f"uses effects {sorted(actual)}",
                decl.line,
                decl.col,
            )
        unaccounted = actual - declared - {"Pure"}
        if unaccounted:
            raise CompileError(
                f"program {decl.name!r} body uses effects {sorted(unaccounted)} "
                f"not listed in `! {{{', '.join(sorted(declared))}}}`",
                decl.line,
                decl.col,
            )

    def _expand_template_calls(
        self, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Inline parametric-program-template call sites in a step list.

        A ``draw v ~ T(args)`` step whose morphism name ``T`` is a
        registered parametric program denotes the instantiation of
        the dependent kernel at the supplied arguments. The body of
        ``T`` is substituted (formal parameters → actual arguments)
        and α-renamed (every locally-bound name is prefixed by
        ``v$``, except the return-variable which is renamed to ``v``
        directly so the call's binding receives the template's
        return value). The renamed step list replaces the call site.

        Recursive template calls (a template body that itself calls
        another template) are handled by post-expansion: after a
        template is inlined its expanded steps are themselves
        recursively expanded, with cycle detection.
        """
        expanded: list[ProgramStep] = []
        for step in steps:
            if (
                isinstance(step, DrawStep)
                and not step.is_observed
                and step.morphism in self._program_templates
            ):
                tmpl = self._program_templates[step.morphism]
                if len(step.vars) != 1:
                    raise CompileError(
                        f"template call {step.morphism!r} may bind only one "
                        f"variable, got tuple {step.vars}",
                        step.line,
                        step.col,
                    )
                bind_name = step.vars[0]
                args = step.args or ()
                inst = self._instantiate_template(tmpl, bind_name, args, step)
                # Recursively expand any nested template calls in the
                # inlined body.
                expanded.extend(self._expand_template_calls(inst))
                continue
            if (
                isinstance(step, PlateDrawStep)
                and step.morphism in self._program_templates
            ):
                raise CompileError(
                    f"template {step.morphism!r} cannot be called from a "
                    f"plate-draw step; use a bare 'draw' inside the template "
                    f"body for the plate or wrap the call in a per-index "
                    f"helper",
                    step.line,
                    step.col,
                )
            expanded.append(step)
        return tuple(expanded)

    def _instantiate_template(
        self,
        tmpl: ProgramDecl,
        bind_name: str,
        args: tuple,
        call_site: ProgramStep,
    ) -> tuple[ProgramStep, ...]:
        """Realise one call site of a parametric program template.

        Categorical denotation: given the dependent kernel
        :math:`\\Pi (p_i : P_i).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))`
        carried by ``tmpl``, return the concrete Kern-morphism at
        ``args`` (a section of the family). The morphism is
        represented as the renamed step list whose internal latents
        contribute their own factors to the caller's joint kernel,
        with the return-variable renamed to ``bind_name`` so the
        call's binding receives the template's output value.

        Substitution + α-renaming together realise the categorical
        substitution lemma: substituting actuals for formals
        commutes with denotation up to renaming-equivalence.
        """
        type_params = tmpl.type_params or ()
        if len(args) != len(type_params):
            raise CompileError(
                f"template {tmpl.name!r} expects {len(type_params)} arguments, "
                f"got {len(args)}",
                call_site.line,
                call_site.col,
            )
        # Build the parameter-substitution environment.
        type_subst: dict[str, ObjectExpr] = {}
        value_subst: dict[str, str | float] = {}
        for param, arg in zip(type_params, args):
            if isinstance(param, ObjectParam):
                # `DrawArgName` carries the identifier text; a bare
                # `str` remains admissible for legacy synthetic
                # steps. Anything else is a literal or nested
                # distribution call, not a valid type-name argument.
                if isinstance(arg, DrawArgName):
                    arg = arg.text
                elif not isinstance(arg, str):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} "
                        f"({param.universe}) requires a type-name argument, "
                        f"got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                # Validate the named object/space matches the declared
                # universe.
                if param.universe == "FinSet" and arg not in self._objects:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"FinSet expects a finite-set object, but {arg!r} is "
                        f"not a declared object",
                        call_site.line,
                        call_site.col,
                    )
                if param.universe == "Space" and arg not in self._spaces:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Space expects a continuous space, but {arg!r} is "
                        f"not a declared space",
                        call_site.line,
                        call_site.col,
                    )
                if (
                    param.universe == "Object"
                    and arg not in self._objects
                    and arg not in self._spaces
                ):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Object expects a declared object or space, got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                type_subst[param.name] = TypeName(
                    name=arg, line=call_site.line, col=call_site.col
                )
            elif isinstance(param, ScalarParam):
                # A scalar parameter admits either a numeric literal
                # (`DrawArgScalar`) or a bound-name reference
                # (`DrawArgName`). Legacy `str` / `float` values from
                # compiler-synthesized call sites are equally
                # admissible.
                if isinstance(arg, DrawArgName):
                    value_subst[param.name] = arg.text
                elif isinstance(arg, DrawArgScalar):
                    value_subst[param.name] = float(arg.value)
                elif isinstance(arg, str):
                    value_subst[param.name] = arg
                else:
                    value_subst[param.name] = float(arg)
            elif isinstance(param, MorphismParam):
                if isinstance(arg, DrawArgName):
                    arg = arg.text
                elif not isinstance(arg, str):
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r} : "
                        f"Mor[...] expects a morphism name, got {arg!r}",
                        call_site.line,
                        call_site.col,
                    )
                if arg not in self._morphisms and arg not in self._program_templates:
                    raise CompileError(
                        f"template {tmpl.name!r}: parameter {param.name!r}: "
                        f"morphism {arg!r} is not declared",
                        call_site.line,
                        call_site.col,
                    )
                value_subst[param.name] = arg
            else:
                raise CompileError(
                    f"template {tmpl.name!r}: unknown parameter kind for "
                    f"{getattr(param, 'name', '?')!r}",
                    call_site.line,
                    call_site.col,
                )
        # Collect all locally-bound names in the template body (latents
        # drawn, plate-draws, lets, observe loop-vars). These are
        # α-renamed to live in the caller's namespace.
        local_names = self._collect_template_local_names(tmpl)
        # The return variable (if a single identifier) receives the
        # call's binding name directly; other locals are namespaced.
        return_var = tmpl.return_vars[0] if len(tmpl.return_vars) == 1 else None
        rename: dict[str, str] = {}
        for nm in local_names:
            if nm == return_var:
                rename[nm] = bind_name
            else:
                rename[nm] = f"{bind_name}${nm}"
        # Expand the template body's BindStep IR into the
        # compiler's internal step shapes first (so the rename pass
        # operates on a uniform IR).
        expanded_body = self._expand_bind_steps(tmpl.draws)
        self._refuse_computation_calls(tmpl.name, expanded_body)
        # Walk the expanded body, applying parameter substitution +
        # α-renaming step by step.
        return tuple(
            self._rename_step(step, type_subst, value_subst, rename)
            for step in expanded_body
        )

    def _inline_pure_calls(
        self, program: str, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Lower every call of a computation to compiled runtime steps.

        A call of a pure computation becomes let steps (its calls of pure
        computations inlined in turn, an ``if`` between pure branches as
        ``select``), by the analysis the transpile planner shares; a call
        of a computation whose only requests are the program's ``Random``
        and ``Score`` runs in place, its sites named ``y$label``; and
        ``traverse`` maps over its items, a drawing callee as plated
        draws (see `lower_calls`). A call that cannot be lowered is left
        in place, and its located refusal is kept under the program's
        name, which fitting the program or instantiating it raises.

        Parameters
        ----------
        program : str
            The program's name.
        steps : tuple[ProgramStep, ...]
            The program's steps, searched through marginalization
            scopes.

        Returns
        -------
        tuple[ProgramStep, ...]
            The steps with every lowerable call lowered.
        """
        if not _calls_a_computation(steps):
            return steps
        module = self.qiec_module
        if module is None or module.gap:
            return steps
        cards = {
            name: obj.cardinality
            for name, obj in self._objects.items()
            if isinstance(obj, FinSet)
        }
        lowered, refusal = lower_calls(program, steps, module, cards, _names_in(steps))
        if refusal is not None:
            self._call_refusals[program] = refusal
        return lowered

    def _refuse_computation_calls(
        self, program: str, steps: tuple[ProgramStep, ...]
    ) -> None:
        """Refuse to instantiate a program that keeps a call of a computation.

        The runtime compiler instantiates a template's steps in place,
        and a call it could not lower to steps has no step to run as.

        Parameters
        ----------
        program : str
            The program instantiated.
        steps : tuple[ProgramStep, ...]
            Its steps, after its calls were lowered.

        Raises
        ------
        CompileError
            The refusal recorded when the call was lowered, located at
            the call and saying why it cannot be fit.
        """
        call = _first_call_step(steps)
        if call is None:
            return
        refusal = self._call_refusals.get(program)
        if refusal is not None:
            raise refusal
        raise CompileError(
            f"program {program!r} runs `let {call.name} <- {call.call.callee}"
            f"(...)`, which the runtime compiler cannot instantiate in place",
            call.line,
            call.col,
        )

    def _collect_template_local_names(self, tmpl: ProgramDecl) -> set[str]:
        """All names bound inside the template body (latents + lets).

        Walks the surface step IR (``SampleStep`` / ``ObserveStep`` /
        ``MarginalizeStep`` / ``LetStep``) and contributes every
        locally-bound name to the alpha-renaming set. Also handles
        the internal IR (``BindStep`` / ``DrawStep`` /
        ``PlateDrawStep`` / etc.) so the same routine can be called
        on already-expanded step lists.
        """
        out: set[str] = set()

        def _walk(steps):
            for step in steps:
                if isinstance(step, SampleStep):
                    out.update(step.vars)
                elif isinstance(step, ObserveStep):
                    out.update(step.vars)
                elif isinstance(step, MarginalizeStep):
                    out.add(step.var)
                    if step.scope:
                        _walk(step.scope)
                elif isinstance(step, BindStep):
                    out.update(step.vars)
                    if step.scope is not None:
                        _walk(step.scope)
                elif isinstance(step, (LetStep, ScoreStep, CallStep)):
                    out.add(step.name)
                # Internal IR (post-expand) — also covered, for the
                # case where _collect_local_names is invoked on
                # already-expanded steps.
                elif isinstance(step, DrawStep):
                    out.update(step.vars)
                elif isinstance(step, PlateDrawStep):
                    out.add(step.name)
                elif isinstance(step, VectorisedObserveStep):
                    out.add(step.index_var)
                    if step.response_var:
                        out.add(step.response_var)
                elif isinstance(step, GroupedMarginalizeStep):
                    out.add(step.var_name)
                    if step.body_ll_var is not None:
                        out.add(step.body_ll_var)
                elif isinstance(step, GroupedLatentInitStep):
                    out.add(step.latent_name)
                elif isinstance(step, GroupedBodyObserveStep):
                    out.add(step.index_var)
                    if step.response_var:
                        out.add(step.response_var)
                    if step.latent_name:
                        out.add(step.latent_name)

        _walk(tmpl.draws)
        return out

    def _rename_type(
        self, texpr: ObjectExpr, type_subst: dict[str, ObjectExpr]
    ) -> ObjectExpr:
        """Substitute object parameters inside a type expression."""
        if isinstance(texpr, TypeName):
            if texpr.name in type_subst:
                return type_subst[texpr.name]
            return texpr
        if isinstance(texpr, ObjectProduct):
            return ObjectProduct(
                components=tuple(
                    self._rename_type(c, type_subst) for c in texpr.components
                ),
                line=texpr.line,
                col=texpr.col,
            )
        return texpr

    def _rename_args(
        self,
        args: tuple | None,
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> tuple | None:
        """Apply parameter substitution and alpha-renaming inside a
        draw-arg list. `args` is a tuple of `DrawArg` tagged variants
        (legacy bare `str` / `float` values also pass through). A
        `DrawArgName` whose `text` matches a substitution key is
        rewritten to the substituted value: a numeric substitute
        becomes `DrawArgScalar`, a string substitute becomes a
        `DrawArgName` with the new text.
        """
        if args is None:
            return None
        out: list = []
        for a in args:
            if isinstance(a, DrawArgNamed):
                renamed = self._rename_args((a.value,), value_subst, rename)
                assert renamed is not None
                out.append(a.with_(value=renamed[0]))
            elif isinstance(a, DrawArgDist):
                out.append(
                    a.with_(args=self._rename_args(a.args, value_subst, rename) or ())
                )
            elif isinstance(a, DrawArgList):
                out.append(
                    a.with_(items=self._rename_args(a.items, value_subst, rename) or ())
                )
            elif isinstance(a, DrawArgName):
                key = a.text
                if key in value_subst:
                    sub = value_subst[key]
                    if isinstance(sub, (int, float)) and not isinstance(sub, bool):
                        out.append(
                            DrawArgScalar(value=float(sub), line=a.line, col=a.col)
                        )
                    else:
                        out.append(a.with_(text=str(sub)))
                elif key in rename:
                    out.append(a.with_(text=rename[key]))
                else:
                    out.append(a)
            elif isinstance(a, DrawArgIndex):
                # Rewrite the base name plus every index identifier
                # under the substitution / rename maps so both the
                # tensor and its indexing plate propagate through
                # template instantiation.
                new_name = a.name
                if new_name in value_subst:
                    sub = value_subst[new_name]
                    new_name = (
                        str(sub) if not isinstance(sub, (int, float)) else new_name
                    )
                elif new_name in rename:
                    new_name = rename[new_name]
                new_indices: list[str] = []
                for ix in a.indices:
                    if ix in value_subst:
                        sub_ix = value_subst[ix]
                        new_indices.append(str(sub_ix))
                    elif ix in rename:
                        new_indices.append(rename[ix])
                    else:
                        new_indices.append(ix)
                out.append(a.with_(name=new_name, indices=tuple(new_indices)))
            elif isinstance(a, str):
                if a in value_subst:
                    out.append(value_subst[a])
                elif a in rename:
                    out.append(rename[a])
                else:
                    out.append(a)
            else:
                out.append(a)
        return tuple(out)

    def _rename_step(
        self,
        step: ProgramStep,
        type_subst: dict[str, ObjectExpr],
        value_subst: dict[str, str | float],
        rename: dict[str, str],
    ) -> ProgramStep:
        """Apply parameter substitution + α-renaming to a single step."""
        if isinstance(step, DrawStep):
            new_vars = tuple(rename.get(v, v) for v in step.vars)
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"draw step morphism {step.morphism!r} substituted to a "
                    f"non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return DrawStep(
                vars=new_vars,
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                is_observed=step.is_observed,
                axes=step.axes,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, PlateDrawStep):
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"plate-draw step morphism {step.morphism!r} substituted "
                    f"to a non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return PlateDrawStep(
                name=rename.get(step.name, step.name),
                index=self._rename_type(step.index, type_subst),
                codomain=self._rename_type(step.codomain, type_subst),
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                axes=step.axes,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, LetStep):
            return LetStep(
                name=rename.get(step.name, step.name),
                value=self._rename_let_expr(
                    step.value, type_subst, value_subst, rename, step.line, step.col
                ),
                line=step.line,
                col=step.col,
            )
        if isinstance(step, ScoreStep):
            return step.with_(
                name=rename.get(step.name, step.name),
                value=self._rename_let_expr(
                    step.value, type_subst, value_subst, rename, step.line, step.col
                ),
            )
        if isinstance(step, VectorisedObserveStep):
            new_morph = value_subst.get(step.morphism, step.morphism)
            if not isinstance(new_morph, str):
                raise CompileError(
                    f"observe step morphism {step.morphism!r} substituted to "
                    f"a non-string value {new_morph!r}",
                    step.line,
                    step.col,
                )
            return VectorisedObserveStep(
                index_var=rename.get(step.index_var, step.index_var),
                index_set=self._rename_type(step.index_set, type_subst),
                morphism=new_morph,
                args=self._rename_args(step.args, value_subst, rename),
                response_var=rename.get(step.response_var, step.response_var),
                axes=step.axes,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, GroupedMarginalizeStep):
            renamed_probs = (
                rename.get(step.probs_var, step.probs_var)
                if step.probs_var is not None
                else None
            )
            renamed_body_ll = (
                rename.get(step.body_ll_var, step.body_ll_var)
                if step.body_ll_var is not None
                else None
            )
            renamed_body_observes: tuple[GroupedObserveEntry, ...] | None = None
            if step.body_observes is not None:
                renamed_body_observes = tuple(
                    GroupedObserveEntry(
                        ll_slot=rename.get(entry.ll_slot, entry.ll_slot),
                        fibration_var=(
                            rename.get(entry.fibration_var, entry.fibration_var)
                            if entry.fibration_var is not None
                            else None
                        ),
                        fibration_axes=(
                            tuple(rename.get(v, v) for v in entry.fibration_axes)
                            if entry.fibration_axes is not None
                            else None
                        ),
                    )
                    for entry in step.body_observes
                )
            return GroupedMarginalizeStep(
                var_name=rename.get(step.var_name, step.var_name),
                class_size=step.class_size,
                probs_var=renamed_probs,
                probs_indices=tuple(rename.get(v, v) for v in step.probs_indices),
                parameterization=step.parameterization,
                over_obj=step.over_obj,
                over_objs=step.over_objs,
                body_ll_var=renamed_body_ll,
                body_observes=renamed_body_observes,
                reduction=step.reduction,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, GroupedLatentInitStep):
            return GroupedLatentInitStep(
                latent_name=rename.get(step.latent_name, step.latent_name),
                class_size=step.class_size,
                line=step.line,
                col=step.col,
            )
        if isinstance(step, GroupedBodyObserveStep):
            return GroupedBodyObserveStep(
                response_var=rename.get(step.response_var, step.response_var),
                morphism=step.morphism,
                args=self._rename_args(step.args, value_subst, rename),
                index_set=self._rename_type(step.index_set, type_subst)
                if step.index_set is not None
                else None,
                index_var=rename.get(step.index_var, step.index_var),
                latent_name=rename.get(step.latent_name, step.latent_name),
                class_size=step.class_size,
                fibration_var=rename.get(step.fibration_var, step.fibration_var)
                if step.fibration_var is not None
                else None,
                fibration_axes=step.fibration_axes,
                ll_slot=step.ll_slot,
                line=step.line,
                col=step.col,
            )
        raise CompileError(
            f"unsupported step kind in template body: {type(step).__name__}",
            getattr(step, "line", 0),
            getattr(step, "col", 0),
        )

    def _rename_let_expr(
        self,
        expr: LetExprNode,
        type_subst: dict[str, ObjectExpr],
        value_subst: dict[str, str | float],
        rename: dict[str, str],
        line: int,
        col: int,
    ) -> LetExprNode:
        """Apply parameter substitution and alpha-renaming inside a let RHS.

        The rewrite is one capture-avoiding substitution over the whole
        expression: a scalar parameter becomes its literal or the
        caller's name, a template local becomes its renamed name
        wherever it occurs (a variable, the array of an index, a call
        argument, a method receiver, a collection item, a lambda or
        factor body), a morphism parameter called as ``f(...)`` becomes
        the morphism it takes, and an object parameter in a factor
        binder's index becomes the object it takes. A lambda parameter
        or factor binder shadows a parameter or local of the same name
        and is renamed if it would capture a substituted name.

        Parameters
        ----------
        expr : LetExprNode
            The let right-hand side from the template body.
        type_subst : dict[str, ObjectExpr]
            The object each object parameter takes.
        value_subst : dict[str, str | float]
            The value each scalar or morphism parameter takes: a
            number, or the name of a bound value or a morphism.
        rename : dict[str, str]
            The caller's name of each name the template binds.
        line : int
            The source line of the step holding the expression.
        col : int
            The source column of the step holding the expression.

        Returns
        -------
        LetExprNode
            The rewritten expression.

        Raises
        ------
        CompileError
            If a parameter that takes a number is called as a function.
        TypeError
            If the expression contains a let-expression variant the
            substitution does not cover; the message names it.
        """
        values: dict[str, LetExprNode] = {}
        callees: dict[str, str] = {}
        for name, target in rename.items():
            values[name] = LetExprVar(name=target)
            callees[name] = target
        for name, sub in value_subst.items():
            if isinstance(sub, str):
                values[name] = LetExprVar(name=sub)
                callees[name] = sub
            else:
                values[name] = LetExprLiteral(value=float(sub))
        for callee in free_let_callees(expr):
            if callee in value_subst and callee not in callees:
                raise CompileError(
                    f"let-expression callee {callee!r} is a parameter that "
                    f"takes the number {value_subst[callee]!r}, not a function",
                    line,
                    col,
                )

        def substitute_objects(node: LetExprNode) -> LetExprNode:
            """Rename the objects a factor's binders range over.

            Parameters
            ----------
            node : LetExprNode
                The expression.

            Returns
            -------
            LetExprNode
                The expression, its factor binders' objects renamed.
            """
            if not isinstance(node, LetExprFactor):
                return node
            return node.with_(
                binders=tuple(
                    binder.with_(index=self._rename_type(binder.index, type_subst))
                    for binder in node.binders
                )
            )

        renamed = substitute_let_expr(expr, values, callees=callees)
        return map_let_expr(renamed, substitute_objects)

    def _compile_morphism_call(self, expr: ExprMorphismCall):
        """Compile ``callee(arg1, arg2, …)`` — currently used to
        invoke a `ContractionDecl` at a define-binding site.

        Resolves ``expr.callee`` against the registered
        contractions, validates that the argument count matches
        the contraction's expected arity, resolves each argument
        against the morphism scope, and runs the einsum-style
        contraction. Returns the result as an
        `ObservedMorphism` with the contraction's declared
        domain and codomain.
        """
        contraction = self._contractions.get(expr.callee)
        if contraction is None:
            # Fall through to parametric-program template
            # invocation: ``define applied = p(f)`` is the existing
            # surface and should keep working when ``p`` is a
            # parametric program rather than a contraction.
            if expr.callee in self._program_templates:
                return self._compile_program_template_call(expr)
            raise CompileError(
                f"morphism-call: undefined {expr.callee!r}; "
                f"contractions: {sorted(self._contractions)}; "
                f"program templates: {sorted(self._program_templates)}",
                expr.line,
                expr.col,
            )
        if len(expr.args) != len(contraction.input_types):
            raise CompileError(
                f"contraction {expr.callee!r}: expected "
                f"{len(contraction.input_types)} arguments "
                f"({', '.join(n for n, _, _ in contraction.input_types)}), "
                f"got {len(expr.args)}",
                expr.line,
                expr.col,
            )
        resolved_morphs: list = []
        for arg_name, (param_name, exp_dom, exp_cod) in zip(
            expr.args, contraction.input_types
        ):
            if arg_name not in self._morphisms:
                raise CompileError(
                    f"contraction {expr.callee!r}: argument "
                    f"{arg_name!r} (for parameter {param_name!r}) "
                    f"is not a declared morphism",
                    expr.line,
                    expr.col,
                )
            morph = self._morphisms[arg_name]
            # Light shape check: matching numel for domain / codomain.
            if _numel_shape(morph.domain.shape) != _numel_shape(
                exp_dom.shape
            ) or _numel_shape(morph.codomain.shape) != _numel_shape(exp_cod.shape):
                raise CompileError(
                    f"contraction {expr.callee!r}: argument {arg_name!r} "
                    f"has shape {tuple(morph.domain.shape)} -> "
                    f"{tuple(morph.codomain.shape)} but parameter "
                    f"{param_name!r} declares {tuple(exp_dom.shape)} -> "
                    f"{tuple(exp_cod.shape)}",
                    expr.line,
                    expr.col,
                )
            resolved_morphs.append(morph)
        # Run the wiring.
        tensors = [m.tensor for m in resolved_morphs]
        result_tensor = contraction.wiring.apply(*tensors)
        return ObservedMorphism(
            contraction.domain,
            contraction.codomain,
            result_tensor,
            algebra=contraction.algebra,
        )

    def _make_template_invoker(self, name: str):
        """Return a Python-callable wrapper around a parametric
        program template, suitable for attaching to a compiled
        `Program` so callers can write
        ``prog.<name>(alpha=0.5, beta=0.1)`` to instantiate the
        template at concrete parameters.
        """
        tmpl = self._program_templates[name]
        type_params = tmpl.type_params or ()
        param_names = tuple(p.name for p in type_params)

        def _invoke(*args, **kwargs) -> Program:
            if args and kwargs:
                raise TypeError(
                    f"template {name!r}: pass either all positional or all "
                    f"keyword arguments, not both"
                )
            if kwargs:
                missing = [n for n in param_names if n not in kwargs]
                extra = [k for k in kwargs if k not in param_names]
                if missing:
                    raise TypeError(f"template {name!r} missing arguments: {missing}")
                if extra:
                    raise TypeError(
                        f"template {name!r} got unexpected arguments: {extra}"
                    )
                ordered = tuple(kwargs[n] for n in param_names)
            else:
                if len(args) != len(param_names):
                    raise TypeError(
                        f"template {name!r} expects {len(param_names)} "
                        f"arguments, got {len(args)}"
                    )
                ordered = args
            normalized = tuple(str(a) if not isinstance(a, str) else a for a in ordered)
            call_expr = ExprMorphismCall(
                callee=name,
                args=normalized,
                line=tmpl.line,
                col=tmpl.col,
            )
            morph = self._compile_program_template_call(call_expr)
            return Program(morph)

        _invoke.__name__ = name
        _invoke.__qualname__ = f"template:{name}"
        return _invoke

    def _compile_program_template_call(self, expr: ExprMorphismCall):
        """Instantiate a parametric program template at a define-binding
        site: ``define applied = p(f, …)`` substitutes the actual
        morphism / object / scalar arguments for the template's
        formal parameters, builds a synthetic non-parametric
        `ProgramDecl`, and compiles it into a runtime
        `MonadicProgram` morphism.

        Realises the dependent-kernel application
        :math:`\\Pi (p:P).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))`
        at the supplied section: the result is the concrete
        Kern-morphism the template denotes at those parameters,
        materialised as a standalone morphism rather than inlined
        into an enclosing program.
        """
        tmpl = self._program_templates[expr.callee]
        type_params = tmpl.type_params or ()
        args = expr.args
        if len(args) != len(type_params):
            raise CompileError(
                f"template {expr.callee!r} expects {len(type_params)} "
                f"arguments, got {len(args)}",
                expr.line,
                expr.col,
            )
        type_subst: dict[str, ObjectExpr] = {}
        value_subst: dict[str, str | float] = {}
        for param, arg in zip(type_params, args):
            if isinstance(param, ObjectParam):
                if param.universe == "FinSet" and arg not in self._objects:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : FinSet expects a finite-set "
                        f"object, but {arg!r} is not a declared object",
                        expr.line,
                        expr.col,
                    )
                if param.universe == "Space" and arg not in self._spaces:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : Space expects a continuous "
                        f"space, but {arg!r} is not a declared space",
                        expr.line,
                        expr.col,
                    )
                if (
                    param.universe == "Object"
                    and arg not in self._objects
                    and arg not in self._spaces
                ):
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r} : Object expects a declared "
                        f"object or space, got {arg!r}",
                        expr.line,
                        expr.col,
                    )
                type_subst[param.name] = TypeName(
                    name=arg, line=expr.line, col=expr.col
                )
            elif isinstance(param, ScalarParam):
                # ``arg`` arrives either as a numeric literal (from a
                # Python invocation) or as a bare identifier naming a
                # let-bound scalar in the caller's scope. Numeric
                # strings parse to floats; everything else stays as a
                # name reference for the caller's bound_vars to
                # resolve at draw-site time.
                if isinstance(arg, (int, float)):
                    value_subst[param.name] = float(arg)
                elif isinstance(arg, str):
                    try:
                        value_subst[param.name] = float(arg)
                    except ValueError:
                        value_subst[param.name] = arg
                else:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r}: scalar argument has "
                        f"unsupported type {type(arg).__name__}",
                        expr.line,
                        expr.col,
                    )
            elif isinstance(param, MorphismParam):
                if arg not in self._morphisms and arg not in self._program_templates:
                    raise CompileError(
                        f"template {expr.callee!r}: parameter "
                        f"{param.name!r}: morphism {arg!r} is not "
                        f"declared",
                        expr.line,
                        expr.col,
                    )
                value_subst[param.name] = arg
            else:
                raise CompileError(
                    f"template {expr.callee!r}: unknown parameter kind "
                    f"for {getattr(param, 'name', '?')!r}",
                    expr.line,
                    expr.col,
                )
        # Substitute parameters through the template body. We do not
        # α-rename locals here: the synthetic program owns its own
        # scope, so the template's local names (return-var, latents,
        # let-bindings) need no further refresh.
        expanded_body = self._expand_bind_steps(tmpl.draws)
        self._refuse_computation_calls(expr.callee, expanded_body)
        empty_rename: dict[str, str] = {}
        substituted_body = tuple(
            self._rename_step(step, type_subst, value_subst, empty_rename)
            for step in expanded_body
        )
        synth_domain = self._rename_type(tmpl.domain, type_subst)
        synth_codomain = self._rename_type(tmpl.codomain, type_subst)
        synth_name = f"__tmpl_call${expr.callee}${expr.line}${expr.col}"
        # Defensively dodge collisions if the same call site is
        # compiled twice for any reason.
        suffix = 0
        unique_name = synth_name
        while unique_name in self._morphisms:
            suffix += 1
            unique_name = f"{synth_name}#{suffix}"
        synth = ProgramDecl(
            name=unique_name,
            params=None,
            domain=synth_domain,
            codomain=synth_codomain,
            draws=substituted_body,
            return_vars=tmpl.return_vars,
            return_labels=tmpl.return_labels,
            options=tmpl.options,
            type_params=None,
            docs=(),
            line=expr.line,
            col=expr.col,
        )
        self._compile_program(synth)
        return self._morphisms.pop(unique_name)

    def _compile_contraction(self, decl: ContractionDecl) -> None:
        """Compile a ``contraction`` declaration into a callable
        registered under ``decl.name``.

        Categorically, a contraction is an n-ary operadic morphism:
        it takes ``len(decl.inputs)`` input morphisms, contracts
        them together under the named composition rule using the
        einsum-style wiring spec, and returns a fresh morphism
        with the declared ``domain -> codomain`` typing.

        The compiled callable accepts that many input morphisms (in
        the order declared in ``decl.inputs``) and returns an
        `ObservedMorphism` whose tensor is the contraction
        result. The callable is registered in the morphism table
        so the user can invoke it like any other morphism:
        ``define out = op_apply(arg1, arg2, kernel)``.
        """
        if decl.name in self._morphisms or decl.name in self._program_templates:
            raise CompileError(
                f"contraction {decl.name!r} already declared as a morphism or program",
                decl.line,
                decl.col,
            )
        check_option_keys(
            decl.options,
            _CONTRACTION_OPTION_KEYS,
            owner=f"contraction {decl.name!r}",
            line=decl.line,
            col=decl.col,
        )
        declared_rule = get_option_name(
            decl.options,
            "rule",
            line=decl.line,
            col=decl.col,
        )
        if declared_rule is None:
            raise CompileError(
                f"contraction {decl.name!r}: required option "
                f"``rule=<NAME>`` is missing",
                decl.line,
                decl.col,
            )
        rule_name = declared_rule.lower()
        if rule_name not in _ALGEBRA_REGISTRY:
            rule_entry = find_option(decl.options, "rule")
            ln = (
                rule_entry.line
                if rule_entry is not None and rule_entry.line
                else decl.line
            )
            cl = (
                rule_entry.col
                if rule_entry is not None and rule_entry.line
                else decl.col
            )
            raise CompileError(
                f"contraction {decl.name!r}: unknown rule "
                f"{declared_rule!r}; available: "
                f"{', '.join(sorted(_ALGEBRA_REGISTRY))}",
                ln,
                cl,
            )
        rule = _ALGEBRA_REGISTRY[rule_name]
        wiring_text = get_option_string(
            decl.options,
            "wiring",
            line=decl.line,
            col=decl.col,
        )
        shared_axes = get_option_name_list(
            decl.options,
            "share",
            line=decl.line,
            col=decl.col,
        )
        if wiring_text:
            # Explicit einsum escape hatch (still supported for
            # contractions the type-driven rule can't express:
            # diagonal extraction, reorderings, etc.).
            wiring_spec = wiring_text
        else:
            try:
                wiring_spec = _infer_wiring_from_signature(
                    inputs=decl.inputs,
                    output_domain=decl.domain,
                    output_codomain=decl.codomain,
                    shared_axes=shared_axes,
                )
            except ValueError as exc:
                raise CompileError(
                    f"contraction {decl.name!r}: cannot infer wiring "
                    f"from typed signature: {exc}. Pass an explicit "
                    f'``wiring "..."`` clause or a ``share`` clause to '
                    f"disambiguate.",
                    decl.line,
                    decl.col,
                ) from exc
        try:
            wiring = EinsumWiring(rule, wiring_spec)
        except ValueError as exc:
            raise CompileError(
                f"contraction {decl.name!r}: invalid wiring {wiring_spec!r}: {exc}",
                decl.line,
                decl.col,
            ) from exc
        expected_arity = wiring.input_arity
        if expected_arity != len(decl.inputs):
            raise CompileError(
                f"contraction {decl.name!r}: wiring spec declares "
                f"{expected_arity} inputs but the parameter list "
                f"has {len(decl.inputs)}",
                decl.line,
                decl.col,
            )
        # Resolve domain / codomain object refs.
        domain_obj = self._resolve_type(decl.domain)
        codomain_obj = self._resolve_type(decl.codomain)
        # Resolve each input's domain / codomain for validation.
        input_types = [
            (
                inp.name,
                self._resolve_type(inp.input_domain),
                self._resolve_type(inp.input_codomain),
            )
            for inp in decl.inputs
        ]
        self._contractions[decl.name] = _CompiledContraction(
            name=decl.name,
            wiring=wiring,
            domain=domain_obj,
            codomain=codomain_obj,
            input_types=input_types,
            algebra=rule,
        )

    def _compile_program(
        self, decl: ProgramDecl, *, step_program: bool = False
    ) -> MonadicProgram | None:
        """Compile a monadic program block into a MonadicProgram.

        A ``sample`` or ``observe`` step that runs a scan is expanded
        first, as the checked elaboration expands it: into a
        ``let h = scan(step, xs)`` over a step program, which compiles
        to a [`Scan`][quivers.continuous.Scan] step. The step program
        draws a morphism cell at the site the scan's binding names, or
        runs a program cell through a [`Call`][quivers.continuous.Call].
        The step's sites are thus sites of this program, reached once
        per position, and a name a program cell reads without binding
        it is this program's binding of that name before the scan, else
        host data.

        Parametric programs (those carrying ``type_params``) are not
        compiled into a runtime ``MonadicProgram`` directly. They
        denote a dependent kernel

        .. math::

            \\Pi (p_1 : P_1) \\ldots \\Pi (p_n : P_n).\\ \\mathbf{Kern}(\\mathrm{dom}(p),\\, \\mathrm{cod}(p))

        in the indexed family of Kleisli arrows over the parameter
        category, and are stored as templates. Each call site of a
        template (a ``draw v ~ template(args)`` step inside another
        program) is realised by substituting the actual arguments
        for the formal parameters and α-renaming all locally-bound
        latents under the call's binding name, then inlining the
        renamed body into the caller's step list. The freshness of
        latent names per call site is the syntactic shadow of the
        fact that distinct call sites contribute distinct factors
        to the parent's joint kernel.

        Parameters
        ----------
        decl : ProgramDecl
            The declaration.
        step_program : bool
            Whether the declaration is a scan's step program, which is
            compiled for the scan that runs it rather than registered,
            and whose calls of programs run in place.

        Returns
        -------
        MonadicProgram | None
            The step program when ``step_program`` is set, else ``None``:
            the program is registered under its name.

        Raises
        ------
        CompileError
            If the declaration does not compile.
        """
        check_option_keys(
            decl.options,
            _PROGRAM_OPTION_KEYS,
            owner=f"program {decl.name!r}",
            line=decl.line,
            col=decl.col,
        )
        if decl.type_params is not None:
            # Parametric program — store as a template; defer body
            # compilation until each call site instantiates it.
            if decl.name in self._morphisms or decl.name in self._program_templates:
                raise CompileError(
                    f"morphism {decl.name!r} already declared",
                    decl.line,
                    decl.col,
                )
            if decl.params is not None:
                raise CompileError(
                    f"parametric program {decl.name!r} cannot also take data parameters",
                    decl.line,
                    decl.col,
                )
            self._program_templates[decl.name] = decl.with_(
                draws=self._inline_pure_calls(decl.name, decl.draws)
            )
            return None
        if decl.name in self._morphisms:
            raise CompileError(
                f"morphism {decl.name!r} already declared", decl.line, decl.col
            )
        domain = self._resolve_any_space(decl.domain)
        codomain = self._resolve_any_space(decl.codomain)
        if decl.params is not None:
            if isinstance(domain, (ProductSet, ProductSpace)):
                if len(decl.params) != len(domain.components):
                    raise CompileError(
                        f"program has {len(decl.params)} params but domain has {len(domain.components)} components",
                        decl.line,
                        decl.col,
                    )
            elif len(decl.params) != 1:
                raise CompileError(
                    f"program has {len(decl.params)} params but domain is not a product type",
                    decl.line,
                    decl.col,
                )
        decl, scan_effects = self._expand_cell_scans(decl)
        decl = decl.with_(draws=self._inline_pure_calls(decl.name, decl.draws))
        decl = decl.with_(draws=self._latent_kernel_draws(decl.draws))
        if not step_program and _calls_a_computation(decl.draws):
            # A program calling a named computation runs as its checked
            # computation: the callee's body is a term of the module,
            # which only the reference machine runs.
            self._bind_checked_program(decl, domain, codomain)
            return None
        bound_vars: dict[str, AnySpace | None] = {}
        if decl.params is not None:
            if isinstance(domain, (ProductSet, ProductSpace)):
                for pname, factor in zip(decl.params, domain.components):
                    bound_vars[pname] = factor
            else:
                bound_vars[decl.params[0]] = domain
        # First, expand the unified surface (BindStep) into the
        # internal IR (DrawStep / PlateDrawStep / VectorisedObserveStep /
        # GroupedMarginalizeStep) that the rest of the compiler consumes.
        # The expansion translates each BindStep based on its mode +
        # index annotation, and inlines marginalize scopes.
        ir_draws = self._expand_bind_steps(decl.draws)
        # Effect-set verification: walk the expanded IR and check
        # that the declared `!` capability set is consistent with
        # the body's actual effect usage.
        self._verify_effects(decl, ir_draws, scan_effects)
        # Then expand parametric-template call sites by inlining the
        # substituted + α-renamed template body. This realises the
        # dependent-kernel application: each call site is a section
        # of the family Π(p:P).Kern(dom(p), cod(p)) at the supplied
        # arguments, contributing its own factors to the parent's
        # joint kernel.
        expanded_draws = self._expand_template_calls(ir_draws)
        # The widths an inline family reads its parameters at come from
        # the checked types: of the host data the program reads, and of
        # each let as its operands type it.
        parameter_spaces = self._host_data_spaces(decl.name)
        self._parameter_space_stack.append(parameter_spaces)
        self._value_extents_stack.append(self._checked_value_extents(decl.name))
        steps: list[Step] = []
        for step in expanded_draws:
            if isinstance(step, PlateDrawStep):
                # draw v : A -> B ~ Family(args).  By the natural iso
                # Kern(1, B^A) ≅ Kern(A, B), the plate variable IS a
                # Kern-morphism A → B; we realise it as a PlateDraw
                # whose codomain is the flat product space of
                # |A| copies of the per-row family's codomain.
                idx_space = self._resolve_plate_index(
                    step.index,
                    f"indexed sample {step.name!r}",
                    step.line,
                    step.col,
                )
                annotated_event = self._annotation_is_event_axis(
                    step, idx_space, bound_vars
                )
                if annotated_event is not None:
                    if step.name in bound_vars:
                        raise CompileError(
                            f"variable {step.name!r} already bound in program",
                            step.line,
                            step.col,
                        )
                    family, step_args = self._resolve_draw_morphism(
                        DrawStep(
                            vars=(step.name,),
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=False,
                            axes=AxisSpec(over=(annotated_event,)),
                            line=step.line,
                            col=step.col,
                        ),
                        bound_vars,
                        idx_space,
                    )
                    if getattr(family.codomain, "dim", None) != idx_space.size:
                        raise CompileError(
                            f"{step.morphism} draw of {step.name!r} is one variate "
                            f"over {annotated_event!r} ({idx_space.size} entries), "
                            f"but the runtime builds this family at "
                            f"{getattr(family.codomain, 'dim', None)} entries; state "
                            f"the event's size through the family's own arguments",
                            step.line,
                            step.col,
                        )
                    bound_vars[step.name] = family.codomain
                    steps.append(
                        Draw(names=(step.name,), morphism=family, args=step_args)
                    )
                    continue
                array = self._independent_over_axes(
                    step.morphism, getattr(step, "axes", None), step.line, step.col
                )
                if array is not None:
                    array_shape, event_axes = array
                    family, step_args = self._resolve_draw_morphism(
                        DrawStep(
                            vars=(step.name,),
                            morphism=step.morphism,
                            args=step.args,
                            is_observed=False,
                            axes=event_axes,
                            line=step.line,
                            col=step.col,
                        ),
                        bound_vars,
                        Euclidean(name=f"_{step.name}", dim=1),
                    )
                    if step.name in bound_vars:
                        raise CompileError(
                            f"variable {step.name!r} already bound in program",
                            step.line,
                            step.col,
                        )
                    index_shape = (idx_space.size, *array_shape)
                    plate = PlateDraw(
                        math.prod(index_shape),
                        family,
                        domain=family.domain,
                        index_shape=index_shape,
                    )
                    bound_vars[step.name] = plate.codomain
                    steps.append(
                        Draw(names=(step.name,), morphism=plate, args=step_args)
                    )
                    continue
                # The per-row codomain `B` is either a declared object /
                # space or an integer literal interpreted as
                # `Euclidean(N)` — the standard convention for
                # continuous per-row families.
                if (
                    isinstance(step.codomain, TypeName)
                    and step.codomain.name.isdigit()
                    and step.codomain.name not in self._objects
                ):
                    cod_space = Euclidean(
                        name=f"_plate_codom_{step.name}",
                        dim=int(step.codomain.name),
                    )
                else:
                    cod_space = self._resolve_any_space(step.codomain)
                # An explicit ``[over=...]`` clause names the per-row
                # event axis; promote it over the placeholder
                # ``codomain`` so families like Dirichlet pick up the
                # correct simplex dimension.
                axes_cod = self._axes_codomain(getattr(step, "axes", None))
                if axes_cod is not None:
                    cod_space = axes_cod
                # A matrix-valued family draws one `K x K` variate, not
                # a plate of them: the index annotation supplies the
                # matrix dimension `K` (there is no other width in
                # scope) and the step compiles to a single draw rather
                # than a plate.
                matrix_spec = self._matrix_family_spec(step.morphism)
                is_matrix_draw = (
                    matrix_spec is not None or step.morphism == "InverseWishart"
                )
                if is_matrix_draw:
                    if axes_cod is not None:
                        raise CompileError(
                            f"matrix-valued family {step.morphism!r} takes a "
                            f"single width, and {step.name!r} names two: the "
                            f"index annotation and the ``[over=...]`` axis. "
                            f"Write one of them: the index annotation for "
                            f"``{step.name} : Dim <- {step.morphism}(...)``, "
                            f"or the axis for ``{step.name} <- "
                            f"{step.morphism}(...) [over=Dim]``",
                            step.line,
                            step.col,
                        )
                    cod_space = idx_space
                # Synthesize a DrawStep so we can reuse the inline /
                # family-registry resolution logic. The synthetic step
                # carries the plate's per-row codomain so the family
                # is built at the right dimensionality.
                _synth = DrawStep(
                    vars=(step.name,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=False,
                    axes=getattr(step, "axes", None),
                    line=step.line,
                    col=step.col,
                )
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, cod_space
                )
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )
                if is_matrix_draw:
                    bound_vars[step.name] = family.codomain
                    steps.append(
                        Draw(names=(step.name,), morphism=family, args=step_args)
                    )
                    continue
                plate = PlateDraw(idx_space.size, family, domain=family.domain)
                bound_vars[step.name] = plate.codomain
                steps.append(Draw(names=(step.name,), morphism=plate, args=step_args))
                continue
            if isinstance(step, GroupedLatentInitStep):
                # Initialise the latent's environment slot to
                # torch.arange(K) so any downstream let-step that
                # references the latent broadcasts across the class
                # axis.
                def _grouped_latent_init(
                    env: dict, _K: int = step.class_size
                ) -> torch.Tensor:
                    return torch.arange(_K, dtype=torch.long)

                if step.latent_name in bound_vars:
                    raise CompileError(
                        f"grouped marginalize: latent {step.latent_name!r} "
                        f"is already bound in scope",
                        step.line,
                        step.col,
                    )
                bound_vars[step.latent_name] = None
                steps.append(
                    Let(name=step.latent_name, value=reading(_grouped_latent_init, ()))
                )
                continue
            if isinstance(step, GroupedBodyObserveStep):
                # The captured observe inside a grouped marginalize
                # block: compute the family's per-row log-likelihood
                # against the supplied response data, broadcasting
                # any (K,)-shaped parameters across the class axis,
                # and store the resulting (N, K) tensor at this
                # observe's dedicated env slot.  The surrounding
                # GroupedMarginalizeStep collects each slot, pairs it with
                # the observe's fibration, and scatter-sums each
                # contribution into the shared (|G|, K) accumulator.
                idx_space = self._resolve_plate_index(
                    step.index_set,
                    f"grouped observe {step.response_var!r}",
                    step.line,
                    step.col,
                )
                _synth = DrawStep(
                    vars=(step.index_var,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=True,
                    line=step.line,
                    col=step.col,
                )
                family, step_args = self._resolve_draw_morphism(
                    _synth, bound_vars, codomain
                )
                # Register the response variable so the runtime can
                # supply data via the observations dict.
                if step.response_var not in bound_vars:
                    bound_vars[step.response_var] = family.codomain
                resp_var = step.response_var
                ll_slot = step.ll_slot or step.latent_name
                num_rows = int(idx_space.size)
                class_size = int(step.class_size or 1)

                def _captured_observe(
                    env: dict,
                    _family=family,
                    _args=step_args,
                    _resp=resp_var,
                    _slot=ll_slot,
                    _N_decl=num_rows,
                    _K=class_size,
                ) -> torch.Tensor:
                    # Take the row count from the response tensor at
                    # runtime: the per-row index axis carries the
                    # data the user actually supplied, which may be
                    # smaller than the declared codomain cardinality
                    # for sparse / minibatch fits.
                    response = env[_resp]
                    _N = int(response.shape[0]) if response.dim() >= 1 else _N_decl
                    # Resolve theta from env. Inside a grouped
                    # marginalize, ``args`` may reference per-class
                    # gathers (``mu[cls]``) that resolve to shape
                    # ``(K, ...)``; the response is shape ``(N,)``.
                    # The per-row per-class log-likelihood requires
                    # broadcasting theta to ``(N, K, ...)`` and the
                    # response to ``(N, 1)`` so the resulting
                    # ``log_prob`` has shape ``(N, K)``.
                    if _args is None:
                        theta = env["_x_input"]
                        parts = [theta]
                    elif len(_args) == 1:
                        parts = [_lookup_arg(env, _args[0])]
                        theta = parts[0]
                    else:
                        parts = [_lookup_arg(env, a) for a in _args]

                    # Detect whether any parameter carries the class
                    # axis. If so, broadcast each part to leading
                    # (N, K) plus any trailing event dims and call
                    # the family's underlying distribution builder
                    # directly so the inline family's split-by-dim
                    # path doesn't shrink the event axis.
                    needs_cross = any(p.dim() >= 1 and p.shape[0] == _K for p in parts)
                    if needs_cross:
                        event_shape: tuple[int, ...] = ()
                        bcast_parts = []
                        for p in parts:
                            if p.dim() == 0:
                                bcast_parts.append(p.expand(_N, _K))
                            elif p.shape[0] == _K:
                                bcast_parts.append(p.unsqueeze(0))
                                if p.dim() > 1:
                                    event_shape = torch.broadcast_shapes(
                                        event_shape, tuple(p.shape[1:])
                                    )
                            elif p.shape[0] == _N:
                                bcast_parts.append(p.unsqueeze(1))
                                if p.dim() > 1:
                                    event_shape = torch.broadcast_shapes(
                                        event_shape, tuple(p.shape[1:])
                                    )
                            else:
                                bcast_parts.append(p)
                        target = (_N, _K) + tuple(event_shape)
                        resolved = []
                        for p in bcast_parts:
                            try:
                                resolved.append(p.expand(target))
                            except RuntimeError:
                                resolved.append(p)
                        # A family applied to literals beside variables
                        # records the literals in its parameter spec;
                        # they take their place among the broadcast
                        # variable parts before the builder runs.
                        spec = getattr(_family, "_param_spec", None)
                        if spec is not None:
                            full: list[torch.Tensor] = []
                            variable_parts = iter(resolved)
                            for kind, literal in spec:
                                if kind == "lit":
                                    full.append(
                                        torch.full(
                                            target,
                                            float(literal),
                                            dtype=resolved[0].dtype
                                            if resolved
                                            else torch.get_default_dtype(),
                                        )
                                    )
                                else:
                                    full.append(next(variable_parts))
                            resolved = full
                        dist = _family._dist_builder(resolved)
                        resp_broadcast = response
                        # Add singleton dims to make response
                        # broadcastable against the (N, K) batch
                        # shape (the response has no K dependence).
                        while resp_broadcast.dim() < len(dist.batch_shape) + len(
                            dist.event_shape
                        ):
                            resp_broadcast = resp_broadcast.unsqueeze(1)
                        if _family._discrete:
                            resp_broadcast = resp_broadcast.float()
                        ll = dist.log_prob(resp_broadcast)
                    else:
                        if len(parts) > 1:
                            common_shape = torch.broadcast_shapes(
                                *(p.shape for p in parts)
                            )
                            expanded = [p.expand(common_shape) for p in parts]
                            theta = torch.stack(expanded, dim=-1)
                        else:
                            theta = parts[0]
                        ll = _family.log_prob(theta, response)
                    # Reshape to (N, K). When the body's per-row
                    # log-likelihood has no class dependence (every
                    # parameter is N-shaped or scalar) we tile across
                    # K so the marginalize's scatter accumulator
                    # always receives a (N, K) tensor.
                    if ll.dim() == 1:
                        ll = ll.unsqueeze(-1).expand(-1, _K)
                    elif ll.dim() == 0:
                        ll = ll.reshape(1, 1).expand(_N, _K)
                    return ll

                bound_vars[ll_slot] = None
                steps.append(
                    Let(
                        name=ll_slot,
                        value=reading(
                            _captured_observe,
                            (resp_var, "_x_input", *_arg_names(step_args or ())),
                        ),
                    )
                )
                continue
            if isinstance(step, VectorisedObserveStep):
                # observe r[n] ~ Family(args) for n in N — the batched-
                # likelihood kernel Φ → G_{≤1}(Φ) with score
                # ∏_n p_F(r_obs(n); θ(n, φ)). Realised as a
                # VectorisedObserve wrapping the per-row family;
                # threads through the existing _StepSpec(is_observed=True)
                # path. The response buffer is supplied at runtime
                # via the `observations` dict on the program.
                idx_space = self._resolve_plate_index(
                    step.index_set,
                    f"indexed observe {step.response_var!r}",
                    step.line,
                    step.col,
                )
                # Leading `over` axes beyond the family's event rank
                # index an independent array of observations per row,
                # as they do for a sample.
                array = self._independent_over_axes(
                    step.morphism, step.axes, step.line, step.col
                )
                array_shape = () if array is None else array[0]
                _synth = DrawStep(
                    vars=(step.index_var,),
                    morphism=step.morphism,
                    args=step.args,
                    is_observed=True,
                    axes=None if array is None else array[1],
                    line=step.line,
                    col=step.col,
                )
                # Use the program's declared codomain as a fallback for
                # type inference on inline distributions; the family's
                # codomain is what actually matters. An array's entries
                # are drawn one variate each.
                family, step_args = self._resolve_draw_morphism(
                    _synth,
                    bound_vars,
                    codomain
                    if array is None
                    else Euclidean(name=f"_{step.response_var}", dim=1),
                    plate_rows=idx_space.size,
                )
                # Build a placeholder response of the right shape; the
                # actual values are supplied at fit time via the
                # `observations[response_var]` dict-entry. The buffer
                # only carries shape information here. Discrete
                # codomains (FinSet) carry one integer per row, so
                # the placeholder is (idx_size,); continuous
                # codomains take the codomain's event shape after
                # the row axis.
                resp_shape: tuple[int, ...]
                if isinstance(family.codomain, SetObject):
                    resp_shape = (idx_space.size,)
                elif hasattr(family.codomain, "dim"):
                    d = int(family.codomain.dim)
                    resp_shape = (idx_space.size,) if d == 1 else (idx_space.size, d)
                else:
                    resp_shape = (idx_space.size,) + tuple(family.codomain.shape)
                resp_shape = (resp_shape[0], *array_shape, *resp_shape[1:])
                placeholder = torch.zeros(*resp_shape)
                vec_obs = VectorisedObserve(
                    family, placeholder, array_shape=array_shape
                )
                # The step's response_var is the data column supplied
                # at fit time. We expose it as the bound name so the
                # runtime's observations[response_var] = data flow
                # automatically clamps the placeholder.
                if step.response_var not in bound_vars:
                    bound_vars[step.response_var] = family.codomain
                steps.append(
                    Observe(
                        names=(step.response_var,), morphism=vec_obs, args=step_args
                    )
                )
                continue
            if isinstance(step, GroupedMarginalizeStep):
                # marginalize v — pushforward G(π_{Φ\\C}). Realised as a
                # deterministic let-step that applies log-sum-exp across
                # the class axis of the named variable's per-class score
                # tensor. The runtime convention is that the body
                # populates `env[v]` with a per-class log-likelihood
                # tensor before this step executes.
                pending = self._ungrouped_marg_bodies.get(id(step))
                if pending is not None:
                    self._compile_ungrouped_marginalize(
                        step, pending, steps, bound_vars, codomain
                    )
                    continue
                if step.var_name not in bound_vars:
                    raise CompileError(
                        f"marginalize: variable {step.var_name!r} not bound",
                        step.line,
                        step.col,
                    )
                target_var = step.var_name
                # Grouped form: `over G` (single plate) or
                # `over G * H` (product plate).  Per-observe
                # fibration data lives in ``body_observes``: each
                # entry is ``(ll_slot, fibration_var,
                # fibration_axes)``.  The codegen iterates the
                # entries, builds a parallel list of (ll, idx)
                # pairs from ``env``, and calls the multi-axis
                # form of ``marginalize_grouped``.
                has_grouping = step.over_obj is not None or step.over_objs is not None
                if has_grouping:
                    over_tuple: tuple[str, ...]
                    if step.over_objs is not None:
                        over_tuple = step.over_objs
                    else:
                        assert step.over_obj is not None
                        over_tuple = (step.over_obj,)
                    for over_name in over_tuple:
                        if over_name not in self._objects:
                            raise CompileError(
                                f"grouped marginalize: `over` object "
                                f"{over_name!r} is not a declared object",
                                step.line,
                                step.col,
                            )
                    if step.probs_var is None or step.probs_var not in bound_vars:
                        raise CompileError(
                            f"grouped marginalize: categorical prior "
                            f"{step.probs_var!r} is not bound in program scope",
                            step.line,
                            step.col,
                        )
                    for index_name in step.probs_indices:
                        if index_name not in bound_vars:
                            raise CompileError(
                                f"grouped marginalize: categorical prior "
                                f"{step.probs_var!r} is indexed by "
                                f"{index_name!r}, which is not bound in "
                                "program scope",
                                step.line,
                                step.col,
                            )
                    if not step.body_observes:
                        raise CompileError(
                            "grouped marginalize: the body must contain "
                            "at least one observe step carrying its own "
                            "`via <idx>` clause",
                            step.line,
                            step.col,
                        )
                    is_product = len(over_tuple) > 1
                    # Verify each per-observe fibration's arity and
                    # bound-ness in scope.  Each entry is either
                    # single-fibration (``fibration_var`` set) or
                    # product-fibration (``fibration_axes`` set
                    # with length equal to ``len(over_tuple)``).
                    nested_projections: dict[
                        str, tuple[tuple[int, ...], tuple[int, ...]]
                    ] = {}
                    for entry in step.body_observes:
                        slot = entry.ll_slot
                        fib_var = entry.fibration_var
                        fib_axes = entry.fibration_axes
                        if fib_var is None and fib_axes is None:
                            # Nested-marginalize entry: the inner
                            # block contributes one aggregate per
                            # position of its own grouping plate,
                            # which must be this block's plate or a
                            # product with it as a factor; the
                            # aggregates are summed along the
                            # projection onto this block's plate.
                            inner_groups = entry.inner_groups
                            if inner_groups is None:
                                raise CompileError(
                                    "grouped marginalize: nested block "
                                    f"at slot {slot!r} has no grouping plate",
                                    step.line,
                                    step.col,
                                )
                            inner_sizes = tuple(
                                int(self._objects[name].size) for name in inner_groups
                            )
                            if all(name in inner_groups for name in over_tuple):
                                nested_projections[slot] = (
                                    inner_sizes,
                                    tuple(
                                        inner_groups.index(name) for name in over_tuple
                                    ),
                                )
                                continue
                            inner_extent = 1
                            for size in inner_sizes:
                                inner_extent *= size
                            outer_extent = 1
                            for name in over_tuple:
                                outer_extent *= int(self._objects[name].size)
                            if inner_extent != outer_extent:
                                raise CompileError(
                                    "grouped marginalize: an inner block "
                                    f"grouped over {inner_groups!r} (extent "
                                    f"{inner_extent}) nests in a block grouped "
                                    f"over {over_tuple!r} (extent "
                                    f"{outer_extent}), but the inner plate must "
                                    "be an axis of the outer plate's extent or "
                                    "a product with the outer plate as a factor",
                                    step.line,
                                    step.col,
                                )
                            # An axis of the same extent, identified
                            # with the outer plate position by position.
                            nested_projections[slot] = ((inner_extent,), (0,))
                            continue
                        if is_product:
                            if fib_axes is None or len(fib_axes) != len(over_tuple):
                                raise CompileError(
                                    "grouped marginalize: observe writing "
                                    f"to slot {slot!r} must declare a "
                                    f"`via product(...)` clause of arity "
                                    f"{len(over_tuple)} matching the "
                                    "product grouping plate",
                                    step.line,
                                    step.col,
                                )
                            # Each axis, like a single ``via``, may name a
                            # bound latent or host data supplied at
                            # runtime through the observations dict.
                        else:
                            if fib_axes is not None:
                                raise CompileError(
                                    "grouped marginalize: `via product(...)` "
                                    "on an observe requires the marginalize "
                                    "header to declare a product grouping "
                                    "plate `over G * H * ...`",
                                    step.line,
                                    step.col,
                                )
                            # ``via`` may reference a bound latent OR a
                            # host-data name supplied at runtime through
                            # the observations dict (the per-row
                            # fibration is usually a `LongTensor` of
                            # row->group indices the user owns).
                    group_sizes = tuple(
                        int(self._objects[name].size) for name in over_tuple
                    )
                    num_classes = step.class_size
                    probs_var = step.probs_var
                    probs_indices = step.probs_indices
                    reduction = step.reduction or "logsumexp"
                    parameterization = step.parameterization
                    observe_specs = tuple(
                        (entry.ll_slot, entry.fibration_var, entry.fibration_axes)
                        for entry in step.body_observes
                    )

                    is_nested_inner = (
                        step.body_ll_var is not None
                        and step.body_ll_var != step.var_name
                    )

                    def _marginalize_grouped_callable(
                        env: dict,
                        _specs: tuple[
                            tuple[str, str | None, tuple[str, ...] | None], ...
                        ] = observe_specs,
                        _nested: dict[
                            str, tuple[tuple[int, ...], tuple[int, ...]]
                        ] = nested_projections,
                        _probs: str = probs_var,
                        _probs_indices: tuple[str, ...] = probs_indices,
                        _sizes: tuple[int, ...] = group_sizes,
                        _k: int = num_classes,
                        _reduction: str = reduction,
                        _parameterization: str = parameterization,
                        _product: bool = is_product,
                        _per_group: bool = is_nested_inner,
                    ) -> torch.Tensor:
                        # Collect per-observe (ll, idx) pairs from
                        # env, with the shape contract that each
                        # ll's trailing axis is the class axis.
                        ll_list: list[torch.Tensor] = []
                        idx_list: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
                        for slot, fib_var, fib_axes in _specs:
                            ll = env[slot]
                            if slot in _nested:
                                # The inner block's per-position
                                # aggregates, summed along the
                                # projection onto this block's plate
                                # and shared across this block's
                                # classes (the inner block has
                                # already integrated its own latent).
                                inner_sizes, kept_axes = _nested[slot]
                                ll = _project_nested_aggregates(
                                    ll, inner_sizes, kept_axes, slot
                                )
                                ll = ll.unsqueeze(-1).expand(-1, _k)
                            if ll.shape[-1] != _k:
                                raise ValueError(
                                    f"grouped marginalize: per-row "
                                    f"per-class log-likelihood at slot "
                                    f"{slot!r} must end with the class "
                                    f"axis of size {_k}; got "
                                    f"{tuple(ll.shape)}"
                                )
                            ll_list.append(ll)
                            if fib_axes is not None:
                                idx_tuple = []
                                for axis_name in fib_axes:
                                    idx = _fibration_index(env, axis_name)
                                    if idx.dim() == 2 and idx.shape[-1] == 1:
                                        idx = idx.squeeze(-1)
                                    idx_tuple.append(idx.to(torch.long))
                                idx_list.append(tuple(idx_tuple))
                            elif fib_var is not None:
                                idx = _fibration_index(env, fib_var)
                                if idx.dim() == 2 and idx.shape[-1] == 1:
                                    idx = idx.squeeze(-1)
                                idx_list.append(idx.to(torch.long))
                            else:
                                # Nested-marginalize entry: the
                                # aggregates already sit at this
                                # block's positions, so the
                                # fibration is the identity.
                                positions = torch.arange(
                                    int(ll.shape[0]),
                                    dtype=torch.long,
                                    device=ll.device,
                                )
                                if _product:
                                    idx_list.append(
                                        tuple(torch.unravel_index(positions, _sizes))
                                    )
                                else:
                                    idx_list.append(positions)
                        probs = env[_probs]
                        for index_name in _probs_indices:
                            probs = probs[env[index_name].to(torch.long)]
                        log_prior = (
                            torch.log_softmax(probs, dim=-1)
                            if _parameterization == "logits"
                            else torch.log(probs.clamp_min(1e-38))
                        )
                        # A per-group prior (the categorical parameter tensor
                        # is indexed by the grouping plate, shape
                        # ``(|G|, K)``) denotes a per-row latent: each
                        # response row draws its own class from its
                        # group's prior ``theta[g(row)]``, so the row's
                        # marginal is ``rho_k(log theta[g(row)][k] +
                        # ll[row, k])`` and the block contributes the
                        # sum over rows. A global ``(K,)`` prior instead
                        # denotes one class shared across each group's
                        # rows, scored by scatter-summing the rows'
                        # log-likelihoods into the group before the
                        # reduction.
                        per_group_prior = (
                            not _product
                            and log_prior.dim() == 2
                            and log_prior.shape[0] == _sizes[0]
                        )
                        if per_group_prior:
                            per_group = log_prior.new_zeros((_sizes[0],))
                            total = log_prior.new_zeros(())
                            for ll, idx in zip(ll_list, idx_list):
                                assert isinstance(idx, torch.Tensor)
                                gathered = log_prior[idx]
                                weighted = gathered + ll
                                if _reduction == "logsumexp":
                                    per_row = torch.logsumexp(weighted, dim=-1)
                                elif _reduction == "sum":
                                    per_row = weighted.sum(dim=-1)
                                else:
                                    per_row = weighted.mean(dim=-1)
                                per_group = per_group.index_add(0, idx, per_row)
                                total = total + per_row.sum()
                            if _per_group:
                                return per_group
                            # The block's total is the rows' own sum: the
                            # per-group scatter is what an enclosing block
                            # reads, and summing the rows directly keeps the
                            # float32 accumulation in row order.
                            return total.reshape(1)
                        result = marginalize_grouped(
                            ll_list,
                            idx_list,
                            log_prior,
                            _sizes if _product else _sizes[0],
                            reduction=_reduction,
                            per_group=_per_group,
                        )
                        # An inner block hands its per-position
                        # aggregates to the enclosing block; the
                        # outermost block contributes a scalar,
                        # wrapped in length 1 so the joint
                        # accumulator broadcasts cleanly.
                        if result.dim() == 0:
                            return result.reshape(1)
                        return result

                    # In a nested-marginalize context, the outer
                    # block's _expand_bind_steps re-points
                    # ``body_ll_var`` at the *outer* latent name so
                    # the inner block's reduction output is bound
                    # there (where the outer block's runtime
                    # callable expects to find it). Otherwise we
                    # emit a fresh ``_marg_<var>`` slot.
                    # Outer block: emit a score step (contributes to
                    # log_joint). Inner block (when body_ll_var
                    # re-points at the outer's expected slot): emit a
                    # let so the outer's callable reads the inner's
                    # reduction from env without double-scoring.
                    marg_name = (
                        step.body_ll_var
                        if is_nested_inner
                        else f"_marg_{step.var_name}"
                    )
                    bound_vars[marg_name] = None
                    marginal_reads: list[str] = [probs_var, *probs_indices]
                    for slot, fib_var, fib_axes in observe_specs:
                        marginal_reads.append(slot)
                        if fib_var is not None:
                            marginal_reads.append(fib_var)
                        if fib_axes is not None:
                            marginal_reads.extend(fib_axes)
                    steps.append(
                        (
                            Score(
                                name=marg_name,
                                score=reading(
                                    _marginalize_grouped_callable, marginal_reads
                                ),
                            )
                            if not is_nested_inner
                            else Let(
                                name=marg_name,
                                value=reading(
                                    _marginalize_grouped_callable, marginal_reads
                                ),
                            )
                        )
                    )
                    continue

                raise CompileError(
                    f"marginalize {target_var!r}: an ungrouped marginalize "
                    "block must reach compilation with its body captured "
                    "(see `_expand_bind_steps`); this step carries neither a "
                    "grouping plate nor a stashed body, which the current "
                    "surface does not admit (e.g. an ungrouped marginalize "
                    "inlined through a parametric template)",
                    step.line,
                    step.col,
                )
            if isinstance(step, CallStep):
                steps.append(self._compile_call(step, bound_vars))
                continue
            if isinstance(step, LetStep):
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )
                scan = scan_let_call(step.value)
                if scan is not None and scan.marginal:
                    steps.append(
                        self._compile_marginal_chain(step, scan, decl, bound_vars)
                    )
                    continue
                if scan is not None and scan.step in self._scan_step_declarations:
                    steps.append(self._compile_scan(step, scan, decl, bound_vars))
                    continue
                # Sequences paired position by position are laid end to
                # end along the feature axis, after the position axis,
                # as per-row vectors bundled by a fan are.
                bundle = positions_let_call(step.value)
                if bundle is not None and not all(
                    isinstance(bound_vars.get(name), ContinuousSpace) for name in bundle
                ):
                    raise CompileError(
                        f"let {step.name!r} pairs {', '.join(bundle)} position by "
                        f"position, which are not all sequences of continuous "
                        f"states bound before it",
                        step.line,
                        step.col,
                    )
                if bundle is None:
                    bundle = _draw_bundle(step.value, bound_vars)
                if bundle is not None:
                    steps.append(Let(name=step.name, value=_concatenation(bundle)))
                    bound_vars[step.name] = ProductSpace(
                        components=tuple(
                            cast(ContinuousSpace, bound_vars[name]) for name in bundle
                        )
                    )
                    continue
                if isinstance(step.value, LetExprVar) and step.value.name in bound_vars:
                    # Alias of a variable bound earlier in the body:
                    # the binding copies the slot and inherits its
                    # type. A name that is *not* bound in the body is
                    # a host-data reference, and takes the compiled
                    # path below so that it resolves against the
                    # conditioning data exactly like the same name
                    # inside a compound expression.
                    bound_vars[step.name] = bound_vars[step.value.name]
                    steps.append(Let(name=step.name, value=step.value.name))
                elif isinstance(step.value, LetExprLiteral):
                    bound_vars[step.name] = None
                    steps.append(Let(name=step.name, value=step.value.value))
                else:
                    # Let-expressions inside a program body may
                    # reference compiled deductions by name (for
                    # `parse(D, ...)` calls), so we pass the
                    # compiler's deductions dict as the static
                    # `globals_` environment for variable
                    # resolution.
                    self._validate_let_expr_vars(step.value, bound_vars, step)
                    deductions_globals = self._let_globals()
                    self._reject_unknown_let_calls(
                        step.value,
                        deductions_globals,
                        bound_vars,
                        "let",
                        step.name,
                        step.line,
                        step.col,
                    )
                    reads = self._let_reads(step.value)
                    compiled_fn = reading(
                        _checked_layout(
                            self._locate_let_failure(
                                self._compile_let_expr(
                                    step.value,
                                    globals_=deductions_globals,
                                ),
                                "let",
                                step.name,
                                step.line,
                                step.col,
                            ),
                            reads,
                            self._value_extents_stack[-1]
                            if self._value_extents_stack
                            else {},
                            step.name,
                        ),
                        reads,
                    )
                    bound_vars[step.name] = None
                    let_space = _let_space(
                        step.value,
                        bound_vars,
                        parameter_spaces,
                        self._function_widths(),
                    )
                    if let_space is not None:
                        parameter_spaces[step.name] = let_space
                    steps.append(Let(name=step.name, value=compiled_fn))
                continue
            if isinstance(step, ScoreStep):
                # Score step: bind ``name`` to the value of the
                # expression (so downstream let/score/return can
                # reference it) AND add the value to ``log_joint``
                # via a `_ScoreSpec`. The compiled callable
                # signature is identical to a let binding's, so the
                # runtime tuple uses the ``morph=None, is_score=True``
                # slot the `MonadicProgram` constructor
                # already recognises.
                if step.name in bound_vars:
                    raise CompileError(
                        f"variable {step.name!r} already bound in program",
                        step.line,
                        step.col,
                    )
                self._validate_let_expr_vars(step.value, bound_vars, step)
                deductions_globals = self._let_globals()
                self._reject_unknown_let_calls(
                    step.value,
                    deductions_globals,
                    bound_vars,
                    "score",
                    step.name,
                    step.line,
                    step.col,
                )
                reads = self._let_reads(step.value)
                compiled_fn = reading(
                    _checked_layout(
                        self._locate_let_failure(
                            self._compile_let_expr(
                                step.value,
                                globals_=deductions_globals,
                            ),
                            "score",
                            step.name,
                            step.line,
                            step.col,
                        ),
                        reads,
                        self._value_extents_stack[-1]
                        if self._value_extents_stack
                        else {},
                        step.name,
                    ),
                    reads,
                )
                bound_vars[step.name] = None
                steps.append(Score(name=step.name, score=compiled_fn))
                continue
            draw = step
            for v in draw.vars:
                if v in bound_vars:
                    raise CompileError(
                        f"variable {v!r} already bound in program", draw.line, draw.col
                    )
            # An argument naming nothing bound earlier is host data, as
            # a free name in a let is: the run reads it from the
            # bindings visible where the program runs, else from the
            # conditioning data.
            array = (
                self._independent_over_axes(
                    draw.morphism, draw.axes, draw.line, draw.col
                )
                if isinstance(draw, DrawStep)
                and not draw.is_observed
                and len(draw.vars) == 1
                else None
            )
            if array is not None and isinstance(draw, DrawStep):
                array_shape, event_axes = array
                family, step_args = self._resolve_draw_morphism(
                    draw.with_(axes=event_axes),
                    bound_vars,
                    Euclidean(name=f"_{draw.vars[0]}", dim=1),
                )
                morph = PlateDraw(
                    math.prod(array_shape),
                    family,
                    domain=family.domain,
                    index_shape=array_shape,
                )
            else:
                morph, step_args = self._resolve_draw_morphism(
                    draw, bound_vars, codomain
                )
            if len(draw.vars) == 1:
                bound_vars[draw.vars[0]] = morph.codomain
            elif isinstance(morph, MonadicProgram) and (not morph._return_is_single):
                if len(draw.vars) != len(morph._return_vars):
                    raise CompileError(
                        f"destructuring {len(draw.vars)} vars but sub-program returns {len(morph._return_vars)}",
                        draw.line,
                        draw.col,
                    )
                for v in draw.vars:
                    bound_vars[v] = None
            elif isinstance(morph.codomain, ProductSet):
                if len(draw.vars) != len(morph.codomain.components):
                    raise CompileError(
                        f"destructuring {len(draw.vars)} vars but codomain has {len(morph.codomain.components)} components",
                        draw.line,
                        draw.col,
                    )
                for v, factor in zip(draw.vars, morph.codomain.components):
                    bound_vars[v] = factor
            else:
                raise CompileError(
                    f"cannot destructure non-product codomain {morph.codomain!r}",
                    draw.line,
                    draw.col,
                )
            steps.append(
                (Observe if draw.is_observed else Draw)(
                    names=draw.vars, morphism=morph, args=step_args
                )
            )
        for rv in decl.return_vars:
            if rv not in bound_vars:
                raise CompileError(
                    f"return variable {rv!r} not bound in program", decl.line, decl.col
                )
        self._parameter_space_stack.pop()
        self._value_extents_stack.pop()
        prog = MonadicProgram(
            domain,
            codomain,
            steps,
            decl.return_vars,
            params=decl.params,
            return_labels=decl.return_labels,
            effect_set=(
                get_program_effects(
                    decl.options,
                    line=decl.line,
                    col=decl.col,
                )
                or frozenset(self._infer_effects(expanded_draws) | scan_effects)
            ),
        )
        # A learned function a let or score calls is evaluated through
        # the let globals, not drawn as a step, so the program holds it
        # as a submodule for its weights to reach the optimizer.
        for callee in _let_callees(expanded_draws):
            function = self._morphisms.get(callee)
            if isinstance(function, LearnedFunction) and not function.latent_sites:
                prog.add_module(f"_function_{callee}", function)
        if step_program:
            return prog
        # Posterior-block routing: `[over=M]` programs go to the
        # posterior registry rather than the morphism registry.
        over_model = get_program_over_model(
            decl.options,
            line=decl.line,
            col=decl.col,
        )
        if over_model is not None:
            if not hasattr(self, "_posteriors"):
                self._posteriors = {}
            if over_model not in self._morphisms:
                raise CompileError(
                    f"posterior block '[over={over_model}]' references "
                    f"undefined model {over_model!r}",
                    decl.line,
                    decl.col,
                )
            self._posteriors[decl.name] = prog
        else:
            self._morphisms[decl.name] = prog
        return None

    def _latent_kernel_draws(
        self, steps: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Draw through each latent-weight kernel at its drawn weights.

        A kernel declared ``[weights=latent]`` reads its network's
        weights from the run's sites rather than from its own
        parameters. Each ``sample`` or ``observe`` through one becomes a
        ``let`` that applies the network, at the drawn weights, to the
        step's arguments, followed by the same step drawn from the
        kernel's family with an identity parameter source over that
        network output, so the family reads the parameters the network
        writes exactly as the kernel would.

        Parameters
        ----------
        steps : tuple[ProgramStep, ...]
            A program body, marginalization scopes included.

        Returns
        -------
        tuple[ProgramStep, ...]
            The body with every such step rewritten.

        Raises
        ------
        CompileError
            If a draw through a latent-weight kernel names no argument,
            or an argument that is not a binding.
        """
        rewritten: list[ProgramStep] = []
        for step in steps:
            if isinstance(step, MarginalizeStep):
                rewritten.append(
                    step.with_(scope=self._latent_kernel_draws(step.scope))
                )
                continue
            if not isinstance(step, SampleStep | ObserveStep):
                rewritten.append(step)
                continue
            network = self._latent_networks.get(step.morphism)
            if network is None:
                rewritten.append(step)
                continue
            names = [
                argument.text
                for argument in (step.args or ())
                if isinstance(argument, DrawArgName)
            ]
            if not names or len(names) != len(step.args or ()):
                raise CompileError(
                    f"the latent-weight kernel {step.morphism!r} is drawn from "
                    f"with its inputs named, as in `{step.morphism}(x)`; each "
                    f"argument is a binding or a host datum",
                    step.line,
                    step.col,
                )
            site = step.vars[0] if step.vars else "value"
            raw = f"{step.morphism}__raw_{site}"
            rewritten.append(
                LetStep(
                    name=raw,
                    value=LetExprCall(
                        func=f"{step.morphism}__network",
                        args=tuple(LetExprVar(name=text) for text in names),
                    ),
                    line=step.line,
                    col=step.col,
                )
            )
            rewritten.append(
                step.with_(
                    morphism=f"{step.morphism}__family",
                    args=(DrawArgName(text=raw),),
                )
            )
        return tuple(rewritten)

    def _register_latent_kernel(self, name: str, kernel: ContinuousMorphism) -> None:
        """Split a latent-weight kernel into its network and its family.

        Parameters
        ----------
        name : str
            The kernel's name.
        kernel : ContinuousMorphism
            The compiled kernel, a conditional family over a parameter
            source.

        Raises
        ------
        CompileError
            If the kernel has no parameter source to split off.
        """
        source = getattr(kernel, "param_source", None)
        if not isinstance(source, torch.nn.Module) or not hasattr(source, "param_dim"):
            raise CompileError(
                f"kernel {name!r} declares latent weights, but its family has no "
                f"parameter source to draw them for"
            )
        width = int(cast(ParamSource, source).param_dim)
        raw_space = Euclidean(name=f"{name}__raw", dim=width)
        construct = cast(Callable[..., ContinuousMorphism], type(kernel))
        family = construct(
            raw_space, kernel.codomain, param_source=IdentitySource(width)
        )
        network = LearnedFunction(
            kernel.domain, raw_space, cast(ParamSource, source), latent=name
        )
        self._morphisms[f"{name}__family"] = family
        self._morphisms[f"{name}__network"] = network
        self._latent_networks[name] = network

    def _expand_cell_scans(self, decl: ProgramDecl) -> tuple[ProgramDecl, set[str]]:
        """Expand the steps of a program that run a scan.

        Parameters
        ----------
        decl : ProgramDecl
            The program.

        Returns
        -------
        tuple[ProgramDecl, set[str]]
            The program with each such step rewritten to its step
            program's scan, and the effects the step programs its scans
            run perform. Every step program the expansion declares is
            recorded for the scans that run it.

        Raises
        ------
        CompileError
            If the expansion fails.
        """
        try:
            draws, declared = expand_program_scans(decl, self._module)
        except StepResolutionError as error:
            raise CompileError(error.detail, decl.line, decl.col) from error
        if draws is decl.draws:
            return decl, set()
        self._scan_step_declarations.update(declared)
        effects: set[str] = set()
        for name in _scanned_step_programs(draws):
            effects |= self._scan_step(name).effect_set or set()
        if _marginal_chains(draws):
            # A marginalized chain scores its observations with its
            # states summed out.
            effects |= {"Score", "Marginal"}
        return decl.with_(draws=draws), effects

    def _scan_step(self, name: str) -> MonadicProgram:
        """The compiled step program of a scan.

        Parameters
        ----------
        name : str
            The step program's name.

        Returns
        -------
        MonadicProgram
            The step program, compiled once.
        """
        compiled = self._scan_step_programs.get(name)
        if compiled is None:
            compiled = self._compile_program(
                self._scan_step_declarations[name], step_program=True
            )
            assert compiled is not None
            self._scan_step_programs[name] = compiled
        return compiled

    def _compile_scan(
        self,
        step: LetStep,
        scan: ScanCall,
        decl: ProgramDecl,
        bound_vars: dict[str, AnySpace | None],
    ) -> Scan:
        """Compile ``let h = scan(step, xs)`` over a scan's step program.

        A scan with ``outputs=all`` binds its every state, which carries
        a position axis after the batch; the name is bound to the
        state's space, the space of each entry, as the sequence a later
        scan reads is.

        Parameters
        ----------
        step : LetStep
            The let.
        scan : ScanCall
            The scan: its step program, its sequence, its initial
            state's name when there is one (a binding of the program,
            or a learned initial state's input), and its options.
        decl : ProgramDecl
            The program the let belongs to.
        bound_vars : dict[str, AnySpace | None]
            The names bound so far; gains the let's.

        Returns
        -------
        Scan
            The scan step.

        Raises
        ------
        CompileError
            If the sequence is neither bound nor the program's input.
        """
        step_name, source_name, init = scan.step, scan.sequence, scan.init
        body = self._scan_step(step_name)
        source: str | None
        if source_name in bound_vars:
            source = source_name
        elif decl.params is None and source_name == _input_name(decl):
            source = None
        else:
            raise CompileError(
                f"scan runs over {source_name!r}, which is neither bound nor the "
                f"program's input",
                step.line,
                step.col,
            )
        initial: torch.nn.Parameter | None = None
        initial_from: str | None = None
        if init is not None and init in bound_vars:
            initial_from = init
        elif init is not None and not is_learned_scan_init(init):
            raise CompileError(
                f"the scan starts from {init!r} (init={init}), which the program "
                f"does not bind before the scan",
                step.line,
                step.col,
            )
        elif init is not None:
            initial = self._scan_initial_states.get(init)
            if initial is None:
                width = getattr(body.codomain, "dim")
                # torch's stubs type the constructor's result as the
                # tensor it wraps; it is the parameter itself.
                created = cast(
                    torch.nn.Parameter, torch.nn.Parameter(torch.zeros(int(width)))
                )
                self._scan_initial_states[init] = created
                initial = created
        bound_vars[step.name] = body.codomain
        return Scan(
            name=step.name,
            step=body,
            source=source,
            initial=initial,
            initial_from=initial_from,
            outputs=scan.outputs,
            reverse=scan.reverse,
        )

    def _compile_marginal_chain(
        self,
        step: LetStep,
        scan: ScanCall,
        decl: ProgramDecl,
        bound_vars: dict[str, AnySpace | None],
    ) -> Score:
        """Compile a marginalized chain to a score of its log marginal.

        The chain's cell returns the latent its block marginalizes, so
        every state of the chain is summed out. Within one run, the
        score evaluates the cell at every position ``t`` and every pair
        of a previous state ``i`` and a next state ``j``: the cell's lets
        at the entry and ``i``, the transition's log probability of
        ``j``, and the block's lets and observations at ``j``, giving
        the weights ``W_t[i, j]``. The forward algorithm then runs in
        log space from the first state's distribution ``p``::

            log_filter = log p
            table_t    = logsumexp_i(log_filter[i] + W_t[i, :])
            evidence_t = logsumexp(table_t)
            log_filter = table_t - evidence_t

        and the score is the sum of the evidences, the log marginal
        likelihood of the chain's observations.

        Parameters
        ----------
        step : LetStep
            The let binding the chain.
        scan : ScanCall
            The marginalized scan: the cell, the sequence, and the first
            state's distribution.
        decl : ProgramDecl
            The program the let belongs to.
        bound_vars : dict[str, AnySpace | None]
            The names bound so far; gains the let's.

        Returns
        -------
        Score
            The score step.

        Raises
        ------
        CompileError
            If the cell is not a chain step the forward algorithm sums
            out, its state is not a finite set, the sequence or the
            first state's distribution is not bound, or the transition
            does not name its parameterization.
        """
        cell_decl = next(
            (
                statement
                for statement in self._module.statements
                if isinstance(statement, ProgramDecl) and statement.name == scan.step
            ),
            None,
        )
        if cell_decl is None:
            raise CompileError(
                f"the marginalized chain runs {scan.step!r}, which is not a program",
                step.line,
                step.col,
            )
        try:
            cell = markov_cell(cell_decl)
        except StepResolutionError as error:
            raise CompileError(error.detail, step.line, step.col) from error
        state_space = (
            self._objects.get(cell.state_object.name)
            if isinstance(cell.state_object, TypeName)
            else None
        )
        if not isinstance(state_space, FinSet):
            raise CompileError(
                f"the marginalized chain over {cell.name!r} sums the states of "
                f"an object that is not a declared finite set",
                step.line,
                step.col,
            )
        count = int(state_space.cardinality)
        entry_space: AnySpace | None = None
        if isinstance(cell.entry_object, TypeName):
            entry_space = self._objects.get(cell.entry_object.name) or self._spaces.get(
                cell.entry_object.name
            )
        if entry_space is None:
            raise CompileError(
                f"the cell {cell.name!r} reads entries of an object that is not "
                f"declared",
                step.line,
                step.col,
            )
        source_key: str
        if scan.sequence in bound_vars:
            source_key = scan.sequence
        elif decl.params is None and scan.sequence == _input_name(decl):
            source_key = "_x_input"
        else:
            raise CompileError(
                f"the marginalized chain runs over {scan.sequence!r}, which is "
                f"neither bound nor the program's input",
                step.line,
                step.col,
            )
        if scan.init is None or scan.init not in bound_vars:
            raise CompileError(
                f"the marginalized chain over {cell.name!r} starts from "
                f"{scan.init!r}, which the program does not bind before the scan",
                step.line,
                step.col,
            )
        entry_width = (
            None if isinstance(entry_space, SetObject) else int(_event_dim(entry_space))
        )
        local_bv: dict[str, AnySpace | None] = dict(bound_vars)
        local_bv[cell.entry] = entry_space
        local_bv[cell.previous] = state_space
        local_bv[cell.latent] = None
        globals_ = self._let_globals()
        extents = self._value_extents_stack[-1] if self._value_extents_stack else {}

        def compile_lets(lets: Sequence[LetStep]) -> tuple[tuple[str, Callable], ...]:
            compiled: list[tuple[str, Callable]] = []
            for let in lets:
                self._reject_unknown_let_calls(
                    let.value, globals_, local_bv, "let", let.name, let.line, let.col
                )
                reads = self._let_reads(let.value)
                compiled.append(
                    (
                        let.name,
                        reading(
                            _checked_layout(
                                self._compile_let_expr(let.value, globals_=globals_),
                                reads,
                                extents,
                                let.name,
                            ),
                            reads,
                        ),
                    )
                )
                local_bv[let.name] = None
            return tuple(compiled)

        prefix = compile_lets(cell.prefix)
        transition = cell.block
        parameterization = (
            transition.args[0].parameter
            if transition.args and isinstance(transition.args[0], DrawArgNamed)
            else None
        )
        if (
            transition.args is None
            or len(transition.args) != 1
            or (parameterization not in ("probs", "logits"))
        ):
            raise CompileError(
                f"the cell {cell.name!r} draws its next state {cell.latent!r} from "
                f"a Categorical with one parameter, probs=... or logits=...",
                transition.line,
                transition.col,
            )
        _, transition_args = self._resolve_draw_morphism(
            DrawStep(
                vars=(cell.latent,),
                morphism=transition.morphism,
                args=list(transition.args),
                is_observed=False,
                line=transition.line,
                col=transition.col,
            ),
            local_bv,
            None,
        )
        assert transition_args is not None and len(transition_args) == 1
        transition_arg = transition_args[0]
        body_lets: list[tuple[str, Callable]] = []
        emissions: list[
            tuple[ContinuousMorphism, tuple[StepArgument, ...] | None, str, bool]
        ] = []
        observed: list[str] = []
        reads: set[str] = set(_arg_names(transition_args))
        prefix_reads: dict[str, frozenset[str]] = {}
        for name, fn in prefix:
            declared = reads_of(fn)
            assert declared is not None
            prefix_reads[name] = declared
            reads |= declared
        body_reads: set[str] = set()
        for inner in transition.scope:
            if isinstance(inner, LetStep):
                ((name, fn),) = compile_lets((inner,))
                body_lets.append((name, fn))
                declared = reads_of(fn)
                assert declared is not None
                body_reads |= declared
                continue
            assert isinstance(inner, ObserveStep)
            response = inner.vars[0]
            if response != cell.entry and response in bound_vars:
                raise CompileError(
                    f"the cell {cell.name!r} observes {response!r} at every "
                    f"position, so the program cannot also bind {response!r}",
                    inner.line,
                    inner.col,
                )
            family, arguments = self._resolve_draw_morphism(
                DrawStep(
                    vars=(response,),
                    morphism=inner.morphism,
                    args=inner.args,
                    is_observed=True,
                    line=inner.line,
                    col=inner.col,
                ),
                local_bv,
                None,
            )
            emissions.append(
                (family, arguments, response, _shared_vector_parameter(family))
            )
            if arguments is not None:
                body_reads |= set(_arg_names(arguments))
            if response != cell.entry:
                observed.append(response)
        reads |= body_reads
        # Which of the cell's values vary with the position (they read
        # the entry or an observation) and which with the previous state.
        by_position: set[str] = {cell.entry, *observed}
        by_previous: set[str] = {cell.previous}
        for name, _ in prefix:
            if prefix_reads[name] & by_position:
                by_position.add(name)
            if prefix_reads[name] & by_previous:
                by_previous.add(name)
        transition_varies = bool(set(_arg_names(transition_args)) & by_position)
        emission_reads_previous = bool(body_reads & by_previous)
        local_names = {cell.entry, cell.previous, cell.latent, "_x_input"}
        local_names |= {name for name, _ in prefix}
        local_names |= {name for name, _ in body_lets}
        chain_reads = (reads - local_names) | {source_key, scan.init, *observed}
        states = tuple(torch.tensor(index) for index in range(count))
        positions_reversed = scan.reverse
        init_name = scan.init

        def _log_emissions(env: dict[str, torch.Tensor]) -> torch.Tensor:
            """The block's log weight at each next state, ``(K,)``."""
            scores: list[torch.Tensor] = []
            for following in states:
                atom = dict(env)
                atom[cell.latent] = following
                for name, fn in body_lets:
                    atom[name] = fn(atom)
                score = torch.zeros(())
                for family, arguments, response, shared in emissions:
                    theta = _observation_parameter(atom, arguments, shared)
                    score = score + family.log_prob(theta, atom[response]).sum()
                scores.append(score)
            return torch.stack(scores)

        def _chain_marginal(env: dict[str, torch.Tensor]) -> torch.Tensor:
            sequence = env[source_key]
            entries = (
                sequence.reshape(-1)
                if entry_width is None
                else sequence.reshape(-1, entry_width)
            )
            length = int(entries.shape[0])
            sequences: dict[str, torch.Tensor] = {}
            for name in observed:
                values = env[name]
                if values.dim() > 0 and int(values.shape[0]) != length:
                    # Data every run shares arrive with the run's axis
                    # of one ahead of the positions.
                    values = values.reshape(values.shape[1:])
                if values.dim() == 0 or int(values.shape[0]) != length:
                    raise ValueError(
                        f"the marginalized chain over {cell.name!r} observes "
                        f"{name!r} at each of {length} positions, but its value "
                        f"has shape {tuple(env[name].shape)}"
                    )
                sequences[name] = values
            start = env[init_name].reshape(count)
            log_filter = torch.log(start) - torch.log(start.sum())
            total = torch.zeros((), dtype=log_filter.dtype, device=log_filter.device)
            order = range(length - 1, -1, -1) if positions_reversed else range(length)
            fixed_transition: torch.Tensor | None = None
            for position in order:
                here = dict(env)
                here[cell.entry] = entries[position]
                for name, values in sequences.items():
                    here[name] = values[position]
                previous_envs: list[dict[str, torch.Tensor]] = []
                log_rows: list[torch.Tensor] = []
                for previous in states:
                    row_env = dict(here)
                    row_env[cell.previous] = previous
                    for name, fn in prefix:
                        row_env[name] = fn(row_env)
                    previous_envs.append(row_env)
                    if fixed_transition is None:
                        weights = _lookup_arg(row_env, transition_arg).reshape(count)
                        log_rows.append(
                            torch.log_softmax(weights, dim=-1)
                            if parameterization == "logits"
                            else torch.log(weights) - torch.log(weights.sum())
                        )
                if fixed_transition is None:
                    log_transition = torch.stack(log_rows)
                    if not transition_varies:
                        fixed_transition = log_transition
                else:
                    log_transition = fixed_transition
                if emission_reads_previous:
                    weights_at = torch.stack(
                        [_log_emissions(row_env) for row_env in previous_envs]
                    )
                else:
                    weights_at = _log_emissions(previous_envs[0]).unsqueeze(0)
                table = torch.logsumexp(
                    log_filter.unsqueeze(-1) + log_transition + weights_at, dim=0
                )
                evidence = torch.logsumexp(table, dim=0)
                total = total + evidence
                log_filter = table - evidence
            return total.reshape(1)

        bound_vars[step.name] = None
        return Score(name=step.name, score=reading(_chain_marginal, chain_reads))

    def _compile_call(
        self, step: CallStep, bound_vars: dict[str, AnySpace | None]
    ) -> Call:
        """Compile ``let y <- p(a, b)`` of a program into a call run in place.

        Parameters
        ----------
        step : CallStep
            The call.
        bound_vars : dict[str, AnySpace | None]
            The names bound so far; gains the call's.

        Returns
        -------
        Call
            The call step.

        Raises
        ------
        CompileError
            If the callee is not a compiled program or an argument is
            not a bound name.
        """
        callee = self._morphisms.get(step.call.callee)
        if not isinstance(callee, MonadicProgram):
            raise CompileError(
                f"`let {step.name} <- {step.call.callee}(...)` calls a named "
                "computation, which only the QIEC route runs; execute the program "
                "through its checked module rather than the runtime compiler",
                step.line,
                step.col,
            )
        arguments: list[str] = []
        for argument in step.call.arguments:
            if not isinstance(argument, LetExprVar) or argument.name not in bound_vars:
                raise CompileError(
                    f"`let {step.name} <- {step.call.callee}(...)` passes an "
                    f"argument that is not a name bound earlier in the program",
                    step.line,
                    step.col,
                )
            arguments.append(argument.name)
        if step.name in bound_vars:
            raise CompileError(
                f"variable {step.name!r} already bound in program",
                step.line,
                step.col,
            )
        bound_vars[step.name] = callee.codomain
        try:
            return Call(name=step.name, program=callee, args=tuple(arguments))
        except ValueError as error:
            raise CompileError(str(error), step.line, step.col) from error

    def _bind_checked_program(
        self,
        decl: ProgramDecl,
        domain: AnySpace,
        codomain: AnySpace,
    ) -> None:
        """Bind a program to its checked computation.

        Parameters
        ----------
        decl : ProgramDecl
            The program declaration.
        domain : AnySpace
            Its resolved domain.
        codomain : AnySpace
            Its resolved codomain.

        Raises
        ------
        CompileError
            If the module's checked projection has no entry for the
            program, because its elaboration fell in a gap.
        """
        module = self.qiec_module
        if module is None or module.gap:
            gap = (
                module.gap if module is not None else "the source has no checked module"
            )
            raise CompileError(
                f"program {decl.name!r} calls a named computation and must run "
                f"as its checked computation, which is unavailable: {gap}",
                decl.line,
                decl.col,
            )
        if all(entry.name != decl.name for entry in module.entries):
            raise CompileError(
                f"program {decl.name!r} has no entry point in the checked module",
                decl.line,
                decl.col,
            )
        self._morphisms[decl.name] = CheckedProgram(
            domain,
            codomain,
            module,
            decl.name,
            refusal=self._call_refusals.get(decl.name),
        )

    def _compile_ungrouped_marginalize(
        self,
        step: GroupedMarginalizeStep,
        pending: tuple[
            str,
            str,
            tuple[DrawArgName | DrawArgIndex | str, ...],
            tuple[ProgramStep, ...],
        ],
        steps: list[Step],
        bound_vars: dict[str, AnySpace | None],
        codomain: SetObject | ContinuousSpace | None,
    ) -> None:
        """Compile an ungrouped ``marginalize`` block into one score step.

        The block integrates the latent out analytically: for each
        value in the prior's finite support, the body's deterministic
        `let` steps are re-evaluated with the latent bound to that
        value and the terminal ``observe`` is scored against the
        clamped response, producing a per-row per-class log-likelihood
        ``(N, K)`` tensor. Adding the per-class log-prior and reducing
        over the class axis (log-sum-exp by default) yields each row's
        marginal log-density; their sum is the block's single
        contribution to the joint. The latent draw and observe stay
        live (so a forward trace still samples them) but are wrapped in
        a [`MarginalizedFactor`][quivers.continuous.morphisms.MarginalizedFactor]
        so neither leaks its raw per-draw factor into the accumulator.

        Supported priors are the finite-support discrete families the
        surface admits at a marginalize head: `Categorical` (support
        ``0 .. K-1`` with per-class prior ``probs[..., k]``) and the
        Bernoulli relaxation family (hard support ``{0, 1}`` with prior
        ``[1 - p, p]``). Any other family raises, since its support is
        not enumerable in closed form here.
        """
        latent_name, prior_family, prior_args, body_steps = pending

        # Split the body into its deterministic `let` prefix and the
        # single terminal `observe`.
        local_bv: dict[str, AnySpace | None] = dict(bound_vars)
        local_bv[latent_name] = None
        # At each atom the latent is one value, so a let over it is typed
        # as its operands give and the observation reads it at that width.
        parameter_spaces = (
            self._parameter_space_stack[-1] if self._parameter_space_stack else {}
        )
        parameter_spaces[latent_name] = Euclidean(name=f"_{latent_name}", dim=1)
        deductions_globals = self._let_globals()
        body_lets: list[tuple[str, Callable]] = []
        observe_step: VectorisedObserveStep | DrawStep | None = None
        for bstep in body_steps:
            if isinstance(bstep, LetStep):
                self._reject_unknown_let_calls(
                    bstep.value,
                    deductions_globals,
                    local_bv,
                    "let",
                    bstep.name,
                    bstep.line,
                    bstep.col,
                )
                body_reads = self._let_reads(bstep.value)
                fn = reading(
                    _checked_layout(
                        self._compile_let_expr(
                            bstep.value, globals_=deductions_globals
                        ),
                        body_reads,
                        (
                            self._value_extents_stack[-1]
                            if self._value_extents_stack
                            else {}
                        ),
                        bstep.name,
                    ),
                    body_reads,
                )
                body_lets.append((bstep.name, fn))
                local_bv[bstep.name] = None
                let_space = _let_space(
                    bstep.value, local_bv, parameter_spaces, self._function_widths()
                )
                if let_space is not None:
                    parameter_spaces[bstep.name] = let_space
            elif isinstance(bstep, VectorisedObserveStep):
                if observe_step is not None:
                    raise CompileError(
                        f"marginalize {latent_name!r}: an ungrouped "
                        "marginalize body admits exactly one observe step",
                        step.line,
                        step.col,
                    )
                observe_step = bstep
            elif isinstance(bstep, DrawStep) and bstep.is_observed:
                if observe_step is not None:
                    raise CompileError(
                        f"marginalize {latent_name!r}: an ungrouped "
                        "marginalize body admits exactly one observe step",
                        step.line,
                        step.col,
                    )
                observe_step = bstep
            else:
                raise CompileError(
                    f"marginalize {latent_name!r}: an ungrouped marginalize "
                    f"body admits only `let` bindings and a single `observe` "
                    f"step; got {type(bstep).__name__}",
                    step.line,
                    step.col,
                )

        # Resolve the terminal observe's likelihood family and its
        # env-resolved argument references. A body with no observe
        # (the latent gates nothing observed) contributes only its
        # prior: the score reduces the log-prior over the support with
        # a zero likelihood.
        obs_family: ContinuousMorphism | None = None
        obs_args: tuple | None = None
        response_var: str | None = None
        if observe_step is not None:
            if isinstance(observe_step, VectorisedObserveStep):
                obs_synth = DrawStep(
                    vars=(observe_step.index_var,),
                    morphism=observe_step.morphism,
                    args=observe_step.args,
                    is_observed=True,
                    line=observe_step.line,
                    col=observe_step.col,
                )
                response_var = observe_step.response_var
            else:
                obs_synth = observe_step
                response_var = observe_step.vars[0]
            obs_family, obs_args = self._resolve_draw_morphism(
                obs_synth, local_bv, codomain
            )

        # Resolve the prior family so the latent keeps a live, but
        # score-suppressed, draw: a forward trace still samples the
        # coordinate, while the block's integrated score step (not the
        # draw) carries the prior's density into the joint.
        prior_synth = DrawStep(
            vars=(latent_name,),
            morphism=prior_family,
            args=list(prior_args),
            is_observed=False,
            line=step.line,
            col=step.col,
        )
        prior_morph, prior_step_args = self._resolve_draw_morphism(
            prior_synth, local_bv, codomain
        )
        if prior_step_args is None or len(prior_step_args) != 1:
            raise CompileError(
                f"marginalize {latent_name!r}: the prior family "
                f"{prior_family!r} must carry exactly one named parameter "
                f"tensor argument",
                step.line,
                step.col,
            )
        prior_arg = prior_step_args[0]
        prior_parameterization = getattr(prior_args[0], "parameter", None)
        if prior_family == "Categorical" and prior_parameterization not in (
            "probs",
            "logits",
        ):
            raise CompileError(
                "Categorical requires exactly one explicit parameterization; "
                "use probs=... or logits=...",
                step.line,
                step.col,
            )

        # A single shared vector-valued likelihood parameter (e.g. a
        # `Categorical` row ``emission_rows[k]`` of shape ``(K_cat,)``)
        # must present a leading batch axis so the inline family reads
        # it as one distribution rather than ``K_cat`` scalar batches;
        # per-row scalar parameters (e.g. a Poisson rate of shape
        # ``(N,)``) need no such reshape. A vector-valued parameter is
        # the single ``var`` slot whose declared width exceeds one.
        shared_vector_theta = _shared_vector_parameter(obs_family)

        _CATEGORICAL = prior_family == "Categorical"
        _BERNOULLI = prior_family in (
            "Bernoulli",
            "ContinuousBernoulli",
            "RelaxedBernoulli",
        )
        if not (_CATEGORICAL or _BERNOULLI):
            raise CompileError(
                f"marginalize {latent_name!r}: prior family {prior_family!r} "
                "has no closed-form finite support for the ungrouped "
                "marginalize path; supported priors are Categorical and the "
                "Bernoulli relaxation family",
                step.line,
                step.col,
            )
        reduction = step.reduction or "logsumexp"

        def _ungrouped_marginal(
            env: dict[str, torch.Tensor],
            _latent: str = latent_name,
            _lets: tuple[tuple[str, Callable], ...] = tuple(body_lets),
            _obs_family: ContinuousMorphism | None = obs_family,
            _obs_args: tuple | None = obs_args,
            _response: str | None = response_var,
            _prior_arg: StepArgument = prior_arg,
            _categorical: bool = _CATEGORICAL,
            _parameterization: str | None = prior_parameterization,
            _reduction: str = reduction,
            _shared_vector: bool = shared_vector_theta,
        ) -> torch.Tensor:
            probs = _lookup_arg(env, _prior_arg)
            if _categorical:
                class_count = int(probs.shape[-1])
                class_values: list[torch.Tensor] = [
                    torch.tensor(k, dtype=torch.long, device=probs.device)
                    for k in range(class_count)
                ]
                log_prior = (
                    torch.log_softmax(probs, dim=-1)
                    if _parameterization == "logits"
                    else torch.log(probs.clamp_min(1e-38))
                )
            else:
                class_values = [
                    torch.zeros((), dtype=probs.dtype, device=probs.device),
                    torch.ones((), dtype=probs.dtype, device=probs.device),
                ]
                log_prior = _binary_log_prior(probs, _parameterization)
            if _obs_family is None or _response is None:
                # No observed likelihood gates the latent: the block
                # contributes only the reduced log-prior over the
                # support (a zero per-class log-likelihood).
                weighted = log_prior
            else:
                response = env[_response]
                ll_cols: list[torch.Tensor] = []
                for value in class_values:
                    local = dict(env)
                    local[_latent] = value
                    for name, fn in _lets:
                        local[name] = fn(local)
                    theta = _observation_parameter(local, _obs_args, _shared_vector)
                    ll_cols.append(_obs_family.log_prob(theta, response))
                per_class = torch.stack(ll_cols, dim=-1)
                if log_prior.dim() < per_class.dim():
                    # One latent for the whole body. The block
                    # declares no index and no grouping plate, so the
                    # rows of the enclosed observe are conditioned on
                    # a single draw and the pushforward reduces the
                    # body's *accumulated* log-likelihood
                    # (`docs/semantics/programs.md` §2.6), which is
                    # `aggr_k [log pi(k) + sum_n l(n, k)]`. Reducing
                    # per row and summing would instead give each row
                    # its own draw, which is what an indexed
                    # `marginalize c : A` or a grouping `over =`
                    # clause asks for.
                    row_axes = tuple(range(per_class.dim() - 1))
                    weighted = log_prior + per_class.sum(dim=row_axes)
                else:
                    # A per-row prior carries the rows' own latents,
                    # so each row reduces on its own.
                    weighted = log_prior + per_class
            if _reduction == "logsumexp":
                per_row = torch.logsumexp(weighted, dim=-1)
            elif _reduction == "sum":
                per_row = weighted.sum(dim=-1)
            else:
                per_row = weighted.mean(dim=-1)
            return per_row.sum().reshape(1)

        # Emit the block's runtime steps in order: the latent
        # coordinate, the body's deterministic lets, the score-
        # suppressed observe, then the integrated marginal score.
        #
        # The latent is bound as a deterministic let whose callable
        # draws from the prior. A forward trace still produces a
        # concrete coordinate (so the terminal observe can sample a
        # response and synthetic-data generation stays intact), but the
        # site reads as deterministic rather than as a free latent: its
        # density is carried by the marginal score, and the reference
        # probe must not flag it as an unclamped, resampled latent. The
        # observe is wrapped in a `MarginalizedFactor` so its clamped /
        # sampled response contributes nothing to the joint either.
        def _latent_prior_sample(
            env: dict[str, torch.Tensor],
            _prior: ContinuousMorphism = prior_morph,
            _arg: StepArgument = prior_arg,
        ) -> torch.Tensor:
            return _prior.rsample(_lookup_arg(env, _arg))

        bound_vars[latent_name] = None
        steps.append(
            Let(
                name=latent_name,
                value=reading(_latent_prior_sample, _arg_names((prior_arg,))),
            )
        )
        marginal_reads: set[str] = {*_arg_names((prior_arg,)), "_x_input"}
        for name, fn in body_lets:
            bound_vars[name] = None
            steps.append(Let(name=name, value=fn))
            declared = reads_of(fn)
            assert declared is not None
            marginal_reads |= declared
        if obs_family is not None and response_var is not None:
            bound_vars[response_var] = None
            steps.append(
                Observe(
                    names=(response_var,),
                    morphism=MarginalizedFactor(obs_family),
                    args=obs_args,
                )
            )
            marginal_reads.add(response_var)
            if obs_args is not None:
                marginal_reads |= set(_arg_names(obs_args))
        marg_name = f"_marg_{latent_name}"
        bound_vars[marg_name] = None
        steps.append(
            Score(name=marg_name, score=reading(_ungrouped_marginal, marginal_reads))
        )

    def _resolve_draw_morphism(
        self,
        draw,
        bound_vars: dict[str, AnySpace | None],
        program_codomain: SetObject | ContinuousSpace | None,
        *,
        plate_rows: int | None = None,
    ) -> tuple:
        """Resolve a draw step's morphism, handling both named morphisms
        and inline distribution families.

        Parameters
        ----------
        draw : DrawStep
            The draw step to resolve.
        bound_vars : dict
            Currently bound variable types.
        program_codomain : object
            The program's declared codomain (for type inference).
        plate_rows : int | None
            The rows of the plate an indexed observe scores the family
            over, one family per row; ``None`` for a draw of its own.
            A vector over the plate's rows filling a rank-0 parameter
            holds one entry per row, which the family reads at its row.

        Returns
        -------
        tuple of (morphism, step_args)
            The compiled morphism and the variable-only args for
            the step spec (None = use program input).
        """
        if draw.morphism in self._morphisms:
            morph = self._morphisms[draw.morphism]
            if isinstance(morph, LearnedFunction):
                raise CompileError(
                    learned_function_draw_message(draw.morphism), draw.line, draw.col
                )
            if isinstance(morph, torch.nn.Module) and any(
                isinstance(module, AttentionSource) for module in morph.modules()
            ):
                raise CompileError(
                    attention_draw_message(draw.morphism), draw.line, draw.col
                )
            if draw.args is not None:
                for a in draw.args:
                    # A named morphism consumes only identifier
                    # references: bare identifiers (`DrawArgName`),
                    # bracket-indexed references (`DrawArgIndex`),
                    # or legacy bare `str` values from compiler-
                    # synthesized steps. Numeric literals, nested
                    # distribution calls, and list literals are
                    # rejected.
                    if isinstance(a, (DrawArgName, DrawArgIndex)) or isinstance(a, str):
                        continue
                    raise CompileError(
                        f"literal argument {a!r} not allowed for "
                        f"named morphism {draw.morphism!r}",
                        draw.line,
                        draw.col,
                    )
            # A bracket-indexed reference becomes an `Indexed` step
            # argument, which the program gathers through at runtime.
            step_args: tuple[StepArgument, ...] | None
            if draw.args is None:
                step_args = None
            else:
                converted: list[StepArgument] = []
                for a in draw.args:
                    if isinstance(a, DrawArgName):
                        converted.append(a.text)
                    elif isinstance(a, DrawArgIndex):
                        converted.append(Indexed(name=a.name, indices=a.indices))
                    else:
                        converted.append(str(a))
                step_args = tuple(converted)
            return (morph, step_args)
        param_names = get_inline_param_names(draw.morphism)
        if param_names is not None:
            if draw.args is None:
                raise CompileError(
                    f"inline distribution {draw.morphism!r} requires arguments (e.g. {draw.morphism}(...))",
                    draw.line,
                    draw.col,
                )
            axes_override = self._axes_codomain(getattr(draw, "axes", None))
            try:
                inline_codomain = self._infer_inline_codomain(
                    draw.morphism,
                    draw.args,
                    draw.vars,
                    program_codomain if axes_override is None else axes_override,
                    event_axis=axes_override,
                    line=draw.line,
                    col=draw.col,
                )
                parameter_spaces = (
                    self._parameter_space_stack[-1]
                    if self._parameter_space_stack
                    else {}
                )
                value_extents = (
                    self._value_extents_stack[-1] if self._value_extents_stack else {}
                )
                variable_types = {
                    **{
                        name: space
                        for name, space in parameter_spaces.items()
                        if bound_vars.get(name) is None
                    },
                    **{k: v for k, v in bound_vars.items() if v is not None},
                }
                if plate_rows is not None:
                    per_row = _per_row_parameters(
                        draw.morphism, draw.args, param_names, value_extents, plate_rows
                    )
                    variable_types = {
                        **variable_types,
                        **{
                            name: Euclidean(
                                name=f"_{name}", dim=plate_rows, plate_rows=plate_rows
                            )
                            for name in per_row
                        },
                    }
                    value_extents = {
                        name: extents
                        for name, extents in value_extents.items()
                        if name not in per_row
                    }
                morph, var_args = make_inline_distribution(
                    draw.morphism,
                    draw.args,
                    inline_codomain,
                    value_extents=value_extents,
                    variable_types=variable_types,
                )
                broadcast = self._broadcast_codomain(draw, morph, axes_override)
                if broadcast is not None:
                    morph, var_args = make_inline_distribution(
                        draw.morphism,
                        draw.args,
                        broadcast,
                        value_extents=value_extents,
                        variable_types=variable_types,
                    )
            except UnsizedParameterError as exc:
                # An inline family the checked program cannot size is
                # refused at the draw; when the checked program itself
                # fails to lower, its located diagnostic says why first.
                _ = self.qiec_module
                raise CompileError(str(exc), draw.line, draw.col) from exc
            except ValueError:
                # When the checked program fails to lower, its located
                # diagnostic says why first.
                _ = self.qiec_module
                raise
            except TypeError as exc:
                # A family constructor rejecting the call site's
                # argument list (too few / too many parameters) is a
                # user error in the QVR source, so it is reported
                # against the draw step rather than escaping as a raw
                # ``TypeError`` from the builder's own signature. The
                # declared parameter list is appended only when the
                # call site's arity disagrees with it, since several
                # families (Dirichlet's flattened concentration
                # vector) legitimately take a different count.
                detail = f"call to {draw.morphism!r} failed: {exc}"
                if len(draw.args) != len(param_names):
                    detail += (
                        f"; {draw.morphism} takes {len(param_names)} "
                        f"argument(s) ({', '.join(param_names)}), and "
                        f"{len(draw.args)} were given"
                    )
                raise CompileError(detail, draw.line, draw.col) from exc
            return (morph, var_args)
        registry = _get_family_registry()
        if draw.morphism in registry:
            raise CompileError(
                f"distribution family {draw.morphism!r} is not supported as an inline distribution; declare it as a continuous morphism instead",
                draw.line,
                draw.col,
            )
        raise CompileError(
            f"undefined morphism or distribution family {draw.morphism!r}",
            draw.line,
            draw.col,
        )

    @staticmethod
    def _broadcast_codomain(
        draw: DrawStep,
        morph: ContinuousMorphism,
        axes_override: AnySpace | None,
    ) -> AnySpace | None:
        """The codomain of a scalar family drawn at vector parameters.

        A scalar family whose parameters are vector-valued bindings, with
        no axis-role clause, draws one independent variate per entry of
        its parameters: its value has their broadcast extent, as the
        checked elaboration types the site.

        Parameters
        ----------
        draw : DrawStep
            The draw.
        morph : ContinuousMorphism
            The family built at the draw's scalar codomain.
        axes_override : AnySpace | None
            The space an axis-role clause names, if any.

        Returns
        -------
        AnySpace | None
            The codomain widened to the parameters' extent, or ``None``
            when every parameter is one value per row.

        Raises
        ------
        CompileError
            If two parameters have different extents of more than one
            entry, which do not broadcast.
        """
        record = FAMILIES.get(draw.morphism)
        spec = getattr(morph, "_param_spec", None)
        codomain = morph.codomain
        if (
            record is None
            or record.event_rank != 0
            or axes_override is not None
            or spec is None
            or not isinstance(codomain, Euclidean | PositiveReals)
            or codomain.dim != 1
        ):
            return None
        extents = list(
            dict.fromkeys(
                int(w) for kind, w in spec if kind == "var" and w is not None and w > 1
            )
        )
        if not extents:
            return None
        if len(extents) > 1:
            raise CompileError(
                f"{draw.morphism} draw of {draw.vars[0]!r} reads parameters of "
                f"{' and '.join(map(str, extents))} entries, which do not "
                f"broadcast to one extent",
                draw.line,
                draw.col,
            )
        return codomain.with_(dim=extents[0])

    def _host_data_spaces(self, program: str) -> dict[str, AnySpace]:
        """The spaces the host data a program reads are checked at.

        Parameters
        ----------
        program : str
            The program's name.

        Returns
        -------
        dict[str, AnySpace]
            Each real-valued host datum of the program's checked entry
            as a Euclidean space: one coordinate for a scalar, and for a
            tensor its entries, read as rows of the plate its checked
            axes name when it ranges over one. Empty when the module has
            no checked entry for the program, or when the module does not
            lower: its diagnostic is then raised where `compile` reads
            the module, after the declarations' own diagnostics.
        """
        try:
            module = self.qiec_module
        except QiecDiagnosticError:
            return {}
        if module is None:
            return {}
        entry = next((e for e in module.entries if e.name == program), None)
        if entry is None:
            return {}
        plates = {axis.name for site in entry.sites for axis in site.batch}
        spaces: dict[str, AnySpace] = {}
        for parameter in entry.parameters:
            if parameter.role != "data":
                continue
            # A datum ranges over a plate when it fills a plated draw's
            # parameter one row at a time, or when it is a vector a let
            # reads along a plate's rows; a tensor over its own axes,
            # such as a Cholesky factor, is one value.
            space = _checked_real_space(
                parameter.name,
                parameter.type,
                parameter.per_row
                or (len(parameter.axes) == 1 and parameter.axes[0] in plates),
            )
            if space is not None:
                spaces[parameter.name] = space
        return spaces

    def _checked_value_extents(self, program: str) -> dict[str, tuple[int, ...]]:
        """The extents of the values a program binds, as its checked body types them.

        A gather ``mu[z]`` passed to a family reads, per index, the
        value left after the gathered axes; the extents say whether that
        is one entry of a vector or one row of a matrix, which a flat
        parameter space does not.

        Parameters
        ----------
        program : str
            The program's name.

        Returns
        -------
        dict[str, tuple[int, ...]]
            Each host datum's, site's, and let's extents by name. Empty
            when the module has no checked computation for the program,
            or when the module does not lower: its diagnostic is then
            raised where `compile` reads the module.
        """
        try:
            module = self.qiec_module
        except QiecDiagnosticError:
            return {}
        if module is None:
            return {}
        entry = next((e for e in module.entries if e.name == program), None)
        if entry is None:
            return {}
        computation = next(
            (c for c in module.computations if c.id == entry.computation), None
        )
        if computation is None:
            return {}
        return _checked_local_extents(computation.parameters, computation.body)

    def _annotation_is_event_axis(
        self,
        step: PlateDrawStep,
        index: SetObject,
        bound_vars: Mapping[str, AnySpace | None],
    ) -> str | None:
        """The object an index annotation names as a vector family's event axis.

        A vector family's draw annotated ``w : K`` with no axis-role
        clause is one variate over ``K`` unless its arguments fix an
        event of another size, as the checked elaboration reads it:
        ``Dirichlet(concentration=1.0)`` annotated ``: K`` is one point
        of the ``K``-simplex, while ``Dirichlet(concentration=[1.0, 2.0,
        3.0])`` annotated ``: Item`` is one three-simplex point per item.
        An argument fixes the event when it is a list, whose length is
        the event's size, or a name bound to a vector.

        Parameters
        ----------
        step : PlateDrawStep
            The annotated draw.
        index : SetObject
            The annotation's object.
        bound_vars : Mapping[str, AnySpace | None]
            The spaces of the names bound so far.

        Returns
        -------
        str | None
            The annotation's object name when it is the event axis;
            ``None`` when it is the plate, when the family has no event
            axis of rank one, or when the draw states its axes.
        """
        record = FAMILIES.get(step.morphism)
        if (
            record is None
            or record.event_rank != 1
            or get_inline_param_names(step.morphism) is None
            or getattr(step, "axes", None) is not None
            or not isinstance(step.index, TypeName)
        ):
            return None
        sizes: set[int] = set()
        for argument in step.args or ():
            value = argument.value if isinstance(argument, DrawArgNamed) else argument
            if isinstance(value, DrawArgList) and value.items:
                sizes.add(len(value.items))
            elif isinstance(value, DrawArgName):
                space = bound_vars.get(value.text)
                if isinstance(space, Euclidean) and space.plate_rows is None:
                    if space.dim > 1:
                        sizes.add(space.dim)
                elif isinstance(space, ContinuousSpace) and not isinstance(
                    space, Euclidean
                ):
                    dim = getattr(space, "dim", None)
                    if isinstance(dim, int) and dim > 1:
                        sizes.add(dim)
        if sizes and sizes != {int(index.size)}:
            return None
        return step.index.name

    def _independent_over_axes(
        self, family: str, axes: AxisSpec | None, line: int, col: int
    ) -> tuple[tuple[int, ...], AxisSpec | None] | None:
        """Split an ``[over=...]`` clause into independent axes and event axes.

        A family of event rank ``r`` drawn ``[over=[X1, ..., Xn]]`` with
        ``n > r`` draws an independent array over the leading
        ``n - r`` axes of one variate on the trailing ``r``: a
        ``Normal`` over ``[B, C]`` is a ``(|B|, |C|)`` array of
        independent normals, a ``Dirichlet`` over ``[B, K]`` one
        ``K``-simplex point per element of ``B``. The value holds every
        listed axis in order, after the plate of an index annotation.

        Parameters
        ----------
        family : str
            The family the step draws from.
        axes : AxisSpec | None
            The step's axis-role clause.
        line : int
            The step's source line, for diagnostics.
        col : int
            The step's source column, for diagnostics.

        Returns
        -------
        tuple[tuple[int, ...], AxisSpec | None] | None
            The extents of the independent axes and the clause naming
            the event axes alone (``None`` for a scalar family), or
            ``None`` when the clause names no independent axis beyond
            the single axis of a scalar family, which the family's own
            codomain carries.

        Raises
        ------
        CompileError
            If a matrix-valued family names independent axes, or an
            axis names no finite object.
        """
        record = FAMILIES.get(family)
        if axes is None or record is None or get_inline_param_names(family) is None:
            return None
        rank = record.event_rank
        over = tuple(axes.over)
        independent = over[: max(0, len(over) - rank)]
        if not independent or (rank == 0 and len(over) == 1):
            return None
        if rank > 1 or self._matrix_family_spec(family) is not None:
            raise CompileError(
                f"matrix-valued family {family!r} draws one variate over its "
                f"event axes and takes no further independent axes, but "
                f"`over` lists {len(over)}: {', '.join(over)}",
                line,
                col,
            )
        sizes: list[int] = []
        for name in independent:
            obj = self._objects.get(name)
            if not isinstance(obj, SetObject):
                raise CompileError(
                    f"`over` axis {name!r} of a {family} draw must name a "
                    f"finite object to index an independent array",
                    line,
                    col,
                )
            sizes.append(int(obj.size))
        event = over[len(independent) :]
        return tuple(sizes), (axes.with_(over=event, iid_over=()) if event else None)

    def _axes_codomain(self, axes):
        """Translate an `AxisSpec`'s ``over`` clause into a
        codomain object that ``_infer_inline_codomain`` can consume.

        ``over=X`` -> the declared object ``X`` (so inline distributions
        whose event size depends on the program codomain pick up the
        named axis instead of the program's own codomain).
        ``over=[X, Y]`` -> a `ProductSet` / `ProductSpace`
        of the named axes.
        """
        if axes is None or not axes.over:
            return None
        comps = []
        for name in axes.over:
            if name in self._objects:
                comps.append(self._objects[name])
            elif name in self._spaces:
                comps.append(self._spaces[name])
            else:
                return None
        if len(comps) == 1:
            return comps[0]
        if any(isinstance(c, ContinuousSpace) for c in comps):
            return ProductSpace(components=tuple(comps))
        return ProductSet(components=tuple(comps))

    @staticmethod
    def _as_float(arg: object) -> float | None:
        """Extract a numeric literal from a draw-step argument. Handles
        both legacy bare `int` / `float` values and the tagged
        [`DrawArgScalar`][quivers.dsl.ast_nodes.DrawArgScalar]
        variant. Returns ``None`` for identifier references
        (`DrawArgName`, bare `str`) and nested distribution calls.
        """
        if isinstance(arg, (int, float)) and not isinstance(arg, bool):
            return float(arg)
        if isinstance(arg, DrawArgNamed):
            return _ProgramsMixin._as_float(arg.value)
        if isinstance(arg, DrawArgScalar):
            return float(arg.value)
        return None

    def _resolve_plate_index(
        self,
        texpr: ObjectExpr,
        surface: str,
        line: int,
        col: int,
    ) -> SetObject:
        """Resolve an index annotation that has to name a plate extent.

        A plate ranges over the elements of a finite set, so the
        annotation in ``v : A <- F(...)`` (and in the indexed-observe
        and grouped-observe forms) must denote a `SetObject`.
        A continuous space has no elements to enumerate, so it is
        rejected here with the offending annotation named, rather than
        reaching the plate constructor and failing on a missing
        ``size`` field.
        """
        space = self._resolve_any_space(texpr)
        if isinstance(space, SetObject):
            return space
        raise CompileError(
            f"{surface}: the plate index {self._object_expr_text(texpr)!r} "
            f"resolves to the continuous space {space}, but a plate index "
            f"must be discrete; annotate the draw with a finite-set object "
            f"(``object N : FinSet k``) and give the per-row value space "
            f"through the family or an ``[over=...]`` axis",
            line,
            col,
        )

    @staticmethod
    def _object_expr_text(texpr: ObjectExpr) -> str:
        """Render an object expression back to its surface spelling for
        diagnostics. Named types print as their name, products as
        ``A * B``; anything else falls back to its node repr.
        """
        if isinstance(texpr, TypeName):
            return texpr.name
        if isinstance(texpr, ObjectProduct):
            return " * ".join(
                _ProgramsMixin._object_expr_text(c) for c in texpr.components
            )
        return str(texpr)

    @staticmethod
    def _matrix_family_spec(family: str) -> FamilySpec | None:
        """The `FamilySpec` of ``family`` when it draws a
        matrix-valued variate, else None.

        A matrix-valued family (LKJCholesky, Wishart) reads its matrix
        dimension ``K`` at construction time and produces a ``(K, K)``
        variate per draw, so its codomain cannot be inferred from the
        argument list the way a scalar family's can.
        """
        spec = FAMILY_REGISTRY.get(family)
        if spec is None or spec.output_kind != "matrix":
            return None
        return spec

    @staticmethod
    def _matrix_codomain(
        spec: FamilySpec,
        dim: int,
        var_name: str,
        line: int,
        col: int,
    ) -> CholeskyFactor | Covariance | LowerTriangular:
        """The codomain manifold a matrix-valued family draws on.

        The family's ``support`` constraint, not its name, fixes the
        manifold: a correlation Cholesky factor lands on
        `CholeskyFactor`, a positive-definite
        variate on `Covariance`, a general
        lower-triangular one on `LowerTriangular`.
        A matrix family advertising some other support has no space to
        map onto, so it raises rather than defaulting to a Euclidean
        codomain whose ``dim`` would silently misconfigure the
        distribution.
        """
        support = spec.support
        if support is _constraints.corr_cholesky:
            return CholeskyFactor(name=f"_{var_name}", dim=dim)
        if support in (
            _constraints.positive_definite,
            _constraints.positive_semidefinite,
        ):
            return Covariance(name=f"_{var_name}", dim=dim)
        if support is _constraints.lower_cholesky:
            return LowerTriangular(name=f"_{var_name}", dim=dim)
        raise CompileError(
            f"matrix-valued family {spec.name!r} advertises support "
            f"{support!r}, which no continuous space represents; the "
            f"compiler cannot give {var_name!r} a codomain",
            line,
            col,
        )

    @staticmethod
    def _axis_event_size(space: object) -> int | None:
        """Cardinality of an explicit ``over=`` event axis, or None.

        A finite-set axis contributes its cardinality, a continuous
        axis its ``dim``, and a product axis the product of its
        components' sizes. Anything else yields None (no resolvable
        event width).

        The product test comes first because `ProductSet` is a
        `SetObject` variant and `ProductSpace` a
        `ContinuousSpace` variant: an axis
        ``[over=[A, B]]`` matches the single-factor branches too, and
        answering there would read a ``cardinality`` field a product
        set does not carry, or sum the component dims where the event
        width of a product axis is their product.
        """
        if isinstance(space, (ProductSet, ProductSpace)):
            total = 1
            for comp in space.components:
                sub = _ProgramsMixin._axis_event_size(comp)
                if sub is None:
                    return None
                total *= sub
            return total
        if isinstance(space, SetObject):
            return int(space.size)
        if isinstance(space, ContinuousSpace):
            return int(space.dim)
        return None

    def _infer_inline_codomain(
        self,
        family: str,
        args: tuple,
        var_names: tuple[str, ...],
        program_codomain: object,
        event_axis: object | None = None,
        line: int = 0,
        col: int = 0,
    ):
        """Infer the codomain for an inline distribution.

        Parameters
        ----------
        family : str
            Distribution family name.
        args : tuple
            Arguments from the draw step.
        var_names : tuple[str, ...]
            Bound variable name(s).
        program_codomain : object
            The program's declared codomain.
        event_axis : object or None
            The object named by an explicit ``[over=...]`` clause,
            when present. A scalar-per-draw family paired with an
            event axis of cardinality ``K`` draws a ``K``-vector of
            independent variates per plate row, so its per-row codomain
            becomes ``K``-dimensional rather than scalar. Families
            whose event dimension already flows from
            ``program_codomain`` (Dirichlet's simplex) ignore this.
        line : int
            Source line of the draw step, for diagnostics.
        col : int
            Source column of the draw step, for diagnostics.

        Returns
        -------
        AnySpace
            The inferred codomain.
        """
        bound = bind_family_arguments(
            family, args, get_inline_param_names(family) or ()
        )
        argument_by_name = dict(bound)
        args = tuple(value for _, value in bound)
        event_size = (
            self._axis_event_size(event_axis) if event_axis is not None else None
        )
        checked_event = self._checked_event_extents(
            family, argument_by_name, var_names[0]
        )
        matrix_spec = self._matrix_family_spec(family)
        if matrix_spec is not None:
            # A matrix-valued family takes its ``K x K`` shape from the
            # width in scope: the ``[over=...]`` axis when one is
            # written, else the extents the checked program gives the
            # draw, otherwise the object the draw is annotated with.
            # None is optional, since the underlying distribution needs
            # ``K`` at construction and a scalar codomain would build it
            # at ``K = 1``.
            matrix_dim = (
                event_size
                if event_size is not None
                else checked_event[-1]
                if checked_event is not None
                else self._axis_event_size(program_codomain)
            )
            if matrix_dim is None or matrix_dim < 2:
                raise CompileError(
                    f"matrix-valued family {family!r} needs a matrix "
                    f"dimension of at least 2 to draw {var_names[0]!r}; "
                    f"annotate the draw with the dimension object "
                    f"(``{var_names[0]} : Dim <- {family}(...)``) or name "
                    f"it with ``[over=Dim]``",
                    line,
                    col,
                )
            return self._matrix_codomain(
                matrix_spec, matrix_dim, var_names[0], line, col
            )
        if family == "InverseWishart":
            # An inverse Wishart draws a K x K positive-definite matrix:
            # K is the ``[over=...]`` axis's size, else the extent the
            # checked program gives the draw, else the codomain's width.
            matrix_dim = (
                event_size
                if event_size is not None
                else checked_event[-1]
                if checked_event is not None
                else self._axis_event_size(program_codomain)
            )
            if matrix_dim is None or matrix_dim < 2:
                raise CompileError(
                    f"InverseWishart draw of {var_names[0]!r} needs a matrix "
                    f"dimension of at least 2: annotate the draw with the "
                    f"dimension object (``{var_names[0]} : Dim <- "
                    f"InverseWishart(...)``) or name it with ``[over=Dim]``",
                    line,
                    col,
                )
            return Covariance(name=f"_{var_names[0]}", dim=matrix_dim)
        if family == "MatrixNormal":
            # A matrix normal draws one R x C matrix, its R C entries the
            # ``[over=[R, C]]`` axes or the checked extents of its mean.
            entries = (
                event_size
                if event_size is not None
                else math.prod(checked_event)
                if checked_event is not None
                else None
            )
            if entries is None:
                raise CompileError(
                    f"MatrixNormal draw of {var_names[0]!r} fixes no matrix "
                    f"shape the compiled runtime can build it at: name its axes "
                    f"with ``[over=[Rows, Cols]]`` or give its 'loc' a fixed shape",
                    line,
                    col,
                )
            return Euclidean(name=f"_{var_names[0]}", dim=entries)
        record = FAMILIES.get(family)
        if (
            record is not None
            and record.event_rank == 1
            and record.event_sources
            and family in _MULTIVARIATE_REAL_FAMILIES
        ):
            # A multivariate real family samples a vector over its event
            # axis: the ``[over=...]`` axis when one is written, else the
            # extent the checked program gives its event source.
            width = (
                event_size
                if event_size is not None
                else checked_event[-1]
                if checked_event is not None
                else None
            )
            if width is None:
                # A checked program that fails to lower says why first.
                _ = self.qiec_module
                raise CompileError(
                    f"{family} draw of {var_names[0]!r} fixes no event size the "
                    f"compiled runtime can build it at: name its axis with "
                    f"``[over=K]``, annotate it (``{var_names[0]} : K <- "
                    f"{family}(...)``), or give its "
                    f"{' or '.join(map(repr, record.event_sources))} a fixed size",
                    line,
                    col,
                )
            return Euclidean(name=f"_{var_names[0]}", dim=width)
        if event_size is not None and event_size > 1:
            if family in (
                "Normal",
                "LogitNormal",
                "Uniform",
                "TruncatedNormal",
                "LogGamma",
                "LogitBeta",
            ):
                return Euclidean(name=f"_{var_names[0]}", dim=event_size)
            if family in (
                "Exponential",
                "HalfCauchy",
                "HalfNormal",
                "LogNormal",
                "Gamma",
            ):
                return PositiveReals(name=f"_{var_names[0]}", dim=event_size)
        if family == "LogitNormal":
            return UnitInterval(f"_{var_names[0]}")
        elif family == "Bernoulli":
            return FinSet(name=f"_{var_names[0]}", cardinality=2)
        elif family == "Uniform":
            low_value = self._as_float(argument_by_name.get("low"))
            high_value = self._as_float(argument_by_name.get("high"))
            if low_value is not None and high_value is not None:
                low, high = (float(low_value), float(high_value))
                if low == 0.0 and high == 1.0:
                    return UnitInterval(f"_{var_names[0]}")
                return Euclidean(name=f"_{var_names[0]}", dim=1, low=low, high=high)
            return UnitInterval(f"_{var_names[0]}")
        elif family == "TruncatedNormal":
            low_value = self._as_float(argument_by_name.get("low"))
            high_value = self._as_float(argument_by_name.get("high"))
            if low_value is not None and high_value is not None:
                low, high = (float(low_value), float(high_value))
                return Euclidean(name=f"_{var_names[0]}", dim=1, low=low, high=high)
            return UnitInterval(f"_{var_names[0]}")
        elif family == "Normal":
            return Euclidean(name=f"_{var_names[0]}", dim=1)
        elif family == "Beta":
            return UnitInterval(f"_{var_names[0]}")
        elif family == "ZeroOneInflatedBeta":
            return UnitInterval(f"_{var_names[0]}")
        elif family in (
            "Exponential",
            "HalfCauchy",
            "HalfNormal",
            "LogNormal",
            "Gamma",
        ):
            return PositiveReals(name=f"_{var_names[0]}", dim=1)
        elif family == "Dirichlet":
            # Inline Dirichlet's simplex dimension:
            #
            # * ``Dirichlet([a_1, …, a_K])`` (parser flattens the
            #   bracketed numeric sequence into K positional literal
            #   floats) → K-simplex.
            # * ``Dirichlet(alpha)`` with a single scalar literal →
            #   simplex dimension comes from the program's declared
            #   codomain (``dim`` for a ContinuousSpace,
            #   ``cardinality`` for a SetObject) of at least two; a
            #   draw that fixes none is refused.
            sim_dim: int | None = None
            n_literals = sum(
                1
                for a in args
                if isinstance(a, (int, float)) or isinstance(a, DrawArgScalar)
            )
            if n_literals >= 2:
                sim_dim = n_literals
            if sim_dim is None:
                for a in args:
                    if isinstance(a, DrawArgList) and a.items:
                        sim_dim = len(a.items)
                        break
                    if isinstance(a, (list, tuple)) and len(a) > 0:
                        sim_dim = len(a)
                        break
            if sim_dim is None and isinstance(program_codomain, ContinuousSpace):
                sim_dim = getattr(program_codomain, "dim", None)
            if sim_dim is None and isinstance(program_codomain, SetObject):
                # A discrete codomain of cardinality `k` indexes a
                # k-simplex of class probabilities.
                sim_dim = getattr(program_codomain, "cardinality", None)
            if sim_dim is None or sim_dim < 2:
                raise CompileError(
                    f"Dirichlet draw of {var_names[0]!r} fixes no simplex "
                    f"dimension: give the concentration as a list, annotate "
                    f"the draw with the simplex's object "
                    f"(``{var_names[0]} : K <- Dirichlet(...)``), or name it "
                    f"with ``[over=K]``",
                    line,
                    col,
                )
            return Simplex(name=f"_{var_names[0]}", dim=sim_dim)
        elif family == "OneHotCategorical":
            values = argument_by_name.get("probs", argument_by_name.get("logits"))
            width = (
                len(values.items)
                if isinstance(values, DrawArgList) and values.items
                else event_size
            )
            if width is None:
                raise CompileError(
                    f"OneHotCategorical draw of {var_names[0]!r} fixes no number "
                    f"of classes: give its probabilities as a list or name the "
                    f"classes' object with ``[over=K]``",
                    line,
                    col,
                )
            return Euclidean(name=f"_{var_names[0]}", dim=width)
        elif family == "Categorical":
            values = argument_by_name.get("probs", argument_by_name.get("logits"))
            if isinstance(values, DrawArgList) and values.items:
                cardinality = len(values.items)
            elif isinstance(program_codomain, SetObject):
                cardinality = int(program_codomain.cardinality)
            else:
                cardinality = 2
            return FinSet(name=f"_{var_names[0]}", cardinality=cardinality)
        else:
            return Euclidean(name=f"_{var_names[0]}", dim=1)

    def _checked_event_extents(
        self,
        family: str,
        arguments: Mapping[str, object],
        site: str,
    ) -> tuple[int, ...] | None:
        """The event extents the checked program gives an inline draw.

        Parameters
        ----------
        family : str
            The family's source name.
        arguments : Mapping[str, object]
            The draw's arguments by parameter name.
        site : str
            The name the draw binds.

        Returns
        -------
        tuple[int, ...] | None
            The trailing event-rank extents of the argument the family's
            event shape comes from, a list literal's length for a vector,
            else of the draw's own checked value; ``None`` when none is
            fixed or the family has no event axes.
        """
        record = FAMILIES.get(family)
        if record is None or record.event_rank == 0:
            return None
        rank = record.event_rank
        extents = self._value_extents_stack[-1] if self._value_extents_stack else {}
        for name in record.event_sources:
            argument = arguments.get(name)
            if isinstance(argument, DrawArgName):
                checked = extents.get(argument.text)
                if checked is not None and len(checked) >= rank:
                    return tuple(checked[len(checked) - rank :])
            if isinstance(argument, DrawArgList) and argument.items and rank == 1:
                return (len(argument.items),)
        own = extents.get(site)
        if own is not None and len(own) >= rank:
            return tuple(own[len(own) - rank :])
        return None

    @staticmethod
    def _locate_let_failure(
        fn: Callable[[dict[str, "LetValue"]], "LetValue"],
        kind: str,
        name: str,
        line: int,
        col: int,
    ) -> Callable[[dict[str, "LetValue"]], "LetValue"]:
        """Give a let / score body's evaluation failures a source location.

        The body is compiled to a closure that runs against the trace
        environment, so a name it cannot resolve is only discovered
        when the program runs, long after the parse position is out of
        reach. Wrapping the closure re-raises any unlocated
        `CompileError` against the binding that
        produced it, which is what makes the diagnostic for a free
        name in ``let w = z + 1.0`` say the same thing, in the same
        place, as the one for ``let w = z``.
        """

        def _run(env: dict[str, "LetValue"]) -> "LetValue":
            try:
                return fn(env)
            except CompileError as exc:
                if exc.line:
                    raise
                raise CompileError(f"{kind} {name!r}: {exc}", line, col) from exc

        return _run

    def _let_globals(self) -> "dict[str, LetValue]":
        """The names a program's let expressions resolve statically.

        Returns
        -------
        dict[str, LetValue]
            The compiled deductions, which ``parse(D, ...)`` reads; every
            learned function (``[role=function]``) by name, called as
            ``f(a, b)`` on one value per domain factor; and the
            index-size resolver ``factor`` binders read.
        """
        found: dict[str, LetValue] = dict(getattr(self, "_deductions", {}))
        for name, morphism in self._morphisms.items():
            if isinstance(morphism, LearnedFunction):
                found[name] = morphism
        found["__index_size__"] = self._resolve_index_size
        return found

    def _let_reads(self, expr: LetExprNode) -> frozenset[str]:
        """The bindings a let expression reads, latent network weights included.

        Parameters
        ----------
        expr : LetExprNode
            The expression.

        Returns
        -------
        frozenset[str]
            The expression's free names, and the weight sites of every
            latent learned function it calls.
        """
        reads = set(_let_expr_reads(expr))
        for callee in free_let_callees(expr):
            function = self._morphisms.get(callee)
            if isinstance(function, LearnedFunction):
                reads |= set(function.latent_sites.values())
        return frozenset(reads)

    def _function_widths(self) -> dict[str, int]:
        """The codomain width of every learned function, by name.

        Returns
        -------
        dict[str, int]
            One entry per ``[role=function]`` morphism.
        """
        return {
            name: int(cast(ContinuousSpace, morphism.codomain).dim)
            for name, morphism in self._morphisms.items()
            if isinstance(morphism, LearnedFunction)
        }

    @staticmethod
    def _reject_unknown_let_calls(
        node: LetExprNode,
        globals_: Mapping[str, LetValue | frozenset[str]],
        bound: Collection[str],
        kind: str,
        name: str,
        line: int,
        col: int,
    ) -> None:
        """Refuse a let or score body that calls a name nothing defines.

        Parameters
        ----------
        node : LetExprNode
            The body.
        globals_ : Mapping[str, LetValue | frozenset[str]]
            The static globals the body is compiled with.
        bound : Collection[str]
            The names bound before the step.
        kind : str
            ``"let"`` or ``"score"``, for the message.
        name : str
            The binding's name.
        line : int
            The step's source line, used when the call has none.
        col : int
            The step's source column, used when the call has none.

        Raises
        ------
        CompileError
            If a call names no builtin, combinator, global, constructor,
            or earlier binding, located at the call.
        """
        call = _unknown_let_call(node, globals_, bound)
        if call is None:
            return
        raise CompileError(
            f"{kind} {name!r}: unknown function {call.func!r} in let expression; "
            f"it names no builtin, no constructor, and nothing bound earlier",
            call.line or line,
            call.col if call.line else col,
        )

    def _validate_let_expr_vars(
        self, node: LetExprNode, bound_vars: dict[str, AnySpace | None], step: LetStep
    ) -> None:
        """Validate that all variables in a let expression are bound.

        The validator tolerates references to compiled deductions
        (via ``self._deductions``) and lambdas that bind their own
        parameter — both of these are resolved at runtime by the
        let-expression evaluator's `globals_` channel and its
        lambda-environment extension.

        Free variables that match none of the above are treated as
        *host-data references*: the value is expected to arrive at
        runtime through [`quivers.inference.condition`][quivers.inference.condition]'s data
        dict (e.g. per-row index arrays for hierarchical regression).
        The runtime raises a clear ``KeyError`` if the data dict
        doesn't supply such a name.
        """
        deductions = getattr(self, "_deductions", {})

        # Inner walker carries a set of locally-bound names from
        # surrounding lambdas, so a lambda's `param` is treated as
        # in-scope inside its body.
        def _walk(node, locals_set: set[str]) -> None:
            if isinstance(node, LetExprVar):
                if node.name in bound_vars or node.name in deductions:
                    return
                if node.name in locals_set:
                    return
                # Otherwise the variable is a host-data reference; the
                # trace runtime resolves it against the conditioning
                # data dict at execution time.
                return
            if isinstance(node, LetExprBinOp):
                _walk(node.left, locals_set)
                _walk(node.right, locals_set)
            elif isinstance(node, LetExprUnaryOp):
                _walk(node.operand, locals_set)
            elif isinstance(node, LetExprTuple):
                for item in node.items:
                    _walk(item, locals_set)
            elif isinstance(node, LetExprCall):
                for arg in node.args:
                    _walk(arg, locals_set)
            elif isinstance(node, LetExprList):
                for item in node.items:
                    _walk(item, locals_set)
            elif isinstance(node, LetExprLambda):
                _walk(node.body, locals_set | {node.param})
            elif isinstance(node, LetExprMethodCall):
                _walk(node.receiver, locals_set)
                for arg in node.args:
                    _walk(arg, locals_set)
            elif isinstance(node, LetExprIndex):
                _walk(node.array, locals_set)
                for idx in node.indices:
                    _walk(idx, locals_set)
            elif isinstance(node, LetExprFactor):
                # Each binder's variable is in scope only inside
                # the factor's body / case values; the index type
                # expression itself is a ObjectExpr, not a let-expr,
                # so we don't walk it for variable references.
                inner = locals_set | {b.var for b in node.binders}
                if node.body is not None:
                    _walk(node.body, inner)
                for case in node.cases:
                    _walk(case.value, inner)
            # LetExprLiteral, LetExprString carry no variables.

        _walk(node, set())

    @staticmethod
    def _compile_let_expr(
        node: LetExprNode,
        globals_: "dict[str, LetValue] | None" = None,
    ) -> Callable[[dict[str, "LetValue"]], "LetValue"]:
        """Compile a let expression tree into a callable.

        The returned callable takes an environment dict mapping
        names to Python values (tensors, strings, lists, lambdas,
        chart views, structured tuples, ...) and returns the
        expression's value. The let-sublanguage is a small typed
        lambda calculus over heterogeneous values; the runtime
        evaluator preserves autograd through all tensor
        operations.

        Supported node kinds:

        * `LetExprLiteral`: numeric literal → tensor.
        * `LetExprString`: string literal → Python str.
        * `LetExprVar`: variable reference → env lookup.
        * `LetExprBinOp`: arithmetic over tensors.
        * `LetExprUnaryOp`: negation.
        * `LetExprList`: list literal → stacked tensor, or a Python
          list when an item is not a tensor.
        * `LetExprLambda`: closure over the let environment.
        * `LetExprMethodCall`: dispatch on receiver type.
        * `LetExprCall`: built-in or constructor mode.
        * `LetExprIndex`: tensor gather.
        """
        if isinstance(node, LetExprLiteral):
            # An integral literal is an integer tensor, so integer
            # arithmetic on it (division above all) stays integral.
            val = int(node.value) if node.integral else node.value

            def _literal(env: dict) -> torch.Tensor:
                for v in env.values():
                    if isinstance(v, torch.Tensor):
                        return torch.tensor(val, device=v.device)
                return torch.tensor(val)

            return _literal
        if isinstance(node, LetExprString):
            val = node.value

            def _string(env: dict) -> str:
                return val

            return _string
        if isinstance(node, LetExprBool):
            flag = node.value

            def _bool(env: dict) -> torch.Tensor:
                return torch.tensor(flag)

            return _bool
        if isinstance(node, LetExprUnit):

            def _unit(env: dict) -> None:
                return None

            return _unit
        if isinstance(node, LetExprTuple):
            item_fns = [
                _ProgramsMixin._compile_let_expr(item, globals_=globals_)
                for item in node.items
            ]

            def _tuple(env: dict) -> tuple[object, ...]:
                return tuple(fn(env) for fn in item_fns)

            return _tuple
        if isinstance(node, LetExprVar):
            name = node.name
            globs = globals_ or {}
            constructors = globs.get(LET_CONSTRUCTORS_KEY, frozenset())
            bound_variables = globs.get(LET_BOUND_VARIABLES_KEY, frozenset())

            def _var(env: dict):
                if name in env:
                    return env[name]
                if name in bound_variables:
                    return variable_term(name)
                if name in constructors:
                    return atom_term(name)
                if name in globs and name not in LET_INTERNAL_KEYS:
                    return globs[name]
                raise CompileError(
                    f"undefined variable {name!r}: it is neither bound "
                    f"earlier in the program body nor supplied under that "
                    f"name in the conditioning data"
                )

            return _var
        if isinstance(node, LetExprList):
            item_fns = [
                _ProgramsMixin._compile_let_expr(it, globals_=globals_)
                for it in node.items
            ]

            def _list(env: dict) -> torch.Tensor | list[LetValue]:
                return _collection_value([fn(env) for fn in item_fns])

            return _list
        if isinstance(node, LetExprLambda):
            param = node.param
            body_fn = _ProgramsMixin._compile_let_expr(node.body, globals_=globals_)

            def _lambda(env: dict):
                # Returns a Python callable closed over the let-env.
                def _closure(arg):
                    extended = dict(env)
                    extended[param] = arg
                    return body_fn(extended)

                return _closure

            return _lambda
        if isinstance(node, LetExprBinOp):
            left_fn = _ProgramsMixin._compile_let_expr(node.left, globals_=globals_)
            right_fn = _ProgramsMixin._compile_let_expr(node.right, globals_=globals_)
            op = node.op

            def _binop(env: dict) -> torch.Tensor:
                l = left_fn(env)
                r = right_fn(env)
                # Promote scalar / int values to tensors.
                if not isinstance(l, torch.Tensor):
                    l = torch.tensor(float(l))
                if not isinstance(r, torch.Tensor):
                    r = torch.tensor(float(r))
                l, r = torch.broadcast_tensors(l, r)
                if op == "+":
                    return l + r
                elif op == "-":
                    return l - r
                elif op == "*":
                    return l * r
                elif op == "/":
                    # Integer division truncates toward zero; real
                    # division is exact.
                    if not (l.is_floating_point() or r.is_floating_point()):
                        return torch.div(l, r, rounding_mode="trunc")
                    return l / r
                elif op == "%":
                    # The remainder of the truncating division.
                    return torch.fmod(l, r)
                elif op == "==":
                    return l == r
                elif op == "!=":
                    return l != r
                elif op == "<":
                    return l < r
                elif op == "<=":
                    return l <= r
                elif op == ">":
                    return l > r
                elif op == ">=":
                    return l >= r
                elif op == "&&":
                    return torch.logical_and(l, r)
                elif op == "||":
                    return torch.logical_or(l, r)
                raise ValueError(f"unknown operator: {op}")

            return _binop
        if isinstance(node, LetExprUnaryOp):
            inner_fn = _ProgramsMixin._compile_let_expr(node.operand, globals_=globals_)
            if node.op == "not":

                def _not(env: dict):
                    v = inner_fn(env)
                    if isinstance(v, torch.Tensor):
                        return torch.logical_not(v)
                    return not v

                return _not

            def _neg(env: dict):
                v = inner_fn(env)
                if isinstance(v, torch.Tensor):
                    return -v
                return -v

            return _neg
        if isinstance(node, LetExprMethodCall):
            recv_fn = _ProgramsMixin._compile_let_expr(node.receiver, globals_=globals_)
            method = node.method
            arg_fns = [
                _ProgramsMixin._compile_let_expr(a, globals_=globals_)
                for a in node.args
            ]

            def _method(env: dict):
                receiver = recv_fn(env)
                args = [fn(env) for fn in arg_fns]
                if (
                    method == "goal_weight"
                    and not args
                    and isinstance(receiver, torch.Tensor)
                ):
                    # In a checked program a deduction's answer is its
                    # goal's inside weight, which ``goal_weight()`` reads.
                    return receiver
                fn = getattr(receiver, method, None)
                if fn is None:
                    raise CompileError(
                        f"object {type(receiver).__name__!r} has no method {method!r}"
                    )
                return fn(*args)

            return _method
        if isinstance(node, LetExprCall):
            func_name = node.func
            arg_fns = [
                _ProgramsMixin._compile_let_expr(a, globals_=globals_)
                for a in node.args
            ]

            # Built-in tensor operations.  Covers the standard
            # `torch.nn.functional` activation pool, common
            # elementwise transformations, and dim=-1 reductions.
            # Contractions, reductions over named axes, and matrix
            # products go through the typed contraction-declaration
            # surface (see § Operadic contractions in
            # docs/semantics/composition-rules.md).
            _TENSOR_BUILTINS = _LET_EXPR_BUILTINS

            # Compile-time arity check for calls into a user-injected
            # callable (program, morphism, encoder, decoder).  Builtins
            # and constructors deliberately accept variable arity, so
            # they are excluded from the check.
            _globs_for_check = globals_ or {}
            _higher_order_or_special = {
                "length",
                "map",
                "filter",
                "fold",
                "logsumexp_over",
                "logsumexp",
                "parse",
                "cholesky_quad_form",
            }
            if (
                func_name in _globs_for_check
                and func_name not in LET_INTERNAL_KEYS
                and func_name not in _TENSOR_BUILTINS
                and func_name not in _higher_order_or_special
                and func_name
                not in _globs_for_check.get(LET_CONSTRUCTORS_KEY, frozenset())
            ):
                _target_for_check = _globs_for_check[func_name]
                if callable(_target_for_check):
                    _arity = _expected_call_arity(_target_for_check)
                    if _arity is not None and _arity != len(arg_fns):
                        raise CompileError(
                            f"call to {func_name!r}: expected {_arity} "
                            f"positional argument(s), got {len(arg_fns)}"
                        )

            def _call(env: dict):
                # Higher-order combinators come first; they consume
                # raw closure args without eager evaluation of the
                # lambda body.
                if func_name == LOG_PROB_CALL:
                    return _family_log_prob([fn(env) for fn in arg_fns])
                if func_name == "length":
                    val = arg_fns[0](env)
                    if isinstance(val, list):
                        return len(val)
                    if isinstance(val, torch.Tensor):
                        return val.shape[0]
                    if isinstance(val, tuple):
                        return len(val)
                    raise CompileError(
                        f"length() does not support {type(val).__name__}"
                    )
                if func_name in ("map", "filter"):
                    coll = arg_fns[0](env)
                    fn = arg_fns[1](env)
                    if func_name == "map":
                        return _collection_value([fn(x) for x in coll])
                    return [x for x in coll if fn(x)]
                if func_name == "fold":
                    # fold(list, init, accumulator_lambda)
                    coll = arg_fns[0](env)
                    init = arg_fns[1](env)
                    fn = arg_fns[2](env)
                    acc = init
                    for x in coll:
                        # The lambda takes one arg (the current
                        # element); inner closures handle accumulation
                        # via additional lambda nesting:
                        #   fold(xs, 0, x -> acc_so_far -> acc + x)
                        # ... this is awkward. Two-argument folds
                        # work better; we expose `fold` as taking a
                        # lambda whose body is itself a lambda
                        # (curried), invoked here as `fn(x)(acc)`.
                        if callable(fn):
                            step = fn(acc)
                            if callable(step):
                                acc = step(x)
                            else:
                                acc = step
                        else:
                            raise CompileError("fold's accumulator must be a lambda")
                    return acc
                if func_name == "logsumexp_over":
                    # logsumexp_over(list, lambda x -> log_weight_x)
                    coll = arg_fns[0](env)
                    fn = arg_fns[1](env)
                    if not isinstance(coll, list | tuple | torch.Tensor):
                        raise CompileError(
                            "logsumexp_over runs over a list, not "
                            f"{type(coll).__name__}"
                        )
                    if not callable(fn):
                        raise CompileError("logsumexp_over's weight must be a lambda")
                    if len(coll) == 0:
                        return torch.tensor(-float("inf"))
                    weights = []
                    for x in coll:
                        w = fn(x)
                        if isinstance(w, torch.Tensor):
                            weights.append(w)
                        elif isinstance(w, int | float):
                            weights.append(torch.tensor(float(w)))
                        else:
                            raise CompileError(
                                "logsumexp_over's weight must be a number, not "
                                f"{type(w).__name__}"
                            )
                    return torch.logsumexp(torch.stack(weights), dim=0)
                if func_name == "logsumexp" and len(arg_fns) > 1:
                    # logsumexp(a, b, ...) over an explicit list of args
                    coll = [fn(env) for fn in arg_fns]
                    coll = [
                        torch.tensor(float(c)) if not isinstance(c, torch.Tensor) else c
                        for c in coll
                    ]
                    return torch.logsumexp(torch.stack(coll), dim=0)
                if func_name == "parse":
                    return _compile_parse_call(node, arg_fns)(env)
                if func_name == "subst":
                    # subst(term, var, value) — capture-avoiding
                    # substitution on a structural LF term. Walks the
                    # term tree, replacing every subterm equal to
                    # ``var`` with ``value``. Bound variables are
                    # tagged ``("var", name)`` with a fresh ``#vN``
                    # canonical name after lexicon alpha-renaming;
                    # nullary constants are tagged ``("atom", name)``.
                    # Because every binder has already been
                    # alpha-renamed to a unique canonical symbol,
                    # structural substitution is capture-avoiding.
                    if len(arg_fns) != 3:
                        raise CompileError(
                            "subst() takes exactly three arguments: term, var, value"
                        )
                    term = arg_fns[0](env)
                    var = arg_fns[1](env)
                    value = arg_fns[2](env)

                    # Structural-equality substitution: the
                    # ``var`` argument is matched against every
                    # subterm by ``==``; matching subterms are
                    # replaced wholesale by ``value``. This handles
                    # tagged variables ``("var", name)``, tagged
                    # constants ``("atom", name)``, and wrapped
                    # forms ``("Var", ("var", name))`` uniformly.
                    # Capture avoidance is automatic because the
                    # lexicon-LF compiler has already alpha-renamed
                    # every binder's bound variable to a fresh
                    # canonical symbol, so no two distinct variables
                    # share a name.
                    def _subst(t, _v=var, _r=value):
                        if t == _v:
                            return _r
                        if isinstance(t, tuple):
                            return tuple(_subst(c) for c in t)
                        return t

                    return _subst(term)
                if func_name == "compose":
                    # compose(D1, D2) — build a deduction system
                    # whose ``axiom_injector`` chains the goal items
                    # of ``D1`` (run on the call-time input) into
                    # the axiom set of ``D2``. The resulting system
                    # shares ``D2``'s semiring, agenda, and goal.
                    if len(arg_fns) != 2:
                        raise CompileError(
                            "compose() takes exactly two arguments: "
                            "deduction systems D1 and D2"
                        )
                    d1 = arg_fns[0](env)
                    d2 = arg_fns[1](env)
                    if not (
                        isinstance(d1, DeductionSystem)
                        and isinstance(d2, DeductionSystem)
                    ):
                        raise CompileError(
                            f"compose(): both arguments must be "
                            f"DeductionSystem instances; got "
                            f"{type(d1).__name__}, {type(d2).__name__}"
                        )

                    def _composed_injector(inp, _d1=d1, _d2=d2):
                        # Run D1 to fixed point; lift each of its
                        # goal items (and their weights) into D2's
                        # axiom list. The implementation defers to
                        # ``D1.__call__`` so the composition picks
                        # up D1's tolerance / max_iterations / etc.
                        sub = _d1(inp)
                        d1_axioms = list(sub.goal_items)
                        # Then thread through D2's own axiom_injector,
                        # which may add additional D2-side axioms
                        # (lexicon entries, structural axioms, etc.).
                        d2_axioms = list(_d2.axiom_injector(inp))
                        return d1_axioms + d2_axioms

                    composed = _dc_replace(d2, axiom_injector=_composed_injector)
                    # Carry across any submodules so the composed
                    # system's ``.parameters()`` walks both factors.
                    for attr in ("_axiom_module", "_rule_module"):
                        for src in (d1, d2):
                            mod = getattr(src, attr, None)
                            if mod is not None:
                                setattr(composed, attr, mod)
                    return composed
                # Standard scalar / tensor builtins.
                if func_name in _TENSOR_BUILTINS:
                    args = [fn(env) for fn in arg_fns]
                    try:
                        return _TENSOR_BUILTINS[func_name](*args)
                    except TypeError as error:
                        raise CompileError(
                            f"builtin {func_name!r} does not accept "
                            f"{len(args)} argument(s): {error}"
                        ) from error
                if func_name == "cholesky_quad_form":
                    args = [fn(env) for fn in arg_fns]
                    # cholesky_quad_form is a tensor builtin; its
                    # operands are guaranteed to be tensors by the
                    # surface contract.  Narrow for the type
                    # checker without runtime cost.
                    L_flat = cast(torch.Tensor, args[0])
                    scale = cast(torch.Tensor, args[1])
                    K = scale.shape[-1]
                    L = L_flat.reshape(*L_flat.shape[:-1], K, K)
                    mask = torch.tril(torch.ones(K, K, device=L.device, dtype=L.dtype))
                    L = L * mask
                    R = L @ L.transpose(-1, -2)
                    D = scale.unsqueeze(-1) * torch.eye(
                        K, device=L.device, dtype=L.dtype
                    )
                    cov = D @ R @ D
                    return cov.reshape(*cov.shape[:-2], K * K)
                # User-defined callable (program, morphism, encoder,
                # decoder) injected via ``globals_``. A deterministic
                # program is a Dirac Kleisli arrow embedding Smooth
                # into Kleisli(Giry); calling it from an encoder body
                # composes the two Smooth pieces and stays in Smooth.
                globs_dict = globals_ or {}
                if func_name in globs_dict and func_name not in LET_INTERNAL_KEYS:
                    target = globs_dict[func_name]
                    if isinstance(target, LearnedFunction):
                        values = [fn(env) for fn in arg_fns]
                        drawn = _drawn_weights(target, env, func_name)
                        return target.apply_values(
                            *(torch.as_tensor(value) for value in values),
                            weights=drawn,
                        )
                    if callable(target):
                        args = [fn(env) for fn in arg_fns]
                        try:
                            return target(*args)
                        except TypeError as exc:
                            raise CompileError(
                                f"call to {func_name!r} failed: {exc}; "
                                f"check that argument count and types match "
                                f"the callee's signature"
                            ) from exc
                        except RuntimeError as exc:
                            # PyTorch tensor-shape errors surface as
                            # RuntimeError; re-raise with the call site
                            # named so the user knows which call broke.
                            raise CompileError(
                                f"call to {func_name!r} failed at runtime: {exc}"
                            ) from exc
                # Constructor mode: build a tuple ``(func_name, *args)``
                # only when ``func_name`` is in the user-declared
                # constructor set (passed via the deduction compiler's
                # internal let-expression environment).
                # The free term algebra over named constructor symbols
                # is thus fully under the user's control: no identifier
                # is silently treated as a constructor.
                constructors = globs_dict.get(LET_CONSTRUCTORS_KEY, frozenset())
                if func_name in constructors:
                    args = [fn(env) for fn in arg_fns]
                    if not args:
                        return atom_term(func_name)
                    return (func_name, *args)
                raise CompileError(
                    f"unknown function {func_name!r} in let expression; "
                    f"declare it as a constructor (e.g., in a deduction's "
                    f"`atoms` block) or use a registered builtin"
                )

            return _call
        if isinstance(node, LetExprFactor):
            # Multi-axis factor: build a finite-domain-indexed
            # tensor by evaluating the body once per tuple of index
            # values.  The dual of `let_index` (LetExprIndex): the
            # left adjoint of indexing in the indexed-family
            # category over FinSet.
            globs = globals_ or {}
            resolver = globs.get("__index_size__")
            if resolver is None:
                raise CompileError(
                    "factor expression compiled without an index-size "
                    "resolver in the let-expression globals; this is a "
                    "compiler integration bug — factor expressions must "
                    "be compiled through `_ProgramsMixin._compile_program`",
                    0,
                    0,
                )
            # Resolve each binder's index type to an integer
            # cardinality at compile time.  Duplicate binder names
            # within a single factor are rejected.
            sizes: list[int] = []
            binder_names: list[str] = []
            for b in node.binders:
                if b.var in binder_names:
                    raise CompileError(
                        f"factor binder name {b.var!r} repeated; each "
                        f"binder must bind a distinct identifier",
                        b.line,
                        b.col,
                    )
                binder_names.append(b.var)
                sizes.append(resolver(b.index))

            if node.cases:
                # Pattern-match form: single-binder, one body per
                # integer label, labels must cover {0, ..., size-1}.
                if len(node.binders) != 1:
                    raise CompileError(
                        f"factor pattern-match form requires exactly one "
                        f"binder; got {len(node.binders)}",
                        node.binders[0].line,
                        node.binders[0].col,
                    )
                size = sizes[0]
                seen_labels: dict[int, "LetFactorCase"] = {}
                for case in node.cases:
                    if not (0 <= case.label < size):
                        raise CompileError(
                            f"factor case label {case.label} out of range "
                            f"[0, {size}); index has cardinality {size}",
                            case.line,
                            case.col,
                        )
                    if case.label in seen_labels:
                        raise CompileError(
                            f"factor case label {case.label} appears more than once",
                            case.line,
                            case.col,
                        )
                    seen_labels[case.label] = case
                missing = sorted(set(range(size)) - set(seen_labels))
                if missing:
                    raise CompileError(
                        f"factor pattern-match must cover every index "
                        f"in [0, {size}); missing labels: {missing!r}",
                        node.binders[0].line,
                        node.binders[0].col,
                    )
                # Compile each case's value once.  Each compiled
                # function takes the current env plus the bound
                # index variable and returns a tensor; the runtime
                # body iterates labels 0..size-1 in order.
                case_fns = [
                    _ProgramsMixin._compile_let_expr(
                        seen_labels[i].value, globals_=globals_
                    )
                    for i in range(size)
                ]
                binder_var = node.binders[0].var

                def _eval_pattern(env: dict) -> torch.Tensor:
                    pieces = []
                    for i, fn in enumerate(case_fns):
                        extended = {**env, binder_var: torch.tensor(i)}
                        pieces.append(cast(torch.Tensor, fn(extended)))
                    return torch.stack(pieces, dim=0)

                return _eval_pattern

            # Uniform body form: a single body expression, evaluated
            # once per tuple of index values from the Cartesian
            # product of the binders.
            if node.body is None:
                raise CompileError(
                    "factor expression has neither a uniform body nor "
                    "a pattern-match block",
                    node.binders[0].line,
                    node.binders[0].col,
                )
            body_fn = _ProgramsMixin._compile_let_expr(node.body, globals_=globals_)
            # Closure-capture the binder names and sizes for the
            # runtime nested loop.
            local_binders = tuple(binder_names)
            local_sizes = tuple(sizes)

            def _eval_uniform(env: dict) -> torch.Tensor:
                # Build a (size_1, ..., size_n, *body_shape) tensor
                # by evaluating body at each index tuple and
                # reshaping.  We iterate in lexicographic order and
                # reshape at the end so torch.stack only needs to
                # see a flat list of body tensors.
                flat: list[torch.Tensor] = []
                for tup in _cartesian_product(*(range(s) for s in local_sizes)):
                    extended = {
                        **env,
                        **{
                            name: torch.tensor(val)
                            for name, val in zip(local_binders, tup)
                        },
                    }
                    flat.append(cast(torch.Tensor, body_fn(extended)))
                stacked = torch.stack(flat, dim=0)
                return stacked.reshape(*local_sizes, *stacked.shape[1:])

            return _eval_uniform
        if isinstance(node, LetExprIndex):
            # Indexed gather along the leading axis of the array.
            # Realises the Kleisli pullback ι^* v = v ∘ ι for a finite
            # fibration ι : N → A and a plate variable v : A → B.
            arr_fn = _ProgramsMixin._compile_let_expr(node.array, globals_=globals_)
            idx_fns = [
                _ProgramsMixin._compile_let_expr(ix, globals_=globals_)
                for ix in node.indices
            ]

            def _index(env: dict) -> LetValue:
                arr = arr_fn(env)
                idx_tensors = [torch.as_tensor(fn(env)) for fn in idx_fns]
                if isinstance(arr, tuple):
                    # A tuple's component is selected by one static
                    # position, as the checked program projects it.
                    if len(idx_tensors) != 1 or idx_tensors[0].numel() != 1:
                        raise CompileError(
                            "a tuple component is selected by one integer"
                        )
                    return arr[int(idx_tensors[0].reshape(()).item())]
                tensor = cast(torch.Tensor, arr)
                long_idx = tuple(
                    ix.to(torch.long) if ix.dtype != torch.long else ix
                    for ix in idx_tensors
                )
                return tensor[long_idx]

            return _index
        raise CompileError(f"unknown let expression node: {type(node).__name__}")

    def _compile_define(self, decl: DefineDecl) -> None:
        """Compile a ``define`` binding with optional where clause.

        The RHS is first classified by surface shape:

        * If it's an expression that denotes a transformation
          (bare reference to a registered trans singleton or
          define-bound trans, a constructor call against the
          transformation catalog, or ``t1 >>> t2``), the binding
          lands in `_transformations`.
        * Otherwise it's compiled as a morphism expression and
          lands in `_morphisms`.

        The two namespaces are disjoint: a name cannot be used as
        both a morphism and a transformation in the same module.
        Where-block entries compile first (innermost bindings feed
        the outer RHS) and are scoped to this binding: after the
        outer RHS compiles, the where-bound names are removed from
        the module namespaces. Each entry must itself be a
        `DefineDecl`.
        """
        where_names: list[str] = []
        for where_decl in decl.where:
            if not isinstance(where_decl, DefineDecl):
                raise CompileError(
                    f"define {decl.name!r}: where-block entries must be "
                    f"define bindings; got {type(where_decl).__name__}",
                    decl.line,
                    decl.col,
                )
            self._compile_define(where_decl)
            where_names.append(where_decl.name)
        if decl.name in self._morphisms or decl.name in self._transformations:
            raise CompileError(f"name {decl.name!r} already bound", decl.line, decl.col)
        if self._scans_marginalized_cell(decl.expr):
            # A marginalized chain is a draw of the program that samples
            # it, which compiles it; it has no morphism of its own.
            return
        try:
            if self._is_trans_expr(decl.expr):
                self._transformations[decl.name] = self._compile_trans_expr(decl.expr)
            else:
                self._morphisms[decl.name] = self._compile_expr(decl.expr)
        finally:
            for name in where_names:
                self._morphisms.pop(name, None)
                self._transformations.pop(name, None)

    def _scans_marginalized_cell(self, expr: Expr) -> bool:
        """Whether an expression names the cell of a marginalized chain.

        Parameters
        ----------
        expr : Expr
            A ``define``'s right-hand side.

        Returns
        -------
        bool
            ``True`` when the expression mentions a program that returns
            the latent it marginalizes, which only a scan drawn by a
            program runs; the expansion of that draw checks the form.
        """
        return bool(_expr_ident_names(expr) & chain_cells(self._module))

    def _is_trans_expr(self, expr) -> bool:
        """Return True iff ``expr`` denotes a transformation value.

        The classification is *purely structural* — based on the
        expression's surface shape — and is the criterion the
        define-binding logic uses to choose between the morphism and
        transformation namespaces.  An `ExprMorphismCall`
        whose callee is in `_trans_constructors` is a
        transformation; the same shape with a callee in
        `_contractions` is a morphism.
        """
        if isinstance(expr, ExprTransCompose):
            return True
        if isinstance(expr, ExprIdent):
            return (
                expr.name in self._trans_singletons
                or expr.name in self._transformations
                or expr.name in self._trans_constructors
            )
        if isinstance(expr, ExprMorphismCall):
            return expr.callee in self._trans_constructors
        return False


__all__ = []
