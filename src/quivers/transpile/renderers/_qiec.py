"""Structural target lowering for executable QIEC computations.

The dynamic targets share one small free-computation ABI, expressed in each
host language and grafted into the renderer's Panproto schema.  The three
static graphical languages accept only the analyzer-proven, effect-free scalar
fragment and emit named pure definitions.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable

import panproto

from quivers.qiec.identifiers import SiteId
from quivers.transpile._api import UnsupportedConstruct
import didactic.api as dx

from quivers.transpile._pipeline import parser_registry
from quivers.transpile.family_spelling import (
    bridge_source,
    helper_families,
    helper_roots,
    spell_distribution,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.dsl.pure_builtins import PRIMITIVE_BUILTINS
from quivers.transpile._builtin_spellings import (
    Dedicated,
    builtin_spelling,
    native_path,
    spell_builtin,
)
from quivers.transpile.ir import (
    IRCall,
    IRMarginalize,
    IRNode,
    IRProgram,
    IRQiecValueExpr,
)
from quivers.transpile.renderers._python_helpers import (
    PyCtx,
    assignment,
    render_let_expr_python,
)
from quivers.transpile.qiec_ir import (
    IRQiecAttachmentRef,
    IRQiecIndexLiteral,
    IRQiecBind,
    IRQiecLocal,
    IRQiecBoolLiteral,
    IRQiecBytesLiteral,
    IRQiecCall,
    IRQiecCase,
    IRQiecComputation,
    IRQiecConstructorValue,
    IRQiecEvidenceValue,
    IRQiecFloatLiteral,
    IRQiecHandle,
    IRQiecIntLiteral,
    IRQiecLiteral,
    IRQiecLiteralValue,
    IRQiecHandlerDef,
    IRQiecIf,
    IRQiecModule,
    IRQiecNamedComputation,
    IRQiecNewInstance,
    IRQiecNullLiteral,
    IRQiecPerform,
    IRQiecPrimitiveApplication,
    IRQiecProjection,
    IRQiecResume,
    IRQiecReturn,
    IRQiecAffineMap,
    IRQiecTableMap,
    IRQiecComprehension,
    IRQiecDistributionValue,
    IRQiecLetValue,
    IRQiecReduction,
    IRQiecRowwise,
    IRQiecGather,
    IRQiecKernelMatrix,
    IRQiecLogDensity,
    IRQiecSegmentSum,
    IRQiecWeightSum,
    IRQiecShapeIndex,
    IRQiecSiteValue,
    IRQiecStatic,
    IRQiecStringLiteral,
    IRQiecTensorValue,
    IRQiecTransportValue,
    IRQiecTupleLiteral,
    IRQiecTupleValue,
    IRQiecTypeApplication,
    IRQiecValue,
    IRQiecVar,
    analyze_qiec_capabilities,
)


_HERE = pathlib.Path(__file__).resolve().parent.parent
_RUNTIMES = {
    "python": _HERE / "runtime_qiec.py",
    "julia": _HERE / "runtime_qiec.jl",
    "javascript": _HERE / "runtime_qiec.js",
    "scheme": _HERE / "runtime_qiec.scm",
}
_DYNAMIC_TARGETS = {
    "pyro": "python",
    "numpyro": "python",
    "pymc": "python",
    "edward2": "python",
    "turing": "julia",
    "gen": "julia",
    "webppl": "javascript",
    "church": "scheme",
}
_ROOT_KINDS = {
    "python": "module",
    "julia": "source_file",
    "javascript": "program",
    "scheme": "program",
}


def qiec_target_name(target: str) -> str:
    """Normalize renderer-internal target labels to public registry keys."""
    return target.removeprefix("qvr-")


def has_runtime_computations(ir: IRProgram) -> bool:
    """Whether a program carries computations the host runtime must run.

    Parameters
    ----------
    ir : IRProgram
        The lowered program.

    Returns
    -------
    bool
        ``True`` when the QIEC module has a computation that is neither a
        program entry point nor one of its marginal helpers.
    """
    return bool(_runtime_computations(ir.module))


def _runtime_computations(
    module: IRQiecModule,
) -> tuple[IRQiecNamedComputation, ...]:
    """The computations a host runtime carries for a module.

    Parameters
    ----------
    module : IRQiecModule
        The module.

    Returns
    -------
    tuple[IRQiecNamedComputation, ...]
        Every computation but the programs and their helpers: a
        program's plan is rendered from its computation, and a program
        another program draws from is planned in place at the call.
    """
    programs = module.program_computations()
    return tuple(
        computation
        for computation in module.computations
        if computation.id.text not in programs
    )


def render_computations_dynamic(
    sb: panproto.SchemaBuilder,
    ir: IRProgram,
    *,
    target: str,
    root: str,
) -> None:
    """Render the module's computations into a dynamic target's root.

    The shared runtime, the target's distribution bridge, and one
    function per computation the plan does not spell itself are
    parsed and placed under ``root``.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    ir : IRProgram
        The lowered root.
    target : str
        The renderer's target label.
    root : str
        The schema vertex the definitions are placed under.

    Raises
    ------
    UnsupportedConstruct
        If the target lacks a capability a computation needs.
    """
    module = ir.module
    if not _runtime_computations(module):
        return
    public_target = qiec_target_name(target)
    diagnostics = analyze_qiec_capabilities(module, public_target)
    if diagnostics:
        raise UnsupportedConstruct(
            f"qvr-{public_target}", [diagnostic.kind for diagnostic in diagnostics]
        )
    grammar = _DYNAMIC_TARGETS[public_target]
    runtime = _RUNTIMES[grammar].read_text()
    definitions = _dynamic_definitions(module, grammar, public_target)
    _graft_parsed_children(
        sb,
        grammar=grammar,
        source=runtime + "\n" + bridge_source(public_target) + "\n" + definitions,
        source_root_kind=_ROOT_KINDS[grammar],
        destination=root,
        prefix=f"qiec_{public_target}",
    )


def needed_computations(ir: IRProgram) -> tuple[IRQiecNamedComputation, ...]:
    """The computations a static target has to carry for a lowered root.

    A static target renders the program's plan, so it needs the
    computations the plan calls, and what those call in turn; a
    computation the program never reaches is not part of its output.
    A module with no program has nothing but its computations, so all
    of them are needed.

    Parameters
    ----------
    ir : IRProgram
        The lowered root.

    Returns
    -------
    tuple[IRQiecNamedComputation, ...]
        The needed computations, in module order.
    """
    module = ir.module
    runtime = _runtime_computations(module)
    if not module.programs:
        return runtime
    by_name = {item.name: item for item in runtime}
    by_id = {item.id.text: item for item in runtime}
    pending = [by_name[name] for name in _called_names(ir.body) if name in by_name]
    needed: dict[str, IRQiecNamedComputation] = {}
    while pending:
        item = pending.pop()
        if item.id.text in needed:
            continue
        needed[item.id.text] = item
        pending.extend(
            by_id[callee] for callee in _callee_ids(item.body) if callee in by_id
        )
    return tuple(item for item in runtime if item.id.text in needed)


def _called_names(body: tuple[IRNode, ...]) -> list[str]:
    """The callees of every call in a plan body, marginalize scopes included.

    Parameters
    ----------
    body : tuple[IRNode, ...]
        The plan body.

    Returns
    -------
    list[str]
        Callee names in plan order.
    """
    names: list[str] = []
    for node in body:
        if isinstance(node, IRCall):
            names.append(node.callee)
        elif isinstance(node, IRMarginalize):
            names.extend(_called_names(node.scope))
    return names


def _callee_ids(node: IRQiecComputation) -> list[str]:
    """The identities of every computation a body calls.

    Parameters
    ----------
    node : IRQiecComputation
        The body.

    Returns
    -------
    list[str]
        Callee identity texts.
    """
    found: list[str] = []
    if isinstance(node, IRQiecCall):
        found.append(node.callee.text)
    elif isinstance(node, IRQiecBind):
        for step in node.steps:
            found.extend(_callee_ids(step.first))
        found.extend(_callee_ids(node.then))
    elif isinstance(node, IRQiecHandle):
        found.extend(_callee_ids(node.computation))
    elif isinstance(node, IRQiecCase):
        for branch in node.branches:
            found.extend(_callee_ids(branch.body))
    elif isinstance(node, IRQiecIf):
        found.extend(_callee_ids(node.then))
        found.extend(_callee_ids(node.otherwise))
    elif isinstance(node, IRQiecNewInstance):
        found.extend(_callee_ids(node.body))
    return found


def carried_computations(
    ir: IRProgram, target: str
) -> tuple[IRQiecNamedComputation, ...]:
    """The computations a static target's output defines.

    Every computation the target can represent is defined, called or
    not, so a module's entry points stay in the output where the
    target has a form for them; one the program never calls and the
    target cannot represent is left out, since it is no part of the
    program the output denotes. A needed computation the target
    cannot represent is refused by
    `refuse_static_gaps`
    before this is read.

    Parameters
    ----------
    ir : IRProgram
        The lowered root.
    target : str
        The renderer's target label.

    Returns
    -------
    tuple[IRQiecNamedComputation, ...]
        The definable computations, in module order.
    """
    public_target = qiec_target_name(target)
    gaps = frozenset(
        diagnostic.computation
        for diagnostic in analyze_qiec_capabilities(ir.module, public_target)
        if diagnostic.computation is not None
    )
    return tuple(
        item for item in _runtime_computations(ir.module) if item.name not in gaps
    )


def refuse_static_gaps(ir: IRProgram, target: str) -> None:
    """Refuse a lowered root whose needed computations a static target lacks.

    Parameters
    ----------
    ir : IRProgram
        The lowered root.
    target : str
        The renderer's target label.

    Raises
    ------
    UnsupportedConstruct
        If the target lacks a capability a needed computation uses.
    """
    public_target = qiec_target_name(target)
    needed = frozenset(item.id.text for item in needed_computations(ir))
    diagnostics = tuple(
        diagnostic
        for diagnostic in analyze_qiec_capabilities(ir.module, public_target)
        if diagnostic.computation is None
        or _computation_id(ir.module, diagnostic.computation) in needed
    )
    if diagnostics:
        raise UnsupportedConstruct(
            f"qvr-{public_target}", [diagnostic.kind for diagnostic in diagnostics]
        )


def _computation_id(module: IRQiecModule, name: str) -> str:
    """The identity of a module computation.

    Parameters
    ----------
    module : IRQiecModule
        The module.
    name : str
        The computation's name.

    Returns
    -------
    str
        Its identity text.
    """
    return next(item.id.text for item in module.computations if item.name == name)


def render_computations_static(
    sb: panproto.SchemaBuilder,
    ir: IRProgram,
    *,
    target: str,
    destination: str,
) -> None:
    """Render the computations a static target needs into its block.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    ir : IRProgram
        The lowered root.
    target : str
        The renderer's target label.
    destination : str
        The schema vertex the definitions are placed under.

    Raises
    ------
    UnsupportedConstruct
        If the target lacks a capability a needed computation uses.
    """
    public_target = qiec_target_name(target)
    refuse_static_gaps(ir, target)
    carried = carried_computations(ir, target)
    if not carried:
        if public_target not in {"bugs", "jags"}:
            return
        source, source_root_kind = "model {\nqiec_declarations <- 0\n}\n", "model_block"
    else:
        source, source_root_kind = _static_definitions(carried, public_target)
    grammar = "bugs" if public_target == "bugs" else public_target
    _graft_parsed_children(
        sb,
        grammar=grammar,
        source=source,
        source_root_kind=source_root_kind,
        destination=destination,
        prefix=f"qiec_{public_target}",
    )


def canonical_operations(module: IRQiecModule) -> tuple[str, str]:
    """The identities of ``Random.sample`` and ``Score.add`` in a module.

    Parameters
    ----------
    module : IRQiecModule
        The module.

    Returns
    -------
    tuple[str, str]
        The sample operation's and the add operation's identity texts.

    Raises
    ------
    UnsupportedConstruct
        If the module declares neither effect.
    """
    found: dict[str, str] = {}
    for effect in module.effects:
        for operation in effect.operations:
            if (effect.ref.name, operation.name) in (
                ("Random", "sample"),
                ("Score", "add"),
            ):
                found[effect.ref.name] = operation.id.text
    if "Random" not in found or "Score" not in found:
        raise UnsupportedConstruct("qvr-lower", ["qiec:call:canonical-effects"])
    return found["Random"], found["Score"]


#: How each host family spells the closed static environment a model
#: body evaluates a checked value under, binds a name, and aliases a
#: model local under the runtime's local naming.
_VALUE_HOSTS: dict[str, tuple[str, str, str]] = {
    "python": (
        "qiec_static = _qvr_qiec_static_environment([], [])\n",
        "{name} = {value}\n",
        "{alias} = {name}\n",
    ),
    "julia": (
        "qiec_static = _qvr_qiec_static_environment([], [])\n",
        "{name} = {value}\n",
        "{alias} = {name}\n",
    ),
    "javascript": (
        "var qiec_static = _qvr_qiec_static_environment([], []);\n",
        "var {name} = {value};\n",
        "var {alias} = {name};\n",
    ),
    "scheme": (
        "(define qiec-static (_qvr-qiec-static-environment '() '()))\n",
        "(define {name} {value})\n",
        "(define {alias} {name})\n",
    ),
}


def _value_locals(node: IRQiecValue) -> tuple[str, ...]:
    """The locals a checked value reads, in first-occurrence order.

    Parameters
    ----------
    node : IRQiecValue
        The value.

    Returns
    -------
    tuple[str, ...]
        The names of every `IRQiecVar` inside it, each once.
    """
    found: list[str] = []

    def walk(item: object) -> None:
        """Visit one node of the value.

        Parameters
        ----------
        item : object
            A value, a container of values, or a scalar.
        """
        if isinstance(item, IRQiecVar):
            if item.local.name not in found:
                found.append(item.local.name)
            return
        if isinstance(item, dx.Model):
            for field in type(item).__field_specs__:
                walk(getattr(item, field))
            return
        if isinstance(item, (tuple, list)):
            for member in item:
                walk(member)

    walk(node)
    return tuple(found)


def value_argument_source(
    host: str,
    target: str,
    name: str,
    node: IRQiecValue,
    model_names: Callable[[str], str],
    *,
    body: str,
    defined: set[str],
) -> str:
    """The statements binding a checked value under a name in a model body.

    The value is rendered as the callee's runtime reads it, with the
    program's locals aliased under the runtime's local naming and the
    closed static environment the model body evaluates under. The
    environment and each alias are bound once per body, since every
    host admits one definition of a name in a body.

    Parameters
    ----------
    host : str
        The host family: ``python``, ``julia``, ``javascript``, or
        ``scheme``.
    target : str
        The transpile target.
    name : str
        The name the value is bound to.
    node : IRQiecValue
        The value.
    model_names : Callable[[str], str]
        How the model body spells a program local.
    body : str
        The body the value is bound in, as the key of ``defined``.
    defined : set[str]
        The bindings every body holds, as ``<body>:<name>`` entries,
        extended with the ones this value binds.

    Returns
    -------
    str
        The host statements, ending in a newline.
    """
    environment, bind, alias = _VALUE_HOSTS[host]
    render = {
        "python": _python_value,
        "julia": _julia_value,
        "javascript": _javascript_value,
        "scheme": _scheme_value,
    }[host]
    source = ""
    if f"{body}:{_ENVIRONMENT_MARKER}" not in defined:
        defined.add(f"{body}:{_ENVIRONMENT_MARKER}")
        source += environment
    for local in _value_locals(node):
        local_name = _local_name(local)
        if f"{body}:{local_name}" in defined:
            continue
        defined.add(f"{body}:{local_name}")
        source += alias.format(alias=local_name, name=model_names(local))
    return source + bind.format(name=name, value=render(target, node))


#: The name under which a body's ``defined`` set records that the
#: closed static environment is bound.
_ENVIRONMENT_MARKER = "<static environment>"


def python_call_source(
    node: IRCall,
    module: IRQiecModule,
    argument_names: tuple[str, ...],
    operations: str,
) -> str:
    """The Python statement calling a computation from the model.

    Parameters
    ----------
    node : IRCall
        The call.
    module : IRQiecModule
        The module the callee belongs to.
    argument_names : tuple[str, ...]
        The names the arguments are bound to in the model body.
    operations : str
        The name bound to the native operation table.

    Returns
    -------
    str
        ``<name> = qiec_<callee>(<arguments>, qiec_static_arguments=...,
        qiec_operations=<operations>)``.
    """
    callee = next(item for item in module.computations if item.name == node.callee)
    arguments = "".join(f"{name}, " for name in argument_names)
    statics = _ir_data(node.static_arguments)
    return (
        f"{node.name} = {_function_name(callee)}({arguments}"
        f"qiec_static_arguments={statics!r}, qiec_operations={operations})\n"
    )


def python_operations_source(
    node: IRCall, module: IRQiecModule, operations: str
) -> str:
    """The Python statement binding the native operation table.

    Parameters
    ----------
    node : IRCall
        A call of the model, which names the program's instances.
    module : IRQiecModule
        The module.
    operations : str
        The name to bind the table to.

    Returns
    -------
    str
        ``<operations> = _qvr_qiec_native_operations(...)``.
    """
    sample, add = canonical_operations(module)
    return (
        f"{operations} = _qvr_qiec_native_operations({node.random_instance!r}, "
        f"{sample!r}, {node.score_instance!r}, {add!r})\n"
    )


def emit_call_python(
    py: PyCtx,
    body: str,
    node: IRCall,
    module: IRQiecModule,
    bound: set[str],
) -> None:
    """Place a call of a computation in a Python model body.

    The first call binds the native operation table, under which the
    callee's draws are the host's sample sites and its scores the
    host's factor terms; an argument that is not a bare name is bound
    first, so the call reads names alone.

    Parameters
    ----------
    py : PyCtx
        The emission context.
    body : str
        The block vertex the statements are placed in.
    node : IRCall
        The call.
    module : IRQiecModule
        The module the callee belongs to.
    bound : set[str]
        The bodies whose operation table is already bound, extended
        with ``body`` once its table is placed, and the names checked
        value arguments have bound in each body.
    """
    names: list[str] = []
    source = ""
    for position, argument in enumerate(node.arguments):
        if isinstance(argument, LetExprVar):
            names.append(argument.name)
            continue
        bound_name = f"{node.name}_arg{position}"
        if isinstance(argument, IRQiecValueExpr):
            source += value_argument_source(
                "python",
                py.target,
                bound_name,
                argument.value,
                lambda local: local,
                body=body,
                defined=bound,
            )
            names.append(bound_name)
            continue
        py.e(
            body,
            assignment(
                py, lhs_name=bound_name, rhs=render_let_expr_python(py, argument)
            ),
            "child_of",
        )
        names.append(bound_name)
    operations = "_qvr_qiec_operations"
    if body not in bound:
        bound.add(body)
        source += python_operations_source(node, module, operations)
    source += python_call_source(node, module, tuple(names), operations)
    graft_python_statements(py.builder, source, body, f"qiec_call_{node.name}")


#: The Julia closures a target's model body hands the native operation
#: table: a draw of a distribution under a site name, and a scored
#: weight. Both are placed inside the model so they close over the
#: host's per-evaluation state.
_JULIA_NATIVE_CLOSURES: dict[str, tuple[str, str]] = {
    "turing": (
        "(label, distribution) -> begin\n"
        "    vn = Turing.DynamicPPL.VarName{Symbol(label)}()\n"
        "    prefixed = Turing.DynamicPPL.prefix(__model__.context, vn)\n"
        "    if Turing.DynamicPPL.contextual_isassumption(__model__.context, prefixed)\n"
        "        value, __varinfo__ = Turing.DynamicPPL.tilde_assume!!("
        "__model__.context, distribution, vn, "
        "Turing.DynamicPPL.VarNamedTuples.NoTemplate(), __varinfo__)\n"
        "    else\n"
        "        supplied = Turing.DynamicPPL.getconditioned_nested(__model__.context, prefixed)\n"
        "        value, __varinfo__ = Turing.DynamicPPL.tilde_observe!!("
        "__model__.context, distribution, supplied, vn, "
        "Turing.DynamicPPL.VarNamedTuples.NoTemplate(), __varinfo__)\n"
        "    end\n"
        "    value\n"
        "end",
        "(label, weight) -> begin\n    Turing.@addlogprob! weight\n    nothing\nend",
    ),
    "gen": (
        "(label, distribution) -> @trace("
        "distribution.distribution(distribution.arguments...), Symbol(label))",
        "(label, weight) -> begin\n"
        "    @trace(_qvr_qiec_factor(weight), :qvr_factor => Symbol(label))\n"
        "    nothing\n"
        "end",
    ),
}


def julia_call_source(
    node: IRCall,
    module: IRQiecModule,
    argument_names: tuple[str, ...],
    operations: str,
) -> str:
    """The Julia statement calling a computation from a model body.

    Parameters
    ----------
    node : IRCall
        The call.
    module : IRQiecModule
        The module the callee belongs to.
    argument_names : tuple[str, ...]
        The names the arguments are bound to in the model body.
    operations : str
        The name bound to the native operation table.

    Returns
    -------
    str
        ``<name> = qiec_<callee>(<arguments>; qiec_static_arguments=...,
        qiec_operations=<operations>)``.
    """
    callee = next(item for item in module.computations if item.name == node.callee)
    arguments = ", ".join(argument_names)
    statics = _julia_data(_ir_data(node.static_arguments))
    return (
        f"{node.name} = {_function_name(callee)}({arguments}; "
        f"qiec_static_arguments={statics}, qiec_operations={operations})\n"
    )


def julia_operations_source(
    node: IRCall, module: IRQiecModule, operations: str, target: str
) -> str:
    """The Julia statement binding the native operation table.

    Parameters
    ----------
    node : IRCall
        A call of the model, which names the program's instances.
    module : IRQiecModule
        The module.
    operations : str
        The name to bind the table to.
    target : str
        The Julia target, which supplies the draw and score closures.

    Returns
    -------
    str
        ``<operations> = _qvr_qiec_native_operations(...)``.
    """
    sample, add = canonical_operations(module)
    draw, score = _JULIA_NATIVE_CLOSURES[target]
    return (
        f"{operations} = _qvr_qiec_native_operations("
        f"{_julia_string(node.random_instance)}, {_julia_string(sample)}, "
        f"{_julia_string(node.score_instance)}, {_julia_string(add)}, "
        f"{draw}, {score})\n"
    )


def emit_call_julia(
    sb: panproto.SchemaBuilder,
    node: IRCall,
    module: IRQiecModule,
    bound: set[str],
    *,
    target: str,
    body: str,
    bind_argument: Callable[[str, LetExprNode], None],
    place: Callable[[str], None],
) -> None:
    """Place a call of a computation in a Julia model body.

    The first call in a body binds the native operation table, whose
    closures draw through the host's own tracing; an argument that is
    not a bare name is bound first, so the call reads names alone.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    node : IRCall
        The call.
    module : IRQiecModule
        The module the callee belongs to.
    bound : set[str]
        The bodies whose operation table is already bound, extended
        with ``body`` once its table is placed, and the names checked
        value arguments have bound in each body.
    target : str
        The Julia target.
    body : str
        The body the statements belong to, as the key of ``bound``.
    bind_argument : Callable[[str, LetExprNode], None]
        Places ``<name> = <expression>`` in the body, for an argument
        that is not a bare name.
    place : Callable[[str], None]
        Places one statement vertex in the body, in order.
    """
    names: list[str] = []
    source = ""
    for position, argument in enumerate(node.arguments):
        if isinstance(argument, LetExprVar):
            names.append(argument.name)
            continue
        bound_name = f"{node.name}_arg{position}"
        if isinstance(argument, IRQiecValueExpr):
            source += value_argument_source(
                "julia",
                target,
                bound_name,
                argument.value,
                lambda local: local,
                body=body,
                defined=bound,
            )
            names.append(bound_name)
            continue
        bind_argument(bound_name, argument)
        names.append(bound_name)
    operations = "_qvr_qiec_operations"
    if body not in bound:
        bound.add(body)
        source += julia_operations_source(node, module, operations, target)
    source += julia_call_source(node, module, tuple(names), operations)
    for statement in _graft_parsed_children(
        sb,
        grammar="julia",
        source=source,
        source_root_kind="source_file",
        destination=None,
        prefix=f"qiec_call_{node.name}",
    ):
        place(statement)


def scheme_call_source(
    node: IRCall,
    module: IRQiecModule,
    argument_names: tuple[str, ...],
    operations: str,
) -> str:
    """The Scheme form calling a computation from a model body.

    Parameters
    ----------
    node : IRCall
        The call.
    module : IRQiecModule
        The module the callee belongs to.
    argument_names : tuple[str, ...]
        The names the arguments are bound to in the model body.
    operations : str
        The name bound to the native operation table.

    Returns
    -------
    str
        ``(define <name> (qiec_<callee> <arguments> <statics> '() '()
        <operations>))``.
    """
    callee = next(item for item in module.computations if item.name == node.callee)
    arguments = "".join(f"{name} " for name in argument_names)
    statics = _scheme_data(_ir_data(node.static_arguments))
    return (
        f"(define {node.name} ({_function_name(callee)} {arguments}"
        f"{statics} '() '() {operations}))\n"
    )


def scheme_operations_source(
    node: IRCall, module: IRQiecModule, operations: str
) -> str:
    """The Scheme form binding the native operation table.

    Parameters
    ----------
    node : IRCall
        A call of the model, which names the program's instances.
    module : IRQiecModule
        The module.
    operations : str
        The name to bind the table to.

    Returns
    -------
    str
        ``(define <operations> (_qvr-qiec-native-operations ...))``.
    """
    sample, add = canonical_operations(module)
    return (
        f"(define {operations} (_qvr-qiec-native-operations "
        f"{_scheme_string(node.random_instance)} {_scheme_string(sample)} "
        f"{_scheme_string(node.score_instance)} {_scheme_string(add)}))\n"
    )


def graft_scheme_forms(
    sb: panproto.SchemaBuilder, source: str, prefix: str
) -> list[str]:
    """Parse Scheme forms into a schema for the caller to place.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    source : str
        The forms.
    prefix : str
        A prefix keeping the parsed vertices' identities apart.

    Returns
    -------
    list[str]
        The top-level form vertices, in source order.
    """
    return _graft_parsed_children(
        sb,
        grammar="scheme",
        source=source,
        source_root_kind="program",
        destination=None,
        prefix=prefix,
    )


def javascript_call_source(
    node: IRCall,
    module: IRQiecModule,
    argument_names: tuple[str, ...],
    operations: str,
) -> str:
    """The WebPPL statement calling a computation from a model body.

    Parameters
    ----------
    node : IRCall
        The call.
    module : IRQiecModule
        The module the callee belongs to.
    argument_names : tuple[str, ...]
        The names the arguments are bound to in the model body.
    operations : str
        The name bound to the native operation table.

    Returns
    -------
    str
        ``var <name> = qiec_<callee>(<arguments>, <statics>, {}, {},
        <operations>);``.
    """
    callee = next(item for item in module.computations if item.name == node.callee)
    arguments = "".join(f"{name}, " for name in argument_names)
    statics = json.dumps(_ir_data(node.static_arguments), ensure_ascii=False)
    return (
        f"var {node.name} = {_function_name(callee)}({arguments}"
        f"{statics}, {{}}, {{}}, {operations});\n"
    )


def javascript_operations_source(
    node: IRCall, module: IRQiecModule, operations: str
) -> str:
    """The WebPPL statement binding the native operation table.

    Parameters
    ----------
    node : IRCall
        A call of the model, which names the program's instances.
    module : IRQiecModule
        The module.
    operations : str
        The name to bind the table to.

    Returns
    -------
    str
        ``var <operations> = _qvr_qiec_native_operations(...);``.
    """
    sample, add = canonical_operations(module)
    return (
        f"var {operations} = _qvr_qiec_native_operations("
        f"{json.dumps(node.random_instance)}, {json.dumps(sample)}, "
        f"{json.dumps(node.score_instance)}, {json.dumps(add)});\n"
    )


def graft_javascript_statements(
    sb: panproto.SchemaBuilder, source: str, destination: str, prefix: str
) -> None:
    """Parse JavaScript statements and place them in a block.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    source : str
        The statements.
    destination : str
        The block vertex the statements are placed in.
    prefix : str
        A prefix keeping the parsed vertices' identities apart.
    """
    _graft_parsed_children(
        sb,
        grammar="javascript",
        source=source,
        source_root_kind="program",
        destination=destination,
        prefix=prefix,
    )


def graft_python_statements(
    sb: panproto.SchemaBuilder, source: str, destination: str, prefix: str
) -> None:
    """Parse Python statements and place them in a body.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    source : str
        The statements.
    destination : str
        The block vertex the statements are placed in.
    prefix : str
        A prefix keeping the parsed vertices' identities apart.
    """
    _graft_parsed_children(
        sb,
        grammar="python",
        source=source,
        source_root_kind="module",
        destination=destination,
        prefix=prefix,
    )


def _graft_parsed_children(
    sb: panproto.SchemaBuilder,
    *,
    grammar: str,
    source: str,
    source_root_kind: str,
    destination: str | None,
    prefix: str,
) -> list[str]:
    """Parse source and graft one root's children into an existing schema.

    Parameters
    ----------
    sb : panproto.SchemaBuilder
        The schema being built.
    grammar : str
        The grammar the source is parsed with.
    source : str
        The source text.
    source_root_kind : str
        The kind of the parsed root whose children are grafted.
    destination : str | None
        The vertex the children hang from, or None to leave them for
        the caller to place.
    prefix : str
        A prefix keeping the grafted vertices' identities apart.

    Returns
    -------
    list[str]
        The grafted top-level vertices, in source order.
    """
    schema = parser_registry().parse_with_protocol(
        grammar, source.encode(), f"<{prefix}>"
    )
    roots = [vertex.id for vertex in schema.vertices if vertex.kind == source_root_kind]
    if len(roots) != 1:
        raise RuntimeError(
            f"QIEC {grammar} fragment has {len(roots)} {source_root_kind!r} roots"
        )
    source_root = roots[0]
    top_level = [edge.tgt for edge in schema.edges if edge.src == source_root]
    top_level.sort(
        key=lambda vertex: next(
            (
                int(constraint.value)
                for constraint in schema.constraints_for(vertex)
                if constraint.sort == "start-byte"
            ),
            0,
        )
    )
    subtree = _reachable(schema, tuple(top_level))
    id_map = {old: f"{prefix}_{position}" for position, old in enumerate(subtree)}
    kinds = {vertex.id: vertex.kind for vertex in schema.vertices}
    for old, new in id_map.items():
        sb.vertex(new, kinds[old])
        for constraint in schema.constraints_for(old):
            sb.constraint(new, constraint.sort, constraint.value)
    for edge in schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    if destination is not None:
        for child in top_level:
            sb.edge(destination, id_map[child], "child_of")
    return [id_map[child] for child in top_level]


def _reachable(schema: panproto.Schema, roots: tuple[str, ...]) -> tuple[str, ...]:
    seen = set(roots)
    ordered = list(roots)
    cursor = 0
    while cursor < len(ordered):
        source = ordered[cursor]
        cursor += 1
        for edge in schema.edges:
            if edge.src == source and edge.tgt not in seen:
                seen.add(edge.tgt)
                ordered.append(edge.tgt)
    return tuple(ordered)


def _safe_name(name: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not candidate or candidate[0].isdigit():
        candidate = "_" + candidate
    return candidate


def _local_name(name: str) -> str:
    """Keep authored locals disjoint from keywords and the generated ABI."""
    return "qv_" + _safe_name(name)


def _function_name(computation: IRQiecNamedComputation) -> str:
    return "qiec_" + _safe_name(computation.name)


def _body_name(computation: IRQiecNamedComputation) -> str:
    """The generated function that builds a computation's body unrun.

    A call needs the callee's computation as data the caller's handlers
    can interpret, not its final value, so every named computation is
    generated twice: this builder, and the public entry that runs it.

    Parameters
    ----------
    computation
        The named computation.

    Returns
    -------
    str
        The builder's host-language name.
    """
    return "qiec_body_" + _safe_name(computation.name)


def _clause_name(handler: IRQiecHandlerDef, operation: str) -> str:
    """The generated function holding one authored clause body.

    Parameters
    ----------
    handler
        The authored handler.
    operation
        The clause's operation, by display name.

    Returns
    -------
    str
        The clause function's host-language name.
    """
    return "qiec_clause_" + _safe_name(handler.name) + "_" + _safe_name(operation)


def _return_clause_name(handler: IRQiecHandlerDef) -> str:
    """The generated function holding an authored return clause body.

    Parameters
    ----------
    handler
        The authored handler.

    Returns
    -------
    str
        The return clause function's host-language name.
    """
    return "qiec_return_" + _safe_name(handler.name)


def _computation_by_id(module: IRQiecModule, callee: str) -> IRQiecNamedComputation:
    """Find a named computation by stable identity.

    Parameters
    ----------
    module
        The module the call is being rendered from.
    callee
        The callee identity, as :attr:`IRQiecId.text`.

    Returns
    -------
    IRQiecNamedComputation
        The computation a call refers to.
    """
    return next(item for item in module.computations if item.id.text == callee)


def _operation_name(module: IRQiecModule, operation: str) -> str:
    """The display name of an operation, by stable identity.

    Parameters
    ----------
    module
        The module declaring the operation's interface.
    operation
        The operation identity, as :attr:`IRQiecId.text`.

    Returns
    -------
    str
        The declared name, or the identity text when no interface in the
        module declares it.
    """
    for effect in module.effects:
        for item in effect.operations:
            if item.id.text == operation:
                return item.name
    return operation


def _ir_data(value: object) -> object:
    """Project typed QIEC IR to lossless host-runtime metadata.

    Didactic's ``model_dump`` is JSON safe, but walking the model itself lets
    us retain stable identifiers in their canonical ``qiec:*`` spelling.
    """
    if hasattr(value, "text") and value.__class__.__name__ == "IRQiecId":
        return value.text  # type: ignore[attr-defined]
    fields = getattr(type(value), "__field_specs__", None)
    if fields is not None:
        return {name: _ir_data(getattr(value, name)) for name in fields}
    if isinstance(value, dict):
        return {_ir_data(key): _ir_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ir_data(item) for item in value]
    return value


def _dynamic_definitions(module: IRQiecModule, grammar: str, target: str) -> str:
    """Generate the module's entry points and clause bodies for one target.

    Parameters
    ----------
    module : IRQiecModule
        The checked module.
    grammar : str
        The host language.
    target : str
        The public target name, which fixes how distributions are spelled.

    Returns
    -------
    str
        The generated definitions, clause bodies first.
    """
    generators: dict[
        str, Callable[[str, IRQiecNamedComputation, IRQiecModule], str]
    ] = {
        "python": _python_definition,
        "julia": _julia_definition,
        "javascript": _javascript_definition,
        "scheme": _scheme_definition,
    }
    handler_generators: dict[
        str, Callable[[str, IRQiecHandlerDef, IRQiecModule], str]
    ] = {
        "python": _python_authored_handler,
        "julia": _julia_authored_handler,
        "javascript": _javascript_authored_handler,
        "scheme": _scheme_authored_handler,
    }
    # Clause bodies are generated before the computations that install
    # them, so a handler is registered by the time any entry point runs.
    authored = [
        handler_generators[grammar](target, handler, module)
        for handler in module.handlers
        if handler.implementation == "authored"
    ]
    # A WebPPL program binds a name once, so its handler table is one
    # declaration listing every authored handler's definition.
    if grammar == "javascript":
        authored.append(_javascript_authored_table(module))
    return "\n".join(
        (
            *authored,
            *(
                generators[grammar](target, item, module)
                for item in _runtime_computations(module)
            ),
        )
    )


def _handler_grades(module: IRQiecModule, handler_id: str) -> dict[str, str]:
    handler = _handler(module, handler_id)
    return {clause.operation.text: clause.grade for clause in handler.clauses}


def _handler(module: IRQiecModule, handler_id: str):
    return next(item for item in module.handlers if item.id.text == handler_id)


def _address(request: object) -> tuple[object, ...]:
    origin = request.origin.origin  # type: ignore[attr-defined]
    static = SiteId.derive(
        origin.source_protocol, origin.module, origin.structural_path, origin.role
    ).to_data()
    dynamic = tuple(
        (frame.scope, frame.key)
        for frame in request.origin.dynamic_path  # type: ignore[attr-defined]
    )
    resumptions = tuple(request.origin.resumption_path)  # type: ignore[attr-defined]
    return (static, dynamic, resumptions)


def _runtime_request(node: IRQiecPerform) -> dict[str, object]:
    request = node.request
    return {
        "instance": request.instance.text,
        "effect": _ir_data(request.effect),
        "operation": request.operation.text,
        "static_arguments": _ir_data(request.static_arguments),
        "result_type": _ir_data(request.result_type),
        "origin": _ir_data(request.origin),
        "address": _address(request),
    }


def _free_runtime_capture(
    node: IRQiecComputation,
    bound: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return outer locals and attachments captured by ``node``."""
    locals_: set[str] = set()
    attachments: set[str] = set()

    def value(item: IRQiecValue, locally_bound: frozenset[str]) -> None:
        if isinstance(item, IRQiecVar):
            if item.local.name not in locally_bound:
                locals_.add(item.local.name)
        elif isinstance(item, IRQiecConstructorValue):
            for field in item.fields:
                value(field, locally_bound)
        elif isinstance(item, IRQiecAttachmentRef):
            attachments.add(item.attachment.text)
        elif isinstance(item, IRQiecTransportValue):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecPrimitiveApplication):
            for argument in item.arguments:
                value(argument, locally_bound)
        elif isinstance(item, IRQiecTupleValue):
            for component in item.items:
                value(component, locally_bound)
        elif isinstance(item, IRQiecProjection):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecTensorValue):
            for entry in item.items:
                value(entry, locally_bound)
        elif isinstance(item, IRQiecDistributionValue):
            for argument in item.arguments:
                value(argument.value, locally_bound)
        elif isinstance(item, IRQiecLogDensity):
            value(item.sampleable, locally_bound)
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecGather | IRQiecSegmentSum):
            value(item.value, locally_bound)
            value(item.index, locally_bound)
        elif isinstance(item, IRQiecWeightSum):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecKernelMatrix):
            value(item.inputs, locally_bound)
        elif isinstance(item, IRQiecAffineMap):
            value(item.weight, locally_bound)
            value(item.bias, locally_bound)
            for source in item.sources:
                value(source, locally_bound)
        elif isinstance(item, IRQiecTableMap):
            value(item.table, locally_bound)
            value(item.index, locally_bound)
        elif isinstance(item, IRQiecReduction | IRQiecRowwise):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecComprehension):
            value(item.body, locally_bound | {item.binder.name})
        elif isinstance(item, IRQiecLetValue):
            for binding in item.bindings:
                value(binding.value, locally_bound)
                locally_bound = locally_bound | {binding.binder.name}
            value(item.body, locally_bound)

    def computation(item: IRQiecComputation, locally_bound: frozenset[str]) -> None:
        if isinstance(item, IRQiecReturn):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecBind):
            for step in item.steps:
                computation(step.first, locally_bound)
                locally_bound = locally_bound | {step.binder.name}
            computation(item.then, locally_bound)
        elif isinstance(item, IRQiecPerform):
            for argument in item.request.arguments:
                value(argument, locally_bound)
        elif isinstance(item, IRQiecHandle):
            computation(item.computation, locally_bound)
        elif isinstance(item, IRQiecCase):
            value(item.scrutinee, locally_bound)
            for branch in item.branches:
                computation(
                    branch.body,
                    locally_bound | {field.name for field in branch.fields},
                )
        elif isinstance(item, IRQiecIf):
            value(item.condition, locally_bound)
            computation(item.then, locally_bound)
            computation(item.otherwise, locally_bound)
        elif isinstance(item, IRQiecCall):
            for argument in item.arguments:
                value(argument, locally_bound)
        elif isinstance(item, IRQiecResume):
            value(item.value, locally_bound)
        elif isinstance(item, IRQiecNewInstance):
            computation(item.body, locally_bound)

    computation(node, bound)
    return tuple(sorted(locals_)), tuple(sorted(attachments))


def qiec_families_used(ir: IRProgram) -> frozenset[str]:
    """The distribution families the program's QIEC computations construct.

    Parameters
    ----------
    ir : IRProgram
        The lowered program.

    Returns
    -------
    frozenset[str]
        Registry names of every family a ``DistributionValue`` in a
        host-runtime computation body or handler clause names. Program
        computations are left out: the renderer emits them from the
        program's plan, which spells their families itself.
    """
    module = ir.module
    if module is None:
        return frozenset()
    found: set[str] = set()

    def value(item: IRQiecValue) -> None:
        if isinstance(item, IRQiecDistributionValue):
            found.add(item.name)
            for argument in item.arguments:
                value(argument.value)
        elif isinstance(item, IRQiecLogDensity):
            value(item.sampleable)
            value(item.value)
        elif isinstance(item, IRQiecGather | IRQiecSegmentSum):
            value(item.value)
            value(item.index)
        elif isinstance(item, IRQiecWeightSum):
            value(item.value)
        elif isinstance(item, IRQiecKernelMatrix):
            value(item.inputs)
        elif isinstance(item, IRQiecAffineMap):
            value(item.weight)
            value(item.bias)
            for source in item.sources:
                value(source)
        elif isinstance(item, IRQiecTableMap):
            value(item.table)
            value(item.index)
        elif isinstance(item, IRQiecReduction | IRQiecRowwise):
            value(item.value)
        elif isinstance(item, IRQiecComprehension):
            value(item.body)
        elif isinstance(item, IRQiecLetValue):
            for binding in item.bindings:
                value(binding.value)
            value(item.body)
        elif isinstance(item, IRQiecConstructorValue):
            for field in item.fields:
                value(field)
        elif isinstance(item, IRQiecTransportValue):
            value(item.value)
        elif isinstance(item, IRQiecPrimitiveApplication):
            for argument in item.arguments:
                value(argument)
        elif isinstance(item, IRQiecTupleValue | IRQiecTensorValue):
            for entry in item.items:
                value(entry)
        elif isinstance(item, IRQiecProjection):
            value(item.value)

    def computation(item: IRQiecComputation) -> None:
        if isinstance(item, IRQiecReturn):
            value(item.value)
        elif isinstance(item, IRQiecBind):
            for step in item.steps:
                computation(step.first)
            computation(item.then)
        elif isinstance(item, IRQiecPerform):
            for argument in item.request.arguments:
                value(argument)
        elif isinstance(item, IRQiecHandle):
            computation(item.computation)
        elif isinstance(item, IRQiecCase):
            value(item.scrutinee)
            for branch in item.branches:
                computation(branch.body)
        elif isinstance(item, IRQiecIf):
            value(item.condition)
            computation(item.then)
            computation(item.otherwise)
        elif isinstance(item, IRQiecCall):
            for argument in item.arguments:
                value(argument)
        elif isinstance(item, IRQiecResume):
            value(item.value)
        elif isinstance(item, IRQiecNewInstance):
            computation(item.body)

    for named in _runtime_computations(module):
        computation(named.body)
    for handler in module.handlers:
        if handler.implementation != "authored":
            continue
        for clause in handler.clauses:
            if clause.body is not None:
                computation(clause.body)
        if handler.return_clause is not None and handler.return_clause.body is not None:
            computation(handler.return_clause.body)
    return frozenset(found)


def qiec_helper_families_used(ir: IRProgram, target: str) -> frozenset[str]:
    """The families the program's QIEC values need grafted helpers for.

    Parameters
    ----------
    ir : IRProgram
        The lowered program.
    target : str
        The public target name.

    Returns
    -------
    frozenset[str]
        The subset of the families used that ``target`` serves through a
        runtime helper rather than a library distribution.
    """
    return qiec_families_used(ir) & helper_families(qiec_target_name(target))


def qiec_helper_roots(ir: IRProgram, target: str) -> frozenset[str]:
    """The runtime helper roots the program's QIEC values need grafted.

    Parameters
    ----------
    ir : IRProgram
        The lowered program.
    target : str
        The public target name.

    Returns
    -------
    frozenset[str]
        Helper class or function names, as the target's runtime file
        defines them.
    """
    return helper_roots(qiec_target_name(target), qiec_families_used(ir))


def _event_shape(result_type: IRQiecStatic) -> tuple[int, ...] | None:
    """The literal event shape a ``Sampleable`` type spells.

    Parameters
    ----------
    result_type : IRQiecStatic
        The type of a distribution construction.

    Returns
    -------
    tuple[int, ...] or None
        The dimensions of ``Sampleable[Tensor[A](shape)]`` when every
        dimension is a literal; ``None`` for a scalar sample type or a
        shape with an open dimension.
    """
    if not (
        isinstance(result_type, IRQiecTypeApplication)
        and result_type.constructor.name == "Sampleable"
        and len(result_type.arguments) == 1
    ):
        return None
    element = result_type.arguments[0]
    if not (
        isinstance(element, IRQiecTypeApplication)
        and element.constructor.name == "Tensor"
        and len(element.arguments) == 2
        and isinstance(element.arguments[1], IRQiecShapeIndex)
    ):
        return None
    dimensions: list[int] = []
    for dimension in element.arguments[1].dimensions:
        if not (
            isinstance(dimension, IRQiecIndexLiteral)
            and isinstance(dimension.value, int)
        ):
            return None
        dimensions.append(dimension.value)
    return tuple(dimensions)


def _extent_literal(term: IRQiecStatic) -> int:
    """The literal value of an index term a runtime needs as a number.

    Parameters
    ----------
    term : IRQiecStatic
        The index term.

    Returns
    -------
    int
        Its value.

    Raises
    ------
    UnsupportedConstruct
        If the term is not a literal natural number; a host runtime
        cannot size a segment or a comprehension by an open index.
    """
    if isinstance(term, IRQiecIndexLiteral) and isinstance(term.value, int):
        return term.value
    raise UnsupportedConstruct("qvr-lower", ["qiec:open-extent"])


def _spelled(
    target: str,
    node: IRQiecDistributionValue,
    render: Callable[[IRQiecValue], str],
) -> str:
    """Spell a distribution construction on rendered arguments.

    Parameters
    ----------
    target : str
        The public target name.
    node : IRQiecDistributionValue
        The construction.
    render : Callable[[IRQiecValue], str]
        Renders one argument in the host language.

    Returns
    -------
    str
        The host expression.
    """
    arguments = {argument.name: render(argument.value) for argument in node.arguments}
    return spell_distribution(
        target, node.name, arguments, _event_shape(node.result_type)
    ).expression


def _branch_static(rendered: str, identifier: str) -> str:
    """Point a branch body's static environment at the branch's own.

    Only the bare identifier moves; runtime helper names that contain it,
    such as the static-environment constructor a nested call uses, are
    left alone.

    Parameters
    ----------
    rendered
        The rendered branch body.
    identifier
        The host-language spelling of the static environment name.

    Returns
    -------
    str
        The body with every bare occurrence renamed to the branch's
        environment.
    """
    pattern = r"(?<![\w-])" + re.escape(identifier) + r"(?![\w-])"
    replacement = identifier.replace("static", "branch-static", 1)
    if "_" in identifier:
        replacement = identifier.replace("static", "branch_static", 1)
    return re.sub(pattern, replacement, rendered)


def _case_metadata(node: IRQiecCase) -> dict[str, object]:
    return {
        "motive": _ir_data(node.motive),
        "branches": {
            branch.constructor.text: {
                "static_arguments": _ir_data(branch.static_arguments),
                "fields": _ir_data(branch.fields),
                "scope": branch.scope.text,
            }
            for branch in node.branches
        },
    }


_PYTHON_ABI = "qiec_static, qiec_attachments, qiec_handlers, qiec_operations"


def _python_definition(
    target: str, item: IRQiecNamedComputation, module: IRQiecModule
) -> str:
    params = [_local_name(parameter.name) for parameter in item.parameters]
    abi = [
        "qiec_static_arguments=None",
        "qiec_attachments=None",
        "qiec_handlers=None",
        "qiec_operations=None",
    ]
    body = _python_computation(target, item.body, module)
    return (
        f"def {_body_name(item)}({', '.join((*params, _PYTHON_ABI))}):\n"
        f"    return {body}\n"
        f"def {_function_name(item)}({', '.join((*params, *abi))}):\n"
        f"    qiec_static = _qvr_qiec_static_environment({_ir_data(item.telescope)!r}, qiec_static_arguments)\n"
        "    qiec_attachments = {} if qiec_attachments is None else qiec_attachments\n"
        "    qiec_handlers = {} if qiec_handlers is None else qiec_handlers\n"
        "    qiec_operations = {} if qiec_operations is None else qiec_operations\n"
        f"    return _qvr_qiec_run(lambda: {_body_name(item)}({', '.join((*params, _PYTHON_ABI))}), qiec_operations)\n"
    )


def _python_authored_handler(
    target: str, handler: IRQiecHandlerDef, module: IRQiecModule
) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        if clause.body is None:
            continue
        params = ", ".join(_local_name(local.name) for local in clause.parameters)
        unpack = f"    ({params},) = qiec_request['arguments']\n" if params else ""
        lines.append(
            f"def {name}(qiec_request, qiec_resume, qiec_context):\n"
            "    qiec_static = qiec_context['static']\n"
            "    qiec_attachments = qiec_context['attachments']\n"
            "    qiec_handlers = qiec_context['handlers']\n"
            "    qiec_operations = qiec_context['operations']\n"
            f"{unpack}"
            f"    return {_python_computation(target, clause.body, module)}\n"
        )
        operations.append(f"{clause.operation.text!r}: {{'invoke': {name}}}")
    return_entry = "None"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"def {name}({binder}, qiec_context):\n"
            "    qiec_static = qiec_context['static']\n"
            "    qiec_attachments = qiec_context['attachments']\n"
            "    qiec_handlers = qiec_context['handlers']\n"
            "    qiec_operations = qiec_context['operations']\n"
            f"    return {_python_computation(target, handler.return_clause.body, module)}\n"
        )
        return_entry = name
    lines.append(
        f"_qvr_qiec_authored[{handler.id.text!r}] = {{"
        f"'operations': {{{', '.join(operations)}}}, "
        f"'return': {return_entry}, 'duplicable_context': True}}\n"
    )
    return "".join(lines)


def _python_computation(
    target: str, node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"_qvr_qiec_pure({_python_value(target, node.value)})"
    if isinstance(node, IRQiecBind):
        rendered = _python_computation(target, node.then, module)
        for position in range(len(node.steps) - 1, -1, -1):
            step = node.steps[position]
            binder = _local_name(step.binder.name)
            rest = IRQiecBind(steps=node.steps[position + 1 :], then=node.then)
            capture_names, attachment_ids = _free_runtime_capture(
                rest, frozenset({step.binder.name})
            )
            captures = (
                "{"
                + ", ".join(f"{name!r}: {_local_name(name)}" for name in capture_names)
                + "}"
            )
            rendered = (
                f"_qvr_qiec_bind({_python_computation(target, step.first, module, tail=False)}, "
                f"lambda {binder}: {rendered}, "
                f"_qvr_qiec_capture({captures}, {attachment_ids!r}, qiec_attachments))"
            )
        return rendered
    if isinstance(node, IRQiecPerform):
        args = ", ".join(
            _python_value(target, value) for value in node.request.arguments
        )
        tuple_args = f"({args},)" if len(node.request.arguments) == 1 else f"({args})"
        request = _runtime_request(node)
        request["arguments"] = None
        rendered = repr(request).replace(
            "'arguments': None", f"'arguments': {tuple_args}"
        )
        return f"_qvr_qiec_effect({rendered}, qiec_static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return (
            f"_qvr_qiec_handle(lambda: {_python_computation(target, node.computation, module)}, "
            f"{node.instance.text!r}, {_ir_data(handler)!r}, "
            f"{_ir_data(node.static_arguments)!r}, qiec_static, qiec_handlers, "
            "qiec_attachments, qiec_operations)"
        )
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_python_value(target, argument)}, " for argument in node.arguments
        )
        return (
            f"_qvr_qiec_enter_call({callee.name!r}, lambda: {_body_name(callee)}("
            f"{arguments}"
            f"_qvr_qiec_static_environment({_ir_data(callee.telescope)!r}, "
            f"_qvr_qiec_specialize({_ir_data(node.static_arguments)!r}, qiec_static)), "
            f"qiec_attachments, qiec_handlers, qiec_operations), {tail!r})"
        )
    if isinstance(node, IRQiecIf):
        return (
            f"_qvr_qiec_if({_python_value(target, node.condition)}, "
            f"lambda: {_python_computation(target, node.then, module, tail=tail)}, "
            f"lambda: {_python_computation(target, node.otherwise, module, tail=tail)})"
        )
    if isinstance(node, IRQiecResume):
        return f"_qvr_qiec_resume(qiec_resume, {_python_value(target, node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return f"_qvr_qiec_instance(lambda: {_python_computation(target, node.body, module)})"
    if isinstance(node, IRQiecCase):
        branches = []
        for branch in node.branches:
            params = ", ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _python_computation(target, branch.body, module), "qiec_static"
            )
            separator = ", " if params else ""
            branches.append(
                f"{branch.constructor.text!r}: lambda qiec_branch_static{separator}{params}: {body}"
            )
        return (
            f"_qvr_qiec_case({_python_value(target, node.scrutinee)}, "
            f"{{{', '.join(branches)}}}, {_case_metadata(node)!r}, qiec_static)"
        )
    raise TypeError(f"unknown QIEC computation {node!r}")


def _python_value(target: str, node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"_qvr_qiec_value({_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _python_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        fields = ", ".join(_python_value(target, field) for field in node.fields)
        fields = f"({fields},)" if len(node.fields) == 1 else f"({fields})"
        return (
            f"_qvr_qiec_constructor({node.constructor.text!r}, "
            f"{_ir_data(node.static_arguments)!r}, {fields}, "
            f"{_ir_data(node.result_type)!r}, qiec_static)"
        )
    if isinstance(node, IRQiecEvidenceValue):
        return f"_qvr_qiec_evidence({_ir_data(node.evidence)!r}, qiec_static)"
    if isinstance(node, IRQiecAttachmentRef):
        return (
            f"_qvr_qiec_attachment(qiec_attachments, {node.attachment.text!r}, "
            f"{_ir_data(node.type)!r}, qiec_static)"
        )
    if isinstance(node, IRQiecTransportValue):
        return (
            f"_qvr_qiec_transport({_ir_data(node.evidence)!r}, "
            f"{_python_value(target, node.value)}, {_ir_data(node.target_type)!r}, qiec_static)"
        )
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = ", ".join(
            _python_value(target, argument) for argument in node.arguments
        )
        return f"_qvr_qiec_primitive({node.name!r}, ({arguments},))"
    if isinstance(node, IRQiecTupleValue):
        items = ", ".join(_python_value(target, item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecProjection):
        return (
            f"_qvr_qiec_project({_python_value(target, node.value)}, {node.position})"
        )
    if isinstance(node, IRQiecTensorValue):
        items = ", ".join(_python_value(target, item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecDistributionValue):
        return _spelled(target, node, lambda item: _python_value(target, item))
    if isinstance(node, IRQiecLogDensity):
        return (
            f"_qvr_qiec_log_density({_python_value(target, node.sampleable)}, "
            f"{_python_value(target, node.value)})"
        )
    if isinstance(node, IRQiecSiteValue):
        return repr(node.label)
    if isinstance(node, IRQiecGather):
        return (
            f"_qvr_qiec_gather({_python_value(target, node.value)}, "
            f"{_python_value(target, node.index)})"
        )
    if isinstance(node, IRQiecWeightSum):
        return f"_qvr_qiec_weight_sum({_python_value(target, node.value)})"
    if isinstance(node, IRQiecSegmentSum):
        return (
            f"_qvr_qiec_segment_sum({_python_value(target, node.value)}, "
            f"{_python_value(target, node.index)}, {_extent_literal(node.groups)})"
        )
    if isinstance(node, IRQiecReduction):
        return (
            f"_qvr_qiec_reduce({node.operator!r}, {_python_value(target, node.value)})"
        )
    if isinstance(node, IRQiecRowwise):
        return (
            f"_qvr_qiec_rowwise({node.operator!r}, {_python_value(target, node.value)})"
        )
    if isinstance(node, IRQiecComprehension):
        binder = _local_name(node.binder.name)
        return (
            f"tuple({_python_value(target, node.body)} for {binder} in "
            f"range({_extent_literal(node.extent)}))"
        )
    if isinstance(node, IRQiecLetValue):
        # Assignment expressions bind in order and are read by the later
        # bindings and the body; the tuple keeps the term one level deep
        # however many bindings it holds.
        bindings = "".join(
            f"({_local_name(binding.binder.name)} := "
            f"{_python_value(target, binding.value)}), "
            for binding in node.bindings
        )
        return f"({bindings}{_python_value(target, node.body)})[-1]"
    raise TypeError(f"unknown QIEC value {node!r}")


def _python_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "None"
    if isinstance(node, IRQiecBytesLiteral):
        return f"bytes.fromhex({node.value!r})"
    if isinstance(node, IRQiecTupleLiteral):
        items = ", ".join(_python_literal(item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    return repr(node.value)  # type: ignore[attr-defined]


_JULIA_ABI = "qiec_static, qiec_attachments, qiec_handlers, qiec_operations"


def _julia_definition(
    target: str, item: IRQiecNamedComputation, module: IRQiecModule
) -> str:
    params = ", ".join(_local_name(parameter.name) for parameter in item.parameters)
    separator = "; " if params else "; "
    body_params = f"{params}, {_JULIA_ABI}" if params else _JULIA_ABI
    return (
        f"function {_body_name(item)}({body_params})\n"
        f"    return {_julia_computation(target, item.body, module)}\n"
        "end\n"
        f"function {_function_name(item)}({params}{separator}qiec_attachments=Dict(), "
        "qiec_handlers=Dict(), qiec_operations=Dict(), qiec_static_arguments=nothing)\n"
        f"    qiec_static = _qvr_qiec_static_environment({_julia_data(_ir_data(item.telescope))}, qiec_static_arguments)\n"
        f"    return _qvr_qiec_run(() -> {_body_name(item)}({body_params}), qiec_operations)\n"
        "end\n"
    )


def _julia_authored_handler(
    target: str, handler: IRQiecHandlerDef, module: IRQiecModule
) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        if clause.body is None:
            continue
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        params = [_local_name(local.name) for local in clause.parameters]
        unpack = "".join(
            f'    {param} = qiec_request["arguments"][{index + 1}]\n'
            for index, param in enumerate(params)
        )
        lines.append(
            f"function {name}(qiec_request, qiec_resume, qiec_context)\n"
            '    qiec_static = qiec_context["static"]\n'
            '    qiec_attachments = qiec_context["attachments"]\n'
            '    qiec_handlers = qiec_context["handlers"]\n'
            '    qiec_operations = qiec_context["operations"]\n'
            f"{unpack}"
            f"    return {_julia_computation(target, clause.body, module)}\n"
            "end\n"
        )
        operations.append(
            f'{_julia_string(clause.operation.text)} => Dict("invoke" => {name})'
        )
    return_entry = "nothing"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"function {name}({binder}, qiec_context)\n"
            '    qiec_static = qiec_context["static"]\n'
            '    qiec_attachments = qiec_context["attachments"]\n'
            '    qiec_handlers = qiec_context["handlers"]\n'
            '    qiec_operations = qiec_context["operations"]\n'
            f"    return {_julia_computation(target, handler.return_clause.body, module)}\n"
            "end\n"
        )
        return_entry = name
    lines.append(
        f"_qvr_qiec_authored[{_julia_string(handler.id.text)}] = Dict("
        f'"operations" => Dict{{String, Any}}({", ".join(operations)}), '
        f'"return" => {return_entry}, "duplicable_context" => true)\n'
    )
    return "".join(lines)


def _julia_computation(
    target: str, node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"_qvr_qiec_pure({_julia_value(target, node.value)})"
    if isinstance(node, IRQiecBind):
        rendered = _julia_computation(target, node.then, module)
        for position in range(len(node.steps) - 1, -1, -1):
            step = node.steps[position]
            binder = _local_name(step.binder.name)
            rest = IRQiecBind(steps=node.steps[position + 1 :], then=node.then)
            capture_names, attachment_ids = _free_runtime_capture(
                rest, frozenset({step.binder.name})
            )
            captures = _julia_data(
                {name: f"__QVR_LOCAL__{_local_name(name)}" for name in capture_names}
            )
            for name in capture_names:
                captures = captures.replace(
                    _julia_string(f"__QVR_LOCAL__{_local_name(name)}"),
                    _local_name(name),
                )
            rendered = f"_qvr_qiec_bind({_julia_computation(target, step.first, module, tail=False)}, {binder} -> {rendered}, _qvr_qiec_capture({captures}, {_julia_data(attachment_ids)}, qiec_attachments))"
        return rendered
    if isinstance(node, IRQiecPerform):
        args = ", ".join(
            _julia_value(target, value) for value in node.request.arguments
        )
        request = _runtime_request(node)
        request["arguments"] = "__QVR_ARGUMENTS__"
        rendered = _julia_data(request).replace(
            _julia_string("__QVR_ARGUMENTS__"), f"Any[{args}]"
        )
        return f"_qvr_qiec_effect({rendered}, qiec_static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return f"_qvr_qiec_handle(() -> {_julia_computation(target, node.computation, module)}, {_julia_string(node.instance.text)}, {_julia_data(_ir_data(handler))}, {_julia_data(_ir_data(node.static_arguments))}, qiec_static, qiec_handlers, qiec_attachments, qiec_operations)"
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_julia_value(target, argument)}, " for argument in node.arguments
        )
        return (
            f"_qvr_qiec_enter_call({_julia_string(callee.name)}, () -> {_body_name(callee)}("
            f"{arguments}"
            f"_qvr_qiec_static_environment({_julia_data(_ir_data(callee.telescope))}, "
            f"_qvr_qiec_specialize({_julia_data(_ir_data(node.static_arguments))}, qiec_static)), "
            f"qiec_attachments, qiec_handlers, qiec_operations), {'true' if tail else 'false'})"
        )
    if isinstance(node, IRQiecIf):
        return f"_qvr_qiec_if({_julia_value(target, node.condition)}, () -> {_julia_computation(target, node.then, module, tail=tail)}, () -> {_julia_computation(target, node.otherwise, module, tail=tail)})"
    if isinstance(node, IRQiecResume):
        return f"_qvr_qiec_resume(qiec_resume, {_julia_value(target, node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return (
            f"_qvr_qiec_instance(() -> {_julia_computation(target, node.body, module)})"
        )
    if isinstance(node, IRQiecCase):
        pairs = []
        for branch in node.branches:
            params = ", ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _julia_computation(target, branch.body, module), "qiec_static"
            )
            branch_params = ", ".join(
                item for item in ("qiec_branch_static", params) if item
            )
            pairs.append(
                f"{_julia_string(branch.constructor.text)} => (({branch_params}) -> {body})"
            )
        return f"_qvr_qiec_case({_julia_value(target, node.scrutinee)}, Dict({', '.join(pairs)}), {_julia_data(_case_metadata(node))}, qiec_static)"
    raise TypeError(f"unknown QIEC computation {node!r}")


def _julia_value(target: str, node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"_qvr_qiec_value({_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _julia_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        return f"_qvr_qiec_constructor({_julia_string(node.constructor.text)}, {_julia_data(_ir_data(node.static_arguments))}, Any[{', '.join(_julia_value(target, field) for field in node.fields)}], {_julia_data(_ir_data(node.result_type))}, qiec_static)"
    if isinstance(node, IRQiecEvidenceValue):
        return (
            f"_qvr_qiec_evidence({_julia_data(_ir_data(node.evidence))}, qiec_static)"
        )
    if isinstance(node, IRQiecAttachmentRef):
        return f"_qvr_qiec_attachment(qiec_attachments, {_julia_string(node.attachment.text)}, {_julia_data(_ir_data(node.type))}, qiec_static)"
    if isinstance(node, IRQiecTransportValue):
        return f"_qvr_qiec_transport({_julia_data(_ir_data(node.evidence))}, {_julia_value(target, node.value)}, {_julia_data(_ir_data(node.target_type))}, qiec_static)"
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = ", ".join(
            _julia_value(target, argument) for argument in node.arguments
        )
        return f"_qvr_qiec_primitive({_julia_string(node.name)}, Any[{arguments}])"
    if isinstance(node, IRQiecTupleValue):
        items = ", ".join(_julia_value(target, item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecProjection):
        return f"_qvr_qiec_project({_julia_value(target, node.value)}, {node.position})"
    if isinstance(node, IRQiecTensorValue):
        items = ", ".join(_julia_value(target, item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecDistributionValue):
        return _spelled(target, node, lambda item: _julia_value(target, item))
    if isinstance(node, IRQiecLogDensity):
        return (
            f"_qvr_qiec_log_density({_julia_value(target, node.sampleable)}, "
            f"{_julia_value(target, node.value)})"
        )
    if isinstance(node, IRQiecSiteValue):
        return _julia_string(node.label)
    if isinstance(node, IRQiecGather):
        return (
            f"_qvr_qiec_gather({_julia_value(target, node.value)}, "
            f"{_julia_value(target, node.index)})"
        )
    if isinstance(node, IRQiecWeightSum):
        return f"_qvr_qiec_weight_sum({_julia_value(target, node.value)})"
    if isinstance(node, IRQiecSegmentSum):
        return (
            f"_qvr_qiec_segment_sum({_julia_value(target, node.value)}, "
            f"{_julia_value(target, node.index)}, {_extent_literal(node.groups)})"
        )
    if isinstance(node, IRQiecReduction):
        return (
            f"_qvr_qiec_reduce({_julia_string(node.operator)}, "
            f"{_julia_value(target, node.value)})"
        )
    if isinstance(node, IRQiecRowwise):
        return (
            f"_qvr_qiec_rowwise({_julia_string(node.operator)}, "
            f"{_julia_value(target, node.value)})"
        )
    if isinstance(node, IRQiecComprehension):
        binder = _local_name(node.binder.name)
        return (
            f"Tuple({_julia_value(target, node.body)} for {binder} in "
            f"0:({_extent_literal(node.extent)} - 1))"
        )
    if isinstance(node, IRQiecLetValue):
        bindings = ", ".join(
            f"{_local_name(binding.binder.name)} = {_julia_value(target, binding.value)}"
            for binding in node.bindings
        )
        return f"(let {bindings}; {_julia_value(target, node.body)} end)"
    raise TypeError(f"unknown QIEC value {node!r}")


def _julia_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _julia_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "nothing"
    if isinstance(node, IRQiecBoolLiteral):
        return "true" if node.value else "false"
    if isinstance(node, IRQiecBytesLiteral):
        return f"hex2bytes({_julia_string(node.value)})"
    if isinstance(node, IRQiecTupleLiteral):
        items = ", ".join(_julia_literal(item) for item in node.items)
        return f"({items},)" if len(node.items) == 1 else f"({items})"
    if isinstance(node, IRQiecStringLiteral):
        return _julia_string(node.value)
    return repr(node.value)  # type: ignore[attr-defined]


def _julia_data(value: object) -> str:
    if isinstance(value, str):
        return _julia_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "nothing"
    if isinstance(value, dict):
        return (
            "Dict("
            + ", ".join(
                f"{_julia_data(key)} => {_julia_data(item)}"
                for key, item in value.items()
            )
            + ")"
        )
    if isinstance(value, (tuple, list)):
        return "Any[" + ", ".join(_julia_data(item) for item in value) + "]"
    return repr(value)


def _javascript_definition(
    target: str, item: IRQiecNamedComputation, module: IRQiecModule
) -> str:
    params = [_local_name(parameter.name) for parameter in item.parameters]
    params.extend(
        (
            "qiec_static_arguments",
            "qiec_attachments_given",
            "qiec_handlers_given",
            "qiec_operations_given",
        )
    )
    body_params = [
        *(_local_name(parameter.name) for parameter in item.parameters),
        "qiec_static",
        "qiec_attachments",
        "qiec_handlers",
        "qiec_operations",
    ]
    return (
        f"var {_body_name(item)} = function({', '.join(body_params)}) {{\n"
        f"  return {_javascript_computation(target, item.body, module)};\n"
        "};\n"
        f"var {_function_name(item)} = function({', '.join(params)}) {{\n"
        f"  var qiec_static = _qvr_qiec_static_environment({json.dumps(_ir_data(item.telescope), ensure_ascii=False)}, qiec_static_arguments);\n"
        "  var qiec_attachments = qiec_attachments_given || {};\n"
        "  var qiec_handlers = qiec_handlers_given || {};\n"
        "  var qiec_operations = qiec_operations_given || {};\n"
        f"  return _qvr_qiec_run(function() {{ return {_body_name(item)}({', '.join(body_params)}); }}, qiec_operations);\n"
        "};\n"
    )


def _javascript_authored_table(module: IRQiecModule) -> str:
    """The table of authored handlers a WebPPL program installs.

    Parameters
    ----------
    module : IRQiecModule
        The checked module.

    Returns
    -------
    str
        ``var _qvr_qiec_authored = {...};`` naming each authored handler's
        definition under its stable identity.
    """
    entries = ", ".join(
        f"{json.dumps(handler.id.text)}: {_authored_handler_name(handler)}"
        for handler in module.handlers
        if handler.implementation == "authored"
    )
    return f"var _qvr_qiec_authored = {{ {entries} }};\n"


def _authored_handler_name(handler: IRQiecHandlerDef) -> str:
    """The WebPPL name holding one authored handler's definition.

    Parameters
    ----------
    handler : IRQiecHandlerDef
        The handler.

    Returns
    -------
    str
        A name derived from the handler's own.
    """
    return f"_qvr_qiec_authored_{_local_name(handler.name)}"


def _javascript_authored_handler(
    target: str, handler: IRQiecHandlerDef, module: IRQiecModule
) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        if clause.body is None:
            continue
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        params = [_local_name(local.name) for local in clause.parameters]
        # WebPPL renames the identifier `arguments`, so the request's
        # field is read under its string key.
        unpack = "".join(
            f'  var {param} = qiec_request["arguments"][{index}];\n'
            for index, param in enumerate(params)
        )
        lines.append(
            f"var {name} = function(qiec_request, qiec_resume, qiec_context) {{\n"
            "  var qiec_static = qiec_context.static;\n"
            "  var qiec_attachments = qiec_context.attachments;\n"
            "  var qiec_handlers = qiec_context.handlers;\n"
            "  var qiec_operations = qiec_context.operations;\n"
            f"{unpack}"
            f"  return {_javascript_computation(target, clause.body, module)};\n"
            "};\n"
        )
        operations.append(f"{json.dumps(clause.operation.text)}: {{ invoke: {name} }}")
    return_entry = "null"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"var {name} = function({binder}, qiec_context) {{\n"
            "  var qiec_static = qiec_context.static;\n"
            "  var qiec_attachments = qiec_context.attachments;\n"
            "  var qiec_handlers = qiec_context.handlers;\n"
            "  var qiec_operations = qiec_context.operations;\n"
            f"  return {_javascript_computation(target, handler.return_clause.body, module)};\n"
            "};\n"
        )
        return_entry = name
    lines.append(
        f"var {_authored_handler_name(handler)} = {{ operations: "
        f'{{ {", ".join(operations)} }}, "return": {return_entry}, '
        "duplicable_context: true };\n"
    )
    return "".join(lines)


def _javascript_computation(
    target: str, node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"_qvr_qiec_pure({_javascript_value(target, node.value)})"
    if isinstance(node, IRQiecBind):
        rendered = _javascript_computation(target, node.then, module)
        for position in range(len(node.steps) - 1, -1, -1):
            step = node.steps[position]
            binder = _local_name(step.binder.name)
            rest = IRQiecBind(steps=node.steps[position + 1 :], then=node.then)
            capture_names, attachment_ids = _free_runtime_capture(
                rest, frozenset({step.binder.name})
            )
            captures = (
                "{"
                + ", ".join(
                    f"{json.dumps(name)}: {_local_name(name)}" for name in capture_names
                )
                + "}"
            )
            rendered = f"_qvr_qiec_bind({_javascript_computation(target, step.first, module, tail=False)}, function({binder}) {{ return {rendered}; }}, _qvr_qiec_capture({captures}, {json.dumps(attachment_ids)}, qiec_attachments))"
        return rendered
    if isinstance(node, IRQiecPerform):
        args = ", ".join(
            _javascript_value(target, value) for value in node.request.arguments
        )
        request = _runtime_request(node)
        request["arguments"] = None
        rendered = json.dumps(request, ensure_ascii=False).replace(
            '"arguments": null', f'"arguments": [{args}]'
        )
        return f"_qvr_qiec_effect({rendered}, qiec_static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return f"_qvr_qiec_handle(function() {{ return {_javascript_computation(target, node.computation, module)}; }}, {json.dumps(node.instance.text)}, {json.dumps(_ir_data(handler), ensure_ascii=False)}, {json.dumps(_ir_data(node.static_arguments), ensure_ascii=False)}, qiec_static, qiec_handlers, qiec_attachments, qiec_operations)"
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_javascript_value(target, argument)}, " for argument in node.arguments
        )
        return (
            f"_qvr_qiec_enter_call({json.dumps(callee.name)}, function() {{ return {_body_name(callee)}("
            f"{arguments}"
            f"_qvr_qiec_static_environment({json.dumps(_ir_data(callee.telescope), ensure_ascii=False)}, "
            f"_qvr_qiec_specialize({json.dumps(_ir_data(node.static_arguments), ensure_ascii=False)}, qiec_static)), "
            f"qiec_attachments, qiec_handlers, qiec_operations); }}, {'true' if tail else 'false'})"
        )
    if isinstance(node, IRQiecIf):
        return f"_qvr_qiec_if({_javascript_value(target, node.condition)}, function() {{ return {_javascript_computation(target, node.then, module, tail=tail)}; }}, function() {{ return {_javascript_computation(target, node.otherwise, module, tail=tail)}; }})"
    if isinstance(node, IRQiecResume):
        return f"_qvr_qiec_resume(qiec_resume, {_javascript_value(target, node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return f"_qvr_qiec_instance(function() {{ return {_javascript_computation(target, node.body, module)}; }})"
    if isinstance(node, IRQiecCase):
        branches = []
        for branch in node.branches:
            params = ", ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _javascript_computation(target, branch.body, module), "qiec_static"
            )
            branches.append(
                f"{json.dumps(branch.constructor.text)}: function(qiec_branch_static{', ' if params else ''}{params}) {{ return {body}; }}"
            )
        return f"_qvr_qiec_case({_javascript_value(target, node.scrutinee)}, {{{', '.join(branches)}}}, {json.dumps(_case_metadata(node), ensure_ascii=False)}, qiec_static)"
    raise TypeError(f"unknown QIEC computation {node!r}")


def _javascript_value(target: str, node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"_qvr_qiec_value({_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _javascript_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        return f"_qvr_qiec_constructor({json.dumps(node.constructor.text)}, {json.dumps(_ir_data(node.static_arguments), ensure_ascii=False)}, [{', '.join(_javascript_value(target, field) for field in node.fields)}], {json.dumps(_ir_data(node.result_type), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecEvidenceValue):
        return f"_qvr_qiec_evidence({json.dumps(_ir_data(node.evidence), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecAttachmentRef):
        return f"_qvr_qiec_attachment(qiec_attachments, {json.dumps(node.attachment.text)}, {json.dumps(_ir_data(node.type), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecTransportValue):
        return f"_qvr_qiec_transport({json.dumps(_ir_data(node.evidence), ensure_ascii=False)}, {_javascript_value(target, node.value)}, {json.dumps(_ir_data(node.target_type), ensure_ascii=False)}, qiec_static)"
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = ", ".join(
            _javascript_value(target, argument) for argument in node.arguments
        )
        return f"_qvr_qiec_primitive({json.dumps(node.name)}, [{arguments}])"
    if isinstance(node, IRQiecTupleValue):
        items = ", ".join(_javascript_value(target, item) for item in node.items)
        return f"Object.freeze([{items}])"
    if isinstance(node, IRQiecProjection):
        return f"_qvr_qiec_project({_javascript_value(target, node.value)}, {node.position})"
    if isinstance(node, IRQiecTensorValue):
        items = ", ".join(_javascript_value(target, item) for item in node.items)
        return f"Object.freeze([{items}])"
    if isinstance(node, IRQiecDistributionValue):
        return _spelled(target, node, lambda item: _javascript_value(target, item))
    if isinstance(node, IRQiecLogDensity):
        return (
            f"_qvr_qiec_log_density({_javascript_value(target, node.sampleable)}, "
            f"{_javascript_value(target, node.value)})"
        )
    if isinstance(node, IRQiecSiteValue):
        return json.dumps(node.label)
    if isinstance(node, IRQiecGather):
        return (
            f"_qvr_qiec_gather({_javascript_value(target, node.value)}, "
            f"{_javascript_value(target, node.index)})"
        )
    if isinstance(node, IRQiecWeightSum):
        return f"_qvr_qiec_weight_sum({_javascript_value(target, node.value)})"
    if isinstance(node, IRQiecSegmentSum):
        return (
            f"_qvr_qiec_segment_sum({_javascript_value(target, node.value)}, "
            f"{_javascript_value(target, node.index)}, {_extent_literal(node.groups)})"
        )
    if isinstance(node, IRQiecReduction):
        return (
            f"_qvr_qiec_reduce({json.dumps(node.operator)}, "
            f"{_javascript_value(target, node.value)})"
        )
    if isinstance(node, IRQiecRowwise):
        return (
            f"_qvr_qiec_rowwise({json.dumps(node.operator)}, "
            f"{_javascript_value(target, node.value)})"
        )
    if isinstance(node, IRQiecComprehension):
        binder = _local_name(node.binder.name)
        return (
            f"Object.freeze(mapN(function({binder}) {{ return "
            f"{_javascript_value(target, node.body)}; }}, {_extent_literal(node.extent)}))"
        )
    if isinstance(node, IRQiecLetValue):
        bindings = " ".join(
            f"var {_local_name(binding.binder.name)} = "
            f"{_javascript_value(target, binding.value)};"
            for binding in node.bindings
        )
        return (
            f"(function() {{ {bindings} return "
            f"{_javascript_value(target, node.body)}; }})()"
        )
    raise TypeError(f"unknown QIEC value {node!r}")


def _javascript_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "null"
    if isinstance(node, IRQiecBytesLiteral):
        return "[" + ", ".join(str(value) for value in bytes.fromhex(node.value)) + "]"
    if isinstance(node, IRQiecTupleLiteral):
        return (
            "Object.freeze(["
            + ", ".join(_javascript_literal(item) for item in node.items)
            + "])"
        )
    return json.dumps(node.value, ensure_ascii=False)  # type: ignore[attr-defined]


def _scheme_definition(
    target: str, item: IRQiecNamedComputation, module: IRQiecModule
) -> str:
    params = " ".join(_local_name(parameter.name) for parameter in item.parameters)
    if params:
        params += " "
    return (
        f"(define ({_body_name(item)} {params}qiec-static qiec-attachments qiec-handlers qiec-operations)\n"
        f"  {_scheme_computation(target, item.body, module)})\n"
        f"(define ({_function_name(item)} {params}. qiec-abi)\n"
        "  (let* ((qiec-static-arguments (if (pair? qiec-abi) (car qiec-abi) #f))\n"
        f"         (qiec-static (_qvr-qiec-static-environment {_scheme_data(_ir_data(item.telescope))} qiec-static-arguments))\n"
        "         (qiec-attachments (if (and (pair? qiec-abi) (pair? (cdr qiec-abi))) (cadr qiec-abi) '()))\n"
        "         (qiec-handlers (if (and (pair? qiec-abi) (pair? (cdr qiec-abi)) (pair? (cddr qiec-abi))) (caddr qiec-abi) '()))\n"
        "         (qiec-operations (if (and (pair? qiec-abi) (pair? (cdr qiec-abi)) (pair? (cddr qiec-abi)) (pair? (cdddr qiec-abi))) (cadddr qiec-abi) '())))\n"
        f"    (_qvr-qiec-run (lambda () ({_body_name(item)} {params}qiec-static qiec-attachments qiec-handlers qiec-operations)) qiec-operations)))\n"
    )


def _scheme_authored_handler(
    target: str, handler: IRQiecHandlerDef, module: IRQiecModule
) -> str:
    """Generate an authored handler's clause bodies and register them.

    Each clause becomes a function with the foreign-clause calling
    convention, so the runtime installs authored and foreign handlers
    alike; the difference is only where the body came from.

    Parameters
    ----------
    handler
        The authored handler declaration.
    module
        The module it belongs to, for resolving names.

    Returns
    -------
    str
        Host-language source defining the clause functions and registering
        the handler under its stable identity.
    """
    lines: list[str] = []
    operations: list[str] = []
    for clause in handler.clauses:
        if clause.body is None:
            continue
        name = _clause_name(handler, _operation_name(module, clause.operation.text))
        params = [_local_name(local.name) for local in clause.parameters]
        bindings = " ".join(
            f'({param} (list-ref (_qvr-qiec-member qiec-request "arguments") {index}))'
            for index, param in enumerate(params)
        )
        lines.append(
            f"(define ({name} qiec-request qiec-resume qiec-context)\n"
            f'  (let* ((qiec-static (_qvr-qiec-member qiec-context "static"))\n'
            f'         (qiec-attachments (_qvr-qiec-member qiec-context "attachments"))\n'
            f'         (qiec-handlers (_qvr-qiec-member qiec-context "handlers"))\n'
            f'         (qiec-operations (_qvr-qiec-member qiec-context "operations"))'
            f"{(' ' + bindings) if bindings else ''})\n"
            f"    {_scheme_computation(target, clause.body, module)}))\n"
        )
        operations.append(
            f'(cons {_scheme_string(clause.operation.text)} (list (cons "invoke" {name})))'
        )
    return_entry = "#f"
    if handler.return_clause is not None:
        name = _return_clause_name(handler)
        binder = _local_name(handler.return_clause.binder.name)
        lines.append(
            f"(define ({name} {binder} qiec-context)\n"
            f'  (let* ((qiec-static (_qvr-qiec-member qiec-context "static"))\n'
            f'         (qiec-attachments (_qvr-qiec-member qiec-context "attachments"))\n'
            f'         (qiec-handlers (_qvr-qiec-member qiec-context "handlers"))\n'
            f'         (qiec-operations (_qvr-qiec-member qiec-context "operations")))\n'
            f"    {_scheme_computation(target, handler.return_clause.body, module)}))\n"
        )
        return_entry = name
    lines.append(
        f"(_qvr-qiec-register-authored {_scheme_string(handler.id.text)} "
        f'(list (cons "operations" (list {" ".join(operations)})) '
        f'(cons "return" {return_entry}) (cons "duplicable_context" #t)))\n'
    )
    return "".join(lines)


def _scheme_computation(
    target: str, node: IRQiecComputation, module: IRQiecModule, tail: bool = True
) -> str:
    if isinstance(node, IRQiecReturn):
        return f"(_qvr-qiec-pure {_scheme_value(target, node.value)})"
    if isinstance(node, IRQiecBind):
        rendered = _scheme_computation(target, node.then, module)
        for position in range(len(node.steps) - 1, -1, -1):
            step = node.steps[position]
            binder = _local_name(step.binder.name)
            rest = IRQiecBind(steps=node.steps[position + 1 :], then=node.then)
            capture_names, attachment_ids = _free_runtime_capture(
                rest, frozenset({step.binder.name})
            )
            captures = (
                "(list "
                + " ".join(
                    f"(cons {_scheme_string(name)} {_local_name(name)})"
                    for name in capture_names
                )
                + ")"
            )
            rendered = f"(_qvr-qiec-bind {_scheme_computation(target, step.first, module, tail=False)} (lambda ({binder}) {rendered}) (_qvr-qiec-capture {captures} {_scheme_data(attachment_ids)} qiec-attachments))"
        return rendered
    if isinstance(node, IRQiecPerform):
        args = " ".join(
            _scheme_value(target, value) for value in node.request.arguments
        )
        request = _runtime_request(node)
        request["arguments"] = "__QVR_ARGUMENTS__"
        rendered = _scheme_data(request).replace(
            _scheme_string("__QVR_ARGUMENTS__"), f"(list {args})"
        )
        return f"(_qvr-qiec-effect {rendered} qiec-static)"
    if isinstance(node, IRQiecHandle):
        handler = _handler(module, node.handler.text)
        return f"(_qvr-qiec-handle (lambda () {_scheme_computation(target, node.computation, module)}) {_scheme_string(node.instance.text)} {_scheme_data(_ir_data(handler))} {_scheme_data(_ir_data(node.static_arguments))} qiec-static qiec-handlers qiec-attachments qiec-operations)"
    if isinstance(node, IRQiecCall):
        callee = _computation_by_id(module, node.callee.text)
        arguments = "".join(
            f"{_scheme_value(target, argument)} " for argument in node.arguments
        )
        return (
            f"(_qvr-qiec-enter-call {_scheme_string(callee.name)} (lambda () ({_body_name(callee)} "
            f"{arguments}"
            f"(_qvr-qiec-static-environment {_scheme_data(_ir_data(callee.telescope))} "
            f"(_qvr-qiec-specialize {_scheme_data(_ir_data(node.static_arguments))} qiec-static)) "
            f"qiec-attachments qiec-handlers qiec-operations)) {'#t' if tail else '#f'})"
        )
    if isinstance(node, IRQiecIf):
        return f"(_qvr-qiec-if {_scheme_value(target, node.condition)} (lambda () {_scheme_computation(target, node.then, module, tail=tail)}) (lambda () {_scheme_computation(target, node.otherwise, module, tail=tail)}))"
    if isinstance(node, IRQiecResume):
        return f"(_qvr-qiec-resume qiec-resume {_scheme_value(target, node.value)})"
    if isinstance(node, IRQiecNewInstance):
        return f"(_qvr-qiec-instance (lambda () {_scheme_computation(target, node.body, module)}))"
    if isinstance(node, IRQiecCase):
        rendered_branches = []
        for branch in node.branches:
            params = " ".join(_local_name(field.name) for field in branch.fields)
            body = _branch_static(
                _scheme_computation(target, branch.body, module), "qiec-static"
            )
            rendered_branches.append(
                f"(cons {_scheme_string(branch.constructor.text)} (lambda (qiec-branch-static{' ' if params else ''}{params}) {body}))"
            )
        branches = " ".join(rendered_branches)
        return f"(_qvr-qiec-case {_scheme_value(target, node.scrutinee)} (list {branches}) {_scheme_data(_case_metadata(node))} qiec-static)"
    raise TypeError(f"unknown QIEC computation {node!r}")


def _scheme_value(target: str, node: IRQiecValue) -> str:
    if isinstance(node, IRQiecVar):
        return f"(_qvr-qiec-value {_local_name(node.local.name)})"
    if isinstance(node, IRQiecLiteralValue):
        return _scheme_literal(node.value)
    if isinstance(node, IRQiecConstructorValue):
        return f"(_qvr-qiec-constructor {_scheme_string(node.constructor.text)} {_scheme_data(_ir_data(node.static_arguments))} (list {' '.join(_scheme_value(target, field) for field in node.fields)}) {_scheme_data(_ir_data(node.result_type))} qiec-static)"
    if isinstance(node, IRQiecEvidenceValue):
        return (
            f"(_qvr-qiec-evidence {_scheme_data(_ir_data(node.evidence))} qiec-static)"
        )
    if isinstance(node, IRQiecAttachmentRef):
        return f"(_qvr-qiec-attachment qiec-attachments {_scheme_string(node.attachment.text)} {_scheme_data(_ir_data(node.type))} qiec-static)"
    if isinstance(node, IRQiecTransportValue):
        return f"(_qvr-qiec-transport {_scheme_data(_ir_data(node.evidence))} {_scheme_value(target, node.value)} {_scheme_data(_ir_data(node.target_type))} qiec-static)"
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = " ".join(
            _scheme_value(target, argument) for argument in node.arguments
        )
        return f"(_qvr-qiec-primitive {_scheme_string(node.name)} (list {arguments}))"
    if isinstance(node, IRQiecTupleValue):
        items = " ".join(_scheme_value(target, item) for item in node.items)
        return f"(_qvr-qiec-tuple (list {items}))"
    if isinstance(node, IRQiecProjection):
        return (
            f"(_qvr-qiec-project {_scheme_value(target, node.value)} {node.position})"
        )
    if isinstance(node, IRQiecTensorValue):
        items = " ".join(_scheme_value(target, item) for item in node.items)
        return f"(_qvr-qiec-tuple (list {items}))"
    if isinstance(node, IRQiecDistributionValue):
        return _spelled(target, node, lambda item: _scheme_value(target, item))
    if isinstance(node, IRQiecLogDensity):
        return (
            f"(_qvr-qiec-log-density {_scheme_value(target, node.sampleable)} "
            f"{_scheme_value(target, node.value)})"
        )
    if isinstance(node, IRQiecSiteValue):
        return _scheme_string(node.label)
    if isinstance(node, IRQiecGather):
        return (
            f"(_qvr-qiec-gather {_scheme_value(target, node.value)} "
            f"{_scheme_value(target, node.index)})"
        )
    if isinstance(node, IRQiecWeightSum):
        return f"(_qvr-qiec-weight-sum {_scheme_value(target, node.value)})"
    if isinstance(node, IRQiecSegmentSum):
        return (
            f"(_qvr-qiec-segment-sum {_scheme_value(target, node.value)} "
            f"{_scheme_value(target, node.index)} {_extent_literal(node.groups)})"
        )
    if isinstance(node, IRQiecReduction):
        return (
            f"(_qvr-qiec-reduce {_scheme_string(node.operator)} "
            f"{_scheme_value(target, node.value)})"
        )
    if isinstance(node, IRQiecRowwise):
        return (
            f"(_qvr-qiec-rowwise {_scheme_string(node.operator)} "
            f"{_scheme_value(target, node.value)})"
        )
    if isinstance(node, IRQiecComprehension):
        binder = _local_name(node.binder.name)
        return (
            f"(_qvr-qiec-tuple (map (lambda ({binder}) {_scheme_value(target, node.body)}) "
            f"(iota {_extent_literal(node.extent)})))"
        )
    if isinstance(node, IRQiecLetValue):
        bindings = " ".join(
            f"({_local_name(binding.binder.name)} {_scheme_value(target, binding.value)})"
            for binding in node.bindings
        )
        return f"(let* ({bindings}) {_scheme_value(target, node.body)})"
    raise TypeError(f"unknown QIEC value {node!r}")


def _scheme_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _scheme_literal(node: IRQiecLiteral) -> str:
    if isinstance(node, IRQiecNullLiteral):
        return "'()"
    if isinstance(node, IRQiecBoolLiteral):
        return "#t" if node.value else "#f"
    if isinstance(node, IRQiecBytesLiteral):
        return (
            "(list " + " ".join(str(value) for value in bytes.fromhex(node.value)) + ")"
        )
    if isinstance(node, IRQiecTupleLiteral):
        return (
            "(_qvr-qiec-tuple (list "
            + " ".join(_scheme_literal(item) for item in node.items)
            + "))"
        )
    if isinstance(node, IRQiecStringLiteral):
        return _scheme_string(node.value)
    return repr(node.value)  # type: ignore[attr-defined]


def _scheme_data(value: object) -> str:
    if isinstance(value, str):
        return _scheme_string(value)
    if isinstance(value, bool):
        return "#t" if value else "#f"
    if value is None:
        return "'()"
    if isinstance(value, dict):
        return (
            "(list "
            + " ".join(
                f"(cons {_scheme_data(key)} {_scheme_data(item)})"
                for key, item in value.items()
            )
            + ")"
        )
    if isinstance(value, (tuple, list)):
        return "(list " + " ".join(_scheme_data(item) for item in value) + ")"
    return repr(value)


def _static_definitions(
    computations: tuple[IRQiecNamedComputation, ...], target: str
) -> tuple[str, str]:
    """The static target's definitions of the computations it needs.

    Parameters
    ----------
    computations : tuple[IRQiecNamedComputation, ...]
        The needed computations, in module order.
    target : str
        The public target name.

    Returns
    -------
    tuple[str, str]
        The source and the kind of its root.
    """
    if target == "stan":
        # Every user-defined function name is in scope throughout the
        # block, so a recursive or mutually recursive computation
        # resolves without a forward declaration.
        definitions = "\n".join(_stan_function(item) for item in computations)
        return f"functions {{\n{definitions}\n}}\n", "functions"
    assignments = []
    for item in computations:
        statements, value = _linearize(item.body)
        environment: dict[str, str] = {}
        for position, (binder, bound) in enumerate(statements):
            target_name = (
                f"qiec_{_safe_name(item.name)}__{_safe_name(binder)}_{position}"
            )
            assignments.append(f"{target_name} <- {_static_value(bound, environment)}")
            environment[binder] = target_name
        assignments.append(
            f"{_function_name(item)} <- {_static_value(value, environment)}"
        )
    return "model {\n" + "\n".join(assignments) + "\n}\n", "model_block"


def _linearize(
    node: IRQiecComputation,
) -> tuple[list[tuple[str, IRQiecValue]], IRQiecValue]:
    """Flatten a pure bind chain into its steps and its result.

    Parameters
    ----------
    node : IRQiecComputation
        A computation of returns and binds.

    Returns
    -------
    tuple[list[tuple[str, IRQiecValue]], IRQiecValue]
        Each binder with the value bound to it, in order, and the
        returned value.

    Raises
    ------
    TypeError
        If the computation has any other form.
    """
    if isinstance(node, IRQiecReturn):
        return [], node.value
    if isinstance(node, IRQiecBind):
        statements: list[tuple[str, IRQiecValue]] = []
        for step in node.steps:
            before, first = _linearize(step.first)
            statements.extend(before)
            statements.append((step.binder.name, first))
        after, result = _linearize(node.then)
        return [*statements, *after], result
    raise TypeError("the QIEC capability analyzer admitted a non-pure computation")


def _static_value(node: IRQiecValue, environment: dict[str, str] | None = None) -> str:
    """A scalar variable or literal as a static target's expression.

    Parameters
    ----------
    node : IRQiecValue
        The value.
    environment : dict[str, str] | None
        The names the graph targets bound each local under.

    Returns
    -------
    str
        The expression.

    Raises
    ------
    TypeError
        If the value is neither a variable nor a numeric scalar.
    """
    if isinstance(node, IRQiecVar):
        return (environment or {}).get(node.local.name, _local_name(node.local.name))
    if not isinstance(node, IRQiecLiteralValue):
        raise TypeError("the QIEC capability analyzer admitted a non-scalar value")
    literal = node.value
    if isinstance(literal, IRQiecBoolLiteral):
        return "1" if literal.value else "0"
    if isinstance(literal, (IRQiecIntLiteral, IRQiecFloatLiteral)):
        return repr(literal.value)
    raise TypeError("the QIEC capability analyzer admitted a non-numeric scalar")


def _stan_type(type_: object) -> str:
    """The Stan type of a monomorphic scalar QIEC type.

    Parameters
    ----------
    type_ : object
        The type.

    Returns
    -------
    str
        ``int`` for `Int` and `Bool`, ``real`` for `Real`.

    Raises
    ------
    TypeError
        If the type is not one of those.
    """
    if not isinstance(type_, IRQiecTypeApplication):
        raise TypeError("the QIEC capability analyzer admitted a non-monomorphic type")
    name = type_.constructor.name
    if name in {"Int", "Bool"}:
        return "int"
    if name == "Real":
        return "real"
    raise TypeError(
        f"the QIEC capability analyzer admitted unsupported Stan type {name}"
    )


#: Stan spellings of the operators, comparisons, conversions, and
#: operand-typed `min` / `max` a Stan function may apply, as templates
#: over the rendered arguments. Booleans are Stan integers, so the
#: logical operators are Stan's. Every other math builtin is spelled by
#: the shared per-target table, through `_stan_builtin`.
_STAN_PRIMITIVES: dict[str, str] = {
    "add_int": "({0} + {1})",
    "sub_int": "({0} - {1})",
    "mul_int": "({0} * {1})",
    "div_int": "({0} %/% {1})",
    "mod_int": "({0} - {1} * ({0} %/% {1}))",
    "neg_int": "(-{0})",
    "min_int": "min({0}, {1})",
    "max_int": "max({0}, {1})",
    "add_real": "({0} + {1})",
    "sub_real": "({0} - {1})",
    "mul_real": "({0} * {1})",
    "div_real": "({0} / {1})",
    "neg_real": "(-{0})",
    "min_real": "fmin({0}, {1})",
    "max_real": "fmax({0}, {1})",
    "eq_int": "({0} == {1})",
    "ne_int": "({0} != {1})",
    "lt_int": "({0} < {1})",
    "le_int": "({0} <= {1})",
    "gt_int": "({0} > {1})",
    "ge_int": "({0} >= {1})",
    "eq_real": "({0} == {1})",
    "ne_real": "({0} != {1})",
    "lt_real": "({0} < {1})",
    "le_real": "({0} <= {1})",
    "gt_real": "({0} > {1})",
    "ge_real": "({0} >= {1})",
    "eq_bool": "({0} == {1})",
    "ne_bool": "({0} != {1})",
    "and": "({0} && {1})",
    "or": "({0} || {1})",
    "not": "(!{0})",
    "int_to_real": "(1.0 * {0})",
}


def _stan_value(node: IRQiecValue, computation: str) -> str:
    """A pure scalar value as a Stan expression.

    Parameters
    ----------
    node : IRQiecValue
        The value.
    computation : str
        The computation it belongs to, for the refusal's kind.

    Returns
    -------
    str
        The Stan expression.

    Raises
    ------
    UnsupportedConstruct
        If a primitive has no Stan spelling.
    """
    if isinstance(node, IRQiecPrimitiveApplication):
        arguments = tuple(
            _stan_value(argument, computation) for argument in node.arguments
        )
        builtin = PRIMITIVE_BUILTINS.get(node.name)
        spelling = (
            None
            if builtin is None
            else builtin_spelling("stan", builtin, len(arguments))
        )
        if (
            builtin is not None
            and spelling is not None
            and not isinstance(spelling, Dedicated)
        ):
            return _stan_builtin(builtin, arguments)
        template = _STAN_PRIMITIVES.get(node.name)
        if template is None:
            raise UnsupportedConstruct(
                "qvr-stan", [f"qiec:stan:primitive:{node.name}:{computation}"]
            )
        return template.format(*arguments)
    return _static_value(node)


def _stan_builtin(builtin: str, arguments: tuple[str, ...]) -> str:
    """A math builtin applied to rendered Stan arguments.

    The builtin is spelled by the shared per-target table: a Stan
    function of its own, or a closed form whose own builtins are
    spelled the same way.

    Parameters
    ----------
    builtin : str
        The QVR builtin.
    arguments : tuple[str, ...]
        The rendered arguments.

    Returns
    -------
    str
        The Stan expression.

    Raises
    ------
    UnsupportedConstruct
        If Stan cannot express the builtin.
    """
    placeholders = tuple(
        LetExprVar(name=f"__qiec_argument{index}") for index in range(len(arguments))
    )
    return _stan_text(
        spell_builtin("stan", builtin, placeholders),
        {
            placeholder.name: text
            for placeholder, text in zip(placeholders, arguments, strict=True)
        },
    )


#: Stan's real-valued pointwise `max` and `min`, which a closed form
#: applies to reals.
_STAN_REAL_POINTWISE: dict[str, str] = {"max": "fmax", "min": "fmin"}


def _stan_text(expr: LetExprNode, arguments: dict[str, str]) -> str:
    """A spelled builtin's expression as Stan text.

    Parameters
    ----------
    expr : LetExprNode
        The call or its closed form, over placeholder variables.
    arguments : dict[str, str]
        The rendered argument each placeholder stands for.

    Returns
    -------
    str
        The Stan expression.

    Raises
    ------
    UnsupportedConstruct
        If the expression holds a node a closed form does not use.
    """
    if isinstance(expr, LetExprVar):
        return arguments[expr.name]
    if isinstance(expr, LetExprLiteral):
        return repr(float(expr.value))
    if isinstance(expr, LetExprBinOp):
        return (
            f"({_stan_text(expr.left, arguments)} {expr.op} "
            f"{_stan_text(expr.right, arguments)})"
        )
    if isinstance(expr, LetExprUnaryOp):
        return f"(-{_stan_text(expr.operand, arguments)})"
    if isinstance(expr, LetExprCall):
        path = native_path("stan", expr.func, len(expr.args))
        name = (
            ".".join(path)
            if path is not None
            else _STAN_REAL_POINTWISE.get(expr.func, expr.func)
        )
        rendered = ", ".join(_stan_text(argument, arguments) for argument in expr.args)
        return f"{name}({rendered})"
    raise UnsupportedConstruct(
        "qvr-stan", [f"qiec:stan:builtin-form:{type(expr).__name__}"]
    )


def _stan_signature(item: IRQiecNamedComputation) -> str:
    """A computation's Stan function signature.

    Parameters
    ----------
    item : IRQiecNamedComputation
        The computation.

    Returns
    -------
    str
        ``<type> qiec_<name>(<typed parameters>)``.
    """
    params = ", ".join(
        f"{_stan_type(parameter.type)} {_local_name(parameter.name)}"
        for parameter in item.parameters
    )
    return f"{_stan_type(item.type.result)} {_function_name(item)}({params})"


def _stan_function(item: IRQiecNamedComputation) -> str:
    """A pure computation as a Stan function definition.

    Parameters
    ----------
    item : IRQiecNamedComputation
        The computation.

    Returns
    -------
    str
        The definition, its binds as typed locals, its conditionals
        as `if` statements, and its calls as calls.
    """
    binders = {local.name: local for local in _bound_locals(item.body)}
    lines = [f"{_stan_signature(item)} {{"]
    lines.extend(_stan_statements(item.body, binders, item.name, None, "  "))
    lines.append("}")
    return "\n".join(lines)


def _stan_statements(
    node: IRQiecComputation,
    binders: dict[str, IRQiecLocal],
    computation: str,
    target: str | None,
    indent: str,
) -> list[str]:
    """The Stan statements running a pure computation.

    Parameters
    ----------
    node : IRQiecComputation
        The computation.
    binders : dict[str, IRQiecLocal]
        Every local the enclosing function's binds introduce.
    computation : str
        The enclosing computation's name, for refusals.
    target : str | None
        The local the computation's result is assigned to, or None to
        return it.
    indent : str
        The indentation of the statements.

    Returns
    -------
    list[str]
        The statements.
    """
    if isinstance(node, IRQiecReturn):
        lines, value = _stan_expression(node.value, computation, indent)
        if target is None:
            return [*lines, f"{indent}return {value};"]
        return [*lines, f"{indent}{target} = {value};"]
    if isinstance(node, IRQiecCall):
        lines, callee = _stan_call(node, computation, indent)
        if target is None:
            return [*lines, f"{indent}return {callee};"]
        return [*lines, f"{indent}{target} = {callee};"]
    if isinstance(node, IRQiecIf):
        lines, condition = _stan_expression(node.condition, computation, indent)
        return [
            *lines,
            f"{indent}if ({condition}) {{",
            *_stan_statements(node.then, binders, computation, target, indent + "  "),
            f"{indent}}} else {{",
            *_stan_statements(
                node.otherwise, binders, computation, target, indent + "  "
            ),
            f"{indent}}}",
        ]
    if isinstance(node, IRQiecBind):
        lines: list[str] = []
        for step in node.steps:
            local = binders[step.binder.name]
            name = _local_name(step.binder.name)
            declared = f"{indent}{_stan_type(local.type)} {name}"
            if isinstance(step.first, IRQiecReturn):
                before, value = _stan_expression(step.first.value, computation, indent)
                lines.extend(before)
                lines.append(f"{declared} = {value};")
            elif isinstance(step.first, IRQiecCall):
                before, callee = _stan_call(step.first, computation, indent)
                lines.extend(before)
                lines.append(f"{declared} = {callee};")
            else:
                lines.append(f"{declared};")
                lines.extend(
                    _stan_statements(step.first, binders, computation, name, indent)
                )
        lines.extend(_stan_statements(node.then, binders, computation, target, indent))
        return lines
    raise TypeError("the QIEC capability analyzer admitted a non-pure computation")


def _stan_call(
    node: IRQiecCall, computation: str, indent: str
) -> tuple[list[str], str]:
    """A call of a module computation as a Stan call expression.

    Parameters
    ----------
    node : IRQiecCall
        The call.
    computation : str
        The calling computation's name, for refusals.
    indent : str
        The indentation of the declarations the arguments need.

    Returns
    -------
    tuple[list[str], str]
        The declarations of the arguments' shared bindings, and
        ``qiec_<callee>(<arguments>)``.

    Raises
    ------
    UnsupportedConstruct
        If an argument applies a primitive with no Stan spelling.
    """
    lines: list[str] = []
    arguments: list[str] = []
    for argument in node.arguments:
        before, rendered = _stan_expression(argument, computation, indent)
        lines.extend(before)
        arguments.append(rendered)
    return lines, f"qiec_{_safe_name(node.name)}({', '.join(arguments)})"


def _stan_expression(
    node: IRQiecValue, computation: str, indent: str
) -> tuple[list[str], str]:
    """A pure scalar value as Stan declarations and one expression.

    Stan expressions have no binding form, so each binding of an
    ``IRQiecLetValue`` becomes a typed local declared ahead of the
    statement that reads the value, and the expression reads it by
    name. The value is scalar: the Stan capability set admits no
    comprehension whose index a binding could read.

    Parameters
    ----------
    node : IRQiecValue
        The value.
    computation : str
        The computation it belongs to, for the refusal's kind.
    indent : str
        The indentation of the declarations.

    Returns
    -------
    tuple[list[str], str]
        The declarations in evaluation order, and the Stan expression.

    Raises
    ------
    UnsupportedConstruct
        If a primitive has no Stan spelling.
    """
    lines: list[str] = []

    def hoisted(item: IRQiecValue) -> IRQiecValue:
        if isinstance(item, IRQiecLetValue):
            for binding in item.bindings:
                bound = _stan_value(hoisted(binding.value), computation)
                lines.append(
                    f"{indent}{_stan_type(binding.binder.type)} "
                    f"{_local_name(binding.binder.name)} = {bound};"
                )
            return hoisted(item.body)
        if isinstance(item, IRQiecPrimitiveApplication):
            return item.with_(
                arguments=tuple(hoisted(argument) for argument in item.arguments)
            )
        return item

    value = _stan_value(hoisted(node), computation)
    return lines, value


def _bound_locals(node: IRQiecComputation) -> tuple[IRQiecLocal, ...]:
    """Every local a pure computation's bind steps introduce.

    Parameters
    ----------
    node : IRQiecComputation
        A computation made of returns, binds, conditionals, and calls.

    Returns
    -------
    tuple[IRQiecLocal, ...]
        The binders in evaluation order, both branches of a
        conditional included.
    """
    if isinstance(node, IRQiecIf):
        return (*_bound_locals(node.then), *_bound_locals(node.otherwise))
    if not isinstance(node, IRQiecBind):
        return ()
    found: list[IRQiecLocal] = []
    for step in node.steps:
        found.append(step.binder)
        found.extend(_bound_locals(step.first))
    found.extend(_bound_locals(node.then))
    return tuple(found)


__all__ = [
    "canonical_operations",
    "emit_call_python",
    "graft_python_statements",
    "python_call_source",
    "python_operations_source",
    "render_computations_dynamic",
    "render_computations_static",
    "has_runtime_computations",
    "qiec_target_name",
]
