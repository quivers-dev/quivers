"""Latent network weights: ``[weights=latent]`` as program sites.

A network kernel (``morphism f : X -> Y [param_source=mlp(...)] ~
Normal``) or a learned function (``[role=function]``) holds its weights
as point estimates by default. Declared ``[weights=latent,
prior_scale=s]``, every weight and bias of its network is instead a
latent sample site with an independent ``Normal(0, s)`` prior, drawn
once per program run and read by every row and position that applies
the network.

The rewrite here is the one place that decides those sites. It adds,
to every program without value parameters that reaches such a
morphism (through a draw, a let call, a ``define`` chain, a scan's
cell, or a called program), one sample step per weight tensor at the
head of its body, and declares a finite object for each axis of each
tensor. The sites are named as the checked elaboration names a
network's weights, ``<f>_param_layer<k>_weight``,
``<f>_param_layer<k>_bias``, ``<f>_param_weight``, and
``<f>_param_bias``, so the checked elaboration reads them where it
would otherwise declare program inputs, and the compiled runtime reads
them from the run's bindings. A program with value parameters (a scan's
cell, a called program) reads the weights its caller draws.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.continuous.deterministic import weight_site_names
from quivers.dsl.ast_nodes import (
    CallStep,
    DefineDecl,
    Expr,
    LetStep,
    MarginalizeStep,
    Module,
    MorphismDecl,
    ObjectDecl,
    ObserveStep,
    OptionCall,
    OptionList,
    OptionName,
    OptionNumber,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    ScoreStep,
    TypeName,
)
from quivers.dsl.let_expr_traversal import free_let_callees, free_let_names
from quivers.dsl.parser import parse
from quivers.dsl.step_resolution import _expr_ident_names

#: The default standard deviation of a latent weight's Normal prior.
DEFAULT_PRIOR_SCALE = 1.0


class LatentWeightsError(ValueError):
    """A ``[weights=latent]`` declaration whose network cannot be sized.

    Parameters
    ----------
    message : str
        What is wrong.
    line : int
        The declaration's line.
    col : int
        The declaration's column.
    """

    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(message)
        self.line = line
        self.col = col


def _option(
    morphism: MorphismDecl, key: str
) -> OptionName | OptionNumber | OptionCall | OptionList | None:
    """The value of one option of a declaration, or ``None``."""
    for entry in morphism.options:
        if entry.key == key:
            value = entry.value
            if isinstance(value, OptionName | OptionNumber | OptionCall | OptionList):
                return value
    return None


def weights_mode(morphism: MorphismDecl) -> str:
    """Whether a morphism's network weights are ``point`` or ``latent``.

    Parameters
    ----------
    morphism : MorphismDecl
        The declaration.

    Returns
    -------
    str
        The ``weights`` option's name, ``point`` when absent.
    """
    value = _option(morphism, "weights")
    return str(value.value) if isinstance(value, OptionName) else "point"


def prior_scale(morphism: MorphismDecl) -> float:
    """The standard deviation of a latent weight's Normal prior.

    Parameters
    ----------
    morphism : MorphismDecl
        The declaration.

    Returns
    -------
    float
        The ``prior_scale`` option, `DEFAULT_PRIOR_SCALE` when absent.
    """
    value = _option(morphism, "prior_scale")
    return (
        float(value.value) if isinstance(value, OptionNumber) else DEFAULT_PRIOR_SCALE
    )


def network_kind(morphism: MorphismDecl) -> str:
    """The network kind a morphism's ``param_source`` names.

    Parameters
    ----------
    morphism : MorphismDecl
        The declaration.

    Returns
    -------
    str
        ``linear`` when the option is absent, else the kind it names.
    """
    value = _option(morphism, "param_source")
    if isinstance(value, OptionCall):
        return value.func
    if isinstance(value, OptionName):
        return str(value.value)
    return "linear"


def hidden_widths(morphism: MorphismDecl) -> tuple[int, ...]:
    """The hidden-layer widths of a morphism's network.

    Parameters
    ----------
    morphism : MorphismDecl
        The declaration.

    Returns
    -------
    tuple[int, ...]
        Empty for a linear network; for ``mlp`` the widths
        ``mlp(a, b, ...)`` lists, else the ``hidden_dim`` option's, else
        two layers of sixty-four.

    Raises
    ------
    LatentWeightsError
        If a width is not a positive integer.
    """
    if network_kind(morphism) != "mlp":
        return ()
    source = _option(morphism, "param_source")
    raw: list[OptionName | OptionNumber | OptionCall | OptionList] = []
    if isinstance(source, OptionCall) and source.args:
        raw = [arg for arg in source.args if isinstance(arg, OptionNumber)]
        if len(raw) != len(source.args):
            raw = []
    else:
        hidden = _option(morphism, "hidden_dim")
        if isinstance(hidden, OptionList):
            raw = [item for item in hidden.items if isinstance(item, OptionNumber)]
        elif hidden is not None:
            raw = [hidden]
    if not raw:
        return (64, 64)
    widths: list[int] = []
    for item in raw:
        if not isinstance(item, OptionNumber) or not float(item.value).is_integer():
            raise LatentWeightsError(
                f"morphism {morphism.names[0]!r} declares a hidden width that is "
                "not an integer",
                morphism.line,
                morphism.col,
            )
        widths.append(int(item.value))
    return tuple(widths)


def latent_morphisms(module: Module) -> dict[str, MorphismDecl]:
    """Every morphism a module declares with ``[weights=latent]``, by name.

    Parameters
    ----------
    module : Module
        The module.

    Returns
    -------
    dict[str, MorphismDecl]
        One entry per declared name.
    """
    found: dict[str, MorphismDecl] = {}
    for statement in module.statements:
        if isinstance(statement, MorphismDecl) and weights_mode(statement) == "latent":
            for name in statement.names:
                found[name] = statement
    return found


def _step_names(steps: tuple[ProgramStep, ...]) -> frozenset[str]:
    """Every morphism, define, or program name a body refers to."""
    names: set[str] = set()
    for step in steps:
        if isinstance(step, SampleStep | ObserveStep):
            names.add(step.morphism)
        if isinstance(step, MarginalizeStep):
            names.add(step.morphism)
            names |= _step_names(step.scope)
        if isinstance(step, LetStep | ScoreStep):
            names |= set(free_let_callees(step.value))
            names |= set(free_let_names(step.value))
        if isinstance(step, CallStep):
            names.add(step.call.callee)
    return frozenset(names)


def _has_scan(expr: Expr) -> bool:
    """Whether a composition expression contains a ``scan``.

    Parameters
    ----------
    expr : Expr
        The expression.

    Returns
    -------
    bool
        ``True`` if a scan occurs anywhere in it.
    """
    pending: list[object] = [expr.model_dump()]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if node.get("kind") == "expr_scan":
                return True
            pending.extend(node.values())
        elif isinstance(node, list | tuple):
            pending.extend(node)
    return False


def _refuse_composite_uses(module: Module, latent: Mapping[str, MorphismDecl]) -> None:
    """Refuse a latent-weight morphism composed outside a scan.

    A program composes a ``define`` that runs a scan step by step, so a
    latent-weight morphism in it is drawn from or called in a program
    step that reads the drawn weights. A ``define`` without a scan, or a
    morphism's ``~ <expression>`` initializer, composes its morphisms as
    one kernel that has no run to read drawn weights from.

    Parameters
    ----------
    module : Module
        The module.
    latent : Mapping[str, MorphismDecl]
        The latent-weight morphisms by name.

    Raises
    ------
    LatentWeightsError
        At the first composition that reads a latent-weight morphism
        outside a scan.
    """
    for statement in module.statements:
        if isinstance(statement, DefineDecl):
            expr, line, col, owner = (
                statement.expr,
                statement.line,
                statement.col,
                f"define {statement.name!r}",
            )
        elif isinstance(statement, MorphismDecl) and statement.init_expr is not None:
            expr, line, col, owner = (
                statement.init_expr,
                statement.line,
                statement.col,
                f"morphism {statement.names[0]!r}",
            )
        else:
            continue
        used = sorted(_expr_ident_names(expr) & set(latent))
        if used and not _has_scan(expr):
            raise LatentWeightsError(
                f"{owner} composes {used[0]!r}, whose weights are latent, outside "
                f"a scan; a composition without a scan is one kernel with no run "
                f"to draw the weights in. Draw from or call {used[0]!r} in a "
                f"program step instead",
                line,
                col,
            )


def _references(module: Module) -> dict[str, frozenset[str]]:
    """The names each program, define, and morphism refers to."""
    graph: dict[str, frozenset[str]] = {}
    for statement in module.statements:
        if isinstance(statement, ProgramDecl):
            graph[statement.name] = _step_names(statement.draws)
        elif isinstance(statement, DefineDecl):
            graph[statement.name] = _expr_ident_names(statement.expr)
        elif isinstance(statement, MorphismDecl) and statement.init_expr is not None:
            for name in statement.names:
                graph[name] = _expr_ident_names(statement.init_expr)
    return graph


def _reaches(
    start: str, targets: Mapping[str, MorphismDecl], graph: Mapping[str, frozenset[str]]
) -> list[str]:
    """The latent morphisms ``start`` reaches, in name order."""
    seen: set[str] = {start}
    frontier = [start]
    found: set[str] = set()
    while frontier:
        name = frontier.pop()
        for reached in graph.get(name, frozenset()):
            if reached in targets:
                found.add(reached)
            if reached not in seen:
                seen.add(reached)
                frontier.append(reached)
    return sorted(found)


def _real_width(expr: object, widths: Mapping[str, int]) -> int | None:
    """The real width of a domain factor or codomain, if it has one."""
    if isinstance(expr, TypeName):
        return widths.get(expr.name)
    return None


def _object_widths(module: Module) -> tuple[dict[str, int], dict[str, int]]:
    """The real widths and finite extents of a module's objects."""
    real: dict[str, int] = {}
    finite: dict[str, int] = {}
    for statement in module.statements:
        if not isinstance(statement, ObjectDecl):
            continue
        init = statement.init
        expr = getattr(init, "expr", None)
        constructor = getattr(expr, "constructor", None)
        args = getattr(expr, "args", ())
        if not args or not str(args[0]).isdigit():
            continue
        size = int(str(args[0]))
        for name in statement.names:
            if constructor == "Real":
                real[name] = size
            elif constructor == "FinSet":
                finite[name] = size
    return real, finite


def network_shape(
    name: str,
    morphism: MorphismDecl,
    output_rows: int,
    real_widths: Mapping[str, int],
) -> tuple[tuple[int, int], ...]:
    """The ``(rows, columns)`` of every layer's weight, hidden layers first.

    Parameters
    ----------
    name : str
        The morphism's name.
    morphism : MorphismDecl
        The declaration.
    output_rows : int
        The output layer's height: the codomain's width for a function,
        one row block per family head for a kernel.
    real_widths : Mapping[str, int]
        The real width of each ``Real N`` object.

    Returns
    -------
    tuple[tuple[int, int], ...]
        One shape per layer.

    Raises
    ------
    LatentWeightsError
        If a domain factor has no fixed real width.
    """
    factors = getattr(morphism.domain, "components", None) or (morphism.domain,)
    fan_in = 0
    for factor in factors:
        width = _real_width(factor, real_widths)
        if width is None:
            raise LatentWeightsError(
                f"morphism {name!r} declares latent weights, but a factor of its "
                "domain is not a `Real N` object of fixed width",
                morphism.line,
                morphism.col,
            )
        fan_in += width
    shapes: list[tuple[int, int]] = []
    for hidden in hidden_widths(morphism):
        shapes.append((hidden, fan_in))
        fan_in = hidden
    shapes.append((output_rows, fan_in))
    return tuple(shapes)


def _output_rows(
    name: str,
    morphism: MorphismDecl,
    real: Mapping[str, int],
    finite: Mapping[str, int],
    heads: Mapping[str, int],
) -> int:
    """The output layer's height for a function or a family kernel."""
    codomain = morphism.codomain
    role = _option(morphism, "role")
    if isinstance(role, OptionName) and role.value == "function":
        width = _real_width(codomain, real)
        if width is None:
            raise LatentWeightsError(
                f"function {name!r} declares latent weights, but its codomain is "
                "not a `Real N` object of fixed width",
                morphism.line,
                morphism.col,
            )
        return width
    family = morphism.init_family.family if morphism.init_family is not None else None
    if family is None:
        family = getattr(morphism.init_expr, "name", None)
    if family is None or family not in heads:
        raise LatentWeightsError(
            f"morphism {name!r} declares latent weights, but it is neither a "
            "learned function nor a kernel `~ Family` of a family whose "
            f"parameters a network supplies ({', '.join(sorted(heads))})",
            morphism.line,
            morphism.col,
        )
    if family in ("Categorical", "Bernoulli"):
        classes = finite.get(codomain.name) if isinstance(codomain, TypeName) else None
        if classes is None:
            raise LatentWeightsError(
                f"kernel {name!r} declares latent weights over {family}, but its "
                "codomain is not a finite object of fixed size",
                morphism.line,
                morphism.col,
            )
        return 1 if family == "Bernoulli" else classes
    width = _real_width(codomain, real)
    if width is None:
        raise LatentWeightsError(
            f"kernel {name!r} declares latent weights, but its codomain is not a "
            "`Real N` object of fixed width",
            morphism.line,
            morphism.col,
        )
    return width * heads[family]


def latent_weight_layout(
    module: Module, heads: Mapping[str, int]
) -> dict[str, tuple[tuple[str, str, int, int], ...]]:
    """The weight sites of every latent-weight morphism of a module.

    Parameters
    ----------
    module : Module
        The module.
    heads : Mapping[str, int]
        The number of parameter heads of each family a kernel's network
        may supply.

    Returns
    -------
    dict[str, tuple[tuple[str, str, int, int], ...]]
        By morphism name, one ``(weight_site, bias_site, rows, columns)``
        per layer, hidden layers first.

    Raises
    ------
    LatentWeightsError
        If a network's shape cannot be read off its declaration.
    """
    real, finite = _object_widths(module)
    layout: dict[str, tuple[tuple[str, str, int, int], ...]] = {}
    for name, morphism in latent_morphisms(module).items():
        rows = _output_rows(name, morphism, real, finite, heads)
        shapes = network_shape(name, morphism, rows, real)
        sites = weight_site_names(name, len(shapes) - 1)
        layout[name] = tuple(
            (weight, bias, height, width)
            for (weight, bias), (height, width) in zip(sites, shapes, strict=True)
        )
    return layout


def expand_latent_weights(module: Module, heads: Mapping[str, int]) -> Module:
    """Draw every latent network weight as a site of the programs using it.

    Parameters
    ----------
    module : Module
        The parsed module.
    heads : Mapping[str, int]
        The number of parameter heads of each family a kernel's network
        may supply.

    Returns
    -------
    Module
        The module unchanged when it declares no latent weights or has
        already been expanded; otherwise with one finite object per axis
        of each weight tensor declared ahead of the first program, and
        each program without value parameters that reaches a
        latent-weight morphism drawing that morphism's weights first.

    Raises
    ------
    LatentWeightsError
        If a network's shape cannot be read off its declaration, or a
        latent-weight morphism is composed outside a scan.
    """
    layout = latent_weight_layout(module, heads)
    if not layout:
        return module
    _refuse_composite_uses(module, latent_morphisms(module))
    declared = {
        name
        for statement in module.statements
        if isinstance(statement, ObjectDecl)
        for name in statement.names
    }
    if any(f"{name}__w0_rows" in declared for name in layout):
        return module
    morphisms = latent_morphisms(module)
    lines: list[str] = []
    steps_text: dict[str, list[str]] = {}
    for name, layers in layout.items():
        scale = prior_scale(morphisms[name])
        text: list[str] = []
        for index, (weight, bias, rows, cols) in enumerate(layers):
            row_axis, col_axis = f"{name}__w{index}_rows", f"{name}__w{index}_cols"
            lines.append(f"object {row_axis} : FinSet {rows}")
            lines.append(f"object {col_axis} : FinSet {cols}")
            text.append(
                f"    sample {weight} : {row_axis} <- Normal(loc=0.0, scale={scale!r}) "
                f"[over={col_axis}, iid_over={row_axis}]"
            )
            text.append(
                f"    sample {bias} <- Normal(loc=0.0, scale={scale!r}) [over={row_axis}]"
            )
        steps_text[name] = text
    first_axis = f"{next(iter(layout))}__w0_rows"
    body = [line for name in layout for line in steps_text[name]]
    first_site = layout[next(iter(layout))][0][0]
    # A carrier program for the generated steps: it is read for its
    # declarations and steps and never compiled.
    parsed = parse(
        "\n".join(
            [
                *lines,
                f"program weights__ : {first_axis} -> {first_axis}",
                *body,
                f"    return {first_site}",
            ]
        )
    )
    objects = [s for s in parsed.statements if isinstance(s, ObjectDecl)]
    program = next(s for s in parsed.statements if isinstance(s, ProgramDecl))
    by_morphism: dict[str, list[ProgramStep]] = {}
    cursor = 0
    for name, layers in layout.items():
        count = 2 * len(layers)
        by_morphism[name] = list(program.draws[cursor : cursor + count])
        cursor += count
    graph = _references(module)
    statements: list[object] = []
    inserted_objects = False
    for statement in module.statements:
        if isinstance(statement, ProgramDecl) and not inserted_objects:
            statements.extend(objects)
            inserted_objects = True
        if isinstance(statement, ProgramDecl) and not statement.params:
            reached = _reaches(statement.name, morphisms, graph)
            if reached:
                prefix: list[ProgramStep] = []
                for name in reached:
                    prefix.extend(
                        step.with_(line=statement.line, col=statement.col)
                        for step in by_morphism[name]
                    )
                statement = statement.with_(draws=(*prefix, *statement.draws))
        statements.append(statement)
    if not inserted_objects:
        statements.extend(objects)
    return module.with_(statements=tuple(statements))


__all__ = []
