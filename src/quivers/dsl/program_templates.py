"""Instantiating a program at a draw from it.

A draw from a program, ``sample theta <- sub(args)`` or
``sample (u, v) <- sub``, denotes the program's body run in place: the
callee's parameters take the draw's arguments, and its steps join the
caller's under names of the caller's own. The instantiation is a
substitution on the callee's source steps followed by an alpha-renaming
of every name the callee binds, which realises the categorical
substitution lemma (substituting actuals for formals commutes with
denotation up to renaming) and gives every draw of a program its own
latents.

The renaming follows one convention on every runtime: a local ``z`` of a
program drawn under the binder ``theta`` becomes ``theta$z``, and the
name the program returns becomes the binder itself, so ``sample theta <-
school_effects(0.6, School)`` binds ``theta`` to the returned value and
contributes the site ``theta$z``. A parenthesized pattern maps the
program's returned names onto its own positionally, and a single binder
taking a pair binds the pair of the renamed components. ``$`` is no
character of a source identifier, so a renamed name never collides with
one the source declares.
"""

from __future__ import annotations

from collections.abc import Mapping

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    AxisSpec,
    CallStep,
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    LetExprFactor,
    LetExprLiteral,
    LetExprNode,
    LetExprTuple,
    LetExprVar,
    LetStep,
    MarginalizeStep,
    ObjectExpr,
    ObjectParam,
    ObjectProduct,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScalarParam,
    ScoreStep,
    TypeName,
)
from quivers.dsl.let_expr_traversal import map_let_expr, substitute_let_expr


class TemplateError(Exception):
    """A draw's arguments or pattern do not fit the program drawn from.

    Parameters
    ----------
    message : str
        What does not fit.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TemplateBindings(dx.Model):
    """What a draw's arguments substitute into the program drawn from.

    Parameters
    ----------
    objects : dict[str, str]
        The object each object parameter takes.
    values : dict[str, LetExprNode]
        The expression each scalar parameter takes.
    morphisms : dict[str, str]
        The declared morphism each morphism parameter takes.
    """

    objects: dict[str, str]
    values: dict[str, LetExprNode]
    morphisms: dict[str, str]


def template_bindings(
    template: ProgramDecl,
    arguments: tuple[DrawArg, ...],
    bound: Mapping[str, LetExprNode],
) -> TemplateBindings:
    """The substitutions a draw's arguments fix for a program's parameters.

    Parameters
    ----------
    template : ProgramDecl
        The program drawn from.
    arguments : tuple[DrawArg, ...]
        The draw's arguments, in the program's parameter order.
    bound : Mapping[str, LetExprNode]
        Every name bound in the caller, each as the expression a
        reference to it denotes.

    Returns
    -------
    TemplateBindings
        The object each object parameter takes, the expression each
        scalar parameter takes, and the morphism each morphism
        parameter takes.

    Raises
    ------
    TemplateError
        If the argument count differs from the parameter count or an
        argument does not fit its parameter.
    """
    parameters = template.type_params or ()
    if len(arguments) != len(parameters):
        raise TemplateError(
            f"program {template.name!r} takes {len(parameters)} template "
            f"parameter(s); the draw supplies {len(arguments)}"
        )
    objects: dict[str, str] = {}
    values: dict[str, LetExprNode] = {}
    morphisms: dict[str, str] = {}
    for argument, parameter in zip(arguments, parameters, strict=True):
        if isinstance(parameter, ObjectParam):
            if not isinstance(argument, DrawArgName):
                raise TemplateError(
                    f"parameter {parameter.name!r} of program {template.name!r} "
                    f"takes an object"
                )
            objects[parameter.name] = argument.text
        elif isinstance(parameter, ScalarParam):
            if isinstance(argument, DrawArgScalar):
                values[parameter.name] = LetExprLiteral(value=argument.value)
            elif isinstance(argument, DrawArgName) and argument.text in bound:
                values[parameter.name] = bound[argument.text]
            else:
                raise TemplateError(
                    f"parameter {parameter.name!r} of program {template.name!r} "
                    f"takes a number or a bound name"
                )
        else:
            if not isinstance(argument, DrawArgName):
                raise TemplateError(
                    f"parameter {parameter.name!r} of program {template.name!r} "
                    f"takes a declared morphism"
                )
            morphisms[parameter.name] = argument.text
    return TemplateBindings(objects=objects, values=values, morphisms=morphisms)


def instantiate_program(
    template: ProgramDecl,
    binders: tuple[str, ...],
    bindings: TemplateBindings,
) -> tuple[ProgramStep, ...]:
    """The steps a draw from a program runs in the caller.

    Parameters
    ----------
    template : ProgramDecl
        The program drawn from.
    binders : tuple[str, ...]
        The draw's pattern: one name, or one per returned name.
    bindings : TemplateBindings
        What the draw's arguments substitute for the program's
        parameters.

    Returns
    -------
    tuple[ProgramStep, ...]
        The program's steps under the caller's names, without its
        return; a single binder taking a pair is bound to the pair of
        the renamed components as a final let.

    Raises
    ------
    TemplateError
        If the pattern's arity is neither one nor the number of names
        the program returns.
    """
    returned = template.return_vars
    prefix = "_".join(binders)
    rename: dict[str, str] = {}
    if len(binders) == len(returned):
        for name, binder in zip(returned, binders, strict=True):
            rename[name] = binder
    elif len(binders) != 1:
        raise TemplateError(
            f"pattern ({', '.join(binders)}) destructures {len(binders)} names, "
            f"but program {template.name!r} returns {len(returned)}"
        )
    for name in _bound_names(template.draws):
        rename.setdefault(name, f"{prefix}${name}")
    steps = tuple(
        _rename_step(step, bindings, rename)
        for step in template.draws
        if not isinstance(step, ReturnStep)
    )
    if len(binders) == 1 and len(returned) != 1:
        steps = (
            *steps,
            LetStep(
                name=binders[0],
                value=LetExprTuple(
                    items=tuple(LetExprVar(name=rename[name]) for name in returned)
                ),
            ),
        )
    return steps


def _bound_names(steps: tuple[ProgramStep, ...]) -> list[str]:
    """Every name the steps bind, marginalization scopes included.

    Parameters
    ----------
    steps : tuple[ProgramStep, ...]
        The steps.

    Returns
    -------
    list[str]
        The names in binding order, each once.
    """
    out: list[str] = []
    for step in steps:
        if isinstance(step, (SampleStep, ObserveStep)):
            out.extend(step.vars)
        elif isinstance(step, MarginalizeStep):
            out.append(step.var)
            out.extend(_bound_names(step.scope))
        elif isinstance(step, (LetStep, ScoreStep, CallStep)):
            out.append(step.name)
    seen: set[str] = set()
    unique: list[str] = []
    for name in out:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _rename_step(
    step: ProgramStep,
    bindings: TemplateBindings,
    rename: Mapping[str, str],
) -> ProgramStep:
    """One step under the substitution and the renaming.

    Parameters
    ----------
    step : ProgramStep
        The step.
    bindings : TemplateBindings
        What the draw's arguments substitute for the program's
        parameters.
    rename : Mapping[str, str]
        The caller's name of each name the program binds.

    Returns
    -------
    ProgramStep
        The rewritten step.

    Raises
    ------
    TemplateError
        If a scalar parameter is drawn through where only a number or
        a name fits.
    """
    objects, values, morphisms = bindings.objects, bindings.values, bindings.morphisms
    if isinstance(step, (SampleStep, ObserveStep)):
        return step.with_(
            vars=tuple(rename.get(name, name) for name in step.vars),
            morphism=_rename_morphism(step.morphism, morphisms, rename),
            args=_rename_args(step.args, values, rename),
            index=None if step.index is None else _rename_object(step.index, objects),
            axes=_rename_axes(step.axes, objects),
        )
    if isinstance(step, MarginalizeStep):
        return step.with_(
            var=rename.get(step.var, step.var),
            morphism=_rename_morphism(step.morphism, morphisms, rename),
            args=_rename_args(step.args, values, rename),
            index=None if step.index is None else _rename_object(step.index, objects),
            over=None if step.over is None else objects.get(step.over, step.over),
            over_objs=(
                None
                if step.over_objs is None
                else tuple(objects.get(name, name) for name in step.over_objs)
            ),
            scope=tuple(_rename_step(inner, bindings, rename) for inner in step.scope),
        )
    if isinstance(step, (LetStep, ScoreStep)):
        return step.with_(
            name=rename.get(step.name, step.name),
            value=_rename_expr(step.value, objects, values, rename),
        )
    if isinstance(step, CallStep):
        return step.with_(
            name=rename.get(step.name, step.name),
            call=step.call.with_(
                arguments=tuple(
                    _rename_expr(argument, objects, values, rename)
                    for argument in step.call.arguments
                )
            ),
        )
    return step


def _rename_morphism(
    name: str, morphisms: Mapping[str, str], rename: Mapping[str, str]
) -> str:
    """The morphism a step draws through, under the substitutions.

    Parameters
    ----------
    name : str
        The name the step names.
    morphisms : Mapping[str, str]
        The declared morphism each morphism parameter takes.
    rename : Mapping[str, str]
        The caller's name of each name the program binds.

    Returns
    -------
    str
        The morphism the parameter takes, the renamed local a
        let-bound morphism was renamed to, or the name itself.
    """
    replacement = morphisms.get(name)
    if replacement is not None:
        return replacement
    return rename.get(name, name)


def _rename_args(
    args: tuple[DrawArg, ...] | None,
    values: Mapping[str, LetExprNode],
    rename: Mapping[str, str],
) -> tuple[DrawArg, ...] | None:
    """Draw arguments under the substitution and the renaming.

    Parameters
    ----------
    args : tuple[DrawArg, ...] | None
        The arguments.
    values : Mapping[str, LetExprNode]
        The expression each scalar parameter takes.
    rename : Mapping[str, str]
        The caller's name of each name the program binds.

    Returns
    -------
    tuple[DrawArg, ...] | None
        The rewritten arguments.

    Raises
    ------
    TemplateError
        If a scalar parameter used as a draw argument takes an
        expression that is neither a number nor a name.
    """
    if args is None:
        return None
    out: list[DrawArg] = []
    for argument in args:
        if isinstance(argument, DrawArgNamed):
            rewritten = _rename_args((argument.value,), values, rename)
            assert rewritten is not None
            out.append(argument.with_(value=rewritten[0]))
        elif isinstance(argument, DrawArgDist):
            out.append(
                argument.with_(args=_rename_args(argument.args, values, rename) or ())
            )
        elif isinstance(argument, DrawArgList):
            out.append(
                argument.with_(items=_rename_args(argument.items, values, rename) or ())
            )
        elif isinstance(argument, DrawArgName):
            replacement = values.get(argument.text)
            if replacement is not None:
                out.append(_draw_argument(replacement, argument))
            else:
                out.append(
                    argument.with_(text=rename.get(argument.text, argument.text))
                )
        elif isinstance(argument, DrawArgIndex):
            out.append(
                argument.with_(
                    name=rename.get(argument.name, argument.name),
                    indices=tuple(
                        rename.get(index, index) for index in argument.indices
                    ),
                )
            )
        else:
            out.append(argument)
    return tuple(out)


def _draw_argument(value: LetExprNode, argument: DrawArgName) -> DrawArg:
    """A scalar parameter's expression as a draw argument.

    Parameters
    ----------
    value : LetExprNode
        The expression the parameter takes.
    argument : DrawArgName
        The argument naming the parameter, for its position.

    Returns
    -------
    DrawArg
        A number or a name.

    Raises
    ------
    TemplateError
        If the expression is neither.
    """
    if isinstance(value, LetExprLiteral):
        return DrawArgScalar(
            value=float(value.value), line=argument.line, col=argument.col
        )
    if isinstance(value, LetExprVar):
        return DrawArgName(text=value.name, line=argument.line, col=argument.col)
    raise TemplateError(
        f"parameter {argument.text!r} is drawn through, so it takes a number or a name"
    )


def _rename_object(expr: ObjectExpr, objects: Mapping[str, str]) -> ObjectExpr:
    """An object expression under the object substitution.

    Parameters
    ----------
    expr : ObjectExpr
        The expression.
    objects : Mapping[str, str]
        The object each object parameter takes.

    Returns
    -------
    ObjectExpr
        The rewritten expression.
    """
    if isinstance(expr, TypeName):
        replacement = objects.get(expr.name)
        return expr if replacement is None else expr.with_(name=replacement)
    if isinstance(expr, ObjectProduct):
        return expr.with_(
            components=tuple(_rename_object(item, objects) for item in expr.components)
        )
    return expr


def _rename_axes(axes: AxisSpec | None, objects: Mapping[str, str]) -> AxisSpec | None:
    """An axis annotation under the object substitution.

    Parameters
    ----------
    axes : AxisSpec | None
        The annotation.
    objects : Mapping[str, str]
        The object each object parameter takes.

    Returns
    -------
    AxisSpec | None
        The rewritten annotation.
    """
    if axes is None:
        return None
    return axes.with_(
        over=tuple(objects.get(name, name) for name in axes.over),
        iid_over=tuple(objects.get(name, name) for name in axes.iid_over),
    )


def _rename_expr(
    expr: LetExprNode,
    objects: Mapping[str, str],
    values: Mapping[str, LetExprNode],
    rename: Mapping[str, str],
) -> LetExprNode:
    """A let expression under the substitution and the renaming.

    One capture-avoiding substitution over the whole expression: a
    scalar parameter becomes the expression it takes and a name the
    program binds becomes the caller's name wherever it occurs free; a
    lambda parameter or factor binder shadows a name of the same
    spelling and is renamed if it would capture a substituted name. An
    object parameter in a factor binder's index becomes the object it
    takes.

    Parameters
    ----------
    expr : LetExprNode
        The expression.
    objects : Mapping[str, str]
        The object each object parameter takes.
    values : Mapping[str, LetExprNode]
        The expression each scalar parameter takes.
    rename : Mapping[str, str]
        The caller's name of each name the program binds.

    Returns
    -------
    LetExprNode
        The rewritten expression.

    Raises
    ------
    TypeError
        If the expression contains a let-expression variant the
        substitution does not cover.
    """
    substitution: dict[str, LetExprNode] = {
        name: LetExprVar(name=target) for name, target in rename.items()
    }
    substitution.update(values)

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
                binder.with_(index=_rename_object(binder.index, objects))
                for binder in node.binders
            )
        )

    return map_let_expr(substitute_let_expr(expr, substitution), substitute_objects)


__all__ = []
