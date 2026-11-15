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
school_effects[School](0.6)`` binds ``theta`` to the returned value and
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
    QiecBinder,
    QiecValueParameter,
    QiecIndexBinder,
    QiecTypeBinder,
    QiecEffectBinder,
    MorphismParam,
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    LetExprFactor,
    LetExprLiteral,
    LetExprBool,
    LetExprUnit,
    LetExprList,
    LetExprIndex,
    LetExprNode,
    LetExprTuple,
    LetExprVar,
    LetStep,
    MarginalizeStep,
    ObjectExpr,
    ObjectCoproduct,
    ObjectSlash,
    ObjectEffectApply,
    ContinuousConstructor,
    DiscreteConstructor,
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
        The expression each runtime value parameter takes.
    morphisms : dict[str, str]
        The declared morphism each morphism parameter takes.
    statics : dict[str, DrawArg]
        The authored argument each dependent static binder takes.
    """

    objects: dict[str, str]
    values: dict[str, LetExprNode]
    morphisms: dict[str, str]
    statics: dict[str, DrawArg] = {}


def static_program_parameters(
    template: ProgramDecl,
) -> tuple[QiecBinder | ObjectParam | MorphismParam, ...]:
    """Return static binders in authored telescope order."""
    return tuple(
        sorted(
            (
                *template.binders,
                *(
                    p
                    for p in template.type_params or ()
                    if isinstance(p, ObjectParam | MorphismParam)
                ),
            ),
            key=lambda parameter: (parameter.line, parameter.col),
        )
    )


def runtime_program_parameters(
    template: ProgramDecl,
) -> tuple[QiecValueParameter | ScalarParam, ...]:
    """Return typed runtime parameters without scalar metadata duplicates."""
    if template.value_parameters:
        return template.value_parameters
    return tuple(p for p in template.type_params or () if isinstance(p, ScalarParam))


def validate_program_application(
    template: ProgramDecl,
    arguments: tuple[DrawArg, ...],
    *,
    static_arguments: tuple[DrawArg, ...] = (),
) -> None:
    """Check argument stages before any source substitution occurs."""
    for stage, parameters, supplied in (
        ("static", static_program_parameters(template), static_arguments),
        ("runtime", runtime_program_parameters(template), arguments),
    ):
        if len(supplied) != len(parameters):
            raise TemplateError(
                f"program {template.name!r} takes {len(parameters)} {stage} "
                f"argument(s); the {stage} application supplies {len(supplied)}"
            )
        for parameter, argument in zip(parameters, supplied, strict=True):
            if stage == "static":
                if isinstance(
                    parameter, ObjectParam | MorphismParam
                ) and not isinstance(argument, DrawArgName):
                    raise TemplateError(
                        f"static parameter {parameter.name!r} takes "
                        + (
                            "an object"
                            if isinstance(parameter, ObjectParam)
                            else "a declared morphism"
                        )
                    )
                if (
                    isinstance(parameter, QiecIndexBinder)
                    and isinstance(argument, DrawArgScalar)
                    and (argument.value < 0 or not argument.value.is_integer())
                ):
                    raise TemplateError(
                        f"static index {parameter.name!r} requires a natural number"
                    )
            elif isinstance(argument, DrawArgNamed):
                raise TemplateError(
                    f"runtime parameters of program {template.name!r} are positional values"
                )


def draw_argument_expression(argument: DrawArg) -> LetExprNode:
    """Represent a value argument with the existing pure expression tree."""
    if isinstance(argument, DrawArgScalar):
        return LetExprLiteral(
            value=argument.value, line=argument.line, col=argument.col
        )
    if isinstance(argument, DrawArgName):
        if argument.text in {"true", "false"}:
            return LetExprBool(
                value=argument.text == "true", line=argument.line, col=argument.col
            )
        if argument.text == "unit":
            return LetExprUnit(line=argument.line, col=argument.col)
        return LetExprVar(name=argument.text, line=argument.line, col=argument.col)
    if isinstance(argument, DrawArgList):
        return LetExprList(
            items=tuple(draw_argument_expression(item) for item in argument.items)
        )
    if isinstance(argument, DrawArgIndex):

        def index_expression(index: str) -> LetExprNode:
            try:
                return LetExprLiteral(value=int(index))
            except ValueError:
                return LetExprVar(name=index)

        return LetExprIndex(
            array=LetExprVar(name=argument.name),
            indices=tuple(index_expression(index) for index in argument.indices),
        )
    raise TemplateError(
        f"a runtime program argument must be a value, not {argument.kind!r}"
    )


def template_bindings(
    template: ProgramDecl,
    arguments: tuple[DrawArg, ...],
    bound: Mapping[str, LetExprNode],
    *,
    static_arguments: tuple[DrawArg, ...] = (),
) -> TemplateBindings:
    """Substitute separately checked static and runtime program arguments.

    The binding records syntax only. The caller checks declared runtime types
    in its typed context before executing the instantiated body.
    """
    validate_program_application(template, arguments, static_arguments=static_arguments)
    objects: dict[str, str] = {}
    values: dict[str, LetExprNode] = {}
    morphisms: dict[str, str] = {}
    statics: dict[str, DrawArg] = {}
    for parameter, argument in zip(
        static_program_parameters(template), static_arguments, strict=True
    ):
        if isinstance(parameter, ObjectParam):
            assert isinstance(argument, DrawArgName)
            objects[parameter.name] = argument.text
        elif isinstance(parameter, MorphismParam):
            assert isinstance(argument, DrawArgName)
            morphisms[parameter.name] = argument.text
        else:
            assert isinstance(
                parameter, QiecIndexBinder | QiecTypeBinder | QiecEffectBinder
            )
            statics[parameter.name] = argument
            if isinstance(parameter, QiecIndexBinder):
                values[parameter.name] = draw_argument_expression(argument)
    for parameter, argument in zip(
        runtime_program_parameters(template), arguments, strict=True
    ):
        expression = draw_argument_expression(argument)
        if isinstance(expression, LetExprVar) and expression.name in bound:
            expression = bound[expression.name]
        values[parameter.name] = expression
    return TemplateBindings(
        objects=objects, values=values, morphisms=morphisms, statics=statics
    )


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
    body_bound = set(_bound_names(template.draws)) - bindings.values.keys()
    aliases: list[tuple[str, str]] = []
    prefix = "_".join(binders)
    rename: dict[str, str] = {}
    if len(binders) == len(returned):
        for name, binder in zip(returned, binders, strict=True):
            if name in body_bound and name not in rename:
                rename[name] = binder
            else:
                aliases.append((binder, name))
    elif len(binders) != 1:
        raise TemplateError(
            f"pattern ({', '.join(binders)}) destructures {len(binders)} names, "
            f"but program {template.name!r} returns {len(returned)}"
        )
    for name in _bound_names(template.draws):
        if name not in bindings.values:
            rename.setdefault(name, f"{prefix}${name}")
    steps = tuple(
        _rename_step(step, bindings, rename)
        for step in template.draws
        if not isinstance(step, ReturnStep)
    )
    steps = (
        *steps,
        *(
            LetStep(
                name=binder,
                value=_rename_expr(
                    LetExprVar(name=name), bindings.objects, bindings.values, rename
                ),
            )
            for binder, name in aliases
            if not (
                isinstance(bindings.values.get(name), LetExprVar)
                and bindings.values[name].name == binder
            )
        ),
    )
    if len(binders) == 1 and len(returned) != 1:
        steps = (
            *steps,
            LetStep(
                name=binders[0],
                value=LetExprTuple(
                    items=tuple(
                        _rename_expr(
                            LetExprVar(name=name),
                            bindings.objects,
                            bindings.values,
                            rename,
                        )
                        for name in returned
                    )
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
    objects = {
        **{
            name: actual.text
            for name, actual in bindings.statics.items()
            if isinstance(actual, DrawArgName)
        },
        **bindings.objects,
    }
    values, morphisms = bindings.values, bindings.morphisms
    if isinstance(step, (SampleStep, ObserveStep)):
        variables: list[str] = []
        for name in step.vars:
            if isinstance(step, ObserveStep) and name in values:
                actual = values[name]
                if not isinstance(actual, LetExprVar):
                    raise TemplateError(
                        f"observed parameter {name!r} requires a named runtime value"
                    )
                variables.append(actual.name)
            else:
                variables.append(rename.get(name, name))
        return step.with_(
            vars=tuple(variables),
            static_args=_rename_static_args(step.static_args, bindings),
            morphism=_rename_morphism(step.morphism, morphisms, rename),
            args=_rename_args(step.args, values, rename),
            index=None if step.index is None else _rename_object(step.index, objects),
            axes=_rename_axes(step.axes, objects),
        )
    if isinstance(step, MarginalizeStep):
        return step.with_(
            static_args=_rename_static_args(step.static_args, bindings),
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


def _rename_static_args(
    arguments: tuple[DrawArg, ...], bindings: TemplateBindings
) -> tuple[DrawArg, ...]:
    """Substitute static names without moving runtime values across stages."""
    names = {**bindings.objects, **bindings.morphisms}
    rewritten: list[DrawArg] = []
    for argument in arguments:
        if isinstance(argument, DrawArgName):
            if argument.text in names:
                argument = argument.with_(text=names[argument.text])
            else:
                argument = bindings.statics.get(argument.text, argument)
        rewritten.append(argument)
    return tuple(rewritten)


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
                    name=_argument_name(argument.name, values, rename),
                    indices=tuple(
                        _argument_name(index, values, rename, index=True)
                        for index in argument.indices
                    ),
                )
            )
        else:
            out.append(argument)
    return tuple(out)


def _argument_name(
    name: str,
    values: Mapping[str, LetExprNode],
    rename: Mapping[str, str],
    *,
    index: bool = False,
) -> str:
    """Substitute an index or array name in the restricted draw syntax."""
    replacement = values.get(name)
    if replacement is None:
        return rename.get(name, name)
    if isinstance(replacement, LetExprVar):
        return replacement.name
    if index and isinstance(replacement, LetExprLiteral):
        numeric = float(replacement.value)
        if numeric.is_integer():
            return str(int(numeric))
    raise TemplateError(f"indexed parameter {name!r} requires a named value")


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


def substitute_program_object(
    expression: ObjectExpr, substitutions: Mapping[str, ObjectExpr]
) -> ObjectExpr:
    """Substitute static objects throughout categorical type syntax.

    Substitution is simultaneous: actual expressions are inserted unchanged
    except for the reference's source span. Constructor dimensions and effect
    names require a named actual; object operators accept arbitrary expressions.
    """
    if not substitutions:
        return expression
    if isinstance(expression, TypeName):
        replacement = substitutions.get(expression.name)
        return (
            expression
            if replacement is None
            else replacement.with_(line=expression.line, col=expression.col)
        )
    if isinstance(expression, ObjectProduct | ObjectCoproduct):
        return expression.with_(
            components=tuple(
                substitute_program_object(item, substitutions)
                for item in expression.components
            )
        )
    if isinstance(expression, ObjectSlash):
        return expression.with_(
            result=substitute_program_object(expression.result, substitutions),
            argument=substitute_program_object(expression.argument, substitutions),
        )

    def name_actual(name: str) -> str:
        replacement = substitutions.get(name)
        if replacement is None:
            return name
        if not isinstance(replacement, TypeName):
            raise TemplateError(
                f"static constructor dimension or effect {name!r} requires a named object or literal"
            )
        return replacement.name

    if isinstance(expression, ObjectEffectApply):
        return expression.with_(
            effect=name_actual(expression.effect),
            args=tuple(
                substitute_program_object(item, substitutions)
                for item in expression.args
            ),
        )
    if isinstance(expression, DiscreteConstructor | ContinuousConstructor):

        def option_actual(value: float | int | str) -> float | int | str:
            if not isinstance(value, str) or value not in substitutions:
                return value
            rewritten = name_actual(value)
            try:
                return int(rewritten)
            except ValueError:
                return rewritten

        return expression.with_(
            args=tuple(name_actual(item) for item in expression.args),
            kwargs={
                key: option_actual(value) for key, value in expression.kwargs.items()
            },
        )
    raise TypeError(f"object substitution does not cover {type(expression).__name__}")


def _rename_object(expr: ObjectExpr, objects: Mapping[str, str]) -> ObjectExpr:
    """Substitute named objects while preserving authored source spans."""
    return substitute_program_object(
        expr, {name: TypeName(name=actual) for name, actual in objects.items()}
    )


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
