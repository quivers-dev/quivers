"""Bidirectional `didactic.api.Lens` from `Formula` to a
QVR [`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module] AST.

The compilation from a formula to a QVR program is a panproto-style
*lens* whose complement is the strict subset of `Formula`
fields that are not recoverable from the emitted
`Module`, packaged as `FormulaData`. The forward
direction produces ``(module, formula_data)`` where the structural
fields of the formula (which columns exist, intercept flag, random
effect group / slope structure, response identifier) are encoded as
QVR latent / let / observe shape and the un-encodable fields (per-row
data, original identifier names lost to `_qvr_name`'s
non-alphanumeric → underscore substitution, the ``term`` / ``name``
presentation labels, the original formula string) ride in
`FormulaData`. The backward direction recovers the structure
by calling `_decode_module` on the target and fuses it with the
complement.

The lens never touches strings: the target is a `Module`
that the existing [`quivers.dsl.compiler.Compiler`][quivers.dsl.compiler.Compiler] consumes
directly, identical in shape to one produced by
[`quivers.dsl.parser.parse`][quivers.dsl.parser.parse].

Emitted structure (one named scalar coefficient per design-matrix
*column*, matching the brms / lme4 canonical layout in which
``poly(x, 2)`` produces two coefficients ``poly(x, 2)_1`` and
``poly(x, 2)_2``):

* One ``object Resp : N`` declaration per response plate.
* One ``object G : K`` declaration per random-effect grouping
  factor (with ``K`` levels).
* For each fixed-design column ``c``: one scalar latent draw
  inside the program body, ``beta_c <- Normal(0, fixed_prior)``,
  whose scale is autoscaled by ``1 / rms(c)`` so the nominal
  value reads in contribution space and means the same thing on
  a raw predictor and on an orthonormal ``poly`` column. An
  explicit per-name prior is emitted as written. The per-row
  covariate values for ``c`` flow in as a free variable via the
  host-data channel (``observations[c]``).
* For each random-effect group ``(slope | g)``: a
  `HalfNormal` scale latent plus a per-level plate draw,
  with the per-row contribution as a plate-gather
  ``alpha_g[g_idx]`` (or ``beta_g_slope[g_idx] * slope`` for a
  random slope). Single-bar blocks couple coefficients through an LKJ
  Cholesky factor; double bars retain independent effects. Explicit shared
  IDs can couple location and auxiliary coefficients.
* One ``observe`` step closes the program with the family's
  observation kernel applied to the inverse-link of the linear
  predictor.
* Categorical responses use category zero as an identified reference
  and lift every fixed and random term over the remaining ``K - 1``
  logits before applying softmax.
* Gaussian mixtures use a configurable component axis, shared softmax
  weights, centered component offsets around the formula predictor,
  and one positive scale per component.

GetPut holds for every `Formula` ``f``: ``backward(*forward(f)) == f``.
"""

from __future__ import annotations

import math
from typing import Literal, Mapping

import didactic.api as dx
import numpy as np
import torch

from quivers.dsl.ast_nodes import (
    AxisSpec,
    ExportDecl,
    ExprIdent,
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    LetStep,
    DiscreteConstructor,
    DrawArgNamed,
    DrawArg,
    DrawArgList,
    DrawArgDist,
    DrawArgScalar,
    Module,
    ObserveStep,
    OptionEntry,
    OptionName,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    atom_to_draw_arg as _to_draw_arg,
    Statement,
    ObjectDecl,
    TypeFromExpr,
    TypeName,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
)
from quivers.dsl.emit import draw_call_to_source
from quivers.dsl.family_schemas import family_parameter_names
from quivers.formulas.family import Family
from quivers.formulas._group_layout import GroupBlockLayout, GroupMemberLayout
from quivers.formulas._ast import (
    _let,
    _var,
    _add,
    _mul,
    _sub,
    _div,
    _call,
    _index,
    _factor,
)
from quivers.formulas._predictor_graph import (
    PredictorGraph,
    ComponentExpression,
    predictor_items,
)
from quivers.dsl.let_expr_traversal import substitute_let_expr

from quivers.formulas.formula import (
    FixedColumn,
    Formula,
    FormulaData,
    RandomTerm,
    _qvr_name,
)


#: Prior families whose last positional argument is a scale, so an
#: autoscaled default divides it by the column's RMS.
_SCALE_LAST_PRIOR_FAMILIES: frozenset[str] = frozenset(
    {"Normal", "Cauchy", "Laplace", "StudentT", "Logistic"}
)


def _autoscaled_prior_args(
    family_name: str,
    args: tuple[str | float, ...],
    column: "np.ndarray",
) -> tuple[str | float, ...]:
    """Rescale a default coefficient prior for its predictor column.

    For a term ``beta * column``, a coefficient scale ``s`` induces contribution
    scale ``s * rms(column)``. If the prior family has a final positional scale
    argument, this function divides that argument by the column RMS. It leaves
    unsupported families, nonnumeric scales, and columns with nonpositive or
    nonfinite RMS unchanged. This follows the default autoscaling convention
    used by `rstanarm`.
    """
    if family_name not in _SCALE_LAST_PRIOR_FAMILIES or not args:
        return args
    scale = args[-1]
    if not isinstance(scale, float):
        return args
    rms = float(np.sqrt(np.mean(np.square(np.asarray(column, dtype=np.float64)))))
    if not np.isfinite(rms) or rms <= 0.0:
        return args
    return (*args[:-1], scale / rms)


def _parse_prior_call(text: str) -> tuple[str, tuple[str | float, ...]]:
    """Split a brms-style prior template ``"Family(arg, arg, ...)"``
    into its family name and a tuple of argument tokens.  Numeric
    tokens become floats; identifier tokens stay as strings so they
    can refer to other latents in the emitted program.
    """
    text = text.strip()
    if "(" not in text or not text.endswith(")"):
        raise ValueError(
            f"compile_formula: prior {text!r} is not in the form "
            f"`Family(arg1, arg2, ...)`"
        )
    family, _, rest = text.partition("(")
    body = rest[:-1]
    args: list[str | float] = []
    named: dict[str, str | float] = {}
    if body.strip():
        for token in body.split(","):
            token = token.strip()
            parameter: str | None = None
            if "=" in token:
                parameter, token = (part.strip() for part in token.split("=", 1))
                if not parameter:
                    raise ValueError(
                        f"compile_formula: prior {text!r} has an empty parameter name"
                    )
            try:
                value: str | float = float(token)
            except ValueError:
                value = token
            if parameter is None:
                if named:
                    raise ValueError(
                        f"compile_formula: prior {text!r} mixes positional and "
                        "named arguments"
                    )
                args.append(value)
            else:
                if args:
                    raise ValueError(
                        f"compile_formula: prior {text!r} mixes positional and "
                        "named arguments"
                    )
                if parameter in named:
                    raise ValueError(
                        f"compile_formula: prior {text!r} repeats parameter "
                        f"{parameter!r}"
                    )
                named[parameter] = value
    family = family.strip()
    if named:
        parameters = family_parameter_names(family)
        if parameters is None or set(named) != set(parameters):
            expected = ", ".join(parameters or ())
            raise ValueError(
                f"compile_formula: prior {text!r} must name exactly ({expected})"
            )
        return family, tuple(named[parameter] for parameter in parameters)
    return family, tuple(args)


def _draw(
    var: str,
    family: str,
    args: tuple[str | float, ...],
    *,
    index: TypeName | None = None,
    mode: Literal["sample", "score", "marginal"] = "sample",
    weights: str | None = None,
) -> SampleStep | ObserveStep:
    """Build a surface program step for a draw of ``var`` from ``family``.

    ``mode="sample"`` / ``"marginal"`` map to `SampleStep`;
    ``mode="score"`` maps to `ObserveStep`. Marginalisation of
    formula draws is not currently emitted by the formula compiler,
    so the marginal mode collapses to sample here. Bare-string and
    bare-float ``args`` are lifted into the tagged
    [`DrawArg`][quivers.dsl.ast_nodes.DrawArg] shape the AST field
    expects.
    """
    parameters = family_parameter_names(family)
    if parameters is None:
        raise ValueError(f"formula family {family!r} has no QVR parameter schema")
    if len(args) != len(parameters):
        raise ValueError(
            f"formula {family} expects {len(parameters)} argument(s) "
            f"({', '.join(parameters)}), got {len(args)}"
        )
    tagged_args = tuple(
        DrawArgNamed(parameter=parameter, value=_to_draw_arg(argument))
        for parameter, argument in zip(parameters, args, strict=True)
    )
    if mode == "score":
        return ObserveStep(
            vars=(var,),
            morphism=family,
            args=tagged_args,
            index=index,
            options=()
            if weights is None
            else (OptionEntry(key="weights", value=OptionName(value=weights)),),
        )
    return SampleStep(
        vars=(var,),
        morphism=family,
        args=tagged_args,
        index=index,
    )


def _auxiliary_draw(var: str, prior: str, *, lower: float | None = None) -> SampleStep:
    """Parse one prior call, retaining the compositional measure algebra."""
    if "(" not in prior.partition("(")[2] and "[" not in prior:
        family, arguments = _parse_prior_call(prior)
        step = _draw(var, family, arguments)
        assert isinstance(step, SampleStep)
    else:
        from quivers.dsl.parser import parse

        module = parse(
            f"program prior : Input -> Output\n    sample {var} <- {prior}\n    return {var}\n"
        )
        if len(module.statements) != 1:
            raise ValueError(
                "an auxiliary prior must contain a single distribution call"
            )
        program = module.statements[0]
        if not isinstance(program, ProgramDecl) or len(program.draws) != 1:
            raise ValueError(
                "an auxiliary prior must contain a single distribution call"
            )
        step = program.draws[0]
        if not isinstance(step, SampleStep) or step.vars != (var,):
            raise ValueError(
                "an auxiliary prior must contain a single distribution call"
            )

        def named_arguments(
            family: str, arguments: tuple[DrawArg, ...]
        ) -> tuple[DrawArg, ...]:
            """Translate formula-prior positions to canonical named draw arguments.

            Parameters
            ----------
            family : str
                Family named by this prior node.
            arguments : tuple[DrawArg, ...]
                Parsed positional or named prior arguments.

            Returns
            -------
            tuple[DrawArg, ...]
                Named arguments with nested distributions translated recursively.

            Raises
            ------
            ValueError
                If a positional prior argument names no canonical parameter.
            """
            parameters = family_parameter_names(family) or ()
            result = []
            for position, argument in enumerate(arguments):
                name = (
                    argument.parameter if isinstance(argument, DrawArgNamed) else None
                )
                value = (
                    argument.value if isinstance(argument, DrawArgNamed) else argument
                )
                if isinstance(value, DrawArgDist):
                    value = value.with_(args=named_arguments(value.family, value.args))
                if isinstance(value, DrawArgList):
                    value = value.with_(
                        items=tuple(
                            item.with_(args=named_arguments(item.family, item.args))
                            if isinstance(item, DrawArgDist)
                            else item
                            for item in value.items
                        )
                    )
                if name is None:
                    if position >= len(parameters):
                        raise ValueError(
                            f"prior family {family!r} has no positional parameter {position}"
                        )
                    name = parameters[position]
                result.append(DrawArgNamed(parameter=name, value=value))
            return tuple(result)

        step = step.with_(args=named_arguments(step.morphism, step.args or ()))
    if lower is not None:
        if step.morphism == "Restrict":
            cuts = [
                arg.value
                for arg in step.args or ()
                if isinstance(arg, DrawArgNamed) and arg.parameter == "low"
            ]
            if (
                len(cuts) != 1
                or not isinstance(cuts[0], DrawArgScalar)
                or cuts[0].value < lower
            ):
                raise ValueError(
                    f"prior for {var!r} must have a literal lower bound at least {lower}"
                )
        else:
            step = SampleStep(
                vars=(var,),
                morphism="Restrict",
                args=(
                    DrawArgNamed(
                        parameter="base",
                        value=DrawArgDist(family=step.morphism, args=step.args or ()),
                    ),
                    DrawArgNamed(parameter="low", value=DrawArgScalar(value=lower)),
                ),
            )
    return step


def _helmert_contrast(cardinality: int) -> np.ndarray:
    """Return an orthonormal sum-zero basis for ``cardinality`` cells.

    Isotropic Normal coefficients in this basis induce the permutation-
    symmetric Gaussian on the sum-zero subspace. The formula mixture uses
    the same basis for logits and location offsets, removing their common
    direction without singling out one component in the default prior.
    """
    basis = np.zeros((cardinality, cardinality - 1), dtype=np.float64)
    for column in range(cardinality - 1):
        scale = np.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


def _apply_link(eta: LetExprNode, link_name: str) -> LetExprNode:
    if link_name == "identity":
        return eta
    if link_name in {"logit", "log", "softmax"}:
        return _call(
            {"logit": "sigmoid", "log": "exp", "softmax": "softmax"}[link_name], eta
        )
    if link_name == "inverse":
        return _div(LetExprLiteral(value=1.0), eta)
    if link_name == "probit":
        return _mul(
            LetExprLiteral(value=0.5),
            _add(
                LetExprLiteral(value=1.0),
                _call("erf", _div(eta, LetExprLiteral(value=math.sqrt(2.0)))),
            ),
        )
    if link_name == "probit_approx":
        return _call(
            "sigmoid",
            _add(
                _mul(LetExprLiteral(value=0.07056), _mul(eta, _mul(eta, eta))),
                _mul(LetExprLiteral(value=1.5976), eta),
            ),
        )
    if link_name == "log1p":
        return _call("expm1", eta)
    if link_name == "cloglog":
        return _mul(
            LetExprLiteral(value=-1.0),
            _call("expm1", _mul(LetExprLiteral(value=-1.0), _call("exp", eta))),
        )
    if link_name == "cauchit":
        return _add(
            LetExprLiteral(value=0.5),
            _div(_call("atan", eta), LetExprLiteral(value=math.pi)),
        )
    if link_name == "sqrt":
        return _mul(eta, eta)
    if link_name == "inverse_square":
        return _div(LetExprLiteral(value=1.0), _call("sqrt", eta))
    if link_name == "logm1":
        return _add(LetExprLiteral(value=1.0), _call("exp", eta))
    if link_name == "softplus":
        nonnegative = LetExprBinOp(op=">=", left=eta, right=LetExprLiteral(value=0.0))
        zero = LetExprLiteral(value=0.0)
        positive = _call("select", nonnegative, eta, zero)
        negative = _call("select", nonnegative, zero, eta)
        return _call(
            "select",
            nonnegative,
            _add(
                eta,
                _call(
                    "log1p", _call("exp", _mul(LetExprLiteral(value=-1.0), positive))
                ),
            ),
            _call("log1p", _call("exp", negative)),
        )
    if link_name == "tan_half":
        return _mul(LetExprLiteral(value=2.0), _call("atan", eta))
    raise ValueError(f"compile_formula: unsupported link {link_name!r}")


def _append_link(
    steps: list[ProgramStep], name: str, eta: LetExprNode, link_name: str
) -> None:
    """Lower inverse links with shared intermediates and safe tail arithmetic."""
    if link_name == "softit":
        positive = f"{name}_link_positive"
        steps.append(_let(positive, _apply_link(eta, "softplus")))
        value = _var(positive)
        steps.append(_let(name, _div(value, _add(LetExprLiteral(value=1.0), value))))
    elif link_name == "squareplus":
        half, scale, root, magnitude = (
            f"{name}_link_{suffix}" for suffix in ("half", "scale", "root", "magnitude")
        )
        steps.append(_let(half, _mul(LetExprLiteral(value=0.5), eta)))
        t = _var(half)
        steps.append(
            _let(scale, _call("max", LetExprLiteral(value=1.0), _call("abs", t)))
        )
        unit = _div(t, _var(scale))
        reciprocal = _div(LetExprLiteral(value=1.0), _var(scale))
        steps.append(
            _let(
                root,
                _mul(
                    _var(scale),
                    _call("sqrt", _add(_mul(unit, unit), _mul(reciprocal, reciprocal))),
                ),
            )
        )
        nonnegative = LetExprBinOp(op=">=", left=t, right=LetExprLiteral(value=0.0))
        # select (rather than abs) retains the right branch's derivative at 0.
        absolute = _call("select", nonnegative, t, _mul(LetExprLiteral(value=-1.0), t))
        steps.append(_let(magnitude, _add(_var(root), absolute)))
        value = _var(magnitude)
        steps.append(
            _let(
                name,
                _call(
                    "select", nonnegative, value, _div(LetExprLiteral(value=1.0), value)
                ),
            )
        )
    else:
        steps.append(_let(name, _apply_link(eta, link_name)))


def _decode_module(module: Module, layouts: tuple[GroupBlockLayout, ...] = ()) -> dict:
    """Recover the structural part of a `Formula` from the
    emitted `Module`.

    Returns a dict carrying the fields the lens forward can deterministically
    produce from a `Formula`:

    * ``n_obs``: the ``Resp`` cardinality.
    * ``group_cardinalities``: ``{qvr_group_name: K}`` for each
      ``object G : K`` declaration preceding the program block.
    * ``fixed_qvr_names``: list of ``(qvr_name, is_intercept)`` in
      emission order, recovered from the latent ``intercept`` /
      ``beta_<qvr_name>`` declarations.
    * ``random_terms_qvr``: list of ``(qvr_group_name, slope_qvr_or_intercept)``
      in emission order, recovered from the
      ``sigma_<g>_<slope>`` + ``alpha_<g>``/``beta_<g>_<slope>`` pattern.
    * ``response_qvr_name``: the QVR-legal identifier of the
      observe step's target.
    * ``observe_family``: the family name on the observe step.

    The decoder is intentionally narrow: it knows the canonical
    emission shape of `FormulaToQVRModule` and recognises that
    shape only. It is not a general QVR parser.
    """
    n_obs: int | None = None
    group_cardinalities: dict[str, int] = {}
    program: ProgramDecl | None = None
    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            init = stmt.init
            if not isinstance(init, TypeFromExpr):
                continue
            type_expr = init.expr
            cardinality: int | None = None
            if isinstance(type_expr, DiscreteConstructor):
                if type_expr.constructor == "FinSet" and len(type_expr.args) == 1:
                    try:
                        cardinality = int(type_expr.args[0])
                    except ValueError:
                        cardinality = None
            elif isinstance(type_expr, TypeName):
                try:
                    cardinality = int(type_expr.name)
                except ValueError:
                    cardinality = None
            if cardinality is None:
                continue
            for decl_name in stmt.names:
                if decl_name == "Resp":
                    n_obs = cardinality
                elif decl_name.startswith("Formula"):
                    # Compiler-owned axes used to assemble family parameters
                    # (cutpoints, spacings, flattened threshold cells) are not
                    # source-level random-effect grouping factors.
                    continue
                else:
                    group_cardinalities[decl_name] = cardinality
        elif isinstance(stmt, ProgramDecl):
            program = stmt
            break

    if program is None or n_obs is None:
        raise ValueError(
            "_decode_module: module is not a formula-emitted program "
            "(missing Resp object or program declaration)"
        )

    # Walk the program steps in emission order. The lens emits, per
    # the layout in `_build_module`, the following pattern:
    #
    #   * One SampleStep per fixed coefficient (`intercept` or
    #     `beta_<qvr_name>`).
    #   * For non-intercepts, a LetStep `<beta_name>_per_row =
    #     beta_name * <qvr_name>` immediately after the SampleStep.
    #   * One SampleStep `sigma_<g>_<slope>` per random-effect entry,
    #     followed by the plate draw (`alpha_<g>` or
    #     `beta_<g>_<slope>`) and a LetStep `<latent_var>_per_row`.
    #   * Aux-family SampleSteps (one per aux param).
    #   * `let eta = ...`, `let mu = ...`.
    #   * A final ObserveStep indexed by `Resp` bearing the response
    #     qvr_name and the observe-family.
    fixed_qvr_names: list[tuple[str, bool]] = []
    random_terms_qvr: list[tuple[str, str]] = []
    response_qvr_name: str = ""
    observe_family: str = ""

    seen_letnames: set[str] = set()
    for step in program.draws:
        if isinstance(step, LetStep):
            seen_letnames.add(step.name)
            continue
        if isinstance(step, ObserveStep):
            if step.index is not None:
                response_qvr_name = step.vars[0]
                observe_family = step.morphism
            continue
        if not isinstance(step, SampleStep):
            continue
        if len(step.vars) != 1:
            continue
        var = step.vars[0]
        # Fixed effects: `intercept` or `beta_<qvr_name>` with no
        # plate index. Random-effect sigmas are also unindexed, but
        # are followed by an indexed plate draw; distinguish by
        # checking the variable name pattern.
        if step.index is None:
            if var == "intercept":
                fixed_qvr_names.append(("", True))
            elif var.startswith("beta_") and not var.startswith("beta_"):
                # unreachable; placeholder to clarify next branch
                pass
            elif var.startswith("sigma_"):
                # random-effect scale; defer to the plate draw that
                # follows, which carries (group, slope) in its name.
                continue
            elif var.startswith("beta_"):
                qvr_name = var.removeprefix("beta_")
                fixed_qvr_names.append((qvr_name, False))
            else:
                # aux family parameter (e.g. `sigma`, `phi`, ...).
                continue
        else:
            # Indexed plate draw: this is the random-effect latent.
            # Centered emission:    `alpha_<g>` or `beta_<g>_<slope>`
            # Non-centered emission: `z_<g>_<slope>`, the unit-scale
            # standard-Normal that the let-expression scales by sigma.
            # `alpha_<g>` ↔ intercept slope; `beta_<g>_<slope>` ↔
            # named slope.
            if not isinstance(step.index, TypeName):
                continue
            qgroup = step.index.name
            if qgroup in {"FormulaCategoryLogit", "FormulaCutpoint"}:
                if var == "intercept":
                    fixed_qvr_names.append(("", True))
                elif var.startswith("beta_"):
                    fixed_qvr_names.append((var.removeprefix("beta_"), False))
                continue
            if qgroup.startswith("FormulaRandomCell_") and var.startswith("z_"):
                for source_group in sorted(group_cardinalities, key=len, reverse=True):
                    prefix = f"z_{source_group}_"
                    if not var.startswith(prefix):
                        continue
                    slope_qvr = var.removeprefix(prefix)
                    random_terms_qvr.append((source_group, slope_qvr))
                    break
                continue
            if var.startswith("alpha_"):
                random_terms_qvr.append((qgroup, "Intercept"))
            elif var.startswith("beta_"):
                # Strip `beta_<qgroup>_` to isolate the slope qvr_name.
                prefix = f"beta_{qgroup}_"
                if var.startswith(prefix):
                    slope_qvr = var.removeprefix(prefix)
                    random_terms_qvr.append((qgroup, slope_qvr))
            elif var.startswith("z_"):
                # Non-centered form: `z_<qgroup>_<slope>`.
                prefix = f"z_{qgroup}_"
                if var.startswith(prefix):
                    slope_qvr = var.removeprefix(prefix)
                    if slope_qvr == "Intercept":
                        random_terms_qvr.append((qgroup, "Intercept"))
                    else:
                        random_terms_qvr.append((qgroup, slope_qvr))

    raw_names = {
        step.vars[0]
        for step in program.draws
        if isinstance(step, SampleStep) and len(step.vars) == 1
    }
    for layout in layouts:
        if not set(layout.raw_sites) <= raw_names:
            continue
        for member in layout.members:
            term = (_qvr_name(layout.group), _qvr_name(member.slope))
            if (
                not member.prefix
                and member.coefficient in seen_letnames
                and term not in random_terms_qvr
            ):
                random_terms_qvr.append(term)

    return {
        "n_obs": n_obs,
        "group_cardinalities": group_cardinalities,
        "fixed_qvr_names": fixed_qvr_names,
        "random_terms_qvr": random_terms_qvr,
        "response_qvr_name": response_qvr_name,
        "observe_family": observe_family,
    }


class FormulaToQVRModule(dx.Lens[Formula, Module, FormulaData]):
    """Translate a `Formula` to a QVR `Module` AST.

    A typed `didactic.api.Lens` whose complement is a
    `FormulaData` carrier containing the fields of the source
    `Formula` that are *not* recoverable from the emitted
    `Module`. The per-row data arrays, the original (pre-
    `_qvr_name`) identifiers, the per-column ``term`` / ``name``
    presentation labels, and the original formula string travel in
    the complement; everything else (which columns there are, the
    intercept / random-term structure, the family choice) is decoded
    back out of the Module by `_decode_module`.

    Parameters
    ----------
    family : Family
        Response family from `quivers.formulas.families`.
    fixed_prior : str
        Default prior for fixed-effect coefficients, in the surface
        form ``"Family(arg, arg, ...)"``; numeric args become floats,
        identifier args stay as variable references in the emitted
        program.
    random_scale_prior : str
        Default prior for random-effect scale parameters.
    correlation_prior : str
        Default LKJ Cholesky prior for correlated coefficient blocks.
    user_priors : Mapping[str, str]
        Per-name prior overrides keyed by the latent's variable
        name in the emitted module.
    reparameterize : {"centered", "noncentered"}
        Conditional Gaussian coordinates for scalar, category-specific and
        component-valued group effects, and grouped threshold log slacks.
    binomial_trials : int or str
        Nonnegative common trial count or a per-row trial-count column.
    thresholds_by : str or None
        Grouping factor whose levels receive partially pooled ordinal
        cutpoint spacings.
    mixture_components : int
        Number of components for ``family="mixture"``; must be at least two.
    predictor_name : str or None
        Host-data name for an external differentiable predictor.
    distributional : Mapping[str, Formula] or None
        Auxiliary-parameter formulas on the same response rows. Each
        predictor receives its own coefficients, covariates, and group effects;
        a ``dpar_<parameter>_`` prefix distinguishes their names. Group
        coding must agree across predictors.

    fixed_parameters : Mapping[str, float] or None
        Auxiliary parameters bound to constants on their natural scale, without
        latent sites or priors. Cannot overlap distributional formulas.

    Raises
    ------
    ValueError
        If an option is out of range or does not apply to ``family``:
        ``thresholds_by`` outside the cumulative family,
        ``binomial_trials`` for a family without trials or outside the
        exact finite-trial range from zero through ``2**53``,
        ``mixture_components`` below two or outside the mixture
        family, or an unknown ``reparameterize`` value.

    Notes
    -----
    GetPut: `backward` ``(forward(f))`` ``=`` ``f`` for every
    `Formula` ``f``. PutGet holds on pairs ``(t, c)`` for
    which ``t`` is in the image of `forward` and ``c`` is the
    corresponding `FormulaData`.
    """

    def __init__(
        self,
        family: Family,
        *,
        fixed_prior: str = "Normal(0.0, 5.0)",
        random_scale_prior: str = "HalfNormal(1.0)",
        user_priors: Mapping[str, str] | None = None,
        reparameterize: Literal["centered", "noncentered"] = "noncentered",
        binomial_trials: int | str | None = None,
        thresholds_by: str | None = None,
        mixture_components: int = 2,
        predictor_name: str | None = None,
        distributional: Mapping[str, Formula] | None = None,
        predictor_graph: PredictorGraph | None = None,
        correlation_prior: str = "LKJCholesky(1.0)",
        fixed_parameters: Mapping[str, float] | None = None,
    ) -> None:
        self._family = family
        if (
            family.name == "frechet"
            and next(aux for aux in family.aux_params if aux.name == "nu").link.name
            != "logm1"
        ):
            raise ValueError(
                "frechet nu regression currently requires the logm1 link to ensure nu > 1"
            )
        self._fixed_parameters = dict(fixed_parameters or {})
        self._correlation_prior = correlation_prior
        self._distributional = dict(distributional or {})
        self._predictor_graph = predictor_graph
        auxiliary_names = {aux.name for aux in family.aux_params}
        expression_roots = (
            set(predictor_graph.roots) - {"mu"}
            if predictor_graph is not None
            else set()
        )
        if expression_roots - auxiliary_names:
            raise ValueError(
                f"unknown distributional graph roots {sorted(expression_roots - auxiliary_names)}"
            )
        if expression_roots & (
            self._distributional.keys() | self._fixed_parameters.keys()
        ):
            raise ValueError(
                "a family parameter cannot have multiple predictor definitions"
            )
        unknown_fixed = self._fixed_parameters.keys() - auxiliary_names
        if unknown_fixed:
            raise ValueError(
                f"unknown fixed auxiliary parameters {sorted(unknown_fixed)}"
            )
        conflicting_fixed = self._fixed_parameters.keys() & self._distributional.keys()
        if conflicting_fixed:
            raise ValueError(
                f"fixed parameters cannot have distributional formulas: {sorted(conflicting_fixed)}"
            )
        for name, value in self._fixed_parameters.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"fixed parameter {name!r} must be a finite real number"
                )
            if name in {"quantile", "bias"}:
                if not 0 < value < 1:
                    raise ValueError(
                        f"fixed parameter {name!r} must be strictly between zero and one"
                    )
            elif name in {"zi", "zoi", "coi", "hu"}:
                if not 0 <= value <= 1:
                    raise ValueError(f"fixed parameter {name!r} must be in [0, 1]")
            elif name in {"alpha", "xi"}:
                pass
            elif name == "ndt":
                if value < 0:
                    raise ValueError("fixed parameter 'ndt' must be nonnegative")
            elif name == "nu" and family.name == "frechet":
                if value <= 1:
                    raise ValueError("frechet fixed parameter 'nu' must exceed one")
            elif value <= 0:
                raise ValueError(f"fixed parameter {name!r} must be strictly positive")
        unknown = self._distributional.keys() - auxiliary_names
        if unknown:
            raise ValueError(
                f"FormulaToQVRModule: unknown distributional parameters {sorted(unknown)}; "
                f"choices are {sorted(auxiliary_names)}"
            )
        self._fixed_prior = fixed_prior
        self._random_scale_prior = random_scale_prior
        self._user_priors: Mapping[str, str] = dict(user_priors or {})
        self._group_layouts: list[GroupBlockLayout] = []
        conflicting = (
            self._distributional.keys() | expression_roots
        ) & self._user_priors.keys()
        if conflicting:
            raise ValueError(
                f"priors on distributionally regressed parameters {sorted(conflicting)} must name their coefficients, such as 'dpar_sigma_intercept'"
            )
        if thresholds_by is not None and (
            family.ordinal is None
            or family.ordinal.kind not in {"cumulative", "hurdle_cumulative"}
        ):
            raise ValueError(
                "FormulaToQVRModule: thresholds_by is only valid for the "
                "cumulative family or hurdle_cumulative family"
            )
        from quivers.formulas._response import resolve_trials

        binomial_trials = resolve_trials(family, binomial_trials)
        if family.trials_parameter is None and binomial_trials != 1:
            raise ValueError(
                "FormulaToQVRModule: binomial_trials is only valid for the "
                "families with a trial-count parameter"
            )
        if binomial_trials is not None and (
            not isinstance(binomial_trials, (int, str))
            or isinstance(binomial_trials, bool)
        ):
            raise ValueError(
                "FormulaToQVRModule: binomial_trials must be a nonnegative integer "
                "or a data-column name"
            )
        if isinstance(binomial_trials, int) and binomial_trials < 0:
            raise ValueError("FormulaToQVRModule: binomial_trials must be nonnegative")
        if isinstance(binomial_trials, int) and binomial_trials > 2**53:
            raise ValueError(
                "FormulaToQVRModule: binomial_trials must be at most 2**53"
            )
        if (
            not isinstance(mixture_components, int)
            or isinstance(mixture_components, bool)
            or mixture_components < 2
        ):
            raise ValueError(
                "FormulaToQVRModule: mixture_components must be an integer "
                "of at least two"
            )
        if family.name != "mixture" and mixture_components != 2:
            raise ValueError(
                "FormulaToQVRModule: mixture_components is only valid for the "
                "mixture family"
            )
        self._binomial_trials = binomial_trials
        self._thresholds_by = thresholds_by
        self._mixture_components = mixture_components
        self._predictor_name = (
            _qvr_name(predictor_name) if predictor_name is not None else None
        )
        if reparameterize not in ("centered", "noncentered"):
            raise ValueError(
                f"FormulaToQVRModule: reparameterize must be 'centered' "
                f"or 'noncentered', got {reparameterize!r}"
            )
        self._reparameterize: Literal["centered", "noncentered"] = reparameterize

    @property
    def group_layouts(self) -> tuple[GroupBlockLayout, ...]:
        """Read the raw-coordinate layouts emitted by the latest forward pass.

        Returns
        -------
        tuple[GroupBlockLayout, ...]
            Immutable group blocks shared with retained prediction plans.
        """
        return tuple(self._group_layouts)

    def forward(self, formula: Formula, /) -> tuple[Module, FormulaData]:
        """Compile a formula to a QVR module and its complement.

        Parameters
        ----------
        formula : Formula
            The parsed formula with its design data.

        Returns
        -------
        tuple[Module, FormulaData]
            The emitted QVR module AST, and the fields of ``formula``
            the module does not encode.

        Raises
        ------
        ValueError
            If the response is incompatible with the family (for
            instance non-integer or non-contiguous category labels
            for a categorical or cumulative family), or a prior does
            not parse as ``"Family(arg, ...)"``.
        """
        self._group_layouts = []
        if (
            self._family.response is not None
            and self._family.response.kind != "categorical"
        ):
            if not formula.response_layout.event_shape:
                raise ValueError(
                    f"{self._family.name} requires an explicit cbind vector response"
                )
            if self._family.response.kind in {"multinomial", "dirichlet_multinomial"}:
                from quivers.formulas._response import count_totals

                totals = count_totals(formula)
                if isinstance(self._binomial_trials, int) and not bool(
                    (totals == self._binomial_trials).all()
                ):
                    raise ValueError(
                        "vector response row totals disagree with supplied trials"
                    )
            else:
                from quivers.formulas._response import validate_simplex

                validate_simplex(formula)
        if formula.response_values.shape[1:] != formula.response_layout.event_shape:
            raise ValueError("response values do not match their declared event layout")
        if formula.response_layout.category_levels:
            if self._family.ordinal is None and self._family.observe_family not in {
                "Bernoulli",
                "Binomial",
                "Categorical",
            }:
                raise ValueError(
                    f"{self._family.name} requires a numeric response, not factor category codes"
                )
            if (
                self._family.observe_family in {"Bernoulli", "Binomial"}
                and len(formula.response_layout.category_levels) != 2
            ):
                raise ValueError(
                    f"{self._family.name} requires exactly two factor response levels"
                )
        if formula.response_layout.event_shape and self._family.name not in {
            "multinomial",
            "dirichlet_multinomial",
            "dirichlet",
            "dirichlet2",
            "logistic_normal",
        }:
            raise ValueError(f"{self._family.name} requires a scalar response event")
        category_columns = tuple(
            col for col in formula.fixed_columns if col.category_specific
        )
        category_terms = tuple(
            term
            for term in formula.random_terms
            if term.column is not None and term.column.category_specific
        )
        if category_columns or category_terms:
            if self._family.ordinal is None:
                raise ValueError("cs terms require an ordinal response family")
            if (
                self._family.ordinal.kind in {"cumulative", "hurdle_cumulative"}
                and any(column.name == "cs(1)" for column in category_columns)
                and any(column.is_intercept for column in formula.fixed_columns)
            ):
                raise ValueError(
                    "fixed cs(1) duplicates the freely estimated cumulative "
                    "threshold profile and proportional intercept; use the ordinary "
                    "intercept and thresholds instead, or use category-specific "
                    "predictors that vary across rows"
                )
            if self._thresholds_by is not None:
                duplicate = next(
                    (
                        term
                        for term in category_terms
                        if term.group == self._thresholds_by
                        and term.column is not None
                        and term.column.name == "cs(1)"
                    ),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(
                        f"category-specific intercept {duplicate.slope!r} for "
                        f"{duplicate.group!r} duplicates the full thresholds_by "
                        "profile; use the threshold profile and its ordinary "
                        "group intercept instead"
                    )
            if (
                self._family.ordinal.kind == "hurdle_cumulative"
                and self._category_count(formula, family=self._family.name) < 3
            ):
                raise ValueError(
                    "cs terms require at least two positive ordinal categories"
                )
        for name, predictor in self._distributional.items():
            if any(col.category_specific for col in predictor.fixed_columns) or any(
                term.column is not None and term.column.category_specific
                for term in predictor.random_terms
            ):
                raise ValueError(f"cs terms are only valid for mu, not {name!r}")
        if self._family.name in {"shifted_lognormal", "frechet"}:
            response = np.asarray(formula.response_values)
            if (
                response.size == 0
                or not np.isfinite(response).all()
                or np.any(response <= 0)
            ):
                raise ValueError(
                    f"{self._family.name} requires finite positive response values"
                )
            ndt = self._fixed_parameters.get("ndt")
            if ndt is not None and np.any(response <= ndt):
                raise ValueError("shifted_lognormal responses must exceed fixed ndt")
        if formula.modifiers.decision is not None and self._family.name != "wiener":
            raise ValueError("a decision response modifier requires the wiener family")
        if self._family.name == "wiener":
            decision = formula.modifiers.decision
            response = np.asarray(formula.response_values)
            if decision is None:
                raise ValueError(
                    "wiener requires a decision column: response | dec(column) or decision=column"
                )
            if (
                response.ndim != 1
                or not np.isfinite(response).all()
                or np.any(response <= 0)
            ):
                raise ValueError("wiener requires finite positive reaction times")
            if (
                decision.data.shape != response.shape
                or decision.data.dtype.kind not in "iu"
                or not np.isin(decision.data, [0, 1]).all()
            ):
                raise ValueError(
                    "decisions must be checked integer 0/1 values aligned with response rows"
                )
            if "ndt" in self._fixed_parameters and np.any(
                response <= self._fixed_parameters["ndt"]
            ):
                raise ValueError("wiener reaction times must exceed fixed ndt")
            aliases = {"formula_response", "formula_decision"}
            if aliases & {
                _qvr_name(formula.response_name),
                self._predictor_name,
                *(column.qvr_name for column in formula.fixed_columns),
            }:
                raise ValueError(
                    "formula_response and formula_decision are reserved response input aliases"
                )
        if self._family.name == "zero_inflated_beta":
            response = np.asarray(formula.response_values)
            if (
                response.size == 0
                or not np.isfinite(response).all()
                or np.any(response < 0)
                or np.any(response >= 1)
            ):
                raise ValueError(
                    "zero_inflated_beta requires finite response values in [0, 1)"
                )
        if formula.modifiers.standard_error is not None:
            error = formula.modifiers.standard_error
            if self._family.name not in {"gaussian", "student_t", "skew_normal"}:
                raise ValueError(
                    "se requires gaussian, student_t, or skew_normal responses"
                )
            if not error.sigma and (
                "sigma" in self._distributional
                or "sigma" in self._fixed_parameters
                or "sigma" in self._user_priors
            ):
                raise ValueError(
                    "se(sigma=False) excludes a separate residual sigma; remove its formula, fixed value, or prior"
                )
        if (
            formula.modifiers.censoring is not None
            or formula.modifiers.truncation is not None
        ) and (
            self._family.ordinal is not None
            or self._family.response is not None
            or self._family.name == "wiener"
        ):
            raise ValueError(
                "censoring/truncation requires an ordinary scalar response family"
            )
        if (
            formula.modifiers.trials is not None
            and self._family.trials_parameter is None
        ):
            raise ValueError(
                "trials requires a response family with finite trial counts"
            )
        role_aliases = {
            column.qvr_name
            for role, column in formula.modifiers.columns()
            if role not in {"weights", "decision"}
        }
        if formula.modifiers.standard_error is not None:
            role_aliases.update({"formula_sigma", "formula_error_reference"})
        collisions = role_aliases & {
            _qvr_name(formula.response_name),
            self._predictor_name,
            *(column.qvr_name for column in formula.fixed_columns),
        }
        if collisions:
            raise ValueError(
                f"{sorted(collisions)} are reserved response input aliases"
            )
        if formula.modifiers.weights is not None:
            weights = formula.modifiers.weights.data
            if weights.shape != formula.response_values.shape[:1]:
                raise ValueError("weights must align with every evaluated response row")
            if (
                weights.dtype.kind not in "iuf"
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
            ):
                raise ValueError("weights must be finite nonnegative real values")
            if (
                _qvr_name(formula.response_name) == "formula_weights"
                or self._predictor_name == "formula_weights"
                or any(
                    column.qvr_name == "formula_weights"
                    for column in formula.fixed_columns
                )
            ):
                raise ValueError(
                    "formula_weights is reserved for response likelihood weights"
                )
        group_codings: dict[str, tuple[str, tuple[str, ...], tuple[int, ...]]] = {}
        for name, _, predictor in self._predictor_items(formula):
            if predictor.response_values.shape != formula.response_values.shape:
                raise ValueError(
                    f"distributional parameter {name!r} has a different row count"
                )
            reserved_offsets = {
                f"formula_offset_{index}" for index, _ in enumerate(predictor.offsets)
            }
            collisions = reserved_offsets & {
                column.qvr_name for column in predictor.fixed_columns
            }
            if collisions:
                raise ValueError(
                    f"offset input names are reserved: {sorted(collisions)}"
                )
            for group, levels in predictor.group_levels.items():
                normalized = _qvr_name(group)
                coding = (group, levels, predictor.group_indices[group])
                existing = group_codings.setdefault(normalized, coding)
                if coding != existing:
                    raise ValueError(
                        f"distributional parameter {name!r} has incompatible coding for {group!r}"
                    )
        if self._distributional and any(
            col.qvr_name.startswith("dpar_") for col in formula.fixed_columns
        ):
            raise ValueError(
                "the 'dpar_' prefix is reserved for distributional coefficients and inputs"
            )
        correlation_names: dict[str, tuple[str, str]] = {}
        for _, prefix, predictor in self._predictor_items(formula):
            for term in predictor.random_terms:
                if term.correlation_id is None:
                    continue
                identity = (
                    f"shared_{term.correlation_id}"
                    if term.shared
                    else prefix + term.correlation_id
                )
                name = f"cor_{_qvr_name(identity)}"
                block = (term.group, identity)
                if correlation_names.setdefault(name, block) != block:
                    raise ValueError(
                        f"correlation name {name!r} is ambiguous across groups or blocks; use distinct shared IDs"
                    )
        module = self._build_module(formula)
        if self._predictor_graph is not None:
            from quivers.dsl.qiec_lowering import lower_qvr_to_qiec

            lower_qvr_to_qiec(module)
        if self._user_priors:
            available = self._prior_calls(formula, module)
            unused = self._user_priors.keys() - available.keys()
            if unused:
                raise ValueError(
                    f"unused prior overrides {sorted(unused)}; available parameters are {sorted(available)}"
                )
        complement = FormulaData(
            formula=formula.formula,
            response_name=formula.response_name,
            response_values=formula.response_values,
            response_layout=formula.response_layout,
            fixed_column_names={
                col.qvr_name: (col.term, col.name) for col in formula.fixed_columns
            },
            fixed_column_data={
                col.qvr_name: col.data
                for col in formula.fixed_columns
                if not col.is_intercept
            },
            group_original_names={_qvr_name(g): g for g in formula.group_levels.keys()},
            group_levels=dict(formula.group_levels),
            group_indices=dict(formula.group_indices),
            group_layouts=self.group_layouts,
            random_terms=formula.random_terms,
            fixed_column_order=tuple(col.qvr_name for col in formula.fixed_columns),
            category_specific_columns=tuple(
                col.qvr_name for col in formula.fixed_columns if col.category_specific
            ),
            offsets=formula.offsets,
            modifiers=formula.modifiers,
        )
        return module, complement

    def get_prior(self, formula: Formula, /) -> dict[str, str]:
        """Inspect the effective priors without compiling or fitting a model.

        Parameters
        ----------
        formula : Formula
            Evaluated response and predictor columns.

        Returns
        -------
        dict[str, str]
            Overrideable parameter names and their canonical distribution
            calls, including the actual scales after column autoscaling.
            Conditional group draws and standard-normal innovations are
            excluded: their laws define the group parameterization.

        Raises
        ------
        ValueError
            If a prior override does not name an available parameter.
        """
        module, _ = self.forward(formula)
        return self._prior_calls(formula, module)

    def _prior_calls(self, formula: Formula, module: Module) -> dict[str, str]:
        """Collect overrideable priors from the same AST used for fitting.

        Parameters
        ----------
        formula : Formula
            Evaluated predictor metadata, combined with auxiliary predictors.
        module : Module
            Emitted native model declarations and program steps.

        Returns
        -------
        dict[str, str]
            Overrideable site names and effective canonical distribution calls.
        """
        formula = self._threshold_location_formula(formula)
        names = {
            aux.name
            for aux in self._family.aux_params
            if aux.name not in self._distributional
            and (
                self._predictor_graph is None
                or aux.name not in self._predictor_graph.roots
            )
            and aux.name not in self._fixed_parameters
        }
        if self._family.ordinal is not None:
            names.update(
                {
                    "cutpoint_log_spacing",
                    "cutpoint_spacing",
                    "cutpoint_offset",
                    "cutpoints_raw",
                    "cutpoint_log_spacing_mean",
                    "cutpoint_log_spacing_scale",
                }
            )
        if self._family.baseline is not None:
            names.add("sbhaz")
        if self._family.name == "mixture":
            names.update(("mixture_logit", "mixture_offset", "mixture_scale"))
        if (
            self._family.response is not None
            and self._family.response.kind == "logistic_normal"
            and len(formula.response_layout.component_names) > 2
        ):
            names.add("rescor")
        for _, prefix, predictor in self._predictor_items(formula):
            for col in predictor.fixed_columns:
                names.add(
                    prefix
                    + ("intercept" if col.is_intercept else f"beta_{col.qvr_name}")
                )
            for term in predictor.random_terms:
                names.add(
                    f"{prefix}sigma_{_qvr_name(term.group)}_{_qvr_name(term.slope)}"
                )
        calls: dict[str, str] = {}
        for statement in module.statements:
            if not isinstance(statement, ProgramDecl):
                continue
            for step in statement.draws:
                if not isinstance(step, SampleStep):
                    continue
                name = step.vars[0]
                if name in names or name.startswith("cor_"):
                    calls[name] = draw_call_to_source(
                        step.morphism, step.args, static_args=step.static_args
                    )
        return calls

    def backward(self, target: Module, complement: FormulaData, /) -> Formula:
        """Recover a formula from a QVR module and its complement.

        The column, intercept, and random-term structure is decoded
        from ``target``; the data arrays and presentation labels come
        from ``complement``.

        Parameters
        ----------
        target : Module
            A QVR module in the image of `forward`.
        complement : FormulaData
            The complement `forward` returned with that module.

        Returns
        -------
        Formula
            The reconstructed formula.
        """
        decoded = _decode_module(target, complement.group_layouts)
        fixed_column_names = complement.fixed_column_names
        fixed_column_data = complement.fixed_column_data

        fixed_columns: list[FixedColumn] = []
        n_obs = decoded["n_obs"]
        ones = np.ones(n_obs, dtype=float)
        for qvr_name, is_intercept in decoded["fixed_qvr_names"]:
            if is_intercept:
                term, name = fixed_column_names.get(
                    "intercept", ("Intercept", "Intercept")
                )
                fixed_columns.append(
                    FixedColumn(
                        term=term,
                        name=name,
                        qvr_name="intercept",
                        is_intercept=True,
                        data=ones,
                    )
                )
            else:
                term, name = fixed_column_names.get(qvr_name, (qvr_name, qvr_name))
                fixed_columns.append(
                    FixedColumn(
                        term=term,
                        name=name,
                        qvr_name=qvr_name,
                        is_intercept=False,
                        category_specific=qvr_name
                        in complement.category_specific_columns,
                        data=fixed_column_data[qvr_name],
                    )
                )

        if complement.fixed_column_order:
            by_name = {column.qvr_name: column for column in fixed_columns}
            fixed_columns = [
                by_name.pop(name)
                for name in complement.fixed_column_order
                if name in by_name
            ]
            fixed_columns.extend(by_name.values())

        random_terms: list[RandomTerm] = []
        for qgroup, slope_qvr in decoded["random_terms_qvr"]:
            group = (
                complement.group_original_names[qgroup]
                if qgroup in complement.group_original_names
                else qgroup
            )
            if slope_qvr == "Intercept":
                random_terms.append(RandomTerm(slope="Intercept", group=group))
            else:
                # Recover the original slope name by inverting via the
                # presentation map when available; otherwise pass the
                # qvr-name through.
                term_name_pairs = {
                    qvr: (term, name)
                    for qvr, (term, name) in fixed_column_names.items()
                }
                _, original_slope = term_name_pairs.get(
                    slope_qvr, (slope_qvr, slope_qvr)
                )
                random_terms.append(RandomTerm(slope=original_slope, group=group))

        original_terms = {
            (_qvr_name(term.group), _qvr_name(term.slope)): term
            for term in complement.random_terms
        }
        random_terms = [
            original_terms.get((_qvr_name(term.group), _qvr_name(term.slope)), term)
            for term in random_terms
        ]
        by_term = {
            (_qvr_name(term.group), _qvr_name(term.slope)): term
            for term in random_terms
        }
        random_terms = [
            by_term.pop(key) for key in original_terms if key in by_term
        ] + list(by_term.values())
        if self._thresholds_by is not None:
            random_terms = [
                term
                for term in random_terms
                if not (
                    term.group == self._thresholds_by
                    and term.slope == "Intercept"
                    and (_qvr_name(term.group), "Intercept") not in original_terms
                )
            ]
        return Formula(
            formula=complement.formula,
            response_name=complement.response_name,
            fixed_columns=tuple(fixed_columns),
            random_terms=tuple(random_terms),
            offsets=complement.offsets,
            modifiers=complement.modifiers,
            response_values=complement.response_values,
            response_layout=complement.response_layout,
            group_levels=complement.group_levels,
            group_indices=complement.group_indices,
        )

    def fixed_column_observations(self, formula: Formula) -> dict[str, torch.Tensor]:
        """Bind retained covariates, offsets, and response likelihood weights.

        Parameters
        ----------
        formula : Formula
            The formula whose fixed-effect columns are bound.

        Returns
        -------
        dict[str, torch.Tensor]
            Numeric predictor and offset columns as ``float32`` tensors,
            keyed by their QVR input names. Auxiliary columns use the prefix
            ``dpar_<parameter>_``. Response weights retain their stored
            ``float64`` precision under the ``formula_weights`` key.
        """
        obs: dict[str, torch.Tensor] = {}
        if (
            formula.response_layout.event_shape
            and self._family.trials_parameter is not None
        ):
            from quivers.formulas._response import count_totals

            obs["formula_trials"] = count_totals(formula)
        if formula.modifiers.weights is not None:
            obs["formula_weights"] = torch.as_tensor(
                formula.modifiers.weights.data.copy()
            )
        if formula.modifiers.decision is not None:
            obs["formula_decision"] = torch.as_tensor(
                formula.modifiers.decision.data.copy(), dtype=torch.bool
            )
        for role, column in formula.modifiers.columns():
            if role not in {"weights", "decision"}:
                obs[column.qvr_name] = torch.as_tensor(column.data.copy())
        for _, prefix, predictor in self._predictor_items(formula):
            for col in predictor.fixed_columns:
                if not col.is_intercept:
                    obs[prefix + col.qvr_name] = torch.as_tensor(
                        col.data.copy(), dtype=torch.float32
                    )
        if self._predictor_graph is not None:
            from quivers.formulas._predictor_graph import graph_input_bindings

            obs.update(
                graph_input_bindings(
                    self._predictor_graph,
                    {
                        name: column.data
                        for name, column in self._predictor_graph.inputs.items()
                    },
                )
            )
        for _, prefix, predictor in self._predictor_items(formula):
            for index, column in enumerate(predictor.offsets):
                obs[f"{prefix}formula_offset_{index}"] = torch.as_tensor(
                    column.data.copy(), dtype=torch.float32
                )
            for term in predictor.random_terms:
                if term.column is not None and not term.column.is_intercept:
                    obs[f"{prefix}random_{term.column.qvr_name}"] = torch.as_tensor(
                        term.column.data.copy(), dtype=torch.float32
                    )
        return obs

    @staticmethod
    def _ordinal_category_count(formula: Formula) -> int:
        return FormulaToQVRModule._category_count(formula, family="cumulative")

    @staticmethod
    def _component_labels(formula: Formula) -> tuple[str, ...]:
        """Read retained event components or scalar factor category labels.

        Parameters
        ----------
        formula : Formula
            Formula carrying the response coordinate layout.

        Returns
        -------
        tuple[str, ...]
            Authored vector labels, factor levels, or numeric category codes.
        """
        if formula.response_layout.component_names:
            return formula.response_layout.component_names
        if formula.response_layout.category_levels:
            return formula.response_layout.category_levels
        return tuple(
            str(index)
            for index in range(
                FormulaToQVRModule._category_count(formula, family="categorical")
            )
        )

    @staticmethod
    def _category_count(formula: Formula, *, family: str) -> int:
        if formula.response_layout.component_names:
            return len(formula.response_layout.component_names)
        values = np.asarray(formula.response_values, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{family} family requires a nonempty, finite response")
        if np.any(values < -(2**63)) or np.any(values >= 2**63):
            raise ValueError(
                f"{family} family requires integer category labels in the Int64 range"
            )
        integers = values.astype(np.int64)
        if not np.equal(values, integers).all():
            raise ValueError(f"{family} family requires integer category labels")
        if formula.response_layout.category_levels:
            count = len(formula.response_layout.category_levels)
            if np.any(integers < 0) or np.any(integers >= count):
                raise ValueError(
                    f"{family} factor response codes exceed the retained levels"
                )
            if count < 2:
                raise ValueError(f"{family} family requires at least two categories")
            return count
        levels = np.unique(integers)
        if levels[0] != 0 or levels[-1] != levels.size - 1:
            raise ValueError(
                f"{family} family requires contiguous category labels "
                f"0..K-1; got {levels.tolist()!r}"
            )
        if levels.size < 2:
            raise ValueError(f"{family} family requires at least two categories")
        return int(levels.size)

    @staticmethod
    def _append_finite_object(
        statements: list[Statement], name: str, cardinality: int
    ) -> None:
        """Create or verify a shared compiler-owned finite axis.

        Parameters
        ----------
        statements : list[Statement]
            Module declarations receiving the finite object if absent.
        name : str
            Shared axis identity.
        cardinality : int
            Exact extent required by the predictor metadata.

        Raises
        ------
        ValueError
            If a previous declaration gives this axis another extent.
        """
        existing = next(
            (
                statement
                for statement in statements
                if isinstance(statement, ObjectDecl) and name in statement.names
            ),
            None,
        )
        if existing is not None:
            if existing.init == TypeFromExpr(
                expr=DiscreteConstructor(constructor="FinSet", args=(str(cardinality),))
            ):
                return
            raise ValueError(
                f"incompatible dimensions for shared formula axis {name!r}"
            )
        statements.append(
            ObjectDecl(
                names=(name,),
                init=TypeFromExpr(
                    expr=DiscreteConstructor(
                        constructor="FinSet", args=(str(cardinality),)
                    )
                ),
            )
        )

    def _append_ordinal_cutpoints(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> str:
        """Emit identified thresholds and return their variable name.

        Cutpoints are represented by a zero origin followed by cumulative
        positive spacings and are centered before use. Centering separates the
        threshold location from the formula intercept. With ``thresholds_by``
        the log spacings receive hierarchical group-level deviations, so every
        group gets an ordered scale without introducing an unidentified random
        threshold shift alongside its random intercept.

        Parameters
        ----------
        formula : Formula
            Response categories, category-specific design and group coding.
        statements : list[Statement]
            Declarations receiving the threshold and profile axes.
        program_steps : list[ProgramStep]
            Steps receiving priors and threshold transformations.

        Returns
        -------
        str
            Shared or rowwise threshold binding; empty for a binary hurdle
            law without a positive-category boundary.

        Raises
        ------
        ValueError
            If a group is absent or a prior call is malformed.
        """
        n_categories = self._category_count(formula, family=self._family.name)
        spec = self._family.ordinal
        assert spec is not None
        if spec.kind == "hurdle_cumulative":
            n_categories -= 1
        n_cutpoints = n_categories - 1
        if n_cutpoints == 0:
            return ""

        ordered = spec.kind in {"cumulative", "hurdle_cumulative"}
        constrained_cs = ordered and (
            any(column.category_specific for column in formula.fixed_columns)
            or any(
                term.column is not None and term.column.category_specific
                for term in formula.random_terms
            )
        )
        cutpoint_index = "FormulaCutpoint"
        self._append_finite_object(statements, cutpoint_index, n_cutpoints)

        if n_cutpoints == 1:
            if self._thresholds_by is not None:
                program_steps.extend(
                    (
                        _let(
                            "cutpoint_matrix",
                            _factor(
                                (
                                    ("threshold_group", _qvr_name(self._thresholds_by)),
                                    ("cutpoint", cutpoint_index),
                                ),
                                body=LetExprLiteral(value=0.0),
                            ),
                        ),
                        _let(
                            "row_cutpoints",
                            _index(
                                "cutpoint_matrix",
                                _var(f"{_qvr_name(self._thresholds_by)}_idx"),
                            ),
                        ),
                    )
                )
                return "row_cutpoints"
            program_steps.append(
                _let(
                    "cutpoints",
                    _factor(
                        (("cutpoint", cutpoint_index),),
                        cases=((0, LetExprLiteral(value=0.0)),),
                    ),
                )
            )
            return "cutpoints"

        if spec.thresholds == "equidistant" and self._thresholds_by is None:
            family_name, args = _parse_prior_call(
                self._user_priors.get(
                    "cutpoint_spacing",
                    "LogNormal(0.0, 1.0)" if ordered else "Normal(0.0, 1.0)",
                )
            )
            if ordered:
                program_steps.append(
                    _auxiliary_draw(
                        "cutpoint_spacing",
                        self._user_priors.get(
                            "cutpoint_spacing", "LogNormal(0.0, 1.0)"
                        ),
                        lower=0.0,
                    )
                )
            else:
                program_steps.append(_draw("cutpoint_spacing", family_name, args))
            equidistant_spacing: LetExprNode = _var("cutpoint_spacing")
            if constrained_cs:
                self._append_ordinal_spacing_bounds(
                    formula, statements, program_steps, n_cutpoints - 1
                )
                equidistant_spacing = _add(
                    _call("max", _var("ordinal_spacing_bound")), equidistant_spacing
                )
            program_steps.append(
                _let(
                    "cutpoints",
                    _factor(
                        (("cutpoint", cutpoint_index),),
                        body=_mul(
                            _sub(
                                _var("cutpoint"),
                                LetExprLiteral(value=(n_cutpoints - 1) / 2),
                            ),
                            equidistant_spacing,
                        ),
                    ),
                )
            )
            return "cutpoints"

        if not ordered:
            if spec.thresholds == "sum_to_zero":
                index = cutpoint_index
                latent = "cutpoints_raw"
                n_free = n_cutpoints
            else:
                index = "FormulaThresholdContrast"
                latent = "cutpoint_offset"
                n_free = n_cutpoints - 1
                self._append_finite_object(statements, index, n_free)
            family_name, args = _parse_prior_call(
                self._user_priors.get(latent, "Normal(0.0, 1.0)")
            )
            program_steps.append(
                _draw(latent, family_name, args, index=TypeName(name=index))
            )
            if spec.thresholds == "sum_to_zero":
                raw = _var(latent)
            else:
                raw = _factor(
                    (("cutpoint", cutpoint_index),),
                    cases=(
                        (0, LetExprLiteral(value=0.0)),
                        *(
                            (
                                position,
                                _index(
                                    latent,
                                    LetExprLiteral(value=position - 1, integral=True),
                                ),
                            )
                            for position in range(1, n_cutpoints)
                        ),
                    ),
                )
            program_steps.append(_let("cutpoints_uncentered", raw))
            program_steps.append(
                _let(
                    "cutpoints",
                    _sub(
                        _var("cutpoints_uncentered"),
                        _call("mean", _var("cutpoints_uncentered")),
                    ),
                )
            )
            return "cutpoints"

        spacing_index = "FormulaSpacing"
        n_spacings = n_cutpoints - 1
        self._append_finite_object(statements, spacing_index, n_spacings)

        if self._thresholds_by is None:
            spacing_family, spacing_args = _parse_prior_call(
                self._user_priors.get("cutpoint_log_spacing", "Normal(0.0, 1.0)")
            )
            program_steps.append(
                _draw(
                    "cutpoint_log_spacing",
                    spacing_family,
                    spacing_args,
                    index=TypeName(name=spacing_index),
                )
            )
            spacing: LetExprNode = _call("exp", _var("cutpoint_log_spacing"))
            if constrained_cs:
                self._append_ordinal_spacing_bounds(
                    formula, statements, program_steps, n_spacings
                )
                spacing = _add(_var("ordinal_spacing_bound"), spacing)
            program_steps.append(
                _let(
                    "cutpoint_cumulative",
                    _call("cumsum", spacing),
                )
            )
            cases = ((0, LetExprLiteral(value=0.0)),) + tuple(
                (
                    cutpoint,
                    _index(
                        "cutpoint_cumulative",
                        LetExprLiteral(value=float(cutpoint - 1), integral=True),
                    ),
                )
                for cutpoint in range(1, n_cutpoints)
            )
            program_steps.append(
                _let(
                    "cutpoints_uncentered",
                    _factor((("cutpoint", cutpoint_index),), cases=cases),
                )
            )
            program_steps.append(
                _let(
                    "cutpoints",
                    _sub(
                        _var("cutpoints_uncentered"),
                        _call("mean", _var("cutpoints_uncentered")),
                    ),
                )
            )
            return "cutpoints"

        group = self._thresholds_by
        if group not in formula.group_levels:
            raise ValueError(
                f"thresholds_by={group!r} must name a grouping factor in "
                f"the formula; choices are {sorted(formula.group_levels)!r}"
            )
        qgroup = _qvr_name(group)
        n_groups = len(formula.group_levels[group])
        # Free shape coordinates can be narrower than the physical gap axis.
        n_shape = 1 if spec.thresholds == "equidistant" else n_spacings
        shape_index = spacing_index
        if n_shape != n_spacings:
            shape_index = "FormulaThresholdShape"
            self._append_finite_object(statements, shape_index, n_shape)
        cell_index = "FormulaThresholdCell"
        self._append_finite_object(statements, cell_index, n_groups * n_shape)

        mean_prior = _auxiliary_draw(
            "cutpoint_log_spacing_mean",
            self._user_priors.get("cutpoint_log_spacing_mean", "Normal(0.0, 1.0)"),
        )
        scale_prior = _auxiliary_draw(
            "cutpoint_log_spacing_scale",
            self._user_priors.get("cutpoint_log_spacing_scale", "HalfNormal(0.5)"),
            lower=0.0 if "cutpoint_log_spacing_scale" in self._user_priors else None,
        )
        program_steps.extend(
            (
                mean_prior.with_(index=TypeName(name=shape_index)),
                scale_prior.with_(index=TypeName(name=shape_index)),
            )
        )
        flat_index = _add(
            _mul(
                _var("threshold_group"),
                LetExprLiteral(value=float(n_shape), integral=True),
            ),
            _var("spacing"),
        )
        threshold_raw = (
            "cutpoint_log_spacing_value"
            if self._reparameterize == "centered"
            else "cutpoint_log_spacing_noise"
        )
        if self._reparameterize == "centered":
            for suffix in ("mean", "scale"):
                program_steps.append(
                    _let(
                        f"cutpoint_log_spacing_cell_{suffix}",
                        _factor(
                            (("threshold_cell", cell_index),),
                            body=_index(
                                f"cutpoint_log_spacing_{suffix}",
                                LetExprBinOp(
                                    op="%",
                                    left=_var("threshold_cell"),
                                    right=LetExprLiteral(value=n_shape, integral=True),
                                ),
                            ),
                        ),
                    )
                )
            program_steps.append(
                _draw(
                    threshold_raw,
                    "Normal",
                    (
                        "cutpoint_log_spacing_cell_mean",
                        "cutpoint_log_spacing_cell_scale",
                    ),
                    index=TypeName(name=cell_index),
                )
            )
        else:
            program_steps.append(
                _draw(
                    threshold_raw, "Normal", (0.0, 1.0), index=TypeName(name=cell_index)
                )
            )
        program_steps.append(
            _let(
                "cutpoint_noise_matrix",
                _factor(
                    (("threshold_group", qgroup), ("spacing", shape_index)),
                    body=_index(threshold_raw, flat_index),
                ),
            )
        )
        log_spacing = _add(
            _index("cutpoint_log_spacing_mean", _var("spacing")),
            _mul(
                _index("cutpoint_log_spacing_scale", _var("spacing")),
                _index(
                    "cutpoint_noise_matrix",
                    _var("threshold_group"),
                    _var("spacing"),
                ),
            ),
        )
        if self._reparameterize == "centered":
            log_spacing = _index(
                "cutpoint_noise_matrix", _var("threshold_group"), _var("spacing")
            )
        program_steps.append(
            _let(
                "cutpoint_log_spacing_matrix",
                _factor(
                    (("threshold_group", qgroup), ("spacing", shape_index)),
                    body=log_spacing,
                ),
            )
        )
        spacing_matrix: LetExprNode = _call("exp", _var("cutpoint_log_spacing_matrix"))
        if spec.thresholds == "equidistant":
            # The one gap must dominate all adjacent CS contrasts. A proper
            # hierarchical log-slack prior remains on its positive excess.
            gap: LetExprNode = _call(
                "exp",
                _index(
                    "cutpoint_log_spacing_matrix",
                    _var("threshold_group"),
                    LetExprLiteral(value=0, integral=True),
                ),
            )
            if constrained_cs:
                self._append_ordinal_spacing_bounds(
                    formula, statements, program_steps, n_spacings
                )
                program_steps.append(
                    _let(
                        "ordinal_group_spacing_bound",
                        _factor(
                            (("threshold_group", qgroup),),
                            body=_call(
                                "max",
                                _index(
                                    "ordinal_spacing_bound", _var("threshold_group")
                                ),
                            ),
                        ),
                    )
                )
                gap = _add(
                    _index("ordinal_group_spacing_bound", _var("threshold_group")),
                    gap,
                )
            spacing_matrix = _factor(
                (("threshold_group", qgroup), ("spacing", spacing_index)),
                body=gap,
            )
        elif constrained_cs:
            self._append_ordinal_spacing_bounds(
                formula, statements, program_steps, n_spacings
            )
            spacing_matrix = _add(_var("ordinal_spacing_bound"), spacing_matrix)
        program_steps.append(
            _let(
                "cutpoint_cumulative_matrix",
                _call("cumsum", spacing_matrix),
            )
        )
        program_steps.append(
            _let(
                "cutpoint_zero_by_group",
                _factor(
                    (("threshold_group", qgroup),),
                    body=LetExprLiteral(value=0.0),
                ),
            )
        )
        position_cases: tuple[tuple[int, LetExprNode], ...] = (
            (0, _var("cutpoint_zero_by_group")),
        ) + tuple(
            (
                cutpoint,
                _factor(
                    (("threshold_group", qgroup),),
                    body=_index(
                        "cutpoint_cumulative_matrix",
                        _var("threshold_group"),
                        LetExprLiteral(value=float(cutpoint - 1), integral=True),
                    ),
                ),
            )
            for cutpoint in range(1, n_cutpoints)
        )
        program_steps.append(
            _let(
                "cutpoints_by_position",
                _factor((("cutpoint", cutpoint_index),), cases=position_cases),
            )
        )
        program_steps.append(
            _let(
                "cutpoint_matrix_uncentered",
                _factor(
                    (("threshold_group", qgroup), ("cutpoint", cutpoint_index)),
                    body=_index(
                        "cutpoints_by_position",
                        _var("cutpoint"),
                        _var("threshold_group"),
                    ),
                ),
            )
        )
        centered_value = _sub(
            _index(
                "cutpoint_matrix_uncentered",
                _var("threshold_group"),
                _var("cutpoint"),
            ),
            _call(
                "mean",
                _index("cutpoint_matrix_uncentered", _var("threshold_group")),
            ),
        )
        program_steps.append(
            _let(
                "cutpoint_matrix",
                _factor(
                    (("threshold_group", qgroup), ("cutpoint", cutpoint_index)),
                    body=centered_value,
                ),
            )
        )
        program_steps.append(
            _let(
                "row_cutpoints",
                _index("cutpoint_matrix", _var(f"{qgroup}_idx")),
            )
        )
        return "row_cutpoints"

    def _threshold_location_formula(self, formula: Formula) -> Formula:
        """Reuse or introduce the one group location of a full threshold profile.

        Parameters
        ----------
        formula : Formula
            Formula whose threshold grouping factor has retained coding.

        Returns
        -------
        Formula
            Formula with exactly one ordinary intercept for the threshold group.

        Raises
        ------
        ValueError
            If the requested threshold group has no evaluated coding.
        """
        group = self._thresholds_by
        if group is None:
            return formula
        if group not in formula.group_levels:
            raise ValueError(
                f"thresholds_by={group!r} requires evaluated group coding; "
                "include the group in the formula or use the dataframe fitting adapter"
            )
        if any(
            term.group == group and term.slope == "Intercept"
            for term in formula.random_terms
        ):
            return formula
        return formula.with_(
            random_terms=(
                *formula.random_terms,
                RandomTerm(group=group, slope="Intercept"),
            )
        )

    def _append_ordinal_spacing_bounds(
        self,
        formula: Formula,
        statements: list[Statement],
        steps: list[ProgramStep],
        n_spacings: int,
    ) -> None:
        """Bound adjacent linear category effects on the retained design.

        Parameters
        ----------
        formula : Formula
            Evaluated design whose rows define the fitting domain.
        statements : list[Statement]
            Module declarations receiving the spacing axis when necessary.
        steps : list[ProgramStep]
            Program steps receiving one linear-time reduction per spacing.
        n_spacings : int
            Number of gaps between adjacent cumulative thresholds.

        Notes
        -----
        The emitted bound is ``max(0, max_i(v[i,j+1]-v[i,j]))``.
        Adding a positive slack gives every strictly feasible threshold gap
        for the fixed linear effects. A prior on the log slack is a proper
        conditional prior, distinct from independently specified threshold
        priors conditioned on the feasible domain.
        """
        axis = "FormulaSpacing"
        if not any(
            isinstance(statement, ObjectDecl) and statement.names == (axis,)
            for statement in statements
        ):
            self._append_finite_object(statements, axis, n_spacings)
        gap = _var("constraint_spacing")

        def difference(row: LetExprNode) -> LetExprNode:
            """Form adjacent transition-effect differences.

            Parameters
            ----------
            row : LetExprNode
                Original fitting row index.

            Returns
            -------
            LetExprNode
                Adjacent effect difference at the bound spacing position.
            """
            return _sub(
                _index(
                    "eta_nonbaseline",
                    row,
                    _add(gap, LetExprLiteral(value=1, integral=True)),
                ),
                _index("eta_nonbaseline", row, gap),
            )

        def reduce_rows(row_axis: str, row: LetExprNode) -> LetExprNode:
            """Reduce a nonempty row partition to nonnegative gap bounds.

            Parameters
            ----------
            row_axis : str
                Finite partition axis.
            row : LetExprNode
                Original row selected at each partition position.

            Returns
            -------
            LetExprNode
                One maximum lower bound for each threshold gap.
            """
            return _factor(
                (("constraint_spacing", axis),),
                body=_call(
                    "max",
                    LetExprLiteral(value=0.0),
                    _call(
                        "max",
                        _factor((("constraint_row", row_axis),), body=difference(row)),
                    ),
                ),
            )

        if self._thresholds_by is not None:
            group = self._thresholds_by
            cases: list[tuple[int, LetExprNode]] = []
            for code in range(len(formula.group_levels[group])):
                positions = tuple(
                    i
                    for i, value in enumerate(formula.group_indices[group])
                    if value == code
                )
                if not positions:
                    bound = _factor(
                        (("constraint_spacing", axis),), body=LetExprLiteral(value=0.0)
                    )
                else:
                    row_axis = f"FormulaConstraintRows_{code}"
                    rows_name = f"ordinal_constraint_rows_{code}"
                    self._append_finite_object(statements, row_axis, len(positions))
                    steps.append(
                        _let(
                            rows_name,
                            _factor(
                                (("constraint_member", row_axis),),
                                cases=tuple(
                                    (i, LetExprLiteral(value=position, integral=True))
                                    for i, position in enumerate(positions)
                                ),
                            ),
                        )
                    )
                    bound = reduce_rows(
                        row_axis, _index(rows_name, _var("constraint_row"))
                    )
                cases.append((code, bound))
            value = _factor(
                (("constraint_group", _qvr_name(group)),), cases=tuple(cases)
            )
        else:
            value = reduce_rows("Resp", _var("constraint_row"))
        steps.append(
            _let(
                "ordinal_spacing_bound",
                value,
            )
        )

    @staticmethod
    def _row_predictor(
        formula: Formula, name: str, row: LetExprNode, *, neural: bool = False
    ) -> LetExprNode:
        """Index predictors whose declared design includes a row axis.

        Parameters
        ----------
        formula : Formula
            Proportional predictor metadata.
        name : str
            Predictor binding.
        row : LetExprNode
            Row index.
        neural : bool
            Whether an external predictor supplies a row axis.

        Returns
        -------
        LetExprNode
            Row-indexed predictor or its shared scalar binding.
        """
        rowwise = (
            neural
            or bool(formula.offsets)
            or any(
                term.column is None or not term.column.category_specific
                for term in formula.random_terms
            )
            or any(
                not column.is_intercept and not column.category_specific
                for column in formula.fixed_columns
            )
        )
        return _index(name, row) if rowwise else _var(name)

    def _append_ordinal_contrasts(
        self, formula: Formula, cutpoints: str, program_steps: list[ProgramStep]
    ) -> None:
        """Materialize and validate each row/transition contrast once.

        Parameters
        ----------
        formula : Formula
            Proportional and category-specific predictor metadata.
        cutpoints : str
            Shared or rowwise threshold binding.
        program_steps : list[ProgramStep]
            Steps receiving checked discrimination, contrasts and cumulative
            row-order validation.
        """
        row, transition = _var("ordinal_row"), _var("ordinal_transition")
        eta = self._row_predictor(
            formula, "mu", row, neural=self._predictor_name is not None
        )
        category_specific = any(
            column.category_specific for column in formula.fixed_columns
        ) or any(
            term.column is not None and term.column.category_specific
            for term in formula.random_terms
        )
        if category_specific:
            eta = _add(eta, _index("eta_nonbaseline", row, transition))
        disc_formula = self._distributional.get("disc")
        program_steps.append(
            _let("ordinal_disc", _call("positive_parameter", _var("disc")))
        )
        disc = (
            self._row_predictor(disc_formula, "ordinal_disc", row)
            if disc_formula is not None
            else _var("ordinal_disc")
        )
        cuts = (
            _index(cutpoints, row, transition)
            if self._thresholds_by is not None
            else _index(cutpoints, transition)
        )
        program_steps.append(
            _let(
                "ordinal_contrasts",
                _factor(
                    (
                        ("ordinal_row", "Resp"),
                        ("ordinal_transition", "FormulaCutpoint"),
                    ),
                    body=_mul(disc, _sub(cuts, eta)),
                ),
            )
        )
        spec = self._family.ordinal
        if (
            category_specific
            and spec is not None
            and spec.kind in {"cumulative", "hurdle_cumulative"}
            and self._category_count(formula, family=self._family.name)
            - int(spec.kind == "hurdle_cumulative")
            > 2
        ):
            gap = _var("ordinal_spacing")
            differences = _factor(
                (("ordinal_row", "Resp"), ("ordinal_spacing", "FormulaSpacing")),
                body=_sub(
                    _index(
                        "ordinal_contrasts",
                        row,
                        _add(gap, LetExprLiteral(value=1, integral=True)),
                    ),
                    _index("ordinal_contrasts", row, gap),
                ),
            )
            program_steps.append(
                _let(
                    "ordinal_effective_spacing",
                    _call("positive_parameter", differences),
                )
            )

    @staticmethod
    def _append_ordinal_cdf(
        argument: LetExprNode, link: str, prefix: str, steps: list[ProgramStep]
    ) -> tuple[str, str]:
        """Bind a shared pair of stable log CDF/survival expressions.

        Parameters
        ----------
        argument : LetExprNode
            Finite ordinal contrasts.
        link : str
            Static inverse link.
        prefix : str
            Namespace of intermediate bindings.
        steps : list[ProgramStep]
            Steps receiving stable probability expressions.

        Returns
        -------
        tuple of str
            Log-CDF and log-survival binding names.

        Raises
        ------
        ValueError
            If the inverse link is unsupported.
        """

        def bind(suffix: str, expression: LetExprNode) -> LetExprVar:
            """Append an intermediate probability binding.

            Parameters
            ----------
            suffix : str
                Local binding label.
            expression : LetExprNode
                Expression to evaluate once.

            Returns
            -------
            LetExprVar
                Reference to the intermediate.
            """
            name = prefix + suffix
            steps.append(_let(name, expression))
            return _var(name)

        def choose(
            condition: LetExprNode, yes: LetExprNode, no: LetExprNode
        ) -> LetExprCall:
            """Select safe elementwise alternatives.

            Parameters
            ----------
            condition : LetExprNode
                Boolean predicate.
            yes : LetExprNode
                Value where the predicate holds.
            no : LetExprNode
                Value elsewhere.

            Returns
            -------
            LetExprCall
                Selection expression.
            """
            return _call("select", condition, yes, no)

        zero, one = LetExprLiteral(value=0.0), LetExprLiteral(value=1.0)
        z = bind("argument", argument)

        def minus(x: LetExprNode) -> LetExprNode:
            """Negate a real expression.

            Parameters
            ----------
            x : LetExprNode
                Expression to negate.

            Returns
            -------
            LetExprNode
                Its additive inverse.
            """
            return _mul(LetExprLiteral(value=-1.0), x)

        if link in {"logit", "probit_approx"}:
            if link == "probit_approx":
                z = bind(
                    "approximation",
                    _add(
                        _mul(
                            LetExprLiteral(value=0.07056),
                            _call("pow", z, LetExprLiteral(value=3.0)),
                        ),
                        _mul(LetExprLiteral(value=1.5976), z),
                    ),
                )
            cdf, sf = _call("logsigmoid", z), _call("logsigmoid", minus(z))
        elif link == "probit":
            cdf, sf = _call("log_ndtr", z), _call("log_ndtr", minus(z))
        elif link == "softit":
            positive = bind("positive", _call("softplus", z))
            small = LetExprBinOp(op="<", left=z, right=LetExprLiteral(value=-35.0))
            safe = choose(small, zero, z)
            log_positive = bind(
                "log_positive", choose(small, z, _call("log", _call("softplus", safe)))
            )
            denominator = bind("denominator", _call("log1p", positive))
            cdf, sf = _sub(log_positive, denominator), minus(denominator)
        elif link == "cauchit":
            magnitude = bind("magnitude", _call("abs", z))
            tail = LetExprBinOp(op=">", left=magnitude, right=one)
            safe = choose(tail, magnitude, one)
            small = bind(
                "small",
                _sub(
                    _call("log", _call("atan", _div(one, safe))),
                    LetExprLiteral(value=math.log(math.pi)),
                ),
            )
            large = bind("large", _call("log1mexp", small))
            negative = LetExprBinOp(op="<", left=z, right=zero)
            regular = _div(
                _call("atan", choose(tail, zero, z)), LetExprLiteral(value=math.pi)
            )
            cdf = choose(
                tail,
                choose(negative, small, large),
                _call("log_probability", _add(LetExprLiteral(value=0.5), regular)),
            )
            sf = choose(
                tail,
                choose(negative, large, small),
                _call("log_probability", _sub(LetExprLiteral(value=0.5), regular)),
            )
        elif link == "cloglog":
            small = LetExprBinOp(op="<", left=z, right=LetExprLiteral(value=-35.0))
            large = LetExprBinOp(op=">", left=z, right=LetExprLiteral(value=80.0))
            safe_cdf = choose(small, zero, choose(large, zero, z))
            regular = _call("log1mexp", minus(_call("exp", safe_cdf)))
            cdf = choose(small, z, choose(large, zero, regular))
            overflow = LetExprBinOp(op=">", left=z, right=LetExprLiteral(value=709.0))
            sf = choose(
                overflow,
                _call("log_probability", zero),
                minus(_call("exp", choose(overflow, zero, z))),
            )
        else:
            raise ValueError(f"unsupported ordinal link {link!r}")
        bind("cdf", cdf)
        bind("sf", sf)
        return prefix + "cdf", prefix + "sf"

    def _append_ordinal_logits(
        self,
        formula: Formula,
        cutpoints: str,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> None:
        """Lower ordinal log masses with shared transitions and one prefix scan.

        Parameters
        ----------
        formula : Formula
            Response and predictor metadata.
        cutpoints : str
            Shared or row-indexed threshold binding.
        statements : list[Statement]
            Declarations receiving the response-category axis.
        program_steps : list[ProgramStep]
            Steps receiving normalized category log masses.
        """
        spec = self._family.ordinal
        assert spec is not None
        categories = self._category_count(formula, family=self._family.name)
        hurdle = spec.kind == "hurdle_cumulative"
        base_categories = categories - int(hurdle)
        self._append_finite_object(statements, "FormulaOrdinalCategory", categories)
        row = _var("ordinal_row")
        cdf_name: str | None = None
        sf_name: str | None = None
        if base_categories > 1:
            self._append_ordinal_contrasts(formula, cutpoints, program_steps)
            argument = _var("ordinal_contrasts")
            if spec.kind in {"acat", "cratio"}:
                argument = _mul(LetExprLiteral(value=-1.0), argument)
            cdf_name, sf_name = self._append_ordinal_cdf(
                argument, spec.link, "ordinal_log_", program_steps
            )
            if spec.kind == "acat":
                program_steps.append(
                    _let(
                        "ordinal_prefix",
                        _call("cumsum", _sub(_var(cdf_name), _var(sf_name))),
                    )
                )
            elif spec.kind in {"sratio", "cratio"}:
                continue_name = sf_name if spec.kind == "sratio" else cdf_name
                program_steps.append(
                    _let("ordinal_prefix", _call("cumsum", _var(continue_name)))
                )

        def cell(name: str, position: int) -> LetExprNode:
            """Select one transition at the current response row.

            Parameters
            ----------
            name : str
                Transition binding.
            position : int
                Zero-based transition position.

            Returns
            -------
            LetExprNode
                Selected transition value.
            """
            return _index(name, row, LetExprLiteral(value=position, integral=True))

        def base_mass(category: int) -> LetExprNode:
            """Construct a category log mass before a hurdle component.

            Parameters
            ----------
            category : int
                Zero-based positive-law category.

            Returns
            -------
            LetExprNode
                Stable category log mass.
            """
            if base_categories == 1:
                return LetExprLiteral(value=0.0)
            assert cdf_name is not None and sf_name is not None
            if spec.kind == "acat":
                return (
                    LetExprLiteral(value=0.0)
                    if category == 0
                    else cell("ordinal_prefix", category - 1)
                )
            if spec.kind in {"sratio", "cratio"}:
                previous = (
                    LetExprLiteral(value=0.0)
                    if category == 0
                    else cell("ordinal_prefix", category - 1)
                )
                stop_name = cdf_name if spec.kind == "sratio" else sf_name
                return (
                    previous
                    if category == base_categories - 1
                    else _add(previous, cell(stop_name, category))
                )
            if category == 0:
                return cell(cdf_name, 0)
            if category == base_categories - 1:
                return cell(sf_name, category - 1)
            if spec.link == "logit":
                lower, upper = (
                    cell("ordinal_contrasts", category - 1),
                    cell("ordinal_contrasts", category),
                )
                return _add(
                    cell(cdf_name, category),
                    cell(sf_name, category - 1),
                    _call("log1mexp", _sub(lower, upper)),
                )
            mirror = LetExprBinOp(
                op=">",
                left=cell(cdf_name, category - 1),
                right=LetExprLiteral(value=-math.log(2)),
            )
            high = _call(
                "select", mirror, cell(sf_name, category - 1), cell(cdf_name, category)
            )
            low = _call(
                "select", mirror, cell(sf_name, category), cell(cdf_name, category - 1)
            )
            return _add(high, _call("log1mexp", _sub(low, high)))

        hu: LetExprNode | None = None
        if hurdle:
            hu_formula = self._distributional.get("hu")
            hu = (
                self._row_predictor(hu_formula, "hu", row)
                if hu_formula is not None
                else _var("hu")
            )
        cases: list[tuple[int, LetExprNode]] = []
        for category in range(categories):
            if hurdle:
                assert hu is not None
                value = (
                    _call("log_probability", hu)
                    if category == 0
                    else _add(
                        _call("log_probability", _sub(LetExprLiteral(value=1.0), hu)),
                        base_mass(category - 1),
                    )
                )
            else:
                value = base_mass(category)
            cases.append((category, _factor((("ordinal_row", "Resp"),), body=value)))
        program_steps.append(
            _let(
                "ordinal_logits_by_category",
                _factor(
                    (("ordinal_category", "FormulaOrdinalCategory"),),
                    cases=tuple(cases),
                ),
            )
        )
        program_steps.append(
            _let(
                "ordinal_logits",
                _factor(
                    (
                        ("ordinal_row", "Resp"),
                        ("ordinal_category", "FormulaOrdinalCategory"),
                    ),
                    body=_index(
                        "ordinal_logits_by_category", _var("ordinal_category"), row
                    ),
                ),
            )
        )

    def _append_categorical_predictor(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
        *,
        ordinal_transitions: bool = False,
        correlated: Mapping[tuple[str, str, str], LetExprNode] | None = None,
        parameter: str = "mu",
        component_count: int | None = None,
        reference: int | None = 0,
        prefix: str = "",
    ) -> None:
        """Emit one shared scalar-component predictor in retained label order.

        Parameters
        ----------
        formula : Formula
            Evaluated design columns and group identities.
        statements : list[Statement]
            Module objects receiving component and innovation axes.
        program_steps : list[ProgramStep]
            Priors, deterministic predictors, and group transformations.
        ordinal_transitions : bool
            Retain transition coordinates for the ordinal likelihood emitter.
        correlated : Mapping or None
            Coefficients already transformed by joint correlation blocks.
        parameter : str
            Location or coordinate-valued auxiliary parameter name.
        component_count : int or None
            Explicit coordinate extent, or the retained response category count.
        reference : int or None
            Zero-predictor component; None estimates every coordinate.
        prefix : str
            Native namespacing for auxiliary designs and coefficients.

        Raises
        ------
        ValueError
            If the response categories or retained component axis are invalid.
        """
        n_categories = component_count or self._category_count(
            formula, family=self._family.name
        )
        if (
            not ordinal_transitions
            and self._family.response is not None
            and parameter == "mu"
        ):
            labels = self._component_labels(formula)
            reference = self._family.response.reference_index(labels)
        coordinates = tuple(
            index for index in range(n_categories) if index != reference
        )
        if (
            ordinal_transitions
            and self._family.ordinal is not None
            and self._family.ordinal.kind == "hurdle_cumulative"
        ):
            n_categories -= 1
        n_logits = n_categories - int(reference is not None)
        category_index = f"{prefix}FormulaCategory"
        logit_index = (
            "FormulaCutpoint"
            if ordinal_transitions
            else f"{prefix}FormulaCategoryLogit"
        )
        if not ordinal_transitions:
            self._append_finite_object(statements, category_index, n_categories)
            self._append_finite_object(statements, logit_index, n_logits)

        cell_terms: list[LetExprNode] = []
        response_var = _var("response")
        logit_var = _var("category_logit")

        for col in formula.fixed_columns:
            beta_name = prefix + (
                "intercept" if col.is_intercept else f"beta_{col.qvr_name}"
            )
            override = self._user_priors.get(beta_name)
            prior_text = override if override is not None else self._fixed_prior
            family_name, args = _parse_prior_call(prior_text)
            if override is None and not col.is_intercept:
                args = _autoscaled_prior_args(family_name, args, col.data)
            program_steps.append(
                _draw(beta_name, family_name, args, index=TypeName(name=logit_index))
            )
            coefficient = _index(beta_name, logit_var)
            if col.is_intercept:
                cell_terms.append(coefficient)
            else:
                cell_terms.append(
                    _mul(coefficient, _index(prefix + col.qvr_name, response_var))
                )

        cell_terms.extend(
            _index(f"{prefix}formula_offset_{index}", response_var)
            for index, _ in enumerate(formula.offsets)
        )

        if (
            self._predictor_name is not None
            and not ordinal_transitions
            and parameter == "mu"
        ):
            predictor_index = _add(
                _mul(
                    response_var,
                    LetExprLiteral(value=float(n_logits), integral=True),
                ),
                logit_var,
            )
            cell_terms.append(_index(self._predictor_name, predictor_index))

        correlated = dict(correlated or {})
        blocks: dict[tuple[str, str], list[tuple[str, RandomTerm]]] = {}
        for term in formula.random_terms:
            if (
                term.correlation_id is not None
                and (prefix, term.group, term.slope) not in correlated
            ):
                identity = (
                    f"shared_{term.correlation_id}"
                    if term.shared
                    else term.correlation_id
                )
                blocks.setdefault((term.group, identity), []).append((prefix, term))
        for (group, identity), members in blocks.items():
            correlated.update(
                self._append_indexed_correlations(
                    formula,
                    group,
                    identity,
                    members,
                    statements,
                    program_steps,
                    coordinate_axis=logit_index,
                    coordinate_count=n_logits,
                )
            )

        for term in formula.random_terms:
            if (prefix, term.group, term.slope) in (correlated or {}):
                continue
            qgroup = _qvr_name(term.group)
            qslope = _qvr_name(term.slope)
            sigma_var = f"{prefix}sigma_{qgroup}_{qslope}"
            sigma_prior_text = self._user_priors.get(
                sigma_var, self._random_scale_prior
            )
            sf_family, sf_args = _parse_prior_call(sigma_prior_text)
            program_steps.append(
                _draw(sigma_var, sf_family, sf_args, index=TypeName(name=logit_index))
            )
            n_groups = len(formula.group_levels[term.group])
            cell_index = f"{prefix}FormulaRandomCell_{qgroup}_{qslope}"
            self._append_finite_object(statements, cell_index, n_groups * n_logits)
            latent = prefix + (
                f"alpha_{qgroup}"
                if term.slope == "Intercept"
                else f"beta_{qgroup}_{qslope}"
            )
            raw_site = (
                f"centered_{latent}"
                if self._reparameterize == "centered"
                else f"{prefix}z_{qgroup}_{qslope}"
            )
            if self._reparameterize == "centered":
                scale_rows = f"{raw_site}_scale"
                program_steps.append(
                    _let(
                        scale_rows,
                        _factor(
                            (("random_cell", cell_index),),
                            body=_index(
                                sigma_var,
                                LetExprBinOp(
                                    op="%",
                                    left=_var("random_cell"),
                                    right=LetExprLiteral(value=n_logits, integral=True),
                                ),
                            ),
                        ),
                    )
                )
                program_steps.append(
                    _draw(
                        raw_site,
                        "Normal",
                        (0.0, scale_rows),
                        index=TypeName(name=cell_index),
                    )
                )
            else:
                program_steps.append(
                    _draw(
                        raw_site, "Normal", (0.0, 1.0), index=TypeName(name=cell_index)
                    )
                )
            self._group_layouts.append(
                GroupBlockLayout(
                    group=term.group,
                    identity=f"independent_{prefix}{qgroup}_{qslope}",
                    coordinate_axis=logit_index,
                    parameterization=self._reparameterize,
                    members=(
                        GroupMemberLayout(
                            prefix=prefix,
                            slope=term.slope,
                            axis=logit_index,
                            width=n_logits,
                            offset=0,
                            coefficient=latent,
                            raw_site=raw_site,
                        ),
                    ),
                )
            )

        for term in formula.random_terms:
            group = term.group
            qgroup = _qvr_name(group)
            slope = term.slope
            qslope = _qvr_name(slope)
            sigma_var = f"{prefix}sigma_{qgroup}_{qslope}"
            z_var = f"{prefix}z_{qgroup}_{qslope}"
            latent = (
                f"{prefix}alpha_{qgroup}"
                if slope == "Intercept"
                else f"{prefix}beta_{qgroup}_{qslope}"
            )
            contribution_name = f"{z_var}_contribution"
            joint_value = (correlated or {}).get((prefix, group, slope))
            if joint_value is not None:
                if slope != "Intercept":
                    column = (
                        f"{prefix}random_{term.column.qvr_name}"
                        if term.column is not None
                        else prefix + qslope
                    )
                    joint_value = _mul(joint_value, _var(column))
                program_steps.append(
                    _let(
                        contribution_name,
                        _factor((("category_logit", logit_index),), body=joint_value),
                    )
                )
                cell_terms.append(_index(contribution_name, logit_var, response_var))
                continue
            column_name = (
                f"{prefix}random_{term.column.qvr_name}"
                if term.column is not None
                else prefix + qslope
            )
            raw_site = (
                f"centered_{latent}" if self._reparameterize == "centered" else z_var
            )
            z_matrix = f"{raw_site}_matrix"
            flat_index = _add(
                _mul(
                    _var("random_group"),
                    LetExprLiteral(value=float(n_logits), integral=True),
                ),
                logit_var,
            )
            program_steps.append(
                _let(
                    z_matrix,
                    _factor(
                        (("category_logit", logit_index), ("random_group", qgroup)),
                        body=_index(raw_site, flat_index),
                    ),
                )
            )
            contribution = _index(z_matrix, logit_var, _var("random_group"))
            if self._reparameterize == "noncentered":
                contribution = _mul(_index(sigma_var, logit_var), contribution)
            value = _factor(
                (("category_logit", logit_index), ("random_group", qgroup)),
                body=contribution,
            )
            program_steps.append(_let(latent, value))
            contribution = _index(latent, logit_var, _var(f"{qgroup}_idx"))
            if slope != "Intercept":
                contribution = _mul(contribution, _var(column_name))
            program_steps.append(
                _let(
                    contribution_name,
                    _factor((("category_logit", logit_index),), body=contribution),
                )
            )
            cell_terms.append(_index(contribution_name, logit_var, response_var))

        program_steps.append(
            _let(
                f"{prefix}eta_nonbaseline",
                _factor(
                    (("response", "Resp"), ("category_logit", logit_index)),
                    body=_add(*cell_terms),
                ),
            )
        )
        if ordinal_transitions:
            return
        self._append_component_response(
            program_steps,
            parameter=parameter,
            prefix=prefix,
            reference=reference,
            coordinates=coordinates,
            category_index=category_index,
        )

    def _append_component_response(
        self,
        program_steps,
        *,
        parameter,
        prefix,
        reference,
        coordinates,
        category_index,
    ):
        """Complete shared labelled predictors with the retained reference and family link.

        Parameters
        ----------
        program_steps : list
            Steps already binding the free coordinate predictor matrix.
        parameter, prefix : str
            Family output and its established namespace.
        reference : int or None
            Identified reference index, or all-coordinate parameterization.
        coordinates : tuple[int, ...]
            Free response coordinates in retained order.
        category_index : str
            Full response component axis.
        """
        response_var = _var("response")
        if (
            self._family.response is not None
            and self._family.response.kind == "logistic_normal"
            and parameter == "mu"
        ):
            program_steps.append(_let("eta", _var(f"{prefix}eta_nonbaseline")))
            program_steps.append(_let("mu", _var("eta")))
            return
        program_steps.append(
            _let(
                f"{prefix}categorical_zero_logit",
                _factor(
                    (("response", "Resp"),),
                    body=LetExprLiteral(value=0.0),
                ),
            )
        )
        category_cases: tuple[tuple[int, LetExprNode], ...] = (
            ()
            if reference is None
            else ((reference, _var(f"{prefix}categorical_zero_logit")),)
        ) + tuple(
            (
                category,
                _factor(
                    (("response", "Resp"),),
                    body=_index(
                        f"{prefix}eta_nonbaseline",
                        response_var,
                        LetExprLiteral(value=float(coordinate), integral=True),
                    ),
                ),
            )
            for coordinate, category in enumerate(coordinates)
        )
        program_steps.append(
            _let(
                f"{prefix}categorical_logits_by_category",
                _factor((("category", category_index),), cases=category_cases),
            )
        )
        program_steps.append(
            _let(
                f"{prefix}eta",
                _factor(
                    (("response", "Resp"), ("category", category_index)),
                    body=_index(
                        f"{prefix}categorical_logits_by_category",
                        _var("category"),
                        response_var,
                    ),
                ),
            )
        )
        link = (
            self._family.location_link
            if parameter == "mu"
            else next(
                aux.link for aux in self._family.aux_params if aux.name == parameter
            )
        )
        _append_link(program_steps, parameter, _var(f"{prefix}eta"), link.name)

    def _append_mixture_parameters(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> tuple[str, str, str]:
        """Emit identified weights, sum-zero offsets, and component scales."""
        component_index = "FormulaComponent"
        contrast_index = "FormulaComponentContrast"
        self._append_finite_object(
            statements, component_index, self._mixture_components
        )
        self._append_finite_object(
            statements, contrast_index, self._mixture_components - 1
        )
        prior_specs = (
            (
                "mixture_logit",
                "mixture_logit",
                "Normal(0.0, 1.0)",
                contrast_index,
            ),
            (
                "mixture_offset",
                "mixture_offset",
                "Normal(0.0, 5.0)",
                contrast_index,
            ),
            (
                "mixture_scale",
                "mixture_scale",
                "HalfCauchy(2.0)",
                component_index,
            ),
        )
        for prior_name, draw_name, default, index_name in prior_specs:
            family_name, args = _parse_prior_call(
                self._user_priors.get(prior_name, default)
            )
            program_steps.append(
                _draw(
                    draw_name,
                    family_name,
                    args,
                    index=TypeName(name=index_name),
                )
            )
        contrast = _helmert_contrast(self._mixture_components)

        def projected_cases(name: str) -> tuple[tuple[int, LetExprNode], ...]:
            cases: list[tuple[int, LetExprNode]] = []
            for component, row in enumerate(contrast):
                terms = tuple(
                    _mul(
                        LetExprLiteral(value=float(weight)),
                        _index(
                            name,
                            LetExprLiteral(value=float(column), integral=True),
                        ),
                    )
                    for column, weight in enumerate(row)
                    if weight != 0.0
                )
                cases.append((component, _add(*terms)))
            return tuple(cases)

        logit_cases = projected_cases("mixture_logit")
        offset_cases = projected_cases("mixture_offset")
        program_steps.extend(
            [
                _let(
                    "mixture_logits",
                    _factor((("component", component_index),), cases=logit_cases),
                ),
                _let(
                    "mixture_weights",
                    _call("softmax", _var("mixture_logits")),
                ),
                _let(
                    "mixture_offsets",
                    _factor((("component", component_index),), cases=offset_cases),
                ),
            ]
        )
        row_varying = (
            self._predictor_name is not None
            or any(not col.is_intercept for col in formula.fixed_columns)
            or bool(formula.random_terms)
        )
        row_eta = _index("eta", _var("response")) if row_varying else _var("eta")
        program_steps.append(
            _let(
                "mixture_locations",
                _factor(
                    (("response", "Resp"), ("component", component_index)),
                    body=_add(
                        row_eta,
                        _index("mixture_offsets", _var("component")),
                    ),
                ),
            )
        )
        return "mixture_weights", "mixture_locations", "mixture_scale"

    def _append_indexed_correlations(
        self,
        formula: Formula,
        group: str,
        identity: str,
        members: list[tuple[str, RandomTerm]],
        statements: list[Statement],
        steps: list[ProgramStep],
        *,
        coordinate_axis: str | None = None,
        coordinate_count: int | None = None,
        coordinate_axes: Mapping[str, tuple[str, int]] | None = None,
    ) -> dict[tuple[str, str, str], LetExprNode]:
        """Correlate scalar and indexed coefficients in one block.

        Parameters
        ----------
        formula : Formula
            Full location predictor and response category metadata.
        group : str
            Shared grouping factor.
        identity : str
            Prior correlation-block identity.
        coordinate_axis : str | None
            Shared vector axis, or transition axes only for category-specific terms.
        coordinate_count : int or None
            Explicit extent of the shared component axis.
        coordinate_axes : Mapping[str, tuple[str, int]] or None
            Predictor-prefix axes and extents for mixed scalar/vector blocks.
        members : list[tuple[str, RandomTerm]]
            Prefix and term pairs in fitted block order.
        statements : list[Statement]
            Module objects receiving finite coordinate axes.
        steps : list[ProgramStep]
            Priors and deterministic transformations to append.

        Returns
        -------
        dict[tuple[str, str, str], LetExprNode]
            Row gathers of scalar or transition-indexed group coefficients.

        Raises
        ------
        ValueError
            If the correlation prior is not an LKJ Cholesky distribution.
        """
        transitions = (
            coordinate_count
            if coordinate_count is not None
            else (
                self._category_count(formula, family=self._family.name) - 1
                if coordinate_axis is not None
                or any(
                    term.column is not None and term.column.category_specific
                    for _, term in members
                )
                else 1
            )
        )
        if (
            coordinate_axis is None
            and self._family.ordinal is not None
            and self._family.ordinal.kind == "hurdle_cumulative"
        ):
            transitions -= 1
        axes = [
            (coordinate_axes or {}).get(prefix)
            or (
                (coordinate_axis, transitions)
                if coordinate_axis is not None
                else ("FormulaCutpoint", transitions)
                if term.column is not None and term.column.category_specific
                else None
            )
            for prefix, term in members
        ]
        widths = [axis[1] if axis is not None else 1 for axis in axes]
        dimension_size = sum(widths)
        if dimension_size < 2:
            return {}
        qgroup = _qvr_name(group)
        correlation = f"cor_{_qvr_name(identity)}"
        dimension = f"FormulaCorrelation_{_qvr_name(identity)}"
        self._append_finite_object(statements, dimension, dimension_size)
        family, arguments = _parse_prior_call(
            self._user_priors.get(correlation, self._correlation_prior)
        )
        if family != "LKJCholesky":
            raise ValueError("correlation priors must use LKJCholesky")
        steps.append(
            _draw(correlation, family, arguments, index=TypeName(name=dimension))
        )
        joint_raw = f"group_coefficients_{_qvr_name(identity)}"
        member_layouts = []
        offset = 0
        for (prefix, term), width, axis in zip(members, widths, axes, strict=True):
            coefficient = prefix + (
                f"alpha_{qgroup}"
                if term.slope == "Intercept"
                else f"beta_{qgroup}_{_qvr_name(term.slope)}"
            )
            member_layouts.append(
                GroupMemberLayout(
                    prefix=prefix,
                    slope=term.slope,
                    axis=axis[0] if axis is not None else None,
                    width=width,
                    offset=offset,
                    coefficient=coefficient,
                    raw_site=joint_raw
                    if self._reparameterize == "centered"
                    else f"{prefix}z_{qgroup}_{_qvr_name(term.slope)}",
                )
            )
            offset += width
        layout = GroupBlockLayout(
            group=group,
            identity=identity,
            coordinate_axis=dimension,
            parameterization=self._reparameterize,
            members=tuple(member_layouts),
        )
        self._group_layouts.append(layout)
        scales: list[str] = []
        innovation_cases: list[tuple[int, LetExprNode]] = []
        coordinate = 0
        for (prefix, term), width, axis in zip(members, widths, axes, strict=True):
            qslope = _qvr_name(term.slope)
            sigma = f"{prefix}sigma_{qgroup}_{qslope}"
            innovation = f"{prefix}z_{qgroup}_{qslope}"
            is_transition = axis is not None
            prior, args = _parse_prior_call(
                self._user_priors.get(sigma, self._random_scale_prior)
            )
            steps.append(
                _draw(
                    sigma,
                    prior,
                    args,
                    index=TypeName(name=axis[0]) if axis is not None else None,
                )
            )
            if self._reparameterize == "centered":
                scales.append(sigma)
                continue
            plate_index = qgroup
            if is_transition:
                plate_index = f"{prefix}FormulaRandomCell_{qgroup}_{qslope}"
                self._append_finite_object(
                    statements, plate_index, len(formula.group_levels[group]) * width
                )
            if self._reparameterize == "noncentered":
                steps.append(
                    _draw(
                        innovation,
                        "Normal",
                        (0.0, 1.0),
                        index=TypeName(name=plate_index),
                    )
                )
            for component in range(width):
                coordinate_index: LetExprNode = _var("random_group")
                if is_transition:
                    coordinate_index = _add(
                        _mul(
                            coordinate_index, LetExprLiteral(value=width, integral=True)
                        ),
                        LetExprLiteral(value=component, integral=True),
                    )
                innovation_cases.append(
                    (
                        coordinate,
                        _factor(
                            (("random_group", qgroup),),
                            body=_index(innovation, coordinate_index),
                        ),
                    )
                )
                coordinate += 1
            scales.append(sigma)
        # Form the block once, multiply once, then gather each member's
        # coordinates. The source grows linearly in the joint dimension;
        # it does not duplicate a triangular sum for every coefficient.
        innovation_matrix = f"__formula_group_{_qvr_name(identity)}_innovations"
        transformed_matrix = f"__formula_group_{_qvr_name(identity)}_transformed"
        if self._reparameterize == "centered":
            scale_cases = []
            for member, sigma in zip(layout.members, scales, strict=True):
                for component in range(member.width):
                    scale_cases.append(
                        (
                            member.offset + component,
                            _index(
                                sigma, LetExprLiteral(value=component, integral=True)
                            )
                            if member.axis is not None
                            else _var(sigma),
                        )
                    )
            scale_vector = f"__formula_group_{_qvr_name(identity)}_scales"
            factor = f"__formula_group_{_qvr_name(identity)}_factor"
            zero = f"__formula_group_{_qvr_name(identity)}_zero"
            steps.extend(
                (
                    _let(
                        scale_vector,
                        _factor(
                            (("group_coordinate", dimension),), cases=tuple(scale_cases)
                        ),
                    ),
                    _let(
                        zero,
                        _factor(
                            (("group_coordinate", dimension),),
                            body=LetExprLiteral(value=0.0),
                        ),
                    ),
                    _let(
                        factor,
                        _factor(
                            (("group_row", dimension), ("group_column", dimension)),
                            body=_mul(
                                _index(scale_vector, _var("group_row")),
                                _index(
                                    correlation, _var("group_row"), _var("group_column")
                                ),
                            ),
                        ),
                    ),
                )
            )
            draw = _draw(
                joint_raw,
                "MultivariateNormal",
                (zero, factor),
                index=TypeName(name=qgroup),
            )
            steps.append(
                draw.with_(
                    axes=AxisSpec(over=(dimension,), iid_over=(qgroup,)),
                    options=(
                        OptionEntry(key="over", value=OptionName(value=dimension)),
                        OptionEntry(key="iid_over", value=OptionName(value=qgroup)),
                    ),
                )
            )
        else:
            steps.append(
                _let(
                    innovation_matrix,
                    _factor(
                        (("group_coordinate", dimension),),
                        cases=tuple(innovation_cases),
                    ),
                )
            )
            steps.append(
                _let(
                    transformed_matrix,
                    _call("matmul", _var(correlation), _var(innovation_matrix)),
                )
            )
        result = {}
        offset = 0
        for position, ((prefix, term), width, axis) in enumerate(
            zip(members, widths, axes, strict=True)
        ):
            is_transition = axis is not None
            latent = prefix + (
                f"alpha_{qgroup}"
                if term.slope == "Intercept"
                else f"beta_{qgroup}_{_qvr_name(term.slope)}"
            )
            member_coordinate: LetExprNode = LetExprLiteral(value=offset, integral=True)
            scale: LetExprNode = _var(scales[position])
            binders = (("random_group", qgroup),)
            if is_transition:
                assert axis is not None
                member_coordinate = _add(member_coordinate, _var("category_logit"))
                scale = _index(scales[position], _var("category_logit"))
                binders = (
                    ("category_logit", axis[0]),
                    *binders,
                )
            coefficient_value = (
                _index(joint_raw, _var("random_group"), member_coordinate)
                if self._reparameterize == "centered"
                else _mul(
                    scale,
                    _index(transformed_matrix, member_coordinate, _var("random_group")),
                )
            )
            value = _factor(binders, body=coefficient_value)
            steps.append(_let(latent, value))
            result[(prefix, group, term.slope)] = (
                _index(latent, _var("category_logit"), _var(f"{qgroup}_idx"))
                if is_transition
                else _index(latent, _var(f"{qgroup}_idx"))
            )
            offset += width
        return result

    def _append_correlations(
        self,
        formula: Formula,
        statements: list[Statement],
        steps: list[ProgramStep],
        *,
        coordinate_axes: Mapping[str, tuple[str, int]] | None = None,
    ) -> dict[tuple[str, str, str], LetExprNode]:
        """Emit LKJ group blocks and their row-wise transformed coefficients.

        Parameters
        ----------
        formula : Formula
            Location predictor, combined with configured auxiliary predictors.
        statements : list[Statement]
            Module statements receiving correlation-dimension objects.
        steps : list[ProgramStep]
            Program steps receiving priors and group transformations.
        coordinate_axes : Mapping[str, tuple[str, int]] or None
            Predictor-prefix axes and extents; absent prefixes are scalar.

        Returns
        -------
        dict[tuple[str, str, str], LetExprNode]
            Coefficients keyed by predictor prefix, group, and slope.

        Raises
        ------
        ValueError
            If a correlation prior uses a different distribution family.
        """
        blocks: dict[tuple[str, str], list[tuple[str, RandomTerm]]] = {}
        for _, prefix, predictor in self._predictor_items(formula):
            for term in predictor.random_terms:
                if term.correlation_id is None:
                    continue
                identity = (
                    f"shared_{term.correlation_id}"
                    if term.shared
                    else prefix + term.correlation_id
                )
                blocks.setdefault((term.group, identity), []).append((prefix, term))
        result: dict[tuple[str, str, str], LetExprNode] = {}
        for (group, identity), members in blocks.items():
            result.update(
                self._append_indexed_correlations(
                    formula,
                    group,
                    identity,
                    members,
                    statements,
                    steps,
                    coordinate_axes=coordinate_axes,
                )
            )
        return result

    def _append_predictor(
        self,
        formula: Formula,
        program_steps: list[ProgramStep],
        *,
        prefix: str = "",
        predictor_name: str | None = None,
        correlated: Mapping[tuple[str, str, str], LetExprNode] | None = None,
    ) -> LetExprNode:
        """Emit one independently named predictor, sharing only grouping axes.

        Parameters
        ----------
        formula : Formula
            Evaluated design columns and group coding.
        program_steps : list[ProgramStep]
            Steps to append coefficient draws and row contributions to.
        prefix : str
            Namespace for this predictor's coefficients and covariates.
        predictor_name : str or None
            Optional host-computed contribution on the predictor scale.
        correlated : Mapping[tuple[str, str, str], LetExprNode] or None
            Row-wise coefficients keyed by predictor prefix, group, and slope.

        Returns
        -------
        LetExprNode
            Sum of the predictor contributions, or scalar zero.
        """
        linear_terms: list[LetExprNode] = [
            _var(f"{prefix}formula_offset_{index}")
            for index, _ in enumerate(formula.offsets)
        ]

        # Fixed effects: one scalar latent per design-matrix column.
        for col in formula.fixed_columns:
            beta_name = prefix + (
                "intercept" if col.is_intercept else f"beta_{col.qvr_name}"
            )
            override = self._user_priors.get(beta_name)
            prior_text = override if override is not None else self._fixed_prior
            family_name, args = _parse_prior_call(prior_text)
            # A default prior is autoscaled to the column it multiplies,
            # so its nominal scale reads in contribution space and means
            # the same thing on a raw predictor and on an orthonormal
            # `poly` column. An explicit prior is the user's statement
            # about that coefficient and is emitted as written.
            if override is None and not col.is_intercept:
                args = _autoscaled_prior_args(family_name, args, col.data)
            program_steps.append(_draw(beta_name, family_name, args))
            if col.is_intercept:
                linear_terms.append(_var(beta_name))
            else:
                contrib_name = f"{beta_name}_per_row"
                program_steps.append(
                    _let(
                        contrib_name, _mul(_var(beta_name), _var(prefix + col.qvr_name))
                    )
                )
                linear_terms.append(_var(contrib_name))

        if predictor_name is not None:
            # A host-attached differentiable predictor contributes on the
            # linear-predictor scale. The free name is supplied by ``fit`` at
            # every optimization step, so gradients remain connected to the
            # external module rather than being frozen into the data frame.
            linear_terms.append(_var(predictor_name))

        # Random effects.  Each (slope | g) entry emits its own scale
        # latent + per-level plate draw + per-row gather; multiple
        # slopes per group are independent (uncorrelated; matches
        # lme4's `(... || g)` semantics).
        #
        # Reparameterisation:
        # * Centered ("centered"): ``alpha_g : g <- Normal(0, sigma_g)``
        #   directly. Its posterior geometry can couple group coefficients
        #   strongly to their scales.
        # * Non-centered ("noncentered", default): emit
        #     z_g_<slope> : g <- Normal(0, 1)
        #     let alpha_g = sigma_g * z_g_<slope>
        #   The per-level draws sit in unit space; ``sigma_g`` scales
        #   the (deterministic) ``alpha_g`` afterwards. Mathematically
        #   identical conditional Gaussian law, with different inference
        #   coordinates. The default remains noncentered.
        for term in formula.random_terms:
            group = term.group
            qgroup = _qvr_name(group)
            slope = term.slope
            qslope = _qvr_name(slope)
            correlated_value = (correlated or {}).get((prefix, group, slope))
            if correlated_value is not None:
                latent_name = (
                    f"{prefix}alpha_{qgroup}"
                    if slope == "Intercept"
                    else f"{prefix}beta_{qgroup}_{qslope}"
                )
                if slope != "Intercept":
                    column_name = (
                        f"random_{term.column.qvr_name}"
                        if term.column is not None
                        else qslope
                    )
                    correlated_value = _mul(
                        correlated_value, _var(prefix + column_name)
                    )
                contribution = f"{latent_name}_per_row"
                program_steps.append(_let(contribution, correlated_value))
                linear_terms.append(_var(contribution))
                continue
            sigma_var = f"{prefix}sigma_{qgroup}_{qslope}"
            sigma_prior_text = self._user_priors.get(
                sigma_var, self._random_scale_prior
            )
            sf_family, sf_args = _parse_prior_call(sigma_prior_text)
            program_steps.append(_draw(sigma_var, sf_family, sf_args))
            if slope == "Intercept":
                latent_var = f"{prefix}alpha_{qgroup}"
            else:
                latent_var = f"{prefix}beta_{qgroup}_{qslope}"
            contrib_name = f"{latent_var}_per_row"
            idx_name = f"{qgroup}_idx"
            if self._reparameterize == "centered":
                program_steps.append(
                    _draw(
                        latent_var,
                        "Normal",
                        (0.0, sigma_var),
                        index=TypeName(name=qgroup),
                    )
                )
                per_level_expr: LetExprNode = LetExprIndex(
                    array=_var(latent_var), indices=(_var(idx_name),)
                )
            else:
                # Materialize group coefficients once per group before the
                # per-response gather. This also retains the native bindings
                # used by posterior coefficient inspection.
                z_var = f"{prefix}z_{qgroup}_{qslope}"
                program_steps.append(
                    _draw(
                        z_var,
                        "Normal",
                        (0.0, 1.0),
                        index=TypeName(name=qgroup),
                    )
                )
                program_steps.append(
                    _let(latent_var, _mul(_var(sigma_var), _var(z_var)))
                )
                per_level_expr = _index(latent_var, _var(idx_name))
            self._group_layouts.append(
                GroupBlockLayout(
                    group=group,
                    identity=f"independent_{prefix}{qgroup}_{qslope}",
                    coordinate_axis=None,
                    parameterization=self._reparameterize,
                    members=(
                        GroupMemberLayout(
                            prefix=prefix,
                            slope=slope,
                            axis=None,
                            width=1,
                            offset=0,
                            coefficient=latent_var,
                            raw_site=latent_var
                            if self._reparameterize == "centered"
                            else f"{prefix}z_{qgroup}_{qslope}",
                        ),
                    ),
                )
            )
            if slope == "Intercept":
                program_steps.append(_let(contrib_name, per_level_expr))
            else:
                program_steps.append(
                    _let(
                        contrib_name,
                        _mul(
                            per_level_expr,
                            _var(
                                prefix
                                + (
                                    f"random_{term.column.qvr_name}"
                                    if term.column is not None
                                    else qslope
                                )
                            ),
                        ),
                    )
                )
            linear_terms.append(_var(contrib_name))

        return _add(*linear_terms) if linear_terms else LetExprLiteral(value=0.0)

    def _runtime_parameters(self, formula: Formula) -> tuple[QiecValueParameter, ...]:
        """Declare checked group-index and response-weight input tensors.

        Parameters
        ----------
        formula : Formula
            Primary predictor, combined with configured auxiliary predictors.

        Returns
        -------
        tuple[QiecValueParameter, ...]
            Checked integer group indices and an optional Real row-weight tensor.
        """
        groups = dict.fromkeys(
            group
            for _, _, predictor in self._predictor_items(formula)
            for group in predictor.group_indices
        )
        index_type = QiecTypeApplication(
            constructor="Tensor",
            static_arguments=(QiecTypeName(name="Int"),),
            indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
        )
        parameters = tuple(
            QiecValueParameter(name=f"{_qvr_name(group)}_idx", type_expr=index_type)
            for group in groups
        )
        if formula.modifiers.decision is not None:
            decision_type = index_type.with_(
                static_arguments=(QiecTypeName(name="Bool"),)
            )
            parameters += (
                QiecValueParameter(name="formula_decision", type_expr=decision_type),
            )
            response_type = QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(QiecTypeName(name="Real"),),
                indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
            )
            parameters += (
                QiecValueParameter(name="formula_response", type_expr=response_type),
            )
        if formula.response_layout.event_shape:
            response_type = QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(
                    QiecTypeName(
                        name="Int"
                        if self._family.response is not None
                        and self._family.response.kind
                        in {"multinomial", "dirichlet_multinomial"}
                        else "Real"
                    ),
                ),
                indices=(
                    QiecShapeIndex(
                        dimensions=(
                            QiecIndexName(name="Resp"),
                            QiecIndexName(name="FormulaCategory"),
                        )
                    ),
                ),
            )
            parameters += (
                QiecValueParameter(
                    name=_qvr_name(formula.response_name), type_expr=response_type
                ),
            )
            if self._family.trials_parameter is not None:
                parameters += (
                    QiecValueParameter(name="formula_trials", type_expr=index_type),
                )
        if formula.modifiers.weights is not None:
            weight_type = QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(QiecTypeName(name="Real"),),
                indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
            )
            parameters += (
                QiecValueParameter(name="formula_weights", type_expr=weight_type),
            )
        if self._predictor_graph is not None:
            for column in self._predictor_graph.inputs.values():
                carrier = (
                    "Bool"
                    if column.data.dtype.kind == "b"
                    else "Int"
                    if column.data.dtype.kind in "iu"
                    else "Real"
                )
                parameters += (
                    QiecValueParameter(
                        name=column.qvr_name,
                        type_expr=index_type.with_(
                            static_arguments=(QiecTypeName(name=carrier),)
                        ),
                    ),
                )
        existing = {parameter.name for parameter in parameters}
        for role, column in formula.modifiers.columns():
            if column.qvr_name in existing:
                continue
            carrier = "Int" if role in {"censoring", "trials"} else "Real"
            role_type = index_type.with_(static_arguments=(QiecTypeName(name=carrier),))
            parameters += (
                QiecValueParameter(name=column.qvr_name, type_expr=role_type),
            )
            existing.add(column.qvr_name)
        return parameters

    def _append_vector_observation(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> None:
        """Lower one named response event using shared component predictors.

        Parameters
        ----------
        formula : Formula
            Vector response layout and evaluated predictors.
        statements : list[Statement]
            Module objects receiving covariance coordinates.
        program_steps : list[ProgramStep]
            Shared priors, predictors, and final response factor.

        Raises
        ------
        ValueError
            If a residual-correlation prior is not an LKJ Cholesky law.
        """
        assert self._family.response is not None
        kind = self._family.response.kind
        if kind == "multinomial":
            args = ("formula_trials", "eta")
        elif kind in {"dirichlet", "dirichlet_multinomial"}:
            phi = (
                _index("phi", _var("response"))
                if "phi" in self._distributional
                or self._predictor_graph is not None
                and "phi" in self._predictor_graph.roots
                else _var("phi")
            )
            concentration = _factor(
                (("response", "Resp"), ("component", "FormulaCategory")),
                body=_mul(_index("mu", _var("response"), _var("component")), phi),
            )
            program_steps.append(_let("formula_concentration", concentration))
            args = (
                ("formula_trials", "formula_concentration")
                if kind == "dirichlet_multinomial"
                else ("formula_concentration",)
            )
        elif kind == "dirichlet2":
            args = ("mu",)
        else:
            width = len(self._component_labels(formula)) - 1
            if width == 1:
                program_steps.append(
                    _let(
                        "formula_alr_correlation",
                        _factor(
                            (
                                ("i", "FormulaCategoryLogit"),
                                ("j", "FormulaCategoryLogit"),
                            ),
                            body=LetExprLiteral(value=1.0),
                        ),
                    )
                )
            else:
                prior, args_prior = _parse_prior_call(
                    self._user_priors.get("rescor", self._correlation_prior)
                )
                if prior != "LKJCholesky":
                    raise ValueError(
                        "ALR residual correlations require an LKJCholesky prior"
                    )
                program_steps.append(
                    _draw(
                        "rescor",
                        prior,
                        args_prior,
                        index=TypeName(name="FormulaCategoryLogit"),
                    )
                )
                program_steps.append(_let("formula_alr_correlation", _var("rescor")))
            sigma = (
                _index("sigma", _var("response"), _var("i"))
                if "sigma" in self._distributional
                or self._predictor_graph is not None
                and isinstance(
                    self._predictor_graph.roots.get("sigma"), ComponentExpression
                )
                else _index("sigma", _var("response"))
                if self._predictor_graph is not None
                and "sigma" in self._predictor_graph.roots
                else _var("sigma")
                if "sigma" in self._fixed_parameters
                else _index("sigma", _var("i"))
            )
            program_steps.append(
                _let(
                    "formula_alr_scale",
                    _factor(
                        (
                            ("response", "Resp"),
                            ("i", "FormulaCategoryLogit"),
                            ("j", "FormulaCategoryLogit"),
                        ),
                        body=_mul(
                            sigma,
                            _index("formula_alr_correlation", _var("i"), _var("j")),
                        ),
                    ),
                )
            )
            reference = self._family.response.reference_index(
                self._component_labels(formula)
            )
            assert reference is not None
            args = ("mu", "formula_alr_scale", reference)
        observation = _draw(
            _qvr_name(formula.response_name),
            self._family.observe_family,
            args,
            index=TypeName(name="Resp"),
            mode="score",
            weights="formula_weights"
            if formula.modifiers.weights is not None
            else None,
        )
        if kind == "multinomial":
            observation = observation.with_(
                args=(
                    DrawArgNamed(
                        parameter="total_count", value=_to_draw_arg("formula_trials")
                    ),
                    DrawArgNamed(parameter="logits", value=_to_draw_arg("eta")),
                )
            )
        observation = observation.with_(
            options=(
                *observation.options,
                OptionEntry(key="over", value=OptionName(value="FormulaCategory")),
            )
        )
        program_steps.append(observation)

    def _predictor_items(self, formula: Formula):
        """Enumerate additive designs once across ordinary and nonlinear parameters.

        Parameters
        ----------
        formula : Formula
            Primary response and ordinary location metadata.

        Returns
        -------
        tuple
            Parameter labels, stable prefixes, and evaluated formulas.
        """
        return predictor_items(formula, self._distributional, self._predictor_graph)

    def _graph_expression_at(self, expression, row, component=None):
        """Project nominal graph inputs to one row and explicitly labelled coordinate.

        Parameters
        ----------
        expression : LetExprNode
            Canonical scalar expression at one result coordinate.
        row : LetExprNode
            Observation-row index.
        component : str or None
            Requested coordinate label, required for vector dependencies.

        Returns
        -------
        LetExprNode
            Capture-safe row expression over existing QVR bindings.

        Raises
        ------
        ValueError
            If a vector dependency has no compatible labelled coordinate.
        """
        graph = self._predictor_graph
        assert graph is not None
        substitution = {
            column.qvr_name: _index(column.qvr_name, row)
            for column in graph.inputs.values()
        }
        for name in (*graph.leaves, *graph.expressions):
            value = graph.expressions.get(name)
            if isinstance(value, ComponentExpression):
                if component is None:
                    # Only reject an actually used vector, not unrelated nodes.
                    from quivers.dsl.let_expr_traversal import free_let_names

                    if f"nlpar_{name}" in free_let_names(expression):
                        raise ValueError(
                            f"nonlinear vector {name!r} requires explicit component expressions"
                        )
                    continue
                if component not in value.components:
                    from quivers.dsl.let_expr_traversal import free_let_names

                    if f"nlpar_{name}" in free_let_names(expression):
                        raise ValueError(
                            f"nonlinear vector {name!r} has no component {component!r}"
                        )
                    continue
                substitution[f"nlpar_{name}"] = _index(
                    f"nlpar_{name}",
                    row,
                    LetExprLiteral(
                        value=value.components.index(component), integral=True
                    ),
                )
            else:
                substitution[f"nlpar_{name}"] = _index(f"nlpar_{name}", row)
        return substitute_let_expr(expression, substitution)

    def _append_graph_value(
        self, name, value, statements, steps, *, axis=None, components=None
    ):
        """Materialize a graph node with explicit row and optional nominal component axes.

        Parameters
        ----------
        name : str
            Existing QVR binding name.
        value : LetExprNode or ComponentExpression
            Scalar-row or explicitly labelled expression.
        statements, steps : list
            Module axes and deterministic program statements.
        axis : str or None
            Shared coordinate axis for a family root, otherwise node-local.
        components : tuple[str, ...] or None
            Required family coordinate labels in canonical retained order.

        Raises
        ------
        ValueError
            If result labels or scalar/vector carriers do not match the root.
        """
        row = _var("graph_row")
        if not isinstance(value, ComponentExpression):
            if components is not None:
                raise ValueError(
                    f"nonlinear {name!r} requires explicit component expressions for {components}"
                )
            steps.append(
                _let(
                    name,
                    _factor(
                        (("graph_row", "Resp"),),
                        body=self._graph_expression_at(value, row),
                    ),
                )
            )
            return
        labels = value.components if components is None else components
        if set(labels) != set(value.components):
            raise ValueError(
                f"nonlinear {name!r} component labels {value.components} do not match expected {labels}"
            )
        axis = axis or f"FormulaGraphCoordinate_{name}"
        self._append_finite_object(statements, axis, len(labels))
        by_coordinate = f"{name}_by_coordinate"
        cases = tuple(
            (
                index,
                _factor(
                    (("graph_row", "Resp"),),
                    body=self._graph_expression_at(
                        value.values[value.components.index(label)], row, label
                    ),
                ),
            )
            for index, label in enumerate(labels)
        )
        steps.append(
            _let(by_coordinate, _factor((("graph_coordinate", axis),), cases=cases))
        )
        steps.append(
            _let(
                name,
                _factor(
                    (("graph_row", "Resp"), ("graph_coordinate", axis)),
                    body=_index(by_coordinate, _var("graph_coordinate"), row),
                ),
            )
        )

    def _append_graph_parameters(self, formula, statements, steps, correlated):
        """Emit nonlinear leaves through existing coefficient and correlation paths.

        Parameters
        ----------
        formula : Formula
            Primary response metadata.
        statements, steps : list
            Existing axes and program statements.
        correlated : Mapping
            Shared group coefficients across all additive parameter leaves.
        """
        graph = self._predictor_graph
        if graph is None:
            return
        for name, leaf in graph.leaves.items():
            value = self._append_predictor(
                leaf, steps, prefix=f"nlpar_{name}_", correlated=correlated
            )
            raw = f"nlpar_{name}_raw"
            steps.append(_let(raw, value))
            steps.append(
                _let(
                    f"nlpar_{name}",
                    _factor(
                        (("graph_row", "Resp"),),
                        body=self._row_predictor(leaf, raw, _var("graph_row")),
                    ),
                )
            )
        for name, expression in graph.expressions.items():
            self._append_graph_value(f"nlpar_{name}", expression, statements, steps)

    def _append_auxiliaries(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
        correlated: Mapping[tuple[str, str, str], LetExprNode],
    ) -> None:
        """Lower shared scalar or coordinate-valued auxiliary parameters.

        Parameters
        ----------
        formula : Formula
            Response metadata and primary design.
        statements : list[Statement]
            Module objects receiving coordinate axes.
        program_steps : list[ProgramStep]
            Program steps receiving priors and predictors.
        correlated : Mapping[tuple[str,str,str],LetExprNode]
            Shared correlated group contributions.
        """
        for aux in self._family.aux_params:
            if (
                aux.name == "sigma"
                and formula.modifiers.standard_error is not None
                and not formula.modifiers.standard_error.sigma
            ):
                continue
            vector_sigma = (
                self._family.response is not None
                and self._family.response.kind == "logistic_normal"
                and aux.name == "sigma"
            )
            if aux.name in self._fixed_parameters:
                program_steps.append(
                    _let(
                        aux.name,
                        LetExprLiteral(value=float(self._fixed_parameters[aux.name])),
                    )
                )
                continue
            if (
                self._predictor_graph is not None
                and aux.name in self._predictor_graph.roots
            ):
                eta_name = f"dpar_{aux.name}_eta"
                value = self._predictor_graph.roots[aux.name]
                labels = None
                if isinstance(value, ComponentExpression):
                    if not vector_sigma:
                        raise ValueError(
                            f"family parameter {aux.name!r} requires a scalar nonlinear root"
                        )
                    assert self._family.response is not None
                    labels = tuple(
                        self._component_labels(formula)[index]
                        for index in self._family.response.coordinate_indices(
                            self._component_labels(formula)
                        )
                    )
                self._append_graph_value(
                    eta_name,
                    value,
                    statements,
                    program_steps,
                    axis=f"dpar_{aux.name}_FormulaCategoryLogit",
                    components=labels,
                )
                _append_link(program_steps, aux.name, _var(eta_name), aux.link.name)
                continue
            predictor = self._distributional.get(aux.name)
            if vector_sigma and predictor is not None:
                self._append_categorical_predictor(
                    predictor,
                    statements,
                    program_steps,
                    parameter=aux.name,
                    component_count=len(self._component_labels(formula)) - 1,
                    reference=None,
                    prefix=f"dpar_{aux.name}_",
                    correlated=correlated,
                )
                continue
            if predictor is None:
                if aux.fixed_default is not None and aux.name not in self._user_priors:
                    program_steps.append(
                        _let(aux.name, LetExprLiteral(value=aux.fixed_default))
                    )
                    continue
                prior = self._user_priors.get(aux.name, aux.prior)
                if (
                    self._family.name == "shifted_lognormal"
                    and aux.name == "ndt"
                    and aux.name not in self._user_priors
                ):
                    prior = f"Uniform(0.0, {float(np.min(formula.response_values))!r})"
                draw = _auxiliary_draw(
                    aux.name,
                    prior,
                    lower=1.0
                    if self._family.name == "frechet" and aux.name == "nu"
                    else None,
                )
                if vector_sigma:
                    draw = draw.with_(index=TypeName(name="FormulaCategoryLogit"))
                program_steps.append(draw)
            else:
                linear_predictor = self._append_predictor(
                    predictor,
                    program_steps,
                    prefix=f"dpar_{aux.name}_",
                    correlated=correlated,
                )
                eta_name = f"dpar_{aux.name}_eta"
                program_steps.append(_let(eta_name, linear_predictor))
                _append_link(program_steps, aux.name, _var(eta_name), aux.link.name)

    def _build_module(self, formula: Formula) -> Module:
        formula = self._threshold_location_formula(formula)
        statements: list[Statement] = []
        n_obs = formula.response_values.shape[0]
        statements.append(
            ObjectDecl(
                names=("Resp",),
                init=TypeFromExpr(
                    expr=DiscreteConstructor(
                        constructor="FinSet",
                        args=(str(int(n_obs)),),
                    )
                ),
            )
        )
        if self._family.baseline is not None and self._family.baseline.by is not None:
            group = self._family.baseline.by
            if group not in formula.group_levels:
                raise ValueError(
                    f"baseline group {group!r} requires retained group coding; "
                    "supply formula_from_data(..., groups=(group,))"
                )
        group_sizes = {
            _qvr_name(group): len(levels)
            for _, _, predictor in self._predictor_items(formula)
            for group, levels in predictor.group_levels.items()
        }
        for qgroup, size in group_sizes.items():
            self._append_finite_object(statements, qgroup, size)

        program_steps: list[ProgramStep] = []
        if self._family.baseline is not None:
            from quivers.formulas._survival import baseline_steps

            basis = self._family.baseline.resolve(formula)
            baseline_objects, baseline_draws = baseline_steps(
                basis,
                prior=_auxiliary_draw("sbhaz", self._user_priors["sbhaz"])
                if "sbhaz" in self._user_priors
                else None,
                group=_qvr_name(self._family.baseline.by)
                if self._family.baseline.by is not None
                else None,
            )
            statements.extend(baseline_objects)
            program_steps.extend(baseline_draws)
        if self._family.response is not None:
            count = self._category_count(formula, family=self._family.name)
            width = len(
                self._family.response.coordinate_indices(
                    self._component_labels(formula)
                )
            )
            self._append_finite_object(statements, "FormulaCategory", count)
            self._append_finite_object(statements, "FormulaCategoryLogit", width)
            axes = {"": ("FormulaCategoryLogit", width)}
            if (
                self._family.response.kind == "logistic_normal"
                and "sigma" in self._distributional
            ):
                self._append_finite_object(
                    statements, "dpar_sigma_FormulaCategoryLogit", width
                )
                axes["dpar_sigma_"] = ("dpar_sigma_FormulaCategoryLogit", width)
            correlated = self._append_correlations(
                formula, statements, program_steps, coordinate_axes=axes
            )
            self._append_graph_parameters(
                formula, statements, program_steps, correlated
            )
            if (
                self._predictor_graph is not None
                and "mu" in self._predictor_graph.roots
            ):
                labels = self._component_labels(formula)
                response = self._family.response
                assert response is not None
                coordinates = response.coordinate_indices(labels)
                expected = tuple(labels[index] for index in coordinates)
                self._append_graph_value(
                    "eta_nonbaseline",
                    self._predictor_graph.roots["mu"],
                    statements,
                    program_steps,
                    axis="FormulaCategoryLogit",
                    components=expected,
                )
                self._append_component_response(
                    program_steps,
                    parameter="mu",
                    prefix="",
                    reference=response.reference_index(labels),
                    coordinates=coordinates,
                    category_index="FormulaCategory",
                )
            else:
                self._append_categorical_predictor(
                    formula, statements, program_steps, correlated=correlated
                )
            if self._family.response.kind != "categorical":
                self._append_auxiliaries(formula, statements, program_steps, correlated)
                self._append_vector_observation(formula, statements, program_steps)
            else:
                program_steps.append(
                    _draw(
                        _qvr_name(formula.response_name),
                        "Categorical",
                        ("mu",),
                        index=TypeName(name="Resp"),
                        mode="score",
                        weights="formula_weights"
                        if formula.modifiers.weights is not None
                        else None,
                    )
                )
            statements.append(
                ProgramDecl(
                    name="model",
                    params=None,
                    value_parameters=self._runtime_parameters(formula),
                    domain=TypeName(name="Resp"),
                    codomain=TypeName(name="Resp"),
                    draws=tuple(program_steps),
                    return_vars=(_qvr_name(formula.response_name),),
                )
            )
            statements.append(ExportDecl(expr=ExprIdent(name="model")))
            return Module(statements=tuple(statements))

        proportional = formula.with_(
            fixed_columns=tuple(
                col for col in formula.fixed_columns if not col.category_specific
            ),
            random_terms=tuple(
                term
                for term in formula.random_terms
                if term.column is None or not term.column.category_specific
            ),
        )
        correlated = self._append_correlations(formula, statements, program_steps)
        self._append_graph_parameters(formula, statements, program_steps, correlated)
        eta_expr = self._append_predictor(
            proportional,
            program_steps,
            predictor_name=self._predictor_name,
            correlated=correlated,
        )

        if self._predictor_graph is not None and "mu" in self._predictor_graph.roots:
            root = self._predictor_graph.roots["mu"]
            if isinstance(root, ComponentExpression):
                raise ValueError(
                    "a scalar response requires a scalar nonlinear mu expression"
                )
            self._append_graph_value("graph_mu_eta", root, statements, program_steps)
            eta_expr = _var("graph_mu_eta")
            if self._predictor_name is not None:
                eta_expr = _add(eta_expr, _var(self._predictor_name))
        self._append_auxiliaries(formula, statements, program_steps, correlated)

        cutpoints_name: str | None = None
        cutpoint_steps: list[ProgramStep] = []
        if self._family.ordinal is not None:
            cutpoints_name = self._append_ordinal_cutpoints(
                formula, statements, cutpoint_steps
            )

        category_columns = tuple(
            col for col in formula.fixed_columns if col.category_specific
        )
        category_terms = tuple(
            term
            for term in formula.random_terms
            if term.column is not None and term.column.category_specific
        )
        if category_columns or category_terms:
            self._append_categorical_predictor(
                formula.with_(
                    fixed_columns=category_columns,
                    random_terms=category_terms,
                    offsets=(),
                ),
                statements,
                program_steps,
                ordinal_transitions=True,
                correlated=correlated,
            )

        program_steps.extend(cutpoint_steps)

        program_steps.append(_let("eta", eta_expr))
        retained_log_rate = (
            self._family.name == "com_poisson" or self._family.baseline is not None
        ) and self._family.location_link.name == "log"
        # A COM log-rate remains useful even when exp(eta) overflows. Its
        # inverse-link prediction is evaluated on request, not as an unused
        # part of every density or native response-mean computation.
        if not retained_log_rate:
            _append_link(
                program_steps,
                "mu",
                _var("eta"),
                "identity"
                if self._family.ordinal is not None
                else self._family.location_link.name,
            )

        if self._family.baseline is not None:
            obs_args = ("baseline_breaks", "baseline_controls", "mu")
        elif self._family.ordinal is not None:
            assert cutpoints_name is not None
            if (
                self._family.ordinal.kind == "cumulative"
                and self._family.ordinal.link in {"logit", "probit"}
            ):
                # Preserve the existing cumulative source and constructor when
                # discrimination is its fixed unit default.
                disc_is_unit = (
                    self._fixed_parameters.get("disc", 1.0) == 1.0
                    and "disc" not in self._distributional
                    and "disc" not in self._user_priors
                    and not category_columns
                    and not category_terms
                )
                if not disc_is_unit:
                    self._append_ordinal_contrasts(
                        formula, cutpoints_name, program_steps
                    )
                    program_steps.append(
                        _let("ordinal_scaled_eta", LetExprLiteral(value=0.0))
                    )
                    obs_args: tuple[str | float, ...] = (
                        "ordinal_scaled_eta",
                        "ordinal_contrasts",
                    )
                else:
                    obs_args = ("mu", cutpoints_name)
            else:
                self._append_ordinal_logits(
                    formula, cutpoints_name, statements, program_steps
                )
                obs_args = ("ordinal_logits",)
        elif self._family.name == "skew_normal":
            program_steps.extend(
                [
                    _let(
                        "skew_delta",
                        _call("sin", _call("atan", _var("alpha"))),
                    ),
                    _let(
                        "skew_center",
                        _mul(
                            _var("skew_delta"),
                            LetExprLiteral(value=math.sqrt(2 / math.pi)),
                        ),
                    ),
                    _let(
                        "skew_scale",
                        _div(
                            _var("sigma"),
                            _call(
                                "sqrt",
                                _sub(
                                    LetExprLiteral(value=1.0),
                                    _mul(_var("skew_center"), _var("skew_center")),
                                ),
                            ),
                        ),
                    ),
                    _let(
                        "skew_location",
                        _sub(_var("mu"), _mul(_var("skew_scale"), _var("skew_center"))),
                    ),
                ]
            )
            obs_args = ("skew_location", "skew_scale", "alpha")
        elif self._family.name == "shifted_lognormal":
            obs_args = ("mu", "sigma", "ndt")
        elif self._family.name == "wiener":
            # The raw positive reaction time and checked decision are explicit
            # inputs. Their signed joint event is an ordinary QVR expression.
            program_steps.append(
                _let(
                    _qvr_name(formula.response_name),
                    _mul(
                        _var("formula_response"),
                        _sub(
                            _mul(
                                LetExprLiteral(value=2.0),
                                _call("real", _var("formula_decision")),
                            ),
                            LetExprLiteral(value=1.0),
                        ),
                    ),
                )
            )
            obs_args = ("bs", "ndt", "bias", "mu")
        elif self._family.name == "frechet":
            program_steps.append(
                _let(
                    "frechet_scale",
                    _mul(
                        _var("mu"),
                        _call(
                            "exp",
                            _mul(
                                LetExprLiteral(value=-1.0),
                                _call(
                                    "lgamma",
                                    _sub(
                                        LetExprLiteral(value=1.0),
                                        _div(LetExprLiteral(value=1.0), _var("nu")),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )
            obs_args = ("frechet_scale", "nu")
        elif self._family.name == "gen_extreme_value":
            obs_args = ("mu", "sigma", "xi")
        elif self._family.name == "zero_inflated_asym_laplace":
            obs_args = ("zi", "mu", "sigma", "quantile")
        elif self._family.name in {
            "negbinomial2",
            "negative_binomial",
            "zero_inflated_negbinomial",
            "hurdle_negbinomial",
        }:
            # QVR follows torch's ``NegativeBinomial(total_count, probs)``
            # convention. The formula family is parameterized by mean ``mu``
            # and a concentration, for which probs = mu / (mu + concentration).
            shape = "disp" if self._family.name == "negative_binomial" else "shape"
            if self._family.name == "negbinomial2":
                shape = "negative_binomial_shape"
                program_steps.append(
                    _let(shape, _div(LetExprLiteral(value=1.0), _var("sigma")))
                )
            program_steps.append(
                _let(
                    "negative_binomial_probs",
                    _div(_var("mu"), _add(_var("mu"), _var(shape))),
                )
            )
            obs_args = (shape, "negative_binomial_probs")
            if self._family.name not in {"negative_binomial", "negbinomial2"}:
                zero = "hu" if self._family.name == "hurdle_negbinomial" else "zi"
                obs_args = (zero, *obs_args)
        elif self._family.name == "exgaussian":
            program_steps.append(
                _let("exgaussian_location", _sub(_var("mu"), _var("beta")))
            )
            program_steps.append(
                _let("exgaussian_rate", _div(LetExprLiteral(value=1.0), _var("beta")))
            )
            obs_args = ("exgaussian_location", "sigma", "exgaussian_rate")
        elif self._family.name == "exponential":
            program_steps.append(
                _let("exponential_rate", _div(LetExprLiteral(value=1.0), _var("mu")))
            )
            obs_args = ("exponential_rate",)
        elif self._family.name == "geometric":
            program_steps.append(
                _let(
                    "geometric_probs",
                    _div(
                        LetExprLiteral(value=1.0),
                        _add(LetExprLiteral(value=1.0), _var("mu")),
                    ),
                )
            )
            obs_args = ("geometric_probs",)
        elif self._family.name == "weibull":
            program_steps.append(
                _let(
                    "weibull_scale",
                    _mul(
                        _var("mu"),
                        _call(
                            "exp",
                            _mul(
                                LetExprLiteral(value=-1.0),
                                _call(
                                    "lgamma",
                                    _add(
                                        LetExprLiteral(value=1.0),
                                        _div(LetExprLiteral(value=1.0), _var("shape")),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )
            obs_args = ("weibull_scale", "shape")
        elif self._family.name == "hurdle_gamma":
            program_steps.append(
                _let("hurdle_gamma_rate", _div(_var("shape"), _var("mu")))
            )
            obs_args = ("hu", "shape", "hurdle_gamma_rate")
        elif self._family.name == "hurdle_lognormal":
            obs_args = ("hu", "mu", "sigma")
        elif self._family.name == "gamma":
            # The formula surface uses the brms mean/shape convention.
            # QVR's Gamma constructor follows torch and accepts
            # ``(concentration, rate)``, so rate = shape / mean.
            program_steps.append(_let("gamma_rate", _div(_var("shape"), _var("mu"))))
            obs_args = ("shape", "gamma_rate")
        elif self._family.name == "beta":
            # Convert mean/precision to the two concentrations expected by
            # QVR: alpha = mu * phi, beta = (1 - mu) * phi.
            program_steps.extend(
                [
                    _let(
                        "beta_concentration1",
                        _mul(_var("mu"), _var("phi")),
                    ),
                    _let(
                        "beta_concentration0",
                        _mul(
                            _sub(LetExprLiteral(value=1.0), _var("mu")),
                            _var("phi"),
                        ),
                    ),
                ]
            )
            obs_args = ("beta_concentration1", "beta_concentration0")
        elif self._family.name == "zero_inflated_beta":
            # A zero-only inflation is exactly the native zero/one mixture
            # with its conditional one-atom probability fixed at zero.
            obs_args = ("mu", "phi", "zi", 0.0)
        elif self._family.name == "student_t":
            # QVR follows torch's ``StudentT(df, loc, scale)`` ordering;
            # the formula surface names those slots ``nu``, ``mu``, and
            # ``sigma`` respectively.
            obs_args = ("nu", "mu", "sigma")
        elif self._family.name in {
            "zero_inflated_poisson",
            "hurdle_poisson",
        }:
            # Both QVR count-mixture constructors take the structural-zero
            # probability first and the Poisson rate second.
            obs_args = ("zi", "mu")
        elif self._family.trials_parameter is not None:
            trials: str | float
            if isinstance(self._binomial_trials, str):
                trials = _qvr_name(self._binomial_trials)
            else:
                assert self._binomial_trials is not None
                trials = float(self._binomial_trials)
            if self._family.observe_family.endswith("BetaBinomial"):
                program_steps.append(
                    _let("beta_binomial_concentration1", _mul(_var("mu"), _var("phi")))
                )
                program_steps.append(
                    _let(
                        "beta_binomial_concentration0",
                        _mul(_sub(LetExprLiteral(value=1.0), _var("mu")), _var("phi")),
                    )
                )
                obs_args = (
                    trials,
                    "beta_binomial_concentration1",
                    "beta_binomial_concentration0",
                )
            else:
                obs_args = (trials, "mu")
            if self._family.name.startswith("zero_inflated_"):
                obs_args = ("zi", *obs_args)
        elif self._family.name == "mixture":
            obs_args = self._append_mixture_parameters(
                formula, statements, program_steps
            )
        else:
            obs_args = (
                ("eta", "shape")
                if retained_log_rate
                else ("mu",) + tuple(self._family.extra_observe_args)
            )
        observation = _draw(
            _qvr_name(formula.response_name),
            self._family.observe_family,
            obs_args,
            index=TypeName(name="Resp"),
            mode="score",
            weights="formula_weights"
            if formula.modifiers.weights is not None
            else None,
        )
        if self._family.baseline is not None:
            observation = observation.with_(
                args=(
                    DrawArgNamed(
                        parameter="breaks", value=_to_draw_arg("baseline_breaks")
                    ),
                    DrawArgNamed(
                        parameter="controls", value=_to_draw_arg("baseline_controls")
                    ),
                    DrawArgNamed(
                        parameter="log_rate" if retained_log_rate else "rate",
                        value=_to_draw_arg("eta" if retained_log_rate else "mu"),
                    ),
                ),
            )
        elif retained_log_rate:
            observation = observation.with_(
                args=(
                    DrawArgNamed(parameter="log_rate", value=_to_draw_arg("eta")),
                    DrawArgNamed(parameter="dispersion", value=_to_draw_arg("shape")),
                ),
            )
        if (
            self._family.ordinal is not None
            and self._family.observe_family == "Categorical"
        ):
            observation = observation.with_(
                args=(
                    DrawArgNamed(
                        parameter="logits", value=_to_draw_arg("ordinal_logits")
                    ),
                )
            )
        if formula.modifiers.standard_error is not None:
            error = formula.modifiers.standard_error
            if not error.sigma:
                program_steps.append(
                    _let("formula_sigma", _var("formula_standard_error"))
                )
            else:
                row = _var("response")
                se = _index("formula_standard_error", row)
                sigma = (
                    _index("sigma", row)
                    if "sigma" in self._distributional
                    else _var("sigma")
                )
                program_steps.append(
                    _let(
                        "formula_error_reference",
                        _factor((("response", "Resp"),), body=_call("max", se, sigma)),
                    )
                )
                scale = _index("formula_error_reference", row)
                combined = _mul(
                    scale,
                    _call(
                        "sqrt",
                        _add(
                            _call("pow", _div(se, scale), LetExprLiteral(value=2.0)),
                            _call("pow", _div(sigma, scale), LetExprLiteral(value=2.0)),
                        ),
                    ),
                )
                program_steps.append(
                    _let(
                        "formula_sigma", _factor((("response", "Resp"),), body=combined)
                    )
                )
            observation = observation.with_(
                args=tuple(
                    argument.with_(value=_to_draw_arg("formula_sigma"))
                    if isinstance(argument, DrawArgNamed)
                    and argument.parameter == "scale"
                    else argument
                    for argument in observation.args or ()
                )
            )
        from quivers.formulas._response_roles import event_option_columns

        observation = observation.with_(
            options=(
                *observation.options,
                *(
                    OptionEntry(key=key, value=OptionName(value=column.qvr_name))
                    for key, column in event_option_columns(formula.modifiers)
                ),
            )
        )
        program_steps.append(observation)

        program_decl = ProgramDecl(
            name="model",
            params=None,
            value_parameters=self._runtime_parameters(formula),
            domain=TypeName(name="Resp"),
            codomain=TypeName(name="Resp"),
            draws=tuple(program_steps),
            return_vars=(_qvr_name(formula.response_name),),
        )
        statements.append(program_decl)
        statements.append(ExportDecl(expr=ExprIdent(name="model")))
        return Module(statements=tuple(statements))


__all__ = [
    "FormulaToQVRModule",
]
