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
    ExportDecl,
    ExprIdent,
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    LetFactorBinder,
    LetFactorCase,
    LetStep,
    DiscreteConstructor,
    DrawArgNamed,
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
    if "(" not in prior.partition("(")[2]:
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


def _let(name: str, value: LetExprNode) -> LetStep:
    return LetStep(name=name, value=value)


def _var(name: str) -> LetExprVar:
    return LetExprVar(name=name)


def _add(*terms: LetExprNode) -> LetExprNode:
    if not terms:
        return LetExprLiteral(value=0.0)
    out = terms[0]
    for t in terms[1:]:
        out = LetExprBinOp(op="+", left=out, right=t)
    return out


def _mul(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="*", left=left, right=right)


def _sub(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="-", left=left, right=right)


def _div(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="/", left=left, right=right)


def _call(name: str, *args: LetExprNode) -> LetExprCall:
    return LetExprCall(func=name, args=tuple(args))


def _index(name: str, *indices: LetExprNode) -> LetExprIndex:
    return LetExprIndex(array=_var(name), indices=tuple(indices))


def _factor(
    binders: tuple[tuple[str, str], ...],
    *,
    body: LetExprNode | None = None,
    cases: tuple[tuple[int, LetExprNode], ...] = (),
) -> LetExprFactor:
    return LetExprFactor(
        binders=tuple(
            LetFactorBinder(var=name, index=TypeName(name=index))
            for name, index in binders
        ),
        body=body,
        cases=tuple(LetFactorCase(label=label, value=value) for label, value in cases),
    )


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


def _decode_module(module: Module) -> dict:
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
            if qgroup == "FormulaCategoryLogit":
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
        Random-effect parameterization. Categorical random effects use the
        non-centered form.
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
        nonnegative signed 64-bit range,
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
        binomial_trials: int | str = 1,
        thresholds_by: str | None = None,
        mixture_components: int = 2,
        predictor_name: str | None = None,
        distributional: Mapping[str, Formula] | None = None,
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
        auxiliary_names = {aux.name for aux in family.aux_params}
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
            if name == "quantile":
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
        conflicting = self._distributional.keys() & self._user_priors.keys()
        if conflicting:
            raise ValueError(
                f"priors on distributionally regressed parameters {sorted(conflicting)} must name their coefficients, such as 'dpar_sigma_intercept'"
            )
        if thresholds_by is not None and family.name != "cumulative":
            raise ValueError(
                "FormulaToQVRModule: thresholds_by is only valid for the "
                "cumulative family"
            )
        if family.trials_parameter is None and binomial_trials != 1:
            raise ValueError(
                "FormulaToQVRModule: binomial_trials is only valid for the "
                "families with a trial-count parameter"
            )
        if not isinstance(binomial_trials, (int, str)) or isinstance(
            binomial_trials, bool
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
        self._reparameterize = reparameterize

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
        if formula.response_weights is not None:
            weights = formula.response_weights.data
            if weights.shape != formula.response_values.shape:
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
        for name, predictor in (("mu", formula), *self._distributional.items()):
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
        for prefix, predictor in (
            ("", formula),
            *(
                (f"dpar_{name}_", predictor)
                for name, predictor in self._distributional.items()
            ),
        ):
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
            random_terms=formula.random_terms,
            offsets=formula.offsets,
            response_weights=formula.response_weights,
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
        names = {
            aux.name
            for aux in self._family.aux_params
            if aux.name not in self._distributional
            and aux.name not in self._fixed_parameters
        }
        if self._family.name == "mixture":
            names.update(("mixture_logit", "mixture_offset", "mixture_scale"))
        for prefix, predictor in (
            ("", formula),
            *((f"dpar_{name}_", item) for name, item in self._distributional.items()),
        ):
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
        decoded = _decode_module(target)
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
                        data=fixed_column_data[qvr_name],
                    )
                )

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
        return Formula(
            formula=complement.formula,
            response_name=complement.response_name,
            fixed_columns=tuple(fixed_columns),
            random_terms=tuple(random_terms),
            offsets=complement.offsets,
            response_weights=complement.response_weights,
            response_values=complement.response_values,
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
        if formula.response_weights is not None:
            obs["formula_weights"] = torch.as_tensor(
                formula.response_weights.data.copy()
            )
        for col in formula.fixed_columns:
            if col.is_intercept:
                continue
            obs[col.qvr_name] = torch.as_tensor(col.data.copy(), dtype=torch.float32)
        for name, predictor in self._distributional.items():
            for col in predictor.fixed_columns:
                if not col.is_intercept:
                    obs[f"dpar_{name}_{col.qvr_name}"] = torch.as_tensor(
                        col.data.copy(), dtype=torch.float32
                    )
        for prefix, predictor in (
            ("", formula),
            *(
                (f"dpar_{name}_", predictor)
                for name, predictor in self._distributional.items()
            ),
        ):
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
    def _category_count(formula: Formula, *, family: str) -> int:
        values = np.asarray(formula.response_values, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{family} family requires a nonempty, finite response")
        integers = values.astype(np.int64)
        if not np.equal(values, integers).all():
            raise ValueError(f"{family} family requires integer category labels")
        levels = np.unique(integers)
        expected = np.arange(int(levels[-1]) + 1)
        if not np.array_equal(levels, expected):
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
        """Emit identified ordered cutpoints and return their variable name.

        Cutpoints are represented by a zero origin followed by cumulative
        positive spacings and are centered before use. Centering separates the
        threshold location from the formula intercept. With ``thresholds_by``
        the log spacings receive hierarchical group-level deviations, so every
        group gets an ordered scale without introducing an unidentified random
        threshold shift alongside its random intercept.
        """
        n_categories = self._ordinal_category_count(formula)
        n_cutpoints = n_categories - 1
        cutpoint_index = "FormulaCutpoint"
        self._append_finite_object(statements, cutpoint_index, n_cutpoints)

        if n_cutpoints == 1:
            if self._thresholds_by is not None:
                raise ValueError(
                    "thresholds_by requires at least three response categories; "
                    "a binary cumulative model has only one threshold"
                )
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

        spacing_index = "FormulaSpacing"
        n_spacings = n_cutpoints - 1
        self._append_finite_object(statements, spacing_index, n_spacings)

        if self._thresholds_by is None:
            program_steps.append(
                _draw(
                    "cutpoint_log_spacing",
                    "Normal",
                    (0.0, 1.0),
                    index=TypeName(name=spacing_index),
                )
            )
            program_steps.append(
                _let(
                    "cutpoint_cumulative",
                    _call("cumsum", _call("exp", _var("cutpoint_log_spacing"))),
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
        cell_index = "FormulaThresholdCell"
        self._append_finite_object(statements, cell_index, n_groups * n_spacings)

        program_steps.extend(
            [
                _draw(
                    "cutpoint_log_spacing_mean",
                    "Normal",
                    (0.0, 1.0),
                    index=TypeName(name=spacing_index),
                ),
                _draw(
                    "cutpoint_log_spacing_scale",
                    "HalfNormal",
                    (0.5,),
                    index=TypeName(name=spacing_index),
                ),
                _draw(
                    "cutpoint_log_spacing_noise",
                    "Normal",
                    (0.0, 1.0),
                    index=TypeName(name=cell_index),
                ),
            ]
        )
        flat_index = _add(
            _mul(
                _var("threshold_group"),
                LetExprLiteral(value=float(n_spacings), integral=True),
            ),
            _var("spacing"),
        )
        program_steps.append(
            _let(
                "cutpoint_noise_matrix",
                _factor(
                    (("threshold_group", qgroup), ("spacing", spacing_index)),
                    body=_index("cutpoint_log_spacing_noise", flat_index),
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
        program_steps.append(
            _let(
                "cutpoint_log_spacing_matrix",
                _factor(
                    (("threshold_group", qgroup), ("spacing", spacing_index)),
                    body=log_spacing,
                ),
            )
        )
        program_steps.append(
            _let(
                "cutpoint_cumulative_matrix",
                _call("cumsum", _call("exp", _var("cutpoint_log_spacing_matrix"))),
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

    def _append_categorical_predictor(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> None:
        """Emit an identified multinomial-logit predictor.

        Category zero is the reference class. Every fixed coefficient and
        random effect therefore has ``K - 1`` entries, and the compiler
        prepends a zero logit before applying softmax. This removes the common
        additive direction that an unconstrained ``K``-logit parameterization
        would leave unidentified.
        """
        n_categories = self._category_count(formula, family="categorical")
        n_logits = n_categories - 1
        category_index = "FormulaCategory"
        logit_index = "FormulaCategoryLogit"
        self._append_finite_object(statements, category_index, n_categories)
        self._append_finite_object(statements, logit_index, n_logits)

        cell_terms: list[LetExprNode] = []
        response_var = _var("response")
        logit_var = _var("category_logit")

        for col in formula.fixed_columns:
            beta_name = "intercept" if col.is_intercept else f"beta_{col.qvr_name}"
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
                cell_terms.append(_mul(coefficient, _index(col.qvr_name, response_var)))

        cell_terms.extend(
            _index(f"formula_offset_{index}", response_var)
            for index, _ in enumerate(formula.offsets)
        )

        if self._predictor_name is not None:
            predictor_index = _add(
                _mul(
                    response_var,
                    LetExprLiteral(value=float(n_logits), integral=True),
                ),
                logit_var,
            )
            cell_terms.append(_index(self._predictor_name, predictor_index))

        if self._reparameterize == "centered" and formula.random_terms:
            raise ValueError(
                "categorical random effects currently require "
                "reparameterize='noncentered'"
            )

        blocks: dict[tuple[str, str], list[RandomTerm]] = {}
        for term in formula.random_terms:
            if term.correlation_id is not None:
                identity = (
                    f"shared_{term.correlation_id}"
                    if term.shared
                    else term.correlation_id
                )
                blocks.setdefault((term.group, identity), []).append(term)
        correlations: dict[tuple[str, str], tuple[str, int, list[RandomTerm]]] = {}
        for (group, identity), members in blocks.items():
            size = len(members) * n_logits
            if size < 2:
                continue
            correlation = f"cor_{_qvr_name(identity)}"
            dimension = f"FormulaCorrelation_{_qvr_name(identity)}"
            self._append_finite_object(statements, dimension, size)
            family, arguments = _parse_prior_call(
                self._user_priors.get(correlation, self._correlation_prior)
            )
            if family != "LKJCholesky":
                raise ValueError("correlation priors must use LKJCholesky")
            program_steps.append(
                _draw(correlation, family, arguments, index=TypeName(name=dimension))
            )
            for position, term in enumerate(members):
                correlations[(group, term.slope)] = (correlation, position, members)

        for term in formula.random_terms:
            qgroup = _qvr_name(term.group)
            qslope = _qvr_name(term.slope)
            sigma_var = f"sigma_{qgroup}_{qslope}"
            sigma_prior_text = self._user_priors.get(
                sigma_var, self._random_scale_prior
            )
            sf_family, sf_args = _parse_prior_call(sigma_prior_text)
            program_steps.append(
                _draw(sigma_var, sf_family, sf_args, index=TypeName(name=logit_index))
            )
            n_groups = len(formula.group_levels[term.group])
            cell_index = f"FormulaRandomCell_{qgroup}_{qslope}"
            self._append_finite_object(statements, cell_index, n_groups * n_logits)
            program_steps.append(
                _draw(
                    f"z_{qgroup}_{qslope}",
                    "Normal",
                    (0.0, 1.0),
                    index=TypeName(name=cell_index),
                )
            )

        for term in formula.random_terms:
            group = term.group
            qgroup = _qvr_name(group)
            slope = term.slope
            qslope = _qvr_name(slope)
            sigma_var = f"sigma_{qgroup}_{qslope}"
            z_var = f"z_{qgroup}_{qslope}"
            latent = (
                f"alpha_{qgroup}" if slope == "Intercept" else f"beta_{qgroup}_{qslope}"
            )
            contribution_name = f"{z_var}_contribution"
            correlated = correlations.get((group, slope))
            column_name = (
                f"random_{term.column.qvr_name}" if term.column is not None else qslope
            )
            if correlated is not None:
                correlation, term_position, members = correlated
                cases: list[tuple[int, LetExprNode]] = []
                for category in range(n_logits):
                    row = term_position * n_logits + category
                    terms: list[LetExprNode] = []
                    for column in range(row + 1):
                        member = members[column // n_logits]
                        member_category = column % n_logits
                        innovation = f"z_{qgroup}_{_qvr_name(member.slope)}"
                        position = _add(
                            _mul(
                                _var("random_group"),
                                LetExprLiteral(value=float(n_logits), integral=True),
                            ),
                            LetExprLiteral(value=float(member_category), integral=True),
                        )
                        terms.append(
                            _mul(
                                _index(
                                    correlation,
                                    LetExprLiteral(value=float(row), integral=True),
                                    LetExprLiteral(value=float(column), integral=True),
                                ),
                                _index(innovation, position),
                            )
                        )
                    contribution = _mul(
                        _index(
                            sigma_var,
                            LetExprLiteral(value=float(category), integral=True),
                        ),
                        _add(*terms),
                    )
                    cases.append(
                        (
                            category,
                            _factor((("random_group", qgroup),), body=contribution),
                        )
                    )
                value = _factor((("category_logit", logit_index),), cases=tuple(cases))
            else:
                z_matrix = f"{z_var}_matrix"
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
                            body=_index(z_var, flat_index),
                        ),
                    )
                )
                contribution = _mul(
                    _index(sigma_var, logit_var),
                    _index(z_matrix, logit_var, _var("random_group")),
                )
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
                "eta_nonbaseline",
                _factor(
                    (("response", "Resp"), ("category_logit", logit_index)),
                    body=_add(*cell_terms),
                ),
            )
        )
        program_steps.append(
            _let(
                "categorical_zero_logit",
                _factor(
                    (("response", "Resp"),),
                    body=LetExprLiteral(value=0.0),
                ),
            )
        )
        category_cases: tuple[tuple[int, LetExprNode], ...] = (
            (0, _var("categorical_zero_logit")),
        ) + tuple(
            (
                category,
                _factor(
                    (("response", "Resp"),),
                    body=_index(
                        "eta_nonbaseline",
                        response_var,
                        LetExprLiteral(value=float(category - 1), integral=True),
                    ),
                ),
            )
            for category in range(1, n_categories)
        )
        program_steps.append(
            _let(
                "categorical_logits_by_category",
                _factor((("category", category_index),), cases=category_cases),
            )
        )
        program_steps.append(
            _let(
                "eta",
                _factor(
                    (("response", "Resp"), ("category", category_index)),
                    body=_index(
                        "categorical_logits_by_category",
                        _var("category"),
                        response_var,
                    ),
                ),
            )
        )
        program_steps.append(_let("mu", _call("softmax", _var("eta"))))

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

    def _append_correlations(
        self,
        formula: Formula,
        statements: list[Statement],
        steps: list[ProgramStep],
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
        for prefix, predictor in (
            ("", formula),
            *(
                (f"dpar_{name}_", predictor)
                for name, predictor in self._distributional.items()
            ),
        ):
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
            if len(members) < 2:
                continue
            qgroup = _qvr_name(group)
            correlation = f"cor_{_qvr_name(identity)}"
            dimension = f"FormulaCorrelation_{_qvr_name(identity)}"
            self._append_finite_object(statements, dimension, len(members))
            family, arguments = _parse_prior_call(
                self._user_priors.get(correlation, self._correlation_prior)
            )
            if family != "LKJCholesky":
                raise ValueError("correlation priors must use LKJCholesky")
            steps.append(
                _draw(correlation, family, arguments, index=TypeName(name=dimension))
            )
            scales: list[str] = []
            innovations: list[str] = []
            for prefix, term in members:
                qslope = _qvr_name(term.slope)
                sigma = f"{prefix}sigma_{qgroup}_{qslope}"
                prior_family, prior_args = _parse_prior_call(
                    self._user_priors.get(sigma, self._random_scale_prior)
                )
                steps.append(_draw(sigma, prior_family, prior_args))
                scales.append(sigma)
                innovations.append(f"{prefix}z_{qgroup}_{qslope}")
            for row, (prefix, term) in enumerate(members):

                def combination(
                    index: LetExprNode | None, *, row: int = row
                ) -> LetExprNode:
                    parts = [
                        _mul(
                            _index(
                                correlation,
                                LetExprLiteral(value=float(row), integral=True),
                                LetExprLiteral(value=float(column), integral=True),
                            ),
                            _index(innovations[column], index)
                            if index is not None
                            else _var(innovations[column]),
                        )
                        for column in range(row)
                    ]
                    return _add(*parts) if parts else LetExprLiteral(value=0.0)

                innovation = innovations[row]
                latent = (
                    f"{prefix}alpha_{qgroup}"
                    if term.slope == "Intercept"
                    else f"{prefix}beta_{qgroup}_{_qvr_name(term.slope)}"
                )
                if self._reparameterize == "noncentered":
                    steps.append(
                        _draw(
                            innovation,
                            "Normal",
                            (0.0, 1.0),
                            index=TypeName(name=qgroup),
                        )
                    )
                    diagonal_term = _mul(
                        _index(
                            correlation,
                            LetExprLiteral(value=float(row), integral=True),
                            LetExprLiteral(value=float(row), integral=True),
                        ),
                        _var(innovation),
                    )
                    steps.append(
                        _let(
                            latent,
                            _mul(
                                _var(scales[row]),
                                _add(combination(None), diagonal_term),
                            ),
                        )
                    )
                else:
                    location = f"{latent}_conditional_location"
                    scale = f"{latent}_conditional_scale"
                    diagonal = _index(
                        correlation,
                        LetExprLiteral(value=float(row), integral=True),
                        LetExprLiteral(value=float(row), integral=True),
                    )
                    steps.append(
                        _let(
                            location,
                            _mul(_var(scales[row]), combination(None)),
                        )
                    )
                    steps.append(_let(scale, _mul(_var(scales[row]), diagonal)))
                    steps.append(
                        _draw(
                            latent,
                            "Normal",
                            (location, scale),
                            index=TypeName(name=qgroup),
                        )
                    )
                    standardized = _sub(
                        _div(_var(latent), _var(scales[row])),
                        combination(None),
                    )
                    steps.append(
                        _let(
                            innovation,
                            _div(standardized, diagonal),
                        )
                    )
                result[(prefix, group, term.slope)] = _index(
                    latent, _var(f"{qgroup}_idx")
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
        #   directly. Easy to read; suffers Neal's funnel under HMC /
        #   NUTS and under-shrinks variance components under VI.
        # * Non-centered ("noncentered", default): emit
        #     z_g_<slope> : g <- Normal(0, 1)
        #     let alpha_g = sigma_g * z_g_<slope>
        #   The per-level draws sit in unit space; ``sigma_g`` scales
        #   the (deterministic) ``alpha_g`` afterwards. Mathematically
        #   identical to centered, but eliminates the funnel and lets
        #   HMC / NUTS sample variance components reliably.  Standard
        #   practice in Stan / brms when divergences appear; we make
        #   it the default so the formula path matches expert advice
        #   out of the box.
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
            for predictor in (formula, *self._distributional.values())
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
        if formula.response_weights is not None:
            weight_type = QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(QiecTypeName(name="Real"),),
                indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
            )
            parameters += (
                QiecValueParameter(name="formula_weights", type_expr=weight_type),
            )
        return parameters

    def _build_module(self, formula: Formula) -> Module:
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
        group_sizes = {
            _qvr_name(group): len(levels)
            for predictor in (formula, *self._distributional.values())
            for group, levels in predictor.group_levels.items()
        }
        for qgroup, size in group_sizes.items():
            self._append_finite_object(statements, qgroup, size)

        program_steps: list[ProgramStep] = []
        if self._family.name == "categorical":
            self._append_categorical_predictor(formula, statements, program_steps)
            program_steps.append(
                _draw(
                    _qvr_name(formula.response_name),
                    "Categorical",
                    ("mu",),
                    index=TypeName(name="Resp"),
                    mode="score",
                    weights="formula_weights"
                    if formula.response_weights is not None
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

        correlated = self._append_correlations(formula, statements, program_steps)
        eta_expr = self._append_predictor(
            formula,
            program_steps,
            predictor_name=self._predictor_name,
            correlated=correlated,
        )

        # Auxiliary family parameters.
        for aux in self._family.aux_params:
            if aux.name in self._fixed_parameters:
                program_steps.append(
                    _let(
                        aux.name,
                        LetExprLiteral(value=float(self._fixed_parameters[aux.name])),
                    )
                )
                continue
            predictor = self._distributional.get(aux.name)
            if predictor is None:
                prior = self._user_priors.get(aux.name, aux.prior)
                if (
                    self._family.name == "shifted_lognormal"
                    and aux.name == "ndt"
                    and aux.name not in self._user_priors
                ):
                    prior = f"Uniform(0.0, {float(np.min(formula.response_values))!r})"
                program_steps.append(
                    _auxiliary_draw(
                        aux.name,
                        prior,
                        lower=1.0
                        if self._family.name == "frechet" and aux.name == "nu"
                        else None,
                    )
                )
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

        cutpoints_name: str | None = None
        if self._family.name == "cumulative":
            cutpoints_name = self._append_ordinal_cutpoints(
                formula, statements, program_steps
            )

        program_steps.append(_let("eta", eta_expr))
        _append_link(
            program_steps,
            "mu",
            _var("eta"),
            "identity"
            if self._family.name == "cumulative"
            else self._family.location_link.name,
        )

        if self._family.name == "cumulative":
            assert cutpoints_name is not None
            obs_args: tuple[str | float, ...] = ("mu", cutpoints_name)
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
            obs_args = ("mu",) + tuple(self._family.extra_observe_args)
        program_steps.append(
            _draw(
                _qvr_name(formula.response_name),
                self._family.observe_family,
                obs_args,
                index=TypeName(name="Resp"),
                mode="score",
                weights="formula_weights"
                if formula.response_weights is not None
                else None,
            )
        )

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
