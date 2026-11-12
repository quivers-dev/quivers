"""Round-trip lowering test: every gallery `.qvr` with a
[`ProgramDecl`][quivers.dsl.ast_nodes.ProgramDecl] lowers without
error, and the lowered IR's input list covers every free name in the
source while the body step count matches the source's expanded step
count (after `expand_composite_lets` plus marginalize unrolling).

A site whose morphism carries a parameter map lowers to more than one
node: the deterministic bindings that compute the family's arguments
from the conditioning row precede the draw. Those bindings carry
names the source never binds, so the step count excludes them and
each one is instead required to be read by a node that follows it.

Parameterised over `docs/examples/source/*.qvr`. Gallery files that
do not declare a program (deduction-only files such as
`ccg.qvr`) are skipped at collection time. Gallery files that
exercise composition constructs the lowering does not yet handle
(`scan`, `fan`, kernel composite-let chains) are reported as
`xfail` with the reported `UnsupportedConstruct` kind.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.dsl.ast_nodes import (
    DecoderDecl,
    DefineDecl,
    DrawArgName,
    EncoderDecl,
    ExprParser,
    LetExprVar,
    LetStep,
    LossDecl,
    MarginalizeStep,
    Module,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScoreStep,
    SignatureDecl,
)
from quivers.dsl.ast_nodes.declarations import ExportDecl
from quivers.dsl.parser import parse
from quivers.dsl.program_templates import instantiate_program, template_bindings
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.ir import (
    IRArg,
    IRArgBroadcast,
    IRArgList,
    IRArgMatrix,
    IRArgRef,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRScore,
)
from quivers.transpile.lower import (
    _names_in_raw_arg,
    free_vars_in_let,
)
from quivers.transpile.plan import Lower


_GALLERY_DIR = pathlib.Path(__file__).resolve().parents[2] / ("docs/examples/source")


def _gallery_paths() -> list[pathlib.Path]:
    """Return every executable `.qvr` gallery example."""
    paths: list[pathlib.Path] = []
    for path in sorted(_GALLERY_DIR.glob("*.qvr")):
        src = path.read_text()
        try:
            module = parse(src)
        except Exception:  # noqa: BLE001
            continue
        if any(
            isinstance(statement, ProgramDecl)
            or (
                isinstance(statement, DefineDecl)
                and isinstance(statement.expr, ExprParser)
            )
            or isinstance(
                statement,
                (SignatureDecl, EncoderDecl, DecoderDecl, LossDecl),
            )
            for statement in module.statements
        ):
            paths.append(path)
    return paths


GALLERY = _gallery_paths()


#: Gallery examples the lowering refuses, and the kind each reports.
#:
#: A refusal is pinned rather than dispatched on: an example that
#: raises where the test only knows "something raised" asserts
#: nothing, and a gap that closes goes unnoticed. Pinned, a closed gap
#: fails here and asks to be lowered for real, and a refusal that
#: changes kind fails rather than passing under the old reason.
#:
#: `program:absent` covers modules with no executable QIEC entry. The
#: rest name a construct the lowering has no form for, and each is the
#: same kind the renderers report for it.
_EXPECTED_LOWER_REFUSAL: dict[str, str] = {
    "bidirectional_rnn_lm": "scan:no-lowering",
    "deep_markov": "param-source:mlp",
    "gru_lm": "scan:no-lowering",
    "hmm": "scan:marginal:no-lowering",
    "lstm_lm": "scan:no-lowering",
    "ccg": "qiec:capability:search",
    "custom_rules": "qiec:capability:search",
    "montague_nli": "qiec:capability:search",
    "multimodal_tlg": "qiec:capability:search",
    "wmcfg": "qiec:capability:search",
    "quantifier_scope": "qiec:capability:search",
    "schema_chart_parser": "qiec:capability:search",
    "seq2seq": "param-source:mlp",
    "stacked_rnn_lm": "scan:no-lowering",
    "tensor_contraction": "program:absent",
    "term_autoencoder": "qiec:capability:neural-attachment",
    "transformer_lm": "param-source:mlp",
    "type_logical": "qiec:capability:search",
    "vae": "param-source:mlp",
    "vanilla_rnn_lm": "scan:no-lowering",
}


@pytest.mark.parametrize("path", GALLERY, ids=[p.stem for p in GALLERY])
def test_lower_roundtrip(path: pathlib.Path) -> None:
    """Lower the gallery example and verify the structural invariants."""
    src = path.read_text()
    module = parse(src)
    expected = _EXPECTED_LOWER_REFUSAL.get(path.stem)
    if expected is not None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            Lower().forward(module)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected) for k in kinds), (
            f"{path.stem}: lowering was expected to refuse with a "
            f"{expected!r} kind and reported {list(kinds)!r}. If the "
            f"gap closed, drop the row so the example is lowered and "
            f"its invariants checked; if a different gap fired, that "
            f"is the thing to look at."
        )
        return
    program = _pick_program(module)
    ir = Lower().forward(module)

    # Structural invariants on the IR shape.
    assert isinstance(ir, IRProgram)
    assert ir.name == program.name
    assert all(isinstance(inp, IRDataInput) for inp in ir.inputs)
    assert all(isinstance(node, IRNode) for node in ir.body)

    # Free names in the (expanded) source must be covered by the
    # IR's `inputs` plus the bound names in the body.
    expanded = expand_composite_lets(module, target="stan")
    expanded_program = _with_program_draws_inlined(
        expanded,
        next(
            s
            for s in expanded.statements
            if isinstance(s, ProgramDecl) and s.name == program.name
        ),
    )
    source_free = _source_free_names(expanded_program)
    ir_bound = _ir_bound_names(ir)
    ir_input_names = {inp.name for inp in ir.inputs}
    missing = source_free - ir_bound - ir_input_names
    assert not missing, (
        f"{path.name}: source free names not covered by IR "
        f"inputs/bindings: {sorted(missing)}"
    )

    # Body step count: every source `ProgramStep` (after composite
    # expansion + marginalize unrolling) maps to one IR site node.
    #
    # A site whose morphism carries a parameter map is preceded by
    # the deterministic bindings that compute the family's arguments
    # from the conditioning row. Those bindings bind names the source
    # never mentions, so they are excluded from the count; in exchange
    # every one of them has to be read by a node that follows it,
    # which is a tighter claim than the count alone would make.
    source_bound = _source_bound_names(expanded_program)
    head_bindings = _param_map_head_bindings(ir.body, source_bound)
    dangling = _unread_bindings(ir.body, head_bindings)
    assert not dangling, (
        f"{path.name}: parameter-map head bindings bound but never "
        f"read: {sorted(dangling)}"
    )
    expected_body_len = _expected_body_step_count(expanded_program)
    body_step_count = _ir_body_step_count(ir.body) - len(head_bindings)
    assert body_step_count == expected_body_len, (
        f"{path.name}: IR body step count {body_step_count} != "
        f"expected source step count {expected_body_len}"
    )


def _with_program_draws_inlined(module: Module, program: ProgramDecl) -> ProgramDecl:
    """The program with every draw from another program run in place.

    The plan runs a drawn program's steps in the caller under the
    caller's names, as the elaboration does, so the source the plan is
    compared against carries those steps rather than the draw.

    Parameters
    ----------
    module : Module
        The expanded module, whose other programs the draws name.
    program : ProgramDecl
        The program.

    Returns
    -------
    ProgramDecl
        The program with each draw from a program replaced by the
        callee's instantiated steps, recursively.
    """
    templates = {
        statement.name: statement
        for statement in module.statements
        if isinstance(statement, ProgramDecl) and statement.name != program.name
    }

    def inlined(steps: tuple[ProgramStep, ...]) -> tuple[ProgramStep, ...]:
        """The steps with every draw from a program run in place.

        Parameters
        ----------
        steps : tuple[ProgramStep, ...]
            The steps.

        Returns
        -------
        tuple[ProgramStep, ...]
            The steps, recursively instantiated.
        """
        out: list[ProgramStep] = []
        for step in steps:
            template = (
                templates.get(step.morphism)
                if isinstance(step, (SampleStep, ObserveStep))
                else None
            )
            if template is None:
                out.append(step)
                continue
            arguments = step.args or ()
            names = {
                argument.text: LetExprVar(name=argument.text)
                for argument in arguments
                if isinstance(argument, DrawArgName)
            }
            bindings = template_bindings(template, arguments, names)
            out.extend(inlined(instantiate_program(template, step.vars, bindings)))
        return tuple(out)

    return program.with_(draws=inlined(program.draws))


def _source_free_names(program: ProgramDecl) -> set[str]:
    """Return the set of free identifier names referenced in the
    program body but not bound by any sample / observe / let / score
    / marginalize step or by a program parameter."""
    bound: set[str] = set()
    if program.params is not None:
        bound.update(program.params)
    if program.type_params is not None:
        bound.update(str(p.name) for p in program.type_params)
    used: list[str] = []
    _collect_step_names(program.draws, bound, used)
    return set(used) - bound


def _collect_step_names(
    steps: tuple[ProgramStep, ...],
    bound: set[str],
    out: list[str],
) -> None:
    for step in steps:
        if isinstance(step, SampleStep):
            for v in step.vars:
                bound.add(v)
            for a in step.args or ():
                out.extend(_names_in_raw_arg(a))
        elif isinstance(step, ObserveStep):
            for v in step.vars:
                bound.add(v)
            for a in step.args or ():
                out.extend(_names_in_raw_arg(a))
            if step.via is not None:
                out.append(step.via)
        elif isinstance(step, MarginalizeStep):
            bound.add(step.var)
            for a in step.args or ():
                out.extend(_names_in_raw_arg(a))
            _collect_step_names(step.scope, bound, out)
        elif isinstance(step, LetStep):
            bound.add(step.name)
            out.extend(free_vars_in_let(step.value))
        elif isinstance(step, ScoreStep):
            bound.add(step.name)
            out.extend(free_vars_in_let(step.value))
        elif isinstance(step, ReturnStep):
            out.extend(step.vars)


def _ir_bound_names(ir: IRProgram) -> set[str]:
    """Return every name bound by a node in the IR body (recursive)."""
    out: set[str] = set()

    def walk(body: tuple[IRNode, ...]) -> None:
        for node in body:
            if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
                out.add(node.name)
            elif isinstance(node, IRScore):
                out.add(node.name)
            elif isinstance(node, IRMarginalize):
                out.add(node.latent)
                walk(node.scope)
            elif isinstance(node, IRDataInput):
                out.add(node.name)

    walk(ir.body)
    return out


def _source_bound_names(program: ProgramDecl) -> set[str]:
    """Every name the source program's steps bind, recursively."""
    out: set[str] = set()
    _collect_step_names(program.draws, out, [])
    return out


def _param_map_head_bindings(
    body: tuple[IRNode, ...], source_bound: set[str]
) -> set[str]:
    """The deterministic bindings lowering introduced on its own.

    A parameter map's heads are computed into fresh bindings named
    after the site they feed, so their names are exactly the
    `IRDeterministic` names the source never binds.
    """
    out: set[str] = set()
    for node in body:
        if isinstance(node, IRDeterministic) and node.name not in source_bound:
            out.add(node.name)
        elif isinstance(node, IRMarginalize):
            out |= _param_map_head_bindings(node.scope, source_bound)
    return out


def _unread_bindings(body: tuple[IRNode, ...], names: set[str]) -> set[str]:
    """Which of `names` nothing in the body reads."""
    return set(names) - _names_read(body)


def _names_read(body: tuple[IRNode, ...]) -> set[str]:
    """Every bound name the body's nodes read."""
    out: set[str] = set()
    for node in body:
        if isinstance(node, (IRDeterministic, IRScore)):
            out |= set(free_vars_in_let(node.expr))
        elif isinstance(node, (IRSample, IRObserve, IRMarginalize)):
            for arg in node.args:
                out |= _ir_arg_names(arg)
            if isinstance(node, IRMarginalize):
                out |= _names_read(node.scope)
        elif isinstance(node, IRReturn):
            out |= set(node.names)
    return out


def _ir_arg_names(arg: IRArg) -> set[str]:
    """Every bound name an IR argument reads."""
    if isinstance(arg, IRArgRef):
        out = {arg.name}
        for index in arg.indices:
            out |= _ir_arg_names(index)
        return out
    if isinstance(arg, IRArgBroadcast):
        return _ir_arg_names(arg.value)
    if isinstance(arg, IRArgList):
        return {n for e in arg.elements for n in _ir_arg_names(e)}
    if isinstance(arg, IRArgMatrix):
        return {n for row in arg.rows for n in _ir_arg_names(row)}
    return set()


def _ir_body_step_count(body: tuple[IRNode, ...]) -> int:
    """Count IR body nodes, excluding the synthesised `IRReturn`."""
    count = 0
    for node in body:
        if isinstance(node, IRReturn):
            continue
        if isinstance(node, IRMarginalize):
            count += 1 + _ir_body_step_count(node.scope)
        else:
            count += 1
    return count


def _expected_body_step_count(program: ProgramDecl) -> int:
    """Count source `ProgramStep`s, recursing into marginalize scopes,
    excluding the bare `return` slot (handled separately)."""
    count = 0
    for step in program.draws:
        if isinstance(step, ReturnStep):
            continue
        if isinstance(step, MarginalizeStep):
            count += 1 + _expected_body_step_count(
                ProgramDecl(
                    name=program.name,
                    domain=program.domain,
                    codomain=program.codomain,
                    draws=step.scope,
                ),
            )
        else:
            count += 1
    return count


def _pick_program(module) -> ProgramDecl:
    """Match `Lower._pick_program`: prefer a program named in an
    `export` declaration; otherwise return the last `ProgramDecl`.
    A fixture that declares both `program lstm_cell` and
    `program lstm_lm` with `export lstm_lm` lowers `lstm_lm`, so the
    test invariants must apply against the same program Lower picked.
    """
    programs = []
    exported_names = set()
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            programs.append(stmt)
        elif isinstance(stmt, ExportDecl):
            if hasattr(stmt, "expr") and hasattr(stmt.expr, "name"):
                exported_names.add(stmt.expr.name)
    if not programs:
        raise AssertionError("no ProgramDecl in module")
    return next(
        (p for p in programs if p.name in exported_names),
        programs[-1],
    )
