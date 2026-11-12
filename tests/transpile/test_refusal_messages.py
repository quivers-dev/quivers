"""Every refusal the transpiler can raise must explain itself.

A refusal reaches the user as
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct], whose
message is rendered from a structured `kinds` list by
[`user_facing_message`][quivers.transpile._diagnostics.user_facing_message].
A kind no renderer explains raises
[`UnregisteredRefusalKind`][quivers.transpile._diagnostics.UnregisteredRefusalKind]
instead of reaching a user as a tag with no account of what to do.

This module holds the line that every kind the transpiler can raise has
an explanation. The kinds are enumerated two ways. Every raise site of
`UnsupportedConstruct` or
[`StepResolutionError`][quivers.dsl.step_resolution.StepResolutionError]
under `src/quivers` is read statically, so a site no example reaches is
still checked; a site that forwards kinds computed elsewhere is named in
`FORWARDING_SITES` with where those kinds come from. The kinds quoted
in `docs/transpile-support.md`, which `test_support_matrix_docs` pins
to a fresh measurement over every gallery program and construct
fixture against every backend, cover the forwarded kinds the pipeline
currently produces.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import quivers.transpile

from quivers.dsl.step_resolution import StepResolutionError
from quivers.transpile._diagnostics import (
    UnregisteredRefusalKind,
    user_facing_message,
)
from tests.transpile import support_matrix

#: Backends the support page reports on, used to check that a kind
#: explains itself whichever target refused it.
TARGETS: tuple[str, ...] = (
    "qvr-stan",
    "qvr-numpyro",
    "qvr-pyro",
    "qvr-pymc",
    "qvr-edward2",
    "qvr-gen",
    "qvr-turing",
    "qvr-webppl",
    "qvr-jags",
    "qvr-bugs",
    "qvr-church",
)


def _emitted_kinds() -> tuple[str, ...]:
    """Every kind quoted in the published support matrix.

    The page records each refusal's reported kinds in fenced `text`
    blocks, one kind per line.
    """
    page = support_matrix.DOC_PATH.read_text(encoding="utf-8")
    kinds: set[str] = set()
    for block in re.findall(r"```text\n(.*?)```", page, re.DOTALL):
        for line in block.splitlines():
            candidate = line.strip().strip("`")
            if candidate and " " not in candidate:
                kinds.add(candidate)
    if not kinds:
        raise AssertionError(
            f"{support_matrix.DOC_PATH.name} quoted no refusal kinds, so "
            f"this test would vacuously pass. The page's format changed; "
            f"update the reader."
        )
    return tuple(sorted(kinds))


#: Root of the package whose raise sites are scanned statically.
PACKAGE_ROOT = Path(quivers.transpile.__file__).parent.parent

#: The exceptions whose second positional argument is a list of kinds.
RAISING_CLASSES: frozenset[str] = frozenset(
    {"UnsupportedConstruct", "StepResolutionError"}
)

#: Raise sites whose kinds are computed elsewhere, by file and enclosing
#: function, with where the kinds come from. Every other site has to
#: spell its kinds as literals or f-strings so this module can read them.
FORWARDING_SITES: dict[tuple[str, str], str] = {
    ("transpile/_expand_composites.py", "expand_composite_lets"): (
        "a StepResolutionError's kinds, whose raise sites are scanned"
    ),
    ("transpile/_resolve.py", "resolve_step_dist"): (
        "a StepResolutionError's kinds, whose raise sites are scanned"
    ),
    ("transpile/_resolve.py", "build_morphism_table"): (
        "a StepResolutionError's kinds, whose raise sites are scanned"
    ),
    ("transpile/_resolve.py", "param_source_kind"): (
        "a StepResolutionError's kinds, whose raise sites are scanned"
    ),
    ("transpile/_support.py", "unsupported_for"): (
        "declaration kinds, each rendered by `_DECLARATION_MEANING`"
    ),
    ("transpile/plan.py", "_refuse_host_runtime_calls"): (
        "QIEC capability diagnostics, measured on the support page"
    ),
    ("transpile/plan.py", "checked_program_plan"): (
        "an elaboration gap's leading kind, a StepResolutionError kind or "
        "one of the elaboration's own gap kinds"
    ),
    ("transpile/plan.py", "checked_module"): (
        "an elaboration diagnostic, rendered as `program:elaboration:...` "
        "unless it leads with a structured kind"
    ),
    ("transpile/plan.py", "forward"): (
        "an elaboration gap's leading kind, a StepResolutionError kind or "
        "one of the elaboration's own gap kinds"
    ),
    ("transpile/renderers/_qiec.py", "render_computations_dynamic"): (
        "QIEC capability diagnostics, measured on the support page"
    ),
    ("transpile/renderers/_qiec.py", "refuse_static_gaps"): (
        "QIEC capability diagnostics, measured on the support page"
    ),
    ("transpile/renderers/bugs.py", "_reject_list_args"): (
        "the family lookup's and the argument emitter's kinds, whose raise "
        "sites are scanned"
    ),
}


def _kind_literal(node: ast.expr) -> str | None:
    """The concrete kind a literal or f-string argument spells.

    An f-string's interpolations are stamped with a placeholder: the
    renderer dispatches on the literal prefixes, so a placeholder in
    a value slot exercises the same branch the real value would.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(part.value) if isinstance(part, ast.Constant) else "X"
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _kind_literal(node.left)
        if left is not None:
            return left
    return None


class _RaiseSite:
    """One call of a kind-carrying exception, read off the source."""

    def __init__(self, path: str, function: str, line: int, call: ast.Call) -> None:
        self.path = path
        self.function = function
        self.line = line
        self.call = call

    def kinds_argument(self) -> ast.expr | None:
        """The call's kinds list, or ``None`` when it passes none."""
        if len(self.call.args) < 2:
            return None
        return self.call.args[1]

    def explanation(self) -> str | None:
        """The literal `explanation=` keyword, if the site passes one."""
        for keyword in self.call.keywords:
            if keyword.arg == "explanation":
                return _kind_literal(keyword.value) or "X"
        return None


def _raise_sites() -> list[_RaiseSite]:
    """Every call of a kind-carrying exception under `src/quivers`.

    Returns
    -------
    list[_RaiseSite]
        The sites, in file and line order.
    """
    sites: list[_RaiseSite] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name in RAISING_CLASSES:
                    sites.append(_RaiseSite(relative, function.name, node.lineno, node))
    unique = {(site.path, site.line, site.call.col_offset): site for site in sites}
    if not unique:
        raise AssertionError(
            f"no kind-carrying raise sites were found under {PACKAGE_ROOT}, "
            f"so this test would vacuously pass."
        )
    return [unique[key] for key in sorted(unique)]


RAISE_SITES: list[_RaiseSite] = _raise_sites()


def _static_kinds() -> tuple[str, ...]:
    """Every kind spelled at a literal raise site, with its explanation.

    The measured page only records kinds the gallery and fixture
    programs actually provoke, so a raise site no example reaches
    would go unchecked. Reading the raise sites off the source covers
    those too, which is how the mis-spelled family kinds were found.
    A site that passes `explanation=` reaches the renderer as the kind
    followed by the explanation, so that form is what is checked.
    """
    kinds: set[str] = set()
    for site in RAISE_SITES:
        argument = site.kinds_argument()
        if not isinstance(argument, ast.List | ast.Tuple):
            continue
        explanation = site.explanation()
        for element in argument.elts:
            kind = _kind_literal(element)
            if kind:
                kinds.add(f"{kind}: {explanation}" if explanation else kind)
    return tuple(sorted(kinds))


EMITTED_KINDS: tuple[str, ...] = tuple(
    sorted(set(_emitted_kinds()) | set(_static_kinds()))
)


def _render(target: str, kind: str) -> str:
    """Render one kind, failing the test when it has no explanation.

    Parameters
    ----------
    target : str
        The backend the refusal is reported for.
    kind : str
        The kind.

    Returns
    -------
    str
        The rendered message.
    """
    try:
        return user_facing_message(target, (kind,))
    except UnregisteredRefusalKind as error:
        pytest.fail(
            f"transpiling to {target} can refuse with kind {kind!r}, which "
            f"has no explanation registered ({error}). Register one in "
            f"`_diagnostics` so the refusal says what the construct means, "
            f"why this target has no form for it, and what to write instead."
        )


@pytest.mark.parametrize("kind", EMITTED_KINDS)
@pytest.mark.parametrize("target", TARGETS)
def test_every_emitted_kind_renders_an_explanation(target: str, kind: str) -> None:
    """No emitted kind lacks a registered explanation."""
    _render(target, kind)


def test_every_raise_site_spells_its_kinds_or_names_their_source() -> None:
    """A raise site the static enumeration cannot read is named.

    A site whose kinds are computed at run time escapes `_static_kinds`,
    so it is accepted only when `FORWARDING_SITES` records where its
    kinds come from; a new computed site fails here until it is spelled
    out or recorded.
    """
    unread: list[str] = []
    for site in RAISE_SITES:
        argument = site.kinds_argument()
        literal = isinstance(argument, ast.List | ast.Tuple) and all(
            _kind_literal(element) is not None and not isinstance(element, ast.Starred)
            for element in argument.elts
        )
        if not literal and (site.path, site.function) not in FORWARDING_SITES:
            unread.append(f"{site.path}:{site.line} in {site.function}")
    assert not unread, (
        f"these raise sites pass kinds this test cannot read: {unread}. "
        f"Spell each kind as a literal or f-string, or record the site in "
        f"FORWARDING_SITES with where its kinds come from."
    )


def test_every_forwarding_site_still_exists() -> None:
    """A recorded forwarding site that no longer raises is dropped.

    Otherwise a stale entry would excuse whatever site later took its
    file and function name.
    """
    present = {(site.path, site.function) for site in RAISE_SITES}
    stale = sorted(set(FORWARDING_SITES) - present)
    assert not stale, f"FORWARDING_SITES names sites that raise nothing: {stale}"


@pytest.mark.parametrize("kind", _static_kinds())
def test_every_static_kind_leads_with_an_identifier(kind: str) -> None:
    """A kind's structured part is an identifier path, never prose.

    The renderer splits a kind at its first colon-space; a raise site
    that puts prose where the kind belongs hands the renderer a head it
    cannot dispatch on.
    """
    structured = kind.partition(": ")[0]
    assert " " not in structured, (
        f"the kind {kind!r} carries prose in its structured part. Pass the "
        f"prose after a colon-space, or as `explanation=` on a "
        f"StepResolutionError."
    )


@pytest.mark.parametrize(
    "kind",
    [
        "no-such-head:anything",
        "data:no-such-form:x",
        "param-source:mlp:no-such-position:f",
        "morphism:no-such-reason:f",
    ],
)
def test_an_unregistered_kind_raises_rather_than_rendering(kind: str) -> None:
    """The renderer refuses to ship a kind it has no explanation for."""
    with pytest.raises(UnregisteredRefusalKind):
        user_facing_message("qvr-stan", (kind,))


def test_a_step_resolution_kind_must_not_carry_prose() -> None:
    """Prose passed where a kind belongs is rejected at the raise site."""
    with pytest.raises(ValueError, match="explanation="):
        StepResolutionError("qvr-transpile", ["param-source:mlp", "the weights are"])


def test_a_step_resolution_explanation_reaches_the_rendered_refusal() -> None:
    """The raise site's account follows the kind's headline, once."""
    error = StepResolutionError(
        "qvr-transpile",
        ["morphism:duplicate:f"],
        explanation="first at line 3, again at line 7.",
    )
    assert error.kinds == ["morphism:duplicate:f"]
    assert error.detail == "morphism:duplicate:f; first at line 3, again at line 7."
    message = user_facing_message("qvr-stan", tuple(error.explained_kinds))
    assert message == (
        "morphism `f` is declared twice: first at line 3, again at line 7."
    )


@pytest.mark.parametrize("kind", EMITTED_KINDS)
@pytest.mark.parametrize("target", TARGETS)
def test_every_explanation_frames_the_tag_as_prose(target: str, kind: str) -> None:
    """A message is prose about the tag, never the tag echoed back.

    Some kinds carry their own detail after a structured head, so a
    message may legitimately quote most of its kind; what it may not
    do is hand the raw tag back, which would leave the user exactly
    where the structured token did.
    """
    message = _render(target, kind)
    assert message.strip() != kind.strip(), (
        f"{target} renders kind {kind!r} as the tag itself, which "
        f"explains nothing the tag did not."
    )
    assert len(message.split()) >= 8, (
        f"{target} renders kind {kind!r} as {message!r}, too terse to "
        f"say what the construct is and what to write instead."
    )
