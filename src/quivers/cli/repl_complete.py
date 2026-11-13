"""Tab-completion sources for the REPL and LSP.

Completions merge three sources:

1. Env names from the active [`ReplSession`][quivers.cli.ReplSession]
   (objects, spaces, morphisms, rules).
2. Keywords pulled live from the QVR Pygments lexer's keyword and
   builtin tables, so adding a new grammar keyword automatically lights
   it up.
3. Meta-command names from the [`ReplSession`][quivers.cli.ReplSession] dispatch table.

Each completion carries an optional one-line documentation string;
prompt_toolkit and the LSP both expose this to the user.
"""

from __future__ import annotations

import glob

import didactic.api as dx

from quivers.analysis.scope import (
    SCOPE_SEPARATOR,
    resolve_scoped_path,
    scope_children,
    top_level_bindings,
)
from quivers.cli import repl_session
from quivers.cli.repl_session import ReplSession
from quivers.dsl.pygments_lexer import (
    _ALGEBRA_NAMES,
    _BUILTIN_FUNCTION_TOKENS,
    _BUILTIN_TYPE_TOKENS,
    _KEYWORD_TOKENS,
)

from quivers.dsl.qiec_tooling import (
    effect_operation_names,
    instance_effect_name,
    qiec_binding_map,
)
from quivers.transpile import available_targets


class Completion(dx.Model):
    """One immutable completion candidate.

    Parameters
    ----------
    text : str
        Insertion text.
    kind : str
        Candidate category, such as an environment name or command.
    detail : str
        Short description shown with the candidate.
    """

    text: str
    kind: str  # "env", "keyword", "type", "function", "namespace", "command", "path"
    detail: str = ""


def public_meta_commands() -> tuple[str, ...]:
    """Return one public spelling for every registered meta-command.

    The dispatch table registers each long spelling before its short aliases.
    Keeping the first name for each handler thus makes completion and the TUI
    palette follow the executable command surface without advertising aliases.

    Returns
    -------
    tuple[str, ...]
        The long spelling of each meta-command, in registration order.
    """
    seen: set[object] = set()
    commands: list[str] = []
    for name, handler in repl_session._META_COMMANDS.items():  # noqa: SLF001
        if handler in seen:
            continue
        seen.add(handler)
        commands.append(name)
    return tuple(commands)


def all_completions(session: ReplSession, prefix: str) -> list[Completion]:
    """Return every candidate whose text starts with ``prefix``.

    The caller (prompt_toolkit Completer, LSP completion handler)
    decides which slice to show.

    Parameters
    ----------
    session : ReplSession
        The session whose environment supplies name completions.
    prefix : str
        The text typed so far.

    Returns
    -------
    list[Completion]
        Meta-command, transpile-target, environment, keyword, and path
        candidates, in that order.
    """
    out: list[Completion] = []
    out.extend(_meta_completions(prefix))
    out.extend(_transpile_target_completions(prefix))
    out.extend(_env_completions(session, prefix))
    out.extend(_keyword_completions(prefix))
    out.extend(_path_completions(prefix))
    return out


def _transpile_target_completions(prefix: str) -> list[Completion]:
    """Complete `:transpile <TAB>` over the registered backend names.

    Activates only when the prefix starts with ``:transpile `` (note
    the trailing space): until the user types the space, the
    :py:func:`_meta_completions` table already completes the command
    itself. After the space, each remaining keystroke filters the
    [`quivers.transpile.available_targets`][quivers.transpile.available_targets]
    list.
    """
    head = ":transpile "
    if not prefix.startswith(head):
        return []
    typed = prefix[len(head) :]
    out: list[Completion] = []
    for target in available_targets():
        if target.startswith(typed):
            out.append(
                Completion(
                    text=head + target,
                    kind="command",
                    detail=f"transpile target: {target}",
                )
            )
    return out


def _meta_completions(prefix: str) -> list[Completion]:
    out: list[Completion] = []
    if prefix.startswith(":"):
        p = prefix[1:]
        for name in public_meta_commands():
            if name.startswith(p):
                out.append(
                    Completion(text=":" + name, kind="command", detail="meta-command")
                )
    return out


def _env_completions(session: ReplSession, prefix: str) -> list[Completion]:
    """Complete bindings from every populated env bucket.

    Two modes:

    * **Bare-name prefix**: ``ld`` -> ``lda`` (program), with detail
      naming the container kind. Also surfaces deep paths whose
      *final* segment matches the prefix, so typing ``thet`` suggests
      ``lda::theta``.
    * **Scope-path prefix**: ``lda::`` lists every child of the
      ``lda`` binding's scope; ``lda::z::`` walks one level deeper.
      Each candidate's text is the full ``::``-path so accepting it
      keeps the path complete.
    """
    out: list[Completion] = []

    # QIEC declarations do not live in the categorical Compiler registries. Expose
    # their top-level names and signature members through the same stream.
    for name, binding in qiec_binding_map(session.module).items():
        if name.startswith(prefix):
            out.append(
                Completion(text=name, kind=binding.semantic_kind, detail=binding.kind)
            )

    # ``perform`` addresses operations through a lexical instance, whose
    # interface is the module's own or the prelude's.
    if "." in prefix:
        instance_name, _, operation_prefix = prefix.partition(".")
        effect_name = instance_effect_name(session.module, instance_name)
        if effect_name is not None:
            out.extend(
                Completion(
                    f"{instance_name}.{operation}",
                    "function",
                    f"operation of {effect_name}",
                )
                for operation in effect_operation_names(session.module, effect_name)
                if operation.startswith(operation_prefix)
            )
        return out

    compiler = session._compiler  # noqa: SLF001
    if compiler is None:
        return out

    # Mode B: scope-path completion. The prefix is ``a::b::`` (or
    # ``a::b::c``); enumerate the children of ``a::b``'s scope
    # whose name starts with the trailing segment.
    if SCOPE_SEPARATOR in prefix:
        head, _, tail = prefix.rpartition(SCOPE_SEPARATOR)
        if head:
            parent = resolve_scoped_path(compiler, head)
            if parent is not None:
                for child_name, child_ref in scope_children(parent).items():
                    if child_name.startswith(tail):
                        out.append(
                            Completion(
                                text=child_ref.path,
                                kind="env",
                                detail=child_ref.kind,
                            )
                        )
        return out

    # Mode A: bare-name completion. Surface top-level bindings whose
    # name starts with the prefix, *and* any deep path whose final
    # segment matches (so users find scoped bindings without
    # knowing the prefix).
    tops = top_level_bindings(compiler)
    for top in tops:
        if top.name.startswith(prefix):
            out.append(Completion(text=top.name, kind="env", detail=top.kind))

    # Surface scope paths whose final segment matches the prefix.
    # Skips duplicates of bare-name matches.
    seen = {c.text for c in out}
    if prefix:
        for top in tops:
            _walk_scope_for_prefix(top, prefix, out, seen)
    return out


def _walk_scope_for_prefix(  # type: ignore[no-untyped-def]
    ref, prefix: str, out: list, seen: set
) -> None:
    """Walk ``ref``'s scope subtree; emit a completion for every
    descendant whose final-segment name starts with ``prefix``."""
    for child_name, child_ref in scope_children(ref).items():
        if child_name.startswith(prefix) and child_ref.path not in seen:
            out.append(
                Completion(text=child_ref.path, kind="env", detail=child_ref.kind)
            )
            seen.add(child_ref.path)
        _walk_scope_for_prefix(child_ref, prefix, out, seen)


def _keyword_completions(prefix: str) -> list[Completion]:
    if not prefix:
        return []
    out: list[Completion] = []
    for kw in sorted(_KEYWORD_TOKENS):
        if kw.startswith(prefix):
            out.append(Completion(text=kw, kind="keyword", detail="keyword"))
    for fn in sorted(_BUILTIN_FUNCTION_TOKENS):
        if fn.startswith(prefix):
            out.append(Completion(text=fn, kind="function", detail="builtin"))
    for ty in sorted(_BUILTIN_TYPE_TOKENS):
        if ty.startswith(prefix):
            out.append(Completion(text=ty, kind="type", detail="builtin type"))
    for ns in sorted(_ALGEBRA_NAMES):
        if ns.startswith(prefix):
            out.append(Completion(text=ns, kind="namespace", detail="algebra"))
    return out


def _path_completions(prefix: str) -> list[Completion]:
    """File-path completions for :load."""
    if "/" not in prefix and not prefix.endswith(".qvr"):
        return []
    out: list[Completion] = []
    for match in sorted(glob.glob(prefix + "*")):
        out.append(Completion(text=match, kind="path", detail="path"))
    return out


__all__ = ["Completion", "all_completions", "public_meta_commands"]
