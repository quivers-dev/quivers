"""Pygments lexer for the QVR domain-specific language.

The lexer drives on the in-tree tree-sitter parser (compiled via
`quivers.dsl._grammar_build`) so it always reflects
the authoritative grammar; there is no regex approximation. When the
shared library cannot be loaded, lexer construction raises with a
typed diagnostic so the failure is visible at the rendering site
rather than silently producing a degraded highlight.

Registers the ``qvr`` language alias so that code blocks tagged
``qvr`` in MkDocs, Sphinx, or any Pygments-based renderer get
proper syntax highlighting.

Registration via the ``[project.entry-points]`` table in
``pyproject.toml``::

    [project.entry-points."pygments.lexers"]
    qvr = "quivers.dsl.pygments_lexer:QvrLexer"
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator

import tree_sitter
from pygments.lexer import Lexer
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
    _TokenType,
)

from quivers.dsl._grammar_build import _parser_artifacts
from quivers.dsl._grammar_introspection import (
    BUILTIN_FUNCTIONS as _GRAMMAR_BUILTIN_FUNCTIONS,
    BUILTIN_TYPES as _GRAMMAR_BUILTIN_TYPES,
    KEYWORDS as _GRAMMAR_KEYWORDS,
    OPERATORS as _GRAMMAR_OPERATORS,
)
from quivers.dsl.family_schemas import DISTRIBUTION_FAMILIES
from quivers.dsl.pure_builtins import PURE_BUILTINS


# ---------------------------------------------------------------------------
# tree-sitter node-kind => Pygments token mapping
# ---------------------------------------------------------------------------
#
# The keyword / operator / builtin sets below are derived from the
# authoritative grammar JSON (see ``_grammar_introspection``) so a grammar
# edit propagates here automatically. The only hand-curated set is
# `_EFFECT_TOKENS`: effects ride as identifier values inside
# ``[effects=[...]]`` option blocks rather than as grammar literals, so
# the grammar walker has no way to surface them.

_EFFECT_TOKENS = frozenset({"Pure", "Sample", "Score", "Marginal"})

# Algebra / semiring identifiers carried by option-block values (e.g.
# ``[semiring=LogProb]``). These live in option-entry value position,
# so they don't surface as grammar literals; the registry below
# mirrors the runtime registry.
_ALGEBRA_NAMES = frozenset(
    {
        "product_fuzzy",
        "boolean",
        "lukasiewicz",
        "godel",
        "tropical",
        "LogProb",
        "Boolean",
        "Viterbi",
        "Counting",
        "ProductFuzzy",
    }
)

# Composition levels ride in ``[level=...]`` option values (e.g.
# ``composition prod [level=algebra]``), so they are likewise
# invisible to the grammar walker; the compiler validates membership.
_COMPOSITION_LEVEL_NAMES = frozenset({"algebra", "semigroupoid", "magmoid", "rule"})

_KEYWORD_TOKENS = _GRAMMAR_KEYWORDS
_BUILTIN_FUNCTION_TOKENS = frozenset(
    (*_GRAMMAR_BUILTIN_FUNCTIONS, *PURE_BUILTINS, "traverse")
)
_BUILTIN_TYPE_TOKENS = _GRAMMAR_BUILTIN_TYPES
_OPERATOR_TOKENS = _GRAMMAR_OPERATORS


def _node_kind_to_pygments_token(
    kind: str,
    text: str,
    parent_kind: str | None,
    field_name: str | None = None,
) -> _TokenType:
    """Map a tree-sitter node kind to a Pygments token type."""
    if kind == "doc_comment":
        return Comment.Doc
    if kind == "line_comment":
        return Comment.Single
    if kind == "block_comment":
        return Comment.Multiline
    if kind in {"pragma_outer", "pragma_inner"}:
        return Name.Decorator
    if kind == "integer":
        return Number.Integer
    if kind == "float":
        return Number.Float
    if kind == "signed_number":
        return Number
    if kind == "string":
        return String
    if kind == "identifier":
        # Context-sensitive tagging: identifiers inside type-like
        # productions colour as types; inside constructors as
        # builtin types/functions; the default is a variable.
        if parent_kind == "qiec_effect_request":
            if field_name == "operation":
                return Name.Function
            if field_name == "instance":
                return Name.Variable
        if parent_kind == "qiec_handler_operation_clause" and field_name == "operation":
            return Name.Function
        if parent_kind == "qiec_call_computation" and field_name == "callee":
            return Name.Builtin if text == "traverse" else Name.Function
        if parent_kind == "let_call" and field_name == "func":
            return Name.Builtin if text in PURE_BUILTINS else Name.Function
        if text in _BUILTIN_FUNCTION_TOKENS:
            return Name.Builtin
        if text in DISTRIBUTION_FAMILIES:
            return Name.Class
        if text in _BUILTIN_TYPE_TOKENS:
            return Name.Class
        if text in _EFFECT_TOKENS:
            return Keyword
        if text in _ALGEBRA_NAMES:
            return String.Symbol
        if parent_kind == "option_entry" and text in _COMPOSITION_LEVEL_NAMES:
            return Keyword
        if parent_kind in {
            "object_atom",
            "object_effect_apply",
            "discrete_constructor",
            "continuous_constructor",
            "sort_decl",
            "constructor_decl",
            "binder_decl",
            "binder_var_decl",
            "binder_arg_decl",
            "vertex_kind_decl",
            "edge_kind_decl",
            "morphism_init_family",
            "index_decl",
            "indexed_family_decl",
            "qiec_type_name",
            "qiec_type_application",
            "qiec_type_binder",
            "qiec_effect_ref",
        }:
            return Name.Class
        if parent_kind in {
            "qiec_index_constructor",
            "qiec_constructor_decl",
            "qiec_constructor_value",
            "qiec_case_branch",
            "qiec_operation_decl",
            "qiec_handler_application",
            "handler_decl",
            "computation_decl",
        }:
            return Name.Function
        if parent_kind == "effect_decl":
            return Name.Class
        if parent_kind in {"pragma_entry", "pragma_outer", "pragma_inner"}:
            return Name.Decorator
        if parent_kind == "schema_parameter":
            return Name.Variable
        if parent_kind in {"enum_set_literal"}:
            return Name.Constant
        return Name.Variable
    if kind in _OPERATOR_TOKENS:
        return Operator
    if kind in {
        "qiec_type_kind",
        "qiec_effect_kind",
        "qiec_nat_sort",
        "qiec_shape_sort",
        "qiec_context_sort",
        "qiec_user_index_sort",
    }:
        return Name.Class
    if kind == "qiec_resumption_grade" or parent_kind == "qiec_resumption_grade":
        return Number
    if kind in {
        "qiec_handler_coverage_key",
        "qiec_handler_forwards_key",
        "qiec_handler_introduces_key",
        "qiec_handler_implementation_key",
    }:
        # The opening bracket and the first option key are one lexical
        # token; it reads as the keyword it names.
        return Keyword
    if kind in {"qiec_bool_literal", "qiec_unit_literal"}:
        return Keyword.Constant
    if kind in {"(", ")", "[", "]", "{", "}", ",", ":", "."}:
        return Punctuation
    if kind in _KEYWORD_TOKENS:
        return Keyword
    # Anonymous tokens (constructors / builtin function heads) come
    # through with the literal text as their node kind.
    if kind in _BUILTIN_TYPE_TOKENS:
        return Name.Class
    if kind in _BUILTIN_FUNCTION_TOKENS:
        return Name.Builtin
    return Text


# ---------------------------------------------------------------------------
# tree-sitter parser singleton
# ---------------------------------------------------------------------------


_TS_PARSER: tuple[tree_sitter.Parser, tree_sitter.Language, ctypes.CDLL] | None = None


def _load_parser() -> tuple[tree_sitter.Parser, tree_sitter.Language, ctypes.CDLL]:
    """Load the in-tree tree-sitter parser. Raises on failure.

    The returned tuple keeps the ``CDLL`` handle alive alongside the
    parser so the language pointer stays valid for the parser's
    lifetime.
    """
    global _TS_PARSER
    if _TS_PARSER is not None:
        return _TS_PARSER

    _grammar_dir, lib_path = _parser_artifacts()
    lib = ctypes.CDLL(str(lib_path))
    lib.tree_sitter_qvr.restype = ctypes.c_void_p
    language_ptr = lib.tree_sitter_qvr()
    # tree-sitter 0.24+ prefers a PyCapsule over a raw integer.
    # Wrap the void* in a capsule so the language handle uses the
    # modern API and we don't emit a DeprecationWarning.
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    PyCapsule_New.restype = ctypes.py_object
    capsule = PyCapsule_New(language_ptr, b"tree_sitter.Language", None)
    language = tree_sitter.Language(capsule)
    parser = tree_sitter.Parser(language)
    _TS_PARSER = (parser, language, lib)
    return _TS_PARSER


# ---------------------------------------------------------------------------
# public Pygments lexer
# ---------------------------------------------------------------------------


class QvrLexer(Lexer):
    """Pygments lexer for ``.qvr`` (quivers DSL) files.

    The lexer is a thin walker over the in-tree tree-sitter parse;
    the grammar is the single source of truth. There is no regex
    approximation: when the shared library can't be loaded, the
    lexer raises a typed exception so the failure is visible at
    the rendering site rather than silently emitting a degraded
    highlight.
    """

    name = "QVR"
    aliases = ["qvr"]
    filenames = ["*.qvr"]
    mimetypes = ["text/x-qvr"]

    def get_tokens_unprocessed(
        self, text: str
    ) -> Iterator[tuple[int, _TokenType, str]]:
        """Yield ``(index, token_type, text)`` tuples for ``text``.

        Parameters
        ----------
        text : str
            The ``.qvr`` source to tokenize.

        Yields
        ------
        tuple[int, _TokenType, str]
            The character offset, Pygments token type, and text of each
            leaf of the parse and of the whitespace between leaves.

        Raises
        ------
        OSError
            If the in-tree tree-sitter parser library cannot be loaded.
        """
        parser, _, _ = _load_parser()
        src_bytes = text.encode("utf-8")
        tree = parser.parse(src_bytes)

        # Walk leaf-first in source order, emitting tokens for
        # each leaf and reproducing inter-leaf whitespace as Text.
        leaves: list[tuple[tree_sitter.Node, str | None, str | None]] = []

        def walk(
            node: tree_sitter.Node,
            parent_kind: str | None,
            field_name: str | None,
        ) -> None:
            if not node.children:
                leaves.append((node, parent_kind, field_name))
                return
            for index, child in enumerate(node.children):
                walk(child, node.type, node.field_name_for_child(index))

        walk(tree.root_node, None, None)

        cursor = 0
        for node, parent_kind, field_name in leaves:
            start = node.start_byte
            end = node.end_byte
            if start > cursor:
                gap = src_bytes[cursor:start].decode("utf-8")
                if gap:
                    yield (_byte_to_char_offset(src_bytes, cursor), Text, gap)
            node_text = src_bytes[start:end].decode("utf-8")
            token = _node_kind_to_pygments_token(
                node.type,
                node_text,
                parent_kind,
                field_name,
            )
            yield (_byte_to_char_offset(src_bytes, start), token, node_text)
            cursor = end
        if cursor < len(src_bytes):
            tail = src_bytes[cursor:].decode("utf-8")
            if tail:
                yield (_byte_to_char_offset(src_bytes, cursor), Text, tail)


def _byte_to_char_offset(source: bytes, byte_offset: int) -> int:
    """Translate a tree-sitter UTF-8 byte offset to a Python string offset."""
    return len(source[:byte_offset].decode("utf-8"))


__all__ = ["QvrLexer"]
