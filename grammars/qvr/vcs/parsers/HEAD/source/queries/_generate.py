"""Regenerate :file:`highlights.scm` from the current grammar.

This script reads :file:`grammars/qvr/src/grammar.json`, walks
each rule's terminal strings, and emits a deterministic
:file:`highlights.scm` consumed by every tree-sitter-driven
highlighter.

Run after any edit to :file:`grammars/qvr/grammar.js`:

.. code-block:: shell

    cd grammars/qvr
    tree-sitter generate          # produces src/grammar.json
    python queries/_generate.py   # rewrites queries/highlights.scm

The hand-curated parts (declaration node patterns binding identifier
roles like ``@type`` / ``@function``) live inline below; the literal
keyword / operator / builtin lists at the top are derived.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quivers.dsl._grammar_introspection import (  # noqa: E402
    BUILTIN_FUNCTIONS,
    BUILTIN_TYPES,
    KEYWORDS,
    OPERATORS,
    SORT_KINDS,
)
from quivers.dsl.family_schemas import DISTRIBUTION_FAMILIES  # noqa: E402


HEADER = """; QVR syntax highlighting queries.
;
; AUTO-GENERATED from grammars/qvr/src/grammar.json by
; grammars/qvr/queries/_generate.py. Do not edit by hand; rerun the
; generator after any grammar change. The literal keyword / builtin
; / operator lists below are derived from the grammar; the structural
; node-pattern rules (binding @type / @function / @variable roles to
; specific decl fields) are emitted from a fixed template inside the
; generator.
;
; Consumed by tree-sitter-driven highlighters: nvim-treesitter, Helix,
; Zed, Emacs treesit, and the in-tree Pygments lexer / REPL highlighter
; (which walk the same grammar through a shared mapping table).
"""

NODE_PATTERNS = """\
; ---------------------------------------------------------------------------
; declarations and identifiers
; ---------------------------------------------------------------------------

(object_decl "object" @keyword)
(index_decl "index" @keyword)
(category_decl    names: (identifier) @type)
(object_decl      names: (identifier) @type)
(rule_decl        name: (identifier) @function)
(rule_decl        variables: (identifier) @variable.parameter)
(schema_decl      name: (identifier) @function)
(schema_parameter names: (identifier) @variable.parameter)
(morphism_decl    names: (identifier) @function)
(define_decl      name: (identifier) @function)
(program_decl     name: (identifier) @function)
(bundle_decl      name: (identifier) @function)
(contraction_decl name: (identifier) @function)
(contraction_input name: (identifier) @function)
(composition_decl name: (identifier) @function)
(composition_rule_entry key: (identifier) @function)
(composition_rule_entry params: (identifier) @variable.parameter)
(enum_set_literal elements: (identifier) @constant)
(free_residuated_expr generators: (identifier) @type)
(free_monoid_expr generators: (identifier) @type)

; QIEC indexed families and effects.
(index_decl name: (identifier) @type)
(qiec_index_constructor name: (identifier) @constructor)
(indexed_family_decl name: (identifier) @type)
(qiec_constructor_decl name: (identifier) @constructor)
(effect_decl name: (identifier) @type)
(qiec_operation_decl name: (identifier) @function.method)
(effect_instance_decl name: (identifier) @variable)
(handler_decl name: (identifier) @function)
(qiec_handler_operation_clause operation: (identifier) @function.method)
(qiec_handler_operation_clause parameters: (qiec_local_binding name: (identifier) @variable.parameter))
(qiec_handler_return_clause binder: (qiec_local_binding name: (identifier) @variable.parameter))
(computation_decl name: (identifier) @function)
(qiec_call_computation callee: (identifier) @function.call)
(qiec_instance_computation name: (identifier) @variable)

; QIEC telescope, type, row, and term positions.
(qiec_type_binder name: (identifier) @type.parameter)
(qiec_index_binder name: (identifier) @variable.parameter)
(qiec_effect_binder name: (identifier) @type.parameter)
(qiec_type_name name: (identifier) @type)
(qiec_type_application constructor: (identifier) @type)
(qiec_effect_ref name: (identifier) @type)
(qiec_row_entry name: (identifier) @variable)
(qiec_effect_row_literal tail: (identifier) @variable)
(qiec_effect_row_literal lacks: (identifier) @variable)
(qiec_value_parameter name: (identifier) @variable.parameter)
(qiec_local_binding name: (identifier) @variable)
(qiec_effect_request instance: (identifier) @variable)
(qiec_effect_request operation: (identifier) @function.method)
(qiec_handler_application name: (identifier) @function)
(qiec_case_branch constructor: (identifier) @constructor)
(qiec_case_static_binder name: (identifier) @variable.parameter)
(qiec_constructor_value constructor: (identifier) @constructor)

; Handler option openers are lexically fused with ``[`` to keep them
; disjoint from static type applications.
(qiec_handler_coverage_key) @keyword
(qiec_handler_forwards_key) @keyword
(qiec_handler_introduces_key) @keyword
(qiec_type_kind) @type.builtin
(qiec_effect_kind) @type.builtin
(qiec_nat_sort) @type.builtin
(qiec_shape_sort) @type.builtin
(qiec_context_sort) @type.builtin
(qiec_resumption_grade) @constant.builtin

; Constructor heads on object expressions.
(discrete_constructor constructor: _ @type.builtin)
(continuous_constructor constructor: _ @type.builtin)

; Object atoms in expression position.
(object_atom (identifier) @type)
(object_effect_apply effect: (identifier) @type)

; Latent / kernel morphism families (initializer).
(morphism_init_family family: (identifier) @type)

; Deduction blocks.
(deduction_decl   name: (identifier) @function)
(deduction_atoms  atoms: (identifier) @constant)
(deduction_rule   name: (identifier) @function)
(deduction_lexicon_from_file path: (string) @string)
(lexicon_entry    words: (string) @string)

; Structural-compression declarations.
(signature_decl   name: (identifier) @type)
(signature_decl   params: (identifier) @type.parameter)
(sort_decl        name: (identifier) @type)
(constructor_decl name: (identifier) @constructor)
(constructor_decl domain: (identifier) @type)
(constructor_decl codomain: (identifier) @type)
(binder_decl      name: (identifier) @constructor)
(binder_decl      codomain: (identifier) @type)
(binder_var_decl  var: (identifier) @variable.parameter)
(binder_var_decl  sort: (identifier) @type)
(binder_arg_decl  arg: (identifier) @variable.parameter)
(binder_arg_decl  sort: (identifier) @type)
(vertex_kind_decl name: (identifier) @type)
(edge_kind_decl   name: (identifier) @type)
(edge_kind_decl   src: (identifier) @type)
(edge_kind_decl   tgt: (identifier) @type)
(encoder_decl     name: (identifier) @function)
(encoder_decl     signature: (identifier) @type)
(encoder_op_rule  op: (identifier) @function)
(decoder_decl     name: (identifier) @function)
(decoder_decl     signature: (identifier) @type)
(loss_decl        name: (identifier) @function)

; Pragmas.
(pragma_outer) @attribute
(pragma_inner) @attribute
(pragma_entry key: (identifier) @attribute)

; Identifier roles in expressions.
(expr_ident (identifier) @variable)
(let_var    (identifier) @variable)
(let_call   func: (identifier) @function.builtin)
(let_bool)  @boolean
(let_unit)  @constant.builtin

; Sort-kind tokens highlight as type qualifiers.
(sort_kind) @type.qualifier

; ---------------------------------------------------------------------------
; literals
; ---------------------------------------------------------------------------

(integer)       @number
(float)         @number
(signed_number) @number
(string)        @string
(line_comment)  @comment
(block_comment) @comment.block
(doc_comment)   @comment.documentation
"""


def _quote_list(items: list[str]) -> str:
    """Format a sorted Scheme string-array for a ``@keyword`` capture."""
    rendered = "\n".join(
        '  "{}"'.format(item.replace("\\", "\\\\").replace('"', '\\"'))
        for item in sorted(items)
    )
    return f"[\n{rendered}\n]"


def _identifier_predicate(capture: str, items: set[str]) -> str:
    """Match named identifiers against a generated finite vocabulary."""
    rendered = "\n".join(
        '    "{}"'.format(item.replace("\\", "\\\\").replace('"', '\\"'))
        for item in sorted(items)
    )
    return f"((identifier) @{capture}\n  (#any-of? @{capture}\n{rendered}))"


def _queryable_literals(items: set[str]) -> list[str]:
    """Keep literals Tree-sitter exposes as anonymous queryable nodes.

    A grammar literal wrapped by a named rule (for instance ``Type`` inside
    ``qiec_type_kind``) is present in grammar.json but intentionally absent
    from node-types.json. Its named wrapper receives a structural capture
    below; emitting the hidden literal in a query would make the query fail
    to compile.
    """
    node_types = json.loads(
        (REPO_ROOT / "grammars/qvr/src/node-types.json").read_text(encoding="utf-8")
    )
    anonymous = {
        node["type"]
        for node in node_types
        if isinstance(node, dict) and node.get("named") is False
    }
    return sorted(items & anonymous)


def render() -> str:
    keyword_lines = [
        "; ---------------------------------------------------------------------------",
        "; keywords (derived from grammar literals)",
        "; ---------------------------------------------------------------------------",
        "",
        _quote_list(_queryable_literals(KEYWORDS - SORT_KINDS)) + " @keyword",
        "",
        "; Sort kinds in structural-compression signatures.",
        _quote_list(_queryable_literals(SORT_KINDS)) + " @type.qualifier",
        "",
        "; ---------------------------------------------------------------------------",
        "; builtin types (constructor / param-kind heads)",
        "; ---------------------------------------------------------------------------",
        "",
        _quote_list(_queryable_literals(BUILTIN_TYPES)) + " @type.builtin",
        "",
        "; Registered distribution-family identifiers.",
        _identifier_predicate("type.builtin", set(DISTRIBUTION_FAMILIES)),
        "",
        "; ---------------------------------------------------------------------------",
        "; builtin functions (combinators, intrinsics)",
        "; ---------------------------------------------------------------------------",
        "",
        _quote_list(_queryable_literals(BUILTIN_FUNCTIONS)) + " @function.builtin",
        "",
        "; ---------------------------------------------------------------------------",
        "; operators",
        "; ---------------------------------------------------------------------------",
        "",
        _quote_list(_queryable_literals(OPERATORS)) + " @operator",
    ]
    return "\n".join([HEADER, *keyword_lines, "", NODE_PATTERNS])


def main() -> int:
    out = THIS_DIR / "highlights.scm"
    out.write_text(render(), encoding="utf-8")
    mirror = (
        REPO_ROOT
        / "editors"
        / "zed-extension-qvr"
        / "languages"
        / "qvr"
        / "highlights.scm"
    )
    if mirror.is_file():
        mirror.write_text(render(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
