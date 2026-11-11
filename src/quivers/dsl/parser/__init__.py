"""Parser for the quivers DSL.

The lexer/parser pipeline is delegated to panproto after overriding its `qvr`
protocol with Quivers' packaged current grammar. The public `parse` entry
point consumes `.qvr` source bytes and returns a `Module` of didactic models.

This package's submodules group the walker logic by topic:

* `._registry` for the panproto registry singleton, ``ParseError``,
  and the ``_Tree`` view that every walker reads from.
* `._helpers` for the low-level helpers ``_required_text``,
  ``_required_field``, ``_field_text``, and ``_walk_draw_arg``.
* `.options` for the ``[k=v, ...]`` option-block walkers.
* `.expressions` for type / space / morphism-expression / let-arith walkers.
* `.program_steps` for program-block step walkers.
* `.statements` for the top-level ``_walk_statement`` dispatcher and
  every per-declaration walker (object, morphism, deduction,
  contraction, signature, encoder, decoder, loss, ...).
* `.core` for the public `parse` / `parse_file` entry points, the
  whole-tree syntax validation that rejects ERROR and missing nodes, and
  the refusal of binders spelled as the reserved let words
  ``RESERVED_LET_WORDS``.

Every public name is re-exported here so ``from quivers.dsl.parser
import X`` keeps working unchanged.
"""

from quivers.dsl.parser.core import RESERVED_LET_WORDS, parse, parse_file
from quivers.dsl.parser._registry import ParseError

__all__ = ["RESERVED_LET_WORDS", "ParseError", "parse", "parse_file"]
