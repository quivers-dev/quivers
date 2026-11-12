# Deductive Systems, Rule Schemas, and Span Components

A [chart parser](../../examples/schema-chart-parser.md) stores partial conclusions and combines them
with [inference rules](../../semantics/schemas.md#6-schema-declarations) [@shieber1995principles; @goodman1999semiring]. In a categorial grammar, for instance, an expression that needs a noun phrase
on its right combines with a noun phrase to give a sentence [@lambek1958mathematics]. The
interfaces below separate the initial facts, the inference steps, the goals, and the schedule, and
the semiring fixes what is accumulated across derivations. The module holds the abstract deductive
system, the rule schemas with their bundles and presets, and the components of CKY parsing over
spans.

## Deductive Systems

Five abstract primitives for weighted deductive parsing, of which the parsers in this package are
instances:

- **`Axiom`:** Creates and populates the initial chart from input.
- **`Deduction`:** A weighted inference step (chart → chart).
- **`Goal`:** Extracts the result from a completed chart.
- **`Schedule`:** Evaluation strategy (CKY bottom-up, Earley, agenda-based).
- **`DeductiveSystem`:** Ties axiom, deductions, goal, schedule, and semiring into a single `nn.Module`.

The [deduction package](deduction.md) exposes these [primitives](deduction/primitives.md)
alongside fitting and sampling operations.

## Rule Schemas

A rule schema is a function from a finite inventory of syntactic categories (`CategorySystem`) to a
set of rules (`RuleSystem`). Each schema states one structural rule pattern, and its rules are the
instances of that pattern over the inventory.

Schemas compose via `|` (union) and can carry default weights via `.weighted(w)`. Grammar formalisms are specific compositions:

- `CCG = EVALUATION | HARMONIC_COMPOSITION | CROSSED_COMPOSITION`
- `LAMBEK = EVALUATION | ADJUNCTION_UNITS | TENSOR_INTRODUCTION | TENSOR_PROJECTION`
- `NL = EVALUATION`
- `LP = EVALUATION | COMMUTATIVE_EVALUATION | ADJUNCTION_UNITS`

About twenty atomic schemas, such as `ForwardApplication` and `BackwardApplication`, compose into
twelve bundled schemas, such as `EVALUATION` and `HARMONIC_COMPOSITION`, which compose into the
grammar presets `CCG`, `LAMBEK`, `NL`, and `LP`.

::: quivers.stochastic.schema
    options:
      filters: ["!^_", "!^[A-Z][A-Z_]+$"]

## Bundled schemas

::: quivers.stochastic
    options:
      members:
        - EVALUATION
        - HARMONIC_COMPOSITION
        - CROSSED_COMPOSITION
        - COMMUTATIVE_EVALUATION
        - ADJUNCTION_UNITS
        - TENSOR_INTRODUCTION
        - TENSOR_PROJECTION
        - UNIT_INTRODUCTION
        - UNIT_ELIMINATION
        - MODAL_INTRODUCTION
        - MODAL_ELIMINATION
        - MODAL_APPLICATION
      show_root_heading: false
      show_root_toc_entry: false

## Grammar presets and the schema registry

::: quivers.stochastic
    options:
      members:
        - CCG
        - LAMBEK
        - NL
        - LP
        - SCHEMA_REGISTRY
      show_root_heading: false
      show_root_toc_entry: false

## CKY Components over Spans {#span-based-cky-components}

Instantiation over spans of the abstract deductive framework for CKY chart parsing:

- **`LexicalAxiom`:** Populates length-1 spans from a learnable lexicon.
- **`BinarySpanDeduction`:** Applies binary structural rules with learnable weights.
- **`UnarySpanDeduction`:** Applies unary rules to convergence.
- **`SpanGoal`:** Extracts the score of the start symbol over the full span.
- **`CKYSchedule`:** Bottom-up evaluation by span length.

::: quivers.stochastic.span
