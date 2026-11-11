; QVR syntax highlighting queries.
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

; ---------------------------------------------------------------------------
; keywords (derived from grammar literals)
; ---------------------------------------------------------------------------

[
  "Nat"
  "aff"
  "as"
  "atoms"
  "attention"
  "authored"
  "binary"
  "binder_select"
  "binders"
  "binds"
  "body"
  "bundle"
  "case"
  "categories"
  "category"
  "ccg"
  "change_base"
  "composition"
  "construct"
  "constructor"
  "constructors"
  "contraction"
  "coverage"
  "curry_left"
  "curry_right"
  "dagger"
  "decoder"
  "deduction"
  "define"
  "depth"
  "dim"
  "edge_kinds"
  "effect"
  "effect_depth"
  "else"
  "encoder"
  "export"
  "factor"
  "false"
  "family"
  "for"
  "foreign"
  "forwards"
  "freeze"
  "from"
  "handle"
  "handler"
  "if"
  "implementation"
  "in"
  "init"
  "instance"
  "introduces"
  "iterations"
  "lacks"
  "lambek"
  "let"
  "lex"
  "lexicon"
  "loss"
  "marginalize"
  "max_length"
  "message"
  "morphism"
  "motive"
  "none"
  "not"
  "observe"
  "omega"
  "op"
  "ops"
  "partial"
  "perform"
  "primitive"
  "program"
  "readout"
  "recurrent"
  "recursive"
  "resume"
  "resumes"
  "return"
  "rule"
  "rules"
  "sample"
  "schema"
  "score"
  "signature"
  "sorts"
  "start"
  "structure"
  "terminal"
  "then"
  "total"
  "trace"
  "true"
  "unary"
  "unknown"
  "update"
  "var_init"
  "vertex_kinds"
  "where"
  "with"
] @keyword

; Sort kinds in structural-compression signatures.
[
  "data"
  "index"
  "object"
] @type.qualifier

; ---------------------------------------------------------------------------
; builtin types (constructor / param-kind heads)
; ---------------------------------------------------------------------------

[
  "Ball"
  "CholeskyFactor"
  "Correlation"
  "Covariance"
  "Diagonal"
  "FinSet"
  "LowerTriangular"
  "Mor"
  "Nat"
  "Object"
  "Orthogonal"
  "Real"
  "Simplex"
  "Space"
  "Sphere"
  "Stiefel"
] @type.builtin

; Registered distribution-family identifiers.
((identifier) @type.builtin
  (#any-of? @type.builtin
    "Bernoulli"
    "Beta"
    "BetaBinomial"
    "Binomial"
    "Categorical"
    "Cauchy"
    "Chi2"
    "ContinuousBernoulli"
    "Dirichlet"
    "Exponential"
    "FisherSnedecor"
    "GP"
    "Gamma"
    "GeneralizedPareto"
    "Geometric"
    "Gumbel"
    "HalfCauchy"
    "HalfNormal"
    "HalfStudentT"
    "Horseshoe"
    "HurdlePoisson"
    "Independent"
    "InverseGamma"
    "InverseWishart"
    "Kumaraswamy"
    "LKJCholesky"
    "LKJCorrelationFactor"
    "Laplace"
    "LogGamma"
    "LogNormal"
    "Logistic"
    "LogisticNormal"
    "LogitBeta"
    "LogitNormal"
    "LowRankMVN"
    "MatrixNormal"
    "Mixture"
    "MixtureNormal"
    "MultivariateNormal"
    "NegativeBinomial"
    "Normal"
    "Normalize"
    "OneHotCategorical"
    "OrderedLogistic"
    "OrderedProbit"
    "Pareto"
    "PointMass"
    "Poisson"
    "Pushforward"
    "RelaxedBernoulli"
    "RelaxedOneHotCategorical"
    "Restrict"
    "StudentT"
    "Transformed"
    "Truncate"
    "Truncated"
    "TruncatedNormal"
    "Uniform"
    "VonMises"
    "Weibull"
    "Wishart"
    "ZeroInflatedPoisson"
    "ZeroOneInflatedBeta"))

; ---------------------------------------------------------------------------
; builtin functions (combinators, intrinsics)
; ---------------------------------------------------------------------------

[
  "FreeMonoid"
  "FreeResiduated"
  "cap"
  "chart_fold"
  "cup"
  "fan"
  "from_data"
  "identity"
  "parser"
  "repeat"
  "scan"
  "stack"
] @function.builtin

; ---------------------------------------------------------------------------
; operators
; ---------------------------------------------------------------------------

[
  "!"
  "!="
  "%"
  "&&"
  "*"
  "+"
  "-"
  "--"
  "->"
  "."
  "/"
  ":"
  "<"
  "<-"
  "<<"
  "<="
  "="
  "=="
  "=>"
  ">"
  ">="
  ">>"
  ">>>"
  "@"
  "\\"
  "|"
  "|-"
  "|->"
  "||"
  "~"
  "⊢"
] @operator

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
