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
  "aff"
  "algebra"
  "arc"
  "as"
  "atoms"
  "attention"
  "authored"
  "auto_intersect"
  "axiom"
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
  "closure"
  "compose"
  "composition"
  "concatenate"
  "condition"
  "conditional_log_prob"
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
  "eps"
  "expected_path_cost"
  "export"
  "factor"
  "false"
  "family"
  "final"
  "for"
  "foreign"
  "forwards"
  "freeze"
  "from"
  "graph"
  "handle"
  "handler"
  "if"
  "implementation"
  "in"
  "init"
  "initial"
  "input"
  "instance"
  "interpret"
  "intersect"
  "introduces"
  "invert"
  "iterations"
  "join"
  "keep_shared"
  "labels"
  "lacks"
  "lambek"
  "let"
  "lex"
  "lexicon"
  "logprob"
  "loss"
  "marginalize"
  "markov"
  "max_length"
  "message"
  "morphism"
  "motive"
  "none"
  "not"
  "observe"
  "observed"
  "omega"
  "op"
  "operation"
  "ops"
  "output"
  "outputs"
  "pair_weight"
  "partial"
  "path_entropy"
  "perform"
  "primitive"
  "program"
  "project"
  "readout"
  "recurrent"
  "recursive"
  "relation"
  "relation_score"
  "resume"
  "resumes"
  "return"
  "reverse"
  "rule"
  "rules"
  "sample"
  "schema"
  "score"
  "signature"
  "sorts"
  "start"
  "states"
  "string_graph"
  "string_graph_score"
  "structure"
  "terminal"
  "then"
  "total"
  "trace"
  "tree_log_prob"
  "tree_partition"
  "trim"
  "true"
  "unary"
  "union"
  "unknown"
  "update"
  "var_init"
  "variable"
  "vertex_kinds"
  "weight"
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
    "AdditiveLogRatioNormal"
    "AsymmetricLaplace"
    "Bernoulli"
    "BernsteinHazard"
    "Beta"
    "BetaBinomial"
    "Binomial"
    "COMPoisson"
    "Categorical"
    "Cauchy"
    "Chi2"
    "ContinuousBernoulli"
    "Dirichlet"
    "DirichletMultinomial"
    "DiscreteWeibull"
    "ExGaussian"
    "Exponential"
    "ExtendedBeta"
    "FisherSnedecor"
    "Frechet"
    "GP"
    "Gamma"
    "GeneralizedExtremeValue"
    "GeneralizedPareto"
    "Geometric"
    "Gumbel"
    "HalfCauchy"
    "HalfNormal"
    "HalfStudentT"
    "Horseshoe"
    "HurdleGamma"
    "HurdleLogNormal"
    "HurdleNegativeBinomial"
    "HurdlePoisson"
    "Independent"
    "InverseGamma"
    "InverseGaussian"
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
    "Multinomial"
    "MultivariateNormal"
    "MultivariateStudentT"
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
    "ShiftedLogNormal"
    "SkewNormal"
    "StudentT"
    "Transformed"
    "Truncate"
    "Truncated"
    "TruncatedNormal"
    "Uniform"
    "VonMises"
    "Weibull"
    "Wiener"
    "Wishart"
    "ZeroInflatedAsymmetricLaplace"
    "ZeroInflatedBetaBinomial"
    "ZeroInflatedBinomial"
    "ZeroInflatedNegativeBinomial"
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

; Checked pure builtins are identifiers rather than grammar literals.
((identifier) @function.builtin
  (#any-of? @function.builtin
    "abs"
    "acos"
    "acosh"
    "asin"
    "asinh"
    "atan"
    "atanh"
    "ceil"
    "cholesky"
    "concat"
    "cos"
    "cosh"
    "cumsum"
    "digamma"
    "elu"
    "erf"
    "erfc"
    "erfinv"
    "exp"
    "expm1"
    "filter"
    "floor"
    "fold"
    "gelu"
    "int"
    "interval_upper"
    "layer_norm"
    "length"
    "lgamma"
    "log"
    "log10"
    "log1mexp"
    "log1p"
    "log2"
    "log_mass"
    "log_ndtr"
    "log_prob"
    "log_probability"
    "log_softmax"
    "logsigmoid"
    "logsumexp"
    "logsumexp_over"
    "map"
    "matmul"
    "max"
    "mean"
    "min"
    "mish"
    "normalize"
    "observation_status"
    "observation_weight"
    "positive_parameter"
    "pow"
    "prod"
    "real"
    "reciprocal"
    "relu"
    "relu6"
    "rms_norm"
    "round"
    "rsqrt"
    "scale_weight"
    "select"
    "selu"
    "sigmoid"
    "sign"
    "silu"
    "sin"
    "sinh"
    "slice"
    "softmax"
    "softplus"
    "softsign"
    "sort"
    "sqrt"
    "square"
    "sum"
    "tan"
    "tanh"
    "trunc"
    "weight"
    "weight_value"))

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
(relation_decl "relation" @keyword)
(relation_decl name: (identifier) @type)
(relation_compose_decl ["relation" "compose"] @keyword)
(relation_compose_decl name: (identifier) @type)
(relation_compose_decl left: (identifier) @type)
(relation_compose_decl right: (identifier) @type)
(relation_transform_decl "relation" @keyword)
(relation_transform_decl name: (identifier) @type)
(relation_unary_operation) @keyword
(relation_binary_operation) @keyword
(relation_unary_recipe source: (identifier) @type)
(relation_projection_recipe "project" @keyword source: (identifier) @type)
(relation_conditioning_recipe "condition" @keyword source: (identifier) @type)
(relation_auto_intersection_recipe "auto_intersect" @keyword source: (identifier) @type)
(relation_binary_recipe source: (identifier) @type other: (identifier) @type)
(relation_join_recipe ["join" "keep_shared"] @keyword source: (identifier) @type other: (identifier) @type)
(relation_algebra) @type.builtin
(relation_tape_direction) @keyword
(relation_tape_decl alphabet: (identifier) @type)
(relation_boundary_kind) @keyword
(relation_boundary_decl weight: (identifier) @variable.parameter)
(relation_arc_decl weight: (identifier) @variable.parameter)
(relation_label "eps" @constant.builtin)
(relation_score_decl relation: (identifier) @type)
(relation_score_decl name: (identifier) @function)
(relation_score_operation) @keyword
(string_graph_decl "string_graph" @keyword)
(string_graph_decl name: (identifier) @type)
(string_graph_variable "variable" @keyword)
(string_graph_variable name: (identifier) @variable)
(string_graph_variable alphabet: (identifier) @type)
(string_graph_factor "factor" @keyword)
(string_graph_factor relation: (identifier) @type)
(string_graph_factor scope: (identifier) @variable)
(string_graph_score_decl graph: (identifier) @type)
(string_graph_score_decl name: (identifier) @function)
(string_graph_score_operation) @keyword
(interpretation_decl name: (identifier) @function)
(interpretation_decl deduction: (identifier) @function)
(interpretation_clause constructor: (identifier) @constructor)
(interpretation_clause operation: (identifier) @function.call)
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
