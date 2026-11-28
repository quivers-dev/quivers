# Rule presentations

A [`RulePresentation`][quivers.qiec.rules.RulePresentation] states a deduction system as QIEC
declarations. A judgment form ([`JudgmentDecl`][quivers.qiec.rules.JudgmentDecl]) is an indexed
family, and a judgment ([`Judgment`][quivers.qiec.rules.Judgment]) is an application of the family
to index arguments. A [`Rule`][quivers.qiec.rules.Rule] is a constructor of its conclusion's family:
its telescope holds its typed schematic variables, its fields are its premise judgments in order
followed by the values it carries, and it may have a side condition, which is a checked QIEC
computation, and a weight parameter. An [`AxiomSlot`][quivers.qiec.rules.AxiomSlot] is a rule with
no premises that is filled at run time, and every input is an
[`AxiomOccurrence`][quivers.qiec.rules.AxiomOccurrence] with its own identity, so two equal inputs
are two axioms.

Proofs are therefore an [inductive family](https://en.wikipedia.org/wiki/Intuitionistic_type_theory#Inductive_types),
and the checker that checks every QIEC declaration checks the presentation:
[`declarations`][quivers.qiec.rules.RulePresentation.declarations] lowers it to families and
constructors, and construction runs [`validate_module`][quivers.qiec.module.validate_module] on the
result. The presentation adds only what the checker cannot see: every variable of a conclusion and
of a side condition is bound by the premises, and every premise pattern can be matched from left to
right without search.

## Semantics

For input occurrences $A_j$ the derivations of a judgment $j$ are the
[initial algebra](https://en.wikipedia.org/wiki/Initial_algebra) of the
[polynomial functor](https://ncatlab.org/nlab/show/polynomial+functor)

$$
(F_A X)_j = A_j + \coprod_{r : (j_1, \dots, j_n) \to j} \prod_{k=1}^{n} X_{j_k},
$$

and a [`HostInterpretation`][quivers.qiec.rules.HostInterpretation], which gives an operation for
each rule and slot, folds a [`Derivation`][quivers.qiec.rules.Derivation] by the unique map out of
it. A derivation is a proof of an [`Item`][quivers.qiec.rules.Item], not the item: one item can have
many derivations, and a nonlinear function of an aggregate over an item's derivations is not in
general a function of the derivations.

A host interpretation is an algebra of the signature functor $F_A$: a carrier per judgment and an
operation per rule. It is neither a weight algebra of [`quivers.core`](../core/algebras.md) (a semiring of
weights, with laws) nor an Eilenberg-Moore algebra of a monad in [`quivers.monadic`](../monadic/algebras.md);
a weight algebra interprets the rules only through the weights of their applications.

## The reference interpretation

[`ReferenceInterpretation`][quivers.qiec.rules.ReferenceInterpretation] grounds every rule against
every derived item, lists every derivation of a goal, and computes an observable by its definition,
$\sum_d w(d)\, h(\mathrm{fold}(d))$. It is an oracle for small systems. Its cost is exponential in
general, and executors are planned from the presentation's analysis instead.

[`closed_derivation_value`][quivers.qiec.rules.closed_derivation_value] makes
an ordinary QIEC constructor value from a closed derivation of an unconditional
presentation without proof equations. The value has the proved judgment's
indexed family type, so a checked computation can take it as an argument. The
caller retains the derivation's occurrence identities and weights separately.
For instance, an expectation of a nonlinear proof encoding applies the
encoding to each proof in the declared support before taking the weighted sum.

## Analysis

[`analysis`][quivers.qiec.rules.RulePresentation.analysis] holds what a plan needs: sizes, the
strongly connected components of the dependency graph of judgment forms, whether each cyclic
component is linear, and the variables each rule's join is on.
[`recursion`][quivers.qiec.rules.PresentationAnalysis.recursion] reads the weakest
[recursion contract](contracts.md#recursion) the family graph guarantees. The family graph
over-approximates the derived hypergraph: span rules are cyclic as families and acyclic as items,
and detecting that needs a measure on the indices, which the analysis does not yet compute.

## Structural signatures

[`presentation_of`][quivers.qiec.rules.presentation_of] gives the presentation of a structural
[`Signature`][quivers.structural.Signature]: each object sort is a judgment form, a constructor is a
rule with no side condition whose object arguments are premises and whose data and index arguments
are values, and a binder is a rule whose scoped arguments are premises with the bound variables as
hypotheses. A signature builds each of its name-keyed tables once, so folding a term costs the size
of the term.

## Deductions and rule schemata

A `deduction` declaration is read once into its presentation, which the compiled
[`DeductionSystem`][quivers.stochastic.agenda.DeductionSystem] keeps as `presentation`. A slot
that holds a chart position is a natural number, and every other argument is a term of one closed
sort, `Term`, whose constructors are the atoms and applied symbols the declaration writes. A
deduction `T -> T` derives terms, so every item whose head holds no position, such as `X/Y` or an
atom, is an instance of one judgment form whose argument is the term, and a variable that is an
item is the same variable wherever it occurs inside a term. An item whose head holds positions,
such as a lexicon's `span(i, j, c)`, is a judgment form of its own. Each sequent is a rule, and
each kind of input is an axiom slot. The
agenda's [`InferenceRule`][quivers.stochastic.agenda.InferenceRule] patterns and the reference
machine's computations are both derived from these rules, so a pattern is read from the source in
one place. A learnable rule's weight is a parameter named by the deduction and the rule.
Judgments partition by form, so a premise or conclusion that is a bare variable, and does not
occur inside a term, ranges over the term form and every form that holds positions: the rule is the
family of its instances, one per choice of a form for each such variable, which becomes the term
itself or the form's head applied to fresh variables (positions where the form has them). One
variable takes one form wherever it occurs, and every instance keeps the rule's weight parameter.
A bare numeral is likewise a term or the position of a form of a single position slot.
[`schema_presentation`][quivers.stochastic.schema.schema_presentation] gives the presentation of a
rule schema instantiated over a category system: one `span` judgment, and one rule per ground
binary or unary instance.

## Contexts and open derivations

An [`OpenDerivation`][quivers.qiec.rules.OpenDerivation] derives an item from a context of
hypotheses. Its [`Hole`][quivers.qiec.rules.Hole]s refer to the context by de Bruijn index,
counted from the last (innermost) hypothesis. A [`Premise`][quivers.qiec.rules.Premise] declares
where its hypotheses go: its `before` block at the left of the enclosing context and its `after`
block at the right, so it is derived in `before + context + after`
([`extend_context`][quivers.qiec.rules.extend_context] is the one place this is decided). In an
ordered context the side matters. Open derivations are the arrows of the free colored nonsymmetric
[multicategory](https://ncatlab.org/nlab/show/multicategory) of the presentation:
[`identity_derivation`][quivers.qiec.rules.identity_derivation] is the identity, and
[`compose_derivations`][quivers.qiec.rules.compose_derivations] plugs a derivation into each
hypothesis in order and concatenates their contexts. Composition is associative and unital, and
an interpretation is a functor out of the multicategory: folding a composite equals folding the
outer derivation at the folds of its arguments. Both laws are checked.

A [`ContextStructure`][quivers.qiec.rules.ContextStructure] declares which structural rules a
presentation's contexts admit: exchange (uses out of order), weakening (a hypothesis unused) and
contraction (one used twice). [`ORDERED`][quivers.qiec.rules.ORDERED] admits none and is the
default, so nothing is inferred from how a context is stored;
[`LINEAR`][quivers.qiec.rules.LINEAR], [`AFFINE`][quivers.qiec.rules.AFFINE],
[`RELEVANT`][quivers.qiec.rules.RELEVANT] and [`CARTESIAN`][quivers.qiec.rules.CARTESIAN] admit
more. [`required_structure`][quivers.qiec.rules.required_structure] reads which rules a derivation
uses, scope by scope, and [`check_structure`][quivers.qiec.rules.check_structure] refuses a use the
presentation does not admit, such as a permutation in an ordered context.
[`rename_hypotheses`][quivers.qiec.rules.rename_hypotheses] moves hypotheses to new positions,
which is how exchange, weakening and contraction are written, and
[`substitute_hypotheses`][quivers.qiec.rules.substitute_hypotheses] replaces every hypothesis by a
derivation in one shared context.

The Lambek calculus is the example. Its slash eliminations are ordinary rules, and its
introductions are premises with one hypothesis: on the right for `a / b`, on the left for
`b \ a`. With [`ORDERED`][quivers.qiec.rules.ORDERED] structure exactly the planar derivations
are admitted: type raising `np |- s / (np \ s)` and `np |- (s / np) \ s` and composition
`s / np, np / n |- s / n` pass the QIEC checker and the structure check, and an independent
cut-free sequent search proves their sequents. `np |- s / (s / np)` needs its hypotheses out of
order; it is refused in an ordered context, admitted in a linear one, and provable only with
permutation. Swapping a left introduction for a right one is refused by the checker, so the two
slashes are not interchangeable.

An interpretation is context indexed: a premise with hypotheses reaches its rule's operation as a
[`Scoped`][quivers.qiec.rules.Scoped] function of the hypotheses' values (the left block, then the
right block), so a binder decides what
its variables stand for, and [`fold_open`][quivers.qiec.rules.HostInterpretation.fold_open] folds an
open derivation at values of its context.

### Checked right-scoped folds

[`scoped_proof_types`][quivers.qiec.scoped_proof_types] derives checked QIEC proof and
environment types from an unconditional, unindexed Cartesian presentation. The
presentation explicitly grants exchange, weakening, and contraction. The adapter admits
right-scoped hypotheses and requires a closed semantic carrier per judgment when interpreting
a proof. A structural signature can use the existing [`presentation_of`][quivers.qiec.rules.presentation_of]
and [`term_derivation`][quivers.qiec.rules.term_derivation] adapters. Set the Cartesian
permissions on that presentation explicitly before preparing the scoped types.

An environment extends a context with a nominal frame that carries the previous environment
and the value of the newest hypothesis. A proof that refers to a hypothesis carries a checked
witness of its exact position in that environment. Induction on the witness shows that lookup
returns the newest value for the innermost reference and visits one preceding frame for each
position it passes. Each premise's context is computed by extending its parent's context in
the order the rule declares. Proof constructors and their stable identities are
shared across choices of semantic carriers. [`ScopedProofTypes.value`][quivers.qiec.ScopedProofTypes.value]
checks an open derivation and retains the original input occurrence identities.

[`scoped_proof_fold`][quivers.qiec.scoped_proof_fold] fixes the carriers and closed effect
rows for an interpretation. [`ScopedProofFold.operation_parameters`][quivers.qiec.ScopedProofFold.operation_parameters]
gives the typed arguments of each rule operation at an arbitrary environment type. Eager
premises run in written order. A scoped premise gives the operation a pair of its child proof
and captured environment. [`ScopedProofFold.applications`][quivers.qiec.ScopedProofFold.applications]
supplies the named computation that takes that pair and one value per introduced hypothesis.
Each application extends the captured environment in order and evaluates the child once.
Treating scoped arguments as functions of their hypotheses requires actions to consume them
through these applications. An action can also inspect the proof in the pair as a proof
observer, so extensional equations are an additional obligation on such actions.

[`ScopedProofFold.assemble`][quivers.qiec.ScopedProofFold.assemble] combines the operations
with the generated folds and checks the module. The resulting computations serialize and
run on the native QIEC evaluator. Applying a scoped argument repeatedly runs its effects
at the corresponding dynamic addresses. Skipping the argument skips those effects.
Applications descend into strict proof subtrees. On finite proofs they terminate when each
operation uses its supplied arguments finitely and terminates apart from these calls.
The adapter checks scope and effect types. Ordered and left-scoped presentations,
term-dependent judgments, and preservation of proof equations require a separate contract.

A `ModelSchema` exposes the selected program’s checked
[`ScopedInterpretationBinding`][quivers.qiec.ScopedInterpretationBinding] records through
`scoped_interpretations`. Reachability follows ordinary calls and calls made in installed
authored handler clauses. Each binding retains the original rule presentation, the
assignment of rules to operation identities, the root computation, the carriers, and the
named instances required by its effect rows. The binding can be serialized alongside the
executable QIEC module. Its [`validate`][quivers.qiec.ScopedInterpretationBinding.validate]
method rechecks the executable declarations before using a restored binding. The
companion has its own wire record and describes the scope and effect contract.
It carries no claims about preservation of proof equations. Its reconstruction
checks the generated environment and fold against the checked computations,
including premise order and the named effect instances.

For the scope comparison, let $\rho$ map hypothesis positions in a context
$\Gamma$ to positions of the same judgment in $\Delta$. Each renamed
hypothesis reads the same value at a valuation $v$ of $\Delta$ as the
original reads at $v\circ\rho$. Lookup follows the membership witness, and
crossing a binder extends $\rho$ by the identity on the newly introduced
positions. The same argument covers substituting a hypothesis by a reference
to another position. It extends to result equality for total pure rule
operations that consume scoped arguments through application. An operation
that inspects proof syntax may distinguish the proofs. This comparison
does not license precomputing a replacement with effects or erasing,
duplicating, or permuting its requests.

For $J$ judgment forms, the membership declarations contain $J(J+1)$ constructors, each with
$J$ uniform carrier types. Each concrete hypothesis reference at depth $d$ adds $d+1$
membership nodes. Lowering also visits all rule nodes and their ordered premise fields once,
sharing context tails in memory. Validation and serialization additionally process the
written type arguments and the nested context types. These folds consume the supplied proofs.
A deduction executor can still use a packed aggregate when its observable and algebra admit
a more efficient computation.

## Binding

Terms of a structural signature use de Bruijn indices.
[`Substitution`][quivers.structural.Substitution] is a simultaneous substitution: the innermost
variables go to replacements and the rest are shifted. [`substitute`][quivers.structural.substitute]
applies it under binders by lifting it, so no replacement is captured, and
[`then`][quivers.structural.signature.Substitution.then] composes two of them. The identity, composition and
renaming laws are checked, and substitution agrees with textbook capture-avoiding substitution on
named terms. [`term_derivation`][quivers.qiec.rules.term_derivation] reads a term as an open
derivation whose holes are its free variables, and substitution of terms corresponds to
substitution of derivations.

## Checking derivations

[`check_derivation`][quivers.qiec.rules.check_derivation] checks a derivation with the QIEC checker:
each node becomes a constructor value of its rule with the rule's telescope instantiated at the
node's binding, and each hypothesis a local of its item's type. A premise with hypotheses is
checked in the context extended by them, since QIEC has no function values; in its parent it
stands for a local of its type.

## Equations

A presentation states the [`Equation`][quivers.qiec.rules.Equation]s between open derivations it
intends an interpretation to preserve. [`check_equation`][quivers.qiec.rules.check_equation] folds
both sides at given values of the context. It returns a [tested](contracts.md#law-claims) claim,
since finitely many cases can refute preservation and cannot establish it, and raises
[`EquationFails`][quivers.qiec.rules.EquationFails] at a counterexample. A proved or decided claim
needs an argument or a decision procedure. Storing contexts as tuples licenses no equation:
exchange, weakening and contraction hold only when the structure declares them.

## Cost

[`PRESENTATION_WORK`][quivers.qiec.rules.PRESENTATION_WORK] states the bounds: construction in
time and space linear in the presentation, matching a premise in the size of its pattern, hashing an
interned item in constant time, and folding a derivation in time and space linear in its nodes. A
presentation is validated and indexed once and never again per firing, fold or refit.

## Extension points

Joint outputs (a rule with several conclusions), negative premises, and coinductive or productive
interpretations are not part of this presentation. A joint output can be stated today as a product
judgment form; negative premises need a stratification that a plan would check; a productive
interpretation needs a final coalgebra in place of the initial algebra. None is claimed, and no
greatest generality is claimed for the presentation.

::: quivers.qiec.rules
