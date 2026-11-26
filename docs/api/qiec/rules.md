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

## The reference interpretation

[`ReferenceInterpretation`][quivers.qiec.rules.ReferenceInterpretation] grounds every rule against
every derived item, lists every derivation of a goal, and computes an observable by its definition,
$\sum_d w(d)\, h(\mathrm{fold}(d))$. It is an oracle for small systems. Its cost is exponential in
general, and executors are planned from the presentation's analysis instead.

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
