# Foreign handler semantics

A foreign handler's clauses run in a runtime provider. Its declaration states what they compute in
its [`foreign`][quivers.qiec.effects.HandlerDef] field, and the core provider builds the handler
from that statement and from nothing else, in particular never from the handler's name.

- [`SemiringSum`][quivers.qiec.foreign.SemiringSum]: a search, resuming once per alternative and
  answering the semiring sum of its completed shots' weights with their number. The number is the
  counting semiring's sum of ones, a structural count that no weight changes.
- [`SemiringProduct`][quivers.qiec.foreign.SemiringProduct]: a collector, multiplying the weights a
  scope adds, in order.
- [`Enumeration`][quivers.qiec.foreign.Enumeration]: an enumeration of a finite draw.
- [`Settlement`][quivers.qiec.foreign.Settlement]: the stop of an iteration, answered with its value
  only when the stop certifies it.
- [`DeductionOutcomes`][quivers.qiec.foreign.DeductionOutcomes] folds a finite packed lexical
  chart in `LogProb` and groups full-span goal masses by a structural key.
- [`DeductionProofs`][quivers.qiec.foreign.DeductionProofs] uses the same chart and returns
  the supported full-span proofs after an exact budget check. Each row binds its proof's indices
  and carries its `LogProb` weight.
- [`DeductionOutcomeGrouping`][quivers.qiec.foreign.DeductionOutcomeGrouping] groups weighted rows
  whose checked keys admit exact structural equality. Ordinary QIEC calls compute the keys before
  this handler runs.

A [`KernelSemiring`][quivers.qiec.foreign.KernelSemiring] names a member of the
[algebra hierarchy](../core/algebras.md) by the subject of its [law claims](contracts.md#law-claims)
and gives the kernel type of its elements. A search requires its semiring to establish
[`SUM_LAWS`][quivers.qiec.foreign.SUM_LAWS] and a collector
[`PRODUCT_LAWS`][quivers.qiec.foreign.PRODUCT_LAWS], and the provider raises
[`MissingLaw`][quivers.qiec.foreign.MissingLaw] for a member that does not. The operations convert
kernel values to `float64` tensors, apply the member's own `join` and `tensor_op`, and convert
back. The conversion is exact or refused: a count is exact below
[`EXACT_COUNT`][quivers.qiec.foreign.EXACT_COUNT], the bound at which the counting algebra also
refuses its closure.

## Finite log expectations

`Enumeration(reduction="log_expectation")` serves an ungrouped
`Random` handler with signature `LogWeight -> LogWeight`. Each
request resumes its continuation once for each positive-weight value
of the unbatched finite draw. If the draw has mass $p(z)$ and the
continuation returns $\ell(z)$ for the value $z$, its answer is

$$
\operatorname{logsumexp}_{z : p(z)>0}
\bigl(\log p(z) + \ell(z)\bigr).
$$

A later request inside the continuation returns the same answer type.
Induction on the finite tree of draws therefore gives the log of the
sum over complete assignments, with each conditional draw mass
appearing once along its branch. Work scales with that tree. An
execution budget can cap it through the usual reference-machine fuel.
The tensor backend preserves score and parameter derivatives on
fixed positive support. A subtree whose scores all have zero weight
returns $-\infty$ through the masked LogProb sum.

## Settled stops

A deduction whose derivations may repeat an item iterates. A stop is certified when the iteration
found no further derivation, so the value sums every derivation, or when the arithmetic is exact and
two iterates agree, since a fixed point reached from below is the least one. Any other stop is by a
control: the declared `tolerance`, or two iterates equal in rounded arithmetic while derivations
keep growing. Its [report][quivers.qiec.builtins.settlement_report] names a
[`ControlReached`][quivers.qiec.contracts.ControlReached] stop, guarantees nothing, and so fails the
guarantee of the [contract][quivers.qiec.builtins.settlement_contract]. The settlement refuses it
with [`ContractNotMet`][quivers.qiec.contracts.ContractNotMet], or, when the caller asks for
[`RETURN_UNESTABLISHED`][quivers.qiec.contracts.FailurePolicy], records the report and
[`run_deduction`][quivers.qiec.program_runtime.run_deduction] returns the value as
[`Unestablished`][quivers.qiec.contracts.Unestablished]. A program that calls a deduction can hold
no unestablished value, so there the stop is refused.

::: quivers.qiec.foreign
