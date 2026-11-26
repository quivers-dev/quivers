# Laws

A [`Law`][quivers.core.laws.Law] is a named statement about a carrier, or about the tensors an
implementation returns. Every member of the [algebra hierarchy](algebras.md) is
[`Lawful`][quivers.core.laws.Lawful]: its class lists the laws it introduces, and the member
reports a [`LawClaim`][quivers.core.laws.LawClaim] for each law it has. The claim's status is
read from its evidence: [`Proved`][quivers.core.laws.Proved] for a written argument,
[`Decided`][quivers.core.laws.Decided] for a decision procedure on a stated domain,
[`Tested`][quivers.core.laws.Tested] for finitely many cases of a larger domain, and
[`Assumed`][quivers.core.laws.Assumed] for a declaration. Proved and decided claims are
established. The arguments of the shipped classes are in
[Semirings, t-norms, and change of base §8](../../semantics/algebras.md#8-proofs-of-the-claimed-laws).

::: quivers.core.laws
    options:
      filters:
        - "!^_"
