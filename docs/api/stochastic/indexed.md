# Indexed charts

An [`IndexedChart`][quivers.stochastic.indexed.IndexedChart] holds the items of a
[`RulePresentation`][quivers.qiec.rules.RulePresentation] and finds, when an item arrives, the
rule instances it completes. A rule whose patterns are relations of their own variables
([`generic_joinable`][quivers.stochastic.indexed.generic_joinable]: no pattern is a sum of two
variables) is joined by [Generic Join](joins.md) from each trigger; any other rule, or every rule
of a chart made with `join="keyed"`, looks its premises up one at a time.

## Generic Join from a trigger

For a rule and a trigger, the premise an arriving item fills, or the conclusion for the backward
pass, [`join_order`][quivers.stochastic.indexed.join_order] puts the trigger's variables first and
the others in the order they first occur. Each other premise has a trie of the items of its form
that match its pattern, keyed by its variables in that order and kept up to date as items arrive.
Matching the trigger binds its variables; each other premise's trie is entered along the bound
prefix of its variables; the remaining variables are bound one at a time, each to the values in
the intersection of the tries that contain it, iterating the smallest and probing the others. A
complete binding determines every premise's item, since the item's arguments are its pattern
evaluated at the binding.

**Lemma (each instance once).** Admit items one at a time, and let each admitted item $x$ trigger
every premise position $t$ whose form it has, enumerating the instances in which $x$ fills $t$,
every premise is admitted, and no premise before $t$ is $x$. Then each instance is enumerated
exactly once.

*Proof.* An instance is a rule with premise items $p_1, \dots, p_n$; its binding determines the
items, and the items determine the binding, since every variable of the rule occurs in a premise.
Generic Join from $t$ with $x$ fixed at $t$ reaches each binding of the other variables consistent
with the admitted items once, by the values of its variables in order, so it yields each choice of
other premise items once. Let $x$ be the last of $p_1, \dots, p_n$ to be admitted and $t$ the least
position with $p_t = x$. Before $x$ is admitted some premise is not, so the instance is not
enumerated. When $x$ is admitted, the join from $t$ yields it, and the leaf test passes since no
premise before $t$ is $x$; the join from a later position $u$ with $p_u = x$ yields it and the test
rejects it, since $p_t = x$ comes before $u$. Items admitted later do not complete it. QED

**Theorem (cost).** Over a run, the lookups and candidates of one rule with $m$ premises and $n$
variables are $O(m N + m^2 n\, \mathrm{AGM})$, where $\mathrm{AGM}$ is the bound of a fractional
edge cover of the body over the final relations and $N$ the most items of a premise's form.

*Proof.* Fix a trigger position $t$ and an arriving item $x$. Its join binds the variables of $t$
to $x$'s values and then runs the levels of Generic Join over the other premises, each relation a
subset of its final one. Every set it intersects at a level is a subset of the set that Generic
Join over the whole body, in the same order and on the final relations, intersects below the same
prefix; and the prefixes it reaches are among those. So its work is at most that of the whole
join below the prefix of $x$'s values, plus $O(m)$ to enter the tries. Distinct items at $t$ have
distinct prefixes, so summing over them gives at most the whole join's work, $O(m n\,
\mathrm{AGM})$ ([Joins](joins.md)), plus $O(m N)$. Summing over the $m$ positions gives the bound.
QED

The plan records, in [`PlanRecord.joins`][quivers.stochastic.planning.PlanRecord], each rule's
algorithm, the weight $\rho^*$ of a least fractional edge cover of its body, and this bound with
$\mathrm{AGM} \le N^{\rho^*}$ ([`rule_joins`][quivers.stochastic.planning.rule_joins]). On the
triangle with a hub value and the skewed triangle, the chart stays within the bound, and a chart
that looks premises up one at a time does $\Theta(N^2)$, which the same bound rejects; on chains,
diamonds and skip graphs, whose step rule has $\rho^* = 1$, the work is linear in the edges.

## Keyed lookups

[`join_plans`][quivers.stochastic.indexed.join_plans] fixes, for every rule and trigger, the order
in which the other premises are looked up and the positions each lookup is keyed on: next the
premise with the most arguments whose variables are already bound. It handles any pattern that
matching binds in premise order, including a sum whose other summand is already bound, and its
instances are enumerated once by the same lemma, with the test applied as each premise is chosen.
[`ChartWork`][quivers.stochastic.indexed.ChartWork] counts the lookups, candidates, instances, side
conditions and index entries of either method.

::: quivers.stochastic.indexed
