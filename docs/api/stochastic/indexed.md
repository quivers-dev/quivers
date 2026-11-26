# Indexed charts

An [`IndexedChart`][quivers.stochastic.indexed.IndexedChart] holds the items of a
[`RulePresentation`][quivers.qiec.rules.RulePresentation] with one index per form and set of
argument positions that a rule's join looks up, so finding the partners of a new item reads only
the items that agree with it on the variables already bound.

[`join_plans`][quivers.stochastic.indexed.join_plans] fixes, for every rule and every premise that
may trigger it, the order in which the other premises are looked up and the positions each lookup
is keyed on: after the trigger, the next premise is the one with the most arguments whose
variables are already bound, and those arguments are its key. A plan with trigger `-1` starts from
the conclusion, which is how a backward pass finds the instances concluding one item. Lookups
build their indexes when first asked and keep them up to date as items are added.

[`IndexedChart.instances`][quivers.stochastic.indexed.IndexedChart.instances] enumerates the
instances a newly admitted item completes, skipping any in which the same item fills an earlier
premise, so each instance is enumerated once over a run (the lemma of [Plans](planning.md)).
[`ChartWork`][quivers.stochastic.indexed.ChartWork] counts the lookups, the candidates they return,
the instances, the side conditions evaluated and the index entries.

::: quivers.stochastic.indexed
