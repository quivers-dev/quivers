# Documentation cross-linking audit

Checked November 6, 2026. This pass follows the accessibility revision recorded in [the section audit](docs-accessibility-audit.md).

## Coverage

Agents revised the guides and QVR reference, semantics and concepts primer, and learning documentation. The parent revised all API pages, consolidated the bibliography, and checked the rendered site. A separate prose review checked whether link labels accurately described the destinations; a separate citation review verified the new resources.

Across 317 Markdown pages in `docs/`:

| Measure | Before this pass | After this pass |
|---|---:|---:|
| Internal Markdown links | 1,515 | 2,084 |
| External Markdown links | 428 | 558 |
| BibTeX entries | 136 | 165 |
| Citation occurrences | 499 | 605 |

This adds 569 internal link occurrences and 130 external link occurrences. Counting both additions and replacements of broad links with specific destinations, 743 internal links were added or retargeted. README navigation also gained two section links; it is outside the table's `docs/` counts.

## What the links connect

- API pages connect the interface to its conceptual explanation, a worked example, or its formal assumptions. The API index links named interfaces to their reference pages.
- Category theory introductions connect to the relevant Milewski chapter or section; the formal definitions and law qualifications remain in the docs.
- Probabilistic programming explanations connect to specific Pyro and Stan sections on guides, ELBO derivations, event and batch shapes, marginalization, parameterization, and predictive checks.
- Gallery pages connect model variants to worked examples elsewhere in the docs and to the relevant external calculation. Links distinguish canonical neural architectures from the simplified stochastic gallery variants.
- The QVR reference connects syntax to tutorials, generated computations, runtime invocation, and the formal typing rules.

External links use native section IDs where available. Twelve links to named Milewski sections use text fragments matched to the exact heading text because the original headings have no IDs. Browsers without text-fragment support open the article; the link labels identify the requested explanation. Links about an entire concept point directly to that concept's chapter.

## Validation

- Strict MkDocs build passed in 67.06 seconds.
- All 743 added or retargeted internal links resolve to existing files and, where specified, IDs in the final rendered HTML.
- All 69 distinct new external destinations match original source pages, native HTML IDs, or verified heading text. The [destination manifest](docs-link-verification.json) records the URLs and referring pages.
- All 29 new bibliography records were checked against publisher, author, or official project sources by the section editors and an independent citation agent. [Citation verification](docs-citation-verification.md#additional-resources-in-the-cross-linking-pass) records the source and publication status for each.
- All 605 rendered citation links on 201 pages resolve. There are no unknown citation keys or unused bibliography entries.
- Twelve bibliography tests pass, including the new regression test for undated resources from the same corporate author. Ruff passes for the bibliography hook and its tests.
- Rendered math and whitespace checks pass. Fenced examples are identical to the snapshot before this linking pass. Runtime source remains unchanged by the documentation work.

The build initially exposed an error when disambiguating undated references. The hook now formats these as `n.d.` and `n.d.-a`, `n.d.-b`, as needed, without adding dates to the source BibTeX. A duplicate PyTorch tutorial entry was consolidated. Existing publisher-access limitations for Neal and Rauch remain documented in the citation report; the 29 new resources have no unresolved verification issue.
