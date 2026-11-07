# Documentation cross-linking audit

Updated November 7, 2026. This report covers the linking pass and the editorial
follow-up to [the accessibility revision](docs-accessibility-audit.md).

## Editorial follow-up

The initial linking pass appended repetitive navigation and reading recommendations
to otherwise substantive explanations. The follow-up was edited directly by the
primary assistant, with no delegated writing. It revised 248 documentation pages,
including all 171 affected API pages.

Links now attach to the concepts, calculations, and operations being explained.
The revisions remove the recurring Milewski recommendations, integrate citations
into supported claims, and rewrite the external-resource additions in the gallery,
tutorials, guides, and semantics. Formal definitions and their qualifications remain.
All fenced examples are byte-identical to the snapshot before this follow-up.

## Current coverage

These counts compare the documentation before the initial linking pass with the
current revision. They describe coverage, not an editorial target.

| Measure | Before linking | Current |
|---|---:|---:|
| Internal Markdown links | 1,515 | 2,022 |
| External Markdown links | 428 | 500 |
| BibTeX entries | 136 | 152 |
| Citation occurrences | 499 | 528 |

The retained additions and replacements include 684 internal link occurrences and
48 distinct external destinations. The [destination manifest](docs-link-verification.json)
records the original-source verification and current referring pages for those
external destinations. The follow-up introduces no new external destination or
bibliographic source.

## Validation

- Strict MkDocs build passed.
- Added or retargeted internal links resolve to files and rendered section IDs.
- All 528 rendered citation links on 184 pages resolve. There are no missing
  citation keys or unused bibliography entries.
- All 13 bibliography and changelog-mirror tests pass.
- Rendered math, public API coverage, and whitespace checks pass.
- Executable examples and runtime source are unchanged by the follow-up.

[Citation verification](docs-citation-verification.md) retains the source records
and publication-status distinctions. Direct publisher access for Neal (2011) and
Rauch, Tung, and Striebel (1965) was retried; both sites returned HTTP 403. These
checks remain incomplete, with the earlier corroborating evidence recorded explicitly.
