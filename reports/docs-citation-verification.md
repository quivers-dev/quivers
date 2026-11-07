# Documentation citation verification

Checked November 6, 2026; editorial follow-up November 7, 2026.

Publisher verification was attempted for every scholarly reference in the revised documentation. The records below identify the publisher metadata, publication artifact, or original resource that was inspected. A check that could not reach the publisher is recorded as incomplete. Publisher-indexed text, publisher-hosted pages, and institutional copies of the published article are distinguished below. Bibliographic checks establish identity and metadata; implementation claims require separate evidence.

Direct publisher access remains incomplete for the Neal (2011) chapter and Rauch, Tung, and Striebel (1965) article. Their records are corroborated by author or bibliographic records, and the checks below identify the unresolved fields. Several other publishers blocked landing-page access; where available, the final published PDF was checked through an author or institutional host. Preprints, forthcoming works, and original tutorials are labeled by publication status.

Duplicate keys from the parallel audits were consolidated in `docs/references.bib`. The retained source reports may describe more than one independent check of the same publication. A source marked removed or replaced is absent from the final bibliography.

## Semantics citation verification

Checked 2026-11-06. “Publisher-indexed” means search returned the publisher page or final PDF content; direct fetch was unavailable where stated. Bibliographic verification is distinct from checking every theorem in full. Unresolved publisher checks are explicitly marked, not certified. Optional fields without verified support were omitted.

| Key | Source and status | Fields checked or corroborated | Claim limitations / corrections |
|---|---|---|---|
| `kelly1982basic` | [Authorized reprint publisher metadata read](https://www.tac.mta.ca/tac/reprints/articles/10/tr10abs.html) | author, title, year, publisher, note, url | Original year, author, title and publisher verified by authorized reprint. Series omitted because landing page mislabels the original series; PDF retrieval timed out. |
| `maclane1998categories` | [Publisher page read](https://link.springer.com/book/10.1007/978-1-4757-4721-8) | author, title, year, publisher, edition, series, volume, doi, url | 1998 is the hardcover second-edition publication date, not the page’s conflicting copyright or ebook date. |
| `giry1982categorical` | [Publisher page read](https://link.springer.com/chapter/10.1007/BFb0092872) | author, title, year, booktitle, editor, series, volume, pages, publisher, doi, url | Monad background; unverified Theorem 5 pinpoint removed. Does not establish arbitrary scored programs are normalizable. |
| `fritz2020synthetic` | [Publisher-indexed metadata and abstract read; DOI fetch blocked](https://www.sciencedirect.com/science/article/pii/S0001870820302656) | author, title, year, journal, volume, pages, doi, url | Markov-category background. Unsupported attribution of numbered strength axioms to section 4 removed. |
| `moggi1991notions` | [Publisher-indexed metadata read; DOI fetch blocked](https://www.sciencedirect.com/science/article/pii/0890540191900524) | author, title, year, journal, volume, number, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `wadler1992essence` | [Publisher-indexed metadata and abstract read; direct fetch blocked](https://doi.org/10.1145/143165.143169) | author, title, year, booktitle, pages, publisher, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `plotkin2009handlers` | [Publisher page read](https://link.springer.com/chapter/10.1007/978-3-642-00590-9_7) | author, title, year, booktitle, series, volume, pages, publisher, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `hughes2000arrows` | [Publisher-indexed metadata and abstract read; direct DOI fetch blocked](https://www.sciencedirect.com/science/article/pii/S0167642399000234) | author, title, year, journal, volume, number, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `joyal1996traced` | [Publisher-indexed metadata read](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/traced-monoidal-categories/2BE85628D269D9FABAB41B6364E117C8) | author, title, year, journal, volume, number, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `lambek1958mathematics` | [Publisher-indexed metadata read; direct fetch blocked](https://www.tandfonline.com/doi/abs/10.1080/00029890.1958.11989160) | author, title, year, journal, volume, number, pages, doi, url | Publisher DOI replaces JSTOR DOI; QVR explicitly uses its own slash operand convention. |
| `lambek1968fixpoint` | [Publisher page read](https://link.springer.com/article/10.1007/BF01110627) | author, title, year, journal, volume, pages, doi, url | Initial algebra fixed-point result; fold uniqueness comes from initiality, not from citing the lemma alone. |
| `plotkin2003algebraic` | [Publisher page read](https://link.springer.com/article/10.1023/A:1023064908962) | author, title, year, journal, volume, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `bauer2015programming` | [Publisher-indexed metadata read](https://www.sciencedirect.com/science/article/pii/S2352220814000194) | author, title, year, journal, volume, number, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `bumford2026effect` | [Publisher-indexed forthcoming record and author page read](https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4) | author, title, year, publisher, series, isbn, note, url | Publisher lists Coming soon and author lists In press. Do not present 2026 as a completed publication; no DOI verified. |
| `battaglia2018relational` | [Author preprint repository read](https://arxiv.org/abs/1806.01261) | author, title, year, howpublished, doi, url | Explicitly preprint. Permutation claims require invariant aggregation and shared equivariant local maps; citation alone is not a proof for arbitrary encoders. |
| `carpenter2017stan` | [Publisher page read](https://www.jstatsoft.org/article/view/v076i01) | author, title, year, journal, volume, number, doi, url | Background reference; does not establish Quivers implementation correctness. Optional pages1--32 omitted because publisher HTML lacks pagination and PDF retrieval failed. |
| `phan2019numpyro` | [Official NumPyro citation and author preprint metadata read](https://arxiv.org/abs/1912.11554) | author, title, year, howpublished, note, doi, url | Corrected Compositional to Composable. Explicitly workshop/preprint, not main NeurIPS proceedings. |
| `bingham2019pyro` | [Publisher page read](https://www.jmlr.org/papers/v20/18-403.html) | author, title, year, journal, volume, number, pages, url | Background reference; does not establish Quivers implementation correctness. |
| `ge2018turing` | [Publisher page read](https://proceedings.mlr.press/v84/ge18b.html) | author, title, year, booktitle, series, volume, pages, publisher, url | Background reference; does not establish Quivers implementation correctness. |
| `cusumanotowner2019gen` | [Official ACM conference record and project BibTeX read](https://pldi19.sigplan.org/details/pldi-2019-papers/45/Gen-A-General-Purpose-Probabilistic-Programming-System-with-Programmable-Inference) | author, title, year, booktitle, pages, publisher, doi, url | ACM DL fetch blocked; official conference verifies title/authors/year. Pagination, DOI and author initials from official project BibTeX at https://www.gen.dev/assets/gen-pldi.txt; published-PDF fetch timed out. |
| `goodman2008church` | [Publisher page read](https://proceedings.mlr.press/r6/goodman08a.html) | author, title, year, booktitle, series, volume, pages, publisher, note, url | Background reference; does not establish Quivers implementation correctness. |
| `goodman2014design` | [Author-published electronic textbook read](https://dippl.org) | author, title, year, howpublished, url | Site supplies 2014 citation. Useful explanatory reference, not peer-reviewed proceedings or source for current backend versions. |
| `kruschke2014doing` | [Publisher-indexed edition record read](https://shop.elsevier.com/books/doing-bayesian-data-analysis/kruschke/978-0-12-405888-0) | author, title, year, publisher, edition, isbn, url | Publisher release date 3 November 2014; copyright says 2015. Cite release year 2014 consistently; omitted unconfirmed book DOI. |
| `lunn2009bugs` | [Publisher-indexed author/title/year/DOI read; published PDF corroborates journal/pages](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.3680) | author, title, year, journal, volume, number, pages, doi, url | Direct publisher retrieval timed out. General BUGS background, not proof of every renderer mapping. |
| `plummer2003jags` | [Official conference published PDF read](https://www.r-project.org/conferences/DSC-2003/Proceedings/Plummer.pdf) | author, title, year, booktitle, editor, url | Removed unsupported pages124--125 (PDF locally numbered1--10). Removed unverified attribution of zero trick to this paper. |
| `tran2018simple` | [Publisher final PDF read](https://proceedings.neurips.cc/paper/2018/file/201e5bacd665709851b77148e225b332-Paper.pdf) | author, title, year, booktitle, volume, url | Final PDF has8 authors; publisher HTML omits last2 and misstates Hoffman middle initial. PDF controls; unverified pages omitted. |
| `salvatier2016pymc3` | [Publisher-indexed final PDF read; direct fetch403](https://peerj.com/articles/cs-55/) | author, title, year, journal, volume, pages, doi, url | Paper concerns PyMC3; current PyMC/PyTensor behavior separately linked to current official docs. |
| `wingate2011lightweight` | [Publisher page read](https://proceedings.mlr.press/v15/wingate11a.html) | author, title, year, booktitle, series, volume, pages, publisher, url | Background reference; does not establish Quivers implementation correctness. |
| `koller2009probabilistic` | [Publisher page read](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/) | author, title, year, publisher, isbn, url | General graphical-model background; removed unsupported chapter4 pinpoint. |
| `shieber1995principles` | [Publisher-indexed metadata/abstract read; author institution confirms names](https://www.sciencedirect.com/science/article/pii/074310669500035I) | author, title, year, journal, volume, number, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `pereira1983parsing` | [Publisher anthology metadata read](https://aclanthology.org/P83-1021/) | author, title, year, booktitle, pages, publisher, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `klein2001parsing` | [Publisher anthology metadata read](https://aclanthology.org/W01-1812/) | author, title, year, booktitle, pages, url | Background reference; does not establish Quivers implementation correctness. |
| `knuth1977generalization` | [Publisher-indexed metadata read](https://www.sciencedirect.com/science/article/abs/pii/0020019077900023) | author, title, year, journal, volume, number, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `nederhof2003weighted` | [Publisher anthology metadata read](https://aclanthology.org/J03-1006/) | author, title, year, journal, volume, number, pages, doi, url | Background reference; does not establish Quivers implementation correctness. |
| `eisner2005compiling` | [Publisher anthology metadata read](https://aclanthology.org/H05-1036/) | author, title, year, booktitle, publisher, pages, url | Publisher PDF https://aclanthology.org/H05-1036.pdf includes Practical before Weighted; restored this word despite omission in ACL landing metadata. PDF is authoritative. |
| `mcallester2002complexity` | [Publisher retrieval unavailable; institution-hosted published PDF read](https://doi.org/10.1145/581771.581774) | author, title, year, journal, volume, number, pages, doi, url | Publisher-source verification incomplete: ACM inaccessible. Final ACM-formatted PDF at https://www.cs.cmu.edu/~fp/courses/lp/handouts/mcallester02jacm.pdf corroborates author/title/year/journal/pages. |
| `kock1972strong` | [Publisher page read](https://link.springer.com/article/10.1007/BF01304852) | author, title, year, journal, volume, pages, doi, url | General strength/commutativity background, not an attribution of exact numbered axioms or a paper-specific Fubini theorem. |
| `kingma2014autoencoding` | [Official ICLR retrospective, author manuscript and institution conference metadata read](https://arxiv.org/abs/1312.6114) | author, title, year, booktitle, note, url | ICLR2014 accepted status/year verified at https://dare.uva.nl/id/cf65ba0f-d88f-4a49-8ebd-3a7fce86edd7; institution says conference submissions on arXiv. Official conference retrospective https://blog.iclr.cc/2024/05/07/iclr-2024-test-of-time-award/ confirms the conference paper and links to this manuscript; no arXiv DOI misrepresented as conference DOI. |
| `robbins1951stochastic` | [Authorized publisher archive indexed issue metadata and published PDF read; main publisher iframe only](https://doi.org/10.1214/aoms/1177729586) | author, title, year, journal, volume, number, pages, doi, url | IMS-authorized archive https://www.jstor.org/stable/i312767 verifies author/title/year/volume/issue/pages; published PDF read at https://www.columbia.edu/~ww2040/8100F16/RM51.pdf. DOI corroborated by registry-derived record, not publisher page. Generic nonconvex SGD convergence attribution removed; cite only stochastic approximation background. |
| `rosenthal1990quantales` | [Publisher-source verification incomplete; library catalog and scanned book metadata agree](https://search.worldcat.org/title/Quantales-and-their-applications/oclc/1504394167) | author, title, year, publisher, isbn, url | No publisher record retrieved after title/publisher searches. Removed from the final bibliography; replaced by publisher-verified `rosenthal1996quantaloids` below. |

| `rosenthal1996quantaloids` | [Publisher-indexed title/author/year/edition/publisher and description read; direct fetch cache miss](https://www.routledge.com/The-Theory-of-Quantaloids/Rosenthal/p/book/9780429157028) | author, title, year, edition, publisher, URL | Verified replacement for1990 source in broad quantale/enrichment reading recommendations. This is a distinct book. Publisher describes quantaloids as a generalization of quantales and identifies enrichment as a central topic. The unverified1990 entry above records a removed citation, not an active bibliography entry. |

| `klein2003astar` | [Publisher anthology page and final PDF read](https://aclanthology.org/N03-1016/) | authors, title, year, venue, pages119--126, URL | Corrected queue docstring: highest priority first requires negating minimization costs; queue construction alone does not establish A* optimality. Publisher PDF checked; no unsupported Knuth theorem attribution. |
| `moortgat1997categorial` | [Publisher-indexed1997 handbook table of contents read; direct403](https://www.sciencedirect.com/book/9780444817143/handbook-of-logic-and-language) | author, title, year, handbook, pages93--177, publisher; editor names corroborated by author institution | General background for modalities in type-logical grammar. No claim a named runtime modality automatically satisfies a logical calculus. No DOI added without verification. |
| `vagner2015algebras` | [Publisher-indexed article metadata and abstract read; direct502](https://www.tac.mta.ca/tac/volumes/30/51/30-51abs.html) | all3authors, title, year, journal, volume30, article51, pages1793--1822, URL | Replaces ambiguous Spivak2014 attribution with a precise published wiring-diagram example. Does not claim this source proves QVR contraction laws. Corrected original docstring’s identification of binary composition with2-cells. |
| `eisner2007transformations` | [Publisher-indexed proceedings PDF title page and article front matter read; direct timeout](https://web.stanford.edu/group/cslipublications/cslipublications/FG/2006/FG.pdf) | authors, title, copyright2007, proceedings FG-2006, editor ShulyWintner, publisherCSLI | Citation distinguishes conference2006 from publication2007; no unverified page range included. General source-to-source transformation background. |

### Autodoc corrections

Removed the nonexistent or unverified Hughes2000 Theorem3.1 and section4 pinpoints, Joyal1996 section3 pinpoint, and Goodman1999 section3 strategy-independence guarantee. Applicative laws now cite `mcbride2008applicative` (verified by guides agent). Traversal equations remain interface requirements without attributing an unverified exact law formulation. Source docstrings use explicit publication links; their API pages supply matching citations from the shared BibTeX bibliography. No executable statements changed.

## Guide/reference publisher verification

Checked 2026-11-06. Publisher landing pages, PDFs, or indexed publisher-page text were inspected with web tools. Bibliographic verification does not establish that every library implementation satisfies a paper’s assumptions.

- `durkan2019splines`: https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `roeder2017sticking`: https://proceedings.neurips.cc/paper/2017/hash/e91068fff3d7fa1594dfdf3b4308433a-Abstract.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `dieng2017chi`: https://proceedings.neurips.cc/paper/2017/hash/35464c848f410e55a13bb9d78e7fddd0-Abstract.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `li2016renyi`: https://proceedings.neurips.cc/paper/2016/hash/7750ca3559e5b8e1f44210283368fc16-Abstract.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `kingma2016iaf`: https://proceedings.neurips.cc/paper/2016/hash/ddeebdeefdb7e7e7a697e1c3e3d8ef54-Abstract.html — NeurIPS publisher title starts Improved, not Improving.
- `papamakarios2017maf`: https://proceedings.neurips.cc/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `kingma2018glow`: https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `zhang2021dais`: https://proceedings.neurips.cc/paper/2021/hash/a1a609f1ac109d0be28d8ae112db1bbb-Abstract.html — NeurIPS proceedings verify authors/title2021. Correct associated preprint2107.10211; API2102.07501 is DIFFERENT Arbel/Matthews/Doucet Annealed Flow Transport Monte Carlo.
- `tucker2019dreg`: https://openreview.net/forum?id=HkG3e205K7 — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `dinh2017realnvp`: https://openreview.net/forum?id=HkpbnH9lx — Official OpenReview PDF header identifies published conferencepaper ICLR2017; replace preprint2016.
- `bornschein2015rws`: https://iclr.cc/archive/www/2015.html — Official ICLR2015 archive verifies authors/title and poster acceptance; cite conference2015, not preprint2014.
- `burda2016iwae`: https://www.iclr.cc/archive/www/2016.html — Official ICLR2016 accepted-paper archive confirms venue/year/title; archived entry orders last two authors differently. Paper author order follows linked arXiv record (Burda, Grosse, Salakhutdinov).
- `daudel2023vriwae`: https://jmlr.org/papers/v24/22-1160.html — JMLR landing confirms24(243):1–83,2023; replace preprint2022.
- `ranganath2014bbvi`: https://proceedings.mlr.press/v33/ranganath14.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `ioffe2015batchnorm`: https://proceedings.mlr.press/v37/ioffe15.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `tierney1986laplace`: https://www.tandfonline.com/doi/abs/10.1080/01621459.1986.10478240 — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `lewandowski2009lkj`: https://www.sciencedirect.com/science/article/pii/S0047259X09000876 — Publisher ScienceDirect search result supplied complete title/authors/volume/issue/pages/DOI; direct fetch403.
- `carvalho2010horseshoe`: https://academic.oup.com/biomet/article-abstract/97/2/465/219397 — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `vehtari2017loo`: https://link.springer.com/article/10.1007/s11222-016-9696-4 — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `vehtari2024psis`: https://www.jmlr.org/papers/v25/19-556.html — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `vehtari2021rhat`: https://doi.org/10.1214/20-BA1221 — LIMITATION: Original Project Euclid article and issue returned iframe-only; DOI confirmed. Metadata corroborated on IMS-hosted Statistical Science37(3) PDF https://imstat.org/publications/sts/sts_37_3/sts_37_3.pdf. Publication-artifact check resolved below.
- `yao2018stacking`: https://doi.org/10.1214/17-BA1091 — LIMITATION: Original Project Euclid returned iframe-only. Authors/title/September2018 verified by publisher ISBA announcement https://bayesian.org/bayesian-analysis-call-for-contributed-discussions-of-using-stacking-to-average-bayesian-predictive-distributions/ and September2018 bulletin. Page range subsequently checked against the publication artifact below.
- `rasmussen2006gp`: https://mitpress.mit.edu/9780262182539/gaussian-processes-for-machine-learning/ — LIMITATION: MIT Press landing confirms authors/title/ISBN and lists release November23,2005; The publisher-linked open edition confirms copyright year2006, as recorded below. DOI is publisher-linked but403; The copyright page was checked in the open edition.
- `bauer2015programming`: https://www.sciencedirect.com/science/article/pii/S2352220814000194 — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `charlow2025scope`: https://academic.oup.com/jos/issue/42/4 — OUP issue42(4) publisher page verifies author/title/353–398/November2025; DOI string containingffad does not imply2023 issue year.
- `bumford2026effect`: https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4 — Publisher lists Coming soon, expected July13,2026; no DOI supplied, so marked forthcoming instead of claiming published.
- `plotkin2003algebraic`: https://link.springer.com/article/10.1023/A:1023064908962 — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `mcbride2008applicative`: https://www.cambridge.org/core/journals/journal-of-functional-programming/article/applicative-programming-with-effects/C80616ACD5687ABDC86D2B341E83D298 — Cambridge publisher PDF gives JFP18(1):1–13,2008, with first onlineMay16,2007; cite issue2008.
- `klein2001parsing`: https://aclanthology.org/W01-1812/ — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `mcallester2002complexity`: https://dl.acm.org/doi/10.1145/581771.581774 — LIMITATION: ACM article returned403; Published PDF subsequently checked as recorded below.
- `eisner2007transformations`: https://web.stanford.edu/group/cslipublications/cslipublications/FG/2006/FG.pdf — Publisher-hosted proceedings PDF search result confirms authors/title/CSLI2007, proceedings FG-2006, first page45. End page85 corroborated in Cambridge publication reference list; direct PDF fetch timed out.
- `pereira1983parsing`: https://aclanthology.org/P83-1021/ — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `knuth1977generalization`: https://www.sciencedirect.com/science/article/abs/pii/0020019077900023 — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `huang2005kbest`: https://aclanthology.org/W05-1506/ — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `nederhof2003weighted`: https://aclanthology.org/J03-1006/ — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `shieber1995principles`: https://www.sciencedirect.com/science/article/pii/074310669500035I — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.
- `tai2015treelstm`: https://aclanthology.org/P15-1150/ — Publisher metadata/PDF verified title, authors, year and listed bibliographic fields.

### API references not included

- Betancourt2017 Conceptual Introduction to Hamiltonian Monte Carlo: arXiv1701.02434 verified primary preprint, no journal/conference publisher located. Prefer Neal chapter if strict publisher-only rule.
- Beskos et al.2013 Optimal tuning of the hybrid Monte Carlo algorithm: DOI10.3150/12-BEJ414; publisher fetch blocked. Author UCL archive supports asymptotic acceptance0.651 in high-dimensional iid product targets, not universal optimal acceptance. Published PDF subsequently checked as recorded below.

### Resolution of publication-artifact checks

The following checks used publisher-formatted versions of record hosted by authors or institutions. Hosting and access limits are recorded for each source.

- `mcallester2002complexity`: ACM version of record inspected at https://www.cs.cmu.edu/~fp/courses/lp/handouts/mcallester02jacm.pdf, first-page journal imprint confirms49(4),July2002,512–537. ACM landing still403.
- `vehtari2021rhat`: Aalto institutional repository explicitly labels Publisher’s PDF/version of record, inspected https://acris.aalto.fi/ws/portalfiles/portal/53922181/Vehtari_Rank_Normalization.euclid.ba.1593828229.pdf; author/title/DOI/2021/16(2)667–718 verified. Euclid landing still iframe-only.
- `yao2018stacking`: publisher-formatted first page indexed at Twente institutional repository https://research.utwente.nl/files/460106188/17-BA1091.pdf confirms Bayesian Analysis2018,13(3),917–1007, authors/title. Direct retrieval403; indexed publication artifact checked, not full paper.
- `rasmussen2006gp`: publisher-linked author-hosted open edition https://gaussianprocess.org/gpml/chapters/RW.pdf explicitly ©2006MIT with bibliographic2006 header. Cite copyright edition2006; MIT store release dateNovember2005 is distinct.
- `beskos2013optimal`: publisher-formatted article inspected https://discovery.ucl.ac.uk/id/eprint/77553/1/Beskos_PillaiRobertsSernaStuart13.pdf, first-page DOI10.3150/12-BEJ414/Bernoulli19(5A)2013,1501–1534. Theorems4.1–4.2 yield asymptotic0.651 under hypotheses; §4.3 notes finite-dimensional practical optima may be higher. Citation must not claim a universal optimal target acceptance.
- `germain2015made`: PMLR publisher https://proceedings.mlr.press/v37/germain15.html confirms Germain/Gregor/Murray/Larochelle, ICML2015,37:881–889. Source incorrectly listed Mnih. arXiv1502.03509 is MADE; it is not the Batch Normalization paper.
- `rezende2015flows`: PMLR publisher https://proceedings.mlr.press/v37/rezende15.html inspected; Rezende and Mohamed, Variational Inference with Normalizing Flows, ICML2015,37:1530–1538 verified.
- `mackay2003information`: Cambridge publisher frontmatter https://assets.cambridge.org/97805216/42989/frontmatter/9780521642989_frontmatter.pdf indexed text confirms exact title, David J.C.MacKay, ISBN9780521642989, copyright/first publication2003. Direct fetch503 but publisher PDF text indexed. No chapter-specific locator claimed without inspecting chapter.

## Learning documentation citation verification

Checked November 6, 2026. Publisher websites, their indexed metadata, and publisher typeset or author originals were inspected. Access limits are explicit below; this ledger does not imply every full text was accessible. Citations establish conceptual background, not validation of Quivers behavior. Shared keys are verified by the semantics or guides agents.

### gelman2013bayesian

- Source: https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### bishop2006pattern

- Source: https://link.springer.com/book/9780387310732
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### blei2017variational

- Source: https://doi.org/10.1080/01621459.2017.1285773
- Status: Publisher endpoint returned403; publisher typeset PDF on author institutional site verified: https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf

### hoffman2013svi

- Source: https://www.jmlr.org/papers/v14/hoffman13a.html
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### hoffman2014nuts

- Source: https://www.jmlr.org/papers/v15/hoffman14a.html
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### ferrari2004beta

- Source: https://www.tandfonline.com/doi/permissions/10.1080/0266476042000214501
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Mean/precision beta regression; excludes exact zero and one outcomes.

### bartholomew2011latent

- Source: https://onlinelibrary.wiley.com/book/10.1002/9781119970583
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### spearman1904intelligence

- Source: https://www.jstor.org/stable/1412107
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Historical factor-analysis motivation; does not validate this particular factor model. Corrected original end page293 to292.

### skellam1948binomial

- Source: https://academic.oup.com/jrsssb/article/10/2/257/7026529
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Beta-binomial construction. Replaced unrelated JSTOR2983694 citation.

### klein2003survival

- Source: https://link.springer.com/book/10.1007/b97377
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### maier2014dirichletreg

- Source: https://research.wu.ac.at/en/publications/dirichletreg-dirichlet-regression-for-compositional-data-in-r-3/
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Institutional working paper, not journal article; gamma normalization/Dirichlet regression context.

### montague1973quantification

- Source: https://link.springer.com/chapter/10.1007/978-94-010-2506-5_10
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### barwise1981quantifiers

- Source: https://link.springer.com/article/10.1007/BF00350139
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### morris2019spatial

- Source: https://www.sciencedirect.com/science/article/pii/S1877584518301175
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### casella1996rao

- Source: https://academic.oup.com/biomet/article-abstract/83/1/81/255465
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Rao-Blackwell variance reduction background; article treats post-simulation accept-reject/Metropolis improvement, not QVR enumeration.

### stephens2000labels

- Source: https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00265
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### goodman1999semiring

- Source: https://aclanthology.org/J99-4004/
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Semiring formulation of parsing, not unconditional equivalence of all search strategies on cyclic charts.

### cho2014encoder

- Source: https://aclanthology.org/D14-1179/
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: GRU architecture motivation; original deterministic gates differ from this probabilistic example.

### bergen2016pragmatic

- Source: https://semprag.org/index.php/sp/article/view/sp.9.20
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### blei2003lda

- Source: https://www.jmlr.org/papers/v3/blei03a.html
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Canonical LDA uses topic choices per token; current gallery program shares a topic per document and is explicitly distinguished.

### salakhutdinov2007pmf

- Source: https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html
- Status: Publisher page and publisher PDF verified; PDF lists Salakhutdinov first, contrary to abstract metadata. Followed PDF author order.

### socher2013tensor

- Source: https://papers.nips.cc/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Neural tensor network motivation; the gallery implements a simpler bilinear contraction.

### sutskever2014sequence

- Source: https://papers.nips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Replaced preprint citation with conference version; gallery does not duplicate original multilayer LSTM.

### vaswani2017attention

- Source: https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### krishnan2017structured

- Source: https://ojs.aaai.org/index.php/AAAI/article/view/10779
- Status: Publisher indexed metadata and publisher PDF verified; direct landing page bot-blocked.

### gelman1996predictive

- Source: https://www3.stat.sinica.edu.tw/statistica/J6N4/J6N41/J6N41.htm
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### kim1998volatility

- Source: https://academic.oup.com/restud/article-abstract/65/3/361/1565336
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### tipping1999ppca

- Source: https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9868.00196
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### hochreiter1997lstm

- Source: https://doi.org/10.1162/neco.1997.9.8.1735
- Status: Publisher indexed full metadata verified; direct MIT endpoint403.
- Scope: Original LSTM architecture; gallery adds stochastic gates and uses a later forget-gate form.

### balakrishnan1992logistic

- Source: https://www.routledge.com/Handbook-of-the-Logistic-Distribution/Balakrishnan/p/book/9780367450458
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### papaspiliopoulos2007parametrization

- Source: https://projecteuclid.org/journals/statistical-science/volume-22/issue-1/A-General-Framework-for-the-Parametrization-of-Hierarchical-Models/10.1214/088342307000000014.pdf
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### carlin1992changepoint

- Source: https://academic.oup.com/jrsssc/article/41/2/389/6990498
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### nelder1972glm

- Source: https://academic.oup.com/jrsssa/article/135/3/370/7110572
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### rubin1981parallel

- Source: https://journals.sagepub.com/doi/10.3102/10769986006004377
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### kumaraswamy1980bounded

- Source: https://www.sciencedirect.com/science/article/pii/0022169480900360
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.

### murphy2007gaussian

- Source: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
- Status: Author-hosted original technical report verified; no claim of peer-reviewed publication.
- Scope: Closed-form conjugate Gaussian updates, with explicit parameterization conventions.

### neal2011mcmc

- Source: https://doi.org/10.1201/b10905-6
- Status: Author original https://glizen.com/radfordneal/ham-mcmc.abstract.html verifies title, author, book, editors, year, chapter and pages. Publisher endpoint403; DOI -6 agrees registry-derived records, but direct publisher DOI verification unavailable. Do not use -7.

### neal2003slice

- Source: https://doi.org/10.1214/aos/1056562461
- Status: Author original https://glizen.com/radfordneal/slice-aos.abstract.html verifies title, author, volume/pages and links publisher record euclid.aos/1056562461. Publisher iframe prevented access; journal publication and discussion range confirmed by author.

### gelman2006variance

- Source: https://doi.org/10.1214/06-BA117A
- Status: Publisher typeset article on author institutional site https://sites.stat.columbia.edu/gelman/bayescomputation/Gelman2006.pdf verifies all fields except DOI, which agrees publisher indexed references. Corrected end page534 to533; publisher landing iframe inaccessible.
- Scope: Half-t priors for scale parameters; not a claim of robust likelihoods for observations.

### mackay1992bayesian

- Source: https://doi.org/10.1162/neco.1992.4.3.448
- Status: Publisher endpoint403; author institution published-version record verifies all metadata: https://authors.library.caltech.edu/records/8n76z-g9k64. Publisher PDF linked there could not be fetched.

### kalman1960filtering

- Source: https://doi.org/10.1115/1.3662552
- Status: Publisher indexed DOI result verifies title/author/content; ASME-supplied NDLI record verifies volume/pages/date. Original paper scan https://skoge.folk.ntnu.no/puublications_others/1960_Kalman%20-%20A%20new%20approach%20to%20linear%20filtering%20and%20prediction%20problems%20-%20Orinal%20version%20with%20comments.pdf. ASME current archive labels journal as Fluids Engineering; retained historical Basic Engineering.

### rauch1965smoothing

- Source: https://arc.aiaa.org/doi/10.2514/3.3166
- Status: UNRESOLVED publisher verification: AIAA landing/PDF403, publisher search yielded no original. Registry-derived library metadata agrees title/year/journal/pages; author order retained from CiNii Crossref record. Independent parent check also could not access original publisher; NII reflects publisher-deposited author order, not inspection of original article.

### lambert1992zip

- Source: https://www.tandfonline.com/doi/abs/10.1080/00401706.1992.10485228
- Status: Publisher page or publisher-indexed metadata verified; title, author, year and listed bibliographic fields checked.
- Scope: Mixture of structural zero and Poisson counts; replaced archival JSTOR DOI with publisher DOI.

### seki1991mcfg

- Source: https://www.sciencedirect.com/science/article/pii/030439759190374B
- Status: Publisher indexed article verifies title/year/journal/volume/pages/DOI; author list confirmed in publisher-hosted citations. Corrected wrong DOI10.1016/0890-5401(91)90031-B.
- Scope: Grammar formalism and generative capacity, not all claims about this weighted implementation.

### moortgat1997types

- Source: https://www.sciencedirect.com/book/9780444817143/handbook-of-logic-and-language
- Status: Publisher indexed table of contents verifies title/author/pages;1997 and editors confirmed author institution. DOI confirmed by author-authored Wiley encyclopedia bibliography. Direct chapter endpoint403.

### birnbaum1968traits

- Source: https://bookstore.emerald.com/statistical-theories-of-mental-test-scores-pb-9781593119348.html
- Status: Reprint publisher table of contents verifies Part V title; original book year/publisher/contributor verified in NLM catalog https://www.ncbi.nlm.nih.gov/nlmcatalog/141772. Original page range could not be verified, so omitted rather than retaining docs397–479.

### maddison2017concrete

- Source: https://openreview.net/forum?id=S1jE5L5gl
- Status: Official ICLR2017 conference poster list verifies accepted title; author arXiv PDF https://arxiv.org/pdf/1611.00712 header confirms ICLR2017, authors and title. OpenReview bot challenge prevented direct access.
- Scope: Biased gradients for relaxed discrete models, not exact marginalization.

### betancourt2015hierarchical

- Source: https://doi.org/10.1201/b18502-5
- Status: Publisher indexed book table of contents verifies chapter/book/editors/year; publisher-supplied OReilly chapter preview https://www.oreilly.com/library/view/current-trends-in/9781482235128/chapter-30.html verifies authors. Direct chapter endpoint inaccessible; DOI/pages corroborated registry record. Replaced2013 preprint with2015 publication.

### rabiner1989hmm

- Source: https://doi.org/10.1109/5.18626
- Status: Publisher indexed IEEE DOI page verifies title/year/venue/volume/pages; publisher PDF mirrored at MIT https://web.mit.edu/6.435/www/Rabiner89.pdf confirms author. Direct IEEE page bot-blocked.


## Additional API citation verification

Checked November 6, 2026. Publication titles and metadata retain their original spelling. Publisher search indexes count as inspected publisher content where explicitly indicated; a blocked request alone does not count as verification. Author-hosted versions of record are identified separately from publisher-hosted pages.

| Key | Evidence | Fields and scope |
|---|---|---|
| `cho2019disintegration` | [Cambridge publisher record](https://doi.org/10.1017/S0960129518000488), full metadata and abstract read | Authors, title, year, journal, volume, issue, pages, DOI. Conditional distributions and Bayesian inversion; existence requires appropriate hypotheses. |
| `bernardo1994bayesian` | [Wiley publisher record](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316870) and [print edition](https://www.wiley.com/en-gb/bayesian-theory-p-9780471924166), publisher-indexed metadata read | Authors, title, original publication May 1994, publisher, DOI. Distinguish later paperback release. |
| `mccullagh1980regression` | [Oxford publisher record](https://academic.oup.com/jrsssb/article/42/2/109/7027621), metadata and abstract read | Author, title, journal, 1980 issue, volume, issue, pages, DOI. Online archive migration date is not publication year. |
| `mullahy1986specification` | [Elsevier publisher record](https://www.sciencedirect.com/science/article/abs/pii/0304407686900023), publisher-indexed metadata read | Author, title, year, journal, volume, issue, pages, DOI. Hurdle models, not an implementation correctness claim. |
| `mclachlan2000finite` | [Wiley publisher record](https://onlinelibrary.wiley.com/doi/book/10.1002/0471721182), publisher-indexed metadata read | Authors, title, first publication September 18, 2000, publisher, DOI. General finite mixture background. |
| `dilavore2025partial` | [Original arXiv preprint record](https://arxiv.org/abs/2502.03477) read | Authors, title, 2025 submission, 2026 revision, DOI. Explicitly a preprint, not a claimed journal publication; distinct from earlier work. |
| `hoffman2019neutra` | [Original preprint](https://arxiv.org/abs/1903.03704) and [authors' institutional publication record](https://research.google/pubs/neutra-lizing-bad-geometry-in-hamiltonian-monte-carlo-using-neural-transport/) read | Authors, title, year, identifier, DOI. Explicitly a preprint; no publisher venue located. |
| `scibior2018functional` | [Author-hosted ACM version of record](https://denotational.co.uk/publications/scibior-kammar-ghahramani-funcitonal-programming-for-modular-bayesian-inference.pdf) read; [institutional copy](https://www.research.ed.ac.uk/files/81876600/Functional_Programming_for_Modular_SCIBIOR_DoA040718_VoR_CC_BY.pdf) corroborates | Authors, title, journal, volume, ICFP issue, article 83, 29 pages, 2018, DOI confirmed by ACM reference format in published PDF. Direct ACM page returned 403. |
| `nguyen2023effect` | [Author-hosted ACM published PDF](https://min-nguyen.github.io/files/papers/haskell23.pdf) read | Authors, title, venue, 2023, publisher, DOI confirmed on first page. Direct ACM page returned 403. Omit page range because PDF uses local numbering and first-page reference format states 15 pages. |
| `maechler2012log1mexp` | [Original CRAN technical note](https://cran.r-project.org/package=Rmpfr/vignettes/log1mexp-note.pdf) read | Author, title, April/October 2012 document date, current typesetting date. Supports stable log1mexp computation, not all distribution implementation claims. |

The former unsupported reference to Welsh et al. (1996) in `Mixture.restrict_to` was removed: that method's mixture reweighting equation follows directly from conditioning and is explained in place. The publisher landing was attempted but blocked; secondary bibliographic agreement was not treated as publisher verification.

- `ospina2010inflated`: [Springer publisher page](https://link.springer.com/article/10.1007/s00362-008-0125-4) read. Authors, title, journal, volume, pages, DOI, and January 2010 issue date verified; first online March 2008 is distinguished from the issue year. The [original brms family documentation](https://paulbuerkner.com/brms/articles/brms_families.html) independently defines the conditional endpoint parameterization. Corrected the API formula that conflated total interval probability with a weighted density.

## Additional resources in the cross-linking pass

Checked November 6, 2026 by the section editors and an independent citation agent. The additional resources were verified on the original publisher, author, or official project site. The table retains the 19 resources still cited after the editorial follow-up. Undated tutorials retain their documented version and access date; no publication year was inferred from the access date. The duplicate PyTorch tutorial record was consolidated into `pytorchAutogradTutorial`.

| Key | Verified original source | Status |
|---|---|---|
| `pyroIntroduction` | [Introduction to Pyro](https://pyro.ai/examples/intro_long.html) | Pyro Tutorials, version 1.9.1; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `pyroSVI` | [SVI Part I: An Introduction to Stochastic Variational Inference in Pyro](https://pyro.ai/examples/svi_part_i.html) | Pyro Tutorials, version 1.9.1; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `pyroTensorShapes` | [Tensor Shapes in Pyro](https://pyro.ai/examples/tensor_shapes.html) | Pyro Tutorials, version 1.9.1; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `pyroEffects` | [Poutine: A Guide to Programming with Effect Handlers in Pyro](https://pyro.ai/examples/effect_handlers.html) | Pyro Tutorials, version 1.9.1; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `pyroEnumeration` | [Inference with Discrete Latent Variables](https://pyro.ai/examples/enumeration.html) | Pyro Tutorials, version 1.9.1; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanPredictiveChecks` | [Posterior and Prior Predictive Checks](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanReparameterization` | [Reparameterization and Change of Variables](https://mc-stan.org/docs/stan-users-guide/reparameterization.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanFiniteMixtures` | [Finite Mixtures](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `pyroVAE` | [Variational Autoencoders](https://pyro.ai/examples/vae.html) | Pyro Tutorials, version 1.9.1; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `pyroBayesianRegression` | [Bayesian Regression - Introduction (Part 1)](https://pyro.ai/examples/bayesian_regression.html) | Pyro Tutorials, version 1.9.1; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanRegression` | [Regression Models](https://mc-stan.org/docs/stan-users-guide/regression.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanTimeSeries` | [Time-Series Models](https://mc-stan.org/docs/stan-users-guide/time-series.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanLatentDiscrete` | [Latent Discrete Parameters](https://mc-stan.org/docs/stan-users-guide/latent-discrete.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanSurvival` | [Survival Models](https://mc-stan.org/docs/stan-users-guide/survival.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `stanEfficiency` | [Efficiency Tuning](https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html) | Stan User's Guide, version 2.40; title, maintainer attribution, and relevant explanation checked. Publication date unspecified. |
| `olah2015lstm` | [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Author-published tutorial: author, title, August 27, 2015. |
| `alammar2018transformer` | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Author-published tutorial: author, title, June 27, 2018; date also checked in the author’s source file. |
| `nltkSyntax2019` | [Analyzing Sentence Structure](https://www.nltk.org/book/ch08.html) | Author-maintained online chapter: authors, copyright 2019, build date September 4, 2019; distinct from the print edition. |
| `pytorchAutogradTutorial` | [Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | Official tutorial: created February 10, 2021; updated January 16, 2024. |

Links to tutorial subsections were checked against the original HTML IDs. The editorial follow-up removed repeated reading recommendations and their unused BibTeX entries, including all Milewski entries. It uses the existing verified resources within the explanations; it introduces no new bibliographic source.

The independent review found no unverifiable new reference, unsupported quotation, or unsupported conceptual attribution. The earlier Neal and Rauch publisher-access limitations remain as documented above. The [link audit](docs-link-audit.md) records coverage and final build checks.

## Editorial follow-up: November 7, 2026

The bibliography now contains 152 entries, all cited. There are 528 citation occurrences across 184 pages. Thirteen uncited entries were removed after references were integrated into the claims they support. Earlier verification records for removed entries remain historical evidence, rather than a claim that those entries are still in the bibliography.

Direct publisher access was retried for Neal (2011) at Taylor & Francis and Rauch, Tung, and Striebel (1965) at AIAA. Both returned HTTP 403. Their publisher-verification limitations remain unresolved; the earlier corroborating records are unchanged.
