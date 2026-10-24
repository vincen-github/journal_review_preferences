# Upstream contributions and reuse audit

## Reviewed ICAI component and optional bridge

Source: [rdnfn/icai](https://github.com/rdnfn/icai), commit **`cb88e6588e9a01e8c1f216ea3cfe1b7461ddee7d`**, package metadata 0.4.8. The paper authors are Arduin Findeis and collaborators; repository metadata lists `rdnfn` and `timokau`. Explicit root license: **Apache-2.0**, retained verbatim in [licenses/ICAI-Apache-2.0.txt](licenses/ICAI-Apache-2.0.txt). The pyproject MIT classifier conflicts with that file; this adaptation follows the root Apache terms. No root NOTICE was listed.

Adapted component: `src/inverse_cai/algorithm/filter.py::filter_according_to_votes` into `journal_prefs/icai_support.py::filter_principles`. Changes: compact expression of the same strict majority, relevance/validity predicates, stable ranking and optional truncation; renamed entrypoint, defaults and nonnegative-integer validation. The component retains Apache-2.0 provenance. The bridge is project-authored. No original prompts, embedding/clustering/voting pipeline, data or dependencies were vendored or executed. [method boundaries](docs/ICAI_ADAPTATION.md).

The full original algorithm was not reproduced or run. ICAI is retained as an optional reference, not the current default; the main method uses absolute-score prediction and existing affine calibration. No score-derived pair labels were constructed.

## Author-released public data (2026-10-23)

The following are **data provenance references**, separate from source-code reuse. The small import/selection helper is authored for this prototype; no dataset repository scripts, dependencies, or parser implementations were copied or executed. Actual records remain gitignored for local research inspection and are not redistributed or included in the project's MIT license.

| Source | Fixed version and retained components | Rights and modifications |
|---|---|---|
| [allenai/PeerRead](https://github.com/allenai/PeerRead); Kang et al., *A Dataset of Peer Reviews (PeerRead)*, NAACL-HLT 2018, [paper](https://aclanthology.org/N18-1149/) | Commit `9bb37751781a900cee9e74ec3105997732c8e8e5`; source README, initial 18-paper pilot and same-year 100-paper expansion (427 review metadata records, 100 selected parsed manuscripts); source paths and verified Git blob/SHA-256 hashes in local manifests | No root LICENSE verified. Repository/paper describe research use; no redistribution or general reuse license is inferred. Raw files unchanged. Derived exports select section text, drop unused identity fields, deduplicate exact projected review entries, separate labels and freeze paper groups. Per-article/review rights and stage uncertainty remain. |
| [Daoze/ReviewRebuttal](https://huggingface.co/datasets/Daoze/ReviewRebuttal); Zhang et al., *Re²: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions*, 2025, [paper](https://arxiv.org/html/2505.07920v1) | Revision `0171bacee24a9d405792a32231dabcdb3788b365`; source card, file metadata and 18,641,943-byte `REVIEWS_test.json` only | Card declares Apache-2.0; this does not replace underlying manuscript/review rights. Raw JSON/card unchanged; a local audit summarizes counts/fields. No manuscript archive or train JSON downloaded. No model training or redistribution. |

No source code or assets from `Jayyeason/get-OpenReview` were used; its affiliation with the dataset authors was not verified. See [PUBLIC_DATA_PILOT.md](docs/PUBLIC_DATA_PILOT.md) for academic references, source boundaries and local receipts. Data ingestion itself does not implement rule learning; separate exploratory score results are documented in the technical reports.

## Public OpenReview ingestion extension (2026-10-23)

The optional public API v2 importer is authored standard-library integration code. [OpenReview's official retrieval documentation](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc) and [Python client interfaces](https://github.com/openreview/openreview-py/blob/master/openreview/api/client.py) were consulted read-only; no SDK source, executable scripts or third-party package were copied, installed or run. Source records are not relicensed as MIT. Per-article license labels, comment/configuration terms and metadata rights remain distinct; see [OPENREVIEW_IMPORT.md](docs/OPENREVIEW_IMPORT.md) and [source terms](https://openreview.net/legal/terms).

The first submission API query was denied HTTP 403 and access stopped. Only a public venue-configuration projection was retained through that API. The independently released data above are a separate source. Future actual API records belong under gitignored `data/openreview/` and must not be included in a source-code publication by default.

Software/license inspection date: **2026-10-23**. Documentation, source text and licenses were inspected read-only. Separately, the bounded public-data files above were retained locally. No downloaded repository scripts were executed, no upstream software was installed, and no model weights were downloaded. Attribution is not a substitute for permission. Original integration code in this folder is MIT licensed; source records and model weights are not relicensed by this project.

## Actual code adaptation: LSAR-public

- Repository: [cgpan/LSAR-public](https://github.com/cgpan/LSAR-public).
- Exact inspected revision for the adaptation: **`ba6ccbb85ddd2cc758c9c3451d3c234e398d3579`** (abbreviated `ba6ccbb`). Commit identity verified from [GitHub's commit patch header](https://github.com/cgpan/LSAR-public/commit/ba6ccbb.patch); source and license rechecked at this revision.
- License: [MIT at that commit](https://github.com/cgpan/LSAR-public/blob/ba6ccbb85ddd2cc758c9c3451d3c234e398d3579/LICENSE), Copyright (c) 2026 Chenguang Pan. Full required notice is preserved in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- Upstream component: [`scripts/calibration_analyze.py`, `_pctl`](https://github.com/cgpan/LSAR-public/blob/ba6ccbb85ddd2cc758c9c3451d3c234e398d3579/scripts/calibration_analyze.py), the small lower-order-statistic percentile helper (raw source lines 72–75, one-based).
- Destination: `journal_prefs/engine.py`, `accepted_percentile`.
- Modifications: renamed function/arguments/locals; added nonempty-input and quantile bounds checks; added attribution/docstring. Retained the sorting and floored percentile-index calculation. No other LSAR code, prompts, venue cutoffs, data, dependencies, or reported results are included.
- Conceptual contribution: published-history score anchoring motivated an optional threshold mode. This MVP instead anchors on accepted TRAIN **human mean scores**, not LSAR's model-scored published-paper corpus. This does not validate a threshold or acceptance probability.

## Interface/concept references; no copied implementation

| Upstream | Evidence inspected and license | Contribution here | What was not reused |
|---|---|---|---|
| [IgorDouven/FeedBackAndForth.jl](https://github.com/IgorDouven/FeedBackAndForth.jl) | [README](https://github.com/IgorDouven/FeedBackAndForth.jl/blob/main/README.md), [module exports](https://github.com/IgorDouven/FeedBackAndForth.jl/blob/main/src/FeedBackAndForth.jl), [MIT license](https://github.com/IgorDouven/FeedBackAndForth.jl/blob/main/LICENSE), Copyright (c) 2026 Igor Douven | Replaceable model/provider boundary and separately supplied venue context | No Julia code, prompts, cloud auto-detection, discussion panel, or cost tables copied |
| [FilippoScaramuzza/agentic-reviewer](https://github.com/FilippoScaramuzza/agentic-reviewer) | [README](https://github.com/FilippoScaramuzza/agentic-reviewer/blob/main/README.md), [scope-explorer prompt](https://github.com/FilippoScaramuzza/agentic-reviewer/blob/main/prompts/scope-explorer.txt), [MIT license](https://github.com/FilippoScaramuzza/agentic-reviewer/blob/main/LICENSE), Copyright (c) 2026 Agentic Reviewer Contributors | Keep journal scope/criteria as explicit external context | No agents, prompt text, workflow scripts, conversion tools, downloads, or skill instructions adopted |
| [EigenTom/ReviewGrounder](https://github.com/EigenTom/ReviewGrounder) | [README and repository layout](https://github.com/EigenTom/ReviewGrounder/blob/main/README.md); root `LICENSE` URL returned 404 and no root license was verified | Comparison reference for backend separation and evidence-grounded review; our evidence contract was independently authored | **No code, prompts, configuration, models, or assets reused.** Permission not established; do not vendor it without a new license audit |

These conceptual references were inspected on their then-current default branches (`main`); no code was copied from them, so no copied-component revision is claimed. Links may move. The pinned LSAR and separately documented ICAI filtering components are the source-code adaptations. No claim is made that these projects provide the complete institutional import/retrieval/calibration/chronological-evaluation combination, or that our combination is novel.

## Authored components and gaps

Authored for this local prototype: input validation, JSONL fixtures, token retrieval, adapter protocol and HTTP client, deterministic mock, affine fit/validation selection, chronological split enforcement, frozen-state checks, CLI, tests and docs. Python standard-library runtime only; no third-party runtime package is vendored.

Deferred: automatic rubric extraction, PDF/document parsing, sophisticated retrieval, complete model/endpoint reproducibility, fine tuning, scientific claim verification, real institutional empirical validation, multi-user security, public precheck, and production journal integration. Upstream benchmarks/results are not results of this prototype.

## Review prompt sources inspected on 2026-10-23

The new JSON prompt loader, field mapping, snapshots and neutral institution example are independently authored; no third-party prompt implementation was copied or executed.

| Source | Permission/version checked | Use here |
|---|---|---|
| [OpenReview official ICLR 2017 form](https://github.com/openreview/openreview-scripts/blob/ac40cf9121585c4beb75ada04099e83c20b260ee/Legacy/venues/ICLR.cc/2017/conference/process/submissionProcess_conference.template#L147-L164) | Commit `ac40cf9121585c4beb75ada04099e83c20b260ee`; file SHA-256 `927448597540d1031eb4909374da78066384994c79d5622c451f2c9c39df676f`. No root license was found in that snapshot. | Historical factual criteria and category meanings independently summarized in `prompts/iclr2017_official_form.v1.json`, with source metadata. No executable code or verbatim upstream prompt incorporated/relicensed. Local gitignored source-audit copies retain original text solely as evidence. Historical-context instruction and JSON format are project additions. This is a review-form condition, not a claimed recovery of a complete prose reviewer guide. |
| [SakanaAI/AI-Scientist, perform_review.py](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py) | Current `main` was inspected read-only. [Current LICENSE](https://github.com/SakanaAI/AI-Scientist/blob/main/LICENSE) is the custom AI Scientist Source Code License v1.0, December 2025; do not assume an older permissive license applies to this snapshot. | Comparison reference for structured review forms. No code/prompt text copied. Negative/positive reviewer personas, NeurIPS categories, few-shot examples, ensembles and reflections were not adopted. |
| [maxidl/OpenReviewer Space, app.py](https://huggingface.co/spaces/maxidl/openreviewer/blob/main/app.py) | Read-only Space source; [README metadata](https://huggingface.co/spaces/maxidl/openreviewer/blob/main/README.md) declares Apache-2.0. The inspected app explicitly defaults to ICLR 2025. | Conceptual reference for editable review-template fields. No code or prompt text copied; no model weights, parsing stack, GPU service or app executed. Its 2025 form is not represented as 2017. |

All runtime templates record their own source/version metadata; factual historical categories do not license the surrounding source code. Existing required LSAR/ICAI notices and the project license remain unchanged. See [prompt interface and source distinctions](docs/PROMPT_INTERFACE.md).

## Prepared AutoCalibrate method adaptation and cached nonlinear comparison

Liu, Yuxuan, et al. *Calibrating LLM-Based Evaluator*. LREC-COLING 2024, pp. 2638–2656, [Algorithm 1](https://aclanthology.org/2024.lrec-main.237.pdf), is the methodological source for `journal_prefs/autocalibrate.py` and the prepared phased pilot. The paper is CC BY-NC 4.0; official code and its code license were not verified. These modules and prompts are independently authored from the method description. No upstream code, verbatim prompt or paper asset was copied, executed, vendored or relicensed. Manuscript-score data, negative MAE, separate validation selection and Codex sampling settings are documented task adaptations, not an official-code reproduction. Two preliminary Sol drafts were generated locally; no rule-conditioned scoring completed. Luna comparison remains unrun. Local experiment artifacts are excluded from this source release.

`journal_prefs/nonlinear_scores.py` contains independently authored standard-library implementations of a small ridge-regression solver and bounded one-dimensional squared-error tree. Official scikit-learn documentation was a conceptual reference; no scikit-learn source was copied. An incompatible installed binary stack was not repaired or used to execute those models. The comparison uses cached scores only; it is neither multidimensional manuscript prediction nor a new LLM experiment. See [cached comparison](docs/EXPERIMENTS.md) and [prepared method](docs/EXPERIMENTS.md). All previously required upstream notices and the MIT license remain unchanged.
