# ICAI: original-method audit and data gap

**Current direction:** ICAI is a retained method reference, not the default. The user chose simple absolute-score calibration and stopped pair construction. Existing PeerRead records do not directly contain ICAI's explicit text-pair preferences. No cross-reviewer absolute scores or accepted/rejected manuscripts have been converted into pairwise labels, and a higher manuscript recommendation is not treated as a preference for a better-written review.

## Inspected primary sources

Findeis, Arduin, Timo Kaufmann, Eyke Hüllermeier, Samuel Albanie, and Robert Mullins. *Inverse Constitutional AI: Compressing Preferences into Principles*. arXiv:2406.06560, 2024; ICLR 2025. [Paper](https://arxiv.org/abs/2406.06560), [v1 method](https://arxiv.org/html/2406.06560v1#S3).

Author repository: [rdnfn/icai](https://github.com/rdnfn/icai), inspected commit **`cb88e6588e9a01e8c1f216ea3cfe1b7461ddee7d`**, package metadata version **0.4.8**. README, root LICENSE, pyproject, algorithm main/proposal/clustering/voting/filter, configuration/prompts, standard input loader and experiment orchestration were read as source text. No upstream package, scripts, models or experiment were installed or executed.

The root LICENSE is **Apache-2.0**. The pyproject also contains a conflicting MIT classifier; this project retains and follows the explicit root Apache-2.0 terms for the adapted component. No root NOTICE file was listed at this revision. See [reuse notice](../THIRD_PARTY_NOTICES.md) and [retained license](../licenses/ICAI-Apache-2.0.txt).

## Original inputs, steps and objective

The author's CSV requires `text_a`, `text_b`, `preferred_text` (`text_a` or `text_b`); `input` is optional. The common example compares responses to the same prompt. Ties are not used by this input path. This application additionally requires a common manuscript/task, paper-group ID and annotation provenance, so its meaning is explicit.

1. Generate candidate natural-language principles from labelled comparisons.
2. Embed principles and cluster them with K-means.
3. Sample a representative principle per cluster (or use the source's configured summary variant).
4. Ask the model which response each principle supports; compare these votes with the original preference annotations.
5. Filter principles using reconstruction support and relevance; retain a compact constitution.

The paper's objective is agreement with original pairwise annotations, equivalently lower reconstruction loss under a compact principle budget. It is not manuscript-score OLS, direct acceptance prediction, or a free-form validation rewrite loop. The inspected implementation uses vote predicates such as `for > against`, valid-vote/relevance thresholds and ranking by `for - against`. Its current defaults include three reannotation seeds; the repository warns that this costs more than older versions. Current source defaults are not claimed identical to the paper's v1 experimental settings.

## What was implemented here

- `icai_support.validate_pairs` and `export_upstream_csv`: explicit, same-manuscript pair validation and exact documented CSV columns; no inferred pair labels.
- `ICAIUpstreamAdapter`: a replaceable preference interface for an explicitly supplied approved original runner, versioned artifact, freeze/save/load and new-paper conditioning. An unconfigured runner fails clearly; there is no substitute algorithm fallback. The bridge does not itself execute proposal, embedding, clustering or voting.
- `filter_principles`: a small reviewed adaptation of the author's actual `algorithm/filter.py` vote predicates/ranking. Its Apache-2.0 provenance and modifications are recorded. This is a reused component, not a full reproduction.
- The generic model/provider interface and fixed three-arm comparison are independent of preference induction and affine score calibration.

The previously authored `EvidenceRules` fit/refine scaffold is retained solely as an **experimental interface scaffold, paused and not the production default**. Its synthetic demo exercises serialization and separation; it is not ICAI and was not used for a formal real-data experiment.

## Optional future faithful run, not the current task

The paper requires compared texts and a preference, not necessarily two assessments. One possible future institutional task would obtain explicit human pairwise judgments between two candidate assessments of the **same manuscript under the same task**. Keep all pairs/versions of a manuscript in one predeclared group. Comparing review quality would learn review-quality preferences; it must not be described as venue acceptance preferences without a corresponding annotation task. Score-derived weak preferences would be a separately labelled methodological adaptation with reviewer/scale/topic confounds, not original human pair labels; this project is not constructing them.

Run the pinned original implementation on learning pairs with an approved LLM and embedding backend. Use validation only for a small predeclared configuration choice, then freeze the constitution. Evaluate on the reserved manuscripts using the same base model/settings/common scoring form: generic, official guideline, guideline plus constitution. Scoring requests contain no target human reviews/scores/outcome; labels are joined only afterward for evaluation. Report every arm and all papers, including no improvement or deterioration.

**Outstanding:** real pairwise annotations; a reviewed/approved original-package runner with its dependencies and embedding backend; a historically matched official 2017 guideline. None is fabricated. The existing 18-paper pilot is score-balanced, manuscript/score stages remain unverified, and this development conversation already encountered held-out labels. Any current run is exploratory and cannot be advertised as blind or representative. No model-quality result has been produced.
