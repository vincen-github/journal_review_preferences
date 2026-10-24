# Public OpenReview import: scope, protocol, and current blocker

The HTTP 403 below records the direct API attempt only. A later, separately authorized author-released PeerRead pilot supplied 18 paired manuscripts, with version readiness still unverified; see [PUBLIC_DATA_PILOT.md](PUBLIC_DATA_PILOT.md). The denied API request was not retried.

This is a small local data-ingestion extension, not preference induction or a model experiment.

## Actual access result on 2026-10-23

The unauthenticated ICLR 2024 venue configuration GET succeeded:

`https://api2.openreview.net/groups?id=ICLR.cc%2F2024%2FConference`

The next request, to inspect two public submission records, returned **HTTP 403 Forbidden**:

`https://api2.openreview.net/notes?invitation=ICLR.cc%2F2024%2FConference%2F-%2FSubmission&limit=2&select=id%2Cforum%2Cnumber%2Ccdate%2Ctcdate%2Cmdate%2Ctmdate%2Cpdate%2Codate%2Clicense%2Creaders%2Ccontent.title%2Ccontent.venueid%2Ccontent.venue%2Ccontent.pdf`

Outcome: **denied**, not pending or uncertain. The response alone does not establish why access was refused. No further API data requests, alternate endpoints, logins, or access workarounds were attempted. The importer has not been run live after this denial. Actual delivered paper/review/PDF counts are all **0**. The successfully observed public group fields are recorded as a labelled response projection in the gitignored local access receipt; this is not a manuscript dataset or a complete raw-response archive. A local parsing smoke check confirms that projection can be read by the new adapter. Actual submission/review parsing remains blocked; its tests use authored API-shaped fixtures.

Local receipt: `data/openreview/iclr_pilot_20261023/manifest.json`, `access_probe.json`, `venue_group_2024.observed_projection.json`, and `actual_group_parse_smoke.json`. These files and any future corpus remain gitignored. They are not covered by the source-code MIT license. No public corpus has been published or relicensed.

## Import interface

`journal_prefs/openreview_import.py` implements a standard-library API v2 client and independent parsing/sampling functions. It uses only unauthenticated GETs to official note/edit/venue-group/form endpoints. It never reads API credentials, retrieves profiles/members, discovers reviewer identities, follows redirects, uses proxies, retries denials, or falls back to another host. Requests are spaced at least 1.5 seconds apart by default, with a bounded request budget. Any HTTP error, including 401/403/429, stops that client for the whole run. It records partial progress and the denied URL/status in a manifest.

The optional CLI is `import-openreview`; the separate flags `--allow-public-network` and `--ack-source-terms` are mandatory. Without both it makes no requests and creates no output. It is never called by `demo`, `review`, or `backtest`. The included `configs/openreview_iclr_pilot.json` describes ICLR 2024 and 2025, a seed of 20261023, and a cap of 300 submission metadata records per year. Live connectivity and paper-scale interoperability have not been validated. Do not rerun the denied action as a workaround; resolve permitted access first.

## Score-balanced, exclusion-filtered sampling

The future live import first reads each venue-year's public official review form. It must establish numeric category labels from that form; missing, ambiguous, or incompatible forms cause a stop rather than a guessed scale. Each year is sampled independently. Raw score strings, numeric values, confidence, stage invitations and creation/modification/publication timestamps are retained. No cross-year normalization or pooling is performed.

Stage: the **latest public Official_Review note at retrieval**, including public revisions. This is not a reconstruction of initial, pre-discussion ratings. A paper's observed mean is the unweighted mean of its valid overall recommendation values. Missing ratings are flagged and never replaced by zero. Papers with missing official reviews or any missing/invalid recommendation are excluded from this pilot; their metadata and reasons remain in the frame.

The raw score range is divided into three equal-width bands: low `[min, edge1)`, middle `[edge1, edge2)`, high `[edge2, max]`. For a *synthetic fixture* scale spanning 1–10, these edges are 1, 4, 7, 10. These are not yet verified as the live 2024/2025 forms. Within each year, the target is three low, three middle and three high papers (18 across two years). The middle target is split into one boundary case and two other middle papers. Boundary means lie inclusively between the official marginal/borderline/weak-accept-or-reject category values. A fixed seed samples uniformly within these disjoint sub-strata. A shortfall is reported without relaxing filters or reallocating papers from another band. The manifest records eligible frame sizes, actual selections, shortfalls, probabilities and inverse-probability weights conditional on this limited eligible frame.

For the requested filter, **every retrieved public official review** must be in an explicit above-borderline accept category, with no missing ratings, and the final status must be ordinary rejection. Such records receive `unanimous_high_score_rejected` in the general parsed data and are excluded **only during pilot selection**. Individual raw scores and review-note IDs remain in the frame for auditing. This public-only import cannot establish whether undisclosed reviews exist. For the synthetic test scale, high means 8 or 10; 6 is marginal and does not qualify. A high mean with a lower individual score is not excluded by this rule. Live categories must first be verified from each official form. The label implies no fraud or mistaken decision; supplementary criteria may explain the outcome.

Withdrawn, desk-rejected, ordinary accepted/rejected and unknown/missing decisions remain distinct. Only ordinary accepted/rejected papers with official decision notes and complete recommendations enter pilot selection. Status and exclusion counts are retained. The metadata frame uses submission number order up to the configured cap, and records API-reported totals and whether coverage is complete. The sample is **score-balanced and exclusion-filtered**, not representative of acceptance probability, rejection prevalence, or general performance. A future independent representative test must not adopt this exclusion filter. Frame weights cannot undo the outcome filter or repair incomplete coverage.

## Versions and evaluation readiness

The mutable current PDF link is never used as the initial manuscript. The adapter inspects public edit metadata for an immutable PDF path from a non-revision submission edit earlier than the earliest official review creation timestamp. It records a reference only, not a claim that the attachment has been downloaded or its public-visibility history proved. Camera-ready/revision edits, mutable links, missing edit timestamps, inaccessible history and missing initial files are flagged.

This metadata pilot downloads **no PDFs** and produces **no evaluation-ready history rows**. Even when an immutable pre-review reference is identified, full text, article-specific usage rights, historical availability and extraction fidelity still need verification before entering backtests. No abstract, current PDF or camera-ready text is silently substituted. Public-review revisions can also discuss revised manuscripts; that alignment requires separate verification. Preference/rubric induction remains unimplemented.

## Future real before/after case study

There are currently **zero data-ready real candidates** because access stopped before any submission records arrived. No improvement example or human-review comparison is reported. Once permitted records and initial versions are available, the evidence chain should be: derive and freeze a rule from older TRAIN examples; use the same fixed model on an untouched later manuscript with and without that rule; identify a concrete manuscript-grounded issue; compare against its actual human review; and report the whole later cohort alongside individual cases. Keep the rule, model, target text/version and prediction inputs fixed and report failures as well as successes. The score-balanced, exclusion-filtered induction pilot cannot serve as the independent representative test. None of this has been run, and no model calls or preference-induction implementation are included in this ingestion step.

## Source licenses and attribution

OpenReview's [Terms of Use](https://openreview.net/legal/terms), checked 2026-10-23, retain article copyright with authors and direct users to each article's license field; a missing article license is unresolved. Their comment/configuration section specifies CC BY 4.0, and metadata uses CC0. We retain available license labels and forum/note links for attribution, keep article rights separate, and do not treat public access as permission to relicense the corpus. The pilot is limited to local research inspection; redistribution and model training are outside this step.

References inspected read-only:

- [Official note/reply retrieval guide](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc).
- [Notes and field-level readership](https://docs.openreview.net/getting-started/objects-in-openreview/introduction-to-notes).
- [Edit history objects](https://docs.openreview.net/getting-started/objects-in-openreview/introduction-to-edits).
- [Official Python client's documented API interfaces](https://github.com/openreview/openreview-py/blob/master/openreview/api/client.py), consulted as an interface reference; no SDK code was copied or executed.
- [ICLR 2024 reviewer guide](https://iclr.cc/Conferences/2024/ReviewerGuide) and [2025 reviewer guide](https://iclr.cc/Conferences/2025/ReviewerGuide), which describe review revisions; they were not treated as verified numeric rating forms.

This extension is authored integration code. Upstream notices and AI-assistance disclosure remain intact.
