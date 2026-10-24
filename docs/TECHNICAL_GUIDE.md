This framework has a [general assessment interface](ASSESSMENT_INTERFACE.md) with optional scores, categorical recommendations, narrative and custom fields. Institutions choose their own model and alignment algorithm. The chronological numeric scoring workflow below is one runnable research example, not a required journal protocol.

# Technical guide

Current main method: [absolute-score prediction and calibration](EXPERIMENTS.md). Replaceable preference interfaces are retained; ICAI and the simplified rule scaffold are inactive references, not default algorithms.

Later real scoring results and the separate Codex route are in [EXPERIMENTS.md](EXPERIMENTS.md). Descriptions of mock/HTTP-adapter testing below refer to those tests, not a claim that no later model call occurred.

Run commands from the project root. This page records implementation details; the short project introduction is in [README.md](../README.md).

## Current prototype — implementation status

Version 0.1.0 is a local CLI skeleton under the MIT license. The bundled demonstration uses 16 fictional manuscripts and a deterministic keyword mock; it verifies software plumbing, not empirical alignment with editors or reviewers. The mock ignores historical labels. Its only data-fitted component is an optional affine mapping from raw scores to human mean scores, selected using validation error. Criteria, weights and default thresholds are manually configured; automatic rubric induction is not implemented, and no base-model weights are trained.

The prototype includes historical-data import, training-only retrieval, chronological evaluation, model interfaces, and a reserved future fine-tuning interface. It provides no validated acceptance probabilities. A public precheck interface, live service, and deployment are future possibilities. The technical sections below describe the current behavior and limits.

## Run the offline demonstration on Windows

Python 3.10+ is sufficient. No packages, model downloads, API keys, GPU, or training are required. Open PowerShell in this folder:

```powershell
python -B -m journal_prefs demo --out my_first_demo
python -B -m unittest discover -s tests -v
```

Or double-click `RUN_DEMO.cmd`. Each double-click creates a new dated output folder. A tested example run is included in `demo_output/`; open [demo_output/report.md](../RUNNING.md) to see it. Every synthetic score/metric is explicitly labelled. The mock deliberately ignores historical labels; it tests the interface, retrieval plumbing, and score calibration, not reasoning quality.

The demo has 16 authored fictional records: 8 TRAIN submissions in 2023, 4 VALIDATION in 2024, 4 TEST in 2025. Its rubric is fixed in 2022. Outputs include:

- `report.md` and `report.json`: split metrics, candidate selection, provenance hashes, false negatives and false positives.
- `predictions.json`: generated text, exact manuscript excerpts, retrieved training IDs, input hashes, and `human_review_required=true` for every score.
- `model_state.json`: frozen external calibration and corpus/rubric/model configuration hashes. No model weights.

Output directories must be new; existing outputs are never overwritten. Use another name when rerunning.

## Public OpenReview ingestion pilot

An optional, separately gated API v2 import adapter now supports a small public ICLR pilot. It retains public reviews, raw scores/confidence, meta-reviews, decisions, timestamps and provenance; samples each venue-year separately across score bands with a fixed seed; and applies the requested unanimous-high-score/rejected exclusion only to pilot selection. The general parser retains those records. It does not infer institutional preferences.

**Direct API status:** the ICLR 2024 venue configuration was publicly readable, but the first submission-record API request returned **HTTP 403 Forbidden**. Retrieval stopped immediately; that request was not retried. No paper records or PDFs were collected through that API. The separately authorized author-released dataset pilot below subsequently supplied real paired records. See [docs/OPENREVIEW_IMPORT.md](OPENREVIEW_IMPORT.md) for the denied action and API protocol. The API command requires explicit public-network and source-terms flags; it is never invoked by offline model inference.

The bounded profile targets **18 papers across ICLR 2024 and 2025**, with three low-, three middle-, and three high-score papers per year, a boundary sub-stratum, fixed seed 20261023, and a metadata frame cap of 300 per year. It uses the latest public official-review snapshot, including revisions, not reconstructed initial ratings. Each year's official rating form must be verified before sampling; raw scales are neither guessed nor pooled. Missing scores are not zero, and stratum shortfalls do not relax exclusions. Frame counts, reasons, conditional selection probabilities and weights go in the manifest. This balanced, exclusion-filtered pilot is not representative of acceptance probability or prevalence and cannot serve as an unbiased test set.

The importer identifies only public, immutable PDF references whose edit metadata precedes every official review; it downloads no PDFs. Current, revised, camera-ready text and abstracts are not substituted for initial full text. Missing versions or review creation times remain unverified and excluded from evaluation-ready inputs. Local records and access receipts live under gitignored `data/openreview/`, retain their source rights, and are not relicensed or redistributed as MIT code.

## Author-released paired-data pilot

The official **PeerRead ICLR 2017** release supplied **18 real manuscripts paired with 55 deduplicated, recommendation-bearing reviews: 7 accepted and 11 rejected papers**. Learning, validation and test each contain six distinct papers, with two low-, two middle- and two high-score papers per group, including boundary cases. Selection uses seed 20261023 and a fixed 96-paper metadata frame. Original source paper partitions are preserved; this is a same-year paper-group split, not a deployment-time chronological backtest. Inputs exclude human reviews, scores and decisions; labels are stored separately.

The learning selection excludes unanimously high-rated rejected papers under a provisional numeric definition (every recommendation at least 8/10); none occurred in its 48-paper frame. The official 2017 category-label boundary is **not verified**, so that part of the requested semantic filter remains unresolved. Validation and test do not apply the exclusion. The balanced pilot is not representative of acceptance rates or unbiased population metrics.

**Paired ingestion is complete; evaluation readiness is not.** Per-paper pre-review manuscript versions and initial/final score stages could not be certified from this release, so all 18 remain marked unverified and the evaluation-ready count is **0**. No dates were invented. No model ran on these records, and automatic preference induction remains unimplemented. The test IDs identify candidates for a future real case, not demonstrated improvements: freeze a rule using learning/validation only, compare the same model with and without it on untouched manuscripts, then compare grounded findings with human reviews and the full test cohort.

A separate bounded inspection retained the **Re²** author release's small review JSON (1,000 paper records, including 655 ICLR records from 2020–2023). It has no manuscript bodies; the large paper archive and training JSON were not downloaded. Actual data remain gitignored and are not included in the code's MIT license. PeerRead has no verified root LICENSE at the pinned revision; the Re² card's Apache-2.0 declaration does not replace underlying article/review rights. Old public papers may have appeared in model pretraining. Exact source commits, hashes, frozen IDs, score stages, sampling details and local paths are in [docs/PUBLIC_DATA_PILOT.md](PUBLIC_DATA_PILOT.md).

## Bring institution-owned historical records

Prepare UTF-8 JSONL, one manuscript version per line, following [docs/DATA_SCHEMA.md](DATA_SCHEMA.md). It includes manuscript text, version/time, text availability, human reviews and scores, decision/time, institutional ownership assertion, and an explicit synthetic flag. Import validation accepts version history; this minimal backtest requires selecting **one frozen pre-review version per manuscript**. It rejects duplicate manuscripts instead of risking version leakage. These offline commands do not download a public corpus.

```powershell
python -B -m journal_prefs validate --data examples/history.synthetic.jsonl --rubric examples/rubric.json
python -B -m journal_prefs backtest --data examples/history.synthetic.jsonl --rubric examples/rubric.json --train-before 2024-01-01T00:00:00Z --validation-before 2025-01-01T00:00:00Z --out my_backtest
python -B -m journal_prefs review --data examples/history.synthetic.jsonl --rubric examples/rubric.json --paper examples/submission.synthetic.json --state my_backtest/model_state.json --out my_review
```

An institution configures `rubric.json`: venue, criteria/descriptions/weights, score range, decision labels, threshold, and when the rubric was available. No ICLR-specific rule exists. Automatic rubric extraction is not implemented. For credible retrospective evaluation, use rules documented before the evaluation period, or derive and freeze them using TRAIN evidence only. Do not reconstruct a rubric from held-out reviews and then call the test untouched.

## What is fitted, selected, and tested?

1. Split submissions chronologically with explicit exclusive cutoffs, and ensure TRAIN labels were available before the TRAIN cutoff and VALIDATION labels before the TEST cutoff. Delayed labels cause a clear error; revise the cohort or cutoff.
2. Retrieve examples only from the frozen TRAIN corpus of this venue, excluding the same manuscript. All example reviews/decisions must be available **strictly before** the query. TRAIN predictions therefore use earlier TRAIN examples; VALIDATION/TEST can use the full eligible TRAIN corpus. That context-size difference is a limitation.
3. Optionally fit a positive affine score mapping on TRAIN model scores and human mean scores. `--calibration none` disables fitting; `affine` fixes that choice; `auto` selects identity versus affine using VALIDATION score MAE. There is no refit on VALIDATION+TRAIN.
4. Freeze parameters before TEST inference. TEST reviews and decisions are excluded from all model requests, retrieval, fitting, and candidate selection; they are used afterward only for metrics. VALIDATION is selection data, not untouched evidence.

The default threshold is institution-configured. Optional `threshold_mode="accepted_quantile"` takes a lower order statistic of accepted TRAIN **human** scores; it is an exploratory external score anchor, not a probability or an editorial verdict. It differs from LSAR's model-scored published anchors. Sparse or accepted-only data cannot establish false-negative performance.

Rules and examples are provided anew through the adapter request. The base LLM does not permanently remember institutional preferences. Changes to the corpus, rubric, or model configuration require new calibration and an untouched test cohort. The frozen-state review command rejects IDs already used in fit/selection/test. This is not an immutable evaluation ledger; manual edits, duplicate content under new IDs, endpoint model changes, and repeated test inspection remain governance responsibilities.

## Models, privacy, and extension points

| Configuration | Behavior |
|---|---|
| `configs/offline.json` (default) | Deterministic in-process mock. No network requests, DNS, telemetry, keys, cloud fallback, or model weights |
| `configs/local.example.json` | Optional client for an already-running OpenAI-compatible loopback server at literal `127.0.0.1` or `[::1]`. Does not install/start a model. No proxies, redirects, API key lookup, or cloud fallback |
| `configs/cloud.DISABLED.example.json` | Disabled template. Requires an explicit HTTPS endpoint/model/key-variable name, `allow_private_data_transfer=true`, and `--allow-cloud` on each CLI run |

The local adapter lets institutions choose the endpoint and open model ID, plus supported `temperature`, `max_tokens`, and optional JSON-object `response_format` settings. It parses structured responses and reports malformed envelopes clearly. Tests cover forwarding and errors using in-memory fake responses; no real model server or cloud provider was contacted. Ollama/vLLM-style API compatibility is an intended integration path, not a tested deployment. Compatible APIs can still differ in parameters and model capabilities. See [the interface notes](PUBLIC_DATA_PILOT.md#interfaces-and-tests).

Cloud mode discloses its destination and transmission of target text, rubric, and selected historical manuscript excerpts/reviews/scores/decisions. It may incur provider charges. Do not enable it for confidential records without institutional authorization. No actual API keys are included or needed for this delivery.

Local-only operation does not guarantee full confidentiality: the operating system, an independently configured local model server, synced folders/backups, other processes, and file access permissions remain outside this prototype's control. Outputs contain excerpts and review text, so treat output directories as sensitive. There is no encryption, authentication, redaction engine, or access-control service. Keep real data outside source control. Prompt instructions and exact-quote checks do not solve prompt injection or verify scientific truth.

`journal_prefs/models.py` defines `ModelAdapter.review(request)` and the reserved `FineTuningExtension.fit(...)` protocol. Replace the adapter in Python or configure the HTTP adapter; there is no automatic loading of downloaded plugins. A future institution-approved fine-tuning component could implement the protocol with separately consented data and compute. No fine-tuning implementation, training command, GPU launch, or background monitoring exists here.

## Reuse and limits

### Project contributions and AI assistance

Wensen Ma contributed the trend hypothesis, needs identification, project idea and direction decisions; he did not write project code. The AI assistance below covers implementation, tests and documentation as actually performed. The market forecast remains unvalidated.

- **dot (OpenAI AI assistant) — concept and coordination support:** helped iteratively clarify the idea, discuss architecture and methods, synthesize research, coordinate implementation work, and draft/refine the project narrative and documentation.
- **Codex coding agents — implementation and review support:** implemented the local CLI and data interfaces, authored unit/integration/privacy tests, and performed independent chronology and privacy code reviews. Concrete findings and verification are recorded in [TEST_RESULTS.md](../TEST_RESULTS.md).

These entries describe human decisions and AI tool assistance. Scientific conclusions and institutional evaluation judgments require human verification. Upstream software contributions and licensed code reuse are documented separately in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); AI assistance does not replace or change those attributions or licenses.

We welcome you to use, improve, and build on this project. If its ideas, design, or implementation inform your research or work, we would appreciate an acknowledgment and a link to [Journal Review Preferences](../README.md) so that readers can trace the contributions. This is a voluntary scholarly attribution request; the applicable license obligations remain those stated in [LICENSE](../LICENSE) and the upstream notices.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for exact provenance, licensing, modifications, and upstream contributions. A tiny MIT-licensed LSAR percentile helper was adapted; other integration code is authored for this prototype. Other projects inform interfaces or remain comparison references; no complete upstream combination was verified and no novelty claim is made.

Remaining work includes real institutional data agreements and de-identification, robust PDF/version ingestion, larger prospective and subgroup evaluations, review-quality evaluation, model reproducibility, retrieval quality, uncertainty/abstention, appeals, and measured false-negative/novelty safeguards. Lexical retrieval and affine scores are deliberately small baselines. Historical decisions can encode bias. Real LLMs may already have encountered public evaluation papers during pretraining; chronological splits alone do not remove that risk. No quality, fairness, acceptance, or workload-saving guarantee is offered.
