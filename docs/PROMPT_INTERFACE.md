# Replaceable review prompts

Three interfaces serve different purposes:

| Interface | Changes |
|---|---|
| Model adapter | Local/cloud provider, model ID and generation settings |
| Preference algorithm | External fitted/configured artifacts; current main method is absolute-score calibration |
| Prompt JSON file | Review criteria, rating anchors and structured output, without editing Python |

Copy [the neutral institution template](../prompts/institution_neutral.v1.json), edit it and increment `version`. Its `instructions` contain the institution's standards; `score_range` and `score_anchors` define the score meanings; `output_schema` defines JSON output. `response_fields` maps your top-level names to `score`, `review`, `evidence`; see [custom field names](../examples/custom_prompt.json). Evidence items retain `quote`/`comment` so manuscript quotations can still be checked. Arbitrary XML/text outputs are not supported.

For an already-running local compatible model service, add this option to the existing `review` or `backtest` command:

```powershell
--config configs/local.example.json --prompt prompts/institution_neutral.v1.json
```

Use your own data/rubric/state/output arguments as documented in the quick start. The prompt and institution rubric must have matching numeric ranges and consistent meanings. The deterministic offline mock does not interpret language prompts and rejects this configuration. No model is installed or started by loading a prompt.

The local/cloud adapter reads `prompt_template` from configuration; the CLI resolves `--prompt` into that embedded object before hashing configuration. It checks mapped score range/type and the canonical review/evidence fields; it is not a complete JSON Schema implementation. Provider support for schema constraints varies. Existing transport and cloud authorization gates remain unchanged.

Runs save `prompt_snapshot.json` with the actual template, template hash, rendered instructions and instructions hash. Chronological backtests also retain that snapshot inside frozen state; changing the template changes the model-configuration hash. The public-paper Codex wrapper accepts the same `--prompt` option and snapshots it before the first request. Its legacy no-`--prompt` behavior remains available for reproducing the earlier generic baseline.

## Default sources and historical ICLR example

The neutral template is project-authored. Structured review forms and editable venue templates were inspected in [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/perform_review.py) and [OpenReviewer](https://huggingface.co/spaces/maxidl/openreviewer/blob/main/app.py); no code or prompt text was transplanted. The former defaults to a negative reviewer persona and a NeurIPS form; its current source license is custom. OpenReviewer's Space declares Apache-2.0 but explicitly defaults to ICLR **2025**. That is its runtime inference template; its training pipeline was not audited here. Neither supplies a 2017 rubric for this experiment.

[The ICLR 2017 template](../prompts/iclr2017_official_form.v1.json) independently summarizes factual categories in the [official historical review-form source](https://github.com/openreview/openreview-scripts/blob/ac40cf9121585c4beb75ada04099e83c20b260ee/Legacy/venues/ICLR.cc/2017/conference/process/submissionProcess_conference.template#L147-L164). Quality, clarity, originality and significance are the criteria; 5/6 straddle acceptance, 7 recommends acceptance, and 8/9/10 describe increasingly high ranks among accepted papers. These differ from the original generic prompt. A separate complete prose reviewer guide has not been located. The explicit historical-knowledge instruction is our addition, using the [official review timeline](https://iclr.cc/archive/www/doku.php%3Fid%3Diclr2017%3Acallforpapers.html). It is an aid, not proof of temporal isolation. Sources/commit/hash and this distinction are stored in the template.

No root license was found in the audited historical repository snapshot. No executable source or verbatim template was incorporated or relicensed; the local, gitignored audit copies are read-only source evidence. See [upstream notices](../THIRD_PARTY_NOTICES.md).
