# Local input contract (schema version 1)

Use UTF-8 JSON. `validate` reads JSONL without calling a model. `examples/history.synthetic.jsonl` contains complete fictional records, and `examples/submission.synthetic.json` is an unlabelled target. Only the Python standard library is required; validation lives in `journal_prefs/data.py`.

One historical JSONL line has this structure (shown multiline for readability):

```json
{
  "schema_version": 1,
  "institution_id": "institution-owned-cohort",
  "venue_id": "your-journal",
  "manuscript_id": "paper-001",
  "version_id": "v1-before-human-review",
  "title": "Example title",
  "text": "Institution-owned manuscript text, available at the prediction time.",
  "version_time": "2023-03-01T09:00:00Z",
  "text_available_at": "2023-03-01T09:00:00Z",
  "submitted_at": "2023-03-02T09:00:00Z",
  "rights_confirmed": true,
  "synthetic": false,
  "reviews": [
    {"text": "Human assessment", "score": 6.0, "available_at": "2023-03-20T09:00:00Z"}
  ],
  "decision": {"label": "accept", "available_at": "2023-03-25T09:00:00Z"}
}
```

All listed fields are required for history. A new target has the same manuscript/rights/time fields and **omits** `reviews` and `decision`. `rights_confirmed` is an institution's assertion that it may use these records for this purpose; the program cannot establish ownership. Avoid reviewer identities and unnecessary personal data. Do not mark real data synthetic.

`manuscript_id` identifies the same work across versions; use stable identifiers. `(manuscript_id, version_id)` must be unique. `version_time` must not be after `submitted_at`. `text_available_at` must strictly precede the simulated submission, as must the rubric's `available_at`. Review/decision availability cannot precede submission. All timestamps require ISO 8601 timezones, such as `Z` or `+08:00`; comparisons normalize to UTC. Record the actual time information became available, not a later export timestamp or an invented early date.

Select one pre-review text snapshot per manuscript for a backtest. Do not use accepted camera-ready text, post-review revisions, appended human reviews, outcome-labelled filenames pasted into text, or response letters as a stand-in for a submitted manuscript. The program rejects duplicate manuscript IDs per run; it does not detect duplicate content under different IDs or hidden outcome information inside prose. Import validation permits multiple versions; automatic version selection and dependence grouping across related manuscripts are deferred.

Scores must be finite numbers within `rubric.score_range`, with higher values meaning stronger assessment. The MVP target is the unweighted mean of human overall scores per paper, not the scores of separate dimensions. Scores, text, and decision must be present; missing-data handling is not implemented. The two permitted decision labels come from the institution's rubric, not a hard-coded conference. For revision/withdrawal/desk-reject categories, define a defensible cohort/label mapping before evaluation; do not silently collapse them after seeing test results.

## Rubric and temporal protocol

`examples/rubric.json` demonstrates required fields: `schema_version`, `institution_id`, `venue_id`, `name`, `source`, `available_at`, `score_range`, `positive_decision`, `negative_decision`, `accept_threshold`, and nonempty `criteria`. Each criterion has unique `id`, a `description`, positive `weight`, and optional `keywords`. Keywords are consumed only by the synthetic mock. A real adapter receives descriptions and weights. `threshold_mode` is `configured` (default) or `accepted_quantile`, with optional quantile in `[0,1]` (default `0.25`). These are external institution-controlled rules; they do not modify the base model.

One run uses exactly one institution/venue. TRAIN submissions are before `--train-before`; VALIDATION submissions are from that time up to `--validation-before`; TEST submissions start at the latter cutoff. Each cohort must be nonempty. All TRAIN reviews/decisions must be available strictly before the first cutoff and all VALIDATION labels before the second; later information causes an error. Retrieval further requires all historical example information to predate the specific query. There is no post-validation refit or expansion of the retrieval corpus with validation/test records.

Model input is an explicit whitelist: target manuscript ID/version/title/text/submission/version time; institution rubric; and top-k historical TRAIN examples with up to 2,000 characters of manuscript excerpt, their human reviews/scores/decision, similarity and availability. Full target text and historical human review text are not truncated. The default k is 3, supported range 0–10. Token Jaccard retrieval is deterministic and simple; non-English segmentation, long-context budgeting, and relevance validation are future work.

Output evidence quotes must be exact nonempty substrings of target text. This checks text location only, not whether a claim is true or a review is useful. `raw_score`, `calibrated_score`, and `meets_score_threshold` are assistance signals, never acceptance probabilities. `human_review_required` is always true. Reports contain fit diagnostics, validation selection metrics, final test metrics, false-negative/false-positive counts, positive recall, cohort IDs, and hashes. Hashes support comparison, not tamper-proof auditing or proof that an endpoint kept the same model weights.
