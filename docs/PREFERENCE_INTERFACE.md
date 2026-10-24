# Preference algorithms and model providers

The interfaces are separate: a preference algorithm creates external conditions; a model provider generates text/scores; affine calibration adjusts numeric outputs. No base weights are trained. **The current main method is simple absolute-score prediction/calibration**, using each same-year manuscript's mean human recommendation as target. Identity versus learning-fitted affine calibration is selected on validation and frozen before test evaluation. ICAI and the evidence-rule scaffold are retained as optional references/interfaces, not active defaults; no score-to-pairwise construction is performed.

Institution review instructions, rating anchors and JSON output fields have a separate [prompt-file interface](PROMPT_INTERFACE.md), shared by compatible local/cloud adapters and the independent Codex CLI scorer.

`PreferenceAlgorithm` exposes `fit(history)`, optional bounded `refine(...)`, `freeze(artifact)` and `condition(artifact, paper)`. Algorithms own their versioned artifact and save/load format. The supplied evidence scaffold has checked export/load helpers; the original-ICAI bridge and minimal custom example have save/load methods. Nothing automatically loads arbitrary plugins.

`examples/custom_preferences.py` is a working minimal **configured** algorithm example, not a learning claim:

```python
from examples.custom_preferences import ConfiguredPreferences

plugin = ConfiguredPreferences(["Check whether the stated claim has supporting evidence."])
state = plugin.freeze(plugin.fit(training_history))
plugin.save(state, "institution_preferences.json")
context = plugin.condition(plugin.load("institution_preferences.json"), new_paper)
```

Replace this object with another algorithm without replacing the scoring provider. `ICAIUpstreamAdapter(runner=approved_original_runner, backend_description=...)` accepts the original runner's `final_principles`; an absent runner does not silently use the simplified scaffold.

## Implemented experimental scaffold

`EvidenceRules(generator)` accepts normalized labelled history (`paper_id`, `title`, `text`, review `text`/`score`, `decision`). A caller-injected `generator(purpose, payload)` returns candidate rules with exact source review/manuscript quotes; per-paper candidates are consolidated. At most one validation refinement is allowed. The artifact records train/selection IDs, source hashes, algorithm version and rules. Freeze prevents ordinary refinement, and new-paper conditions expose rule IDs/principles only. Artifacts retain labelled source records locally and must be protected accordingly. Checksums detect accidental changes; they are not signatures.

This is **not ICAI** and its real experiment is paused. `SyntheticRuleGenerator` merely copies and consolidates fixture reasons; `MockModel` ignores those rules. To check the software flow offline:

```powershell
python -B -m journal_prefs preference-demo --out my_preference_demo
```

Its equal-arm result is expected synthetic plumbing. It does not show learned scientific preferences or improvement.

## Fair scoring and evaluation

`compare_inputs(...)` receives label-free manuscripts, one common scoring form, guideline provenance, a frozen preference artifact and one model/configuration. All three arms share the same provider, settings and empty historical-example list; only guideline/principle context changes. Order rotates by a rule fixed before outputs. Configuration/artifact hashes must remain unchanged. Scoring is independent of affine calibration.

`evaluate_comparison(...)` accepts human labels only after predictions exist. It reports all arms' score MAE, configured-threshold agreement, false negatives and false positives, and preserves negative or zero deltas. These are small-pilot score/outcome comparisons, not a validated review-quality or acceptance-probability measure. Target-label poisoning tests change metrics without changing requests or frozen predictions.

Prior coordinator label exposure is explicit and distinct from evidence of scoring-session exposure. The report does not certify strict blinding; request fingerprints, observed tool events and their limits are described in the [pilot audit](EXPERIMENTS.md). Public-data pretraining contamination remains possible. Version limitations remain descriptive and do not get replaced by invented dates.

`OpenAICompatibleModel.generate_json(payload, instructions)` provides the same checked local/cloud transport for an external generator. Local mode keeps its loopback/proxy/redirect/no-fallback behavior. The Sol connectivity check used Codex CLI separately; no native project HTTP/Sol integration is claimed tested.
