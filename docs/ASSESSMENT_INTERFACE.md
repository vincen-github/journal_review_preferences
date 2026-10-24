# Generic structured assessment

This small interface lets an institution define its instructions and output fields. Numeric scores are optional. The runnable example uses a categorical recommendation, an editor note, and custom flags; it contains no real manuscript or evaluation result. Institutions choose their own alignment method and model. This interface does not implement preference learning or a backtest.

From the project directory, this example runs entirely offline:

```python
from journal_prefs.data import read_json
from journal_prefs.assessment import assess

task = read_json("examples/assessment_task.synthetic.json")
request = read_json("examples/assessment_request.synthetic.json")
artifact = assess(task, request)
print(artifact["output"])
assert artifact["synthetic_output"] and artifact["human_review_required"]
```

The task's `offline_response` must explicitly contain `synthetic: true` and a schema-valid `output`. Offline mode never constructs or calls a model. Local/cloud modes reuse `make_model` and `generate_json`, including the existing literal-loopback and explicit cloud-permission gates. The configured offline output is never sent to a real model. No endpoint or provider is selected automatically.

The request must contain exactly `target` and `history`. A manuscript contains only `id`, `title`, and `text`; target labels and arbitrary fields are rejected. Every historical example declares `split: "TRAIN"` and contains a `manuscript` plus its structured `output`. IDs must be unique and exclude the target. The caller is responsible for truthful TRAIN membership, paper identity, data rights, and model suitability. Artifacts snapshot the task/request and hashes, so they may contain sensitive TRAIN content and should remain local when appropriate. Hashes are checksums, not signatures.

The explicit schema subset supports `object`, `array`, `string`, `number`, `integer`, and `boolean`, plus scalar `enum` and optional `description`. Objects declare `properties`, optional `required`, and `additionalProperties: false`. Strings require `maxLength` (0–20,000); arrays require `items` and `maxItems` (0–100); optional `minLength`/`minItems` may narrow these bounds. Numbers require finite `minimum` and `maximum`. A schema has at most 256 nodes, six nesting levels, 64 properties per object, and 64 enum choices. Other JSON Schema keywords are rejected rather than silently ignored. History is limited to 64 explicitly supplied examples.

`evaluate_metrics(task, predictions, targets)` accepts equally sized, already paired lists of schema-valid output objects. Optional metric configurations name a top-level output field and `accuracy` (enum/boolean only), `MAE`, or `RMSE` (numeric only). For example: `{"field": "recommendation", "metric": "accuracy"}`. With no metric configuration it returns `{}`; it never invents a text-quality measure. These metrics measure agreement with supplied targets, not scientific validity or acceptance probabilities. A human editor must review every suggestion; this module takes no editorial action.

## CLI example

```powershell
python -B -m journal_prefs assess --task examples/assessment_task.synthetic.json --request examples/assessment_request.synthetic.json --out my_first_demo
```

This configured synthetic example is the generic framework path. The separate `demo`, `review` and `backtest` commands retain a numeric-score demonstration; their numeric schema is not required by `assess`. Institutions supply the preference algorithm, training evidence and task metrics rather than adopting a mandated alignment method.
