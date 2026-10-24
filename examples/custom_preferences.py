"""Minimal institution-configured plugin example; this does not learn rules.

Inject an instance into compare_inputs(..., algorithm=plugin, artifact=state).
No automatic importing of arbitrary plugins occurs in the CLI.
"""
from copy import deepcopy
from journal_prefs.data import fingerprint, read_json, write_json


class ConfiguredPreferences:
    def __init__(self, principles):
        self.principles = list(principles)

    def fit(self, history):
        return {"schema_version": 1, "algorithm": "institution_configured_example",
                "train_ids": [r["paper_id"] for r in history],
                "principles": self.principles[:], "frozen": False}

    def refine(self, artifact, validation, assessments=None):
        return deepcopy(artifact)  # No automatic validation-based change in this example.

    def freeze(self, artifact):
        result = deepcopy(artifact)
        result["frozen"] = True
        return result

    def condition(self, artifact, paper):
        if not artifact["frozen"] or paper["paper_id"] in artifact["train_ids"]:
            raise ValueError("Use frozen preferences on a new paper")
        return [{"rule_id": str(i + 1), "principle": text} for i, text in enumerate(artifact["principles"])]

    def save(self, artifact, path):
        write_json(path, {"artifact": artifact, "sha256": fingerprint(artifact)})

    def load(self, path):
        record = read_json(path)
        if fingerprint(record["artifact"]) != record["sha256"]:
            raise ValueError("Preference checksum mismatch")
        return record["artifact"]
