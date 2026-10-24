"""Small, provider-independent JSON prompt files; no code or template evaluation."""

import json
from copy import deepcopy

from .data import fingerprint, nonempty, number, read_json


def validate_template(template):
    if not isinstance(template, dict) or template.get("schema_version") != 1:
        raise ValueError("Prompt template requires schema_version=1")
    for key in ("template_id", "version", "instructions"):
        nonempty(template.get(key), "prompt " + key)
    score_range = template.get("score_range")
    if not isinstance(score_range, list) or len(score_range) != 2:
        raise ValueError("Prompt score_range must contain two numbers")
    low, high = [number(v, "prompt score limit") for v in score_range]
    if low >= high:
        raise ValueError("Prompt score_range must increase")
    anchors = template.get("score_anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("Prompt requires explicit score_anchors")
    seen = set()
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise ValueError("Prompt score anchor must be an object")
        score = number(anchor.get("score"), "prompt anchor score")
        if score in seen or not low <= score <= high:
            raise ValueError("Prompt anchor score is duplicated or outside range")
        seen.add(score)
        nonempty(anchor.get("meaning"), "prompt anchor meaning")
    fields = template.get("response_fields", {k: k for k in ("score", "review", "evidence")})
    if not isinstance(fields, dict) or set(fields) != {"score", "review", "evidence"}:
        raise ValueError("Prompt response_fields must map score, review and evidence")
    for value in fields.values():
        nonempty(value, "response field name")
    if len(set(fields.values())) != 3:
        raise ValueError("Prompt response field names must be distinct")
    schema = template.get("output_schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Prompt output_schema must be a JSON object schema")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not set(fields.values()) <= set(properties):
        raise ValueError("Prompt schema must declare all response fields")
    if not set(fields.values()) <= set(schema.get("required", [])):
        raise ValueError("Prompt schema must require all response fields")
    score_schema = properties[fields["score"]]
    if not isinstance(score_schema, dict) or score_schema.get("type") not in ("number", "integer") or score_schema.get("minimum") != low or score_schema.get("maximum") != high:
        raise ValueError("Prompt score schema must match score_range")
    if properties[fields["review"]].get("type") != "string" or properties[fields["evidence"]].get("type") != "array":
        raise ValueError("Prompt review/evidence fields must be string/array")
    if not isinstance(template.get("sources"), list):
        raise ValueError("Prompt sources must be a list; original prompts may use an empty list")
    # Also reject non-JSON/nonfinite content before any model request or snapshot.
    fingerprint(template)
    return deepcopy(template)


def load_template(path):
    return validate_template(read_json(path))


def render_instructions(template):
    template = validate_template(template)
    return (template["instructions"] + "\nSCORING_RANGE: " + json.dumps(template["score_range"])
            + "\nRATING_ANCHORS: " + json.dumps(template["score_anchors"], ensure_ascii=False)
            + "\nReturn ONLY JSON matching this output schema: "
            + json.dumps(template["output_schema"], ensure_ascii=False))


def normalize_response(response, template):
    if not isinstance(response, dict):
        raise ValueError("Prompt response must be a JSON object")
    fields = template.get("response_fields", {k: k for k in ("score", "review", "evidence")})
    try:
        result = {key: response[value] for key, value in fields.items()}
    except KeyError:
        raise ValueError("Model output is missing a configured response field") from None
    score = number(result["score"], "model score")
    low, high = template["score_range"]
    if not low <= score <= high:
        raise ValueError("Model score is outside prompt score_range")
    if template["output_schema"]["properties"][fields["score"]]["type"] == "integer" and not score.is_integer():
        raise ValueError("Prompt requires an integer recommendation score")
    return result


def snapshot(template):
    template = validate_template(template)
    return {"template": template, "template_sha256": fingerprint(template),
            "rendered_instructions": render_instructions(template),
            "rendered_instructions_sha256": fingerprint(render_instructions(template))}
