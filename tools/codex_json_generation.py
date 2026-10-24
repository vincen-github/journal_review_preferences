"""One explicit, isolated Codex JSON generation; no dataset or credential reads.

The caller supplies the training-only payload and owns call-budget/domain
validation. This module never retries, resumes, selects data, or reads labels.
"""

import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from journal_prefs.data import fingerprint, write_json


SETTINGS = {
    "model": "gpt-6.1-sol", "reasoning_effort": "low", "service_tier": "default",
    "forced_auth": "chatgpt", "web_search": "disabled", "shell_tool": False,
    "apps": False, "plugins": False, "multi_agent": False,
    "permissions": "root deny; minimal/runtime read; isolated workspace read; command network disabled",
    "temperature": "not exposed by Codex; not set",
}

_CONFIGS = [
    'forced_login_method="chatgpt"', 'model_provider="openai"', 'model_reasoning_effort="low"',
    'service_tier="default"', 'web_search="disabled"', 'default_permissions="scoring"',
    'permissions.scoring.filesystem={":root"="deny",":minimal"="read",":workspace_roots"={"."="read"}}',
    'permissions.scoring.network.enabled=false', 'features.shell_tool=false', 'features.multi_agent=false',
    'features.apps=false', 'features.plugins=false', 'features.remote_plugin=false',
    'features.browser_use=false', 'features.computer_use=false', 'features.memories=false',
    'tools.view_image=false', 'mcp_servers={}', 'project_doc_max_bytes=0',
]
_EVENTS = {"thread.started", "turn.started", "turn.completed", "turn.failed",
           "item.started", "item.updated", "item.completed", "error"}
_TOOLS = {"command_execution", "mcp_tool_call", "web_search", "tool_call", "file_change",
          "file_read", "browser_use", "computer_use", "image_view"}
_USAGE = {"input_tokens", "cached_input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
          "reasoning_output_tokens", "cache_write_input_tokens"}


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        # allow_nan=False in fingerprint below checks floating point finiteness.
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("Generation values must be JSON with string object keys")


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _notice(message):
    """Persist fixed diagnostic codes, never arbitrary CLI messages/identities."""
    message = _text(message).lower()
    for text, code in (("permission denied", "permission_denied"), ("rate limit", "rate_limit"),
                       ("not logged in", "authentication_required"), ("invalid", "invalid_configuration")):
        if text in message:
            return code
    return "other_cli_notice"


def _collect(stdout, stderr, output, receipt):
    messages, events, tools, notices, usage = [], [], [], [], {}
    unknown = 0
    for line in _text(stdout).splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _EVENTS:
            unknown += 1
            continue
        events.append(event_type)
        if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {key: value for key, value in event["usage"].items()
                     if key in _USAGE and type(value) is int and value >= 0}
        if event_type in ("turn.failed", "error"):
            detail = event.get("error", {})
            notices.append(_notice(detail.get("message") if isinstance(detail, dict) else detail))
        if event_type != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            tools.append("unrecognized_item_type")
            continue
        item_type = item.get("type")
        if item_type == "agent_message":
            if isinstance(item.get("text"), str):
                messages.append(item["text"])
        elif isinstance(item_type, str) and item_type in _TOOLS:
            tools.append(item_type)
        elif item_type == "error":
            notices.append(_notice(item.get("message")))
        elif item_type not in ("reasoning", "plan"):
            tools.append("unrecognized_item_type")
    receipt.update({"event_types": events, "tool_events": tools, "cli_notices": notices,
                    "usage": usage, "unrecognized_event_count": unknown,
                    "reported_sol_slug": bool(re.search(r'\bslug="?gpt-6\.1-sol(?=["\s,}\]]|$)', _text(stderr)))})
    receipt['reported_model_slug'] = bool(re.search(r'\bslug="?' + re.escape(receipt['settings']['model']) + r'(?=["\s,}\]]|$)', _text(stderr)))
    if messages:
        raw = messages[-1]
        (output / "response.raw.txt").write_text(raw, encoding="utf-8")
        receipt["raw_response_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return raw
    return None


def generate(payload, instructions, schema, output, codex, allow_quota=False, model=None):
    """Return (JSON object, receipt) after exactly one permitted subprocess.

    The CLI receives the supplied output schema. Semantic rule/evidence
    validation remains the caller's responsibility. A fresh output directory
    prevents accidental reuse/retry of an earlier or uncertain invocation.
    """
    if allow_quota is not True:
        raise ValueError("Explicit allow_quota=True is required")
    model = model or SETTINGS['model']
    if model not in ('gpt-6.1-sol', 'gpt-6-luna'):
        raise ValueError('Model must be explicitly verified for this bounded trial')
    if not isinstance(payload, dict):
        raise ValueError("Generation payload must be a JSON object")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Generation instructions must be a nonempty string")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Generation output schema must describe an object")
    if not isinstance(codex, (str, Path)) or not str(codex).strip():
        raise ValueError("An explicit Codex executable is required")
    _json_value(payload)
    _json_value(schema)
    payload_hash, schema_hash = fingerprint(payload), fingerprint(schema)
    request = instructions + "\nGENERATION_DATA_JSON:\n" + json.dumps(payload, ensure_ascii=False, allow_nan=False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    isolated = output / "isolated"
    isolated.mkdir()
    schema_path = isolated / "response.schema.json"
    write_json(schema_path, schema)
    write_json(output / "response.schema.json", schema)
    snapshot = {"instructions": instructions, "payload": deepcopy(payload), "request": request,
                "instructions_sha256": fingerprint(instructions), "payload_sha256": payload_hash,
                "schema_sha256": schema_hash, "request_sha256": fingerprint(request)}
    write_json(output / "prompt_snapshot.json", snapshot)
    receipt = {"schema_version": 1, "status": "started", "settings": dict(deepcopy(SETTINGS), model=model),
               "instructions_sha256": snapshot["instructions_sha256"], "payload_sha256": payload_hash,
               "schema_sha256": schema_hash, "request_sha256": snapshot["request_sha256"],
               "prompt_snapshot_sha256": fingerprint(snapshot), "invocation_config_sha256": fingerprint(_CONFIGS),
               "subprocess_attempts": 1, "timeout_seconds": 300, "exit_code": None,
               "event_types": [], "tool_events": [], "cli_notices": [], "usage": {},
               "reported_sol_slug": False, "automatic_retry": False,
               "route": "Fresh ephemeral Codex CLI using existing ChatGPT sign-in; supplied payload only",
               "validation": "JSON object; schema supplied to CLI; caller validates domain evidence"}
    write_json(output / "receipt.json", receipt)
    args = [str(codex), "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
            "--color", "never", "--json", "--model", model, "--output-schema", str(schema_path)]
    for config in _CONFIGS:
        args.extend(["-c", config])
    args.append("-")
    error_code = "process_launch_error"
    try:
        try:
            result = subprocess.run(args, input=request, text=True, encoding="utf-8", errors="replace",
                                    capture_output=True, cwd=isolated, timeout=300)
        except subprocess.TimeoutExpired as exc:
            _collect(exc.stdout, exc.stderr, output, receipt)
            error_code = "timeout_uncertain"
            raise
        receipt["exit_code"] = result.returncode
        raw = _collect(result.stdout, result.stderr, output, receipt)
        if result.returncode != 0:
            error_code = "nonzero_exit"
            raise ValueError("nonzero_exit")
        if receipt["tool_events"]:
            error_code = "tool_or_unrecognized_item"
            raise ValueError("tool_or_unrecognized_item")
        if "turn.failed" in receipt["event_types"]:
            error_code = "turn_failed"
            raise ValueError("turn_failed")
        if not receipt["reported_model_slug"]:
            error_code = "model_route_not_reported"
            raise ValueError("model_route_not_reported")
        if raw is None:
            error_code = "missing_final_response"
            raise ValueError("missing_final_response")
        error_code = "invalid_json_object"
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise ValueError("invalid_json_object")
        _json_value(response)
        receipt["response_sha256"] = fingerprint(response)
        write_json(output / "response.json", response)
        receipt["status"] = "completed"
        write_json(output / "receipt.json", receipt)
        return response, deepcopy(receipt)
    except Exception as exc:
        receipt.update({"status": "failed_or_uncertain_no_retry", "error_code": error_code,
                        "error_type": type(exc).__name__})
        write_json(output / "receipt.json", receipt)
        raise ValueError("Codex JSON generation failed; see the local receipt; no automatic retry") from None
