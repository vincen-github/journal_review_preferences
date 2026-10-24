"""Deterministic quote formatting repair; never change words, scores or reviews."""
import re
from copy import deepcopy


def restore_source_whitespace(response, text):
    result, repairs = deepcopy(response), []
    if not isinstance(result, dict) or not isinstance(result.get("evidence"), list):
        return result, repairs
    for item in result["evidence"]:
        if not isinstance(item, dict) or not isinstance(item.get("quote"), str):
            continue
        quote = item["quote"]
        if quote in text or not quote.strip():
            continue
        # Exact non-whitespace characters, in the same contiguous order.
        pattern = r"\s+".join(re.escape(part) for part in re.split(r"\s+", quote.strip()))
        match = re.search(pattern, text)
        if match:
            item["quote"] = match.group()
            repairs.append({"original_quote": quote, "source_quote": match.group(),
                            "start": match.start(), "end": match.end()})
    return result, repairs


def quarantine_unmatched_quotes(response, text):
    """Retain verified fragments; preserve rejected fragments in a separate audit."""
    result, rejected = deepcopy(response), []
    if not isinstance(result, dict) or not isinstance(result.get("evidence"), list):
        return result, rejected
    retained = []
    for item in result["evidence"]:
        if isinstance(item, dict) and isinstance(item.get("quote"), str) and item["quote"].strip() and item["quote"] not in text:
            rejected.append(dict(item, reason="Exact quote absent from supplied manuscript; not verified or repaired"))
        else:
            retained.append(item)
    result["evidence"] = retained
    return result, rejected
