"""Author-created fixture generator. No external input, networking, or models.

The fixture intentionally has imperfect label agreement. This is plumbing, not
evidence of institutional preference alignment. Run only to regenerate examples.
"""

import json
from pathlib import Path


def build():
    counts = [(1, 1, 1), (4, 3, 3), (2, 2, 1), (3, 4, 4),
              (0, 2, 2), (4, 4, 4), (2, 3, 2), (3, 1, 3),
              (1, 2, 1), (4, 3, 4), (2, 2, 3), (3, 3, 2),
              (4, 4, 3), (0, 1, 0), (2, 3, 4), (3, 2, 2)]
    vocabulary = [("baseline", "randomized", "replication", "uncertainty"),
                  ("limitations", "protocol", "dataset", "code"),
                  ("theory", "comparison", "evidence", "application")]
    noise = [0.2, -0.3, 0.6, -0.4, 0.1, -0.1, 0.4, -0.2,
             0.3, -0.4, 0.1, 0.5, -0.2, 0.4, -0.5, 0.2]
    rows = []
    for index, count in enumerate(counts):
        year = 2023 if index < 8 else (2024 if index < 12 else 2025)
        month = index + 1 if index < 8 else (index - 7 if index < 12 else index - 11)
        keywords = [word for words, n in zip(vocabulary, count) for word in words[:n]]
        raw = 1 + 9 * (0.1 + 0.8 * sum(w * c / 4 for w, c in zip((0.5, 0.3, 0.2), count)))
        human_score = round(max(1, min(10, 0.82 * raw + 1.1 + noise[index])), 2)
        # These synthetic human labels also contain contradictions to illustrate uncertainty.
        decision = "accept" if human_score >= 6 else "decline"
        if index in (6, 14):
            decision = "decline" if decision == "accept" else "accept"
        stamp = f"{year}-{month:02d}"
        rows.append({
            "schema_version": 1, "institution_id": "synthetic-institution",
            "venue_id": "synthetic-interdisciplinary-journal", "manuscript_id": f"synthetic-{index + 1:02d}",
            "version_id": "v1-pre-review", "title": f"SYNTHETIC study {index + 1}: learning and evaluation",
            "text": "SYNTHETIC MANUSCRIPT FOR SOFTWARE DEMONSTRATION ONLY. Learning study discussing "
                    + ", ".join(keywords) + ". All claims and data are fictional.",
            "version_time": stamp + "-01T00:00:00Z", "text_available_at": stamp + "-01T00:00:00Z",
            "submitted_at": stamp + "-02T00:00:00Z", "rights_confirmed": True, "synthetic": True,
            "reviews": [{"text": "SYNTHETIC HUMAN REVIEW: fictional assessment of reporting and study design.",
                         "score": human_score, "available_at": stamp + "-15T00:00:00Z"}],
            "decision": {"label": decision, "available_at": stamp + "-20T00:00:00Z"}})
    return rows


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "examples" / "history.synthetic.jsonl"
    if path.exists():
        raise SystemExit("Refusing to overwrite an existing fixture")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build()), encoding="utf-8")
    print(f"Wrote {len(build())} synthetic records to {path}")
