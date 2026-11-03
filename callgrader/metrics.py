"""Per-label precision, recall and F1 on the calls where the label applies."""

from __future__ import annotations


def score(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(y_true), "positives": tp + fn, "accuracy": round(correct / max(len(y_true), 1), 3),
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def report(rows: list[dict], preds: list[dict[str, bool]], labels: list[str]) -> dict[str, dict]:
    out = {}
    for lab in labels:
        pairs = [(r["labels"][lab], p[lab]) for r, p in zip(rows, preds) if r["labels"][lab] is not None]
        out[lab] = score([t for t, _ in pairs], [p for _, p in pairs])
    out["macro_f1"] = round(sum(out[l]["f1"] for l in labels) / len(labels), 3)
    return out
