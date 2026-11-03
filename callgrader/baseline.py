"""Baseline: TF-IDF features and one logistic regression per label (scikit-learn)."""

from __future__ import annotations

import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from callgrader.data import LABELS, read
from callgrader.metrics import report


def train(rows: list[dict]) -> dict:
    models = {}
    for lab in LABELS:
        xs = [r["text"] for r in rows if r["labels"][lab] is not None]
        ys = [r["labels"][lab] for r in rows if r["labels"][lab] is not None]
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0))
        models[lab] = m.fit(xs, ys)
    return models


def predict(models: dict, rows: list[dict]) -> list[dict[str, bool]]:
    texts = [r["text"] for r in rows]
    cols = {lab: models[lab].predict(texts) for lab in LABELS}
    return [{lab: bool(cols[lab][i]) for lab in LABELS} for i in range(len(rows))]


if __name__ == "__main__":
    train_rows, test_rows = read("train") + read("val"), read("test")
    t0 = time.perf_counter(); models = train(train_rows); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); preds = predict(models, test_rows); ms = (time.perf_counter() - t0) / len(test_rows) * 1000
    out = {"model": "tfidf+logreg", "train_calls": len(train_rows), "test_calls": len(test_rows),
           "fit_seconds": round(fit_s, 2), "ms_per_call": round(ms, 3), "metrics": report(test_rows, preds, LABELS)}
    json.dump(out, open("results/baseline.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
