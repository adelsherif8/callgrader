"""5-fold cross-validation grouped by scenario.

One fixed test split of ~300 calls holds only a handful of rare positives (missed handoffs, made-up answers),
so a single score moves a lot with which scenarios happened to land in test. Here every scenario is tested
exactly once, by a model that never saw it: within each category, scenarios are dealt round-robin into 5 folds
(largest first), so each fold gets a share of every kind of call. Scores are pooled over all out-of-fold
predictions, with the per-fold spread and a bootstrap 95% interval on macro F1.

    python -m callgrader.cv                 # scikit-learn baseline only (seconds)
    python -m callgrader.cv --encoder       # plus ModernBERT, 5 trainings (~2 h on an M3)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics

from callgrader import baseline
from callgrader.data import LABELS, read
from callgrader.metrics import report

K = 5


def all_rows() -> list[dict]:
    return read("train") + read("val") + read("test")


def folds(rows: list[dict], k: int = K) -> dict[str, int]:
    """scenario -> fold. Whole scenarios stay together; each category is spread over all folds."""
    sizes: dict[str, dict[str, int]] = {}
    for r in rows:
        sizes.setdefault(r["category"], {}).setdefault(r["scenario"], 0)
        sizes[r["category"]][r["scenario"]] += 1
    out, offset = {}, 0
    for cat in sorted(sizes):
        for i, sc in enumerate(sorted(sizes[cat], key=lambda s: (-sizes[cat][s], s))):
            out[sc] = (i + offset) % k
        offset += len(sizes[cat])          # start each category on a different fold, so folds stay even
    return out


def bootstrap_macro(rows, preds, n=1000, seed=0) -> tuple[float, float]:
    rnd, scores = random.Random(seed), []
    for _ in range(n):
        idx = [rnd.randrange(len(rows)) for _ in rows]
        scores.append(report([rows[i] for i in idx], [preds[i] for i in idx], LABELS)["macro_f1"])
    scores.sort()
    return scores[int(0.025 * n)], scores[int(0.975 * n)]


def majority(train_rows, test_rows):
    """The floor: predict each label's most common value in training, for every call."""
    common = {}
    for lab in LABELS:
        ys = [r["labels"][lab] for r in train_rows if r["labels"][lab] is not None]
        common[lab] = sum(ys) > len(ys) / 2
    return [dict(common) for _ in test_rows]


def run_baseline(tr, va, te):
    return baseline.predict(baseline.train(tr + va), te)


def run_encoder(tr, va, te):
    from callgrader import encoder
    from transformers import AutoTokenizer
    res = encoder.train(epochs=3, splits=(tr, va, te), out="models/cv-encoder", result_path=None)
    # rebuild predictions from the saved best model, so they can be pooled across folds
    import torch
    tok = AutoTokenizer.from_pretrained(encoder.MODEL)
    model = encoder.Grader().to(encoder.device())
    model.load_state_dict(torch.load("models/cv-encoder/grader.pt", map_location=encoder.device()))
    preds = encoder.to_preds(encoder.probabilities(model, tok, te), res["thresholds"])
    del model
    return preds


def cross_validate(name, fn, rows, k=K, log=print, ckpt=None):
    """With `ckpt` (a folder, e.g. on Google Drive) each finished fold's predictions are saved, and a rerun skips the
    folds already done, so a reset only loses the fold that was running."""
    fold_of = folds(rows, k)
    oof: list[tuple[dict, dict]] = []
    per_fold = []
    for f in range(k):
        te = [r for r in rows if fold_of[r["scenario"]] == f]
        va = [r for r in rows if fold_of[r["scenario"]] == (f + 1) % k]       # threshold tuning (encoder)
        tr = [r for r in rows if fold_of[r["scenario"]] not in (f, (f + 1) % k)]
        done = os.path.join(ckpt, f"{name}-fold{f + 1}.json") if ckpt else None
        if done and os.path.exists(done):
            preds = json.load(open(done))
            log({"model": name, "fold": f + 1, "resumed": True})
        else:
            preds = fn(tr, va, te)
            if done:
                os.makedirs(ckpt, exist_ok=True)
                json.dump(preds, open(done + ".tmp", "w")); os.replace(done + ".tmp", done)
        per_fold.append(report(te, preds, LABELS)["macro_f1"])
        oof += list(zip(te, preds))
        log({"model": name, "fold": f + 1, "test_calls": len(te), "macro_f1": per_fold[-1]})
    rows_o, preds_o = [r for r, _ in oof], [p for _, p in oof]
    lo, hi = bootstrap_macro(rows_o, preds_o)
    return {"model": name, "folds": k, "calls": len(rows_o), "per_fold_macro_f1": per_fold,
            "fold_mean": round(statistics.mean(per_fold), 3), "fold_std": round(statistics.stdev(per_fold), 3),
            "macro_f1_95ci": [lo, hi], "metrics": report(rows_o, preds_o, LABELS)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", action="store_true")
    ap.add_argument("--ckpt", help="folder for per-fold checkpoints (e.g. on Google Drive); reruns skip finished folds")
    a = ap.parse_args()
    rows = all_rows()
    out = {"majority": cross_validate("majority", lambda tr, va, te: majority(tr + va, te), rows),
           "baseline": cross_validate("tfidf+logreg", run_baseline, rows)}
    if a.encoder:
        out["encoder"] = cross_validate("modernbert", run_encoder, rows, log=lambda d: print(d, flush=True), ckpt=a.ckpt)
    json.dump(out, open("results/cv.json" if a.encoder else "results/cv-fast.json", "w"), indent=2)
    for v in out.values():
        print(v["model"], v["metrics"]["macro_f1"], "folds", v["fold_mean"], "+/-", v["fold_std"], "CI", v["macro_f1_95ci"])
