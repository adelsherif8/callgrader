"""Fine-tune ModernBERT as a multi-label call grader, with a plain PyTorch training loop.

One sigmoid output per label. Labels that don't apply to a call (None) are masked out of the loss.
Long calls keep their first 192 and last 576 tokens: the opening sets the task, the end holds the handoff,
the save and the goodbye. Thresholds are tuned per label on the validation split.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from callgrader.data import LABELS, read
from callgrader.metrics import report

MODEL = "answerdotai/ModernBERT-base"
HEAD, TAIL = 192, 576     # 768 tokens: fits the M3's shared 8 GB; covers ~85% of calls whole


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class Grader(nn.Module):
    def __init__(self, base: str = MODEL):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        self.encoder.gradient_checkpointing_enable()   # recompute activations: fits an 8 GB laptop
        self.head = nn.Linear(self.encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)       # mean over real tokens
        return self.head(pooled)


def encode(tok, texts: list[str]) -> dict[str, torch.Tensor]:
    ids = []
    for t in texts:
        x = tok(t, add_special_tokens=False)["input_ids"]
        if len(x) > HEAD + TAIL - 2:
            x = x[:HEAD] + x[-(TAIL - 2):]
        ids.append([tok.cls_token_id] + x + [tok.sep_token_id])
    width = max(map(len, ids))
    input_ids = torch.full((len(ids), width), tok.pad_token_id)
    attn = torch.zeros((len(ids), width), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = torch.tensor(x)
        attn[i, :len(x)] = 1
    return {"input_ids": input_ids, "attention_mask": attn}


def targets(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.tensor([[float(bool(r["labels"][l])) for l in LABELS] for r in rows])
    known = torch.tensor([[float(r["labels"][l] is not None) for l in LABELS] for r in rows])
    return y, known


@torch.no_grad()
def probabilities(model, tok, rows, batch=2) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        enc = {k: v.to(device()) for k, v in encode(tok, [r["text"] for r in rows[i:i + batch]]).items()}
        out.append(torch.sigmoid(model(**enc)).float().cpu())
    return torch.cat(out)


def tune_thresholds(probs: torch.Tensor, rows: list[dict]) -> dict[str, float]:
    """Per label, the threshold with the best F1 on validation; 0.5 when validation has no positives."""
    best = {}
    for j, lab in enumerate(LABELS):
        idx = [i for i, r in enumerate(rows) if r["labels"][lab] is not None]
        ys = [rows[i]["labels"][lab] for i in idx]
        if not any(ys):
            best[lab] = 0.5
            continue
        scores = []
        for th in [x / 20 for x in range(2, 19)]:
            pred = [bool(probs[i, j] >= th) for i in idx]
            tp = sum(p and y for p, y in zip(pred, ys)); fp = sum(p and not y for p, y in zip(pred, ys))
            fn = sum(y and not p for p, y in zip(pred, ys))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            scores.append((f1, -abs(th - 0.5), th))
        best[lab] = max(scores)[2]
    return best


def to_preds(probs: torch.Tensor, th: dict[str, float]) -> list[dict[str, bool]]:
    return [{lab: bool(probs[i, j] >= th[lab]) for j, lab in enumerate(LABELS)} for i in range(len(probs))]


def train(epochs=4, lr=3e-5, batch=1, accum=16, seed=0, out="models/encoder", splits=None,
          result_path="results/encoder.json"):
    """Train on `splits` = (train, val, test) rows (default: the saved split). Returns the result dict."""
    random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Grader().to(device())
    tr, va, te = splits or (read("train"), read("val"), read("test"))
    y_all, k_all = targets(tr)
    pos = (y_all * k_all).sum(0); neg = ((1 - y_all) * k_all).sum(0)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=(neg / pos.clamp(min=1)).clamp(max=10).to(device()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(tr) // (batch * accum) + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, history, t0 = -1.0, [], time.perf_counter()
    for ep in range(epochs):
        model.train(); order = list(range(len(tr))); random.shuffle(order); running = 0.0
        for n, i in enumerate(range(0, len(order), batch)):
            rows = [tr[j] for j in order[i:i + batch]]
            enc = {k: v.to(device()) for k, v in encode(tok, [r["text"] for r in rows]).items()}
            y, known = (t.to(device()) for t in targets(rows))
            loss = (loss_fn(model(**enc), y) * known).sum() / known.sum().clamp(min=1)
            (loss / accum).backward(); running += loss.item()
            if (n + 1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
            if (n + 1) % 50 == 0:
                print({"epoch": ep + 1, "batch": n + 1, "of": len(order) // batch + 1, "loss": round(running / (n + 1), 4),
                       "minutes": round((time.perf_counter() - t0) / 60, 1)}, flush=True)
        p_va = probabilities(model, tok, va)
        th = tune_thresholds(p_va, va)
        val = report(va, to_preds(p_va, th), LABELS)
        history.append({"epoch": ep + 1, "train_loss": round(running / (n + 1), 4), "val_macro_f1": val["macro_f1"],
                        "minutes": round((time.perf_counter() - t0) / 60, 1)})
        print(history[-1], flush=True)
        if val["macro_f1"] > best:
            best = val["macro_f1"]; os.makedirs(out, exist_ok=True)
            torch.save(model.state_dict(), f"{out}/grader.pt"); json.dump(th, open(f"{out}/thresholds.json", "w"))
    model.load_state_dict(torch.load(f"{out}/grader.pt", map_location=device()))
    th = json.load(open(f"{out}/thresholds.json"))
    t1 = time.perf_counter(); p_te = probabilities(model, tok, te); ms = (time.perf_counter() - t1) / len(te) * 1000
    res = {"model": MODEL + " (fine-tuned, PyTorch, " + str(device()) + ")", "train_calls": len(tr), "test_calls": len(te),
           "epochs": epochs, "history": history, "thresholds": th, "ms_per_call": round(ms, 1),
           "metrics": report(te, to_preds(p_te, th), LABELS)}
    if result_path:
        json.dump(res, open(result_path, "w"), indent=2)
    print(json.dumps(res["metrics"], indent=2), flush=True)
    del model
    if device().type == "mps":
        torch.mps.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=4); a = ap.parse_args()
    train(epochs=a.epochs)
