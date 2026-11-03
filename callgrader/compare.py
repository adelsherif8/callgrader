"""Collect every results/*.json into one comparison table (markdown) for the README."""

import json
import os

from callgrader.data import LABELS

ROWS = [("baseline.json", "TF-IDF + logistic regression (scikit-learn)"),
        ("encoder.json", "ModernBERT-base fine-tuned (PyTorch)"),
        ("llm.json", "Qwen2.5-0.5B + LoRA, JSON answer")]


def table(results_dir="results") -> str:
    head = "| Model | " + " | ".join(LABELS) + " | Macro F1 | ms / call |\n|---|" + "---|" * (len(LABELS) + 2)
    lines = [head]
    for fname, name in ROWS:
        p = os.path.join(results_dir, fname)
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); m = r["metrics"]
        cells = " | ".join(f"{m[l]['f1']:.2f}" for l in LABELS)
        lines.append(f"| {name} | {cells} | **{m['macro_f1']:.2f}** | {r.get('ms_per_call', '')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
