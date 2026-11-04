"""Build the CallGrader dataset from VoiceDesk's saved evaluation runs.

Each VoiceDesk eval run has simulated callers talking to the demo voice agents, with the full transcript
(PII already redacted by Presidio) and labels from the harness: code checks for tool calls and handoffs,
an LLM judge for task success and hallucination. Those labels are the targets here ("silver" labels:
they come from the harness, not from a human).

Splits are by scenario, so test calls come from situations the models never saw in training.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass

from callgrader.context import card

# What CallGrader predicts for each call. None = not applicable to that call (e.g. no injection attempt).
LABELS = ["passed", "task_success", "missed_handoff", "hallucinated", "injection_resisted"]

DEFAULT_RUNS = os.path.expanduser("~/Desktop/GHAI/voicedesk/evals/results")


@dataclass
class Call:
    id: str
    agent: str
    scenario: str
    category: str
    text: str
    labels: dict[str, bool | None]


def render(transcript: list[dict]) -> str:
    """One line per turn, with the agent's tool calls, as a grader would read the call log."""
    lines = []
    for t in transcript:
        who = "AGENT" if t.get("role") == "agent" else "CALLER"
        text = (t.get("text") or "").strip()
        if text:
            lines.append(f"{who}: {text}")
        for tc in t.get("tool_calls") or []:
            args = tc.get("args") or tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"raw": args}
            shown = ", ".join(f"{k}={v}" for k, v in args.items() if k != "execution_message")
            lines.append(f"TOOL {tc.get('name', '?')}({shown})")
    return "\n".join(lines)


def load_calls(runs_dir: str = DEFAULT_RUNS) -> list[Call]:
    calls, seen = [], set()
    for path in sorted(glob.glob(os.path.join(runs_dir, "run-*.json"))):
        for r in json.load(open(path)).get("results") or []:
            if not r.get("transcript") or r.get("error"):
                continue
            call = render(r["transcript"])
            digest = hashlib.sha1(call.encode()).hexdigest()[:12]
            text = card(r["agent"]) + "\n\n" + call     # the business facts the label judge also had
            if digest in seen:          # the same conversation saved by two runs
                continue
            seen.add(digest)
            calls.append(Call(
                id=digest, agent=r["agent"], scenario=r["scenario"], category=r["category"], text=text,
                labels={k: r.get(k) if isinstance(r.get(k), bool) else None for k in LABELS},
            ))
    return calls


def assign_splits(calls: list[Call]) -> dict[str, str]:
    """Scenario-level split, stratified by category: within each category, scenarios (in a stable hashed order)
    fill ~70% of its calls into train, ~15% into validation and the rest into test. Whole scenarios stay together,
    so test calls come from situations never seen in training, and every category appears in every split."""
    by_cat: dict[str, dict[str, int]] = {}
    for c in calls:
        by_cat.setdefault(c.category, {}).setdefault(c.scenario, 0)
        by_cat[c.category][c.scenario] += 1
    split: dict[str, str] = {}
    for cat, scen in by_cat.items():
        total, done = sum(scen.values()), 0
        order = sorted(scen, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        for i, sc in enumerate(order):
            share = done / total
            # small categories still get one scenario in validation and test when they have three or more
            if len(order) >= 3 and i == len(order) - 1:
                split[sc] = "test"
            elif len(order) >= 3 and i == len(order) - 2:
                split[sc] = "val"
            else:
                split[sc] = "train" if share < 0.70 else "val" if share < 0.85 else "test"
            done += scen[sc]
    return split


def build(out_dir: str = "data", runs_dir: str = DEFAULT_RUNS) -> dict[str, int]:
    os.makedirs(out_dir, exist_ok=True)
    counts: dict[str, int] = {}
    files = {s: open(os.path.join(out_dir, f"{s}.jsonl"), "w") for s in ("train", "val", "test")}
    calls = load_calls(runs_dir)
    splits = assign_splits(calls)
    for c in calls:
        s = splits[c.scenario]
        files[s].write(json.dumps(c.__dict__) + "\n")
        counts[s] = counts.get(s, 0) + 1
    for f in files.values():
        f.close()
    return counts


def read(split: str, data_dir: str = "data") -> list[dict]:
    return [json.loads(line) for line in open(os.path.join(data_dir, f"{split}.jsonl"))]


if __name__ == "__main__":
    print(build())
