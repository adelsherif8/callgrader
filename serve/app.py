"""CallGrader HTTP service: grade a call, show why, and grade every Retell call automatically.

    uvicorn serve.app:app --port 8000          # then open http://localhost:8000 for the demo page

    POST /grade            {"transcript": "...", "agent": "dental" | "card": "BUSINESS: ...", "explain": true}
    POST /retell/webhook   Retell's call_analyzed event, signed with X-Retell-Signature (RETELL_API_KEY)
    GET  /calls            the latest graded calls (newest first)

Env: GRADER_DIR (model folder, default models/encoder), RETELL_API_KEY, CARDS_FILE (JSON: Retell agent_id ->
fact card text), GRADED_LOG (JSONL file the graded calls are appended to, default results/graded.jsonl).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from callgrader import explain, retell
from callgrader.context import CARDS, card as demo_card
from callgrader.data import LABELS

MODEL_DIR = os.getenv("GRADER_DIR", "models/encoder")
HERE = os.path.dirname(__file__)


class CallIn(BaseModel):
    transcript: str = Field(min_length=1, max_length=60_000, description="One turn per line: 'AGENT: ...' / 'CALLER: ...'")
    agent: str | None = Field(None, description="A demo business (recovery, dental, receptionist) to use its fact card")
    card: str | None = Field(None, max_length=4_000, description="Your own fact card: prices, hours, handoff rules")
    explain: bool = Field(False, description="Also return the lines that drove each verdict (slower: N+1 passes)")


class Verdict(BaseModel):
    labels: dict[str, bool]
    scores: dict[str, float]
    flagged: list[str]
    evidence: dict[str, list[dict]] | None = None
    ms: float


app = FastAPI(title="CallGrader", description="Grades AI voice-agent calls: pass/fail, made-up answers, missed handoffs.")
_state: dict = {}
_lock = threading.Lock()
_recent: deque = deque(maxlen=200)


def _load():
    """The fine-tuned ModernBERT grader, loaded once. Tests swap `_state['score']` for a stub."""
    with _lock:
        if "score" not in _state:
            import torch
            from transformers import AutoTokenizer
            from callgrader.encoder import MODEL, Grader, device, encode

            model = Grader(MODEL)
            model.load_state_dict(torch.load(f"{MODEL_DIR}/grader.pt", map_location="cpu"))
            model = model.to(device()).eval()
            tok = AutoTokenizer.from_pretrained(MODEL)

            @torch.no_grad()
            def score(texts: list[str]) -> list[list[float]]:
                out = []
                for i in range(0, len(texts), 4):
                    enc = {k: v.to(device()) for k, v in encode(tok, texts[i:i + 4]).items()}
                    out += torch.sigmoid(model(**enc)).float().cpu().tolist()
                return out

            _state.update(score=score, th=json.load(open(f"{MODEL_DIR}/thresholds.json")))
    return _state


# What a True verdict means for the business: these are the ones a person should look at.
BAD = {"passed": False, "task_success": False, "missed_handoff": True, "hallucinated": True, "injection_resisted": False}


def grade_text(text: str, want_evidence: bool = False) -> Verdict:
    s, t0 = _load(), time.perf_counter()
    probs = s["score"]([text])[0]
    labels = {l: p >= s["th"][l] for l, p in zip(LABELS, probs)}
    ev = None
    if want_evidence:
        ev = explain.evidence(text, s["score"], LABELS, labels)
    return Verdict(labels=labels, scores={l: round(p, 3) for l, p in zip(LABELS, probs)},
                   flagged=[l for l in LABELS if labels[l] == BAD[l]], evidence=ev,
                   ms=round((time.perf_counter() - t0) * 1000, 1))


def with_card(transcript: str, agent: str | None = None, card: str | None = None) -> str:
    c = (card or "").strip() or demo_card(agent or "")
    if c and not c.startswith("BUSINESS:"):
        c = "BUSINESS: " + c
    return (c + "\n\n" if c else "") + transcript.strip()


def _cards() -> dict[str, str]:
    path = os.getenv("CARDS_FILE", "")
    return json.load(open(path)) if path and os.path.exists(path) else {}


def _keep(entry: dict) -> None:
    _recent.appendleft(entry)
    path = os.getenv("GRADED_LOG", "results/graded.jsonl")
    try:
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


@app.get("/")
def page():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/healthz")
def health():
    return {"ok": os.path.exists(f"{MODEL_DIR}/grader.pt") or "score" in _state}


@app.get("/demo-agents")
def demo_agents():
    return {k: v for k, v in CARDS.items()}


@app.get("/samples")
def samples(n: int = 8):
    """A few held-out test calls (the grader never trained on them), one per kind, for the demo page."""
    path, seen, out = "data/test.jsonl", set(), []
    if not os.path.exists(path):
        return out
    for line in open(path):
        r = json.loads(line)
        if r["category"] in seen:
            continue
        seen.add(r["category"])
        out.append({"agent": r["agent"], "category": r["category"], "scenario": r["scenario"],
                    "transcript": "\n".join(explain.split_card(r["text"])[1]),
                    "expected": r["labels"]})
        if len(out) >= n:
            break
    return out


@app.post("/grade", response_model=Verdict)
def grade(call: CallIn):
    return grade_text(with_card(call.transcript, call.agent, call.card), call.explain)


@app.post("/retell/webhook")
async def retell_webhook(request: Request):
    raw = await request.body()
    if not retell.signed(raw, request.headers.get("x-retell-signature", ""), os.getenv("RETELL_API_KEY", "").strip()):
        return JSONResponse({"ok": False, "error": "bad signature"}, status_code=401)
    body = json.loads(raw or b"{}")
    if body.get("event") != "call_analyzed":          # transcript is final only once the call is analysed
        return {"ok": True, "graded": False}
    call = body.get("call") or {}
    text = retell.transcript(call)
    if not text.strip():
        return {"ok": True, "graded": False}
    card = _cards().get(call.get("agent_id", ""), "")
    v = grade_text(with_card(text, card=card), want_evidence=True)
    _keep({"call_id": call.get("call_id"), "agent_id": call.get("agent_id"), "at": int(time.time()),
           "had_card": bool(card), **v.model_dump()})
    return {"ok": True, "graded": True, "flagged": v.flagged}


@app.get("/calls")
def calls(limit: int = 50):
    return list(_recent)[:max(1, min(limit, 200))]
