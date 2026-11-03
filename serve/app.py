"""HTTP API for the fine-tuned grader: POST a call transcript, get back the label verdicts.

    uvicorn serve.app:app --port 8000
    curl -s localhost:8000/grade -H 'content-type: application/json' -d '{"transcript": "AGENT: ...\\nCALLER: ..."}'
"""

from __future__ import annotations

import json
import os
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from callgrader.data import LABELS
from callgrader.encoder import MODEL, Grader, device, encode

MODEL_DIR = os.getenv("GRADER_DIR", "models/encoder")


class CallIn(BaseModel):
    transcript: str = Field(min_length=1, max_length=60_000, description="One turn per line: 'AGENT: ...' / 'CALLER: ...'")


class Verdict(BaseModel):
    labels: dict[str, bool]
    scores: dict[str, float]
    ms: float


app = FastAPI(title="CallGrader")
_state: dict = {}


def _load():
    if not _state:
        model = Grader(MODEL)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/grader.pt", map_location="cpu"))
        _state.update(model=model.to(device()).eval(), tok=AutoTokenizer.from_pretrained(MODEL),
                      th=json.load(open(f"{MODEL_DIR}/thresholds.json")))
    return _state


@app.get("/healthz")
def health():
    return {"ok": os.path.exists(f"{MODEL_DIR}/grader.pt")}


@app.post("/grade", response_model=Verdict)
@torch.no_grad()
def grade(call: CallIn):
    s, t0 = _load(), time.perf_counter()
    enc = {k: v.to(device()) for k, v in encode(s["tok"], [call.transcript]).items()}
    probs = torch.sigmoid(s["model"](**enc))[0].float().cpu().tolist()
    return Verdict(labels={l: p >= s["th"][l] for l, p in zip(LABELS, probs)},
                   scores={l: round(p, 3) for l, p in zip(LABELS, probs)},
                   ms=round((time.perf_counter() - t0) * 1000, 1))
