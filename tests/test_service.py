"""Retell adapter, webhook signature, evidence lines, folds, and the HTTP API (with a stub model)."""

import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from callgrader import explain, retell
from callgrader.cv import folds
from callgrader.data import LABELS

RETELL_CALL = {
    "call_id": "c1", "agent_id": "agent_abc",
    "transcript_with_tool_calls": [
        {"role": "agent", "content": "Brightside Dental, how can I help?"},
        {"role": "user", "content": "Do you do implants on Saturday?"},
        {"role": "tool_call_invocation", "name": "book_patient", "arguments": '{"name": "Ana", "slot": "Mon 9am"}'},
        {"role": "tool_call_result", "content": "booked"},
        {"role": "agent", "content": "Booked you Monday at 9."},
    ],
}


def test_retell_call_renders_like_training_data():
    assert retell.transcript(RETELL_CALL).splitlines() == [
        "AGENT: Brightside Dental, how can I help?", "CALLER: Do you do implants on Saturday?",
        "TOOL book_patient(name=Ana, slot=Mon 9am)", "AGENT: Booked you Monday at 9."]


def test_retell_plain_text_transcript_fallback():
    assert retell.transcript({"transcript": "Agent: Hi\nUser: Hello there"}) == "AGENT: Hi\nCALLER: Hello there"


def sign(body: bytes, key: str, ts: int | None = None) -> str:
    ts = ts or int(time.time() * 1000)
    return f"v={ts},d=" + hmac.new(key.encode(), body + str(ts).encode(), hashlib.sha256).hexdigest()


def test_signature_accepts_good_and_rejects_wrong_key_stale_and_malformed():
    body = b'{"event":"call_analyzed"}'
    assert retell.signed(body, sign(body, "k"), "k")
    assert not retell.signed(body, sign(body, "other"), "k")
    assert not retell.signed(body, sign(body, "k", ts=int(time.time() * 1000) - 10 * 60 * 1000), "k")
    assert not retell.signed(body, "garbage", "k")
    assert not retell.signed(body, sign(body, "k"), "")          # no key configured: nobody gets in


def test_evidence_finds_the_line_that_moves_the_score():
    text = "BUSINESS: Brightside Dental. $95 exam.\n\nAGENT: Hi\nAGENT: Implants are $499 today only\nCALLER: Great"

    def score(texts):          # 'hallucinated' is high only while the made-up price is present
        return [[0.9 if ("$499" in t and l == "hallucinated") else 0.5 for l in LABELS] for t in texts]

    ev = explain.evidence(text, score, LABELS, {l: l == "hallucinated" for l in LABELS})
    assert ev["hallucinated"][0] == {"line": 2, "text": "AGENT: Implants are $499 today only", "effect": 0.4}
    assert ev["task_success"] == []          # nothing moves the other labels


def test_folds_keep_scenarios_whole_and_spread_each_category():
    rows = [{"scenario": f"{c}-{i}", "category": c} for c in "ab" for i in range(10) for _ in range(3)]
    f = folds(rows, 5)
    assert set(f.values()) == set(range(5))
    for c in "ab":
        assert {f[f"{c}-{i}"] for i in range(10)} == set(range(5))


@pytest.fixture
def client(monkeypatch, tmp_path):
    from serve import app as svc

    def score(texts):
        return [[0.2, 0.3, 0.1, 0.95 if "$499" in t else 0.05, 0.5] for t in texts]

    monkeypatch.setitem(svc._state, "score", score)
    monkeypatch.setitem(svc._state, "th", {l: 0.5 for l in LABELS})
    monkeypatch.setenv("RETELL_API_KEY", "key_test")
    monkeypatch.setenv("GRADED_LOG", str(tmp_path / "graded.jsonl"))
    cards = tmp_path / "cards.json"
    cards.write_text(json.dumps({"agent_abc": "BUSINESS: Brightside Dental. Only price: $95 exam."}))
    monkeypatch.setenv("CARDS_FILE", str(cards))
    svc._recent.clear()
    return TestClient(svc.app)


def test_grade_flags_made_up_price_and_shows_the_line(client):
    r = client.post("/grade", json={"transcript": "AGENT: Hi\nAGENT: Implants are $499\nCALLER: ok",
                                    "agent": "dental", "explain": True}).json()
    assert r["labels"]["hallucinated"] is True and "hallucinated" in r["flagged"]
    assert r["evidence"]["hallucinated"][0]["text"] == "AGENT: Implants are $499"


def test_retell_webhook_needs_a_signature_and_grades_analysed_calls(client):
    body = json.dumps({"event": "call_analyzed", "call": RETELL_CALL}).encode()
    assert client.post("/retell/webhook", content=body).status_code == 401
    r = client.post("/retell/webhook", content=body, headers={"x-retell-signature": sign(body, "key_test")})
    assert r.json() == {"ok": True, "graded": True, "flagged": ["passed", "task_success"]}
    saved = client.get("/calls").json()[0]
    assert saved["call_id"] == "c1" and saved["had_card"] is True


def test_retell_webhook_ignores_other_events(client):
    body = json.dumps({"event": "call_started", "call": RETELL_CALL}).encode()
    r = client.post("/retell/webhook", content=body, headers={"x-retell-signature": sign(body, "key_test")})
    assert r.json() == {"ok": True, "graded": False}


def test_cross_validation_resumes_finished_folds_from_checkpoints(tmp_path):
    from callgrader.cv import cross_validate
    rows = [{"scenario": f"s{i}", "category": "c", "labels": {l: bool(i % 2) for l in LABELS}} for i in range(10)]
    calls = []

    def fn(tr, va, te):
        calls.append(len(te))
        return [{l: True for l in LABELS} for _ in te]

    first = cross_validate("m", fn, rows, k=5, log=lambda d: None, ckpt=str(tmp_path))
    assert len(calls) == 5 and len(list(tmp_path.glob("m-fold*.json"))) == 5
    (tmp_path / "m-fold3.json").unlink()                     # pretend the reset happened during fold 3
    again = cross_validate("m", fn, rows, k=5, log=lambda d: None, ckpt=str(tmp_path))
    assert len(calls) == 6                                    # only fold 3 was trained again
    assert again["metrics"] == first["metrics"]
