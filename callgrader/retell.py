"""Turn a Retell call (webhook payload or get-call response) into the transcript CallGrader reads.

Retell's `transcript_with_tool_calls` is a list of utterances ("agent" / "user") interleaved with
"tool_call_invocation" and "tool_call_result" entries. The grader was trained on VoiceDesk's rendering:
`AGENT: ...`, `CALLER: ...` and `TOOL name(k=v)` lines, so this module produces exactly that.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time

from callgrader.data import render

_SIG = re.compile(r"v=(\d+),d=([0-9a-fA-F]+)")
MAX_AGE_MS = 5 * 60 * 1000


def turns(call: dict) -> list[dict]:
    """Retell call -> [{"role", "text", "tool_calls"}], the shape `data.render` takes."""
    items = call.get("transcript_with_tool_calls") or call.get("transcript_object") or []
    out: list[dict] = []
    for it in items:
        role = it.get("role")
        if role in ("agent", "user"):
            out.append({"role": "agent" if role == "agent" else "caller", "text": it.get("content") or ""})
        elif role == "tool_call_invocation":
            args = it.get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                args = {"raw": args}
            out.append({"role": "agent", "text": "", "tool_calls": [{"name": it.get("name", "?"), "args": args}]})
        # tool_call_result entries are the server's reply, not something the agent said: skipped, as in training
    return out


def transcript(call: dict) -> str:
    """Retell call -> the text block the grader reads (without the business card)."""
    if not (call.get("transcript_with_tool_calls") or call.get("transcript_object")) and call.get("transcript"):
        # plain-text fallback: Retell's "Agent: ... / User: ..." lines
        lines = []
        for line in str(call["transcript"]).splitlines():
            who, _, text = line.partition(":")
            if text.strip():
                lines.append(f"{'AGENT' if who.strip().lower() == 'agent' else 'CALLER'}: {text.strip()}")
        return "\n".join(lines)
    return render(turns(call))


def signed(raw: bytes, signature: str, api_key: str, now_ms: float | None = None) -> bool:
    """X-Retell-Signature: v=<ms>,d=<hex HMAC-SHA256 of body + timestamp, keyed by the Retell API key>.
    Rejects a missing key, a malformed header, a stale timestamp (replay) and a wrong digest."""
    m = _SIG.fullmatch((signature or "").strip())
    now = time.time() * 1000 if now_ms is None else now_ms
    if not api_key or not m or abs(now - int(m[1])) > MAX_AGE_MS:
        return False
    digest = hmac.new(api_key.encode(), raw + m[1].encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, m[2].lower())
