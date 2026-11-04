"""Which lines of the call drove each verdict: leave-one-line-out (occlusion) attribution.

For a call with N lines, the grader scores N extra copies, each with one line removed. A line's effect on a
label is how far the probability moves when it is gone: if removing the agent's "$49 off today" line drops
`hallucinated` from 0.91 to 0.22, that line is the evidence. It is model-agnostic, needs no gradients, and
its output is something a business owner can read: the exact sentence that failed the call.

The business card is never removed (it is reference, not evidence); only the call lines are.
"""

from __future__ import annotations

from typing import Callable

Scorer = Callable[[list[str]], list[list[float]]]     # texts -> per-text label probabilities


def split_card(text: str) -> tuple[str, list[str]]:
    """'card\\n\\ncall lines' -> (card, lines). Text without a card (no 'BUSINESS:' header) is all call."""
    if text.startswith("BUSINESS:") and "\n\n" in text:
        card, call = text.split("\n\n", 1)
        return card, call.splitlines()
    return "", text.splitlines()


def join(card: str, lines: list[str]) -> str:
    return (card + "\n\n" if card else "") + "\n".join(lines)


def evidence(text: str, score: Scorer, labels: list[str], verdict: dict[str, bool], top: int = 3,
             min_effect: float = 0.02) -> dict[str, list[dict]]:
    """For each label, up to `top` lines that most support the verdict the grader gave.

    effect > 0 means the line pushed the probability toward the verdict (toward True for a True verdict,
    toward False for a False one)."""
    card, lines = split_card(text)
    variants = [join(card, lines)] + [join(card, lines[:i] + lines[i + 1:]) for i in range(len(lines))]
    probs = score(variants)
    full, out = probs[0], {}
    for j, lab in enumerate(labels):
        sign = 1 if verdict[lab] else -1
        ranked = sorted(((sign * (full[j] - probs[i + 1][j]), i) for i in range(len(lines))), reverse=True)
        out[lab] = [{"line": i + 1, "text": lines[i], "effect": round(e, 3)}
                    for e, i in ranked[:top] if e >= min_effect]
    return out
