"""The dataset builder: transcripts render with tool calls, duplicates drop, and no scenario leaks across splits."""

import json

from callgrader.data import Call, LABELS, assign_splits, load_calls, render
from callgrader.metrics import report, score


def test_render_shows_turns_and_tool_calls_without_the_spoken_filler():
    text = render([
        {"role": "agent", "text": "Blue Ridge Plumbing, what's going on?"},
        {"role": "caller", "text": "Water everywhere"},
        {"role": "agent", "text": "", "tool_calls": [{"name": "book_appointment",
                                                       "args": {"window": "2-4pm", "execution_message": "One moment"}}]},
    ])
    assert text.splitlines() == ["AGENT: Blue Ridge Plumbing, what's going on?", "CALLER: Water everywhere",
                                 "TOOL book_appointment(window=2-4pm)"]


def run_file(tmp_path, results):
    (tmp_path / "run-1.json").write_text(json.dumps({"results": results}))
    return str(tmp_path)


def result(scenario, text, **labels):
    return {"agent": "recovery", "scenario": scenario, "category": "emergency", "error": None,
            "transcript": [{"role": "caller", "text": text}], **labels}


def test_duplicates_and_errored_calls_are_dropped_and_missing_labels_are_none(tmp_path):
    runs = run_file(tmp_path, [result("a", "same", passed=True), result("a", "same", passed=True),
                               {**result("b", "x"), "error": "timeout"}, result("c", "other", passed=False)])
    calls = load_calls(runs)
    assert [c.scenario for c in calls] == ["a", "c"]
    assert calls[0].labels["passed"] is True and calls[0].labels["injection_resisted"] is None
    assert set(calls[0].labels) == set(LABELS)


def test_whole_scenarios_stay_in_one_split_and_every_split_is_used():
    calls = [Call(f"{s}-{i}", "recovery", s, cat, "t", {}) for cat in ("a", "b") for s in (f"{cat}1", f"{cat}2", f"{cat}3", f"{cat}4")
             for i in range(5)]
    splits = assign_splits(calls)
    assert set(splits) == {c.scenario for c in calls}
    assert set(splits.values()) == {"train", "val", "test"}
    assert assign_splits(calls) == splits                     # stable between runs


def test_metrics_skip_calls_where_a_label_does_not_apply():
    assert score([True, False, True, False], [True, True, False, False]) == {
        "n": 4, "positives": 2, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    rows = [{"labels": {"x": True}}, {"labels": {"x": None}}, {"labels": {"x": False}}]
    out = report(rows, [{"x": True}, {"x": True}, {"x": False}], ["x"])
    assert out["x"]["n"] == 2 and out["x"]["f1"] == 1.0 and out["macro_f1"] == 1.0
