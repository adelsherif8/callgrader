from callgrader.llm import answer, parse


def test_answers_round_trip_and_unreadable_output_counts_as_false():
    labels = {"passed": True, "task_success": None, "missed_handoff": False, "hallucinated": True, "injection_resisted": None}
    assert parse(answer(labels)) == {"passed": True, "task_success": False, "missed_handoff": False,
                                     "hallucinated": True, "injection_resisted": False}
    assert parse('Sure! {"passed": true} hope that helps') ["passed"] is True
    assert set(parse("no json here").values()) == {False}
