"""Offline contracts for the local Laya decision backend. No network, no model load."""

import pytest

from jev_ultrafast import laya, model
from jev_ultrafast.browser import fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def big_page(clicks, fills=0):
    actions = [
        {"id": f"f{n}", "kind": "fill", "label": f"Field {n}", "role": "textbox", "value": "", "node": n + 1}
        for n in range(fills)
    ] + [
        {"id": f"c{n}", "kind": "click", "label": f"Button {n}", "role": "button", "value": "", "node": 1000 + n}
        for n in range(clicks)
    ]
    actions.append({"id": "wait", "kind": "wait", "label": "Wait"})
    state = {
        "url": "https://example.test/", "title": "Search", "text": "Search", "scroll": {"y": 0}, "actions": actions
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def test_one_batched_call_carries_all_heads(monkeypatch):
    calls = []

    def fake(state, questions):
        calls.append(questions)
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(questions["click_target"]["criteria"], "2"),
                "type_text_target": choice(questions["type_text_target"]["criteria"], "1"),
            },
            "usage": {},
        }

    monkeypatch.setattr(laya, "_predict", fake)
    d = laya.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert set(calls[0]) == {"operation", "click_target", "type_text_target"}
    assert set(d) == {
        "choice", "operation", "target", "confidence", "probabilities", "operation_probabilities",
        "target_probabilities", "target_confidence", "raw_answers", "model", "usage", "latency_ms",
        "request", "backend", "stop_override",
    }
    assert d["backend"] == "laya"
    assert d["operation"] == "CLICK" and d["target"] == "2" and d["choice"] == "e3"


def test_tournament_chunks_large_heads(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_OPTIONS", "10")
    calls = []

    def fake(state, questions):
        calls.append(questions)
        answers = {}
        for qid, q in questions.items():
            if qid == "operation":
                answers[qid] = choice(q["criteria"], "CLICK")
            elif qid == "runoff":
                answers[qid] = choice(q["criteria"], "11")
            else:  # chunk questions: the first criterion wins its chunk
                answers[qid] = choice(q["criteria"], next(iter(q["criteria"])))
        return {"model": "test", "answers": answers, "usage": {}}

    monkeypatch.setattr(laya, "_predict", fake)
    d = laya.choose(big_page(clicks=15), "Find a book", [])
    assert "click_target__chunk0" in calls[0] and "click_target__chunk1" in calls[0]
    assert "click_target" not in calls[0]
    assert len(calls) == 2
    assert list(calls[1]) == ["runoff"]
    assert set(calls[1]["runoff"]["criteria"]) == {"1", "11"}
    assert d["target"] == "11"
    assert set(d["probabilities"]) == {f"c{n}" for n in range(15)}
    assert sum(d["probabilities"].values()) == pytest.approx(1.0, abs=0.02)


def test_runoff_only_for_chosen_operation(monkeypatch):
    monkeypatch.setenv("LAYA_MAX_OPTIONS", "10")
    calls = []

    def fake(state, questions):
        calls.append(questions)
        answers = {}
        for qid, q in questions.items():
            if qid == "operation":
                answers[qid] = choice(q["criteria"], "TYPE_TEXT")
            elif qid == "runoff":
                answers[qid] = choice(q["criteria"], next(iter(q["criteria"])))
            else:
                answers[qid] = choice(q["criteria"], next(iter(q["criteria"])))
        return {"model": "test", "answers": answers, "usage": {}}

    monkeypatch.setattr(laya, "_predict", fake)
    state = big_page(clicks=15, fills=12)
    text_indices = set(model.action_space(state["actions"])[1]["TYPE_TEXT"])
    d = laya.choose(state, "Find a book", [])
    assert set(calls[1]["runoff"]["criteria"]) <= text_indices
    assert d["operation"] == "TYPE_TEXT"


def test_invalid_laya_answer_rejected(monkeypatch):
    def fake(state, questions):
        answers = {"operation": {"choice": "invented", "confidence": 1.0,
                                 "probabilities": {i: 0.5 for i in questions["operation"]["criteria"]}}}
        return {"model": "test", "answers": answers, "usage": {}}

    monkeypatch.setattr(laya, "_predict", fake)
    with pytest.raises(ValueError, match="Invalid Laya"):
        laya.choose(page(), "Find a book", [])


def test_direct_head_validates_full_index_set(monkeypatch):
    def fake(state, questions):
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(["999"], "999"),
            },
            "usage": {},
        }

    monkeypatch.setattr(laya, "_predict", fake)
    with pytest.raises(ValueError, match="Invalid Laya"):
        laya.choose(page(), "Find a book", [])


def test_pack_state_packs_decision_first_and_truncates():
    state = {
        "title": "Search",
        "url": "https://example.test/",
        "text": "x" * 5000,
        "actions": [{"id": "e1", "kind": "fill", "label": "City", "role": "textbox", "value": "Zürich", "node": 1}],
    }
    packed = laya.pack_state(state, "g" * 1000, [])
    assert list(packed) == ["goal", "recent_actions", "page", "field_values", "page_text"]
    assert len(packed["goal"]) <= 200
    assert len(packed["page_text"]) <= 500
    assert "Zürich" in packed["field_values"]


def _stop_guard_fake(probs):
    def fake(state, questions):
        answers = {
            "operation": {"choice": "DONE", "confidence": 0.1, "probabilities": probs},
            "click_target": choice(questions["click_target"]["criteria"], "2"),
            "type_text_target": choice(questions["type_text_target"]["criteria"], "1"),
        }
        return {"model": "test", "answers": answers, "usage": {}}
    return fake


# DONE is the argmax in both (validate_choice demands it); only its margin varies.
LOW_STOP = {"CLICK": 0.25, "TYPE_TEXT": 0.13, "WAIT": 0.13, "DONE": 0.32, "BLOCKED": 0.17}
HIGH_STOP = {"CLICK": 0.08, "TYPE_TEXT": 0.04, "WAIT": 0.04, "DONE": 0.8, "BLOCKED": 0.04}


def test_low_confidence_stop_falls_back_to_actionable(monkeypatch):
    monkeypatch.setattr(laya, "_predict", _stop_guard_fake(LOW_STOP))
    d = laya.choose(page(), "Find a book", [])
    assert d["operation"] == "CLICK" and d["target"] == "2" and d["choice"] == "e3"
    assert d["stop_override"]["from"] == "DONE"
    assert d["stop_override"]["probability"] == pytest.approx(0.32)


def test_confident_stop_is_respected(monkeypatch):
    monkeypatch.setattr(laya, "_predict", _stop_guard_fake(HIGH_STOP))
    d = laya.choose(page(), "Find a book", [])
    assert d["operation"] == "DONE" and d["choice"] == "DONE"
    assert d["stop_override"] is None


def test_stop_guard_threshold_is_env_tunable(monkeypatch):
    monkeypatch.setenv("LAYA_STOP_MIN_P", "0.2")
    monkeypatch.setattr(laya, "_predict", _stop_guard_fake(LOW_STOP))
    d = laya.choose(page(), "Find a book", [])
    assert d["operation"] == "DONE"
    assert d["stop_override"] is None


def test_dispatch_routes_to_laya_by_default(monkeypatch):
    monkeypatch.delenv("JEV_BACKEND", raising=False)

    def fake(state, questions):
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(questions["type_text_target"]["criteria"], "1"),
            },
            "usage": {},
        }

    monkeypatch.setattr(laya, "_predict", fake)
    d = model.choose(page(), "Find a book", [])
    assert d["backend"] == "laya"
