"""Tests for ConversationRecord output models (Task 49)."""

from __future__ import annotations

import json

import pytest

from tooluse_gen.agents.conversation_models import Conversation, ConversationMetadata, Message
from tooluse_gen.agents.execution_models import ToolCallRequest
from tooluse_gen.core.output_models import (
    ConversationRecord,
    from_conversation,
    validate_conversation_record,
    validate_record,
)
from tooluse_gen.evaluation.models import JudgeScores as EvalJudgeScores

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conversation() -> Conversation:
    tc = ToolCallRequest(
        endpoint_id="h/s", tool_id="h", tool_name="H", endpoint_name="S",
        arguments={"city": "Paris"},
    )
    conv = Conversation(messages=[
        Message(role="user", content="Find hotels"),
        Message(role="assistant", tool_calls=[tc]),
        Message(role="tool", tool_call_id=tc.call_id, tool_output={"id": "htl_1"}),
        Message(role="assistant", content="Found hotel."),
    ])
    conv.metadata = ConversationMetadata(
        seed=42, tools_used=["h"], num_turns=4, num_tool_calls=1,
    )
    return conv


def _make_valid_dict() -> dict:
    return {
        "conversation_id": "c1",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "judge_scores": {"tool_correctness": 4, "argument_grounding": 3,
                         "task_completion": 5, "naturalness": 4},
        "metadata": {"seed": 42},
    }


# ===================================================================
# ConversationRecord
# ===================================================================


class TestConversationRecord:
    def test_construction(self):
        r = ConversationRecord(
            conversation_id="c1",
            messages=[{"role": "user", "content": "hi"}],
            judge_scores={"tool_correctness": 4},
            metadata={"seed": 42},
        )
        assert r.conversation_id == "c1"
        assert len(r.messages) == 1

    def test_to_jsonl(self):
        r = ConversationRecord(
            conversation_id="c1",
            messages=[{"role": "user", "content": "hi"}],
        )
        s = r.to_jsonl()
        parsed = json.loads(s)
        assert parsed["conversation_id"] == "c1"

    def test_to_dict(self):
        r = ConversationRecord(
            conversation_id="c1",
            messages=[{"role": "user", "content": "hi"}],
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "conversation_id" in d

    def test_keys(self):
        r = ConversationRecord(
            conversation_id="c1",
            messages=[{"role": "user", "content": "hi"}],
            judge_scores={"tc": 5},
            metadata={"seed": 1},
        )
        d = r.to_dict()
        assert set(d.keys()) == {"conversation_id", "messages", "judge_scores", "metadata"}

    def test_judge_scores_none(self):
        r = ConversationRecord(
            conversation_id="c1", messages=[{"role": "user", "content": "hi"}],
        )
        assert r.judge_scores is None

    def test_serialization_round_trip(self):
        r = ConversationRecord(
            conversation_id="c1",
            messages=[{"role": "user", "content": "hi"}],
            judge_scores={"tc": 3},
        )
        data = r.model_dump()
        restored = ConversationRecord.model_validate(data)
        assert restored.conversation_id == "c1"


# ===================================================================
# from_conversation
# ===================================================================


class TestFromConversation:
    def test_converts(self):
        conv = _make_conversation()
        record = from_conversation(conv)
        assert isinstance(record, ConversationRecord)

    def test_conversation_id_preserved(self):
        conv = _make_conversation()
        record = from_conversation(conv)
        assert record.conversation_id == conv.conversation_id

    def test_messages_serialized(self):
        conv = _make_conversation()
        record = from_conversation(conv)
        assert isinstance(record.messages, list)
        assert len(record.messages) == 4

    def test_messages_have_role_content(self):
        conv = _make_conversation()
        record = from_conversation(conv)
        for msg in record.messages:
            assert "role" in msg
            assert "content" in msg

    def test_tool_calls_have_endpoint(self):
        conv = _make_conversation()
        record = from_conversation(conv)
        tc_msg = record.messages[1]
        assert "tool_calls" in tc_msg
        assert tc_msg["tool_calls"][0]["endpoint"] == "h/s"
        assert "arguments" in tc_msg["tool_calls"][0]

    def test_eval_scores_used(self):
        conv = _make_conversation()
        scores = EvalJudgeScores(
            tool_correctness=5, argument_grounding=4,
            task_completion=5, naturalness=4,
        )
        record = from_conversation(conv, eval_scores=scores)
        assert record.judge_scores == {
            "tool_correctness": 5, "argument_grounding": 4,
            "task_completion": 5, "naturalness": 4,
        }

    def test_fallback_to_conv_scores(self):
        from tooluse_gen.agents.conversation_models import (
            JudgeScores as AgentJudgeScores,
        )

        conv = _make_conversation()
        conv.judge_scores = AgentJudgeScores(
            naturalness=4.0, tool_correctness=3.0,
        )
        record = from_conversation(conv)
        assert record.judge_scores is not None
        assert "naturalness" in record.judge_scores

    def test_no_scores(self):
        conv = _make_conversation()
        record = from_conversation(conv)
        assert record.judge_scores is None


# ===================================================================
# validate_record
# ===================================================================


class TestValidateRecord:
    def test_valid(self):
        ok, errs = validate_record(_make_valid_dict())
        assert ok and errs == []

    def test_missing_conversation_id(self):
        d = _make_valid_dict()
        del d["conversation_id"]
        ok, errs = validate_record(d)
        assert not ok
        assert any("conversation_id" in e for e in errs)

    def test_empty_conversation_id(self):
        d = _make_valid_dict()
        d["conversation_id"] = ""
        ok, errs = validate_record(d)
        assert not ok

    def test_missing_messages(self):
        d = _make_valid_dict()
        del d["messages"]
        ok, errs = validate_record(d)
        assert not ok

    def test_empty_messages(self):
        d = _make_valid_dict()
        d["messages"] = []
        ok, errs = validate_record(d)
        assert not ok

    def test_message_missing_role(self):
        d = _make_valid_dict()
        d["messages"] = [{"content": "hi"}]
        ok, errs = validate_record(d)
        assert not ok
        assert any("role" in e for e in errs)

    def test_invalid_role(self):
        d = _make_valid_dict()
        d["messages"] = [{"role": "system", "content": "hi"}]
        ok, errs = validate_record(d)
        assert not ok
        assert any("invalid role" in e for e in errs)

    def test_tool_call_missing_endpoint(self):
        d = _make_valid_dict()
        d["messages"] = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"arguments": {}}]},
        ]
        ok, errs = validate_record(d)
        assert not ok
        assert any("endpoint" in e for e in errs)

    def test_valid_tool_call(self):
        d = _make_valid_dict()
        d["messages"] = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"endpoint": "h/s", "arguments": {"city": "Paris"}}]},
        ]
        ok, errs = validate_record(d)
        assert ok

    def test_judge_scores_none_valid(self):
        d = _make_valid_dict()
        d["judge_scores"] = None
        ok, errs = validate_record(d)
        assert ok


# ===================================================================
# validate_conversation_record
# ===================================================================


class TestValidateConversationRecord:
    def test_valid(self):
        conv = _make_conversation()
        record = from_conversation(conv)
        ok, errs = validate_conversation_record(record)
        assert ok and errs == []

    def test_delegates(self):
        record = ConversationRecord(
            conversation_id="c1",
            messages=[{"role": "user", "content": "hi"}],
        )
        ok, errs = validate_conversation_record(record)
        assert ok

    def test_invalid(self):
        record = ConversationRecord(
            conversation_id="",
            messages=[],
        )
        ok, errs = validate_conversation_record(record)
        assert not ok
        assert len(errs) > 0


# ===================================================================
# Schema compliance with spec
# ===================================================================


class TestSchemaCompliance:
    def test_spec_example(self):
        record = ConversationRecord(
            conversation_id="conv_0042",
            messages=[
                {"role": "user", "content": "Find me a hotel in Paris"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"endpoint": "hotels/search", "arguments": {"city": "Paris"}}
                ]},
                {"role": "tool", "content": {"results": [{"id": "htl_881"}]}},
                {"role": "assistant", "content": "Booked!"},
            ],
            judge_scores={"tool_correctness": 4, "argument_grounding": 5,
                          "task_completion": 5, "naturalness": 4},
            metadata={"seed": 42, "tools_used": ["hotels"], "num_turns": 4},
        )
        ok, errs = validate_conversation_record(record)
        assert ok and errs == []

    def test_required_fields(self):
        record = from_conversation(_make_conversation())
        d = record.to_dict()
        assert "conversation_id" in d
        assert "messages" in d
        assert "metadata" in d

    def test_tool_content_dict(self):
        record = from_conversation(_make_conversation())
        tool_msgs = [m for m in record.messages if m["role"] == "tool"]
        for m in tool_msgs:
            assert isinstance(m["content"], dict)

    def test_round_trip(self):
        conv = _make_conversation()
        scores = EvalJudgeScores(
            tool_correctness=5, argument_grounding=4,
            task_completion=5, naturalness=4,
        )
        record = from_conversation(conv, eval_scores=scores)
        json_str = record.to_jsonl()
        parsed = json.loads(json_str)
        ok, errs = validate_record(parsed)
        assert ok and errs == []
        assert parsed["judge_scores"]["tool_correctness"] == 5
