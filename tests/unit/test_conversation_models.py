"""Tests for conversation data models (Task 35)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tooluse_gen.agents.conversation_models import (
    Conversation,
    ConversationMetadata,
    GenerationConfig,
    JudgeScores,
    Message,
)
from tooluse_gen.agents.execution_models import ToolCallRequest
from tooluse_gen.graph.chain_models import ChainPattern, ChainStep, ToolChain
from tooluse_gen.registry.models import HttpMethod

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call_request(
    endpoint_id: str = "hotels/search",
    tool_id: str = "hotels",
    tool_name: str = "Hotels API",
    endpoint_name: str = "Search Hotels",
    arguments: dict | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        endpoint_id=endpoint_id,
        tool_id=tool_id,
        tool_name=tool_name,
        endpoint_name=endpoint_name,
        method=HttpMethod.GET,
        path="/hotels/search",
        arguments=arguments or {"city": "Paris", "max_price": 200},
    )


def _make_chain() -> ToolChain:
    return ToolChain(
        chain_id="test_chain",
        steps=[
            ChainStep(
                endpoint_id="hotels/search",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Search Hotels",
                method=HttpMethod.GET,
                path="/hotels/search",
                expected_params=["city"],
                domain="Travel",
            ),
            ChainStep(
                endpoint_id="hotels/book",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Book Hotel",
                method=HttpMethod.POST,
                path="/hotels/book",
                expected_params=["hotel_id"],
                domain="Travel",
            ),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


# ===================================================================
# Message
# ===================================================================


class TestMessageConstruction:
    def test_user_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.tool_output is None

    def test_assistant_message_with_content(self):
        msg = Message(role="assistant", content="Sure, let me help.")
        assert msg.role == "assistant"
        assert msg.content == "Sure, let me help."

    def test_assistant_message_with_tool_calls(self):
        tc = _make_tool_call_request()
        msg = Message(role="assistant", tool_calls=[tc])
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].endpoint_id == "hotels/search"

    def test_tool_message_with_output(self):
        msg = Message(
            role="tool",
            tool_call_id="abc-123",
            tool_output={"results": [{"id": "htl_1"}]},
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "abc-123"
        assert msg.tool_output == {"results": [{"id": "htl_1"}]}

    def test_default_metadata_empty(self):
        msg = Message(role="user", content="hi")
        assert msg.metadata == {}

    def test_custom_metadata(self):
        msg = Message(role="user", content="hi", metadata={"source": "test"})
        assert msg.metadata["source"] == "test"

    def test_serialization_round_trip(self):
        tc = _make_tool_call_request()
        msg = Message(role="assistant", tool_calls=[tc])
        data = msg.model_dump()
        restored = Message.model_validate(data)
        assert restored.role == "assistant"
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 1

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="system", content="nope")  # type: ignore[arg-type]


class TestMessageToJsonl:
    def test_user_message_jsonl(self):
        msg = Message(role="user", content="Find hotels")
        d = msg.to_jsonl_dict()
        assert d["role"] == "user"
        assert d["content"] == "Find hotels"
        assert "tool_calls" not in d
        assert "tool_call_id" not in d

    def test_assistant_content_only_jsonl(self):
        msg = Message(role="assistant", content="What's your budget?")
        d = msg.to_jsonl_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "What's your budget?"
        assert "tool_calls" not in d

    def test_assistant_tool_calls_jsonl(self):
        tc = _make_tool_call_request()
        msg = Message(role="assistant", tool_calls=[tc])
        d = msg.to_jsonl_dict()
        assert d["role"] == "assistant"
        assert d["content"] is None
        assert len(d["tool_calls"]) == 1
        tc_dict = d["tool_calls"][0]
        assert tc_dict["endpoint"] == "hotels/search"
        assert tc_dict["arguments"] == {"city": "Paris", "max_price": 200}
        assert tc_dict["tool_name"] == "Hotels API"
        assert "call_id" in tc_dict

    def test_tool_message_jsonl_content_is_output(self):
        output = {"results": [{"id": "htl_1", "name": "Grand Hotel"}]}
        msg = Message(role="tool", tool_call_id="call-1", tool_output=output)
        d = msg.to_jsonl_dict()
        assert d["role"] == "tool"
        assert d["content"] == output
        assert isinstance(d["content"], dict)
        assert d["tool_call_id"] == "call-1"

    def test_tool_message_without_call_id(self):
        msg = Message(role="tool", tool_output={"status": "ok"})
        d = msg.to_jsonl_dict()
        assert "tool_call_id" not in d

    def test_omits_empty_tool_calls(self):
        msg = Message(role="assistant", content="ok", tool_calls=[])
        d = msg.to_jsonl_dict()
        assert "tool_calls" not in d

    def test_multiple_tool_calls(self):
        tc1 = _make_tool_call_request(endpoint_id="a/get", tool_id="a", tool_name="A")
        tc2 = _make_tool_call_request(endpoint_id="b/post", tool_id="b", tool_name="B")
        msg = Message(role="assistant", tool_calls=[tc1, tc2])
        d = msg.to_jsonl_dict()
        assert len(d["tool_calls"]) == 2
        assert d["tool_calls"][0]["endpoint"] == "a/get"
        assert d["tool_calls"][1]["endpoint"] == "b/post"


# ===================================================================
# JudgeScores
# ===================================================================


class TestJudgeScores:
    def test_defaults_all_zero(self):
        s = JudgeScores()
        assert s.naturalness == 0.0
        assert s.tool_correctness == 0.0
        assert s.task_completion == 0.0
        assert s.coherence == 0.0
        assert s.grounding_accuracy == 0.0

    def test_construction_all_fields(self):
        s = JudgeScores(
            naturalness=4.2,
            tool_correctness=4.8,
            task_completion=5.0,
            coherence=4.5,
            grounding_accuracy=4.6,
        )
        assert s.naturalness == 4.2
        assert s.task_completion == 5.0

    def test_mean_score(self):
        s = JudgeScores(
            naturalness=4.0,
            tool_correctness=3.0,
            task_completion=5.0,
            coherence=2.0,
            grounding_accuracy=1.0,
        )
        assert s.mean_score == 3.0

    def test_mean_score_zeros(self):
        assert JudgeScores().mean_score == 0.0

    def test_mean_score_perfect(self):
        s = JudgeScores(
            naturalness=5.0,
            tool_correctness=5.0,
            task_completion=5.0,
            coherence=5.0,
            grounding_accuracy=5.0,
        )
        assert s.mean_score == 5.0

    def test_scores_dict_keys(self):
        s = JudgeScores(naturalness=1.0, tool_correctness=2.0)
        d = s.scores_dict
        assert set(d.keys()) == {
            "naturalness",
            "tool_correctness",
            "task_completion",
            "coherence",
            "grounding_accuracy",
        }
        assert d["naturalness"] == 1.0
        assert d["tool_correctness"] == 2.0

    def test_passes_threshold_true(self):
        s = JudgeScores(
            naturalness=4.0,
            tool_correctness=4.0,
            task_completion=4.0,
            coherence=4.0,
            grounding_accuracy=4.0,
        )
        assert s.passes_threshold(4.0)
        assert s.passes_threshold(3.5)

    def test_passes_threshold_false(self):
        s = JudgeScores()
        assert not s.passes_threshold(1.0)

    def test_passes_threshold_boundary(self):
        s = JudgeScores(
            naturalness=3.5,
            tool_correctness=3.5,
            task_completion=3.5,
            coherence=3.5,
            grounding_accuracy=3.5,
        )
        assert s.passes_threshold(3.5)
        assert not s.passes_threshold(3.6)

    def test_validation_too_high(self):
        with pytest.raises(ValidationError):
            JudgeScores(naturalness=6.0)

    def test_validation_negative(self):
        with pytest.raises(ValidationError):
            JudgeScores(tool_correctness=-1.0)

    def test_serialization_round_trip(self):
        s = JudgeScores(naturalness=4.2, coherence=3.8)
        data = s.model_dump()
        restored = JudgeScores.model_validate(data)
        assert restored.naturalness == 4.2
        assert restored.coherence == 3.8


# ===================================================================
# ConversationMetadata
# ===================================================================


class TestConversationMetadata:
    def test_defaults(self):
        meta = ConversationMetadata()
        assert meta.seed == 0
        assert meta.tools_used == []
        assert meta.domains == []
        assert meta.num_turns == 0
        assert meta.num_tool_calls == 0
        assert meta.num_distinct_tools == 0
        assert meta.pattern == ""
        assert meta.generation_time_ms == 0
        assert meta.attempt_number == 1
        assert meta.config == {}

    def test_from_conversation_tools_used(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[
                    _make_tool_call_request(tool_id="a"),
                    _make_tool_call_request(tool_id="b"),
                ],
            ),
            Message(role="tool", tool_output={"ok": True}),
            Message(
                role="assistant",
                tool_calls=[_make_tool_call_request(tool_id="a")],
            ),
        ]
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation(msgs, chain)
        assert sorted(meta.tools_used) == ["a", "b"]

    def test_from_conversation_num_turns(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="thanks"),
        ]
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation(msgs, chain)
        assert meta.num_turns == 3

    def test_from_conversation_num_tool_calls(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[_make_tool_call_request()],
            ),
            Message(role="tool", tool_output={}),
            Message(
                role="assistant",
                tool_calls=[_make_tool_call_request(tool_id="b")],
            ),
            Message(role="tool", tool_output={}),
            Message(role="assistant", content="done"),
        ]
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation(msgs, chain)
        assert meta.num_tool_calls == 2

    def test_from_conversation_distinct_tools(self):
        msgs = [
            Message(
                role="assistant",
                tool_calls=[
                    _make_tool_call_request(tool_id="x"),
                    _make_tool_call_request(tool_id="y"),
                ],
            ),
            Message(
                role="assistant",
                tool_calls=[_make_tool_call_request(tool_id="x")],
            ),
        ]
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation(msgs, chain)
        assert meta.num_distinct_tools == 2

    def test_from_conversation_domains(self):
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation([], chain)
        assert meta.domains == ["Travel"]

    def test_from_conversation_pattern(self):
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation([], chain)
        assert meta.pattern == "sequential"

    def test_from_conversation_seed_and_time(self):
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation(
            [], chain, seed=99, generation_time_ms=1500, attempt_number=3
        )
        assert meta.seed == 99
        assert meta.generation_time_ms == 1500
        assert meta.attempt_number == 3

    def test_from_conversation_config(self):
        chain = _make_chain()
        meta = ConversationMetadata.from_conversation(
            [], chain, config={"temperature": 0.7}
        )
        assert meta.config == {"temperature": 0.7}

    def test_serialization_round_trip(self):
        meta = ConversationMetadata(
            seed=42, tools_used=["a"], num_turns=5, pattern="sequential"
        )
        data = meta.model_dump()
        restored = ConversationMetadata.model_validate(data)
        assert restored.seed == 42
        assert restored.tools_used == ["a"]


# ===================================================================
# Conversation
# ===================================================================


class TestConversation:
    def test_default_construction(self):
        conv = Conversation()
        assert conv.conversation_id  # non-empty UUID
        assert conv.messages == []
        assert conv.chain is None
        assert conv.judge_scores is None
        assert conv.turn_count == 0

    def test_auto_generated_id(self):
        c1 = Conversation()
        c2 = Conversation()
        assert c1.conversation_id != c2.conversation_id

    def test_add_user_message(self):
        conv = Conversation()
        msg = conv.add_user_message("hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert conv.turn_count == 1
        assert conv.messages[0] is msg

    def test_add_assistant_message_content(self):
        conv = Conversation()
        msg = conv.add_assistant_message(content="Sure!")
        assert msg.role == "assistant"
        assert msg.content == "Sure!"
        assert msg.tool_calls is None

    def test_add_assistant_message_tool_calls(self):
        conv = Conversation()
        tc = _make_tool_call_request()
        msg = conv.add_assistant_message(tool_calls=[tc])
        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_add_tool_message(self):
        conv = Conversation()
        msg = conv.add_tool_message("call-1", {"status": "ok"})
        assert msg.role == "tool"
        assert msg.tool_call_id == "call-1"
        assert msg.tool_output == {"status": "ok"}

    def test_turn_count(self):
        conv = Conversation()
        assert conv.turn_count == 0
        conv.add_user_message("a")
        assert conv.turn_count == 1
        conv.add_assistant_message(content="b")
        assert conv.turn_count == 2
        conv.add_tool_message("c", {})
        assert conv.turn_count == 3


class TestConversationJsonl:
    def test_to_jsonl_dict_structure(self):
        conv = Conversation()
        conv.add_user_message("hi")
        d = conv.to_jsonl_dict()
        assert "conversation_id" in d
        assert "messages" in d
        assert "judge_scores" in d
        assert "metadata" in d
        assert d["judge_scores"] is None

    def test_to_jsonl_dict_with_scores(self):
        conv = Conversation()
        conv.judge_scores = JudgeScores(naturalness=4.0, task_completion=5.0)
        d = conv.to_jsonl_dict()
        assert d["judge_scores"]["naturalness"] == 4.0
        assert d["judge_scores"]["task_completion"] == 5.0

    def test_to_jsonl_dict_without_scores(self):
        conv = Conversation()
        d = conv.to_jsonl_dict()
        assert d["judge_scores"] is None

    def test_to_jsonl_valid_json(self):
        conv = Conversation()
        conv.add_user_message("hello")
        json_str = conv.to_jsonl()
        parsed = json.loads(json_str)
        assert parsed["conversation_id"] == conv.conversation_id
        assert len(parsed["messages"]) == 1

    def test_to_jsonl_round_trip(self):
        conv = Conversation()
        conv.add_user_message("test")
        conv.add_assistant_message(content="reply")
        json_str = conv.to_jsonl()
        parsed = json.loads(json_str)
        assert parsed["messages"][0]["role"] == "user"
        assert parsed["messages"][1]["role"] == "assistant"
        assert parsed["messages"][1]["content"] == "reply"

    def test_metadata_in_jsonl(self):
        conv = Conversation()
        conv.metadata = ConversationMetadata(seed=42, num_turns=3)
        d = conv.to_jsonl_dict()
        assert d["metadata"]["seed"] == 42
        assert d["metadata"]["num_turns"] == 3


class TestConversationFullFlow:
    """End-to-end conversation flow matching the project spec."""

    def test_full_flow_jsonl(self):
        conv = Conversation()

        # User request
        conv.add_user_message("Find me a hotel in Paris for next weekend")
        # Disambiguation
        conv.add_assistant_message(content="What's your budget range?")
        conv.add_user_message("Under 200 euros per night")

        # Tool call: search
        search_req = ToolCallRequest(
            endpoint_id="hotels/search",
            tool_id="hotels",
            tool_name="Hotels API",
            endpoint_name="Search Hotels",
            method=HttpMethod.GET,
            path="/hotels/search",
            arguments={"city": "Paris", "max_price": 200, "currency": "EUR"},
        )
        conv.add_assistant_message(tool_calls=[search_req])
        conv.add_tool_message(
            tool_call_id=search_req.call_id,
            output={
                "results": [
                    {"id": "htl_881", "name": "Hotel du Marais", "price": 175}
                ]
            },
        )

        # Tool call: book
        book_req = ToolCallRequest(
            endpoint_id="hotels/book",
            tool_id="hotels",
            tool_name="Hotels API",
            endpoint_name="Book Hotel",
            method=HttpMethod.POST,
            path="/hotels/book",
            arguments={"hotel_id": "htl_881", "check_in": "2026-04-11"},
        )
        conv.add_assistant_message(tool_calls=[book_req])
        conv.add_tool_message(
            tool_call_id=book_req.call_id,
            output={"booking_id": "bk_3391", "status": "confirmed"},
        )

        # Final answer
        conv.add_assistant_message(
            content="I've booked Hotel du Marais for Apr 11. Confirmation: bk_3391."
        )

        assert conv.turn_count == 8

        # Set scores
        conv.judge_scores = JudgeScores(
            naturalness=4.2,
            tool_correctness=4.8,
            task_completion=5.0,
            coherence=4.5,
            grounding_accuracy=4.6,
        )

        # Set metadata
        chain = _make_chain()
        conv.chain = chain
        conv.metadata = ConversationMetadata.from_conversation(
            conv.messages, chain, seed=42
        )

        # Verify JSONL output
        record = conv.to_jsonl_dict()
        msgs = record["messages"]

        assert len(msgs) == 8
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Find me a hotel in Paris for next weekend"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "What's your budget range?"
        assert msgs[2]["role"] == "user"
        assert msgs[3]["role"] == "assistant"
        assert "tool_calls" in msgs[3]
        assert msgs[3]["tool_calls"][0]["endpoint"] == "hotels/search"
        assert msgs[3]["tool_calls"][0]["arguments"]["city"] == "Paris"
        assert msgs[4]["role"] == "tool"
        assert isinstance(msgs[4]["content"], dict)
        assert msgs[4]["content"]["results"][0]["id"] == "htl_881"
        assert msgs[5]["role"] == "assistant"
        assert msgs[5]["tool_calls"][0]["endpoint"] == "hotels/book"
        assert msgs[6]["role"] == "tool"
        assert msgs[6]["content"]["booking_id"] == "bk_3391"
        assert msgs[7]["role"] == "assistant"
        assert "bk_3391" in msgs[7]["content"]

        # Scores
        assert record["judge_scores"]["naturalness"] == 4.2
        assert record["judge_scores"]["task_completion"] == 5.0

        # Metadata
        assert record["metadata"]["num_turns"] == 8
        assert record["metadata"]["num_tool_calls"] == 2
        assert record["metadata"]["tools_used"] == ["hotels"]
        assert record["metadata"]["domains"] == ["Travel"]
        assert record["metadata"]["pattern"] == "sequential"

    def test_roles_in_order(self):
        conv = Conversation()
        conv.add_user_message("q")
        tc = _make_tool_call_request()
        conv.add_assistant_message(tool_calls=[tc])
        conv.add_tool_message(tc.call_id, {"data": 1})
        conv.add_assistant_message(content="done")

        roles = [m["role"] for m in conv.to_jsonl_dict()["messages"]]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_tool_calls_have_required_keys(self):
        tc = _make_tool_call_request()
        conv = Conversation()
        conv.add_assistant_message(tool_calls=[tc])
        d = conv.to_jsonl_dict()
        tc_dict = d["messages"][0]["tool_calls"][0]
        assert "endpoint" in tc_dict
        assert "arguments" in tc_dict
        assert "tool_name" in tc_dict
        assert "call_id" in tc_dict

    def test_tool_content_is_dict(self):
        conv = Conversation()
        conv.add_tool_message("c1", {"key": "value"})
        d = conv.to_jsonl_dict()
        assert isinstance(d["messages"][0]["content"], dict)

    def test_json_serializable(self):
        conv = Conversation()
        conv.add_user_message("hi")
        tc = _make_tool_call_request()
        conv.add_assistant_message(tool_calls=[tc])
        conv.add_tool_message(tc.call_id, {"result": [1, 2, 3]})
        conv.add_assistant_message(content="done")
        conv.judge_scores = JudgeScores(naturalness=3.0)

        json_str = conv.to_jsonl()
        parsed = json.loads(json_str)
        assert parsed["conversation_id"] == conv.conversation_id
        assert len(parsed["messages"]) == 4
        assert parsed["judge_scores"]["naturalness"] == 3.0


# ===================================================================
# GenerationConfig
# ===================================================================


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.max_turns == 15
        assert cfg.include_disambiguation is True
        assert cfg.disambiguation_probability == 0.3
        assert cfg.temperature == 0.7
        assert cfg.require_final_answer is True
        assert cfg.min_tool_calls == 1
        assert cfg.max_consecutive_tool_calls == 3

    def test_custom_values(self):
        cfg = GenerationConfig(
            max_turns=30,
            include_disambiguation=False,
            disambiguation_probability=0.5,
            temperature=1.0,
            require_final_answer=False,
            min_tool_calls=0,
            max_consecutive_tool_calls=5,
        )
        assert cfg.max_turns == 30
        assert cfg.include_disambiguation is False
        assert cfg.temperature == 1.0

    def test_max_turns_min(self):
        with pytest.raises(ValidationError):
            GenerationConfig(max_turns=0)

    def test_disambiguation_probability_bounds(self):
        GenerationConfig(disambiguation_probability=0.0)  # ok
        GenerationConfig(disambiguation_probability=1.0)  # ok
        with pytest.raises(ValidationError):
            GenerationConfig(disambiguation_probability=-0.1)
        with pytest.raises(ValidationError):
            GenerationConfig(disambiguation_probability=1.1)

    def test_temperature_bounds(self):
        GenerationConfig(temperature=0.0)  # ok
        GenerationConfig(temperature=2.0)  # ok
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=2.1)

    def test_serialization_round_trip(self):
        cfg = GenerationConfig(max_turns=10, temperature=1.5)
        data = cfg.model_dump()
        restored = GenerationConfig.model_validate(data)
        assert restored.max_turns == 10
        assert restored.temperature == 1.5
