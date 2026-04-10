"""Tests for the JudgeAgent (Task 45)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tooluse_gen.agents.conversation_models import Conversation, Message
from tooluse_gen.agents.execution_models import ToolCallRequest
from tooluse_gen.evaluation.judge import JudgeAgent
from tooluse_gen.evaluation.models import JudgeScores

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(
    response_text: str = (
        '{"tool_correctness": 4, "argument_grounding": 3,'
        ' "task_completion": 5, "naturalness": 4, "reasoning": "Good conversation"}'
    ),
) -> MagicMock:
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    mock.chat.completions.create.return_value = mock_response
    return mock


def _make_conversation() -> Conversation:
    tc = ToolCallRequest(
        endpoint_id="h/s",
        tool_id="h",
        tool_name="H",
        endpoint_name="S",
        arguments={"city": "Paris"},
    )
    return Conversation(
        messages=[
            Message(role="user", content="Find me a hotel"),
            Message(role="assistant", tool_calls=[tc]),
            Message(
                role="tool",
                tool_call_id=tc.call_id,
                tool_output={"id": "htl_1", "name": "Grand Hotel"},
            ),
            Message(role="assistant", content="Found Grand Hotel for you."),
        ]
    )


def _make_minimal_conversation() -> Conversation:
    tc = ToolCallRequest(
        endpoint_id="x", tool_id="x", tool_name="X", endpoint_name="X"
    )
    return Conversation(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", tool_calls=[tc]),
            Message(role="tool", tool_call_id=tc.call_id, tool_output={"ok": True}),
            Message(role="assistant", content="done"),
        ]
    )


def _make_rich_conversation() -> Conversation:
    tc1 = ToolCallRequest(
        endpoint_id="a/s", tool_id="a", tool_name="A", endpoint_name="Search",
        arguments={"city": "Rome"},
    )
    tc2 = ToolCallRequest(
        endpoint_id="b/b", tool_id="b", tool_name="B", endpoint_name="Book",
        arguments={"id": "htl_1"},
    )
    return Conversation(
        messages=[
            Message(role="user", content="Plan a trip to Rome"),
            Message(role="assistant", content="What is your budget?"),
            Message(role="user", content="Under 200"),
            Message(role="assistant", tool_calls=[tc1]),
            Message(
                role="tool",
                tool_call_id=tc1.call_id,
                tool_output={"id": "htl_1", "name": "Roma Inn"},
            ),
            Message(role="user", content="Book it"),
            Message(role="assistant", tool_calls=[tc2]),
            Message(
                role="tool",
                tool_call_id=tc2.call_id,
                tool_output={"booking": "BK-1"},
            ),
            Message(role="assistant", content="Booked Roma Inn. Ref: BK-1."),
        ]
    )


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_offline(self):
        j = JudgeAgent()
        assert j._client is None

    def test_with_client(self):
        mock = _make_mock_client()
        j = JudgeAgent(llm_client=mock)
        assert j._client is mock

    def test_custom_model_temp(self):
        j = JudgeAgent(model="gpt-4o-mini", temperature=0.1)
        assert j._model == "gpt-4o-mini"
        assert j._temperature == 0.1


# ===================================================================
# score — offline mode
# ===================================================================


class TestScoreOffline:
    def test_returns_judge_scores(self):
        s = JudgeAgent().score(_make_conversation())
        assert isinstance(s, JudgeScores)

    def test_scores_in_range(self):
        s = JudgeAgent().score(_make_conversation())
        for v in s.scores_dict.values():
            assert 1 <= v <= 5

    def test_has_reasoning(self):
        s = JudgeAgent().score(_make_conversation())
        assert isinstance(s.reasoning, str)
        assert len(s.reasoning) > 0

    def test_better_scores_for_rich_conversation(self):
        simple = JudgeAgent().score(_make_minimal_conversation())
        rich = JudgeAgent().score(_make_rich_conversation())
        assert rich.average >= simple.average

    def test_minimal_gets_lower(self):
        s = JudgeAgent().score(_make_minimal_conversation())
        # Minimal: 1 tool call, 1 tool, no disambig, no follow-up
        assert s.tool_correctness <= 4
        assert s.naturalness <= 3

    def test_deterministic(self):
        conv = _make_conversation()
        s1 = JudgeAgent().score(conv)
        s2 = JudgeAgent().score(conv)
        assert s1.scores_dict == s2.scores_dict


# ===================================================================
# score — LLM mode
# ===================================================================


class TestScoreLLM:
    def test_calls_llm(self):
        mock = _make_mock_client()
        JudgeAgent(llm_client=mock).score(_make_conversation())
        mock.chat.completions.create.assert_called_once()

    def test_returns_parsed_scores(self):
        mock = _make_mock_client()
        s = JudgeAgent(llm_client=mock).score(_make_conversation())
        assert s.tool_correctness == 4
        assert s.argument_grounding == 3
        assert s.task_completion == 5
        assert s.naturalness == 4
        assert s.reasoning == "Good conversation"

    def test_handles_none_content(self):
        mock = _make_mock_client()
        mock.chat.completions.create.return_value.choices[
            0
        ].message.content = None
        s = JudgeAgent(llm_client=mock).score(_make_conversation())
        assert s.reasoning == "Failed to parse judge response"

    def test_correct_parameters(self):
        mock = _make_mock_client()
        j = JudgeAgent(llm_client=mock, model="test-model", temperature=0.1)
        j.score(_make_conversation())
        call_kwargs = mock.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.1


# ===================================================================
# _build_judge_prompt
# ===================================================================


class TestBuildJudgePrompt:
    def test_returns_list(self):
        p = JudgeAgent()._build_judge_prompt(_make_conversation())
        assert isinstance(p, list)
        assert len(p) == 2

    def test_system_has_rubric(self):
        p = JudgeAgent()._build_judge_prompt(_make_conversation())
        sys = p[0]["content"]
        assert "Tool Selection Correctness" in sys
        assert "Argument Grounding" in sys
        assert "Task Completion" in sys
        assert "Naturalness" in sys

    def test_user_has_conversation(self):
        p = JudgeAgent()._build_judge_prompt(_make_conversation())
        user = p[1]["content"]
        assert "user" in user
        assert "assistant" in user

    def test_system_instructs_json(self):
        p = JudgeAgent()._build_judge_prompt(_make_conversation())
        assert "JSON" in p[0]["content"]


# ===================================================================
# _parse_scores
# ===================================================================


class TestParseScores:
    def test_valid_json(self):
        s = JudgeAgent()._parse_scores(
            '{"tool_correctness": 4, "argument_grounding": 3,'
            ' "task_completion": 5, "naturalness": 4, "reasoning": "ok"}'
        )
        assert s.tool_correctness == 4
        assert s.task_completion == 5

    def test_extracts_all_dimensions(self):
        s = JudgeAgent()._parse_scores(
            '{"tool_correctness": 2, "argument_grounding": 3,'
            ' "task_completion": 4, "naturalness": 5}'
        )
        assert s.tool_correctness == 2
        assert s.argument_grounding == 3
        assert s.task_completion == 4
        assert s.naturalness == 5

    def test_extracts_reasoning(self):
        s = JudgeAgent()._parse_scores(
            '{"tool_correctness": 3, "argument_grounding": 3,'
            ' "task_completion": 3, "naturalness": 3, "reasoning": "decent"}'
        )
        assert s.reasoning == "decent"

    def test_clamps_low(self):
        s = JudgeAgent()._parse_scores(
            '{"tool_correctness": 0, "argument_grounding": -1,'
            ' "task_completion": 3, "naturalness": 3}'
        )
        assert s.tool_correctness == 1
        assert s.argument_grounding == 1

    def test_clamps_high(self):
        s = JudgeAgent()._parse_scores(
            '{"tool_correctness": 6, "argument_grounding": 10,'
            ' "task_completion": 3, "naturalness": 3}'
        )
        assert s.tool_correctness == 5
        assert s.argument_grounding == 5

    def test_malformed_json(self):
        s = JudgeAgent()._parse_scores("not json at all")
        assert s.reasoning == "Failed to parse judge response"

    def test_empty_string(self):
        s = JudgeAgent()._parse_scores("")
        assert s.reasoning == "Failed to parse judge response"

    def test_regex_fallback(self):
        text = 'Some text "tool_correctness": 4 and "naturalness": 3 end'
        s = JudgeAgent()._parse_scores(text)
        assert s.tool_correctness == 4
        assert s.naturalness == 3


# ===================================================================
# score_batch
# ===================================================================


class TestScoreBatch:
    def test_returns_list(self):
        convs = [_make_conversation(), _make_conversation()]
        result = JudgeAgent().score_batch(convs)
        assert isinstance(result, list)

    def test_length_matches(self):
        convs = [_make_conversation()] * 3
        assert len(JudgeAgent().score_batch(convs)) == 3

    def test_each_is_judge_scores(self):
        convs = [_make_conversation()]
        result = JudgeAgent().score_batch(convs)
        assert isinstance(result[0], JudgeScores)


# ===================================================================
# aggregate_scores
# ===================================================================


class TestAggregateScores:
    def test_correct_mean(self):
        s1 = JudgeScores(
            tool_correctness=4, argument_grounding=4,
            task_completion=4, naturalness=4,
        )
        s2 = JudgeScores(
            tool_correctness=2, argument_grounding=2,
            task_completion=2, naturalness=2,
        )
        agg = JudgeAgent().aggregate_scores([s1, s2])
        assert agg.tool_correctness == 3

    def test_returns_judge_scores(self):
        s = JudgeScores(tool_correctness=5)
        agg = JudgeAgent().aggregate_scores([s])
        assert isinstance(agg, JudgeScores)

    def test_single_score(self):
        s = JudgeScores(
            tool_correctness=4, argument_grounding=3,
            task_completion=5, naturalness=2,
        )
        agg = JudgeAgent().aggregate_scores([s])
        assert agg.tool_correctness == 4

    def test_empty_list(self):
        agg = JudgeAgent().aggregate_scores([])
        assert agg.tool_correctness == 1  # defaults


# ===================================================================
# Integration
# ===================================================================


class TestIntegration:
    def test_score_well_formed(self):
        s = JudgeAgent().score(_make_conversation())
        assert s.average >= 2.0

    def test_batch(self):
        convs = [_make_conversation(), _make_rich_conversation()]
        scores = JudgeAgent().score_batch(convs)
        assert len(scores) == 2
        assert all(s.average >= 1.0 for s in scores)

    def test_reasonable_for_rich(self):
        s = JudgeAgent().score(_make_rich_conversation())
        # Rich conv has 2 tool calls, 2 tools, disambig, user follow-ups
        assert s.tool_correctness >= 3
        assert s.naturalness >= 3
