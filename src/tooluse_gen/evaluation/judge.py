"""LLM-as-judge scoring agent.

:class:`JudgeAgent` evaluates conversations on four quality dimensions
using either an OpenAI-compatible LLM or offline heuristics.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.evaluation.models import JudgeScores
from tooluse_gen.utils.logging import get_logger

logger = get_logger("evaluation.judge")

_RUBRIC = """\
You are evaluating a synthetic conversation for training data quality.
Score each dimension 1-5:

Tool Selection Correctness: Did the assistant pick appropriate tools for the user's request?
1=completely wrong tools, 3=partially correct, 5=perfect tool selection

Argument Grounding: Are tool call arguments valid and reference real values from prior outputs?
1=hallucinated arguments, 3=some grounded, 5=all arguments properly grounded

Task Completion: Did the conversation achieve the user's stated goal?
1=goal not addressed, 3=partially achieved, 5=fully completed

Naturalness: Does the conversation flow naturally?
1=robotic/incoherent, 3=acceptable, 5=indistinguishable from real conversation

Respond with ONLY a JSON object in this exact format:
{"tool_correctness": <int>, "argument_grounding": <int>, \
"task_completion": <int>, "naturalness": <int>, "reasoning": "<brief explanation>"}\
"""


def _clamp(value: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, value))


class JudgeAgent:
    """Scores conversations using an LLM-as-judge or offline heuristics."""

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._temperature = temperature
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, conversation: Conversation) -> JudgeScores:
        """Score a single conversation.

        If the LLM call fails, falls back to offline heuristic scoring
        so the pipeline can continue without crashing.
        """
        if self._client is not None:
            try:
                return self._score_with_llm(conversation)
            except Exception as exc:
                self._logger.warning(
                    "Judge LLM call failed for %s, falling back to offline: %s",
                    conversation.conversation_id,
                    exc,
                )
                return self._score_offline(conversation)
        return self._score_offline(conversation)

    def score_batch(
        self, conversations: list[Conversation]
    ) -> list[JudgeScores]:
        """Score a batch, logging progress every 10 items."""
        results: list[JudgeScores] = []
        for i, conv in enumerate(conversations):
            results.append(self.score(conv))
            if (i + 1) % 10 == 0 or i == len(conversations) - 1:
                self._logger.info(
                    "Scored %d/%d conversations", i + 1, len(conversations)
                )
        return results

    def aggregate_scores(self, scores: list[JudgeScores]) -> JudgeScores:
        """Compute per-dimension mean across *scores*, rounded to int."""
        if not scores:
            return JudgeScores()
        n = len(scores)
        return JudgeScores(
            tool_correctness=_clamp(
                round(sum(s.tool_correctness for s in scores) / n)
            ),
            argument_grounding=_clamp(
                round(sum(s.argument_grounding for s in scores) / n)
            ),
            task_completion=_clamp(
                round(sum(s.task_completion for s in scores) / n)
            ),
            naturalness=_clamp(
                round(sum(s.naturalness for s in scores) / n)
            ),
        )

    # ------------------------------------------------------------------
    # LLM scoring
    # ------------------------------------------------------------------

    def _score_with_llm(self, conversation: Conversation) -> JudgeScores:
        prompt = self._build_judge_prompt(conversation)
        text = self._call_llm(prompt)
        return self._parse_scores(text)

    def _build_judge_prompt(
        self, conversation: Conversation
    ) -> list[dict[str, str]]:
        msgs = conversation.to_jsonl_dict().get("messages", [])
        conv_json = json.dumps(msgs, default=str, indent=2)
        return [
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": conv_json},
        ]

    def _parse_scores(self, response_text: str) -> JudgeScores:
        """Extract scores from the judge's response text."""
        if not response_text.strip():
            return JudgeScores(reasoning="Failed to parse judge response")

        # Try JSON parsing first.
        try:
            data = json.loads(response_text)
            return self._scores_from_dict(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: regex extraction.
        data: dict[str, Any] = {}
        for key in (
            "tool_correctness",
            "argument_grounding",
            "task_completion",
            "naturalness",
        ):
            m = re.search(rf'"{key}"\s*:\s*(\d+)', response_text)
            if m:
                data[key] = int(m.group(1))
        m_reason = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response_text)
        if m_reason:
            data["reasoning"] = m_reason.group(1)

        if data:
            return self._scores_from_dict(data)

        return JudgeScores(reasoning="Failed to parse judge response")

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=500,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()

    # ------------------------------------------------------------------
    # Offline (heuristic) scoring
    # ------------------------------------------------------------------

    def _score_offline(self, conversation: Conversation) -> JudgeScores:
        msgs = conversation.messages
        reasons: list[str] = []

        # Counts used across dimensions.
        tc_msgs = [m for m in msgs if m.role == "assistant" and m.tool_calls]
        tool_call_count = len(tc_msgs)
        tool_ids: set[str] = set()
        for m in tc_msgs:
            if m.tool_calls:
                for tc in m.tool_calls:
                    tool_ids.add(tc.tool_id)
        user_msgs = [m for m in msgs if m.role == "user"]
        has_disambig = any(
            m.role == "assistant"
            and m.content
            and "?" in m.content
            and not m.tool_calls
            for m in msgs
        )

        # tool_correctness
        tc_score = 3
        if tool_call_count >= 2:
            tc_score += 1
        if len(tool_ids) >= 2:
            tc_score += 1
        tc_score = _clamp(tc_score)
        reasons.append(f"tool_correctness={tc_score}: {tool_call_count} calls, {len(tool_ids)} tools")

        # argument_grounding
        ag_score = 2
        # Check if any argument value appears in a prior tool output.
        tool_output_values: set[str] = set()
        for m in msgs:
            if m.role == "tool" and m.tool_output:
                for v in m.tool_output.values():
                    if isinstance(v, str) and v:
                        tool_output_values.add(v)
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    for v in tc.arguments.values():
                        if isinstance(v, str) and v in tool_output_values:
                            ag_score += 1
                            break
                    break  # only check once per message
        gs = getattr(conversation.metadata, "grounding_stats", {})
        if isinstance(gs, dict) and gs.get("grounded_args", 0) > 0:
            ag_score += 1
        ag_score = _clamp(ag_score)
        reasons.append(f"argument_grounding={ag_score}")

        # task_completion
        comp_score = 3
        if msgs and msgs[-1].role == "assistant" and not msgs[-1].tool_calls:
            comp_score += 1
        if len(msgs) >= 3:
            comp_score += 1
        comp_score = _clamp(comp_score)
        reasons.append(f"task_completion={comp_score}")

        # naturalness
        nat_score = 2
        if msgs and msgs[0].role == "user":
            nat_score += 1
        if len(user_msgs) > 1:
            nat_score += 1
        if has_disambig:
            nat_score += 1
        nat_score = _clamp(nat_score)
        reasons.append(f"naturalness={nat_score}")

        return JudgeScores(
            tool_correctness=tc_score,
            argument_grounding=ag_score,
            task_completion=comp_score,
            naturalness=nat_score,
            reasoning="; ".join(reasons),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scores_from_dict(data: dict[str, Any]) -> JudgeScores:
        return JudgeScores(
            tool_correctness=_clamp(int(data.get("tool_correctness", 1))),
            argument_grounding=_clamp(int(data.get("argument_grounding", 1))),
            task_completion=_clamp(int(data.get("task_completion", 1))),
            naturalness=_clamp(int(data.get("naturalness", 1))),
            reasoning=str(data.get("reasoning", "")),
        )
