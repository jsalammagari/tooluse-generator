"""Conversation data models for the multi-agent generator.

:class:`Message` represents a single turn in a conversation (user,
assistant, or tool).  :class:`Conversation` aggregates messages with
metadata and judge scores, and can serialize to the JSONL output format
required by the project spec.

:class:`JudgeScores` captures LLM-as-judge quality ratings on five
dimensions.  :class:`ConversationMetadata` summarises a conversation
for analysis.  :class:`GenerationConfig` controls generation behaviour.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from tooluse_gen.agents.execution_models import ToolCallRequest
from tooluse_gen.graph.chain_models import ToolChain

# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(use_enum_values=True)

    role: Literal["user", "assistant", "tool"] = Field(
        ..., description="Who sent the message."
    )
    content: str | None = Field(default=None, description="Text content.")
    tool_calls: list[ToolCallRequest] | None = Field(
        default=None, description="Tool invocations (assistant only)."
    )
    tool_call_id: str | None = Field(
        default=None, description="Links a tool response to a specific call."
    )
    tool_output: dict[str, Any] | None = Field(
        default=None, description="Structured response payload (tool only)."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary per-message metadata."
    )

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Serialize to the JSONL output format.

        * Tool messages use ``tool_output`` as ``content``.
        * Assistant tool-call messages include a ``tool_calls`` list with
          ``endpoint``, ``arguments``, ``tool_name``, and ``call_id``.
        * ``None`` values are omitted to keep output compact.
        """
        result: dict[str, Any] = {"role": self.role}

        # Content — for tool messages, use tool_output as content.
        if self.tool_output is not None:
            result["content"] = self.tool_output
        else:
            result["content"] = self.content

        # Tool calls on assistant messages.
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "endpoint": tc.endpoint_id,
                    "arguments": tc.arguments,
                    "tool_name": tc.tool_name,
                    "call_id": tc.call_id,
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id

        return result


# ---------------------------------------------------------------------------
# JudgeScores
# ---------------------------------------------------------------------------


class JudgeScores(BaseModel):
    """LLM-as-judge quality scores (each 0–5)."""

    model_config = ConfigDict(use_enum_values=True)

    naturalness: float = Field(default=0.0, ge=0.0, le=5.0, description="How natural.")
    tool_correctness: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Correct tools & arguments."
    )
    task_completion: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Task actually completed."
    )
    coherence: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Logical flow & consistency."
    )
    grounding_accuracy: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Tool outputs correctly referenced."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_score(self) -> float:
        """Average of all five dimension scores."""
        return (
            self.naturalness
            + self.tool_correctness
            + self.task_completion
            + self.coherence
            + self.grounding_accuracy
        ) / 5.0

    @property
    def scores_dict(self) -> dict[str, float]:
        """All scores as a plain dict for serialization."""
        return {
            "naturalness": self.naturalness,
            "tool_correctness": self.tool_correctness,
            "task_completion": self.task_completion,
            "coherence": self.coherence,
            "grounding_accuracy": self.grounding_accuracy,
        }

    def passes_threshold(self, threshold: float) -> bool:
        """Return ``True`` when :pyattr:`mean_score` ≥ *threshold*."""
        return self.mean_score >= threshold


# ---------------------------------------------------------------------------
# ConversationMetadata
# ---------------------------------------------------------------------------


class ConversationMetadata(BaseModel):
    """Summary metadata for a generated conversation."""

    model_config = ConfigDict(use_enum_values=True)

    seed: int = Field(default=0, description="Random seed used.")
    tools_used: list[str] = Field(default_factory=list, description="Tool IDs used.")
    domains: list[str] = Field(default_factory=list, description="Domains involved.")
    num_turns: int = Field(default=0, description="Total message count.")
    num_tool_calls: int = Field(default=0, description="Number of tool invocations.")
    num_distinct_tools: int = Field(default=0, description="Unique tools used.")
    pattern: str = Field(default="", description="ChainPattern value.")
    generation_time_ms: int = Field(default=0, description="Generation time (ms).")
    attempt_number: int = Field(default=1, description="Retry attempt that produced this.")
    config: dict[str, Any] = Field(default_factory=dict, description="Config snapshot.")
    endpoints_called: list[str] = Field(
        default_factory=list, description="Endpoint IDs called in order."
    )
    disambiguation_count: int = Field(
        default=0, description="Number of disambiguation exchanges."
    )
    timed_out: bool = Field(default=False, description="Whether this conversation hit the timeout.")
    grounding_stats: dict[str, int] = Field(
        default_factory=dict,
        description="Grounding statistics: grounded_args, fresh_args, total_args.",
    )

    @classmethod
    def from_conversation(
        cls,
        messages: list[Message],
        chain: ToolChain,
        seed: int = 0,
        generation_time_ms: int = 0,
        attempt_number: int = 1,
        config: dict[str, Any] | None = None,
        endpoints_called: list[str] | None = None,
        disambiguation_count: int = 0,
        grounding_stats: dict[str, int] | None = None,
    ) -> ConversationMetadata:
        """Compute metadata from *messages* and *chain*."""
        tool_ids: set[str] = set()
        tool_call_count = 0
        ep_ids: list[str] = []
        for msg in messages:
            if msg.tool_calls:
                tool_call_count += 1
                for tc in msg.tool_calls:
                    tool_ids.add(tc.tool_id)
                    ep_ids.append(tc.endpoint_id)

        sorted_tools = sorted(tool_ids)
        return cls(
            seed=seed,
            tools_used=sorted_tools,
            domains=list(chain.domains_involved),
            num_turns=len(messages),
            num_tool_calls=tool_call_count,
            num_distinct_tools=len(sorted_tools),
            pattern=chain.pattern if isinstance(chain.pattern, str) else chain.pattern,
            generation_time_ms=generation_time_ms,
            attempt_number=attempt_number,
            config=config or {},
            endpoints_called=endpoints_called if endpoints_called is not None else ep_ids,
            disambiguation_count=disambiguation_count,
            grounding_stats=grounding_stats or {},
        )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class Conversation(BaseModel):
    """A complete generated conversation."""

    model_config = ConfigDict(use_enum_values=True)

    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique conversation ID.",
    )
    messages: list[Message] = Field(default_factory=list, description="Ordered messages.")
    chain: ToolChain | None = Field(
        default=None, description="Sampled chain driving this conversation."
    )
    judge_scores: JudgeScores | None = Field(default=None, description="Quality scores.")
    metadata: ConversationMetadata = Field(
        default_factory=ConversationMetadata, description="Conversation metadata."
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 creation timestamp.",
    )

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> Message:
        """Append a user message and return it."""
        msg = Message(role="user", content=content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str | None = None,
        tool_calls: list[ToolCallRequest] | None = None,
    ) -> Message:
        """Append an assistant message and return it."""
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        self.messages.append(msg)
        return msg

    def add_tool_message(
        self,
        tool_call_id: str,
        output: dict[str, Any],
    ) -> Message:
        """Append a tool response message and return it."""
        msg = Message(role="tool", tool_call_id=tool_call_id, tool_output=output)
        self.messages.append(msg)
        return msg

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.messages)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Produce the JSONL output record."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [msg.to_jsonl_dict() for msg in self.messages],
            "judge_scores": self.judge_scores.scores_dict if self.judge_scores else None,
            "metadata": self.metadata.model_dump(),
        }

    def to_jsonl(self) -> str:
        """Serialize to a single JSON line."""
        return json.dumps(self.to_jsonl_dict(), default=str)


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------


class GenerationConfig(BaseModel):
    """Controls how conversations are generated."""

    model_config = ConfigDict(use_enum_values=True)

    max_turns: int = Field(default=15, ge=1, description="Max messages before forced stop.")
    include_disambiguation: bool = Field(
        default=True, description="Include assistant disambiguation questions."
    )
    disambiguation_probability: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Probability of a disambiguation turn."
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature."
    )
    require_final_answer: bool = Field(
        default=True, description="Conversation must end with assistant text."
    )
    min_tool_calls: int = Field(
        default=1, ge=0, description="Minimum tool calls per conversation."
    )
    max_consecutive_tool_calls: int = Field(
        default=3, ge=1, description="Max tool calls before requiring text."
    )
