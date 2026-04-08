"""Data models for tool execution.

:class:`ToolCallRequest` and :class:`ToolCallResponse` represent an
individual tool invocation.  :class:`ConversationContext` tracks the
full state of a multi-turn conversation being generated — messages,
tool outputs, grounding values, and the driving tool chain.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.graph.chain_models import ChainStep, ToolChain
from tooluse_gen.registry.models import HttpMethod

# ---------------------------------------------------------------------------
# ToolCallRequest
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    """A request to invoke a tool endpoint."""

    model_config = ConfigDict(use_enum_values=True)

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique call ID.")
    endpoint_id: str = Field(..., description="Endpoint identifier.")
    tool_id: str = Field(..., description="Parent tool identifier.")
    tool_name: str = Field(..., description="Human-readable tool name.")
    endpoint_name: str = Field(..., description="Human-readable endpoint name.")
    method: HttpMethod = Field(default=HttpMethod.GET, description="HTTP method.")
    path: str = Field(default="", description="URL path template.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Key-value arguments for the call."
    )

    @classmethod
    def from_chain_step(
        cls,
        step: ChainStep,
        arguments: dict[str, Any] | None = None,
    ) -> ToolCallRequest:
        """Build a request from a :class:`ChainStep`."""
        return cls(
            endpoint_id=step.endpoint_id,
            tool_id=step.tool_id,
            tool_name=step.tool_name,
            endpoint_name=step.endpoint_name,
            method=step.method,
            path=step.path,
            arguments=dict(arguments) if arguments else {},
        )


# ---------------------------------------------------------------------------
# ToolCallResponse
# ---------------------------------------------------------------------------


class ToolCallResponse(BaseModel):
    """The result of a tool call."""

    model_config = ConfigDict(use_enum_values=True)

    call_id: str = Field(..., description="Matches the request's call_id.")
    status_code: int = Field(default=200, description="HTTP-style status code.")
    data: dict[str, Any] = Field(default_factory=dict, description="Response payload.")
    generated_ids: dict[str, str] = Field(
        default_factory=dict, description="IDs created in this response."
    )
    extractable_values: dict[str, Any] = Field(
        default_factory=dict, description="Values usable by downstream steps."
    )
    error: str | None = Field(default=None, description="Error message if call failed.")

    @property
    def is_success(self) -> bool:
        """True when status is 2xx and no error message is set."""
        return 200 <= self.status_code < 300 and self.error is None


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------


class ConversationContext(BaseModel):
    """Tracks the full state of a conversation being generated."""

    model_config = ConfigDict(use_enum_values=True)

    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique conversation ID."
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Role-tagged messages."
    )
    generated_ids: dict[str, str] = Field(
        default_factory=dict, description="Accumulated entity IDs across all tool calls."
    )
    tool_outputs: list[ToolCallResponse] = Field(
        default_factory=list, description="All tool responses in order."
    )
    grounding_values: dict[str, Any] = Field(
        default_factory=dict,
        description="All extractable values with provenance keys.",
    )
    chain: ToolChain | None = Field(
        default=None, description="The sampled chain driving this conversation."
    )
    current_step: int = Field(default=0, description="Index of the current step.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 creation timestamp.",
    )

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a role-tagged message."""
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    # ------------------------------------------------------------------
    # Tool output tracking
    # ------------------------------------------------------------------

    def add_tool_output(self, response: ToolCallResponse) -> None:
        """Record a tool response and merge its values into context."""
        self.tool_outputs.append(response)

        # Merge generated IDs
        self.generated_ids.update(response.generated_ids)

        # Merge extractable values — both raw and step-prefixed
        for key, value in response.extractable_values.items():
            self.grounding_values[f"step_{self.current_step}.{key}"] = value
            self.grounding_values[key] = value

        # Add a tool message
        self.add_message(
            "tool",
            json.dumps(response.data, default=str),
            tool_calls=[{"call_id": response.call_id}],
        )

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    def get_available_values(self) -> dict[str, Any]:
        """All grounding values and generated IDs combined."""
        result: dict[str, Any] = {}
        result.update(self.grounding_values)
        result.update(self.generated_ids)
        return result

    def get_history_for_prompt(self) -> list[dict[str, Any]]:
        """Return a copy of the message history."""
        return list(self.messages)

    def get_last_tool_output(self) -> ToolCallResponse | None:
        """Return the most recent tool response, or ``None``."""
        return self.tool_outputs[-1] if self.tool_outputs else None

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def advance_step(self) -> None:
        """Move to the next chain step."""
        self.current_step += 1
