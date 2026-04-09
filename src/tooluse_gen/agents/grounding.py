"""Grounding value formatting and provenance tracking.

Provides utilities for presenting grounding values (IDs, names, etc.
extracted from prior tool calls) in human-readable prompt fragments
and structured dicts for OpenAI function calling.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.agents.execution_models import ConversationContext, ToolCallResponse

# ---------------------------------------------------------------------------
# Provenance model
# ---------------------------------------------------------------------------


class ValueProvenance(BaseModel):
    """Tracks where a grounding value came from."""

    model_config = ConfigDict(use_enum_values=True)

    value_key: str = Field(..., description="Key name, e.g. 'hotel_id'.")
    value: Any = Field(..., description="The actual value.")
    source_endpoint: str = Field(..., description="Endpoint that produced this value.")
    step_index: int = Field(..., description="Conversation step index.")
    value_type: str = Field(default="", description="Optional type hint: 'id', 'name', etc.")


# ---------------------------------------------------------------------------
# Grounding tracker
# ---------------------------------------------------------------------------


class GroundingTracker:
    """Tracks provenance of grounding values across conversation steps."""

    def __init__(self) -> None:
        self._provenance: dict[str, ValueProvenance] = {}

    def track_value(
        self,
        value_key: str,
        value: Any,
        source_endpoint: str,
        step_index: int,
        value_type: str = "",
    ) -> None:
        """Record provenance for a single value."""
        prov = ValueProvenance(
            value_key=value_key,
            value=value,
            source_endpoint=source_endpoint,
            step_index=step_index,
            value_type=value_type,
        )
        self._provenance[value_key] = prov
        self._provenance[f"step_{step_index}.{value_key}"] = prov

    def track_from_response(
        self,
        response: ToolCallResponse,
        source_endpoint: str,
        step_index: int,
    ) -> None:
        """Record provenance for all extractable values and generated IDs."""
        for key, value in response.extractable_values.items():
            self.track_value(key, value, source_endpoint, step_index)
        for key, value in response.generated_ids.items():
            self.track_value(key, value, source_endpoint, step_index, value_type="id")

    def get_provenance(self, value_key: str) -> ValueProvenance | None:
        """Return provenance for *value_key*, or ``None``."""
        return self._provenance.get(value_key)

    def get_all_provenance(self) -> dict[str, ValueProvenance]:
        """Return a copy of all tracked provenance."""
        return dict(self._provenance)

    def reset(self) -> None:
        """Clear all provenance records."""
        self._provenance.clear()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_MAX_VALUE_LEN = 50


def format_value_for_prompt(value: Any) -> str:
    """Convert *value* to a prompt-friendly string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        items = [str(v) for v in value[:3]]
        tail = ", ..." if len(value) > 3 else ""
        return "[" + ", ".join(items) + tail + "]"
    if isinstance(value, dict):
        keys = ", ".join(str(k) for k in value)
        return "{" + keys + "}"
    s = str(value)
    if len(s) > _MAX_VALUE_LEN:
        return s[:_MAX_VALUE_LEN] + "..."
    return s


def format_available_values(
    context: ConversationContext,
    tracker: GroundingTracker | None = None,
) -> str:
    """Human-readable prompt fragment listing available grounding values."""
    available = context.get_available_values()
    if not available:
        return "No values available from prior tool calls."

    # Filter out step-prefixed keys
    filtered = {k: v for k, v in available.items() if "." not in k}
    if not filtered:
        return "No values available from prior tool calls."

    lines = ["Available values from prior tool calls:"]
    for key in sorted(filtered):
        val_str = format_value_for_prompt(filtered[key])
        if tracker is not None:
            prov = tracker.get_provenance(key)
            if prov is not None:
                lines.append(
                    f"- {key}: {val_str} (from {prov.source_endpoint}, step {prov.step_index})"
                )
                continue
        lines.append(f"- {key}: {val_str}")

    return "\n".join(lines)


def format_grounding_context(context: ConversationContext) -> dict[str, Any]:
    """Structured dict for OpenAI function calling context injection."""
    available = context.get_available_values()
    filtered = {k: v for k, v in available.items() if "." not in k}

    return {
        "available_values": filtered,
        "generated_ids": dict(context.generated_ids),
        "current_step": context.current_step,
        "prior_tool_calls": len(context.tool_outputs),
    }
