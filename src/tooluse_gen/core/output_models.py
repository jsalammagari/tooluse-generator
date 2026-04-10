"""Output models and validation for JSONL conversation records.

:class:`ConversationRecord` is the final serialization format matching
the project spec.  :func:`from_conversation` converts an internal
:class:`Conversation` into a record, and :func:`validate_record` checks
a raw dict against the expected schema.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.evaluation.models import JudgeScores as EvalJudgeScores

# ---------------------------------------------------------------------------
# ConversationRecord
# ---------------------------------------------------------------------------


class ConversationRecord(BaseModel):
    """Final JSONL output schema for a generated conversation."""

    model_config = ConfigDict(use_enum_values=True)

    conversation_id: str = Field(..., description="Unique conversation ID.")
    messages: list[dict[str, Any]] = Field(..., description="Serialized messages.")
    judge_scores: dict[str, int] | None = Field(
        default=None, description="Judge dimension scores (no reasoning)."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Serialized metadata."
    )

    def to_jsonl(self) -> str:
        """Serialize to a single JSON line."""
        return json.dumps(self.model_dump(), default=str)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


def from_conversation(
    conversation: Conversation,
    eval_scores: EvalJudgeScores | None = None,
) -> ConversationRecord:
    """Convert an internal :class:`Conversation` to a :class:`ConversationRecord`."""
    base = conversation.to_jsonl_dict()

    scores: dict[str, int] | None = None
    if eval_scores is not None:
        scores = eval_scores.scores_dict
    elif base.get("judge_scores") is not None:
        scores = base["judge_scores"]

    return ConversationRecord(
        conversation_id=base["conversation_id"],
        messages=base["messages"],
        judge_scores=scores,
        metadata=base.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_ROLES = {"user", "assistant", "tool"}


def validate_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a raw dict against the expected JSONL schema.

    Returns ``(True, [])`` when valid, ``(False, errors)`` otherwise.
    """
    errors: list[str] = []

    # conversation_id
    cid = record.get("conversation_id")
    if not isinstance(cid, str) or not cid:
        errors.append("conversation_id must be a non-empty string")

    # messages
    msgs = record.get("messages")
    if not isinstance(msgs, list):
        errors.append("messages must be a list")
    elif len(msgs) == 0:
        errors.append("messages must not be empty")
    else:
        for i, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                errors.append(f"messages[{i}] must be a dict")
                continue
            if "role" not in msg:
                errors.append(f"messages[{i}] missing 'role'")
            elif msg["role"] not in _VALID_ROLES:
                errors.append(
                    f"messages[{i}] invalid role '{msg['role']}'"
                )
            if "content" not in msg:
                errors.append(f"messages[{i}] missing 'content'")

            tc = msg.get("tool_calls")
            if tc is not None:
                if not isinstance(tc, list):
                    errors.append(f"messages[{i}] tool_calls must be a list")
                else:
                    for j, call in enumerate(tc):
                        if not isinstance(call, dict):
                            errors.append(
                                f"messages[{i}].tool_calls[{j}] must be a dict"
                            )
                            continue
                        if "endpoint" not in call:
                            errors.append(
                                f"messages[{i}].tool_calls[{j}] missing 'endpoint'"
                            )
                        if "arguments" not in call:
                            errors.append(
                                f"messages[{i}].tool_calls[{j}] missing 'arguments'"
                            )

    # judge_scores
    js = record.get("judge_scores")
    if js is not None and not isinstance(js, dict):
        errors.append("judge_scores must be a dict or null")

    # metadata
    meta = record.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        errors.append("metadata must be a dict")

    return (len(errors) == 0, errors)


def validate_conversation_record(
    record: ConversationRecord,
) -> tuple[bool, list[str]]:
    """Validate a :class:`ConversationRecord` instance."""
    return validate_record(record.to_dict())
