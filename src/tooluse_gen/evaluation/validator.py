"""Structural conversation validator.

:class:`ConversationValidator` performs pre-judge structural checks on
generated conversations: message ordering, tool-call validity, grounding
consistency, minimum requirements, and completeness.
"""

from __future__ import annotations

from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.evaluation.models import ValidationResult
from tooluse_gen.registry.registry import ToolRegistry
from tooluse_gen.utils.logging import get_logger

logger = get_logger("evaluation.validator")


class ConversationValidator:
    """Pre-judge structural validation for generated conversations."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, conversation: Conversation) -> ValidationResult:
        """Run all structural checks and return a :class:`ValidationResult`."""
        errors: list[str] = []
        errors.extend(self._check_message_structure(conversation))
        errors.extend(self._check_tool_call_validity(conversation))
        errors.extend(self._check_grounding_consistency(conversation))
        errors.extend(self._check_minimum_requirements(conversation))
        errors.extend(self._check_conversation_completeness(conversation))
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    # ------------------------------------------------------------------
    # Message structure
    # ------------------------------------------------------------------

    def _check_message_structure(self, conversation: Conversation) -> list[str]:
        errors: list[str] = []
        msgs = conversation.messages

        if not msgs:
            errors.append("Conversation has no messages")
            return errors

        for i in range(len(msgs) - 1):
            if msgs[i].role == "user" and msgs[i + 1].role == "user":
                errors.append(
                    f"Consecutive user messages at positions {i} and {i + 1}"
                )

        for i, msg in enumerate(msgs):
            if msg.role == "tool":
                found = False
                for j in range(i - 1, -1, -1):
                    if msgs[j].role == "assistant" and msgs[j].tool_calls:
                        found = True
                        break
                    if msgs[j].role == "user":
                        break
                if not found:
                    errors.append(
                        f"Tool message at position {i} has no preceding "
                        "assistant tool call"
                    )

        for i, msg in enumerate(msgs):
            if msg.role == "assistant" and msg.tool_calls is not None and len(msg.tool_calls) == 0:
                errors.append(
                    f"Assistant message at position {i} has empty tool_calls list"
                )

        for i, msg in enumerate(msgs):
            if msg.role == "user" and not msg.content:
                errors.append(f"User message at position {i} has no content")

        return errors

    # ------------------------------------------------------------------
    # Tool call validity
    # ------------------------------------------------------------------

    def _check_tool_call_validity(self, conversation: Conversation) -> list[str]:
        if self._registry is None:
            return []

        errors: list[str] = []
        for msg in conversation.messages:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                ep = self._registry.get_endpoint(tc.endpoint_id)
                if ep is None:
                    errors.append(
                        f"Endpoint '{tc.endpoint_id}' not found in registry"
                    )
                    continue
                for param in ep.required_parameters:
                    if param not in tc.arguments:
                        errors.append(
                            f"Missing required parameter '{param}' "
                            f"for endpoint '{tc.endpoint_id}'"
                        )
        return errors

    # ------------------------------------------------------------------
    # Grounding consistency
    # ------------------------------------------------------------------

    def _check_grounding_consistency(self, conversation: Conversation) -> list[str]:
        errors: list[str] = []
        msgs = conversation.messages
        last_tool_call_idx: int | None = None

        for i, msg in enumerate(msgs):
            if msg.role == "assistant" and msg.tool_calls:
                if last_tool_call_idx is not None:
                    # Check there is at least one tool response between
                    # the previous tool call and this one.
                    has_tool_response = any(
                        msgs[k].role == "tool"
                        for k in range(last_tool_call_idx + 1, i)
                    )
                    if not has_tool_response:
                        errors.append(
                            f"Assistant tool call at position {i} has no "
                            "preceding tool response for previous call"
                        )
                last_tool_call_idx = i

        return errors

    # ------------------------------------------------------------------
    # Minimum requirements
    # ------------------------------------------------------------------

    def _check_minimum_requirements(self, conversation: Conversation) -> list[str]:
        errors: list[str] = []
        tool_call_count = sum(
            1
            for msg in conversation.messages
            if msg.role == "assistant" and msg.tool_calls
        )
        if tool_call_count == 0:
            errors.append("No tool calls in conversation")
        return errors

    # ------------------------------------------------------------------
    # Conversation completeness
    # ------------------------------------------------------------------

    def _check_conversation_completeness(
        self, conversation: Conversation
    ) -> list[str]:
        errors: list[str] = []
        msgs = conversation.messages

        if not msgs:
            return errors  # already caught in structure check

        if msgs[0].role != "user":
            errors.append("Conversation does not start with a user message")

        has_assistant = any(m.role == "assistant" for m in msgs)
        if not has_assistant:
            errors.append("Conversation has no assistant messages")

        if msgs[-1].role != "assistant":
            errors.append("Conversation does not end with an assistant message")

        return errors
