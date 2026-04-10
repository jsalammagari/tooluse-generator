"""Conversation orchestrator.

:class:`ConversationOrchestrator` is the centralized controller that
coordinates :class:`UserSimulator`, :class:`AssistantAgent`, and
:class:`ToolExecutor` to produce complete multi-turn conversations
from a sampled :class:`ToolChain`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.conversation_models import (
    Conversation,
    ConversationMetadata,
    GenerationConfig,
)
from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.agents.grounding import GroundingTracker
from tooluse_gen.agents.state_machine import (
    ConversationEvent,
    ConversationState,
    ConversationStateMachine,
)
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.graph.chain_models import ChainStep, ParallelGroup, ToolChain
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.orchestrator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_steps(chain: ToolChain) -> list[ChainStep]:
    """Flatten chain steps, expanding :class:`ParallelGroup` instances."""
    result: list[ChainStep] = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            result.extend(item.steps)
        else:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------


class OrchestratorConfig(BaseModel):
    """Configuration for the conversation orchestrator."""

    model_config = ConfigDict(use_enum_values=True)

    max_turns: int = Field(default=15, ge=1, description="Max messages before forced stop.")
    max_consecutive_tool_calls: int = Field(
        default=5, ge=1, description="Max tool calls in a row."
    )
    require_disambiguation: bool = Field(
        default=True, description="Whether to include disambiguation."
    )
    disambiguation_probability: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Probability of disambiguation."
    )
    require_final_answer: bool = Field(
        default=True, description="Must end with assistant text."
    )
    min_tool_calls: int = Field(default=2, ge=0, description="Minimum tool calls.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature.")


# ---------------------------------------------------------------------------
# ConversationOrchestrator
# ---------------------------------------------------------------------------


class ConversationOrchestrator:
    """Coordinates agents to produce complete multi-turn conversations."""

    def __init__(
        self,
        user_sim: UserSimulator,
        assistant: AssistantAgent,
        executor: ToolExecutor,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._user_sim = user_sim
        self._assistant = assistant
        self._executor = executor
        self._config = config or OrchestratorConfig()
        self._tracker = GroundingTracker()
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_conversation(
        self,
        chain: ToolChain,
        seed: int = 42,
        attempt_number: int = 1,
    ) -> Conversation:
        """Generate a complete conversation driven by *chain*."""
        rng = np.random.default_rng(seed)
        context = ConversationContext(chain=chain)
        conv = Conversation(chain=chain)
        gen_config = self._make_gen_config()
        self._tracker = GroundingTracker()

        t0 = time.monotonic()
        loop_stats = self._run_loop(conv, context, chain, rng, gen_config)

        # Force final answer if required and missing.
        needs_final = (
            not conv.messages
            or conv.messages[-1].role != "assistant"
            or conv.messages[-1].tool_calls
        )
        if self._config.require_final_answer and needs_final:
            self._force_final_answer(conv, context, chain, rng, gen_config)

        generation_time_ms = int((time.monotonic() - t0) * 1000)

        conv.metadata = ConversationMetadata.from_conversation(
            conv.messages,
            chain,
            seed=seed,
            generation_time_ms=generation_time_ms,
            attempt_number=attempt_number,
            config=self._config.model_dump(),
            endpoints_called=loop_stats["endpoints_called"],
            disambiguation_count=loop_stats["disambiguation_count"],
            grounding_stats=loop_stats["grounding_stats"],
        )
        return conv

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        conv: Conversation,
        context: ConversationContext,
        chain: ToolChain,
        rng: np.random.Generator,
        gen_config: GenerationConfig,
    ) -> dict[str, Any]:
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)

        # Metadata tracking.
        endpoints_called: list[str] = []
        disambiguation_count = 0
        grounded_args = 0
        fresh_args = 0
        total_args = 0

        # Step 1: initial user request.
        msg = self._user_sim.generate_initial_request(chain, context, rng)
        conv.add_user_message(msg)
        context.add_message("user", msg)
        sm.transition(ConversationEvent.USER_MESSAGE)

        flat = _flatten_steps(chain)

        # Step 2: main loop.
        safety = gen_config.max_turns + 10  # hard guard against infinite loops
        iterations = 0
        while not sm.is_terminal:
            iterations += 1
            if iterations > safety:
                self._logger.warning("Safety limit reached, breaking loop.")
                sm.transition(ConversationEvent.ERROR)
                break

            # Check completion before assistant turn.
            if conv.turn_count >= gen_config.max_turns:
                sm.transition(ConversationEvent.MAX_TURNS_REACHED)
                break
            if context.current_step >= len(flat):
                sm.transition(ConversationEvent.CHAIN_COMPLETE)
                break

            # Ensure we're in ASSISTANT_TURN before generating a response.
            if sm.state == ConversationState.USER_TURN:
                sm.transition(ConversationEvent.USER_MESSAGE)

            resp = self._assistant.generate_response(context, rng, gen_config)

            # Disambiguation.
            if resp.is_disambiguation:
                sm.transition(ConversationEvent.ASSISTANT_DISAMBIGUATE)
                disambiguation_count += 1
                conv.add_assistant_message(content=resp.content)
                context.add_message("assistant", resp.content or "")
                clarif = self._user_sim.generate_clarification_response(
                    context, resp.content or "", rng
                )
                conv.add_user_message(clarif)
                context.add_message("user", clarif)
                sm.transition(ConversationEvent.USER_CLARIFICATION)
                continue

            # Tool calls.
            if resp.tool_calls:
                sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
                conv.add_assistant_message(tool_calls=resp.tool_calls)
                context.add_message("assistant", "", tool_calls=[
                    {"endpoint": tc.endpoint_id, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ])

                for call in resp.tool_calls:
                    endpoints_called.append(call.endpoint_id)
                    # Count grounding before execution mutates context.
                    available = context.get_available_values()
                    for key in call.arguments:
                        if key in available:
                            grounded_args += 1
                        else:
                            fresh_args += 1
                    total_args += len(call.arguments)

                    result = self._executor.execute(call, context, rng)
                    conv.add_tool_message(
                        tool_call_id=call.call_id, output=result.data
                    )
                    context.add_tool_output(result)
                    self._tracker.track_from_response(
                        result, call.endpoint_id, context.current_step
                    )

                context.advance_step()
                sm.transition(ConversationEvent.TOOL_RESULT)

                # Check if chain is complete after execution.
                if context.current_step >= len(flat):
                    sm.transition(ConversationEvent.CHAIN_COMPLETE)
                    break

                # Optional user follow-up between steps.
                remaining = len(flat) - context.current_step
                if remaining > 0 and conv.turn_count < gen_config.max_turns - 2:
                    follow = self._user_sim.generate_follow_up(context, rng)
                    conv.add_user_message(follow)
                    context.add_message("user", follow)
                    sm.transition(ConversationEvent.USER_MESSAGE)
                continue

            # Final answer.
            if resp.is_final_answer:
                sm.transition(ConversationEvent.ASSISTANT_FINAL)
                conv.add_assistant_message(content=resp.content)
                context.add_message("assistant", resp.content or "")
                break

            # Plain text response.
            sm.transition(ConversationEvent.ASSISTANT_TEXT)
            conv.add_assistant_message(content=resp.content)
            context.add_message("assistant", resp.content or "")

        return {
            "endpoints_called": endpoints_called,
            "disambiguation_count": disambiguation_count,
            "grounding_stats": {
                "grounded_args": grounded_args,
                "fresh_args": fresh_args,
                "total_args": total_args,
            },
        }

    # ------------------------------------------------------------------
    # Completion detection
    # ------------------------------------------------------------------

    def _is_complete(
        self,
        context: ConversationContext,
        conv: Conversation,
        gen_config: GenerationConfig,
        flat: list[ChainStep],
    ) -> bool:
        # Turn limit.
        if conv.turn_count >= gen_config.max_turns:
            return True

        # All chain steps executed.
        if context.current_step >= len(flat):
            return True

        # Last message was a final answer.
        if conv.messages:
            last = conv.messages[-1]
            if last.metadata.get("is_final_answer"):
                return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_gen_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_turns=self._config.max_turns,
            include_disambiguation=self._config.require_disambiguation,
            disambiguation_probability=self._config.disambiguation_probability,
            temperature=self._config.temperature,
            require_final_answer=self._config.require_final_answer,
            min_tool_calls=self._config.min_tool_calls,
            max_consecutive_tool_calls=self._config.max_consecutive_tool_calls,
        )

    def _force_final_answer(
        self,
        conv: Conversation,
        context: ConversationContext,
        chain: ToolChain,
        rng: np.random.Generator,
        gen_config: GenerationConfig,
    ) -> None:
        """Force the assistant to produce a final answer."""
        # Push current_step past all steps so generate_response returns final.
        flat = _flatten_steps(chain)
        while context.current_step < len(flat):
            context.advance_step()

        resp = self._assistant.generate_response(context, rng, gen_config)
        conv.add_assistant_message(content=resp.content)
        context.add_message("assistant", resp.content or "")
