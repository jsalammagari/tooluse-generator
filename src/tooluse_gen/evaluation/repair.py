"""Retry / repair loop for conversation quality.

:class:`RepairLoop` evaluates a conversation with structural validation
and LLM-as-judge scoring, then regenerates with feedback when the
conversation fails to meet quality thresholds.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.agents.orchestrator import ConversationOrchestrator
from tooluse_gen.evaluation.judge import JudgeAgent
from tooluse_gen.evaluation.models import (
    EvaluationConfig,
    EvaluationResult,
    JudgeScores,
)
from tooluse_gen.evaluation.validator import ConversationValidator
from tooluse_gen.graph.chain_models import ToolChain
from tooluse_gen.utils.logging import get_logger

logger = get_logger("evaluation.repair")


# ---------------------------------------------------------------------------
# RepairStats
# ---------------------------------------------------------------------------


class RepairStats(BaseModel):
    """Tracks repair statistics across evaluations."""

    model_config = ConfigDict(use_enum_values=True)

    total_attempts: int = Field(default=0, description="Total regeneration attempts.")
    structural_repairs: int = Field(default=0, description="Structural failures.")
    quality_repairs: int = Field(default=0, description="Quality failures.")
    successful_repairs: int = Field(default=0, description="Repairs that passed.")
    failed_repairs: int = Field(default=0, description="Still failed after retries.")
    attempts_distribution: dict[int, int] = Field(
        default_factory=dict,
        description="Conversations that passed on attempt N.",
    )


# ---------------------------------------------------------------------------
# RepairLoop
# ---------------------------------------------------------------------------


class RepairLoop:
    """Evaluates conversations and repairs them via retry with feedback."""

    def __init__(
        self,
        orchestrator: ConversationOrchestrator,
        validator: ConversationValidator,
        judge: JudgeAgent,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._validator = validator
        self._judge = judge
        self._config = config or EvaluationConfig()
        self._stats = RepairStats()
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_and_repair(
        self,
        conversation: Conversation,
        chain: ToolChain,
        seed: int = 42,
    ) -> tuple[Conversation, EvaluationResult]:
        """Evaluate *conversation* and retry up to ``max_retries`` times."""
        max_attempts = self._config.max_retries + 1
        scores: JudgeScores | None = None

        for attempt in range(max_attempts):
            # Step 1 — structural validation.
            if self._config.validate_structure:
                validation = self._validator.validate(conversation)
                if not validation.valid:
                    self._logger.warning(
                        "Structural issues (attempt %d): %s",
                        attempt + 1,
                        validation.errors,
                    )
                    self._stats.structural_repairs += 1
                    if attempt < max_attempts - 1:
                        conversation = self._regenerate_with_feedback(
                            chain, seed + attempt + 1, validation.errors
                        )
                    continue

            # Step 2 — quality scoring.
            scores = self._judge.score(conversation)
            if scores.passes_threshold(self._config.min_score):
                attempt_num = attempt + 1
                self._stats.successful_repairs += 1 if attempt > 0 else 0
                self._stats.attempts_distribution[attempt_num] = (
                    self._stats.attempts_distribution.get(attempt_num, 0) + 1
                )
                return conversation, EvaluationResult(
                    conversation_id=conversation.conversation_id,
                    scores=scores,
                    passed=True,
                    attempt_number=attempt_num,
                )

            # Quality failed — regenerate if attempts remain.
            self._logger.warning(
                "Quality below threshold (attempt %d): avg=%.2f < %.2f — %s",
                attempt + 1,
                scores.average,
                self._config.min_score,
                scores.reasoning,
            )
            self._stats.quality_repairs += 1
            if attempt < max_attempts - 1:
                conversation = self._regenerate_with_feedback(
                    chain, seed + attempt + 1, [scores.reasoning]
                )

        # All attempts exhausted.
        self._stats.failed_repairs += 1
        return conversation, EvaluationResult(
            conversation_id=conversation.conversation_id,
            scores=scores,
            passed=False,
            failure_reasons=["max_retries_exceeded"],
            attempt_number=max_attempts,
        )

    def evaluate_and_repair_batch(
        self,
        conversations: list[tuple[Conversation, ToolChain]],
        seed: int = 42,
    ) -> list[tuple[Conversation, EvaluationResult]]:
        """Evaluate and repair a batch of conversations."""
        results: list[tuple[Conversation, EvaluationResult]] = []
        for i, (conv, chain) in enumerate(conversations):
            conv_seed = seed + i * 100
            results.append(self.evaluate_and_repair(conv, chain, seed=conv_seed))
            if (i + 1) % 10 == 0 or i == len(conversations) - 1:
                self._logger.info(
                    "Evaluated %d/%d conversations", i + 1, len(conversations)
                )
        return results

    def get_stats(self) -> RepairStats:
        """Return a copy of the current repair statistics."""
        return self._stats.model_copy()

    def reset_stats(self) -> None:
        """Reset statistics to zero."""
        self._stats = RepairStats()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _regenerate_with_feedback(
        self,
        chain: ToolChain,
        seed: int,
        feedback: list[str],
    ) -> Conversation:
        """Regenerate the conversation with a different seed."""
        self._stats.total_attempts += 1
        # Store feedback in chain metadata (orchestrator may use it in LLM mode).
        chain_copy = chain.model_copy(deep=True)
        chain_copy.metadata["repair_feedback"] = feedback
        return self._orchestrator.generate_conversation(chain_copy, seed=seed)

    def _identify_problematic_turn(
        self,
        conversation: Conversation,
        scores: JudgeScores,
    ) -> int | None:
        """Heuristic: identify the turn most likely causing quality issues."""
        msgs = conversation.messages
        if not msgs:
            return None

        if scores.naturalness <= 2:
            for i, m in enumerate(msgs):
                if m.role == "user":
                    return i

        if scores.argument_grounding <= 2:
            for i, m in enumerate(msgs):
                if m.role == "assistant" and m.tool_calls:
                    return i

        if scores.task_completion <= 2:
            return len(msgs) - 1

        return None
