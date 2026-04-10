"""Unified evaluation pipeline.

:class:`EvaluationPipeline` combines structural validation, LLM-as-judge
scoring, and an optional retry/repair loop into a single entry point for
evaluating generated conversations.
"""

from __future__ import annotations

from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.evaluation.judge import JudgeAgent
from tooluse_gen.evaluation.models import (
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
)
from tooluse_gen.evaluation.repair import RepairLoop
from tooluse_gen.evaluation.validator import ConversationValidator
from tooluse_gen.graph.chain_models import ToolChain
from tooluse_gen.utils.logging import get_logger

logger = get_logger("evaluation.pipeline")


class EvaluationPipeline:
    """Unified evaluation pipeline: validate → judge → repair."""

    def __init__(
        self,
        validator: ConversationValidator,
        judge: JudgeAgent,
        repair_loop: RepairLoop | None = None,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._validator = validator
        self._judge = judge
        self._repair_loop = repair_loop
        self._config = config or EvaluationConfig()
        self._results: list[EvaluationResult] = []
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        conversation: Conversation,
        chain: ToolChain | None = None,
        seed: int = 42,
    ) -> EvaluationResult:
        """Validate, score, and optionally repair a single conversation."""
        result = self._evaluate(conversation, chain, seed)
        self._results.append(result)
        return result

    def _evaluate(
        self,
        conversation: Conversation,
        chain: ToolChain | None,
        seed: int,
    ) -> EvaluationResult:
        """Core evaluation logic (does not accumulate results)."""
        # Step 1 — structural validation.
        if self._config.validate_structure:
            validation = self._validator.validate(conversation)
            if not validation.valid:
                if self._repair_loop is not None and chain is not None:
                    _, result = self._repair_loop.evaluate_and_repair(
                        conversation, chain, seed=seed
                    )
                    return result
                return EvaluationResult(
                    conversation_id=conversation.conversation_id,
                    passed=False,
                    validation_errors=validation.errors,
                    failure_reasons=["structural_validation_failed"],
                )

        # Step 2 — quality scoring.
        scores = self._judge.score(conversation)

        # Step 3 — threshold check.
        if scores.passes_threshold(self._config.min_score):
            return EvaluationResult(
                conversation_id=conversation.conversation_id,
                scores=scores,
                passed=True,
            )

        # Quality failed — attempt repair if available.
        if self._repair_loop is not None and chain is not None:
            _, result = self._repair_loop.evaluate_and_repair(
                conversation, chain, seed=seed
            )
            return result

        return EvaluationResult(
            conversation_id=conversation.conversation_id,
            scores=scores,
            passed=False,
            failure_reasons=[
                f"quality_below_threshold: {scores.average:.2f} < {self._config.min_score}"
            ],
        )

    def evaluate_batch(
        self,
        conversations: list[Conversation],
        chains: list[ToolChain | None] | None = None,
        seed: int = 42,
    ) -> EvaluationReport:
        """Evaluate a batch and return an aggregate report."""
        for i, conv in enumerate(conversations):
            chain = chains[i] if chains is not None else None
            self.evaluate_single(conv, chain=chain, seed=seed + i)
            if (i + 1) % 10 == 0 or i == len(conversations) - 1:
                self._logger.info(
                    "Evaluated %d/%d conversations", i + 1, len(conversations)
                )
        return self.generate_report()

    def generate_report(self) -> EvaluationReport:
        """Build an :class:`EvaluationReport` from accumulated results."""
        repair_stats: dict[str, int] | None = None
        if self._repair_loop is not None:
            raw = self._repair_loop.get_stats().model_dump()
            repair_stats = {
                k: v for k, v in raw.items() if isinstance(v, int)
            }
        return EvaluationReport.from_results(
            self._results, repair_stats=repair_stats
        )

    def get_results(self) -> list[EvaluationResult]:
        """Return a copy of accumulated evaluation results."""
        return list(self._results)

    def reset(self) -> None:
        """Clear accumulated results and repair stats."""
        self._results.clear()
        if self._repair_loop is not None:
            self._repair_loop.reset_stats()
