"""Evaluation pipeline data models.

:class:`JudgeScores` captures integer 1–5 LLM-as-judge ratings with
a ``reasoning`` explanation.  :class:`EvaluationResult` wraps the
outcome for a single conversation.  :class:`ValidationResult` records
structural checks while :class:`RepairNeeded` and :class:`Accepted`
signal the pipeline's next action.

:class:`EvaluationConfig` controls thresholds and retry behaviour.
:class:`EvaluationReport` aggregates batch-level statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field

# ---------------------------------------------------------------------------
# JudgeScores
# ---------------------------------------------------------------------------


class JudgeScores(BaseModel):
    """LLM-as-judge quality scores on a 1–5 integer scale."""

    model_config = ConfigDict(use_enum_values=True)

    tool_correctness: int = Field(
        default=1, ge=1, le=5, description="Did assistant pick appropriate tools?"
    )
    argument_grounding: int = Field(
        default=1, ge=1, le=5, description="Are arguments valid and grounded?"
    )
    task_completion: int = Field(
        default=1, ge=1, le=5, description="Did conversation achieve user's goal?"
    )
    naturalness: int = Field(
        default=1, ge=1, le=5, description="Does it read like a real conversation?"
    )
    reasoning: str = Field(default="", description="Brief judge explanation.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average(self) -> float:
        """Mean of the four dimension scores."""
        return (
            self.tool_correctness
            + self.argument_grounding
            + self.task_completion
            + self.naturalness
        ) / 4.0

    @property
    def scores_dict(self) -> dict[str, int]:
        """Score values keyed by dimension (excludes reasoning)."""
        return {
            "tool_correctness": self.tool_correctness,
            "argument_grounding": self.argument_grounding,
            "task_completion": self.task_completion,
            "naturalness": self.naturalness,
        }

    def passes_threshold(self, threshold: float) -> bool:
        """Return ``True`` when :pyattr:`average` ≥ *threshold*."""
        return self.average >= threshold


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Outcome of evaluating a single conversation."""

    model_config = ConfigDict(use_enum_values=True)

    conversation_id: str = Field(..., description="Conversation identifier.")
    scores: JudgeScores | None = Field(default=None, description="Judge scores.")
    passed: bool = Field(default=False, description="Passed quality threshold.")
    failure_reasons: list[str] = Field(
        default_factory=list, description="Why it failed."
    )
    attempt_number: int = Field(default=1, description="Attempt that produced this.")
    validation_errors: list[str] = Field(
        default_factory=list, description="Structural validation errors."
    )


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Structural validation result."""

    model_config = ConfigDict(use_enum_values=True)

    valid: bool = Field(default=True, description="Structurally valid.")
    errors: list[str] = Field(default_factory=list, description="Issues found.")

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.errors)


# ---------------------------------------------------------------------------
# RepairNeeded
# ---------------------------------------------------------------------------


class RepairNeeded(BaseModel):
    """Signals that a conversation needs repair."""

    model_config = ConfigDict(use_enum_values=True)

    reason: str = Field(..., description="'structural' or 'quality'.")
    scores: JudgeScores | None = Field(
        default=None, description="Scores if quality failure."
    )
    validation_errors: list[str] = Field(
        default_factory=list, description="Errors if structural failure."
    )
    conversation_id: str = Field(default="", description="Conversation to repair.")

    @property
    def is_structural(self) -> bool:
        """True when the failure is structural."""
        return self.reason == "structural"

    @property
    def is_quality(self) -> bool:
        """True when the failure is quality-related."""
        return self.reason == "quality"


# ---------------------------------------------------------------------------
# Accepted
# ---------------------------------------------------------------------------


class Accepted(BaseModel):
    """Signals that a conversation passed evaluation."""

    model_config = ConfigDict(use_enum_values=True)

    conversation_id: str = Field(..., description="Conversation identifier.")
    scores: JudgeScores = Field(..., description="Passing scores.")


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation pipeline."""

    model_config = ConfigDict(use_enum_values=True)

    min_score: float = Field(
        default=3.5, ge=1.0, le=5.0, description="Minimum average to pass."
    )
    model: str = Field(default="gpt-4o", description="LLM model for judging.")
    max_retries: int = Field(default=3, ge=0, description="Max repair attempts.")
    temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Judge temperature."
    )
    validate_structure: bool = Field(
        default=True, description="Run structural validation first."
    )


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------


class EvaluationReport(BaseModel):
    """Aggregate report for a batch of evaluations."""

    model_config = ConfigDict(use_enum_values=True)

    total: int = Field(default=0, description="Total evaluated.")
    passed: int = Field(default=0, description="Passed.")
    failed: int = Field(default=0, description="Failed after retries.")
    discarded: int = Field(default=0, description="Discarded (unfixable).")
    average_scores: JudgeScores | None = Field(
        default=None, description="Mean scores."
    )
    score_distribution: dict[str, dict[int, int]] = Field(
        default_factory=dict, description="Per-dimension score counts."
    )
    repair_stats: dict[str, int] = Field(
        default_factory=dict, description="Repair statistics."
    )

    @property
    def pass_rate(self) -> float:
        """Fraction of conversations that passed."""
        return self.passed / max(self.total, 1)

    @classmethod
    def from_results(
        cls,
        results: list[EvaluationResult],
        repair_stats: dict[str, int] | None = None,
    ) -> EvaluationReport:
        """Compute an aggregate report from individual results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        # Collect scores from results that have them.
        scored: list[JudgeScores] = [
            r.scores for r in results if r.scores is not None
        ]

        avg: JudgeScores | None = None
        dist: dict[str, dict[int, int]] = {}

        if scored:
            avg = JudgeScores(
                tool_correctness=round(
                    sum(s.tool_correctness for s in scored) / len(scored)
                ),
                argument_grounding=round(
                    sum(s.argument_grounding for s in scored) / len(scored)
                ),
                task_completion=round(
                    sum(s.task_completion for s in scored) / len(scored)
                ),
                naturalness=round(
                    sum(s.naturalness for s in scored) / len(scored)
                ),
            )

            for dim in (
                "tool_correctness",
                "argument_grounding",
                "task_completion",
                "naturalness",
            ):
                counts: dict[int, int] = {v: 0 for v in range(1, 6)}
                for s in scored:
                    val = getattr(s, dim)
                    counts[val] = counts.get(val, 0) + 1
                dist[dim] = counts

        return cls(
            total=total,
            passed=passed,
            failed=failed,
            average_scores=avg,
            score_distribution=dist,
            repair_stats=repair_stats or {},
        )
