"""Tests for evaluation pipeline models (Task 43)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tooluse_gen.evaluation.models import (
    Accepted,
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
    JudgeScores,
    RepairNeeded,
    ValidationResult,
)

pytestmark = pytest.mark.unit


# ===================================================================
# JudgeScores
# ===================================================================


class TestJudgeScores:
    def test_defaults(self):
        s = JudgeScores()
        assert s.tool_correctness == 1
        assert s.argument_grounding == 1
        assert s.task_completion == 1
        assert s.naturalness == 1
        assert s.reasoning == ""

    def test_construction_all_fields(self):
        s = JudgeScores(
            tool_correctness=4,
            argument_grounding=5,
            task_completion=3,
            naturalness=4,
            reasoning="Solid conversation.",
        )
        assert s.tool_correctness == 4
        assert s.reasoning == "Solid conversation."

    def test_average(self):
        s = JudgeScores(
            tool_correctness=4,
            argument_grounding=5,
            task_completion=3,
            naturalness=4,
        )
        assert s.average == 4.0

    def test_average_perfect(self):
        s = JudgeScores(
            tool_correctness=5,
            argument_grounding=5,
            task_completion=5,
            naturalness=5,
        )
        assert s.average == 5.0

    def test_average_minimum(self):
        s = JudgeScores()
        assert s.average == 1.0

    def test_scores_dict_keys(self):
        s = JudgeScores(
            tool_correctness=3,
            argument_grounding=4,
            task_completion=2,
            naturalness=5,
            reasoning="test",
        )
        d = s.scores_dict
        assert set(d.keys()) == {
            "tool_correctness",
            "argument_grounding",
            "task_completion",
            "naturalness",
        }
        assert "reasoning" not in d
        assert d["tool_correctness"] == 3

    def test_passes_threshold_true(self):
        s = JudgeScores(
            tool_correctness=4,
            argument_grounding=4,
            task_completion=4,
            naturalness=4,
        )
        assert s.passes_threshold(4.0)
        assert s.passes_threshold(3.5)

    def test_passes_threshold_false(self):
        s = JudgeScores()
        assert not s.passes_threshold(2.0)

    def test_passes_threshold_boundary(self):
        s = JudgeScores(
            tool_correctness=3,
            argument_grounding=4,
            task_completion=3,
            naturalness=4,
        )
        assert s.average == 3.5
        assert s.passes_threshold(3.5)
        assert not s.passes_threshold(3.6)

    def test_validation_too_low(self):
        with pytest.raises(ValidationError):
            JudgeScores(tool_correctness=0)

    def test_validation_too_high(self):
        with pytest.raises(ValidationError):
            JudgeScores(naturalness=6)

    def test_serialization_round_trip(self):
        s = JudgeScores(
            tool_correctness=5,
            argument_grounding=3,
            reasoning="ok",
        )
        data = s.model_dump()
        restored = JudgeScores.model_validate(data)
        assert restored.tool_correctness == 5
        assert restored.reasoning == "ok"


# ===================================================================
# EvaluationResult
# ===================================================================


class TestEvaluationResult:
    def test_default(self):
        r = EvaluationResult(conversation_id="c1")
        assert r.conversation_id == "c1"
        assert r.scores is None
        assert r.passed is False
        assert r.failure_reasons == []
        assert r.attempt_number == 1

    def test_passed(self):
        s = JudgeScores(
            tool_correctness=5,
            argument_grounding=4,
            task_completion=5,
            naturalness=4,
        )
        r = EvaluationResult(conversation_id="c1", scores=s, passed=True)
        assert r.passed
        assert r.scores is not None
        assert r.scores.average == 4.5

    def test_failed_with_reasons(self):
        r = EvaluationResult(
            conversation_id="c2",
            passed=False,
            failure_reasons=["low quality", "missing tools"],
        )
        assert not r.passed
        assert len(r.failure_reasons) == 2

    def test_with_validation_errors(self):
        r = EvaluationResult(
            conversation_id="c3",
            validation_errors=["no user message"],
        )
        assert r.validation_errors == ["no user message"]

    def test_serialization_round_trip(self):
        s = JudgeScores(tool_correctness=3, argument_grounding=3)
        r = EvaluationResult(conversation_id="c1", scores=s, passed=False)
        data = r.model_dump()
        restored = EvaluationResult.model_validate(data)
        assert restored.conversation_id == "c1"
        assert restored.scores is not None


# ===================================================================
# ValidationResult
# ===================================================================


class TestValidationResult:
    def test_valid(self):
        v = ValidationResult()
        assert v.valid is True
        assert v.errors == []

    def test_invalid_with_errors(self):
        v = ValidationResult(
            valid=False,
            errors=["no user message", "missing tool response"],
        )
        assert not v.valid
        assert len(v.errors) == 2

    def test_error_count(self):
        v = ValidationResult(errors=["a", "b", "c"])
        assert v.error_count == 3

    def test_empty_errors_valid(self):
        v = ValidationResult(valid=True, errors=[])
        assert v.error_count == 0


# ===================================================================
# RepairNeeded
# ===================================================================


class TestRepairNeeded:
    def test_structural(self):
        r = RepairNeeded(
            reason="structural",
            validation_errors=["missing tool response"],
        )
        assert r.reason == "structural"
        assert r.is_structural
        assert not r.is_quality
        assert r.scores is None

    def test_quality_with_scores(self):
        s = JudgeScores(tool_correctness=2, argument_grounding=1)
        r = RepairNeeded(reason="quality", scores=s)
        assert r.is_quality
        assert not r.is_structural
        assert r.scores is not None

    def test_is_structural_property(self):
        r = RepairNeeded(reason="structural")
        assert r.is_structural is True

    def test_is_quality_property(self):
        r = RepairNeeded(reason="quality")
        assert r.is_quality is True

    def test_with_conversation_id(self):
        r = RepairNeeded(reason="structural", conversation_id="c42")
        assert r.conversation_id == "c42"


# ===================================================================
# Accepted
# ===================================================================


class TestAccepted:
    def test_construction(self):
        s = JudgeScores(
            tool_correctness=5,
            argument_grounding=5,
            task_completion=5,
            naturalness=5,
        )
        a = Accepted(conversation_id="c1", scores=s)
        assert a.conversation_id == "c1"
        assert a.scores.average == 5.0

    def test_has_fields(self):
        s = JudgeScores(tool_correctness=4, argument_grounding=4)
        a = Accepted(conversation_id="c2", scores=s)
        assert isinstance(a.conversation_id, str)
        assert isinstance(a.scores, JudgeScores)

    def test_serialization_round_trip(self):
        s = JudgeScores(tool_correctness=3, argument_grounding=3)
        a = Accepted(conversation_id="c1", scores=s)
        data = a.model_dump()
        restored = Accepted.model_validate(data)
        assert restored.conversation_id == "c1"


# ===================================================================
# EvaluationConfig
# ===================================================================


class TestEvaluationConfig:
    def test_defaults(self):
        c = EvaluationConfig()
        assert c.min_score == 3.5
        assert c.model == "gpt-4o"
        assert c.max_retries == 3
        assert c.temperature == 0.3
        assert c.validate_structure is True

    def test_custom(self):
        c = EvaluationConfig(
            min_score=4.0,
            model="gpt-4o-mini",
            max_retries=5,
            temperature=0.1,
            validate_structure=False,
        )
        assert c.min_score == 4.0
        assert c.model == "gpt-4o-mini"

    def test_min_score_bounds(self):
        EvaluationConfig(min_score=1.0)
        EvaluationConfig(min_score=5.0)
        with pytest.raises(ValidationError):
            EvaluationConfig(min_score=0.5)
        with pytest.raises(ValidationError):
            EvaluationConfig(min_score=5.5)

    def test_max_retries_bounds(self):
        EvaluationConfig(max_retries=0)
        with pytest.raises(ValidationError):
            EvaluationConfig(max_retries=-1)


# ===================================================================
# EvaluationReport
# ===================================================================


class TestEvaluationReport:
    def test_defaults(self):
        r = EvaluationReport()
        assert r.total == 0
        assert r.passed == 0
        assert r.failed == 0
        assert r.discarded == 0
        assert r.average_scores is None
        assert r.score_distribution == {}
        assert r.repair_stats == {}

    def test_pass_rate(self):
        r = EvaluationReport(total=10, passed=7, failed=3)
        assert r.pass_rate == 0.7

    def test_pass_rate_zero_total(self):
        r = EvaluationReport()
        assert r.pass_rate == 0.0

    def test_from_results_mixed(self):
        results = [
            EvaluationResult(
                conversation_id="c1",
                scores=JudgeScores(
                    tool_correctness=5,
                    argument_grounding=4,
                    task_completion=5,
                    naturalness=4,
                ),
                passed=True,
            ),
            EvaluationResult(
                conversation_id="c2",
                scores=JudgeScores(
                    tool_correctness=2,
                    argument_grounding=2,
                    task_completion=2,
                    naturalness=2,
                ),
                passed=False,
                failure_reasons=["low"],
            ),
            EvaluationResult(
                conversation_id="c3",
                scores=JudgeScores(
                    tool_correctness=4,
                    argument_grounding=4,
                    task_completion=4,
                    naturalness=4,
                ),
                passed=True,
            ),
        ]
        report = EvaluationReport.from_results(results)
        assert report.total == 3
        assert report.passed == 2
        assert report.failed == 1

    def test_from_results_average_scores(self):
        results = [
            EvaluationResult(
                conversation_id="c1",
                scores=JudgeScores(
                    tool_correctness=4,
                    argument_grounding=4,
                    task_completion=4,
                    naturalness=4,
                ),
                passed=True,
            ),
            EvaluationResult(
                conversation_id="c2",
                scores=JudgeScores(
                    tool_correctness=2,
                    argument_grounding=2,
                    task_completion=2,
                    naturalness=2,
                ),
                passed=False,
            ),
        ]
        report = EvaluationReport.from_results(results)
        assert report.average_scores is not None
        # (4+2)/2 = 3 for each dimension
        assert report.average_scores.tool_correctness == 3

    def test_from_results_score_distribution(self):
        results = [
            EvaluationResult(
                conversation_id="c1",
                scores=JudgeScores(
                    tool_correctness=5,
                    argument_grounding=3,
                    task_completion=5,
                    naturalness=4,
                ),
                passed=True,
            ),
            EvaluationResult(
                conversation_id="c2",
                scores=JudgeScores(
                    tool_correctness=5,
                    argument_grounding=4,
                    task_completion=3,
                    naturalness=4,
                ),
                passed=True,
            ),
        ]
        report = EvaluationReport.from_results(results)
        dist = report.score_distribution
        assert "tool_correctness" in dist
        assert dist["tool_correctness"][5] == 2
        assert dist["tool_correctness"][1] == 0

    def test_from_results_no_scores(self):
        results = [
            EvaluationResult(conversation_id="c1"),
        ]
        report = EvaluationReport.from_results(results)
        assert report.total == 1
        assert report.average_scores is None
        assert report.score_distribution == {}

    def test_from_results_with_repair_stats(self):
        results = [
            EvaluationResult(conversation_id="c1", passed=True),
        ]
        stats = {"total_repairs": 2, "successful_repairs": 1}
        report = EvaluationReport.from_results(results, repair_stats=stats)
        assert report.repair_stats == stats
