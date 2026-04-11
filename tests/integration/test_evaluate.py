"""Integration tests for the ``tooluse evaluate`` CLI command (Task 60)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from tooluse_gen.cli.main import app

pytestmark = pytest.mark.integration

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixture JSONL data
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "conversation_id": "conv_001",
        "messages": [
            {"role": "user", "content": "Find me a hotel in Paris"},
            {"role": "assistant", "content": "What is your budget?"},
            {"role": "user", "content": "Under 200 euros"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "endpoint": "hotels/search",
                        "arguments": {"city": "Paris", "max_price": 200},
                        "tool_name": "Hotels API",
                        "call_id": "c1",
                    }
                ],
            },
            {"role": "tool", "content": {"results": [{"id": "h1", "name": "Hotel Marais"}]}},
            {"role": "assistant", "content": "I found Hotel Marais for you!"},
        ],
        "judge_scores": {
            "tool_correctness": 4,
            "argument_grounding": 3,
            "task_completion": 5,
            "naturalness": 4,
        },
        "metadata": {
            "seed": 42,
            "tools_used": ["hotels_api"],
            "domains": ["Travel"],
            "num_turns": 6,
        },
    },
    {
        "conversation_id": "conv_002",
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "endpoint": "weather/current",
                        "arguments": {"city": "Tokyo"},
                        "tool_name": "Weather API",
                        "call_id": "c2",
                    }
                ],
            },
            {"role": "tool", "content": {"temp": 22, "condition": "sunny"}},
            {"role": "assistant", "content": "It's 22C and sunny in Tokyo!"},
        ],
        "judge_scores": {
            "tool_correctness": 5,
            "argument_grounding": 4,
            "task_completion": 5,
            "naturalness": 5,
        },
        "metadata": {
            "seed": 43,
            "tools_used": ["weather_api"],
            "domains": ["Weather"],
            "num_turns": 4,
        },
    },
    {
        "conversation_id": "conv_003",
        "messages": [
            {"role": "user", "content": "Book a flight to London"},
            {"role": "assistant", "content": "When would you like to travel?"},
            {"role": "user", "content": "Next Friday"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "endpoint": "flights/search",
                        "arguments": {"destination": "London"},
                        "tool_name": "Flights API",
                        "call_id": "c3",
                    }
                ],
            },
            {"role": "tool", "content": {"flights": [{"id": "f1", "price": 150}]}},
            {"role": "assistant", "content": "Found a flight for 150 EUR!"},
        ],
        "judge_scores": None,
        "metadata": {
            "seed": 44,
            "tools_used": ["flights_api"],
            "domains": ["Travel"],
            "num_turns": 6,
        },
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conversations_jsonl(tmp_path: Path) -> Path:
    """Create a fixture JSONL file with sample conversations."""
    path = tmp_path / "conversations.jsonl"
    with open(path, "w") as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")
    return path


@pytest.fixture()
def conversations_jsonl_no_scores(tmp_path: Path) -> Path:
    """Fixture JSONL where all records lack judge_scores."""
    path = tmp_path / "no_scores.jsonl"
    with open(path, "w") as f:
        for rec in _SAMPLE_RECORDS:
            copy = dict(rec)
            copy["judge_scores"] = None
            f.write(json.dumps(copy) + "\n")
    return path


# ---------------------------------------------------------------------------
# Mock embedding service (for E2E test)
# ---------------------------------------------------------------------------


class _MockEmbeddingService:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self._model = None
        self._cache_dir = None
        self.model_name = "mock"

    def _get_model(self) -> object:
        return None

    def embed_text(self, text: str) -> list[float]:
        h = hash(text)
        rng = np.random.default_rng(abs(h) % (2**31))
        vec = rng.standard_normal(384).tolist()
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_batch(
        self, texts: list[str], batch_size: int = 256, show_progress: bool = False
    ) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def compute_similarity(self, emb_a: list[float], emb_b: list[float]) -> float:
        a = np.asarray(emb_a, dtype=np.float64)
        b = np.asarray(emb_b, dtype=np.float64)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return 0.0 if na == 0.0 or nb == 0.0 else float(np.dot(a, b) / (na * nb))

    def compute_similarity_matrix(
        self, embeddings: list[list[float]]
    ) -> np.ndarray:  # type: ignore[type-arg]
        if not embeddings:
            return np.empty((0, 0), dtype=np.float64)
        mat = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = mat / norms
        return normed @ normed.T  # type: ignore[no-any-return]


# ===================================================================
# Tests
# ===================================================================


class TestEvaluateCommand:
    """Core evaluate command tests."""

    def test_evaluate_runs_successfully(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, ["evaluate", str(conversations_jsonl)])
        assert result.exit_code == 0, result.output

    def test_evaluate_displays_report(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, ["evaluate", str(conversations_jsonl)])
        assert result.exit_code == 0
        assert "Evaluation Report" in result.output or "Total" in result.output

    def test_evaluate_table_format(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, [
            "evaluate", str(conversations_jsonl), "--format", "table",
        ])
        assert result.exit_code == 0
        assert "Evaluation Report" in result.output

    def test_evaluate_json_format(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, [
            "evaluate", str(conversations_jsonl), "--format", "json",
        ])
        assert result.exit_code == 0
        # Output should contain valid JSON with report data
        # Find the JSON block in the output (may have Rich markup around it)
        assert "total" in result.output
        assert "pass_rate" in result.output

    def test_evaluate_markdown_format(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, [
            "evaluate", str(conversations_jsonl), "--format", "markdown",
        ])
        assert result.exit_code == 0
        assert "# Evaluation Report" in result.output
        assert "| Metric | Value |" in result.output

    def test_evaluate_rescore(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, [
            "evaluate", str(conversations_jsonl), "--rescore",
        ])
        assert result.exit_code == 0
        # Should show "Rescored" in output
        output_lower = result.output.lower()
        assert "rescored" in output_lower or "scored" in output_lower

    def test_evaluate_shows_pass_rate(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, ["evaluate", str(conversations_jsonl)])
        assert result.exit_code == 0
        assert "Pass rate" in result.output or "pass_rate" in result.output

    def test_evaluate_shows_score_dimensions(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, ["evaluate", str(conversations_jsonl)])
        assert result.exit_code == 0
        output = result.output
        # At least some score dimensions should appear
        assert "tool_correctness" in output or "Mean score" in output

    def test_evaluate_shows_diversity(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, ["evaluate", str(conversations_jsonl)])
        assert result.exit_code == 0
        assert "entropy" in result.output.lower() or "Unique tools" in result.output


class TestEvaluateOutput:
    """Tests for --output flag (enriched JSONL)."""

    def test_writes_enriched_output(
        self, conversations_jsonl: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enriched.jsonl"
        result = runner.invoke(app, [
            "evaluate", str(conversations_jsonl), "--output", str(out),
        ])
        assert result.exit_code == 0
        assert out.exists()

    def test_enriched_has_scores(
        self, conversations_jsonl: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enriched.jsonl"
        runner.invoke(app, [
            "evaluate", str(conversations_jsonl), "--output", str(out), "--rescore",
        ])
        records = []
        for line in out.read_text().strip().split("\n"):
            data = json.loads(line)
            if "conversation_id" in data:
                records.append(data)
        assert len(records) == len(_SAMPLE_RECORDS)
        for rec in records:
            assert rec["judge_scores"] is not None
            assert "tool_correctness" in rec["judge_scores"]

    def test_enriched_record_count(
        self, conversations_jsonl: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "enriched.jsonl"
        runner.invoke(app, [
            "evaluate", str(conversations_jsonl), "--output", str(out),
        ])
        records = [
            json.loads(line) for line in out.read_text().strip().split("\n")
            if "conversation_id" in line
        ]
        assert len(records) == len(_SAMPLE_RECORDS)


class TestEvaluateAutoScore:
    """Tests for automatic scoring when records lack scores."""

    def test_scores_when_missing(
        self, conversations_jsonl_no_scores: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "scored.jsonl"
        result = runner.invoke(app, [
            "evaluate", str(conversations_jsonl_no_scores),
            "--output", str(out),
        ])
        assert result.exit_code == 0
        records = [
            json.loads(line) for line in out.read_text().strip().split("\n")
            if "conversation_id" in line
        ]
        for rec in records:
            assert rec["judge_scores"] is not None


class TestEvaluateErrorHandling:
    """Tests for error conditions."""

    def test_missing_input_file(self) -> None:
        result = runner.invoke(app, ["evaluate", "/tmp/nonexistent_xyz.jsonl"])
        assert result.exit_code != 0

    def test_empty_input_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(app, ["evaluate", str(empty)])
        assert result.exit_code != 0

    def test_invalid_format_option(self) -> None:
        result = runner.invoke(app, ["evaluate", "f.jsonl", "--format", "xml"])
        assert result.exit_code == 1


class TestEvaluateQuietMode:
    """Tests for --quiet flag."""

    def test_quiet_suppresses_output(
        self, conversations_jsonl: Path
    ) -> None:
        result = runner.invoke(app, [
            "--quiet", "evaluate", str(conversations_jsonl),
        ])
        assert result.exit_code == 0
        assert "Evaluation Report" not in result.output


class TestEvaluateEndToEnd:
    """Full pipeline: build -> generate -> evaluate."""

    @pytest.fixture(autouse=True)
    def _mock_embeddings(self) -> object:
        with (
            patch("tooluse_gen.graph.builder.EmbeddingService", _MockEmbeddingService),
            patch("tooluse_gen.graph.embeddings.EmbeddingService", _MockEmbeddingService),
        ):
            yield

    def test_build_generate_evaluate_pipeline(self, tmp_path: Path) -> None:
        # Create fixture tools
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i, tool in enumerate([
            {
                "tool_name": "Weather API",
                "tool_description": "Weather forecasts",
                "api_list": [{
                    "name": "current",
                    "url": "/current",
                    "description": "Current weather",
                    "method": "GET",
                    "required_parameters": [
                        {"name": "city", "type": "STRING", "description": "City", "default": ""}
                    ],
                    "optional_parameters": [],
                }],
            },
            {
                "tool_name": "Hotels API",
                "tool_description": "Hotel booking",
                "api_list": [
                    {
                        "name": "search",
                        "url": "/search",
                        "description": "Search hotels",
                        "method": "GET",
                        "required_parameters": [
                            {
                                "name": "city",
                                "type": "STRING",
                                "description": "City",
                                "default": "",
                            }
                        ],
                        "optional_parameters": [],
                    },
                    {
                        "name": "book",
                        "url": "/book",
                        "description": "Book hotel",
                        "method": "POST",
                        "required_parameters": [
                            {
                                "name": "hotel_id",
                                "type": "STRING",
                                "description": "ID",
                                "default": "",
                            }
                        ],
                        "optional_parameters": [],
                    },
                ],
            },
        ]):
            (data_dir / f"tool_{i}.json").write_text(json.dumps(tool))

        build_out = tmp_path / "build"
        gen_out = tmp_path / "conversations.jsonl"

        # Build
        r = runner.invoke(app, [
            "build", "--input-dir", str(data_dir),
            "--output-dir", str(build_out),
            "--force", "--similarity-threshold", "0.1",
        ])
        assert r.exit_code == 0, f"Build failed:\n{r.output}"

        # Generate
        r = runner.invoke(app, [
            "generate", "--output", str(gen_out),
            "--build-dir", str(build_out),
            "--count", "3", "--seed", "42",
            "--min-steps", "1", "--max-steps", "3",
        ])
        assert r.exit_code == 0, f"Generate failed:\n{r.output}"

        # Evaluate
        r = runner.invoke(app, ["evaluate", str(gen_out)])
        assert r.exit_code == 0, f"Evaluate failed:\n{r.output}"
        assert "Evaluation Report" in r.output

        # Evaluate with enriched output
        eval_out = tmp_path / "evaluated.jsonl"
        r = runner.invoke(app, [
            "evaluate", str(gen_out), "--output", str(eval_out), "--rescore",
        ])
        assert r.exit_code == 0
        assert eval_out.exists()
        records = [
            json.loads(line) for line in eval_out.read_text().strip().split("\n")
            if "conversation_id" in line
        ]
        assert len(records) >= 1
        for rec in records:
            assert rec.get("judge_scores") is not None
