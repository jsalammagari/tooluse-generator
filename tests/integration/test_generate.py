"""Integration tests for the ``tooluse generate`` CLI command (Task 59)."""

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
# Mock embedding service (same as test_build.py)
# ---------------------------------------------------------------------------


class _MockEmbeddingService:
    """Deterministic hash-based embeddings for fast, offline testing."""

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
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def compute_similarity(self, emb_a: list[float], emb_b: list[float]) -> float:
        a = np.asarray(emb_a, dtype=np.float64)
        b = np.asarray(emb_b, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

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

    def save_embeddings(
        self, embeddings: dict[str, list[float]], output_path: Path
    ) -> None:
        import joblib

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(embeddings), output_path)

    def load_embeddings(self, input_path: Path) -> dict[str, list[float]]:
        import joblib

        return joblib.load(input_path)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Fixture tool data
# ---------------------------------------------------------------------------

_WEATHER_TOOL = {
    "tool_name": "Weather API",
    "tool_description": "Get weather forecasts and current conditions",
    "home_url": "https://api.weather.example.com",
    "api_list": [
        {
            "name": "current_weather",
            "url": "https://api.weather.example.com/current",
            "description": "Get current weather for a city",
            "method": "GET",
            "required_parameters": [
                {"name": "city", "type": "STRING", "description": "City name", "default": ""}
            ],
            "optional_parameters": [
                {
                    "name": "units",
                    "type": "STRING",
                    "description": "Temperature units",
                    "default": "metric",
                }
            ],
        },
        {
            "name": "forecast",
            "url": "https://api.weather.example.com/forecast",
            "description": "Get weather forecast for upcoming days",
            "method": "GET",
            "required_parameters": [
                {"name": "city", "type": "STRING", "description": "City name", "default": ""}
            ],
            "optional_parameters": [
                {"name": "days", "type": "NUMBER", "description": "Forecast days", "default": "5"}
            ],
        },
    ],
}

_HOTEL_TOOL = {
    "tool_name": "Hotel Booking API",
    "tool_description": "Search and book hotels worldwide",
    "home_url": "https://api.hotels.example.com",
    "api_list": [
        {
            "name": "search_hotels",
            "url": "https://api.hotels.example.com/search",
            "description": "Search for available hotels in a city",
            "method": "GET",
            "required_parameters": [
                {"name": "city", "type": "STRING", "description": "City name", "default": ""},
                {
                    "name": "check_in",
                    "type": "STRING",
                    "description": "Check-in date",
                    "default": "",
                },
            ],
            "optional_parameters": [
                {
                    "name": "max_price",
                    "type": "NUMBER",
                    "description": "Max price per night",
                    "default": "",
                }
            ],
        },
        {
            "name": "book_hotel",
            "url": "https://api.hotels.example.com/book",
            "description": "Book a hotel room",
            "method": "POST",
            "required_parameters": [
                {
                    "name": "hotel_id",
                    "type": "STRING",
                    "description": "Hotel identifier",
                    "default": "",
                },
                {
                    "name": "guest_name",
                    "type": "STRING",
                    "description": "Guest full name",
                    "default": "",
                },
            ],
            "optional_parameters": [],
        },
    ],
}

_FLIGHT_TOOL = {
    "tool_name": "Flight Search API",
    "tool_description": "Search for flights between airports",
    "home_url": "https://api.flights.example.com",
    "api_list": [
        {
            "name": "search_flights",
            "url": "https://api.flights.example.com/search",
            "description": "Search for flights between two airports",
            "method": "GET",
            "required_parameters": [
                {
                    "name": "origin",
                    "type": "STRING",
                    "description": "Origin airport code",
                    "default": "",
                },
                {
                    "name": "destination",
                    "type": "STRING",
                    "description": "Destination airport code",
                    "default": "",
                },
            ],
            "optional_parameters": [],
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_embeddings() -> object:
    """Patch EmbeddingService everywhere so tests never download a real model."""
    with (
        patch("tooluse_gen.graph.builder.EmbeddingService", _MockEmbeddingService),
        patch("tooluse_gen.graph.embeddings.EmbeddingService", _MockEmbeddingService),
    ):
        yield


@pytest.fixture()
def build_dir(tmp_path: Path) -> Path:
    """Build artifacts from fixture tool data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name, tool in [
        ("weather.json", _WEATHER_TOOL),
        ("hotels.json", _HOTEL_TOOL),
        ("flights.json", _FLIGHT_TOOL),
    ]:
        (data_dir / name).write_text(json.dumps(tool))

    out = tmp_path / "build"
    result = runner.invoke(app, [
        "build",
        "--input-dir", str(data_dir),
        "--output-dir", str(out),
        "--force",
        "--similarity-threshold", "0.1",  # low threshold for more edges in small graph
    ])
    assert result.exit_code == 0, f"Build fixture failed:\n{result.output}"
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_generate(
    build_dir: Path,
    output_path: Path,
    extra_args: list[str] | None = None,
) -> object:
    """Invoke ``tooluse generate`` and return the CliRunner result."""
    args = [
        "generate",
        "--output", str(output_path),
        "--build-dir", str(build_dir),
        "--count", "3",
        "--seed", "42",
        "--min-steps", "1",
        "--max-steps", "3",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(app, args)


def _read_records(path: Path) -> list[dict]:
    """Read JSONL, returning only conversation records (skip header)."""
    records: list[dict] = []
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        if "conversation_id" in data:
            records.append(data)
    return records


# ===================================================================
# Tests
# ===================================================================


class TestGenerateCommand:
    """Core generate command tests."""

    def test_generate_creates_output_file(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(build_dir, out)
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_generate_writes_records(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        records = _read_records(out)
        assert len(records) >= 1

    def test_generate_record_format(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        records = _read_records(out)
        for rec in records:
            assert "conversation_id" in rec
            assert "messages" in rec
            assert isinstance(rec["messages"], list)
            assert len(rec["messages"]) >= 1
            # metadata should be present
            assert "metadata" in rec

    def test_generate_messages_have_roles(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        records = _read_records(out)
        valid_roles = {"user", "assistant", "tool"}
        for rec in records:
            for msg in rec["messages"]:
                assert "role" in msg
                assert msg["role"] in valid_roles

    def test_generate_has_tool_calls(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        records = _read_records(out)
        # At least one conversation should have tool calls
        has_tc = any(
            any(msg.get("tool_calls") for msg in rec["messages"])
            for rec in records
        )
        assert has_tc, "No conversations contain tool calls"

    def test_generate_metadata_present(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        records = _read_records(out)
        for rec in records:
            assert isinstance(rec.get("metadata"), dict)

    def test_generate_summary_output(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(build_dir, out)
        assert result.exit_code == 0
        # Summary table should be in the output
        assert "Generation Summary" in result.output or "Generated" in result.output

    def test_generate_respects_seed(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out1 = tmp_path / "out1.jsonl"
        out2 = tmp_path / "out2.jsonl"
        _run_generate(build_dir, out1)
        _run_generate(build_dir, out2)
        r1 = _read_records(out1)
        r2 = _read_records(out2)
        assert len(r1) == len(r2)
        # Same seed should produce same number of messages per conversation
        lens1 = [len(r["messages"]) for r in r1]
        lens2 = [len(r["messages"]) for r in r2]
        assert lens1 == lens2

    def test_generate_respects_count(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = runner.invoke(app, [
            "generate",
            "--output", str(out),
            "--build-dir", str(build_dir),
            "--count", "2",
            "--seed", "42",
            "--min-steps", "1",
            "--max-steps", "3",
        ])
        assert result.exit_code == 0
        records = _read_records(out)
        assert len(records) == 2

    def test_generate_has_header(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        first_line = out.read_text().strip().split("\n")[0]
        header = json.loads(first_line)
        # Header should have run config metadata, not a conversation
        assert "conversation_id" not in header or "seed" in header


class TestGenerateSteering:
    """Tests for steering flag."""

    def test_steering_enabled_by_default(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(build_dir, out)
        assert result.exit_code == 0
        assert "enabled" in result.output.lower() or "True" in result.output

    def test_no_steering_flag(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(
            build_dir, out, ["--no-cross-conversation-steering"],
        )
        assert result.exit_code == 0
        assert "disabled" in result.output.lower() or "False" in result.output


class TestGenerateErrorHandling:
    """Tests for error conditions."""

    def test_missing_build_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "generate",
            "--output", str(tmp_path / "out.jsonl"),
            "--build-dir", str(tmp_path / "nonexistent"),
        ])
        assert result.exit_code != 0

    def test_missing_registry(self, tmp_path: Path) -> None:
        # Create build_dir with graph but no registry
        bd = tmp_path / "bad_build"
        bd.mkdir()
        (bd / "graph.pkl").write_bytes(b"fake")
        result = runner.invoke(app, [
            "generate",
            "--output", str(tmp_path / "out.jsonl"),
            "--build-dir", str(bd),
        ])
        assert result.exit_code != 0

    def test_missing_graph(self, build_dir: Path, tmp_path: Path) -> None:
        # Remove graph from a valid build_dir copy
        import shutil

        bd = tmp_path / "bad_build"
        shutil.copytree(build_dir, bd)
        (bd / "graph.pkl").unlink()
        result = runner.invoke(app, [
            "generate",
            "--output", str(tmp_path / "out.jsonl"),
            "--build-dir", str(bd),
        ])
        assert result.exit_code != 0


class TestGenerateQuietMode:
    """Tests for --quiet flag."""

    def test_quiet_suppresses_output(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = runner.invoke(app, [
            "--quiet",
            "generate",
            "--output", str(out),
            "--build-dir", str(build_dir),
            "--count", "2",
            "--seed", "42",
        ])
        assert result.exit_code == 0
        # Quiet mode should suppress summary tables
        assert "Generation Summary" not in result.output


class TestGenerateJSONLOutput:
    """Verify the output JSONL matches the spec format."""

    def test_output_is_valid_jsonl(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        for line in out.read_text().strip().split("\n"):
            data = json.loads(line)  # should not raise
            assert isinstance(data, dict)

    def test_judge_scores_present(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        _run_generate(build_dir, out)
        records = _read_records(out)
        # At least some records should have judge scores
        with_scores = [r for r in records if r.get("judge_scores") is not None]
        assert len(with_scores) >= 1
