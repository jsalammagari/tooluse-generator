"""CLI integration tests — cross-command pipeline, flag combos, error cases (Task 63)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from tooluse_gen.cli.main import app

pytestmark = pytest.mark.integration

runner = CliRunner()

# ---------------------------------------------------------------------------
# Mock embedding service
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
            "url": "/current",
            "description": "Get current weather for a city",
            "method": "GET",
            "required_parameters": [
                {"name": "city", "type": "STRING", "description": "City name", "default": ""}
            ],
            "optional_parameters": [
                {"name": "units", "type": "STRING", "description": "Units", "default": "metric"}
            ],
        },
        {
            "name": "forecast",
            "url": "/forecast",
            "description": "Get weather forecast",
            "method": "GET",
            "required_parameters": [
                {"name": "city", "type": "STRING", "description": "City name", "default": ""}
            ],
            "optional_parameters": [],
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
            "url": "/search",
            "description": "Search for hotels in a city",
            "method": "GET",
            "required_parameters": [
                {"name": "city", "type": "STRING", "description": "City", "default": ""},
                {"name": "check_in", "type": "STRING", "description": "Check-in date", "default": ""},
            ],
            "optional_parameters": [
                {"name": "max_price", "type": "NUMBER", "description": "Max price", "default": ""}
            ],
        },
        {
            "name": "book_hotel",
            "url": "/book",
            "description": "Book a hotel room",
            "method": "POST",
            "required_parameters": [
                {"name": "hotel_id", "type": "STRING", "description": "Hotel ID", "default": ""},
                {"name": "guest_name", "type": "STRING", "description": "Guest name", "default": ""},
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
            "url": "/search",
            "description": "Search flights between airports",
            "method": "GET",
            "required_parameters": [
                {"name": "origin", "type": "STRING", "description": "Origin", "default": ""},
                {"name": "destination", "type": "STRING", "description": "Dest", "default": ""},
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
    with (
        patch("tooluse_gen.graph.builder.EmbeddingService", _MockEmbeddingService),
        patch("tooluse_gen.graph.embeddings.EmbeddingService", _MockEmbeddingService),
    ):
        yield


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Write fixture ToolBench JSON files."""
    d = tmp_path / "data"
    d.mkdir()
    for name, tool in [
        ("weather.json", _WEATHER_TOOL),
        ("hotels.json", _HOTEL_TOOL),
        ("flights.json", _FLIGHT_TOOL),
    ]:
        (d / name).write_text(json.dumps(tool))
    return d


@pytest.fixture()
def build_dir(data_dir: Path, tmp_path: Path) -> Path:
    """Run build and return the artifacts directory."""
    out = tmp_path / "build"
    result = runner.invoke(app, [
        "build", "--input-dir", str(data_dir), "--output-dir", str(out),
        "--force", "--similarity-threshold", "0.1",
    ])
    assert result.exit_code == 0, f"Build fixture failed:\n{result.output}"
    return out


def _run_generate(
    build_dir: Path, output: Path, extra: list[str] | None = None
) -> object:
    args = [
        "generate", "--output", str(output), "--build-dir", str(build_dir),
        "--count", "3", "--seed", "42", "--min-steps", "1", "--max-steps", "3",
    ]
    if extra:
        args.extend(extra)
    return runner.invoke(app, args)


@pytest.fixture()
def generated_jsonl(build_dir: Path, tmp_path: Path) -> Path:
    """Run generate and return the output JSONL path."""
    out = tmp_path / "conversations.jsonl"
    result = _run_generate(build_dir, out)
    assert result.exit_code == 0, f"Generate fixture failed:\n{result.output}"
    return out


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        if "conversation_id" in data:
            records.append(data)
    return records


# ===================================================================
# Full pipeline
# ===================================================================


class TestFullPipeline:
    """End-to-end: build -> generate -> evaluate."""

    def test_build_then_generate(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(build_dir, out)
        assert result.exit_code == 0
        assert out.exists()
        assert len(_read_records(out)) >= 1

    def test_build_generate_evaluate(self, generated_jsonl: Path) -> None:
        result = runner.invoke(app, ["evaluate", str(generated_jsonl)])
        assert result.exit_code == 0

    def test_pipeline_output_matches_spec(self, generated_jsonl: Path) -> None:
        for rec in _read_records(generated_jsonl):
            assert "conversation_id" in rec
            assert "messages" in rec
            assert isinstance(rec["messages"], list)
            assert "metadata" in rec

    def test_pipeline_messages_have_tool_calls(
        self, generated_jsonl: Path
    ) -> None:
        records = _read_records(generated_jsonl)
        has_tc = any(
            any(m.get("tool_calls") for m in rec["messages"])
            for rec in records
        )
        assert has_tc, "No conversations have tool calls"

    def test_pipeline_deterministic_with_seed(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out1 = tmp_path / "a.jsonl"
        out2 = tmp_path / "b.jsonl"
        _run_generate(build_dir, out1)
        _run_generate(build_dir, out2)
        assert len(_read_records(out1)) == len(_read_records(out2))

    def test_evaluate_generated_output(
        self, generated_jsonl: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, [
            "evaluate", str(generated_jsonl), "--format", "json",
        ])
        assert result.exit_code == 0
        assert "total" in result.output

    def test_evaluate_enriched_round_trip(
        self, generated_jsonl: Path, tmp_path: Path
    ) -> None:
        enriched = tmp_path / "enriched.jsonl"
        r1 = runner.invoke(app, [
            "evaluate", str(generated_jsonl), "--output", str(enriched), "--rescore",
        ])
        assert r1.exit_code == 0
        assert enriched.exists()

        # Re-evaluate the enriched file
        r2 = runner.invoke(app, ["evaluate", str(enriched)])
        assert r2.exit_code == 0


# ===================================================================
# Flag combinations
# ===================================================================


class TestFlagCombinations:
    """Test that CLI flags combine correctly."""

    def test_generate_no_steering(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(
            build_dir, out, ["--no-cross-conversation-steering"]
        )
        assert result.exit_code == 0
        assert len(_read_records(out)) >= 1

    def test_generate_no_cache(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(build_dir, out, ["--no-cache"])
        assert result.exit_code == 0
        assert "disabled" in result.output.lower()

    def test_generate_no_steering_no_cache(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_generate(
            build_dir, out,
            ["--no-cross-conversation-steering", "--no-cache"],
        )
        assert result.exit_code == 0
        assert len(_read_records(out)) >= 1

    def test_generate_custom_steps(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        result = runner.invoke(app, [
            "generate", "--output", str(out), "--build-dir", str(build_dir),
            "--count", "2", "--seed", "42",
            "--min-steps", "1", "--max-steps", "2",
        ])
        assert result.exit_code == 0

    def test_generate_custom_seed(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        out_a = tmp_path / "a.jsonl"
        out_b = tmp_path / "b.jsonl"
        _run_generate(build_dir, out_a)
        # Different seed
        result = runner.invoke(app, [
            "generate", "--output", str(out_b), "--build-dir", str(build_dir),
            "--count", "3", "--seed", "99", "--min-steps", "1", "--max-steps", "3",
        ])
        assert result.exit_code == 0
        # Both should produce records (content may differ)
        assert len(_read_records(out_a)) >= 1
        assert len(_read_records(out_b)) >= 1

    def test_evaluate_json_format(self, generated_jsonl: Path) -> None:
        result = runner.invoke(app, [
            "evaluate", str(generated_jsonl), "--format", "json",
        ])
        assert result.exit_code == 0
        assert "total" in result.output
        assert "pass_rate" in result.output
        assert "diversity" in result.output

    def test_evaluate_markdown_format(self, generated_jsonl: Path) -> None:
        result = runner.invoke(app, [
            "evaluate", str(generated_jsonl), "--format", "markdown",
        ])
        assert result.exit_code == 0
        assert "# Evaluation Report" in result.output
        assert "| Metric | Value |" in result.output

    def test_evaluate_rescore_flag(self, generated_jsonl: Path) -> None:
        result = runner.invoke(app, [
            "evaluate", str(generated_jsonl), "--rescore",
        ])
        assert result.exit_code == 0
        lower = result.output.lower()
        assert "rescored" in lower or "scored" in lower

    def test_build_with_generate_pools(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_pools"
        result = runner.invoke(app, [
            "build", "--input-dir", str(data_dir), "--output-dir", str(out),
            "--force", "--similarity-threshold", "0.1", "--generate-pools",
        ])
        assert result.exit_code == 0
        assert (out / "value_pools.json").exists()

    def test_quiet_flag_all_commands(
        self, data_dir: Path, build_dir: Path,
        generated_jsonl: Path, tmp_path: Path,
    ) -> None:
        # build
        out_b = tmp_path / "qbuild"
        r = runner.invoke(app, [
            "--quiet", "build", "--input-dir", str(data_dir),
            "--output-dir", str(out_b), "--force", "--similarity-threshold", "0.1",
        ])
        assert r.exit_code == 0
        assert "Build Summary" not in r.output

        # generate
        out_g = tmp_path / "qgen.jsonl"
        r = runner.invoke(app, [
            "--quiet", "generate", "--output", str(out_g),
            "--build-dir", str(build_dir),
            "--count", "2", "--seed", "42", "--min-steps", "1", "--max-steps", "3",
        ])
        assert r.exit_code == 0
        assert "Generation Summary" not in r.output

        # evaluate
        r = runner.invoke(app, [
            "--quiet", "evaluate", str(generated_jsonl),
        ])
        assert r.exit_code == 0
        assert "Evaluation Report" not in r.output

    def test_verbose_flag(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "vbuild"
        result = runner.invoke(app, [
            "-v", "build", "--input-dir", str(data_dir),
            "--output-dir", str(out), "--force", "--similarity-threshold", "0.1",
        ])
        assert result.exit_code == 0

    def test_config_flag(self, data_dir: Path, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("seed: 99\nverbose: 0\n")
        out = tmp_path / "cbuild"
        result = runner.invoke(app, [
            "--config", str(cfg), "build", "--input-dir", str(data_dir),
            "--output-dir", str(out), "--force", "--similarity-threshold", "0.1",
        ])
        assert result.exit_code == 0

    def test_config_from_flag(
        self, generated_jsonl: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, [
            "--config-from", str(generated_jsonl),
            "evaluate", str(generated_jsonl),
        ])
        assert result.exit_code == 0


# ===================================================================
# Error cases
# ===================================================================


class TestErrorCases:
    """Error handling across commands."""

    def test_build_missing_input(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "build", "--input-dir", str(tmp_path / "nope"),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_generate_missing_build_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "generate", "--output", str(tmp_path / "out.jsonl"),
            "--build-dir", str(tmp_path / "nope"),
        ])
        assert result.exit_code != 0

    def test_generate_missing_registry(self, tmp_path: Path) -> None:
        bd = tmp_path / "bad_build"
        bd.mkdir()
        (bd / "graph.pkl").write_bytes(b"fake")
        result = runner.invoke(app, [
            "generate", "--output", str(tmp_path / "out.jsonl"),
            "--build-dir", str(bd),
        ])
        assert result.exit_code != 0

    def test_generate_missing_graph(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        bd = tmp_path / "no_graph"
        shutil.copytree(build_dir, bd)
        (bd / "graph.pkl").unlink()
        result = runner.invoke(app, [
            "generate", "--output", str(tmp_path / "out.jsonl"),
            "--build-dir", str(bd),
        ])
        assert result.exit_code != 0

    def test_evaluate_missing_input(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "evaluate", str(tmp_path / "nope.jsonl"),
        ])
        assert result.exit_code != 0

    def test_evaluate_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(app, ["evaluate", str(empty)])
        assert result.exit_code != 0

    def test_generate_min_gt_max_steps(
        self, build_dir: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, [
            "generate", "--output", str(tmp_path / "out.jsonl"),
            "--build-dir", str(build_dir),
            "--min-steps", "5", "--max-steps", "2",
        ])
        assert result.exit_code == 1

    def test_evaluate_invalid_format(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "evaluate", str(tmp_path / "f.jsonl"), "--format", "xml",
        ])
        assert result.exit_code == 1

    def test_build_no_force_existing(
        self, data_dir: Path, build_dir: Path,
    ) -> None:
        # build_dir already has artifacts — re-running without --force fails
        result = runner.invoke(app, [
            "build", "--input-dir", str(data_dir),
            "--output-dir", str(build_dir),
        ])
        assert result.exit_code != 0
