"""Integration tests for the ``tooluse build`` CLI command (Task 58)."""

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
# Mock embedding service — avoids downloading real model in tests
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
# Fixtures
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
                {"name": "days", "type": "NUMBER", "description": "Number of days", "default": "5"}
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
                    "description": "Maximum price per night",
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


@pytest.fixture()
def fixture_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with 3 ToolBench-format JSON files."""
    data_dir = tmp_path / "toolbench_data"
    data_dir.mkdir()
    for name, tool in [
        ("weather.json", _WEATHER_TOOL),
        ("hotels.json", _HOTEL_TOOL),
        ("flights.json", _FLIGHT_TOOL),
    ]:
        (data_dir / name).write_text(json.dumps(tool))
    return data_dir


@pytest.fixture(autouse=True)
def _mock_embeddings() -> object:
    """Patch EmbeddingService everywhere so tests never download a real model."""
    with (
        patch(
            "tooluse_gen.graph.builder.EmbeddingService",
            _MockEmbeddingService,
        ),
        patch(
            "tooluse_gen.graph.embeddings.EmbeddingService",
            _MockEmbeddingService,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_build(
    fixture_data_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> object:
    """Invoke ``tooluse build`` and return the CliRunner result."""
    args = [
        "build",
        "--input-dir", str(fixture_data_dir),
        "--output-dir", str(output_dir),
        "--force",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(app, args)


# ===================================================================
# Tests
# ===================================================================


class TestBuildCommand:
    """Core build command tests."""

    def test_build_creates_output_dir(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        result = _run_build(fixture_data_dir, out)
        assert result.exit_code == 0, result.output
        assert out.is_dir()

    def test_build_creates_registry(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)
        assert (out / "registry.json").exists()

    def test_build_creates_graph(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)
        assert (out / "graph.pkl").exists()

    def test_build_creates_embeddings(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)
        assert (out / "embeddings.joblib").exists()

    def test_build_registry_loadable(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)

        from tooluse_gen.registry.serialization import load_registry

        reg, meta = load_registry(out / "registry.json")
        assert len(reg) >= 1

    def test_build_graph_loadable(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)

        from tooluse_gen.graph.persistence import load_graph

        graph, meta = load_graph(out / "graph.pkl")
        assert graph.number_of_nodes() > 0

    def test_build_embeddings_loadable(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)

        from tooluse_gen.graph.persistence import load_embeddings

        emb = load_embeddings(out / "embeddings.joblib")
        assert len(emb) > 0

    def test_build_summary_output(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        result = _run_build(fixture_data_dir, out)
        assert result.exit_code == 0
        output = result.output
        # Summary should mention tools and endpoints
        assert "tool" in output.lower() or "Tools" in output

    def test_build_artifacts_have_content(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)
        for name in ["registry.json", "graph.pkl", "embeddings.joblib"]:
            p = out / name
            assert p.exists(), f"{name} missing"
            assert p.stat().st_size > 0, f"{name} is empty"


class TestBuildForceFlag:
    """Tests for --force behavior."""

    def test_force_overwrites(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)
        size1 = (out / "registry.json").stat().st_size

        # Second run with --force should succeed
        result = _run_build(fixture_data_dir, out)
        assert result.exit_code == 0
        size2 = (out / "registry.json").stat().st_size
        assert size2 > 0
        # Sizes should be roughly equal (same data)
        assert abs(size1 - size2) < size1 * 0.1

    def test_no_force_fails_if_exists(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)

        # Run again WITHOUT --force
        result = runner.invoke(app, [
            "build",
            "--input-dir", str(fixture_data_dir),
            "--output-dir", str(out),
        ])
        assert result.exit_code != 0


class TestBuildErrorHandling:
    """Tests for error conditions."""

    def test_invalid_input_dir(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "build",
            "--input-dir", str(tmp_path / "nonexistent"),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_empty_input_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(app, [
            "build",
            "--input-dir", str(empty_dir),
            "--output-dir", str(tmp_path / "out"),
            "--force",
        ])
        # Should fail because no tools pass quality filter
        assert result.exit_code != 0


class TestBuildGeneratePools:
    """Tests for --generate-pools flag."""

    def test_generate_pools_creates_file(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        result = _run_build(fixture_data_dir, out, ["--generate-pools"])
        assert result.exit_code == 0
        assert (out / "value_pools.json").exists()
        assert (out / "value_pools.json").stat().st_size > 0

    def test_no_pools_by_default(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)
        assert not (out / "value_pools.json").exists()


class TestBuildGraphStructure:
    """Verify built graph has the expected structure."""

    def test_graph_has_tool_and_endpoint_nodes(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)

        from tooluse_gen.graph.persistence import load_graph

        graph, _ = load_graph(out / "graph.pkl")
        tool_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "tool"
        ]
        ep_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"
        ]
        assert len(tool_nodes) >= 1
        assert len(ep_nodes) >= 1

    def test_graph_has_edges(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        _run_build(fixture_data_dir, out)

        from tooluse_gen.graph.persistence import load_graph

        graph, _ = load_graph(out / "graph.pkl")
        assert graph.number_of_edges() > 0


class TestBuildQuietMode:
    """Tests for --quiet flag."""

    def test_quiet_suppresses_output(
        self, fixture_data_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "build_out"
        result = runner.invoke(app, [
            "--quiet",
            "build",
            "--input-dir", str(fixture_data_dir),
            "--output-dir", str(out),
            "--force",
        ])
        assert result.exit_code == 0
        # Quiet mode should produce minimal or no output
        # (no Rich panels/tables)
        assert "Build Summary" not in result.output
