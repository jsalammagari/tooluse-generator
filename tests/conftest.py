"""Root-level test configuration — shared fixtures, markers, and E2E gating.

Provides:

* ``--run-e2e`` CLI flag so ``@pytest.mark.e2e`` tests are skipped by default.
* :class:`MockEmbeddingService` — deterministic hash-based embeddings.
* ``toolbench_subset`` / ``toolbench_data_dir`` — real or synthetic tool data.
* ``build_artifacts`` — pre-built registry + graph for integration tests.

.. note::

   pytest-xdist is not yet a dependency.  Adding it and running with
   ``-n auto`` is a future improvement for parallelising E2E tests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# --run-e2e flag
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-e2e`` CLI option."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests (skipped by default).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.e2e`` tests unless ``--run-e2e`` is passed."""
    if config.getoption("--run-e2e"):
        return
    skip_e2e = pytest.mark.skip(reason="Need --run-e2e option to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


# ---------------------------------------------------------------------------
# MockEmbeddingService
# ---------------------------------------------------------------------------


class MockEmbeddingService:
    """Deterministic hash-based embeddings for testing without real models."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._model = None
        self._cache_dir = None
        self.model_name = "mock"

    def _get_model(self) -> object:
        return None

    def embed_text(self, text: str) -> list[float]:
        import numpy as np

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
        import numpy as np

        a = np.asarray(emb_a, dtype=np.float64)
        b = np.asarray(emb_b, dtype=np.float64)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return 0.0 if na == 0.0 or nb == 0.0 else float(np.dot(a, b) / (na * nb))

    def compute_similarity_matrix(self, embeddings: list[list[float]]) -> Any:
        import numpy as np

        if not embeddings:
            return np.empty((0, 0), dtype=np.float64)
        mat = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = mat / norms
        return normed @ normed.T


# ---------------------------------------------------------------------------
# Fixtures — MockEmbeddingService
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_embedding_service() -> type:
    """Return the :class:`MockEmbeddingService` class for patching."""
    return MockEmbeddingService


# ---------------------------------------------------------------------------
# Fixtures — ToolBench data subset
# ---------------------------------------------------------------------------

_TOOLBENCH_ROOT = Path(__file__).resolve().parent.parent / "data" / "toolenv" / "tools"
_SUBSET_CATEGORIES = ["Finance", "Food", "Weather", "Travel", "Sports"]
_TOOLS_PER_CATEGORY = 10

# Minimal synthetic tools used when real data is unavailable.
_FALLBACK_TOOLS = [
    {
        "tool_name": "Weather API",
        "tool_description": "Get weather forecasts",
        "home_url": "https://api.weather.example.com",
        "api_list": [
            {
                "name": "current_weather",
                "url": "/current",
                "description": "Get current weather for a city",
                "method": "GET",
                "required_parameters": [
                    {"name": "city", "type": "STRING", "description": "City", "default": ""}
                ],
                "optional_parameters": [],
            },
        ],
    },
    {
        "tool_name": "Hotel Booking API",
        "tool_description": "Search and book hotels",
        "home_url": "https://api.hotels.example.com",
        "api_list": [
            {
                "name": "search_hotels",
                "url": "/search",
                "description": "Search hotels",
                "method": "GET",
                "required_parameters": [
                    {"name": "city", "type": "STRING", "description": "City", "default": ""}
                ],
                "optional_parameters": [],
            },
            {
                "name": "book_hotel",
                "url": "/book",
                "description": "Book a hotel",
                "method": "POST",
                "required_parameters": [
                    {"name": "hotel_id", "type": "STRING", "description": "ID", "default": ""}
                ],
                "optional_parameters": [],
            },
        ],
    },
    {
        "tool_name": "Flight Search API",
        "tool_description": "Search for flights",
        "home_url": "https://api.flights.example.com",
        "api_list": [
            {
                "name": "search_flights",
                "url": "/search",
                "description": "Search flights",
                "method": "GET",
                "required_parameters": [
                    {"name": "origin", "type": "STRING", "description": "Origin", "default": ""},
                    {"name": "dest", "type": "STRING", "description": "Dest", "default": ""},
                ],
                "optional_parameters": [],
            },
        ],
    },
]


@pytest.fixture(scope="session")
def toolbench_subset(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Copy ~50 real ToolBench tools from diverse categories to a temp dir.

    Returns ``None`` when the ToolBench data checkout is absent (e.g. in CI).
    """
    if not _TOOLBENCH_ROOT.is_dir():
        return None

    dest = tmp_path_factory.mktemp("toolbench_subset")
    total = 0
    for category in _SUBSET_CATEGORIES:
        cat_dir = _TOOLBENCH_ROOT / category
        if not cat_dir.is_dir():
            continue
        json_files = sorted(cat_dir.glob("*.json"))[:_TOOLS_PER_CATEGORY]
        if not json_files:
            continue
        cat_dest = dest / category
        cat_dest.mkdir(parents=True, exist_ok=True)
        for f in json_files:
            shutil.copy2(f, cat_dest / f.name)
            total += 1

    return dest if total > 0 else None


@pytest.fixture()
def toolbench_data_dir(toolbench_subset: Path | None, tmp_path: Path) -> Path:
    """Return a test-local directory with ToolBench tool JSON files.

    Uses a copy of the real-data subset when available, otherwise creates
    minimal synthetic tools so tests never fail due to missing data.
    """
    if toolbench_subset is not None:
        dest = tmp_path / "toolbench"
        shutil.copytree(toolbench_subset, dest)
        return dest

    # Fallback: write synthetic tool files
    dest = tmp_path / "toolbench"
    dest.mkdir()
    for i, tool in enumerate(_FALLBACK_TOOLS):
        (dest / f"tool_{i}.json").write_text(json.dumps(tool))
    return dest


# ---------------------------------------------------------------------------
# Fixtures — build artifacts
# ---------------------------------------------------------------------------


@pytest.fixture()
def build_artifacts(
    toolbench_data_dir: Path,
    tmp_path: Path,
    mock_embedding_service: type,
) -> Path:
    """Build registry + graph from ToolBench data and return the output dir."""
    from unittest.mock import patch

    from typer.testing import CliRunner

    from tooluse_gen.cli.main import app

    out = tmp_path / "build"
    cli_runner = CliRunner()
    with (
        patch("tooluse_gen.graph.builder.EmbeddingService", mock_embedding_service),
        patch("tooluse_gen.graph.embeddings.EmbeddingService", mock_embedding_service),
    ):
        result = cli_runner.invoke(app, [
            "build",
            "--input-dir", str(toolbench_data_dir),
            "--output-dir", str(out),
            "--force",
            "--similarity-threshold", "0.1",
        ])
    assert result.exit_code == 0, f"Build failed:\n{result.output}"
    return out
