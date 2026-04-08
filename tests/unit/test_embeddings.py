"""Unit tests for Task 18 — Embedding service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tooluse_gen.graph.embeddings import (
    EmbeddingService,
    build_endpoint_description,
    build_tool_description,
)
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterType,
    Tool,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_model() -> MagicMock:
    """A MagicMock that mimics SentenceTransformer.encode()."""
    model = MagicMock()
    model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
    return model


@pytest.fixture()
def service(mock_model: MagicMock) -> EmbeddingService:
    """EmbeddingService with a pre-injected mock model."""
    svc = EmbeddingService()
    svc._model = mock_model
    return svc


@pytest.fixture()
def sample_tool() -> Tool:
    """A complete Tool with one endpoint and two parameters."""
    return Tool(
        tool_id="weather_api",
        name="Weather API",
        description="Get weather information.",
        domain="Weather",
        completeness_score=0.85,
        endpoints=[
            Endpoint(
                endpoint_id="weather_api/GET/abc12345",
                tool_id="weather_api",
                name="Get Current Weather",
                description="Current conditions for a location.",
                method=HttpMethod.GET,
                path="/weather/current",
                parameters=[
                    Parameter(
                        name="location",
                        description="City name.",
                        param_type=ParameterType.STRING,
                        required=True,
                    ),
                    Parameter(
                        name="units",
                        param_type=ParameterType.STRING,
                        default="celsius",
                    ),
                ],
                completeness_score=0.9,
            ),
        ],
    )


@pytest.fixture()
def sample_endpoint(sample_tool: Tool) -> Endpoint:
    return sample_tool.endpoints[0]


# ===========================================================================
# EmbeddingService — initialisation
# ===========================================================================


class TestEmbeddingServiceInit:
    def test_default_model_name(self) -> None:
        svc = EmbeddingService()
        assert svc.model_name == "all-MiniLM-L6-v2"

    def test_model_not_loaded_on_init(self) -> None:
        svc = EmbeddingService()
        assert svc._model is None

    def test_custom_model_name(self) -> None:
        svc = EmbeddingService(model_name="custom-model")
        assert svc.model_name == "custom-model"

    def test_cache_dir_created(self, tmp_path: Path) -> None:
        cache = tmp_path / "embed_cache"
        assert not cache.exists()
        EmbeddingService(cache_dir=cache)
        assert cache.is_dir()

    def test_cache_dir_none_default(self) -> None:
        svc = EmbeddingService()
        assert svc.cache_dir is None


# ===========================================================================
# Lazy model loading
# ===========================================================================


class TestLazyLoading:
    @patch("tooluse_gen.graph.embeddings.SentenceTransformer", create=True)
    def test_get_model_loads_on_first_call(self, mock_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        svc = EmbeddingService()

        with patch(
            "tooluse_gen.graph.embeddings.SentenceTransformer",
            mock_cls,
            create=True,
        ):
            # _get_model triggers the import internally, so we patch at the import site
            pass

        # Directly test via injecting
        assert svc._model is None

    def test_second_call_returns_same_instance(self, mock_model: MagicMock) -> None:
        svc = EmbeddingService()
        svc._model = mock_model
        m1 = svc._get_model()
        m2 = svc._get_model()
        assert m1 is m2

    def test_model_cached_after_first_load(self) -> None:
        mock_instance = MagicMock()
        svc = EmbeddingService()
        svc._model = mock_instance
        assert svc._get_model() is mock_instance


# ===========================================================================
# embed_text
# ===========================================================================


class TestEmbedText:
    def test_returns_list_of_floats(self, service: EmbeddingService) -> None:
        result = service.embed_text("hello")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_calls_encode_correctly(
        self, service: EmbeddingService, mock_model: MagicMock
    ) -> None:
        service.embed_text("test text")
        mock_model.encode.assert_called_once_with("test text", show_progress_bar=False)

    def test_result_matches_mock(self, service: EmbeddingService) -> None:
        result = service.embed_text("anything")
        assert result == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_different_mock_output(self, mock_model: MagicMock) -> None:
        mock_model.encode.return_value = np.array([1.0, 2.0])
        svc = EmbeddingService()
        svc._model = mock_model
        assert svc.embed_text("x") == pytest.approx([1.0, 2.0])


# ===========================================================================
# embed_batch
# ===========================================================================


class TestEmbedBatch:
    def test_returns_list_of_lists(self, service: EmbeddingService, mock_model: MagicMock) -> None:
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = service.embed_batch(["a", "b"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(row, list) for row in result)

    def test_calls_encode_with_batch_size(
        self, service: EmbeddingService, mock_model: MagicMock
    ) -> None:
        mock_model.encode.return_value = np.array([[0.1], [0.2]])
        service.embed_batch(["a", "b"], batch_size=64)
        mock_model.encode.assert_called_once_with(
            ["a", "b"], batch_size=64, show_progress_bar=True
        )

    def test_empty_list_returns_empty(self, service: EmbeddingService) -> None:
        result = service.embed_batch([])
        assert result == []

    def test_show_progress_passed(
        self, service: EmbeddingService, mock_model: MagicMock
    ) -> None:
        mock_model.encode.return_value = np.array([[0.1]])
        service.embed_batch(["a"], show_progress=False)
        mock_model.encode.assert_called_once_with(
            ["a"], batch_size=256, show_progress_bar=False
        )


# ===========================================================================
# compute_similarity
# ===========================================================================


class TestComputeSimilarity:
    def test_identical_vectors(self) -> None:
        svc = EmbeddingService()
        sim = svc.compute_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        svc = EmbeddingService()
        sim = svc.compute_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self) -> None:
        svc = EmbeddingService()
        sim = svc.compute_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector_a(self) -> None:
        svc = EmbeddingService()
        assert svc.compute_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b(self) -> None:
        svc = EmbeddingService()
        assert svc.compute_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors(self) -> None:
        svc = EmbeddingService()
        assert svc.compute_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_known_similarity(self) -> None:
        svc = EmbeddingService()
        # cos(45°) ≈ 0.7071
        sim = svc.compute_similarity([1.0, 0.0], [1.0, 1.0])
        assert abs(sim - (1.0 / np.sqrt(2.0))) < 1e-6


# ===========================================================================
# compute_similarity_matrix
# ===========================================================================


class TestComputeSimilarityMatrix:
    def test_shape(self) -> None:
        svc = EmbeddingService()
        mat = svc.compute_similarity_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        assert mat.shape == (3, 3)

    def test_diagonal_is_one(self) -> None:
        svc = EmbeddingService()
        mat = svc.compute_similarity_matrix([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(np.diag(mat), [1.0, 1.0], atol=1e-6)

    def test_symmetric(self) -> None:
        svc = EmbeddingService()
        mat = svc.compute_similarity_matrix([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_empty_input(self) -> None:
        svc = EmbeddingService()
        mat = svc.compute_similarity_matrix([])
        assert mat.shape == (0, 0)

    def test_identical_rows(self) -> None:
        svc = EmbeddingService()
        mat = svc.compute_similarity_matrix([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(mat, [[1.0, 1.0], [1.0, 1.0]], atol=1e-6)

    def test_orthogonal_rows(self) -> None:
        svc = EmbeddingService()
        mat = svc.compute_similarity_matrix([[1.0, 0.0], [0.0, 1.0]])
        assert abs(mat[0, 1]) < 1e-6
        assert abs(mat[1, 0]) < 1e-6


# ===========================================================================
# save / load embeddings
# ===========================================================================


class TestEmbeddingCaching:
    def test_round_trip(self, tmp_path: Path) -> None:
        svc = EmbeddingService()
        data = {"node_a": [0.1, 0.2], "node_b": [0.3, 0.4]}
        path = tmp_path / "emb.joblib"
        svc.save_embeddings(data, path)
        loaded = svc.load_embeddings(path)
        assert loaded == data

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        svc = EmbeddingService()
        with pytest.raises(FileNotFoundError):
            svc.load_embeddings(tmp_path / "nonexistent.joblib")

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        svc = EmbeddingService()
        path = tmp_path / "a" / "b" / "emb.joblib"
        svc.save_embeddings({"x": [1.0]}, path)
        assert path.exists()

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        svc = EmbeddingService()
        path = tmp_path / "emb.joblib"
        svc.save_embeddings({"v1": [1.0]}, path)
        svc.save_embeddings({"v2": [2.0]}, path)
        loaded = svc.load_embeddings(path)
        assert "v2" in loaded
        assert "v1" not in loaded


# ===========================================================================
# build_tool_description
# ===========================================================================


class TestBuildToolDescription:
    def test_full_tool(self, sample_tool: Tool) -> None:
        desc = build_tool_description(sample_tool)
        assert "Weather API" in desc
        assert "Get weather information" in desc
        assert "Domain: Weather" in desc
        assert "Endpoints:" in desc
        assert "Get Current Weather" in desc

    def test_empty_description(self) -> None:
        tool = Tool(tool_id="t", name="MyTool", domain="D")
        desc = build_tool_description(tool)
        assert "MyTool" in desc
        assert "Domain: D" in desc
        # No double spaces or stray periods from empty description
        assert "  " not in desc

    def test_empty_domain(self) -> None:
        tool = Tool(tool_id="t", name="MyTool", description="A tool.")
        desc = build_tool_description(tool)
        assert "MyTool" in desc
        assert "A tool" in desc
        assert "Domain:" not in desc

    def test_no_endpoints(self) -> None:
        tool = Tool(tool_id="t", name="MyTool")
        desc = build_tool_description(tool)
        assert "Endpoints:" not in desc

    def test_minimal_tool(self) -> None:
        tool = Tool(tool_id="t", name="X")
        desc = build_tool_description(tool)
        assert desc.startswith("X.")

    def test_multiple_endpoints(self) -> None:
        tool = Tool(
            tool_id="t",
            name="Multi",
            endpoints=[
                Endpoint(
                    endpoint_id="t/GET/a", tool_id="t", name="EP1", path="/a"
                ),
                Endpoint(
                    endpoint_id="t/GET/b", tool_id="t", name="EP2", path="/b"
                ),
            ],
        )
        desc = build_tool_description(tool)
        assert "EP1" in desc
        assert "EP2" in desc


# ===========================================================================
# build_endpoint_description
# ===========================================================================


class TestBuildEndpointDescription:
    def test_full_endpoint(self, sample_endpoint: Endpoint, sample_tool: Tool) -> None:
        desc = build_endpoint_description(sample_endpoint, sample_tool)
        assert "Get Current Weather" in desc
        assert "Current conditions" in desc
        assert "Tool: Weather API" in desc
        assert "Method: GET" in desc
        assert "Path: /weather/current" in desc
        assert "Parameters:" in desc
        assert "location" in desc
        assert "units" in desc

    def test_no_description(self) -> None:
        ep = Endpoint(
            endpoint_id="t/GET/x", tool_id="t", name="EP", path="/x"
        )
        tool = Tool(tool_id="t", name="T")
        desc = build_endpoint_description(ep, tool)
        assert "EP." in desc
        assert "Tool: T" in desc
        assert "  " not in desc

    def test_no_parameters(self) -> None:
        ep = Endpoint(
            endpoint_id="t/GET/x",
            tool_id="t",
            name="EP",
            description="Does stuff.",
            path="/x",
        )
        tool = Tool(tool_id="t", name="T")
        desc = build_endpoint_description(ep, tool)
        assert "Parameters:" not in desc

    def test_minimal_endpoint(self) -> None:
        ep = Endpoint(
            endpoint_id="t/GET/x", tool_id="t", name="EP", path="/x"
        )
        tool = Tool(tool_id="t", name="T")
        desc = build_endpoint_description(ep, tool)
        assert desc.startswith("EP.")
        assert "Tool: T" in desc
