"""Tests for test infrastructure (Task 64)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ===================================================================
# Marker recognition
# ===================================================================


class TestMarkers:
    def test_unit_marker_exists(self) -> None:
        """This test is decorated with @pytest.mark.unit — running proves it works."""

    @pytest.mark.integration
    def test_integration_marker_recognized(self) -> None:
        """pytest does not warn about the 'integration' marker."""

    @pytest.mark.e2e
    def test_e2e_marker_recognized(self) -> None:
        """pytest does not warn about the 'e2e' marker.

        This test will be *skipped* unless ``--run-e2e`` is passed, which
        itself proves that the marker and the skip mechanism both work.
        """


# ===================================================================
# MockEmbeddingService (from root conftest)
# ===================================================================


class TestMockEmbeddingService:
    def test_embed_text_returns_list(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        result = svc.embed_text("hello world")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_text_deterministic(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        a = svc.embed_text("test")
        b = svc.embed_text("test")
        assert a == b

    def test_embed_text_different_inputs(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        a = svc.embed_text("hello")
        b = svc.embed_text("world")
        assert a != b

    def test_embed_batch(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        results = svc.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 384 for v in results)

    def test_compute_similarity_self(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        a = svc.embed_text("hello")
        assert abs(svc.compute_similarity(a, a) - 1.0) < 1e-6

    def test_compute_similarity_different(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        a = svc.embed_text("alpha")
        b = svc.embed_text("beta")
        sim = svc.compute_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_compute_similarity_matrix(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        vecs = svc.embed_batch(["a", "b"])
        mat = svc.compute_similarity_matrix(vecs)
        assert mat.shape == (2, 2)

    def test_compute_similarity_matrix_empty(self, mock_embedding_service: type) -> None:
        svc = mock_embedding_service()
        mat = svc.compute_similarity_matrix([])
        assert mat.shape == (0, 0)


# ===================================================================
# ToolBench subset fixtures
# ===================================================================


class TestToolbenchSubset:
    def test_subset_fixture_returns_path_or_none(
        self, toolbench_subset: Path | None
    ) -> None:
        assert toolbench_subset is None or isinstance(toolbench_subset, Path)

    def test_subset_has_json_files(self, toolbench_subset: Path | None) -> None:
        if toolbench_subset is None:
            pytest.skip("ToolBench data not available")
        files = list(toolbench_subset.rglob("*.json"))
        assert len(files) >= 10

    def test_subset_has_multiple_categories(
        self, toolbench_subset: Path | None
    ) -> None:
        if toolbench_subset is None:
            pytest.skip("ToolBench data not available")
        categories = [d.name for d in toolbench_subset.iterdir() if d.is_dir()]
        assert len(categories) >= 2

    def test_data_dir_always_returns_path(self, toolbench_data_dir: Path) -> None:
        """toolbench_data_dir always returns a usable Path (with fallback)."""
        assert isinstance(toolbench_data_dir, Path)
        assert toolbench_data_dir.is_dir()
        # Must contain at least one JSON file
        json_files = list(toolbench_data_dir.rglob("*.json"))
        assert len(json_files) >= 1


# ===================================================================
# Build artifacts fixture
# ===================================================================


class TestBuildArtifacts:
    def test_build_artifacts_has_registry(self, build_artifacts: Path) -> None:
        assert (build_artifacts / "registry.json").exists()
        assert (build_artifacts / "registry.json").stat().st_size > 0

    def test_build_artifacts_has_graph(self, build_artifacts: Path) -> None:
        assert (build_artifacts / "graph.pkl").exists()
        assert (build_artifacts / "graph.pkl").stat().st_size > 0

    def test_build_artifacts_has_embeddings(self, build_artifacts: Path) -> None:
        assert (build_artifacts / "embeddings.joblib").exists()
        assert (build_artifacts / "embeddings.joblib").stat().st_size > 0


# ===================================================================
# E2E skip behaviour
# ===================================================================


class TestE2ESkipBehavior:
    @pytest.mark.e2e
    def test_e2e_skipped_without_flag(self) -> None:
        """This should be skipped unless --run-e2e is passed."""
