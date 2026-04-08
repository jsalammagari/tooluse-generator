"""Embedding service for tool graph construction.

Wraps ``sentence-transformers`` with lazy model loading, disk caching
via ``joblib``, and cosine-similarity helpers.  Two free functions
build the description text fed into the embedding model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt

from tooluse_gen.registry.models import Endpoint, Tool
from tooluse_gen.utils.logging import get_logger

logger = get_logger("graph.embeddings")

# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Manages text embedding using sentence-transformers with disk caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None  # SentenceTransformer, lazy-loaded

        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # -- lazy loading -------------------------------------------------------

    def _get_model(self) -> Any:
        """Return the sentence-transformers model, loading it on first call."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    # -- embedding ----------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string and return a list of floats."""
        model = self._get_model()
        vec: npt.NDArray[np.floating[Any]] = model.encode(text, show_progress_bar=False)
        return vec.tolist()  # type: ignore[no-any-return]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Embed a list of texts and return a list of float-lists."""
        if not texts:
            return []
        logger.info("Embedding %d texts with batch_size=%d", len(texts), batch_size)
        model = self._get_model()
        vecs: npt.NDArray[np.floating[Any]] = model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress
        )
        return vecs.tolist()  # type: ignore[no-any-return]

    # -- similarity ---------------------------------------------------------

    def compute_similarity(self, emb_a: list[float], emb_b: list[float]) -> float:
        """Cosine similarity between two vectors. Returns 0.0 for zero-norm."""
        a = np.asarray(emb_a, dtype=np.float64)
        b = np.asarray(emb_b, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_similarity_matrix(
        self, embeddings: list[list[float]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Pairwise cosine similarity matrix for *embeddings*."""
        if not embeddings:
            return np.empty((0, 0), dtype=np.float64)
        mat = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = mat / norms
        return normed @ normed.T  # type: ignore[no-any-return]

    # -- caching ------------------------------------------------------------

    def save_embeddings(
        self, embeddings: dict[str, list[float]], output_path: Path
    ) -> None:
        """Persist embeddings to disk via joblib."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving embeddings to %s", output_path)
        joblib.dump(embeddings, output_path)

    def load_embeddings(self, input_path: Path) -> dict[str, list[float]]:
        """Load embeddings from disk. Raises ``FileNotFoundError`` if missing."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Embedding cache not found: {input_path}")
        logger.info("Loading embeddings from %s", input_path)
        return joblib.load(input_path)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Description builders
# ---------------------------------------------------------------------------

_MULTI_SPACE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    """Strip and collapse whitespace."""
    return _MULTI_SPACE.sub(" ", text).strip()


def build_tool_description(tool: Tool) -> str:
    """Build a single text string suitable for embedding a :class:`Tool`."""
    parts: list[str] = [tool.name + "."]
    if tool.description:
        parts.append(tool.description)
    if tool.domain:
        parts.append(f"Domain: {tool.domain}.")
    if tool.endpoints:
        ep_names = ", ".join(ep.name for ep in tool.endpoints)
        parts.append(f"Endpoints: {ep_names}.")
    return _collapse(" ".join(parts))


def build_endpoint_description(endpoint: Endpoint, tool: Tool) -> str:
    """Build a single text string suitable for embedding an :class:`Endpoint`."""
    parts: list[str] = [endpoint.name + "."]
    if endpoint.description:
        parts.append(endpoint.description)
    parts.append(f"Tool: {tool.name}.")
    method = endpoint.method if isinstance(endpoint.method, str) else endpoint.method
    parts.append(f"Method: {method}.")
    parts.append(f"Path: {endpoint.path}.")
    if endpoint.parameters:
        param_names = ", ".join(p.name for p in endpoint.parameters)
        parts.append(f"Parameters: {param_names}.")
    return _collapse(" ".join(parts))
