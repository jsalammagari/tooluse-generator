"""Deterministic disk-backed prompt cache.

:class:`PromptCache` stores LLM prompt/response pairs as human-readable
JSON files keyed by a SHA-256 hash of the serialised prompt.  Supports
optional TTL-based invalidation and a ``--no-cache`` disable flag.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tooluse_gen.utils.logging import get_logger

logger = get_logger("core.cache")


class PromptCache:
    """Deterministic disk-backed cache for LLM prompt/response pairs."""

    def __init__(
        self,
        cache_dir: Path | str,
        enabled: bool = True,
        ttl_seconds: float | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = enabled
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._logger = logger

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def hash_prompt(
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> str:
        """SHA-256 hash (first 16 hex chars) of the canonical prompt."""
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model,
            **kwargs,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Get / Put
    # ------------------------------------------------------------------

    def get(self, prompt_hash: str) -> dict[str, Any] | None:
        """Return the cached response, or ``None`` on miss."""
        if not self._enabled:
            return None

        path = self._cache_dir / f"{prompt_hash}.json"
        if not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            self._logger.warning("Corrupt cache file %s — deleting", path)
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

        # TTL check.
        if self._ttl is not None:
            cached_at = data.get("cached_at", "")
            try:
                ts = datetime.fromisoformat(cached_at)
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age > self._ttl:
                    path.unlink(missing_ok=True)
                    self._misses += 1
                    return None
            except (ValueError, TypeError):
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

        self._hits += 1
        return data.get("response")  # type: ignore[no-any-return]

    def put(self, prompt_hash: str, response: dict[str, Any]) -> None:
        """Write a response to the cache."""
        if not self._enabled:
            return

        path = self._cache_dir / f"{prompt_hash}.json"
        data = {
            "response": response,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, default=str))
        self._logger.debug("Cached %s", prompt_hash)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """Delete all cached entries and reset counters."""
        count = 0
        for f in self._cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        self._hits = 0
        self._misses = 0
        self._logger.info("Cleared %d cache entries", count)
        return count

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": self.size,
            "enabled": self._enabled,
            "cache_dir": str(self._cache_dir),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether caching is active."""
        return self._enabled

    @property
    def size(self) -> int:
        """Number of cached entries on disk."""
        return len(list(self._cache_dir.glob("*.json")))
