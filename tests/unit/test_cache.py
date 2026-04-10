"""Tests for PromptCache (Task 54)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tooluse_gen.core.cache import PromptCache

pytestmark = pytest.mark.unit


# ===================================================================
# hash_prompt
# ===================================================================


class TestHashPrompt:
    def test_returns_string(self):
        h = PromptCache.hash_prompt([{"role": "user", "content": "hi"}], model="gpt-4o")
        assert isinstance(h, str) and len(h) == 16

    def test_same_input_same_hash(self):
        msgs = [{"role": "user", "content": "hello"}]
        h1 = PromptCache.hash_prompt(msgs, model="gpt-4o")
        h2 = PromptCache.hash_prompt(msgs, model="gpt-4o")
        assert h1 == h2

    def test_different_messages(self):
        h1 = PromptCache.hash_prompt([{"role": "user", "content": "a"}], model="m")
        h2 = PromptCache.hash_prompt([{"role": "user", "content": "b"}], model="m")
        assert h1 != h2

    def test_different_model(self):
        msgs = [{"role": "user", "content": "x"}]
        h1 = PromptCache.hash_prompt(msgs, model="gpt-4o")
        h2 = PromptCache.hash_prompt(msgs, model="gpt-3.5")
        assert h1 != h2


# ===================================================================
# get / put — basic
# ===================================================================


class TestGetPutBasic:
    def test_put_then_get(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("abc", {"content": "hello"})
        assert c.get("abc") == {"content": "hello"}

    def test_get_miss(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        assert c.get("missing") is None

    def test_put_overwrites(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("k", {"v": 1})
        c.put("k", {"v": 2})
        assert c.get("k") == {"v": 2}

    def test_response_preserved(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        resp = {"content": "test", "tool_calls": [{"name": "f"}], "finish_reason": "stop"}
        c.put("h1", resp)
        assert c.get("h1") == resp

    def test_multiple_entries(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("a", {"v": "a"})
        c.put("b", {"v": "b"})
        assert c.get("a") == {"v": "a"}
        assert c.get("b") == {"v": "b"}

    def test_file_is_json(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("j", {"x": 1})
        data = json.loads((tmp_path / "j.json").read_text())
        assert "response" in data
        assert "cached_at" in data
        assert data["response"] == {"x": 1}


# ===================================================================
# Enabled / disabled
# ===================================================================


class TestDisabled:
    def test_get_returns_none(self, tmp_path: Path):
        c = PromptCache(tmp_path, enabled=False)
        c._cache_dir.joinpath("x.json").write_text('{"response":{},"cached_at":"t"}')
        assert c.get("x") is None

    def test_put_noop(self, tmp_path: Path):
        c = PromptCache(tmp_path, enabled=False)
        c.put("k", {"v": 1})
        assert c.size == 0

    def test_stats_disabled(self, tmp_path: Path):
        c = PromptCache(tmp_path, enabled=False)
        assert c.stats()["enabled"] is False


# ===================================================================
# TTL
# ===================================================================


class TestTTL:
    def test_within_ttl(self, tmp_path: Path):
        c = PromptCache(tmp_path, ttl_seconds=10.0)
        c.put("t", {"ok": True})
        assert c.get("t") == {"ok": True}

    def test_expired(self, tmp_path: Path):
        c = PromptCache(tmp_path, ttl_seconds=0.3)
        c.put("t", {"ok": True})
        time.sleep(0.4)
        assert c.get("t") is None

    def test_expired_deletes_file(self, tmp_path: Path):
        c = PromptCache(tmp_path, ttl_seconds=0.3)
        c.put("t", {"ok": True})
        time.sleep(0.4)
        c.get("t")
        assert not (tmp_path / "t.json").exists()

    def test_no_ttl_never_expires(self, tmp_path: Path):
        c = PromptCache(tmp_path, ttl_seconds=None)
        c.put("t", {"ok": True})
        # No sleep needed — no TTL means always valid
        assert c.get("t") == {"ok": True}


# ===================================================================
# clear
# ===================================================================


class TestClear:
    def test_removes_all(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("a", {"v": 1})
        c.put("b", {"v": 2})
        c.clear()
        assert c.size == 0

    def test_resets_stats(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("a", {})
        c.get("a")
        c.get("missing")
        c.clear()
        s = c.stats()
        assert s["hits"] == 0 and s["misses"] == 0

    def test_returns_count(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("a", {})
        c.put("b", {})
        assert c.clear() == 2


# ===================================================================
# stats
# ===================================================================


class TestStats:
    def test_initial_zeros(self, tmp_path: Path):
        s = PromptCache(tmp_path).stats()
        assert s["hits"] == 0 and s["misses"] == 0 and s["size"] == 0

    def test_tracks_hits_misses(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("k", {})
        c.get("k")      # hit
        c.get("miss")   # miss
        s = c.stats()
        assert s["hits"] == 1 and s["misses"] == 1

    def test_size(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("a", {})
        c.put("b", {})
        assert c.stats()["size"] == 2


# ===================================================================
# Disk persistence
# ===================================================================


class TestPersistence:
    def test_new_instance_sees_entries(self, tmp_path: Path):
        c1 = PromptCache(tmp_path)
        c1.put("persist", {"v": 42})
        c2 = PromptCache(tmp_path)
        assert c2.get("persist") == {"v": 42}

    def test_survives_across_instances(self, tmp_path: Path):
        PromptCache(tmp_path).put("k", {"x": 1})
        assert PromptCache(tmp_path).get("k") == {"x": 1}

    def test_corrupt_file(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("not json {{{")
        c = PromptCache(tmp_path)
        assert c.get("bad") is None
        assert not (tmp_path / "bad.json").exists()


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_messages(self):
        h = PromptCache.hash_prompt([], model="m")
        assert isinstance(h, str) and len(h) == 16

    def test_long_prompt(self):
        msgs = [{"role": "user", "content": "x" * 100000}]
        h = PromptCache.hash_prompt(msgs, model="m")
        assert isinstance(h, str) and len(h) == 16

    def test_creates_dir(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        c = PromptCache(nested)
        assert nested.exists()
        c.put("k", {})
        assert c.size == 1

    def test_put_same_key_twice(self, tmp_path: Path):
        c = PromptCache(tmp_path)
        c.put("k", {"v": 1})
        c.put("k", {"v": 2})
        assert c.get("k") == {"v": 2}
        assert c.size == 1
