"""Tests for LLMRecorder (Task 56)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.fake_llm import FakeLLM, FakeLLMResponse
from tests.helpers.vcr_recorder import CassetteNotFoundError, LLMRecorder

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MSGS = [{"role": "user", "content": "hello"}]
_MODEL = "gpt-4o"


def _record_one(tmp_path: Path, content: str = "recorded") -> None:
    """Record a single cassette."""
    fake = FakeLLM(responses=[FakeLLMResponse(content=content)])
    rec = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
    rec.chat_completion(_MSGS, model=_MODEL)


# ===================================================================
# CassetteNotFoundError
# ===================================================================


class TestCassetteNotFoundError:
    def test_message(self):
        e = CassetteNotFoundError("missing abc")
        assert "missing abc" in str(e)

    def test_raised(self, tmp_path: Path):
        r = LLMRecorder(tmp_path, replay=True, record=False)
        with pytest.raises(CassetteNotFoundError):
            r.chat_completion(_MSGS, model=_MODEL)


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_default_replay(self, tmp_path: Path):
        r = LLMRecorder(tmp_path)
        assert r.is_replaying and not r.is_recording

    def test_record_mode(self, tmp_path: Path):
        r = LLMRecorder(tmp_path, record=True, replay=False)
        assert r.is_recording and not r.is_replaying

    def test_both(self, tmp_path: Path):
        r = LLMRecorder(tmp_path, record=True, replay=True)
        assert r.is_recording and r.is_replaying


# ===================================================================
# Replay mode
# ===================================================================


class TestReplay:
    def test_returns_saved(self, tmp_path: Path):
        _record_one(tmp_path, "saved")
        r = LLMRecorder(tmp_path, replay=True)
        result = r.chat_completion(_MSGS, model=_MODEL)
        assert result["content"] == "saved"

    def test_not_found_raises(self, tmp_path: Path):
        r = LLMRecorder(tmp_path, replay=True, record=False)
        with pytest.raises(CassetteNotFoundError):
            r.chat_completion([{"role": "user", "content": "unknown"}], model="m")

    def test_multiple_coexist(self, tmp_path: Path):
        fake = FakeLLM(responses=[
            FakeLLMResponse(content="a"),
            FakeLLMResponse(content="b"),
        ])
        rec = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        rec.chat_completion([{"role": "user", "content": "msg_a"}], model="m")
        rec.chat_completion([{"role": "user", "content": "msg_b"}], model="m")

        replay = LLMRecorder(tmp_path, replay=True)
        ra = replay.chat_completion([{"role": "user", "content": "msg_a"}], model="m")
        rb = replay.chat_completion([{"role": "user", "content": "msg_b"}], model="m")
        assert ra["content"] == "a"
        assert rb["content"] == "b"

    def test_response_matches(self, tmp_path: Path):
        _record_one(tmp_path, "exact")
        r = LLMRecorder(tmp_path, replay=True)
        result = r.chat_completion(_MSGS, model=_MODEL)
        assert result == {"content": "exact", "tool_calls": None, "finish_reason": "stop",
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    def test_replay_count(self, tmp_path: Path):
        _record_one(tmp_path)
        r = LLMRecorder(tmp_path, replay=True)
        r.chat_completion(_MSGS, model=_MODEL)
        r.chat_completion(_MSGS, model=_MODEL)
        assert r.replay_count == 2


# ===================================================================
# Record mode
# ===================================================================


class TestRecord:
    def test_calls_client(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="from client")])
        rec = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        result = rec.chat_completion(_MSGS, model=_MODEL)
        assert result["content"] == "from client"
        assert fake.call_count == 1

    def test_cassette_file_exists(self, tmp_path: Path):
        _record_one(tmp_path)
        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_cassette_valid_json(self, tmp_path: Path):
        _record_one(tmp_path)
        f = next(tmp_path.glob("*.json"))
        data = json.loads(f.read_text())
        assert "response" in data
        assert "recorded_at" in data

    def test_returns_response(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="ret")])
        rec = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        r = rec.chat_completion(_MSGS, model=_MODEL)
        assert r["content"] == "ret"

    def test_record_count(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="a"), FakeLLMResponse(content="b")])
        rec = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        rec.chat_completion([{"role": "user", "content": "x"}], model="m")
        rec.chat_completion([{"role": "user", "content": "y"}], model="m")
        assert rec.record_count == 2


# ===================================================================
# Record + Replay
# ===================================================================


class TestRecordReplay:
    def test_replay_hit(self, tmp_path: Path):
        _record_one(tmp_path, "cached")
        fake = FakeLLM(responses=[FakeLLMResponse(content="should not use")])
        r = LLMRecorder(tmp_path, record=True, replay=True, client=fake)
        result = r.chat_completion(_MSGS, model=_MODEL)
        assert result["content"] == "cached"
        assert fake.call_count == 0  # client not called

    def test_falls_back_to_record(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="fresh")])
        r = LLMRecorder(tmp_path, record=True, replay=True, client=fake)
        result = r.chat_completion(_MSGS, model=_MODEL)
        assert result["content"] == "fresh"
        assert r.record_count == 1

    def test_second_call_replays(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="once")])
        r = LLMRecorder(tmp_path, record=True, replay=True, client=fake)
        r.chat_completion(_MSGS, model=_MODEL)  # records
        r.chat_completion(_MSGS, model=_MODEL)  # replays
        assert r.record_count == 1
        assert r.replay_count == 1


# ===================================================================
# OpenAI interface
# ===================================================================


class TestOpenAI:
    def test_create_works(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="openai")])
        r = LLMRecorder(tmp_path, record=True, replay=True, client=fake)
        resp = r.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
        )
        assert resp.choices[0].message.content == "openai"

    def test_content(self, tmp_path: Path):
        _record_one(tmp_path, "oa_content")
        r = LLMRecorder(tmp_path, replay=True)
        resp = r.chat.completions.create(model=_MODEL, messages=_MSGS)
        assert resp.choices[0].message.content == "oa_content"

    def test_tool_calls(self, tmp_path: Path):
        tc = [{"name": "f", "arguments": '{"a":1}', "id": "c1"}]
        fake = FakeLLM(responses=[FakeLLMResponse(tool_calls=tc, finish_reason="tool_calls")])
        r = LLMRecorder(tmp_path, record=True, replay=True, client=fake)
        resp = r.chat.completions.create(model="m", messages=_MSGS)
        assert resp.choices[0].message.tool_calls is not None
        assert resp.choices[0].message.tool_calls[0].function.name == "f"


# ===================================================================
# Cassette management
# ===================================================================


class TestCassetteManagement:
    def test_list_cassettes(self, tmp_path: Path):
        _record_one(tmp_path)
        r = LLMRecorder(tmp_path)
        assert len(r.list_cassettes()) == 1

    def test_clear_cassettes(self, tmp_path: Path):
        _record_one(tmp_path)
        r = LLMRecorder(tmp_path)
        count = r.clear_cassettes()
        assert count == 1
        assert len(r.list_cassettes()) == 0

    def test_save_creates_file(self, tmp_path: Path):
        r = LLMRecorder(tmp_path)
        r.save_cassette("test_hash", {"content": "saved"})
        assert (tmp_path / "test_hash.json").exists()

    def test_load_reads_file(self, tmp_path: Path):
        r = LLMRecorder(tmp_path)
        r.save_cassette("ld", {"content": "loaded"})
        assert r.load_cassette("ld") == {"content": "loaded"}


# ===================================================================
# Stats
# ===================================================================


class TestStats:
    def test_initial_zeros(self, tmp_path: Path):
        s = LLMRecorder(tmp_path).stats()
        assert s == {"calls": 0, "replays": 0, "records": 0, "cassettes": 0}

    def test_tracks(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="x")])
        r = LLMRecorder(tmp_path, record=True, replay=True, client=fake)
        r.chat_completion(_MSGS, model=_MODEL)  # record
        r.chat_completion(_MSGS, model=_MODEL)  # replay
        s = r.stats()
        assert s["calls"] == 2
        assert s["records"] == 1
        assert s["replays"] == 1

    def test_call_count_total(self, tmp_path: Path):
        _record_one(tmp_path)
        r = LLMRecorder(tmp_path, replay=True)
        r.chat_completion(_MSGS, model=_MODEL)
        r.chat_completion(_MSGS, model=_MODEL)
        assert r.call_count == 2


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_corrupt_cassette(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("{{{not json")
        r = LLMRecorder(tmp_path)
        assert r.load_cassette("bad") is None

    def test_empty_messages(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="empty")])
        r = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        result = r.chat_completion([], model="m")
        assert result["content"] == "empty"

    def test_with_functions(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="fn")])
        r = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        result = r.chat_completion_with_functions(
            _MSGS, functions=[{"name": "f", "parameters": {}}], model="m",
        )
        assert result["content"] == "fn"
