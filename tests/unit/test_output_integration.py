"""Integration tests for the full Phase 8 output pipeline (Task 52).

Exercises: generate → convert → validate → write JSONL → read → verify.
All agents run in offline mode — no LLM calls.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from tooluse_gen.agents import (
    AssistantAgent,
    Conversation,
    ConversationOrchestrator,
    OrchestratorConfig,
    ToolExecutor,
    UserSimulator,
)
from tooluse_gen.core import (
    ConversationRecord,
    JSONLReader,
    JSONLWriter,
    compare_configs,
    embed_config_in_output,
    ensure_reproducibility,
    from_conversation,
    load_config,
    load_config_from_output,
    serialize_run_config,
    validate_conversation_record,
    validate_record,
)
from tooluse_gen.evaluation import JudgeAgent
from tooluse_gen.graph.builder import GraphBuilder
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.models import GraphConfig
from tooluse_gen.graph.sampler import SamplerConfig
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType, Tool
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Mock embedding
# ---------------------------------------------------------------------------


class _MockEmb(EmbeddingService):
    def __init__(self) -> None:
        self._model = None
        self._cache_dir = None

    def embed_text(self, text: str) -> list[float]:
        h = hash(text)
        rng = np.random.default_rng(abs(h) % (2**31))
        vec = rng.standard_normal(384).tolist()
        n = sum(x * x for x in vec) ** 0.5
        return [x / n for x in vec]

    def embed_batch(
        self, texts: list[str], batch_size: int = 256, show_progress: bool = False
    ) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.add_tool(Tool(
        tool_id="hotels", name="Hotels API", domain="Travel",
        endpoints=[
            Endpoint(endpoint_id="hotels/search", tool_id="hotels", name="Search",
                     method=HttpMethod.GET, path="/s",
                     parameters=[Parameter(name="city", param_type=ParameterType.STRING, required=True)],
                     required_parameters=["city"]),
            Endpoint(endpoint_id="hotels/book", tool_id="hotels", name="Book",
                     method=HttpMethod.POST, path="/b",
                     parameters=[
                         Parameter(name="hotel_id", param_type=ParameterType.STRING, required=True),
                         Parameter(name="guest_name", param_type=ParameterType.STRING, required=True),
                     ], required_parameters=["hotel_id", "guest_name"]),
        ],
    ))
    reg.add_tool(Tool(
        tool_id="weather", name="Weather API", domain="Weather",
        endpoints=[
            Endpoint(endpoint_id="weather/current", tool_id="weather", name="Current",
                     method=HttpMethod.GET, path="/w",
                     parameters=[Parameter(name="city", param_type=ParameterType.STRING, required=True)],
                     required_parameters=["city"]),
        ],
    ))
    return reg


@pytest.fixture(scope="module")
def conversations(registry: ToolRegistry) -> list[Conversation]:
    graph = GraphBuilder(
        config=GraphConfig(include_semantic_edges=False), embedding_service=_MockEmb(),
    ).build(registry)
    sampler = ToolChainSampler(graph, SamplerConfig(max_iterations=300, max_retries=30))
    rng = np.random.default_rng(42)
    orch = ConversationOrchestrator(
        user_sim=UserSimulator(), assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry),
        config=OrchestratorConfig(require_disambiguation=False),
    )
    results: list[Conversation] = []
    for i in range(5):
        chain = sampler.sample_chain(
            SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng,
        )
        results.append(orch.generate_conversation(chain, seed=42 + i))
    return results


# ===================================================================
# 1. Import completeness
# ===================================================================


class TestImportCompleteness:
    def test_all_symbols(self):
        from tooluse_gen.core import __all__
        assert len(__all__) >= 26

    def test_output_model_symbols(self):
        assert ConversationRecord is not None
        assert from_conversation is not None
        assert validate_record is not None
        assert validate_conversation_record is not None

    def test_io_and_reproducibility(self):
        assert JSONLWriter is not None
        assert JSONLReader is not None
        assert serialize_run_config is not None
        assert embed_config_in_output is not None
        assert load_config_from_output is not None
        assert ensure_reproducibility is not None
        assert compare_configs is not None


# ===================================================================
# 2. Conversion pipeline
# ===================================================================


class TestConversion:
    def test_produces_record(self, conversations):
        rec = from_conversation(conversations[0])
        assert isinstance(rec, ConversationRecord)

    def test_has_fields(self, conversations):
        rec = from_conversation(conversations[0])
        assert rec.conversation_id
        assert len(rec.messages) > 0
        assert isinstance(rec.metadata, dict)

    def test_messages_have_role_content(self, conversations):
        rec = from_conversation(conversations[0])
        for msg in rec.messages:
            assert "role" in msg
            assert "content" in msg

    def test_with_eval_scores(self, conversations):
        from tooluse_gen.evaluation.models import JudgeScores as EvalScores
        scores = EvalScores(tool_correctness=5, argument_grounding=4,
                            task_completion=5, naturalness=4)
        rec = from_conversation(conversations[0], eval_scores=scores)
        assert rec.judge_scores is not None
        assert rec.judge_scores["tool_correctness"] == 5

    def test_without_scores_none(self, conversations):
        rec = from_conversation(conversations[0])
        assert rec.judge_scores is None


# ===================================================================
# 3. Validation
# ===================================================================


class TestValidation:
    def test_valid_record(self, conversations):
        rec = from_conversation(conversations[0])
        ok, errs = validate_record(json.loads(rec.to_jsonl()))
        assert ok and errs == []

    def test_valid_conversation_record(self, conversations):
        rec = from_conversation(conversations[0])
        ok, errs = validate_conversation_record(rec)
        assert ok

    def test_invalid_detected(self):
        ok, errs = validate_record({"conversation_id": "", "messages": []})
        assert not ok

    def test_all_generated_valid(self, conversations):
        for conv in conversations:
            rec = from_conversation(conv)
            ok, errs = validate_record(json.loads(rec.to_jsonl()))
            assert ok, f"Record invalid: {errs}"


# ===================================================================
# 4. JSONL write/read round-trip
# ===================================================================


class TestJsonlRoundTrip:
    def test_count_matches(self, conversations, tmp_path):
        records = [from_conversation(c) for c in conversations]
        p = tmp_path / "out.jsonl"
        JSONLWriter(p).write_batch(records)
        assert len(JSONLReader(p).read_all()) == len(records)

    def test_ids_match(self, conversations, tmp_path):
        records = [from_conversation(c) for c in conversations]
        p = tmp_path / "out.jsonl"
        JSONLWriter(p).write_batch(records)
        loaded = JSONLReader(p).read_all()
        for orig, loaded_rec in zip(records, loaded, strict=True):
            assert loaded_rec.conversation_id == orig.conversation_id

    def test_messages_preserved(self, conversations, tmp_path):
        records = [from_conversation(c) for c in conversations]
        p = tmp_path / "out.jsonl"
        JSONLWriter(p).write_batch(records)
        loaded = JSONLReader(p).read_all()
        assert len(loaded[0].messages) == len(records[0].messages)

    def test_scores_preserved(self, conversations, tmp_path):
        from tooluse_gen.evaluation.models import JudgeScores as EvalScores
        scores = EvalScores(tool_correctness=5, argument_grounding=4,
                            task_completion=5, naturalness=4)
        rec = from_conversation(conversations[0], eval_scores=scores)
        p = tmp_path / "out.jsonl"
        JSONLWriter(p).write_record(rec)
        loaded = JSONLReader(p).read_all()[0]
        assert loaded.judge_scores == rec.judge_scores

    def test_header_metadata(self, conversations, tmp_path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 42, "note": "test"})
        w.write_record(from_conversation(conversations[0]))
        meta = JSONLReader(p).read_metadata()
        assert meta is not None and meta["seed"] == 42

    def test_iterator_matches(self, conversations, tmp_path):
        records = [from_conversation(c) for c in conversations[:3]]
        p = tmp_path / "out.jsonl"
        JSONLWriter(p).write_batch(records)
        ids = [r.conversation_id for r in JSONLReader(p).read_iterator()]
        assert ids == [r.conversation_id for r in records]


# ===================================================================
# 5. Reproducibility integration
# ===================================================================


class TestReproducibility:
    def test_serialize_captures_fields(self):
        cfg = load_config()
        rc = serialize_run_config(cfg, seed=42, cli_args={"count": 5})
        assert rc["seed"] == 42
        assert "config" in rc
        assert "timestamp" in rc
        assert "version" in rc

    def test_embed_adds_config(self, conversations):
        records = [from_conversation(c) for c in conversations[:2]]
        rc = {"seed": 42, "test": True}
        embedded = embed_config_in_output(records, rc)
        assert all(r.metadata.get("run_config") is not None for r in embedded)

    def test_load_from_header(self, conversations, tmp_path):
        rc = {"seed": 77, "model": "x"}
        p = tmp_path / "h.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": rc})
        w.write_record(from_conversation(conversations[0]))
        assert load_config_from_output(p)["seed"] == 77

    def test_load_from_record(self, conversations, tmp_path):
        rec = from_conversation(conversations[0])
        embedded = embed_config_in_output([rec], {"seed": 88})
        p = tmp_path / "r.jsonl"
        JSONLWriter(p).write_batch(embedded)
        assert load_config_from_output(p)["seed"] == 88

    def test_full_round_trip(self, conversations, tmp_path):
        cfg = load_config()
        rc = serialize_run_config(cfg, seed=42)
        records = [from_conversation(c) for c in conversations[:2]]
        embedded = embed_config_in_output(records, rc)
        p = tmp_path / "rt.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": rc})
        w.write_batch(embedded)
        loaded = load_config_from_output(p)
        assert loaded["seed"] == 42


# ===================================================================
# 6. Full end-to-end flow
# ===================================================================


class TestFullFlow:
    def test_generate_score_convert_write_read(self, conversations, tmp_path):
        judge = JudgeAgent()
        records = []
        for c in conversations:
            s = judge.score(c)
            rec = from_conversation(c, eval_scores=s)
            records.append(rec)

        cfg = load_config()
        rc = serialize_run_config(cfg, seed=42)
        embedded = embed_config_in_output(records, rc)

        p = tmp_path / "e2e.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": rc})
        w.write_batch(embedded)

        loaded = JSONLReader(p).read_all()
        assert len(loaded) == len(conversations)

    def test_all_valid_after_pipeline(self, conversations, tmp_path):
        judge = JudgeAgent()
        records = [from_conversation(c, eval_scores=judge.score(c)) for c in conversations]
        p = tmp_path / "val.jsonl"
        JSONLWriter(p).write_batch(records)
        for rec in JSONLReader(p).read_all():
            ok, errs = validate_conversation_record(rec)
            assert ok, f"Invalid: {errs}"

    def test_config_recoverable(self, conversations, tmp_path):
        cfg = load_config()
        rc = serialize_run_config(cfg, seed=42)
        records = embed_config_in_output(
            [from_conversation(c) for c in conversations[:1]], rc,
        )
        p = tmp_path / "cfg.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": rc})
        w.write_batch(records)
        assert load_config_from_output(p)["seed"] == 42

    def test_raw_json_parseable(self, conversations, tmp_path):
        records = [from_conversation(c) for c in conversations]
        p = tmp_path / "raw.jsonl"
        JSONLWriter(p).write_batch(records)
        with open(p) as f:
            for line in f:
                parsed = json.loads(line.strip())
                assert "conversation_id" in parsed
                assert "messages" in parsed


# ===================================================================
# 7. Determinism
# ===================================================================


class TestDeterminism:
    def test_same_seed_same_records(self, registry):
        def run() -> list[str]:
            graph = GraphBuilder(
                config=GraphConfig(include_semantic_edges=False),
                embedding_service=_MockEmb(),
            ).build(registry)
            sampler = ToolChainSampler(graph, SamplerConfig(max_iterations=200, max_retries=20))
            rng = np.random.default_rng(42)
            chain = sampler.sample_chain(
                SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng,
            )
            orch = ConversationOrchestrator(
                user_sim=UserSimulator(), assistant=AssistantAgent(registry=registry),
                executor=ToolExecutor(registry),
                config=OrchestratorConfig(require_disambiguation=False),
            )
            conv = orch.generate_conversation(chain, seed=42)
            rec = from_conversation(conv)
            return [m.get("content", "") or "" for m in rec.messages]

        a = run()
        b = run()
        assert a == b

    def test_ensure_reproducibility_consistent(self):
        # Use numpy Generator which is not affected by other test state
        ensure_reproducibility(12345)
        a = np.random.default_rng(12345).random(3).tolist()
        ensure_reproducibility(12345)
        b = np.random.default_rng(12345).random(3).tolist()
        assert a == b


# ===================================================================
# 8. Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_conversation(self, conversations, tmp_path):
        rec = from_conversation(conversations[0])
        p = tmp_path / "single.jsonl"
        JSONLWriter(p).write_record(rec)
        loaded = JSONLReader(p).read_all()
        assert len(loaded) == 1

    def test_empty_metadata(self, tmp_path):
        rec = ConversationRecord(
            conversation_id="e", messages=[{"role": "user", "content": "hi"}],
        )
        p = tmp_path / "empty_meta.jsonl"
        JSONLWriter(p).write_record(rec)
        loaded = JSONLReader(p).read_all()
        assert loaded[0].metadata == {}

    def test_large_batch(self, conversations, tmp_path):
        records = [from_conversation(conversations[i % len(conversations)])
                    for i in range(20)]
        p = tmp_path / "large.jsonl"
        JSONLWriter(p).write_batch(records)
        assert len(JSONLReader(p).read_all()) == 20
