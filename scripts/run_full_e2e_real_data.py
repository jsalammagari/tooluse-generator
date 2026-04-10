"""
Full end-to-end pipeline with REAL ToolBench data — Tasks 1-42.

Loads actual ToolBench JSON, builds the full pipeline, generates
conversations via the multi-agent orchestrator, and writes JSONL output.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "toolenv" / "tools"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.is_dir():
    print(f"ERROR: ToolBench data not found at {DATA_DIR}")
    sys.exit(1)

# =========================================================================
print("=" * 70)
print("PHASE 1: Load Real ToolBench Data")
print("=" * 70)

from tooluse_gen.registry.loader import ToolBenchLoader, LoaderConfig
from tooluse_gen.registry.registry import ToolRegistry
from tooluse_gen.registry.completeness import CompletenessCalculator, get_quality_tier
from tooluse_gen.utils.seeding import set_global_seed

set_global_seed(42)

DOMAINS = ["Finance", "Travel", "Food", "Weather", "Sports", "Entertainment"]

loader = ToolBenchLoader(LoaderConfig(
    strict_mode=False, infer_types=True,
    min_endpoints=2, max_endpoints=15,
    calculate_completeness=True, min_completeness=0.5,
    max_tools=8,
))

all_tools = []
for domain in DOMAINS:
    domain_dir = DATA_DIR / domain
    if not domain_dir.is_dir():
        continue
    tools = loader.load_directory(domain_dir, recursive=True, progress=False)
    print(f"  {domain}: {len(tools)} tools")
    all_tools.extend(tools)

registry = ToolRegistry()
registry.add_tools(all_tools)
print(f"\n  Total: {len(registry)} tools, "
      f"{sum(len(t.endpoints) for t in registry.tools())} endpoints, "
      f"{len(registry.domains)} domains")

top5 = sorted(registry.tools(), key=lambda t: t.completeness_score, reverse=True)[:5]
for t in top5:
    print(f"    {t.name[:40]:<42s} score={t.completeness_score:.3f} eps={len(t.endpoints)}")
print()

# =========================================================================
print("=" * 70)
print("PHASE 2: Build Tool Graph")
print("=" * 70)

from tooluse_gen.graph.models import GraphConfig
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.builder import GraphBuilder
from tooluse_gen.graph.queries import get_graph_stats


class MockEmb(EmbeddingService):
    def __init__(self) -> None:
        self._model = None
        self._cache_dir = None

    def embed_text(self, text: str) -> list[float]:
        h = hash(text)
        rng = np.random.default_rng(abs(h) % (2**31))
        vec = rng.standard_normal(384).tolist()
        n = sum(x * x for x in vec) ** 0.5
        return [x / n for x in vec]

    def embed_batch(self, texts: list[str], **kw: object) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


t0 = time.time()
graph = GraphBuilder(
    config=GraphConfig(
        include_tool_nodes=True, include_domain_edges=True,
        include_semantic_edges=False, max_edges_per_node=50,
    ),
    embedding_service=MockEmb(),
).build(registry)
build_time = time.time() - t0

gstats = get_graph_stats(graph)
print(f"  Nodes: {gstats.tool_node_count} tools + {gstats.endpoint_node_count} endpoints")
print(f"  Edges: {graph.number_of_edges()}")
print(f"  Components: {gstats.connected_components}")
print(f"  Build time: {build_time:.2f}s")
print()

# =========================================================================
print("=" * 70)
print("PHASE 3: Sample Tool Chains")
print("=" * 70)

from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.sampler import SamplerConfig
from tooluse_gen.graph.diversity import DiversitySteeringConfig
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.patterns import chain_to_description

sampler = ToolChainSampler(
    graph, SamplerConfig(max_iterations=500, max_retries=50),
    DiversitySteeringConfig(enabled=True, weight_decay=0.85),
)
rng = np.random.default_rng(42)
NUM_CONVS = 10
constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)

chains = sampler.sample_batch(constraints, count=NUM_CONVS, rng=rng)
diversity = sampler.get_diversity_report()
print(f"  Sampled {len(chains)} chains")
print(f"  Diversity: entropy={diversity.tool_entropy:.2f}, "
      f"domain_cov={diversity.domain_coverage:.2f}")
for i, ch in enumerate(chains[:5]):
    print(f"    Chain {i}: {ch.total_step_count} steps, "
          f"tools={ch.tool_ids}, desc={chain_to_description(ch)[:70]}")
print()

# =========================================================================
print("=" * 70)
print("PHASE 4: Generate Conversations (Multi-Agent Orchestrator)")
print("=" * 70)

from tooluse_gen.agents import (
    AssistantAgent, BatchGenerator, ConversationOrchestrator,
    OrchestratorConfig, ToolExecutor, UserSimulator,
)

# Run A: Steering ON, no disambiguation
orch_a = ConversationOrchestrator(
    user_sim=UserSimulator(),
    assistant=AssistantAgent(registry=registry),
    executor=ToolExecutor(registry),
    config=OrchestratorConfig(
        require_disambiguation=False, require_final_answer=True, max_turns=15,
    ),
)
gen_a = BatchGenerator(orchestrator=orch_a, sampler=sampler,
                       diversity_config=DiversitySteeringConfig(enabled=True))

t0 = time.time()
batch_a = gen_a.generate_batch(
    count=NUM_CONVS, constraints=constraints, seed=42, steering_enabled=True,
)
gen_time = time.time() - t0
stats_a = gen_a.get_batch_stats()

print(f"  Run A (no disambig): {stats_a.total_generated} convs in {gen_time:.2f}s")
print(f"    Avg turns: {stats_a.average_turns:.1f}, "
      f"Avg tool calls: {stats_a.average_tool_calls:.1f}")
print(f"    Tools: {stats_a.tools_coverage}, Domains: {stats_a.domain_coverage}")

# Run B: Steering ON, with disambiguation
orch_b = ConversationOrchestrator(
    user_sim=UserSimulator(),
    assistant=AssistantAgent(registry=registry),
    executor=ToolExecutor(registry),
    config=OrchestratorConfig(
        require_disambiguation=True, disambiguation_probability=1.0,
        require_final_answer=True, max_turns=15,
    ),
)
gen_b = BatchGenerator(orchestrator=orch_b, sampler=sampler)
batch_b = gen_b.generate_batch(
    count=5, constraints=constraints, seed=42, steering_enabled=True,
)
stats_b = gen_b.get_batch_stats()
print(f"  Run B (with disambig): {stats_b.total_generated} convs, "
      f"avg {stats_b.average_turns:.1f} turns")
print()

# =========================================================================
print("=" * 70)
print("PHASE 5: Write JSONL Output")
print("=" * 70)

out_path = OUTPUT_DIR / "real_toolbench_conversations_v2.jsonl"
all_convs = batch_a + batch_b
with open(out_path, "w") as f:
    for conv in all_convs:
        f.write(conv.to_jsonl() + "\n")

print(f"  Output: {out_path}")
print(f"  Records: {len(all_convs)}")
print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")
print()

# =========================================================================
print("=" * 70)
print("PHASE 6: Display Conversations")
print("=" * 70)

for i, conv in enumerate(all_convs):
    record = conv.to_jsonl_dict()
    m = conv.metadata
    label = "A" if i < len(batch_a) else "B"
    idx = i if i < len(batch_a) else i - len(batch_a)

    print(f"\n--- Conversation {label}-{idx} (seed={m.seed}) ---")
    print(f"    Tools: {m.tools_used} | Domains: {m.domains} | "
          f"Pattern: {m.pattern}")
    print(f"    Turns: {m.num_turns} | Tool calls: {m.num_tool_calls} | "
          f"Disambig: {m.disambiguation_count}")
    print(f"    Endpoints: {m.endpoints_called}")
    print(f"    Grounding: {m.grounding_stats}")
    print(f"    Messages:")

    for msg in record["messages"]:
        role = msg["role"]
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            args_str = json.dumps(tc["arguments"], default=str)
            if len(args_str) > 60:
                args_str = args_str[:57] + "..."
            print(f"      [{role}] CALL {tc['tool_name']}.{tc['endpoint']}({args_str})")
        elif isinstance(msg.get("content"), dict):
            content_str = json.dumps(msg["content"], default=str)
            if len(content_str) > 80:
                content_str = content_str[:77] + "..."
            print(f"      [{role}] {content_str}")
        else:
            text = str(msg.get("content", ""))
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"      [{role}] {text}")

# =========================================================================
print()
print("=" * 70)
print("PHASE 7: Validation & Statistics")
print("=" * 70)

errors = []
multi_tool = 0
multi_step = 0
total_grounded = 0
total_fresh = 0
total_args = 0
all_tools_used: set[str] = set()
all_domains_used: set[str] = set()
all_endpoints: set[str] = set()

for i, conv in enumerate(all_convs):
    record = conv.to_jsonl_dict()
    m = conv.metadata

    # Structure
    if "conversation_id" not in record:
        errors.append(f"Conv {i}: missing conversation_id")
    if not record.get("messages"):
        errors.append(f"Conv {i}: no messages")

    # Roles
    roles = [msg["role"] for msg in record["messages"]]
    if "user" not in roles:
        errors.append(f"Conv {i}: no user")
    if "assistant" not in roles:
        errors.append(f"Conv {i}: no assistant")
    if "tool" not in roles:
        errors.append(f"Conv {i}: no tool")
    if roles[0] != "user":
        errors.append(f"Conv {i}: first msg not user")

    # Last is assistant text
    last = record["messages"][-1]
    if last["role"] != "assistant" or last.get("content") is None:
        errors.append(f"Conv {i}: last msg not assistant text")

    # Tool call format
    for msg in record["messages"]:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if "endpoint" not in tc or "arguments" not in tc:
                    errors.append(f"Conv {i}: bad tool_call format")
        if msg["role"] == "tool" and not isinstance(msg.get("content"), dict):
            errors.append(f"Conv {i}: tool content not dict")

    # Metadata
    if m.num_turns != len(record["messages"]):
        errors.append(f"Conv {i}: turns mismatch")
    if m.num_tool_calls != len(m.endpoints_called):
        errors.append(f"Conv {i}: tool_calls != endpoints count")
    gs = m.grounding_stats
    if gs["total_args"] != gs["grounded_args"] + gs["fresh_args"]:
        errors.append(f"Conv {i}: grounding sum mismatch")

    # JSON
    try:
        json.loads(conv.to_jsonl())
    except json.JSONDecodeError:
        errors.append(f"Conv {i}: invalid JSON")

    # Stats
    if m.num_distinct_tools >= 2:
        multi_tool += 1
    if m.num_tool_calls >= 3:
        multi_step += 1
    total_grounded += gs["grounded_args"]
    total_fresh += gs["fresh_args"]
    total_args += gs["total_args"]
    all_tools_used.update(m.tools_used)
    all_domains_used.update(m.domains)
    all_endpoints.update(m.endpoints_called)

n = len(all_convs)
if errors:
    print(f"  ERRORS ({len(errors)}):")
    for e in errors[:10]:
        print(f"    {e}")
else:
    print("  All conversations valid: PASS")

print(f"\n  Dataset Statistics:")
print(f"    Total conversations:  {n}")
print(f"    Multi-tool (>=2):     {multi_tool}/{n} ({100*multi_tool/n:.0f}%)")
print(f"    Multi-step (>=3):     {multi_step}/{n} ({100*multi_step/n:.0f}%)")
print(f"    Avg turns (Run A):    {stats_a.average_turns:.1f}")
print(f"    Avg turns (Run B):    {stats_b.average_turns:.1f}")
print(f"    Total grounded args:  {total_grounded}")
print(f"    Total fresh args:     {total_fresh}")
print(f"    Total args:           {total_args}")
print(f"    Grounding rate:       {100*total_grounded/max(total_args,1):.1f}%")
print(f"    Distinct tools used:  {len(all_tools_used)}")
print(f"    Distinct domains:     {len(all_domains_used)}")
print(f"    Distinct endpoints:   {len(all_endpoints)}")
print(f"    Tools: {sorted(all_tools_used)[:10]}")
if len(all_tools_used) > 10:
    print(f"           ... and {len(all_tools_used) - 10} more")
print(f"    Domains: {sorted(all_domains_used)}")
print(f"    Endpoints: {sorted(all_endpoints)[:10]}")
if len(all_endpoints) > 10:
    print(f"               ... and {len(all_endpoints) - 10} more")

print()
print("=" * 70)
if errors:
    print(f"COMPLETED WITH {len(errors)} ERRORS")
else:
    print("ALL PHASES PASSED")
print("=" * 70)
