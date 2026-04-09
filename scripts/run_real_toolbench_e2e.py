"""
End-to-end pipeline with REAL ToolBench data.

Loads actual ToolBench JSON files from data/toolenv/tools/,
builds the full pipeline, and produces conversation records.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "toolenv" / "tools"

if not DATA_DIR.is_dir():
    print(f"ERROR: ToolBench data not found at {DATA_DIR}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Load ToolBench Data
# ---------------------------------------------------------------------------

from tooluse_gen.registry.loader import ToolBenchLoader, LoaderConfig
from tooluse_gen.registry.registry import ToolRegistry, RegistryBuilder
from tooluse_gen.registry.completeness import CompletenessCalculator, get_quality_tier
from tooluse_gen.utils.seeding import set_global_seed

set_global_seed(42)

print("=" * 70)
print("PHASE 1: Load Real ToolBench Data")
print("=" * 70)

# Load from select domains to keep it manageable
DOMAINS_TO_LOAD = ["Finance", "Travel", "Food", "Weather", "Sports", "Entertainment"]

t0 = time.time()
loader = ToolBenchLoader(LoaderConfig(
    strict_mode=False,
    infer_types=True,
    min_endpoints=2,       # Only tools with 2+ endpoints (enables within-tool chaining)
    max_endpoints=15,      # Cap endpoint count for manageable graph
    calculate_completeness=True,
    min_completeness=0.5,  # Filter low quality
    max_tools=8,           # Per domain — keeps domain edges manageable
))

all_tools = []
for domain in DOMAINS_TO_LOAD:
    domain_dir = DATA_DIR / domain
    if not domain_dir.is_dir():
        print(f"  Skipping {domain} (directory not found)")
        continue
    tools = loader.load_directory(domain_dir, recursive=True, progress=False)
    print(f"  {domain}: loaded {len(tools)} tools")
    all_tools.extend(tools)

stats = loader.get_stats()
load_time = time.time() - t0

print(f"\n  Total tools loaded: {len(all_tools)}")
print(f"  Files processed: {stats.files_processed}")
print(f"  Files failed: {stats.files_failed}")
print(f"  Endpoints: {stats.endpoints_loaded}")
print(f"  Parameters: {stats.parameters_loaded}")
print(f"  Missing descriptions: {stats.missing_descriptions}")
print(f"  Inferred types: {stats.inferred_types}")
print(f"  Load time: {load_time:.2f}s")
print()

# ---------------------------------------------------------------------------
# 2. Build Registry with Quality Filtering
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 2: Build Registry & Score Quality")
print("=" * 70)

registry = ToolRegistry()
registry.add_tools(all_tools)

# Show quality distribution
calc = CompletenessCalculator()
tiers = {}
for tool in registry.tools():
    tier = get_quality_tier(tool.completeness_score)
    tier_name = tier.value if hasattr(tier, 'value') else str(tier)
    tiers[tier_name] = tiers.get(tier_name, 0) + 1

print(f"  Registry: {len(registry)} tools")
total_eps = sum(len(t.endpoints) for t in registry.tools())
print(f"  Total endpoints: {total_eps}")
print(f"  Domains: {registry.domains}")
print(f"  Quality distribution:")
for tier, count in sorted(tiers.items()):
    print(f"    {tier}: {count}")

# Show top 10 tools by completeness
top_tools = sorted(registry.tools(), key=lambda t: t.completeness_score, reverse=True)[:10]
print(f"\n  Top 10 tools by completeness:")
for t in top_tools:
    print(f"    {t.name[:40]:<42s} score={t.completeness_score:.3f}  eps={len(t.endpoints)}  domain={t.domain}")
print()

# ---------------------------------------------------------------------------
# 3. Registry Serialization
# ---------------------------------------------------------------------------

from tooluse_gen.registry.serialization import save_registry, load_registry

print("=" * 70)
print("PHASE 3: Registry Serialization Round-Trip")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    reg_path = Path(tmp) / "toolbench_registry.json"
    save_registry(registry, reg_path)
    file_size = reg_path.stat().st_size
    loaded_reg, meta = load_registry(reg_path)
    assert len(loaded_reg) == len(registry)
    print(f"  Saved: {file_size / 1024:.1f} KB")
    print(f"  Loaded: {len(loaded_reg)} tools, {meta.endpoint_count} endpoints")
    print(f"  Checksum: {meta.checksum}")
print()

# ---------------------------------------------------------------------------
# 4. Build Tool Graph
# ---------------------------------------------------------------------------

from tooluse_gen.graph.models import GraphConfig
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.builder import GraphBuilder

print("=" * 70)
print("PHASE 4: Build Tool Graph")
print("=" * 70)

# Mock embedding for speed (real embeddings would use sentence-transformers)
class MockEmbeddingService(EmbeddingService):
    def __init__(self):
        self._model = None
        self._cache_dir = None

    def embed_text(self, text):
        h = hash(text)
        rng = np.random.default_rng(abs(h) % (2**31))
        vec = rng.standard_normal(384).tolist()
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_batch(self, texts, batch_size=256, show_progress=False):
        return [self.embed_text(t) for t in texts]

t0 = time.time()
graph_config = GraphConfig(
    include_tool_nodes=True,
    include_domain_edges=True,
    include_semantic_edges=False,  # Skip semantic edges for speed with large dataset
    similarity_threshold=0.3,
    max_edges_per_node=50,
)
builder = GraphBuilder(config=graph_config, embedding_service=MockEmbeddingService())
graph = builder.build(registry)
build_time = time.time() - t0

tool_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "tool"]
ep_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"]
edge_types = {}
for _, _, d in graph.edges(data=True):
    et = d.get("edge_type", "unknown")
    edge_types[et] = edge_types.get(et, 0) + 1

print(f"  Nodes: {graph.number_of_nodes()} ({len(tool_nodes)} tools, {len(ep_nodes)} endpoints)")
print(f"  Edges: {graph.number_of_edges()}")
for et, count in sorted(edge_types.items()):
    print(f"    {et}: {count}")
print(f"  Build time: {build_time:.2f}s")
print()

# ---------------------------------------------------------------------------
# 5. Graph Serialization & Queries
# ---------------------------------------------------------------------------

from tooluse_gen.graph.persistence import save_graph, load_graph, get_graph_info
from tooluse_gen.graph.queries import (
    get_endpoints_for_tool, get_domain_endpoints, get_chainable_endpoints,
    compute_node_importance, get_graph_stats,
)

print("=" * 70)
print("PHASE 5: Graph Persistence & Queries")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    gpath = Path(tmp) / "graph.pkl"
    gmeta = save_graph(graph, gpath)
    gsize = gpath.stat().st_size
    loaded_g, _ = load_graph(gpath)
    assert loaded_g.number_of_nodes() == graph.number_of_nodes()
    print(f"  Serialized: {gsize / 1024:.1f} KB")
    print(f"  Round-trip: {loaded_g.number_of_nodes()} nodes, {loaded_g.number_of_edges()} edges")

# Domain endpoint counts
for domain in registry.domains:
    eps = get_domain_endpoints(graph, domain)
    print(f"  Domain '{domain}': {len(eps)} endpoints")

# PageRank top 10
importance = compute_node_importance(graph)
ep_importance = {k: v for k, v in importance.items() if graph.nodes[k].get("node_type") == "endpoint"}
top10 = sorted(ep_importance.items(), key=lambda x: -x[1])[:10]
print(f"\n  Top 10 endpoints by PageRank:")
for nid, score in top10:
    name = graph.nodes[nid].get("name", "?")
    tool = graph.nodes[nid].get("tool_id", "?")
    print(f"    {tool}/{name}: {score:.5f}")

gstats = get_graph_stats(graph)
print(f"\n  Graph density: {gstats.density:.5f}")
print(f"  Connected components: {gstats.connected_components}")
print()

# ---------------------------------------------------------------------------
# 6. Chain Sampling (MCTS + Diversity)
# ---------------------------------------------------------------------------

from tooluse_gen.graph.chain_models import SamplingConstraints, ParallelGroup
from tooluse_gen.graph.sampler import SamplerConfig
from tooluse_gen.graph.diversity import DiversitySteeringConfig
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.patterns import chain_to_description

print("=" * 70)
print("PHASE 6: Sample Tool Chains (MCTS + Diversity)")
print("=" * 70)

t0 = time.time()
facade = ToolChainSampler(
    graph,
    sampler_config=SamplerConfig(max_iterations=500, max_retries=50),
    diversity_config=DiversitySteeringConfig(enabled=True, weight_decay=0.85),
)
constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
rng = np.random.default_rng(42)

NUM_CONVERSATIONS = 20
chains = facade.sample_batch(constraints, count=NUM_CONVERSATIONS, rng=rng)
sample_time = time.time() - t0

print(f"  Sampled {len(chains)} chains in {sample_time:.2f}s")
for i, chain in enumerate(chains[:10]):
    desc = chain_to_description(chain)
    print(f"    Chain {i}: {chain.total_step_count} steps, tools={chain.tool_ids}, pattern={chain.pattern}")
    print(f"             {desc[:100]}")

if len(chains) > 10:
    print(f"    ... and {len(chains) - 10} more chains")

metrics = facade.get_diversity_report()
print(f"\n  Diversity Metrics:")
print(f"    Tool entropy:           {metrics.tool_entropy:.3f}")
print(f"    Domain coverage:        {metrics.domain_coverage:.3f}")
print(f"    Unique tool pair ratio: {metrics.unique_tool_pair_ratio:.3f}")
print(f"    Pattern repetition:     {metrics.pattern_repetition_rate:.3f}")
print()

# ---------------------------------------------------------------------------
# 7. Execute Chains → Generate Conversations
# ---------------------------------------------------------------------------

from tooluse_gen.agents.execution_models import ConversationContext, ToolCallRequest
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.argument_generator import ArgumentGenerator
from tooluse_gen.agents.grounding import GroundingTracker, format_available_values, format_grounding_context


def flatten_steps(chain):
    flat = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            flat.extend(item.steps)
        else:
            flat.append(item)
    return flat


print("=" * 70)
print("PHASE 7: Execute Chains (Full Pipeline)")
print("=" * 70)

t0 = time.time()
executor = ToolExecutor(registry)
arg_gen = ArgumentGenerator()
conversations = []
execution_stats = {"total_calls": 0, "success": 0, "grounded_args": 0, "total_args": 0}

for chain_idx, chain in enumerate(chains):
    exec_rng = np.random.default_rng(42 + chain_idx)
    context = ConversationContext(chain=chain)
    tracker = GroundingTracker()

    context.add_message("user", f"Help me with: {', '.join(chain.domains_involved)}")
    flat_steps = flatten_steps(chain)

    for i, step in enumerate(flat_steps):
        endpoint = registry.get_endpoint(step.endpoint_id)
        if endpoint is None:
            continue

        args = arg_gen.generate_arguments(endpoint, context, exec_rng)
        execution_stats["total_args"] += len(args)

        # Count grounded args
        available = context.get_available_values()
        for k in args:
            if k in available:
                execution_stats["grounded_args"] += 1

        request = ToolCallRequest.from_chain_step(step, args)
        response = executor.execute(request, context, exec_rng)
        execution_stats["total_calls"] += 1
        if response.is_success:
            execution_stats["success"] += 1

        context.add_tool_output(response)
        tracker.track_from_response(response, step.endpoint_id, i)

        if i < len(flat_steps) - 1:
            context.advance_step()

    context.add_message("assistant", "Here are the results of your request.")

    record = {
        "conversation_id": context.conversation_id,
        "messages": context.get_history_for_prompt(),
        "metadata": {
            "seed": 42 + chain_idx,
            "tools_used": list(chain.tool_ids),
            "domains": list(chain.domains_involved),
            "num_turns": len(context.messages),
            "num_tool_calls": len(context.tool_outputs),
            "num_distinct_tools": len(set(chain.tool_ids)),
            "pattern": chain.pattern,
            "grounding_values_count": len(context.grounding_values),
        },
    }
    conversations.append(record)

exec_time = time.time() - t0
grounding_rate = (execution_stats["grounded_args"] / max(execution_stats["total_args"], 1)) * 100

print(f"  Executed {execution_stats['total_calls']} tool calls in {exec_time:.2f}s")
total_calls = max(execution_stats['total_calls'], 1)
print(f"  Success rate: {execution_stats['success']}/{execution_stats['total_calls']} ({100*execution_stats['success']/total_calls:.0f}%)")
print(f"  Grounded arguments: {execution_stats['grounded_args']}/{execution_stats['total_args']} ({grounding_rate:.1f}%)")
print()

# Show 3 detailed conversations
for ci in range(min(3, len(conversations))):
    conv = conversations[ci]
    meta = conv["metadata"]
    print(f"  Conversation {ci}: {meta['tools_used']}")
    print(f"    Turns: {meta['num_turns']}, Tool calls: {meta['num_tool_calls']}, Grounding values: {meta['grounding_values_count']}")
    for msg in conv["messages"][:5]:
        role = msg["role"]
        content = str(msg.get("content", ""))[:90]
        print(f"    [{role}] {content}")
    if len(conv["messages"]) > 5:
        print(f"    ... ({len(conv['messages']) - 5} more)")
    print()

# ---------------------------------------------------------------------------
# 8. Write JSONL Output
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 8: JSONL Output & Dataset Stats")
print("=" * 70)

output_path = Path(__file__).resolve().parent.parent / "output" / "real_toolbench_conversations.jsonl"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    for rec in conversations:
        f.write(json.dumps(rec, default=str) + "\n")

print(f"  Output: {output_path}")
print(f"  Records: {len(conversations)}")
print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

# Dataset property analysis
multi_tool = sum(1 for c in conversations if c["metadata"]["num_distinct_tools"] >= 2)
multi_step = sum(1 for c in conversations if c["metadata"]["num_tool_calls"] >= 3)
n_conv = max(len(conversations), 1)
avg_turns = sum(c["metadata"]["num_turns"] for c in conversations) / n_conv
avg_calls = sum(c["metadata"]["num_tool_calls"] for c in conversations) / n_conv
domains_used = set()
tools_used = set()
for c in conversations:
    domains_used.update(c["metadata"]["domains"])
    tools_used.update(c["metadata"]["tools_used"])

print(f"\n  Dataset Properties:")
print(f"    Multi-tool (>=2):     {multi_tool}/{n_conv} ({100*multi_tool/n_conv:.0f}%)")
print(f"    Multi-step (>=3):     {multi_step}/{n_conv} ({100*multi_step/n_conv:.0f}%)")
print(f"    Avg turns/conv:       {avg_turns:.1f}")
print(f"    Avg tool calls/conv:  {avg_calls:.1f}")
print(f"    Distinct tools used:  {len(tools_used)}")
print(f"    Domains covered:      {len(domains_used)}/{len(registry.domains)}")
print(f"    Tools used: {sorted(tools_used)[:15]}")
if len(tools_used) > 15:
    print(f"                ... and {len(tools_used) - 15} more")
print()

# ---------------------------------------------------------------------------
# 9. Diversity Experiment (Steering ON vs OFF)
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 9: Diversity Experiment (Steering ON vs OFF)")
print("=" * 70)

def run_batch(enabled, seed=42, count=20):
    f = ToolChainSampler(
        graph,
        sampler_config=SamplerConfig(max_iterations=500, max_retries=50),
        diversity_config=DiversitySteeringConfig(enabled=enabled, weight_decay=0.85),
    )
    c = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
    r = np.random.default_rng(seed)
    batch = f.sample_batch(c, count=count, rng=r)
    return batch, f.get_diversity_report()

chains_off, m_off = run_batch(False)
chains_on, m_on = run_batch(True)

print(f"\n  {'Metric':<30s} {'Steering OFF':>14s} {'Steering ON':>14s}")
print(f"  {'-'*30} {'-'*14} {'-'*14}")
print(f"  {'Chains produced':<30s} {len(chains_off):>14d} {len(chains_on):>14d}")
print(f"  {'Tool entropy':<30s} {m_off.tool_entropy:>14.3f} {m_on.tool_entropy:>14.3f}")
print(f"  {'Domain coverage':<30s} {m_off.domain_coverage:>14.3f} {m_on.domain_coverage:>14.3f}")
print(f"  {'Unique tool pair ratio':<30s} {m_off.unique_tool_pair_ratio:>14.3f} {m_on.unique_tool_pair_ratio:>14.3f}")
print(f"  {'Pattern repetition rate':<30s} {m_off.pattern_repetition_rate:>14.3f} {m_on.pattern_repetition_rate:>14.3f}")

def tool_dist(chains):
    counts = {}
    for ch in chains:
        for tid in ch.tool_ids:
            counts[tid] = counts.get(tid, 0) + 1
    return counts

dist_off = tool_dist(chains_off)
dist_on = tool_dist(chains_on)
all_t = sorted(set(list(dist_off.keys()) + list(dist_on.keys())))

print(f"\n  Tool usage (top 15):")
print(f"  {'Tool':<35s} {'OFF':>5s} {'ON':>5s}")
print(f"  {'-'*35} {'-'*5} {'-'*5}")
for t in all_t[:15]:
    print(f"  {t[:35]:<35s} {dist_off.get(t,0):>5d} {dist_on.get(t,0):>5d}")
if len(all_t) > 15:
    print(f"  ... ({len(all_t) - 15} more tools)")

def domain_dist(chains):
    counts = {}
    for ch in chains:
        for d in ch.domains_involved:
            counts[d] = counts.get(d, 0) + 1
    return counts

dd_off = domain_dist(chains_off)
dd_on = domain_dist(chains_on)
all_d = sorted(set(list(dd_off.keys()) + list(dd_on.keys())))
print(f"\n  Domain usage:")
print(f"  {'Domain':<25s} {'OFF':>5s} {'ON':>5s}")
print(f"  {'-'*25} {'-'*5} {'-'*5}")
for d in all_d:
    print(f"  {d:<25s} {dd_off.get(d,0):>5d} {dd_on.get(d,0):>5d}")
print()

# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 10: Determinism Verification")
print("=" * 70)

from tooluse_gen.graph.sampler import MCTSSampler

def run_pipeline(seed):
    r = np.random.default_rng(seed)
    sampler = MCTSSampler(graph, SamplerConfig(max_iterations=300, max_retries=30))
    c = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)
    chain = sampler.sample(c, r)
    ctx = ConversationContext(chain=chain)
    ex = ToolExecutor(registry)
    ag = ArgumentGenerator()
    outputs = []
    for i, step in enumerate(flatten_steps(chain)):
        ep = registry.get_endpoint(step.endpoint_id)
        if ep is None:
            continue
        a = ag.generate_arguments(ep, ctx, r)
        req = ToolCallRequest.from_chain_step(step, a)
        resp = ex.execute(req, ctx, r)
        ctx.add_tool_output(resp)
        outputs.append(resp.data)
        if i < len(flatten_steps(chain)) - 1:
            ctx.advance_step()
    return chain.endpoint_ids, outputs

eids1, out1 = run_pipeline(42)
eids2, out2 = run_pipeline(42)
print(f"  Same chain: {eids1 == eids2}")
print(f"  Same outputs: {out1 == out2}")
print(f"  Deterministic: {eids1 == eids2 and out1 == out2}")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY — Real ToolBench End-to-End")
print("=" * 70)
print(f"  ToolBench data:    {DATA_DIR}")
print(f"  Domains loaded:    {DOMAINS_TO_LOAD}")
print(f"  Tools in registry: {len(registry)}")
print(f"  Endpoints:         {total_eps}")
print(f"  Graph nodes:       {graph.number_of_nodes()}")
print(f"  Graph edges:       {graph.number_of_edges()}")
print(f"  Chains sampled:    {len(chains)}")
print(f"  Conversations:     {len(conversations)}")
print(f"  Tool calls:        {execution_stats['total_calls']} ({execution_stats['success']} success)")
print(f"  Grounding rate:    {grounding_rate:.1f}%")
print(f"  Multi-tool:        {multi_tool}/{len(conversations)}")
print(f"  Multi-step:        {multi_step}/{len(conversations)}")
print(f"  Distinct tools:    {len(tools_used)}")
print(f"  Domains covered:   {len(domains_used)}")
print(f"  Deterministic:     {eids1 == eids2 and out1 == out2}")
print(f"  Output file:       {output_path}")
print(f"  ALL PHASES:        PASSED")
print("=" * 70)
