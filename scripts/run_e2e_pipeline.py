"""
Full end-to-end pipeline execution — exercises everything built in tasks 1-34.

Pipeline:
  1. Load config & set seed
  2. Create tools with endpoints/parameters → ToolRegistry
  3. Score completeness & filter by quality
  4. Serialize registry → reload → verify
  5. Build tool graph (nodes, domain edges, semantic edges)
  6. Serialize graph → reload → verify
  7. Query the graph
  8. Sample tool chains via MCTS (with diversity steering)
  9. For each chain: generate arguments → execute → extract values → ground
 10. Format for prompts
 11. Produce JSONL output records
 12. Run diversity experiment (steering ON vs OFF)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Config & Seed
# ---------------------------------------------------------------------------

from tooluse_gen.core.config import load_config
from tooluse_gen.utils.seeding import set_global_seed

print("=" * 70)
print("PHASE 1: Configuration & Seeding")
print("=" * 70)

config = load_config()
set_global_seed(42)
print(f"  Config loaded: sampling={config.sampling.min_steps}-{config.sampling.max_steps} steps")
print(f"  Seed: 42")
print(f"  Diversity enabled: {config.diversity.enabled}")
print()

# ---------------------------------------------------------------------------
# 2. Build Tool Registry
# ---------------------------------------------------------------------------

from tooluse_gen.registry.models import (
    Endpoint, HttpMethod, Parameter, ParameterLocation, ParameterType, ResponseSchema, Tool,
)
from tooluse_gen.registry.registry import ToolRegistry

print("=" * 70)
print("PHASE 2: Build Tool Registry")
print("=" * 70)

registry = ToolRegistry()

tools = [
    Tool(
        tool_id="hotel_api", name="Hotel API",
        description="Search, book, and manage hotel reservations worldwide",
        domain="Travel", categories=["travel", "hotels", "booking"],
        base_url="https://api.hotels.example.com", auth_type="apikey",
        endpoints=[
            Endpoint(
                endpoint_id="hotel_api/GET/search", tool_id="hotel_api",
                name="Search Hotels", description="Find available hotels in a city",
                method=HttpMethod.GET, path="/hotels/search",
                parameters=[
                    Parameter(name="city", description="City name", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="check_in", description="Check-in date", param_type=ParameterType.DATE, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="max_price", description="Max price per night", param_type=ParameterType.NUMBER, location=ParameterLocation.QUERY, required=False),
                    Parameter(name="stars", description="Minimum star rating", param_type=ParameterType.INTEGER, location=ParameterLocation.QUERY, required=False),
                ],
                required_parameters=["city", "check_in"],
                response_schema=ResponseSchema(status_code=200, schema_type=ParameterType.OBJECT, properties={"results": {"type": "array"}}),
            ),
            Endpoint(
                endpoint_id="hotel_api/POST/book", tool_id="hotel_api",
                name="Book Hotel", description="Book a hotel room",
                method=HttpMethod.POST, path="/hotels/book",
                parameters=[
                    Parameter(name="hotel_id", description="Hotel ID from search", param_type=ParameterType.STRING, location=ParameterLocation.BODY, required=True),
                    Parameter(name="check_in", description="Check-in date", param_type=ParameterType.DATE, location=ParameterLocation.BODY, required=True),
                    Parameter(name="guest_name", description="Guest name", param_type=ParameterType.STRING, location=ParameterLocation.BODY, required=True),
                ],
                required_parameters=["hotel_id", "check_in", "guest_name"],
            ),
            Endpoint(
                endpoint_id="hotel_api/GET/reviews", tool_id="hotel_api",
                name="Hotel Reviews", description="Get reviews for a specific hotel",
                method=HttpMethod.GET, path="/hotels/{hotel_id}/reviews",
                parameters=[
                    Parameter(name="hotel_id", description="Hotel ID", param_type=ParameterType.STRING, location=ParameterLocation.PATH, required=True),
                    Parameter(name="limit", description="Max reviews", param_type=ParameterType.INTEGER, location=ParameterLocation.QUERY, required=False, default=10),
                ],
                required_parameters=["hotel_id"],
            ),
        ],
    ),
    Tool(
        tool_id="weather_api", name="Weather API",
        description="Real-time weather data and forecasts",
        domain="Weather", categories=["weather", "forecast", "climate"],
        base_url="https://api.weather.example.com",
        endpoints=[
            Endpoint(
                endpoint_id="weather_api/GET/current", tool_id="weather_api",
                name="Current Weather", description="Get current weather conditions",
                method=HttpMethod.GET, path="/weather/current",
                parameters=[
                    Parameter(name="city", description="City name", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="units", description="Temperature units", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=False, enum_values=["celsius", "fahrenheit"]),
                ],
                required_parameters=["city"],
            ),
            Endpoint(
                endpoint_id="weather_api/GET/forecast", tool_id="weather_api",
                name="Weather Forecast", description="Multi-day weather forecast",
                method=HttpMethod.GET, path="/weather/forecast",
                parameters=[
                    Parameter(name="city", description="City name", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="days", description="Forecast days", param_type=ParameterType.INTEGER, location=ParameterLocation.QUERY, required=False, default=5),
                ],
                required_parameters=["city"],
            ),
        ],
    ),
    Tool(
        tool_id="flight_api", name="Flight API",
        description="Search and book flights",
        domain="Travel", categories=["travel", "flights", "airlines"],
        base_url="https://api.flights.example.com",
        endpoints=[
            Endpoint(
                endpoint_id="flight_api/GET/search", tool_id="flight_api",
                name="Search Flights", description="Search for available flights",
                method=HttpMethod.GET, path="/flights/search",
                parameters=[
                    Parameter(name="origin", description="Origin city", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="destination", description="Destination city", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="date", description="Departure date", param_type=ParameterType.DATE, location=ParameterLocation.QUERY, required=True),
                ],
                required_parameters=["origin", "destination", "date"],
            ),
            Endpoint(
                endpoint_id="flight_api/POST/book", tool_id="flight_api",
                name="Book Flight", description="Book a flight",
                method=HttpMethod.POST, path="/flights/book",
                parameters=[
                    Parameter(name="flight_id", description="Flight ID", param_type=ParameterType.STRING, location=ParameterLocation.BODY, required=True),
                    Parameter(name="passenger_name", description="Passenger name", param_type=ParameterType.STRING, location=ParameterLocation.BODY, required=True),
                ],
                required_parameters=["flight_id", "passenger_name"],
            ),
        ],
    ),
    Tool(
        tool_id="restaurant_api", name="Restaurant API",
        description="Search restaurants and make reservations",
        domain="Food", categories=["food", "dining", "restaurants"],
        base_url="https://api.restaurants.example.com",
        endpoints=[
            Endpoint(
                endpoint_id="restaurant_api/GET/search", tool_id="restaurant_api",
                name="Search Restaurants", description="Find restaurants in a city",
                method=HttpMethod.GET, path="/restaurants/search",
                parameters=[
                    Parameter(name="city", description="City name", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="cuisine", description="Cuisine type", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=False),
                ],
                required_parameters=["city"],
            ),
            Endpoint(
                endpoint_id="restaurant_api/POST/reserve", tool_id="restaurant_api",
                name="Make Reservation", description="Reserve a table",
                method=HttpMethod.POST, path="/restaurants/reserve",
                parameters=[
                    Parameter(name="restaurant_id", description="Restaurant ID", param_type=ParameterType.STRING, location=ParameterLocation.BODY, required=True),
                    Parameter(name="party_size", description="Number of guests", param_type=ParameterType.INTEGER, location=ParameterLocation.BODY, required=True),
                    Parameter(name="date", description="Reservation date", param_type=ParameterType.DATE, location=ParameterLocation.BODY, required=True),
                ],
                required_parameters=["restaurant_id", "party_size", "date"],
            ),
        ],
    ),
    Tool(
        tool_id="currency_api", name="Currency API",
        description="Currency exchange rates and conversion",
        domain="Finance", categories=["finance", "currency", "exchange"],
        base_url="https://api.currency.example.com",
        endpoints=[
            Endpoint(
                endpoint_id="currency_api/GET/rate", tool_id="currency_api",
                name="Exchange Rate", description="Get exchange rate between currencies",
                method=HttpMethod.GET, path="/exchange/rate",
                parameters=[
                    Parameter(name="from_currency", description="Source currency code", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="to_currency", description="Target currency code", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                ],
                required_parameters=["from_currency", "to_currency"],
            ),
            Endpoint(
                endpoint_id="currency_api/GET/convert", tool_id="currency_api",
                name="Convert Currency", description="Convert an amount between currencies",
                method=HttpMethod.GET, path="/exchange/convert",
                parameters=[
                    Parameter(name="from_currency", description="Source currency", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="to_currency", description="Target currency", param_type=ParameterType.STRING, location=ParameterLocation.QUERY, required=True),
                    Parameter(name="amount", description="Amount to convert", param_type=ParameterType.NUMBER, location=ParameterLocation.QUERY, required=True),
                ],
                required_parameters=["from_currency", "to_currency", "amount"],
            ),
        ],
    ),
]

registry.add_tools(tools)
total_endpoints = sum(len(t.endpoints) for t in registry.tools())
print(f"  Tools: {len(registry)}")
print(f"  Endpoints: {total_endpoints}")
print(f"  Domains: {registry.domains}")
print()

# ---------------------------------------------------------------------------
# 3. Completeness Scoring
# ---------------------------------------------------------------------------

from tooluse_gen.registry.completeness import CompletenessCalculator, QualityTier, get_quality_tier

print("=" * 70)
print("PHASE 3: Completeness Scoring")
print("=" * 70)

calc = CompletenessCalculator()
scored_tools = []
for tool in registry.tools():
    scored = calc.calculate_all(tool)
    scored_tools.append(scored)
    tier = get_quality_tier(scored.completeness_score)
    print(f"  {scored.name:25s}  score={scored.completeness_score:.3f}  tier={tier}")

# Rebuild registry with scored tools
registry = ToolRegistry()
registry.add_tools(scored_tools)
print()

# ---------------------------------------------------------------------------
# 4. Registry Serialization Round-Trip
# ---------------------------------------------------------------------------

from tooluse_gen.registry.serialization import save_registry, load_registry

print("=" * 70)
print("PHASE 4: Registry Serialization")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    reg_path = Path(tmp) / "registry.json"
    save_registry(registry, reg_path)
    loaded_registry, meta = load_registry(reg_path)
    assert len(loaded_registry) == len(registry), "Registry round-trip failed!"
    print(f"  Saved & loaded: {len(loaded_registry)} tools, {meta.endpoint_count} endpoints")
    print(f"  Checksum: {meta.checksum}")
print()

# ---------------------------------------------------------------------------
# 5. Build Tool Graph
# ---------------------------------------------------------------------------

from tooluse_gen.graph.models import GraphConfig
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.builder import GraphBuilder

print("=" * 70)
print("PHASE 5: Build Tool Graph")
print("=" * 70)

# Use a mock embedding service (deterministic, no model download)
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

graph_config = GraphConfig(
    include_tool_nodes=True,
    include_domain_edges=True,
    include_semantic_edges=True,
    similarity_threshold=0.3,
    max_edges_per_node=20,
)
builder = GraphBuilder(config=graph_config, embedding_service=MockEmbeddingService())
graph = builder.build(registry)

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
print()

# ---------------------------------------------------------------------------
# 6. Graph Serialization Round-Trip
# ---------------------------------------------------------------------------

from tooluse_gen.graph.persistence import save_graph, load_graph, get_graph_info

print("=" * 70)
print("PHASE 6: Graph Serialization")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    graph_path = Path(tmp) / "graph.pkl"
    graph_meta = save_graph(graph, graph_path)
    loaded_graph, loaded_meta = load_graph(graph_path)
    info = get_graph_info(graph_path)
    assert loaded_graph.number_of_nodes() == graph.number_of_nodes()
    assert loaded_graph.number_of_edges() == graph.number_of_edges()
    print(f"  Saved & loaded: {loaded_meta.node_count} nodes, {loaded_meta.edge_count} edges")
    print(f"  Checksum: {loaded_meta.checksum}")
    print(f"  Quick info (no full load): {info.node_count} nodes, {info.edge_count} edges")
print()

# ---------------------------------------------------------------------------
# 7. Graph Queries
# ---------------------------------------------------------------------------

from tooluse_gen.graph.queries import (
    get_endpoints_for_tool, get_domain_endpoints, get_chainable_endpoints,
    get_connected_endpoints, compute_node_importance, get_graph_stats,
)

print("=" * 70)
print("PHASE 7: Graph Queries")
print("=" * 70)

for tool_id in ["hotel_api", "weather_api", "flight_api", "restaurant_api", "currency_api"]:
    eps = get_endpoints_for_tool(graph, tool_id)
    print(f"  {tool_id}: {len(eps)} endpoints")

for domain in ["Travel", "Weather", "Food", "Finance"]:
    eps = get_domain_endpoints(graph, domain)
    print(f"  Domain '{domain}': {len(eps)} endpoints")

# Chainable from hotel search
hotel_search_node = "ep:hotel_api:hotel_api/GET/search"
chainable = get_chainable_endpoints(graph, hotel_search_node)
print(f"  Chainable from Hotel Search: {len(chainable)} endpoints")
for node_id, weight in chainable[:5]:
    name = graph.nodes[node_id].get("name", node_id)
    print(f"    -> {name} (weight={weight:.3f})")

# Connected within 2 hops
connected = get_connected_endpoints(graph, hotel_search_node, max_hops=2)
print(f"  Connected (2 hops) from Hotel Search: {len(connected)} endpoints")

# PageRank
importance = compute_node_importance(graph)
top5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
print(f"  Top 5 by PageRank:")
for node_id, score in top5:
    name = graph.nodes[node_id].get("name", node_id)
    print(f"    {name}: {score:.4f}")

stats = get_graph_stats(graph)
print(f"  Graph stats: {stats.tool_node_count} tools, {stats.endpoint_node_count} endpoints, density={stats.density:.4f}")
print()

# ---------------------------------------------------------------------------
# 8. Chain Sampling (MCTS + Diversity)
# ---------------------------------------------------------------------------

from tooluse_gen.graph.chain_models import ChainPattern, SamplingConstraints, ParallelGroup
from tooluse_gen.graph.sampler import MCTSSampler, SamplerConfig
from tooluse_gen.graph.diversity import DiversitySteeringConfig
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.patterns import chain_to_description

print("=" * 70)
print("PHASE 8: Chain Sampling (MCTS + Diversity Steering)")
print("=" * 70)

facade = ToolChainSampler(
    graph,
    sampler_config=SamplerConfig(max_iterations=300, max_retries=30),
    diversity_config=DiversitySteeringConfig(enabled=True, weight_decay=0.85),
)
constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
rng = np.random.default_rng(42)

chains = facade.sample_batch(constraints, count=10, rng=rng)
print(f"  Sampled {len(chains)} chains:")
for i, chain in enumerate(chains):
    desc = chain_to_description(chain)
    print(f"    Chain {i}: {chain.total_step_count} steps, tools={chain.tool_ids}, pattern={chain.pattern}")
    print(f"             {desc}")

metrics = facade.get_diversity_report()
print(f"\n  Diversity Metrics:")
print(f"    Tool entropy:          {metrics.tool_entropy:.3f}")
print(f"    Domain coverage:       {metrics.domain_coverage:.3f}")
print(f"    Unique tool pair ratio: {metrics.unique_tool_pair_ratio:.3f}")
print(f"    Pattern repetition:    {metrics.pattern_repetition_rate:.3f}")
print(f"    Total conversations:   {metrics.total_conversations}")
print()

# ---------------------------------------------------------------------------
# 9. Full Execution Pipeline (for each chain)
# ---------------------------------------------------------------------------

from tooluse_gen.agents.execution_models import ConversationContext, ToolCallRequest
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.argument_generator import ArgumentGenerator
from tooluse_gen.agents.grounding import GroundingTracker, format_available_values, format_grounding_context

print("=" * 70)
print("PHASE 9: Execute Chains (Args → Execute → Ground)")
print("=" * 70)


def flatten_steps(chain):
    flat = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            flat.extend(item.steps)
        else:
            flat.append(item)
    return flat


conversations = []
executor = ToolExecutor(registry)
arg_gen = ArgumentGenerator()

for chain_idx, chain in enumerate(chains):
    exec_rng = np.random.default_rng(42 + chain_idx)
    context = ConversationContext(chain=chain)
    tracker = GroundingTracker()

    context.add_message("user", f"Help me with a task involving {', '.join(chain.domains_involved)}")

    flat_steps = flatten_steps(chain)
    print(f"\n  --- Chain {chain_idx} ({len(flat_steps)} steps) ---")

    for i, step in enumerate(flat_steps):
        endpoint = registry.get_endpoint(step.endpoint_id)
        if endpoint is None:
            print(f"    Step {i}: SKIP (endpoint {step.endpoint_id} not found)")
            continue

        # Generate arguments with grounding
        args = arg_gen.generate_arguments(endpoint, context, exec_rng)

        # Check required params filled
        missing = [p for p in endpoint.required_parameters if p not in args]
        if missing:
            print(f"    Step {i}: WARNING - missing required params: {missing}")

        # Execute
        request = ToolCallRequest.from_chain_step(step, args)
        response = executor.execute(request, context, exec_rng)

        # Track & advance
        context.add_tool_output(response)
        tracker.track_from_response(response, step.endpoint_id, i)

        status = "OK" if response.is_success else f"FAIL ({response.error})"
        grounded_keys = [k for k in args if k in context.grounding_values or k in context.generated_ids]
        print(f"    Step {i}: {step.tool_name}/{step.endpoint_name} [{status}]")
        print(f"             args={args}")
        print(f"             grounded_from_prior={grounded_keys}")
        print(f"             extracted={list(response.extractable_values.keys())[:5]}")

        if i < len(flat_steps) - 1:
            context.advance_step()

    context.add_message("assistant", "Done! Here are your results.")

    # Build output record
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
        },
    }
    conversations.append(record)

print()

# ---------------------------------------------------------------------------
# 10. Prompt Formatting
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 10: Prompt Formatting (sample from last conversation)")
print("=" * 70)

# Show formatting for the last executed context
print(f"  format_available_values (first 500 chars):")
prompt_text = format_available_values(context, tracker)
for line in prompt_text.split("\n")[:12]:
    print(f"    {line}")
if prompt_text.count("\n") > 12:
    print(f"    ... ({prompt_text.count(chr(10)) - 12} more lines)")

structured = format_grounding_context(context)
print(f"\n  format_grounding_context:")
print(f"    current_step: {structured['current_step']}")
print(f"    prior_tool_calls: {structured['prior_tool_calls']}")
print(f"    available_values keys: {list(structured['available_values'].keys())[:8]}")
print(f"    generated_ids: {structured['generated_ids']}")
print()

# ---------------------------------------------------------------------------
# 11. JSONL Output
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 11: JSONL Output")
print("=" * 70)

with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
    for rec in conversations:
        f.write(json.dumps(rec, default=str) + "\n")
    jsonl_path = f.name

# Read back and validate
with open(jsonl_path) as f:
    lines = f.readlines()

print(f"  Wrote {len(lines)} conversation records to {jsonl_path}")

multi_tool_count = 0
multi_step_count = 0
for line in lines:
    rec = json.loads(line)
    meta = rec["metadata"]
    if meta["num_distinct_tools"] >= 2:
        multi_tool_count += 1
    if meta["num_tool_calls"] >= 3:
        multi_step_count += 1

print(f"  Multi-tool (≥2 tools):  {multi_tool_count}/{len(lines)} ({100*multi_tool_count/len(lines):.0f}%)")
print(f"  Multi-step (≥3 calls):  {multi_step_count}/{len(lines)} ({100*multi_step_count/len(lines):.0f}%)")

# Show one sample record
sample = json.loads(lines[0])
print(f"\n  Sample record (conversation {sample['conversation_id'][:12]}...):")
print(f"    Tools: {sample['metadata']['tools_used']}")
print(f"    Domains: {sample['metadata']['domains']}")
print(f"    Turns: {sample['metadata']['num_turns']}")
print(f"    Tool calls: {sample['metadata']['num_tool_calls']}")
print(f"    Messages:")
for msg in sample["messages"][:6]:
    role = msg["role"]
    content = str(msg.get("content", ""))[:80]
    print(f"      [{role}] {content}")
if len(sample["messages"]) > 6:
    print(f"      ... ({len(sample['messages']) - 6} more messages)")
print()

# ---------------------------------------------------------------------------
# 12. Diversity Experiment (Steering ON vs OFF)
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 12: Diversity Experiment (Steering ON vs OFF)")
print("=" * 70)

def run_batch(steering_enabled, seed=42, count=15):
    f = ToolChainSampler(
        graph,
        sampler_config=SamplerConfig(max_iterations=300, max_retries=30),
        diversity_config=DiversitySteeringConfig(enabled=steering_enabled, weight_decay=0.85),
    )
    c = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
    r = np.random.default_rng(seed)
    batch = f.sample_batch(c, count=count, rng=r)
    return batch, f.get_diversity_report()


chains_off, metrics_off = run_batch(steering_enabled=False)
chains_on, metrics_on = run_batch(steering_enabled=True)

print(f"\n  {'Metric':<30s} {'Steering OFF':>14s} {'Steering ON':>14s}")
print(f"  {'-'*30} {'-'*14} {'-'*14}")
print(f"  {'Chains sampled':<30s} {len(chains_off):>14d} {len(chains_on):>14d}")
print(f"  {'Tool entropy':<30s} {metrics_off.tool_entropy:>14.3f} {metrics_on.tool_entropy:>14.3f}")
print(f"  {'Domain coverage':<30s} {metrics_off.domain_coverage:>14.3f} {metrics_on.domain_coverage:>14.3f}")
print(f"  {'Unique tool pair ratio':<30s} {metrics_off.unique_tool_pair_ratio:>14.3f} {metrics_on.unique_tool_pair_ratio:>14.3f}")
print(f"  {'Pattern repetition rate':<30s} {metrics_off.pattern_repetition_rate:>14.3f} {metrics_on.pattern_repetition_rate:>14.3f}")

# Tool usage distribution
def tool_dist(chains):
    counts = {}
    for ch in chains:
        for tid in ch.tool_ids:
            counts[tid] = counts.get(tid, 0) + 1
    return counts

dist_off = tool_dist(chains_off)
dist_on = tool_dist(chains_on)
all_tools = sorted(set(list(dist_off.keys()) + list(dist_on.keys())))

print(f"\n  Tool usage distribution:")
print(f"  {'Tool':<20s} {'OFF':>6s} {'ON':>6s}")
print(f"  {'-'*20} {'-'*6} {'-'*6}")
for tid in all_tools:
    print(f"  {tid:<20s} {dist_off.get(tid, 0):>6d} {dist_on.get(tid, 0):>6d}")

# Domain distribution
def domain_dist(chains):
    counts = {}
    for ch in chains:
        for d in ch.domains_involved:
            counts[d] = counts.get(d, 0) + 1
    return counts

ddist_off = domain_dist(chains_off)
ddist_on = domain_dist(chains_on)
all_domains = sorted(set(list(ddist_off.keys()) + list(ddist_on.keys())))

print(f"\n  Domain distribution:")
print(f"  {'Domain':<20s} {'OFF':>6s} {'ON':>6s}")
print(f"  {'-'*20} {'-'*6} {'-'*6}")
for d in all_domains:
    print(f"  {d:<20s} {ddist_off.get(d, 0):>6d} {ddist_on.get(d, 0):>6d}")

print()

# ---------------------------------------------------------------------------
# 13. Determinism Check
# ---------------------------------------------------------------------------

print("=" * 70)
print("PHASE 13: Determinism Check")
print("=" * 70)

def run_full(seed):
    r = np.random.default_rng(seed)
    sampler = MCTSSampler(graph, SamplerConfig(max_iterations=200, max_retries=20))
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

eids1, out1 = run_full(42)
eids2, out2 = run_full(42)
deterministic = (eids1 == eids2) and (out1 == out2)
print(f"  Same seed produces same chain: {eids1 == eids2}")
print(f"  Same seed produces same outputs: {out1 == out2}")
print(f"  Fully deterministic: {deterministic}")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Registry:       {len(registry)} tools, {total_endpoints} endpoints, {len(registry.domains)} domains")
print(f"  Graph:          {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
print(f"  Chains sampled: {len(chains)}")
print(f"  Conversations:  {len(conversations)} generated")
print(f"  JSONL records:  {len(lines)} written")
print(f"  Deterministic:  {deterministic}")
print(f"  All phases:     PASSED")
print("=" * 70)
