"""End-to-end test exercising the full pipeline built in tasks 1–34.

Pipeline flow:
    1. Create realistic tools with endpoints/parameters → ToolRegistry
    2. Build tool graph (nodes, edges, embeddings) → nx.DiGraph
    3. Sample tool chains via MCTS → ToolChain
    4. Execute chains: generate arguments → mock execute → extract values
    5. Verify grounding propagation across steps
    6. Verify diversity steering across a batch
    7. Verify serialization round-trips
    8. Verify formatting for downstream prompt injection
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterLocation,
    ParameterType,
    ResponseSchema,
    Tool,
)
from tooluse_gen.registry.completeness import CompletenessCalculator, QualityTier
from tooluse_gen.registry.registry import ToolRegistry
from tooluse_gen.registry.serialization import save_registry, load_registry

from tooluse_gen.graph.models import GraphConfig
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.builder import GraphBuilder
from tooluse_gen.graph.persistence import save_graph, load_graph
from tooluse_gen.graph.queries import (
    get_endpoints_for_tool,
    get_chainable_endpoints,
    get_domain_endpoints,
    get_graph_stats,
)
from tooluse_gen.graph.chain_models import ChainPattern, SamplingConstraints
from tooluse_gen.graph.sampler import MCTSSampler, SamplerConfig
from tooluse_gen.graph.diversity import (
    DiversitySteeringConfig,
    DiversityTracker,
    build_steering_prompt,
)
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.patterns import chain_to_description

from tooluse_gen.agents.execution_models import (
    ConversationContext,
    ToolCallRequest,
)
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.argument_generator import ArgumentGenerator
from tooluse_gen.agents.value_generator import ValuePool, SchemaBasedGenerator
from tooluse_gen.agents.grounding import (
    GroundingTracker,
    format_available_values,
    format_grounding_context,
)

from tooluse_gen.graph.chain_models import ParallelGroup
from tooluse_gen.core.config import load_config
from tooluse_gen.utils.seeding import set_global_seed


def _flatten_steps(chain):
    """Flatten chain steps, expanding ParallelGroups into individual ChainSteps."""
    flat = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            flat.extend(item.steps)
        else:
            flat.append(item)
    return flat


# ---------------------------------------------------------------------------
# Fixtures: build a realistic multi-domain tool registry
# ---------------------------------------------------------------------------


def _build_travel_tool() -> Tool:
    """Travel API with search and booking endpoints."""
    return Tool(
        tool_id="travel_api",
        name="Travel API",
        description="Search and book hotels, flights, and car rentals",
        domain="Travel",
        categories=["travel", "booking", "hotels"],
        base_url="https://api.travel.example.com",
        auth_type="apikey",
        endpoints=[
            Endpoint(
                endpoint_id="travel_api/GET/search_hotels",
                tool_id="travel_api",
                name="Search Hotels",
                description="Search for available hotels in a city",
                method=HttpMethod.GET,
                path="/hotels/search",
                parameters=[
                    Parameter(
                        name="city",
                        description="City to search",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.QUERY,
                        required=True,
                    ),
                    Parameter(
                        name="max_price",
                        description="Max price per night",
                        param_type=ParameterType.NUMBER,
                        location=ParameterLocation.QUERY,
                        required=False,
                    ),
                    Parameter(
                        name="check_in",
                        description="Check-in date",
                        param_type=ParameterType.DATE,
                        location=ParameterLocation.QUERY,
                        required=False,
                    ),
                ],
                required_parameters=["city"],
                response_schema=ResponseSchema(
                    status_code=200,
                    schema_type=ParameterType.OBJECT,
                    properties={
                        "results": {"type": "array", "description": "List of hotels"},
                    },
                ),
            ),
            Endpoint(
                endpoint_id="travel_api/POST/book_hotel",
                tool_id="travel_api",
                name="Book Hotel",
                description="Book a hotel room",
                method=HttpMethod.POST,
                path="/hotels/book",
                parameters=[
                    Parameter(
                        name="hotel_id",
                        description="Hotel identifier",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.BODY,
                        required=True,
                    ),
                    Parameter(
                        name="check_in",
                        description="Check-in date",
                        param_type=ParameterType.DATE,
                        location=ParameterLocation.BODY,
                        required=True,
                    ),
                    Parameter(
                        name="guest_name",
                        description="Guest full name",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.BODY,
                        required=True,
                    ),
                ],
                required_parameters=["hotel_id", "check_in", "guest_name"],
            ),
            Endpoint(
                endpoint_id="travel_api/GET/hotel_reviews",
                tool_id="travel_api",
                name="Hotel Reviews",
                description="Get reviews for a hotel",
                method=HttpMethod.GET,
                path="/hotels/{hotel_id}/reviews",
                parameters=[
                    Parameter(
                        name="hotel_id",
                        description="Hotel identifier",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.PATH,
                        required=True,
                    ),
                ],
                required_parameters=["hotel_id"],
            ),
        ],
    )


def _build_weather_tool() -> Tool:
    """Weather API with current and forecast endpoints."""
    return Tool(
        tool_id="weather_api",
        name="Weather API",
        description="Get current weather and forecasts",
        domain="Weather",
        categories=["weather", "forecast"],
        base_url="https://api.weather.example.com",
        endpoints=[
            Endpoint(
                endpoint_id="weather_api/GET/current",
                tool_id="weather_api",
                name="Current Weather",
                description="Get current weather for a city",
                method=HttpMethod.GET,
                path="/weather/current",
                parameters=[
                    Parameter(
                        name="city",
                        description="City name",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.QUERY,
                        required=True,
                    ),
                ],
                required_parameters=["city"],
            ),
            Endpoint(
                endpoint_id="weather_api/GET/forecast",
                tool_id="weather_api",
                name="Weather Forecast",
                description="Get weather forecast for a city",
                method=HttpMethod.GET,
                path="/weather/forecast",
                parameters=[
                    Parameter(
                        name="city",
                        description="City name",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.QUERY,
                        required=True,
                    ),
                    Parameter(
                        name="days",
                        description="Number of forecast days",
                        param_type=ParameterType.INTEGER,
                        location=ParameterLocation.QUERY,
                        required=False,
                        default=5,
                    ),
                ],
                required_parameters=["city"],
            ),
        ],
    )


def _build_finance_tool() -> Tool:
    """Finance API with stock and currency endpoints."""
    return Tool(
        tool_id="finance_api",
        name="Finance API",
        description="Stock quotes and currency exchange rates",
        domain="Finance",
        categories=["finance", "stocks", "currency"],
        base_url="https://api.finance.example.com",
        endpoints=[
            Endpoint(
                endpoint_id="finance_api/GET/stock_quote",
                tool_id="finance_api",
                name="Stock Quote",
                description="Get current stock price",
                method=HttpMethod.GET,
                path="/stocks/{symbol}",
                parameters=[
                    Parameter(
                        name="symbol",
                        description="Stock ticker symbol",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.PATH,
                        required=True,
                    ),
                ],
                required_parameters=["symbol"],
            ),
            Endpoint(
                endpoint_id="finance_api/GET/exchange_rate",
                tool_id="finance_api",
                name="Exchange Rate",
                description="Get currency exchange rate",
                method=HttpMethod.GET,
                path="/exchange",
                parameters=[
                    Parameter(
                        name="from_currency",
                        description="Source currency",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.QUERY,
                        required=True,
                    ),
                    Parameter(
                        name="to_currency",
                        description="Target currency",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.QUERY,
                        required=True,
                    ),
                ],
                required_parameters=["from_currency", "to_currency"],
            ),
        ],
    )


def _build_food_tool() -> Tool:
    """Food delivery API with search and order endpoints."""
    return Tool(
        tool_id="food_api",
        name="Food Delivery API",
        description="Search restaurants and place food orders",
        domain="Food",
        categories=["food", "delivery", "restaurants"],
        base_url="https://api.food.example.com",
        endpoints=[
            Endpoint(
                endpoint_id="food_api/GET/search_restaurants",
                tool_id="food_api",
                name="Search Restaurants",
                description="Search for restaurants near a location",
                method=HttpMethod.GET,
                path="/restaurants/search",
                parameters=[
                    Parameter(
                        name="city",
                        description="City name",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.QUERY,
                        required=True,
                    ),
                    Parameter(
                        name="cuisine",
                        description="Cuisine type",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.QUERY,
                        required=False,
                    ),
                ],
                required_parameters=["city"],
            ),
            Endpoint(
                endpoint_id="food_api/POST/place_order",
                tool_id="food_api",
                name="Place Order",
                description="Place a food order",
                method=HttpMethod.POST,
                path="/orders",
                parameters=[
                    Parameter(
                        name="restaurant_id",
                        description="Restaurant ID",
                        param_type=ParameterType.STRING,
                        location=ParameterLocation.BODY,
                        required=True,
                    ),
                    Parameter(
                        name="items",
                        description="List of menu items",
                        param_type=ParameterType.ARRAY,
                        location=ParameterLocation.BODY,
                        required=True,
                    ),
                ],
                required_parameters=["restaurant_id", "items"],
            ),
        ],
    )


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    """Build a multi-domain registry with 4 tools and 9 endpoints."""
    reg = ToolRegistry()
    tools = [
        _build_travel_tool(),
        _build_weather_tool(),
        _build_finance_tool(),
        _build_food_tool(),
    ]
    # Score completeness
    calc = CompletenessCalculator()
    for tool in tools:
        tool = calc.calculate_all(tool)
    reg.add_tools(tools)
    return reg


class _MockEmbeddingService(EmbeddingService):
    """Deterministic mock embedding service for testing without model loading."""

    def __init__(self) -> None:
        # Don't call super().__init__() to avoid model loading
        self._model = None
        self._cache_dir = None

    def embed_text(self, text: str) -> list[float]:
        """Deterministic hash-based embedding."""
        h = hash(text)
        rng = np.random.default_rng(abs(h) % (2**31))
        vec = rng.standard_normal(384).tolist()
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


@pytest.fixture(scope="module")
def graph(registry):
    """Build a tool graph from the registry."""
    config = GraphConfig(
        include_tool_nodes=True,
        include_domain_edges=True,
        include_semantic_edges=True,
        similarity_threshold=0.3,  # Low threshold so hash-based embeddings produce edges
        max_edges_per_node=20,
    )
    builder = GraphBuilder(config=config, embedding_service=_MockEmbeddingService())
    return builder.build(registry)


# ---------------------------------------------------------------------------
# Phase 1: Registry Tests
# ---------------------------------------------------------------------------


class TestRegistryE2E:
    """Verify the registry was built correctly."""

    def test_tool_count(self, registry):
        assert len(registry) == 4

    def test_endpoint_count(self, registry):
        total = sum(len(t.endpoints) for t in registry.tools())
        assert total == 9

    def test_tool_lookup(self, registry):
        t = registry.get_tool("travel_api")
        assert t is not None
        assert t.domain == "Travel"

    def test_endpoint_lookup(self, registry):
        ep = registry.get_endpoint("travel_api/GET/search_hotels")
        assert ep is not None
        assert ep.name == "Search Hotels"

    def test_completeness_scored(self, registry):
        for tool in registry.tools():
            assert tool.completeness_score > 0

    def test_domain_filtering(self, registry):
        travel = registry.get_tools_by_domain("Travel")
        assert len(travel) == 1
        assert travel[0].tool_id == "travel_api"

    def test_registry_serialization_roundtrip(self, registry):
        """Save and reload the registry, verify it survives."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            save_registry(registry, path)
            loaded, metadata = load_registry(path)
            assert len(loaded) == len(registry)
            for tool in registry.tools():
                loaded_tool = loaded.get_tool(tool.tool_id)
                assert loaded_tool is not None
                assert len(loaded_tool.endpoints) == len(tool.endpoints)


# ---------------------------------------------------------------------------
# Phase 2: Graph Construction Tests
# ---------------------------------------------------------------------------


class TestGraphE2E:
    """Verify the tool graph was built correctly."""

    def test_graph_has_nodes(self, graph):
        assert graph.number_of_nodes() > 0

    def test_graph_has_tool_nodes(self, graph):
        tool_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "tool"]
        assert len(tool_nodes) == 4

    def test_graph_has_endpoint_nodes(self, graph):
        ep_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"]
        assert len(ep_nodes) == 9

    def test_graph_has_edges(self, graph):
        assert graph.number_of_edges() > 0

    def test_graph_stats(self, graph):
        stats = get_graph_stats(graph)
        assert stats.tool_node_count == 4
        assert stats.endpoint_node_count == 9

    def test_endpoint_queries(self, graph):
        eps = get_endpoints_for_tool(graph, "travel_api")
        assert len(eps) == 3

    def test_domain_endpoints(self, graph):
        eps = get_domain_endpoints(graph, "Travel")
        assert len(eps) >= 3  # All travel endpoints

    def test_chainable_endpoints(self, graph):
        chainable = get_chainable_endpoints(graph, "ep:travel_api:travel_api/GET/search_hotels")
        # Should find neighboring endpoints via domain/semantic edges
        assert len(chainable) > 0

    def test_graph_persistence_roundtrip(self, graph):
        """Save and reload the graph."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.pkl"
            save_graph(graph, path)
            loaded, metadata = load_graph(path)
            assert loaded.number_of_nodes() == graph.number_of_nodes()
            assert loaded.number_of_edges() == graph.number_of_edges()


# ---------------------------------------------------------------------------
# Phase 3: Chain Sampling Tests
# ---------------------------------------------------------------------------


class TestChainSamplingE2E:
    """Test MCTS-based chain sampling from the graph."""

    def test_sample_single_chain(self, graph):
        """Sample a single chain and verify structure."""
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
        rng = np.random.default_rng(42)

        chain = sampler.sample(constraints, rng)
        assert chain is not None
        assert len(chain.steps) >= 2
        assert len(chain.steps) <= 4
        assert chain.pattern == ChainPattern.SEQUENTIAL

    def test_sample_with_domain_constraint(self, graph):
        """Sample only from the Travel domain."""
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(
            min_steps=2, max_steps=3, min_tools=1, domains=["Travel"]
        )
        rng = np.random.default_rng(123)

        chain = sampler.sample(constraints, rng)
        assert chain is not None
        for step in chain.steps:
            assert step.domain == "Travel" or step.tool_id == "travel_api"

    def test_chain_description(self, graph):
        """Verify human-readable chain description."""
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)
        rng = np.random.default_rng(99)

        chain = sampler.sample(constraints, rng)
        desc = chain_to_description(chain)
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "->" in desc or "→" in desc or len(chain.steps) == 1

    def test_facade_batch_sampling_with_diversity(self, graph):
        """Sample a batch and verify diversity tracking."""
        facade = ToolChainSampler(
            graph,
            sampler_config=SamplerConfig(max_iterations=200, max_retries=20),
            diversity_config=DiversitySteeringConfig(enabled=True, weight_decay=0.9),
        )
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
        rng = np.random.default_rng(42)

        chains = facade.sample_batch(constraints, count=10, rng=rng)
        assert len(chains) >= 5  # Some may fail, but most should succeed

        metrics = facade.get_diversity_report()
        assert metrics.total_conversations == len(chains)
        assert metrics.tool_entropy >= 0

    def test_diversity_steering_disabled_vs_enabled(self, graph):
        """Compare batch with and without diversity steering."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)

        # Run A: disabled
        facade_a = ToolChainSampler(
            graph,
            sampler_config=SamplerConfig(max_iterations=200, max_retries=20),
            diversity_config=DiversitySteeringConfig(enabled=False),
        )
        chains_a = facade_a.sample_batch(constraints, count=8, rng=rng_a)
        metrics_a = facade_a.get_diversity_report()

        # Run B: enabled
        facade_b = ToolChainSampler(
            graph,
            sampler_config=SamplerConfig(max_iterations=200, max_retries=20),
            diversity_config=DiversitySteeringConfig(enabled=True, weight_decay=0.8),
        )
        chains_b = facade_b.sample_batch(constraints, count=8, rng=rng_b)
        metrics_b = facade_b.get_diversity_report()

        # Both should produce chains
        assert len(chains_a) >= 3
        assert len(chains_b) >= 3

        # Metrics should be computable for both
        assert metrics_a.total_conversations > 0
        assert metrics_b.total_conversations > 0


# ---------------------------------------------------------------------------
# Phase 4: Full Execution Pipeline Tests
# ---------------------------------------------------------------------------


class TestExecutionPipelineE2E:
    """Test the full execute loop: sample → generate args → execute → ground."""

    def _execute_chain(self, chain, registry, rng):
        """Run a single chain through the full execution pipeline."""
        context = ConversationContext(chain=chain)
        executor = ToolExecutor(registry)
        arg_gen = ArgumentGenerator()
        tracker = GroundingTracker()

        flat_steps = _flatten_steps(chain)
        results = []
        for i, step in enumerate(flat_steps):
            endpoint = registry.get_endpoint(step.endpoint_id)
            assert endpoint is not None, f"Endpoint {step.endpoint_id} not found in registry"

            # Generate arguments (with grounding from prior steps)
            args = arg_gen.generate_arguments(endpoint, context, rng)
            assert isinstance(args, dict)

            # All required params should be filled
            for req in endpoint.required_parameters:
                assert req in args, f"Required param '{req}' missing for {endpoint.name}"

            # Create request and execute
            request = ToolCallRequest.from_chain_step(step, args)
            assert request.endpoint_id == step.endpoint_id

            response = executor.execute(request, context, rng)
            assert response.is_success, f"Execution failed: {response.error}"
            assert isinstance(response.data, dict)

            # Track grounding and add to context
            context.add_tool_output(response)
            tracker.track_from_response(response, step.endpoint_id, i)

            results.append({
                "step": i,
                "endpoint": step.endpoint_name,
                "tool": step.tool_name,
                "args": args,
                "response_keys": list(response.data.keys()),
                "extractable": dict(response.extractable_values),
                "generated_ids": dict(response.generated_ids),
            })

            if i < len(flat_steps) - 1:
                context.advance_step()

        return context, tracker, results

    def test_single_chain_execution(self, graph, registry):
        """Execute a single sampled chain end-to-end."""
        rng = np.random.default_rng(42)
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)

        chain = sampler.sample(constraints, rng)
        context, tracker, results = self._execute_chain(chain, registry, rng)

        # Verify context accumulated data
        flat_count = len(_flatten_steps(chain))
        assert len(context.tool_outputs) == flat_count
        assert len(context.messages) >= flat_count  # At least one message per step

        # Verify grounding values accumulated
        if len(chain.steps) >= 2:
            values = context.get_available_values()
            assert len(values) > 0, "No grounding values accumulated"

    def test_grounding_propagation(self, graph, registry):
        """Verify values from step N are available in step N+1."""
        rng = np.random.default_rng(77)
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)

        chain = sampler.sample(constraints, rng)
        context, tracker, results = self._execute_chain(chain, registry, rng)

        # After first step, there should be step-prefixed grounding values
        if len(results) >= 2:
            step0_keys = [k for k in context.grounding_values if k.startswith("step_0.")]
            assert len(step0_keys) > 0 or len(context.generated_ids) > 0, \
                "No grounding values produced by step 0"

    def test_multiple_chain_executions(self, graph, registry):
        """Execute multiple chains to test variety and stability."""
        facade = ToolChainSampler(
            graph,
            sampler_config=SamplerConfig(max_iterations=200, max_retries=20),
            diversity_config=DiversitySteeringConfig(enabled=True),
        )
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
        rng = np.random.default_rng(42)

        chains = facade.sample_batch(constraints, count=5, rng=rng)

        all_tools_used = set()
        all_domains = set()
        execution_count = 0

        for chain in chains:
            exec_rng = np.random.default_rng(42 + execution_count)
            context, tracker, results = self._execute_chain(chain, registry, exec_rng)
            execution_count += 1

            for step in _flatten_steps(chain):
                all_tools_used.add(step.tool_id)
                if step.domain:
                    all_domains.add(step.domain)

        # Should use multiple tools across the batch
        assert len(all_tools_used) >= 2, f"Only used tools: {all_tools_used}"

    def test_formatting_for_prompts(self, graph, registry):
        """Verify grounding formatters produce valid output."""
        rng = np.random.default_rng(42)
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)

        chain = sampler.sample(constraints, rng)
        context, tracker, results = self._execute_chain(chain, registry, rng)

        # Human-readable format
        prompt_text = format_available_values(context, tracker)
        assert isinstance(prompt_text, str)

        # Structured format for function calling
        structured = format_grounding_context(context)
        assert isinstance(structured, dict)
        assert "available_values" in structured
        assert "current_step" in structured

    def test_conversation_context_messages(self, graph, registry):
        """Verify conversation context accumulates proper message history."""
        rng = np.random.default_rng(42)
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)

        chain = sampler.sample(constraints, rng)

        context = ConversationContext(chain=chain)
        executor = ToolExecutor(registry)
        arg_gen = ArgumentGenerator()

        # Simulate a real conversation with user/assistant/tool messages
        context.add_message("user", "Help me plan a trip")

        for i, step in enumerate(_flatten_steps(chain)):
            endpoint = registry.get_endpoint(step.endpoint_id)
            args = arg_gen.generate_arguments(endpoint, context, rng)
            request = ToolCallRequest.from_chain_step(step, args)

            context.add_message("assistant", None, tool_calls=[{
                "endpoint": step.endpoint_id,
                "arguments": args,
            }])

            response = executor.execute(request, context, rng)
            context.add_tool_output(response)

            if i < len(_flatten_steps(chain)) - 1:
                context.advance_step()

        # Final assistant message
        context.add_message("assistant", "Here are your results!")

        history = context.get_history_for_prompt()
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles


# ---------------------------------------------------------------------------
# Phase 5: Output Format Validation
# ---------------------------------------------------------------------------


class TestOutputFormatE2E:
    """Verify generated data matches the expected JSONL output format."""

    def test_conversation_record_structure(self, graph, registry):
        """Build a conversation record matching the spec's example format."""
        rng = np.random.default_rng(42)
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)

        chain = sampler.sample(constraints, rng)
        context = ConversationContext(chain=chain)
        executor = ToolExecutor(registry)
        arg_gen = ArgumentGenerator()

        context.add_message("user", "I need help with something")

        for i, step in enumerate(_flatten_steps(chain)):
            endpoint = registry.get_endpoint(step.endpoint_id)
            args = arg_gen.generate_arguments(endpoint, context, rng)
            request = ToolCallRequest.from_chain_step(step, args)
            response = executor.execute(request, context, rng)
            context.add_tool_output(response)
            if i < len(_flatten_steps(chain)) - 1:
                context.advance_step()

        context.add_message("assistant", "Task completed.")

        # Build the output record
        record = {
            "conversation_id": context.conversation_id,
            "messages": context.get_history_for_prompt(),
            "metadata": {
                "seed": 42,
                "tools_used": list(chain.tool_ids),
                "num_turns": len(context.messages),
                "num_tool_calls": len(context.tool_outputs),
                "domains": list(chain.domains_involved),
                "pattern": chain.pattern,
            },
        }

        # Validate structure
        assert "conversation_id" in record
        assert "messages" in record
        assert "metadata" in record
        assert len(record["messages"]) >= 3  # user + tool(s) + assistant

        # Should be JSON-serializable
        json_str = json.dumps(record, default=str)
        parsed = json.loads(json_str)
        assert parsed["conversation_id"] == context.conversation_id

        # Metadata checks
        assert len(record["metadata"]["tools_used"]) >= 1
        assert record["metadata"]["num_tool_calls"] >= 2

    def test_batch_jsonl_output(self, graph, registry):
        """Generate a batch and write to JSONL, simulating the generate command."""
        facade = ToolChainSampler(
            graph,
            sampler_config=SamplerConfig(max_iterations=200, max_retries=20),
            diversity_config=DiversitySteeringConfig(enabled=True),
        )
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)
        rng = np.random.default_rng(42)

        chains = facade.sample_batch(constraints, count=5, rng=rng)
        records = []

        for idx, chain in enumerate(chains):
            exec_rng = np.random.default_rng(42 + idx)
            context = ConversationContext(chain=chain)
            executor = ToolExecutor(registry)
            arg_gen = ArgumentGenerator()

            context.add_message("user", f"Request #{idx}")
            for i, step in enumerate(_flatten_steps(chain)):
                endpoint = registry.get_endpoint(step.endpoint_id)
                args = arg_gen.generate_arguments(endpoint, context, exec_rng)
                request = ToolCallRequest.from_chain_step(step, args)
                response = executor.execute(request, context, exec_rng)
                context.add_tool_output(response)
                if i < len(_flatten_steps(chain)) - 1:
                    context.advance_step()

            records.append({
                "conversation_id": context.conversation_id,
                "messages": context.get_history_for_prompt(),
                "metadata": {
                    "seed": 42 + idx,
                    "tools_used": list(chain.tool_ids),
                    "num_turns": len(context.messages),
                    "num_tool_calls": len(context.tool_outputs),
                },
            })

        # Write to JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
            jsonl_path = f.name

        # Read back and verify
        with open(jsonl_path) as f:
            lines = f.readlines()

        assert len(lines) == len(records)
        for line in lines:
            parsed = json.loads(line)
            assert "conversation_id" in parsed
            assert "messages" in parsed
            assert len(parsed["messages"]) >= 2


# ---------------------------------------------------------------------------
# Phase 6: Determinism & Reproducibility
# ---------------------------------------------------------------------------


class TestDeterminismE2E:
    """Verify same seed produces same output."""

    def test_deterministic_chain_sampling(self, graph):
        """Same seed → same chain."""
        config = SamplerConfig(max_iterations=200, max_retries=20)
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)

        sampler1 = MCTSSampler(graph, config)
        chain1 = sampler1.sample(constraints, np.random.default_rng(42))

        sampler2 = MCTSSampler(graph, config)
        chain2 = sampler2.sample(constraints, np.random.default_rng(42))

        assert chain1.endpoint_ids == chain2.endpoint_ids

    def test_deterministic_execution(self, graph, registry):
        """Same seed → same execution results."""
        config = SamplerConfig(max_iterations=200, max_retries=20)
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)

        def run_pipeline(seed):
            rng = np.random.default_rng(seed)
            sampler = MCTSSampler(graph, config)
            chain = sampler.sample(constraints, rng)
            context = ConversationContext(chain=chain)
            executor = ToolExecutor(registry)
            arg_gen = ArgumentGenerator()

            outputs = []
            for i, step in enumerate(_flatten_steps(chain)):
                endpoint = registry.get_endpoint(step.endpoint_id)
                args = arg_gen.generate_arguments(endpoint, context, rng)
                request = ToolCallRequest.from_chain_step(step, args)
                response = executor.execute(request, context, rng)
                context.add_tool_output(response)
                outputs.append(response.data)
                if i < len(_flatten_steps(chain)) - 1:
                    context.advance_step()
            return outputs

        run1 = run_pipeline(42)
        run2 = run_pipeline(42)
        assert run1 == run2


# ---------------------------------------------------------------------------
# Phase 7: Config & Seeding Integration
# ---------------------------------------------------------------------------


class TestConfigE2E:
    """Test configuration loading and seed management."""

    def test_default_config_loads(self):
        config = load_config()
        assert config is not None
        assert config.sampling.min_steps >= 1
        assert config.sampling.max_steps >= config.sampling.min_steps

    def test_global_seed(self):
        set_global_seed(42)
        import random
        a = random.random()
        set_global_seed(42)
        b = random.random()
        assert a == b
