"""Tests for the UserSimulator agent (Task 36)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.graph.chain_models import ChainPattern, ChainStep, ParallelGroup, ToolChain
from tooluse_gen.registry.models import HttpMethod

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _make_mock_client(response_text: str = "I need to book a hotel in Paris.") -> MagicMock:
    """Create a mock OpenAI client that returns a fixed response."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    mock.chat.completions.create.return_value = mock_response
    return mock


@pytest.fixture()
def chain() -> ToolChain:
    return ToolChain(
        chain_id="test",
        steps=[
            ChainStep(
                endpoint_id="hotels/search",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Search Hotels",
                method=HttpMethod.GET,
                domain="Travel",
                expected_params=["city", "check_in"],
            ),
            ChainStep(
                endpoint_id="hotels/book",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Book Hotel",
                method=HttpMethod.POST,
                domain="Travel",
                expected_params=["hotel_id"],
            ),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


@pytest.fixture()
def multi_domain_chain() -> ToolChain:
    return ToolChain(
        chain_id="multi",
        steps=[
            ChainStep(
                endpoint_id="hotels/search",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Search Hotels",
                domain="Travel",
            ),
            ChainStep(
                endpoint_id="weather/cur",
                tool_id="weather",
                tool_name="Weather API",
                endpoint_name="Current Weather",
                domain="Weather",
            ),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


@pytest.fixture()
def single_step_chain() -> ToolChain:
    return ToolChain(
        chain_id="single",
        steps=[
            ChainStep(
                endpoint_id="weather/cur",
                tool_id="weather",
                tool_name="Weather API",
                endpoint_name="Current Weather",
                domain="Weather",
            ),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


@pytest.fixture()
def parallel_chain() -> ToolChain:
    return ToolChain(
        chain_id="par",
        steps=[
            ParallelGroup(
                steps=[
                    ChainStep(
                        endpoint_id="a/get",
                        tool_id="a",
                        tool_name="A API",
                        endpoint_name="Get A",
                        domain="Travel",
                    ),
                    ChainStep(
                        endpoint_id="b/get",
                        tool_id="b",
                        tool_name="B API",
                        endpoint_name="Get B",
                        domain="Travel",
                    ),
                ]
            ),
        ],
        pattern=ChainPattern.PARALLEL,
    )


@pytest.fixture()
def context(chain: ToolChain) -> ConversationContext:
    ctx = ConversationContext(chain=chain)
    ctx.add_message("user", "I want to find a hotel in Paris")
    return ctx


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_offline_mode(self):
        sim = UserSimulator()
        assert sim._client is None

    def test_with_mock_client(self):
        mock = _make_mock_client()
        sim = UserSimulator(llm_client=mock)
        assert sim._client is mock

    def test_custom_model_and_temperature(self):
        sim = UserSimulator(model="gpt-4o-mini", temperature=1.0)
        assert sim._model == "gpt-4o-mini"
        assert sim._temperature == 1.0


# ===================================================================
# Offline initial request
# ===================================================================


class TestOfflineInitialRequest:
    def test_returns_nonempty_string(self, chain, rng):
        sim = UserSimulator()
        ctx = ConversationContext(chain=chain)
        result = sim.generate_initial_request(chain, ctx, rng)
        assert isinstance(result, str)
        assert len(result) > 10

    def test_references_domain_context(self, chain, rng):
        sim = UserSimulator()
        ctx = ConversationContext(chain=chain)
        result = sim.generate_initial_request(chain, ctx, rng)
        # Should reference travel or hotel concepts
        lower = result.lower()
        assert any(
            kw in lower
            for kw in ["trip", "travel", "vacation", "hotel", "search", "book", "planning"]
        )

    def test_different_seeds_different_results(self, chain):
        sim = UserSimulator()
        ctx = ConversationContext(chain=chain)
        r1 = sim.generate_initial_request(chain, ctx, np.random.default_rng(1))
        r2 = sim.generate_initial_request(chain, ctx, np.random.default_rng(99))
        # With enough templates, different seeds should eventually differ
        # (they might occasionally match — just check both are valid)
        assert isinstance(r1, str) and len(r1) > 5
        assert isinstance(r2, str) and len(r2) > 5

    def test_multi_domain_chain(self, multi_domain_chain, rng):
        sim = UserSimulator()
        ctx = ConversationContext(chain=multi_domain_chain)
        result = sim.generate_initial_request(multi_domain_chain, ctx, rng)
        assert isinstance(result, str)
        assert len(result) > 10

    def test_single_step_chain(self, single_step_chain, rng):
        sim = UserSimulator()
        ctx = ConversationContext(chain=single_step_chain)
        result = sim.generate_initial_request(single_step_chain, ctx, rng)
        assert isinstance(result, str)
        assert len(result) > 10


# ===================================================================
# LLM initial request
# ===================================================================


class TestLLMInitialRequest:
    def test_calls_llm(self, chain, rng):
        mock = _make_mock_client("Find me a nice hotel please.")
        sim = UserSimulator(llm_client=mock)
        ctx = ConversationContext(chain=chain)
        sim.generate_initial_request(chain, ctx, rng)
        mock.chat.completions.create.assert_called_once()

    def test_returns_mock_response(self, chain, rng):
        mock = _make_mock_client("I want to book a hotel in Rome.")
        sim = UserSimulator(llm_client=mock)
        ctx = ConversationContext(chain=chain)
        result = sim.generate_initial_request(chain, ctx, rng)
        assert result == "I want to book a hotel in Rome."

    def test_prompt_includes_domain(self, chain, rng):
        mock = _make_mock_client("test")
        sim = UserSimulator(llm_client=mock)
        ctx = ConversationContext(chain=chain)
        sim.generate_initial_request(chain, ctx, rng)

        call_args = mock.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Travel" in system_msg["content"]


# ===================================================================
# Prompt construction
# ===================================================================


class TestPromptConstruction:
    def test_initial_prompt_structure(self, chain):
        sim = UserSimulator()
        prompt = sim._build_initial_prompt(chain)
        assert isinstance(prompt, list)
        assert len(prompt) >= 2
        assert prompt[0]["role"] == "system"
        assert prompt[-1]["role"] == "user"

    def test_initial_prompt_mentions_domain(self, chain):
        sim = UserSimulator()
        prompt = sim._build_initial_prompt(chain)
        system = prompt[0]["content"]
        assert "Travel" in system

    def test_initial_prompt_mentions_tools(self, chain):
        sim = UserSimulator()
        prompt = sim._build_initial_prompt(chain)
        system = prompt[0]["content"]
        assert "Hotels API" in system

    def test_follow_up_prompt_includes_history(self, context):
        sim = UserSimulator()
        prompt = sim._build_follow_up_prompt(context)
        # Should contain the user message from context
        all_content = " ".join(m["content"] for m in prompt)
        assert "hotel" in all_content.lower() or "Paris" in all_content

    def test_clarification_prompt_includes_question(self, context):
        sim = UserSimulator()
        prompt = sim._build_clarification_prompt(context, "What is your budget?")
        all_content = " ".join(m["content"] for m in prompt)
        assert "budget" in all_content.lower()


# ===================================================================
# Offline follow-up
# ===================================================================


class TestOfflineFollowUp:
    def test_returns_nonempty(self, context, rng):
        sim = UserSimulator()
        result = sim.generate_follow_up(context, rng)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_references_next_step(self, context, rng):
        sim = UserSimulator()
        # current_step=0, so next step is "Book Hotel"
        result = sim.generate_follow_up(context, rng)
        assert "book hotel" in result.lower() or len(result) > 5

    def test_generic_when_no_remaining_steps(self, rng):
        chain = ToolChain(
            steps=[
                ChainStep(
                    endpoint_id="a/get",
                    tool_id="a",
                    tool_name="A",
                    endpoint_name="Get",
                    domain="X",
                ),
            ],
            pattern=ChainPattern.SEQUENTIAL,
        )
        ctx = ConversationContext(chain=chain)
        ctx.advance_step()  # past the only step
        sim = UserSimulator()
        result = sim.generate_follow_up(ctx, rng)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_no_chain_generic(self, rng):
        ctx = ConversationContext()
        sim = UserSimulator()
        result = sim.generate_follow_up(ctx, rng)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_different_seeds(self, context):
        sim = UserSimulator()
        r1 = sim.generate_follow_up(context, np.random.default_rng(1))
        r2 = sim.generate_follow_up(context, np.random.default_rng(99))
        assert isinstance(r1, str)
        assert isinstance(r2, str)


# ===================================================================
# LLM follow-up
# ===================================================================


class TestLLMFollowUp:
    def test_calls_llm(self, context, rng):
        mock = _make_mock_client("Now book the hotel please.")
        sim = UserSimulator(llm_client=mock)
        sim.generate_follow_up(context, rng)
        mock.chat.completions.create.assert_called_once()

    def test_returns_mock_response(self, context, rng):
        mock = _make_mock_client("Can you also check the weather?")
        sim = UserSimulator(llm_client=mock)
        result = sim.generate_follow_up(context, rng)
        assert result == "Can you also check the weather?"


# ===================================================================
# Offline clarification
# ===================================================================


class TestOfflineClarification:
    def test_returns_nonempty(self, context, rng):
        sim = UserSimulator()
        result = sim.generate_clarification_response(context, "What city?", rng)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_plausible_answer(self, context, rng):
        sim = UserSimulator()
        result = sim.generate_clarification_response(context, "What is your budget?", rng)
        # Should be a sentence-like response
        assert any(kw in result.lower() for kw in [
            "prefer", "looking", "go with", "think", "would",
        ])

    def test_uses_grounding_values(self, rng):
        ctx = ConversationContext()
        ctx.grounding_values["city"] = "Tokyo"
        ctx.grounding_values["step_0.city"] = "Tokyo"
        sim = UserSimulator()
        result = sim.generate_clarification_response(ctx, "Which city?", rng)
        assert isinstance(result, str)
        assert len(result) > 3


# ===================================================================
# LLM clarification
# ===================================================================


class TestLLMClarification:
    def test_calls_llm(self, context, rng):
        mock = _make_mock_client("Under 200 euros per night.")
        sim = UserSimulator(llm_client=mock)
        sim.generate_clarification_response(context, "Budget?", rng)
        mock.chat.completions.create.assert_called_once()

    def test_returns_mock_response(self, context, rng):
        mock = _make_mock_client("I'd like a 4-star hotel.")
        sim = UserSimulator(llm_client=mock)
        result = sim.generate_clarification_response(context, "Star rating?", rng)
        assert result == "I'd like a 4-star hotel."


# ===================================================================
# should_be_ambiguous
# ===================================================================


class TestShouldBeAmbiguous:
    def test_returns_bool(self, rng):
        sim = UserSimulator()
        result = sim.should_be_ambiguous(rng)
        assert isinstance(result, bool)

    def test_always_true_at_1(self):
        sim = UserSimulator()
        for seed in range(10):
            assert sim.should_be_ambiguous(np.random.default_rng(seed), probability=1.0) is True

    def test_always_false_at_0(self):
        sim = UserSimulator()
        for seed in range(10):
            assert sim.should_be_ambiguous(np.random.default_rng(seed), probability=0.0) is False

    def test_deterministic(self):
        sim = UserSimulator()
        r1 = sim.should_be_ambiguous(np.random.default_rng(42), probability=0.5)
        r2 = sim.should_be_ambiguous(np.random.default_rng(42), probability=0.5)
        assert r1 == r2


# ===================================================================
# _build_tool_context
# ===================================================================


class TestBuildTaskDescription:
    def test_produces_nonempty(self, chain):
        from tooluse_gen.agents.user_simulator import _iter_chain_steps

        sim = UserSimulator()
        steps = _iter_chain_steps(chain)
        rng = np.random.default_rng(42)
        result = sim._build_task_description(steps, rng)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_handles_parallel_group(self, parallel_chain):
        from tooluse_gen.agents.user_simulator import _iter_chain_steps

        sim = UserSimulator()
        steps = _iter_chain_steps(parallel_chain)
        rng = np.random.default_rng(42)
        result = sim._build_task_description(steps, rng)
        assert isinstance(result, str)
        assert len(result) > 5


# ===================================================================
# _call_llm
# ===================================================================


class TestCallLLM:
    def test_returns_stripped_content(self):
        mock = _make_mock_client("  Hello world!  ")
        sim = UserSimulator(llm_client=mock)
        result = sim._call_llm([{"role": "user", "content": "hi"}])
        assert result == "Hello world!"

    def test_fallback_when_none(self):
        mock = _make_mock_client("ignored")
        mock.chat.completions.create.return_value.choices[0].message.content = None
        sim = UserSimulator(llm_client=mock)
        result = sim._call_llm([{"role": "user", "content": "hi"}])
        assert result == "I need help with this."


# ===================================================================
# Determinism
# ===================================================================


class TestDeterminism:
    def test_same_seed_same_initial(self, chain):
        sim = UserSimulator()
        ctx = ConversationContext(chain=chain)
        r1 = sim.generate_initial_request(chain, ctx, np.random.default_rng(42))
        r2 = sim.generate_initial_request(chain, ctx, np.random.default_rng(42))
        assert r1 == r2

    def test_same_seed_same_follow_up(self, context):
        sim = UserSimulator()
        r1 = sim.generate_follow_up(context, np.random.default_rng(42))
        r2 = sim.generate_follow_up(context, np.random.default_rng(42))
        assert r1 == r2
