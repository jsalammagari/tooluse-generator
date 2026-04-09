"""User simulator agent.

:class:`UserSimulator` generates realistic user messages — initial
requests, follow-ups, and clarification responses — to drive multi-turn
conversations.  It supports two modes:

* **LLM mode** — uses an OpenAI-compatible client for natural generation.
* **Offline mode** — uses seeded, template-based generation when no
  client is provided.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.graph.chain_models import ChainStep, ParallelGroup, ToolChain
from tooluse_gen.graph.patterns import chain_to_description
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.user_simulator")

# ---------------------------------------------------------------------------
# Domain-keyed template pools
# ---------------------------------------------------------------------------

_DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "Travel": [
        "I'm planning a trip and need help finding {tool_context}.",
        "Can you help me with travel arrangements? I need {tool_context}.",
        "I want to plan a vacation. Help me with {tool_context}.",
        "I'm traveling soon and need to {tool_context}.",
    ],
    "Weather": [
        "What's the weather like? I need {tool_context}.",
        "Can you check the weather for me? I need {tool_context}.",
        "I'd like to know the weather conditions. Please {tool_context}.",
    ],
    "Finance": [
        "I need help with some financial information. Can you {tool_context}?",
        "Help me with finance — I want to {tool_context}.",
        "I'm looking into my finances. Can you {tool_context}?",
    ],
    "Food": [
        "I'm looking for food options. Can you help me {tool_context}?",
        "Help me find somewhere to eat. I need {tool_context}.",
        "I'm hungry and need to {tool_context}.",
    ],
    "Sports": [
        "I want to check some sports info. Can you {tool_context}?",
        "Help me find sports data — I need to {tool_context}.",
    ],
    "Entertainment": [
        "I'm looking for something fun. Can you {tool_context}?",
        "Help me with entertainment options — I want to {tool_context}.",
    ],
}

_GENERIC_TEMPLATES: list[str] = [
    "I need help with something. Can you {tool_context}?",
    "Can you assist me? I'd like to {tool_context}.",
    "I need to {tool_context}. Can you help?",
]

_FOLLOW_UP_TEMPLATES: list[str] = [
    "Great, thanks! Can you also {next_action}?",
    "That looks good. Now I'd like to {next_action}.",
    "Perfect. What about {next_action}?",
    "Thanks for that. I also need to {next_action}.",
    "Awesome! Now please {next_action}.",
]

_GENERIC_FOLLOW_UP: list[str] = [
    "Can you tell me more about the results?",
    "What else can you help me with?",
    "Is there anything else you can find for me?",
    "Thanks! Any other details you can provide?",
]

_CLARIFICATION_TEMPLATES: list[str] = [
    "Sure, I'd prefer {detail}.",
    "I'm looking for {detail}.",
    "Let's go with {detail}.",
    "I think {detail} would work.",
    "I'd say {detail}.",
]

_GENERIC_DETAILS: list[str] = [
    "something affordable",
    "the most popular option",
    "whatever you recommend",
    "the best available option",
    "something highly rated",
]


# ---------------------------------------------------------------------------
# UserSimulator
# ---------------------------------------------------------------------------


class UserSimulator:
    """Generates realistic user messages via LLM to drive conversations."""

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._temperature = temperature
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_initial_request(
        self,
        chain: ToolChain,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> str:
        """Generate a natural user request for the sampled *chain*."""
        if self._client is not None:
            return self._call_llm(self._build_initial_prompt(chain))
        return self._generate_initial_offline(chain, rng)

    def generate_follow_up(
        self,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> str:
        """Generate a follow-up message based on conversation history."""
        if self._client is not None:
            return self._call_llm(self._build_follow_up_prompt(context))
        return self._generate_follow_up_offline(context, rng)

    def generate_clarification_response(
        self,
        context: ConversationContext,
        question: str,
        rng: np.random.Generator,
    ) -> str:
        """Respond to an assistant's clarification *question*."""
        if self._client is not None:
            return self._call_llm(self._build_clarification_prompt(context, question))
        return self._generate_clarification_offline(context, question, rng)

    def should_be_ambiguous(
        self,
        rng: np.random.Generator,
        probability: float = 0.3,
    ) -> bool:
        """Return ``True`` with the given *probability*."""
        return bool(rng.random() < probability)

    # ------------------------------------------------------------------
    # Prompt builders (LLM mode)
    # ------------------------------------------------------------------

    def _build_initial_prompt(self, chain: ToolChain) -> list[dict[str, str]]:
        desc = chain_to_description(chain)
        domains = ", ".join(chain.domains_involved) if chain.domains_involved else "general"
        tool_names = ", ".join(dict.fromkeys(
            s.tool_name
            for s in _iter_chain_steps(chain)
        ))

        system = (
            "You are simulating a user who needs help. "
            "Generate a single natural request (1\u20133 sentences) that would require "
            "using these tools. Do NOT mention tool names or endpoints \u2014 describe "
            "your need naturally.\n\n"
            f"Domain(s): {domains}\n"
            f"Tools available: {tool_names}\n"
            f"Tool chain: {desc}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "Generate a realistic user request."},
        ]

    def _build_follow_up_prompt(
        self, context: ConversationContext
    ) -> list[dict[str, str]]:
        history = context.get_history_for_prompt()
        values = context.get_available_values()

        system = (
            "You are simulating a user in an ongoing conversation. "
            "Based on the conversation so far and the last tool results, "
            "generate a natural follow-up request (1\u20132 sentences). "
            "Stay in character.\n\n"
            f"Available data from prior tool calls: {json.dumps(values, default=str)[:500]}"
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in history:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": "Generate a follow-up request."})
        return messages

    def _build_clarification_prompt(
        self, context: ConversationContext, question: str
    ) -> list[dict[str, str]]:
        history = context.get_history_for_prompt()

        system = (
            "You are simulating a user. The assistant asked you a clarifying "
            "question. Provide a brief, natural answer (1 sentence) with "
            "specific details.\n\n"
            f"The assistant asked: \"{question}\""
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in history:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({
            "role": "user",
            "content": "Answer the clarifying question briefly.",
        })
        return messages

    # ------------------------------------------------------------------
    # Offline generators (template-based)
    # ------------------------------------------------------------------

    def _generate_initial_offline(
        self, chain: ToolChain, rng: np.random.Generator
    ) -> str:
        tool_context = self._build_tool_context(chain)
        domains = chain.domains_involved

        # Pick a template pool based on the first domain.
        templates = _GENERIC_TEMPLATES
        if domains:
            templates = _DOMAIN_TEMPLATES.get(domains[0], _GENERIC_TEMPLATES)

        idx = int(rng.integers(0, len(templates)))
        return templates[idx].format(tool_context=tool_context)

    def _generate_follow_up_offline(
        self, context: ConversationContext, rng: np.random.Generator
    ) -> str:
        chain = context.chain
        if chain is None:
            idx = int(rng.integers(0, len(_GENERIC_FOLLOW_UP)))
            return _GENERIC_FOLLOW_UP[idx]

        flat = _iter_chain_steps(chain)
        step_idx = context.current_step + 1
        if step_idx < len(flat):
            next_action = flat[step_idx].endpoint_name.lower()
            idx = int(rng.integers(0, len(_FOLLOW_UP_TEMPLATES)))
            return _FOLLOW_UP_TEMPLATES[idx].format(next_action=next_action)

        idx = int(rng.integers(0, len(_GENERIC_FOLLOW_UP)))
        return _GENERIC_FOLLOW_UP[idx]

    def _generate_clarification_offline(
        self,
        context: ConversationContext,
        question: str,
        rng: np.random.Generator,
    ) -> str:
        # Try to pick a detail from grounding values.
        values = context.get_available_values()
        if values:
            # Pick a value that's a simple string or number.
            candidates = [
                str(v)
                for k, v in values.items()
                if not k.startswith("step_") and isinstance(v, (str, int, float)) and str(v)
            ]
            if candidates:
                detail = str(candidates[int(rng.integers(0, len(candidates)))])
                idx = int(rng.integers(0, len(_CLARIFICATION_TEMPLATES)))
                return _CLARIFICATION_TEMPLATES[idx].format(detail=detail)

        detail = _GENERIC_DETAILS[int(rng.integers(0, len(_GENERIC_DETAILS)))]
        idx = int(rng.integers(0, len(_CLARIFICATION_TEMPLATES)))
        return _CLARIFICATION_TEMPLATES[idx].format(detail=detail)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM and return the response text."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=200,
        )
        content = response.choices[0].message.content
        if content is None:
            return "I need help with this."
        return content.strip()

    def _build_tool_context(self, chain: ToolChain) -> str:
        """Lowercase natural-language summary of chain endpoint names."""
        names = [s.endpoint_name for s in _iter_chain_steps(chain)]
        return " and ".join(names).lower()


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _iter_chain_steps(chain: ToolChain) -> list[ChainStep]:
    """Flatten chain steps, expanding :class:`ParallelGroup` instances."""
    result: list[ChainStep] = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            result.extend(item.steps)
        else:
            result.append(item)
    return result
