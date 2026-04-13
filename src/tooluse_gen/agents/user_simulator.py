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
import re
from typing import Any

import numpy as np

from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.graph.chain_models import ChainStep, ParallelGroup, ToolChain
from tooluse_gen.graph.patterns import chain_to_description
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.user_simulator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_chain_steps(chain: ToolChain) -> list[ChainStep]:
    """Flatten chain into a list of :class:`ChainStep`."""
    result: list[ChainStep] = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            result.extend(item.steps)
        else:
            result.append(item)
    return result


def _humanize_name(name: str) -> str:
    """Convert an endpoint/tool name to natural language.

    'get_weather_forecast' → 'get the weather forecast'
    'SearchHotels' → 'search hotels'
    'license_plate_lookup' → 'look up a license plate'
    """
    # CamelCase → spaced
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    # Underscores/hyphens → spaces
    s = s.replace("_", " ").replace("-", " ")
    s = s.lower().strip()
    # Strip leading slashes and path-like prefixes
    s = re.sub(r"^/+", "", s)
    # Clean up common patterns
    s = re.sub(r"\bget\b", "get", s)
    s = re.sub(r"\bsearch\b", "search for", s)
    s = re.sub(r"\blookup\b", "look up", s)
    s = re.sub(r"\bfetch\b", "fetch", s)
    s = re.sub(r"\blist\b", "list", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_garbage_name(name: str) -> bool:
    """Detect names that are too garbled/short/meaningless to show to users.

    Examples of garbage: 'rererer', 'tg', '/card content', 'scott',
    'this free fier', 'getglobalachievementpercentagesforapp'
    """
    s = name.strip().lower()
    # Too short (single word < 4 chars that isn't a known verb)
    if len(s) < 4 and s not in ("get", "set", "add", "run", "put"):
        return True
    # All one word with no spaces/underscores/camelCase and very long (concatenated)
    if " " not in s and "_" not in s and "-" not in s and len(s) > 25:
        return True
    # Contains path-like patterns
    if s.startswith("/") or "//" in s:
        return True
    # Looks like gibberish (no vowels, or repeated chars)
    vowels = sum(1 for c in s if c in "aeiou")
    if len(s) > 4 and vowels < len(s) * 0.15:
        return True
    # Known garbage patterns from ToolBench
    if any(kw in s for kw in ["602917ac", "version 2", "copy", "..."]):
        return True
    return False


def _is_clean_description(desc: str) -> bool:
    """Check if a ToolBench description is clean enough to use in user messages."""
    if not desc or len(desc) < 5:
        return False
    # Reject descriptions with code/JSON fragments
    if any(c in desc for c in ["{", "}", "\"", "\\", "=", "<", ">", "//"]):
        return False
    # Reject descriptions that are clearly technical/API docs
    if any(kw in desc.lower() for kw in [
        "api", "endpoint", "http", "json", "parameter", "request",
        "response", "returns", "w...", "...", "base64", "png",
    ]):
        return False
    # Reject very long descriptions (likely copy-paste from docs)
    if len(desc) > 100:
        return False
    # Reject descriptions that start with articles/prepositions (likely sentence fragments)
    if desc.strip().startswith(("The ", "A ", "An ", "This ")):
        # These can be OK if they're short and readable
        if len(desc) > 60:
            return False
    return True


def _describe_step_naturally(step: ChainStep) -> str:
    """Build a natural description of what a chain step does."""
    name = _humanize_name(step.endpoint_name)
    tool = _humanize_name(step.tool_name)

    # Only use the description if it's genuinely clean and readable
    if step.description and _is_clean_description(step.description):
        desc = step.description.strip().rstrip(".").lower()
        if len(desc) < 50:
            return desc

    # If the endpoint name is garbage, fall back to tool name + domain
    if _is_garbage_name(step.endpoint_name):
        if step.domain:
            domain_lower = step.domain.lower().replace("_", " ")
            if _is_garbage_name(step.tool_name):
                return f"look up {domain_lower} information"
            return f"use {tool} for {domain_lower} data"
        if not _is_garbage_name(step.tool_name):
            return f"get data from {tool}"
        return "look up some information"

    # If the tool name is garbage but endpoint is OK, just use endpoint
    if _is_garbage_name(step.tool_name):
        return name

    # Both are usable — combine them
    if any(w in name for w in tool.split() if len(w) > 3):
        return name
    return f"use {tool} to {name}"


# ---------------------------------------------------------------------------
# Domain-specific request templates
# ---------------------------------------------------------------------------

# Each template takes {task_description} — a natural-language summary of what
# the user wants to accomplish (NOT raw tool/endpoint names).
_DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "Travel": [
        "I'm planning a trip and need help. Specifically, I need to {task_description}.",
        "Can you help me with travel arrangements? I want to {task_description}.",
        "I'm traveling soon. Could you {task_description} for me?",
    ],
    "Weather": [
        "Can you check the weather for me? I'd like to {task_description}.",
        "I need some weather information — could you {task_description}?",
    ],
    "Finance": [
        "I need help with some financial information. Could you {task_description}?",
        "I'm looking into finances. Can you {task_description}?",
    ],
    "Food": [
        "I'm looking for food options. Can you {task_description}?",
        "Help me find somewhere to eat — I want to {task_description}.",
    ],
    "Sports": [
        "I want to check some sports info. Can you {task_description}?",
        "I'm interested in sports data. Could you {task_description}?",
    ],
    "Entertainment": [
        "I'm looking for something fun. Can you {task_description}?",
        "Help me find some entertainment — I'd like to {task_description}.",
    ],
    "Music": [
        "I'm looking for music. Can you {task_description}?",
        "I need help finding some music — could you {task_description}?",
    ],
    "Data": [
        "I need some data. Can you {task_description}?",
        "Could you help me look up some information? I want to {task_description}.",
    ],
    "Business": [
        "I need some business information. Could you {task_description}?",
        "I'm doing some business research. Can you {task_description}?",
    ],
    "Education": [
        "I'm researching something for study purposes. Can you {task_description}?",
        "I need educational information. Could you {task_description}?",
    ],
    "Location": [
        "I need location-based information. Could you {task_description}?",
        "Can you help me find a place? I want to {task_description}.",
    ],
    "Communication": [
        "I need to send some communications. Can you {task_description}?",
        "Help me with messaging — I want to {task_description}.",
    ],
}

_GENERIC_TEMPLATES: list[str] = [
    "Hi, I need some help. Could you {task_description}?",
    "Can you assist me with something? I'd like to {task_description}.",
    "I need to {task_description}. Can you help me with that?",
    "Hey, could you help me {task_description}?",
]

_FOLLOW_UP_TEMPLATES: list[str] = [
    "Great, thanks! Now could you also {next_task}?",
    "That looks good. Next, I'd like you to {next_task}.",
    "Perfect, thanks. Can you also {next_task}?",
    "Thanks for that. I also need you to {next_task}.",
    "Awesome! Now please {next_task}.",
]

_FOLLOW_UP_WITH_CONTEXT: list[str] = [
    "Great, now using those results, could you {next_task}?",
    "Thanks! Based on that, can you {next_task}?",
    "Perfect. Now I'd like you to take those details and {next_task}.",
    "Good, that's helpful. Next, please {next_task}.",
]

_GENERIC_FOLLOW_UP: list[str] = [
    "Can you tell me more about the results?",
    "What else can you help me with?",
    "Is there anything else you can find for me?",
    "Thanks! Any other details you can provide?",
]

_DISAMBIGUATION_TEMPLATES: list[str] = [
    "Sure, I'd prefer {detail}.",
    "I'm looking for {detail}.",
    "Let's go with {detail}.",
    "I think {detail} would work.",
    "I'd say {detail}.",
]

_CLARIFICATION_TEMPLATES = _DISAMBIGUATION_TEMPLATES

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
        self._logger.debug("Generating initial request for chain: %d steps", chain.total_step_count)
        if self._client is not None:
            result = self._call_llm(self._build_initial_prompt(chain))
        else:
            result = self._generate_initial_offline(chain, rng)
        self._logger.debug("User initial msg: %s", result[:80])
        return result

    def generate_follow_up(
        self,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> str:
        """Generate a follow-up message based on conversation history."""
        self._logger.debug("Generating follow-up (step %d)", context.current_step)
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
        self._logger.debug("Clarifying: %s", question[:80])
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
            "You are simulating a user. Based on the conversation so far, "
            "generate a brief follow-up message (1 sentence). "
            "Reference specific results if available. Be natural."
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in history:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({
            "role": "user",
            "content": "Generate a follow-up request.",
        })
        return messages

    def _build_clarification_prompt(
        self, context: ConversationContext, question: str
    ) -> list[dict[str, str]]:
        history = context.get_history_for_prompt()

        system = (
            "You are simulating a user. The assistant asked you a clarifying "
            "question. Answer it briefly and naturally (1 sentence)."
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in history:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({
            "role": "assistant",
            "content": question,
        })
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
        steps = _iter_chain_steps(chain)
        task_description = self._build_task_description(steps, rng)
        domains = chain.domains_involved

        # Pick a template pool based on the first domain.
        templates = _GENERIC_TEMPLATES
        if domains:
            for d in domains:
                if d in _DOMAIN_TEMPLATES:
                    templates = _DOMAIN_TEMPLATES[d]
                    break

        idx = int(rng.integers(0, len(templates)))
        return templates[idx].format(task_description=task_description)

    def _generate_follow_up_offline(
        self, context: ConversationContext, rng: np.random.Generator
    ) -> str:
        chain = context.chain
        if chain is None:
            idx = int(rng.integers(0, len(_GENERIC_FOLLOW_UP)))
            return _GENERIC_FOLLOW_UP[idx]

        flat = _iter_chain_steps(chain)
        step_idx = context.current_step
        if step_idx < len(flat):
            next_step = flat[step_idx]
            next_task = _describe_step_naturally(next_step)

            # If there are previous tool outputs, reference them
            has_prior_results = len(context.tool_outputs) > 0
            if has_prior_results and rng.random() > 0.4:
                templates = _FOLLOW_UP_WITH_CONTEXT
            else:
                templates = _FOLLOW_UP_TEMPLATES

            idx = int(rng.integers(0, len(templates)))
            return templates[idx].format(next_task=next_task)

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

    def _build_task_description(
        self, steps: list[ChainStep], rng: np.random.Generator
    ) -> str:
        """Build a natural task description from chain steps.

        Instead of 'heroes and default and license plate lookup', produce
        something like 'search for superhero information, then look up a
        license plate'.
        """
        if not steps:
            return "help me with a task"

        # Describe each step naturally, deduplicating similar descriptions
        descriptions: list[str] = []
        seen_tools: set[str] = set()
        for step in steps:
            desc = _describe_step_naturally(step)
            # Skip near-duplicates (same tool, similar action)
            tool_key = step.tool_id
            if tool_key in seen_tools and len(descriptions) >= 2:
                continue
            seen_tools.add(tool_key)
            descriptions.append(desc)

        if len(descriptions) == 1:
            return descriptions[0]
        if len(descriptions) == 2:
            return f"{descriptions[0]} and then {descriptions[1]}"

        # For 3+ steps, mention first, a middle one, and last with connectors
        first = descriptions[0]
        middle = descriptions[1:-1]
        last = descriptions[-1]

        # Randomly decide whether to enumerate all or summarize
        if len(descriptions) <= 4 or rng.random() > 0.5:
            parts = [first]
            for m in middle:
                parts.append(f"then {m}")
            parts.append(f"and finally {last}")
            return ", ".join(parts)
        else:
            return f"{first}, then {middle[0]}, and a couple more things after that"

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
