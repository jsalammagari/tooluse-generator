"""Core tool-calling assistant agent.

:class:`AssistantAgent` follows the sampled tool chain, emitting tool
calls, disambiguation questions, or final text summaries depending on
the conversation state.  It supports two modes:

* **LLM mode** — uses an OpenAI-compatible client for natural generation.
* **Offline mode** — uses template-based generation when no client is
  provided.

:class:`AssistantResponse` is the structured output returned by the
agent on every turn.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.agents.conversation_models import GenerationConfig
from tooluse_gen.agents.execution_models import (
    ConversationContext,
    ToolCallRequest,
)
from tooluse_gen.agents.grounding import format_available_values
from tooluse_gen.graph.chain_models import ChainStep, ParallelGroup, ToolChain
from tooluse_gen.registry.registry import ToolRegistry
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.assistant_agent")

# ---------------------------------------------------------------------------
# AssistantResponse
# ---------------------------------------------------------------------------


class AssistantResponse(BaseModel):
    """Structured output from the assistant on a single turn."""

    model_config = ConfigDict(use_enum_values=True)

    content: str | None = Field(default=None, description="Text content.")
    tool_calls: list[ToolCallRequest] | None = Field(
        default=None, description="Tool invocations to make."
    )
    is_disambiguation: bool = Field(
        default=False, description="True when asking a clarifying question."
    )
    is_final_answer: bool = Field(
        default=False, description="True when this is the final summary."
    )


# ---------------------------------------------------------------------------
# Template pools
# ---------------------------------------------------------------------------

_DISAMBIGUATION_TEMPLATES: list[str] = [
    "Could you please specify {param_desc}?",
    "I need a bit more information. What {param_desc} would you prefer?",
    "Before I proceed, could you tell me {param_desc}?",
    "What {param_desc} should I use?",
]

_FINAL_ANSWER_TEMPLATES: list[str] = [
    "I've completed your request. Here's a summary of what I found: {summary}",
    "All done! {summary}",
    "Here are the results of your request. {summary}",
    "I've finished helping you. {summary}",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _flatten_steps(chain: ToolChain) -> list[ChainStep]:
    """Flatten chain steps, expanding :class:`ParallelGroup` instances."""
    result: list[ChainStep] = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            result.extend(item.steps)
        else:
            result.append(item)
    return result


_JSON_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "file": "string",
    "date": "string",
    "datetime": "string",
    "unknown": "string",
}


def _ptype_to_json(ptype: str) -> str:
    """Map a :class:`ParameterType` string value to a JSON Schema type."""
    return _JSON_TYPE_MAP.get(ptype, "string")


# ---------------------------------------------------------------------------
# AssistantAgent
# ---------------------------------------------------------------------------


class AssistantAgent:
    """Core tool-calling assistant that follows the sampled chain."""

    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: Any | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> None:
        self._registry = registry
        self._client = llm_client
        self._model = model
        self._temperature = temperature
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_response(
        self,
        context: ConversationContext,
        rng: np.random.Generator,
        gen_config: GenerationConfig | None = None,
    ) -> AssistantResponse:
        """Decide the assistant's next action and return a response."""
        chain = context.chain
        config = gen_config or GenerationConfig()

        # (a) Final answer when chain is done or absent.
        if chain is None:
            resp = self._generate_final_answer(context)
            self._logger.debug("Assistant: final answer (no chain)")
            return resp

        flat = _flatten_steps(chain)
        self._logger.debug(
            "Assistant generating response (step %d/%d)", context.current_step, len(flat),
        )

        if context.current_step >= len(flat):
            resp = self._generate_final_answer(context)
            self._logger.debug("Assistant: final answer (chain complete)")
            return resp

        # (b) Disambiguation.
        if config.include_disambiguation and self._should_disambiguate(
            context, rng, config
        ):
            resp = self._generate_disambiguation(context, rng)
            self._logger.debug("Assistant: disambiguation question")
            return resp

        # (c) Tool call for the next chain step.
        resp = self._generate_tool_call(context, rng)
        if resp.tool_calls:
            self._logger.debug(
                "Assistant: %d tool call(s): %s",
                len(resp.tool_calls),
                [tc.endpoint_id for tc in resp.tool_calls],
            )
        return resp

    # ------------------------------------------------------------------
    # Tool call generation
    # ------------------------------------------------------------------

    def _generate_tool_call(
        self,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> AssistantResponse:
        chain = context.chain
        assert chain is not None
        flat = _flatten_steps(chain)
        step = flat[context.current_step]

        endpoint = self._registry.get_endpoint(step.endpoint_id)

        if self._client is not None and endpoint is not None:
            return self._generate_tool_call_llm(context, step, endpoint, rng)

        # Offline mode — build arguments from grounding + placeholders.
        arguments: dict[str, Any] = {}
        available = context.get_available_values()
        has_prior_values = len(available) > 0

        if endpoint is not None:
            for param in endpoint.parameters:
                # Always fill required params. For optional params:
                # - If we have grounding values, fill more aggressively (80%)
                #   to demonstrate coherent chaining
                # - Otherwise, 50% chance
                fill_prob = 0.8 if has_prior_values else 0.5
                if param.required or bool(rng.random() < fill_prob):
                    arguments[param.name] = self._resolve_argument(param, context, rng)
        else:
            # Endpoint not in registry — fill expected_params with placeholders.
            for pname in step.expected_params:
                arguments[pname] = f"{pname}_value"

        request = ToolCallRequest.from_chain_step(step, arguments)
        return AssistantResponse(tool_calls=[request])

    def _generate_tool_call_llm(
        self,
        context: ConversationContext,
        step: ChainStep,
        endpoint: Any,
        rng: np.random.Generator,
    ) -> AssistantResponse:
        """Use the LLM to generate a tool call."""
        chain = context.chain
        assert chain is not None

        system = self._build_system_prompt(context)
        history = context.get_history_for_prompt()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in history:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        tools = self._build_tools_schema(chain)
        response = self._call_llm(messages, tools=tools)

        # Try to parse tool calls from the response.
        choice = response.choices[0]
        if choice.message.tool_calls:
            # Use the first tool call from the LLM.
            tc = choice.message.tool_calls[0]
            import json as _json

            try:
                args = _json.loads(tc.function.arguments)
            except Exception:
                args = {}
            request = ToolCallRequest.from_chain_step(step, args)
            return AssistantResponse(tool_calls=[request])

        # Fallback to offline if LLM didn't produce tool calls.
        return self._generate_tool_call(
            context._replace_client(None) if hasattr(context, "_replace_client") else context,
            rng,
        )

    # ------------------------------------------------------------------
    # Disambiguation
    # ------------------------------------------------------------------

    def _generate_disambiguation(
        self,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> AssistantResponse:
        chain = context.chain
        assert chain is not None
        flat = _flatten_steps(chain)
        step = flat[context.current_step]
        endpoint = self._registry.get_endpoint(step.endpoint_id)

        if self._client is not None:
            system = (
                "You are a helpful assistant. The user made a request but you need "
                "more information. Ask a brief clarifying question about the missing "
                "details (1 sentence)."
            )
            history = context.get_history_for_prompt()
            messages: list[dict[str, str]] = [{"role": "system", "content": system}]
            for msg in history:
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
            messages.append({
                "role": "user",
                "content": "Ask a clarifying question.",
            })
            resp = self._call_llm(messages)
            text = resp.choices[0].message.content
            question = text.strip() if text else "Could you provide more details?"
            return AssistantResponse(content=question, is_disambiguation=True)

        # Offline mode — build question from unresolved required params.
        available = context.get_available_values()
        unresolved: list[str] = []
        if endpoint is not None:
            for pname in endpoint.required_parameters:
                if pname not in available:
                    unresolved.append(pname)

        if not unresolved:
            unresolved = [p for p in step.expected_params if p not in available]

        if unresolved:
            param_desc = "the " + " and ".join(
                p.replace("_", " ") for p in unresolved
            )
        else:
            param_desc = "any preferences you have"

        idx = int(rng.integers(0, len(_DISAMBIGUATION_TEMPLATES)))
        question = _DISAMBIGUATION_TEMPLATES[idx].format(param_desc=param_desc)
        return AssistantResponse(content=question, is_disambiguation=True)

    # ------------------------------------------------------------------
    # Final answer
    # ------------------------------------------------------------------

    def _generate_final_answer(
        self, context: ConversationContext
    ) -> AssistantResponse:
        if self._client is not None:
            system = (
                "You are a helpful assistant. Summarize what you accomplished "
                "for the user in 1-2 sentences based on the conversation."
            )
            history = context.get_history_for_prompt()
            messages: list[dict[str, str]] = [{"role": "system", "content": system}]
            for msg in history:
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": "Summarize the results."})
            resp = self._call_llm(messages)
            text = resp.choices[0].message.content
            answer = text.strip() if text else "I've completed your request."
            return AssistantResponse(content=answer, is_final_answer=True)

        summary = self._build_summary(context)
        idx = len(context.tool_outputs) % len(_FINAL_ANSWER_TEMPLATES)
        answer = _FINAL_ANSWER_TEMPLATES[idx].format(summary=summary)
        return AssistantResponse(content=answer, is_final_answer=True)

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _should_disambiguate(
        self,
        context: ConversationContext,
        rng: np.random.Generator,
        gen_config: GenerationConfig | None = None,
    ) -> bool:
        # Only on the first step with no tool outputs yet.
        if context.current_step != 0 or len(context.tool_outputs) > 0:
            return False

        # Check if already disambiguated (assistant message containing '?').
        for msg in context.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "") or ""
                if "?" in content:
                    return False

        prob = gen_config.disambiguation_probability if gen_config else 0.3
        return bool(rng.random() < prob)

    # ------------------------------------------------------------------
    # Prompt / schema builders
    # ------------------------------------------------------------------

    def _build_system_prompt(self, context: ConversationContext) -> str:
        chain = context.chain
        tool_descriptions: list[str] = []
        if chain is not None:
            seen: set[str] = set()
            for step in _flatten_steps(chain):
                if step.endpoint_id in seen:
                    continue
                seen.add(step.endpoint_id)
                ep = self._registry.get_endpoint(step.endpoint_id)
                desc = ep.description if ep and ep.description else step.endpoint_name
                tool_descriptions.append(f"- {step.tool_name}: {step.endpoint_name} — {desc}")

        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "(none)"
        values_text = format_available_values(context)

        return (
            "You are a helpful assistant with access to the following tools:\n"
            f"{tools_text}\n\n"
            "Use them to help the user. When calling a tool, provide all "
            "required arguments.\n\n"
            f"{values_text}"
        )

    @staticmethod
    def _sanitize_function_name(name: str) -> str:
        """Sanitize a name to match OpenAI's function name pattern: ^[a-zA-Z0-9_-]+$."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    def _build_tools_schema(self, chain: ToolChain) -> list[dict[str, Any]]:
        """Convert chain endpoints to OpenAI function-calling format."""
        schemas: list[dict[str, Any]] = []
        seen: set[str] = set()
        for step in _flatten_steps(chain):
            if step.endpoint_id in seen:
                continue
            seen.add(step.endpoint_id)

            ep = self._registry.get_endpoint(step.endpoint_id)
            if ep is None:
                continue

            properties: dict[str, Any] = {}
            required_params: list[str] = []
            for param in ep.parameters:
                safe_name = self._sanitize_function_name(param.name)
                prop: dict[str, str] = {
                    "type": _ptype_to_json(
                        param.param_type
                        if isinstance(param.param_type, str)
                        else param.param_type
                    ),
                }
                if param.description:
                    prop["description"] = param.description
                properties[safe_name] = prop
                if param.name in ep.required_parameters:
                    required_params.append(safe_name)

            func_name = self._sanitize_function_name(ep.endpoint_id)
            schemas.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": ep.description or ep.name,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_params,
                    },
                },
            })
        return schemas

    # ------------------------------------------------------------------
    # Argument resolution
    # ------------------------------------------------------------------

    def _resolve_argument(
        self,
        param: Any,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> Any:
        """Resolve a single parameter value from grounding or placeholder.

        Priority: exact name match → semantic match (ID→ID, name→name) →
        enum → default → type-based placeholder.
        """
        available = context.get_available_values()
        pname: str = param.name
        pname_lower: str = pname.lower()
        ptype: str = (
            param.param_type
            if isinstance(param.param_type, str)
            else str(param.param_type)
        )

        # 1. Exact name match.
        if pname in available:
            return available[pname]

        # 2. Substring match on names (skip step-prefixed keys).
        for key, val in available.items():
            if "." in key:
                continue
            if pname_lower in key.lower() or key.lower() in pname_lower:
                return val

        # 3. Semantic type match — if param expects an ID, use any available ID.
        #    This is the key grounding mechanism: step N's "hotel_id" gets
        #    resolved to the "id" from step N-1's response.
        if "id" in pname_lower:
            for key, val in available.items():
                if "." in key:
                    continue
                if "id" in key.lower() and isinstance(val, str):
                    return val

        if any(kw in pname_lower for kw in ("name", "title", "label")):
            for key, val in available.items():
                if "." in key:
                    continue
                if any(kw in key.lower() for kw in ("name", "title")) and isinstance(val, str):
                    return val

        if any(kw in pname_lower for kw in ("city", "location", "place", "destination", "address")):
            for key, val in available.items():
                if "." in key:
                    continue
                if any(kw in key.lower() for kw in ("city", "location", "destination")) and isinstance(val, str):
                    return val

        if any(kw in pname_lower for kw in ("date", "time", "start", "end", "check")):
            for key, val in available.items():
                if "." in key:
                    continue
                if any(kw in key.lower() for kw in ("date", "time", "created")) and isinstance(val, str):
                    return val

        if any(kw in pname_lower for kw in ("price", "cost", "amount", "budget")):
            for key, val in available.items():
                if "." in key:
                    continue
                if any(kw in key.lower() for kw in ("price", "cost", "amount")) and isinstance(val, (int, float)):
                    return val

        if any(kw in pname_lower for kw in ("status", "state")):
            for key, val in available.items():
                if "." in key:
                    continue
                if "status" in key.lower() and isinstance(val, str):
                    return val

        if any(kw in pname_lower for kw in ("email", "mail")):
            for key, val in available.items():
                if "." in key:
                    continue
                if "email" in key.lower() and isinstance(val, str):
                    return val

        if any(kw in pname_lower for kw in ("url", "link", "href")):
            for key, val in available.items():
                if "." in key:
                    continue
                if any(kw in key.lower() for kw in ("url", "link")) and isinstance(val, str):
                    return val

        # 4. Enum.
        if param.enum_values:
            idx = int(rng.integers(0, len(param.enum_values)))
            return param.enum_values[idx]

        # 5. Default.
        if param.default is not None:
            return param.default

        # 6. Type-based placeholder.
        return self._placeholder(ptype, pname, rng)

    @staticmethod
    def _placeholder(ptype: str, pname: str, rng: np.random.Generator) -> Any:
        """Generate a type-appropriate placeholder value."""
        if ptype == "integer":
            return int(rng.integers(1, 100))
        if ptype == "number":
            return round(float(rng.uniform(1.0, 100.0)), 2)
        if ptype == "boolean":
            return bool(rng.random() > 0.5)
        if ptype in ("date", "datetime"):
            return "2024-06-15"
        if ptype == "array":
            return []
        if ptype == "object":
            return {}
        # string / unknown
        return f"{pname}_value"

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Call the OpenAI-compatible client."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": 300,
        }
        if tools:
            kwargs["tools"] = tools
        return self._client.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(self, context: ConversationContext) -> str:
        """Build a natural summary from tool outputs."""
        if not context.tool_outputs:
            return "I wasn't able to find any results."

        # Collect meaningful values from outputs
        highlights: list[str] = []
        for resp in context.tool_outputs:
            data = resp.data
            # Look for the most informative fields
            name = data.get("name") or data.get("title")
            status = data.get("status")
            item_id = None
            for k, v in data.items():
                if "id" in k.lower() and isinstance(v, str):
                    item_id = v
                    break

            if name and status:
                highlights.append(f"{name} (status: {status})")
            elif name:
                highlights.append(str(name))

            # Check for list results
            results = data.get("results")
            if isinstance(results, list) and results:
                count = data.get("count", len(results))
                first_name = results[0].get("name", "item") if isinstance(results[0], dict) else "item"
                highlights.append(f"found {count} results including {first_name}")

            if item_id and not highlights:
                highlights.append(f"reference ID: {item_id}")

        if not highlights:
            return "I've completed all the requested tasks successfully."

        # Build natural summary
        if len(highlights) == 1:
            return f"Here's what I found: {highlights[0]}."
        elif len(highlights) == 2:
            return f"Here's what I found: {highlights[0]}, and {highlights[1]}."
        else:
            main = ", ".join(highlights[:2])
            return f"Here's a summary: {main}, and {len(highlights) - 2} more result(s)."
