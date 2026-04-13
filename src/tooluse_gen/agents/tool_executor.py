"""Mock tool execution agent.

:class:`ToolExecutor` processes :class:`ToolCallRequest` objects,
generates grounded mock responses via :class:`SchemaBasedGenerator`,
extracts referenceable values, and maintains session state consistency
through :class:`ConversationContext`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from tooluse_gen.agents.execution_models import (
    ConversationContext,
    ToolCallRequest,
    ToolCallResponse,
)
from tooluse_gen.agents.value_generator import SchemaBasedGenerator
from tooluse_gen.registry.models import Endpoint
from tooluse_gen.registry.registry import ToolRegistry
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.tool_executor")

_ID_PATTERN = re.compile(r"^[A-Z]{2,5}-\d{3,5}$")

# Keys whose values are likely useful for downstream chaining
_EXTRACTABLE_SUBSTRINGS = (
    "id", "name", "title", "url", "link", "date", "time",
    "price", "cost", "amount", "status", "email", "count", "total",
)


class ToolExecutor:
    """Mock execution agent that processes tool call requests and produces grounded responses."""

    def __init__(
        self,
        registry: ToolRegistry,
        generator: SchemaBasedGenerator | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
    ) -> None:
        self._registry = registry
        self._generator = generator or SchemaBasedGenerator()
        self._use_llm = use_llm
        self._llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        request: ToolCallRequest,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> ToolCallResponse:
        """Execute a tool call and return a mock response."""
        endpoint = self._registry.get_endpoint(request.endpoint_id)
        if endpoint is None:
            return ToolCallResponse(
                call_id=request.call_id,
                status_code=404,
                error=f"Endpoint {request.endpoint_id} not found",
            )

        if self._use_llm and self._llm_client is not None:
            response = self._execute_llm_based(request, endpoint, context)
        else:
            response = self._execute_schema_based(request, endpoint, context, rng)

        # Extract values and attach
        extracted = self._extract_values(response, endpoint)
        response = response.model_copy(update={"extractable_values": extracted})

        # Validate (log only, still return)
        if not self._validate_response(response, endpoint):
            logger.debug(
                "Response validation warning for %s (%s)",
                request.endpoint_id,
                request.call_id,
            )

        return response

    # ------------------------------------------------------------------
    # Schema-based execution
    # ------------------------------------------------------------------

    def _execute_schema_based(
        self,
        request: ToolCallRequest,
        endpoint: Endpoint,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> ToolCallResponse:
        # Determine domain from the tool for domain-aware mock responses
        tool = self._registry.get_endpoint_tool(request.endpoint_id)
        domain = tool.domain if tool and tool.domain else ""

        data = self._generator.generate_response(
            endpoint, request.arguments, context, rng, domain=domain
        )

        # Detect generated IDs in response data
        generated_ids: dict[str, str] = {}
        for key, value in data.items():
            if "id" in key.lower() and isinstance(value, str) and _ID_PATTERN.match(value):
                generated_ids[key] = value

        return ToolCallResponse(
            call_id=request.call_id,
            status_code=200,
            data=data,
            generated_ids=generated_ids,
        )

    # ------------------------------------------------------------------
    # LLM-based execution (placeholder)
    # ------------------------------------------------------------------

    def _execute_llm_based(
        self,
        request: ToolCallRequest,
        endpoint: Endpoint,
        context: ConversationContext,
    ) -> ToolCallResponse:
        logger.warning("LLM-based execution not yet implemented, returning placeholder")
        return ToolCallResponse(
            call_id=request.call_id,
            status_code=200,
            data={"message": "LLM-based execution not yet implemented"},
        )

    # ------------------------------------------------------------------
    # Value extraction
    # ------------------------------------------------------------------

    def _extract_values(
        self,
        response: ToolCallResponse,
        endpoint: Endpoint,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if (
            endpoint.response_schema is not None
            and endpoint.response_schema.properties
        ):
            # Schema-guided: extract matching keys from data
            for key in endpoint.response_schema.properties:
                if key in response.data:
                    result[key] = response.data[key]
        else:
            result.update(self._extract_values_heuristic(response.data))

        # Always include generated IDs
        result.update(response.generated_ids)
        return result

    def _extract_values_heuristic(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract likely-useful values from response data by key name."""
        result: dict[str, Any] = {}

        for key, value in data.items():
            k_lower = key.lower()
            if any(sub in k_lower for sub in _EXTRACTABLE_SUBSTRINGS):
                result[key] = value

        # Handle nested "results" lists: extract from first item
        results_list = data.get("results")
        if isinstance(results_list, list) and results_list:
            first = results_list[0]
            if isinstance(first, dict):
                for key, value in first.items():
                    k_lower = key.lower()
                    if any(sub in k_lower for sub in _EXTRACTABLE_SUBSTRINGS):
                        result[f"results_0_{key}"] = value

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_response(
        self,
        response: ToolCallResponse,
        endpoint: Endpoint,
    ) -> bool:
        if response.error is not None:
            return True  # error responses don't need structural validation

        if not response.data:
            logger.debug("Empty response data for %s", endpoint.endpoint_id)
            return False

        method = endpoint.method if isinstance(endpoint.method, str) else endpoint.method

        if method == "DELETE" and "status" not in response.data:
            logger.debug("DELETE response missing 'status' key for %s", endpoint.endpoint_id)
            return False

        if method == "POST" and "id" not in response.data:
            logger.debug("POST response missing 'id' key for %s", endpoint.endpoint_id)
            return False

        return True
