"""Argument generation for tool calls.

:class:`ArgumentGenerator` produces realistic arguments for tool calls
by resolving grounding values from :class:`ConversationContext` when
available and falling back to :class:`ValuePool` generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.agents.value_generator import ValuePool
from tooluse_gen.registry.models import Endpoint, Parameter, ParameterType
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.argument_generator")


class ArgumentGenerator:
    """Generates realistic arguments for tool calls using grounding and value pools."""

    def __init__(self, pool: ValuePool | None = None) -> None:
        self._pool = pool or ValuePool()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_arguments(
        self,
        endpoint: Endpoint,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Produce arguments for a tool call to *endpoint*."""
        result = self._fill_required_params(endpoint, context, rng)
        result.update(self._fill_optional_params(endpoint, context, rng))
        return result

    # ------------------------------------------------------------------
    # Required parameters
    # ------------------------------------------------------------------

    def _fill_required_params(
        self,
        endpoint: Endpoint,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        required_names = set(endpoint.required_parameters)
        result: dict[str, Any] = {}

        for param in endpoint.parameters:
            if not param.required and param.name not in required_names:
                continue
            result[param.name] = self._resolve_param_value(param, context, rng)

        return result

    # ------------------------------------------------------------------
    # Optional parameters
    # ------------------------------------------------------------------

    def _fill_optional_params(
        self,
        endpoint: Endpoint,
        context: ConversationContext,
        rng: np.random.Generator,
        include_probability: float = 0.5,
    ) -> dict[str, Any]:
        required_names = set(endpoint.required_parameters)
        result: dict[str, Any] = {}

        for param in endpoint.parameters:
            if param.required or param.name in required_names:
                continue
            if rng.random() >= include_probability:
                continue
            result[param.name] = self._resolve_param_value(param, context, rng)

        return result

    # ------------------------------------------------------------------
    # Value resolution
    # ------------------------------------------------------------------

    def _resolve_param_value(
        self,
        param: Parameter,
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> Any:
        """Resolve a single parameter value: grounding → enum → default → fresh."""
        # 1. Grounding
        grounded = self._resolve_grounded_value(param, context)
        if grounded is not None:
            return grounded

        # 2. Enum
        if param.enum_values:
            return param.enum_values[int(rng.integers(len(param.enum_values)))]

        # 3. Default
        if param.default is not None:
            return param.default

        # 4. Fresh
        return self._generate_fresh_value(param, rng)

    def _resolve_grounded_value(
        self,
        param: Parameter,
        context: ConversationContext,
    ) -> Any | None:
        """Return a grounding value matching *param*, or ``None``."""
        available = context.get_available_values()
        if not available:
            return None

        match_key = self._match_param_to_grounding(
            param.name, param.param_type, available
        )
        if match_key is not None:
            return available[match_key]
        return None

    def _generate_fresh_value(
        self,
        param: Parameter,
        rng: np.random.Generator,
    ) -> Any:
        """Generate a fresh value from the pool based on type and name."""
        name = param.name.lower()
        # use_enum_values=True means param_type is a str at runtime
        ptype = str(param.param_type)

        # ID heuristic (before type check)
        if "id" in name:
            prefix = param.name.upper()[:3]
            return f"{prefix}-{int(rng.integers(1000, 9999))}"

        if ptype == "integer":
            return self._pool.get("integer", rng)
        if ptype == "number":
            return self._pool.get("price", rng)
        if ptype == "boolean":
            return self._pool.get("boolean", rng)
        if ptype in ("date", "datetime"):
            return self._pool.get("date", rng)
        if ptype == "array":
            return [self._pool.get("description", rng)]
        if ptype == "object":
            return {"key": self._pool.get("description", rng)}

        # STRING or fallback — use name-based fuzzy lookup
        return self._pool.get(param.name, rng)

    # ------------------------------------------------------------------
    # Grounding matching
    # ------------------------------------------------------------------

    def _match_param_to_grounding(
        self,
        param_name: str,
        param_type: str | ParameterType,  # noqa: ARG002
        available_values: dict[str, Any],
    ) -> str | None:
        """Find the best matching key in *available_values* for *param_name*."""
        # Exact match
        if param_name in available_values:
            return param_name

        # Step-prefix match (e.g. step_0.city → city)
        suffix = f".{param_name}"
        for key in available_values:
            if key.endswith(suffix):
                return key

        # Substring match
        for key in available_values:
            # Skip step-prefixed keys for substring (avoid false positives)
            bare_key = key.split(".")[-1] if "." in key else key
            if param_name in bare_key or bare_key in param_name:
                return key

        return None
