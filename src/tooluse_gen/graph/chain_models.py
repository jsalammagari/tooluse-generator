"""Data models for tool-chain sampling.

A *tool chain* is an ordered sequence of :class:`ChainStep` (individual
endpoint invocations) and :class:`ParallelGroup` (concurrent calls)
sampled from the tool graph.  Four :class:`ChainPattern` variants
describe the execution topology.

:class:`SamplingConstraints` configures which chains the sampler may
produce (domain filters, step limits, quality thresholds, etc.).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from tooluse_gen.registry.completeness import QualityTier
from tooluse_gen.registry.models import HttpMethod

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ChainPattern(str, Enum):
    """Execution topology of a tool chain."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BRANCH_AND_MERGE = "branch_and_merge"
    ITERATIVE = "iterative"


# ---------------------------------------------------------------------------
# ChainStep
# ---------------------------------------------------------------------------


class ChainStep(BaseModel):
    """A single endpoint invocation in a tool chain."""

    model_config = ConfigDict(use_enum_values=True)

    endpoint_id: str = Field(..., description="Endpoint identifier.")
    tool_id: str = Field(..., description="Parent tool identifier.")
    tool_name: str = Field(..., description="Human-readable tool name.")
    endpoint_name: str = Field(..., description="Human-readable endpoint name.")
    method: HttpMethod = Field(default=HttpMethod.GET, description="HTTP method.")
    path: str = Field(default="", description="URL path template.")
    expected_params: list[str] = Field(
        default_factory=list, description="Parameter names this step expects."
    )
    expected_output_types: list[str] = Field(
        default_factory=list,
        description="Output types this step produces (ParameterType string values).",
    )
    description: str = Field(default="", description="What this step does.")
    domain: str = Field(default="", description="Domain of the parent tool.")

    @classmethod
    def from_graph_node(cls, graph: nx.DiGraph, endpoint_node_id: str) -> ChainStep:
        """Construct a :class:`ChainStep` from a graph endpoint node."""
        data = graph.nodes[endpoint_node_id]
        tool_id: str = data.get("tool_id", "")
        tool_node_id = f"tool:{tool_id}"
        if tool_node_id in graph:
            tool_name: str = graph.nodes[tool_node_id].get("name", tool_id)
        else:
            tool_name = tool_id

        return cls(
            endpoint_id=data.get("endpoint_id", ""),
            tool_id=tool_id,
            tool_name=tool_name,
            endpoint_name=data.get("name", ""),
            method=data.get("method", "GET"),
            path=data.get("path", ""),
            expected_params=list(data.get("parameter_names", [])),
            expected_output_types=list(data.get("extractable_output_types", [])),
            description=data.get("description", ""),
            domain=data.get("domain", ""),
        )


# ---------------------------------------------------------------------------
# ParallelGroup
# ---------------------------------------------------------------------------


class ParallelGroup(BaseModel):
    """Two or more steps executed in parallel."""

    model_config = ConfigDict(use_enum_values=True)

    steps: list[ChainStep] = Field(..., description="Steps executed concurrently (min 2).")

    @field_validator("steps")
    @classmethod
    def _at_least_two(cls, v: list[ChainStep]) -> list[ChainStep]:
        if len(v) < 2:
            raise ValueError("ParallelGroup requires at least 2 steps.")
        return v

    @computed_field
    @property
    def step_count(self) -> int:
        """Number of parallel steps."""
        return len(self.steps)

    @computed_field
    @property
    def tool_ids(self) -> list[str]:
        """Sorted unique tool IDs in this group."""
        return sorted({s.tool_id for s in self.steps})

    @computed_field
    @property
    def endpoint_ids(self) -> list[str]:
        """Endpoint IDs in order."""
        return [s.endpoint_id for s in self.steps]


# ---------------------------------------------------------------------------
# ToolChain
# ---------------------------------------------------------------------------


def _iter_chain_steps(steps: list[ChainStep | ParallelGroup]) -> list[ChainStep]:
    """Flatten all :class:`ChainStep` instances from a mixed list."""
    result: list[ChainStep] = []
    for item in steps:
        if isinstance(item, ParallelGroup):
            result.extend(item.steps)
        else:
            result.append(item)
    return result


class ToolChain(BaseModel):
    """A complete sampled chain of tool invocations."""

    model_config = ConfigDict(use_enum_values=True)

    chain_id: str = Field(default="", description="Unique chain identifier.")
    steps: list[ChainStep | ParallelGroup] = Field(
        ..., description="Ordered list of steps and parallel groups."
    )
    pattern: ChainPattern = Field(..., description="Execution pattern.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata.")

    @computed_field
    @property
    def domains_involved(self) -> list[str]:
        """Sorted unique domains across all steps."""
        return sorted({s.domain for s in _iter_chain_steps(self.steps) if s.domain})

    @computed_field
    @property
    def total_step_count(self) -> int:
        """Total individual ChainSteps (ParallelGroup steps counted separately)."""
        return len(_iter_chain_steps(self.steps))

    @computed_field
    @property
    def tool_ids(self) -> list[str]:
        """Sorted unique tool IDs across all steps."""
        return sorted({s.tool_id for s in _iter_chain_steps(self.steps)})

    @computed_field
    @property
    def endpoint_ids(self) -> list[str]:
        """Flat list of endpoint IDs in order."""
        return [s.endpoint_id for s in _iter_chain_steps(self.steps)]

    @property
    def is_multi_tool(self) -> bool:
        """True when more than one unique tool is used."""
        return len(self.tool_ids) > 1

    @property
    def is_cross_domain(self) -> bool:
        """True when more than one domain is involved."""
        return len(self.domains_involved) > 1


# ---------------------------------------------------------------------------
# SamplingConstraints
# ---------------------------------------------------------------------------


class SamplingConstraints(BaseModel):
    """Configuration for the chain sampler."""

    model_config = ConfigDict(use_enum_values=True)

    domains: list[str] | None = Field(default=None, description="Restrict to these domains.")
    min_steps: int = Field(default=2, ge=1, description="Minimum total step count.")
    max_steps: int = Field(default=5, ge=1, description="Maximum total step count.")
    min_tools: int = Field(default=2, ge=1, description="Minimum unique tools.")
    required_tools: list[str] | None = Field(
        default=None, description="Tools that must appear."
    )
    excluded_tools: list[str] | None = Field(
        default=None, description="Tools that must not appear."
    )
    required_patterns: list[ChainPattern] | None = Field(
        default=None, description="Only generate these patterns."
    )
    quality_threshold: QualityTier = Field(
        default=QualityTier.FAIR, description="Minimum quality tier for tools."
    )

    @model_validator(mode="after")
    def _check_step_range(self) -> SamplingConstraints:
        if self.max_steps < self.min_steps:
            raise ValueError(
                f"max_steps ({self.max_steps}) must be >= min_steps ({self.min_steps})."
            )
        return self
