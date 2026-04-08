"""Central Tool Registry with multi-index lookup.

The :class:`ToolRegistry` stores :class:`Tool` objects and maintains
secondary indexes (by endpoint ID, domain, quality tier, HTTP method)
for efficient filtering.  All mutating operations invalidate the stats
cache so :attr:`stats` always reflects current contents.

:class:`RegistryBuilder` provides a fluent API for loading, scoring,
filtering, and constructing a ready-to-use registry in one pipeline.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tooluse_gen.registry.completeness import (
    CompletenessCalculator,
    QualityTier,
    get_quality_tier,
)
from tooluse_gen.registry.loader import LoaderConfig, ToolBenchLoader
from tooluse_gen.registry.models import Endpoint, HttpMethod, Tool

# ---------------------------------------------------------------------------
# RegistryStats
# ---------------------------------------------------------------------------

_TIER_ORDER: dict[QualityTier, int] = {
    QualityTier.EXCELLENT: 4,
    QualityTier.GOOD: 3,
    QualityTier.FAIR: 2,
    QualityTier.POOR: 1,
    QualityTier.MINIMAL: 0,
}


@dataclass
class RegistryStats:
    """Aggregate statistics about registry contents."""

    total_tools: int = 0
    total_endpoints: int = 0
    total_parameters: int = 0

    quality_distribution: dict[str, int] = field(default_factory=dict)
    average_completeness: float = 0.0

    domain_distribution: dict[str, int] = field(default_factory=dict)
    total_domains: int = 0

    method_distribution: dict[str, int] = field(default_factory=dict)
    avg_endpoints_per_tool: float = 0.0
    avg_params_per_endpoint: float = 0.0

    endpoints_with_response_schema: int = 0
    params_with_types: int = 0
    params_with_descriptions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tools": self.total_tools,
            "total_endpoints": self.total_endpoints,
            "total_parameters": self.total_parameters,
            "quality_distribution": self.quality_distribution,
            "average_completeness": self.average_completeness,
            "domain_distribution": self.domain_distribution,
            "total_domains": self.total_domains,
            "method_distribution": self.method_distribution,
            "avg_endpoints_per_tool": self.avg_endpoints_per_tool,
            "avg_params_per_endpoint": self.avg_params_per_endpoint,
            "endpoints_with_response_schema": self.endpoints_with_response_schema,
            "params_with_types": self.params_with_types,
            "params_with_descriptions": self.params_with_descriptions,
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Tools: {self.total_tools}",
            f"Endpoints: {self.total_endpoints}",
            f"Parameters: {self.total_parameters}",
            f"Domains: {self.total_domains}",
            f"Avg completeness: {self.average_completeness:.2f}",
            f"Avg endpoints/tool: {self.avg_endpoints_per_tool:.1f}",
            f"Avg params/endpoint: {self.avg_params_per_endpoint:.1f}",
            f"Endpoints with response schema: {self.endpoints_with_response_schema}",
            f"Params with types: {self.params_with_types}",
            f"Params with descriptions: {self.params_with_descriptions}",
        ]
        if self.quality_distribution:
            dist = ", ".join(f"{k}: {v}" for k, v in sorted(self.quality_distribution.items()))
            lines.append(f"Quality: {dist}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Central registry with multi-index lookup for tools and endpoints."""

    def __init__(self) -> None:
        # Primary storage
        self._tools: dict[str, Tool] = {}

        # Indexes
        self._endpoints: dict[str, Endpoint] = {}
        self._endpoint_to_tool: dict[str, str] = {}
        self._domains: dict[str, set[str]] = defaultdict(set)
        self._quality_tiers: dict[QualityTier, set[str]] = defaultdict(set)
        self._methods: dict[str, set[str]] = defaultdict(set)

        # Metadata
        self._created_at: datetime = datetime.now()
        self._source_info: dict[str, Any] = {}
        self._stats_cache: RegistryStats | None = None

    # === Tool Operations ===================================================

    def add_tool(self, tool: Tool) -> None:
        """Add a tool and update all indexes.

        Raises :class:`ValueError` if a tool with the same ID exists.
        """
        if tool.tool_id in self._tools:
            raise ValueError(f"Tool '{tool.tool_id}' already exists in registry.")
        self._tools[tool.tool_id] = tool
        self._index_tool(tool)
        self._stats_cache = None

    def add_tools(self, tools: Iterable[Tool]) -> int:
        """Add multiple tools, skipping duplicates. Return count added."""
        count = 0
        for tool in tools:
            if tool.tool_id not in self._tools:
                self._tools[tool.tool_id] = tool
                self._index_tool(tool)
                count += 1
        if count:
            self._stats_cache = None
        return count

    def get_tool(self, tool_id: str) -> Tool | None:
        """Return the tool with *tool_id*, or ``None``."""
        return self._tools.get(tool_id)

    def get_tool_or_raise(self, tool_id: str) -> Tool:
        """Return the tool with *tool_id*, or raise :class:`KeyError`."""
        try:
            return self._tools[tool_id]
        except KeyError:
            raise KeyError(f"Tool '{tool_id}' not found in registry.") from None

    def has_tool(self, tool_id: str) -> bool:
        return tool_id in self._tools

    def remove_tool(self, tool_id: str) -> bool:
        """Remove a tool and clean up indexes. Return ``True`` if removed."""
        tool = self._tools.pop(tool_id, None)
        if tool is None:
            return False
        self._deindex_tool(tool)
        self._stats_cache = None
        return True

    # === Endpoint Operations ===============================================

    def get_endpoint(self, endpoint_id: str) -> Endpoint | None:
        return self._endpoints.get(endpoint_id)

    def get_endpoint_tool(self, endpoint_id: str) -> Tool | None:
        tool_id = self._endpoint_to_tool.get(endpoint_id)
        if tool_id is None:
            return None
        return self._tools.get(tool_id)

    def get_endpoints_by_method(self, method: HttpMethod) -> list[Endpoint]:
        key = method.value if isinstance(method, HttpMethod) else str(method).upper()
        return [self._endpoints[eid] for eid in self._methods.get(key, set()) if eid in self._endpoints]

    # === Filtering =========================================================

    def get_tools_by_domain(self, domain: str) -> list[Tool]:
        tool_ids = self._domains.get(domain, set())
        return [self._tools[tid] for tid in tool_ids if tid in self._tools]

    def get_tools_by_quality(self, min_tier: QualityTier = QualityTier.FAIR) -> list[Tool]:
        min_rank = _TIER_ORDER[min_tier]
        result: list[Tool] = []
        for tier, rank in _TIER_ORDER.items():
            if rank >= min_rank:
                for tid in self._quality_tiers.get(tier, set()):
                    t = self._tools.get(tid)
                    if t is not None:
                        result.append(t)
        return result

    def filter_tools(
        self,
        domains: list[str] | None = None,
        min_quality: QualityTier | None = None,
        min_endpoints: int | None = None,
        max_endpoints: int | None = None,
        has_response_schema: bool | None = None,
    ) -> list[Tool]:
        """Filter tools by multiple criteria (AND logic)."""
        candidates: Iterable[Tool]

        if domains is not None:
            ids: set[str] = set()
            for d in domains:
                ids.update(self._domains.get(d, set()))
            candidates = [self._tools[tid] for tid in ids if tid in self._tools]
        else:
            candidates = self._tools.values()

        results: list[Tool] = []
        for tool in candidates:
            if min_quality is not None:
                tier = get_quality_tier(tool.completeness_score)
                if _TIER_ORDER[tier] < _TIER_ORDER[min_quality]:
                    continue
            ep_count = len(tool.endpoints)
            if min_endpoints is not None and ep_count < min_endpoints:
                continue
            if max_endpoints is not None and ep_count > max_endpoints:
                continue
            if has_response_schema is not None:
                has_rs = any(ep.response_schema is not None for ep in tool.endpoints)
                if has_rs != has_response_schema:
                    continue
            results.append(tool)
        return results

    def filter_endpoints(
        self,
        tool_ids: list[str] | None = None,
        domains: list[str] | None = None,
        methods: list[HttpMethod] | None = None,
        min_params: int | None = None,
        has_response_schema: bool | None = None,
    ) -> list[Endpoint]:
        """Filter endpoints by multiple criteria (AND logic)."""
        # Determine candidate endpoint IDs
        if tool_ids is not None:
            candidate_eids = {
                eid for eid, tid in self._endpoint_to_tool.items() if tid in set(tool_ids)
            }
        elif domains is not None:
            tids: set[str] = set()
            for d in domains:
                tids.update(self._domains.get(d, set()))
            candidate_eids = {
                eid for eid, tid in self._endpoint_to_tool.items() if tid in tids
            }
        else:
            candidate_eids = set(self._endpoints.keys())

        if methods is not None:
            method_eids: set[str] = set()
            for m in methods:
                key = m.value if isinstance(m, HttpMethod) else str(m).upper()
                method_eids.update(self._methods.get(key, set()))
            candidate_eids &= method_eids

        results: list[Endpoint] = []
        for eid in candidate_eids:
            ep = self._endpoints.get(eid)
            if ep is None:
                continue
            if min_params is not None and len(ep.parameters) < min_params:
                continue
            if has_response_schema is not None:
                has_rs = ep.response_schema is not None
                if has_rs != has_response_schema:
                    continue
            results.append(ep)
        return results

    # === Iteration =========================================================

    def tools(self) -> Iterator[Tool]:
        yield from self._tools.values()

    def endpoints(self) -> Iterator[Endpoint]:
        yield from self._endpoints.values()

    def tool_items(self) -> Iterator[tuple[str, Tool]]:
        yield from self._tools.items()

    def endpoint_items(self) -> Iterator[tuple[str, Endpoint]]:
        yield from self._endpoints.items()

    # === Statistics =========================================================

    @property
    def stats(self) -> RegistryStats:
        """Registry statistics (cached; invalidated on mutation)."""
        if self._stats_cache is None:
            self._stats_cache = self._compute_stats()
        return self._stats_cache

    @property
    def domains(self) -> list[str]:
        return sorted(d for d, ids in self._domains.items() if ids)

    def domain_counts(self) -> dict[str, int]:
        return {d: len(ids) for d, ids in self._domains.items() if ids}

    # === Serialization =====================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": [t.model_dump() for t in self._tools.values()],
            "created_at": self._created_at.isoformat(),
            "source_info": self._source_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolRegistry:
        registry = cls()
        for raw_tool in data.get("tools", []):
            tool = Tool.model_validate(raw_tool)
            registry.add_tools([tool])
        registry._source_info = data.get("source_info", {})
        return registry

    # === Dunder ============================================================

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._tools

    def __repr__(self) -> str:
        return (
            f"ToolRegistry(tools={len(self._tools)}, "
            f"endpoints={len(self._endpoints)}, "
            f"domains={len(self.domains)})"
        )

    # === Private ===========================================================

    def _index_tool(self, tool: Tool) -> None:
        """Add *tool* to all secondary indexes."""
        tier = get_quality_tier(tool.completeness_score)
        self._quality_tiers[tier].add(tool.tool_id)

        if tool.domain:
            self._domains[tool.domain].add(tool.tool_id)

        for ep in tool.endpoints:
            self._endpoints[ep.endpoint_id] = ep
            self._endpoint_to_tool[ep.endpoint_id] = tool.tool_id
            method_key = ep.method if isinstance(ep.method, str) else ep.method.value
            self._methods[method_key].add(ep.endpoint_id)

    def _deindex_tool(self, tool: Tool) -> None:
        """Remove *tool* from all secondary indexes."""
        for tier_set in self._quality_tiers.values():
            tier_set.discard(tool.tool_id)

        if tool.domain and tool.tool_id in self._domains.get(tool.domain, set()):
            self._domains[tool.domain].discard(tool.tool_id)

        for ep in tool.endpoints:
            self._endpoints.pop(ep.endpoint_id, None)
            self._endpoint_to_tool.pop(ep.endpoint_id, None)
            for method_set in self._methods.values():
                method_set.discard(ep.endpoint_id)

    def _compute_stats(self) -> RegistryStats:
        """Build stats from current registry contents."""
        total_tools = len(self._tools)
        total_endpoints = 0
        total_params = 0
        completeness_sum = 0.0
        quality_dist: dict[str, int] = defaultdict(int)
        domain_dist: dict[str, int] = defaultdict(int)
        method_dist: dict[str, int] = defaultdict(int)
        endpoints_with_rs = 0
        params_with_types = 0
        params_with_descs = 0

        for tool in self._tools.values():
            completeness_sum += tool.completeness_score
            tier = get_quality_tier(tool.completeness_score)
            quality_dist[tier.value] += 1
            if tool.domain:
                domain_dist[tool.domain] += 1
            for ep in tool.endpoints:
                total_endpoints += 1
                method_key = ep.method if isinstance(ep.method, str) else ep.method.value
                method_dist[method_key] += 1
                if ep.response_schema is not None:
                    endpoints_with_rs += 1
                for p in ep.parameters:
                    total_params += 1
                    if p.has_type:
                        params_with_types += 1
                    if p.description:
                        params_with_descs += 1

        avg_comp = (completeness_sum / total_tools) if total_tools else 0.0
        avg_ep = (total_endpoints / total_tools) if total_tools else 0.0
        avg_params = (total_params / total_endpoints) if total_endpoints else 0.0

        return RegistryStats(
            total_tools=total_tools,
            total_endpoints=total_endpoints,
            total_parameters=total_params,
            quality_distribution=dict(quality_dist),
            average_completeness=round(avg_comp, 4),
            domain_distribution=dict(domain_dist),
            total_domains=len([d for d, ids in self._domains.items() if ids]),
            method_distribution=dict(method_dist),
            avg_endpoints_per_tool=round(avg_ep, 2),
            avg_params_per_endpoint=round(avg_params, 2),
            endpoints_with_response_schema=endpoints_with_rs,
            params_with_types=params_with_types,
            params_with_descriptions=params_with_descs,
        )


# ---------------------------------------------------------------------------
# RegistryBuilder
# ---------------------------------------------------------------------------


class RegistryBuilder:
    """Fluent builder for constructing a :class:`ToolRegistry`.

    Usage::

        registry = (
            RegistryBuilder()
            .load_from_directory("data/toolbench")
            .calculate_completeness()
            .filter_by_quality(QualityTier.FAIR)
            .build()
        )
    """

    def __init__(self) -> None:
        self._tools: list[Tool] = []
        self._loader_config = LoaderConfig()
        self._completeness_calc = CompletenessCalculator()
        self._min_quality: QualityTier | None = None
        self._domains_filter: list[str] | None = None

    def with_loader_config(self, config: LoaderConfig) -> RegistryBuilder:
        self._loader_config = config
        return self

    def load_from_directory(self, path: Path | str) -> RegistryBuilder:
        loader = ToolBenchLoader(self._loader_config)
        self._tools.extend(loader.load_directory(Path(path)))
        return self

    def load_from_file(self, path: Path | str) -> RegistryBuilder:
        loader = ToolBenchLoader(self._loader_config)
        self._tools.extend(loader.load_file(Path(path)))
        return self

    def add_tools(self, tools: list[Tool]) -> RegistryBuilder:
        self._tools.extend(tools)
        return self

    def calculate_completeness(self) -> RegistryBuilder:
        for tool in self._tools:
            self._completeness_calc.calculate_all(tool)
        return self

    def filter_by_quality(self, min_tier: QualityTier) -> RegistryBuilder:
        self._min_quality = min_tier
        return self

    def filter_by_domains(self, domains: list[str]) -> RegistryBuilder:
        self._domains_filter = domains
        return self

    def build(self) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in self._tools:
            if self._min_quality is not None:
                tier = get_quality_tier(tool.completeness_score)
                if _TIER_ORDER[tier] < _TIER_ORDER[self._min_quality]:
                    continue
            if self._domains_filter is not None and tool.domain not in self._domains_filter:
                continue
            registry.add_tools([tool])
        return registry


# ---------------------------------------------------------------------------
# Random access helpers
# ---------------------------------------------------------------------------


def get_random_tool(
    registry: ToolRegistry,
    rng: random.Random | None = None,
    domain: str | None = None,
    min_quality: QualityTier | None = None,
) -> Tool:
    """Return a random tool, optionally filtered by domain / quality.

    Raises :class:`ValueError` when no tools match the filters.
    """
    candidates = registry.filter_tools(
        domains=[domain] if domain else None,
        min_quality=min_quality,
    )
    if not candidates:
        raise ValueError("No tools match the given filters.")
    r = rng or random.Random()
    return r.choice(candidates)


def get_random_endpoint(
    registry: ToolRegistry,
    rng: random.Random | None = None,
    tool_id: str | None = None,
    method: HttpMethod | None = None,
) -> Endpoint:
    """Return a random endpoint, optionally filtered.

    Raises :class:`ValueError` when no endpoints match.
    """
    candidates = registry.filter_endpoints(
        tool_ids=[tool_id] if tool_id else None,
        methods=[method] if method else None,
    )
    if not candidates:
        raise ValueError("No endpoints match the given filters.")
    r = rng or random.Random()
    return r.choice(candidates)
