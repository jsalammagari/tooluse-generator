"""Unit tests for the ToolRegistry (Task 14)."""

from __future__ import annotations

import random

import pytest

from tooluse_gen.registry.completeness import QualityTier
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    Tool,
)
from tooluse_gen.registry.registry import (
    RegistryBuilder,
    RegistryStats,
    ToolRegistry,
    get_random_endpoint,
    get_random_tool,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _ep(eid: str = "t/GET/x", tool_id: str = "t", method: HttpMethod = HttpMethod.GET,
        path: str = "/x", params: list[Parameter] | None = None,
        response_schema: object = None) -> Endpoint:
    return Endpoint(
        endpoint_id=eid, tool_id=tool_id, name=eid, path=path,
        method=method, parameters=params or [],
        response_schema=response_schema,
    )


def _tool(tid: str = "t", name: str = "T", domain: str = "D",
          endpoints: list[Endpoint] | None = None,
          completeness_score: float = 0.5) -> Tool:
    return Tool(
        tool_id=tid, name=name, domain=domain,
        endpoints=endpoints or [_ep(eid=f"{tid}/GET/x", tool_id=tid)],
        completeness_score=completeness_score,
    )


# ---------------------------------------------------------------------------
# ToolRegistry — tool operations
# ---------------------------------------------------------------------------


class TestToolOperations:
    def test_add_and_get(self):
        reg = ToolRegistry()
        t = _tool()
        reg.add_tool(t)
        assert reg.get_tool("t") is t
        assert reg.has_tool("t") is True
        assert len(reg) == 1

    def test_add_duplicate_raises(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        with pytest.raises(ValueError):
            reg.add_tool(_tool())

    def test_add_tools_bulk(self):
        reg = ToolRegistry()
        count = reg.add_tools([_tool("a"), _tool("b"), _tool("a")])
        assert count == 2  # duplicate skipped
        assert len(reg) == 2

    def test_get_tool_missing(self):
        reg = ToolRegistry()
        assert reg.get_tool("nope") is None

    def test_get_tool_or_raise(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        assert reg.get_tool_or_raise("t").tool_id == "t"
        with pytest.raises(KeyError):
            reg.get_tool_or_raise("nope")

    def test_remove_tool(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        assert reg.remove_tool("t") is True
        assert reg.has_tool("t") is False
        assert len(reg) == 0
        assert reg.remove_tool("t") is False  # already removed

    def test_contains(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        assert "t" in reg
        assert "nope" not in reg


# ---------------------------------------------------------------------------
# Endpoint operations
# ---------------------------------------------------------------------------


class TestEndpointOperations:
    def test_get_endpoint(self):
        reg = ToolRegistry()
        ep = _ep()
        reg.add_tool(_tool(endpoints=[ep]))
        assert reg.get_endpoint("t/GET/x") is ep

    def test_get_endpoint_missing(self):
        reg = ToolRegistry()
        assert reg.get_endpoint("nope") is None

    def test_get_endpoint_tool(self):
        reg = ToolRegistry()
        t = _tool()
        reg.add_tool(t)
        assert reg.get_endpoint_tool("t/GET/x") is t

    def test_get_endpoint_tool_missing(self):
        reg = ToolRegistry()
        assert reg.get_endpoint_tool("nope") is None

    def test_get_endpoints_by_method(self):
        ep_get = _ep(eid="t/GET/a", method=HttpMethod.GET)
        ep_post = _ep(eid="t/POST/b", method=HttpMethod.POST)
        reg = ToolRegistry()
        reg.add_tool(_tool(endpoints=[ep_get, ep_post]))
        gets = reg.get_endpoints_by_method(HttpMethod.GET)
        assert len(gets) == 1
        assert gets[0].endpoint_id == "t/GET/a"


# ---------------------------------------------------------------------------
# Filtering — tools
# ---------------------------------------------------------------------------


class TestFilterTools:
    def test_by_domain(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a", domain="Weather"), _tool("b", domain="Finance")])
        result = reg.get_tools_by_domain("Weather")
        assert len(result) == 1 and result[0].tool_id == "a"

    def test_by_quality(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("hi", completeness_score=0.9), _tool("lo", completeness_score=0.1)])
        result = reg.get_tools_by_quality(QualityTier.GOOD)
        assert len(result) == 1 and result[0].tool_id == "hi"

    def test_filter_tools_multi(self):
        reg = ToolRegistry()
        reg.add_tools([
            _tool("a", domain="Weather", completeness_score=0.9),
            _tool("b", domain="Weather", completeness_score=0.1),
            _tool("c", domain="Finance", completeness_score=0.9),
        ])
        result = reg.filter_tools(domains=["Weather"], min_quality=QualityTier.GOOD)
        assert len(result) == 1 and result[0].tool_id == "a"

    def test_filter_tools_min_max_endpoints(self):
        reg = ToolRegistry()
        ep1 = _ep(eid="t1/GET/a", tool_id="t1")
        ep2 = _ep(eid="t1/GET/b", tool_id="t1")
        reg.add_tools([_tool("t1", endpoints=[ep1, ep2]), _tool("t2", endpoints=[_ep(eid="t2/GET/x", tool_id="t2")])])
        assert len(reg.filter_tools(min_endpoints=2)) == 1
        assert len(reg.filter_tools(max_endpoints=1)) == 1

    def test_filter_tools_has_response_schema(self):
        from tooluse_gen.registry.models import ResponseSchema as LRS
        reg = ToolRegistry()
        ep_rs = _ep(eid="a/GET/x", tool_id="a", response_schema=LRS())
        ep_no = _ep(eid="b/GET/x", tool_id="b")
        reg.add_tools([_tool("a", endpoints=[ep_rs]), _tool("b", endpoints=[ep_no])])
        assert len(reg.filter_tools(has_response_schema=True)) == 1
        assert len(reg.filter_tools(has_response_schema=False)) == 1

    def test_filter_tools_empty(self):
        reg = ToolRegistry()
        assert reg.filter_tools(domains=["Nope"]) == []


# ---------------------------------------------------------------------------
# Filtering — endpoints
# ---------------------------------------------------------------------------


class TestFilterEndpoints:
    def test_by_tool_ids(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a"), _tool("b")])
        result = reg.filter_endpoints(tool_ids=["a"])
        assert all(reg.get_endpoint_tool(ep.endpoint_id).tool_id == "a" for ep in result)

    def test_by_methods(self):
        ep_get = _ep(eid="t/GET/a", method=HttpMethod.GET)
        ep_post = _ep(eid="t/POST/b", method=HttpMethod.POST)
        reg = ToolRegistry()
        reg.add_tool(_tool(endpoints=[ep_get, ep_post]))
        result = reg.filter_endpoints(methods=[HttpMethod.POST])
        assert len(result) == 1 and result[0].method == "POST"

    def test_by_min_params(self):
        ep1 = _ep(eid="t/GET/a", params=[Parameter(name="q")])
        ep2 = _ep(eid="t/GET/b")
        reg = ToolRegistry()
        reg.add_tool(_tool(endpoints=[ep1, ep2]))
        result = reg.filter_endpoints(min_params=1)
        assert len(result) == 1

    def test_by_domains(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a", domain="Weather"), _tool("b", domain="Finance")])
        result = reg.filter_endpoints(domains=["Weather"])
        assert len(result) == 1

    def test_by_has_response_schema(self):
        from tooluse_gen.registry.models import ResponseSchema as LRS
        ep_rs = _ep(eid="t/GET/a", response_schema=LRS())
        ep_no = _ep(eid="t/GET/b")
        reg = ToolRegistry()
        reg.add_tool(_tool(endpoints=[ep_rs, ep_no]))
        assert len(reg.filter_endpoints(has_response_schema=True)) == 1
        assert len(reg.filter_endpoints(has_response_schema=False)) == 1


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestIteration:
    def test_tools_iterator(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a"), _tool("b")])
        assert len(list(reg.tools())) == 2

    def test_endpoints_iterator(self):
        reg = ToolRegistry()
        reg.add_tool(_tool(endpoints=[_ep(eid="t/GET/a"), _ep(eid="t/POST/b", method=HttpMethod.POST)]))
        assert len(list(reg.endpoints())) == 2

    def test_tool_items(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        items = list(reg.tool_items())
        assert items[0] == ("t", reg.get_tool("t"))

    def test_endpoint_items(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        items = list(reg.endpoint_items())
        assert len(items) == 1
        assert items[0][0] == "t/GET/x"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestRegistryStats:
    def test_stats_computed(self):
        reg = ToolRegistry()
        reg.add_tools([
            _tool("a", domain="Weather", completeness_score=0.9,
                  endpoints=[_ep(eid="a/GET/x", tool_id="a", params=[Parameter(name="q", has_type=True, description="A query parameter for searching.")])]),
            _tool("b", domain="Finance", completeness_score=0.3),
        ])
        s = reg.stats
        assert s.total_tools == 2
        assert s.total_endpoints == 2
        assert s.total_parameters == 1
        assert s.total_domains == 2
        assert s.average_completeness > 0
        assert s.avg_endpoints_per_tool == 1.0
        assert s.params_with_types == 1
        assert s.params_with_descriptions == 1

    def test_stats_cached(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        s1 = reg.stats
        s2 = reg.stats
        assert s1 is s2  # same object

    def test_stats_invalidated(self):
        reg = ToolRegistry()
        reg.add_tool(_tool("a"))
        s1 = reg.stats
        reg.add_tool(_tool("b"))
        s2 = reg.stats
        assert s1 is not s2

    def test_stats_to_dict(self):
        s = RegistryStats(total_tools=1)
        d = s.to_dict()
        assert d["total_tools"] == 1

    def test_stats_summary(self):
        s = RegistryStats(total_tools=5, total_endpoints=10, total_parameters=20,
                          total_domains=3, average_completeness=0.75,
                          avg_endpoints_per_tool=2.0, avg_params_per_endpoint=2.0,
                          endpoints_with_response_schema=5, params_with_types=15,
                          params_with_descriptions=10)
        summary = s.summary()
        assert "Tools: 5" in summary
        assert "Endpoints: 10" in summary

    def test_domains_property(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a", domain="B"), _tool("b", domain="A")])
        assert reg.domains == ["A", "B"]  # sorted

    def test_domain_counts(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a", domain="Weather"), _tool("b", domain="Weather"), _tool("c", domain="Finance")])
        counts = reg.domain_counts()
        assert counts["Weather"] == 2
        assert counts["Finance"] == 1

    def test_empty_registry_stats(self):
        s = ToolRegistry().stats
        assert s.total_tools == 0
        assert s.average_completeness == 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        data = reg.to_dict()
        assert len(data["tools"]) == 1
        reg2 = ToolRegistry.from_dict(data)
        assert len(reg2) == 1
        assert reg2.has_tool("t")

    def test_repr(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        r = repr(reg)
        assert "tools=1" in r
        assert "endpoints=1" in r


# ---------------------------------------------------------------------------
# RegistryBuilder
# ---------------------------------------------------------------------------


class TestRegistryBuilder:
    def test_add_tools_and_build(self):
        reg = RegistryBuilder().add_tools([_tool("a"), _tool("b")]).build()
        assert len(reg) == 2

    def test_filter_by_quality(self):
        tools = [_tool("hi", completeness_score=0.9), _tool("lo", completeness_score=0.1)]
        reg = RegistryBuilder().add_tools(tools).filter_by_quality(QualityTier.GOOD).build()
        assert len(reg) == 1 and reg.has_tool("hi")

    def test_filter_by_domains(self):
        tools = [_tool("a", domain="Weather"), _tool("b", domain="Finance")]
        reg = RegistryBuilder().add_tools(tools).filter_by_domains(["Weather"]).build()
        assert len(reg) == 1 and reg.has_tool("a")

    def test_calculate_completeness(self):
        t = _tool(completeness_score=0.0)
        reg = RegistryBuilder().add_tools([t]).calculate_completeness().build()
        loaded = reg.get_tool("t")
        assert loaded is not None
        assert loaded.completeness_score > 0.0

    def test_chaining(self):
        reg = (
            RegistryBuilder()
            .add_tools([_tool("a", domain="W", completeness_score=0.9)])
            .filter_by_quality(QualityTier.GOOD)
            .filter_by_domains(["W"])
            .build()
        )
        assert len(reg) == 1

    def test_load_from_file(self, tmp_path):
        import json
        data = {"name": "T", "api_list": [{"name": "ep", "url": "/x"}]}
        f = tmp_path / "t.json"
        f.write_text(json.dumps(data))
        reg = RegistryBuilder().load_from_file(f).build()
        assert len(reg) >= 1

    def test_load_from_directory(self, tmp_path):
        import json
        data = {"name": "T", "api_list": [{"name": "ep", "url": "/x"}]}
        (tmp_path / "t.json").write_text(json.dumps(data))
        reg = RegistryBuilder().load_from_directory(tmp_path).build()
        assert len(reg) >= 1


# ---------------------------------------------------------------------------
# Random access
# ---------------------------------------------------------------------------


class TestRandomAccess:
    def test_get_random_tool(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a"), _tool("b")])
        t = get_random_tool(reg, rng=random.Random(42))
        assert t.tool_id in ("a", "b")

    def test_get_random_tool_with_domain(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a", domain="Weather"), _tool("b", domain="Finance")])
        t = get_random_tool(reg, rng=random.Random(42), domain="Weather")
        assert t.tool_id == "a"

    def test_get_random_tool_no_match(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        with pytest.raises(ValueError):
            get_random_tool(reg, domain="Nope")

    def test_get_random_endpoint(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        ep = get_random_endpoint(reg, rng=random.Random(42))
        assert ep.endpoint_id == "t/GET/x"

    def test_get_random_endpoint_by_tool(self):
        reg = ToolRegistry()
        reg.add_tools([_tool("a"), _tool("b")])
        ep = get_random_endpoint(reg, rng=random.Random(42), tool_id="a")
        assert reg.get_endpoint_tool(ep.endpoint_id).tool_id == "a"

    def test_get_random_endpoint_no_match(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        with pytest.raises(ValueError):
            get_random_endpoint(reg, method=HttpMethod.DELETE)


# ---------------------------------------------------------------------------
# Index cleanup on remove
# ---------------------------------------------------------------------------


class TestIndexCleanup:
    def test_remove_cleans_endpoints(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        assert reg.get_endpoint("t/GET/x") is not None
        reg.remove_tool("t")
        assert reg.get_endpoint("t/GET/x") is None

    def test_remove_cleans_domain_index(self):
        reg = ToolRegistry()
        reg.add_tool(_tool(domain="Weather"))
        assert len(reg.get_tools_by_domain("Weather")) == 1
        reg.remove_tool("t")
        assert len(reg.get_tools_by_domain("Weather")) == 0

    def test_remove_cleans_method_index(self):
        reg = ToolRegistry()
        reg.add_tool(_tool())
        assert len(reg.get_endpoints_by_method(HttpMethod.GET)) == 1
        reg.remove_tool("t")
        assert len(reg.get_endpoints_by_method(HttpMethod.GET)) == 0


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_exports():
    from tooluse_gen.registry import (
        RegistryBuilder,
        ToolRegistry,
    )
    assert ToolRegistry is not None
    assert RegistryBuilder is not None
