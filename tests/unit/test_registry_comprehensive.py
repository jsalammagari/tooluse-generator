"""Comprehensive cross-component tests for the registry pipeline (Task 16).

Covers ToolBench edge cases: missing fields, inconsistent naming,
malformed JSON, empty arrays, Unicode, very large tools, null values,
duplicate parameters, alternative field names, and normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooluse_gen.registry.completeness import (
    CompletenessCalculator,
    QualityTier,
    get_quality_tier,
    get_score_breakdown,
)
from tooluse_gen.registry.loader import (
    LoaderConfig,
    ToolBenchLoader,
    ToolBenchLoaderError,
)
from tooluse_gen.registry.models import (
    ParameterType,
)
from tooluse_gen.registry.registry import (
    RegistryBuilder,
    ToolRegistry,
    get_random_endpoint,
    get_random_tool,
)
from tooluse_gen.registry.serialization import (
    ChecksumError,
    SerializationFormat,
    VersionIncompatibleError,
    load_registry,
    save_registry,
)
from tooluse_gen.registry.type_inference import ParameterTypeInferrer

pytestmark = pytest.mark.unit


# =========================================================================
# Loader tests
# =========================================================================


class TestToolBenchLoaderComplete:
    """Loading well-formed, complete tools."""

    def test_load_complete_tool(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        assert len(tools) == 1
        t = tools[0]
        assert t.name == "Weather API"
        assert t.domain == "Weather"
        assert t.base_url == "https://api.weather.com"
        assert len(t.endpoints) == 1

        ep = t.endpoints[0]
        assert ep.name == "Get Current Weather"
        assert ep.method == "GET"
        assert ep.path == "/weather/current"
        assert len(ep.parameters) == 2
        assert ep.response_schema is not None

        loc_param = next(p for p in ep.parameters if p.name == "location")
        assert loc_param.required is True
        assert loc_param.param_type == ParameterType.STRING
        assert loc_param.has_type is True

    def test_load_complete_tool_stats(self, temp_data_dir: Path, loader: ToolBenchLoader):
        loader.load_file(temp_data_dir / "complete.json")
        stats = loader.get_stats()
        assert stats.files_processed == 1
        assert stats.files_failed == 0
        assert stats.tools_loaded == 1
        assert stats.endpoints_loaded == 1
        assert stats.parameters_loaded == 2

    def test_load_minimal_tool(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "minimal.json")
        assert len(tools) == 1
        t = tools[0]
        assert t.name == "Minimal API"
        assert t.description == ""
        assert t.domain == ""
        assert len(t.endpoints) == 1
        assert t.endpoints[0].path == "/data"

    def test_load_empty_tool_graceful(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "empty.json")
        assert tools == []
        stats = loader.get_stats()
        assert stats.tools_skipped >= 1

    def test_load_empty_tool_strict(self, temp_data_dir: Path, strict_loader: ToolBenchLoader):
        with pytest.raises(ToolBenchLoaderError):
            strict_loader.load_file(temp_data_dir / "empty.json")

    def test_load_unicode_tool(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "unicode.json")
        assert len(tools) == 1
        t = tools[0]
        assert "日本語" in t.name
        assert "émojis" in t.description or "mojis" in t.description
        assert len(t.endpoints) == 1

    def test_load_multiple_tools_from_array(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "multiple.json")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "Weather API" in names
        assert "Minimal API" in names

    def test_load_directory_recursive(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_directory(temp_data_dir, recursive=True)
        # Should load from all valid files including subdir/nested.json
        assert len(tools) >= 5  # at least complete, minimal, unicode, alt_format, nested...

    def test_load_directory_non_recursive(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools_rec = loader.load_directory(temp_data_dir, recursive=True)
        loader2 = ToolBenchLoader(LoaderConfig())
        tools_flat = loader2.load_directory(temp_data_dir, recursive=False)
        assert len(tools_flat) < len(tools_rec)


# =========================================================================
# Edge case tests
# =========================================================================


class TestLoaderEdgeCases:
    def test_malformed_endpoints_graceful(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "malformed_ep.json")
        # "endpoints": "not_an_array" → tool has no valid endpoints → filtered by min_endpoints
        assert len(tools) == 0

    def test_missing_file(self, loader: ToolBenchLoader):
        tools = loader.load_file(Path("/nonexistent/file.json"))
        assert tools == []
        assert loader.get_stats().files_failed == 1

    def test_missing_file_strict(self, strict_loader: ToolBenchLoader):
        with pytest.raises(ToolBenchLoaderError):
            strict_loader.load_file(Path("/nonexistent/file.json"))

    def test_invalid_json(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "invalid.json")
        assert tools == []
        assert loader.get_stats().files_failed == 1

    def test_large_tool_filtering(self, temp_data_dir: Path):
        loader = ToolBenchLoader(LoaderConfig(max_endpoints=100))
        tools = loader.load_file(temp_data_dir / "large.json")
        assert len(tools) == 0
        assert loader.get_stats().tools_skipped >= 1

    def test_large_tool_accepted_with_high_limit(self, temp_data_dir: Path):
        loader = ToolBenchLoader(LoaderConfig(max_endpoints=200))
        tools = loader.load_file(temp_data_dir / "large.json")
        assert len(tools) == 1
        assert len(tools[0].endpoints) == 150

    def test_duplicate_parameter_names(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "dup_params.json")
        assert len(tools) == 1
        # Both params loaded (normalizer doesn't deduplicate)
        assert len(tools[0].endpoints[0].parameters) == 2

    def test_null_values(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "null_values.json")
        assert len(tools) == 1
        t = tools[0]
        assert t.description == ""  # None → ""
        assert t.domain == ""
        ep = t.endpoints[0]
        assert ep.method == "GET"  # None → default GET
        p = ep.parameters[0]
        assert p.description == ""

    def test_max_tools_limit(self, temp_data_dir: Path):
        loader = ToolBenchLoader(LoaderConfig(max_tools=2))
        tools = loader.load_directory(temp_data_dir)
        assert len(tools) == 2

    def test_min_completeness_filter(self, temp_data_dir: Path):
        loader = ToolBenchLoader(LoaderConfig(min_completeness=0.99))
        tools = loader.load_directory(temp_data_dir)
        assert len(tools) == 0


# =========================================================================
# Alternative field names
# =========================================================================


class TestAlternativeFieldNames:
    def test_alt_format_tool(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "alt_format.json")
        assert len(tools) == 1
        t = tools[0]
        assert t.name == "AltFormat"
        assert "alternative" in t.description.lower()
        assert t.domain == "Testing"
        assert "alt.api.com" in t.base_url
        assert len(t.endpoints) == 1
        ep = t.endpoints[0]
        assert ep.name == "DoThing"
        assert ep.method == "POST"
        assert ep.path == "/do"
        assert len(ep.parameters) == 1
        assert ep.parameters[0].required is True


# =========================================================================
# Normalization tests
# =========================================================================


class TestFieldNormalization:
    def test_http_method_normalization(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "method_variants.json")
        assert len(tools) == 1
        methods = [ep.method for ep in tools[0].endpoints]
        assert "GET" in methods
        assert "POST" in methods
        assert "PUT" in methods
        assert "DELETE" in methods
        # no_method defaults to GET
        no_method_ep = next(ep for ep in tools[0].endpoints if ep.name == "no_method")
        assert no_method_ep.method == "GET"

    def test_path_normalization(self, temp_data_dir: Path, loader: ToolBenchLoader):
        """The loader preserves paths as-is from source; verify they are stored."""
        tools = loader.load_file(temp_data_dir / "path_variants.json")
        assert len(tools) == 1
        eps = {ep.name: ep for ep in tools[0].endpoints}

        # Brace-style preserved
        assert "{id}" in eps["brace"].path
        # Colon/angle stored as-is (loader does not rewrite param syntax)
        assert eps["colon"].path  # non-empty
        assert eps["angle"].path

        # Paths are stored (query string may or may not be stripped by loader)
        assert eps["query"].path

        # Verify all endpoints have non-empty paths
        for ep in tools[0].endpoints:
            assert ep.path

    def test_parameter_type_normalization(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "type_variants.json")
        assert len(tools) == 1
        params = {p.name: p for p in tools[0].endpoints[0].parameters}

        assert params["a"].param_type == ParameterType.STRING
        assert params["b"].param_type == ParameterType.INTEGER
        assert params["c"].param_type == ParameterType.BOOLEAN
        assert params["d"].param_type == ParameterType.NUMBER
        assert params["e"].param_type == ParameterType.ARRAY
        assert params["f"].param_type == ParameterType.OBJECT
        assert params["g"].param_type == ParameterType.DATETIME
        assert params["h"].param_type == ParameterType.UNKNOWN
        # "i" has no type → inferred
        assert params["i"].has_type is True  # inferred
        assert params["i"].inferred_type is True


# =========================================================================
# Type inference tests
# =========================================================================


class TestTypeInferenceIntegration:
    def setup_method(self):
        self.inf = ParameterTypeInferrer()

    def test_infer_id_type(self):
        r = self.inf.infer_type("user_id")
        assert r.inferred_type == ParameterType.STRING
        assert r.confidence >= 0.8

    def test_infer_count_type(self):
        for name in ("limit", "offset", "page", "item_count"):
            r = self.inf.infer_type(name)
            assert r.inferred_type == ParameterType.INTEGER, f"Failed for {name}"

    def test_infer_boolean_type(self):
        for name in ("is_active", "has_items"):
            r = self.inf.infer_type(name)
            assert r.inferred_type == ParameterType.BOOLEAN, f"Failed for {name}"

    def test_infer_from_default(self):
        assert self.inf.infer_type("x", default_value=42).inferred_type == ParameterType.INTEGER
        assert self.inf.infer_type("x", default_value=True).inferred_type == ParameterType.BOOLEAN
        assert self.inf.infer_type("x", default_value=3.14).inferred_type == ParameterType.NUMBER

    def test_infer_from_enum(self):
        r = self.inf.infer_type("status", enum_values=["active", "inactive"])
        assert r.inferred_type == ParameterType.STRING
        r2 = self.inf.infer_type("level", enum_values=["1", "2", "3"])
        assert r2.inferred_type == ParameterType.INTEGER

    def test_confidence_combination(self):
        # name "is_active" → BOOLEAN + default True → BOOLEAN: boosted
        r = self.inf.infer_type("is_active", default_value=True)
        assert r.inferred_type == ParameterType.BOOLEAN
        assert r.confidence > 0.9
        assert len(r.evidences) >= 2

    def test_conflicting_evidence_highest_wins(self):
        # default=42 (0.95 INTEGER) vs name "query" (0.8 STRING)
        r = self.inf.infer_type("query", default_value=42)
        assert r.inferred_type == ParameterType.INTEGER

    def test_description_inference(self):
        r = self.inf.infer_type("x", description="Enter a number between 1 and 100")
        assert r.inferred_type == ParameterType.INTEGER

    def test_min_confidence_fallback(self):
        strict = ParameterTypeInferrer(min_confidence=0.99)
        r = strict.infer_type("unknown_field_xyz")
        assert r.inferred_type == ParameterType.STRING


# =========================================================================
# Registry tests
# =========================================================================


class TestToolRegistryIntegration:
    def test_add_and_get_tool(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        assert len(reg) == 1
        t = reg.get_tool(tools[0].tool_id)
        assert t is not None
        assert t.name == "Weather API"

    def test_duplicate_tool_raises(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tool(tools[0])
        with pytest.raises(ValueError):
            reg.add_tool(tools[0])

    def test_endpoint_indexing(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        # Endpoint should be indexed
        for ep in tools[0].endpoints:
            assert reg.get_endpoint(ep.endpoint_id) is ep
            assert reg.get_endpoint_tool(ep.endpoint_id) is tools[0]

    def test_domain_filtering(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools_w = loader.load_file(temp_data_dir / "complete.json")
        loader2 = ToolBenchLoader(LoaderConfig())
        tools_a = loader2.load_file(temp_data_dir / "alt_format.json")
        reg = ToolRegistry()
        reg.add_tools(tools_w)
        reg.add_tools(tools_a)
        weather = reg.get_tools_by_domain("Weather")
        assert len(weather) == 1
        assert weather[0].domain == "Weather"

    def test_quality_filtering(self):
        from tooluse_gen.registry.models import Endpoint as Ep
        from tooluse_gen.registry.models import Tool as T

        ep = Ep(endpoint_id="a/GET/x", tool_id="a", name="E", path="/x")
        t_hi = T(tool_id="hi", name="Hi", endpoints=[ep], completeness_score=0.9)
        ep2 = Ep(endpoint_id="b/GET/x", tool_id="b", name="E", path="/x")
        t_lo = T(tool_id="lo", name="Lo", endpoints=[ep2], completeness_score=0.1)
        reg = ToolRegistry()
        reg.add_tools([t_hi, t_lo])
        good = reg.get_tools_by_quality(QualityTier.GOOD)
        assert len(good) == 1 and good[0].tool_id == "hi"

    def test_statistics(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        s = reg.stats
        assert s.total_tools == 1
        assert s.total_endpoints == 1
        assert s.total_parameters == 2
        assert s.total_domains == 1

    def test_remove_cleans_indexes(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        tid = tools[0].tool_id
        eid = tools[0].endpoints[0].endpoint_id
        reg.remove_tool(tid)
        assert reg.get_tool(tid) is None
        assert reg.get_endpoint(eid) is None
        assert len(reg) == 0

    def test_filter_tools_combined(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_directory(temp_data_dir)
        reg = ToolRegistry()
        reg.add_tools(tools)
        # Filter by domain + min_endpoints
        filtered = reg.filter_tools(domains=["Weather"], min_endpoints=1)
        assert all(t.domain == "Weather" for t in filtered)

    def test_random_tool(self, temp_data_dir: Path, loader: ToolBenchLoader):
        import random

        tools = loader.load_directory(temp_data_dir)
        reg = ToolRegistry()
        reg.add_tools(tools)
        t = get_random_tool(reg, rng=random.Random(42))
        assert t is not None

    def test_random_endpoint(self, temp_data_dir: Path, loader: ToolBenchLoader):
        import random

        tools = loader.load_directory(temp_data_dir)
        reg = ToolRegistry()
        reg.add_tools(tools)
        ep = get_random_endpoint(reg, rng=random.Random(42))
        assert ep is not None


# =========================================================================
# Serialization integration tests
# =========================================================================


class TestSerializationIntegration:
    def test_json_roundtrip(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        out = temp_data_dir / "output" / "reg.json"
        save_registry(reg, out, fmt=SerializationFormat.JSON)
        reg2, meta = load_registry(out)
        assert len(reg2) == len(reg)
        assert meta.tool_count == 1

    def test_pickle_roundtrip(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        out = temp_data_dir / "output" / "reg.pkl"
        save_registry(reg, out, fmt=SerializationFormat.PICKLE)
        reg2, meta = load_registry(out)
        assert len(reg2) == len(reg)

    def test_format_autodetection(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        json_out = temp_data_dir / "out.json"
        pkl_out = temp_data_dir / "out.pkl"
        save_registry(reg, json_out)
        save_registry(reg, pkl_out)
        r1, _ = load_registry(json_out)
        r2, _ = load_registry(pkl_out)
        assert len(r1) == len(r2)

    def test_version_compatibility(self, temp_data_dir: Path):
        reg = ToolRegistry()
        reg.add_tool(
            __import__("tooluse_gen.registry.models", fromlist=["Tool"]).Tool(
                tool_id="t",
                name="T",
                endpoints=[
                    __import__("tooluse_gen.registry.models", fromlist=["Endpoint"]).Endpoint(
                        endpoint_id="t/GET/x",
                        tool_id="t",
                        name="E",
                        path="/x",
                    )
                ],
            )
        )
        out = temp_data_dir / "ver.json"
        save_registry(reg, out)
        data = json.loads(out.read_text())
        data["metadata"]["version"] = "2.0.0"
        out.write_text(json.dumps(data))
        with pytest.raises(VersionIncompatibleError):
            load_registry(out)

    def test_checksum_verification(self, temp_data_dir: Path):
        from tooluse_gen.registry.models import Endpoint as Ep
        from tooluse_gen.registry.models import Tool as T

        reg = ToolRegistry()
        reg.add_tool(
            T(
                tool_id="t",
                name="T",
                endpoints=[Ep(endpoint_id="t/GET/x", tool_id="t", name="E", path="/x")],
            )
        )
        out = temp_data_dir / "cs.json"
        save_registry(reg, out)
        data = json.loads(out.read_text())
        data["registry"]["tools"][0]["name"] = "TAMPERED"
        out.write_text(json.dumps(data))
        with pytest.raises(ChecksumError):
            load_registry(out)

    def test_data_preserved_after_roundtrip(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        reg = ToolRegistry()
        reg.add_tools(tools)
        out = temp_data_dir / "preserve.json"
        save_registry(reg, out)
        reg2, _ = load_registry(out)
        t1 = list(reg.tools())[0]
        t2 = list(reg2.tools())[0]
        assert t1.name == t2.name
        assert t1.domain == t2.domain
        assert len(t1.endpoints) == len(t2.endpoints)
        assert t1.endpoints[0].name == t2.endpoints[0].name
        assert len(t1.endpoints[0].parameters) == len(t2.endpoints[0].parameters)


# =========================================================================
# Completeness tests
# =========================================================================


class TestCompletenessIntegration:
    def test_complete_tool_high_score(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        assert len(tools) == 1
        t = tools[0]
        assert t.completeness_score > 0.5

    def test_minimal_tool_low_score(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "minimal.json")
        assert len(tools) == 1
        t = tools[0]
        assert t.completeness_score < 0.5

    def test_quality_tiers(self):
        assert get_quality_tier(0.9) == QualityTier.EXCELLENT
        assert get_quality_tier(0.7) == QualityTier.GOOD
        assert get_quality_tier(0.5) == QualityTier.FAIR
        assert get_quality_tier(0.3) == QualityTier.POOR
        assert get_quality_tier(0.1) == QualityTier.MINIMAL

    def test_score_breakdown(self, temp_data_dir: Path, loader: ToolBenchLoader):
        tools = loader.load_file(temp_data_dir / "complete.json")
        bd = get_score_breakdown(tools[0])
        assert bd.total_score > 0
        assert len(bd.component_scores) > 0
        assert bd.quality_tier == get_quality_tier(bd.total_score)
        assert isinstance(bd.recommendations, list)

    def test_calculate_all_updates_scores(self):
        from tooluse_gen.registry.models import Endpoint as Ep
        from tooluse_gen.registry.models import Parameter as P
        from tooluse_gen.registry.models import Tool as T

        t = T(
            tool_id="t",
            name="TestAPI",
            description="A test API for verifying completeness score calculation.",
            domain="Testing",
            endpoints=[
                Ep(
                    endpoint_id="t/GET/x",
                    tool_id="t",
                    name="GetData",
                    description="Retrieve data from the testing API service.",
                    path="/data",
                    parameters=[
                        P(name="q", description="Search query string for filtering.", has_type=True)
                    ],
                ),
            ],
        )
        assert t.completeness_score == 0.0
        calc = CompletenessCalculator()
        calc.calculate_all(t)
        assert t.completeness_score > 0.5
        assert t.endpoints[0].completeness_score > 0.0


# =========================================================================
# Builder integration tests
# =========================================================================


class TestRegistryBuilderIntegration:
    def test_full_pipeline(self, temp_data_dir: Path):
        reg = RegistryBuilder().load_from_directory(temp_data_dir).calculate_completeness().build()
        assert len(reg) > 0
        # All tools should have scores
        for t in reg.tools():
            assert t.completeness_score > 0.0

    def test_pipeline_with_quality_filter(self, temp_data_dir: Path):
        reg_all = (
            RegistryBuilder().load_from_directory(temp_data_dir).calculate_completeness().build()
        )
        reg_good = (
            RegistryBuilder()
            .load_from_directory(temp_data_dir)
            .calculate_completeness()
            .filter_by_quality(QualityTier.GOOD)
            .build()
        )
        assert len(reg_good) <= len(reg_all)
        for t in reg_good.tools():
            assert get_quality_tier(t.completeness_score) in (
                QualityTier.EXCELLENT,
                QualityTier.GOOD,
            )

    def test_pipeline_with_domain_filter(self, temp_data_dir: Path):
        reg = (
            RegistryBuilder()
            .load_from_directory(temp_data_dir)
            .filter_by_domains(["Weather"])
            .build()
        )
        for t in reg.tools():
            assert t.domain == "Weather"

    def test_pipeline_load_from_file(self, temp_data_dir: Path):
        reg = RegistryBuilder().load_from_file(temp_data_dir / "complete.json").build()
        assert len(reg) == 1
