"""Unit tests for completeness scoring (Task 10)."""

from __future__ import annotations

import pytest

from tooluse_gen.registry.completeness import (
    DEFAULT_WEIGHTS,
    CompletenessCalculator,
    CompletenessWeights,
    QualityTier,
    ScoreBreakdown,
    count_documented_params,
    filter_by_quality,
    generate_quality_report,
    get_quality_tier,
    get_score_breakdown,
    is_explicit_type,
    is_meaningful_description,
)
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterType,
    Tool,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _param(
    name: str = "q",
    description: str = "",
    param_type: ParameterType = ParameterType.STRING,
    has_type: bool = False,
    inferred_type: bool = False,
    required: bool = False,
    default: object = None,
    has_description: bool = False,
) -> Parameter:
    return Parameter(
        name=name,
        description=description,
        param_type=param_type,
        has_type=has_type,
        inferred_type=inferred_type,
        required=required,
        default=default,
        has_description=has_description,
    )


def _endpoint(
    endpoint_id: str = "t/GET/abc",
    tool_id: str = "t",
    name: str = "Search",
    description: str = "",
    path: str = "/search",
    method: HttpMethod = HttpMethod.GET,
    parameters: list[Parameter] | None = None,
    response_schema: object = None,
    completeness_score: float = 0.0,
) -> Endpoint:
    return Endpoint(
        endpoint_id=endpoint_id,
        tool_id=tool_id,
        name=name,
        description=description,
        path=path,
        method=method,
        parameters=parameters or [],
        response_schema=response_schema,
        completeness_score=completeness_score,
    )


def _tool(
    tool_id: str = "t",
    name: str = "TestTool",
    description: str = "",
    domain: str = "",
    endpoints: list[Endpoint] | None = None,
    completeness_score: float = 0.0,
) -> Tool:
    return Tool(
        tool_id=tool_id,
        name=name,
        description=description,
        domain=domain,
        endpoints=endpoints or [],
        completeness_score=completeness_score,
    )


def _rich_param() -> Parameter:
    """A fully-documented parameter."""
    return _param(
        name="query",
        description="The search query string to filter results by keyword.",
        param_type=ParameterType.STRING,
        has_type=True,
        inferred_type=False,
        required=True,
        default=None,
    )


def _rich_endpoint() -> Endpoint:
    """A fully-documented endpoint."""
    from tooluse_gen.registry.models import ResponseSchema as LegacyRS

    return _endpoint(
        name="Search",
        description="Search for items by keyword with optional pagination support.",
        path="/items/search",
        method=HttpMethod.GET,
        parameters=[_rich_param()],
        response_schema=LegacyRS(status_code=200),
    )


def _rich_tool() -> Tool:
    """A fully-documented tool."""
    return _tool(
        name="ItemsAPI",
        description="A comprehensive API for managing and searching items in the catalog.",
        domain="E-Commerce",
        endpoints=[_rich_endpoint()],
    )


# ---------------------------------------------------------------------------
# is_meaningful_description
# ---------------------------------------------------------------------------


class TestIsMeaningfulDescription:
    def test_empty(self):
        assert is_meaningful_description("", "name") is False

    def test_whitespace(self):
        assert is_meaningful_description("   ", "name") is False

    def test_too_short(self):
        assert is_meaningful_description("short", "name") is False

    def test_equals_name(self):
        assert is_meaningful_description("Search", "Search") is False
        assert is_meaningful_description("search", "Search") is False

    def test_trivial_prefix(self):
        assert is_meaningful_description("The Search", "Search") is False
        assert is_meaningful_description("A Search", "Search") is False
        assert is_meaningful_description("An Search", "Search") is False
        assert is_meaningful_description("This Search", "Search") is False

    def test_meaningful(self):
        assert is_meaningful_description("Search for items by keyword", "Search") is True

    def test_long_enough_different(self):
        assert is_meaningful_description("1234567890", "other") is True


# ---------------------------------------------------------------------------
# is_explicit_type
# ---------------------------------------------------------------------------


class TestIsExplicitType:
    def test_explicit(self):
        p = _param(has_type=True, inferred_type=False)
        assert is_explicit_type(p) is True

    def test_inferred(self):
        p = _param(has_type=True, inferred_type=True)
        assert is_explicit_type(p) is False

    def test_no_type(self):
        p = _param(has_type=False, inferred_type=False)
        assert is_explicit_type(p) is False


# ---------------------------------------------------------------------------
# count_documented_params
# ---------------------------------------------------------------------------


class TestCountDocumentedParams:
    def test_no_params(self):
        ep = _endpoint(parameters=[])
        assert count_documented_params(ep) == (0, 0)

    def test_all_documented(self):
        p = _param(name="q", description="Search query for filtering results.")
        ep = _endpoint(parameters=[p])
        assert count_documented_params(ep) == (1, 1)

    def test_none_documented(self):
        p = _param(name="q", description="")
        ep = _endpoint(parameters=[p])
        assert count_documented_params(ep) == (0, 1)

    def test_mixed(self):
        p1 = _param(name="q", description="Search query for filtering results.")
        p2 = _param(name="limit", description="")
        ep = _endpoint(parameters=[p1, p2])
        assert count_documented_params(ep) == (1, 2)


# ---------------------------------------------------------------------------
# get_quality_tier
# ---------------------------------------------------------------------------


class TestGetQualityTier:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (1.0, QualityTier.EXCELLENT),
            (0.8, QualityTier.EXCELLENT),
            (0.79, QualityTier.GOOD),
            (0.6, QualityTier.GOOD),
            (0.59, QualityTier.FAIR),
            (0.4, QualityTier.FAIR),
            (0.39, QualityTier.POOR),
            (0.2, QualityTier.POOR),
            (0.19, QualityTier.MINIMAL),
            (0.0, QualityTier.MINIMAL),
        ],
    )
    def test_thresholds(self, score: float, expected: QualityTier):
        assert get_quality_tier(score) == expected


# ---------------------------------------------------------------------------
# CompletenessCalculator — parameter
# ---------------------------------------------------------------------------


class TestParameterScore:
    def test_bare_param(self):
        """Param with only a name scores just the name weight."""
        calc = CompletenessCalculator()
        p = _param()
        score = calc.calculate_parameter_score(p)
        assert score == pytest.approx(DEFAULT_WEIGHTS.param_name, abs=1e-3)

    def test_rich_param(self):
        calc = CompletenessCalculator()
        p = _rich_param()
        score = calc.calculate_parameter_score(p)
        # name + description (meaningful) + type (explicit) + required
        expected = (
            DEFAULT_WEIGHTS.param_name
            + DEFAULT_WEIGHTS.param_description
            + DEFAULT_WEIGHTS.param_type
            + DEFAULT_WEIGHTS.param_required_flag
        )
        assert score == pytest.approx(expected, abs=1e-3)

    def test_param_with_default(self):
        calc = CompletenessCalculator()
        p = _param(default="foo")
        score = calc.calculate_parameter_score(p)
        expected = DEFAULT_WEIGHTS.param_name + DEFAULT_WEIGHTS.param_default_value
        assert score == pytest.approx(expected, abs=1e-3)

    def test_unknown_type_not_scored(self):
        calc = CompletenessCalculator()
        p = _param(param_type=ParameterType.UNKNOWN, has_type=True, inferred_type=False)
        score = calc.calculate_parameter_score(p)
        assert score == pytest.approx(DEFAULT_WEIGHTS.param_name, abs=1e-3)

    def test_score_bounds(self):
        calc = CompletenessCalculator()
        p = _param()
        assert 0.0 <= calc.calculate_parameter_score(p) <= 1.0

    def test_score_capped_at_one(self):
        """Even with inflated weights, score never exceeds 1.0."""
        heavy = CompletenessWeights(
            param_name=0.5,
            param_description=0.5,
            param_type=0.5,
            param_required_flag=0.5,
            param_default_value=0.5,
        )
        calc = CompletenessCalculator(heavy)
        p = _rich_param()
        assert calc.calculate_parameter_score(p) <= 1.0


# ---------------------------------------------------------------------------
# CompletenessCalculator — endpoint
# ---------------------------------------------------------------------------


class TestEndpointScore:
    def test_bare_endpoint(self):
        calc = CompletenessCalculator()
        ep = _endpoint()
        score = calc.calculate_endpoint_score(ep)
        # name + path + method
        expected = (
            DEFAULT_WEIGHTS.endpoint_name
            + DEFAULT_WEIGHTS.endpoint_path
            + DEFAULT_WEIGHTS.endpoint_method
        )
        assert score == pytest.approx(expected, abs=1e-3)

    def test_rich_endpoint(self):
        calc = CompletenessCalculator()
        ep = _rich_endpoint()
        score = calc.calculate_endpoint_score(ep)
        assert score > 0.5

    def test_no_params_no_penalty(self):
        """Endpoints with zero params don't lose points for params."""
        calc = CompletenessCalculator()
        ep = _endpoint(parameters=[])
        score = calc.calculate_endpoint_score(ep)
        # No parameter contribution, but no negative either
        assert score >= 0.0

    def test_response_schema_boost(self):
        from tooluse_gen.registry.models import ResponseSchema as LegacyRS

        calc = CompletenessCalculator()
        ep_no_rs = _endpoint()
        ep_with_rs = _endpoint(response_schema=LegacyRS(status_code=200))
        assert calc.calculate_endpoint_score(ep_with_rs) > calc.calculate_endpoint_score(ep_no_rs)

    def test_empty_path_not_scored(self):
        calc = CompletenessCalculator()
        ep = _endpoint(path="")
        score = calc.calculate_endpoint_score(ep)
        # Should not include path weight
        assert score < (
            DEFAULT_WEIGHTS.endpoint_name
            + DEFAULT_WEIGHTS.endpoint_path
            + DEFAULT_WEIGHTS.endpoint_method
        )

    def test_slash_only_path_not_scored(self):
        calc = CompletenessCalculator()
        ep = _endpoint(path="/")
        score = calc.calculate_endpoint_score(ep)
        ep2 = _endpoint(path="/items")
        assert calc.calculate_endpoint_score(ep2) > score


# ---------------------------------------------------------------------------
# CompletenessCalculator — tool
# ---------------------------------------------------------------------------


class TestToolScore:
    def test_bare_tool(self):
        calc = CompletenessCalculator()
        t = _tool()
        score = calc.calculate_tool_score(t)
        # Only name
        assert score == pytest.approx(DEFAULT_WEIGHTS.tool_name, abs=1e-3)

    def test_rich_tool(self):
        calc = CompletenessCalculator()
        t = _rich_tool()
        score = calc.calculate_tool_score(t)
        assert score > 0.6

    def test_no_endpoints_no_endpoint_quality(self):
        calc = CompletenessCalculator()
        t = _tool()
        score = calc.calculate_tool_score(t)
        # Should not include has_endpoints or endpoint_quality
        assert score < DEFAULT_WEIGHTS.tool_name + DEFAULT_WEIGHTS.tool_has_endpoints + 0.01

    def test_tool_with_domain(self):
        calc = CompletenessCalculator()
        t1 = _tool(domain="")
        t2 = _tool(domain="Finance")
        assert calc.calculate_tool_score(t2) > calc.calculate_tool_score(t1)

    def test_tool_score_capped(self):
        heavy = CompletenessWeights(
            tool_name=0.5,
            tool_description=0.5,
            tool_domain=0.5,
            tool_has_endpoints=0.5,
            endpoint_quality=0.5,
        )
        calc = CompletenessCalculator(heavy)
        t = _rich_tool()
        assert calc.calculate_tool_score(t) <= 1.0


# ---------------------------------------------------------------------------
# calculate_all
# ---------------------------------------------------------------------------


class TestCalculateAll:
    def test_assigns_scores_in_place(self):
        calc = CompletenessCalculator()
        t = _rich_tool()
        assert t.completeness_score == 0.0
        assert t.endpoints[0].completeness_score == 0.0

        result = calc.calculate_all(t)
        assert result is t
        assert t.completeness_score > 0.0
        assert t.endpoints[0].completeness_score > 0.0

    def test_empty_tool(self):
        calc = CompletenessCalculator()
        t = _tool()
        calc.calculate_all(t)
        assert t.completeness_score > 0.0  # at least name weight


# ---------------------------------------------------------------------------
# filter_by_quality
# ---------------------------------------------------------------------------


class TestFilterByQuality:
    def test_filter_default_fair(self):
        t_good = _tool(completeness_score=0.7)
        t_poor = _tool(completeness_score=0.1)
        result = filter_by_quality([t_good, t_poor])
        assert len(result) == 1
        assert result[0] is t_good

    def test_filter_minimal(self):
        t = _tool(completeness_score=0.05)
        result = filter_by_quality([t], min_tier=QualityTier.MINIMAL)
        assert len(result) == 1

    def test_filter_excellent(self):
        t1 = _tool(completeness_score=0.9)
        t2 = _tool(completeness_score=0.7)
        result = filter_by_quality([t1, t2], min_tier=QualityTier.EXCELLENT)
        assert len(result) == 1
        assert result[0] is t1

    def test_filter_empty_list(self):
        assert filter_by_quality([]) == []


# ---------------------------------------------------------------------------
# get_score_breakdown
# ---------------------------------------------------------------------------


class TestScoreBreakdown:
    def test_bare_tool_breakdown(self):
        t = _tool()
        bd = get_score_breakdown(t)
        assert isinstance(bd, ScoreBreakdown)
        assert bd.total_score > 0.0  # name always present
        assert "tool.description" in bd.missing_fields
        assert "tool.domain" in bd.missing_fields
        assert "tool.endpoints" in bd.missing_fields
        assert bd.quality_tier == get_quality_tier(bd.total_score)
        assert len(bd.recommendations) > 0

    def test_rich_tool_breakdown(self):
        t = _rich_tool()
        bd = get_score_breakdown(t)
        assert bd.total_score > 0.6
        assert "tool.name" not in bd.missing_fields
        assert "tool.description" not in bd.missing_fields
        assert "tool.domain" not in bd.missing_fields

    def test_breakdown_component_scores_sum(self):
        t = _rich_tool()
        bd = get_score_breakdown(t)
        total_from_components = sum(bd.component_scores.values())
        assert abs(total_from_components - bd.total_score) < 0.01

    def test_custom_calculator(self):
        t = _tool()
        heavy = CompletenessWeights(tool_name=0.9)
        calc = CompletenessCalculator(heavy)
        bd = get_score_breakdown(t, calc)
        assert bd.component_scores["tool_name"] == 0.9


# ---------------------------------------------------------------------------
# generate_quality_report
# ---------------------------------------------------------------------------


class TestQualityReport:
    def test_empty_tools(self):
        report = generate_quality_report([])
        assert report["total_tools"] == 0
        assert report["average_score"] == 0.0
        assert sum(report["score_histogram"]) == 0

    def test_single_tool(self):
        t = _tool(completeness_score=0.75)
        report = generate_quality_report([t])
        assert report["total_tools"] == 1
        assert report["average_score"] == 0.75
        assert report["tier_distribution"]["good"] == 1
        assert sum(report["score_histogram"]) == 1

    def test_multiple_tools(self):
        tools = [
            _tool(completeness_score=0.9),
            _tool(completeness_score=0.5),
            _tool(completeness_score=0.1),
        ]
        report = generate_quality_report(tools)
        assert report["total_tools"] == 3
        assert report["tier_distribution"]["excellent"] == 1
        assert report["tier_distribution"]["fair"] == 1
        assert report["tier_distribution"]["minimal"] == 1
        assert sum(report["score_histogram"]) == 3

    def test_histogram_buckets(self):
        report = generate_quality_report([])
        assert len(report["score_histogram"]) == 10

    def test_top_issues(self):
        tools = [_tool(), _tool()]  # both missing desc, domain, endpoints
        report = generate_quality_report(tools)
        assert isinstance(report["top_issues"], list)
        assert len(report["top_issues"]) > 0

    def test_score_1_goes_to_last_bucket(self):
        t = _tool(completeness_score=1.0)
        report = generate_quality_report([t])
        assert report["score_histogram"][9] == 1


# ---------------------------------------------------------------------------
# DEFAULT_WEIGHTS
# ---------------------------------------------------------------------------


def test_default_weights_exist():
    assert isinstance(DEFAULT_WEIGHTS, CompletenessWeights)
    assert DEFAULT_WEIGHTS.tool_name == 0.10
    assert DEFAULT_WEIGHTS.endpoint_parameters == 0.35
    assert DEFAULT_WEIGHTS.param_type == 0.30


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_exports():
    from tooluse_gen.registry import (
        CompletenessCalculator,
        QualityTier,
    )

    assert CompletenessCalculator is not None
    assert QualityTier.EXCELLENT == "excellent"
