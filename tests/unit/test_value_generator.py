"""Unit tests for Task 30 — Value generation engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.agents.value_generator import SchemaBasedGenerator, ValuePool
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    ResponseSchema,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> ConversationContext:
    return ConversationContext()


def _ep(
    name: str = "Get Data",
    method: HttpMethod = HttpMethod.GET,
    path: str = "/data",
    response_schema: ResponseSchema | None = None,
) -> Endpoint:
    return Endpoint(
        endpoint_id="e1", tool_id="t1", name=name,
        method=method, path=path, response_schema=response_schema,
    )


# ===========================================================================
# ValuePool — get
# ===========================================================================


class TestPoolGet:
    def test_city(self) -> None:
        pool = ValuePool()
        city = pool.get("city", np.random.default_rng(42))
        assert isinstance(city, str) and len(city) > 0

    def test_person_name(self) -> None:
        pool = ValuePool()
        name = pool.get("person_name", np.random.default_rng(42))
        assert isinstance(name, str) and " " in name

    def test_price_numeric(self) -> None:
        pool = ValuePool()
        price = pool.get("price", np.random.default_rng(42))
        assert isinstance(price, (int, float))

    def test_fuzzy_match(self) -> None:
        pool = ValuePool()
        val = pool.get("city_name", np.random.default_rng(42))
        assert isinstance(val, str) and len(val) > 0

    def test_unknown_fallback(self) -> None:
        pool = ValuePool()
        val = pool.get("zzz_xyzzy_unknown", np.random.default_rng(42))
        assert val == "zzz_xyzzy_unknown_001"


# ===========================================================================
# ValuePool — get_id
# ===========================================================================


class TestPoolGetId:
    def test_generates_new(self) -> None:
        pool = ValuePool()
        ctx = _ctx()
        new_id = pool.get_id("hotel", ctx, np.random.default_rng(42))
        assert new_id.startswith("HOT")
        assert "-" in new_id

    def test_reuses_existing(self) -> None:
        pool = ValuePool()
        ctx = _ctx()
        id1 = pool.get_id("hotel", ctx, np.random.default_rng(42))
        id2 = pool.get_id("hotel", ctx, np.random.default_rng(99))
        assert id1 == id2

    def test_format(self) -> None:
        pool = ValuePool()
        ctx = _ctx()
        new_id = pool.get_id("booking", ctx, np.random.default_rng(42))
        parts = new_id.split("-")
        assert len(parts) == 2
        assert parts[0] == "BOO"
        assert parts[1].isdigit()

    def test_stores_in_context(self) -> None:
        pool = ValuePool()
        ctx = _ctx()
        new_id = pool.get_id("order", ctx, np.random.default_rng(42))
        assert ctx.generated_ids["order"] == new_id


# ===========================================================================
# ValuePool — save/load
# ===========================================================================


class TestPoolSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        pool = ValuePool()
        p = tmp_path / "pools.json"
        pool.save(p)
        loaded = ValuePool.load(p)
        rng = np.random.default_rng(42)
        assert loaded.get("city", rng) == pool.get("city", np.random.default_rng(42))

    def test_load_from_file(self, tmp_path: Path) -> None:
        p = tmp_path / "pools.json"
        p.write_text('{"custom_key": ["val1", "val2"]}')
        loaded = ValuePool.load(p)
        val = loaded.get("custom_key", np.random.default_rng(42))
        assert val in ("val1", "val2")

    def test_custom_seed_preserved(self, tmp_path: Path) -> None:
        pool = ValuePool(seed_data={"fruit": ["apple", "banana"]})
        p = tmp_path / "pools.json"
        pool.save(p)
        loaded = ValuePool.load(p)
        val = loaded.get("fruit", np.random.default_rng(42))
        assert val in ("apple", "banana")


# ===========================================================================
# ValuePool — deterministic
# ===========================================================================


class TestPoolDeterministic:
    def test_same_seed(self) -> None:
        pool = ValuePool()
        v1 = pool.get("city", np.random.default_rng(42))
        v2 = pool.get("city", np.random.default_rng(42))
        assert v1 == v2

    def test_different_seed(self) -> None:
        pool = ValuePool()
        # With enough pool items, different seeds should usually differ
        values = {pool.get("city", np.random.default_rng(i)) for i in range(20)}
        assert len(values) > 1


# ===========================================================================
# SchemaBasedGenerator — generate_response
# ===========================================================================


class TestGenerateResponse:
    def test_returns_dict(self) -> None:
        gen = SchemaBasedGenerator()
        resp = gen.generate_response(_ep(), {}, _ctx(), np.random.default_rng(42))
        assert isinstance(resp, dict)

    def test_grounding_propagated(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(method=HttpMethod.POST, path="/bookings")
        resp = gen.generate_response(ep, {"hotel_id": "HOT-1234"}, _ctx(), np.random.default_rng(42))
        assert resp.get("hotel_id") == "HOT-1234"

    def test_get_plural_list(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(name="List Hotels", method=HttpMethod.GET, path="/hotels")
        resp = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(42))
        assert "results" in resp
        assert isinstance(resp["results"], list)

    def test_post_object(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(name="Create Booking", method=HttpMethod.POST, path="/bookings")
        resp = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(42))
        assert "id" in resp

    def test_delete_status(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(name="Delete Booking", method=HttpMethod.DELETE, path="/bookings/1")
        resp = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(42))
        assert resp["status"] == "success"

    def test_schema_properties_used(self) -> None:
        gen = SchemaBasedGenerator()
        schema = ResponseSchema(properties={"email": "string", "city": "string"})
        ep = _ep(response_schema=schema)
        resp = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(42))
        assert "email" in resp
        assert "city" in resp


# ===========================================================================
# SchemaBasedGenerator — _generate_value_for_key
# ===========================================================================


class TestGenerateValueForKey:
    def test_email(self) -> None:
        gen = SchemaBasedGenerator()
        val = gen._generate_value_for_key("email", np.random.default_rng(42), _ctx())
        assert "@" in str(val)

    def test_price(self) -> None:
        gen = SchemaBasedGenerator()
        val = gen._generate_value_for_key("total_price", np.random.default_rng(42), _ctx())
        assert isinstance(val, (int, float))

    def test_city_name(self) -> None:
        gen = SchemaBasedGenerator()
        val = gen._generate_value_for_key("city_name", np.random.default_rng(42), _ctx())
        assert isinstance(val, str) and len(val) > 0

    def test_hotel_id(self) -> None:
        gen = SchemaBasedGenerator()
        ctx = _ctx()
        val = gen._generate_value_for_key("hotel_id", np.random.default_rng(42), ctx)
        assert isinstance(val, str) and "-" in val

    def test_status(self) -> None:
        gen = SchemaBasedGenerator()
        val = gen._generate_value_for_key("status", np.random.default_rng(42), _ctx())
        assert isinstance(val, str)


# ===========================================================================
# SchemaBasedGenerator — list vs object vs status
# ===========================================================================


class TestResponseStructures:
    def test_list_has_results(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(method=HttpMethod.GET, path="/items")
        resp = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(42))
        assert "results" in resp
        assert "count" in resp
        assert len(resp["results"]) >= 2

    def test_object_has_id(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(method=HttpMethod.POST, path="/items")
        resp = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(42))
        assert "id" in resp

    def test_status_has_status(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(method=HttpMethod.DELETE, path="/items/1")
        resp = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(42))
        assert "status" in resp
        assert resp["status"] == "success"


# ===========================================================================
# Deterministic generation
# ===========================================================================


class TestDeterministic:
    def test_same_seed_same_response(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(method=HttpMethod.GET, path="/hotels")
        r1 = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(99))
        r2 = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(99))
        assert r1 == r2

    def test_different_seed_different_response(self) -> None:
        gen = SchemaBasedGenerator()
        ep = _ep(method=HttpMethod.GET, path="/hotels")
        responses = set()
        for seed in range(10):
            r = gen.generate_response(ep, {}, _ctx(), np.random.default_rng(seed))
            responses.add(str(r))
        assert len(responses) > 1
