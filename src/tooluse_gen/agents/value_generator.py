"""Value generation engine for mock tool responses.

:class:`ValuePool` holds pre-built pools of realistic values organised
by type.  :class:`SchemaBasedGenerator` uses those pools together with
endpoint schemas and conversation context to produce grounded mock
responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.registry.models import Endpoint
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.value_generator")

# ---------------------------------------------------------------------------
# Default value pools
# ---------------------------------------------------------------------------

_DEFAULT_POOLS: dict[str, list[Any]] = {
    "city": [
        "New York", "London", "Tokyo", "Paris", "Sydney", "Mumbai", "Berlin",
        "Toronto", "Singapore", "Dubai", "San Francisco", "Barcelona", "Seoul",
        "Amsterdam", "Rome", "Bangkok", "Cape Town", "Mexico City", "Istanbul",
        "Buenos Aires",
    ],
    "person_name": [
        "Alice Johnson", "Bob Smith", "Carlos Garcia", "Diana Chen",
        "Erik Müller", "Fatima Al-Hassan", "George Tanaka", "Hannah Kim",
        "Ivan Petrov", "Julia Santos", "Kevin O'Brien", "Leila Patel",
    ],
    "company": [
        "Acme Corp", "TechVentures Inc", "GlobalTrade LLC", "Pinnacle Systems",
        "NovaTech Solutions", "BlueOcean Analytics", "Summit Digital",
        "Horizon Partners", "CrestWave Technologies", "Apex Dynamics",
    ],
    "hotel": [
        "Grand Hotel", "Seaside Resort", "Mountain Lodge", "City Central Inn",
        "Royal Palace Hotel", "Sunset Beach Resort", "The Ritz Suites",
        "Harbor View Hotel", "Alpine Retreat", "Palm Garden Resort",
    ],
    "product": [
        "Widget Pro", "SmartGadget X1", "EcoClean Solution", "TurboCharge 3000",
        "AeroFit Tracker", "CloudSync Platform", "DataVault Enterprise",
        "StreamLine App", "PowerGrid Module", "ZenFlow Monitor",
    ],
    "price": [
        9.99, 14.99, 19.99, 29.99, 49.99, 79.99, 99.99, 149.99,
        199.99, 249.99, 349.99, 499.99, 599.99, 799.99, 999.99,
    ],
    "currency": ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "INR", "KRW"],
    "date": [
        "2024-01-15", "2024-02-20", "2024-03-22", "2024-04-10",
        "2024-05-05", "2024-06-18", "2024-07-30", "2024-08-12",
        "2024-09-25", "2024-10-08", "2024-11-14", "2024-12-01",
    ],
    "url": [
        "https://example.com/resource/1", "https://api.example.com/v2/data",
        "https://cdn.example.org/files/doc.pdf", "https://app.example.io/dashboard",
        "https://portal.example.net/account", "https://store.example.com/items/42",
        "https://blog.example.com/posts/hello-world",
    ],
    "email": [
        "alice@example.com", "bob@test.org", "carlos@mail.io",
        "diana@company.co", "erik@webservice.net", "fatima@domain.com",
        "george@startup.io", "hannah@enterprise.biz", "ivan@cloud.dev",
        "julia@platform.app",
    ],
    "phone": [
        "+1-555-0101", "+1-555-0142", "+44-20-7946-0958", "+81-3-1234-5678",
        "+61-2-9876-5432", "+49-30-1234-5678", "+33-1-2345-6789",
        "+91-22-1234-5678", "+86-10-1234-5678", "+82-2-1234-5678",
    ],
    "status": ["active", "pending", "completed", "cancelled", "processing",
               "approved", "rejected", "in_progress", "scheduled", "delivered"],
    "rating": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "description": [
        "A high-quality service with excellent reviews.",
        "Premium offering designed for professionals.",
        "Reliable and efficient solution for everyday needs.",
        "Award-winning product with global recognition.",
        "Cutting-edge technology at an affordable price.",
        "Trusted by millions of customers worldwide.",
        "Innovative approach to solving common challenges.",
        "Industry-leading performance and reliability.",
    ],
    "country": [
        "United States", "United Kingdom", "Japan", "Germany", "France",
        "Australia", "Canada", "India", "Brazil", "South Korea",
        "Italy", "Spain", "Netherlands", "Switzerland", "Singapore",
    ],
    "boolean": [True, False],
    "integer": [1, 2, 3, 5, 10, 15, 25, 50, 75, 100, 150, 250, 500, 1000],
}

# Mapping from common substrings to pool keys for fuzzy matching
_FUZZY_MAP: dict[str, str] = {
    "city": "city",
    "hotel": "hotel",
    "person": "person_name",
    "user": "person_name",
    "author": "person_name",
    "company": "company",
    "organisation": "company",
    "organization": "company",
    "product": "product",
    "item": "product",
    "price": "price",
    "cost": "price",
    "amount": "price",
    "currency": "currency",
    "date": "date",
    "time": "date",
    "url": "url",
    "link": "url",
    "email": "email",
    "phone": "phone",
    "status": "status",
    "state": "status",
    "rating": "rating",
    "score": "rating",
    "description": "description",
    "summary": "description",
    "country": "country",
    "nation": "country",
    "name": "person_name",
}


# ---------------------------------------------------------------------------
# ValuePool
# ---------------------------------------------------------------------------


class ValuePool:
    """Pool of realistic values organised by type for mock response generation."""

    def __init__(self, seed_data: dict[str, list[Any]] | None = None) -> None:
        self._pools: dict[str, list[Any]] = dict(seed_data) if seed_data else dict(_DEFAULT_POOLS)

    def get(self, value_type: str, rng: np.random.Generator) -> Any:
        """Sample a random value for *value_type*, with fuzzy fallback."""
        pool = self._pools.get(value_type)
        if pool:
            return pool[int(rng.integers(len(pool)))]

        # Fuzzy match
        vt_lower = value_type.lower()
        for substr, key in _FUZZY_MAP.items():
            if substr in vt_lower:
                pool = self._pools.get(key)
                if pool:
                    return pool[int(rng.integers(len(pool)))]

        return f"{value_type}_001"

    def get_id(
        self, entity_type: str, context: ConversationContext, rng: np.random.Generator
    ) -> str:
        """Return an existing or newly generated ID for *entity_type*."""
        existing = context.generated_ids.get(entity_type)
        if existing is not None:
            return existing
        prefix = entity_type.upper()[:3]
        new_id = f"{prefix}-{int(rng.integers(1000, 9999))}"
        context.generated_ids[entity_type] = new_id
        return new_id

    def save(self, path: Path) -> None:
        """Persist pools to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._pools, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ValuePool:
        """Load pools from a JSON file."""
        data: dict[str, list[Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(seed_data=data)


# ---------------------------------------------------------------------------
# SchemaBasedGenerator
# ---------------------------------------------------------------------------


class SchemaBasedGenerator:
    """Generates mock tool responses using endpoint schemas and value pools."""

    def __init__(self, pool: ValuePool | None = None) -> None:
        self._pool = pool or ValuePool()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_response(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Produce a mock response dict for *endpoint*."""
        if endpoint.response_schema and endpoint.response_schema.properties:
            result = self._generate_from_schema(endpoint, arguments, context, rng)
        else:
            structure = self._infer_response_structure(endpoint)
            if structure == "list":
                result = self._generate_list_response(endpoint, arguments, context, rng)
            elif structure == "status":
                result = self._generate_status_response(endpoint, arguments, context, rng)
            else:
                result = self._generate_object_response(endpoint, arguments, context, rng)

        # Grounding: propagate argument values into response
        for key, value in arguments.items():
            if isinstance(value, str):
                result[key] = value

        return result

    # ------------------------------------------------------------------
    # Schema-driven
    # ------------------------------------------------------------------

    def _generate_from_schema(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        schema = endpoint.response_schema
        if schema is None:
            return {}
        result: dict[str, Any] = {}
        for key in schema.properties:
            result[key] = self._generate_value_for_key(key, rng, context)
        return result

    # ------------------------------------------------------------------
    # Key-based heuristic
    # ------------------------------------------------------------------

    def _generate_value_for_key(
        self, key: str, rng: np.random.Generator, context: ConversationContext
    ) -> Any:
        k = key.lower()
        pool = self._pool

        if "id" in k:
            return pool.get_id(key, context, rng)
        if "city" in k and "name" in k:
            return pool.get("city", rng)
        if "name" in k:
            return pool.get("person_name", rng)
        if "email" in k:
            return pool.get("email", rng)
        if "phone" in k:
            return pool.get("phone", rng)
        if "url" in k or "link" in k:
            return pool.get("url", rng)
        if "price" in k or "cost" in k or "amount" in k:
            return pool.get("price", rng)
        if "date" in k or "time" in k:
            return pool.get("date", rng)
        if "status" in k:
            return pool.get("status", rng)
        if "rating" in k or "score" in k:
            return pool.get("rating", rng)
        if "description" in k or "summary" in k:
            return pool.get("description", rng)
        if "count" in k or "total" in k or "quantity" in k:
            return pool.get("integer", rng)
        if "country" in k:
            return pool.get("country", rng)
        if "city" in k:
            return pool.get("city", rng)
        if "active" in k or "enabled" in k or k.startswith("is_"):
            return pool.get("boolean", rng)
        return pool.get("description", rng)

    # ------------------------------------------------------------------
    # Structure inference
    # ------------------------------------------------------------------

    def _infer_response_structure(self, endpoint: Endpoint) -> str:
        method = endpoint.method if isinstance(endpoint.method, str) else endpoint.method
        path = endpoint.path.rstrip("/")

        if method == "DELETE":
            return "status"
        if method in ("POST", "PUT"):
            return "object"
        # GET with plural-looking path → list
        if method == "GET" and path and path.split("/")[-1].endswith("s"):
            return "list"
        return "object"

    # ------------------------------------------------------------------
    # Response generators
    # ------------------------------------------------------------------

    def _generate_list_response(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        count = int(rng.integers(2, 6))
        items: list[dict[str, Any]] = []
        for _ in range(count):
            item: dict[str, Any] = {
                "id": self._pool.get_id(f"{endpoint.name}_item", context, rng),
                "name": self._pool.get("product", rng),
                "status": self._pool.get("status", rng),
            }
            # Ensure unique IDs per item by clearing from context for next iteration
            context.generated_ids.pop(f"{endpoint.name}_item", None)
            items.append(item)
        return {"results": items, "count": count}

    def _generate_object_response(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        entity = endpoint.name.lower().replace(" ", "_")
        result: dict[str, Any] = {
            "id": self._pool.get_id(entity, context, rng),
            "name": self._pool.get("product", rng),
            "status": self._pool.get("status", rng),
            "created_at": self._pool.get("date", rng),
        }
        return result

    def _generate_status_response(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        context: ConversationContext,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "message": f"{endpoint.name} completed successfully",
        }
