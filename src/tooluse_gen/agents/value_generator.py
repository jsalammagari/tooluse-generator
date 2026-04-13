"""Value generation engine for mock tool responses.

:class:`ValuePool` holds pre-built pools of realistic values organised
by type.  :class:`SchemaBasedGenerator` uses those pools together with
endpoint schemas and conversation context to produce grounded mock
responses.
"""

from __future__ import annotations

import json
import re
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

# Domain-specific name pools so a music tool returns song names, not "TurboCharge 3000"
_DOMAIN_NAMES: dict[str, list[str]] = {
    "Music": [
        "Bohemian Rhapsody", "Stairway to Heaven", "Hotel California",
        "Imagine", "Yesterday", "Smells Like Teen Spirit", "Billie Jean",
        "Like a Rolling Stone", "Purple Rain", "Hey Jude",
    ],
    "Entertainment": [
        "The Shawshank Redemption", "Inception", "The Dark Knight",
        "Pulp Fiction", "Forrest Gump", "The Matrix", "Interstellar",
        "The Godfather", "Fight Club", "Parasite",
    ],
    "Food": [
        "Margherita Pizza", "Pad Thai", "Sushi Platter", "Caesar Salad",
        "Beef Bourguignon", "Fish and Chips", "Chicken Tikka Masala",
        "Tacos al Pastor", "Ramen Bowl", "Croissant",
    ],
    "Sports": [
        "Manchester United vs Liverpool", "Super Bowl LVIII", "NBA Finals Game 5",
        "Wimbledon Final", "World Cup Semi-Final", "Champions League Draw",
        "Olympics 100m Sprint", "Tour de France Stage 12",
    ],
    "Travel": [
        "Paris City Tour", "Tokyo Express Pass", "Caribbean Cruise",
        "Alpine Hiking Trail", "Safari Adventure", "Beach Resort Package",
        "Historical Walking Tour", "Northern Lights Experience",
    ],
    "Finance": [
        "AAPL (Apple Inc.)", "GOOGL (Alphabet)", "TSLA (Tesla Inc.)",
        "S&P 500 Index Fund", "Bitcoin (BTC)", "US Treasury Bond 10Y",
        "MSFT (Microsoft Corp)", "AMZN (Amazon.com)",
    ],
    "Education": [
        "Introduction to Python", "Advanced Mathematics", "World History 101",
        "Machine Learning Fundamentals", "Creative Writing Workshop",
        "Data Science Bootcamp", "Biology Lab Manual",
    ],
    "Location": [
        "Times Square, New York", "Eiffel Tower, Paris", "Shibuya Crossing, Tokyo",
        "Big Ben, London", "Colosseum, Rome", "Opera House, Sydney",
        "Golden Gate Bridge, San Francisco",
    ],
    "Commerce": [
        "Premium Wireless Headphones", "Organic Cotton T-Shirt",
        "Stainless Steel Water Bottle", "Leather Messenger Bag",
        "Smart Home Hub", "Running Shoes Pro", "Ceramic Coffee Mug Set",
    ],
    "Business": [
        "Q4 Revenue Report", "Marketing Strategy Deck", "Board Meeting Minutes",
        "Annual Budget Proposal", "Client Engagement Summary",
        "Product Roadmap 2024", "Team Performance Review",
    ],
    "Cryptography": [
        "SHA-256 Hash Result", "RSA-2048 Key Pair", "AES Encrypted Payload",
        "Digital Signature (ECDSA)", "Base64 Decoded Message",
        "HMAC Authentication Token", "PGP Encrypted Document",
    ],
    "Cybersecurity": [
        "Vulnerability Scan Report", "Threat Assessment Summary",
        "Firewall Rule Update", "Intrusion Detection Alert",
        "SSL Certificate Status", "Access Audit Log",
    ],
    "Database": [
        "Users Table Query", "Customer Records Export", "Transaction Log Entry",
        "Schema Migration Result", "Index Optimization Report",
    ],
    "Email": [
        "Newsletter Campaign", "Welcome Email Template", "Password Reset Email",
        "Order Confirmation", "Meeting Invitation",
    ],
    "Gaming": [
        "World of Warcraft", "Minecraft", "League of Legends",
        "Fortnite Battle Royale", "The Legend of Zelda", "Elden Ring",
    ],
    "Health_and_Fitness": [
        "Daily Step Count", "Heart Rate Summary", "Workout Plan",
        "Nutrition Analysis", "Sleep Quality Report", "BMI Calculator Result",
    ],
    "Advertising": [
        "Summer Sale Campaign", "Brand Awareness Ad Set", "Retargeting Campaign",
        "Social Media Promotion", "Holiday Special Offer",
    ],
    "Search": [
        "Web Search Results", "Image Search Gallery", "News Article Digest",
        "Product Comparison", "Local Business Listing",
    ],
}

# Domain-specific entity type for ID prefixes
_DOMAIN_ID_PREFIX: dict[str, str] = {
    "Music": "TRK",
    "Entertainment": "MOV",
    "Food": "RST",
    "Sports": "EVT",
    "Travel": "BKG",
    "Finance": "TXN",
    "Education": "CRS",
    "Location": "LOC",
    "Commerce": "ORD",
    "Business": "DOC",
    "Cryptography": "KEY",
    "Cybersecurity": "SEC",
    "Database": "REC",
    "Email": "MSG",
    "Gaming": "GAM",
    "Health_and_Fitness": "FIT",
    "Advertising": "ADC",
    "Search": "RES",
    "Data": "DAT",
    "Communication": "COM",
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

    def get_domain_name(
        self, domain: str, rng: np.random.Generator, endpoint_name: str = ""
    ) -> str:
        """Return a domain-appropriate name (e.g., song title for Music)."""
        names = _DOMAIN_NAMES.get(domain)
        if names:
            return names[int(rng.integers(len(names)))]
        # For unmapped domains, generate a contextual name from the endpoint
        if endpoint_name:
            # Humanize the endpoint name for a more relevant result
            parts = re.sub(r"[_\-/]", " ", endpoint_name).strip().split()
            if len(parts) >= 2:
                label = " ".join(w.capitalize() for w in parts[:3])
                return f"{label} Result"
        # Last resort: domain-based generic
        if domain:
            clean_domain = domain.replace("_", " ")
            return f"{clean_domain} Service"
        return self.get("product", rng)

    def get_domain_id(
        self, domain: str, entity_type: str, context: ConversationContext, rng: np.random.Generator
    ) -> str:
        """Return an ID with a domain-appropriate prefix."""
        prefix = _DOMAIN_ID_PREFIX.get(domain, entity_type.upper()[:3])
        new_id = f"{prefix}-{int(rng.integers(1000, 9999))}"
        return new_id

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
        domain: str = "",
    ) -> dict[str, Any]:
        """Produce a mock response dict for *endpoint*."""
        ep_name = endpoint.name
        if endpoint.response_schema and endpoint.response_schema.properties:
            result = self._generate_from_schema(endpoint, arguments, context, rng, domain)
        else:
            structure = self._infer_response_structure(endpoint)
            if structure == "list":
                result = self._generate_list_response(endpoint, arguments, context, rng, domain, ep_name)
            elif structure == "status":
                result = self._generate_status_response(endpoint, arguments, context, rng)
            else:
                result = self._generate_object_response(endpoint, arguments, context, rng, domain, ep_name)

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
        domain: str = "",
    ) -> dict[str, Any]:
        schema = endpoint.response_schema
        if schema is None:
            return {}
        result: dict[str, Any] = {}
        ep_name = endpoint.name
        # Skip meta-schema fields that aren't actual response data
        _SKIP_KEYS = {"type", "properties", "required", "items", "additionalProperties"}
        for key in schema.properties:
            if key in _SKIP_KEYS:
                continue
            result[key] = self._generate_value_for_key(key, rng, context, domain, ep_name)
        # If schema only had meta-fields, generate a sensible default response
        if not result:
            result = self._generate_object_response(endpoint, arguments, context, rng, domain, ep_name)
        return result

    # ------------------------------------------------------------------
    # Key-based heuristic
    # ------------------------------------------------------------------

    def _generate_value_for_key(
        self, key: str, rng: np.random.Generator, context: ConversationContext,
        domain: str = "", ep_name: str = "",
    ) -> Any:
        k = key.lower()
        pool = self._pool

        if "id" in k:
            if domain:
                return pool.get_domain_id(domain, key, context, rng)
            return pool.get_id(key, context, rng)
        if "title" in k or ("name" in k and domain):
            return pool.get_domain_name(domain, rng, ep_name)
        if "city" in k and "name" in k:
            return pool.get("city", rng)
        if "name" in k:
            if domain:
                return pool.get_domain_name(domain, rng, ep_name)
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
        domain: str = "",
        ep_name: str = "",
    ) -> dict[str, Any]:
        count = int(rng.integers(2, 6))
        items: list[dict[str, Any]] = []
        for _ in range(count):
            item_name = self._pool.get_domain_name(domain, rng, ep_name)
            item: dict[str, Any] = {
                "id": self._pool.get_domain_id(domain, endpoint.name, context, rng) if domain
                else self._pool.get_id(f"{endpoint.name}_item", context, rng),
                "name": item_name,
                "status": self._pool.get("status", rng),
            }
            # Add domain-relevant extra fields
            if domain in ("Finance", "Commerce"):
                item["price"] = self._pool.get("price", rng)
                item["currency"] = self._pool.get("currency", rng)
            elif domain in ("Music", "Entertainment"):
                item["rating"] = self._pool.get("rating", rng)
            elif domain in ("Travel", "Location"):
                item["city"] = self._pool.get("city", rng)
            # Ensure unique IDs per item
            context.generated_ids.pop(f"{endpoint.name}_item", None)
            items.append(item)
        return {"results": items, "count": count}

    def _generate_object_response(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        context: ConversationContext,
        rng: np.random.Generator,
        domain: str = "",
        ep_name: str = "",
    ) -> dict[str, Any]:
        entity = endpoint.name.lower().replace(" ", "_")
        item_name = self._pool.get_domain_name(domain, rng, ep_name)
        result: dict[str, Any] = {
            "id": self._pool.get_domain_id(domain, entity, context, rng) if domain
            else self._pool.get_id(entity, context, rng),
            "name": item_name,
            "status": self._pool.get("status", rng),
            "created_at": self._pool.get("date", rng),
        }
        # Add domain-specific fields
        if domain in ("Finance", "Commerce"):
            result["amount"] = self._pool.get("price", rng)
            result["currency"] = self._pool.get("currency", rng)
        elif domain == "Travel":
            result["destination"] = self._pool.get("city", rng)
        elif domain in ("Music", "Entertainment"):
            result["rating"] = self._pool.get("rating", rng)
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
