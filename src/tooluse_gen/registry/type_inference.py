"""Parameter type inference heuristics.

When ToolBench parameters lack an explicit type, this module infers the
most likely :class:`ParameterType` from multiple evidence sources
(name patterns, default values, enum values, description text, examples,
and parameter location).  Each source produces a :class:`TypeEvidence`
with a confidence score; the :class:`ParameterTypeInferrer` combines
all evidence and returns the highest-confidence result.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from tooluse_gen.registry.models import (
    Endpoint,
    ParameterLocation,
    ParameterType,
)

# ---------------------------------------------------------------------------
# Inference rules
# ---------------------------------------------------------------------------


@dataclass
class InferenceRule:
    """A single name-based type inference rule."""

    name: str
    pattern: str | re.Pattern[str]
    inferred_type: ParameterType
    confidence: float  # 0.0 – 1.0
    priority: int  # lower = higher priority

    def matches(self, text: str) -> bool:
        """Return ``True`` when *text* matches this rule's pattern."""
        if isinstance(self.pattern, re.Pattern):
            return self.pattern.search(text) is not None
        return re.search(self.pattern, text, re.IGNORECASE) is not None


# Comprehensive name-based rules, ordered by priority then confidence.
NAME_RULES: list[InferenceRule] = [
    # ── ID patterns → STRING ──────────────────────────────────────────
    InferenceRule("uuid_pattern", r"uuid|guid", ParameterType.STRING, 0.95, 1),
    InferenceRule("id_exact", r"^id$", ParameterType.STRING, 0.9, 1),
    InferenceRule("id_suffix", r"[_-]id$", ParameterType.STRING, 0.9, 1),
    # ── Pagination / count → INTEGER ──────────────────────────────────
    InferenceRule(
        "pagination",
        r"^(limit|offset|page|per_page|page_size|skip|top)$",
        ParameterType.INTEGER,
        0.95,
        1,
    ),
    InferenceRule("count_suffix", r"[_-]?count$", ParameterType.INTEGER, 0.9, 2),
    InferenceRule("num_prefix", r"^num[_-]", ParameterType.INTEGER, 0.85, 2),
    InferenceRule(
        "size_pattern",
        r"^(size|length|width|height|max|min|total)$",
        ParameterType.INTEGER,
        0.85,
        2,
    ),
    InferenceRule(
        "year_month_day", r"^(year|month|day|hour|minute)$", ParameterType.INTEGER, 0.8, 3
    ),
    # ── Boolean patterns ──────────────────────────────────────────────
    InferenceRule("is_prefix", r"^is[_-]", ParameterType.BOOLEAN, 0.9, 2),
    InferenceRule("has_prefix", r"^has[_-]", ParameterType.BOOLEAN, 0.9, 2),
    InferenceRule(
        "bool_keywords",
        r"^(enable|disable|active|verbose|recursive|force|dry_run|include|exclude|ascending|descending)$",
        ParameterType.BOOLEAN,
        0.85,
        2,
    ),
    # ── Date / time patterns ──────────────────────────────────────────
    InferenceRule("timestamp_pattern", r"timestamp", ParameterType.DATETIME, 0.9, 2),
    InferenceRule("datetime_suffix", r"[_-](at|on)$", ParameterType.DATETIME, 0.8, 3),
    InferenceRule("date_suffix", r"[_-]?date$", ParameterType.DATE, 0.9, 2),
    InferenceRule(
        "date_keywords",
        r"^(created|updated|modified|expires|start|end)$",
        ParameterType.DATE,
        0.75,
        3,
    ),
    # ── Numeric patterns ──────────────────────────────────────────────
    InferenceRule(
        "price_pattern", r"(price|cost|amount|fee|rate|balance)", ParameterType.NUMBER, 0.85, 2
    ),
    InferenceRule(
        "lat_lon",
        r"^(lat|latitude|lon|lng|longitude)$",
        ParameterType.NUMBER,
        0.95,
        1,
    ),
    InferenceRule(
        "score_weight", r"(score|weight|ratio|percent|factor)", ParameterType.NUMBER, 0.8, 3
    ),
    # ── Array patterns ────────────────────────────────────────────────
    InferenceRule("ids_suffix", r"[_-]?ids$", ParameterType.ARRAY, 0.9, 2),
    InferenceRule("list_suffix", r"[_-]?list$", ParameterType.ARRAY, 0.85, 2),
    InferenceRule("items_suffix", r"[_-]?items$", ParameterType.ARRAY, 0.8, 3),
    InferenceRule(
        "tags_categories", r"^(tags|categories|labels|fields|columns)$", ParameterType.ARRAY, 0.8, 3
    ),
    # ── Object patterns ───────────────────────────────────────────────
    InferenceRule(
        "body_data",
        r"^(body|data|payload|metadata|config|options|settings|filter)$",
        ParameterType.OBJECT,
        0.75,
        3,
    ),
    # ── String fallbacks (low priority) ───────────────────────────────
    InferenceRule("name_pattern", r"[_-]?name$", ParameterType.STRING, 0.8, 3),
    InferenceRule("email_pattern", r"email", ParameterType.STRING, 0.9, 2),
    InferenceRule("url_pattern", r"(url|link|href|uri)$", ParameterType.STRING, 0.85, 2),
    InferenceRule(
        "token_key", r"(token|key|secret|password|api_key)$", ParameterType.STRING, 0.85, 2
    ),
    InferenceRule("query_search", r"^(query|search|q|keyword|term)$", ParameterType.STRING, 0.8, 3),
    InferenceRule(
        "format_type",
        r"^(format|type|kind|sort|order|status|state|mode|lang|locale|currency)$",
        ParameterType.STRING,
        0.75,
        3,
    ),
]

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass
class TypeEvidence:
    """Evidence for a type inference from a single source."""

    source: str  # "name_pattern", "default_value", "enum_values", etc.
    inferred_type: ParameterType
    confidence: float
    reasoning: str


@dataclass
class InferenceResult:
    """Aggregated result of type inference."""

    inferred_type: ParameterType
    confidence: float
    is_inferred: bool
    evidences: list[TypeEvidence] = field(default_factory=list)

    @property
    def primary_evidence(self) -> TypeEvidence | None:
        """Highest-confidence evidence, or ``None``."""
        if not self.evidences:
            return None
        return max(self.evidences, key=lambda e: e.confidence)

    @property
    def reasoning(self) -> str:
        """Human-readable explanation of the inference."""
        if not self.evidences:
            return "No evidence available."
        primary = self.primary_evidence
        assert primary is not None
        parts = [f"Inferred as {self.inferred_type} (confidence {self.confidence:.2f})."]
        parts.append(f"Primary: {primary.reasoning}")
        if len(self.evidences) > 1:
            others = [e for e in self.evidences if e is not primary]
            parts.append("Supporting: " + "; ".join(e.reasoning for e in others))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Description keywords
# ---------------------------------------------------------------------------

_DESC_PATTERNS: list[tuple[re.Pattern[str], ParameterType, float]] = [
    (re.compile(r"\bnumber\b|\binteger\b|\bint\b", re.I), ParameterType.INTEGER, 0.7),
    (re.compile(r"\btrue\b.*\bfalse\b|\bboolean\b|\bbool\b", re.I), ParameterType.BOOLEAN, 0.75),
    (re.compile(r"\bdate\b|\biso\s*8601\b", re.I), ParameterType.DATE, 0.7),
    (re.compile(r"\btimestamp\b|\bdatetime\b", re.I), ParameterType.DATETIME, 0.7),
    (re.compile(r"\bfloat\b|\bdecimal\b|\bdouble\b", re.I), ParameterType.NUMBER, 0.7),
    (re.compile(r"\barray\b|\blist\b|\bcomma[- ]?separated\b", re.I), ParameterType.ARRAY, 0.7),
    (re.compile(r"\bjson\b|\bobject\b|\bdict\b", re.I), ParameterType.OBJECT, 0.65),
    (re.compile(r"\burl\b|\buri\b|\blink\b", re.I), ParameterType.STRING, 0.65),
]

# ---------------------------------------------------------------------------
# Inferrer
# ---------------------------------------------------------------------------

_PYTHON_TYPE_MAP: dict[type, ParameterType] = {
    bool: ParameterType.BOOLEAN,
    int: ParameterType.INTEGER,
    float: ParameterType.NUMBER,
    str: ParameterType.STRING,
    list: ParameterType.ARRAY,
    dict: ParameterType.OBJECT,
}


class ParameterTypeInferrer:
    """Infer parameter types by combining multiple heuristic sources."""

    def __init__(
        self,
        name_rules: list[InferenceRule] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        rules = name_rules if name_rules is not None else NAME_RULES
        self.name_rules = sorted(rules, key=lambda r: r.priority)
        self.min_confidence = min_confidence

    # -- public API ---------------------------------------------------------

    def infer_type(
        self,
        name: str,
        description: str = "",
        default_value: Any = None,
        enum_values: list[str] | None = None,
        location: ParameterLocation = ParameterLocation.QUERY,
        examples: list[Any] | None = None,
    ) -> InferenceResult:
        """Infer a parameter's type from all available signals."""
        evidences: list[TypeEvidence] = []

        # 1. Default value (strongest signal)
        if default_value is not None:
            evidences.append(self._infer_from_default(default_value))

        # 2. Enum values
        if enum_values:
            evidences.append(self._infer_from_enum(enum_values))

        # 3. Examples
        if examples:
            evidences.append(self._infer_from_examples(examples))

        # 4. Name patterns
        name_ev = self._infer_from_name(name)
        if name_ev is not None:
            evidences.append(name_ev)

        # 5. Description
        if description:
            desc_ev = self._infer_from_description(description)
            if desc_ev is not None:
                evidences.append(desc_ev)

        # 6. Context (location)
        evidences.append(self._infer_from_context(location))

        return self._combine_evidence(evidences)

    # -- evidence sources ---------------------------------------------------

    def _infer_from_name(self, name: str) -> TypeEvidence | None:
        """Apply name-based rules; return first match per priority."""
        lower = name.lower().strip()
        if not lower:
            return None
        for rule in self.name_rules:
            if rule.matches(lower):
                return TypeEvidence(
                    source="name_pattern",
                    inferred_type=rule.inferred_type,
                    confidence=rule.confidence,
                    reasoning=f"Name '{name}' matches rule '{rule.name}'.",
                )
        return None

    def _infer_from_default(self, default_value: Any) -> TypeEvidence:
        """Infer type from a default value's Python type."""
        # bool must be checked before int (bool is a subclass of int)
        pt = self._python_type_to_param_type(type(default_value))
        # Extra: if it's a string that looks numeric, note it
        reasoning = f"Default value {default_value!r} is {type(default_value).__name__}."
        if isinstance(default_value, str):
            if self._is_integer_string(default_value):
                pt = ParameterType.INTEGER
                reasoning = f"Default value '{default_value}' is a numeric string (integer)."
            elif self._is_numeric_string(default_value):
                pt = ParameterType.NUMBER
                reasoning = f"Default value '{default_value}' is a numeric string."
        return TypeEvidence(
            source="default_value",
            inferred_type=pt,
            confidence=0.95,
            reasoning=reasoning,
        )

    def _infer_from_enum(self, enum_values: list[str]) -> TypeEvidence:
        """Infer type from enum values — almost always STRING."""
        # Check if all values are numeric
        if all(self._is_integer_string(str(v)) for v in enum_values):
            return TypeEvidence(
                source="enum_values",
                inferred_type=ParameterType.INTEGER,
                confidence=0.85,
                reasoning=f"All {len(enum_values)} enum values are integers.",
            )
        if all(self._is_numeric_string(str(v)) for v in enum_values):
            return TypeEvidence(
                source="enum_values",
                inferred_type=ParameterType.NUMBER,
                confidence=0.85,
                reasoning=f"All {len(enum_values)} enum values are numeric.",
            )
        if all(str(v).lower() in ("true", "false") for v in enum_values):
            return TypeEvidence(
                source="enum_values",
                inferred_type=ParameterType.BOOLEAN,
                confidence=0.9,
                reasoning="Enum values are boolean strings.",
            )
        return TypeEvidence(
            source="enum_values",
            inferred_type=ParameterType.STRING,
            confidence=0.9,
            reasoning=f"Enum with {len(enum_values)} string values.",
        )

    def _infer_from_description(self, description: str) -> TypeEvidence | None:
        """Search description for type-hinting keywords."""
        for pattern, pt, conf in _DESC_PATTERNS:
            if pattern.search(description):
                return TypeEvidence(
                    source="description",
                    inferred_type=pt,
                    confidence=conf,
                    reasoning=f"Description contains keyword matching {pt}.",
                )
        return None

    def _infer_from_examples(self, examples: list[Any]) -> TypeEvidence:
        """Infer type from example values (majority vote)."""
        type_counts: dict[ParameterType, int] = defaultdict(int)
        for ex in examples:
            pt = self._python_type_to_param_type(type(ex))
            if isinstance(ex, str):
                if self._is_integer_string(ex):
                    pt = ParameterType.INTEGER
                elif self._is_numeric_string(ex):
                    pt = ParameterType.NUMBER
            type_counts[pt] += 1
        best = max(type_counts, key=lambda t: type_counts[t])
        return TypeEvidence(
            source="examples",
            inferred_type=best,
            confidence=0.85,
            reasoning=f"Majority of {len(examples)} examples are {best}.",
        )

    def _infer_from_context(self, location: ParameterLocation) -> TypeEvidence:
        """Infer type from parameter location (weak signal)."""
        if location == ParameterLocation.BODY:
            return TypeEvidence(
                source="context",
                inferred_type=ParameterType.OBJECT,
                confidence=0.4,
                reasoning="Body parameters are often objects.",
            )
        if location == ParameterLocation.PATH:
            return TypeEvidence(
                source="context",
                inferred_type=ParameterType.STRING,
                confidence=0.5,
                reasoning="Path parameters are usually strings.",
            )
        if location == ParameterLocation.HEADER:
            return TypeEvidence(
                source="context",
                inferred_type=ParameterType.STRING,
                confidence=0.6,
                reasoning="Header values are strings.",
            )
        return TypeEvidence(
            source="context",
            inferred_type=ParameterType.STRING,
            confidence=0.3,
            reasoning="Default assumption for query parameters.",
        )

    # -- combining ----------------------------------------------------------

    def _combine_evidence(self, evidences: list[TypeEvidence]) -> InferenceResult:
        """Combine evidence: group by type, pick highest aggregate confidence."""
        if not evidences:
            return InferenceResult(
                inferred_type=ParameterType.STRING,
                confidence=0.0,
                is_inferred=True,
                evidences=[],
            )

        # Group by type, aggregate confidence (max per type, boosted by count)
        type_scores: dict[ParameterType, float] = defaultdict(float)
        type_evidences: dict[ParameterType, list[TypeEvidence]] = defaultdict(list)

        for ev in evidences:
            type_evidences[ev.inferred_type].append(ev)
            # Take the max confidence for each type, then boost slightly per extra evidence
            current_max = type_scores[ev.inferred_type]
            type_scores[ev.inferred_type] = max(current_max, ev.confidence)

        # Boost for multiple agreeing sources (capped at +0.1)
        for pt, evs in type_evidences.items():
            if len(evs) > 1:
                boost = min(0.05 * (len(evs) - 1), 0.1)
                type_scores[pt] = min(type_scores[pt] + boost, 1.0)

        best_type = max(type_scores, key=lambda t: type_scores[t])
        best_confidence = type_scores[best_type]

        if best_confidence < self.min_confidence:
            return InferenceResult(
                inferred_type=ParameterType.STRING,
                confidence=best_confidence,
                is_inferred=True,
                evidences=evidences,
            )

        return InferenceResult(
            inferred_type=best_type,
            confidence=round(best_confidence, 4),
            is_inferred=True,
            evidences=evidences,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _python_type_to_param_type(py_type: type) -> ParameterType:
        """Map a Python type to a :class:`ParameterType`."""
        # bool before int (bool is subclass of int)
        if py_type is bool:
            return ParameterType.BOOLEAN
        return _PYTHON_TYPE_MAP.get(py_type, ParameterType.STRING)

    @staticmethod
    def _is_numeric_string(value: str) -> bool:
        """True when *value* represents a number (int or float)."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_integer_string(value: str) -> bool:
        """True when *value* represents an integer."""
        try:
            int(value)
            return "." not in value
        except (ValueError, TypeError):
            return False


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def infer_endpoint_parameter_types(
    endpoint: Endpoint,
    inferrer: ParameterTypeInferrer | None = None,
) -> Endpoint:
    """Infer types for all parameters lacking explicit types.

    Modifies *endpoint* in-place and returns it.
    """
    inf = inferrer or ParameterTypeInferrer()
    for param in endpoint.parameters:
        if param.has_type and not param.inferred_type:
            continue  # already has an explicit type
        result = inf.infer_type(
            name=param.name,
            description=param.description,
            default_value=param.default,
            enum_values=param.enum_values,
            location=ParameterLocation(param.location)
            if isinstance(param.location, str)
            else param.location,
        )
        if result.confidence >= inf.min_confidence:
            param.param_type = result.inferred_type
            param.has_type = True
            param.inferred_type = True
    return endpoint
