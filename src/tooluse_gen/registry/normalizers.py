"""Field normalisation utilities for ToolBench data.

Each normaliser class handles one category of field (text, types, paths,
values).  :class:`FieldNormalizer` composes them all and tracks
statistics so the loader can report how much cleaning was required.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from tooluse_gen.registry.models import (
    HttpMethod,
    ParameterLocation,
    ParameterType,
)

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

# Characters not allowed in names (keep alphanum, space, dash, underscore)
_INVALID_NAME_RE = re.compile(r"[^\w\s\-]", re.UNICODE)
# Collapse runs of whitespace
_MULTI_WS_RE = re.compile(r"\s+")
# Characters not allowed in identifiers (keep alphanum, dash, underscore)
_INVALID_ID_RE = re.compile(r"[^\w\-]", re.UNICODE)
# Escaped unicode literals: \u00e9, \U000000e9
_ESCAPED_UNICODE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
# Trivial description that just echoes the name
_TRIVIAL_DESC_RE = re.compile(r"^(the|a|an|this)\s+", re.IGNORECASE)


class TextNormalizer:
    """Normalise text fields (names, descriptions, identifiers)."""

    @staticmethod
    def normalize_name(raw_name: str | None) -> str:
        """Return a clean name string.

        * Strip whitespace
        * Remove characters not in ``[\\w\\s\\-]``
        * Collapse internal whitespace
        * Truncate to 200 characters
        * ``None`` / empty → ``""``
        """
        if not raw_name:
            return ""
        text = raw_name.strip()
        text = _INVALID_NAME_RE.sub("", text)
        text = _MULTI_WS_RE.sub(" ", text).strip()
        return text[:200]

    @staticmethod
    def normalize_description(raw_desc: str | None, name: str = "") -> str:
        """Return a clean description string.

        * Strip whitespace, fix encoding, decode HTML entities
        * Collapse internal whitespace
        * Remove descriptions that merely repeat *name*
        * Truncate to 2 000 characters
        * ``None`` / empty → ``""``
        """
        if not raw_desc:
            return ""
        text = TextNormalizer.fix_encoding(raw_desc)
        text = html.unescape(text)
        text = _MULTI_WS_RE.sub(" ", text).strip()
        if not text:
            return ""
        # Discard if it just repeats the name
        if name:
            core = _TRIVIAL_DESC_RE.sub("", text).strip()
            if core.lower() == name.lower():
                return ""
        return text[:2000]

    @staticmethod
    def normalize_identifier(raw_id: str | None, fallback_source: str = "") -> str:
        """Return a clean, lower-case identifier.

        * Strip whitespace, replace spaces with underscores
        * Remove chars not in ``[\\w\\-]``
        * Lower-case
        * If empty, derive from *fallback_source*
        """
        src = (raw_id or "").strip()
        if not src and fallback_source:
            src = fallback_source.strip()
        src = src.replace(" ", "_")
        src = _INVALID_ID_RE.sub("", src)
        src = src.strip("_-").lower()
        return src

    @staticmethod
    def fix_encoding(text: str) -> str:
        """Best-effort repair of common encoding problems.

        Handles:
        * UTF-8 bytes mis-decoded as Latin-1 (e.g. ``Ã©`` → ``é``)
        * Escaped unicode literals (``\\\\u00e9`` → ``é``)
        * HTML entities (``&amp;`` → ``&``)
        """
        # Try to fix mojibake (Latin-1 misread of UTF-8)
        try:
            candidate = text.encode("latin-1").decode("utf-8")
            # Only accept if it actually changed and looks shorter/simpler
            if candidate != text:
                text = candidate
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        # Escaped unicode: \u00e9 → é
        text = _ESCAPED_UNICODE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)

        # HTML entities
        text = html.unescape(text)

        return text


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TypeNormalizer:
    """Normalise type-related fields."""

    TYPE_MAPPINGS: dict[str, ParameterType] = {
        # String
        "string": ParameterType.STRING,
        "str": ParameterType.STRING,
        "text": ParameterType.STRING,
        # Integer
        "integer": ParameterType.INTEGER,
        "int": ParameterType.INTEGER,
        "int32": ParameterType.INTEGER,
        "int64": ParameterType.INTEGER,
        "long": ParameterType.INTEGER,
        # Number
        "number": ParameterType.NUMBER,
        "float": ParameterType.NUMBER,
        "double": ParameterType.NUMBER,
        "decimal": ParameterType.NUMBER,
        # Boolean
        "boolean": ParameterType.BOOLEAN,
        "bool": ParameterType.BOOLEAN,
        # Array
        "array": ParameterType.ARRAY,
        "list": ParameterType.ARRAY,
        # Object
        "object": ParameterType.OBJECT,
        "dict": ParameterType.OBJECT,
        "json": ParameterType.OBJECT,
        "map": ParameterType.OBJECT,
        # File
        "file": ParameterType.FILE,
        "binary": ParameterType.FILE,
        # Date
        "date": ParameterType.DATE,
        "datetime": ParameterType.DATETIME,
        "date-time": ParameterType.DATETIME,
    }

    _METHOD_MAP: dict[str, HttpMethod] = {
        "get": HttpMethod.GET,
        "post": HttpMethod.POST,
        "put": HttpMethod.PUT,
        "delete": HttpMethod.DELETE,
        "patch": HttpMethod.PATCH,
        "head": HttpMethod.HEAD,
        "options": HttpMethod.OPTIONS,
    }

    _LOCATION_MAP: dict[str, ParameterLocation] = {
        "query": ParameterLocation.QUERY,
        "path": ParameterLocation.PATH,
        "header": ParameterLocation.HEADER,
        "body": ParameterLocation.BODY,
        "formdata": ParameterLocation.FORM,
        "form": ParameterLocation.FORM,
    }

    @classmethod
    def normalize_parameter_type(cls, raw_type: str | None) -> tuple[ParameterType, bool]:
        """Normalise a parameter type string.

        Returns ``(normalised_type, was_explicit)`` where
        *was_explicit* is ``True`` when the type was found in the
        mapping table.
        """
        if raw_type is None:
            return ParameterType.UNKNOWN, False
        key = str(raw_type).strip().lower()
        if not key:
            return ParameterType.UNKNOWN, False
        pt = cls.TYPE_MAPPINGS.get(key)
        if pt is not None:
            return pt, True
        return ParameterType.UNKNOWN, False

    @classmethod
    def normalize_http_method(cls, raw_method: str | None) -> HttpMethod:
        """Normalise an HTTP method string, defaulting to GET."""
        if not raw_method:
            return HttpMethod.GET
        key = str(raw_method).strip().lower()
        return cls._METHOD_MAP.get(key, HttpMethod.GET)

    @classmethod
    def normalize_location(cls, raw_location: str | None) -> ParameterLocation:
        """Normalise a parameter location string, defaulting to QUERY."""
        if not raw_location:
            return ParameterLocation.QUERY
        key = str(raw_location).strip().lower()
        return cls._LOCATION_MAP.get(key, ParameterLocation.QUERY)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Path parameters in various styles: {id}, :id, <id>
_COLON_PARAM_RE = re.compile(r":([a-zA-Z_]\w*)")
_ANGLE_PARAM_RE = re.compile(r"<([^>]+)>")
_BRACE_PARAM_RE = re.compile(r"\{([^}]+)\}")


class PathNormalizer:
    """Normalise URL and path fields."""

    @staticmethod
    def normalize_base_url(raw_url: str | None) -> str:
        """Return a clean base URL.

        * Strip whitespace and trailing slashes
        * Prefix ``https://`` when no scheme is present
        * Handle protocol-relative ``//host`` URLs
        * Return ``""`` for ``None``/empty
        """
        if not raw_url:
            return ""
        url = raw_url.strip().rstrip("/")
        if not url:
            return ""
        if url.startswith("//"):
            url = "https:" + url
        elif not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    @staticmethod
    def normalize_endpoint_path(raw_path: str | None) -> str:
        """Return a clean endpoint path.

        * Ensure leading ``/``
        * Remove trailing ``/`` (unless the path is ``/``)
        * Unify parameter styles to ``{name}``
        * Strip query strings
        * Decode percent-encoded characters
        """
        if not raw_path:
            return "/"
        path = raw_path.strip()
        # Strip query string
        path = path.split("?", 1)[0]
        # Decode percent-encoding
        path = unquote(path)
        # Unify param styles → {name}
        path = _COLON_PARAM_RE.sub(r"{\1}", path)
        path = _ANGLE_PARAM_RE.sub(r"{\1}", path)
        # Leading slash
        if not path.startswith("/"):
            path = "/" + path
        # Trailing slash
        if len(path) > 1:
            path = path.rstrip("/")
        return path

    @staticmethod
    def extract_path_parameters(path: str) -> list[str]:
        """Extract parameter names from a path template.

        Supports ``{name}``, ``:name``, and ``<name>`` styles.

        Example::

            extract_path_parameters("/users/{user_id}/posts/{post_id}")
            # → ["user_id", "post_id"]
        """
        names: list[str] = []
        names.extend(_BRACE_PARAM_RE.findall(path))
        names.extend(_COLON_PARAM_RE.findall(path))
        names.extend(_ANGLE_PARAM_RE.findall(path))
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


class ValueNormalizer:
    """Normalise parameter values (defaults, examples, enums)."""

    @staticmethod
    def normalize_default_value(raw_value: Any, param_type: ParameterType) -> Any:
        """Coerce *raw_value* to match *param_type* where possible.

        Returns ``None`` when coercion fails.
        """
        if raw_value is None:
            return None

        try:
            if param_type in (ParameterType.INTEGER, "integer"):
                return int(raw_value)
            if param_type in (ParameterType.NUMBER, "number"):
                return float(raw_value)
            if param_type in (ParameterType.BOOLEAN, "boolean"):
                if isinstance(raw_value, bool):
                    return raw_value
                return str(raw_value).strip().lower() in ("true", "1", "yes")
            if param_type in (ParameterType.STRING, "string"):
                return str(raw_value)
        except (ValueError, TypeError):
            return None

        return raw_value

    @staticmethod
    def normalize_enum_values(raw_enum: Any) -> list[str] | None:
        """Normalise enum / allowed-values to a ``list[str]``.

        Handles:
        * ``list`` → stringify each element
        * Comma-separated ``str`` → split and strip
        * Single scalar → wrap in a one-element list
        * ``None`` / empty → ``None``
        """
        if raw_enum is None:
            return None
        if isinstance(raw_enum, list):
            values = [str(v).strip() for v in raw_enum if v is not None]
            return values if values else None
        if isinstance(raw_enum, str):
            if not raw_enum.strip():
                return None
            if "," in raw_enum:
                values = [v.strip() for v in raw_enum.split(",") if v.strip()]
                return values if values else None
            return [raw_enum.strip()]
        return [str(raw_enum)]

    @staticmethod
    def normalize_required_flag(raw_value: Any) -> bool:
        """Coerce *raw_value* to a boolean required flag.

        Handles booleans, strings (``"true"``/``"false"``/``"yes"``/``"no"``/
        ``"optional"``), and integers (``1``/``0``).
        """
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, int):
            return raw_value != 0
        if isinstance(raw_value, str):
            lower = raw_value.strip().lower()
            if lower == "optional":
                return False
            return lower in ("true", "1", "yes")
        return bool(raw_value)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class NormalizationStats:
    """Track normalisation actions for reporting."""

    text_cleaned: int = 0
    encoding_fixed: int = 0
    types_normalized: int = 0
    types_defaulted: int = 0
    paths_normalized: int = 0
    values_coerced: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "text_cleaned": self.text_cleaned,
            "encoding_fixed": self.encoding_fixed,
            "types_normalized": self.types_normalized,
            "types_defaulted": self.types_defaulted,
            "paths_normalized": self.paths_normalized,
            "values_coerced": self.values_coerced,
        }


# ---------------------------------------------------------------------------
# Composite normaliser
# ---------------------------------------------------------------------------


class FieldNormalizer:
    """Compose all normaliser classes with statistics tracking."""

    def __init__(self) -> None:
        self.text = TextNormalizer()
        self.types = TypeNormalizer()
        self.paths = PathNormalizer()
        self.values = ValueNormalizer()
        self.stats = NormalizationStats()

    def normalize_tool_fields(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Return a new dict with normalised tool-level fields.

        The original *raw* dict is stored under ``"raw_schema"``.
        """
        name = self.text.normalize_name(raw.get("name") or raw.get("tool_name"))
        self.stats.text_cleaned += 1

        desc = self.text.normalize_description(
            raw.get("description") or raw.get("desc") or raw.get("summary"),
            name=name,
        )
        self.stats.text_cleaned += 1

        tool_id = self.text.normalize_identifier(
            raw.get("tool_id") or raw.get("id"),
            fallback_source=name,
        )
        self.stats.text_cleaned += 1

        base_url = self.paths.normalize_base_url(
            raw.get("base_url") or raw.get("api_host") or raw.get("host")
        )
        if base_url:
            self.stats.paths_normalized += 1

        domain = self.text.normalize_name(
            raw.get("domain") or raw.get("category") or raw.get("category_name")
        )

        return {
            "tool_id": tool_id,
            "name": name,
            "description": desc,
            "domain": domain,
            "base_url": base_url,
            "raw_schema": raw,
        }

    def normalize_endpoint_fields(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Return a new dict with normalised endpoint-level fields."""
        name = self.text.normalize_name(
            raw.get("name") or raw.get("api_name") or raw.get("operation_id")
        )
        self.stats.text_cleaned += 1

        desc = self.text.normalize_description(
            raw.get("description") or raw.get("api_description") or raw.get("desc"),
            name=name,
        )
        self.stats.text_cleaned += 1

        method = self.types.normalize_http_method(raw.get("method") or raw.get("http_method"))
        self.stats.types_normalized += 1

        path = self.paths.normalize_endpoint_path(
            raw.get("path") or raw.get("url") or raw.get("api_url") or raw.get("endpoint")
        )
        self.stats.paths_normalized += 1

        return {
            "name": name,
            "description": desc,
            "method": method,
            "path": path,
            "raw_definition": raw,
        }

    def normalize_parameter_fields(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Return a new dict with normalised parameter-level fields."""
        name = self.text.normalize_name(raw.get("name") or raw.get("param_name") or raw.get("key"))
        self.stats.text_cleaned += 1

        desc = self.text.normalize_description(
            raw.get("description") or raw.get("desc"),
            name=name,
        )
        self.stats.text_cleaned += 1

        raw_type = raw.get("type") or raw.get("param_type") or raw.get("data_type")
        param_type, was_explicit = self.types.normalize_parameter_type(raw_type)
        if was_explicit:
            self.stats.types_normalized += 1
        else:
            self.stats.types_defaulted += 1

        location = self.types.normalize_location(
            raw.get("in") or raw.get("location") or raw.get("param_in")
        )

        required = self.values.normalize_required_flag(
            raw.get("required", raw.get("is_required", False))
        )

        raw_default = raw.get("default") or raw.get("default_value")
        default = self.values.normalize_default_value(raw_default, param_type)
        if default is not None and default != raw_default:
            self.stats.values_coerced += 1

        enum_values = self.values.normalize_enum_values(raw.get("enum"))

        return {
            "name": name,
            "description": desc,
            "param_type": param_type,
            "location": location,
            "required": required,
            "default": default,
            "enum_values": enum_values,
            "has_type": was_explicit,
            "raw_definition": raw,
        }
