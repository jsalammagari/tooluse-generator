"""ToolBench JSON loader.

Ingests raw ToolBench API definitions from JSON files — one-per-tool or
bulk — and produces normalised :class:`Tool` objects.  The loader is
deliberately lenient: malformed entries are logged and skipped (unless
``strict_mode`` is enabled), and missing fields are filled with safe
defaults so downstream code can rely on a uniform schema.

Supported input formats
-----------------------
- **single_tool** — a dict with tool-level keys (``name``, ``endpoints``, …).
- **tool_list** — a JSON array of tool dicts.
- **toolbench_v1** — ``{"tools": [...]}``.
- **toolbench_v2** — ``{"api_list": [...]}``.
- **openapi** — dict with ``"openapi"`` or ``"swagger"`` key.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tooluse_gen.registry.completeness import CompletenessCalculator
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterLocation,
    ParameterType,
    ResponseSchema,
    Tool,
    generate_endpoint_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolBenchLoaderError(Exception):
    """Base exception for loader errors."""


class InvalidToolError(ToolBenchLoaderError):
    """Raised when a tool cannot be parsed."""

    def __init__(self, message: str, raw_data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_data = raw_data


class MissingRequiredFieldError(ToolBenchLoaderError):
    """Raised when a required field is missing."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoaderConfig:
    """Configuration for the ToolBench loader."""

    # Parsing behaviour
    strict_mode: bool = False
    infer_types: bool = True

    # Filtering
    min_endpoints: int = 1
    max_endpoints: int = 100

    # Quality
    calculate_completeness: bool = True
    min_completeness: float = 0.0

    # Limits
    max_tools: int | None = None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class LoaderStats:
    """Statistics from a load operation."""

    files_processed: int = 0
    files_failed: int = 0
    tools_loaded: int = 0
    tools_skipped: int = 0
    endpoints_loaded: int = 0
    parameters_loaded: int = 0

    # Quality distribution
    quality_distribution: dict[str, int] = field(default_factory=dict)

    # Common issues
    missing_descriptions: int = 0
    missing_types: int = 0
    inferred_types: int = 0
    missing_response_schemas: int = 0

    # Errors
    errors: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging / serialisation."""
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "tools_loaded": self.tools_loaded,
            "tools_skipped": self.tools_skipped,
            "endpoints_loaded": self.endpoints_loaded,
            "parameters_loaded": self.parameters_loaded,
            "quality_distribution": self.quality_distribution,
            "missing_descriptions": self.missing_descriptions,
            "missing_types": self.missing_types,
            "inferred_types": self.inferred_types,
            "missing_response_schemas": self.missing_response_schemas,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, ParameterType] = {
    "string": ParameterType.STRING,
    "str": ParameterType.STRING,
    "text": ParameterType.STRING,
    "integer": ParameterType.INTEGER,
    "int": ParameterType.INTEGER,
    "long": ParameterType.INTEGER,
    "number": ParameterType.NUMBER,
    "float": ParameterType.NUMBER,
    "double": ParameterType.NUMBER,
    "decimal": ParameterType.NUMBER,
    "boolean": ParameterType.BOOLEAN,
    "bool": ParameterType.BOOLEAN,
    "array": ParameterType.ARRAY,
    "list": ParameterType.ARRAY,
    "object": ParameterType.OBJECT,
    "dict": ParameterType.OBJECT,
    "map": ParameterType.OBJECT,
    "file": ParameterType.FILE,
    "binary": ParameterType.FILE,
    "date": ParameterType.DATE,
    "datetime": ParameterType.DATETIME,
    "date-time": ParameterType.DATETIME,
}

_LOCATION_MAP: dict[str, ParameterLocation] = {
    "query": ParameterLocation.QUERY,
    "path": ParameterLocation.PATH,
    "header": ParameterLocation.HEADER,
    "body": ParameterLocation.BODY,
    "formdata": ParameterLocation.FORM,
    "form": ParameterLocation.FORM,
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


class ParameterTypeInferrer:
    """Infer parameter types from names and default values."""

    _ID_RE = re.compile(r"(^id$|_id$|Id$|uuid|guid)", re.IGNORECASE)
    _BOOL_RE = re.compile(r"^(is_|has_|enable|disable|flag|active|verbose)", re.IGNORECASE)
    _NUM_RE = re.compile(
        r"(count|total|limit|offset|page|size|width|height|lat|lon|price|amount|num)",
        re.IGNORECASE,
    )

    def infer(self, name: str, default: Any | None = None) -> ParameterType:
        """Return best-guess type for a parameter."""
        if default is not None:
            if isinstance(default, bool):
                return ParameterType.BOOLEAN
            if isinstance(default, int):
                return ParameterType.INTEGER
            if isinstance(default, float):
                return ParameterType.NUMBER
            if isinstance(default, list):
                return ParameterType.ARRAY
            if isinstance(default, dict):
                return ParameterType.OBJECT

        if self._BOOL_RE.search(name):
            return ParameterType.BOOLEAN
        if self._NUM_RE.search(name):
            return ParameterType.NUMBER
        if self._ID_RE.search(name):
            return ParameterType.STRING

        return ParameterType.STRING


# ---------------------------------------------------------------------------
# RawToolParser
# ---------------------------------------------------------------------------


class RawToolParser:
    """Parse raw ToolBench JSON files into intermediate dicts."""

    def __init__(self, config: LoaderConfig) -> None:
        self.config = config

    def parse_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Read and parse a single JSON file, returning raw tool dicts."""
        text = file_path.read_text(encoding="utf-8")
        data = json.loads(text)
        fmt = self.detect_format(data)
        return self._extract_tools(data, fmt)

    def parse_directory(self, dir_path: Path, recursive: bool = True) -> Iterator[dict[str, Any]]:
        """Yield raw tool dicts from all JSON files under *dir_path*."""
        pattern = "**/*.json" if recursive else "*.json"
        for path in sorted(dir_path.glob(pattern)):
            if not path.is_file():
                continue
            try:
                yield from self.parse_file(path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", path, exc)

    def detect_format(self, data: dict[str, Any] | list[Any]) -> str:
        """Detect the data format of a parsed JSON payload."""
        if isinstance(data, list):
            return "tool_list"
        if not isinstance(data, dict):
            return "unknown"
        if "openapi" in data or "swagger" in data:
            return "openapi"
        if "tools" in data and isinstance(data["tools"], list):
            return "toolbench_v1"
        if "api_list" in data and isinstance(data["api_list"], list):
            # Distinguish single-tool-with-endpoints from a list of tools.
            # If the dict also carries a tool-level name, treat it as a
            # single tool whose api_list contains endpoints.
            if any(k in data for k in ("name", "tool_name", "api_name", "title")):
                return "single_tool"
            return "toolbench_v2"
        # Looks like a single tool dict
        return "single_tool"

    # -- private ------------------------------------------------------------

    def _extract_tools(
        self, data: dict[str, Any] | list[Any], fmt: str
    ) -> list[dict[str, Any]]:
        if fmt == "tool_list" and isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            if fmt == "toolbench_v1":
                return [d for d in data.get("tools", []) if isinstance(d, dict)]
            if fmt == "toolbench_v2":
                return [d for d in data.get("api_list", []) if isinstance(d, dict)]
            if fmt in ("openapi", "single_tool"):
                return [data]
        return []


# ---------------------------------------------------------------------------
# ToolNormalizer
# ---------------------------------------------------------------------------


class ToolNormalizer:
    """Normalise raw dicts into :class:`Tool` model instances."""

    TOOL_FIELD_MAPPINGS: dict[str, list[str]] = {
        "tool_id": ["tool_id", "id", "api_id", "tool_name_id"],
        "name": ["name", "tool_name", "api_name", "title"],
        "description": ["description", "desc", "tool_description", "summary"],
        "domain": ["domain", "category", "category_name", "type"],
        "base_url": ["base_url", "api_host", "host", "server_url"],
        "endpoints": ["endpoints", "api_list", "apis", "operations", "paths"],
    }

    ENDPOINT_FIELD_MAPPINGS: dict[str, list[str]] = {
        "name": ["name", "api_name", "operation_id", "summary"],
        "description": ["description", "api_description", "desc"],
        "method": ["method", "http_method", "request_method"],
        "path": ["path", "url", "api_url", "endpoint"],
        "parameters": ["parameters", "params", "required_parameters"],
    }

    PARAM_FIELD_MAPPINGS: dict[str, list[str]] = {
        "name": ["name", "param_name", "parameter_name", "key"],
        "description": ["description", "desc", "parameter_description"],
        "type": ["type", "param_type", "data_type", "schema.type"],
        "required": ["required", "is_required", "optional"],
        "default": ["default", "default_value", "example"],
        "location": ["in", "location", "param_in"],
    }

    def __init__(self, config: LoaderConfig) -> None:
        self.config = config
        self.type_inferrer = ParameterTypeInferrer() if config.infer_types else None

    # -- public API ---------------------------------------------------------

    def normalize_tool(self, raw: dict[str, Any], source_file: str = "") -> Tool | None:
        """Normalise a raw tool dict into a :class:`Tool`.

        Returns ``None`` when the tool is invalid and ``strict_mode`` is
        ``False``.  Raises :class:`InvalidToolError` otherwise.
        """
        try:
            name = self._get_field(raw, "name", self.TOOL_FIELD_MAPPINGS) or ""
            tool_id = self._get_field(raw, "tool_id", self.TOOL_FIELD_MAPPINGS) or ""
            if not tool_id:
                # Derive an id from the name
                tool_id = re.sub(r"\W+", "_", name.lower()).strip("_") if name else ""
            if not tool_id:
                raise InvalidToolError("Tool has no id or name", raw_data=raw)

            description = str(
                self._get_field(raw, "description", self.TOOL_FIELD_MAPPINGS) or ""
            )
            domain = str(self._get_field(raw, "domain", self.TOOL_FIELD_MAPPINGS) or "")
            # Derive domain from source file path if not in the JSON
            if not domain and source_file:
                domain = self._domain_from_path(source_file)
            base_url = str(self._get_field(raw, "base_url", self.TOOL_FIELD_MAPPINGS) or "")

            raw_endpoints = self._get_field(raw, "endpoints", self.TOOL_FIELD_MAPPINGS)
            endpoints: list[Endpoint] = []
            if isinstance(raw_endpoints, list):
                for raw_ep in raw_endpoints:
                    if not isinstance(raw_ep, dict):
                        continue
                    ep = self.normalize_endpoint(raw_ep, tool_id)
                    if ep is not None:
                        endpoints.append(ep)

            return Tool(
                tool_id=tool_id,
                name=name or tool_id,
                description=description,
                domain=domain,
                base_url=base_url,
                endpoints=endpoints,
                raw_schema=raw,
                source_file=source_file,
            )
        except InvalidToolError:
            if self.config.strict_mode:
                raise
            logger.debug("Skipping invalid tool: %s", raw.get("name", "<unknown>"))
            return None
        except Exception as exc:
            if self.config.strict_mode:
                raise InvalidToolError(str(exc), raw_data=raw) from exc
            logger.debug("Error normalising tool: %s", exc)
            return None

    def normalize_endpoint(
        self, raw: dict[str, Any], tool_id: str
    ) -> Endpoint | None:
        """Normalise a raw endpoint dict into an :class:`Endpoint`."""
        try:
            name = str(self._get_field(raw, "name", self.ENDPOINT_FIELD_MAPPINGS) or "")
            description = str(
                self._get_field(raw, "description", self.ENDPOINT_FIELD_MAPPINGS) or ""
            )
            method_raw = str(
                self._get_field(raw, "method", self.ENDPOINT_FIELD_MAPPINGS) or "GET"
            )
            method = _METHOD_MAP.get(method_raw.lower(), HttpMethod.GET)
            raw_path = str(self._get_field(raw, "path", self.ENDPOINT_FIELD_MAPPINGS) or "/")
            path = self._extract_path_from_url(raw_path)

            endpoint_id = generate_endpoint_id(tool_id, method.value, path)

            # Merge required_parameters + optional_parameters (ToolBench format)
            # or fall back to a single "parameters" list
            parameters: list[Parameter] = []
            required_names: list[str] = []

            req_params = raw.get("required_parameters")
            opt_params = raw.get("optional_parameters")
            if isinstance(req_params, list) or isinstance(opt_params, list):
                for raw_p in req_params or []:
                    if isinstance(raw_p, dict):
                        raw_p.setdefault("required", True)
                        p = self.normalize_parameter(raw_p)
                        p.required = True
                        parameters.append(p)
                        required_names.append(p.name)
                    elif isinstance(raw_p, str):
                        parameters.append(Parameter(name=raw_p, required=True))
                        required_names.append(raw_p)
                for raw_p in opt_params or []:
                    if isinstance(raw_p, dict):
                        raw_p.setdefault("required", False)
                        p = self.normalize_parameter(raw_p)
                        p.required = False
                        parameters.append(p)
                    elif isinstance(raw_p, str):
                        parameters.append(Parameter(name=raw_p, required=False))
            else:
                raw_params = self._get_field(raw, "parameters", self.ENDPOINT_FIELD_MAPPINGS)
                if isinstance(raw_params, list):
                    for raw_p in raw_params:
                        if isinstance(raw_p, dict):
                            p = self.normalize_parameter(raw_p)
                            parameters.append(p)
                            if p.required:
                                required_names.append(p.name)
                        elif isinstance(raw_p, str):
                            parameters.append(Parameter(name=raw_p, required=True))
                            required_names.append(raw_p)

            # Response schema (lightweight probe)
            response_schema = self._extract_response_schema(raw)

            return Endpoint(
                endpoint_id=endpoint_id,
                tool_id=tool_id,
                name=name or path,
                description=description,
                method=method,
                path=path,
                parameters=parameters,
                required_parameters=required_names,
                response_schema=response_schema,
                raw_definition=raw,
            )
        except Exception as exc:
            if self.config.strict_mode:
                raise InvalidToolError(f"Bad endpoint: {exc}", raw_data=raw) from exc
            logger.debug("Skipping bad endpoint: %s", exc)
            return None

    def normalize_parameter(self, raw: dict[str, Any]) -> Parameter:
        """Normalise a raw parameter dict into a :class:`Parameter`."""
        name = str(self._get_field(raw, "name", self.PARAM_FIELD_MAPPINGS) or "unknown")
        description = str(
            self._get_field(raw, "description", self.PARAM_FIELD_MAPPINGS) or ""
        )

        # Type
        raw_type = self._get_field(raw, "type", self.PARAM_FIELD_MAPPINGS)
        has_type = raw_type is not None and str(raw_type).strip() != ""
        inferred_type = False
        if has_type:
            param_type = _TYPE_MAP.get(str(raw_type).lower().strip(), ParameterType.UNKNOWN)
        elif self.type_inferrer is not None:
            default_val = self._get_field(raw, "default", self.PARAM_FIELD_MAPPINGS)
            param_type = self.type_inferrer.infer(name, default_val)
            inferred_type = True
            has_type = True  # mark as present (inferred)
        else:
            param_type = ParameterType.UNKNOWN

        # Required
        req_raw = self._get_field(raw, "required", self.PARAM_FIELD_MAPPINGS)
        if req_raw is not None:
            required = not _to_bool(req_raw) if "optional" in raw else _to_bool(req_raw)
        else:
            required = False

        # Default value
        default = self._get_field(raw, "default", self.PARAM_FIELD_MAPPINGS)

        # Location
        loc_raw = self._get_field(raw, "location", self.PARAM_FIELD_MAPPINGS)
        location = _LOCATION_MAP.get(str(loc_raw).lower().strip(), ParameterLocation.QUERY) if loc_raw else ParameterLocation.QUERY

        # Enum
        enum_values = raw.get("enum")
        enum_values = [str(v) for v in enum_values] if isinstance(enum_values, list) else None

        return Parameter(
            name=name,
            description=description,
            param_type=param_type,
            location=location,
            required=required,
            default=default,
            enum_values=enum_values,
            has_description=bool(description.strip()),
            has_type=has_type,
            inferred_type=inferred_type,
            raw_definition=raw,
        )

    # -- field lookup -------------------------------------------------------

    def _get_field(
        self,
        raw: dict[str, Any],
        field_name: str,
        mappings: dict[str, list[str]],
        default: Any = None,
    ) -> Any:
        """Look up *field_name* in *raw* trying every alias in *mappings*."""
        for alias in mappings.get(field_name, [field_name]):
            val = self._get_nested(raw, alias) if "." in alias else raw.get(alias)
            if val is not None:
                return val
        return default

    @staticmethod
    def _get_nested(raw: dict[str, Any], path: str) -> Any:
        """Traverse *raw* by dot-separated *path*."""
        current: Any = raw
        for key in path.split("."):
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current

    # -- response schema ----------------------------------------------------

    @staticmethod
    def _extract_response_schema(raw: dict[str, Any]) -> ResponseSchema | None:
        """Try to extract a lightweight response schema from *raw*."""
        # ToolBench format: "schema" field with response structure
        schema_val = raw.get("schema")
        if isinstance(schema_val, dict) and schema_val:
            status = raw.get("statuscode", 200)
            return ResponseSchema(
                status_code=int(status) if status else 200,
                properties=schema_val,
            )

        for key in ("response", "response_schema", "responses", "output"):
            val = raw.get(key)
            if val is None:
                continue
            if isinstance(val, dict):
                status = val.get("status_code", val.get("code", 200))
                desc = val.get("description", "")
                return ResponseSchema(
                    status_code=int(status) if status else 200,
                    description=str(desc),
                )
            # Truthy but non-dict: at least mark that schema exists
            return ResponseSchema()
        return None

    # -- path helpers -------------------------------------------------------

    @staticmethod
    def _extract_path_from_url(raw_path: str) -> str:
        """Extract the path component from a value that might be a full URL."""
        if raw_path.startswith(("http://", "https://")):
            # Strip scheme + host → keep path
            from urllib.parse import urlparse

            parsed = urlparse(raw_path)
            path = parsed.path or "/"
            return path
        return raw_path if raw_path else "/"

    @staticmethod
    def _domain_from_path(source_file: str) -> str:
        """Derive a domain/category from the source file's parent directory.

        Only activates when the path looks like a ToolBench layout, i.e.
        contains ``toolenv/tools/<Category>/file.json``.  Returns ``""``
        for paths that don't match this pattern.
        """
        parts = Path(source_file).parts
        for i, part in enumerate(parts):
            if part == "tools" and i + 2 < len(parts):
                # parts[i] == "tools", parts[i+1] == category, parts[i+2] == file
                return parts[i + 1].replace("_", " ")
        return ""


# ---------------------------------------------------------------------------
# ToolBenchLoader
# ---------------------------------------------------------------------------


class ToolBenchLoader:
    """Main loader for ToolBench data.

    Usage::

        loader = ToolBenchLoader()
        tools = loader.load_directory(Path("data/toolbench"))
        print(loader.get_stats().to_dict())
    """

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()
        self.parser = RawToolParser(self.config)
        self.normalizer = ToolNormalizer(self.config)
        self.completeness_calc = CompletenessCalculator()
        self.stats = LoaderStats()

    # -- public API ---------------------------------------------------------

    def load_file(self, file_path: Path) -> list[Tool]:
        """Load tools from a single JSON file."""
        self.stats = LoaderStats()
        file_path = Path(file_path)
        try:
            raw_tools = self.parser.parse_file(file_path)
            self.stats.files_processed = 1
        except (json.JSONDecodeError, OSError) as exc:
            self.stats.files_processed = 1
            self.stats.files_failed = 1
            self.stats.errors.append((str(file_path), str(exc)))
            if self.config.strict_mode:
                raise ToolBenchLoaderError(f"Failed to read {file_path}: {exc}") from exc
            logger.warning("Failed to read %s: %s", file_path, exc)
            return []

        return self._process_raw_tools(raw_tools, str(file_path))

    def load_directory(
        self,
        dir_path: Path,
        recursive: bool = True,
        progress: bool = True,
    ) -> list[Tool]:
        """Load all tools from a directory of JSON files."""
        self.stats = LoaderStats()
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ToolBenchLoaderError(f"Not a directory: {dir_path}")

        pattern = "**/*.json" if recursive else "*.json"
        json_files = sorted(dir_path.glob(pattern))
        all_tools: list[Tool] = []

        for fpath in json_files:
            if not fpath.is_file():
                continue
            self.stats.files_processed += 1
            try:
                raw_tools = self.parser.parse_file(fpath)
            except (json.JSONDecodeError, OSError) as exc:
                self.stats.files_failed += 1
                self.stats.errors.append((str(fpath), str(exc)))
                logger.warning("Failed to read %s: %s", fpath, exc)
                continue

            tools = self._process_raw_tools(raw_tools, str(fpath))
            all_tools.extend(tools)

            if self.config.max_tools and len(all_tools) >= self.config.max_tools:
                all_tools = all_tools[: self.config.max_tools]
                break

        return all_tools

    def load_from_glob(self, pattern: str) -> list[Tool]:
        """Load tools from files matching a glob *pattern*."""
        self.stats = LoaderStats()
        all_tools: list[Tool] = []

        for fpath in sorted(Path(".").glob(pattern)):
            if not fpath.is_file():
                continue
            self.stats.files_processed += 1
            try:
                raw_tools = self.parser.parse_file(fpath)
            except (json.JSONDecodeError, OSError) as exc:
                self.stats.files_failed += 1
                self.stats.errors.append((str(fpath), str(exc)))
                continue
            tools = self._process_raw_tools(raw_tools, str(fpath))
            all_tools.extend(tools)

            if self.config.max_tools and len(all_tools) >= self.config.max_tools:
                all_tools = all_tools[: self.config.max_tools]
                break

        return all_tools

    def get_stats(self) -> LoaderStats:
        """Return statistics from the most recent load operation."""
        return self.stats

    # -- private ------------------------------------------------------------

    def _process_raw_tools(
        self, raw_tools: list[dict[str, Any]], source: str
    ) -> list[Tool]:
        """Normalise, score, and filter a batch of raw tool dicts."""
        results: list[Tool] = []
        for raw in raw_tools:
            tool = self.normalizer.normalize_tool(raw, source_file=source)
            if tool is None:
                self.stats.tools_skipped += 1
                continue

            # Endpoint count filter
            ep_count = len(tool.endpoints)
            if ep_count < self.config.min_endpoints:
                self.stats.tools_skipped += 1
                continue
            if ep_count > self.config.max_endpoints:
                self.stats.tools_skipped += 1
                continue

            # Completeness scoring
            if self.config.calculate_completeness:
                self.completeness_calc.calculate_all(tool)
                if tool.completeness_score < self.config.min_completeness:
                    self.stats.tools_skipped += 1
                    continue

            # Collect stats
            self._collect_stats(tool)
            results.append(tool)

        return results

    def _collect_stats(self, tool: Tool) -> None:
        """Update running statistics with data from *tool*."""
        self.stats.tools_loaded += 1
        tier = _score_to_tier_label(tool.completeness_score)
        self.stats.quality_distribution[tier] = (
            self.stats.quality_distribution.get(tier, 0) + 1
        )

        for ep in tool.endpoints:
            self.stats.endpoints_loaded += 1
            if ep.response_schema is None:
                self.stats.missing_response_schemas += 1
            for param in ep.parameters:
                self.stats.parameters_loaded += 1
                if not param.has_description:
                    self.stats.missing_descriptions += 1
                if not param.has_type:
                    self.stats.missing_types += 1
                elif param.inferred_type:
                    self.stats.inferred_types += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_bool(value: Any) -> bool:
    """Coerce *value* to bool, handling strings like ``"true"``/``"false"``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower().strip() in ("true", "1", "yes")
    return bool(value)


def _score_to_tier_label(score: float) -> str:
    """Return tier label for a score (mirrors QualityTier thresholds)."""
    if score >= 0.8:
        return "excellent"
    if score >= 0.6:
        return "good"
    if score >= 0.4:
        return "fair"
    if score >= 0.2:
        return "poor"
    return "minimal"
