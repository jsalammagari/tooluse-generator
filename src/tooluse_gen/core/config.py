"""Configuration management for tooluse-generator.

Loads from a YAML file, validates with Pydantic, and supports CLI overrides
via double-underscore notation (e.g. models__assistant="gpt-4o-mini").
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Default config file location
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = Path("config/default.yaml")


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """LLM model identifiers used by each agent role."""

    assistant: str = Field(
        default="gpt-4o",
        description="Model for the assistant agent that drives tool-use decisions.",
    )
    judge: str = Field(
        default="gpt-4o",
        description="Model for the LLM-as-judge quality scorer.",
    )
    user_simulator: str = Field(
        default="gpt-4o-mini",
        description="Model for simulating user turns in the conversation.",
    )
    mock_generator: str = Field(
        default="gpt-4o-mini",
        description="Model for generating schema-consistent mock tool responses.",
    )
    embedding: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model for semantic edge construction.",
    )


class QualityConfig(BaseModel):
    """Quality thresholds and LLM-as-judge settings."""

    min_score: float = Field(
        default=3.5,
        description="Minimum average judge score (1–5) to accept a conversation.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum repair attempts before discarding a conversation.",
    )
    dimensions: list[str] = Field(
        default=["tool_correctness", "argument_grounding", "task_completion", "naturalness"],
        description="Scoring dimensions evaluated by the LLM judge.",
    )

    @field_validator("min_score")
    @classmethod
    def check_min_score(cls, v: float) -> float:
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"min_score must be between 1.0 and 5.0, got {v}")
        return v

    @field_validator("max_retries")
    @classmethod
    def check_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_retries must be >= 0, got {v}")
        return v


class SamplingConfig(BaseModel):
    """Tool-chain sampling parameters."""

    min_steps: int = Field(
        default=2,
        description="Minimum number of tool calls per conversation.",
    )
    max_steps: int = Field(
        default=5,
        description="Maximum number of tool calls per conversation.",
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Cosine similarity threshold for adding semantic edges (0–1).",
    )
    domains: list[str] | None = Field(
        default=None,
        description="Restrict sampling to these ToolBench domains. None = all domains.",
    )
    excluded_tools: list[str] | None = Field(
        default=None,
        description="Tool endpoint IDs to exclude from sampling.",
    )

    @field_validator("similarity_threshold")
    @classmethod
    def check_similarity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {v}")
        return v

    @model_validator(mode="after")
    def check_steps_order(self) -> SamplingConfig:
        if self.min_steps >= self.max_steps:
            raise ValueError(f"min_steps ({self.min_steps}) must be < max_steps ({self.max_steps})")
        return self


class DiversityConfig(BaseModel):
    """Cross-conversation diversity steering settings."""

    enabled: bool = Field(
        default=True,
        description="Enable cross-conversation diversity steering.",
    )
    weight_decay: float = Field(
        default=0.9,
        description=(
            "Exponential decay for inverse-frequency tool weights. "
            "Higher = faster decay of recency penalty."
        ),
    )
    min_domain_coverage: float = Field(
        default=0.5,
        description=(
            "Target fraction of domains that must appear in the corpus "
            "before any domain is repeated."
        ),
    )


class PathsConfig(BaseModel):
    """File-system paths used throughout the pipeline."""

    model_config = ConfigDict(validate_default=True)

    build_dir: Path = Field(
        default=Path("./output/build"),
        description="Directory for built artifacts (graph, registry, indexes).",
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Root output directory for generated conversations.",
    )
    cache_dir: Path = Field(
        default=Path("./.cache"),
        description="Directory for intermediate caches (embeddings, mock responses).",
    )

    @field_validator("build_dir", "output_dir", "cache_dir", mode="before")
    @classmethod
    def expand_and_create(cls, v: Any) -> Path:
        """Expand environment variables, resolve to absolute path, and mkdir."""
        path = Path(os.path.expandvars(str(v))).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


class AppConfig(BaseModel):
    """Root application configuration."""

    models: ModelConfig = Field(default_factory=ModelConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    seed: int = Field(default=42, description="Global random seed for reproducibility.")
    verbose: int = Field(default=0, description="Verbosity level (0 = quiet, 2 = debug).")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Resolution order:
    1. ``config_path`` if explicitly provided.
    2. ``config/default.yaml`` if it exists.
    3. Pure Pydantic defaults (no file required).

    Environment variables are expanded in all path fields.
    """
    resolved = config_path or (_DEFAULT_CONFIG if _DEFAULT_CONFIG.exists() else None)

    if resolved is None:
        return AppConfig()

    resolved = Path(os.path.expandvars(str(resolved))).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    raw: dict[str, Any] = yaml.safe_load(resolved.read_text()) or {}
    return AppConfig.model_validate(raw)


def merge_cli_overrides(config: AppConfig, **overrides: Any) -> AppConfig:
    """Return a new AppConfig with CLI overrides applied.

    Nested fields are addressed with double-underscore separators::

        merge_cli_overrides(cfg, models__assistant="gpt-4o-mini", seed=7)

    Unknown keys are silently ignored so callers can pass the full CLI
    namespace without filtering.
    """
    # Convert config to a nested dict, then apply overrides
    data = config.model_dump()

    for key, value in overrides.items():
        if value is None:
            continue  # Unset CLI flags leave the loaded value intact
        parts = key.split("__")
        node: dict[str, Any] = data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                break  # Unknown nested key — skip
            node = node[part]
        else:
            leaf = parts[-1]
            if leaf in node:
                node[leaf] = value

    return AppConfig.model_validate(data)


def export_config(config: AppConfig) -> dict[str, Any]:
    """Serialise config to a plain dict for embedding in output metadata.

    Path objects are converted to strings so the result is JSON-serialisable.
    """

    def _convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    result = _convert(config.model_dump())
    assert isinstance(result, dict)
    return result
