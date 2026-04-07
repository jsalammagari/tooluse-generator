"""Core package — configuration management, secrets, and API clients."""

from tooluse_gen.core.clients import ClientManager
from tooluse_gen.core.config import (
    AppConfig,
    DiversityConfig,
    ModelConfig,
    PathsConfig,
    QualityConfig,
    SamplingConfig,
    export_config,
    load_config,
    merge_cli_overrides,
)
from tooluse_gen.core.secrets import (
    Secrets,
    get_instructor_client,
    get_openai_client,
    load_secrets,
    validate_api_keys,
)

__all__ = [
    # Config
    "AppConfig",
    "ModelConfig",
    "QualityConfig",
    "SamplingConfig",
    "DiversityConfig",
    "PathsConfig",
    "load_config",
    "merge_cli_overrides",
    "export_config",
    # Secrets
    "Secrets",
    "load_secrets",
    "get_openai_client",
    "get_instructor_client",
    "validate_api_keys",
    # Clients
    "ClientManager",
]
