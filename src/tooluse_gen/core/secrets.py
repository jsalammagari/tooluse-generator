"""Secure environment variable handling for API keys and secrets.

Keys are stored as ``pydantic.SecretStr`` so they are never exposed in
repr/logging output.  Access the raw value only when constructing an API
client, via ``secret.get_secret_value()``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import instructor
from openai import AuthenticationError, OpenAI
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secrets model
# ---------------------------------------------------------------------------


class Secrets(BaseSettings):
    """API keys and tokens loaded from environment variables or a .env file.

    Required:
        OPENAI_API_KEY

    Optional:
        ANTHROPIC_API_KEY
        HUGGINGFACE_TOKEN
    """

    openai_api_key: SecretStr
    anthropic_api_key: SecretStr | None = None
    huggingface_token: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Prevent values from leaking into stack traces
        hide_input_in_errors=True,
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_secrets(env_file: Path | None = None) -> Secrets:
    """Load secrets from environment variables, optionally overriding with a .env file.

    Args:
        env_file: Path to a ``.env`` file.  ``None`` uses the default ``.env``
                  in the working directory (if present).

    Returns:
        Validated :class:`Secrets` instance.

    Raises:
        SystemExit: With a clear, key-safe error message if ``OPENAI_API_KEY``
                    is missing.
    """
    if env_file is not None and not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")

    try:
        # pydantic-settings accepts _env_file as a constructor kwarg to
        # override the model_config value at runtime.
        if env_file is not None:
            return Secrets(_env_file=str(env_file))
        return Secrets()
    except ValidationError as exc:
        # Extract field names from the error without exposing any values
        missing = [
            e["loc"][0]
            for e in exc.errors()
            if e.get("type") in ("missing", "value_error")
        ]
        msg = (
            "Missing required environment variable(s): "
            + ", ".join(str(f).upper() for f in missing)
            + "\nSet them in your environment or in a .env file. "
            "See .env.example for the full list."
        )
        raise OSError(msg) from None


def get_openai_client(secrets: Secrets) -> OpenAI:
    """Return a configured :class:`openai.OpenAI` client.

    The raw key is accessed exactly once here and never stored elsewhere.
    """
    return OpenAI(api_key=secrets.openai_api_key.get_secret_value())


def get_instructor_client(secrets: Secrets) -> instructor.Instructor:
    """Return an ``instructor``-wrapped OpenAI client for structured outputs."""
    openai_client = get_openai_client(secrets)
    return instructor.from_openai(openai_client)


def validate_api_keys(secrets: Secrets) -> bool:
    """Verify that the OpenAI key is accepted by making a minimal API call.

    Args:
        secrets: Loaded :class:`Secrets` instance.

    Returns:
        ``True`` if the key is valid.

    Raises:
        EnvironmentError: If the key is rejected by the API, with a message
                          that does **not** include the key value.
    """
    client = get_openai_client(secrets)
    try:
        # Cheapest possible call: list available models
        client.models.list()
        logger.debug("OpenAI API key validated successfully.")
        return True
    except AuthenticationError:
        raise OSError(
            "OPENAI_API_KEY was rejected by the OpenAI API. "
            "Check that the key is correct and has not been revoked."
        ) from None
    except Exception as exc:
        raise OSError(
            f"Could not validate OPENAI_API_KEY: {type(exc).__name__}. "
            "Check network connectivity and key permissions."
        ) from None
