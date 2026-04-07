"""Lazy API client manager.

Clients are constructed on first access so that importing this module never
makes a network call or requires environment variables to be set.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel

from tooluse_gen.core.config import AppConfig
from tooluse_gen.core.secrets import Secrets, get_instructor_client, get_openai_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClientManager:
    """Manages OpenAI and instructor API clients with lazy initialisation.

    Clients are created on first access and reused for the lifetime of this
    object.  The raw API key is never stored; it lives only inside the
    :class:`~tooluse_gen.core.secrets.Secrets` model as a ``SecretStr``.

    Example::

        mgr = ClientManager(secrets=load_secrets(), config=load_config())
        text = mgr.get_completion([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, secrets: Secrets, config: AppConfig) -> None:
        self._secrets = secrets
        self._config = config
        self._openai: OpenAI | None = None
        self._instructor: instructor.Instructor | None = None

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def openai(self) -> OpenAI:
        """Return the :class:`openai.OpenAI` client, initialising on first use."""
        if self._openai is None:
            logger.debug("Initialising OpenAI client.")
            self._openai = get_openai_client(self._secrets)
        return self._openai

    @property
    def instructor(self) -> instructor.Instructor:
        """Return the instructor-wrapped client for structured outputs."""
        if self._instructor is None:
            logger.debug("Initialising instructor client.")
            self._instructor = get_instructor_client(self._secrets)
        return self._instructor

    # ------------------------------------------------------------------
    # Completion helpers
    # ------------------------------------------------------------------

    def get_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the text content of a single chat completion.

        Args:
            messages: OpenAI-format message list.
            model: Model identifier.  Defaults to ``config.models.assistant``.
            **kwargs: Extra arguments forwarded to ``chat.completions.create``.

        Returns:
            The string content of the first choice's message.
        """
        resolved_model = model or self._config.models.assistant
        logger.debug("get_completion: model=%s, messages=%d", resolved_model, len(messages))

        response = self.openai.chat.completions.create(
            model=resolved_model,
            messages=messages,
            **kwargs,
        )
        content: str | None = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned an empty response.")
        return content

    def get_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        model: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Return a structured response validated against a Pydantic model.

        Uses the ``instructor`` library to enforce the schema via OpenAI's
        structured-output / function-calling API.

        Args:
            messages: OpenAI-format message list.
            response_model: Pydantic model class the response must conform to.
            model: Model identifier.  Defaults to ``config.models.assistant``.
            **kwargs: Extra arguments forwarded to ``instructor.chat.completions.create``.

        Returns:
            A validated instance of *response_model*.
        """
        resolved_model = model or self._config.models.assistant
        logger.debug(
            "get_structured: model=%s, schema=%s, messages=%d",
            resolved_model,
            response_model.__name__,
            len(messages),
        )

        result: T = self.instructor.chat.completions.create(
            model=resolved_model,
            response_model=response_model,
            messages=messages,
            **kwargs,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        openai_ready = self._openai is not None
        instructor_ready = self._instructor is not None
        return (
            f"ClientManager(model={self._config.models.assistant!r}, "
            f"openai_ready={openai_ready}, instructor_ready={instructor_ready})"
        )
