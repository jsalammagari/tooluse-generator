"""Unit tests for secrets and client manager (Task 5).

All OpenAI network calls are mocked — no real API key is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from tooluse_gen.core import AppConfig, ClientManager, Secrets, load_secrets
from tooluse_gen.core.secrets import (
    get_instructor_client,
    get_openai_client,
    validate_api_keys,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_KEY = "sk-test-fake-key-1234567890"


def make_secrets(**kwargs: str) -> Secrets:
    """Build a Secrets instance without touching env or .env file."""
    return Secrets.model_construct(
        openai_api_key=SecretStr(kwargs.get("openai_api_key", FAKE_KEY)),
        anthropic_api_key=(
            SecretStr(kwargs["anthropic_api_key"]) if "anthropic_api_key" in kwargs else None
        ),
        huggingface_token=(
            SecretStr(kwargs["huggingface_token"]) if "huggingface_token" in kwargs else None
        ),
    )


# ---------------------------------------------------------------------------
# Secrets model
# ---------------------------------------------------------------------------


def test_secrets_fields_exist():
    s = make_secrets()
    assert hasattr(s, "openai_api_key")
    assert hasattr(s, "anthropic_api_key")
    assert hasattr(s, "huggingface_token")


def test_secrets_key_is_secret_str():
    s = make_secrets()
    assert isinstance(s.openai_api_key, SecretStr)


def test_secrets_repr_hides_value():
    s = make_secrets()
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s)


def test_secrets_get_secret_value():
    s = make_secrets()
    assert s.openai_api_key.get_secret_value() == FAKE_KEY


def test_secrets_optional_fields_default_none():
    s = make_secrets()
    assert s.anthropic_api_key is None
    assert s.huggingface_token is None


def test_secrets_optional_fields_set():
    s = make_secrets(anthropic_api_key="sk-ant-test", huggingface_token="hf_test")
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert s.huggingface_token is not None
    assert s.huggingface_token.get_secret_value() == "hf_test"


# ---------------------------------------------------------------------------
# load_secrets
# ---------------------------------------------------------------------------


def test_load_secrets_from_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENAI_API_KEY={FAKE_KEY}\n")
    secrets = load_secrets(env_file=env_file)
    assert secrets.openai_api_key.get_secret_value() == FAKE_KEY


def test_load_secrets_missing_env_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_secrets(env_file=tmp_path / "nonexistent.env")


def test_load_secrets_missing_required_key_raises(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-test\n")  # missing OPENAI_API_KEY
    with pytest.raises(OSError, match="OPENAI_API_KEY"):
        load_secrets(env_file=env_file)


def test_load_secrets_with_all_optional_fields(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"OPENAI_API_KEY={FAKE_KEY}\n"
        "ANTHROPIC_API_KEY=sk-ant-test\n"
        "HUGGINGFACE_TOKEN=hf_test\n"
    )
    secrets = load_secrets(env_file=env_file)
    assert secrets.openai_api_key.get_secret_value() == FAKE_KEY
    assert secrets.anthropic_api_key is not None
    assert secrets.huggingface_token is not None


# ---------------------------------------------------------------------------
# get_openai_client
# ---------------------------------------------------------------------------


def test_get_openai_client_returns_client():
    s = make_secrets()
    client = get_openai_client(s)
    from openai import OpenAI

    assert isinstance(client, OpenAI)


def test_get_openai_client_uses_key():
    s = make_secrets()
    client = get_openai_client(s)
    assert client.api_key == FAKE_KEY


# ---------------------------------------------------------------------------
# get_instructor_client
# ---------------------------------------------------------------------------


def test_get_instructor_client_returns_instructor():
    import instructor

    s = make_secrets()
    client = get_instructor_client(s)
    assert isinstance(client, instructor.Instructor)


# ---------------------------------------------------------------------------
# validate_api_keys
# ---------------------------------------------------------------------------


def test_validate_api_keys_success():
    s = make_secrets()
    with patch("tooluse_gen.core.secrets.get_openai_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_factory.return_value = mock_client
        result = validate_api_keys(s)
    assert result is True


def test_validate_api_keys_auth_error_raises():
    from openai import AuthenticationError

    s = make_secrets()
    with patch("tooluse_gen.core.secrets.get_openai_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = AuthenticationError(
            message="invalid key", response=MagicMock(status_code=401), body={}
        )
        mock_factory.return_value = mock_client
        with pytest.raises(OSError, match="rejected"):
            validate_api_keys(s)


def test_validate_api_keys_network_error_raises():
    s = make_secrets()
    with patch("tooluse_gen.core.secrets.get_openai_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ConnectionError("network down")
        mock_factory.return_value = mock_client
        with pytest.raises(OSError, match="Could not validate"):
            validate_api_keys(s)


# ---------------------------------------------------------------------------
# ClientManager
# ---------------------------------------------------------------------------


def make_manager() -> ClientManager:
    return ClientManager(secrets=make_secrets(), config=AppConfig())


def test_client_manager_repr():
    mgr = make_manager()
    r = repr(mgr)
    assert "ClientManager" in r
    assert "openai_ready=False" in r
    assert "instructor_ready=False" in r


def test_client_manager_openai_lazy():
    mgr = make_manager()
    assert mgr._openai is None
    _ = mgr.openai
    assert mgr._openai is not None


def test_client_manager_instructor_lazy():
    mgr = make_manager()
    assert mgr._instructor is None
    _ = mgr.instructor
    assert mgr._instructor is not None


def test_client_manager_openai_cached():
    mgr = make_manager()
    c1 = mgr.openai
    c2 = mgr.openai
    assert c1 is c2


def test_client_manager_instructor_cached():
    mgr = make_manager()
    c1 = mgr.instructor
    c2 = mgr.instructor
    assert c1 is c2


def test_client_manager_get_completion():
    mgr = make_manager()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello, world!"

    with patch.object(mgr, "_openai") as mock_openai:
        mgr._openai = mock_openai
        mock_openai.chat.completions.create.return_value = mock_response
        result = mgr.get_completion([{"role": "user", "content": "Hi"}])

    assert result == "Hello, world!"
    mock_openai.chat.completions.create.assert_called_once()


def test_client_manager_get_completion_uses_default_model():
    mgr = make_manager()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"

    with patch.object(mgr, "_openai") as mock_openai:
        mgr._openai = mock_openai
        mock_openai.chat.completions.create.return_value = mock_response
        mgr.get_completion([{"role": "user", "content": "Hi"}])
        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == mgr._config.models.assistant


def test_client_manager_get_completion_custom_model():
    mgr = make_manager()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"

    with patch.object(mgr, "_openai") as mock_openai:
        mgr._openai = mock_openai
        mock_openai.chat.completions.create.return_value = mock_response
        mgr.get_completion([{"role": "user", "content": "Hi"}], model="gpt-4o-mini")
        call_kwargs = mock_openai.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"


def test_client_manager_get_completion_empty_response_raises():
    mgr = make_manager()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    with patch.object(mgr, "_openai") as mock_openai:
        mgr._openai = mock_openai
        mock_openai.chat.completions.create.return_value = mock_response
        with pytest.raises(ValueError, match="empty"):
            mgr.get_completion([{"role": "user", "content": "Hi"}])


def test_client_manager_get_structured():
    from pydantic import BaseModel

    class Reply(BaseModel):
        answer: str

    mgr = make_manager()
    expected = Reply(answer="42")

    with patch.object(mgr, "_instructor") as mock_inst:
        mgr._instructor = mock_inst
        mock_inst.chat.completions.create.return_value = expected
        result = mgr.get_structured(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            response_model=Reply,
        )
    assert result.answer == "42"
    mock_inst.chat.completions.create.assert_called_once()
