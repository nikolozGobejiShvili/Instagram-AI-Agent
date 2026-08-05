"""Setup-token support, ported from sityvis-akademia-ai-agent.

`claude setup-token` mints an OAuth credential (`sk-ant-oat01-…`) billed
against a Claude Pro/Max subscription rather than API credits. One variable
accepts either shape; the prefix decides the transport.

Each case here covers a failure that does not point at its own cause.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.providers import anthropic_auth  # noqa: E402
from app.services.providers.anthropic_text import AnthropicTextProvider  # noqa: E402
from app.services.providers.base import ProviderNotConfigured  # noqa: E402

SETUP_TOKEN = "sk-ant-oat01-EXAMPLE-not-a-real-token"
API_KEY = "sk-ant-api03-EXAMPLE-not-a-real-key"

SECTIONS = [
    {"role": "system", "content": [{"type": "input_text", "text": "You are an agent."}]},
    {"role": "user", "content": [{"type": "input_text", "text": "გამარჯობა"}]},
]


class StubAnthropic:
    def __init__(self):
        self.captured = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ------------------------------------------------------------- sanitizing


@pytest.mark.parametrize("dirty,clean", [
    ("﻿" + SETUP_TOKEN, SETUP_TOKEN),          # UTF-8 BOM from a shell pipe
    (SETUP_TOKEN + "\n", SETUP_TOKEN),               # trailing newline
    ("  " + SETUP_TOKEN + "  ", SETUP_TOKEN),        # padding
    ("sk-ant-oat01-​abc", "sk-ant-oat01-abc"),  # zero-width space
    ("sk-ant -oat", "sk-ant-oat"),              # non-breaking space
])
def test_invisible_characters_are_stripped(dirty, clean):
    """A clipboard round-trip corrupts the auth header; the API then says 401
    without mentioning whitespace."""
    assert anthropic_auth.sanitize_token(dirty) == clean


def test_sanitizing_handles_empty_input():
    assert anthropic_auth.sanitize_token(None) == ""
    assert anthropic_auth.sanitize_token("") == ""


# ------------------------------------------------------ credential shapes


def test_setup_token_is_recognised_by_prefix():
    assert anthropic_auth.is_setup_token(SETUP_TOKEN) is True
    assert anthropic_auth.is_setup_token(API_KEY) is False
    assert anthropic_auth.is_setup_token("") is False


def test_setup_token_authenticates_as_bearer_with_oauth_beta():
    kwargs = anthropic_auth.client_kwargs(SETUP_TOKEN)
    assert kwargs["auth_token"] == SETUP_TOKEN
    assert "oauth-2025-04-20" in kwargs["default_headers"]["anthropic-beta"]
    assert "api_key" not in kwargs


def test_api_key_uses_the_classic_header():
    kwargs = anthropic_auth.client_kwargs(API_KEY)
    assert kwargs["api_key"] == API_KEY
    assert "auth_token" not in kwargs
    assert "default_headers" not in kwargs


def test_either_variable_name_is_accepted(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SETUP_TOKEN)
    assert anthropic_auth.auth_token() == SETUP_TOKEN

    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    assert anthropic_auth.auth_token() == API_KEY, "ANTHROPIC_API_KEY must remain a working alias"


def test_auth_kind_never_reveals_the_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SETUP_TOKEN)
    kind = anthropic_auth.auth_kind()
    assert kind == "setup-token"
    assert SETUP_TOKEN not in kind

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", API_KEY)
    assert anthropic_auth.auth_kind() == "api-key"

    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    assert anthropic_auth.auth_kind() == "none"


# --------------------------------------------------- billing attribution


def test_billing_attribution_is_prepended_for_setup_tokens():
    """Without it a setup token reaches Haiku but 400s on Sonnet/Opus."""
    result = anthropic_auth.apply_billing_attribution("You are an agent.", SETUP_TOKEN)
    assert result.startswith(anthropic_auth.CC_BILLING_HEADER)
    assert result.endswith("You are an agent.")


def test_billing_attribution_survives_an_empty_system_prompt():
    assert anthropic_auth.apply_billing_attribution("", SETUP_TOKEN) == anthropic_auth.CC_BILLING_HEADER


def test_api_keys_are_left_untouched():
    assert anthropic_auth.apply_billing_attribution("You are an agent.", API_KEY) == "You are an agent."


def test_generation_carries_billing_attribution_on_the_real_call_path(monkeypatch):
    """The attribution has to be applied where requests are actually built —
    a helper that exists but is never called protects nothing."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SETUP_TOKEN)
    client = StubAnthropic()

    AnthropicTextProvider(client=client).generate(
        task_type="chat", response_input=SECTIONS, max_output_tokens=64
    )

    assert client.captured["system"].startswith(anthropic_auth.CC_BILLING_HEADER)
    assert "You are an agent." in client.captured["system"]


def test_generation_with_an_api_key_sends_no_attribution(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    client = StubAnthropic()

    AnthropicTextProvider(client=client).generate(
        task_type="chat", response_input=SECTIONS, max_output_tokens=64
    )

    assert not client.captured["system"].startswith(anthropic_auth.CC_BILLING_HEADER)


def test_missing_credential_names_the_documented_variable():
    with pytest.raises(ProviderNotConfigured) as exc:
        AnthropicTextProvider().generate(
            task_type="chat", response_input=SECTIONS, max_output_tokens=64
        )
    assert exc.value.missing == "ANTHROPIC_AUTH_TOKEN"
