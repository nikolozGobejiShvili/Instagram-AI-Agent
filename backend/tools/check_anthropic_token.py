"""Verify the Anthropic credential before routing the agent to Claude.

    python -m tools.check_anthropic_token

Run from the ``backend/`` directory.

Why a dedicated check: a Claude **setup token** can be perfectly valid and
still fail on the model you configured. Without the billing attribution that
``providers.anthropic_auth`` injects, an ``sk-ant-oat`` token reaches Haiku but
answers 400 for Sonnet/Opus. Credential and model have to be proven together,
which is exactly what this does.

Exit code 0 on success, 1 on any failure. The token is never printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.providers import anthropic_auth  # noqa: E402
from app.services.providers.anthropic_text import AnthropicTextProvider  # noqa: E402
from app.services.providers.base import ProviderError  # noqa: E402


def _print_config(provider: AnthropicTextProvider) -> None:
    print("Anthropic configuration")
    print(f"  PRIMARY_LLM_PROVIDER : {os.getenv('PRIMARY_LLM_PROVIDER', 'openai')}")
    print(f"  ANTHROPIC_MODEL      : {provider.model_name}")
    print(f"  credential kind      : {anthropic_auth.auth_kind()}")
    print()


def _check_completion(provider: AnthropicTextProvider) -> bool:
    print("[1] plain completion ...", end=" ", flush=True)

    sections = [
        {"role": "system", "content": [{"type": "input_text", "text": "Respond with OK"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "test"}]},
    ]
    try:
        result = provider.generate(task_type="chat", response_input=sections, max_output_tokens=32)
    except ProviderError as exc:
        print("FAILED")
        print(f"    {exc.safe_message}  (code={exc.code})")
        print()
        print("    Common causes:")
        print("      * token wrapped across lines, or pasted with quotes")
        print("      * setup token expired — regenerate with `claude setup-token`")
        print("      * ANTHROPIC_MODEL not available to this credential")
        print("      * billing attribution missing — a setup token then reaches")
        print("        Haiku but 400s on Sonnet/Opus")
        return False

    print("OK")
    print(f"    {result['model_name']} responded: {result['reply'][:60]}")
    usage = result.get("usage") or {}
    if usage:
        print(f"    tokens in/out: {usage.get('input_tokens')}/{usage.get('output_tokens')}")
    return True


def main() -> int:
    provider = AnthropicTextProvider()
    _print_config(provider)

    if not anthropic_auth.is_configured():
        print("FAILED: ANTHROPIC_AUTH_TOKEN is empty.")
        print("Generate a setup token with:  claude setup-token")
        print("(or paste a classic API key from console.anthropic.com — either works)")
        return 1

    if not _check_completion(provider):
        return 1

    print()
    print("Ready. Set PRIMARY_LLM_PROVIDER=anthropic to route the agent to Claude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
