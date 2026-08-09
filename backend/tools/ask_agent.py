"""Talk to the deployed agent from a terminal.

    python -m tools.ask_agent carousel "3 რჩევა ძლიერი კაუჭისთვის"
    python -m tools.ask_agent caption "ახალი ჭიქის გაშვება" --niche "კერამიკა, თბილისი"
    python -m tools.ask_agent public_profile_analysis "ნაიკივით მინდა" --handle nike
    python -m tools.ask_agent --list

Run from the ``backend/`` directory.

Why this exists: the agent has no interface. It is an HTTP API behind a site
key, and every call also needs a short-lived user token — so trying it by hand
means three requests before the first question. Langflow's chat is not an
alternative: it no longer sits on the generation path, so a message typed there
exercises none of this.

The two credentials are read from the environment. Do not paste them as
arguments — a command line ends up in shell history and in `ps` output:

    $env:SITE_API_KEY  = "..."      # PowerShell
    $env:AGENT_BASE_URL = "https://<your service>.up.railway.app"

Exit code 0 when the generation succeeds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_BASE_URL = "https://instagram-ai-agent-production-5979.up.railway.app"

# Kept here rather than fetched so `--list` works even when the service is down,
# which is exactly when someone reaches for a diagnostic tool.
TASK_TYPES = [
    ("chat", "free-form question"),
    ("caption", "a caption, ready to post"),
    ("carousel", "carousel copy plus rendered slide images"),
    ("reel_idea", "reel ideas for the account"),
    ("reel_script", "a reel script ready to record"),
    ("reel_feedback", "review of an existing reel"),
    ("profile_audit", "audit of the connected profile"),
    ("content_plan", "30-day plan"),
    ("performance_summary", "what worked and what did not"),
    ("public_profile_analysis", "read another public account and adapt it (needs --handle)"),
]


def _fail(message: str) -> int:
    print(f"FAILED: {message}", file=sys.stderr)
    return 1


def _render(result: dict) -> None:
    reply = result.get("reply") or ""
    print("\n--- reply ---")
    print(reply.strip() or "(empty)")

    structured = result.get("structured_output")
    if isinstance(structured, dict) and structured:
        print("\n--- structured output ---")
        print(json.dumps(structured, ensure_ascii=False, indent=2))

        # Surfaced separately because a slide with no image_url is the visible
        # symptom of an unset image key, and it is easy to miss inside the JSON.
        slides = structured.get("slides") or []
        if slides:
            missing = [s.get("slide_number") for s in slides if not s.get("image_url")]
            print(f"\nslides: {len(slides)}   without an image: {missing or 'none'}")

    print("\n--- diagnostics ---")
    for label, key in (
        ("provider", "model_provider"),
        ("model", "model_name"),
        ("parse status", "parse_status"),
    ):
        if result.get(key):
            print(f"  {label:14}: {result[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one request to the deployed agent.")
    parser.add_argument("task_type", nargs="?", help="see --list")
    parser.add_argument("message", nargs="?", help="what you are asking for")
    parser.add_argument("--user", default="manual-test", help="which customer to bill and read context for")
    parser.add_argument("--niche", default=None)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--handle", default=None, help="public_profile_analysis only: the account to learn from")
    parser.add_argument("--slides", type=int, default=None)
    parser.add_argument("--no-images", action="store_true", help="carousel: skip image generation")
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for the answer")
    parser.add_argument("--list", action="store_true", help="show the task types and exit")
    args = parser.parse_args()

    if args.list:
        print("Task types:")
        for name, description in TASK_TYPES:
            print(f"  {name:26} {description}")
        return 0

    if not args.task_type or not args.message:
        parser.print_usage()
        return _fail("a task type and a message are required (see --list)")

    base_url = (os.getenv("AGENT_BASE_URL", "").strip() or DEFAULT_BASE_URL).rstrip("/")
    site_key = os.getenv("SITE_API_KEY", "").strip()
    if not site_key:
        return _fail("SITE_API_KEY is not set. Copy it from Railway > Variables.")

    headers = {"X-Site-Key": site_key, "Content-Type": "application/json"}

    with httpx.Client(timeout=60) as client:
        # A user token, because every route that spends credits reads the
        # customer from the token rather than the body.
        try:
            token_response = client.post(
                f"{base_url}/api/v1/auth/user-token", headers=headers, json={"user_id": args.user}
            )
            token_response.raise_for_status()
            headers["X-User-Token"] = token_response.json()["user_token"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                return _fail("SITE_API_KEY was rejected. Check it against Railway > Variables.")
            return _fail(f"could not get a user token: HTTP {exc.response.status_code}")
        except httpx.HTTPError as exc:
            return _fail(f"could not reach {base_url}: {type(exc).__name__}")

        body = {"user_id": args.user, "task_type": args.task_type, "message": args.message}
        for key, value in (
            ("niche", args.niche),
            ("goal", args.goal),
            ("reference_handle", args.handle),
            ("slide_count", args.slides),
        ):
            if value is not None:
                body[key] = value
        if args.no_images:
            body["generate_images"] = False

        print(f"asking {args.task_type} as user '{args.user}' ...")
        try:
            created = client.post(f"{base_url}/api/v1/generation-jobs", headers=headers, json=body)
            created.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("error", {}).get("details") or ""
            except Exception:  # noqa: BLE001 - the message is best-effort
                detail = exc.response.text[:200]
            return _fail(f"HTTP {exc.response.status_code} {detail}")

        job_id = created.json()["job_id"]

        # Polling, because generation is asynchronous by design: a carousel runs
        # a text call plus an image per slide, far longer than a request can be
        # held open.
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            time.sleep(4)
            job = client.get(f"{base_url}/api/v1/generation-jobs/{job_id}", headers=headers).json()
            status = job.get("status")
            if status == "succeeded":
                _render(job.get("result") or {})
                return 0
            if status == "failed":
                return _fail(job.get("error") or "the job failed with no message")
            print(f"  {status} ...")

    return _fail(f"still running after {args.timeout}s — raise --timeout")


if __name__ == "__main__":
    raise SystemExit(main())
