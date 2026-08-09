"""Turning a Stripe payment into a plan.

Two rules shape everything here.

**The webhook is the source of truth, not the browser.** A customer returning to
a success page proves they reached a URL, nothing more — that URL can be opened
directly. Only Stripe's server-to-server webhook confirms money moved, so
nothing grants a plan except a verified webhook.

**A webhook is a public endpoint.** Its URL is guessable and Stripe cannot send a
site key, so the signature *is* the authentication. Without verification, anyone
who found the path could POST a fabricated "payment succeeded" and hand
themselves the agency plan. Verification is therefore mandatory: an unset secret
refuses every event rather than accepting them unchecked.

Stripe delivers events more than once by design, and out of order. Every event id
is recorded before its effect is applied, so a redelivery is dropped instead of
extending a subscription twice.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Events that change what a customer is entitled to. Anything else is
# acknowledged and ignored -- Stripe sends dozens of types, and reacting to ones
# we do not understand is how a plan gets revoked by an unrelated invoice event.
SUBSCRIPTION_ACTIVE_EVENTS = {
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "invoice.payment_succeeded",
}
SUBSCRIPTION_ENDED_EVENTS = {
    "customer.subscription.deleted",
}

# Stripe's own tolerance, and the reason the timestamp is signed at all: without
# it a captured request could be replayed indefinitely.
SIGNATURE_TOLERANCE_SECONDS = 300


class PaymentError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PaymentService:
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS processed_payment_events (
            event_id     TEXT PRIMARY KEY,
            event_type   TEXT NOT NULL,
            user_id      TEXT,
            plan         TEXT,
            processed_at TEXT NOT NULL
        )
    """

    def __init__(
        self,
        *,
        db_path: Path | str | None = None,
        webhook_secret: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        self.db_path = Path(db_path or os.getenv("PAYMENTS_DB_PATH") or (data_dir / "payments.sqlite3"))
        self._webhook_secret = (
            webhook_secret if webhook_secret is not None else os.getenv("STRIPE_WEBHOOK_SECRET", "")
        ).strip()
        # Separate from the webhook secret: one signs incoming events, the other
        # authorises outgoing calls. A deployment can hold either alone.
        self._api_key = (api_key if api_key is not None else os.getenv("STRIPE_API_KEY", "")).strip()
        self.timeout = float(timeout or os.getenv("STRIPE_TIMEOUT_SECONDS", "20") or 20)
        self._initialise_storage()

    # ------------------------------------------------------------------ storage
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialise_storage(self) -> None:
        with self._connect() as conn:
            conn.execute(self._SCHEMA)

    def claim_event(self, event_id: str, event_type: str) -> bool:
        """Record this event, returning False if it was already handled.

        The insert *is* the lock: two concurrent deliveries of the same event
        both reach here, and the primary key lets exactly one through. Checking
        first and inserting after would let both pass in the gap.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "INSERT OR IGNORE INTO processed_payment_events "
                "(event_id, event_type, processed_at) VALUES (?, ?, datetime('now'))",
                (event_id, event_type),
            )
            claimed = cursor.rowcount == 1
            conn.execute("COMMIT")
        if not claimed:
            logger.info("Ignoring duplicate Stripe event event_id=%s", event_id)
        return claimed

    def record_outcome(self, event_id: str, *, user_id: str | None, plan: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE processed_payment_events SET user_id = ?, plan = ? WHERE event_id = ?",
                (user_id, plan, event_id),
            )

    # ---------------------------------------------------------------- signature
    def verify_signature(self, payload: bytes, signature_header: str, *, now: float | None = None) -> dict:
        """Check Stripe's signature and return the parsed event.

        Fails closed on an unset secret: a webhook whose signature is not
        checked is an open endpoint that grants plans, and the URL is guessable.
        """
        if not self._webhook_secret:
            raise PaymentError(
                "Payments are not accepting webhooks.",
                status_code=503,
            )

        timestamp, signatures = self._parse_signature_header(signature_header)
        current = int(now if now is not None else time.time())
        if abs(current - timestamp) > SIGNATURE_TOLERANCE_SECONDS:
            # Without this a captured request stays valid forever.
            raise PaymentError("Webhook signature is outside the accepted time window.")

        expected = hmac.new(
            self._webhook_secret.encode(),
            f"{timestamp}.".encode() + payload,
            hashlib.sha256,
        ).hexdigest()
        # Stripe may send several signatures during a secret rotation; any one
        # matching is a valid request.
        if not any(hmac.compare_digest(expected, candidate) for candidate in signatures):
            raise PaymentError("Webhook signature verification failed.")

        try:
            event = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise PaymentError("Webhook body was not valid JSON.") from exc
        if not isinstance(event, dict) or not event.get("id"):
            raise PaymentError("Webhook body was not a Stripe event.")
        return event

    @staticmethod
    def _parse_signature_header(header: str) -> tuple[int, list[str]]:
        timestamp = None
        signatures = []
        for part in (header or "").split(","):
            key, _, value = part.strip().partition("=")
            if key == "t":
                try:
                    timestamp = int(value)
                except ValueError:
                    raise PaymentError("Webhook signature header is malformed.") from None
            elif key == "v1":
                signatures.append(value)

        if timestamp is None or not signatures:
            raise PaymentError("Webhook signature header is missing or malformed.")
        return timestamp, signatures

    # -------------------------------------------------------------------- plans
    @staticmethod
    def price_for_plan(plan: str) -> str | None:
        """The Stripe price to charge for a plan — the reverse of the webhook's
        lookup, read from the same variable so the two cannot disagree."""
        for price_id, mapped_plan in PaymentService._price_map().items():
            if mapped_plan == (plan or "").strip():
                return price_id
        return None

    @staticmethod
    def _price_map() -> dict[str, str]:
        mapping = {}
        for entry in os.getenv("STRIPE_PRICE_MAP", "").split(","):
            key, _, value = entry.strip().partition("=")
            if key and value:
                mapping[key.strip()] = value.strip()
        return mapping

    def create_checkout_session(self, *, user_id: str, plan: str, success_url: str, cancel_url: str) -> dict:
        """Start a subscription purchase for one customer.

        The webhook was built first and could only ever react; without this the
        website had to create the session itself, and the one thing it must not
        get wrong is the metadata — a session created without ``user_id`` pays
        real money and grants nothing, which the webhook can only log after the
        fact. Creating it here makes that impossible to forget.

        The id is written into **both** the session metadata and
        ``subscription_data.metadata``. Checkout events carry the former;
        renewal and cancellation events are emitted against the subscription and
        carry only the latter. Setting just one means the first payment works and
        every event afterwards arrives unattributable.

        Form-encoded HTTP rather than the Stripe SDK: this is one call, and the
        SDK is a large dependency to add for it.
        """
        api_key = (self._api_key or "").strip()
        if not api_key:
            raise PaymentError("Payments are not configured.", status_code=503)

        price_id = self.price_for_plan(plan)
        if not price_id:
            # Refused rather than defaulted. Guessing a price is how a customer
            # is charged for a tier they did not choose.
            raise PaymentError(
                f"No Stripe price is mapped to plan '{plan}'. Set STRIPE_PRICE_MAP.",
                status_code=503,
            )

        form = {
            "mode": "subscription",
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "client_reference_id": user_id,
            "metadata[user_id]": user_id,
            "subscription_data[metadata][user_id]": user_id,
        }

        try:
            import httpx

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    "https://api.stripe.com/v1/checkout/sessions",
                    data=form,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                session = response.json()
        except Exception as exc:  # noqa: BLE001 - normalised for the caller
            logger.warning("Stripe checkout creation failed: %s", type(exc).__name__)
            raise PaymentError("Could not start checkout. Please try again shortly.", status_code=502) from exc

        url = session.get("url")
        if not url:
            raise PaymentError("Stripe did not return a checkout url.", status_code=502)

        logger.info("Checkout session created user_id=%s plan=%s", user_id, plan)
        return {"checkout_url": url, "session_id": str(session.get("id") or ""), "plan": plan}

    @staticmethod
    def plan_for_price(price_id: str | None) -> str | None:
        """Map a Stripe price to a plan.

        Configured as ``STRIPE_PRICE_MAP`` (``price_x=creator,price_y=pro``)
        rather than inferred from an amount: prices change with currency,
        discounts and tax, and a customer who paid a promotional rate must still
        land on the tier they bought.
        """
        raw = os.getenv("STRIPE_PRICE_MAP", "")
        mapping = {}
        for entry in raw.split(","):
            key, _, value = entry.strip().partition("=")
            if key and value:
                mapping[key.strip()] = value.strip()
        return mapping.get((price_id or "").strip())

    @staticmethod
    def extract_user_id(event_object: dict) -> str | None:
        """Find our customer id in a Stripe object.

        Read from metadata, which the checkout session sets, because Stripe's
        own customer id is not ours and a lookup by email would merge two people
        who share an inbox.
        """
        metadata = event_object.get("metadata") or {}
        candidate = metadata.get("user_id") or metadata.get("app_user_id")
        if candidate:
            return str(candidate).strip() or None

        subscription_details = event_object.get("subscription_details") or {}
        nested = (subscription_details.get("metadata") or {}).get("user_id")
        return str(nested).strip() if nested else None

    @staticmethod
    def extract_price_id(event_object: dict) -> str | None:
        items = ((event_object.get("items") or {}).get("data")) or []
        for item in items:
            price = (item or {}).get("price") or {}
            if price.get("id"):
                return str(price["id"])

        lines = ((event_object.get("lines") or {}).get("data")) or []
        for line in lines:
            price = (line or {}).get("price") or {}
            if price.get("id"):
                return str(price["id"])

        return None
