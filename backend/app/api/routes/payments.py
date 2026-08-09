"""Payment webhook and user tokens — the two things selling requires.

The webhook is exempt from the site key (``app/api/site_auth.py``): Stripe
cannot send one, and its signature is the stronger authentication anyway. That
exemption is why signature verification here is mandatory rather than
best-effort.
"""
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from app.api.internal_auth import require_internal_admin_access  # noqa: F401  (imported for parity)
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.api.user_auth import resolve_user_id
from app.schemas.payment import (
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    PaymentWebhookResponse,
    UserTokenRequest,
    UserTokenResponse,
)
from app.services.billing_service import BillingService
from app.services.payment_service import (
    SUBSCRIPTION_ACTIVE_EVENTS,
    SUBSCRIPTION_ENDED_EVENTS,
    PaymentError,
    PaymentService,
)
from app.services.user_token_service import UserTokenError, UserTokenNotConfigured, UserTokenService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["payments"])

payment_service = PaymentService()
billing_service = BillingService()
user_token_service = UserTokenService()


@router.post("/auth/user-token", response_model=UserTokenResponse, responses=STANDARD_ERROR_RESPONSES)
def issue_user_token(payload: UserTokenRequest):
    """Mint a short-lived token for one customer.

    Site-key protected, so only the website can ask. The website is still
    trusted to know who is logged in -- that cannot be removed without it
    handing over its own login -- but from here on the token decides the
    customer on every request, so a bug that sends the wrong id is refused
    instead of silently spending someone else's credits.
    """
    try:
        return user_token_service.issue(payload.user_id)
    except UserTokenNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UserTokenError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/payments/checkout", response_model=CheckoutSessionResponse, responses=STANDARD_ERROR_RESPONSES)
def create_checkout(payload: CheckoutSessionRequest, request: Request):
    """Start a subscription purchase and return the URL to send the customer to.

    The customer comes from the user token, not the body, for the same reason
    every other money path does: this decides whose account gets the plan, and a
    website bug that named the wrong id would sell one customer a subscription
    on another's account.
    """
    user_id = resolve_user_id(request, payload.user_id)

    if payload.plan not in BillingService.PLAN_DEFAULTS:
        raise HTTPException(status_code=400, detail=f"Unknown plan '{payload.plan}'")
    if payload.plan == "trial":
        # Refused explicitly: a paid checkout for the free tier would take money
        # for something already given away.
        raise HTTPException(status_code=400, detail="The trial plan is not purchasable")

    try:
        return payment_service.create_checkout_session(
            user_id=user_id,
            plan=payload.plan,
            success_url=payload.success_url,
            cancel_url=payload.cancel_url,
        )
    except PaymentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/payments/stripe/webhook", response_model=PaymentWebhookResponse, include_in_schema=False)
async def stripe_webhook(request: Request, stripe_signature: str = Header(default="", alias="Stripe-Signature")):
    # Raw bytes, not the parsed body: the signature covers the exact payload
    # Stripe sent, and re-serialising JSON changes whitespace and key order
    # enough to invalidate it.
    payload = await request.body()

    try:
        event = payment_service.verify_signature(payload, stripe_signature)
    except PaymentError as exc:
        logger.warning("Rejected Stripe webhook: %s", exc.message)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    event_id = str(event.get("id"))
    event_type = str(event.get("type") or "")

    if not payment_service.claim_event(event_id, event_type):
        # Stripe redelivers by design. Answering 200 stops it retrying an event
        # that is already applied.
        return {"received": True, "applied": False, "reason": "duplicate"}

    applied, reason = _apply_event(event_id, event_type, (event.get("data") or {}).get("object") or {})
    return {"received": True, "applied": applied, "reason": reason}


def _apply_event(event_id: str, event_type: str, event_object: dict) -> tuple[bool, str]:
    """Grant or end a plan. Never raises: an exception here makes Stripe retry
    an event already recorded as claimed, which would never succeed."""
    if event_type not in SUBSCRIPTION_ACTIVE_EVENTS and event_type not in SUBSCRIPTION_ENDED_EVENTS:
        return False, "ignored_event_type"

    user_id = PaymentService.extract_user_id(event_object)
    if not user_id:
        # Worth an error log rather than a shrug: it means checkout was created
        # without metadata, so a paying customer is about to receive nothing.
        logger.error("Stripe event carried no user_id event_id=%s type=%s", event_id, event_type)
        return False, "missing_user_id"

    try:
        if event_type in SUBSCRIPTION_ENDED_EVENTS:
            billing_service.set_plan(user_id, {"current_plan": "trial", "plan_status": "cancelled"})
            payment_service.record_outcome(event_id, user_id=user_id, plan="trial")
            logger.info("Subscription ended user_id=%s event_id=%s", user_id, event_id)
            return True, "subscription_ended"

        plan = PaymentService.plan_for_price(PaymentService.extract_price_id(event_object))
        if not plan:
            logger.error(
                "No plan mapped for Stripe price event_id=%s user_id=%s -- set STRIPE_PRICE_MAP",
                event_id,
                user_id,
            )
            return False, "unmapped_price"

        billing_service.set_plan(user_id, {"current_plan": plan, "plan_status": "active"})
        payment_service.record_outcome(event_id, user_id=user_id, plan=plan)
        logger.info("Plan granted from payment user_id=%s plan=%s event_id=%s", user_id, plan, event_id)
        return True, "plan_granted"
    except Exception:  # noqa: BLE001 - a retry cannot fix this, so do not ask for one
        logger.exception("Failed to apply Stripe event event_id=%s", event_id)
        return False, "apply_failed"
