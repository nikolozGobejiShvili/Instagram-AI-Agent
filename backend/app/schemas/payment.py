from pydantic import BaseModel, ConfigDict, Field


class UserTokenRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"user_id": "user-1"}},
    )

    user_id: str = Field(..., min_length=1)


class UserTokenResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_token": "eyJzdWIiOiJ1c2VyLTEi....abc123",
                "user_id": "user-1",
                "expires_at": 1785000000,
                "expires_in": 3600,
            }
        },
    )

    user_token: str
    user_id: str
    # Absolute and relative both: the absolute value survives a clock the client
    # cannot compare against, the relative one is what a refresh timer needs.
    expires_at: int
    expires_in: int


class CheckoutSessionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_id": "user-1",
                "plan": "pro",
                "success_url": "https://tichu.example/agent?checkout=done",
                "cancel_url": "https://tichu.example/pricing",
            }
        },
    )

    user_id: str = Field(..., min_length=1)
    # Not a price id: the website should ask for the tier it is selling, and the
    # backend owns which Stripe price that is. Letting the caller name a price
    # would put the plan/price mapping in two places.
    plan: str = Field(..., min_length=1)
    success_url: str = Field(..., min_length=1)
    cancel_url: str = Field(..., min_length=1)


class CheckoutSessionResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "checkout_url": "https://checkout.stripe.com/c/pay/cs_test_...",
                "session_id": "cs_test_a1b2c3",
                "plan": "pro",
            }
        },
    )

    checkout_url: str
    session_id: str
    plan: str


class PaymentWebhookResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"received": True, "applied": True, "reason": "plan_granted"}
        },
    )

    received: bool
    # Separate from `received` on purpose. Stripe only needs to know the event
    # arrived; whether it changed anything is what an operator reading the
    # dashboard needs, and collapsing the two hides every ignored event.
    applied: bool
    reason: str
