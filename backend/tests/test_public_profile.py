"""Reading another account, and being honest about what Meta did not return.

The feature the subscription is really sold on — "build my shop the way Nike's
page is built" — so the cases here are about the two ways it goes wrong
silently: accepting input customers actually paste, and letting the model talk
about metrics Meta never provided.
"""
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.public_profile_service import PublicProfileService  # noqa: E402


class StubConnections:
    def __init__(self, payload=None, ig_id="17841400000000000"):
        self.payload = payload
        self.requested = {}
        self._ig_id = ig_id

    def get_connection_for_meta(self, user_id, account_id):
        return {"access_token": "tok", "instagram_business_account_id": self._ig_id}

    def meta_get(self, **kwargs):
        self.requested = kwargs
        return self.payload


def _payload(**overrides):
    base = {
        "business_discovery": {
            "username": "nike",
            "name": "Nike",
            "biography": "Just Do It",
            "followers_count": 302_000_000,
            "media_count": 1200,
            "media": {
                "data": [
                    {
                        "id": "1",
                        "caption": "Every athlete. Every day.",
                        "like_count": 40_000,
                        "comments_count": 300,
                        "media_type": "VIDEO",
                        "media_product_type": "REELS",
                        "permalink": "https://instagram.com/p/1",
                        "timestamp": "2026-08-01T10:00:00+0000",
                    },
                    {
                        "id": "2",
                        "caption": "New drop.",
                        "like_count": 12_000,
                        "comments_count": 80,
                        "media_type": "IMAGE",
                        "media_product_type": "FEED",
                        "permalink": "https://instagram.com/p/2",
                        "timestamp": "2026-07-30T10:00:00+0000",
                    },
                ]
            },
        }
    }
    base["business_discovery"].update(overrides)
    return base


# ------------------------------------------------------------------- input


@pytest.mark.parametrize("given", [
    "nike",
    "@nike",
    "https://www.instagram.com/nike/",
    "instagram.com/nike",
    "https://instagram.com/nike/?hl=en",
    "  @nike  ",
])
def test_every_way_a_customer_writes_a_handle_is_accepted(given):
    """People paste a handle, an @handle, or the URL from the address bar.
    Rejecting two of three reads as the feature being broken."""
    assert PublicProfileService.normalize_handle(given) == "nike"


@pytest.mark.parametrize("bad", ["", "   ", "has spaces", "way" * 20, "bad/slash"])
def test_input_that_is_not_a_handle_is_refused(bad):
    with pytest.raises(HTTPException) as exc:
        PublicProfileService.normalize_handle(bad)
    assert exc.value.status_code == 400


# ------------------------------------------------------------------ lookup


def test_the_request_asks_meta_for_the_named_account():
    stub = StubConnections(_payload())

    PublicProfileService(stub).fetch(user_id="u", account_id="a", handle="@nike")

    fields = stub.requested["fields"]
    assert "business_discovery.username(nike)" in fields
    # Captions are the substance -- they carry the hook style and CTA pattern the
    # customer wants translated. A request without them returns numbers only.
    assert "caption" in fields
    assert "like_count" in fields


def test_a_personal_account_is_named_as_the_reason_not_reported_as_missing():
    """No retry fixes a personal account, and "not found" sends the customer
    hunting for a typo that is not there."""
    stub = StubConnections({"id": "123"})  # business_discovery absent

    with pytest.raises(HTTPException) as exc:
        PublicProfileService(stub).fetch(user_id="u", account_id="a", handle="someone")

    assert exc.value.status_code == 404
    assert "Business and Creator" in exc.value.detail


def test_a_connection_without_a_business_account_id_fails_before_calling_meta():
    stub = StubConnections(_payload(), ig_id="")

    with pytest.raises(HTTPException) as exc:
        PublicProfileService(stub).fetch(user_id="u", account_id="a", handle="nike")

    assert exc.value.status_code == 400
    assert stub.requested == {}, "no request should have been made"


def test_the_profile_and_its_posts_come_back_shaped():
    profile = PublicProfileService(StubConnections(_payload())).fetch(
        user_id="u", account_id="a", handle="nike"
    )

    assert profile["handle"] == "nike"
    assert profile["followers_count"] == 302_000_000
    assert [p["product_type"] for p in profile["posts"]] == ["REELS", "FEED"]
    assert profile["posts"][0]["like_count"] == 40_000


# ------------------------------------------------------- honesty about data


def test_the_payload_states_which_metrics_meta_withheld():
    """Meta returns no reach or impressions for an account the customer does not
    own. Leaving that unsaid is how a competitor analysis ends up quoting reach
    figures that were invented."""
    profile = PublicProfileService(StubConnections(_payload())).fetch(
        user_id="u", account_id="a", handle="nike"
    )

    assert "reach" in profile["unavailable_metrics"]
    assert "impressions" in profile["unavailable_metrics"]
    assert profile["available_metrics"] == ["like_count", "comment_count"]


def test_the_prompt_context_carries_the_same_warning():
    service = PublicProfileService(StubConnections(_payload()))
    context = service.as_prompt_context(service.fetch(user_id="u", account_id="a", handle="nike"))

    assert "Do not state or estimate reach" in context["metrics_note"]


def test_the_prompt_context_summarises_the_content_mix():
    """The question is how the page is *built*, so the mix of formats matters as
    much as any single post."""
    service = PublicProfileService(StubConnections(_payload()))
    context = service.as_prompt_context(service.fetch(user_id="u", account_id="a", handle="nike"))

    assert context["content_mix"] == {"REELS": 1, "FEED": 1}
    assert context["median_likes"] in (12_000, 40_000)
    assert context["recent_captions"][0]["caption"].startswith("Every athlete")


def test_the_prompt_context_drops_what_the_model_cannot_use():
    """Post ids and permalinks spend context and cannot inform an answer."""
    service = PublicProfileService(StubConnections(_payload()))
    context = service.as_prompt_context(service.fetch(user_id="u", account_id="a", handle="nike"))

    rendered = str(context)
    assert "permalink" not in rendered
    assert "post_id" not in rendered


# ------------------------------------------------------- wired to the agent


def test_the_task_is_schema_enforced_like_every_other_structured_task():
    """Without a schema it falls back to heading parsing, which Sonnet does not
    follow -- the failure that made carousels return nothing."""
    from app.services.llm_service import LLMService

    schema = LLMService()._structured_schema("public_profile_analysis")

    assert schema is not None
    props = schema["properties"]["structured_output"]["properties"]
    assert "adapted_plan" in props
    assert "first_three_posts" in props


def test_the_output_is_shaped_as_a_plan_not_a_report():
    """The customer asked how to build *their* page. A list of observations
    about the reference account is a report they cannot act on."""
    from app.schemas.agent import PublicProfileAnalysisStructuredOutput

    fields = set(PublicProfileAnalysisStructuredOutput.model_fields)

    assert {"adapted_plan", "first_three_posts", "what_does_not_transfer"} <= fields


def test_the_instruction_forbids_inventing_reach():
    """Meta withholds reach for accounts the customer does not own, so any
    reach figure in a competitor analysis is fabricated."""
    from app.services.langflow_service import LangflowService

    instruction = LangflowService()._task_instruction("public_profile_analysis").lower()

    assert "reach" in instruction
    assert "never state or estimate" in instruction


def test_the_instruction_warns_that_scale_does_not_transfer():
    """A small shop copying a global brand's awareness posts reaches nobody."""
    instruction_source = __import__(
        "app.services.langflow_service", fromlist=["LangflowService"]
    ).LangflowService()._task_instruction("public_profile_analysis").lower()

    assert "scale" in instruction_source


def test_the_task_costs_more_than_an_audit():
    """It adds a Meta round trip and a high-effort pass over the captions."""
    from app.services.billing_service import BillingService

    assert BillingService.generation_cost("public_profile_analysis") > BillingService.generation_cost("profile_audit")


def test_only_paid_tiers_get_it():
    """It is the clearest reason to upgrade, so it must not be in the trial."""
    from app.services.billing_service import BillingService

    trial = BillingService.PLAN_DEFAULTS["trial"]["allowed_task_types"]
    pro = BillingService.PLAN_DEFAULTS["pro"]["allowed_task_types"]

    assert "public_profile_analysis" not in trial
    assert "public_profile_analysis" in pro


def test_the_reference_data_reaches_the_prompt():
    """Threading it only as far as the service would leave the model answering
    from what it remembers about the brand -- the generic reply the customer
    could have had for free."""
    from app.services.llm_service import LLMService

    service = PublicProfileService(StubConnections(_payload()))
    context = service.as_prompt_context(service.fetch(user_id="u", account_id="a", handle="nike"))

    sections = LLMService()._build_prompt_sections(
        message="build my sports shop like this",
        task_type="public_profile_analysis",
        public_profile_context=context,
    )
    joined = "\n".join(s["content"] for s in sections)

    assert "@nike" in joined
    assert "Every athlete" in joined
    assert "Do not state or estimate reach" in joined


def test_no_reference_data_means_no_section():
    from app.services.llm_service import LLMService

    names = [
        s["name"]
        for s in LLMService()._build_prompt_sections(
            message="x", task_type="public_profile_analysis", public_profile_context=None
        )
    ]

    assert "public_profile_context" not in names


def test_an_empty_profile_produces_no_context():
    """An empty block reads to the model as data that exists and is blank, which
    invites it to fill the gap."""
    assert PublicProfileService(StubConnections()).as_prompt_context({}) is None
