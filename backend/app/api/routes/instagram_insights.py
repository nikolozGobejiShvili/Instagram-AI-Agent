from fastapi import APIRouter

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.instagram_insights import InstagramInsightsResponse
from app.services.instagram_insights_service import InstagramInsightsService

router = APIRouter(prefix="/api/v1/instagram-insights", tags=["instagram-insights"])

instagram_insights_service = InstagramInsightsService()


@router.get("/{user_id}", response_model=InstagramInsightsResponse, responses=STANDARD_ERROR_RESPONSES)
def get_instagram_insights(user_id: str, account_id: str | None = None, period: str = "30d"):
    return instagram_insights_service.get_insights(user_id, account_id, period)
