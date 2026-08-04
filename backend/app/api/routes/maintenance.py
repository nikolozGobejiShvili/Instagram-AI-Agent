from fastapi import APIRouter, Depends

from app.api.internal_auth import require_internal_admin_access
from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.maintenance import MaintenanceJobRunResponse, MaintenanceStatusResponse
from app.services.maintenance_service import MaintenanceService

router = APIRouter(prefix="/api/v1/maintenance", tags=["maintenance"])

maintenance_service = MaintenanceService()

# These are cron jobs exposed over HTTP. monthly-usage-reset clears the usage
# counter for *every* user, so leaving them open let any caller hand out free
# quota to the whole customer base.
ADMIN_ONLY = [Depends(require_internal_admin_access)]


@router.post(
    "/run/monthly-usage-reset",
    response_model=MaintenanceJobRunResponse,
    responses=STANDARD_ERROR_RESPONSES,
    dependencies=ADMIN_ONLY,
)
def run_monthly_usage_reset():
    return maintenance_service.run_monthly_usage_reset()


@router.post(
    "/run/context-freshness-scan",
    response_model=MaintenanceJobRunResponse,
    responses=STANDARD_ERROR_RESPONSES,
    dependencies=ADMIN_ONLY,
)
def run_context_freshness_scan():
    return maintenance_service.run_context_freshness_scan()


@router.post(
    "/run/context-refresh",
    response_model=MaintenanceJobRunResponse,
    responses=STANDARD_ERROR_RESPONSES,
    dependencies=ADMIN_ONLY,
)
def run_context_refresh():
    return maintenance_service.run_context_refresh()


@router.post(
    "/run/connection-health-scan",
    response_model=MaintenanceJobRunResponse,
    responses=STANDARD_ERROR_RESPONSES,
    dependencies=ADMIN_ONLY,
)
def run_connection_health_scan():
    return maintenance_service.run_connection_health_scan()


@router.get("/status", response_model=MaintenanceStatusResponse, responses=STANDARD_ERROR_RESPONSES)
def get_maintenance_status():
    return maintenance_service.get_status()
