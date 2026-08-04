from fastapi import APIRouter

from app.schemas.api_error import STANDARD_ERROR_RESPONSES
from app.schemas.maintenance import MaintenanceJobRunResponse, MaintenanceStatusResponse
from app.services.maintenance_service import MaintenanceService

router = APIRouter(prefix="/api/v1/maintenance", tags=["maintenance"])

maintenance_service = MaintenanceService()


@router.post("/run/monthly-usage-reset", response_model=MaintenanceJobRunResponse, responses=STANDARD_ERROR_RESPONSES)
def run_monthly_usage_reset():
    return maintenance_service.run_monthly_usage_reset()


@router.post("/run/context-freshness-scan", response_model=MaintenanceJobRunResponse, responses=STANDARD_ERROR_RESPONSES)
def run_context_freshness_scan():
    return maintenance_service.run_context_freshness_scan()


@router.post("/run/context-refresh", response_model=MaintenanceJobRunResponse, responses=STANDARD_ERROR_RESPONSES)
def run_context_refresh():
    return maintenance_service.run_context_refresh()


@router.post("/run/connection-health-scan", response_model=MaintenanceJobRunResponse, responses=STANDARD_ERROR_RESPONSES)
def run_connection_health_scan():
    return maintenance_service.run_connection_health_scan()


@router.get("/status", response_model=MaintenanceStatusResponse, responses=STANDARD_ERROR_RESPONSES)
def get_maintenance_status():
    return maintenance_service.get_status()
