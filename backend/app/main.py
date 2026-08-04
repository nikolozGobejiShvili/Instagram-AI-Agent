import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv

# Loaded before the route modules are imported, not after. Services are
# constructed at import time and some read configuration in their constructor
# (BillingService resolves BILLING_DB_PATH), so a value set only in .env would
# otherwise be missed on startup.
load_dotenv()

from app.api.routes.connected_accounts import router as connected_accounts_router  # noqa: E402
from app.api.routes.app_bootstrap import router as app_bootstrap_router
from app.api.routes.billing import router as billing_router
from app.api.routes.generation_history import router as generation_history_router
from app.api.routes.internal_generation_debug import router as internal_generation_debug_router
from app.api.routes.knowledge_packs import router as knowledge_packs_router
from app.api.routes.maintenance import router as maintenance_router
from app.api.routes.instagram_connection import router as instagram_connection_router
from app.api.routes.instagram_context_sync import router as instagram_context_sync_router
from app.api.routes.instagram_debug import router as instagram_debug_router
from app.api.routes.instagram_insights import router as instagram_insights_router
from app.api.routes.instagram_media import router as instagram_media_router
from app.api.routes.instagram_profile import router as instagram_profile_router
from app.api.routes.recent_posts_context import router as recent_posts_context_router
from app.api.routes.recent_content_context import router as recent_content_context_router
from app.api.routes.agent import router as agent_router
from app.api.routes.profile_context import router as profile_context_router
from app.api.routes.carousel_media import router as carousel_media_router  # noqa: E402
from app.api.routes.generation_jobs import job_worker, router as generation_jobs_router  # noqa: E402
from app.error_handling import http_exception_handler, request_validation_handler, unhandled_exception_handler  # noqa: E402
from app.api.routes.link_context import router as link_context_router  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run the background job worker alongside the API.

    Disabled by setting JOB_WORKER_ENABLED=false -- tests drive the worker
    explicitly so results are deterministic, and a deployment that runs the
    worker as a separate process wants the web replicas to leave it alone.
    """
    worker_enabled = os.getenv("JOB_WORKER_ENABLED", "true").strip().lower() == "true"
    if worker_enabled:
        job_worker.start()
    try:
        yield
    finally:
        if worker_enabled:
            job_worker.stop()


app = FastAPI(
    title=os.getenv("APP_NAME", "instagram-agent"),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, request_validation_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(agent_router)
app.include_router(app_bootstrap_router)
app.include_router(billing_router)
app.include_router(connected_accounts_router)
app.include_router(carousel_media_router)
app.include_router(generation_history_router)
app.include_router(generation_jobs_router)
app.include_router(internal_generation_debug_router)
app.include_router(knowledge_packs_router)
app.include_router(maintenance_router)
app.include_router(instagram_connection_router)
app.include_router(instagram_context_sync_router)
app.include_router(instagram_debug_router)
app.include_router(instagram_insights_router)
app.include_router(instagram_media_router)
app.include_router(instagram_profile_router)
app.include_router(recent_posts_context_router)
app.include_router(profile_context_router)
app.include_router(link_context_router)
app.include_router(recent_content_context_router)

@app.get("/")
def root():
    return {"message": "Instagram Agent backend is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": os.getenv("APP_NAME", "instagram-agent"),
        "version": "0.1.0",
    }
