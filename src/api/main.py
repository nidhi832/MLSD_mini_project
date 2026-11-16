"""
Main FastAPI Application Entrypoint (F1, F8, A4, D3, D5).
"""

import time
import uuid
import os
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse

os.environ.setdefault("OTEL_SERVICE_NAME", "claim-api-gateway")
from src.core.tracing import init_tracing  # noqa: E402

init_tracing()

from src.api.routes import router as api_router  # noqa: E402
from src.core.observability import get_prometheus_metrics, logger  # noqa: E402
from src.core.shared_state import get_redis  # noqa: E402

app = FastAPI(
    title="Air Passenger Claim Triage Service",
    description="Multi-Agent Production Service for DGCA CAR Claim Triage (UE24AM342AA5 Mini Project 04)",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_and_timing_middleware(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID", f"corr-{uuid.uuid4().hex[:8]}")
    request.state.correlation_id = corr_id
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    response.headers["X-Correlation-ID"] = corr_id
    response.headers["X-Response-Time-Seconds"] = f"{duration:.4f}"
    return response


# Include API v1 router
app.include_router(api_router)

# Mount static directory for UI Client (A8)
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
assets_dir = os.path.join(static_dir, "assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")



@app.get("/health")
async def health_check():
    """Health endpoint for container readiness and liveness checks (D3, F8)."""
    return {
        "status": "healthy",
        "service": "air-passenger-claim-triage",
        "queue_backend": "redis" if get_redis() is not None else "in-memory (inline execution)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scraping endpoint (D5)."""
    data = get_prometheus_metrics()
    return Response(content=data, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the web client (A8)."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
    return HTMLResponse("<h2>Air Passenger Claim Triage Service API Running</h2><p><a href='/docs'>Swagger Docs</a></p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
