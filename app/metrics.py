# app/metrics.py
import time
from fastapi import Request
from fastapi.responses import Response
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

# ---- Metrics ----
REQUEST_COUNT = Counter(
    "app_requests_total", "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Request latency by endpoint",
    ["endpoint"], buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
)

PREDICTIONS_TOTAL = Counter(
    "model_predictions_total", "Number of predictions served"
)

INFERENCE_SECONDS = Histogram(
    "model_inference_seconds", "Model inference latency (s)"
)

APP_HEALTH = Gauge("app_health", "1 if app is healthy, else 0")
APP_HEALTH.set(1)
# ------------------

def install_metrics(app):
    """Attach a function-based HTTP middleware that records count & latency."""
    @app.middleware("http")
    async def _mw(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.url.path).observe(elapsed)
        return response

def metrics_endpoint():
    """FastAPI-compatible handler for /metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
