from contextlib import asynccontextmanager
import asyncio
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.runtime_config import load_env_file

load_env_file()

from app.api import admin, audit_log, auth, inspections, locks, notifications, obligating_decisions, recommendations, reports, risk_exposure, schedules, slowniki
from app.db_backup import is_backup_in_progress, run_daily_db_backup_once
from app.db import init_db
from app.schedule_engine import run_due_schedules_once


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()

    stop_event = asyncio.Event()

    async def _worker() -> None:
        while not stop_event.is_set():
            try:
                await asyncio.to_thread(run_due_schedules_once)
            except Exception as exc:  # noqa: BLE001
                print(f"[SCHEDULE-WORKER] {exc}")
            try:
                await asyncio.to_thread(run_daily_db_backup_once)
            except Exception as exc:  # noqa: BLE001
                print(f"[DB-BACKUP-WORKER] {exc}")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=60)
            except TimeoutError:
                pass

    task = asyncio.create_task(_worker())
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Rejestr Backend", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def structured_http_exception_handler(_: Request, exc: HTTPException):
    if exc.status_code in {409, 423} and isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def _env_truthy(name: str, default: str = "0") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _cors_origins() -> list[str]:
    raw = (os.getenv("CORS_ALLOWED_ORIGINS") or "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]

    env_name = (os.getenv("APP_ENV") or "dev").strip().lower()
    if env_name in {"dev", "local", "test"}:
        return [
            "http://localhost:3002",
            "http://127.0.0.1:3002",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    return []


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_https(request: Request, call_next):
    if not _env_truthy("ENFORCE_HTTPS", "0"):
        return await call_next(request)

    # Behind reverse proxy respect forwarded protocol if available.
    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    is_https = request.url.scheme == "https" or forwarded_proto == "https"
    if is_https:
        return await call_next(request)

    return JSONResponse(status_code=400, content={"detail": "HTTPS is required"})


@app.middleware("http")
async def maintenance_during_db_backup(request: Request, call_next):
    if not _env_truthy("DB_BACKUP_BLOCK_REQUESTS", "1"):
        return await call_next(request)

    if not is_backup_in_progress():
        return await call_next(request)

    if request.url.path == "/health":
        return await call_next(request)

    return JSONResponse(status_code=503, content={"detail": "Service temporarily unavailable: database backup in progress"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(inspections.router)
app.include_router(recommendations.router)
app.include_router(obligating_decisions.router)
app.include_router(risk_exposure.router)
app.include_router(slowniki.router)
app.include_router(reports.router)
app.include_router(notifications.router)
app.include_router(schedules.router)
app.include_router(locks.router)
app.include_router(audit_log.router)
