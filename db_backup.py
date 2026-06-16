from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.database.connection import BASE_DIR, DB_PATH


logger = logging.getLogger(__name__)
_BACKUP_IN_PROGRESS = threading.Event()


def _env_truthy(name: str, default: str = "0") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_timezone() -> ZoneInfo | timezone:
    try:
        return ZoneInfo("Europe/Warsaw")
    except ZoneInfoNotFoundError:
        logger.warning("DB backup timezone Europe/Warsaw unavailable, fallback UTC")
        return timezone.utc


def _parse_backup_time(raw: str) -> tuple[int, int]:
    text = str(raw or "").strip()
    if not text:
        return 23, 0

    parts = text.split(":", 1)
    if len(parts) != 2:
        raise ValueError("DB_BACKUP_TIME must be in HH:MM format")

    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("DB_BACKUP_TIME must be in HH:MM 24h format")
    return hour, minute


def _resolve_backup_dir() -> Path:
    raw = (os.getenv("DB_BACKUP_DIR") or "Backups").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _lock_file_path(backup_dir: Path) -> Path:
    return backup_dir / ".db_backup.lock"


def _state_file_path(backup_dir: Path) -> Path:
    return backup_dir / ".db_backup_state.json"


def is_backup_in_progress() -> bool:
    return _BACKUP_IN_PROGRESS.is_set()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")


def _acquire_process_lock(lock_path: Path, *, stale_after_seconds: int = 21600) -> bool:
    payload = {"pid": os.getpid(), "createdAt": int(time.time())}

    def _create_lock() -> bool:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
            return True
        except Exception:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    if _create_lock():
        return True

    # Try to recover stale lock once.
    try:
        raw = lock_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        created_at = int(data.get("createdAt") or 0)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        created_at = 0

    now_ts = int(time.time())
    if created_at <= 0 or now_ts - created_at > stale_after_seconds:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            return False
        return _create_lock()

    return False


def _release_process_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove backup lock file: %s", lock_path)


def _cleanup_old_backups(backup_dir: Path, prefix: str, keep_days: int, now: datetime) -> None:
    if keep_days <= 0:
        return

    cutoff = now - timedelta(days=keep_days)
    pattern = f"{prefix}-*.db"
    for item in backup_dir.glob(pattern):
        if not item.is_file():
            continue
        modified = datetime.fromtimestamp(item.stat().st_mtime, tz=now.tzinfo)
        if modified < cutoff:
            try:
                item.unlink()
            except OSError as exc:
                logger.warning("Failed to remove old DB backup %s: %s", item, exc)


def _safe_unlink(path: Path, *, retries: int = 5, delay_seconds: float = 0.05) -> bool:
    for _ in range(max(1, retries)):
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            time.sleep(delay_seconds)
    return False


def _safe_replace(src: Path, dst: Path, *, retries: int = 5, delay_seconds: float = 0.05) -> None:
    last_exc: OSError | None = None
    for _ in range(max(1, retries)):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(delay_seconds)
    assert last_exc is not None
    raise last_exc


def _create_backup_file(target_file: Path) -> None:
    # Write into a temp file first and atomically move it into place.
    temp_file = target_file.with_name(f"{target_file.name}.{secrets.token_hex(4)}.tmp")
    try:
        source_conn = sqlite3.connect(DB_PATH)
        try:
            backup_conn = sqlite3.connect(temp_file)
            try:
                source_conn.backup(backup_conn)
            finally:
                backup_conn.close()
        finally:
            source_conn.close()

        _safe_replace(temp_file, target_file)
    except Exception:
        if not _safe_unlink(temp_file):
            logger.warning("Failed to remove temp DB backup file: %s", temp_file)
        raise


def run_daily_db_backup_once() -> None:
    if not _env_truthy("DB_BACKUP_ENABLED", "0"):
        return

    tz = _resolve_timezone()
    now = datetime.now(tz)

    try:
        target_hour, target_minute = _parse_backup_time(os.getenv("DB_BACKUP_TIME") or "23:00")
    except ValueError as exc:
        logger.error("DB backup disabled due to invalid DB_BACKUP_TIME: %s", exc)
        return

    backup_dir = _resolve_backup_dir()
    backup_dir.mkdir(parents=True, exist_ok=True)

    lock_path = _lock_file_path(backup_dir)
    if not _acquire_process_lock(lock_path):
        return

    started_at = time.perf_counter()
    _BACKUP_IN_PROGRESS.set()
    try:
        state_path = _state_file_path(backup_dir)
        state = _load_state(state_path)
        today_key = now.date().isoformat()

        scheduled_for_today = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        if now < scheduled_for_today:
            return

        if str(state.get("last_backup_date") or "") == today_key:
            return

        prefix = (os.getenv("DB_BACKUP_FILENAME_PREFIX") or "rejestr-backup").strip() or "rejestr-backup"
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        target_file = backup_dir / f"{prefix}-{timestamp}.db"

        logger.info("DB backup started: target=%s", target_file)
        try:
            _create_backup_file(target_file)
        except Exception as exc:  # noqa: BLE001
            logger.exception("DB backup failed: target=%s error=%s", target_file, exc)
            return

        state["last_backup_date"] = today_key
        state["last_backup_file"] = str(target_file)
        _save_state(state_path, state)

        keep_days_raw = (os.getenv("DB_BACKUP_KEEP_DAYS") or "0").strip()
        try:
            keep_days = int(keep_days_raw)
        except ValueError:
            logger.warning("Invalid DB_BACKUP_KEEP_DAYS=%r, cleanup disabled", keep_days_raw)
            keep_days = 0
        _cleanup_old_backups(backup_dir, prefix, keep_days, now)

        elapsed = time.perf_counter() - started_at
        logger.info("DB backup created: %s (%.3fs)", target_file, elapsed)
    finally:
        _BACKUP_IN_PROGRESS.clear()
        _release_process_lock(lock_path)
