from __future__ import annotations

import json
import random
import string
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import median
from typing import Any

BASE_URL = "http://127.0.0.1:8002"
ADMIN_LOGIN = "admin"
REQUEST_TIMEOUT_SECONDS = 20


@dataclass
class StepResult:
    name: str
    passed: bool
    details: dict[str, Any]


@dataclass
class UpdateAttempt:
    operator: str
    status_code: int
    elapsed_ms: float
    response: dict[str, Any] | list[Any] | str | None


def _url(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}{path}"


def _request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any]:
    body: bytes | None = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(_url(path), data=body, method=method.upper(), headers=req_headers)

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return resp.status, None
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body_obj: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body_obj = raw
        return exc.code, body_obj
    except (urllib.error.URLError, OSError, socket.error) as exc:
        # 599 = network/connectivity error in client-side transport.
        return 599, {"error": str(exc), "kind": "network"}


def _must_request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    expected_status: int | tuple[int, ...] = 200,
) -> Any:
    status, body = _request_json(method, path, payload=payload, headers=headers)
    allowed = (expected_status,) if isinstance(expected_status, int) else expected_status
    if status not in allowed:
        raise RuntimeError(f"HTTP {status}: {body}")
    return body


def _random_suffix(size: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(size))


def _operator_headers(login: str) -> dict[str, str]:
    return {"X-Operator-Login": login}


def _admin_headers() -> dict[str, str]:
    return _operator_headers(ADMIN_LOGIN)


def _bootstrap_admin_id() -> int:
    data = _must_request_json("POST", "/api/auth/login", payload={"login": ADMIN_LOGIN}, expected_status=200)
    return int(data["user"]["id"])


def _team_id(code: str = "A") -> int:
    teams = _must_request_json("GET", "/api/admin/teams", headers=_admin_headers(), expected_status=200)
    for team in teams:
        if str(team.get("kod")) == code:
            return int(team["id"])
    raise RuntimeError(f"Team with code {code} not found")


def _create_user(team_id: int, role_id: int = 1) -> str:
    for _ in range(20):
        suffix = _random_suffix(10)
        login = f"load_{suffix}"
        payload = {
            "login": login,
            "imie": "Load",
            "nazwisko": suffix,
            "email": f"{login}@example.local",
            "rolaId": role_id,
            "zespolId": team_id,
            "aktywny": True,
        }
        status, body = _request_json("POST", "/api/admin/users", payload=payload, headers=_admin_headers())
        if status == 200:
            return login
        if status == 409:
            continue
        raise RuntimeError(f"HTTP {status}: {body}")
    raise RuntimeError("Unable to create unique synthetic user after 20 attempts")


def _create_inspection(admin_id: int, label: str) -> int:
    payload = {
        "nazwaPodmiotu": label,
        "typInspekcji": "Kontrola",
        "poczatekInspekcji": "2026-05-01",
        "koniecInspekcji": "2026-05-20",
        "osobaKierujacaUserId": admin_id,
    }
    data = _must_request_json(
        "POST",
        "/api/structure/inspections",
        payload=payload,
        headers=_admin_headers(),
        expected_status=201,
    )
    return int(data["id"])


def _create_recommendation(inspection_id: int, pozycja: int) -> int:
    payload = {
        "inspectionId": inspection_id,
        "pozycja": pozycja,
        "komentarz": f"seed-{pozycja}",
    }
    data = _must_request_json("POST", "/api/recommendations", payload=payload, headers=_admin_headers(), expected_status=201)
    return int(data["id"])


def _acquire_lock(module_name: str, record_id: int, login: str) -> tuple[int, Any]:
    return _request_json("POST", f"/api/locks/{module_name}/{record_id}/acquire", headers=_operator_headers(login))


def _release_lock(module_name: str, record_id: int, login: str, lock_token: str | None) -> None:
    if not lock_token:
        return
    _request_json(
        "POST",
        f"/api/locks/{module_name}/{record_id}/release",
        payload={"lockToken": lock_token},
        headers=_operator_headers(login),
    )


def _update_recommendation_once(recommendation_id: int, login: str, index: int) -> UpdateAttempt:
    import time

    started = time.perf_counter()
    lock_token: str | None = None
    try:
        detail_status, detail = _request_json(
            "GET",
            f"/api/recommendations/{recommendation_id}",
            headers=_operator_headers(login),
        )
        if detail_status != 200:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return UpdateAttempt(login, detail_status, elapsed_ms, detail)

        expected_updated_at = str(detail.get("zaktualizowanoO") or "")
        lock_status, lock_body = _acquire_lock("recommendations", recommendation_id, login)
        if lock_status != 200:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return UpdateAttempt(login, lock_status, elapsed_ms, lock_body)

        lock_token = str(lock_body.get("lockToken") or "")
        update_payload = {
            "komentarz": f"parallel-update-{index}-{_random_suffix(4)}",
            "lockToken": lock_token,
            "expectedUpdatedAt": expected_updated_at,
        }
        status, body = _request_json(
            "PUT",
            f"/api/recommendations/{recommendation_id}",
            payload=update_payload,
            headers=_operator_headers(login),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return UpdateAttempt(login, status, elapsed_ms, body)
    finally:
        _release_lock("recommendations", recommendation_id, login, lock_token)


def _percentile_ms(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * p)
    return ordered[idx]


def test_same_record_write_collision(operators: list[str], recommendation_id: int) -> StepResult:
    attempts: list[UpdateAttempt] = []
    with ThreadPoolExecutor(max_workers=len(operators)) as pool:
        futures = [pool.submit(_update_recommendation_once, recommendation_id, login, idx) for idx, login in enumerate(operators, start=1)]
        for future in as_completed(futures):
            attempts.append(future.result())

    status_counts: dict[int, int] = {}
    for attempt in attempts:
        status_counts[attempt.status_code] = status_counts.get(attempt.status_code, 0) + 1

    succeeded = status_counts.get(200, 0)
    had_unexpected_5xx = any(code >= 500 for code in status_counts)
    allowed_codes = {200, 409, 423}
    had_unexpected_code = any(code not in allowed_codes for code in status_counts)
    passed = succeeded >= 1 and not had_unexpected_5xx and not had_unexpected_code

    return StepResult(
        name="same_record_write_collision",
        passed=passed,
        details={
            "attempts": len(attempts),
            "statusCounts": status_counts,
            "successes": succeeded,
            "p50Ms": round(median([a.elapsed_ms for a in attempts]), 2) if attempts else 0.0,
            "p95Ms": round(_percentile_ms([a.elapsed_ms for a in attempts], 0.95), 2),
        },
    )


def test_parallel_writes_different_records(operators: list[str], inspection_id: int, records_to_create: int) -> StepResult:
    rec_ids = [_create_recommendation(inspection_id, pozycja=100 + i) for i in range(records_to_create)]

    attempts: list[UpdateAttempt] = []
    max_workers = min(len(rec_ids), len(operators))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for idx, rec_id in enumerate(rec_ids, start=1):
            operator = operators[idx % len(operators)]
            futures.append(pool.submit(_update_recommendation_once, rec_id, operator, idx))

        for future in as_completed(futures):
            attempts.append(future.result())

    status_counts: dict[int, int] = {}
    for attempt in attempts:
        status_counts[attempt.status_code] = status_counts.get(attempt.status_code, 0) + 1

    successes = status_counts.get(200, 0)
    failed_5xx = sum(count for code, count in status_counts.items() if code >= 500)
    passed = successes >= int(0.9 * len(attempts)) and failed_5xx == 0

    elapsed_values = [a.elapsed_ms for a in attempts]
    return StepResult(
        name="parallel_writes_different_records",
        passed=passed,
        details={
            "attempts": len(attempts),
            "statusCounts": status_counts,
            "successRate": round((successes / len(attempts)) * 100.0, 2) if attempts else 0.0,
            "p50Ms": round(median(elapsed_values), 2) if elapsed_values else 0.0,
            "p95Ms": round(_percentile_ms(elapsed_values, 0.95), 2),
            "p99Ms": round(_percentile_ms(elapsed_values, 0.99), 2),
        },
    )
