from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from app.db import get_connection
from app.schedule_engine import (
    WARSAW_TZ,
    RECIPIENT_STRATEGY_AUTHOR_ONLY,
    RECIPIENT_STRATEGY_INSPECTION_CONTEXT,
    list_template_variables,
    list_schedule_date_fields,
    resolve_module_type,
    resolve_recipient_strategy,
    run_due_schedules_once,
    validate_template_text,
)

router = APIRouter()


class ScheduleDateFieldItem(BaseModel):
    code: str
    label: str


class ScheduleRead(BaseModel):
    id: int
    name: str
    moduleType: str
    dateFieldA: str
    checkEmptyField: str
    daysDifference: int
    subjectTemplate: str
    bodyTemplate: str
    sendHour: int
    sendMinute: int
    targetInspectionLeader: bool
    targetInspectionTeam: bool
    fallbackRecipient: Literal["author"]
    enabled: bool
    createdByUserId: int
    lastRunDate: str | None = None
    createdAt: str
    updatedAt: str
    recipientStrategy: Literal["inspection_context", "author_only"]


class ScheduleCreateRequest(BaseModel):
    name: str
    dateFieldA: str
    checkEmptyField: str
    daysDifference: int
    subjectTemplate: str
    bodyTemplate: str
    sendHour: int
    sendMinute: int = 0
    targetInspectionLeader: bool = True
    targetInspectionTeam: bool = False
    fallbackRecipient: Literal["author"] = "author"
    enabled: bool = True


class ScheduleUpdateRequest(BaseModel):
    name: str | None = None
    dateFieldA: str | None = None
    checkEmptyField: str | None = None
    daysDifference: int | None = None
    subjectTemplate: str | None = None
    bodyTemplate: str | None = None
    sendHour: int | None = None
    sendMinute: int | None = None
    targetInspectionLeader: bool | None = None
    targetInspectionTeam: bool | None = None
    fallbackRecipient: Literal["author"] | None = None
    enabled: bool | None = None


class ScheduleRuleRead(BaseModel):
    id: int
    scheduleId: int
    daysDifference: int
    subjectTemplate: str
    bodyTemplate: str
    enabled: bool
    createdAt: str
    updatedAt: str


class ScheduleRuleCreateRequest(BaseModel):
    daysDifference: int
    subjectTemplate: str
    bodyTemplate: str
    enabled: bool = True


class ScheduleRuleUpdateRequest(BaseModel):
    daysDifference: int | None = None
    subjectTemplate: str | None = None
    bodyTemplate: str | None = None
    enabled: bool | None = None


class ScheduleDispatchRead(BaseModel):
    id: int
    runId: int | None = None
    scheduleId: int
    ruleId: int
    moduleType: str
    recordId: int
    inspectionId: str | None = None
    recommendationId: str | None = None
    sanctionRequestId: str | None = None
    recipientEmail: str
    recipientType: str | None = None
    status: str
    errorMessage: str | None = None
    renderedSubject: str | None = None
    renderedBody: str | None = None
    createdAt: str


class ScheduleRunNowResponse(BaseModel):
    ok: bool
    matched: int
    sent: int
    runAt: str


class TemplateVariableItem(BaseModel):
    token: str
    label: str


class ScheduleDispatchKpiResponse(BaseModel):
    scheduleId: int
    total: int
    sent: int
    failed: int
    successRate: float
    byModule: list[dict[str, Any]]
    byRecipientType: list[dict[str, Any]]


def _resolve_operator(conn: Any, x_operator_login: str) -> dict[str, Any]:
    login = (x_operator_login or "").strip()
    if not login:
        raise HTTPException(status_code=401, detail="Operator nie istnieje")

    row = conn.execute(
        """
        SELECT id, login, rola_id, aktywny
        FROM users
        WHERE lower(login)=lower(?)
        LIMIT 1
        """,
        (login,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Operator nie istnieje")

    operator = dict(row)
    if int(operator["aktywny"]) != 1:
        raise HTTPException(status_code=403, detail="Operator jest nieaktywny")
    return operator


def _ensure_director(operator: dict[str, Any]) -> None:
    if int(operator["rola_id"]) != 3:
        raise HTTPException(status_code=403, detail="Brak uprawnien")


def _map_schedule(row: dict[str, Any]) -> dict[str, Any]:
    recipient_strategy = resolve_recipient_strategy(str(row["date_field_a"]), str(row["date_field_b"]))
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "moduleType": str(row["module_type"]),
        "dateFieldA": str(row["date_field_a"]),
        "checkEmptyField": str(row["date_field_b"]),
        "daysDifference": int(row["days_difference"]),
        "subjectTemplate": str(row["subject_template"]),
        "bodyTemplate": str(row["body_template"]),
        "sendHour": int(row["send_hour"]),
        "sendMinute": int(row["send_minute"]),
        "targetInspectionLeader": bool(row["target_inspection_leader"]),
        "targetInspectionTeam": bool(row["target_inspection_team"]),
        "fallbackRecipient": str(row["fallback_recipient"]),
        "enabled": bool(row["enabled"]),
        "createdByUserId": int(row["created_by_user_id"]),
        "lastRunDate": row["last_run_date"],
        "createdAt": str(row["created_at"]),
        "updatedAt": str(row["updated_at"]),
        "recipientStrategy": recipient_strategy,
    }


def _map_rule(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "scheduleId": int(row["schedule_id"]),
        "daysDifference": int(row["days_difference"]),
        "subjectTemplate": str(row["subject_template"]),
        "bodyTemplate": str(row["body_template"]),
        "enabled": bool(row["enabled"]),
        "createdAt": str(row["created_at"]),
        "updatedAt": str(row["updated_at"]),
    }


def _map_recipient_type_label(recipient_type: str | None) -> str | None:
    if recipient_type is None:
        return None
    labels = {
        "inspection_leader": "Kierujacy",
        "inspection_team": "Zespol",
        "inspection_leader_team": "Kierujacy, Zespol",
        "author": "Fallback: autor",
    }
    return labels.get(recipient_type, recipient_type)


def _dispatch_created_at_local(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return text

    formats = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        local_dt = parsed.replace(tzinfo=timezone.utc).astimezone(WARSAW_TZ)
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")

    try:
        parsed_iso = datetime.fromisoformat(text)
    except ValueError:
        return text

    if parsed_iso.tzinfo is None:
        parsed_iso = parsed_iso.replace(tzinfo=timezone.utc)
    local_dt = parsed_iso.astimezone(WARSAW_TZ)
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")


def _build_dispatch_filter_sql(
    schedule_id: int,
    period: Literal["week", "month", "year", "all"],
    status: Literal["sent", "failed"] | None,
    recipient_email: str | None,
    recipient_type: Literal["inspection_leader", "inspection_team", "author"] | None,
    date_from: str | None,
    date_to: str | None,
) -> tuple[str, list[Any]]:
    where_parts = ["d.schedule_id = ?"]
    params: list[Any] = [int(schedule_id)]

    if period == "week":
        where_parts.append("datetime(d.created_at) >= datetime('now', '-7 days')")
    elif period == "month":
        where_parts.append("datetime(d.created_at) >= datetime('now', '-1 month')")
    elif period == "year":
        where_parts.append("datetime(d.created_at) >= datetime('now', '-1 year')")

    if status is not None:
        where_parts.append("d.status = ?")
        params.append(status)

    if recipient_email is not None and recipient_email.strip():
        where_parts.append("lower(d.recipient_email) LIKE ?")
        params.append(f"%{recipient_email.strip().lower()}%")

    if recipient_type == "inspection_leader":
        where_parts.append("d.recipient_type IN ('inspection_leader', 'inspection_leader_team')")
    elif recipient_type == "inspection_team":
        where_parts.append("d.recipient_type IN ('inspection_team', 'inspection_leader_team')")
    elif recipient_type == "author":
        where_parts.append("d.recipient_type = 'author'")

    if date_from is not None and date_from.strip():
        where_parts.append("date(d.created_at) >= date(?)")
        params.append(date_from.strip())
    if date_to is not None and date_to.strip():
        where_parts.append("date(d.created_at) <= date(?)")
        params.append(date_to.strip())

    return " AND ".join(where_parts), params


@router.get("/api/admin/schedules/date-fields", response_model=list[ScheduleDateFieldItem])
def get_schedule_date_fields(
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, str]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)
    return list_schedule_date_fields()


@router.get("/api/admin/schedules/template-variables", response_model=list[TemplateVariableItem])
def get_template_variables(
    date_field_a: str = Query(..., alias="dateFieldA"),
    check_empty_field: str = Query(..., alias="checkEmptyField"),
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, str]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)
    selected_check_field = (check_empty_field or "").strip()
    if not selected_check_field:
        raise HTTPException(status_code=400, detail="checkEmptyField jest wymagane")

    try:
        return list_template_variables(date_field_a, selected_check_field)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/admin/schedules", response_model=list[ScheduleRead])
def list_schedules(
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)
        rows = conn.execute(
            """
                        SELECT id, name, module_type, date_field_a, date_field_b, send_hour, send_minute,
                     days_difference, subject_template, body_template,
                   target_inspection_leader, target_inspection_team,
                   fallback_recipient, enabled, created_by_user_id,
                   last_run_date, created_at, updated_at
            FROM notification_schedules
            ORDER BY id ASC
            """
        ).fetchall()
    return [_map_schedule(dict(row)) for row in rows]


@router.post("/api/admin/schedules", response_model=ScheduleRead, status_code=201)
def create_schedule(
    payload: ScheduleCreateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    if payload.sendHour < 0 or payload.sendHour > 23:
        raise HTTPException(status_code=400, detail="sendHour musi byc w zakresie 0..23")
    if payload.sendMinute < 0 or payload.sendMinute > 59:
        raise HTTPException(status_code=400, detail="sendMinute musi byc w zakresie 0..59")
    if payload.daysDifference < 0:
        raise HTTPException(status_code=400, detail="daysDifference musi byc >= 0")

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nazwa harmonogramu jest wymagana")
    subject = payload.subjectTemplate.strip()
    body = payload.bodyTemplate.strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subjectTemplate jest wymagany")
    if not body:
        raise HTTPException(status_code=400, detail="bodyTemplate jest wymagany")

    try:
        module_type = resolve_module_type(payload.dateFieldA, payload.checkEmptyField)
        recipient_strategy = resolve_recipient_strategy(payload.dateFieldA, payload.checkEmptyField)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if recipient_strategy == RECIPIENT_STRATEGY_INSPECTION_CONTEXT:
        if not payload.targetInspectionLeader and not payload.targetInspectionTeam:
            raise HTTPException(
                status_code=400,
                detail="Dla harmonogramu z data inspekcji wybierz: kierujacy inspekcja i/lub zespol inspekcji",
            )

    target_leader = 1 if payload.targetInspectionLeader else 0
    target_team = 1 if payload.targetInspectionTeam else 0
    if recipient_strategy == RECIPIENT_STRATEGY_AUTHOR_ONLY:
        target_leader = 0
        target_team = 0

    unknown_subject = validate_template_text(subject, payload.dateFieldA, payload.checkEmptyField)
    if unknown_subject:
        raise HTTPException(status_code=400, detail=f"Nieznane tokeny w subjectTemplate: {', '.join(sorted(unknown_subject))}")
    unknown_body = validate_template_text(body, payload.dateFieldA, payload.checkEmptyField)
    if unknown_body:
        raise HTTPException(status_code=400, detail=f"Nieznane tokeny w bodyTemplate: {', '.join(sorted(unknown_body))}")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        cursor = conn.execute(
            """
            INSERT INTO notification_schedules (
                name, module_type, date_field_a, date_field_b, days_difference,
                subject_template, body_template, send_hour, send_minute,
                target_inspection_leader, target_inspection_team,
                fallback_recipient, enabled, created_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                module_type,
                payload.dateFieldA,
                payload.checkEmptyField,
                int(payload.daysDifference),
                subject,
                body,
                int(payload.sendHour),
                int(payload.sendMinute),
                target_leader,
                target_team,
                payload.fallbackRecipient,
                1 if payload.enabled else 0,
                int(operator["id"]),
            ),
        )
        schedule_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO notification_schedule_rules (
                schedule_id, days_difference, subject_template, body_template, enabled
            ) VALUES (?, ?, ?, ?, 1)
            """,
            (schedule_id, int(payload.daysDifference), subject, body),
        )
        conn.commit()

        row = conn.execute(
            """
                        SELECT id, name, module_type, date_field_a, date_field_b, send_hour, send_minute,
                     days_difference, subject_template, body_template,
                   target_inspection_leader, target_inspection_team,
                   fallback_recipient, enabled, created_by_user_id,
                   last_run_date, created_at, updated_at
            FROM notification_schedules
            WHERE id = ?
            LIMIT 1
            """,
            (schedule_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac utworzonego harmonogramu")
    return _map_schedule(dict(row))


@router.put("/api/admin/schedules/{schedule_id}", response_model=ScheduleRead)
def update_schedule(
    schedule_id: int,
    payload: ScheduleUpdateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Brak pol do aktualizacji")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        current_row = conn.execute(
            """
                 SELECT id, name, module_type, date_field_a, date_field_b, send_hour, send_minute,
                   days_difference, subject_template, body_template,
                   target_inspection_leader, target_inspection_team,
                   fallback_recipient, enabled, created_by_user_id,
                   last_run_date, created_at, updated_at
            FROM notification_schedules
            WHERE id = ?
            LIMIT 1
            """,
            (int(schedule_id),),
        ).fetchone()
        if current_row is None:
            raise HTTPException(status_code=404, detail="Harmonogram nie istnieje")

        current = dict(current_row)
        next_name = current["name"]
        next_date_a = current["date_field_a"]
        next_date_b = current["date_field_b"]
        next_days_difference = int(current["days_difference"])
        next_subject = str(current["subject_template"])
        next_body = str(current["body_template"])
        next_send_hour = int(current["send_hour"])
        next_send_minute = int(current["send_minute"])
        next_target_leader = int(current["target_inspection_leader"])
        next_target_team = int(current["target_inspection_team"])
        next_fallback = current["fallback_recipient"]
        next_enabled = int(current["enabled"])

        if "name" in fields:
            next_name = str(fields["name"] or "").strip()
            if not next_name:
                raise HTTPException(status_code=400, detail="Nazwa harmonogramu jest wymagana")

        if "dateFieldA" in fields:
            next_date_a = str(fields["dateFieldA"])
        if "checkEmptyField" in fields:
            next_date_b = str(fields["checkEmptyField"])
        if "daysDifference" in fields:
            next_days_difference = int(fields["daysDifference"])
            if next_days_difference < 0:
                raise HTTPException(status_code=400, detail="daysDifference musi byc >= 0")
        if "subjectTemplate" in fields:
            next_subject = str(fields["subjectTemplate"] or "").strip()
            if not next_subject:
                raise HTTPException(status_code=400, detail="subjectTemplate jest wymagany")
        if "bodyTemplate" in fields:
            next_body = str(fields["bodyTemplate"] or "").strip()
            if not next_body:
                raise HTTPException(status_code=400, detail="bodyTemplate jest wymagany")

        try:
            next_module_type = resolve_module_type(str(next_date_a), str(next_date_b))
            recipient_strategy = resolve_recipient_strategy(str(next_date_a), str(next_date_b))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if "sendHour" in fields:
            next_send_hour = int(fields["sendHour"])
            if next_send_hour < 0 or next_send_hour > 23:
                raise HTTPException(status_code=400, detail="sendHour musi byc w zakresie 0..23")
        if "sendMinute" in fields:
            next_send_minute = int(fields["sendMinute"])
            if next_send_minute < 0 or next_send_minute > 59:
                raise HTTPException(status_code=400, detail="sendMinute musi byc w zakresie 0..59")

        if "targetInspectionLeader" in fields:
            next_target_leader = 1 if fields["targetInspectionLeader"] else 0
        if "targetInspectionTeam" in fields:
            next_target_team = 1 if fields["targetInspectionTeam"] else 0
        if "fallbackRecipient" in fields:
            next_fallback = fields["fallbackRecipient"]
        if "enabled" in fields:
            next_enabled = 1 if fields["enabled"] else 0

        if recipient_strategy == RECIPIENT_STRATEGY_INSPECTION_CONTEXT:
            if next_target_leader == 0 and next_target_team == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Dla harmonogramu z data inspekcji wybierz: kierujacy inspekcja i/lub zespol inspekcji",
                )
        if recipient_strategy == RECIPIENT_STRATEGY_AUTHOR_ONLY:
            next_target_leader = 0
            next_target_team = 0

        reset_last_run_date = (
            (
                str(current["date_field_a"]) != str(next_date_a)
                or str(current["date_field_b"]) != str(next_date_b)
                or int(current["days_difference"]) != int(next_days_difference)
                or int(current["send_hour"]) != int(next_send_hour)
                or int(current["send_minute"]) != int(next_send_minute)
                or int(current["target_inspection_leader"]) != int(next_target_leader)
                or int(current["target_inspection_team"]) != int(next_target_team)
                or str(current["fallback_recipient"]) != str(next_fallback)
                or int(current["enabled"]) != int(next_enabled)
            )
            and int(next_enabled) == 1
        )

        unknown_subject = validate_template_text(next_subject, str(next_date_a), str(next_date_b))
        if unknown_subject:
            raise HTTPException(status_code=400, detail=f"Nieznane tokeny w subjectTemplate: {', '.join(sorted(unknown_subject))}")
        unknown_body = validate_template_text(next_body, str(next_date_a), str(next_date_b))
        if unknown_body:
            raise HTTPException(status_code=400, detail=f"Nieznane tokeny w bodyTemplate: {', '.join(sorted(unknown_body))}")

        conn.execute(
            """
            UPDATE notification_schedules
            SET name = ?,
                module_type = ?,
                date_field_a = ?,
                date_field_b = ?,
                days_difference = ?,
                subject_template = ?,
                body_template = ?,
                send_hour = ?,
                send_minute = ?,
                target_inspection_leader = ?,
                target_inspection_team = ?,
                fallback_recipient = ?,
                enabled = ?,
                last_run_date = CASE WHEN ? = 1 THEN NULL ELSE last_run_date END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                next_name,
                next_module_type,
                next_date_a,
                next_date_b,
                next_days_difference,
                next_subject,
                next_body,
                next_send_hour,
                next_send_minute,
                next_target_leader,
                next_target_team,
                next_fallback,
                next_enabled,
                1 if reset_last_run_date else 0,
                int(schedule_id),
            ),
        )

        existing_rule = conn.execute(
            "SELECT id FROM notification_schedule_rules WHERE schedule_id = ? ORDER BY id ASC LIMIT 1",
            (int(schedule_id),),
        ).fetchone()
        if existing_rule is None:
            conn.execute(
                """
                INSERT INTO notification_schedule_rules (
                    schedule_id, days_difference, subject_template, body_template, enabled
                ) VALUES (?, ?, ?, ?, 1)
                """,
                (int(schedule_id), next_days_difference, next_subject, next_body),
            )
        else:
            conn.execute(
                """
                UPDATE notification_schedule_rules
                SET days_difference = ?,
                    subject_template = ?,
                    body_template = ?,
                    enabled = 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (next_days_difference, next_subject, next_body, int(existing_rule[0])),
            )
        conn.commit()

        row = conn.execute(
            """
                        SELECT id, name, module_type, date_field_a, date_field_b, send_hour, send_minute,
                     days_difference, subject_template, body_template,
                   target_inspection_leader, target_inspection_team,
                   fallback_recipient, enabled, created_by_user_id,
                   last_run_date, created_at, updated_at
            FROM notification_schedules
            WHERE id = ?
            LIMIT 1
            """,
            (int(schedule_id),),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac zaktualizowanego harmonogramu")
    return _map_schedule(dict(row))


@router.delete("/api/admin/schedules/{schedule_id}")
def delete_schedule(
    schedule_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, bool]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        row = conn.execute("SELECT id FROM notification_schedules WHERE id = ? LIMIT 1", (int(schedule_id),)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Harmonogram nie istnieje")

        conn.execute("DELETE FROM notification_schedules WHERE id = ?", (int(schedule_id),))
        conn.commit()

    return {"ok": True}


@router.get("/api/admin/schedules/{schedule_id}/rules", response_model=list[ScheduleRuleRead])
def list_schedule_rules(
    schedule_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    raise HTTPException(status_code=410, detail="Osobne reguly sa wylaczone: uzyj pol harmonogramu")


@router.post("/api/admin/schedules/{schedule_id}/rules", response_model=ScheduleRuleRead, status_code=201)
def create_schedule_rule(
    schedule_id: int,
    payload: ScheduleRuleCreateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    raise HTTPException(status_code=410, detail="Osobne reguly sa wylaczone: uzyj pol harmonogramu")


@router.put("/api/admin/schedules/{schedule_id}/rules/{rule_id}", response_model=ScheduleRuleRead)
def update_schedule_rule(
    schedule_id: int,
    rule_id: int,
    payload: ScheduleRuleUpdateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    raise HTTPException(status_code=410, detail="Osobne reguly sa wylaczone: uzyj pol harmonogramu")


@router.delete("/api/admin/schedules/{schedule_id}/rules/{rule_id}")
def delete_schedule_rule(
    schedule_id: int,
    rule_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, bool]:
    raise HTTPException(status_code=410, detail="Osobne reguly sa wylaczone: uzyj pol harmonogramu")


@router.get("/api/admin/schedules/{schedule_id}/dispatches", response_model=list[ScheduleDispatchRead])
def list_schedule_dispatches(
    schedule_id: int,
    limit: int = Query(default=100, ge=1, le=1000),
    period: Literal["week", "month", "year", "all"] = Query(default="all"),
    status: Literal["sent", "failed"] | None = Query(default=None),
    recipient_email: str | None = Query(default=None, alias="recipientEmail"),
    recipient_type: Literal["inspection_leader", "inspection_team", "author"] | None = Query(default=None, alias="recipientType"),
    date_from: str | None = Query(default=None, alias="dateFrom"),
    date_to: str | None = Query(default=None, alias="dateTo"),
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        exists = conn.execute("SELECT id FROM notification_schedules WHERE id = ? LIMIT 1", (int(schedule_id),)).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Harmonogram nie istnieje")

        where_sql, params = _build_dispatch_filter_sql(
            schedule_id,
            period,
            status,
            recipient_email,
            recipient_type,
            date_from,
            date_to,
        )

        params.append(int(limit))
        sql = f"""
            SELECT d.id, d.run_id, d.schedule_id, d.rule_id, d.module_type, d.record_id,
                   CASE
                       WHEN d.module_type = 'inspections' THEN i.kod_inspekcji
                       WHEN d.module_type = 'recommendations' THEN ir.kod_inspekcji
                       WHEN d.module_type = 'risk_exposure' THEN iw.kod_inspekcji
                       ELSE NULL
                   END AS inspection_code,
                   CASE WHEN d.module_type = 'recommendations' THEN r.kod_zalecenia ELSE NULL END AS recommendation_code,
                   CASE WHEN d.module_type = 'risk_exposure' THEN w.kod_sankcji ELSE NULL END AS sanction_request_code,
                   d.recipient_email,
                   COALESCE(
                       d.recipient_type,
                       CASE
                           WHEN lower(COALESCE(author_i.email, author_r.email, author_w.email, '')) = lower(d.recipient_email) THEN 'author'
                           WHEN lower(COALESCE(leader_u.email, '')) = lower(d.recipient_email)
                                AND EXISTS (
                                    SELECT 1
                                    FROM inspection_members im
                                    JOIN users um ON um.id = im.user_id
                                    WHERE im.inspection_id = dispatch_inspection.id
                                      AND lower(um.email) = lower(d.recipient_email)
                                ) THEN 'inspection_leader_team'
                           WHEN lower(COALESCE(leader_u.email, '')) = lower(d.recipient_email) THEN 'inspection_leader'
                           WHEN EXISTS (
                               SELECT 1
                               FROM inspection_members im
                               JOIN users um ON um.id = im.user_id
                               WHERE im.inspection_id = dispatch_inspection.id
                                 AND lower(um.email) = lower(d.recipient_email)
                           ) THEN 'inspection_team'
                           ELSE CASE
                               WHEN s.target_inspection_leader = 1 AND s.target_inspection_team = 1 THEN 'inspection_leader_team'
                               WHEN s.target_inspection_leader = 1 THEN 'inspection_leader'
                               WHEN s.target_inspection_team = 1 THEN 'inspection_team'
                               ELSE 'author'
                           END
                       END
                   ) AS effective_recipient_type,
                   d.status, d.error_message,
                   d.rendered_subject, d.rendered_body, d.created_at
            FROM notification_schedule_dispatches d
            JOIN notification_schedules s ON s.id = d.schedule_id
            LEFT JOIN inspections i ON d.module_type = 'inspections' AND i.id = d.record_id
            LEFT JOIN recommendations r ON d.module_type = 'recommendations' AND r.id = d.record_id
            LEFT JOIN inspections ir ON d.module_type = 'recommendations' AND ir.id = r.inspection_id
            LEFT JOIN risk_exposure_requests w ON d.module_type = 'risk_exposure' AND w.id = d.record_id
            LEFT JOIN inspections iw ON d.module_type = 'risk_exposure' AND iw.id = w.inspection_id
            LEFT JOIN inspections dispatch_inspection ON dispatch_inspection.id =
                CASE
                    WHEN d.module_type = 'inspections' THEN d.record_id
                    WHEN d.module_type = 'recommendations' THEN r.inspection_id
                    WHEN d.module_type = 'risk_exposure' THEN w.inspection_id
                    ELSE NULL
                END
            LEFT JOIN users leader_u ON leader_u.id = dispatch_inspection.osoba_kierujaca_user_id
            LEFT JOIN users author_i ON d.module_type = 'inspections' AND author_i.id = i.created_by_user_id
            LEFT JOIN users author_r ON d.module_type = 'recommendations' AND author_r.id = r.created_by_user_id
            LEFT JOIN users author_w ON d.module_type = 'risk_exposure' AND author_w.id = w.utworzono_przez_user_id
            WHERE {where_sql}
            ORDER BY d.id DESC
            LIMIT ?
        """
        rows = conn.execute(sql, tuple(params)).fetchall()

    return [
        {
            "id": int(row["id"]),
            "runId": row["run_id"],
            "scheduleId": int(row["schedule_id"]),
            "ruleId": int(row["rule_id"]),
            "moduleType": str(row["module_type"]),
            "recordId": int(row["record_id"]),
            "inspectionId": row["inspection_code"],
            "recommendationId": row["recommendation_code"],
            "sanctionRequestId": row["sanction_request_code"],
            "recipientEmail": str(row["recipient_email"]),
            "recipientType": _map_recipient_type_label(row["effective_recipient_type"]),
            "status": str(row["status"]),
            "errorMessage": row["error_message"],
            "renderedSubject": row["rendered_subject"],
            "renderedBody": row["rendered_body"],
            "createdAt": _dispatch_created_at_local(row["created_at"]),
        }
        for row in rows
    ]


@router.get("/api/admin/schedules/{schedule_id}/dispatches/kpi", response_model=ScheduleDispatchKpiResponse)
def get_schedule_dispatches_kpi(
    schedule_id: int,
    period: Literal["week", "month", "year", "all"] = Query(default="all"),
    status: Literal["sent", "failed"] | None = Query(default=None),
    recipient_email: str | None = Query(default=None, alias="recipientEmail"),
    recipient_type: Literal["inspection_leader", "inspection_team", "author"] | None = Query(default=None, alias="recipientType"),
    date_from: str | None = Query(default=None, alias="dateFrom"),
    date_to: str | None = Query(default=None, alias="dateTo"),
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        exists = conn.execute("SELECT id FROM notification_schedules WHERE id = ? LIMIT 1", (int(schedule_id),)).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Harmonogram nie istnieje")

        where_sql, params = _build_dispatch_filter_sql(
            schedule_id,
            period,
            status,
            recipient_email,
            recipient_type,
            date_from,
            date_to,
        )

        summary = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN d.status = 'sent' THEN 1 ELSE 0 END) AS sent,
                SUM(CASE WHEN d.status = 'failed' THEN 1 ELSE 0 END) AS failed
            FROM notification_schedule_dispatches d
            WHERE {where_sql}
            """,
            tuple(params),
        ).fetchone()

        by_module = conn.execute(
            f"""
            SELECT d.module_type AS moduleType, COUNT(*) AS total
            FROM notification_schedule_dispatches d
            WHERE {where_sql}
            GROUP BY d.module_type
            ORDER BY total DESC
            """,
            tuple(params),
        ).fetchall()

        by_recipient_type = conn.execute(
            f"""
            SELECT COALESCE(d.recipient_type, 'unknown') AS recipientType, COUNT(*) AS total
            FROM notification_schedule_dispatches d
            WHERE {where_sql}
            GROUP BY COALESCE(d.recipient_type, 'unknown')
            ORDER BY total DESC
            """,
            tuple(params),
        ).fetchall()

    total = int(summary["total"] or 0)
    sent_count = int(summary["sent"] or 0)
    failed_count = int(summary["failed"] or 0)
    success_rate = round((sent_count / total) * 100.0, 2) if total > 0 else 0.0

    return {
        "scheduleId": int(schedule_id),
        "total": total,
        "sent": sent_count,
        "failed": failed_count,
        "successRate": success_rate,
        "byModule": [{"moduleType": str(row["moduleType"]), "total": int(row["total"])} for row in by_module],
        "byRecipientType": [
            {"recipientType": str(row["recipientType"]), "total": int(row["total"])} for row in by_recipient_type
        ],
    }


@router.post("/api/admin/schedules/run-now", response_model=ScheduleRunNowResponse)
def run_schedules_now(
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

    result = run_due_schedules_once(trigger_type="manual")
    return {
        "ok": True,
        "matched": int(result.get("matched", 0)),
        "sent": int(result.get("sent", 0)),
        "runAt": datetime.now().isoformat(timespec="seconds"),
    }
