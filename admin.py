from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import os
from pathlib import Path
import secrets
import smtplib
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.auth_security import hash_password, hash_token, revoke_user_sessions
from app.db import get_connection
from app.permissions import (
    PERMISSION_CATALOG,
    ROLE_EXTERNAL_USER,
    get_user_permissions,
    normalize_permission_codes,
    set_user_permissions,
)


ROLE_ID_TO_NAME = {
    1: "inspector",
    2: "team_lead",
    3: "director",
    4: "external_user",
}

ACCOUNT_TYPE_DIU = "diu"
ACCOUNT_TYPE_OBSERVER = "observer"
ACCOUNT_TYPE_TECHNICAL = "technical"
LIST_VISIBILITY_VISIBLE = "visible"
LIST_VISIBILITY_HIDDEN = "hidden"

TEAM_LEAD_MANAGEABLE_PERMISSION_CODES: tuple[str, ...] = (
    "registry.inspections.read",
    "registry.recommendations.read",
    "registry.obligating_decisions.read",
    "registry.risk_exposure.read",
    "reports.executed_inspections.read",
)

router = APIRouter()


def _audit(event: str, **fields: str) -> None:
    pairs = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[AUDIT] event={event} {pairs}".strip())


class TeamRead(BaseModel):
    id: int
    kod: str
    nazwa: str
    kierownikUserId: int | None = None


class TeamCreateRequest(BaseModel):
    kod: str
    nazwa: str
    kierownikUserId: int | None = None


class TeamUpdateRequest(BaseModel):
    kod: str | None = None
    nazwa: str | None = None
    kierownikUserId: int | None = None


class UserRead(BaseModel):
    id: int
    login: str
    imie: str
    nazwisko: str
    email: str | None = None
    rolaId: int
    rola: str
    zespolId: int | None = None
    zespolSkroconaNazwa: str | None = None
    zespolPelnaNazwa: str | None = None
    aktywny: bool
    accountType: Literal["diu", "observer", "technical"]
    departmentCode: str | None = None
    departmentLabel: str | None = None
    createdByLogin: str | None = None
    createdByOperator: bool | None = None
    listVisibility: Literal["visible", "hidden"]
    profileChangedAt: str | None = None
    profileChangedBy: dict[str, Any] | None = None


class UserCreateRequest(BaseModel):
    login: str
    imie: str
    nazwisko: str
    email: str | None = None
    password: str | None = None
    rolaId: int
    zespolId: int | None = None
    aktywny: bool = True
    accountType: Literal["diu", "observer", "technical"] | None = None
    departmentCode: str | None = None
    listVisibility: Literal["visible", "hidden"] | None = None


class UserUpdateRequest(BaseModel):
    login: str | None = None
    imie: str | None = None
    nazwisko: str | None = None
    email: str | None = None
    rolaId: int | None = None
    zespolId: int | None = None
    aktywny: bool | None = None
    accountType: Literal["diu", "observer", "technical"] | None = None
    departmentCode: str | None = None
    listVisibility: Literal["visible", "hidden"] | None = None


class UserProfileHistoryRead(BaseModel):
    id: int
    validFrom: str
    validTo: str | None = None
    login: str | None = None
    imie: str | None = None
    nazwisko: str | None = None
    email: str | None = None
    rolaId: int | None = None
    aktywny: bool | None = None
    accountType: Literal["diu", "observer", "technical"]
    zespolId: int | None = None
    departmentCode: str | None = None
    departmentLabel: str | None = None
    listVisibility: Literal["visible", "hidden"]
    permissions: list[str] = []
    changedBy: dict[str, Any] | None = None
    changedAt: str


class UserProfileHistoryEventChangeRead(BaseModel):
    field: str
    label: str
    before: str | None = None
    after: str | None = None


class UserProfileHistoryEventRead(BaseModel):
    eventId: int
    changedAt: str
    changedBy: str | None = None
    changes: list[UserProfileHistoryEventChangeRead]


class UserPasswordUpdateRequest(BaseModel):
    password: str


class UserPermissionsRead(BaseModel):
    userId: int
    permissions: list[str]


class UserPermissionsUpdateRequest(BaseModel):
    permissions: list[str]


class PermissionCatalogItem(BaseModel):
    code: str
    label: str
    group: str


class ExternalUserCreateRequest(BaseModel):
    login: str
    imie: str
    nazwisko: str
    email: str
    permissions: list[str]
    aktywny: bool = True
    frontendInviteUrl: str | None = None


class ExternalUserCreateResponse(BaseModel):
    user: UserRead
    invite: SendUserInviteResponse


class SendUserInviteRequest(BaseModel):
    frontendInviteUrl: str | None = None


class SendUserInviteResponse(BaseModel):
    ok: bool
    expiresAt: str
    delivery: str
    inviteLink: str | None = None


def _invite_ttl_hours() -> int:
    raw = (os.getenv("AUTH_INVITE_TTL_HOURS") or "24").strip()
    try:
        parsed = int(raw)
    except ValueError:
        return 24
    return parsed if parsed > 0 else 24


def _build_invite_link(token: str, frontend_invite_url: str | None) -> str:
    base = (frontend_invite_url or os.getenv("FRONTEND_INVITE_URL") or "http://localhost:3002/set-password").strip()
    if not base:
        base = "http://localhost:3002/set-password"

    parts = urlsplit(base)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["token"] = token
    updated_query = urlencode(query)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, updated_query, parts.fragment))


def _create_and_send_invite_for_user(
    conn: Any,
    *,
    user_id: int,
    login: str,
    email: str,
    operator_user_id: int,
    frontend_invite_url: str | None,
) -> tuple[str, str, str]:
    token = secrets.token_urlsafe(48)
    token_hash = hash_token(token)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=_invite_ttl_hours())).isoformat(timespec="seconds")

    conn.execute(
        """
        UPDATE user_invites
        SET used_at = CURRENT_TIMESTAMP
        WHERE user_id = ? AND used_at IS NULL
        """,
        (int(user_id),),
    )
    conn.execute(
        """
        INSERT INTO user_invites (user_id, token_hash, expires_at, created_by_user_id)
        VALUES (?, ?, ?, ?)
        """,
        (int(user_id), token_hash, expires_at, int(operator_user_id)),
    )

    invite_link = _build_invite_link(token, frontend_invite_url)
    delivery = _send_invite_email(email, login, invite_link, expires_at)
    return delivery, expires_at, invite_link


def _send_invite_email(to_email: str, login: str, invite_link: str, expires_at: str) -> str:
    mode = (os.getenv("INVITE_EMAIL_MODE") or "log").strip().lower()
    if mode not in {"log", "smtp"}:
        mode = "log"

    if mode == "log":
        print(
            "[INVITE-EMAIL]",
            f"to={to_email}",
            f"login={login}",
            f"expires_at={expires_at}",
            f"link={invite_link}",
        )
        return "log"

    smtp_host = (os.getenv("SMTP_HOST") or "").strip()
    smtp_port_raw = (os.getenv("SMTP_PORT") or "465").strip()
    smtp_login = (os.getenv("SMTP_LOGIN") or "").strip()
    smtp_pass = os.getenv("SMTP_PASS") or ""
    if not smtp_pass:
        secret_file_raw = (os.getenv("SMTP_PASS_FILE") or "").strip()
        if secret_file_raw:
            try:
                smtp_pass = Path(secret_file_raw).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise HTTPException(status_code=500, detail="Nie mozna odczytac SMTP_PASS_FILE") from exc
    smtp_login_mail = (os.getenv("SMTP_LOGIN_MAIL") or "").strip()
    from_email = (os.getenv("FROM_EMAIL") or smtp_login_mail or "").strip()
    smtp_security = (os.getenv("SMTP_SECURITY") or "starttls").strip().lower()

    if not smtp_host or not smtp_login or not smtp_pass or not from_email:
        raise HTTPException(status_code=500, detail="Brak konfiguracji SMTP do wysylki zaproszen")

    try:
        smtp_port = int(smtp_port_raw)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Niepoprawna konfiguracja SMTP_PORT") from exc

    if smtp_security not in {"starttls", "ssl", "plain"}:
        raise HTTPException(status_code=500, detail="Niepoprawna konfiguracja SMTP_SECURITY")

    base_dir = Path(__file__).resolve().parents[2]
    template_path = Path(
        (os.getenv("INVITE_EMAIL_TEMPLATE_FILE") or str(base_dir / "config" / "invite_email_template.txt")).strip()
    )
    if template_path.exists() and template_path.is_file():
        template_text = template_path.read_text(encoding="utf-8")
    else:
        template_text = "\n".join(
            [
                "Witaj {{LOGIN}},",
                "",
                "Twoje konto zostalo utworzone. Ustaw haslo przez ponizszy link:",
                "{{INVITE_LINK}}",
                "",
                "Link wygasa: {{EXPIRES_AT}}",
            ]
        )

    body = (
        template_text.replace("{{LOGIN}}", login)
        .replace("{{INVITE_LINK}}", invite_link)
        .replace("{{EXPIRES_AT}}", expires_at)
    )

    msg = EmailMessage()
    msg["Subject"] = (os.getenv("INVITE_EMAIL_SUBJECT") or "Zaproszenie do ustawienia hasla").strip()
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    try:
        if smtp_security == "ssl":
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
                server.login(smtp_login, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                if smtp_security == "starttls":
                    server.starttls()
                    server.ehlo()
                server.login(smtp_login, smtp_pass)
                server.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        raise HTTPException(status_code=502, detail="Nie udalo sie wyslac zaproszenia email") from exc

    return "smtp"


def _resolve_operator(
    conn: Any,
    x_operator_login: str,
) -> dict[str, Any]:
    login = x_operator_login.strip()
    if not login:
        raise HTTPException(status_code=401, detail="Operator nie istnieje")

    operator = conn.execute(
        """
        SELECT id, login, rola_id, zespol_id, aktywny
        FROM users
        WHERE lower(login) = lower(?)
        LIMIT 1
        """,
        (login,),
    ).fetchone()

    if operator is None:
        raise HTTPException(status_code=401, detail="Operator nie istnieje")

    operator_data = dict(operator)
    if int(operator_data["aktywny"]) != 1:
        raise HTTPException(status_code=403, detail="Operator jest nieaktywny")

    return operator_data


def _map_user_row(row: dict[str, Any]) -> dict[str, Any]:
    account_type = str(row.get("account_type") or ACCOUNT_TYPE_DIU)
    exposed_login = str(row["login"] or "")
    if not bool(row["aktywny"]) or account_type == ACCOUNT_TYPE_TECHNICAL:
        exposed_login = ""

    normalized_department_code = str(row.get("department_short") or "").strip() or None
    if normalized_department_code is None and row.get("department_label") is not None:
        normalized_department_code = str(row.get("department_label") or "").strip() or None
    if normalized_department_code is None:
        normalized_department_code = str(row.get("department_code") or "").strip() or None

    normalized_team_id = row["zespol_id"] if account_type == ACCOUNT_TYPE_DIU else None
    normalized_team_short = (
        row["zespol_skrot_slownik"] or row["zespol_kod_slownik"] or row["zespol_kod_baza"]
        if account_type == ACCOUNT_TYPE_DIU
        else None
    )
    normalized_team_full = (row["zespol_nazwa_slownik"] or row["zespol_nazwa_baza"]) if account_type == ACCOUNT_TYPE_DIU else None

    profile_changed_by = None
    if row.get("profile_changed_by_user_id") is not None:
        profile_changed_by = {
            "userId": int(row["profile_changed_by_user_id"]),
            "login": row.get("profile_changed_by_login"),
        }
    return {
        "id": row["id"],
        "login": exposed_login,
        "imie": row["imie"],
        "nazwisko": row["nazwisko"],
        "email": row["email"],
        "rolaId": int(row["rola_id"]),
        "rola": ROLE_ID_TO_NAME.get(int(row["rola_id"]), "unknown"),
        "zespolId": normalized_team_id,
        "zespolSkroconaNazwa": normalized_team_short,
        "zespolPelnaNazwa": normalized_team_full,
        "aktywny": bool(row["aktywny"]),
        "accountType": account_type,
        "departmentCode": normalized_department_code,
        "departmentLabel": row.get("department_label"),
        "createdByLogin": row.get("created_by_login"),
        "createdByOperator": row.get("created_by_operator"),
        "listVisibility": str(row["list_visibility"] or LIST_VISIBILITY_VISIBLE),
        "profileChangedAt": row.get("profile_changed_at"),
        "profileChangedBy": profile_changed_by,
    }


def _user_select_sql(where_clause: str = "") -> str:
    return f"""
        SELECT
            u.id,
            u.login,
            u.imie,
            u.nazwisko,
            u.email,
            u.rola_id,
            u.zespol_id,
            u.aktywny,
            u.account_type,
            u.department_code,
            u.created_by_user_id,
            u.list_visibility,
            u.profile_changed_at,
            u.profile_changed_by_user_id,
            pbu.login AS profile_changed_by_login,
            cbu.login AS created_by_login,
            dep.skrot_pozycji AS department_short,
            dep.nazwa_pozycji AS department_label,
            t.kod AS zespol_kod_baza,
            t.nazwa AS zespol_nazwa_baza,
            sp.kod_pozycji AS zespol_kod_slownik,
            sp.skrot_pozycji AS zespol_skrot_slownik,
            sp.nazwa_pozycji AS zespol_nazwa_slownik
        FROM users u
        LEFT JOIN teams t ON t.id = u.zespol_id
        LEFT JOIN slownik_pozycje sp
            ON sp.id = t.slownik_pozycja_id
           AND sp.kod_typu = 'zespoly'
        LEFT JOIN slownik_pozycje dep
            ON lower(dep.kod_typu) = 'department_ogolne'
           AND lower(dep.kod_pozycji) = lower(COALESCE(u.department_code, ''))
        LEFT JOIN users pbu ON pbu.id = u.profile_changed_by_user_id
        LEFT JOIN users cbu ON cbu.id = u.created_by_user_id
        {where_clause}
    """


def _resolve_department_entry(conn: Any, department_value: str | None) -> tuple[str, str] | None:
    value = str(department_value or "").strip()
    if not value:
        return None

    # First try canonical dictionary code.
    row = conn.execute(
        """
        SELECT kod_pozycji, nazwa_pozycji
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'department_ogolne'
          AND lower(kod_pozycji) = lower(?)
        LIMIT 1
        """,
        (value,),
    ).fetchone()
    if row is not None:
        return str(row["kod_pozycji"]), str(row["nazwa_pozycji"])

    # Backward-compatible fallback for clients sending short label values.
    row = conn.execute(
        """
        SELECT kod_pozycji, nazwa_pozycji
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'department_ogolne'
          AND lower(COALESCE(skrot_pozycji, '')) = lower(?)
        ORDER BY id ASC
        LIMIT 1
        """,
        (value,),
    ).fetchone()
    if row is None:
        return None
    return str(row["kod_pozycji"]), str(row["nazwa_pozycji"])


def _resolve_department_label(conn: Any, department_code: str | None) -> str | None:
    resolved = _resolve_department_entry(conn, department_code)
    if resolved is None:
        return None
    _, label = resolved
    return label


def _require_diu_department(conn: Any) -> tuple[str, str]:
    configured_diu_code = (os.getenv("DIU_DEPARTMENT_CODE") or "DIU").strip()
    if not configured_diu_code:
        configured_diu_code = "DIU"

    row = conn.execute(
        """
        SELECT kod_pozycji, nazwa_pozycji
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'department_ogolne'
          AND lower(kod_pozycji) = lower(?)
        LIMIT 1
        """
        ,
        (configured_diu_code,),
    ).fetchone()
    if row is None:
        # Backward-compatible fallback: infer DIU row by shortcut if explicit code is not configured/present.
        row = conn.execute(
            """
            SELECT kod_pozycji, nazwa_pozycji
            FROM slownik_pozycje
            WHERE lower(kod_typu) = 'department_ogolne'
              AND (
                    upper(skrot_pozycji) = 'DIU'
                    OR upper(kod_pozycji) = 'DIU'
                  )
            ORDER BY CASE WHEN upper(skrot_pozycji) = 'DIU' THEN 0 ELSE 1 END, id ASC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=422,
            detail=f"Brak pozycji {configured_diu_code} w slowniku department_ogolne",
        )
    return str(row["kod_pozycji"]), str(row["nazwa_pozycji"])


def _validate_profile_state(
    conn: Any,
    *,
    account_type: str,
    department_code: str | None,
    email: str | None,
    list_visibility: str,
    aktywny: int,
    rola_id: int,
    zespol_id: int | None,
    previous_account_type: str,
) -> tuple[str, str | None, str | None, str, int | None]:
    normalized_account_type = str(account_type or "").strip().lower()
    if normalized_account_type not in {ACCOUNT_TYPE_DIU, ACCOUNT_TYPE_OBSERVER, ACCOUNT_TYPE_TECHNICAL}:
        raise HTTPException(status_code=400, detail="Niepoprawny accountType")

    normalized_visibility = str(list_visibility or "").strip().lower()
    if normalized_visibility not in {LIST_VISIBILITY_VISIBLE, LIST_VISIBILITY_HIDDEN}:
        raise HTTPException(status_code=400, detail="Niepoprawny listVisibility")

    normalized_department_code = str(department_code or "").strip() or None
    normalized_email = str(email or "").strip() or None
    normalized_team_id = zespol_id

    if int(aktywny) != 1 and normalized_email is not None:
        raise HTTPException(status_code=422, detail="Dla konta nieaktywnego email musi byc null")

    if normalized_account_type == ACCOUNT_TYPE_DIU:
        diu_code, diu_label = _require_diu_department(conn)
        if normalized_department_code is None:
            normalized_department_code = diu_code
        normalized_input = str(normalized_department_code).strip().lower()
        accepted_values = {str(diu_code).strip().lower(), "diu"}
        if normalized_input not in accepted_values:
            raise HTTPException(status_code=422, detail=f"Dla accountType=diu departmentCode musi byc {diu_code}")
        normalized_department_code = diu_code
        if rola_id == 3:
            normalized_team_id = None
        elif rola_id in (1, 2) and normalized_team_id is None:
            raise HTTPException(status_code=422, detail="Dla tej roli zespolId jest wymagane")
        return normalized_account_type, normalized_department_code, diu_label, normalized_visibility, normalized_team_id

    if normalized_account_type == ACCOUNT_TYPE_OBSERVER and int(aktywny) != 1:
        raise HTTPException(status_code=422, detail="Observer nie moze byc nieaktywny")

    if normalized_account_type == ACCOUNT_TYPE_TECHNICAL:
        if int(aktywny) != 0:
            raise HTTPException(status_code=422, detail="Dla accountType=technical aktywny musi byc false")
        if normalized_email is not None:
            raise HTTPException(status_code=422, detail="Dla accountType=technical email musi byc null")

    if int(aktywny) != 1 and previous_account_type == ACCOUNT_TYPE_OBSERVER and normalized_account_type == ACCOUNT_TYPE_OBSERVER:
        raise HTTPException(
            status_code=422,
            detail="Nie mozna dezaktywowac observer bez jednoczesnej zmiany accountType na diu lub technical",
        )

    normalized_team_id = None
    department_label = None
    if normalized_department_code is not None:
        resolved_department = _resolve_department_entry(conn, normalized_department_code)
        if resolved_department is None:
            raise HTTPException(status_code=422, detail="Nie znaleziono departmentCode w slowniku department_ogolne")
        normalized_department_code, department_label = resolved_department
    if normalized_department_code is not None and department_label is None:
        raise HTTPException(status_code=422, detail="Nie znaleziono departmentCode w slowniku department_ogolne")

    return normalized_account_type, normalized_department_code, department_label, normalized_visibility, normalized_team_id



def _write_profile_history_version(
    conn: Any,
    *,
    user_id: int,
    login: str | None,
    imie: str | None,
    nazwisko: str | None,
    email: str | None,
    rola_id: int | None,
    aktywny: int,
    account_type: str,
    zespol_id: int | None,
    department_code: str | None,
    department_label: str | None,
    list_visibility: str,
    permissions_codes: list[str] | None,
    changed_by_user_id: int,
    changed_by_login: str,
    changed_at: str,
) -> None:
    conn.execute(
        """
        UPDATE user_profile_history
        SET valid_to = ?
        WHERE user_id = ?
          AND valid_to IS NULL
        """,
        (changed_at, int(user_id)),
    )

    conn.execute(
        """
        INSERT INTO user_profile_history (
            user_id,
            valid_from,
            valid_to,
            login,
            imie,
            nazwisko,
            email,
            rola_id,
            aktywny,
            account_type,
            zespol_id,
            department_code,
            department_label,
            list_visibility,
            permissions_codes,
            changed_by_user_id,
            changed_by_login,
            changed_at
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            changed_at,
            login,
            imie,
            nazwisko,
            email,
            int(rola_id) if rola_id is not None else None,
            int(aktywny),
            account_type,
            zespol_id,
            department_code,
            department_label,
            list_visibility,
            ";".join(sorted(set(permissions_codes or []))) or None,
            int(changed_by_user_id),
            changed_by_login,
            changed_at,
        ),
    )


def _append_current_profile_history(
    conn: Any,
    *,
    user_id: int,
    changed_by_user_id: int,
    changed_by_login: str,
) -> None:
    row = conn.execute(
        """
        SELECT login, imie, nazwisko, email, rola_id, aktywny,
               account_type, zespol_id, department_code, list_visibility
        FROM users
        WHERE id = ?
        LIMIT 1
        """,
        (int(user_id),),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Uzytkownik nie istnieje")

    account_type = str(row["account_type"] or ACCOUNT_TYPE_DIU)
    zespol_id = row["zespol_id"]
    department_code = row["department_code"]
    department_label = _resolve_department_label(conn, department_code)
    list_visibility = str(row["list_visibility"] or LIST_VISIBILITY_VISIBLE)
    changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    permissions_codes = get_user_permissions(conn, int(user_id))

    _write_profile_history_version(
        conn,
        user_id=int(user_id),
        login=str(row["login"] or "").strip() or None,
        imie=str(row["imie"] or "").strip() or None,
        nazwisko=str(row["nazwisko"] or "").strip() or None,
        email=str(row["email"] or "").strip() or None,
        rola_id=int(row["rola_id"]) if row["rola_id"] is not None else None,
        aktywny=int(row["aktywny"]),
        account_type=account_type,
        zespol_id=zespol_id,
        department_code=department_code,
        department_label=department_label,
        list_visibility=list_visibility,
        permissions_codes=permissions_codes,
        changed_by_user_id=int(changed_by_user_id),
        changed_by_login=changed_by_login,
        changed_at=changed_at,
    )


def _authorize_profile_history_access(conn: Any, user_id: int, operator: dict[str, Any]) -> dict[str, Any]:
    operator_role_id = int(operator["rola_id"])

    target_row = conn.execute(
        "SELECT id, zespol_id, created_by_user_id FROM users WHERE id = ? LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if target_row is None:
        raise HTTPException(status_code=404, detail="Uzytkownik nie istnieje")
    target = dict(target_row)

    if operator_role_id == 3:
        return target
    if operator_role_id == 2:
        team_ids = _resolve_operator_team_ids(conn, operator)
        if not _can_team_lead_manage_user(operator=operator, team_ids=team_ids, target=target):
            raise HTTPException(status_code=403, detail="Kierownik moze podgladac historie tylko swojego zespolu")
        return target

    raise HTTPException(status_code=403, detail="Brak uprawnien")


def _profile_history_field_label(field: str) -> str:
    labels = {
        "login": "Login",
        "imie": "Imie",
        "nazwisko": "Nazwisko",
        "email": "E-mail",
        "rolaId": "Rola",
        "aktywny": "Dostep do konta",
        "accountType": "Typ konta",
        "zespolId": "Zespol",
        "departmentCode": "Departament",
        "listVisibility": "Widocznosc",
        "permissions": "Widocznosc modulow",
    }
    return labels.get(field, field)


def _serialize_profile_history_value(field: str, value: Any, team_names: dict[int, str]) -> str | None:
    if value is None:
        return None
    if field == "zespolId":
        team_id = int(value)
        team_name = team_names.get(team_id)
        return f"{team_name} ({team_id})" if team_name else str(team_id)
    if field == "aktywny":
        return "Aktywny" if int(value) == 1 else "Nieaktywny"
    if field == "permissions":
        if isinstance(value, list):
            return ", ".join(value)
        return str(value)
    return str(value)

    conn.execute(
        """
        UPDATE users
        SET profile_changed_at = ?,
            profile_changed_by_user_id = ?
        WHERE id = ?
        """,
        (changed_at, int(changed_by_user_id), int(user_id)),
    )


def _prepare_login_for_persistence(
    conn: Any,
    *,
    requested_login: str | None,
    current_login: str | None,
    user_id: int | None,
    account_type: str,
    aktywny: int,
) -> str:
    normalized = str(requested_login or "").strip().lower()
    allow_blank_login = int(aktywny) != 1 or account_type == ACCOUNT_TYPE_TECHNICAL

    if allow_blank_login:
        if normalized:
            raise HTTPException(status_code=422, detail="Dla konta nieaktywnego lub accountType=technical login musi byc pusty")

        # Always rotate to an internal placeholder login for inactive/technical
        # accounts so previous active login is not retained.
        prefix = "technical" if account_type == ACCOUNT_TYPE_TECHNICAL else "inactive"
        while True:
            suffix = f"{str(user_id)}_{secrets.token_hex(4)}" if user_id is not None else secrets.token_hex(8)
            candidate = f"__{prefix}_{suffix}".lower()
            exists = conn.execute(
                "SELECT id FROM users WHERE lower(login)=lower(?) LIMIT 1",
                (candidate,),
            ).fetchone()
            if exists is None:
                return candidate

    if not normalized or " " in normalized:
        raise HTTPException(status_code=422, detail="Dla aktywnego konta login jest wymagany i nie moze zawierac spacji")

    if user_id is None:
        conflict = conn.execute(
            "SELECT id FROM users WHERE lower(login)=lower(?) LIMIT 1",
            (normalized,),
        ).fetchone()
    else:
        conflict = conn.execute(
            "SELECT id FROM users WHERE lower(login)=lower(?) AND id <> ? LIMIT 1",
            (normalized, int(user_id)),
        ).fetchone()
    if conflict is not None:
        raise HTTPException(status_code=409, detail="Login juz istnieje")

    return normalized

    conn.execute(
        """
        INSERT INTO user_profile_history (
            user_id,
            valid_from,
            valid_to,
            account_type,
            department_code,
            department_label,
            list_visibility,
            changed_by_user_id,
            changed_by_login,
            changed_at
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id),
            changed_at,
            account_type,
            department_code,
            department_label,
            list_visibility,
            int(changed_by_user_id),
            changed_by_login,
            changed_at,
        ),
    )


def _ensure_director(operator: dict[str, Any]) -> None:
    if int(operator["rola_id"]) != 3:
        raise HTTPException(status_code=403, detail="Brak uprawnien")


def _resolve_operator_team_ids(conn: Any, operator: dict[str, Any]) -> set[int]:
    team_ids: set[int] = set()
    if operator.get("zespol_id") is not None:
        team_ids.add(int(operator["zespol_id"]))

    managed_rows = conn.execute(
        "SELECT id FROM teams WHERE kierownik_user_id = ?",
        (int(operator["id"]),),
    ).fetchall()
    team_ids.update(int(row["id"]) for row in managed_rows)
    return team_ids


def _can_team_lead_manage_user(*, operator: dict[str, Any], team_ids: set[int], target: dict[str, Any]) -> bool:
    target_team_id = target.get("zespol_id")
    if target_team_id is not None and int(target_team_id) in team_ids:
        return True
    created_by = target.get("created_by_user_id")
    return created_by is not None and int(created_by) == int(operator["id"])


def _resolve_operator_for_permissions(conn: Any, x_operator_login: str) -> dict[str, Any]:
    login = x_operator_login.strip()
    if not login:
        raise HTTPException(status_code=401, detail="Sesja wygasla lub operator nie istnieje. Wymagany re-login")

    row = conn.execute(
        """
        SELECT id, login, rola_id, zespol_id, aktywny
        FROM users
        WHERE lower(login) = lower(?)
        LIMIT 1
        """,
        (login,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Sesja wygasla lub operator nie istnieje. Wymagany re-login")

    operator = dict(row)
    if int(operator["aktywny"]) != 1:
        raise HTTPException(status_code=403, detail="Operator jest nieaktywny. Wymagany re-login")

    return {
        "id": int(operator["id"]),
        "login": operator["login"],
        "rola_id": int(operator["rola_id"]),
        "zespol_id": operator["zespol_id"],
    }


def _supports_module_permissions(*, role_id: int, account_type: str | None) -> bool:
    normalized_account_type = str(account_type or "").strip().lower()
    return int(role_id) == ROLE_EXTERNAL_USER or normalized_account_type == ACCOUNT_TYPE_OBSERVER


def _ensure_permissions_access_to_target(conn: Any, operator: dict[str, Any], user_id: int) -> tuple[dict[str, Any], bool]:
    row = conn.execute(
        "SELECT id, rola_id, account_type, zespol_id, created_by_user_id FROM users WHERE id = ? LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Uzytkownik nie istnieje")

    target = dict(row)
    operator_role_id = int(operator["rola_id"])
    is_team_lead = operator_role_id == 2

    if operator_role_id == 3:
        return target, False
    if is_team_lead:
        team_ids = _resolve_operator_team_ids(conn, operator)
        if not _can_team_lead_manage_user(operator=operator, team_ids=team_ids, target=target):
            raise HTTPException(status_code=403, detail="Brak uprawnien")
        return target, True

    raise HTTPException(status_code=403, detail="Brak uprawnien")


def _sync_user_as_team_leader(conn: Any, team_id: int, kierownik_user_id: int | None) -> None:
    if kierownik_user_id is None:
        return

    user_row = conn.execute(
        """
        SELECT id, aktywny, rola_id, zespol_id
        FROM users
        WHERE id = ?
        LIMIT 1
        """,
        (kierownik_user_id,),
    ).fetchone()
    if user_row is None:
        raise HTTPException(status_code=404, detail="Kierownik nie istnieje")

    user = dict(user_row)
    if int(user["aktywny"]) != 1:
        raise HTTPException(status_code=400, detail="Kierownik musi byc aktywny")

    conflict_team = conn.execute(
        "SELECT id FROM teams WHERE kierownik_user_id = ? AND id <> ? LIMIT 1",
        (kierownik_user_id, team_id),
    ).fetchone()
    if conflict_team is not None:
        raise HTTPException(status_code=409, detail="Uzytkownik jest juz kierownikiem innego zespolu")

    # Backend keeps data consistent automatically: assigning a leader also updates
    # user's team and role to team_lead.
    if user["zespol_id"] != team_id or int(user["rola_id"]) != 2:
        conn.execute(
            """
            UPDATE users
            SET zespol_id = ?,
                rola_id = 2,
                zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (team_id, kierownik_user_id),
        )


@router.get("/api/admin/teams", response_model=list[TeamRead])
def list_teams() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                t.id,
                t.kod,
                t.nazwa,
                t.kierownik_user_id
            FROM teams t
            ORDER BY t.id ASC
            """
        ).fetchall()

    return [
        {
            "id": row["id"],
            "kod": row["kod"],
            "nazwa": row["nazwa"],
            "kierownikUserId": row["kierownik_user_id"],
        }
        for row in rows
    ]


@router.post("/api/admin/teams", response_model=TeamRead, status_code=201)
def create_team(
    payload: TeamCreateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    kod = payload.kod.strip().upper()
    nazwa = payload.nazwa.strip()

    if not kod or " " in kod:
        raise HTTPException(status_code=400, detail="Kod jest wymagany i nie moze zawierac spacji")
    if not nazwa:
        raise HTTPException(status_code=400, detail="Nazwa jest wymagana")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        existing_code = conn.execute(
            "SELECT id FROM teams WHERE lower(kod) = lower(?) LIMIT 1",
            (kod,),
        ).fetchone()
        if existing_code is not None:
            raise HTTPException(status_code=409, detail="Kod zespolu juz istnieje")

        existing_name = conn.execute(
            "SELECT id FROM teams WHERE lower(nazwa) = lower(?) LIMIT 1",
            (nazwa,),
        ).fetchone()
        if existing_name is not None:
            raise HTTPException(status_code=409, detail="Nazwa zespolu juz istnieje")

        cursor = conn.execute(
            "INSERT INTO teams (kod, nazwa, kierownik_user_id) VALUES (?, ?, NULL)",
            (kod, nazwa),
        )
        team_id = int(cursor.lastrowid)

        _sync_user_as_team_leader(conn, team_id, payload.kierownikUserId)

        if payload.kierownikUserId is not None:
            conn.execute(
                "UPDATE teams SET kierownik_user_id = ?, zaktualizowano_o = CURRENT_TIMESTAMP WHERE id = ?",
                (payload.kierownikUserId, team_id),
            )

        conn.commit()

        created = conn.execute(
            "SELECT id, kod, nazwa, kierownik_user_id FROM teams WHERE id = ? LIMIT 1",
            (team_id,),
        ).fetchone()

    if created is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac utworzonego zespolu")

    row = dict(created)
    return {
        "id": row["id"],
        "kod": row["kod"],
        "nazwa": row["nazwa"],
        "kierownikUserId": row["kierownik_user_id"],
    }


@router.put("/api/admin/teams/{team_id}", response_model=TeamRead)
def update_team(
    team_id: int,
    payload: TeamUpdateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Brak pol do aktualizacji")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        existing_team = conn.execute(
            "SELECT id, kod, nazwa, kierownik_user_id FROM teams WHERE id = ? LIMIT 1",
            (team_id,),
        ).fetchone()
        if existing_team is None:
            raise HTTPException(status_code=404, detail="Zespol nie istnieje")

        team = dict(existing_team)
        next_kod = team["kod"]
        next_nazwa = team["nazwa"]
        next_kierownik = team["kierownik_user_id"]

        if "kod" in fields:
            next_kod = fields["kod"].strip().upper()
            if not next_kod or " " in next_kod:
                raise HTTPException(status_code=400, detail="Kod jest wymagany i nie moze zawierac spacji")
            code_conflict = conn.execute(
                "SELECT id FROM teams WHERE lower(kod) = lower(?) AND id <> ? LIMIT 1",
                (next_kod, team_id),
            ).fetchone()
            if code_conflict is not None:
                raise HTTPException(status_code=409, detail="Kod zespolu juz istnieje")

        if "nazwa" in fields:
            next_nazwa = fields["nazwa"].strip()
            if not next_nazwa:
                raise HTTPException(status_code=400, detail="Nazwa jest wymagana")
            name_conflict = conn.execute(
                "SELECT id FROM teams WHERE lower(nazwa) = lower(?) AND id <> ? LIMIT 1",
                (next_nazwa, team_id),
            ).fetchone()
            if name_conflict is not None:
                raise HTTPException(status_code=409, detail="Nazwa zespolu juz istnieje")

        if "kierownikUserId" in fields:
            next_kierownik = fields["kierownikUserId"]
            _sync_user_as_team_leader(conn, team_id, next_kierownik)

        conn.execute(
            """
            UPDATE teams
            SET kod = ?, nazwa = ?, kierownik_user_id = ?, zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (next_kod, next_nazwa, next_kierownik, team_id),
        )
        conn.commit()

        updated = conn.execute(
            "SELECT id, kod, nazwa, kierownik_user_id FROM teams WHERE id = ? LIMIT 1",
            (team_id,),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac zaktualizowanego zespolu")

    row = dict(updated)
    return {
        "id": row["id"],
        "kod": row["kod"],
        "nazwa": row["nazwa"],
        "kierownikUserId": row["kierownik_user_id"],
    }


@router.delete("/api/admin/teams/{team_id}")
def delete_team(
    team_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, bool]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        existing_team = conn.execute(
            "SELECT id FROM teams WHERE id = ? LIMIT 1",
            (team_id,),
        ).fetchone()
        if existing_team is None:
            raise HTTPException(status_code=404, detail="Zespol nie istnieje")

        active_users = conn.execute(
            "SELECT COUNT(1) AS cnt FROM users WHERE zespol_id = ? AND aktywny = 1",
            (team_id,),
        ).fetchone()
        if active_users is not None and int(active_users["cnt"]) > 0:
            raise HTTPException(status_code=409, detail="Nie mozna usunac zespolu z aktywnymi uzytkownikami")

        conn.execute("DELETE FROM teams WHERE id = ?", (team_id,))
        conn.commit()

    return {"ok": True}


@router.get("/api/admin/users", response_model=list[UserRead])
def list_users(
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        operator_role_id = int(operator["rola_id"])
        operator_team_id = operator["zespol_id"]

        if operator_role_id == 3:
            where_clause = ""
            params: tuple[Any, ...] = ()
        elif operator_role_id == 2:
            team_ids = _resolve_operator_team_ids(conn, operator)
            if team_ids:
                placeholders = ", ".join("?" for _ in team_ids)
                where_clause = f"WHERE (u.zespol_id IN ({placeholders}) OR u.created_by_user_id = ?)"
                params = tuple(sorted(team_ids)) + (int(operator["id"]),)
            else:
                where_clause = "WHERE u.created_by_user_id = ?"
                params = (int(operator["id"]),)
        else:
            raise HTTPException(status_code=403, detail="Brak uprawnien do podgladu uzytkownikow")

        rows = conn.execute(
            _user_select_sql(where_clause) + " ORDER BY u.id ASC",
            params,
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        creator_id = row_dict.get("created_by_user_id")
        row_dict["created_by_operator"] = bool(creator_id is not None and int(creator_id) == int(operator["id"]))
        result.append(_map_user_row(row_dict))
    return result


@router.post("/api/admin/users", response_model=UserRead)
def create_user(
    payload: UserCreateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    requested_login = payload.login
    imie = payload.imie.strip()
    nazwisko = payload.nazwisko.strip()

    if not imie:
        raise HTTPException(status_code=400, detail="Imie jest wymagane")
    if not nazwisko:
        raise HTTPException(status_code=400, detail="Nazwisko jest wymagane")
    if payload.rolaId not in (1, 2, 3, ROLE_EXTERNAL_USER):
        raise HTTPException(status_code=400, detail="Niepoprawna rolaId")

    next_account_type = payload.accountType or (ACCOUNT_TYPE_OBSERVER if payload.rolaId == ROLE_EXTERNAL_USER else ACCOUNT_TYPE_DIU)
    next_rola_id = int(payload.rolaId)
    next_list_visibility = payload.listVisibility or LIST_VISIBILITY_VISIBLE
    next_department_code = payload.departmentCode

    with get_connection() as conn:
        operator_data = _resolve_operator(conn, x_operator_login)

        operator_role_id = int(operator_data["rola_id"])
        operator_team_id = operator_data["zespol_id"]

        if operator_role_id not in (2, 3):
            raise HTTPException(status_code=403, detail="Brak uprawnien do dodawania uzytkownikow")

        target_team_id = payload.zespolId

        # Technical accounts are non-operational placeholders and always use inspector role semantics.
        if str(next_account_type).strip().lower() == ACCOUNT_TYPE_TECHNICAL:
            next_rola_id = 1

        if next_rola_id in (3, ROLE_EXTERNAL_USER):
            if operator_role_id != 3:
                raise HTTPException(status_code=403, detail="Tylko director moze tworzyc role director lub external")
            target_team_id = None

        if target_team_id is not None:
            team_row = conn.execute(
                "SELECT id FROM teams WHERE id = ? LIMIT 1",
                (target_team_id,),
            ).fetchone()
            if team_row is None:
                raise HTTPException(status_code=404, detail="Zespol nie istnieje")

        if operator_role_id == 2:
            if operator_team_id is None:
                raise HTTPException(status_code=403, detail="Kierownik bez przypisanego zespolu")
            if str(next_account_type).strip().lower() == ACCOUNT_TYPE_TECHNICAL:
                if target_team_id is not None:
                    raise HTTPException(status_code=403, detail="Kierownik moze tworzyc konto technical tylko bez przypisanego zespolu")
            elif target_team_id != operator_team_id:
                raise HTTPException(status_code=403, detail="Kierownik moze dodawac tylko do swojego zespolu")
            if str(next_account_type).strip().lower() != ACCOUNT_TYPE_TECHNICAL and next_rola_id != 1:
                raise HTTPException(status_code=403, detail="Kierownik moze tworzyc tylko role inspector")

        (
            next_account_type,
            next_department_code,
            next_department_label,
            next_list_visibility,
            target_team_id,
        ) = _validate_profile_state(
            conn,
            account_type=next_account_type,
            department_code=next_department_code,
            email=str(payload.email) if payload.email is not None else None,
            list_visibility=next_list_visibility,
            aktywny=1 if payload.aktywny else 0,
            rola_id=next_rola_id,
            zespol_id=target_team_id,
            previous_account_type=next_account_type,
        )

        if target_team_id is not None:
            team_row = conn.execute(
                "SELECT id FROM teams WHERE id = ? LIMIT 1",
                (target_team_id,),
            ).fetchone()
            if team_row is None:
                raise HTTPException(status_code=404, detail="Zespol nie istnieje")

        login = _prepare_login_for_persistence(
            conn,
            requested_login=requested_login,
            current_login=None,
            user_id=None,
            account_type=next_account_type,
            aktywny=1 if payload.aktywny else 0,
        )

        password_hash_value: str | None = None
        if payload.password is not None:
            try:
                password_hash_value = hash_password(payload.password)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        cursor = conn.execute(
            """
            INSERT INTO users (
                login,
                imie,
                nazwisko,
                email,
                password_hash,
                rola_id,
                zespol_id,
                created_by_user_id,
                aktywny,
                account_type,
                department_code,
                list_visibility,
                profile_changed_at,
                profile_changed_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                login,
                imie,
                nazwisko,
                str(payload.email) if payload.email is not None else None,
                password_hash_value,
                next_rola_id,
                target_team_id,
                int(operator_data["id"]),
                1 if payload.aktywny else 0,
                next_account_type,
                next_department_code,
                next_list_visibility,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                int(operator_data["id"]),
            ),
        )
        user_id = int(cursor.lastrowid)

        changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_profile_history_version(
            conn,
            user_id=user_id,
            login=login,
            imie=imie,
            nazwisko=nazwisko,
            email=str(payload.email) if payload.email is not None else None,
            rola_id=next_rola_id,
            aktywny=1 if payload.aktywny else 0,
            account_type=next_account_type,
            zespol_id=target_team_id,
            department_code=next_department_code,
            department_label=next_department_label,
            list_visibility=next_list_visibility,
            permissions_codes=[],
            changed_by_user_id=int(operator_data["id"]),
            changed_by_login=str(operator_data["login"]),
            changed_at=changed_at,
        )
        conn.execute(
            """
            UPDATE users
            SET profile_changed_at = ?, profile_changed_by_user_id = ?
            WHERE id = ?
            """,
            (changed_at, int(operator_data["id"]), user_id),
        )
        conn.commit()

        created = conn.execute(
            _user_select_sql("WHERE u.id = ?") + " LIMIT 1",
            (user_id,),
        ).fetchone()

    if created is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac utworzonego uzytkownika")

    return _map_user_row(dict(created))


@router.put("/api/admin/users/{user_id}", response_model=UserRead)
def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Brak pol do aktualizacji")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        operator_role_id = int(operator["rola_id"])
        operator_team_id = operator["zespol_id"]

        if operator_role_id not in (2, 3):
            raise HTTPException(status_code=403, detail="Brak uprawnien do edycji uzytkownikow")

        target = conn.execute(
            """
            SELECT
                id,
                login,
                imie,
                nazwisko,
                email,
                rola_id,
                zespol_id,
                created_by_user_id,
                aktywny,
                account_type,
                department_code,
                list_visibility,
                profile_changed_at,
                profile_changed_by_user_id
            FROM users
            WHERE id = ?
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if target is None:
            raise HTTPException(status_code=404, detail="Uzytkownik nie istnieje")

        current = dict(target)

        if operator_role_id == 2:
            team_ids = _resolve_operator_team_ids(conn, operator)
            if not _can_team_lead_manage_user(operator=operator, team_ids=team_ids, target=current):
                raise HTTPException(status_code=403, detail="Kierownik moze edytowac tylko uzytkownikow ze swojego zespolu")

        next_login = current["login"]
        next_imie = current["imie"]
        next_nazwisko = current["nazwisko"]
        next_email = current["email"]
        next_rola_id = int(current["rola_id"])
        next_zespol_id = current["zespol_id"]
        next_aktywny = int(current["aktywny"])
        next_account_type = str(current.get("account_type") or (ACCOUNT_TYPE_OBSERVER if next_rola_id == ROLE_EXTERNAL_USER else ACCOUNT_TYPE_DIU))
        next_department_code = current.get("department_code")
        next_list_visibility = str(current.get("list_visibility") or LIST_VISIBILITY_VISIBLE)

        requested_login = next_login
        if "login" in fields:
            requested_login = fields["login"]

        if "imie" in fields:
            next_imie = (fields["imie"] or "").strip()
            if not next_imie:
                raise HTTPException(status_code=400, detail="Imie jest wymagane")

        if "nazwisko" in fields:
            next_nazwisko = (fields["nazwisko"] or "").strip()
            if not next_nazwisko:
                raise HTTPException(status_code=400, detail="Nazwisko jest wymagane")

        if "email" in fields:
            next_email = str(fields["email"] or "").strip() or None

        if "rolaId" in fields:
            if fields["rolaId"] not in (1, 2, 3, ROLE_EXTERNAL_USER):
                raise HTTPException(status_code=400, detail="Niepoprawna rolaId")
            next_rola_id = int(fields["rolaId"])

        if "zespolId" in fields:
            next_zespol_id = fields["zespolId"]

        if "aktywny" in fields:
            next_aktywny = 1 if fields["aktywny"] else 0

        if "accountType" in fields:
            next_account_type = str(fields["accountType"])
        if "departmentCode" in fields:
            next_department_code = str(fields["departmentCode"] or "").strip() or None
        if "listVisibility" in fields:
            next_list_visibility = str(fields["listVisibility"])

        if operator_role_id == 2:
            team_ids = _resolve_operator_team_ids(conn, operator)
            if "zespolId" in fields and next_zespol_id is not None and int(next_zespol_id) not in team_ids:
                raise HTTPException(status_code=403, detail="Kierownik nie moze przenosic do innego zespolu")
            if "rolaId" in fields and next_rola_id != 1:
                raise HTTPException(status_code=403, detail="Kierownik moze ustawic tylko role inspector")

        (
            next_account_type,
            next_department_code,
            next_department_label,
            next_list_visibility,
            next_zespol_id,
        ) = _validate_profile_state(
            conn,
            account_type=next_account_type,
            department_code=next_department_code,
            email=next_email,
            list_visibility=next_list_visibility,
            aktywny=next_aktywny,
            rola_id=next_rola_id,
            zespol_id=next_zespol_id,
            previous_account_type=str(current.get("account_type") or ""),
        )

        next_login = _prepare_login_for_persistence(
            conn,
            requested_login=str(requested_login or ""),
            current_login=str(current.get("login") or ""),
            user_id=int(user_id),
            account_type=next_account_type,
            aktywny=next_aktywny,
        )

        if next_zespol_id is not None:
            team_exists = conn.execute(
                "SELECT id FROM teams WHERE id = ? LIMIT 1",
                (next_zespol_id,),
            ).fetchone()
            if team_exists is None:
                raise HTTPException(status_code=404, detail="Zespol nie istnieje")

        changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        conn.execute(
            """
            UPDATE users
            SET login = ?,
                imie = ?,
                nazwisko = ?,
                email = ?,
                rola_id = ?,
                zespol_id = ?,
                aktywny = ?,
                account_type = ?,
                department_code = ?,
                list_visibility = ?,
                profile_changed_at = ?,
                profile_changed_by_user_id = ?,
                zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                next_login,
                next_imie,
                next_nazwisko,
                next_email,
                next_rola_id,
                next_zespol_id,
                next_aktywny,
                next_account_type,
                next_department_code,
                next_list_visibility,
                changed_at,
                int(operator["id"]),
                user_id,
            ),
        )

        # Business rule: every user edit creates a profile history version,
        # even when profile values are unchanged.
        _write_profile_history_version(
            conn,
            user_id=int(user_id),
            login=next_login,
            imie=next_imie,
            nazwisko=next_nazwisko,
            email=next_email,
            rola_id=next_rola_id,
            aktywny=next_aktywny,
            account_type=next_account_type,
            zespol_id=next_zespol_id,
            department_code=next_department_code,
            department_label=next_department_label,
            list_visibility=next_list_visibility,
            permissions_codes=get_user_permissions(conn, int(user_id)),
            changed_by_user_id=int(operator["id"]),
            changed_by_login=str(operator["login"]),
            changed_at=changed_at,
        )
        revoke_user_sessions(conn, int(user_id))
        conn.commit()

        updated = conn.execute(
            _user_select_sql("WHERE u.id = ?") + " LIMIT 1",
            (user_id,),
        ).fetchone()

    if updated is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac zaktualizowanego uzytkownika")

    return _map_user_row(dict(updated))


@router.get("/api/admin/users/{user_id}/profile-history", response_model=list[UserProfileHistoryRead])
def get_user_profile_history(
    user_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _authorize_profile_history_access(conn, user_id, operator)

        rows = conn.execute(
            """
            SELECT
                h.id,
                h.valid_from,
                h.valid_to,
                h.login,
                h.imie,
                h.nazwisko,
                h.email,
                h.rola_id,
                h.aktywny,
                h.account_type,
                h.zespol_id,
                h.department_code,
                h.department_label,
                h.list_visibility,
                h.permissions_codes,
                h.changed_by_user_id,
                h.changed_by_login,
                h.changed_at,
                u.login AS changed_by_login_live
            FROM user_profile_history h
            LEFT JOIN users u ON u.id = h.changed_by_user_id
            WHERE h.user_id = ?
            ORDER BY datetime(h.valid_from) ASC, h.id ASC
            """,
            (int(user_id),),
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        changed_by = None
        if row["changed_by_user_id"] is not None or row["changed_by_login"] is not None:
            changed_by = {
                "userId": int(row["changed_by_user_id"]) if row["changed_by_user_id"] is not None else None,
                "login": row["changed_by_login"] or row["changed_by_login_live"],
            }
        result.append(
            {
                "id": int(row["id"]),
                "validFrom": str(row["valid_from"]),
                "validTo": row["valid_to"],
                "login": row["login"],
                "imie": row["imie"],
                "nazwisko": row["nazwisko"],
                "email": row["email"],
                "rolaId": int(row["rola_id"]) if row["rola_id"] is not None else None,
                "aktywny": bool(row["aktywny"]) if row["aktywny"] is not None else None,
                "accountType": str(row["account_type"]),
                "zespolId": row["zespol_id"],
                "departmentCode": row["department_code"],
                "departmentLabel": row["department_label"],
                "listVisibility": str(row["list_visibility"]),
                "permissions": [part for part in str(row["permissions_codes"] or "").split(";") if part],
                "changedBy": changed_by,
                "changedAt": str(row["changed_at"]),
            }
        )
    return result


@router.get("/api/admin/users/{user_id}/profile-history/events", response_model=list[UserProfileHistoryEventRead])
def get_user_profile_history_events(
    user_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _authorize_profile_history_access(conn, user_id, operator)

        rows = conn.execute(
            """
            SELECT
                h.id,
                h.login,
                h.imie,
                h.nazwisko,
                h.email,
                h.rola_id,
                h.aktywny,
                h.account_type,
                h.zespol_id,
                h.department_code,
                h.list_visibility,
                h.permissions_codes,
                h.changed_by_login,
                h.changed_at,
                u.login AS changed_by_login_live
            FROM user_profile_history h
            LEFT JOIN users u ON u.id = h.changed_by_user_id
            WHERE h.user_id = ?
            ORDER BY datetime(h.valid_from) ASC, h.id ASC
            """,
            (int(user_id),),
        ).fetchall()

        team_name_rows = conn.execute("SELECT id, nazwa FROM teams").fetchall()
        team_names = {int(row["id"]): str(row["nazwa"]) for row in team_name_rows}

    tracked_fields = (
        "login",
        "imie",
        "nazwisko",
        "email",
        "rolaId",
        "aktywny",
        "accountType",
        "zespolId",
        "departmentCode",
        "listVisibility",
        "permissions",
    )

    events: list[dict[str, Any]] = []
    previous_state: dict[str, Any] | None = None
    for raw in rows:
        row = dict(raw)
        current_state = {
            "login": row.get("login"),
            "imie": row.get("imie"),
            "nazwisko": row.get("nazwisko"),
            "email": row.get("email"),
            "rolaId": int(row["rola_id"]) if row.get("rola_id") is not None else None,
            "aktywny": int(row["aktywny"]) if row.get("aktywny") is not None else None,
            "accountType": str(row["account_type"]),
            "zespolId": row["zespol_id"],
            "departmentCode": row["department_code"],
            "listVisibility": str(row["list_visibility"]),
            "permissions": [part for part in str(row.get("permissions_codes") or "").split(";") if part],
        }

        if previous_state is None:
            previous_state = current_state
            continue

        changes: list[dict[str, Any]] = []
        for field in tracked_fields:
            before = previous_state.get(field)
            after = current_state.get(field)
            if before == after:
                continue
            changes.append(
                {
                    "field": field,
                    "label": _profile_history_field_label(field),
                    "before": _serialize_profile_history_value(field, before, team_names),
                    "after": _serialize_profile_history_value(field, after, team_names),
                }
            )

        if changes:
            events.append(
                {
                    "eventId": int(row["id"]),
                    "changedAt": str(row["changed_at"]),
                    "changedBy": row["changed_by_login"] or row["changed_by_login_live"],
                    "changes": changes,
                }
            )

        previous_state = current_state

    events.sort(key=lambda item: str(item["changedAt"]), reverse=True)
    return events


@router.get("/api/admin/permissions/catalog", response_model=list[PermissionCatalogItem])
def list_permission_catalog(
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> list[dict[str, str]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        if int(operator["rola_id"]) == 3:
            return [dict(item) for item in PERMISSION_CATALOG]
        if int(operator["rola_id"]) == 2:
            allowed = set(TEAM_LEAD_MANAGEABLE_PERMISSION_CODES)
            return [dict(item) for item in PERMISSION_CATALOG if str(item.get("code") or "") in allowed]
        raise HTTPException(status_code=403, detail="Brak uprawnien")


@router.get("/api/admin/users/{user_id}/permissions", response_model=UserPermissionsRead)
def get_user_permissions_for_admin(
    user_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator_for_permissions(conn, x_operator_login)
        profile, team_lead_mode = _ensure_permissions_access_to_target(conn, operator, int(user_id))

        if not _supports_module_permissions(role_id=int(profile["rola_id"]), account_type=profile["account_type"]):
            raise HTTPException(status_code=400, detail="Uprawnienia per-modul sa dostepne tylko dla observer/external_user")

        permissions = get_user_permissions(conn, int(user_id))
        if team_lead_mode:
            allowed = set(TEAM_LEAD_MANAGEABLE_PERMISSION_CODES)
            permissions = [code for code in permissions if code in allowed]

    return {
        "userId": int(user_id),
        "permissions": permissions,
    }


@router.put("/api/admin/users/{user_id}/permissions", response_model=UserPermissionsRead)
def update_user_permissions_for_admin(
    user_id: int,
    payload: UserPermissionsUpdateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator_for_permissions(conn, x_operator_login)
        user_row, team_lead_mode = _ensure_permissions_access_to_target(conn, operator, int(user_id))
        if not _supports_module_permissions(role_id=int(user_row["rola_id"]), account_type=user_row["account_type"]):
            raise HTTPException(status_code=400, detail="Uprawnienia per-modul sa dostepne tylko dla observer/external_user")

        codes = normalize_permission_codes(payload.permissions)
        if team_lead_mode:
            allowed = set(TEAM_LEAD_MANAGEABLE_PERMISSION_CODES)
            disallowed = sorted(code for code in codes if code not in allowed)
            if disallowed:
                raise HTTPException(status_code=403, detail="Kierownik moze zarzadzac tylko wybranymi uprawnieniami")
        set_user_permissions(conn, int(user_id), codes, int(operator["id"]))

        # Keep a consistent audit trail in profile-history for observer permission edits.
        _append_current_profile_history(
            conn,
            user_id=int(user_id),
            changed_by_user_id=int(operator["id"]),
            changed_by_login=str(operator["login"]),
        )
        revoke_user_sessions(conn, int(user_id))
        conn.commit()

    return {
        "userId": int(user_id),
        "permissions": codes,
    }


@router.post("/api/admin/users/external", response_model=ExternalUserCreateResponse, status_code=201)
def create_external_user(
    payload: ExternalUserCreateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    login = payload.login.strip().lower()
    imie = payload.imie.strip()
    nazwisko = payload.nazwisko.strip()
    email = payload.email.strip()
    if not login or " " in login:
        raise HTTPException(status_code=400, detail="Login jest wymagany i nie moze zawierac spacji")
    if not imie:
        raise HTTPException(status_code=400, detail="Imie jest wymagane")
    if not nazwisko:
        raise HTTPException(status_code=400, detail="Nazwisko jest wymagane")
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Email jest wymagany i musi byc poprawny")
    if not payload.aktywny:
        raise HTTPException(status_code=422, detail="Observer nie moze byc nieaktywny")

    codes = normalize_permission_codes(payload.permissions)

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        operator_role_id = int(operator["rola_id"])
        if operator_role_id not in (2, 3):
            raise HTTPException(status_code=403, detail="Brak uprawnien")

        if operator_role_id == 2:
            team_ids = _resolve_operator_team_ids(conn, operator)
            if not team_ids:
                raise HTTPException(status_code=403, detail="Kierownik bez przypisanego zespolu")

            allowed = set(TEAM_LEAD_MANAGEABLE_PERMISSION_CODES)
            disallowed = sorted(code for code in codes if code not in allowed)
            if disallowed:
                raise HTTPException(status_code=403, detail="Kierownik moze nadawac tylko wybrane uprawnienia")

        existing_login = conn.execute(
            "SELECT id FROM users WHERE lower(login) = lower(?) LIMIT 1",
            (login,),
        ).fetchone()
        if existing_login is not None:
            raise HTTPException(status_code=409, detail="Login juz istnieje")

        cursor = conn.execute(
            """
            INSERT INTO users (
                login,
                imie,
                nazwisko,
                email,
                password_hash,
                rola_id,
                zespol_id,
                created_by_user_id,
                aktywny,
                account_type,
                department_code,
                list_visibility,
                profile_changed_at,
                profile_changed_by_user_id
            ) VALUES (?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                login,
                imie,
                nazwisko,
                email,
                ROLE_EXTERNAL_USER,
                int(operator["id"]),
                1 if payload.aktywny else 0,
                ACCOUNT_TYPE_OBSERVER,
                None,
                LIST_VISIBILITY_VISIBLE,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                int(operator["id"]),
            ),
        )
        user_id = int(cursor.lastrowid)

        changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_profile_history_version(
            conn,
            user_id=user_id,
            login=login,
            imie=imie,
            nazwisko=nazwisko,
            email=email,
            rola_id=ROLE_EXTERNAL_USER,
            aktywny=1 if payload.aktywny else 0,
            account_type=ACCOUNT_TYPE_OBSERVER,
            zespol_id=None,
            department_code=None,
            department_label=None,
            list_visibility=LIST_VISIBILITY_VISIBLE,
            permissions_codes=codes,
            changed_by_user_id=int(operator["id"]),
            changed_by_login=str(operator["login"]),
            changed_at=changed_at,
        )
        conn.execute(
            """
            UPDATE users
            SET profile_changed_at = ?, profile_changed_by_user_id = ?
            WHERE id = ?
            """,
            (changed_at, int(operator["id"]), user_id),
        )

        set_user_permissions(conn, user_id, codes, int(operator["id"]))

        delivery, expires_at, invite_link = _create_and_send_invite_for_user(
            conn,
            user_id=user_id,
            login=login,
            email=email,
            operator_user_id=int(operator["id"]),
            frontend_invite_url=payload.frontendInviteUrl,
        )

        conn.commit()

        created = conn.execute(
            _user_select_sql("WHERE u.id = ?") + " LIMIT 1",
            (user_id,),
        ).fetchone()

    if created is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac utworzonego uzytkownika")

    invite_payload: dict[str, Any] = {
        "ok": True,
        "expiresAt": expires_at,
        "delivery": delivery,
    }
    if delivery == "log":
        invite_payload["inviteLink"] = invite_link

    _audit(
        "external_user_created",
        operator_id=str(operator["id"]),
        target_user_id=str(user_id),
        delivery=delivery,
    )

    return {
        "user": _map_user_row(dict(created)),
        "invite": invite_payload,
    }


@router.put("/api/admin/users/{user_id}/password")
def update_user_password(
    user_id: int,
    payload: UserPasswordUpdateRequest,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, bool]:
    try:
        password_hash_value = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        _ensure_director(operator)

        exists = conn.execute("SELECT id FROM users WHERE id = ? LIMIT 1", (user_id,)).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Uzytkownik nie istnieje")

        conn.execute(
            """
            UPDATE users
            SET password_hash = ?,
                zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (password_hash_value, user_id),
        )
        revoke_user_sessions(conn, user_id)
        _audit("admin_password_reset", operator_id=str(operator["id"]), target_user_id=str(user_id))
        conn.commit()

    return {"ok": True}


@router.post("/api/admin/users/{user_id}/send-invite", response_model=SendUserInviteResponse)
def send_user_invite(
    user_id: int,
    payload: SendUserInviteRequest | None = None,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        operator_role_id = int(operator["rola_id"])
        operator_user_id = int(operator["id"])

        if operator_role_id not in (2, 3):
            raise HTTPException(status_code=403, detail="Brak uprawnien do wysylki zaproszen")

        row = conn.execute(
            """
            SELECT id, login, email, rola_id, zespol_id, aktywny, account_type, created_by_user_id
            FROM users
            WHERE id = ?
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Uzytkownik nie istnieje")

        target = dict(row)
        target_account_type = str(target.get("account_type") or "").strip().lower()
        if target_account_type not in {ACCOUNT_TYPE_DIU, ACCOUNT_TYPE_OBSERVER, ACCOUNT_TYPE_TECHNICAL}:
            raise HTTPException(status_code=422, detail="Niepoprawny accountType uzytkownika")
        if target_account_type == ACCOUNT_TYPE_TECHNICAL:
            raise HTTPException(status_code=422, detail="Dla accountType=technical wysylka zaproszenia jest niedozwolona")
        if int(target["aktywny"]) != 1:
            raise HTTPException(status_code=400, detail="Zaproszenie mozna wyslac tylko do aktywnego uzytkownika")

        login = str(target.get("login") or "").strip().lower()
        if not login or login.startswith("__inactive_") or login.startswith("__technical_"):
            raise HTTPException(status_code=400, detail="Uzytkownik musi miec poprawny login przed wyslaniem zaproszenia")

        email = str(target.get("email") or "").strip()
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Uzytkownik musi miec poprawny email")

        if operator_role_id == 2:
            team_ids = _resolve_operator_team_ids(conn, operator)
            if not _can_team_lead_manage_user(operator=operator, team_ids=team_ids, target=target):
                raise HTTPException(status_code=403, detail="Kierownik moze zapraszac tylko swoj zespol lub uzytkownikow dodanych przez siebie")

        delivery, expires_at, invite_link = _create_and_send_invite_for_user(
            conn,
            user_id=int(user_id),
            login=login,
            email=email,
            operator_user_id=operator_user_id,
            frontend_invite_url=payload.frontendInviteUrl if payload is not None else None,
        )
        _audit(
            "invite_sent",
            operator_id=str(operator_user_id),
            target_user_id=str(user_id),
            delivery=delivery,
        )

        conn.commit()

    response: dict[str, Any] = {
        "ok": True,
        "expiresAt": expires_at,
        "delivery": delivery,
    }
    if delivery == "log":
        # In local/dev mode expose link for manual testing without SMTP.
        response["inviteLink"] = invite_link
    return response
