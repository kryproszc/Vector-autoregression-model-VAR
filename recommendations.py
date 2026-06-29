from __future__ import annotations

from datetime import date
import os
import re
from app.permissions import PERMISSION_RECOMMENDATIONS_READ, require_permission, require_write_access
import unicodedata
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.audit import (
    AKCJA_CREATE,
    AKCJA_DELETE,
    AKCJA_UPDATE,
    REJESTR_ZALECENIA,
    build_create_changes,
    build_recommendation_changes,
    new_session_id,
    write_audit_log,
)
from app.database import get_connection
from app.record_locks import assert_expected_updated_at, assert_lock_for_save, now_rfc3339_utc_ms

router = APIRouter()

FORBIDDEN_INSPECTION_STATUS_CODES = {
    "CLOSED_WITH_RECOMMENDATIONS",
    "CLOSED_WITHOUT_RECOMMENDATIONS",
}
FORBIDDEN_INSPECTION_STATUS_LABEL_KEYS = {
    "zamkniete - wydano zalecenia",
    "zamkniete - brak zalecen",
}
INSPECTION_STATUS_BLOCK_ERROR_CODE = "INSPECTION_STATUS_BLOCKS_OPERATION"


class RecommendationCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "inspectionId": 1,
                "pozycja": 1,
                "dataZalecen": "2026-10-31",
                "statusId": 1,
                "komentarz": "Opis zalecenia",
                "terminyWykonaniaZalecenList": ["2026-10-01", "2026-10-05"],
                "dataAkceptacjiNotyWeryfikacjiList": ["2026-10-10"],
            }
        },
    )

    inspectionId: int | None = None
    inspectionTeamIds: list[int] | None = None
    inspection_team_ids: list[int] | None = None
    pozycja: int = Field(ge=1)
    nazwaPodmiotuId: int | None = None
    dataZalecen: str | None = None
    terminyWykonaniaZalecenList: list[str] | None = None
    statusId: int
    komentarz: str | None = None
    dataAkceptacjiNotyWeryfikacjiList: list[str] | None = None
    brakTerminowWykonaniaZalecen: bool = False
    brakDatAkceptacjiNotyWeryfikacji: bool = False


class RecommendationUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "pozycja": 2,
                "statusId": 2,
                "komentarz": "Po weryfikacji",
                "terminyWykonaniaZalecenList": ["2026-10-15"],
                "dataAkceptacjiNotyWeryfikacjiList": ["2026-10-20", "2026-10-22"],
            }
        }
    )

    inspectionId: int | None = None
    inspectionTeamIds: list[int] | None = None
    inspection_team_ids: list[int] | None = None
    lockToken: str | None = None
    expectedUpdatedAt: str | None = None
    pozycja: int | None = Field(default=None, ge=1)
    nazwaPodmiotuId: int | None = None
    dataZalecen: str | None = None
    terminyWykonaniaZalecenList: list[str] | None = None
    statusId: int | None = None
    komentarz: str | None = None
    dataAkceptacjiNotyWeryfikacjiList: list[str] | None = None
    brakTerminowWykonaniaZalecen: bool | None = None
    brakDatAkceptacjiNotyWeryfikacji: bool | None = None


class RecommendationRead(BaseModel):
    id: int
    lp: int
    kodZalecenia: str | None = None
    canEdit: bool
    inspectionId: int | None = None
    inspectionTeamIds: list[int]
    inspectionLp: int | None = None
    inspectionKod: str | None = None
    pozycja: int
    nazwaPodmiotuId: int | None = None
    nazwaPodmiotu: str | None = None
    nazwaPodmiotuSkrocona: str | None = None
    nazwaPodmiotuSkrot: str | None = None
    dataZalecen: str | None = None
    terminyWykonaniaZalecenList: list[str]
    # Legacy aliases kept for backward compatibility.
    terminWykonaniaZalecen: str | None = None
    statusId: int | None = None
    status: str | None = None
    statusSkrocona: str | None = None
    statusSkrot: str | None = None
    komentarz: str | None = None
    dataZalecenList: list[str]
    dataAkceptacjiNotyWeryfikacjiList: list[str]
    brakTerminowWykonaniaZalecen: bool
    brakDatAkceptacjiNotyWeryfikacji: bool
    utworzonoO: str
    zaktualizowanoO: str


class RecommendationListResponse(BaseModel):
    items: list[RecommendationRead]
    total: int


class RecommendationInspectionOption(BaseModel):
    id: int
    lp: int
    kodInspekcji: str | None = None
    nazwaPodmiotu: str
    nazwaPodmiotuSkrocona: str | None = None
    nazwaPodmiotuSkrot: str | None = None
    poczatekInspekcji: str
    koniecInspekcji: str
    osobaKierujacaUserId: int | None = None
    osobaKierujaca: str | None = None


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _raise_contract_422(
    code: str,
    message: str,
    *,
    field: str,
    value: Any = None,
    kod_typu: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "field": field,
        "value": value,
    }
    if kod_typu is not None:
        payload["kodTypu"] = kod_typu
    raise HTTPException(status_code=422, detail=payload)


def _slug_code(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    code = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_").upper()
    return code or "POZYCJA"


def _status_label_key(value: str | None) -> str | None:
    normalized = _norm(value)
    if normalized is None:
        return None
    ascii_value = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.casefold().split())


def _parse_status_codes_env(name: str) -> set[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return set()
    values: set[str] = set()
    for token in re.split(r"[,;\n]", raw):
        cleaned = token.strip().upper()
        if cleaned:
            values.add(cleaned)
    return values


def _blocked_status_codes_for_operator(operator: dict[str, Any] | None) -> set[str]:
    blocked = _parse_status_codes_env("RECOMMENDATIONS_BLOCKED_INSPECTION_STATUS_CODES")
    if operator is None:
        return blocked

    role_id_raw = operator.get("rola_id")
    if role_id_raw is None:
        return blocked

    role_id = int(role_id_raw)
    blocked |= _parse_status_codes_env(f"RECOMMENDATIONS_BLOCKED_INSPECTION_STATUS_CODES_ROLE_{role_id}")
    return blocked


def _inspection_status_info(conn: Any, inspection_id: int) -> tuple[str | None, str | None]:
    row = conn.execute(
        """
        SELECT sp.kod_pozycji AS status_code, sp.nazwa_pozycji AS status_label
        FROM inspections i
        LEFT JOIN slownik_pozycje sp ON sp.id = i.status_inspekcji_id
        WHERE i.id = ?
        LIMIT 1
        """,
        (inspection_id,),
    ).fetchone()
    if row is None:
        return None, None

    code_value = _norm(row["status_code"])
    label_value = _norm(row["status_label"])
    return (code_value.upper() if code_value else None), label_value


def _inspection_status_is_forbidden(
    conn: Any,
    inspection_id: int,
    operator: dict[str, Any] | None = None,
) -> tuple[bool, str | None, str | None]:
    status_code, status_label = _inspection_status_info(conn, inspection_id)
    label_key = _status_label_key(status_label)
    blocked_codes = set(FORBIDDEN_INSPECTION_STATUS_CODES)
    blocked_codes |= _blocked_status_codes_for_operator(operator)
    blocked = (status_code in blocked_codes) or (label_key in FORBIDDEN_INSPECTION_STATUS_LABEL_KEYS)
    return blocked, status_code, status_label


def _raise_inspection_status_block(inspection_id: int, status_code: str | None, status_label: str | None) -> None:
    raise HTTPException(
        status_code=409,
        detail={
            "code": INSPECTION_STATUS_BLOCK_ERROR_CODE,
            "message": "Nie mozna zapisac rekordu dla inspekcji zamknietej.",
            "inspectionId": inspection_id,
            "inspectionStatusCode": status_code,
            "inspectionStatus": status_label,
        },
    )


def _resolve_operator(conn: Any, operator_login: str | None) -> dict[str, Any]:
    login = (operator_login or "").strip()
    if not login:
        raise HTTPException(status_code=401, detail="Operator nie istnieje")

    row = conn.execute(
        """
        SELECT id, login, rola_id, zespol_id, aktywny
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

    return {
        "id": int(operator["id"]),
        "login": operator["login"],
        "rola_id": int(operator["rola_id"]),
        "zespol_id": operator["zespol_id"],
    }


def _ensure_director(operator: dict[str, Any]) -> None:
    if int(operator["rola_id"]) != 3:
        raise HTTPException(status_code=403, detail="Brak uprawnien")


def _resolve_slownik_item_id(conn: Any, kod_typu: str, raw_value: str | None) -> int | None:
    value = _norm(raw_value)
    if value is None or value.lower() == "brak":
        return None

    row = conn.execute(
        """
        SELECT id FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?) AND lower(nazwa_pozycji) = lower(?)
        LIMIT 1
        """,
        (kod_typu, value),
    ).fetchone()
    if row is not None:
        return int(row["id"])

    base_code = _slug_code(value)
    code = base_code
    suffix = 2
    while True:
        exists = conn.execute(
            """
            SELECT id FROM slownik_pozycje
            WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?)
            LIMIT 1
            """,
            (kod_typu, code),
        ).fetchone()
        if exists is None:
            break
        code = f"{base_code}_{suffix}"
        suffix += 1

    max_row = conn.execute(
        "SELECT COALESCE(MAX(kolejnosc), 0) AS max_kolejnosc FROM slownik_pozycje WHERE lower(kod_typu) = lower(?)",
        (kod_typu,),
    ).fetchone()
    next_order = int(max_row["max_kolejnosc"]) + 1

    cursor = conn.execute(
        """
        INSERT INTO slownik_pozycje
        (kod_typu, kod_pozycji, nazwa_pozycji, kolejnosc, aktywny)
        VALUES (?, ?, ?, ?, 1)
        """,
        (kod_typu, code, value, next_order),
    )
    return int(cursor.lastrowid)


def _resolve_dictionary_id_strict(
    conn: Any,
    *,
    kod_typu: str,
    field_name: str,
    id_value: int | None,
    required: bool,
) -> int | None:
    if id_value is None:
        if required:
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                f"Pole {field_name} jest wymagane.",
                field=field_name,
                value=id_value,
                kod_typu=kod_typu,
            )
        return None

    row = conn.execute(
        """
        SELECT id, aktywny
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?) AND id = ?
        LIMIT 1
        """,
        (kod_typu, int(id_value)),
    ).fetchone()
    if row is None or int(row["aktywny"] or 0) != 1:
        _raise_contract_422(
            "UNKNOWN_DICTIONARY_ID",
            f"{field_name} nie wskazuje aktywnej pozycji slownika.",
            field=field_name,
            value=id_value,
            kod_typu=kod_typu,
        )
    return int(row["id"])


def _normalize_date_list(raw_values: list[str] | None, field_name: str) -> list[str]:
    if raw_values is None:
        return []

    normalized: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            _raise_contract_422(
                "INVALID_DATE_VALUE",
                f"{field_name} zawiera niepoprawny typ.",
                field=field_name,
                value=raw,
            )
        value = raw.strip()
        if not value:
            _raise_contract_422(
                "INVALID_DATE_VALUE",
                f"{field_name} nie moze zawierac pustych wartosci.",
                field=field_name,
                value=raw,
            )
        if value.lower() in {"brak", "0", "0000-00-00"}:
            _raise_contract_422(
                "INVALID_DATE_VALUE",
                f"{field_name} ma niepoprawny format daty.",
                field=field_name,
                value=raw,
            )
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            _raise_contract_422(
                "INVALID_DATE_VALUE",
                f"{field_name} ma niepoprawny format daty.",
                field=field_name,
                value=raw,
            )
        normalized.append(parsed.isoformat())

    # Keep deterministic order and remove duplicates.
    return sorted(set(normalized))


def _normalize_date_absence_state(
    *,
    dates: list[str],
    brak: bool,
    list_field_name: str,
    bool_field_name: str,
) -> tuple[list[str], bool]:
    if brak and dates:
        _raise_contract_422(
            "INVALID_ABSENCE_COMBINATION",
            f"Gdy {bool_field_name}=true, {list_field_name} musi byc puste.",
            field=bool_field_name,
            value=True,
        )
    if brak:
        return [], True
    if dates:
        return dates, False
    return [], False


def _validate_optional_iso_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        _raise_contract_422(
            "INVALID_DATE_VALUE",
            f"{field_name} ma niepoprawny format daty.",
            field=field_name,
            value=value,
        )
    return parsed.isoformat()


def _resolve_single_data_zalecen(
    *,
    data_zalecen: str | None,
) -> str | None:
    return _validate_optional_iso_date(data_zalecen, "dataZalecen") if data_zalecen is not None else None


def _normalize_and_validate_recommendation_team_ids(
    conn: Any,
    *,
    team_ids: list[int] | None,
    team_ids_legacy: list[int] | None,
    field_sent: bool,
) -> list[int]:
    if team_ids is not None and team_ids_legacy is not None:
        normalized_primary = sorted(set(int(x) for x in team_ids))
        normalized_legacy = sorted(set(int(x) for x in team_ids_legacy))
        if normalized_primary != normalized_legacy:
            _raise_contract_422(
                "CONFLICTING_TEAM_IDS",
                "inspectionTeamIds i inspection_team_ids wskazuja rozne wartosci.",
                field="inspectionTeamIds",
                value={"inspectionTeamIds": team_ids, "inspection_team_ids": team_ids_legacy},
            )

    selected = team_ids if team_ids is not None else team_ids_legacy
    if field_sent and selected is None:
        _raise_contract_422(
            "INVALID_TEAM_IDS",
            "Pole inspectionTeamIds nie moze byc null.",
            field="inspectionTeamIds",
            value=selected,
        )
    if selected is None:
        return []

    normalized_ids = sorted(set(int(x) for x in selected))
    if not normalized_ids:
        return []

    placeholders = ",".join(["?"] * len(normalized_ids))
    rows = conn.execute(
        f"SELECT id FROM teams WHERE id IN ({placeholders})",
        tuple(normalized_ids),
    ).fetchall()
    existing = {int(row["id"]) for row in rows}
    missing = [team_id for team_id in normalized_ids if team_id not in existing]
    if missing:
        _raise_contract_422(
            "UNKNOWN_TEAM_ID",
            "inspectionTeamIds zawiera nieistniejace teams.id.",
            field="inspectionTeamIds",
            value=missing,
        )

    return normalized_ids


def _sync_recommendation_teams(conn: Any, recommendation_id: int, team_ids: list[int]) -> None:
    conn.execute("DELETE FROM recommendation_teams WHERE recommendation_id = ?", (recommendation_id,))
    for team_id in team_ids:
        conn.execute(
            "INSERT OR IGNORE INTO recommendation_teams (recommendation_id, team_id) VALUES (?, ?)",
            (recommendation_id, int(team_id)),
        )


def _get_recommendation_team_ids(conn: Any, recommendation_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT team_id FROM recommendation_teams WHERE recommendation_id = ? ORDER BY team_id ASC",
        (recommendation_id,),
    ).fetchall()
    return [int(row["team_id"]) for row in rows]


def _resolve_terminy_wykonania_list(
    *,
    terminy_wykonania_list: list[str] | None,
) -> list[str]:
    return _normalize_date_list(terminy_wykonania_list, "terminyWykonaniaZalecenList") if terminy_wykonania_list is not None else []


def _recommendation_code_year(
    data_zalecen: str | None,
    inspection_start: str | None,
    terminy_wykonania_list: list[str],
) -> str:
    year_source = data_zalecen
    if not year_source:
        year_source = terminy_wykonania_list[0] if terminy_wykonania_list else None
    if not year_source:
        year_source = inspection_start

    raw = str(year_source or "").strip()
    if len(raw) >= 4 and raw[:4].isdigit():
        return raw[:4]
    return str(date.today().year)


def _next_recommendation_code(conn: Any, year: str) -> str:
    prefix = "Z"
    pattern = f"{prefix}/{year}/%"
    rows = conn.execute(
        "SELECT kod_zalecenia FROM recommendations WHERE kod_zalecenia LIKE ?",
        (pattern,),
    ).fetchall()

    max_seq = 0
    for row in rows:
        raw = str(row["kod_zalecenia"] or "").strip()
        parts = raw.split("/")
        if len(parts) != 3 or parts[0] != prefix or parts[1] != year:
            continue
        try:
            seq = int(parts[2])
        except ValueError:
            continue
        max_seq = max(max_seq, seq)

    return f"{prefix}/{year}/{max_seq + 1}"


def _sync_multi_dates(
    conn: Any,
    recommendation_id: int,
    date_type: Literal["TERMIN_WYKONANIA_ZALECEN", "AKCEPTACJA_NOTY_WERYFIKACJI"],
    target_dates: list[str],
    operator_user_id: int,
) -> None:
    existing_rows = conn.execute(
        """
        SELECT id, date_value
        FROM recommendation_multi_dates
        WHERE recommendation_id = ? AND date_type = ?
        """,
        (recommendation_id, date_type),
    ).fetchall()

    existing_by_value = {str(r["date_value"]): int(r["id"]) for r in existing_rows}
    target_set = set(target_dates)
    existing_set = set(existing_by_value.keys())

    to_delete = sorted(existing_set - target_set)
    to_insert = sorted(target_set - existing_set)

    for date_value in to_delete:
        conn.execute(
            "DELETE FROM recommendation_multi_dates WHERE id = ?",
            (existing_by_value[date_value],),
        )

    for date_value in to_insert:
        conn.execute(
            """
            INSERT INTO recommendation_multi_dates
            (recommendation_id, date_type, date_value, created_by_user_id, updated_by_user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (recommendation_id, date_type, date_value, operator_user_id, operator_user_id),
        )


def _parse_dates_csv(csv_value: str | None) -> list[str]:
    if not csv_value:
        return []
    return [part for part in str(csv_value).split(",") if part]


def _first_date_or_none(values: list[str]) -> str | None:
    return values[0] if values else None


def _sync_inspection_recommendation_dates(
    conn: Any,
    inspection_id: int | None,
    operator_user_id: int,
) -> None:
    if inspection_id is None:
        return

    rows = conn.execute(
        """
                SELECT DISTINCT r.data_zalecen AS date_value
                FROM recommendations r
        WHERE r.inspection_id = ?
                    AND r.data_zalecen IS NOT NULL
                    AND trim(r.data_zalecen) <> ''
                ORDER BY r.data_zalecen ASC
        """,
        (inspection_id,),
    ).fetchall()
    target_dates = [str(r["date_value"]) for r in rows]

    existing_rows = conn.execute(
        """
        SELECT id, date_value
        FROM inspection_multi_dates
        WHERE inspection_id = ?
          AND date_type = 'ZALECENIE'
        """,
        (inspection_id,),
    ).fetchall()
    existing_by_value = {str(r["date_value"]): int(r["id"]) for r in existing_rows}

    target_set = set(target_dates)
    existing_set = set(existing_by_value.keys())

    to_delete = sorted(existing_set - target_set)
    to_insert = sorted(target_set - existing_set)

    for date_value in to_delete:
        conn.execute(
            "DELETE FROM inspection_multi_dates WHERE id = ?",
            (existing_by_value[date_value],),
        )

    for date_value in to_insert:
        conn.execute(
            """
            INSERT INTO inspection_multi_dates
            (inspection_id, date_type, date_value, created_by_user_id, updated_by_user_id)
            VALUES (?, 'ZALECENIE', ?, ?, ?)
            """,
            (inspection_id, date_value, operator_user_id, operator_user_id),
        )

    latest_date = target_dates[-1] if target_dates else None
    conn.execute(
        "UPDATE inspections SET data_zalecen = ? WHERE id = ?",
        (latest_date, inspection_id),
    )


def _can_edit_recommendation(
    conn: Any,
    inspection_id: int | None,
    operator: dict[str, Any],
    created_by_user_id: int | None,
) -> bool:
    # dyrektor: pelny dostep do wszystkich inspekcji
    if operator["rola_id"] == 3:
        return True

    if inspection_id is None:
        if created_by_user_id is None:
            return False

        if int(created_by_user_id) == operator["id"]:
            return True

        if operator["rola_id"] == 2:
            author_row = conn.execute(
                "SELECT zespol_id, created_by_user_id FROM users WHERE id = ? LIMIT 1",
                (int(created_by_user_id),),
            ).fetchone()
            if author_row is None:
                return False
            created_by = author_row["created_by_user_id"]
            if created_by is not None and int(created_by) == int(operator["id"]):
                return True
            if operator["zespol_id"] is None or author_row["zespol_id"] is None:
                return False
            return int(author_row["zespol_id"]) == int(operator["zespol_id"])

        return False

    if created_by_user_id is None:
        inspection_row = conn.execute(
            "SELECT created_by_user_id FROM inspections WHERE id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()
        if inspection_row is not None and inspection_row["created_by_user_id"] is not None:
            created_by_user_id = int(inspection_row["created_by_user_id"])

    # kierownik: moze dodawac/edytowac zalecenia dla inspekcji,
    # gdzie osoba kierujaca LUB dowolny czlonek skladu jest z jego zespolu.
    if operator["rola_id"] == 2:
        if created_by_user_id is not None:
            author_row = conn.execute(
                "SELECT zespol_id, created_by_user_id FROM users WHERE id = ? LIMIT 1",
                (int(created_by_user_id),),
            ).fetchone()
            if author_row is not None:
                author_created_by = author_row["created_by_user_id"]
                if author_created_by is not None and int(author_created_by) == int(operator["id"]):
                    return True
                if operator["zespol_id"] is not None and author_row["zespol_id"] is not None:
                    if int(author_row["zespol_id"]) == int(operator["zespol_id"]):
                        return True

        leader_row = conn.execute(
            """
            SELECT 1
            FROM inspections i
            JOIN users u ON u.id = i.osoba_kierujaca_user_id
            WHERE i.id = ?
                            AND (
                                        (u.zespol_id = ?)
                                        OR (u.created_by_user_id = ?)
                                    )
            LIMIT 1
            """,
                        (inspection_id, operator["zespol_id"], int(operator["id"])),
        ).fetchone()
        if leader_row is not None:
            return True

        team_member_row = conn.execute(
            """
            SELECT 1
            FROM inspection_members im
            JOIN users u ON u.id = im.user_id
            WHERE im.inspection_id = ?
                            AND (
                                        (u.zespol_id = ?)
                                        OR (u.created_by_user_id = ?)
                                    )
            LIMIT 1
            """,
                        (inspection_id, operator["zespol_id"], int(operator["id"])),
        ).fetchone()
        return team_member_row is not None

    # zwykly uzytkownik: inspekcje utworzone przez siebie lub takie, w ktorych jest czlonkiem
    if created_by_user_id is not None and int(created_by_user_id) == operator["id"]:
        return True

    member_row = conn.execute(
        "SELECT 1 FROM inspection_members WHERE inspection_id = ? AND user_id = ? LIMIT 1",
        (inspection_id, operator["id"]),
    ).fetchone()
    return member_row is not None


def _base_select_sql() -> str:
    return """
        SELECT
            r.id,
            r.inspection_id,
            (
                SELECT group_concat(x.tid, ',')
                FROM (
                    SELECT rt.team_id AS tid
                    FROM recommendation_teams rt
                    WHERE rt.recommendation_id = r.id
                    ORDER BY rt.team_id ASC
                ) x
            ) AS inspection_team_ids_csv,
            i.lp AS inspection_lp,
            i.kod_inspekcji AS inspection_kod,
            r.kod_zalecenia,
            r.created_by_user_id,
            r.pozycja,
            r.nazwa_podmiotu_id,
            np.nazwa_pozycji AS nazwa_podmiotu_nazwa,
            np.skrot_pozycji AS nazwa_podmiotu_skrot,
            r.data_zalecen,
            r.brak_terminow_wykonania_zalecen,
            r.brak_dat_akceptacji_noty_weryfikacji,
            r.status_zalecenia_id,
            st.nazwa_pozycji AS status_nazwa,
            st.skrot_pozycji AS status_skrot,
            r.komentarz,
            r.utworzono_o,
            r.zaktualizowano_o,
            (
                SELECT group_concat(x.dv, ',')
                FROM (
                    SELECT rmd.date_value AS dv
                    FROM recommendation_multi_dates rmd
                    WHERE rmd.recommendation_id = r.id
                                            AND rmd.date_type = 'TERMIN_WYKONANIA_ZALECEN'
                    ORDER BY rmd.date_value ASC
                ) x
            ) AS data_zalecen_list_csv,
            (
                SELECT group_concat(x.dv, ',')
                FROM (
                    SELECT rmd.date_value AS dv
                    FROM recommendation_multi_dates rmd
                    WHERE rmd.recommendation_id = r.id
                      AND rmd.date_type = 'AKCEPTACJA_NOTY_WERYFIKACJI'
                    ORDER BY rmd.date_value ASC
                ) x
            ) AS data_akceptacji_weryfikacji_list_csv
        FROM recommendations r
        LEFT JOIN inspections i ON i.id = r.inspection_id
        LEFT JOIN slownik_pozycje np ON np.id = r.nazwa_podmiotu_id
        LEFT JOIN slownik_pozycje st ON st.id = r.status_zalecenia_id
    """


def _row_to_payload(row: dict[str, Any], lp: int, can_edit: bool) -> dict[str, Any]:
    terminy_wykonania_list = _parse_dates_csv(row.get("data_zalecen_list_csv"))
    data_zalecen_single = row.get("data_zalecen")
    termin_wykonania_single = _first_date_or_none(terminy_wykonania_list)
    brak_terminow = int(row.get("brak_terminow_wykonania_zalecen") or 0) == 1
    brak_akceptacji = int(row.get("brak_dat_akceptacji_noty_weryfikacji") or 0) == 1
    inspection_team_ids_csv = row.get("inspection_team_ids_csv")
    inspection_team_ids: list[int] = []
    if inspection_team_ids_csv:
        inspection_team_ids = [int(x) for x in str(inspection_team_ids_csv).split(",") if x.strip()]
    return {
        "id": int(row["id"]),
        "lp": lp,
        "kodZalecenia": row.get("kod_zalecenia"),
        "canEdit": can_edit,
        "inspectionId": int(row["inspection_id"]) if row.get("inspection_id") is not None else None,
        "inspectionTeamIds": inspection_team_ids,
        "inspectionLp": int(row["inspection_lp"]) if row.get("inspection_lp") is not None else None,
        "inspectionKod": row.get("inspection_kod"),
        "pozycja": int(row["pozycja"]),
        "nazwaPodmiotuId": int(row["nazwa_podmiotu_id"]) if row.get("nazwa_podmiotu_id") is not None else None,
        "nazwaPodmiotu": row.get("nazwa_podmiotu_nazwa") or "brak",
        "nazwaPodmiotuSkrocona": row.get("nazwa_podmiotu_skrot"),
        "nazwaPodmiotuSkrot": row.get("nazwa_podmiotu_skrot"),
        "dataZalecen": data_zalecen_single,
        "terminyWykonaniaZalecenList": terminy_wykonania_list,
        # Legacy aliases.
        "terminWykonaniaZalecen": termin_wykonania_single,
        "statusId": int(row["status_zalecenia_id"]) if row.get("status_zalecenia_id") is not None else None,
        "status": row.get("status_nazwa") or "brak",
        "statusSkrocona": row.get("status_skrot"),
        "statusSkrot": row.get("status_skrot"),
        "komentarz": row.get("komentarz"),
        "dataZalecenList": terminy_wykonania_list,
        "dataAkceptacjiNotyWeryfikacjiList": _parse_dates_csv(row.get("data_akceptacji_weryfikacji_list_csv")),
        "brakTerminowWykonaniaZalecen": brak_terminow,
        "brakDatAkceptacjiNotyWeryfikacji": brak_akceptacji,
        "utworzonoO": row.get("utworzono_o"),
        "zaktualizowanoO": row.get("zaktualizowano_o"),
    }


@router.get(
    "/api/recommendations/available-inspections",
    response_model=list[RecommendationInspectionOption],
)
def list_available_inspections_for_recommendations(
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)
        require_permission(conn, operator, PERMISSION_RECOMMENDATIONS_READ)
        rows = conn.execute(
            """
            SELECT
                i.id,
                i.lp,
                i.kod_inspekcji,
                np.nazwa_pozycji AS nazwa_podmiotu_nazwa,
                np.skrot_pozycji AS nazwa_podmiotu_skrot,
                i.poczatek_inspekcji,
                i.koniec_inspekcji,
                i.osoba_kierujaca_user_id,
                trim(ulead.imie || ' ' || ulead.nazwisko) AS osoba_kierujaca_full,
                ulead.login AS osoba_kierujaca_login
            FROM inspections i
            LEFT JOIN slownik_pozycje np ON np.id = i.nazwa_podmiotu_id
            LEFT JOIN users ulead ON ulead.id = i.osoba_kierujaca_user_id
            ORDER BY i.lp ASC, i.id ASC
            """
        ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            inspection_id = int(row_dict["id"])
            if not _can_edit_recommendation(conn, inspection_id, operator, None):
                continue
            blocked, _, _ = _inspection_status_is_forbidden(conn, inspection_id, operator)
            if blocked:
                continue

            leader_name = _norm(row_dict.get("osoba_kierujaca_full")) or row_dict.get("osoba_kierujaca_login")
            items.append(
                {
                    "id": inspection_id,
                    "lp": int(row_dict["lp"]),
                    "kodInspekcji": row_dict.get("kod_inspekcji"),
                    "nazwaPodmiotu": row_dict.get("nazwa_podmiotu_nazwa") or "brak",
                    "nazwaPodmiotuSkrocona": row_dict.get("nazwa_podmiotu_skrot"),
                    "nazwaPodmiotuSkrot": row_dict.get("nazwa_podmiotu_skrot"),
                    "poczatekInspekcji": row_dict.get("poczatek_inspekcji"),
                    "koniecInspekcji": row_dict.get("koniec_inspekcji"),
                    "osobaKierujacaUserId": row_dict.get("osoba_kierujaca_user_id"),
                    "osobaKierujaca": leader_name,
                }
            )

    return items


@router.get("/api/recommendations", response_model=RecommendationListResponse)
def list_recommendations(
    inspectionId: int | None = Query(default=None),
    status: str | None = Query(default=None),
    nazwaPodmiotu: str | None = Query(default=None),
    sortBy: str = Query(default="dataZalecen"),
    sortOrder: str = Query(default="desc"),
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    allowed_sort_columns = {
        "id": "r.id",
        "pozycja": "r.pozycja",
        "dataZalecen": "r.data_zalecen",
        "terminWykonaniaZalecen": "(SELECT MIN(rmd.date_value) FROM recommendation_multi_dates rmd WHERE rmd.recommendation_id = r.id AND rmd.date_type = 'TERMIN_WYKONANIA_ZALECEN')",
        "utworzonoO": "r.utworzono_o",
        "zaktualizowanoO": "r.zaktualizowano_o",
    }
    if sortBy not in allowed_sort_columns:
        raise HTTPException(status_code=400, detail="Niepoprawny sortBy")

    direction = sortOrder.lower()
    if direction not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="Niepoprawny sortOrder")

    where_parts: list[str] = []
    params: list[Any] = []

    if inspectionId is not None:
        where_parts.append("r.inspection_id = ?")
        params.append(int(inspectionId))

    status_norm = _norm(status)
    if status_norm is not None:
        where_parts.append("lower(st.nazwa_pozycji) LIKE lower(?)")
        params.append(f"%{status_norm}%")

    podmiot_norm = _norm(nazwaPodmiotu)
    if podmiot_norm is not None:
        where_parts.append("lower(np.nazwa_pozycji) LIKE lower(?)")
        params.append(f"%{podmiot_norm}%")

    where_sql = ""
    if where_parts:
        where_sql = " WHERE " + " AND ".join(where_parts)

    order_sql = f" ORDER BY {allowed_sort_columns[sortBy]} {direction.upper()}, r.id DESC"

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_RECOMMENDATIONS_READ)
        rows = conn.execute(_base_select_sql() + where_sql + order_sql, tuple(params)).fetchall()

        items: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            row_dict = dict(row)
            inspection_id = int(row_dict["inspection_id"]) if row_dict.get("inspection_id") is not None else None
            created_by = int(row_dict["created_by_user_id"]) if row_dict.get("created_by_user_id") is not None else None
            can_edit = _can_edit_recommendation(conn, inspection_id, operator, created_by)
            items.append(_row_to_payload(row_dict, lp=index, can_edit=can_edit))

    return {"items": items, "total": len(items)}


@router.get("/api/recommendations/{recommendation_id}", response_model=RecommendationRead)
def get_recommendation(
    recommendation_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_RECOMMENDATIONS_READ)
        row = conn.execute(
            _base_select_sql() + " WHERE r.id = ? LIMIT 1",
            (recommendation_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Recommendation not found")

        row_dict = dict(row)
        inspection_id = int(row_dict["inspection_id"]) if row_dict.get("inspection_id") is not None else None
        created_by = int(row_dict["created_by_user_id"]) if row_dict.get("created_by_user_id") is not None else None
        can_edit = _can_edit_recommendation(conn, inspection_id, operator, created_by)

    return _row_to_payload(row_dict, lp=1, can_edit=can_edit)


@router.post(
    "/api/recommendations",
    response_model=RecommendationRead,
    status_code=201,
    responses={
        400: {"content": {"application/json": {"example": {"detail": "Bledne dane"}}}},
        401: {"content": {"application/json": {"example": {"detail": "Operator nie istnieje"}}}},
        403: {"content": {"application/json": {"example": {"detail": "Brak uprawnien"}}}},
        404: {"content": {"application/json": {"example": {"detail": "Inspection not found"}}}},
    },
)
def create_recommendation(
    payload: RecommendationCreate,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    terminy_wykonania_list = _resolve_terminy_wykonania_list(
        terminy_wykonania_list=payload.terminyWykonaniaZalecenList,
    )
    data_akceptacji_list = _normalize_date_list(
        payload.dataAkceptacjiNotyWeryfikacjiList,
        "dataAkceptacjiNotyWeryfikacjiList",
    )
    data_zalecen_single = _resolve_single_data_zalecen(
        data_zalecen=payload.dataZalecen,
    )
    terminy_wykonania_list, brak_terminow = _normalize_date_absence_state(
        dates=terminy_wykonania_list,
        brak=bool(payload.brakTerminowWykonaniaZalecen),
        list_field_name="terminyWykonaniaZalecenList",
        bool_field_name="brakTerminowWykonaniaZalecen",
    )
    data_akceptacji_list, brak_akceptacji = _normalize_date_absence_state(
        dates=data_akceptacji_list,
        brak=bool(payload.brakDatAkceptacjiNotyWeryfikacji),
        list_field_name="dataAkceptacjiNotyWeryfikacjiList",
        bool_field_name="brakDatAkceptacjiNotyWeryfikacji",
    )
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)

        resolved_inspection_id: int | None = None
        resolved_nazwa_podmiotu_id: int | None = None
        inspection_row = None

        if payload.inspectionId is not None:
            inspection_row = conn.execute(
                "SELECT id, nazwa_podmiotu_id, created_by_user_id, poczatek_inspekcji FROM inspections WHERE id = ? LIMIT 1",
                (payload.inspectionId,),
            ).fetchone()
            if inspection_row is None:
                raise HTTPException(status_code=404, detail="Inspection not found")

            resolved_inspection_id = int(inspection_row["id"])
            if not _can_edit_recommendation(
                conn,
                resolved_inspection_id,
                operator,
                int(inspection_row["created_by_user_id"]) if inspection_row["created_by_user_id"] is not None else None,
            ):
                raise HTTPException(status_code=403, detail="Brak uprawnien do tej inspekcji")

            blocked, status_code, status_label = _inspection_status_is_forbidden(conn, resolved_inspection_id, operator)
            if blocked:
                _raise_inspection_status_block(resolved_inspection_id, status_code, status_label)

            if inspection_row["nazwa_podmiotu_id"] is not None:
                resolved_nazwa_podmiotu_id = int(inspection_row["nazwa_podmiotu_id"])
            else:
                resolved_nazwa_podmiotu_id = _resolve_dictionary_id_strict(
                    conn,
                    kod_typu="nazwy_podmiotow",
                    field_name="nazwaPodmiotuId",
                    id_value=payload.nazwaPodmiotuId,
                    required=True,
                )
        else:
            resolved_nazwa_podmiotu_id = _resolve_dictionary_id_strict(
                conn,
                kod_typu="nazwy_podmiotow",
                field_name="nazwaPodmiotuId",
                id_value=payload.nazwaPodmiotuId,
                required=True,
            )

        status_id = _resolve_dictionary_id_strict(
            conn,
            kod_typu="statusy_zalecen",
            field_name="statusId",
            id_value=payload.statusId,
            required=True,
        )
        inspection_team_ids = _normalize_and_validate_recommendation_team_ids(
            conn,
            team_ids=payload.inspectionTeamIds,
            team_ids_legacy=payload.inspection_team_ids,
            field_sent=(payload.inspectionTeamIds is not None or payload.inspection_team_ids is not None),
        )
        inspection_start = str(inspection_row["poczatek_inspekcji"]) if inspection_row is not None else None
        recommendation_year = _recommendation_code_year(data_zalecen_single, inspection_start, terminy_wykonania_list)

        cursor = conn.execute(
            """
            INSERT INTO recommendations (
                inspection_id,
                pozycja,
                kod_zalecenia,
                nazwa_podmiotu_id,
                data_zalecen,
                brak_terminow_wykonania_zalecen,
                brak_dat_akceptacji_noty_weryfikacji,
                status_zalecenia_id,
                komentarz,
                created_by_user_id,
                updated_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_inspection_id,
                payload.pozycja,
                _next_recommendation_code(conn, recommendation_year),
                resolved_nazwa_podmiotu_id,
                data_zalecen_single,
                1 if brak_terminow else 0,
                1 if brak_akceptacji else 0,
                status_id,
                payload.komentarz,
                operator["id"],
                operator["id"],
            ),
        )
        recommendation_id = int(cursor.lastrowid)
        _sync_recommendation_teams(conn, recommendation_id, inspection_team_ids)

        _sync_multi_dates(conn, recommendation_id, "TERMIN_WYKONANIA_ZALECEN", terminy_wykonania_list, operator["id"])
        _sync_multi_dates(
            conn,
            recommendation_id,
            "AKCEPTACJA_NOTY_WERYFIKACJI",
            data_akceptacji_list,
            operator["id"],
        )
        _sync_inspection_recommendation_dates(conn, resolved_inspection_id, operator["id"])
        kod_row = conn.execute(
            "SELECT kod_zalecenia FROM recommendations WHERE id = ? LIMIT 1",
            (recommendation_id,),
        ).fetchone()
        rekord_kod_cr = str(kod_row["kod_zalecenia"]) if kod_row and kod_row["kod_zalecenia"] else str(recommendation_id)
        row = conn.execute(
            _base_select_sql() + " WHERE r.id = ? LIMIT 1",
            (recommendation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="Failed to fetch created recommendation")

        created_payload = _row_to_payload(dict(row), lp=1, can_edit=True)
        changes = build_create_changes(
            [
                ("Kod zalecenia", created_payload.get("kodZalecenia")),
                ("Kod inspekcji", created_payload.get("inspectionKod")),
                ("Zespoły inspekcji", created_payload.get("inspectionTeamIds")),
                ("Pozycja", created_payload.get("pozycja")),
                ("Nazwa podmiotu", created_payload.get("nazwaPodmiotu")),
                ("Data zaleceń", created_payload.get("dataZalecen")),
                ("Terminy wykonania zaleceń", created_payload.get("terminyWykonaniaZalecenList")),
                ("Daty akceptacji noty weryfikacji", created_payload.get("dataAkceptacjiNotyWeryfikacjiList")),
                ("Status", created_payload.get("status")),
                ("Komentarz", created_payload.get("komentarz")),
            ]
        )

        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_CREATE,
                        REJESTR_ZALECENIA, rekord_kod_cr, changes)
        conn.commit()

    if row is None:
        raise HTTPException(status_code=500, detail="Failed to fetch created recommendation")

    return _row_to_payload(dict(row), lp=1, can_edit=True)


@router.put("/api/recommendations/{recommendation_id}", response_model=RecommendationRead)
def update_recommendation(
    recommendation_id: int,
    payload: RecommendationUpdate,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    fields = payload.model_dump(exclude_unset=True)
    lock_token = str(fields.pop("lockToken", "") or "")
    expected_updated_at = fields.pop("expectedUpdatedAt", None)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT id, inspection_id, created_by_user_id, nazwa_podmiotu_id, status_zalecenia_id, data_zalecen, pozycja, komentarz, kod_zalecenia, brak_terminow_wykonania_zalecen, brak_dat_akceptacji_noty_weryfikacji, zaktualizowano_o FROM recommendations WHERE id = ? LIMIT 1",
            (recommendation_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail="Recommendation not found")

        current_dict = dict(current)
        inspection_id = int(current_dict["inspection_id"]) if current_dict.get("inspection_id") is not None else None
        created_by = int(current_dict["created_by_user_id"]) if current_dict.get("created_by_user_id") is not None else None
        if not _can_edit_recommendation(conn, inspection_id, operator, created_by):
            raise HTTPException(status_code=403, detail="Brak uprawnien do tej inspekcji")

        assert_lock_for_save(conn, "recommendations", recommendation_id, operator, lock_token)
        assert_expected_updated_at(expected_updated_at, str(current_dict.get("zaktualizowano_o") or ""))

        # Pobierz daty przed zmianą (dla audit log)
        _dz_before = conn.execute(
            "SELECT date_value FROM recommendation_multi_dates WHERE recommendation_id=? AND date_type='TERMIN_WYKONANIA_ZALECEN' ORDER BY date_value",
            (recommendation_id,),
        ).fetchall()
        _da_before = conn.execute(
            "SELECT date_value FROM recommendation_multi_dates WHERE recommendation_id=? AND date_type='AKCEPTACJA_NOTY_WERYFIKACJI' ORDER BY date_value",
            (recommendation_id,),
        ).fetchall()
        dates_z_before = ", ".join(r["date_value"] for r in _dz_before) or None
        dates_a_before = ", ".join(r["date_value"] for r in _da_before) or None

        previous_inspection_id = inspection_id
        next_inspection_id = inspection_id
        next_nazwa_podmiotu_id = current_dict.get("nazwa_podmiotu_id")
        inspection_id_changed = False
        current_team_ids = _get_recommendation_team_ids(conn, recommendation_id)
        current_dict["_inspection_team_ids"] = ", ".join(str(x) for x in current_team_ids) or None
        next_team_ids = current_team_ids
        teams_updated = False

        set_parts: list[str] = []
        values: list[Any] = []
        row_touched = False

        if "inspectionId" in fields:
            if fields["inspectionId"] is None:
                next_inspection_id = None
            else:
                target_inspection = conn.execute(
                    "SELECT id, nazwa_podmiotu_id, created_by_user_id FROM inspections WHERE id = ? LIMIT 1",
                    (int(fields["inspectionId"]),),
                ).fetchone()
                if target_inspection is None:
                    raise HTTPException(status_code=404, detail="Inspection not found")
                target_inspection_id = int(target_inspection["id"])
                target_created_by = (
                    int(target_inspection["created_by_user_id"])
                    if target_inspection["created_by_user_id"] is not None
                    else None
                )
                if not _can_edit_recommendation(conn, target_inspection_id, operator, target_created_by):
                    raise HTTPException(status_code=403, detail="Brak uprawnien do tej inspekcji")

                next_inspection_id = target_inspection_id
                if target_inspection["nazwa_podmiotu_id"] is not None:
                    next_nazwa_podmiotu_id = int(target_inspection["nazwa_podmiotu_id"])

            set_parts.append("inspection_id = ?")
            values.append(next_inspection_id)
            inspection_id_changed = next_inspection_id != previous_inspection_id

        # Closed-inspection guard applies only when switching target inspection.
        if inspection_id_changed and next_inspection_id is not None:
            blocked, status_code, status_label = _inspection_status_is_forbidden(conn, next_inspection_id, operator)
            if blocked:
                _raise_inspection_status_block(next_inspection_id, status_code, status_label)

        if "pozycja" in fields:
            if fields["pozycja"] is None or int(fields["pozycja"]) < 1:
                _raise_contract_422(
                    "INVALID_VALUE",
                    "pozycja musi byc >= 1.",
                    field="pozycja",
                    value=fields.get("pozycja"),
                )
            set_parts.append("pozycja = ?")
            values.append(int(fields["pozycja"]))

        if "nazwaPodmiotuId" in fields:
            nazwa_id = _resolve_dictionary_id_strict(
                conn,
                kod_typu="nazwy_podmiotow",
                field_name="nazwaPodmiotuId",
                id_value=fields.get("nazwaPodmiotuId"),
                required=next_inspection_id is None,
            )
            next_nazwa_podmiotu_id = nazwa_id
            set_parts.append("nazwa_podmiotu_id = ?")
            values.append(nazwa_id)

        if next_inspection_id is None and next_nazwa_podmiotu_id is None:
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                "Pole nazwaPodmiotuId jest wymagane gdy inspectionId jest null.",
                field="nazwaPodmiotuId",
                value=None,
                kod_typu="nazwy_podmiotow",
            )

        single_date_payload_present = "dataZalecen" in fields
        if single_date_payload_present:
            resolved_data_zalecen = _resolve_single_data_zalecen(
                data_zalecen=fields.get("dataZalecen"),
            )
            set_parts.append("data_zalecen = ?")
            values.append(resolved_data_zalecen)

        if "statusId" in fields:
            current_status_id = (
                int(current_dict["status_zalecenia_id"])
                if current_dict.get("status_zalecenia_id") is not None
                else None
            )

            status_id = _resolve_dictionary_id_strict(
                conn,
                kod_typu="statusy_zalecen",
                field_name="statusId",
                id_value=fields.get("statusId"),
                required=True,
            )
            if current_status_id != int(status_id):
                set_parts.append("status_zalecenia_id = ?")
                values.append(status_id)
            else:
                fields.pop("statusId", None)

        if "komentarz" in fields:
            set_parts.append("komentarz = ?")
            values.append(fields["komentarz"])

        if "inspectionTeamIds" in fields or "inspection_team_ids" in fields:
            next_team_ids = _normalize_and_validate_recommendation_team_ids(
                conn,
                team_ids=fields.get("inspectionTeamIds"),
                team_ids_legacy=fields.get("inspection_team_ids"),
                field_sent=True,
            )
            teams_updated = next_team_ids != current_team_ids

        current_brak_terminow = int(current_dict.get("brak_terminow_wykonania_zalecen") or 0) == 1
        current_brak_akceptacji = int(current_dict.get("brak_dat_akceptacji_noty_weryfikacji") or 0) == 1

        if set_parts:
            set_parts.append("updated_by_user_id = ?")
            values.append(operator["id"])
            set_parts.append("zaktualizowano_o = ?")
            values.append(now_rfc3339_utc_ms())
            values.append(recommendation_id)
            conn.execute(
                f"UPDATE recommendations SET {', '.join(set_parts)} WHERE id = ?",
                tuple(values),
            )
            row_touched = True

        multi_dates_payload_present = "terminyWykonaniaZalecenList" in fields
        bool_columns_update: dict[str, int] = {}
        if multi_dates_payload_present:
            terminy_wykonania = _resolve_terminy_wykonania_list(
                terminy_wykonania_list=fields.get("terminyWykonaniaZalecenList"),
            )
            terminy_wykonania, brak_terminow_after = _normalize_date_absence_state(
                dates=terminy_wykonania,
                brak=bool(fields.get("brakTerminowWykonaniaZalecen", current_brak_terminow)),
                list_field_name="terminyWykonaniaZalecenList",
                bool_field_name="brakTerminowWykonaniaZalecen",
            )
            _sync_multi_dates(conn, recommendation_id, "TERMIN_WYKONANIA_ZALECEN", terminy_wykonania, operator["id"])
            bool_columns_update["brak_terminow_wykonania_zalecen"] = 1 if brak_terminow_after else 0
        elif "brakTerminowWykonaniaZalecen" in fields:
            brak_terminow_after = bool(fields["brakTerminowWykonaniaZalecen"])
            if brak_terminow_after:
                _sync_multi_dates(conn, recommendation_id, "TERMIN_WYKONANIA_ZALECEN", [], operator["id"])
            bool_columns_update["brak_terminow_wykonania_zalecen"] = 1 if brak_terminow_after else 0

        if "dataAkceptacjiNotyWeryfikacjiList" in fields:
            data_akceptacji = _normalize_date_list(
                fields["dataAkceptacjiNotyWeryfikacjiList"],
                "dataAkceptacjiNotyWeryfikacjiList",
            )
            data_akceptacji, brak_akceptacji_after = _normalize_date_absence_state(
                dates=data_akceptacji,
                brak=bool(fields.get("brakDatAkceptacjiNotyWeryfikacji", current_brak_akceptacji)),
                list_field_name="dataAkceptacjiNotyWeryfikacjiList",
                bool_field_name="brakDatAkceptacjiNotyWeryfikacji",
            )
            _sync_multi_dates(
                conn,
                recommendation_id,
                "AKCEPTACJA_NOTY_WERYFIKACJI",
                data_akceptacji,
                operator["id"],
            )
            bool_columns_update["brak_dat_akceptacji_noty_weryfikacji"] = 1 if brak_akceptacji_after else 0
        elif "brakDatAkceptacjiNotyWeryfikacji" in fields:
            brak_akceptacji_after = bool(fields["brakDatAkceptacjiNotyWeryfikacji"])
            if brak_akceptacji_after:
                _sync_multi_dates(conn, recommendation_id, "AKCEPTACJA_NOTY_WERYFIKACJI", [], operator["id"])
            bool_columns_update["brak_dat_akceptacji_noty_weryfikacji"] = 1 if brak_akceptacji_after else 0

        if bool_columns_update:
            set_sql = ", ".join(f"{column} = ?" for column in bool_columns_update.keys())
            conn.execute(
                f"UPDATE recommendations SET {set_sql}, updated_by_user_id = ?, zaktualizowano_o = ? WHERE id = ?",
                (*bool_columns_update.values(), operator["id"], now_rfc3339_utc_ms(), recommendation_id),
            )
            row_touched = True

        for affected_inspection_id in {previous_inspection_id, next_inspection_id}:
            _sync_inspection_recommendation_dates(conn, affected_inspection_id, operator["id"])

        if teams_updated:
            _sync_recommendation_teams(conn, recommendation_id, next_team_ids)
            conn.execute(
                "UPDATE recommendations SET updated_by_user_id = ?, zaktualizowano_o = ? WHERE id = ?",
                (operator["id"], now_rfc3339_utc_ms(), recommendation_id),
            )
            row_touched = True

        if not row_touched and (
            multi_dates_payload_present or "dataAkceptacjiNotyWeryfikacjiList" in fields
        ):
            conn.execute(
                "UPDATE recommendations SET updated_by_user_id = ?, zaktualizowano_o = ? WHERE id = ?",
                (operator["id"], now_rfc3339_utc_ms(), recommendation_id),
            )

        # --- Audit log ---
        rekord_kod_u = str(current_dict.get("kod_zalecenia") or recommendation_id)
        audit_fields = dict(fields)
        if single_date_payload_present:
            audit_fields["dataZalecen"] = resolved_data_zalecen
        if multi_dates_payload_present:
            audit_fields["terminyWykonaniaZalecenList"] = terminy_wykonania
        if teams_updated:
            audit_fields["inspectionTeamIds"] = next_team_ids
        changes_r = build_recommendation_changes(conn, current_dict, audit_fields, dates_z_before, dates_a_before)
        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_UPDATE,
                        REJESTR_ZALECENIA, rekord_kod_u, changes_r)
        # --- koniec audit log ---

        conn.commit()

        row = conn.execute(
            _base_select_sql() + " WHERE r.id = ? LIMIT 1",
            (recommendation_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    return _row_to_payload(dict(row), lp=1, can_edit=True)


@router.delete("/api/recommendations/{recommendation_id}")
def delete_recommendation(
    recommendation_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, bool]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)
        _ensure_director(operator)
        row = conn.execute(
            "SELECT id, inspection_id, created_by_user_id, kod_zalecenia FROM recommendations WHERE id = ? LIMIT 1",
            (recommendation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Recommendation not found")

        inspection_id = int(row["inspection_id"]) if row["inspection_id"] is not None else None

        conn.execute("DELETE FROM recommendations WHERE id = ?", (recommendation_id,))
        _sync_inspection_recommendation_dates(conn, inspection_id, operator["id"])
        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_DELETE,
                        REJESTR_ZALECENIA, str(row["kod_zalecenia"] or recommendation_id), [])
        conn.commit()

    return {"ok": True}
