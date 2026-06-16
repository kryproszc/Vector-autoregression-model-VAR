from __future__ import annotations

from datetime import date
import os
import re
import unicodedata
from typing import Any, Literal, Sequence

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.audit import (
    AKCJA_CREATE,
    AKCJA_DELETE,
    AKCJA_UPDATE,
    REJESTR_WNIOSKI,
    build_create_changes,
    build_risk_exposure_changes,
    new_session_id,
    write_audit_log,
)
from app.database import get_connection
from app.permissions import PERMISSION_RISK_EXPOSURE_READ, require_permission, require_write_access
from app.record_locks import assert_expected_updated_at, assert_lock_for_save, now_rfc3339_utc_ms

router = APIRouter()

ROZSTRZYGNIECIE_WNIOSKU_KOD_TYPU = "rozstrzygniecie_wniosku_sankcyjnego_I"
FORBIDDEN_INSPECTION_STATUS_CODES = {
  
}
FORBIDDEN_INSPECTION_STATUS_LABEL_KEYS = {
    
}
INSPECTION_STATUS_BLOCK_ERROR_CODE = "INSPECTION_STATUS_BLOCKS_OPERATION"


class RiskExposureCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inspectionId": 1,
                "nazwaPodmiotuObjetegoInspekcja": "Podmiot X",
                "nazwaPodmiotuObjetegoSankcjaList": ["Podmiot Y", "Podmiot Z"],
                "dataWniosku": "2026-12-01",
                "wniosekDo": "Departament A",
                "sankcjaList": ["Kara pieniezna"],
                "podstawaPrawnaSankcjiList": ["Art. 12"],
                "naruszeniaSkutkujaceSankcjaList": ["Brak raportu"],
                "czyMamyInformacjeOWszczeciuPostepowania": "Tak",
                "rozstrzygniecie": "W toku",
                "komentarz": "Komentarz",
            }
        }
    )

    inspectionId: int | None = None
    nazwaPodmiotuObjetegoInspekcjaId: int | None = None
    nazwaPodmiotuObjetegoInspekcja: str | None = None
    nazwaPodmiotuObjetegoSankcjaIds: list[int] | None = None
    nazwaPodmiotuObjetegoSankcjaList: list[str] | None = None
    dataWniosku: str | None = None
    wniosekDoId: int | None = None
    wniosekDo: str | None = None
    sankcjaIds: list[int] | None = None
    sankcjaList: list[str] | None = None
    podstawaPrawnaSankcjiIds: list[int] | None = None
    podstawaPrawnaSankcjiList: list[str] | None = None
    naruszeniaSkutkujaceSankcjaIds: list[int] | None = None
    naruszeniaSkutkujaceSankcjaList: list[str] | None = None
    czyMamyInformacjeOWszczeciuPostepowaniaId: int | None = None
    czyMamyInformacjeOWszczeciuPostepowania: str | None = None
    rozstrzygniecieId: int | None = None
    rozstrzygniecie: str | None = None
    komentarz: str | None = None


class RiskExposureUpdate(BaseModel):
    lockToken: str | None = None
    expectedUpdatedAt: str | None = None
    inspectionId: int | None = None
    nazwaPodmiotuObjetegoInspekcjaId: int | None = None
    nazwaPodmiotuObjetegoInspekcja: str | None = None
    nazwaPodmiotuObjetegoSankcjaIds: list[int] | None = None
    nazwaPodmiotuObjetegoSankcjaList: list[str] | None = None
    dataWniosku: str | None = None
    wniosekDoId: int | None = None
    wniosekDo: str | None = None
    sankcjaIds: list[int] | None = None
    sankcjaList: list[str] | None = None
    podstawaPrawnaSankcjiIds: list[int] | None = None
    podstawaPrawnaSankcjiList: list[str] | None = None
    naruszeniaSkutkujaceSankcjaIds: list[int] | None = None
    naruszeniaSkutkujaceSankcjaList: list[str] | None = None
    czyMamyInformacjeOWszczeciuPostepowaniaId: int | None = None
    czyMamyInformacjeOWszczeciuPostepowania: str | None = None
    rozstrzygniecieId: int | None = None
    rozstrzygniecie: str | None = None
    komentarz: str | None = None


class RiskExposureRead(BaseModel):
    id: int
    lp: int
    kodSankcji: str | None = None
    canEdit: bool
    inspectionId: int | None = None
    inspectionLp: int | None = None
    inspectionKod: str | None = None
    nazwaPodmiotuObjetegoInspekcjaId: int | None = None
    nazwaPodmiotuObjetegoInspekcja: str | None = None
    nazwaPodmiotuObjetegoInspekcjaSkrocona: str | None = None
    nazwaPodmiotuObjetegoInspekcjaSkrot: str | None = None
    nazwaPodmiotuObjetegoSankcjaIds: list[int]
    nazwaPodmiotuObjetegoSankcjaList: list[str]
    nazwaPodmiotuObjetegoSankcjaListSkrocona: list[str]
    nazwaPodmiotuObjetegoSankcjaListSkrot: list[str]
    dataWniosku: str | None = None
    wniosekDoId: int | None = None
    wniosekDo: str | None = None
    wniosekDoSkrocona: str | None = None
    wniosekDoSkrot: str | None = None
    sankcjaIds: list[int]
    sankcjaList: list[str]
    sankcjaListSkrocona: list[str]
    sankcjaListSkrot: list[str]
    podstawaPrawnaSankcjiIds: list[int]
    podstawaPrawnaSankcjiList: list[str]
    podstawaPrawnaSankcjiListSkrocona: list[str]
    podstawaPrawnaSankcjiListSkrot: list[str]
    naruszeniaSkutkujaceSankcjaIds: list[int]
    naruszeniaSkutkujaceSankcjaList: list[str]
    naruszeniaSkutkujaceSankcjaListSkrocona: list[str]
    naruszeniaSkutkujaceSankcjaListSkrot: list[str]
    czyMamyInformacjeOWszczeciuPostepowaniaId: int | None = None
    czyMamyInformacjeOWszczeciuPostepowania: str | None = None
    czyMamyInformacjeOWszczeciuPostepowaniaSkrocona: str | None = None
    czyMamyInformacjeOWszczeciuPostepowaniaSkrot: str | None = None
    rozstrzygniecieId: int | None = None
    rozstrzygniecie: str | None = None
    rozstrzygniecieSkrocona: str | None = None
    rozstrzygniecieSkrot: str | None = None
    komentarz: str | None = None
    utworzonoO: str
    zaktualizowanoO: str


class RiskExposureListResponse(BaseModel):
    items: list[RiskExposureRead]
    total: int


class RiskExposureInspectionOption(BaseModel):
    id: int
    lp: int
    kodInspekcji: str | None = None
    nazwaPodmiotu: str
    poczatekInspekcji: str
    koniecInspekcji: str
    osobaKierujacaUserId: int | None = None
    osobaKierujaca: str | None = None


class SanctionEntityOption(BaseModel):
    value: str
    label: str
    source: Literal["inspections", "sanctions", "historical"]
    active: bool


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _normalize_entity_text(value: str | None) -> str | None:
    cleaned = _norm(value)
    if cleaned is None:
        return None
    # Keep original diacritics but normalize repeated whitespace.
    return " ".join(cleaned.split())


def _entity_dedupe_key(value: str) -> str:
    normalized = _normalize_entity_text(value) or ""
    tokens = [token for token in normalized.casefold().split(" ") if token]
    if tokens and len(set(tokens)) == 1:
        tokens = [tokens[0]]
    return " ".join(tokens)


def _merge_entity_option(
    existing: dict[str, Any],
    *,
    value: str,
    label: str,
    source: Literal["inspections", "sanctions", "historical"],
    active: bool,
) -> None:
    existing["active"] = bool(existing.get("active", False) or active)

    current_label = _normalize_entity_text(existing.get("label")) or value
    candidate_label = _normalize_entity_text(label) or value
    if (len(candidate_label), candidate_label.casefold(), candidate_label) < (
        len(current_label),
        current_label.casefold(),
        current_label,
    ):
        existing["label"] = candidate_label


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
    blocked = _parse_status_codes_env("RISK_EXPOSURE_BLOCKED_INSPECTION_STATUS_CODES")
    if operator is None:
        return blocked

    role_id_raw = operator.get("rola_id")
    if role_id_raw is None:
        return blocked

    role_id = int(role_id_raw)
    blocked |= _parse_status_codes_env(f"RISK_EXPOSURE_BLOCKED_INSPECTION_STATUS_CODES_ROLE_{role_id}")
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


def _validate_optional_iso_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} ma niepoprawny format daty") from exc
    return parsed.isoformat()


def _normalize_text_list(raw_values: list[str] | None, field_name: str) -> list[str]:
    if raw_values is None:
        return []

    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        if not isinstance(raw, str):
            raise HTTPException(status_code=400, detail=f"{field_name} zawiera niepoprawny typ")
        value = raw.strip()
        if not value:
            raise HTTPException(status_code=400, detail=f"{field_name} nie moze zawierac pustych wartosci")
        if value not in seen:
            seen.add(value)
            values.append(value)

    # Stabilne, deterministiczne sortowanie dla payloadu i odczytu.
    values.sort(key=lambda v: (v.casefold(), v))
    return values


def _normalize_int_list(raw_values: list[int] | None, field_name: str) -> list[int]:
    if raw_values is None:
        return []

    values: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"{field_name} zawiera niepoprawny typ") from exc
        if value <= 0:
            raise HTTPException(status_code=400, detail=f"{field_name} zawiera niepoprawne id")
        if value not in seen:
            seen.add(value)
            values.append(value)

    values.sort()
    return values


def _parse_csv(csv_value: str | None) -> list[str]:
    if not csv_value:
        return []
    values: list[str] = []
    for part in str(csv_value).split(";"):
        cleaned = part.strip()
        if cleaned:
            values.append(cleaned)
    return values


def _parse_int_csv(csv_value: str | None) -> list[int]:
    if not csv_value:
        return []
    values: list[int] = []
    for part in str(csv_value).split(";"):
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        values.append(value)
    return values


def _next_lp(conn: Any) -> int:
    row = conn.execute("SELECT COALESCE(MAX(lp), 0) AS max_lp FROM risk_exposure_requests").fetchone()
    return int(row["max_lp"]) + 1


def _next_sanction_code(conn: Any, data_wniosku: str | None) -> str:
    if data_wniosku:
        try:
            year = str(date.fromisoformat(data_wniosku).year)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="dataWniosku ma niepoprawny format daty") from exc
    else:
        year = str(date.today().year)

    prefix = "WS"
    pattern = f"{prefix}/{year}/%"
    rows = conn.execute(
        "SELECT kod_sankcji FROM risk_exposure_requests WHERE kod_sankcji LIKE ?",
        (pattern,),
    ).fetchall()

    max_seq = 0
    for row in rows:
        raw = str(row["kod_sankcji"] or "").strip()
        parts = raw.split("/")
        if len(parts) != 3 or parts[0] != prefix or parts[1] != year:
            continue
        try:
            seq = int(parts[2])
        except ValueError:
            continue
        max_seq = max(max_seq, seq)

    return f"{prefix}/{year}/{max_seq + 1}"


def _resolve_or_create_slownik_item_id(conn: Any, kod_typu: str, raw_value: str | None) -> int | None:
    value = _norm(raw_value)
    if value is None:
        return None

    row = conn.execute(
        """
        SELECT id
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?)
          AND lower(nazwa_pozycji) = lower(?)
        LIMIT 1
        """,
        (kod_typu, value),
    ).fetchone()
    if row is not None:
        return int(row["id"])

    base_code = _slug_code(value)
    candidate = base_code
    suffix = 2
    while True:
        exists = conn.execute(
            """
            SELECT id
            FROM slownik_pozycje
            WHERE lower(kod_typu) = lower(?)
              AND lower(kod_pozycji) = lower(?)
            LIMIT 1
            """,
            (kod_typu, candidate),
        ).fetchone()
        if exists is None:
            break
        candidate = f"{base_code}_{suffix}"
        suffix += 1

    max_row = conn.execute(
        """
        SELECT COALESCE(MAX(kolejnosc), 0) AS max_kolejnosc
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?)
        """,
        (kod_typu,),
    ).fetchone()
    next_order = int(max_row["max_kolejnosc"]) + 1

    cursor = conn.execute(
        """
        INSERT INTO slownik_pozycje
        (kod_typu, kod_pozycji, nazwa_pozycji, kolejnosc, aktywny)
        VALUES (?, ?, ?, ?, 1)
        """,
        (kod_typu, candidate, value, next_order),
    )
    return int(cursor.lastrowid)


def _validate_slownik_item_id(
    conn: Any,
    kod_typu: str | Sequence[str],
    slownik_id: int,
    field_name: str,
) -> int:
    if isinstance(kod_typu, str):
        row = conn.execute(
            """
            SELECT id
            FROM slownik_pozycje
            WHERE id = ?
              AND lower(kod_typu) = lower(?)
            LIMIT 1
            """,
            (slownik_id, kod_typu),
        ).fetchone()
    else:
        normalized = [str(item).strip().lower() for item in kod_typu if str(item).strip()]
        if not normalized:
            row = None
        else:
            placeholders = ",".join("?" for _ in normalized)
            row = conn.execute(
                f"""
                SELECT id
                FROM slownik_pozycje
                WHERE id = ?
                  AND lower(kod_typu) IN ({placeholders})
                LIMIT 1
                """,
                (slownik_id, *normalized),
            ).fetchone()

    if row is None:
        raise HTTPException(status_code=400, detail=f"{field_name} zawiera niepoprawne id")
    return int(row["id"])


def _resolve_single_slownik_id(
    conn: Any,
    kod_typu: str | Sequence[str],
    raw_id: int | None,
    raw_name: str | None,
    id_field_name: str,
) -> int | None:
    if raw_id is not None:
        return _validate_slownik_item_id(conn, kod_typu, int(raw_id), id_field_name)

    if isinstance(kod_typu, str):
        return _resolve_or_create_slownik_item_id(conn, kod_typu, raw_name)

    value = _norm(raw_name)
    if value is None:
        return None

    original = [str(item).strip() for item in kod_typu if str(item).strip()]
    if not original:
        return None
    normalized = [item.lower() for item in original]

    placeholders = ",".join("?" for _ in normalized)
    existing = conn.execute(
        f"""
        SELECT id
        FROM slownik_pozycje
        WHERE lower(kod_typu) IN ({placeholders})
          AND lower(nazwa_pozycji) = lower(?)
        LIMIT 1
        """,
        (*normalized, value),
    ).fetchone()
    if existing is not None:
        return int(existing["id"])

    return _resolve_or_create_slownik_item_id(conn, original[0], raw_name)


def _resolve_multi_slownik_ids(
    conn: Any,
    kod_typu: str,
    raw_ids: list[int] | None,
    raw_names: list[str] | None,
    ids_field_name: str,
    names_field_name: str,
) -> list[int]:
    if raw_ids is not None:
        ids = _normalize_int_list(raw_ids, ids_field_name)
        return [_validate_slownik_item_id(conn, kod_typu, slownik_id, ids_field_name) for slownik_id in ids]

    names = _normalize_text_list(raw_names, names_field_name)
    resolved: list[int] = []
    for name in names:
        slownik_id = _resolve_or_create_slownik_item_id(conn, kod_typu, name)
        if slownik_id is not None:
            resolved.append(slownik_id)
    resolved.sort()
    return resolved


def _can_access_inspection_for_risk_exposure(conn: Any, inspection_id: int, operator: dict[str, Any]) -> bool:
    if operator["rola_id"] == 3:
        return True

    if operator["rola_id"] == 2:
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

    inspection_row = conn.execute(
        "SELECT created_by_user_id FROM inspections WHERE id = ? LIMIT 1",
        (inspection_id,),
    ).fetchone()
    if inspection_row is not None and inspection_row["created_by_user_id"] is not None:
        if int(inspection_row["created_by_user_id"]) == operator["id"]:
            return True

    member_row = conn.execute(
        "SELECT 1 FROM inspection_members WHERE inspection_id = ? AND user_id = ? LIMIT 1",
        (inspection_id, operator["id"]),
    ).fetchone()
    return member_row is not None


def _can_edit_risk_exposure(
    conn: Any,
    inspection_id: int | None,
    operator: dict[str, Any],
    created_by_user_id: int | None,
) -> bool:
    if operator["rola_id"] == 3:
        return True

    if inspection_id is None:
        return created_by_user_id is not None and int(created_by_user_id) == operator["id"]

    return _can_access_inspection_for_risk_exposure(conn, inspection_id, operator)


def _resolve_inspection_subject_id(conn: Any, inspection_id: int) -> int | None:
    row = conn.execute(
        """
        SELECT i.nazwa_podmiotu_id
        FROM inspections i
        WHERE i.id = ?
        LIMIT 1
        """,
        (inspection_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    if row["nazwa_podmiotu_id"] is None:
        return None
    return int(row["nazwa_podmiotu_id"])


def _sync_multi_values(
    conn: Any,
    risk_exposure_id: int,
    value_type: Literal[
        "NAZWA_PODMIOTU_OBJETEGO_SANKCJA",
        "SANKCJA",
        "PODSTAWA_PRAWNA_SANKCJI",
        "NARUSZENIA_SKUTKUJACE_SANKCJA",
    ],
    slownik_ids: list[int],
    operator_user_id: int,
) -> None:
    value_type_map: dict[str, tuple[str, str]] = {
        "NAZWA_PODMIOTU_OBJETEGO_SANKCJA": ("nazwy_podmiotow_sankcje", "risk_exposure_sanction_subjects"),
        "SANKCJA": ("sankcja", "risk_exposure_sanctions"),
        "PODSTAWA_PRAWNA_SANKCJI": ("podstawa_prawna_sankcji", "risk_exposure_legal_bases"),
        "NARUSZENIA_SKUTKUJACE_SANKCJA": ("naruszenia_skutkujace_sankcja", "risk_exposure_violations"),
    }
    kod_typu, relation_table = value_type_map[value_type]

    existing_rows = conn.execute(
        f"""
        SELECT id, slownik_pozycja_id
        FROM {relation_table}
        WHERE risk_exposure_id = ?
        """,
        (risk_exposure_id,),
    ).fetchall()

    existing_by_value = {int(r["slownik_pozycja_id"]): int(r["id"]) for r in existing_rows}
    target_set = set(slownik_ids)
    existing_set = set(existing_by_value.keys())

    to_delete = sorted(existing_set - target_set)
    to_insert = sorted(target_set - existing_set)

    for slownik_id in to_delete:
        conn.execute(
            f"DELETE FROM {relation_table} WHERE id = ?",
            (existing_by_value[slownik_id],),
        )

    if to_insert:
        placeholders = ",".join("?" for _ in to_insert)
        rows = conn.execute(
            f"""
            SELECT id, nazwa_pozycji
            FROM slownik_pozycje
            WHERE lower(kod_typu) = lower(?)
              AND id IN ({placeholders})
            """,
            (kod_typu, *to_insert),
        ).fetchall()
        names_by_id = {int(r["id"]): str(r["nazwa_pozycji"]) for r in rows}
    else:
        names_by_id = {}

    for slownik_id in to_insert:
        if slownik_id not in names_by_id:
            raise HTTPException(status_code=400, detail="Lista wartosci zawiera niepoprawne id")
        conn.execute(
            f"""
            INSERT INTO {relation_table} (
                risk_exposure_id,
                slownik_pozycja_id,
                created_by_user_id,
                updated_by_user_id
            ) VALUES (?, ?, ?, ?)
            """,
            (
                risk_exposure_id,
                slownik_id,
                operator_user_id,
                operator_user_id,
            ),
        )


def _base_select_sql() -> str:
    return """
        SELECT
            r.id,
            r.lp,
            r.kod_sankcji,
            r.inspection_id,
            i.lp AS inspection_lp,
            i.kod_inspekcji AS inspection_kod,
                        r.nazwa_podmiotu_objetego_inspekcja_id,
                        npoi.nazwa_pozycji AS nazwa_podmiotu_objetego_inspekcja,
                        npoi.skrot_pozycji AS nazwa_podmiotu_objetego_inspekcja_skrot,
            r.data_wniosku,
                        r.wniosek_do_id,
                        wd.nazwa_pozycji AS wniosek_do,
                        wd.skrot_pozycji AS wniosek_do_skrot,
                        r.czy_mamy_informacje_o_wszczeciu_postepowania_id,
                        wsz.nazwa_pozycji AS czy_mamy_informacje_o_wszczeciu_postepowania,
                        wsz.skrot_pozycji AS czy_mamy_informacje_o_wszczeciu_postepowania_skrot,
                        r.rozstrzygniecie_id,
                        roz.nazwa_pozycji AS rozstrzygniecie,
                        roz.skrot_pozycji AS rozstrzygniecie_skrot,
            r.komentarz,
            r.utworzono_o,
            r.zaktualizowano_o,
            r.utworzono_przez_user_id,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT CAST(rel.slownik_pozycja_id AS TEXT) AS v
                    FROM risk_exposure_sanction_subjects rel
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS nazwa_podmiotu_objetego_sankcja_ids_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.nazwa_pozycji AS v
                    FROM risk_exposure_sanction_subjects rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY v ASC
                ) x
            ) AS nazwa_podmiotu_objetego_sankcja_list_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.skrot_pozycji AS v
                    FROM risk_exposure_sanction_subjects rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                      AND trim(COALESCE(sp.skrot_pozycji, '')) <> ''
                    ORDER BY v ASC
                ) x
            ) AS nazwa_podmiotu_objetego_sankcja_list_skrot_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT CAST(rel.slownik_pozycja_id AS TEXT) AS v
                    FROM risk_exposure_sanctions rel
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS sankcja_ids_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.nazwa_pozycji AS v
                    FROM risk_exposure_sanctions rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY v ASC
                ) x
            ) AS sankcja_list_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.skrot_pozycji AS v
                    FROM risk_exposure_sanctions rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY v ASC
                ) x
            ) AS sankcja_list_skrot_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT CAST(rel.slownik_pozycja_id AS TEXT) AS v
                    FROM risk_exposure_legal_bases rel
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS podstawa_prawna_sankcji_ids_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.nazwa_pozycji AS v
                    FROM risk_exposure_legal_bases rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY v ASC
                ) x
            ) AS podstawa_prawna_sankcji_list_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.skrot_pozycji AS v
                    FROM risk_exposure_legal_bases rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY v ASC
                ) x
            ) AS podstawa_prawna_sankcji_list_skrot_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT CAST(rel.slownik_pozycja_id AS TEXT) AS v
                    FROM risk_exposure_violations rel
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS naruszenia_skutkujace_sankcja_ids_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.nazwa_pozycji AS v
                    FROM risk_exposure_violations rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY v ASC
                ) x
            ) AS naruszenia_skutkujace_sankcja_list_csv
            ,(
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.skrot_pozycji AS v
                    FROM risk_exposure_violations rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY v ASC
                ) x
            ) AS naruszenia_skutkujace_sankcja_list_skrot_csv
        FROM risk_exposure_requests r
        LEFT JOIN inspections i ON i.id = r.inspection_id
                LEFT JOIN slownik_pozycje npoi ON npoi.id = r.nazwa_podmiotu_objetego_inspekcja_id
                LEFT JOIN slownik_pozycje wd ON wd.id = r.wniosek_do_id
                LEFT JOIN slownik_pozycje wsz ON wsz.id = r.czy_mamy_informacje_o_wszczeciu_postepowania_id
                LEFT JOIN slownik_pozycje roz ON roz.id = r.rozstrzygniecie_id
    """


def _row_to_payload(row: dict[str, Any], can_edit: bool) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "lp": int(row["lp"]),
        "kodSankcji": row.get("kod_sankcji"),
        "canEdit": can_edit,
        "inspectionId": int(row["inspection_id"]) if row.get("inspection_id") is not None else None,
        "inspectionLp": int(row["inspection_lp"]) if row.get("inspection_lp") is not None else None,
        "inspectionKod": row.get("inspection_kod"),
        "nazwaPodmiotuObjetegoInspekcjaId": (
            int(row["nazwa_podmiotu_objetego_inspekcja_id"])
            if row.get("nazwa_podmiotu_objetego_inspekcja_id") is not None
            else None
        ),
        "nazwaPodmiotuObjetegoInspekcja": row.get("nazwa_podmiotu_objetego_inspekcja"),
        "nazwaPodmiotuObjetegoInspekcjaSkrocona": row.get("nazwa_podmiotu_objetego_inspekcja_skrot"),
        "nazwaPodmiotuObjetegoInspekcjaSkrot": row.get("nazwa_podmiotu_objetego_inspekcja_skrot"),
        "nazwaPodmiotuObjetegoSankcjaIds": _parse_int_csv(row.get("nazwa_podmiotu_objetego_sankcja_ids_csv")),
        "nazwaPodmiotuObjetegoSankcjaList": _parse_csv(row.get("nazwa_podmiotu_objetego_sankcja_list_csv")),
        "nazwaPodmiotuObjetegoSankcjaListSkrocona": _parse_csv(
            row.get("nazwa_podmiotu_objetego_sankcja_list_skrot_csv")
        ),
        "nazwaPodmiotuObjetegoSankcjaListSkrot": _parse_csv(
            row.get("nazwa_podmiotu_objetego_sankcja_list_skrot_csv")
        ),
        "dataWniosku": row.get("data_wniosku"),
        "wniosekDoId": int(row["wniosek_do_id"]) if row.get("wniosek_do_id") is not None else None,
        "wniosekDo": row.get("wniosek_do"),
        "wniosekDoSkrocona": row.get("wniosek_do_skrot"),
        "wniosekDoSkrot": row.get("wniosek_do_skrot"),
        "sankcjaIds": _parse_int_csv(row.get("sankcja_ids_csv")),
        "sankcjaList": _parse_csv(row.get("sankcja_list_csv")),
        "sankcjaListSkrocona": _parse_csv(row.get("sankcja_list_skrot_csv")),
        "sankcjaListSkrot": _parse_csv(row.get("sankcja_list_skrot_csv")),
        "podstawaPrawnaSankcjiIds": _parse_int_csv(row.get("podstawa_prawna_sankcji_ids_csv")),
        "podstawaPrawnaSankcjiList": _parse_csv(row.get("podstawa_prawna_sankcji_list_csv")),
        "podstawaPrawnaSankcjiListSkrocona": _parse_csv(row.get("podstawa_prawna_sankcji_list_skrot_csv")),
        "podstawaPrawnaSankcjiListSkrot": _parse_csv(row.get("podstawa_prawna_sankcji_list_skrot_csv")),
        "naruszeniaSkutkujaceSankcjaIds": _parse_int_csv(row.get("naruszenia_skutkujace_sankcja_ids_csv")),
        "naruszeniaSkutkujaceSankcjaList": _parse_csv(row.get("naruszenia_skutkujace_sankcja_list_csv")),
        "naruszeniaSkutkujaceSankcjaListSkrocona": _parse_csv(
            row.get("naruszenia_skutkujace_sankcja_list_skrot_csv")
        ),
        "naruszeniaSkutkujaceSankcjaListSkrot": _parse_csv(
            row.get("naruszenia_skutkujace_sankcja_list_skrot_csv")
        ),
        "czyMamyInformacjeOWszczeciuPostepowaniaId": (
            int(row["czy_mamy_informacje_o_wszczeciu_postepowania_id"])
            if row.get("czy_mamy_informacje_o_wszczeciu_postepowania_id") is not None
            else None
        ),
        "czyMamyInformacjeOWszczeciuPostepowania": row.get("czy_mamy_informacje_o_wszczeciu_postepowania"),
        "czyMamyInformacjeOWszczeciuPostepowaniaSkrocona": row.get(
            "czy_mamy_informacje_o_wszczeciu_postepowania_skrot"
        ),
        "czyMamyInformacjeOWszczeciuPostepowaniaSkrot": row.get(
            "czy_mamy_informacje_o_wszczeciu_postepowania_skrot"
        ),
        "rozstrzygniecieId": int(row["rozstrzygniecie_id"]) if row.get("rozstrzygniecie_id") is not None else None,
        "rozstrzygniecie": row.get("rozstrzygniecie"),
        "rozstrzygniecieSkrocona": row.get("rozstrzygniecie_skrot"),
        "rozstrzygniecieSkrot": row.get("rozstrzygniecie_skrot"),
        "komentarz": row.get("komentarz"),
        "utworzonoO": row.get("utworzono_o"),
        "zaktualizowanoO": row.get("zaktualizowano_o"),
    }


@router.get("/api/sanction-requests", response_model=RiskExposureListResponse)
@router.get("/api/risk-exposure", response_model=RiskExposureListResponse)
def list_risk_exposure(
    sortBy: str = Query(default="lp"),
    sortOrder: str = Query(default="asc"),
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    allowed_sort_columns = {
        "lp": "r.lp",
        "id": "r.id",
        "kodSankcji": "r.kod_sankcji",
        "dataWniosku": "r.data_wniosku",
        "inspectionLp": "i.lp",
        "inspectionKod": "i.kod_inspekcji",
        "nazwaPodmiotuObjetegoInspekcja": "npoi.nazwa_pozycji",
    }
    if sortBy not in allowed_sort_columns:
        raise HTTPException(status_code=400, detail="Niepoprawny sortBy")

    direction = sortOrder.lower()
    if direction not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="Niepoprawny sortOrder")

    order_sql = f" ORDER BY {allowed_sort_columns[sortBy]} {direction.upper()}, r.id ASC"

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_RISK_EXPOSURE_READ)
        rows = conn.execute(_base_select_sql() + order_sql).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            inspection_id = int(row_dict["inspection_id"]) if row_dict.get("inspection_id") is not None else None
            created_by = (
                int(row_dict["utworzono_przez_user_id"]) if row_dict.get("utworzono_przez_user_id") is not None else None
            )
            can_edit = _can_edit_risk_exposure(conn, inspection_id, operator, created_by)
            items.append(_row_to_payload(row_dict, can_edit=can_edit))

    return {"items": items, "total": len(items)}


@router.get(
    "/api/sanction-requests/available-inspections",
    response_model=list[RiskExposureInspectionOption],
)
@router.get(
    "/api/risk-exposure/available-inspections",
    response_model=list[RiskExposureInspectionOption],
)
def list_available_inspections_for_risk_exposure(
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_RISK_EXPOSURE_READ)
        rows = conn.execute(
            """
            SELECT
                i.id,
                i.lp,
                i.kod_inspekcji,
                np.nazwa_pozycji AS nazwa_podmiotu_nazwa,
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
            if not _can_access_inspection_for_risk_exposure(conn, inspection_id, operator):
                continue
            blocked, _, _ = _inspection_status_is_forbidden(conn, inspection_id, operator)
            if blocked:
                continue
            items.append(
                {
                    "id": inspection_id,
                    "lp": int(row_dict["lp"]),
                    "kodInspekcji": row_dict.get("kod_inspekcji"),
                    "nazwaPodmiotu": row_dict.get("nazwa_podmiotu_nazwa") or "brak",
                    "poczatekInspekcji": row_dict.get("poczatek_inspekcji"),
                    "koniecInspekcji": row_dict.get("koniec_inspekcji"),
                    "osobaKierujacaUserId": row_dict.get("osoba_kierujaca_user_id"),
                    "osobaKierujaca": _norm(row_dict.get("osoba_kierujaca_full")) or row_dict.get("osoba_kierujaca_login"),
                }
            )

    return items


@router.get(
    "/api/sanction-requests/entity-options",
    response_model=list[SanctionEntityOption],
)
@router.get(
    "/api/risk-exposure/entity-options",
    response_model=list[SanctionEntityOption],
)
def list_sanction_request_entity_options(
    include_historical: bool = Query(default=True, alias="includeHistorical"),
    limit: int | None = Query(default=None, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    source_priority = {
        "inspections": 0,
        "sanctions": 1,
        "historical": 2,
    }
    by_key: dict[str, dict[str, Any]] = {}

    def add_option(
        *,
        value: str | None,
        label: str | None,
        source: Literal["inspections", "sanctions", "historical"],
        active: bool,
    ) -> None:
        normalized_value = _normalize_entity_text(value)
        if normalized_value is None:
            return
        normalized_label = _normalize_entity_text(label) or normalized_value
        key = _entity_dedupe_key(normalized_value)
        if not key:
            return

        existing = by_key.get(key)
        if existing is None:
            by_key[key] = {
                "value": normalized_value,
                "label": normalized_label,
                "source": source,
                "active": bool(active),
            }
            return

        if source_priority[source] < source_priority[str(existing["source"])]:
            existing["source"] = source
            existing["value"] = normalized_value
        _merge_entity_option(
            existing,
            value=normalized_value,
            label=normalized_label,
            source=source,
            active=active,
        )

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_RISK_EXPOSURE_READ)

        inspection_dictionary_rows = conn.execute(
            """
            SELECT nazwa_pozycji AS value, skrot_pozycji AS short_value, aktywny
            FROM slownik_pozycje
            WHERE lower(kod_typu) = 'nazwy_podmiotow'
            ORDER BY id ASC
            """
        ).fetchall()
        for row in inspection_dictionary_rows:
            add_option(
                value=row["value"],
                label=row["short_value"] or row["value"],
                source="inspections",
                active=bool(int(row["aktywny"])),
            )

        sanctions_rows = conn.execute(
            """
            SELECT nazwa_pozycji AS value, skrot_pozycji AS short_value, aktywny
            FROM slownik_pozycje
            WHERE lower(kod_typu) = 'nazwy_podmiotow_sankcje'
              AND aktywny = 1
            ORDER BY id ASC
            """
        ).fetchall()
        for row in sanctions_rows:
            add_option(
                value=row["value"],
                label=row["short_value"] or row["value"],
                source="sanctions",
                active=bool(int(row["aktywny"])),
            )

        if include_historical:
            historical_rows = conn.execute(
                """
                SELECT DISTINCT sp.nazwa_pozycji AS value, sp.skrot_pozycji AS short_value, sp.aktywny
                FROM risk_exposure_sanction_subjects rel
                JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                ORDER BY sp.id ASC
                """
            ).fetchall()
            for row in historical_rows:
                add_option(
                    value=row["value"],
                    label=row["short_value"] or row["value"],
                    source="historical",
                    active=bool(int(row["aktywny"])),
                )

    items = list(by_key.values())
    items.sort(key=lambda item: (str(item["label"]).casefold(), str(item["label"])))

    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return items


@router.get("/api/sanction-requests/{risk_exposure_id}", response_model=RiskExposureRead)
@router.get("/api/risk-exposure/{risk_exposure_id}", response_model=RiskExposureRead)
def get_risk_exposure(
    risk_exposure_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_RISK_EXPOSURE_READ)
        row = conn.execute(_base_select_sql() + " WHERE r.id = ? LIMIT 1", (risk_exposure_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Risk exposure request not found")

        row_dict = dict(row)
        inspection_id = int(row_dict["inspection_id"]) if row_dict.get("inspection_id") is not None else None
        created_by = int(row_dict["utworzono_przez_user_id"]) if row_dict.get("utworzono_przez_user_id") is not None else None
        can_edit = _can_edit_risk_exposure(conn, inspection_id, operator, created_by)

    return _row_to_payload(row_dict, can_edit=can_edit)


@router.post("/api/sanction-requests", response_model=RiskExposureRead, status_code=201)
@router.post("/api/risk-exposure", response_model=RiskExposureRead, status_code=201)
def create_risk_exposure(
    payload: RiskExposureCreate,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    data_wniosku = _validate_optional_iso_date(payload.dataWniosku, "dataWniosku")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)

        nazwa_sankcja_ids = _resolve_multi_slownik_ids(
            conn,
            "nazwy_podmiotow_sankcje",
            payload.nazwaPodmiotuObjetegoSankcjaIds,
            payload.nazwaPodmiotuObjetegoSankcjaList,
            "nazwaPodmiotuObjetegoSankcjaIds",
            "nazwaPodmiotuObjetegoSankcjaList",
        )
        sankcja_ids = _resolve_multi_slownik_ids(
            conn,
            "sankcja",
            payload.sankcjaIds,
            payload.sankcjaList,
            "sankcjaIds",
            "sankcjaList",
        )
        podstawa_ids = _resolve_multi_slownik_ids(
            conn,
            "podstawa_prawna_sankcji",
            payload.podstawaPrawnaSankcjiIds,
            payload.podstawaPrawnaSankcjiList,
            "podstawaPrawnaSankcjiIds",
            "podstawaPrawnaSankcjiList",
        )
        naruszenia_ids = _resolve_multi_slownik_ids(
            conn,
            "naruszenia_skutkujace_sankcja",
            payload.naruszeniaSkutkujaceSankcjaIds,
            payload.naruszeniaSkutkujaceSankcjaList,
            "naruszeniaSkutkujaceSankcjaIds",
            "naruszeniaSkutkujaceSankcjaList",
        )

        resolved_inspection_id: int | None = None
        if payload.inspectionId is not None:
            inspection_row = conn.execute("SELECT id FROM inspections WHERE id = ? LIMIT 1", (payload.inspectionId,)).fetchone()
            if inspection_row is None:
                raise HTTPException(status_code=404, detail="Inspection not found")
            resolved_inspection_id = int(inspection_row["id"])
            if not _can_access_inspection_for_risk_exposure(conn, resolved_inspection_id, operator):
                raise HTTPException(status_code=403, detail="Brak uprawnien do tej inspekcji")
            blocked, status_code, status_label = _inspection_status_is_forbidden(conn, resolved_inspection_id, operator)
            if blocked:
                _raise_inspection_status_block(resolved_inspection_id, status_code, status_label)
            nazwa_objetego_inspekcja_id = _resolve_inspection_subject_id(conn, resolved_inspection_id)
        else:
            nazwa_objetego_inspekcja_id = _resolve_single_slownik_id(
                conn,
                "nazwy_podmiotow",
                payload.nazwaPodmiotuObjetegoInspekcjaId,
                payload.nazwaPodmiotuObjetegoInspekcja,
                "nazwaPodmiotuObjetegoInspekcjaId",
            )

        wniosek_do_id = _resolve_single_slownik_id(
            conn,
            "department",
            payload.wniosekDoId,
            payload.wniosekDo,
            "wniosekDoId",
        )
        wszczecie_id = _resolve_single_slownik_id(
            conn,
            "informacja_o_wszczeciu_postepowania_sankcyjnego",
            payload.czyMamyInformacjeOWszczeciuPostepowaniaId,
            payload.czyMamyInformacjeOWszczeciuPostepowania,
            "czyMamyInformacjeOWszczeciuPostepowaniaId",
        )
        rozstrzygniecie_id = _resolve_single_slownik_id(
            conn,
            ROZSTRZYGNIECIE_WNIOSKU_KOD_TYPU,
            payload.rozstrzygniecieId,
            payload.rozstrzygniecie,
            "rozstrzygniecieId",
        )

        cursor = conn.execute(
            """
            INSERT INTO risk_exposure_requests (
                lp,
                kod_sankcji,
                inspection_id,
                nazwa_podmiotu_objetego_inspekcja_id,
                data_wniosku,
                wniosek_do_id,
                czy_mamy_informacje_o_wszczeciu_postepowania_id,
                rozstrzygniecie_id,
                komentarz,
                utworzono_przez_user_id,
                zaktualizowano_przez_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _next_lp(conn),
                _next_sanction_code(conn, data_wniosku),
                resolved_inspection_id,
                nazwa_objetego_inspekcja_id,
                data_wniosku,
                wniosek_do_id,
                wszczecie_id,
                rozstrzygniecie_id,
                payload.komentarz,
                operator["id"],
                operator["id"],
            ),
        )
        risk_exposure_id = int(cursor.lastrowid)

        _sync_multi_values(conn, risk_exposure_id, "NAZWA_PODMIOTU_OBJETEGO_SANKCJA", nazwa_sankcja_ids, operator["id"])
        _sync_multi_values(conn, risk_exposure_id, "SANKCJA", sankcja_ids, operator["id"])
        _sync_multi_values(conn, risk_exposure_id, "PODSTAWA_PRAWNA_SANKCJI", podstawa_ids, operator["id"])
        _sync_multi_values(conn, risk_exposure_id, "NARUSZENIA_SKUTKUJACE_SANKCJA", naruszenia_ids, operator["id"])
        kod_row = conn.execute(
            "SELECT kod_sankcji FROM risk_exposure_requests WHERE id = ? LIMIT 1",
            (risk_exposure_id,),
        ).fetchone()
        rekord_kod_cr = str(kod_row["kod_sankcji"]) if kod_row and kod_row["kod_sankcji"] else str(risk_exposure_id)
        row = conn.execute(_base_select_sql() + " WHERE r.id = ? LIMIT 1", (risk_exposure_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="Failed to fetch created risk exposure request")

        created_payload = _row_to_payload(dict(row), can_edit=True)
        changes = build_create_changes(
            [
                ("Kod sankcji", created_payload.get("kodSankcji")),
                ("Kod inspekcji", created_payload.get("inspectionKod")),
                ("Nazwa podmiotu objętego inspekcją", created_payload.get("nazwaPodmiotuObjetegoInspekcja")),
                ("Data wniosku", created_payload.get("dataWniosku")),
                ("Wniosek do", created_payload.get("wniosekDo")),
                ("Informacja o wszczęciu postępowania", created_payload.get("czyMamyInformacjeOWszczeciuPostepowania")),
                ("Rozstrzygnięcie", created_payload.get("rozstrzygniecie")),
                ("Podmioty objęte sankcją", created_payload.get("nazwaPodmiotuObjetegoSankcjaList")),
                ("Sankcje", created_payload.get("sankcjaList")),
                ("Podstawy prawne", created_payload.get("podstawaPrawnaSankcjiList")),
                ("Naruszenia", created_payload.get("naruszeniaSkutkujaceSankcjaList")),
                ("Komentarz", created_payload.get("komentarz")),
            ]
        )

        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_CREATE,
                        REJESTR_WNIOSKI, rekord_kod_cr, changes)
        conn.commit()

    if row is None:
        raise HTTPException(status_code=500, detail="Failed to fetch created risk exposure request")

    return _row_to_payload(dict(row), can_edit=True)


@router.put("/api/sanction-requests/{risk_exposure_id}", response_model=RiskExposureRead)
@router.put("/api/risk-exposure/{risk_exposure_id}", response_model=RiskExposureRead)
def update_risk_exposure(
    risk_exposure_id: int,
    payload: RiskExposureUpdate,
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
            """SELECT id, inspection_id, kod_sankcji, nazwa_podmiotu_objetego_inspekcja_id,
                      data_wniosku, wniosek_do_id, czy_mamy_informacje_o_wszczeciu_postepowania_id,
                      rozstrzygniecie_id, komentarz, utworzono_przez_user_id, zaktualizowano_o
               FROM risk_exposure_requests WHERE id = ? LIMIT 1""",
            (risk_exposure_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail="Risk exposure request not found")

        current_dict = dict(current)
        current_inspection_id = (
            int(current_dict["inspection_id"]) if current_dict.get("inspection_id") is not None else None
        )
        created_by = (
            int(current_dict["utworzono_przez_user_id"])
            if current_dict.get("utworzono_przez_user_id") is not None
            else None
        )
        if not _can_edit_risk_exposure(conn, current_inspection_id, operator, created_by):
            raise HTTPException(status_code=403, detail="Brak uprawnien")

        assert_lock_for_save(conn, "sanction-requests", risk_exposure_id, operator, lock_token)
        assert_expected_updated_at(expected_updated_at, str(current_dict.get("zaktualizowano_o") or ""))

        # Pobierz multi-values przed zmianą (dla audit log)
        def _fetch_mv_before(table: str) -> str | None:
            rows = conn.execute(
                f"SELECT sp.nazwa_pozycji FROM {table} t JOIN slownik_pozycje sp ON sp.id = t.slownik_pozycja_id WHERE t.risk_exposure_id = ? ORDER BY sp.nazwa_pozycji",
                (risk_exposure_id,),
            ).fetchall()
            return ", ".join(r["nazwa_pozycji"] for r in rows) or None

        subjects_before = _fetch_mv_before("risk_exposure_sanction_subjects")
        sanctions_before = _fetch_mv_before("risk_exposure_sanctions")
        legal_bases_before = _fetch_mv_before("risk_exposure_legal_bases")
        violations_before = _fetch_mv_before("risk_exposure_violations")

        next_inspection_id = current_inspection_id
        set_parts: list[str] = []
        values: list[Any] = []
        row_touched = False

        if "inspectionId" in fields:
            if fields["inspectionId"] is None:
                next_inspection_id = None
                set_parts.append("inspection_id = ?")
                values.append(None)
            else:
                target_inspection = conn.execute(
                    "SELECT id FROM inspections WHERE id = ? LIMIT 1",
                    (int(fields["inspectionId"]),),
                ).fetchone()
                if target_inspection is None:
                    raise HTTPException(status_code=404, detail="Inspection not found")
                next_inspection_id = int(target_inspection["id"])
                if not _can_access_inspection_for_risk_exposure(conn, next_inspection_id, operator):
                    raise HTTPException(status_code=403, detail="Brak uprawnien do tej inspekcji")
                set_parts.append("inspection_id = ?")
                values.append(next_inspection_id)

        if next_inspection_id is not None:
            blocked, status_code, status_label = _inspection_status_is_forbidden(conn, next_inspection_id, operator)
            if blocked:
                _raise_inspection_status_block(next_inspection_id, status_code, status_label)

        if next_inspection_id is not None:
            # Pole pochodne z inspekcji.
            set_parts.append("nazwa_podmiotu_objetego_inspekcja_id = ?")
            values.append(_resolve_inspection_subject_id(conn, next_inspection_id))
        elif "nazwaPodmiotuObjetegoInspekcjaId" in fields or "nazwaPodmiotuObjetegoInspekcja" in fields:
            set_parts.append("nazwa_podmiotu_objetego_inspekcja_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "nazwy_podmiotow",
                    fields.get("nazwaPodmiotuObjetegoInspekcjaId"),
                    fields.get("nazwaPodmiotuObjetegoInspekcja"),
                    "nazwaPodmiotuObjetegoInspekcjaId",
                )
            )

        if "dataWniosku" in fields:
            set_parts.append("data_wniosku = ?")
            values.append(_validate_optional_iso_date(fields["dataWniosku"], "dataWniosku"))
        if "wniosekDoId" in fields or "wniosekDo" in fields:
            set_parts.append("wniosek_do_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "department",
                    fields.get("wniosekDoId"),
                    fields.get("wniosekDo"),
                    "wniosekDoId",
                )
            )
        if "czyMamyInformacjeOWszczeciuPostepowaniaId" in fields or "czyMamyInformacjeOWszczeciuPostepowania" in fields:
            set_parts.append("czy_mamy_informacje_o_wszczeciu_postepowania_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "informacja_o_wszczeciu_postepowania_sankcyjnego",
                    fields.get("czyMamyInformacjeOWszczeciuPostepowaniaId"),
                    fields.get("czyMamyInformacjeOWszczeciuPostepowania"),
                    "czyMamyInformacjeOWszczeciuPostepowaniaId",
                )
            )
        if "rozstrzygniecieId" in fields or "rozstrzygniecie" in fields:
            set_parts.append("rozstrzygniecie_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    ROZSTRZYGNIECIE_WNIOSKU_KOD_TYPU,
                    fields.get("rozstrzygniecieId"),
                    fields.get("rozstrzygniecie"),
                    "rozstrzygniecieId",
                )
            )
        if "komentarz" in fields:
            set_parts.append("komentarz = ?")
            values.append(fields["komentarz"])

        if set_parts:
            set_parts.append("zaktualizowano_przez_user_id = ?")
            values.append(operator["id"])
            set_parts.append("zaktualizowano_o = ?")
            values.append(now_rfc3339_utc_ms())
            values.append(risk_exposure_id)
            conn.execute(
                f"UPDATE risk_exposure_requests SET {', '.join(set_parts)} WHERE id = ?",
                tuple(values),
            )
            row_touched = True

        if "nazwaPodmiotuObjetegoSankcjaIds" in fields or "nazwaPodmiotuObjetegoSankcjaList" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "NAZWA_PODMIOTU_OBJETEGO_SANKCJA",
                _resolve_multi_slownik_ids(
                    conn,
                    "nazwy_podmiotow_sankcje",
                    fields.get("nazwaPodmiotuObjetegoSankcjaIds"),
                    fields.get("nazwaPodmiotuObjetegoSankcjaList"),
                    "nazwaPodmiotuObjetegoSankcjaIds",
                    "nazwaPodmiotuObjetegoSankcjaList",
                ),
                operator["id"],
            )
        if "sankcjaIds" in fields or "sankcjaList" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "SANKCJA",
                _resolve_multi_slownik_ids(
                    conn,
                    "sankcja",
                    fields.get("sankcjaIds"),
                    fields.get("sankcjaList"),
                    "sankcjaIds",
                    "sankcjaList",
                ),
                operator["id"],
            )
        if "podstawaPrawnaSankcjiIds" in fields or "podstawaPrawnaSankcjiList" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "PODSTAWA_PRAWNA_SANKCJI",
                _resolve_multi_slownik_ids(
                    conn,
                    "podstawa_prawna_sankcji",
                    fields.get("podstawaPrawnaSankcjiIds"),
                    fields.get("podstawaPrawnaSankcjiList"),
                    "podstawaPrawnaSankcjiIds",
                    "podstawaPrawnaSankcjiList",
                ),
                operator["id"],
            )
        if "naruszeniaSkutkujaceSankcjaIds" in fields or "naruszeniaSkutkujaceSankcjaList" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "NARUSZENIA_SKUTKUJACE_SANKCJA",
                _resolve_multi_slownik_ids(
                    conn,
                    "naruszenia_skutkujace_sankcja",
                    fields.get("naruszeniaSkutkujaceSankcjaIds"),
                    fields.get("naruszeniaSkutkujaceSankcjaList"),
                    "naruszeniaSkutkujaceSankcjaIds",
                    "naruszeniaSkutkujaceSankcjaList",
                ),
                operator["id"],
            )

        if not row_touched and (
            "nazwaPodmiotuObjetegoSankcjaIds" in fields
            or "nazwaPodmiotuObjetegoSankcjaList" in fields
            or "sankcjaIds" in fields
            or "sankcjaList" in fields
            or "podstawaPrawnaSankcjiIds" in fields
            or "podstawaPrawnaSankcjiList" in fields
            or "naruszeniaSkutkujaceSankcjaIds" in fields
            or "naruszeniaSkutkujaceSankcjaList" in fields
        ):
            conn.execute(
                "UPDATE risk_exposure_requests SET zaktualizowano_przez_user_id = ?, zaktualizowano_o = ? WHERE id = ?",
                (operator["id"], now_rfc3339_utc_ms(), risk_exposure_id),
            )

        # Pobierz multi-values po zmianie (dla audit log)
        subjects_after = _fetch_mv_before("risk_exposure_sanction_subjects")
        sanctions_after = _fetch_mv_before("risk_exposure_sanctions")
        legal_bases_after = _fetch_mv_before("risk_exposure_legal_bases")
        violations_after = _fetch_mv_before("risk_exposure_violations")

        # --- Audit log ---
        rekord_kod_u = str(current_dict.get("kod_sankcji") or risk_exposure_id)
        changes_re = build_risk_exposure_changes(
            conn, current_dict, fields,
            subjects_before, subjects_after,
            sanctions_before, sanctions_after,
            legal_bases_before, legal_bases_after,
            violations_before, violations_after,
        )
        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_UPDATE,
                        REJESTR_WNIOSKI, rekord_kod_u, changes_re)
        # --- koniec audit log ---

        conn.commit()
        row = conn.execute(_base_select_sql() + " WHERE r.id = ? LIMIT 1", (risk_exposure_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Risk exposure request not found")

    return _row_to_payload(dict(row), can_edit=True)


@router.delete("/api/sanction-requests/{risk_exposure_id}")
@router.delete("/api/risk-exposure/{risk_exposure_id}")
def delete_risk_exposure(
    risk_exposure_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, bool]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)
        _ensure_director(operator)
        row = conn.execute(
            "SELECT inspection_id, kod_sankcji, utworzono_przez_user_id FROM risk_exposure_requests WHERE id = ? LIMIT 1",
            (risk_exposure_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Risk exposure request not found")

        conn.execute("DELETE FROM risk_exposure_requests WHERE id = ?", (risk_exposure_id,))
        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_DELETE,
                        REJESTR_WNIOSKI, str(row["kod_sankcji"] or risk_exposure_id), [])
        conn.commit()

    return {"ok": True}
