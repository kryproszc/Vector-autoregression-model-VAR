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
        extra="forbid",
        json_schema_extra={
            "example": {
                "inspectionId": 1,
                "nazwaPodmiotuObjetegoInspekcjaId": 10,
                "nazwaPodmiotuObjetegoSankcjaIds": [11, 12],
                "dataWniosku": "2026-12-01",
                "wniosekDoId": 21,
                "sankcjaIds": [31],
                "podstawaPrawnaSankcjiIds": [41],
                "naruszeniaSkutkujaceSankcjaIds": [51],
                "czyMamyInformacjeOWszczeciuPostepowaniaId": 61,
                "rozstrzygniecieId": 71,
                "komentarz": "Komentarz",
            }
        }
    )

    inspectionId: int | None = None
    nazwaPodmiotuObjetegoInspekcjaId: int | None = None
    nazwaPodmiotuObjetegoSankcjaIds: list[int] | None = None
    dataWniosku: str | None = None
    wniosekDoId: int | None = None
    sankcjaIds: list[int] | None = None
    podstawaPrawnaSankcjiIds: list[int] | None = None
    naruszeniaSkutkujaceSankcjaIds: list[int] | None = None
    czyMamyInformacjeOWszczeciuPostepowaniaId: int | None = None
    rozstrzygniecieId: int | None = None
    komentarz: str | None = None


class RiskExposureUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lockToken: str | None = None
    expectedUpdatedAt: str | None = None
    inspectionId: int | None = None
    nazwaPodmiotuObjetegoInspekcjaId: int | None = None
    nazwaPodmiotuObjetegoSankcjaIds: list[int] | None = None
    dataWniosku: str | None = None
    wniosekDoId: int | None = None
    sankcjaIds: list[int] | None = None
    podstawaPrawnaSankcjiIds: list[int] | None = None
    naruszeniaSkutkujaceSankcjaIds: list[int] | None = None
    czyMamyInformacjeOWszczeciuPostepowaniaId: int | None = None
    rozstrzygniecieId: int | None = None
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
    nazwaPodmiotuSkrocona: str | None = None
    nazwaPodmiotuSkrot: str | None = None
    poczatekInspekcji: str
    koniecInspekcji: str
    osobaKierujacaUserId: int | None = None
    osobaKierujaca: str | None = None


class SanctionEntityOption(BaseModel):
    value: str
    label: str
    source: Literal["inspections", "sanctions", "historical"]
    active: bool


class RiskExposureLegacyTranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nazwaPodmiotuObjetegoInspekcja: str | None = None
    nazwaPodmiotuObjetegoSankcjaList: list[str] | None = None
    wniosekDo: str | None = None
    sankcjaList: list[str] | None = None
    podstawaPrawnaSankcjiList: list[str] | None = None
    naruszeniaSkutkujaceSankcjaList: list[str] | None = None
    czyMamyInformacjeOWszczeciuPostepowania: str | None = None
    rozstrzygniecie: str | None = None


class RiskExposureLegacyTranslationResponse(BaseModel):
    nazwaPodmiotuObjetegoInspekcjaId: int | None = None
    nazwaPodmiotuObjetegoSankcjaIds: list[int] = Field(default_factory=list)
    wniosekDoId: int | None = None
    sankcjaIds: list[int] = Field(default_factory=list)
    podstawaPrawnaSankcjiIds: list[int] = Field(default_factory=list)
    naruszeniaSkutkujaceSankcjaIds: list[int] = Field(default_factory=list)
    czyMamyInformacjeOWszczeciuPostepowaniaId: int | None = None
    rozstrzygniecieId: int | None = None
    unresolved: dict[str, list[str]] = Field(default_factory=dict)


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
        _raise_contract_422(
            "INVALID_DATE_VALUE",
            f"{field_name} ma niepoprawny format daty.",
            field=field_name,
            value=value,
        )
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
            _raise_contract_422(
                "INVALID_VALUE",
                f"{field_name} zawiera niepoprawny typ.",
                field=field_name,
                value=raw,
            )
        if value <= 0:
            _raise_contract_422(
                "INVALID_VALUE",
                f"{field_name} zawiera niepoprawne id.",
                field=field_name,
                value=raw,
            )
        if value not in seen:
            seen.add(value)
            values.append(value)

    values.sort()
    return values


def _normalize_text_input_list(raw_values: list[str] | None, field_name: str) -> list[str]:
    if raw_values is None:
        return []

    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        if not isinstance(raw, str):
            _raise_contract_422(
                "INVALID_VALUE",
                f"{field_name} zawiera niepoprawny typ.",
                field=field_name,
                value=raw,
            )
        cleaned = _normalize_entity_text(raw)
        if cleaned is None:
            _raise_contract_422(
                "INVALID_VALUE",
                f"{field_name} nie moze zawierac pustych wartosci.",
                field=field_name,
                value=raw,
            )
        dedupe_key = cleaned.casefold()
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            values.append(cleaned)
    return values


def _parse_csv(csv_value: str | None) -> list[str]:
    if not csv_value:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for part in str(csv_value).split(";"):
        cleaned = part.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            values.append(cleaned)
    return values


def _parse_int_csv(csv_value: str | None) -> list[int]:
    if not csv_value:
        return []
    values: list[int] = []
    seen: set[int] = set()
    for part in str(csv_value).split(";"):
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0 and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _lookup_slownik_id_by_name(
    conn: Any,
    kod_typu: str | Sequence[str],
    raw_name: str | None,
) -> tuple[int | None, str | None]:
    normalized_name = _normalize_entity_text(raw_name)
    if normalized_name is None:
        return None, None

    if isinstance(kod_typu, str):
        rows = conn.execute(
            """
            SELECT id, nazwa_pozycji
            FROM slownik_pozycje
            WHERE lower(kod_typu) = lower(?)
              AND aktywny = 1
            ORDER BY id ASC
            """,
            (kod_typu,),
        ).fetchall()
    else:
        normalized_types = [str(item).strip().lower() for item in kod_typu if str(item).strip()]
        if not normalized_types:
            return None, normalized_name
        placeholders = ",".join("?" for _ in normalized_types)
        rows = conn.execute(
            f"""
            SELECT id, nazwa_pozycji
            FROM slownik_pozycje
            WHERE lower(kod_typu) IN ({placeholders})
              AND aktywny = 1
            ORDER BY id ASC
            """,
            tuple(normalized_types),
        ).fetchall()

    target_key = normalized_name.casefold()
    for row in rows:
        candidate_name = _normalize_entity_text(row["nazwa_pozycji"]) or ""
        if candidate_name.casefold() == target_key:
            return int(row["id"]), None
    return None, normalized_name


def _lookup_slownik_ids_by_names(
    conn: Any,
    kod_typu: str | Sequence[str],
    raw_values: list[str] | None,
    field_name: str,
) -> tuple[list[int], list[str]]:
    normalized_values = _normalize_text_input_list(raw_values, field_name)
    ids: list[int] = []
    unresolved: list[str] = []
    for value in normalized_values:
        resolved_id, unresolved_name = _lookup_slownik_id_by_name(conn, kod_typu, value)
        if resolved_id is None:
            if unresolved_name is not None:
                unresolved.append(unresolved_name)
            continue
        ids.append(resolved_id)
    return ids, unresolved


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
              AND aktywny = 1
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
                  AND aktywny = 1
                LIMIT 1
                """,
                (slownik_id, *normalized),
            ).fetchone()

    if row is None:
        resolved_kod_typu = kod_typu if isinstance(kod_typu, str) else ",".join(str(item) for item in kod_typu)
        _raise_contract_422(
            "UNKNOWN_DICTIONARY_ID",
            f"{field_name} nie wskazuje aktywnej pozycji slownika.",
            field=field_name,
            value=slownik_id,
            kod_typu=str(resolved_kod_typu),
        )
    return int(row["id"])


def _resolve_single_slownik_id(
    conn: Any,
    kod_typu: str | Sequence[str],
    raw_id: int | None,
    id_field_name: str,
    required: bool,
) -> int | None:
    if raw_id is not None:
        try:
            raw_id = int(raw_id)
        except (TypeError, ValueError):
            _raise_contract_422(
                "INVALID_VALUE",
                f"{id_field_name} zawiera niepoprawny typ.",
                field=id_field_name,
                value=raw_id,
            )
        if raw_id <= 0:
            _raise_contract_422(
                "INVALID_VALUE",
                f"{id_field_name} zawiera niepoprawne id.",
                field=id_field_name,
                value=raw_id,
            )

    if raw_id is None:
        if required:
            resolved_kod_typu = kod_typu if isinstance(kod_typu, str) else ",".join(str(item) for item in kod_typu)
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                f"Pole {id_field_name} jest wymagane.",
                field=id_field_name,
                value=raw_id,
                kod_typu=str(resolved_kod_typu),
            )
        return None
    return _validate_slownik_item_id(conn, kod_typu, int(raw_id), id_field_name)


def _resolve_multi_slownik_ids(
    conn: Any,
    kod_typu: str | Sequence[str],
    raw_ids: list[int] | None,
    ids_field_name: str,
) -> list[int]:
    ids = _normalize_int_list(raw_ids, ids_field_name)
    return [_validate_slownik_item_id(conn, kod_typu, slownik_id, ids_field_name) for slownik_id in ids]


def _can_access_inspection_for_risk_exposure(conn: Any, inspection_id: int, operator: dict[str, Any]) -> bool:
    if operator["rola_id"] == 3:
        return True

    if operator["rola_id"] == 2:
        inspection_author_row = conn.execute(
            """
            SELECT u.zespol_id, u.created_by_user_id
            FROM inspections i
            JOIN users u ON u.id = i.created_by_user_id
            WHERE i.id = ?
            LIMIT 1
            """,
            (inspection_id,),
        ).fetchone()
        if inspection_author_row is not None:
            author_created_by = inspection_author_row["created_by_user_id"]
            if author_created_by is not None and int(author_created_by) == int(operator["id"]):
                return True
            if operator["zespol_id"] is not None and inspection_author_row["zespol_id"] is not None:
                if int(inspection_author_row["zespol_id"]) == int(operator["zespol_id"]):
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
    value_type_map: dict[str, tuple[str | Sequence[str], str]] = {
        "NAZWA_PODMIOTU_OBJETEGO_SANKCJA": (
            ("nazwy_podmiotow_sankcje", "nazwy_podmiotow"),
            "risk_exposure_sanction_subjects",
        ),
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
        if isinstance(kod_typu, str):
            rows = conn.execute(
                f"""
                SELECT id, nazwa_pozycji
                FROM slownik_pozycje
                WHERE lower(kod_typu) = lower(?)
                  AND id IN ({placeholders})
                """,
                (kod_typu, *to_insert),
            ).fetchall()
        else:
            normalized = [str(item).strip().lower() for item in kod_typu if str(item).strip()]
            if not normalized:
                rows = []
            else:
                type_placeholders = ",".join("?" for _ in normalized)
                rows = conn.execute(
                    f"""
                    SELECT id, nazwa_pozycji
                    FROM slownik_pozycje
                    WHERE lower(kod_typu) IN ({type_placeholders})
                      AND id IN ({placeholders})
                    """,
                    (*normalized, *to_insert),
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
                    ORDER BY rel.slownik_pozycja_id ASC
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
                                        ORDER BY rel.slownik_pozycja_id ASC
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
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS sankcja_list_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.skrot_pozycji AS v
                    FROM risk_exposure_sanctions rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY rel.slownik_pozycja_id ASC
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
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS podstawa_prawna_sankcji_list_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.skrot_pozycji AS v
                    FROM risk_exposure_legal_bases rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY rel.slownik_pozycja_id ASC
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
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS naruszenia_skutkujace_sankcja_list_csv
            ,(
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.skrot_pozycji AS v
                    FROM risk_exposure_violations rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.risk_exposure_id = r.id
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS naruszenia_skutkujace_sankcja_list_skrot_csv
        FROM risk_exposure_requests r
        LEFT JOIN inspections i ON i.id = r.inspection_id
                LEFT JOIN slownik_pozycje npoi ON npoi.id = r.nazwa_podmiotu_objetego_inspekcja_id
                LEFT JOIN slownik_pozycje wd ON wd.id = r.wniosek_do_id
                LEFT JOIN slownik_pozycje wsz ON wsz.id = r.czy_mamy_informacje_o_wszczeciu_postepowania_id
                LEFT JOIN slownik_pozycje roz ON roz.id = r.rozstrzygniecie_id
    """


def _load_multi_values_for_payload(conn: Any, risk_exposure_id: int) -> dict[str, list[Any]]:
    def load_rows(relation_table: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"""
            SELECT rel.slownik_pozycja_id AS slownik_id,
                   sp.nazwa_pozycji AS nazwa,
                   sp.skrot_pozycji AS skrot
            FROM {relation_table} rel
            JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
            WHERE rel.risk_exposure_id = ?
            ORDER BY rel.slownik_pozycja_id ASC
            """,
            (risk_exposure_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def build_lists(rows: list[dict[str, Any]]) -> tuple[list[int], list[str], list[str]]:
        ids: list[int] = []
        names: list[str] = []
        shorts: list[str] = []
        seen: set[int] = set()
        for row in rows:
            slownik_id = int(row["slownik_id"])
            if slownik_id in seen:
                continue
            seen.add(slownik_id)
            name = _normalize_entity_text(row.get("nazwa"))
            short = _normalize_entity_text(row.get("skrot"))
            if name is None:
                continue
            ids.append(slownik_id)
            names.append(name)
            shorts.append(short or name)
        return ids, names, shorts

    sanction_subject_rows = load_rows("risk_exposure_sanction_subjects")
    sanction_rows = load_rows("risk_exposure_sanctions")
    legal_base_rows = load_rows("risk_exposure_legal_bases")
    violation_rows = load_rows("risk_exposure_violations")

    sanction_subject_ids, sanction_subject_names, sanction_subject_shorts = build_lists(sanction_subject_rows)
    sanction_ids, sanction_names, sanction_shorts = build_lists(sanction_rows)
    legal_base_ids, legal_base_names, legal_base_shorts = build_lists(legal_base_rows)
    violation_ids, violation_names, violation_shorts = build_lists(violation_rows)

    return {
        "nazwaPodmiotuObjetegoSankcjaIds": sanction_subject_ids,
        "nazwaPodmiotuObjetegoSankcjaList": sanction_subject_names,
        "nazwaPodmiotuObjetegoSankcjaListSkrot": sanction_subject_shorts,
        "sankcjaIds": sanction_ids,
        "sankcjaList": sanction_names,
        "sankcjaListSkrot": sanction_shorts,
        "podstawaPrawnaSankcjiIds": legal_base_ids,
        "podstawaPrawnaSankcjiList": legal_base_names,
        "podstawaPrawnaSankcjiListSkrot": legal_base_shorts,
        "naruszeniaSkutkujaceSankcjaIds": violation_ids,
        "naruszeniaSkutkujaceSankcjaList": violation_names,
        "naruszeniaSkutkujaceSankcjaListSkrot": violation_shorts,
    }


def _row_to_payload(conn: Any, row: dict[str, Any], can_edit: bool) -> dict[str, Any]:
    multi = _load_multi_values_for_payload(conn, int(row["id"]))
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
        "nazwaPodmiotuObjetegoSankcjaIds": multi["nazwaPodmiotuObjetegoSankcjaIds"],
        "nazwaPodmiotuObjetegoSankcjaList": multi["nazwaPodmiotuObjetegoSankcjaList"],
        "nazwaPodmiotuObjetegoSankcjaListSkrocona": multi["nazwaPodmiotuObjetegoSankcjaListSkrot"],
        "nazwaPodmiotuObjetegoSankcjaListSkrot": multi["nazwaPodmiotuObjetegoSankcjaListSkrot"],
        "dataWniosku": row.get("data_wniosku"),
        "wniosekDoId": int(row["wniosek_do_id"]) if row.get("wniosek_do_id") is not None else None,
        "wniosekDo": row.get("wniosek_do"),
        "wniosekDoSkrocona": row.get("wniosek_do_skrot"),
        "wniosekDoSkrot": row.get("wniosek_do_skrot"),
        "sankcjaIds": multi["sankcjaIds"],
        "sankcjaList": multi["sankcjaList"],
        "sankcjaListSkrocona": multi["sankcjaListSkrot"],
        "sankcjaListSkrot": multi["sankcjaListSkrot"],
        "podstawaPrawnaSankcjiIds": multi["podstawaPrawnaSankcjiIds"],
        "podstawaPrawnaSankcjiList": multi["podstawaPrawnaSankcjiList"],
        "podstawaPrawnaSankcjiListSkrocona": multi["podstawaPrawnaSankcjiListSkrot"],
        "podstawaPrawnaSankcjiListSkrot": multi["podstawaPrawnaSankcjiListSkrot"],
        "naruszeniaSkutkujaceSankcjaIds": multi["naruszeniaSkutkujaceSankcjaIds"],
        "naruszeniaSkutkujaceSankcjaList": multi["naruszeniaSkutkujaceSankcjaList"],
        "naruszeniaSkutkujaceSankcjaListSkrocona": multi["naruszeniaSkutkujaceSankcjaListSkrot"],
        "naruszeniaSkutkujaceSankcjaListSkrot": multi["naruszeniaSkutkujaceSankcjaListSkrot"],
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
    sortBy: str = Query(default="dataWniosku"),
    sortOrder: str = Query(default="desc"),
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
            items.append(_row_to_payload(conn, row_dict, can_edit=can_edit))

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
        require_write_access(conn, operator)
        require_permission(conn, operator, PERMISSION_RISK_EXPOSURE_READ)
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
            if not _can_edit_risk_exposure(conn, inspection_id, operator, None):
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
                    "nazwaPodmiotuSkrocona": row_dict.get("nazwa_podmiotu_skrot"),
                    "nazwaPodmiotuSkrot": row_dict.get("nazwa_podmiotu_skrot"),
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


@router.post(
    "/api/sanction-requests/translate-legacy-values",
    response_model=RiskExposureLegacyTranslationResponse,
)
@router.post(
    "/api/risk-exposure/translate-legacy-values",
    response_model=RiskExposureLegacyTranslationResponse,
)
def translate_legacy_risk_exposure_values(
    payload: RiskExposureLegacyTranslationRequest,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_RISK_EXPOSURE_READ)

        unresolved: dict[str, list[str]] = {}

        nazwa_objetego_id, unresolved_nazwa_objetego = _lookup_slownik_id_by_name(
            conn,
            "nazwy_podmiotow",
            payload.nazwaPodmiotuObjetegoInspekcja,
        )
        if unresolved_nazwa_objetego is not None:
            unresolved["nazwaPodmiotuObjetegoInspekcja"] = [unresolved_nazwa_objetego]

        nazwy_sankcja_ids, unresolved_nazwy_sankcja = _lookup_slownik_ids_by_names(
            conn,
            ("nazwy_podmiotow_sankcje", "nazwy_podmiotow"),
            payload.nazwaPodmiotuObjetegoSankcjaList,
            "nazwaPodmiotuObjetegoSankcjaList",
        )
        if unresolved_nazwy_sankcja:
            unresolved["nazwaPodmiotuObjetegoSankcjaList"] = unresolved_nazwy_sankcja

        wniosek_do_id, unresolved_wniosek_do = _lookup_slownik_id_by_name(
            conn,
            "department",
            payload.wniosekDo,
        )
        if unresolved_wniosek_do is not None:
            unresolved["wniosekDo"] = [unresolved_wniosek_do]

        sankcja_ids, unresolved_sankcje = _lookup_slownik_ids_by_names(
            conn,
            "sankcja",
            payload.sankcjaList,
            "sankcjaList",
        )
        if unresolved_sankcje:
            unresolved["sankcjaList"] = unresolved_sankcje

        podstawa_ids, unresolved_podstawy = _lookup_slownik_ids_by_names(
            conn,
            "podstawa_prawna_sankcji",
            payload.podstawaPrawnaSankcjiList,
            "podstawaPrawnaSankcjiList",
        )
        if unresolved_podstawy:
            unresolved["podstawaPrawnaSankcjiList"] = unresolved_podstawy

        naruszenia_ids, unresolved_naruszenia = _lookup_slownik_ids_by_names(
            conn,
            "naruszenia_skutkujace_sankcja",
            payload.naruszeniaSkutkujaceSankcjaList,
            "naruszeniaSkutkujaceSankcjaList",
        )
        if unresolved_naruszenia:
            unresolved["naruszeniaSkutkujaceSankcjaList"] = unresolved_naruszenia

        wszczecie_id, unresolved_wszczecie = _lookup_slownik_id_by_name(
            conn,
            "informacja_o_wszczeciu_postepowania_sankcyjnego",
            payload.czyMamyInformacjeOWszczeciuPostepowania,
        )
        if unresolved_wszczecie is not None:
            unresolved["czyMamyInformacjeOWszczeciuPostepowania"] = [unresolved_wszczecie]

        rozstrzygniecie_id, unresolved_rozstrzygniecie = _lookup_slownik_id_by_name(
            conn,
            ROZSTRZYGNIECIE_WNIOSKU_KOD_TYPU,
            payload.rozstrzygniecie,
        )
        if unresolved_rozstrzygniecie is not None:
            unresolved["rozstrzygniecie"] = [unresolved_rozstrzygniecie]

        return {
            "nazwaPodmiotuObjetegoInspekcjaId": nazwa_objetego_id,
            "nazwaPodmiotuObjetegoSankcjaIds": nazwy_sankcja_ids,
            "wniosekDoId": wniosek_do_id,
            "sankcjaIds": sankcja_ids,
            "podstawaPrawnaSankcjiIds": podstawa_ids,
            "naruszeniaSkutkujaceSankcjaIds": naruszenia_ids,
            "czyMamyInformacjeOWszczeciuPostepowaniaId": wszczecie_id,
            "rozstrzygniecieId": rozstrzygniecie_id,
            "unresolved": unresolved,
        }


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

    return _row_to_payload(conn, row_dict, can_edit=can_edit)


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
            ("nazwy_podmiotow_sankcje", "nazwy_podmiotow"),
            payload.nazwaPodmiotuObjetegoSankcjaIds,
            "nazwaPodmiotuObjetegoSankcjaIds",
        )
        sankcja_ids = _resolve_multi_slownik_ids(
            conn,
            "sankcja",
            payload.sankcjaIds,
            "sankcjaIds",
        )
        podstawa_ids = _resolve_multi_slownik_ids(
            conn,
            "podstawa_prawna_sankcji",
            payload.podstawaPrawnaSankcjiIds,
            "podstawaPrawnaSankcjiIds",
        )
        naruszenia_ids = _resolve_multi_slownik_ids(
            conn,
            "naruszenia_skutkujace_sankcja",
            payload.naruszeniaSkutkujaceSankcjaIds,
            "naruszeniaSkutkujaceSankcjaIds",
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
                "nazwaPodmiotuObjetegoInspekcjaId",
                required=True,
            )

        wniosek_do_id = _resolve_single_slownik_id(
            conn,
            "department",
            payload.wniosekDoId,
            "wniosekDoId",
            required=False,
        )
        wszczecie_id = _resolve_single_slownik_id(
            conn,
            "informacja_o_wszczeciu_postepowania_sankcyjnego",
            payload.czyMamyInformacjeOWszczeciuPostepowaniaId,
            "czyMamyInformacjeOWszczeciuPostepowaniaId",
            required=False,
        )
        rozstrzygniecie_id = _resolve_single_slownik_id(
            conn,
            ROZSTRZYGNIECIE_WNIOSKU_KOD_TYPU,
            payload.rozstrzygniecieId,
            "rozstrzygniecieId",
            required=False,
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

        created_payload = _row_to_payload(conn, dict(row), can_edit=True)
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

    return _row_to_payload(conn, dict(row), can_edit=True)


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
        next_subject_id = int(current_dict["nazwa_podmiotu_objetego_inspekcja_id"]) if current_dict.get("nazwa_podmiotu_objetego_inspekcja_id") is not None else None
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
            next_subject_id = _resolve_inspection_subject_id(conn, next_inspection_id)
            values.append(next_subject_id)
        elif "nazwaPodmiotuObjetegoInspekcjaId" in fields:
            set_parts.append("nazwa_podmiotu_objetego_inspekcja_id = ?")
            next_subject_id = _resolve_single_slownik_id(
                conn,
                "nazwy_podmiotow",
                fields.get("nazwaPodmiotuObjetegoInspekcjaId"),
                "nazwaPodmiotuObjetegoInspekcjaId",
                required=True,
            )
            values.append(next_subject_id)

        if next_inspection_id is None and next_subject_id is None:
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                "Pole nazwaPodmiotuObjetegoInspekcjaId jest wymagane gdy inspectionId jest null.",
                field="nazwaPodmiotuObjetegoInspekcjaId",
                value=None,
                kod_typu="nazwy_podmiotow",
            )

        if "dataWniosku" in fields:
            set_parts.append("data_wniosku = ?")
            values.append(_validate_optional_iso_date(fields["dataWniosku"], "dataWniosku"))
        if "wniosekDoId" in fields:
            set_parts.append("wniosek_do_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "department",
                    fields.get("wniosekDoId"),
                    "wniosekDoId",
                    required=False,
                )
            )
        if "czyMamyInformacjeOWszczeciuPostepowaniaId" in fields:
            set_parts.append("czy_mamy_informacje_o_wszczeciu_postepowania_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "informacja_o_wszczeciu_postepowania_sankcyjnego",
                    fields.get("czyMamyInformacjeOWszczeciuPostepowaniaId"),
                    "czyMamyInformacjeOWszczeciuPostepowaniaId",
                    required=False,
                )
            )
        if "rozstrzygniecieId" in fields:
            set_parts.append("rozstrzygniecie_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    ROZSTRZYGNIECIE_WNIOSKU_KOD_TYPU,
                    fields.get("rozstrzygniecieId"),
                    "rozstrzygniecieId",
                    required=False,
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

        if "nazwaPodmiotuObjetegoSankcjaIds" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "NAZWA_PODMIOTU_OBJETEGO_SANKCJA",
                _resolve_multi_slownik_ids(
                    conn,
                    ("nazwy_podmiotow_sankcje", "nazwy_podmiotow"),
                    fields.get("nazwaPodmiotuObjetegoSankcjaIds"),
                    "nazwaPodmiotuObjetegoSankcjaIds",
                ),
                operator["id"],
            )
        if "sankcjaIds" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "SANKCJA",
                _resolve_multi_slownik_ids(
                    conn,
                    "sankcja",
                    fields.get("sankcjaIds"),
                    "sankcjaIds",
                ),
                operator["id"],
            )
        if "podstawaPrawnaSankcjiIds" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "PODSTAWA_PRAWNA_SANKCJI",
                _resolve_multi_slownik_ids(
                    conn,
                    "podstawa_prawna_sankcji",
                    fields.get("podstawaPrawnaSankcjiIds"),
                    "podstawaPrawnaSankcjiIds",
                ),
                operator["id"],
            )
        if "naruszeniaSkutkujaceSankcjaIds" in fields:
            _sync_multi_values(
                conn,
                risk_exposure_id,
                "NARUSZENIA_SKUTKUJACE_SANKCJA",
                _resolve_multi_slownik_ids(
                    conn,
                    "naruszenia_skutkujace_sankcja",
                    fields.get("naruszeniaSkutkujaceSankcjaIds"),
                    "naruszeniaSkutkujaceSankcjaIds",
                ),
                operator["id"],
            )

        if not row_touched and (
            "nazwaPodmiotuObjetegoSankcjaIds" in fields
            or "sankcjaIds" in fields
            or "podstawaPrawnaSankcjiIds" in fields
            or "naruszeniaSkutkujaceSankcjaIds" in fields
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

    with get_connection() as conn:
        fetched_row = conn.execute(_base_select_sql() + " WHERE r.id = ? LIMIT 1", (risk_exposure_id,)).fetchone()
        if fetched_row is None:
            raise HTTPException(status_code=404, detail="Risk exposure request not found")
        return _row_to_payload(conn, dict(fetched_row), can_edit=True)


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
