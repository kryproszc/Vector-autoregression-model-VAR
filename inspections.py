from __future__ import annotations

from datetime import date
import logging
import os
import re
import unicodedata
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict

from app.audit import (
    AKCJA_CREATE, AKCJA_DELETE, AKCJA_UPDATE,
    REJESTR_INSPEKCJE,
    build_create_changes,
    new_session_id,
    write_audit_log,
)
from app.database import get_connection
from app.permissions import PERMISSION_INSPECTIONS_READ, require_permission, require_write_access
from app.record_locks import assert_expected_updated_at, assert_lock_for_save, now_rfc3339_utc_ms

router = APIRouter()
logger = logging.getLogger(__name__)

_ALLOWED_INSPECTION_TYPE_BY_NORMALIZED: dict[str, tuple[str, str, int]] = {
    "kontrola": ("Kontrola", "KONTROLA", 1),
    "wizyta nadzorcza": ("Wizyta nadzorcza", "WIZYTA_NADZORCZA", 2),
}

_SZCZEGOLY_DOTYCZACE_ZAKRESU_MAX_LEN = 2000
_INSPECTION_STATUS_RELATIONS_ERROR_CODE = "INSPECTION_STATUS_RELATIONS_VALIDATION_FAILED"
_INSPECTION_STATUS_RELATIONS_ERROR_CODE_ID = 1100
_VIOLATION_RECOMMENDATIONS_REQUIRED_MISSING_ID = 1001
_VIOLATION_RECOMMENDATIONS_FORBIDDEN_PRESENT_ID = 1002
_VIOLATION_SANCTIONS_REQUIRED_MISSING_ID = 1003
_VIOLATION_SANCTIONS_FORBIDDEN_PRESENT_ID = 1004
_SCOPE_LIST_SEPARATOR = "\x1f"


def _env_truthy(name: str, default: str = "0") -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_INSPECTIONS_LEGACY_SCOPE_TEXT_SPLIT_ENABLED = _env_truthy(
    "INSPECTIONS_LEGACY_SCOPE_TEXT_SPLIT_ENABLED",
    "1",
)


def _warn_legacy_scope_text_usage(
    *,
    path: str,
    operator_login: str | None,
    inspection_id: int | None,
) -> None:
    logger.warning(
        "Deprecated payload field 'zakresInspekcji' used on %s; "
        "prefer 'zakresInspekcjiIds' or 'zakresInspekcjiList'. "
        "operator=%r inspection_id=%r",
        path,
        operator_login,
        inspection_id,
    )


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


def _parse_status_relation_rules_env(name: str) -> dict[str, tuple[str, str]]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return {}

    rules: dict[str, tuple[str, str]] = {}
    for token in re.split(r"[;\n]", raw):
        item = token.strip()
        if not item:
            continue

        if ":" not in item:
            raise HTTPException(
                status_code=500,
                detail=f"Niepoprawny format {name}: '{item}'. Oczekiwano CODE:R+,S-",
            )

        code_raw, spec_raw = item.split(":", 1)
        code = code_raw.strip().upper()
        spec_parts = [p.strip().upper() for p in re.split(r"[,|]", spec_raw) if p.strip()]
        if not code:
            raise HTTPException(status_code=500, detail=f"Niepoprawny format {name}: pusty kod statusu")

        rec_expectation = "any"
        san_expectation = "any"
        for part in spec_parts:
            if part == "R+":
                rec_expectation = "present"
            elif part == "R-":
                rec_expectation = "absent"
            elif part == "S+":
                san_expectation = "present"
            elif part == "S-":
                san_expectation = "absent"
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Niepoprawny format {name}: nieznany token '{part}' dla statusu {code}",
                )

        rules[code] = (rec_expectation, san_expectation)

    return rules


_INSPECTION_STATUS_REQUIRES_RECOMMENDATIONS_CODES = _parse_status_codes_env(
    "INSPECTIONS_STATUS_REQUIRES_RECOMMENDATIONS_CODES"
)
_INSPECTION_STATUS_FORBIDS_RECOMMENDATIONS_CODES = _parse_status_codes_env(
    "INSPECTIONS_STATUS_FORBIDS_RECOMMENDATIONS_CODES"
)
_INSPECTION_STATUS_REQUIRES_SANCTIONS_CODES = _parse_status_codes_env(
    "INSPECTIONS_STATUS_REQUIRES_SANCTIONS_CODES"
)
_INSPECTION_STATUS_FORBIDS_SANCTIONS_CODES = _parse_status_codes_env(
    "INSPECTIONS_STATUS_FORBIDS_SANCTIONS_CODES"
)
_INSPECTION_STATUS_RELATION_RULES = _parse_status_relation_rules_env(
    "INSPECTIONS_STATUS_RELATION_RULES"
)


class InspectionStructureCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nazwaPodmiotu: str
    typInspekcjiId: int | None = None
    typInspekcjiKodPozycji: str | None = None
    typInspekcji: str | None = None
    zakresInspekcji: str | None = None
    zakresInspekcjiList: list[str] | None = None
    zakresInspekcjiIds: list[int] | None = None
    poczatekInspekcji: str
    koniecInspekcji: str
    osobaKierujacaUserId: int | None = None
    teamMemberUserIds: list[int] | None = None
    inspectionTeamIds: list[int] | None = None
    inspection_team_ids: list[int] | None = None
    forceOperatorAsLeader: bool = False
    osobaKierujaca: str | None = None
    skladZespolu: str | None = None
    rynekId: int | None = None
    rynekKodPozycji: str | None = None
    rynek: str | None = None
    rodzajPodmiotuId: int | None = None
    rodzajPodmiotuKodPozycji: str | None = None
    rodzajPodmiotu: str | None = None
    aspektKonsumencki: str | None = None
    dataProtokolu: str | None = None
    dataDoreczeniaProtokolu: str | None = None
    dataAkceptacjiSprawozdania: str | None = None
    dataDoreczeniaPisma: str | None = None
    brakDataDoreczeniaPisma: bool = False
    dataWyslaniaPismaZZastrzezeniami: str | None = None
    brakDataWyslaniaPismaZZastrzezeniami: bool = False
    dataPismaZastrzezenia: str | None = None
    brakDataPismaZastrzezenia: bool = False
    dataWplywuPisma: str | None = None
    brakDataWplywuPisma: bool = False
    dataWyslaniaPismaZOdpowiedzia: str | None = None
    brakDataWyslaniaPismaZOdpowiedzia: bool = False
    dataPismaZOdpowiedzia: str | None = None
    brakDataPismaZOdpowiedzia: bool = False
    dataAkceptacjiNotyList: list[str] | None = None
    brakDatAkceptacjiNoty: bool = False
    dataZalecenList: list[str] | None = None
    dataAkceptacjiNoty: str | None = None
    dataZalecen: str | None = None
    statusId: int | None = None
    statusKodPozycji: str | None = None
    status: str | None = None
    komentarz: str | None = None
    szczegolyDotyczaceZakresu: str | None = None


class InspectionStructureUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lockToken: str | None = None
    expectedUpdatedAt: str | None = None
    nazwaPodmiotu: str | None = None
    typInspekcjiId: int | None = None
    typInspekcjiKodPozycji: str | None = None
    typInspekcji: str | None = None
    zakresInspekcji: str | None = None
    zakresInspekcjiList: list[str] | None = None
    zakresInspekcjiIds: list[int] | None = None
    poczatekInspekcji: str | None = None
    koniecInspekcji: str | None = None
    osobaKierujacaUserId: int | None = None
    teamMemberUserIds: list[int] | None = None
    inspectionTeamIds: list[int] | None = None
    inspection_team_ids: list[int] | None = None
    forceOperatorAsLeader: bool = False
    osobaKierujaca: str | None = None
    skladZespolu: str | None = None
    rynekId: int | None = None
    rynekKodPozycji: str | None = None
    rynek: str | None = None
    rodzajPodmiotuId: int | None = None
    rodzajPodmiotuKodPozycji: str | None = None
    rodzajPodmiotu: str | None = None
    aspektKonsumencki: str | None = None
    dataProtokolu: str | None = None
    dataDoreczeniaProtokolu: str | None = None
    dataAkceptacjiSprawozdania: str | None = None
    dataDoreczeniaPisma: str | None = None
    brakDataDoreczeniaPisma: bool | None = None
    dataWyslaniaPismaZZastrzezeniami: str | None = None
    brakDataWyslaniaPismaZZastrzezeniami: bool | None = None
    dataPismaZastrzezenia: str | None = None
    brakDataPismaZastrzezenia: bool | None = None
    dataWplywuPisma: str | None = None
    brakDataWplywuPisma: bool | None = None
    dataWyslaniaPismaZOdpowiedzia: str | None = None
    brakDataWyslaniaPismaZOdpowiedzia: bool | None = None
    dataPismaZOdpowiedzia: str | None = None
    brakDataPismaZOdpowiedzia: bool | None = None
    dataAkceptacjiNotyList: list[str] | None = None
    brakDatAkceptacjiNoty: bool | None = None
    dataZalecenList: list[str] | None = None
    dataAkceptacjiNoty: str | None = None
    dataZalecen: str | None = None
    statusId: int | None = None
    statusKodPozycji: str | None = None
    status: str | None = None
    komentarz: str | None = None
    szczegolyDotyczaceZakresu: str | None = None


class InspectionStructureRead(BaseModel):
    id: int
    kodInspekcji: str | None = None
    canEdit: bool
    nazwaPodmiotu: str
    nazwaPodmiotuSkrocona: str | None = None
    typInspekcjiId: int | None = None
    typInspekcjiKodPozycji: str | None = None
    typInspekcji: str | None = None
    typInspekcjiSkrocona: str | None = None
    zakresInspekcji: str | None = None
    zakresInspekcjiList: list[str]
    zakresInspekcjiSkrocona: str | None = None
    zakresInspekcjiIds: list[int]
    poczatekInspekcji: str
    koniecInspekcji: str
    osobaKierujacaUserId: int | None = None
    teamMemberUserIds: list[int]
    inspectionTeamIds: list[int]
    inspection_team_ids: list[int]
    inspectionTeamsList: list[str]
    osobaKierujaca: str
    skladZespolu: str
    rynekId: int | None = None
    rynekKodPozycji: str | None = None
    rynek: str | None = None
    rodzajPodmiotuId: int | None = None
    rodzajPodmiotuKodPozycji: str | None = None
    rodzajPodmiotu: str | None = None
    rodzajPodmiotuSkrocona: str | None = None
    aspektKonsumencki: str | None = None
    dataProtokolu: str | None = None
    dataDoreczeniaProtokolu: str | None = None
    dataAkceptacjiSprawozdania: str | None = None
    dataDoreczeniaPisma: str | None = None
    brakDataDoreczeniaPisma: bool
    dataWyslaniaPismaZZastrzezeniami: str | None = None
    brakDataWyslaniaPismaZZastrzezeniami: bool
    dataPismaZastrzezenia: str | None = None
    brakDataPismaZastrzezenia: bool
    dataWplywuPisma: str | None = None
    brakDataWplywuPisma: bool
    dataWyslaniaPismaZOdpowiedzia: str | None = None
    brakDataWyslaniaPismaZOdpowiedzia: bool
    dataPismaZOdpowiedzia: str | None = None
    brakDataPismaZOdpowiedzia: bool
    dataAkceptacjiNotyList: list[str]
    brakDatAkceptacjiNoty: bool
    dataZalecenList: list[str]
    dataAkceptacjiNoty: str | None = None
    dataZalecen: str | None = None
    statusId: int | None = None
    statusKodPozycji: str | None = None
    status: str | None = None
    statusSkrocona: str | None = None
    komentarz: str | None = None
    szczegolyDotyczaceZakresu: str | None = None
    zaktualizowanoO: str | None = None


class InspectionStructureListResponse(BaseModel):
    items: list[InspectionStructureRead]
    total: int


class InspectionPeopleOption(BaseModel):
    id: int
    login: str
    displayName: str
    active: bool
    canBeLeader: bool
    listVisibility: str
    visibleOnList: bool
    createdByOperator: bool
    createdByLogin: str | None = None
    teamId: int | None = None
    teamName: str | None = None
    accountType: str | None = None


class InspectionTeamOption(BaseModel):
    id: int
    code: str
    name: str
    isActive: bool


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _normalize_optional_text_with_limit(value: str | None, field_name: str, max_len: int) -> str | None:
    normalized = _norm(value)
    if normalized is None:
        return None
    if len(normalized) > max_len:
        raise HTTPException(status_code=422, detail=f"{field_name} przekracza limit {max_len} znakow")
    return normalized


def _raise_business_403(code: str, detail: str, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "code": code,
        "detail": detail,
    }
    payload.update(extra)
    raise HTTPException(status_code=403, detail=payload)


def _slug_code(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    code = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_").upper()
    return code or "POZYCJA"


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


def _normalize_optional_code(raw_value: str | None) -> str | None:
    value = _norm(raw_value)
    if value is None or value.lower() == "brak":
        return None
    return value.upper()


def _resolve_dictionary_row_by_id(conn: Any, kod_typu: str, item_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, kod_pozycji, nazwa_pozycji, aktywny
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?) AND id = ?
        LIMIT 1
        """,
        (kod_typu, int(item_id)),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def _resolve_dictionary_row_by_code(conn: Any, kod_typu: str, code_value: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, kod_pozycji, nazwa_pozycji, aktywny
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?)
        LIMIT 1
        """,
        (kod_typu, code_value),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def _resolve_dictionary_row_by_name(conn: Any, kod_typu: str, name_value: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, kod_pozycji, nazwa_pozycji, aktywny
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?) AND lower(nazwa_pozycji) = lower(?)
        LIMIT 1
        """,
        (kod_typu, name_value),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def _raise_dictionary_422(code: str, message: str, *, field_base: str, kod_typu: str) -> None:
    raise HTTPException(
        status_code=422,
        detail={
            "code": code,
            "message": message,
            "field": field_base,
            "kodTypu": kod_typu,
        },
    )


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


def _resolve_dictionary_id_strict(
    conn: Any,
    *,
    kod_typu: str,
    field_base: str,
    id_value: int | None,
    code_value: str | None,
    name_value: str | None,
    required: bool,
) -> int | None:
    normalized_code = _normalize_optional_code(code_value)
    normalized_name = _norm(name_value)
    has_legacy_text = bool(normalized_code) or (normalized_name is not None and normalized_name.lower() != "brak")

    if id_value is None:
        if has_legacy_text:
            _raise_contract_422(
                "LEGACY_TEXT_ONLY_VALUE",
                f"Pole {field_base}Id jest wymagane. Wartosci tekstowe/kodowe nie sa akceptowane.",
                field=f"{field_base}Id",
                value=normalized_code or normalized_name,
                kod_typu=kod_typu,
            )
        if required:
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                f"Pole {field_base}Id jest wymagane.",
                field=f"{field_base}Id",
                value=id_value,
                kod_typu=kod_typu,
            )
        return None

    row = _resolve_dictionary_row_by_id(conn, kod_typu, int(id_value))
    if row is None or int(row.get("aktywny") or 0) != 1:
        _raise_contract_422(
            "UNKNOWN_DICTIONARY_ID",
            f"{field_base}Id nie wskazuje aktywnej pozycji slownika.",
            field=f"{field_base}Id",
            value=id_value,
            kod_typu=kod_typu,
        )
    return int(row["id"])


def _resolve_scope_ids_strict(
    conn: Any,
    *,
    scope_ids: list[int] | None,
    scope_list: list[str] | None,
    scope_text: str | None,
    required: bool,
) -> list[int]:
    if scope_ids is None:
        if scope_list is not None or _norm(scope_text) is not None:
            _raise_contract_422(
                "LEGACY_TEXT_ONLY_VALUE",
                "Pole zakresInspekcjiIds jest wymagane. Pola tekstowe zakresu nie sa akceptowane.",
                field="zakresInspekcjiIds",
                value=scope_list if scope_list is not None else scope_text,
                kod_typu="zakresy_inspekcji",
            )
        if required:
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                "Pole zakresInspekcjiIds jest wymagane.",
                field="zakresInspekcjiIds",
                value=scope_ids,
                kod_typu="zakresy_inspekcji",
            )
        return []

    if scope_list is not None or _norm(scope_text) is not None:
        _raise_contract_422(
            "LEGACY_TEXT_ONLY_VALUE",
            "Dozwolone jest tylko pole zakresInspekcjiIds.",
            field="zakresInspekcjiIds",
            value=scope_list if scope_list is not None else scope_text,
            kod_typu="zakresy_inspekcji",
        )
    return _validate_scope_ids(conn, scope_ids)


def _resolve_dictionary_id_priority(
    conn: Any,
    *,
    kod_typu: str,
    field_base: str,
    id_value: int | None,
    code_value: str | None,
    name_value: str | None,
    required: bool = False,
    allow_name_autocreate: bool = False,
) -> int | None:
    candidate_ids: list[int] = []

    if id_value is not None:
        row = _resolve_dictionary_row_by_id(conn, kod_typu, int(id_value))
        if row is None:
            _raise_dictionary_422(
                "DICTIONARY_VALUE_NOT_FOUND",
                f"{field_base}Id wskazuje nieistniejaca pozycje slownika.",
                field_base=field_base,
                kod_typu=kod_typu,
            )
        if int(row.get("aktywny") or 0) != 1:
            _raise_dictionary_422(
                "DICTIONARY_VALUE_NOT_ACTIVE",
                f"{field_base}Id wskazuje nieaktywna pozycje slownika.",
                field_base=field_base,
                kod_typu=kod_typu,
            )
        candidate_ids.append(int(row["id"]))

    normalized_code = _normalize_optional_code(code_value)
    if normalized_code is not None:
        row = _resolve_dictionary_row_by_code(conn, kod_typu, normalized_code)
        if row is None:
            _raise_dictionary_422(
                "DICTIONARY_VALUE_NOT_FOUND",
                f"{field_base}KodPozycji nie wskazuje pozycji slownika.",
                field_base=field_base,
                kod_typu=kod_typu,
            )
        if int(row.get("aktywny") or 0) != 1:
            _raise_dictionary_422(
                "DICTIONARY_VALUE_NOT_ACTIVE",
                f"{field_base}KodPozycji wskazuje nieaktywna pozycje slownika.",
                field_base=field_base,
                kod_typu=kod_typu,
            )
        candidate_ids.append(int(row["id"]))

    normalized_name = _norm(name_value)
    if normalized_name is not None and normalized_name.lower() != "brak":
        row = _resolve_dictionary_row_by_name(conn, kod_typu, normalized_name)
        if row is None and allow_name_autocreate:
            created_id = _resolve_slownik_item_id(conn, kod_typu, normalized_name)
            if created_id is not None:
                row = _resolve_dictionary_row_by_id(conn, kod_typu, created_id)
        if row is None:
            _raise_dictionary_422(
                "DICTIONARY_VALUE_NOT_FOUND",
                f"{field_base} nie wskazuje pozycji slownika.",
                field_base=field_base,
                kod_typu=kod_typu,
            )
        if int(row.get("aktywny") or 0) != 1:
            _raise_dictionary_422(
                "DICTIONARY_VALUE_NOT_ACTIVE",
                f"{field_base} wskazuje nieaktywna pozycje slownika.",
                field_base=field_base,
                kod_typu=kod_typu,
            )
        candidate_ids.append(int(row["id"]))

    if not candidate_ids:
        if required:
            _raise_dictionary_422(
                "DICTIONARY_VALUE_REQUIRED",
                f"{field_base} jest wymagane (Id, KodPozycji lub Nazwa).",
                field_base=field_base,
                kod_typu=kod_typu,
            )
        return None

    unique_ids = list(dict.fromkeys(candidate_ids))
    if len(unique_ids) > 1:
        _raise_dictionary_422(
            "DICTIONARY_CONFLICT",
            f"Sprzeczne wartosci dla {field_base}: Id/KodPozycji/Nazwa wskazuja rozne pozycje.",
            field_base=field_base,
            kod_typu=kod_typu,
        )
    return unique_ids[0]


def _resolve_inspection_type_id_priority(
    conn: Any,
    *,
    field_base: str,
    id_value: int | None,
    code_value: str | None,
    name_value: str | None,
    required: bool = False,
) -> int | None:
    normalized_name: str | None = None
    if _norm(name_value) is not None and str(name_value).strip().lower() != "brak":
        normalized_key = _normalize_inspection_type_name(name_value)
        canonical_name, _, _ = _ALLOWED_INSPECTION_TYPE_BY_NORMALIZED[normalized_key]
        normalized_name = canonical_name

    resolved_id = _resolve_dictionary_id_priority(
        conn,
        kod_typu="typy_inspekcji",
        field_base=field_base,
        id_value=id_value,
        code_value=code_value,
        name_value=normalized_name,
        required=required,
        allow_name_autocreate=True,
    )
    if resolved_id is None:
        return None

    row = _resolve_dictionary_row_by_id(conn, "typy_inspekcji", resolved_id)
    if row is None:
        _raise_dictionary_422(
            "DICTIONARY_VALUE_NOT_FOUND",
            f"{field_base} wskazuje nieistniejaca pozycje slownika.",
            field_base=field_base,
            kod_typu="typy_inspekcji",
        )

    try:
        _normalize_inspection_type_name(str(row.get("nazwa_pozycji") or ""))
    except HTTPException:
        _raise_dictionary_422(
            "DICTIONARY_VALUE_NOT_ALLOWED",
            "Dozwolone typy inspekcji: Kontrola, Wizyta nadzorcza.",
            field_base=field_base,
            kod_typu="typy_inspekcji",
        )

    return resolved_id


def _status_code_by_id(conn: Any, status_id: int | None) -> str | None:
    if status_id is None:
        return None
    row = conn.execute(
        "SELECT kod_pozycji FROM slownik_pozycje WHERE id = ? LIMIT 1",
        (int(status_id),),
    ).fetchone()
    if row is None:
        return None
    return str(row["kod_pozycji"] or "").strip().upper() or None


def _count_related_by_inspection(conn: Any, table_name: str, inspection_id: int) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) AS total FROM {table_name} WHERE inspection_id = ?",
        (int(inspection_id),),
    ).fetchone()
    return int(row["total"]) if row is not None else 0


def _status_expectations_by_code(status_code: str | None) -> tuple[str, str]:
    code = str(status_code or "").strip().upper()
    if not code:
        return "any", "any"

    if code in _INSPECTION_STATUS_RELATION_RULES:
        return _INSPECTION_STATUS_RELATION_RULES[code]

    if (
        code in _INSPECTION_STATUS_REQUIRES_RECOMMENDATIONS_CODES
        and code in _INSPECTION_STATUS_FORBIDS_RECOMMENDATIONS_CODES
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Konflikt konfiguracji statusu inspekcji: "
                f"{code} jednoczesnie wymaga i zabrania zalecen"
            ),
        )

    if code in _INSPECTION_STATUS_REQUIRES_RECOMMENDATIONS_CODES:
        recommendations_expectation = "present"
    elif code in _INSPECTION_STATUS_FORBIDS_RECOMMENDATIONS_CODES:
        recommendations_expectation = "absent"
    else:
        recommendations_expectation = "any"

    if code in _INSPECTION_STATUS_REQUIRES_SANCTIONS_CODES and code in _INSPECTION_STATUS_FORBIDS_SANCTIONS_CODES:
        raise HTTPException(
            status_code=500,
            detail=(
                "Konflikt konfiguracji statusu inspekcji: "
                f"{code} jednoczesnie wymaga i zabrania wnioskow sankcyjnych"
            ),
        )

    if code in _INSPECTION_STATUS_REQUIRES_SANCTIONS_CODES:
        sanctions_expectation = "present"
    elif code in _INSPECTION_STATUS_FORBIDS_SANCTIONS_CODES:
        sanctions_expectation = "absent"
    else:
        sanctions_expectation = "any"

    return recommendations_expectation, sanctions_expectation


def _validate_status_relations_for_save(
    conn: Any,
    *,
    inspection_id: int | None,
    status_id: int | None,
) -> None:
    status_code = _status_code_by_id(conn, status_id)
    recommendations_expectation, sanctions_expectation = _status_expectations_by_code(status_code)

    if recommendations_expectation == "any" and sanctions_expectation == "any":
        return

    recommendations_count = 0
    sanctions_count = 0
    if inspection_id is not None:
        recommendations_count = _count_related_by_inspection(conn, "recommendations", int(inspection_id))
        sanctions_count = _count_related_by_inspection(conn, "risk_exposure_requests", int(inspection_id))

    violations: list[dict[str, Any]] = []
    if recommendations_expectation == "present" and recommendations_count == 0:
        violations.append(
            {
                "violationCode": "RECOMMENDATIONS_REQUIRED_MISSING",
                "violationCodeId": _VIOLATION_RECOMMENDATIONS_REQUIRED_MISSING_ID,
                "entity": "recommendations",
                "expected": "present",
                "actualCount": recommendations_count,
                "message": "Status wymaga co najmniej jednego zalecenia, ale nie znaleziono powiazanych zaleceń.",
            }
        )
    elif recommendations_expectation == "absent" and recommendations_count > 0:
        violations.append(
            {
                "violationCode": "RECOMMENDATIONS_FORBIDDEN_PRESENT",
                "violationCodeId": _VIOLATION_RECOMMENDATIONS_FORBIDDEN_PRESENT_ID,
                "entity": "recommendations",
                "expected": "absent",
                "actualCount": recommendations_count,
                "message": "Status wymaga braku zaleceń, ale dla tej inspekcji istnieja powiazane zalecenia.",
            }
        )

    if sanctions_expectation == "present" and sanctions_count == 0:
        violations.append(
            {
                "violationCode": "SANCTIONS_REQUIRED_MISSING",
                "violationCodeId": _VIOLATION_SANCTIONS_REQUIRED_MISSING_ID,
                "entity": "sanctionRequests",
                "expected": "present",
                "actualCount": sanctions_count,
                "message": "Status wymaga co najmniej jednego wniosku sankcyjnego, ale nie znaleziono powiazanych wnioskow.",
            }
        )
    elif sanctions_expectation == "absent" and sanctions_count > 0:
        violations.append(
            {
                "violationCode": "SANCTIONS_FORBIDDEN_PRESENT",
                "violationCodeId": _VIOLATION_SANCTIONS_FORBIDDEN_PRESENT_ID,
                "entity": "sanctionRequests",
                "expected": "absent",
                "actualCount": sanctions_count,
                "message": "Status wymaga braku wnioskow sankcyjnych, ale dla tej inspekcji istnieja powiazane wnioski.",
            }
        )

    if not violations:
        return

    raise HTTPException(
        status_code=409,
        detail={
            "code": _INSPECTION_STATUS_RELATIONS_ERROR_CODE,
            "codeId": _INSPECTION_STATUS_RELATIONS_ERROR_CODE_ID,
            "message": "Wybrany status nie jest zgodny z powiazaniami inspekcji.",
            "inspectionId": inspection_id,
            "statusId": status_id,
            "statusCode": status_code,
            "expectations": {
                "recommendations": recommendations_expectation,
                "sanctionRequests": sanctions_expectation,
            },
            "counts": {
                "recommendations": recommendations_count,
                "sanctionRequests": sanctions_count,
            },
            "violations": violations,
        },
    )


def _normalize_inspection_type_name(raw_value: str | None) -> str:
    value = _norm(raw_value)
    if value is None or value.lower() == "brak":
        raise HTTPException(status_code=400, detail="typInspekcji jest wymagane")

    normalized = " ".join(value.casefold().split())
    if normalized not in _ALLOWED_INSPECTION_TYPE_BY_NORMALIZED:
        raise HTTPException(status_code=400, detail="Dozwolone typy inspekcji: Kontrola, Wizyta nadzorcza")
    return normalized


def _resolve_inspection_type_id(conn: Any, raw_value: str | None) -> int:
    normalized = _normalize_inspection_type_name(raw_value)
    canonical_name, canonical_code, canonical_order = _ALLOWED_INSPECTION_TYPE_BY_NORMALIZED[normalized]

    row = conn.execute(
        """
        SELECT id
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'typy_inspekcji'
          AND lower(nazwa_pozycji) = lower(?)
        LIMIT 1
        """,
        (canonical_name,),
    ).fetchone()
    if row is not None:
        return int(row["id"])

    cursor = conn.execute(
        """
        INSERT INTO slownik_pozycje
        (kod_typu, kod_pozycji, nazwa_pozycji, kolejnosc, aktywny)
        VALUES ('typy_inspekcji', ?, ?, ?, 1)
        """,
        (canonical_code, canonical_name, canonical_order),
    )
    return int(cursor.lastrowid)


def _resolve_user_id_by_text(conn: Any, raw_value: str | None) -> int | None:
    value = _norm(raw_value)
    if value is None or value.lower() == "brak":
        return None

    row = conn.execute(
        "SELECT id FROM users WHERE lower(login) = lower(?) LIMIT 1",
        (value,),
    ).fetchone()
    if row is not None:
        return int(row["id"])

    row = conn.execute(
        """
        SELECT id FROM users
        WHERE lower(trim(imie || ' ' || nazwisko)) = lower(?)
        LIMIT 1
        """,
        (value,),
    ).fetchone()
    if row is not None:
        return int(row["id"])

    return None


def _get_member_ids(conn: Any, inspection_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT user_id FROM inspection_members WHERE inspection_id = ? ORDER BY id ASC",
        (inspection_id,),
    ).fetchall()
    return [int(r["user_id"]) for r in rows]


def _get_scope_ids(conn: Any, inspection_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT scope_id FROM inspection_scopes WHERE inspection_id = ? ORDER BY id ASC",
        (inspection_id,),
    ).fetchall()
    return [int(r["scope_id"]) for r in rows]


def _get_inspection_team_ids(conn: Any, inspection_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT team_id FROM inspection_teams WHERE inspection_id = ? ORDER BY team_id ASC",
        (inspection_id,),
    ).fetchall()
    return [int(r["team_id"]) for r in rows]


def _sync_inspection_teams(conn: Any, inspection_id: int, team_ids: list[int]) -> None:
    conn.execute("DELETE FROM inspection_teams WHERE inspection_id = ?", (inspection_id,))
    for team_id in team_ids:
        conn.execute(
            "INSERT OR IGNORE INTO inspection_teams (inspection_id, team_id) VALUES (?, ?)",
            (inspection_id, int(team_id)),
        )


def _validate_scope_ids(conn: Any, scope_ids: list[int]) -> list[int]:
    if not scope_ids:
        return []

    deduped = list(dict.fromkeys(int(x) for x in scope_ids))
    placeholders = ",".join(["?"] * len(deduped))
    rows = conn.execute(
        f"""
        SELECT id, aktywny
        FROM slownik_pozycje
        WHERE kod_typu = 'zakresy_inspekcji' AND id IN ({placeholders})
        """,
        tuple(deduped),
    ).fetchall()
    found_rows = {int(r["id"]): int(r["aktywny"] or 0) for r in rows}
    missing = [scope_id for scope_id in deduped if scope_id not in found_rows]
    inactive = [scope_id for scope_id in deduped if scope_id in found_rows and found_rows[scope_id] != 1]

    if missing:
        _raise_contract_422(
            "UNKNOWN_SCOPE_ID",
            "zakresInspekcjiIds zawiera nieistniejace pozycje slownika.",
            field="zakresInspekcjiIds",
            value=missing,
            kod_typu="zakresy_inspekcji",
        )
    if inactive:
        _raise_contract_422(
            "INACTIVE_DICTIONARY_ENTRY",
            "zakresInspekcjiIds zawiera nieaktywne pozycje slownika.",
            field="zakresInspekcjiIds",
            value=inactive,
            kod_typu="zakresy_inspekcji",
        )
    return deduped


def _normalize_and_validate_inspection_team_ids(
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
            "MISSING_DICTIONARY_ID",
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


def _resolve_scope_ids_from_text(conn: Any, raw_value: str | None) -> list[int]:
    value = _norm(raw_value)
    if value is None or value.lower() == "brak":
        return []

    if not _INSPECTIONS_LEGACY_SCOPE_TEXT_SPLIT_ENABLED:
        scope_id = _resolve_slownik_item_id(conn, "zakresy_inspekcji", value)
        return [scope_id] if scope_id is not None else []

    logger.warning(
        "Legacy parsing of zakresInspekcji string is used. "
        "Prefer zakresInspekcjiIds or zakresInspekcjiList. value=%r",
        value,
    )

    parts = [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
    if not parts:
        return []

    resolved: list[int] = []
    for part in parts:
        scope_id = _resolve_slownik_item_id(conn, "zakresy_inspekcji", part)
        if scope_id is not None:
            resolved.append(scope_id)
    return list(dict.fromkeys(resolved))


def _normalize_scope_text_list(raw_values: list[str] | None, field_name: str) -> list[str]:
    if raw_values is None:
        return []

    normalized: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            raise HTTPException(status_code=422, detail=f"{field_name} zawiera niepoprawny typ")
        value = raw.strip()
        if not value:
            raise HTTPException(status_code=422, detail=f"{field_name} nie moze zawierac pustych wartosci")
        if value.lower() == "brak":
            continue
        normalized.append(value)

    # Preserve order and remove duplicates.
    return list(dict.fromkeys(normalized))


def _resolve_scope_ids_from_list(conn: Any, raw_values: list[str] | None) -> list[int]:
    names = _normalize_scope_text_list(raw_values, "zakresInspekcjiList")
    resolved: list[int] = []
    for name in names:
        scope_id = _resolve_slownik_item_id(conn, "zakresy_inspekcji", name)
        if scope_id is not None:
            resolved.append(scope_id)
    return list(dict.fromkeys(resolved))


def _sync_scopes(conn: Any, inspection_id: int, scope_ids: list[int]) -> None:
    conn.execute("DELETE FROM inspection_scopes WHERE inspection_id = ?", (inspection_id,))
    for scope_id in scope_ids:
        conn.execute(
            "INSERT OR IGNORE INTO inspection_scopes (inspection_id, scope_id) VALUES (?, ?)",
            (inspection_id, scope_id),
        )


def _normalize_date_list(raw_values: list[str] | None, field_name: str, *, status_code: int = 400) -> list[str]:
    if raw_values is None:
        return []

    normalized: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            raise HTTPException(status_code=status_code, detail=f"{field_name} zawiera niepoprawny typ")
        value = raw.strip()
        if not value:
            raise HTTPException(status_code=status_code, detail=f"{field_name} nie moze zawierac pustych wartosci")
        if value.lower() in {"brak", "brak pisma", "0", "0000-00-00"}:
            raise HTTPException(status_code=status_code, detail=f"{field_name} ma niepoprawny format daty")
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise HTTPException(status_code=status_code, detail=f"{field_name} ma niepoprawny format daty") from exc
        normalized_value = parsed.isoformat()
        normalized.append(normalized_value)

    if len(normalized) != len(set(normalized)):
        raise HTTPException(status_code=status_code, detail=f"{field_name} zawiera duplikaty")

    normalized.sort()
    return normalized


def _normalize_optional_iso_date(raw_value: str | None, field_name: str) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise HTTPException(status_code=422, detail=f"{field_name} zawiera niepoprawny typ")
    value = raw_value.strip()
    if not value:
        return None
    if value.lower() in {"brak pisma", "0", "0000-00-00"}:
        raise HTTPException(status_code=422, detail=f"{field_name} ma niepoprawny format daty")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} ma niepoprawny format daty") from exc
    return parsed.isoformat()


def _is_legacy_missing_date_marker(raw_value: Any) -> bool:
    if raw_value is None:
        return False
    cleaned = str(raw_value).strip().lower()
    return cleaned in {"brak", "brak pisma", "0", "0000-00-00"}


def _normalize_optional_iso_date_from_db(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    cleaned = str(raw_value).strip()
    if not cleaned:
        return None
    if _is_legacy_missing_date_marker(cleaned):
        return None
    try:
        parsed = date.fromisoformat(cleaned[:10])
    except ValueError:
        return None
    return parsed.isoformat()


def _normalize_single_date_absence_state(
    *,
    date_value: str | None,
    brak_flag: bool,
    date_field_name: str,
    bool_field_name: str,
) -> tuple[str | None, bool]:
    if brak_flag and date_value is not None:
        raise HTTPException(status_code=422, detail=f"Gdy {bool_field_name}=true, {date_field_name} musi byc null")
    if date_value is not None:
        return date_value, False
    if brak_flag:
        return None, True
    return None, False


def _normalize_date_list_absence_state(
    *,
    date_values: list[str],
    brak_flag: bool,
    list_field_name: str,
    bool_field_name: str,
) -> tuple[list[str], bool]:
    if brak_flag and date_values:
        raise HTTPException(status_code=422, detail=f"Gdy {bool_field_name}=true, {list_field_name} musi byc []")
    if date_values:
        return date_values, False
    if brak_flag:
        return [], True
    return [], False


def _validate_inspection_date_range(start_value: str, end_value: str) -> None:
    try:
        start_date = date.fromisoformat(start_value)
        end_date = date.fromisoformat(end_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Niepoprawny format daty inspekcji") from exc

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="koniecInspekcji nie moze byc wczesniej niz poczatekInspekcji")


def _parse_dates_csv(csv_value: str | None) -> list[str]:
    if not csv_value:
        return []
    return [part for part in str(csv_value).split(",") if part]


def _next_inspection_lp(conn: Any) -> int:
    row = conn.execute("SELECT COALESCE(MAX(lp), 0) AS max_lp FROM inspections").fetchone()
    return int(row["max_lp"]) + 1


def _inspection_prefix_from_typ(typ_inspekcji: str | None) -> str:
    normalized = " ".join((typ_inspekcji or "").casefold().split())
    if normalized == "kontrola":
        return "K"
    if normalized == "wizyta nadzorcza":
        return "WN"
    raise HTTPException(status_code=400, detail="Dozwolone typy inspekcji: Kontrola, Wizyta nadzorcza")


def _next_inspection_code(conn: Any, typ_inspekcji_id: int | None, poczatek_inspekcji: str) -> str:
    try:
        year = str(date.fromisoformat(poczatek_inspekcji).year)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Niepoprawny format daty inspekcji") from exc

    typ_nazwa: str | None = None
    if typ_inspekcji_id is not None:
        row = conn.execute(
            "SELECT nazwa_pozycji FROM slownik_pozycje WHERE id = ? LIMIT 1",
            (typ_inspekcji_id,),
        ).fetchone()
        if row is not None:
            typ_nazwa = row["nazwa_pozycji"]

    prefix = _inspection_prefix_from_typ(typ_nazwa)
    pattern = f"{prefix}/{year}/%"
    existing = conn.execute(
        "SELECT kod_inspekcji FROM inspections WHERE kod_inspekcji LIKE ?",
        (pattern,),
    ).fetchall()

    max_seq = 0
    for row in existing:
        raw_code = str(row["kod_inspekcji"] or "").strip()
        parts = raw_code.split("/")
        if len(parts) != 3:
            continue
        if parts[0] != prefix or parts[1] != year:
            continue
        try:
            seq = int(parts[2])
        except ValueError:
            continue
        if seq > max_seq:
            max_seq = seq

    return f"{prefix}/{year}/{max_seq + 1}"


def _sync_multi_dates(
    conn: Any,
    inspection_id: int,
    date_type: str,
    target_dates: list[str],
    operator_user_id: int,
) -> None:
    existing_rows = conn.execute(
        """
        SELECT id, date_value
        FROM inspection_multi_dates
        WHERE inspection_id = ? AND date_type = ?
        """,
        (inspection_id, date_type),
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
            VALUES (?, ?, ?, ?, ?)
            """,
            (inspection_id, date_type, date_value, operator_user_id, operator_user_id),
        )


def _resolve_operator(conn: Any, operator_login: str | None) -> dict[str, Any]:
    login = (operator_login or "").strip()
    if not login:
        raise HTTPException(status_code=401, detail="Operator nie istnieje")

    row = conn.execute(
        """
        SELECT id, login, rola_id, zespol_id, aktywny, account_type
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
        "account_type": str(operator.get("account_type") or "diu").strip().lower(),
    }


def _is_inspector_diu_operator(operator: dict[str, Any]) -> bool:
    return int(operator.get("rola_id", 0)) == 1 and str(operator.get("account_type") or "").strip().lower() == "diu"


def _ensure_director(operator: dict[str, Any]) -> None:
    if int(operator["rola_id"]) != 3:
        raise HTTPException(status_code=403, detail="Brak uprawnien")


def _resolve_people_options_operator(conn: Any, operator_login: str | None) -> dict[str, Any]:
    login = (operator_login or "").strip()
    if not login:
        raise HTTPException(status_code=422, detail="Brak lub nieprawidlowy X-Operator-Login")

    row = conn.execute(
        """
        SELECT id, login, rola_id, zespol_id, aktywny, account_type
        FROM users
        WHERE lower(login)=lower(?)
        LIMIT 1
        """,
        (login,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Brak autoryzacji operatora")

    operator = dict(row)
    if int(operator["aktywny"]) != 1:
        raise HTTPException(status_code=403, detail="Operator nieaktywny lub brak dostepu")

    return {
        "id": int(operator["id"]),
        "login": operator["login"],
        "rola_id": int(operator["rola_id"]),
        "zespol_id": operator.get("zespol_id"),
        "account_type": str(operator.get("account_type") or "diu").strip().lower(),
    }


def _build_display_name(imie: str | None, nazwisko: str | None, login: str) -> str:
    full_name = " ".join(part for part in [
        (imie or "").strip(),
        (nazwisko or "").strip(),
    ] if part).strip()
    return full_name or login


def _resolve_team_names_by_ids(conn: Any, team_ids: list[int]) -> list[str]:
    if not team_ids:
        return []

    placeholders = ",".join(["?"] * len(team_ids))
    rows = conn.execute(
        f"SELECT id, nazwa FROM teams WHERE id IN ({placeholders})",
        tuple(int(team_id) for team_id in team_ids),
    ).fetchall()
    by_id = {int(row["id"]): str(row["nazwa"] or "").strip() for row in rows}
    result: list[str] = []
    for team_id in team_ids:
        name = by_id.get(int(team_id))
        if name:
            result.append(name)
    return result


def _validate_active_user_ids(conn: Any, user_ids: list[int]) -> None:
    for user_id in user_ids:
        row = conn.execute(
            "SELECT id, aktywny FROM users WHERE id=? LIMIT 1",
            (user_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} nie istnieje")
        if int(row["aktywny"]) != 1:
            raise HTTPException(status_code=403, detail=f"User {user_id} jest nieaktywny")


def _is_user_active(conn: Any, user_id: int) -> bool:
    row = conn.execute(
        "SELECT aktywny FROM users WHERE id = ? LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"User {int(user_id)} nie istnieje")
    return int(row["aktywny"]) == 1


def _can_team_lead_assign_own_technical_inactive_leader(conn: Any, operator: dict[str, Any], leader_user_id: int) -> bool:
    if int(operator.get("rola_id", 0)) != 2:
        return False

    row = conn.execute(
        """
        SELECT created_by_user_id, account_type
        FROM users
        WHERE id = ?
        LIMIT 1
        """,
        (int(leader_user_id),),
    ).fetchone()
    if row is None:
        return False

    created_by = row["created_by_user_id"]
    account_type = str(row["account_type"] or "").strip().lower()
    return created_by is not None and int(created_by) == int(operator["id"]) and account_type == "technical"


def _validate_leader_activity(conn: Any, operator: dict[str, Any], leader_user_id: int) -> None:
    row = conn.execute(
        "SELECT id, aktywny, account_type, created_by_user_id FROM users WHERE id = ? LIMIT 1",
        (int(leader_user_id),),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"User {int(leader_user_id)} nie istnieje")

    if int(row["aktywny"]) == 1:
        return

    account_type = str(row["account_type"] or "").strip().lower()
    if account_type in {"technical", "diu"}:
        # Directors can always point to historical placeholders (technical/inactive DIU).
        if int(operator.get("rola_id", 0)) == 3:
            return

        created_by = row["created_by_user_id"]
        # Team lead can point only to placeholders created by that team lead.
        if int(operator.get("rola_id", 0)) == 2 and created_by is not None and int(created_by) == int(operator["id"]):
            return

    _raise_business_403(
        "INACTIVE_USER_NOT_ALLOWED",
        "Wskazany użytkownik nieaktywny nie jest dozwolony w tym scenariuszu.",
    )


def _is_user_visible_on_list(conn: Any, user_id: int) -> bool:
    row = conn.execute(
        "SELECT list_visibility FROM users WHERE id = ? LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} nie istnieje")
    return str(row["list_visibility"] or "visible").strip().lower() != "hidden"


def _validate_leader_visibility(conn: Any, leader_user_id: int) -> None:
    if _is_user_visible_on_list(conn, int(leader_user_id)):
        return
    _raise_business_403(
        "USER_HIDDEN_NOT_ALLOWED",
        "Wskazany użytkownik ma listVisibility=hidden i nie może być osobą kierującą.",
    )


def _validate_visible_member_ids(conn: Any, user_ids: list[int]) -> None:
    for user_id in user_ids:
        if not _is_user_visible_on_list(conn, int(user_id)):
            raise HTTPException(
                status_code=422,
                detail=f"User {int(user_id)} ma listVisibility=hidden i nie moze byc czlonkiem skladu zespolu",
            )


def _enforce_leader_scope_by_role(conn: Any, operator: dict[str, Any], leader_user_id: int) -> None:
    # director: pelny zakres lidera wg zasad backendu
    if operator["rola_id"] == 3:
        return

    # team_lead: scope by team for active users, and own-created inactive technical/DIU placeholders
    if operator["rola_id"] == 2:
        if leader_user_id == operator["id"]:
            return

        target_row = conn.execute(
            "SELECT zespol_id, created_by_user_id, account_type, aktywny FROM users WHERE id = ? LIMIT 1",
            (leader_user_id,),
        ).fetchone()
        if target_row is None:
            raise HTTPException(status_code=404, detail=f"User {leader_user_id} nie istnieje")

        created_by = target_row["created_by_user_id"]

        # Team lead scope includes any user explicitly created by that operator.
        if created_by is not None and int(created_by) == int(operator["id"]):
            return

        operator_team_id = operator.get("zespol_id")
        target_team_id = target_row["zespol_id"]
        if operator_team_id is not None and target_team_id is not None and int(target_team_id) == int(operator_team_id):
            return

        _raise_business_403(
            "LEADER_OUT_OF_SCOPE",
            "Wskazana osoba kierująca jest poza zakresem operatora.",
        )
        return

    # pozostale role: zachowanie bez zmian
    return


def _enforce_member_scope_by_role(conn: Any, operator: dict[str, Any], member_ids: list[int]) -> None:
    if not member_ids:
        return

    if int(operator.get("rola_id", 0)) == 3:
        return

    if int(operator.get("rola_id", 0)) == 2:
        placeholders = ",".join("?" for _ in member_ids)
        rows = conn.execute(
            f"SELECT id, zespol_id, created_by_user_id FROM users WHERE id IN ({placeholders})",
            tuple(int(uid) for uid in member_ids),
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        for member_id in member_ids:
            row = by_id.get(int(member_id))
            if row is None:
                raise HTTPException(status_code=404, detail=f"User {int(member_id)} nie istnieje")
            # For inspection team composition, a team lead may include any existing
            # user as long as other validators pass (especially listVisibility=visible).
            continue


def _base_select_sql() -> str:
    return """
        SELECT
            i.id,
            i.lp,
            i.kod_inspekcji,
            i.created_by_user_id,
            np.nazwa_pozycji AS nazwa_podmiotu_nazwa,
            np.skrot_pozycji AS nazwa_podmiotu_skrot,
            i.typ_inspekcji_id,
            ti.kod_pozycji AS typ_inspekcji_kod_pozycji,
            ti.nazwa_pozycji AS typ_inspekcji_nazwa,
            ti.skrot_pozycji AS typ_inspekcji_skrot,
            (
                SELECT group_concat(x.scope_name, '; ')
                FROM (
                    SELECT sp.nazwa_pozycji AS scope_name
                    FROM inspection_scopes isc
                    JOIN slownik_pozycje sp ON sp.id = isc.scope_id
                    WHERE isc.inspection_id = i.id
                    ORDER BY lower(sp.nazwa_pozycji), sp.id
                ) x
            ) AS zakres_inspekcji_nazwa,
            (
                SELECT group_concat(x.scope_name, char(31))
                FROM (
                    SELECT sp.nazwa_pozycji AS scope_name
                    FROM inspection_scopes isc
                    JOIN slownik_pozycje sp ON sp.id = isc.scope_id
                    WHERE isc.inspection_id = i.id
                    ORDER BY lower(sp.nazwa_pozycji), sp.id
                ) x
            ) AS zakres_inspekcji_list_raw,
            (
                SELECT group_concat(x.scope_short, '; ')
                FROM (
                    SELECT sp.skrot_pozycji AS scope_short
                    FROM inspection_scopes isc
                    JOIN slownik_pozycje sp ON sp.id = isc.scope_id
                    WHERE isc.inspection_id = i.id
                      AND trim(COALESCE(sp.skrot_pozycji, '')) <> ''
                    ORDER BY lower(sp.skrot_pozycji), sp.id
                ) x
            ) AS zakres_inspekcji_skrot,
            i.poczatek_inspekcji,
            i.koniec_inspekcji,
            i.osoba_kierujaca_user_id,
            (
                SELECT group_concat(x.tid, ',')
                FROM (
                    SELECT it.team_id AS tid
                    FROM inspection_teams it
                    WHERE it.inspection_id = i.id
                    ORDER BY it.team_id ASC
                ) x
            ) AS inspection_team_ids_csv,
            (
                SELECT group_concat(x.tname, char(31))
                FROM (
                    SELECT t.nazwa AS tname
                    FROM inspection_teams it
                    JOIN teams t ON t.id = it.team_id
                    WHERE it.inspection_id = i.id
                    ORDER BY it.team_id ASC
                ) x
            ) AS inspection_team_names_raw,
            ulead.imie AS lead_imie,
            ulead.nazwisko AS lead_nazwisko,
            i.rynek_id,
            r.kod_pozycji AS rynek_kod_pozycji,
            r.nazwa_pozycji AS rynek_nazwa,
            i.rodzaj_podmiotu_id,
            rp.kod_pozycji AS rodzaj_podmiotu_kod_pozycji,
            rp.nazwa_pozycji AS rodzaj_podmiotu_nazwa,
            rp.skrot_pozycji AS rodzaj_podmiotu_skrot,
            i.aspekt_konsumencki,
            i.komentarz,
            i.szczegoly_dotyczace_zakresu,
            i.status_inspekcji_id,
            si.kod_pozycji AS status_kod_pozycji,
            si.nazwa_pozycji AS status_nazwa,
            si.skrot_pozycji AS status_skrot,
            i.data_protokolu_sprawozdania,
            i.data_doreczenia_protokolu,
            i.data_akceptacji_sprawozdania,
            i.data_doreczenia_pisma,
            i.brak_data_doreczenia_pisma,
            i.data_wyslania_pisma_z_zastrzezeniami,
            i.brak_data_wyslania_pisma_z_zastrzezeniami,
            i.data_pisma_zastrzezenia,
            i.brak_data_pisma_zastrzezenia,
            i.data_wplywu_pisma,
            i.brak_data_wplywu_pisma,
            i.data_wyslania_pisma_z_odpowiedzia,
            i.brak_data_wyslania_pisma_z_odpowiedzia,
            i.data_pisma_z_odpowiedzia,
            i.brak_data_pisma_z_odpowiedzia,
            i.brak_dat_akceptacji_noty,
            i.data_akceptacji_noty,
            i.data_zalecen,
            i.zaktualizowano_o,
            (
                SELECT group_concat(x.dv, ',')
                FROM (
                    SELECT imd.date_value AS dv
                    FROM inspection_multi_dates imd
                    WHERE imd.inspection_id = i.id
                      AND imd.date_type = 'AKCEPTACJA_NOTY'
                    ORDER BY imd.date_value ASC
                ) x
            ) AS data_akceptacji_noty_list_csv,
            (
                SELECT group_concat(x.dv, ',')
                FROM (
                    SELECT imd.date_value AS dv
                    FROM inspection_multi_dates imd
                    WHERE imd.inspection_id = i.id
                      AND imd.date_type = 'ZALECENIE'
                    ORDER BY imd.date_value ASC
                ) x
            ) AS data_zalecen_list_csv,
            (
                SELECT group_concat(x.uid, ',')
                FROM (
                    SELECT u.id AS uid
                    FROM inspection_members im
                    JOIN users u ON u.id = im.user_id
                    WHERE im.inspection_id = i.id
                    ORDER BY lower(u.nazwisko), lower(u.imie), u.id
                ) x
            ) AS team_member_ids_csv,
            (
                SELECT group_concat(x.sid, ',')
                FROM (
                    SELECT isc.scope_id AS sid
                    FROM inspection_scopes isc
                    WHERE isc.inspection_id = i.id
                    ORDER BY isc.id ASC
                ) x
            ) AS scope_ids_csv,
            (
                SELECT group_concat(x.full_name, '; ')
                FROM (
                    SELECT trim(u.imie || ' ' || u.nazwisko) AS full_name
                    FROM inspection_members im
                    JOIN users u ON u.id = im.user_id
                    WHERE im.inspection_id = i.id
                    ORDER BY lower(u.nazwisko), lower(u.imie), u.id
                ) x
            ) AS sklad_zespolu_z_relacji
        FROM inspections i
        LEFT JOIN slownik_pozycje np ON np.id = i.nazwa_podmiotu_id
        LEFT JOIN slownik_pozycje ti ON ti.id = i.typ_inspekcji_id
        LEFT JOIN users ulead ON ulead.id = i.osoba_kierujaca_user_id
        LEFT JOIN slownik_pozycje r ON r.id = i.rynek_id
        LEFT JOIN slownik_pozycje rp ON rp.id = i.rodzaj_podmiotu_id
        LEFT JOIN slownik_pozycje si ON si.id = i.status_inspekcji_id
    """


def _can_edit_inspection(conn: Any, inspection_id: int, operator: dict[str, Any], created_by_user_id: int | None = None) -> bool:
    if operator["rola_id"] == 3:
        return True

    if created_by_user_id is None:
        row = conn.execute(
            "SELECT created_by_user_id FROM inspections WHERE id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()
        if row is None:
            return False
        created_by_user_id = row["created_by_user_id"]

    if created_by_user_id is not None and int(created_by_user_id) == operator["id"]:
        return True

    member_row = conn.execute(
        "SELECT 1 FROM inspection_members WHERE inspection_id = ? AND user_id = ? LIMIT 1",
        (inspection_id, operator["id"]),
    ).fetchone()
    if member_row is not None:
        return True

    if operator["rola_id"] == 2 and operator["zespol_id"] is not None:
        lead_row = conn.execute(
            """
            SELECT 1
            FROM inspection_members im
            JOIN users u ON u.id = im.user_id
            WHERE im.inspection_id = ?
                            AND (
                                        u.zespol_id = ?
                                        OR u.created_by_user_id = ?
                                    )
            LIMIT 1
            """,
                        (inspection_id, operator["zespol_id"], operator["id"]),
        ).fetchone()
        if lead_row is not None:
            return True

    return False


def _row_to_structure_payload(conn: Any, row: dict[str, Any], can_edit: bool) -> dict[str, Any]:
    lead_full = _norm(" ".join([row.get("lead_imie") or "", row.get("lead_nazwisko") or ""]).strip())
    sklad_rel = _norm(row.get("sklad_zespolu_z_relacji"))

    member_ids_csv = row.get("team_member_ids_csv")
    member_ids: list[int] = []
    if member_ids_csv:
        member_ids = [int(x) for x in str(member_ids_csv).split(",") if x.strip()]

    scope_ids_csv = row.get("scope_ids_csv")
    scope_ids: list[int] = []
    if scope_ids_csv:
        scope_ids = [int(x) for x in str(scope_ids_csv).split(",") if x.strip()]

    inspection_team_ids_csv = row.get("inspection_team_ids_csv")
    inspection_team_ids: list[int] = []
    if inspection_team_ids_csv:
        inspection_team_ids = [int(x) for x in str(inspection_team_ids_csv).split(",") if x.strip()]

    inspection_team_names_raw = row.get("inspection_team_names_raw")
    inspection_team_names: list[str] = []
    if inspection_team_names_raw:
        inspection_team_names = [x for x in str(inspection_team_names_raw).split(_SCOPE_LIST_SEPARATOR) if x]

    if not inspection_team_names and inspection_team_ids:
        inspection_team_names = _resolve_team_names_by_ids(conn, inspection_team_ids)

    scope_names_raw = row.get("zakres_inspekcji_list_raw")
    scope_names: list[str] = []
    if scope_names_raw:
        scope_names = [x for x in str(scope_names_raw).split(_SCOPE_LIST_SEPARATOR) if x]

    data_akceptacji_noty_list = _parse_dates_csv(row.get("data_akceptacji_noty_list_csv"))
    data_zalecen_list = _parse_dates_csv(row.get("data_zalecen_list_csv"))
    raw_data_wyslania_pisma_z_odpowiedzia = row.get("data_wyslania_pisma_z_odpowiedzia")
    legacy_brak_data_wyslania_pisma_z_odpowiedzia = _is_legacy_missing_date_marker(raw_data_wyslania_pisma_z_odpowiedzia)
    brak_data_wyslania_pisma_z_odpowiedzia = (
        int(row.get("brak_data_wyslania_pisma_z_odpowiedzia") or 0) == 1
    ) or legacy_brak_data_wyslania_pisma_z_odpowiedzia
    data_wyslania_pisma_z_odpowiedzia = (
        None if brak_data_wyslania_pisma_z_odpowiedzia else raw_data_wyslania_pisma_z_odpowiedzia
    )

    return {
        "id": row["id"],
        "kodInspekcji": row.get("kod_inspekcji"),
        "canEdit": can_edit,
        "nazwaPodmiotu": row.get("nazwa_podmiotu_nazwa") or "brak",
        "nazwaPodmiotuSkrocona": row.get("nazwa_podmiotu_skrot"),
        "typInspekcjiId": row.get("typ_inspekcji_id"),
        "typInspekcjiKodPozycji": row.get("typ_inspekcji_kod_pozycji"),
        "typInspekcji": row.get("typ_inspekcji_nazwa") or "brak",
        "typInspekcjiSkrocona": row.get("typ_inspekcji_skrot"),
        "zakresInspekcji": row.get("zakres_inspekcji_nazwa") or "brak",
        "zakresInspekcjiList": scope_names,
        "zakresInspekcjiSkrocona": row.get("zakres_inspekcji_skrot"),
        "zakresInspekcjiIds": scope_ids,
        "poczatekInspekcji": row.get("poczatek_inspekcji"),
        "koniecInspekcji": row.get("koniec_inspekcji"),
        "osobaKierujacaUserId": row.get("osoba_kierujaca_user_id"),
        "teamMemberUserIds": member_ids,
        "inspectionTeamIds": inspection_team_ids,
        "inspection_team_ids": inspection_team_ids,
        "inspectionTeamsList": inspection_team_names,
        "osobaKierujaca": lead_full or "brak",
        "skladZespolu": sklad_rel or "brak",
        "rynekId": row.get("rynek_id"),
        "rynekKodPozycji": row.get("rynek_kod_pozycji"),
        "rynek": row.get("rynek_nazwa") or "brak",
        "rodzajPodmiotuId": row.get("rodzaj_podmiotu_id"),
        "rodzajPodmiotuKodPozycji": row.get("rodzaj_podmiotu_kod_pozycji"),
        "rodzajPodmiotu": row.get("rodzaj_podmiotu_nazwa") or "brak",
        "rodzajPodmiotuSkrocona": row.get("rodzaj_podmiotu_skrot"),
        "aspektKonsumencki": row.get("aspekt_konsumencki"),
        "dataProtokolu": row.get("data_protokolu_sprawozdania"),
        "dataDoreczeniaProtokolu": row.get("data_doreczenia_protokolu"),
        "dataAkceptacjiSprawozdania": row.get("data_akceptacji_sprawozdania"),
        "dataDoreczeniaPisma": row.get("data_doreczenia_pisma"),
        "brakDataDoreczeniaPisma": int(row.get("brak_data_doreczenia_pisma") or 0) == 1,
        "dataWyslaniaPismaZZastrzezeniami": row.get("data_wyslania_pisma_z_zastrzezeniami"),
        "brakDataWyslaniaPismaZZastrzezeniami": int(row.get("brak_data_wyslania_pisma_z_zastrzezeniami") or 0) == 1,
        "dataPismaZastrzezenia": row.get("data_pisma_zastrzezenia"),
        "brakDataPismaZastrzezenia": int(row.get("brak_data_pisma_zastrzezenia") or 0) == 1,
        "dataWplywuPisma": row.get("data_wplywu_pisma"),
        "brakDataWplywuPisma": int(row.get("brak_data_wplywu_pisma") or 0) == 1,
        "dataWyslaniaPismaZOdpowiedzia": data_wyslania_pisma_z_odpowiedzia,
        "brakDataWyslaniaPismaZOdpowiedzia": brak_data_wyslania_pisma_z_odpowiedzia,
        "dataPismaZOdpowiedzia": row.get("data_pisma_z_odpowiedzia"),
        "brakDataPismaZOdpowiedzia": int(row.get("brak_data_pisma_z_odpowiedzia") or 0) == 1,
        "dataAkceptacjiNotyList": data_akceptacji_noty_list,
        "brakDatAkceptacjiNoty": int(row.get("brak_dat_akceptacji_noty") or 0) == 1,
        "dataZalecenList": data_zalecen_list,
        # Legacy single-date fields are kept for transitional compatibility.
        "dataAkceptacjiNoty": data_akceptacji_noty_list[-1] if data_akceptacji_noty_list else None,
        "dataZalecen": row.get("data_zalecen") or (data_zalecen_list[-1] if data_zalecen_list else None),
        "statusId": row.get("status_inspekcji_id"),
        "statusKodPozycji": row.get("status_kod_pozycji"),
        "status": row.get("status_nazwa") or "brak",
        "statusSkrocona": row.get("status_skrot"),
        "komentarz": row.get("komentarz"),
        "szczegolyDotyczaceZakresu": row.get("szczegoly_dotyczace_zakresu"),
        "zaktualizowanoO": row.get("zaktualizowano_o"),
    }


def _insert_or_replace_members(conn: Any, inspection_id: int, member_ids: list[int]) -> None:
    conn.execute("DELETE FROM inspection_members WHERE inspection_id = ?", (inspection_id,))
    for user_id in member_ids:
        conn.execute(
            "INSERT OR IGNORE INTO inspection_members (inspection_id, user_id) VALUES (?, ?)",
            (inspection_id, user_id),
        )


def _resolve_leader_and_members(
    conn: Any,
    payload: InspectionStructureCreate | InspectionStructureUpdate,
    fields_present: set[str] | None,
    operator: dict[str, Any],
    current_leader_user_id: int | None = None,
    current_member_ids: list[int] | None = None,
    allow_force_operator: bool = False,
) -> tuple[int | None, list[int] | None, bool]:
    is_update = fields_present is not None
    leader_was_explicitly_sent = fields_present is not None and "osobaKierujacaUserId" in fields_present

    force_operator = bool(getattr(payload, "forceOperatorAsLeader", False)) and allow_force_operator
    leader_user_id = getattr(payload, "osobaKierujacaUserId", None)

    if force_operator:
        leader_user_id = operator["id"]
    elif leader_user_id is None and is_update:
        leader_user_id = current_leader_user_id

    if leader_user_id is None and not is_update:
        raise HTTPException(status_code=400, detail="osobaKierujacaUserId jest wymagane")

    if _is_inspector_diu_operator(operator):
        if leader_user_id is None:
            leader_user_id = int(operator["id"]) if not is_update else current_leader_user_id

        leader_changed_by_request = (
            leader_was_explicitly_sent
            and (
                (int(current_leader_user_id) if current_leader_user_id is not None else None)
                != (int(leader_user_id) if leader_user_id is not None else None)
            )
        )
        enforce_self_leader_rule = (not is_update) or leader_changed_by_request or force_operator

        if enforce_self_leader_rule and (leader_user_id is None or int(leader_user_id) != int(operator["id"])):
            _raise_business_403(
                "INSPECTOR_LEADER_MUST_BE_SELF",
                "Dla aktywnego inspektora DIU osoba kierująca musi być zalogowanym operatorem.",
                requiredLeaderUserId=int(operator["id"]),
            )

        if not is_update:
            leader_user_id = int(operator["id"])

    member_ids = getattr(payload, "teamMemberUserIds", None)
    updated_members_flag = False
    if member_ids is not None:
        updated_members_flag = True
        member_ids = [int(x) for x in member_ids]
    elif not is_update:
        if leader_user_id is not None and _is_user_visible_on_list(conn, leader_user_id):
            member_ids = [leader_user_id]
        else:
            member_ids = []
        updated_members_flag = True
    else:
        member_ids = current_member_ids

    leader_changed_by_request = (
        leader_was_explicitly_sent
        and (
            (int(current_leader_user_id) if current_leader_user_id is not None else None)
            != (int(leader_user_id) if leader_user_id is not None else None)
        )
    )
    leader_scope_should_be_checked = (not is_update) or leader_changed_by_request or force_operator
    if leader_user_id is not None and leader_scope_should_be_checked:
        _enforce_leader_scope_by_role(conn, operator, leader_user_id)
        _validate_leader_visibility(conn, leader_user_id)

    if member_ids is not None:
        seen: set[int] = set()
        deduped: list[int] = []
        for uid in member_ids:
            if uid not in seen:
                seen.add(uid)
                deduped.append(uid)
        member_ids = deduped
        if (
            leader_user_id is not None
            and leader_user_id not in member_ids
            and _is_user_visible_on_list(conn, leader_user_id)
        ):
            member_ids.append(leader_user_id)
        if member_ids:
            if updated_members_flag or not is_update:
                _validate_visible_member_ids(conn, member_ids)
            _enforce_member_scope_by_role(conn, operator, member_ids)

    return leader_user_id, member_ids, updated_members_flag


def _enforce_operator_in_created_team(operator: dict[str, Any], leader_user_id: int | None, member_ids: list[int] | None) -> None:
    members = set(member_ids or [])
    operator_id = int(operator["id"])
    if leader_user_id is not None and int(leader_user_id) == operator_id:
        return
    if operator_id in members:
        return
    raise HTTPException(
        status_code=403,
        detail="Operator musi byc osoba kierujaca lub czlonkiem skladu zespolu inspekcyjnego",
    )


def _is_technical_user(conn: Any, user_id: int | None) -> bool:
    if user_id is None:
        return False
    row = conn.execute(
        "SELECT account_type FROM users WHERE id = ? LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if row is None:
        return False
    return str(row["account_type"] or "").strip().lower() == "technical"


def _can_bypass_operator_team_guard_for_inactive_leader(
    conn: Any,
    operator: dict[str, Any],
    leader_user_id: int | None,
) -> bool:
    if leader_user_id is None:
        return False

    row = conn.execute(
        """
        SELECT aktywny, account_type, created_by_user_id
        FROM users
        WHERE id = ?
        LIMIT 1
        """,
        (int(leader_user_id),),
    ).fetchone()
    if row is None:
        return False

    if int(row["aktywny"]) == 1:
        return False

    account_type = str(row["account_type"] or "").strip().lower()
    if account_type not in {"technical", "diu"}:
        return False

    role_id = int(operator.get("rola_id", 0))
    if role_id == 3:
        return True
    if role_id != 2:
        return False

    created_by = row["created_by_user_id"]
    return created_by is not None and int(created_by) == int(operator["id"])


@router.get("/api/structure/inspections", response_model=InspectionStructureListResponse)
def list_structure_inspections(
    sortBy: str = Query(default="poczatekInspekcji"),
    sortOrder: str = Query(default="asc"),
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    poczatek_inspekcji_sort_sql = """
        CASE
            WHEN i.poczatek_inspekcji LIKE '__.__.____'
                THEN substr(i.poczatek_inspekcji, 7, 4) || '-' || substr(i.poczatek_inspekcji, 4, 2) || '-' || substr(i.poczatek_inspekcji, 1, 2)
            ELSE i.poczatek_inspekcji
        END
    """.strip()
    allowed_sort_columns = {
        "lp": "i.lp",
        "kodInspekcji": "i.kod_inspekcji",
        "id": "i.id",
        "poczatekInspekcji": poczatek_inspekcji_sort_sql,
        "koniecInspekcji": "i.koniec_inspekcji",
    }
    sort_aliases = {
        "kod_inspekcji": "kodInspekcji",
        "poczatek_inspekcji": "poczatekInspekcji",
        "koniec_inspekcji": "koniecInspekcji",
        "startDate": "poczatekInspekcji",
    }

    normalized_sort_by = sort_aliases.get((sortBy or "").strip(), sortBy)
    if normalized_sort_by not in allowed_sort_columns:
        raise HTTPException(status_code=400, detail="Niepoprawny sortBy")

    direction = sortOrder.lower()
    if direction not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="Niepoprawny sortOrder")

    order_sql = f" ORDER BY {allowed_sort_columns[normalized_sort_by]} {direction.upper()}, i.id ASC"

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_INSPECTIONS_READ)
        rows = conn.execute(_base_select_sql() + order_sql).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            can_edit = _can_edit_inspection(
                conn,
                inspection_id=row_dict["id"],
                operator=operator,
                created_by_user_id=row_dict.get("created_by_user_id"),
            )
            items.append(_row_to_structure_payload(conn, row_dict, can_edit=can_edit))

    return {"items": items, "total": len(items)}


@router.get("/api/structure/inspections/{inspection_id}", response_model=InspectionStructureRead)
def get_structure_inspection(
    inspection_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_INSPECTIONS_READ)
        row = conn.execute(
            _base_select_sql() + " WHERE i.id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Inspection not found")

        row_dict = dict(row)
        can_edit = _can_edit_inspection(
            conn=conn,
            inspection_id=row_dict["id"],
            operator=operator,
            created_by_user_id=row_dict.get("created_by_user_id"),
        )

        payload = _row_to_structure_payload(conn, row_dict, can_edit=can_edit)

    return payload


@router.post("/api/structure/inspections", response_model=InspectionStructureRead, status_code=201)
def create_structure_inspection(
    payload: InspectionStructureCreate,
    response: Response,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)
        _validate_inspection_date_range(payload.poczatekInspekcji, payload.koniecInspekcji)
        nazwa_podmiotu_id = _resolve_slownik_item_id(conn, "nazwy_podmiotow", payload.nazwaPodmiotu)
        typ_inspekcji_id = _resolve_dictionary_id_strict(
            conn,
            kod_typu="typy_inspekcji",
            field_base="typInspekcji",
            id_value=payload.typInspekcjiId,
            code_value=payload.typInspekcjiKodPozycji,
            name_value=payload.typInspekcji,
            required=True,
        )
        typ_row = _resolve_dictionary_row_by_id(conn, "typy_inspekcji", int(typ_inspekcji_id))
        if typ_row is None:
            _raise_contract_422(
                "UNKNOWN_DICTIONARY_ID",
                "typInspekcjiId nie wskazuje aktywnej pozycji slownika.",
                field="typInspekcjiId",
                value=payload.typInspekcjiId,
                kod_typu="typy_inspekcji",
            )
        try:
            _normalize_inspection_type_name(str(typ_row.get("nazwa_pozycji") or ""))
        except HTTPException:
            _raise_contract_422(
                "UNKNOWN_DICTIONARY_ID",
                "Dozwolone typy inspekcji: Kontrola, Wizyta nadzorcza.",
                field="typInspekcjiId",
                value=payload.typInspekcjiId,
                kod_typu="typy_inspekcji",
            )

        scope_ids = _resolve_scope_ids_strict(
            conn,
            scope_ids=payload.zakresInspekcjiIds,
            scope_list=payload.zakresInspekcjiList,
            scope_text=payload.zakresInspekcji,
            required=True,
        )
        zakres_inspekcji_id = scope_ids[0] if scope_ids else None
        rynek_id = _resolve_dictionary_id_strict(
            conn,
            kod_typu="rynki",
            field_base="rynek",
            id_value=payload.rynekId,
            code_value=payload.rynekKodPozycji,
            name_value=payload.rynek,
            required=False,
        )
        rodzaj_podmiotu_id = _resolve_dictionary_id_strict(
            conn,
            kod_typu="rodzaje_podmiotu",
            field_base="rodzajPodmiotu",
            id_value=payload.rodzajPodmiotuId,
            code_value=payload.rodzajPodmiotuKodPozycji,
            name_value=payload.rodzajPodmiotu,
            required=False,
        )
        status_id = _resolve_dictionary_id_strict(
            conn,
            kod_typu="statusy_inspekcji",
            field_base="status",
            id_value=payload.statusId,
            code_value=payload.statusKodPozycji,
            name_value=payload.status,
            required=True,
        )
        _validate_status_relations_for_save(conn, inspection_id=None, status_id=status_id)
        szczegoly_dotyczace_zakresu = _normalize_optional_text_with_limit(
            payload.szczegolyDotyczaceZakresu,
            "szczegolyDotyczaceZakresu",
            _SZCZEGOLY_DOTYCZACE_ZAKRESU_MAX_LEN,
        )

        raw_akceptacje = payload.dataAkceptacjiNotyList
        if raw_akceptacje is None and payload.dataAkceptacjiNoty:
            raw_akceptacje = [payload.dataAkceptacjiNoty]
        data_akceptacji_noty_list = _normalize_date_list(raw_akceptacje, "dataAkceptacjiNotyList", status_code=422)
        data_akceptacji_noty_list, brak_dat_akceptacji_noty = _normalize_date_list_absence_state(
            date_values=data_akceptacji_noty_list,
            brak_flag=bool(payload.brakDatAkceptacjiNoty),
            list_field_name="dataAkceptacjiNotyList",
            bool_field_name="brakDatAkceptacjiNoty",
        )

        data_doreczenia_pisma, brak_data_doreczenia_pisma = _normalize_single_date_absence_state(
            date_value=_normalize_optional_iso_date(payload.dataDoreczeniaPisma, "dataDoreczeniaPisma"),
            brak_flag=bool(payload.brakDataDoreczeniaPisma),
            date_field_name="dataDoreczeniaPisma",
            bool_field_name="brakDataDoreczeniaPisma",
        )
        data_wyslania_pisma_z_zastrzezeniami, brak_data_wyslania_pisma_z_zastrzezeniami = _normalize_single_date_absence_state(
            date_value=_normalize_optional_iso_date(payload.dataWyslaniaPismaZZastrzezeniami, "dataWyslaniaPismaZZastrzezeniami"),
            brak_flag=bool(payload.brakDataWyslaniaPismaZZastrzezeniami),
            date_field_name="dataWyslaniaPismaZZastrzezeniami",
            bool_field_name="brakDataWyslaniaPismaZZastrzezeniami",
        )
        data_pisma_zastrzezenia, brak_data_pisma_zastrzezenia = _normalize_single_date_absence_state(
            date_value=_normalize_optional_iso_date(payload.dataPismaZastrzezenia, "dataPismaZastrzezenia"),
            brak_flag=bool(payload.brakDataPismaZastrzezenia),
            date_field_name="dataPismaZastrzezenia",
            bool_field_name="brakDataPismaZastrzezenia",
        )
        data_wplywu_pisma, brak_data_wplywu_pisma = _normalize_single_date_absence_state(
            date_value=_normalize_optional_iso_date(payload.dataWplywuPisma, "dataWplywuPisma"),
            brak_flag=bool(payload.brakDataWplywuPisma),
            date_field_name="dataWplywuPisma",
            bool_field_name="brakDataWplywuPisma",
        )
        data_wyslania_pisma_z_odpowiedzia, brak_data_wyslania_pisma_z_odpowiedzia = _normalize_single_date_absence_state(
            date_value=_normalize_optional_iso_date(payload.dataWyslaniaPismaZOdpowiedzia, "dataWyslaniaPismaZOdpowiedzia"),
            brak_flag=bool(payload.brakDataWyslaniaPismaZOdpowiedzia),
            date_field_name="dataWyslaniaPismaZOdpowiedzia",
            bool_field_name="brakDataWyslaniaPismaZOdpowiedzia",
        )
        data_pisma_z_odpowiedzia, brak_data_pisma_z_odpowiedzia = _normalize_single_date_absence_state(
            date_value=_normalize_optional_iso_date(payload.dataPismaZOdpowiedzia, "dataPismaZOdpowiedzia"),
            brak_flag=bool(payload.brakDataPismaZOdpowiedzia),
            date_field_name="dataPismaZOdpowiedzia",
            bool_field_name="brakDataPismaZOdpowiedzia",
        )

        if payload.teamMemberUserIds is None:
            if _norm(payload.skladZespolu) is not None:
                _raise_contract_422(
                    "LEGACY_TEXT_ONLY_VALUE",
                    "Pole teamMemberUserIds jest wymagane. Pole skladZespolu nie jest akceptowane przy zapisie.",
                    field="teamMemberUserIds",
                    value=payload.skladZespolu,
                )
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                "Pole teamMemberUserIds jest wymagane.",
                field="teamMemberUserIds",
                value=payload.teamMemberUserIds,
            )

        if payload.osobaKierujacaUserId is None and _norm(payload.osobaKierujaca) is not None and not payload.forceOperatorAsLeader:
            _raise_contract_422(
                "LEGACY_TEXT_ONLY_VALUE",
                "Pole osobaKierujacaUserId jest wymagane. Pole osobaKierujaca nie jest akceptowane przy zapisie.",
                field="osobaKierujacaUserId",
                value=payload.osobaKierujaca,
            )

        leader_user_id, member_ids, _ = _resolve_leader_and_members(
            conn,
            payload,
            fields_present=None,
            operator=operator,
            current_leader_user_id=None,
            current_member_ids=None,
            allow_force_operator=True,
        )
        inspection_team_ids = _normalize_and_validate_inspection_team_ids(
            conn,
            team_ids=payload.inspectionTeamIds,
            team_ids_legacy=payload.inspection_team_ids,
            field_sent=(payload.inspectionTeamIds is not None or payload.inspection_team_ids is not None),
        )

        cursor = conn.execute(
            """
            INSERT INTO inspections (
                lp,
                kod_inspekcji,
                created_by_user_id,
                nazwa_podmiotu_id,
                typ_inspekcji_id,
                zakres_inspekcji_id,
                poczatek_inspekcji,
                koniec_inspekcji,
                osoba_kierujaca_user_id,
                rynek_id,
                rodzaj_podmiotu_id,
                aspekt_konsumencki,
                komentarz,
                szczegoly_dotyczace_zakresu,
                status_inspekcji_id,
                data_protokolu_sprawozdania,
                data_doreczenia_protokolu,
                data_akceptacji_sprawozdania,
                data_doreczenia_pisma,
                brak_data_doreczenia_pisma,
                data_wyslania_pisma_z_zastrzezeniami,
                brak_data_wyslania_pisma_z_zastrzezeniami,
                data_pisma_zastrzezenia,
                brak_data_pisma_zastrzezenia,
                data_wplywu_pisma,
                brak_data_wplywu_pisma,
                data_wyslania_pisma_z_odpowiedzia,
                brak_data_wyslania_pisma_z_odpowiedzia,
                data_pisma_z_odpowiedzia,
                brak_data_pisma_z_odpowiedzia,
                brak_dat_akceptacji_noty,
                data_akceptacji_noty,
                data_zalecen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _next_inspection_lp(conn),
                _next_inspection_code(conn, typ_inspekcji_id, payload.poczatekInspekcji),
                operator["id"],
                nazwa_podmiotu_id,
                typ_inspekcji_id,
                zakres_inspekcji_id,
                payload.poczatekInspekcji,
                payload.koniecInspekcji,
                leader_user_id,
                rynek_id,
                rodzaj_podmiotu_id,
                payload.aspektKonsumencki,
                payload.komentarz,
                szczegoly_dotyczace_zakresu,
                status_id,
                payload.dataProtokolu,
                payload.dataDoreczeniaProtokolu,
                payload.dataAkceptacjiSprawozdania,
                data_doreczenia_pisma,
                1 if brak_data_doreczenia_pisma else 0,
                data_wyslania_pisma_z_zastrzezeniami,
                1 if brak_data_wyslania_pisma_z_zastrzezeniami else 0,
                data_pisma_zastrzezenia,
                1 if brak_data_pisma_zastrzezenia else 0,
                data_wplywu_pisma,
                1 if brak_data_wplywu_pisma else 0,
                data_wyslania_pisma_z_odpowiedzia,
                1 if brak_data_wyslania_pisma_z_odpowiedzia else 0,
                data_pisma_z_odpowiedzia,
                1 if brak_data_pisma_z_odpowiedzia else 0,
                1 if brak_dat_akceptacji_noty else 0,
                (data_akceptacji_noty_list[-1] if data_akceptacji_noty_list else None),
                None,
            ),
        )

        inspection_id = int(cursor.lastrowid)
        _insert_or_replace_members(conn, inspection_id, member_ids or [])
        _sync_scopes(conn, inspection_id, scope_ids)
        _sync_inspection_teams(conn, inspection_id, inspection_team_ids)
        _sync_multi_dates(conn, inspection_id, "AKCEPTACJA_NOTY", data_akceptacji_noty_list, operator["id"])

        kod_row = conn.execute("SELECT kod_inspekcji FROM inspections WHERE id = ? LIMIT 1", (inspection_id,)).fetchone()
        rekord_kod = str(kod_row["kod_inspekcji"]) if kod_row and kod_row["kod_inspekcji"] else str(inspection_id)
        row = conn.execute(
            _base_select_sql() + " WHERE i.id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="Failed to fetch created inspection")

        created_payload = _row_to_structure_payload(conn, dict(row), can_edit=True)
        changes = build_create_changes(
            [
                ("Kod inspekcji", created_payload.get("kodInspekcji")),
                ("Nazwa podmiotu", created_payload.get("nazwaPodmiotu")),
                ("Typ inspekcji", created_payload.get("typInspekcji")),
                ("Zakres inspekcji", created_payload.get("zakresInspekcji")),
                ("Początek inspekcji", created_payload.get("poczatekInspekcji")),
                ("Koniec inspekcji", created_payload.get("koniecInspekcji")),
                ("Osoba kierująca", created_payload.get("osobaKierujaca")),
                ("Skład zespołu", created_payload.get("skladZespolu")),
                ("Zespoły inspekcji", created_payload.get("inspectionTeamIds")),
                ("Rynek", created_payload.get("rynek")),
                ("Rodzaj podmiotu", created_payload.get("rodzajPodmiotu")),
                ("Status", created_payload.get("status")),
                ("Daty akceptacji noty", created_payload.get("dataAkceptacjiNotyList")),
                ("Daty zaleceń", created_payload.get("dataZalecenList")),
                ("Komentarz", created_payload.get("komentarz")),
                ("Szczegóły dotyczące zakresu", created_payload.get("szczegolyDotyczaceZakresu")),
            ]
        )

        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_CREATE,
                        REJESTR_INSPEKCJE, rekord_kod, changes)
        conn.commit()

    if row is None:
        raise HTTPException(status_code=500, detail="Failed to fetch created inspection")

    response.status_code = 201
    return created_payload


@router.get("/api/inspections/people-options", response_model=list[InspectionPeopleOption])
def list_inspection_people_options(
    includeInactive: bool = False,
    inspectionId: int | None = Query(default=None, ge=1),
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_people_options_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_INSPECTIONS_READ)

        # Backward-compatibility: endpoint keeps returning users regardless of activity.
        # UI should decide based on active/canBeLeader/visibleOnList.
        _ = includeInactive

        rows = conn.execute(
            f"""
            SELECT
                u.id,
                u.login,
                u.imie,
                u.nazwisko,
                u.aktywny,
                u.account_type,
                u.list_visibility,
                u.created_by_user_id,
                cbu.login AS created_by_login,
                u.zespol_id,
                COALESCE(sp.nazwa_pozycji, t.nazwa) AS team_name
            FROM users u
            LEFT JOIN teams t ON t.id = u.zespol_id
            LEFT JOIN users cbu ON cbu.id = u.created_by_user_id
            LEFT JOIN slownik_pozycje sp
                ON sp.id = t.slownik_pozycja_id
               AND sp.kod_typu = 'zespoly'
            """,
        ).fetchall()

        def _can_be_leader_for_operator(candidate_user_id: int) -> bool:
            try:
                if _is_inspector_diu_operator(operator) and int(candidate_user_id) != int(operator["id"]):
                    return False
                _enforce_leader_scope_by_role(conn, operator, int(candidate_user_id))
                _validate_leader_visibility(conn, int(candidate_user_id))
                return True
            except HTTPException:
                return False

        items: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            active = bool(row_dict["aktywny"])
            list_visibility = str(row_dict.get("list_visibility") or "visible")
            visible_on_list = list_visibility.strip().lower() != "hidden"
            created_by_operator = row_dict.get("created_by_user_id") is not None and int(row_dict["created_by_user_id"]) == int(operator["id"])
            account_type = str(row_dict.get("account_type") or "").strip().lower()
            can_be_leader = _can_be_leader_for_operator(int(row_dict["id"]))
            items.append(
                {
                    "id": int(row_dict["id"]),
                    "login": row_dict["login"],
                    "displayName": _build_display_name(row_dict.get("imie"), row_dict.get("nazwisko"), row_dict["login"]),
                    "active": active,
                    "canBeLeader": can_be_leader,
                    "listVisibility": list_visibility,
                    "visibleOnList": visible_on_list,
                    "createdByOperator": created_by_operator,
                    "createdByLogin": row_dict.get("created_by_login"),
                    "teamId": row_dict["zespol_id"],
                    "teamName": row_dict["team_name"],
                    "accountType": account_type or None,
                }
            )

        items.sort(key=lambda item: (item["displayName"].casefold(), item["id"]))

        if inspectionId is not None:
            leader_row = conn.execute(
                "SELECT osoba_kierujaca_user_id FROM inspections WHERE id = ? LIMIT 1",
                (int(inspectionId),),
            ).fetchone()
            if leader_row is None:
                raise HTTPException(status_code=404, detail="Inspection not found")

            current_leader_id = leader_row["osoba_kierujaca_user_id"]
            if current_leader_id is not None:
                current_leader_id = int(current_leader_id)
                leader_index = next((idx for idx, item in enumerate(items) if int(item["id"]) == current_leader_id), None)
                if leader_index is not None and leader_index > 0:
                    leader_item = items.pop(leader_index)
                    items.insert(0, leader_item)

        return items


@router.get("/api/inspections/team-options", response_model=list[InspectionTeamOption])
@router.get("/api/structure/inspections/team-options", response_model=list[InspectionTeamOption])
def list_inspection_team_options(
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_people_options_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_INSPECTIONS_READ)

        rows = conn.execute(
            """
            SELECT
                t.id,
                t.kod,
                t.nazwa
            FROM teams t
            ORDER BY t.id ASC
            """
        ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "id": int(row["id"]),
                    "code": str(row["kod"] or "").strip(),
                    "name": str(row["nazwa"] or "").strip(),
                    "isActive": True,
                }
            )

        items.sort(key=lambda item: item["id"])
        return items


@router.put("/api/structure/inspections/{inspection_id}", response_model=InspectionStructureRead)
def update_structure_inspection(
    inspection_id: int,
    payload: InspectionStructureUpdate,
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

        def _members_as_text() -> str | None:
            rows = conn.execute(
                """
                SELECT u.imie, u.nazwisko, u.login
                FROM inspection_members im
                JOIN users u ON u.id = im.user_id
                WHERE im.inspection_id = ?
                ORDER BY u.id
                """,
                (inspection_id,),
            ).fetchall()
            return ", ".join((f"{r['imie'] or ''} {r['nazwisko'] or ''}".strip() or r["login"]) for r in rows) or None

        def _scopes_as_text() -> str | None:
            rows = conn.execute(
                """
                SELECT sp.nazwa_pozycji
                FROM inspection_scopes isc
                JOIN slownik_pozycje sp ON sp.id = isc.scope_id
                WHERE isc.inspection_id = ?
                ORDER BY sp.nazwa_pozycji
                """,
                (inspection_id,),
            ).fetchall()
            return "; ".join(str(r["nazwa_pozycji"]) for r in rows) or None

        def _multi_dates_as_text(date_type: str) -> str | None:
            rows = conn.execute(
                """
                SELECT date_value
                FROM inspection_multi_dates
                WHERE inspection_id = ? AND date_type = ?
                ORDER BY date_value
                """,
                (inspection_id, date_type),
            ).fetchall()
            return ", ".join(str(r["date_value"]) for r in rows) or None

        def _multi_dates_as_list(date_type: str) -> list[str]:
            rows = conn.execute(
                """
                SELECT date_value
                FROM inspection_multi_dates
                WHERE inspection_id = ? AND date_type = ?
                ORDER BY date_value
                """,
                (inspection_id, date_type),
            ).fetchall()
            return [str(r["date_value"]) for r in rows]

        def _slownik_name_by_id(item_id: int | None) -> str | None:
            if item_id is None:
                return None
            row = conn.execute(
                "SELECT nazwa_pozycji FROM slownik_pozycje WHERE id = ? LIMIT 1",
                (int(item_id),),
            ).fetchone()
            return str(row["nazwa_pozycji"]) if row is not None else None

        def _user_display_by_id(user_id: int | None) -> str | None:
            if user_id is None:
                return None
            row = conn.execute(
                "SELECT imie, nazwisko, login FROM users WHERE id = ? LIMIT 1",
                (int(user_id),),
            ).fetchone()
            if row is None:
                return None
            full_name = f"{(row['imie'] or '').strip()} {(row['nazwisko'] or '').strip()}".strip()
            return full_name or str(row["login"])

        current_row = conn.execute(
            """
            SELECT id, created_by_user_id, osoba_kierujaca_user_id, poczatek_inspekcji, koniec_inspekcji,
                                     zaktualizowano_o, nazwa_podmiotu_id, typ_inspekcji_id, zakres_inspekcji_id,
                                                                         status_inspekcji_id, rynek_id, rodzaj_podmiotu_id, aspekt_konsumencki,
                     komentarz, szczegoly_dotyczace_zakresu, data_protokolu_sprawozdania, data_doreczenia_protokolu,
                   data_akceptacji_sprawozdania, data_doreczenia_pisma, data_pisma_zastrzezenia,
                     data_wplywu_pisma, data_wyslania_pisma_z_zastrzezeniami,
                                         data_wyslania_pisma_z_odpowiedzia, data_pisma_z_odpowiedzia,
                                                                                 brak_data_doreczenia_pisma, brak_data_wyslania_pisma_z_zastrzezeniami, brak_data_pisma_zastrzezenia,
                                                                                 brak_data_wplywu_pisma, brak_data_wyslania_pisma_z_odpowiedzia, brak_data_pisma_z_odpowiedzia,
                                                                                 brak_dat_akceptacji_noty, kod_inspekcji
            FROM inspections
            WHERE id = ?
            """,
            (inspection_id,),
        ).fetchone()
        if current_row is None:
            raise HTTPException(status_code=404, detail="Inspection not found")

        current = dict(current_row)
        members_before_str = _members_as_text()
        scopes_before_str = _scopes_as_text()
        dates_akceptacji_before = _multi_dates_as_text("AKCEPTACJA_NOTY")
        dates_zalecen_before = _multi_dates_as_text("ZALECENIE")
        dates_akceptacji_before_list = _multi_dates_as_list("AKCEPTACJA_NOTY")
        next_start_date = fields.get("poczatekInspekcji", current.get("poczatek_inspekcji"))
        next_end_date = fields.get("koniecInspekcji", current.get("koniec_inspekcji"))
        if next_start_date and next_end_date:
            _validate_inspection_date_range(next_start_date, next_end_date)

        if not _can_edit_inspection(
            conn,
            inspection_id=inspection_id,
            operator=operator,
            created_by_user_id=current.get("created_by_user_id"),
        ):
            raise HTTPException(status_code=403, detail="Brak uprawnien do edycji tej inspekcji")

        if "teamMemberUserIds" in fields and fields.get("teamMemberUserIds") is None:
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                "Pole teamMemberUserIds nie moze byc null.",
                field="teamMemberUserIds",
                value=fields.get("teamMemberUserIds"),
            )
        if "skladZespolu" in fields and "teamMemberUserIds" not in fields:
            _raise_contract_422(
                "LEGACY_TEXT_ONLY_VALUE",
                "Do aktualizacji skladu zespolu uzyj pola teamMemberUserIds.",
                field="teamMemberUserIds",
                value=fields.get("skladZespolu"),
            )
        if "osobaKierujacaUserId" in fields and fields.get("osobaKierujacaUserId") is None:
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                "Pole osobaKierujacaUserId nie moze byc null.",
                field="osobaKierujacaUserId",
                value=fields.get("osobaKierujacaUserId"),
            )
        if "osobaKierujaca" in fields and "osobaKierujacaUserId" not in fields:
            _raise_contract_422(
                "LEGACY_TEXT_ONLY_VALUE",
                "Do aktualizacji osoby kierujacej uzyj pola osobaKierujacaUserId.",
                field="osobaKierujacaUserId",
                value=fields.get("osobaKierujaca"),
            )
        if (
            ("inspectionTeamIds" in fields and fields.get("inspectionTeamIds") is None)
            or ("inspection_team_ids" in fields and fields.get("inspection_team_ids") is None)
        ):
            _raise_contract_422(
                "MISSING_DICTIONARY_ID",
                "Pole inspectionTeamIds nie moze byc null.",
                field="inspectionTeamIds",
                value=fields.get("inspectionTeamIds") if "inspectionTeamIds" in fields else fields.get("inspection_team_ids"),
            )

        assert_lock_for_save(conn, "inspections", inspection_id, operator, lock_token)
        assert_expected_updated_at(expected_updated_at, str(current.get("zaktualizowano_o") or ""))

        leader_user_id, member_ids, members_updated = _resolve_leader_and_members(
            conn,
            payload,
            set(fields.keys()),
            operator,
            current_leader_user_id=current.get("osoba_kierujaca_user_id"),
            current_member_ids=_get_member_ids(conn, inspection_id),
            allow_force_operator=False,
        )

        scopes_updated = False
        next_scope_ids = _get_scope_ids(conn, inspection_id)

        set_parts: list[str] = []
        values: list[Any] = []

        date_lists_updated = False
        data_akceptacji_noty_list: list[str] | None = None
        row_touched = False
        current_status_id = int(current["status_inspekcji_id"]) if current.get("status_inspekcji_id") is not None else None
        next_status_id = current_status_id
        current_inspection_team_ids = _get_inspection_team_ids(conn, inspection_id)
        inspection_teams_updated = False
        next_inspection_team_ids = current_inspection_team_ids

        if "nazwaPodmiotu" in fields:
            set_parts.append("nazwa_podmiotu_id = ?")
            values.append(_resolve_slownik_item_id(conn, "nazwy_podmiotow", fields["nazwaPodmiotu"]))
        typ_field_present = (
            "typInspekcjiId" in fields
            or "typInspekcjiKodPozycji" in fields
            or "typInspekcji" in fields
        )
        if typ_field_present:
            typ_inspekcji_id = _resolve_dictionary_id_strict(
                conn,
                kod_typu="typy_inspekcji",
                field_base="typInspekcji",
                id_value=fields.get("typInspekcjiId"),
                code_value=fields.get("typInspekcjiKodPozycji"),
                name_value=fields.get("typInspekcji"),
                required=True,
            )
            typ_row = _resolve_dictionary_row_by_id(conn, "typy_inspekcji", int(typ_inspekcji_id))
            if typ_row is None:
                _raise_contract_422(
                    "UNKNOWN_DICTIONARY_ID",
                    "typInspekcjiId nie wskazuje aktywnej pozycji slownika.",
                    field="typInspekcjiId",
                    value=fields.get("typInspekcjiId"),
                    kod_typu="typy_inspekcji",
                )
            try:
                _normalize_inspection_type_name(str(typ_row.get("nazwa_pozycji") or ""))
            except HTTPException:
                _raise_contract_422(
                    "UNKNOWN_DICTIONARY_ID",
                    "Dozwolone typy inspekcji: Kontrola, Wizyta nadzorcza.",
                    field="typInspekcjiId",
                    value=fields.get("typInspekcjiId"),
                    kod_typu="typy_inspekcji",
                )
            set_parts.append("typ_inspekcji_id = ?")
            values.append(typ_inspekcji_id)
        scope_field_present = (
            "zakresInspekcjiIds" in fields
            or "zakresInspekcjiList" in fields
            or "zakresInspekcji" in fields
        )
        if scope_field_present:
            next_scope_ids = _resolve_scope_ids_strict(
                conn,
                scope_ids=fields.get("zakresInspekcjiIds"),
                scope_list=fields.get("zakresInspekcjiList"),
                scope_text=fields.get("zakresInspekcji"),
                required=True,
            )
            scopes_updated = True
            set_parts.append("zakres_inspekcji_id = ?")
            values.append(next_scope_ids[0] if next_scope_ids else None)
        if "poczatekInspekcji" in fields:
            set_parts.append("poczatek_inspekcji = ?")
            values.append(fields["poczatekInspekcji"])
        if "koniecInspekcji" in fields:
            set_parts.append("koniec_inspekcji = ?")
            values.append(fields["koniecInspekcji"])
        if "osobaKierujacaUserId" in fields:
            set_parts.append("osoba_kierujaca_user_id = ?")
            values.append(leader_user_id)
        rynek_field_present = (
            "rynekId" in fields
            or "rynekKodPozycji" in fields
            or "rynek" in fields
        )
        if rynek_field_present:
            rynek_id = _resolve_dictionary_id_strict(
                conn,
                kod_typu="rynki",
                field_base="rynek",
                id_value=fields.get("rynekId"),
                code_value=fields.get("rynekKodPozycji"),
                name_value=fields.get("rynek"),
                required=False,
            )
            set_parts.append("rynek_id = ?")
            values.append(rynek_id)
        rodzaj_field_present = (
            "rodzajPodmiotuId" in fields
            or "rodzajPodmiotuKodPozycji" in fields
            or "rodzajPodmiotu" in fields
        )
        if rodzaj_field_present:
            rodzaj_podmiotu_id = _resolve_dictionary_id_strict(
                conn,
                kod_typu="rodzaje_podmiotu",
                field_base="rodzajPodmiotu",
                id_value=fields.get("rodzajPodmiotuId"),
                code_value=fields.get("rodzajPodmiotuKodPozycji"),
                name_value=fields.get("rodzajPodmiotu"),
                required=False,
            )
            set_parts.append("rodzaj_podmiotu_id = ?")
            values.append(rodzaj_podmiotu_id)
        if "aspektKonsumencki" in fields:
            set_parts.append("aspekt_konsumencki = ?")
            values.append(fields["aspektKonsumencki"])
        inspection_team_field_present = "inspectionTeamIds" in fields or "inspection_team_ids" in fields
        if inspection_team_field_present:
            next_inspection_team_ids = _normalize_and_validate_inspection_team_ids(
                conn,
                team_ids=fields.get("inspectionTeamIds"),
                team_ids_legacy=fields.get("inspection_team_ids"),
                field_sent=True,
            )
            if next_inspection_team_ids != current_inspection_team_ids:
                inspection_teams_updated = True
        status_field_present = (
            "statusId" in fields
            or "statusKodPozycji" in fields
            or "status" in fields
        )
        if status_field_present:
            next_status_id = _resolve_dictionary_id_strict(
                conn,
                kod_typu="statusy_inspekcji",
                field_base="status",
                id_value=fields.get("statusId"),
                code_value=fields.get("statusKodPozycji"),
                name_value=fields.get("status"),
                required=True,
            )
            set_parts.append("status_inspekcji_id = ?")
            values.append(next_status_id)
            if next_status_id != current_status_id:
                _validate_status_relations_for_save(conn, inspection_id=inspection_id, status_id=next_status_id)
        if "komentarz" in fields:
            set_parts.append("komentarz = ?")
            values.append(fields["komentarz"])
        if "szczegolyDotyczaceZakresu" in fields:
            set_parts.append("szczegoly_dotyczace_zakresu = ?")
            values.append(
                _normalize_optional_text_with_limit(
                    fields["szczegolyDotyczaceZakresu"],
                    "szczegolyDotyczaceZakresu",
                    _SZCZEGOLY_DOTYCZACE_ZAKRESU_MAX_LEN,
                )
            )
        if "dataProtokolu" in fields:
            set_parts.append("data_protokolu_sprawozdania = ?")
            values.append(fields["dataProtokolu"])
        if "dataDoreczeniaProtokolu" in fields:
            set_parts.append("data_doreczenia_protokolu = ?")
            values.append(fields["dataDoreczeniaProtokolu"])
        if "dataAkceptacjiSprawozdania" in fields:
            set_parts.append("data_akceptacji_sprawozdania = ?")
            values.append(fields["dataAkceptacjiSprawozdania"])
        current_pairs = {
            "dataDoreczeniaPisma": (
                _normalize_optional_iso_date(current.get("data_doreczenia_pisma"), "dataDoreczeniaPisma"),
                int(current.get("brak_data_doreczenia_pisma") or 0) == 1,
                "data_doreczenia_pisma",
                "brak_data_doreczenia_pisma",
                "brakDataDoreczeniaPisma",
            ),
            "dataPismaZastrzezenia": (
                _normalize_optional_iso_date(current.get("data_pisma_zastrzezenia"), "dataPismaZastrzezenia"),
                int(current.get("brak_data_pisma_zastrzezenia") or 0) == 1,
                "data_pisma_zastrzezenia",
                "brak_data_pisma_zastrzezenia",
                "brakDataPismaZastrzezenia",
            ),
            "dataWyslaniaPismaZZastrzezeniami": (
                _normalize_optional_iso_date(current.get("data_wyslania_pisma_z_zastrzezeniami"), "dataWyslaniaPismaZZastrzezeniami"),
                int(current.get("brak_data_wyslania_pisma_z_zastrzezeniami") or 0) == 1,
                "data_wyslania_pisma_z_zastrzezeniami",
                "brak_data_wyslania_pisma_z_zastrzezeniami",
                "brakDataWyslaniaPismaZZastrzezeniami",
            ),
            "dataWplywuPisma": (
                _normalize_optional_iso_date(current.get("data_wplywu_pisma"), "dataWplywuPisma"),
                int(current.get("brak_data_wplywu_pisma") or 0) == 1,
                "data_wplywu_pisma",
                "brak_data_wplywu_pisma",
                "brakDataWplywuPisma",
            ),
            "dataPismaZOdpowiedzia": (
                _normalize_optional_iso_date(current.get("data_pisma_z_odpowiedzia"), "dataPismaZOdpowiedzia"),
                int(current.get("brak_data_pisma_z_odpowiedzia") or 0) == 1,
                "data_pisma_z_odpowiedzia",
                "brak_data_pisma_z_odpowiedzia",
                "brakDataPismaZOdpowiedzia",
            ),
            "dataWyslaniaPismaZOdpowiedzia": (
                _normalize_optional_iso_date_from_db(current.get("data_wyslania_pisma_z_odpowiedzia")),
                (int(current.get("brak_data_wyslania_pisma_z_odpowiedzia") or 0) == 1)
                or _is_legacy_missing_date_marker(current.get("data_wyslania_pisma_z_odpowiedzia")),
                "data_wyslania_pisma_z_odpowiedzia",
                "brak_data_wyslania_pisma_z_odpowiedzia",
                "brakDataWyslaniaPismaZOdpowiedzia",
            ),
        }

        for api_date_field, (current_date, current_brak, db_date_col, db_brak_col, api_brak_field) in current_pairs.items():
            date_present = api_date_field in fields
            brak_present = api_brak_field in fields
            if not date_present and not brak_present:
                continue

            incoming_date = _normalize_optional_iso_date(fields.get(api_date_field) if date_present else current_date, api_date_field)
            incoming_brak = bool(fields.get(api_brak_field)) if brak_present else current_brak
            normalized_date, normalized_brak = _normalize_single_date_absence_state(
                date_value=incoming_date,
                brak_flag=incoming_brak,
                date_field_name=api_date_field,
                bool_field_name=api_brak_field,
            )

            if normalized_date != current_date:
                set_parts.append(f"{db_date_col} = ?")
                values.append(normalized_date)
            if normalized_brak != current_brak:
                set_parts.append(f"{db_brak_col} = ?")
                values.append(1 if normalized_brak else 0)
        if (
            "dataAkceptacjiNotyList" in fields
            or "dataAkceptacjiNoty" in fields
            or "brakDatAkceptacjiNoty" in fields
        ):
            if "dataAkceptacjiNotyList" in fields:
                raw_akceptacje_update = fields["dataAkceptacjiNotyList"]
            elif "dataAkceptacjiNoty" in fields:
                raw_akceptacje_update = [fields["dataAkceptacjiNoty"]] if fields["dataAkceptacjiNoty"] else []
            else:
                raw_akceptacje_update = dates_akceptacji_before_list

            normalized_akceptacje = _normalize_date_list(
                raw_akceptacje_update,
                "dataAkceptacjiNotyList",
                status_code=422,
            )
            normalized_akceptacje, normalized_brak_akceptacji = _normalize_date_list_absence_state(
                date_values=normalized_akceptacje,
                brak_flag=(
                    bool(fields["brakDatAkceptacjiNoty"])
                    if "brakDatAkceptacjiNoty" in fields
                    else int(current.get("brak_dat_akceptacji_noty") or 0) == 1
                ),
                list_field_name="dataAkceptacjiNotyList",
                bool_field_name="brakDatAkceptacjiNoty",
            )

            if normalized_akceptacje != dates_akceptacji_before_list:
                data_akceptacji_noty_list = normalized_akceptacje
                date_lists_updated = True
                set_parts.append("data_akceptacji_noty = ?")
                values.append(normalized_akceptacje[-1] if normalized_akceptacje else None)

            current_brak_akceptacji = int(current.get("brak_dat_akceptacji_noty") or 0) == 1
            if normalized_brak_akceptacji != current_brak_akceptacji:
                set_parts.append("brak_dat_akceptacji_noty = ?")
                values.append(1 if normalized_brak_akceptacji else 0)

        if set_parts:
            set_parts.append("zaktualizowano_o = ?")
            values.append(now_rfc3339_utc_ms())
            values.append(inspection_id)
            conn.execute(
                f"UPDATE inspections SET {', '.join(set_parts)} WHERE id = ?",
                tuple(values),
            )
            row_touched = True

        if members_updated and member_ids is not None:
            _insert_or_replace_members(conn, inspection_id, member_ids)

        if scopes_updated:
            _sync_scopes(conn, inspection_id, next_scope_ids)

        if inspection_teams_updated:
            _sync_inspection_teams(conn, inspection_id, next_inspection_team_ids)

        if date_lists_updated:
            if data_akceptacji_noty_list is not None:
                _sync_multi_dates(conn, inspection_id, "AKCEPTACJA_NOTY", data_akceptacji_noty_list, operator["id"])

        if not row_touched and (members_updated or scopes_updated or date_lists_updated):
            conn.execute(
                "UPDATE inspections SET zaktualizowano_o = ? WHERE id = ?",
                (now_rfc3339_utc_ms(), inspection_id),
            )

        # --- Audit log ---
        updated_row = conn.execute(
            """
            SELECT id, created_by_user_id, osoba_kierujaca_user_id, poczatek_inspekcji, koniec_inspekcji,
                                     zaktualizowano_o, nazwa_podmiotu_id, typ_inspekcji_id, zakres_inspekcji_id,
                                                                         status_inspekcji_id, rynek_id, rodzaj_podmiotu_id, aspekt_konsumencki,
                     komentarz, szczegoly_dotyczace_zakresu, data_protokolu_sprawozdania, data_doreczenia_protokolu,
                   data_akceptacji_sprawozdania, data_doreczenia_pisma, data_pisma_zastrzezenia,
                     data_wplywu_pisma, data_wyslania_pisma_z_zastrzezeniami,
                                         data_wyslania_pisma_z_odpowiedzia, data_pisma_z_odpowiedzia,
                                                                                 brak_data_doreczenia_pisma, brak_data_wyslania_pisma_z_zastrzezeniami, brak_data_pisma_zastrzezenia,
                                                                                 brak_data_wplywu_pisma, brak_data_wyslania_pisma_z_odpowiedzia, brak_data_pisma_z_odpowiedzia,
                                                                                 brak_dat_akceptacji_noty, kod_inspekcji
            FROM inspections
            WHERE id = ?
            """,
            (inspection_id,),
        ).fetchone()
        if updated_row is None:
            raise HTTPException(status_code=404, detail="Inspection not found")

        updated = dict(updated_row)
        members_after_str = _members_as_text()
        scopes_after_str = _scopes_as_text()
        dates_akceptacji_after = _multi_dates_as_text("AKCEPTACJA_NOTY")
        dates_zalecen_after = _multi_dates_as_text("ZALECENIE")

        changes: list[dict[str, Any]] = []

        def _add_change(field_label: str, before_value: Any, after_value: Any) -> None:
            if str(before_value or "") == str(after_value or ""):
                return
            changes.append({"pole": field_label, "przed": before_value, "po": after_value})

        _add_change(
            "Nazwa podmiotu",
            _slownik_name_by_id(current.get("nazwa_podmiotu_id")),
            _slownik_name_by_id(updated.get("nazwa_podmiotu_id")),
        )
        _add_change(
            "Typ inspekcji",
            _slownik_name_by_id(current.get("typ_inspekcji_id")),
            _slownik_name_by_id(updated.get("typ_inspekcji_id")),
        )
        _add_change("Początek inspekcji", current.get("poczatek_inspekcji"), updated.get("poczatek_inspekcji"))
        _add_change("Koniec inspekcji", current.get("koniec_inspekcji"), updated.get("koniec_inspekcji"))
        _add_change(
            "Osoba kierująca",
            _user_display_by_id(current.get("osoba_kierujaca_user_id")),
            _user_display_by_id(updated.get("osoba_kierujaca_user_id")),
        )
        _add_change(
            "Zespoły inspekcji",
            current_inspection_team_ids,
            _get_inspection_team_ids(conn, inspection_id),
        )
        _add_change("Skład zespołu", members_before_str, members_after_str)
        _add_change("Zakres inspekcji", scopes_before_str, scopes_after_str)
        _add_change(
            "Rynek",
            _slownik_name_by_id(current.get("rynek_id")),
            _slownik_name_by_id(updated.get("rynek_id")),
        )
        _add_change(
            "Rodzaj podmiotu",
            _slownik_name_by_id(current.get("rodzaj_podmiotu_id")),
            _slownik_name_by_id(updated.get("rodzaj_podmiotu_id")),
        )
        _add_change("Aspekt konsumencki", current.get("aspekt_konsumencki"), updated.get("aspekt_konsumencki"))
        _add_change(
            "Status",
            _slownik_name_by_id(current.get("status_inspekcji_id")),
            _slownik_name_by_id(updated.get("status_inspekcji_id")),
        )
        _add_change("Komentarz", current.get("komentarz"), updated.get("komentarz"))
        _add_change(
            "Szczegóły dotyczące zakresu",
            current.get("szczegoly_dotyczace_zakresu"),
            updated.get("szczegoly_dotyczace_zakresu"),
        )
        _add_change("Data protokołu", current.get("data_protokolu_sprawozdania"), updated.get("data_protokolu_sprawozdania"))
        _add_change("Data doręczenia protokołu", current.get("data_doreczenia_protokolu"), updated.get("data_doreczenia_protokolu"))
        _add_change(
            "Data akceptacji sprawozdania",
            current.get("data_akceptacji_sprawozdania"),
            updated.get("data_akceptacji_sprawozdania"),
        )
        _add_change("Data doręczenia pisma", current.get("data_doreczenia_pisma"), updated.get("data_doreczenia_pisma"))
        _add_change("Brak daty doręczenia pisma", current.get("brak_data_doreczenia_pisma"), updated.get("brak_data_doreczenia_pisma"))
        _add_change(
            "Data wysłania pisma z zastrzeżeniami",
            current.get("data_wyslania_pisma_z_zastrzezeniami"),
            updated.get("data_wyslania_pisma_z_zastrzezeniami"),
        )
        _add_change(
            "Brak daty wysłania pisma z zastrzeżeniami",
            current.get("brak_data_wyslania_pisma_z_zastrzezeniami"),
            updated.get("brak_data_wyslania_pisma_z_zastrzezeniami"),
        )
        _add_change("Data pisma zastrzeżenia", current.get("data_pisma_zastrzezenia"), updated.get("data_pisma_zastrzezenia"))
        _add_change("Brak daty pisma zastrzeżenia", current.get("brak_data_pisma_zastrzezenia"), updated.get("brak_data_pisma_zastrzezenia"))
        _add_change("Data wpływu pisma", current.get("data_wplywu_pisma"), updated.get("data_wplywu_pisma"))
        _add_change("Brak daty wpływu pisma", current.get("brak_data_wplywu_pisma"), updated.get("brak_data_wplywu_pisma"))
        _add_change(
            "Data wysłania pisma z odpowiedzią",
            current.get("data_wyslania_pisma_z_odpowiedzia"),
            updated.get("data_wyslania_pisma_z_odpowiedzia"),
        )
        _add_change(
            "Brak daty wysłania pisma z odpowiedzią",
            current.get("brak_data_wyslania_pisma_z_odpowiedzia"),
            updated.get("brak_data_wyslania_pisma_z_odpowiedzia"),
        )
        _add_change("Data pisma z odpowiedzią", current.get("data_pisma_z_odpowiedzia"), updated.get("data_pisma_z_odpowiedzia"))
        _add_change("Brak daty pisma z odpowiedzią", current.get("brak_data_pisma_z_odpowiedzia"), updated.get("brak_data_pisma_z_odpowiedzia"))
        _add_change("Daty akceptacji noty", dates_akceptacji_before, dates_akceptacji_after)
        _add_change("Brak dat akceptacji noty", current.get("brak_dat_akceptacji_noty"), updated.get("brak_dat_akceptacji_noty"))
        _add_change("Daty zaleceń", dates_zalecen_before, dates_zalecen_after)

        if changes:
            rekord_kod = str(updated.get("kod_inspekcji") or inspection_id)
            write_audit_log(
                conn,
                new_session_id(),
                operator["login"],
                AKCJA_UPDATE,
                REJESTR_INSPEKCJE,
                rekord_kod,
                changes,
            )
        # --- koniec audit log ---

        conn.commit()

        row = conn.execute(
            _base_select_sql() + " WHERE i.id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Inspection not found")

        response_payload = _row_to_structure_payload(conn, dict(row), can_edit=True)

    return response_payload


@router.get("/api/inspections", response_model=InspectionStructureListResponse)
def list_inspections_compat(
    sortBy: str = Query(default="poczatekInspekcji"),
    sortOrder: str = Query(default="asc"),
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    return list_structure_inspections(sortBy, sortOrder, x_operator_login)


@router.get("/api/inspections/{inspection_id}", response_model=InspectionStructureRead)
def get_inspection_compat(
    inspection_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    return get_structure_inspection(inspection_id, x_operator_login)


@router.post("/api/inspections", response_model=InspectionStructureRead, status_code=201)
def create_inspection_compat(
    payload: InspectionStructureCreate,
    response: Response,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    return create_structure_inspection(payload, response, x_operator_login)


@router.delete("/api/structure/inspections/{inspection_id}")
def delete_structure_inspection(
    inspection_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)
        _ensure_director(operator)

        inspection_row = conn.execute(
            "SELECT id, kod_inspekcji FROM inspections WHERE id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()
        if inspection_row is None:
            raise HTTPException(status_code=404, detail="Inspection not found")

        recommendations_row = conn.execute(
            "SELECT COUNT(*) AS total FROM recommendations WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()
        deleted_recommendations = int(recommendations_row["total"]) if recommendations_row is not None else 0

        risk_row = conn.execute(
            "SELECT COUNT(*) AS total FROM risk_exposure_requests WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()
        deleted_risk_exposure_requests = int(risk_row["total"]) if risk_row is not None else 0

        # risk_exposure_requests uses ON DELETE SET NULL, so remove linked records explicitly.
        conn.execute(
            "DELETE FROM risk_exposure_requests WHERE inspection_id = ?",
            (inspection_id,),
        )
        rekord_kod_del = str(inspection_row["kod_inspekcji"]) if inspection_row["kod_inspekcji"] else str(inspection_id)
        conn.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_DELETE,
                        REJESTR_INSPEKCJE, rekord_kod_del, [])
        conn.commit()

    return {
        "id": inspection_id,
        "deletedRecommendations": deleted_recommendations,
        "deletedSanctionRequests": deleted_risk_exposure_requests,
    }


@router.delete("/api/inspections/{inspection_id}")
def delete_inspection_compat(
    inspection_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    return delete_structure_inspection(inspection_id, x_operator_login)
