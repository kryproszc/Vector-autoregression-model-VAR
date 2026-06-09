from __future__ import annotations

from datetime import date
import logging
import os
import re
import unicodedata
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.audit import (
    AKCJA_CREATE,
    AKCJA_DELETE,
    AKCJA_UPDATE,
    REJESTR_DECYZJE,
    build_create_changes,
    build_decision_changes,
    new_session_id,
    write_audit_log,
)
from app.database import get_connection
from app.permissions import ROLE_EXTERNAL_USER, PERMISSION_OBLIGATING_DECISIONS_READ, require_permission, require_write_access
from app.record_locks import assert_expected_updated_at, assert_lock_for_save, now_rfc3339_utc_ms

router = APIRouter()
logger = logging.getLogger(__name__)

RECOMMENDATION_STATUS_BLOCK_ERROR_CODE = "RECOMMENDATION_STATUS_BLOCKS_DECISION_LINK"

_SECOND_INSTANCE_FIELDS: set[str] = {
    "osobyProwadzaceIIInstancjeIds",
    "osobyProwadzaceIIInstancjeList",
    "dataDecyzjiIIInstancji",
    "dataDoreczeniaDecyzjiIIInstancji",
    "rozstrzygniecieIIId",
    "rozstrzygniecieII",
}

_SECOND_INSTANCE_ASSIGNMENT_FIELDS: set[str] = {
    "osobyProwadzaceIIInstancjeIds",
    "osobyProwadzaceIIInstancjeList",
}

_FIRST_INSTANCE_FIELDS: set[str] = {
    "recommendationKodZalecenia",
    "nazwaPodmiotuId",
    "nazwaPodmiotu",
    "liczbaZalecen",
    "dataWszczeciaPostepowaniaIInstancji",
    "osobyProwadzaceIInstancjeIds",
    "osobyProwadzaceIInstancjeList",
    "dataDecyzjiIInstancji",
    "dataDoreczeniaDecyzjiIInstancji",
    "rozstrzygniecieIId",
    "rozstrzygniecieI",
    "dataWnioskuPonowneRozpatrzenie",
    "dataWplywuWnioskuPonowneRozpatrzenie",
}

_FIRST_INSTANCE_ASSIGNMENT_FIELDS: set[str] = {
    "osobyProwadzaceIInstancjeIds",
    "osobyProwadzaceIInstancjeList",
}


class ObligatingDecisionCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendationKodZalecenia": "Z/2026/12",
                "nazwaPodmiotu": "Podmiot A",
                "liczbaZalecen": 2,
                "dataWszczeciaPostepowaniaIInstancji": "2026-06-01",
                "osobyProwadzaceIInstancjeList": ["Jan Kowalski"],
                "dataDecyzjiIInstancji": "2026-06-10",
                "rozstrzygniecieI": "Utrzymano",
                "dataWnioskuPonowneRozpatrzenie": "2026-06-20",
                "dataWplywuWnioskuPonowneRozpatrzenie": "2026-06-22",
                "osobyProwadzaceIIInstancjeList": ["Anna Nowak"],
                "dataDecyzjiIIInstancji": "2026-07-01",
                "rozstrzygniecieII": "Zmieniono",
                "komentarz": "Komentarz",
            }
        }
    )

    recommendationKodZalecenia: str | None = None
    nazwaPodmiotuId: int | None = None
    nazwaPodmiotu: str | None = None
    liczbaZalecen: int | None = Field(default=None, ge=0)
    dataWszczeciaPostepowaniaIInstancji: str | None = None
    osobyProwadzaceIInstancjeIds: list[int] | None = None
    osobyProwadzaceIInstancjeList: list[str] | None = None
    dataDecyzjiIInstancji: str | None = None
    dataDoreczeniaDecyzjiIInstancji: str | None = None
    rozstrzygniecieIId: int | None = None
    rozstrzygniecieI: str | None = None
    dataWnioskuPonowneRozpatrzenie: str | None = None
    dataWplywuWnioskuPonowneRozpatrzenie: str | None = None
    osobyProwadzaceIIInstancjeIds: list[int] | None = None
    osobyProwadzaceIIInstancjeList: list[str] | None = None
    dataDecyzjiIIInstancji: str | None = None
    dataDoreczeniaDecyzjiIIInstancji: str | None = None
    rozstrzygniecieIIId: int | None = None
    rozstrzygniecieII: str | None = None
    komentarz: str | None = None


class ObligatingDecisionUpdate(BaseModel):
    lockToken: str | None = None
    expectedUpdatedAt: str | None = None
    recommendationKodZalecenia: str | None = None
    nazwaPodmiotuId: int | None = None
    nazwaPodmiotu: str | None = None
    liczbaZalecen: int | None = Field(default=None, ge=0)
    dataWszczeciaPostepowaniaIInstancji: str | None = None
    osobyProwadzaceIInstancjeIds: list[int] | None = None
    osobyProwadzaceIInstancjeList: list[str] | None = None
    dataDecyzjiIInstancji: str | None = None
    dataDoreczeniaDecyzjiIInstancji: str | None = None
    rozstrzygniecieIId: int | None = None
    rozstrzygniecieI: str | None = None
    dataWnioskuPonowneRozpatrzenie: str | None = None
    dataWplywuWnioskuPonowneRozpatrzenie: str | None = None
    osobyProwadzaceIIInstancjeIds: list[int] | None = None
    osobyProwadzaceIIInstancjeList: list[str] | None = None
    dataDecyzjiIIInstancji: str | None = None
    dataDoreczeniaDecyzjiIIInstancji: str | None = None
    rozstrzygniecieIIId: int | None = None
    rozstrzygniecieII: str | None = None
    komentarz: str | None = None


class ObligatingDecisionRead(BaseModel):
    id: int
    kodDecyzji: str | None = None
    canEdit: bool
    canEditIInstance: bool
    canAssignIInstancePeople: bool
    canEditIIInstance: bool
    canAssignIIInstancePeople: bool
    canEditComment: bool
    recommendationKodZalecenia: str | None = None
    nazwaPodmiotuId: int | None = None
    nazwaPodmiotu: str | None = None
    nazwaPodmiotuSkrocona: str | None = None
    nazwaPodmiotuSkrot: str | None = None
    liczbaZalecen: int | None = None
    dataWszczeciaPostepowaniaIInstancji: str | None = None
    osobyProwadzaceIInstancjeIds: list[int]
    osobyProwadzaceIInstancjeList: list[str]
    dataDecyzjiIInstancji: str | None = None
    dataDoreczeniaDecyzjiIInstancji: str | None = None
    rozstrzygniecieIId: int | None = None
    rozstrzygniecieI: str | None = None
    rozstrzygniecieISkrocona: str | None = None
    rozstrzygniecieISkrot: str | None = None
    dataWnioskuPonowneRozpatrzenie: str | None = None
    dataWplywuWnioskuPonowneRozpatrzenie: str | None = None
    osobyProwadzaceIIInstancjeIds: list[int]
    osobyProwadzaceIIInstancjeList: list[str]
    dataDecyzjiIIInstancji: str | None = None
    dataDoreczeniaDecyzjiIIInstancji: str | None = None
    rozstrzygniecieIIId: int | None = None
    rozstrzygniecieII: str | None = None
    rozstrzygniecieIISkrocona: str | None = None
    rozstrzygniecieIISkrot: str | None = None
    komentarz: str | None = None
    utworzonoO: str
    zaktualizowanoO: str


class ObligatingDecisionListResponse(BaseModel):
    items: list[ObligatingDecisionRead]
    total: int


class DecisionRecommendationOption(BaseModel):
    id: int
    kodZalecenia: str
    pozycja: int
    inspectionId: int | None = None
    inspectionKod: str | None = None
    nazwaPodmiotu: str
    nazwaPodmiotuSkrocona: str | None = None
    nazwaPodmiotuSkrot: str | None = None
    canCreateDecisionForRecommendation: bool
    createDeniedReasonCode: str | None = None


class DecisionPersonOption(BaseModel):
    id: int
    userId: int
    displayName: str
    label: str
    login: str | None = None
    accountType: str | None = None
    listVisibility: str
    createdByOperator: bool
    creatorLogin: str | None = None
    kodPozycji: str
    nazwaPozycji: str
    skrotPozycji: str | None = None


class DecisionPeopleOptionsResponse(BaseModel):
    items: list[DecisionPersonOption]


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


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


def _blocked_recommendation_status_codes_for_operator(operator: dict[str, Any] | None) -> set[str]:
    blocked = _parse_status_codes_env("OBLIGATING_DECISIONS_BLOCKED_RECOMMENDATION_STATUS_CODES")
    if operator is None:
        return blocked

    role_id_raw = operator.get("rola_id")
    if role_id_raw is None:
        return blocked

    role_id = int(role_id_raw)
    blocked |= _parse_status_codes_env(f"OBLIGATING_DECISIONS_BLOCKED_RECOMMENDATION_STATUS_CODES_ROLE_{role_id}")
    return blocked


def _recommendation_status_by_code(conn: Any, recommendation_kod: str | None) -> tuple[str | None, str | None]:
    code_value = _norm(recommendation_kod)
    if code_value is None:
        return None, None

    row = conn.execute(
        """
        SELECT sp.kod_pozycji AS status_code, sp.nazwa_pozycji AS status_label
        FROM recommendations r
        LEFT JOIN slownik_pozycje sp ON sp.id = r.status_zalecenia_id
        WHERE r.kod_zalecenia = ?
        LIMIT 1
        """,
        (code_value,),
    ).fetchone()
    if row is None:
        return None, None

    code_value = _norm(row["status_code"])
    label_value = _norm(row["status_label"])
    return (code_value.upper() if code_value else None), label_value


def _recommendation_status_is_blocked(
    conn: Any,
    recommendation_kod: str | None,
    operator: dict[str, Any] | None = None,
) -> tuple[bool, str | None, str | None]:
    status_code, status_label = _recommendation_status_by_code(conn, recommendation_kod)
    blocked_codes = _blocked_recommendation_status_codes_for_operator(operator)
    return (status_code in blocked_codes), status_code, status_label


def _ensure_recommendation_status_link_allowed(conn: Any, operator: dict[str, Any], recommendation_kod: str | None) -> None:
    blocked, status_code, status_label = _recommendation_status_is_blocked(
        conn,
        recommendation_kod,
        operator,
    )
    if not blocked:
        return

    _raise_domain_error(
        409,
        RECOMMENDATION_STATUS_BLOCK_ERROR_CODE,
        "Powiazanie decyzji z zaleceniem jest zablokowane dla statusu zalecenia.",
        recommendationKodZalecenia=_norm(recommendation_kod),
        recommendationStatusCode=status_code,
        recommendationStatus=status_label,
    )


def _domain_detail(code: str, message: str, **extra: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update(extra)
    return detail


def _raise_domain_error(status_code: int, code: str, message: str, **extra: Any) -> None:
    raise HTTPException(status_code=status_code, detail=_domain_detail(code, message, **extra))


def _ensure_second_instance_write_allowed(operator: dict[str, Any], fields: dict[str, Any]) -> None:
    if not any(field in fields for field in _SECOND_INSTANCE_FIELDS):
        return

    if int(operator.get("rola_id", 0)) not in {2, 3}:
        _raise_domain_error(
            403,
            "PERMISSION_DENIED_II_INSTANCE",
            "Pola II instancji moze edytowac tylko kierownik lub dyrektor",
        )


def _is_decision_person_ii(conn: Any, decision_id: int, operator: dict[str, Any]) -> bool:
    code = f"OSOBA_{int(operator['id'])}"
    row = conn.execute(
        """
        SELECT 1
        FROM obligating_decisions_persons_ii rel
        JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
        WHERE rel.obligating_decision_id = ?
          AND lower(sp.kod_typu) = 'osoby'
          AND (
                lower(sp.kod_pozycji) = lower(?)
                OR lower(COALESCE(sp.skrot_pozycji, '')) = lower(?)
              )
        LIMIT 1
        """,
        (int(decision_id), code, str(operator.get("login") or "")),
    ).fetchone()
    return row is not None


def _is_team_lead_of_decision_instance(conn: Any, operator: dict[str, Any], decision_id: int, instance: str) -> bool:
    if not _is_team_lead(operator):
        return False

    relation_table = {
        "I": "obligating_decisions_persons_i",
        "II": "obligating_decisions_persons_ii",
    }[instance]

    row = conn.execute(
        f"""
        SELECT 1
        FROM {relation_table} rel
        JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
        JOIN users u
          ON (
              lower(sp.kod_pozycji) = lower('OSOBA_' || CAST(u.id AS TEXT))
              OR lower(COALESCE(sp.skrot_pozycji, '')) = lower(u.login)
             )
        WHERE rel.obligating_decision_id = ?
                    AND (
                                u.zespol_id = ?
                                OR u.created_by_user_id = ?
                            )
        LIMIT 1
        """,
                (int(decision_id), int(operator["zespol_id"]), int(operator["id"])),
    ).fetchone()
    return row is not None


def _is_team_lead_of_slownik_users(conn: Any, operator: dict[str, Any], slownik_ids: list[int]) -> bool:
    if not _is_team_lead(operator) or not slownik_ids:
        return False

    placeholders = ",".join(["?"] * len(slownik_ids))
    row = conn.execute(
        f"""
        SELECT 1
        FROM slownik_pozycje sp
        JOIN users u
          ON (
              lower(sp.kod_pozycji) = lower('OSOBA_' || CAST(u.id AS TEXT))
              OR lower(COALESCE(sp.skrot_pozycji, '')) = lower(u.login)
             )
        WHERE sp.id IN ({placeholders})
                    AND (
                                u.zespol_id = ?
                                OR u.created_by_user_id = ?
                            )
        LIMIT 1
        """,
                tuple(list(slownik_ids) + [int(operator["zespol_id"]), int(operator["id"])]),
    ).fetchone()
    return row is not None


def _has_appeal_inflow_date(conn: Any, decision_id: int) -> bool:
    row = conn.execute(
        """
        SELECT data_wplywu_wniosku_ponowne_rozpatrzenie
        FROM obligating_decisions
        WHERE id = ?
        LIMIT 1
        """,
        (int(decision_id),),
    ).fetchone()
    if row is None:
        return False
    value = str(row["data_wplywu_wniosku_ponowne_rozpatrzenie"] or "").strip()
    return value != ""


def _has_decision_persons_for_instance(conn: Any, decision_id: int, instance: str) -> bool:
    relation_table = {
        "I": "obligating_decisions_persons_i",
        "II": "obligating_decisions_persons_ii",
    }[instance]
    row = conn.execute(
        f"SELECT 1 FROM {relation_table} WHERE obligating_decision_id = ? LIMIT 1",
        (int(decision_id),),
    ).fetchone()
    return row is not None


def _can_access_ii_instance(conn: Any, operator: dict[str, Any], decision_id: int) -> bool:
    if _is_director(operator):
        return True

    # II instancja jest aktywna dopiero po wpływie wniosku o ponowne rozpatrzenie.
    if not _has_appeal_inflow_date(conn, decision_id):
        return False

    has_ii_people = _has_decision_persons_for_instance(conn, decision_id, "II")
    if has_ii_people:
        if _is_decision_person_ii(conn, decision_id, operator):
            return True
        return _is_team_lead_of_decision_instance(conn, operator, decision_id, "II")

    # Gdy II jest puste: merytorycznie II mogą rozpocząć osoby z I i ich kierownicy.
    if _is_decision_person_i(conn, decision_id, operator):
        return True
    return _is_team_lead_of_decision_instance(conn, operator, decision_id, "I")


def _can_assign_ii_instance_people(conn: Any, operator: dict[str, Any], decision_id: int) -> bool:
    if _is_director(operator):
        return True
    if not _is_team_lead(operator):
        return False
    if not _has_appeal_inflow_date(conn, decision_id):
        return False
    # Business rule: once appeal inflow date is present, team leads of I-instance
    # people can keep managing II-instance assignments as well.
    if _is_team_lead_of_decision_instance(conn, operator, decision_id, "I"):
        return True

    has_ii_people = _has_decision_persons_for_instance(conn, decision_id, "II")
    if has_ii_people:
        return _is_team_lead_of_decision_instance(conn, operator, decision_id, "II")
    return _is_team_lead_of_decision_instance(conn, operator, decision_id, "I")


def _can_edit_instance_i(conn: Any, operator: dict[str, Any], decision_id: int) -> bool:
    if _is_director(operator):
        return True
    if _is_decision_person_i(conn, decision_id, operator):
        return True
    return _is_team_lead_of_decision_instance(conn, operator, decision_id, "I")


def _can_edit_instance_ii(conn: Any, operator: dict[str, Any], decision_id: int) -> bool:
    return _can_access_ii_instance(conn, operator, decision_id)


def _can_assign_i_instance_people(
    conn: Any,
    operator: dict[str, Any],
    decision_id: int,
) -> bool:
    if _is_director(operator):
        return True
    if not _is_team_lead(operator):
        return False
    return _is_team_lead_of_decision_instance(conn, operator, decision_id, "I")


def _can_assign_i_instance_people_for_decision(
    conn: Any,
    operator: dict[str, Any],
    decision_id: int,
    recommendation_kod: str | None,
) -> bool:
    if _is_director(operator):
        return True
    if not _is_team_lead(operator):
        return False

    if _can_manage_decision_by_recommendation_code(conn, operator, recommendation_kod):
        return True
    return _is_team_lead_of_decision_instance(conn, operator, decision_id, "I")


def _ensure_update_instance_permissions(
    conn: Any,
    operator: dict[str, Any],
    decision_id: int,
    recommendation_kod: str | None,
    fields: dict[str, Any],
) -> None:
    touches_i = any(field in fields for field in _FIRST_INSTANCE_FIELDS)
    touches_i_assignment = any(field in fields for field in _FIRST_INSTANCE_ASSIGNMENT_FIELDS)
    touches_ii = any(field in fields for field in _SECOND_INSTANCE_FIELDS)
    touches_ii_assignment = any(field in fields for field in _SECOND_INSTANCE_ASSIGNMENT_FIELDS)
    touches_comment = "komentarz" in fields

    if touches_i:
        if touches_i_assignment and not _can_assign_i_instance_people_for_decision(
            conn,
            operator,
            decision_id,
            recommendation_kod,
        ):
            _raise_domain_error(
                403,
                "PERMISSION_DENIED_I_INSTANCE",
                "Brak uprawnien do przypisania osob I instancji",
            )

        non_assignment_i = any(
            field in fields for field in (_FIRST_INSTANCE_FIELDS - _FIRST_INSTANCE_ASSIGNMENT_FIELDS)
        )
        if non_assignment_i and not _can_edit_instance_i(conn, operator, decision_id):
            _raise_domain_error(403, "PERMISSION_DENIED_I_INSTANCE", "Brak uprawnien do edycji I instancji")

    if touches_ii:
        if touches_ii_assignment and not _can_assign_ii_instance_people(conn, operator, decision_id):
            _raise_domain_error(
                403,
                "PERMISSION_DENIED_II_INSTANCE",
                "Brak uprawnien do przypisania osob II instancji",
            )

        non_assignment_ii = any(
            field in fields for field in (_SECOND_INSTANCE_FIELDS - _SECOND_INSTANCE_ASSIGNMENT_FIELDS)
        )
        if non_assignment_ii and not _can_edit_instance_ii(conn, operator, decision_id):
            _raise_domain_error(403, "PERMISSION_DENIED_II_INSTANCE", "Brak uprawnien do edycji II instancji")

    if touches_comment and not (
        _can_edit_instance_i(conn, operator, decision_id)
        or _can_edit_instance_ii(conn, operator, decision_id)
    ):
        _raise_domain_error(403, "PERMISSION_DENIED_I_INSTANCE", "Brak uprawnien do edycji komentarza")


def _slug_code(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    code = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_").upper()
    return code or "POZYCJA"


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


def _validate_optional_iso_date(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} ma niepoprawny format daty") from exc
    return parsed.isoformat()


def _parse_csv(csv_value: str | None) -> list[str]:
    if not csv_value:
        return []
    return [part for part in str(csv_value).split(";") if part]


def _parse_int_csv(csv_value: str | None) -> list[int]:
    if not csv_value:
        return []
    values: list[int] = []
    for part in str(csv_value).split(";"):
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return values


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


def _can_edit_fallback(conn: Any, operator: dict[str, Any], created_by_user_id: int | None) -> bool:
    if operator["rola_id"] == 3:
        return True

    if created_by_user_id is None:
        return False

    if int(created_by_user_id) == operator["id"]:
        return True

    if operator["rola_id"] == 2:
        if operator["zespol_id"] is None:
            return False
        author_row = conn.execute(
            "SELECT zespol_id FROM users WHERE id = ? LIMIT 1",
            (int(created_by_user_id),),
        ).fetchone()
        if author_row is None or author_row["zespol_id"] is None:
            return False
        return int(author_row["zespol_id"]) == int(operator["zespol_id"])

    return False


def _is_director(operator: dict[str, Any]) -> bool:
    return int(operator.get("rola_id", 0)) == 3


def _is_team_lead(operator: dict[str, Any]) -> bool:
    return int(operator.get("rola_id", 0)) == 2 and operator.get("zespol_id") is not None


def _normalize_list_visibility(value: str | None) -> str:
    normalized = str(value or "visible").strip().lower()
    return "hidden" if normalized == "hidden" else "visible"


def _display_name_for_person_option(row: dict[str, Any]) -> str:
    named = str(row.get("nazwa_pozycji") or "").strip()
    if named:
        return named
    full_name = f"{str(row.get('imie') or '').strip()} {str(row.get('nazwisko') or '').strip()}".strip()
    return full_name or str(row.get("login") or "")


def _ensure_person_slownik_id_for_user(conn: Any, user_row: dict[str, Any]) -> int:
    code = f"OSOBA_{int(user_row['user_id'])}"
    login_value = str(user_row.get("login") or "").strip()
    existing = conn.execute(
        """
        SELECT id
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'osoby'
          AND (
                lower(kod_pozycji) = lower(?)
                OR lower(COALESCE(skrot_pozycji, '')) = lower(?)
              )
        ORDER BY id ASC
        LIMIT 1
        """,
        (code, login_value),
    ).fetchone()
    if existing is not None:
        return int(existing["id"])

    display_name = f"{str(user_row.get('imie') or '').strip()} {str(user_row.get('nazwisko') or '').strip()}".strip() or code
    max_row = conn.execute(
        """
        SELECT COALESCE(MAX(kolejnosc), 0) AS max_kolejnosc
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'osoby'
        """
    ).fetchone()
    next_order = int(max_row["max_kolejnosc"]) + 1

    cursor = conn.execute(
        """
        INSERT INTO slownik_pozycje
        (kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
        VALUES ('osoby', ?, ?, ?, ?, 1)
        """,
        (code, display_name, login_value, next_order),
    )
    return int(cursor.lastrowid)


def _is_user_in_decision_people_scope(operator: dict[str, Any], row: dict[str, Any]) -> bool:
    user_id = int(row["user_id"])
    if _is_director(operator):
        return True

    if _is_team_lead(operator):
        in_team = row.get("zespol_id") is not None and int(row["zespol_id"]) == int(operator["zespol_id"])
        created_by_operator = row.get("created_by_user_id") is not None and int(row["created_by_user_id"]) == int(operator["id"])
        return bool(in_team or created_by_operator)

    return user_id == int(operator["id"])


def _build_available_decision_people(
    conn: Any,
    operator: dict[str, Any],
    endpoint_name: str,
    team_lead_full_scope: bool = False,
    force_full_scope: bool = False,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            u.id AS user_id,
            u.login,
            u.imie,
            u.nazwisko,
            u.account_type,
            u.list_visibility,
            u.created_by_user_id,
            cbu.login AS creator_login,
            u.zespol_id
        FROM users u
        LEFT JOIN users cbu ON cbu.id = u.created_by_user_id
        WHERE u.id IS NOT NULL
        ORDER BY u.id ASC
        """
    ).fetchall()

    candidates_before_filter = len(rows)

    visible_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        list_visibility = _normalize_list_visibility(row.get("list_visibility"))
        if list_visibility == "hidden":
            continue
        row["list_visibility"] = list_visibility
        visible_rows.append(row)

    candidates_after_hidden = len(visible_rows)

    if force_full_scope:
        scoped_rows = list(visible_rows)
    elif team_lead_full_scope and _is_team_lead(operator):
        scoped_rows = list(visible_rows)
    else:
        scoped_rows = [row for row in visible_rows if _is_user_in_decision_people_scope(operator, row)]
    candidates_after_rbac = len(scoped_rows)

    # Source data is users table; keep an explicit dedup stage for diagnostics clarity.
    deduplicated_rows = list(scoped_rows)
    candidates_after_dedup = len(deduplicated_rows)

    logger.info(
        "decision_people_options endpoint=%s operatorLogin=%s role=%s before_filters_count=%s after_hidden_filter_count=%s after_scope_filter_count=%s after_dedup_count=%s",
        endpoint_name,
        operator.get("login"),
        operator.get("rola_id"),
        candidates_before_filter,
        candidates_after_hidden,
        candidates_after_rbac,
        candidates_after_dedup,
    )

    items: list[dict[str, Any]] = []
    for row in deduplicated_rows:
        slownik_id = _ensure_person_slownik_id_for_user(conn, row)
        display_name = _display_name_for_person_option(row)
        account_type_raw = str(row.get("account_type") or "").strip().lower()
        person_code = f"OSOBA_{int(row['user_id'])}"
        item = {
            "id": int(slownik_id),
            "userId": int(row["user_id"]),
            "displayName": display_name,
            "label": display_name,
            "login": row.get("login"),
            "accountType": account_type_raw or None,
            "listVisibility": str(row["list_visibility"]),
            "createdByOperator": row.get("created_by_user_id") is not None and int(row["created_by_user_id"]) == int(operator["id"]),
            "creatorLogin": row.get("creator_login"),
            "kodPozycji": person_code,
            "nazwaPozycji": display_name,
            "skrotPozycji": row.get("login"),
        }

        if not isinstance(item.get("id"), int) or item["id"] <= 0:
            _raise_domain_error(
                500,
                "PEOPLE_OPTIONS_CONTRACT_VIOLATION",
                "People options item has invalid id",
                endpoint=endpoint_name,
                record=item,
            )
        if not isinstance(item.get("displayName"), str) or not str(item["displayName"]).strip():
            _raise_domain_error(
                500,
                "PEOPLE_OPTIONS_CONTRACT_VIOLATION",
                "People options item has invalid displayName",
                endpoint=endpoint_name,
                record=item,
            )
        if not isinstance(item.get("listVisibility"), str) or not str(item["listVisibility"]).strip():
            _raise_domain_error(
                500,
                "PEOPLE_OPTIONS_CONTRACT_VIOLATION",
                "People options item has invalid listVisibility",
                endpoint=endpoint_name,
                record=item,
            )

        items.append(item)

    return items


def _available_first_instance_people(conn: Any, operator: dict[str, Any]) -> list[dict[str, Any]]:
    return _build_available_decision_people(
        conn,
        operator,
        endpoint_name="first_instance",
        team_lead_full_scope=True,
    )


def _available_second_instance_people(conn: Any, operator: dict[str, Any]) -> list[dict[str, Any]]:
    return _build_available_decision_people(
        conn,
        operator,
        endpoint_name="second_instance",
        team_lead_full_scope=True,
    )


def _allowed_first_instance_person_ids(conn: Any, operator: dict[str, Any]) -> set[int]:
    strict_items = _build_available_decision_people(
        conn,
        operator,
        endpoint_name="scope_validation",
        team_lead_full_scope=False,
    )
    return {int(item["id"]) for item in strict_items}


def _ensure_first_instance_person_scope(conn: Any, operator: dict[str, Any], slownik_ids: list[int]) -> None:
    if not slownik_ids:
        return

    allowed_ids = _allowed_first_instance_person_ids(conn, operator)
    disallowed_ids = sorted({int(value) for value in slownik_ids if int(value) not in allowed_ids})
    if disallowed_ids:
        _raise_domain_error(
            403,
            "LEADER_OUT_OF_SCOPE",
            "Brak uprawnien do przypisania wybranych osob",
            invalidIds=disallowed_ids,
        )


def _ensure_first_instance_person_scope_on_create(conn: Any, operator: dict[str, Any], slownik_ids: list[int]) -> None:
    if not slownik_ids:
        return
    # UX exception for "Dodaj decyzje zobowiazujaca": team lead can select
    # any visible user on create, like director. Hidden users remain blocked
    # by _validate_slownik_item_id.
    if _is_team_lead(operator):
        return
    _ensure_first_instance_person_scope(conn, operator, slownik_ids)


def _is_author_of_unlinked_recommendation(
    conn: Any,
    operator: dict[str, Any],
    recommendation_kod: str | None,
) -> bool:
    scope = _recommendation_scope_by_code(conn, recommendation_kod)
    if scope is None:
        return False
    inspection_id, recommendation_created_by = scope
    if inspection_id is not None or recommendation_created_by is None:
        return False
    return int(recommendation_created_by) == int(operator["id"])


def _ensure_first_instance_person_scope_on_create_for_recommendation(
    conn: Any,
    operator: dict[str, Any],
    recommendation_kod: str | None,
    slownik_ids: list[int],
) -> None:
    if not slownik_ids:
        return
    if _is_author_of_unlinked_recommendation(conn, operator, recommendation_kod):
        return
    _ensure_first_instance_person_scope_on_create(conn, operator, slownik_ids)


def _recommendation_scope_by_code(
    conn: Any,
    recommendation_kod: str | None,
) -> tuple[int | None, int | None] | None:
    value = _norm(recommendation_kod)
    if value is None:
        return None

    row = conn.execute(
        """
        SELECT inspection_id, created_by_user_id
        FROM recommendations
        WHERE kod_zalecenia = ?
        LIMIT 1
        """,
        (value,),
    ).fetchone()
    if row is None:
        return None

    inspection_id = int(row["inspection_id"]) if row["inspection_id"] is not None else None
    recommendation_created_by = int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
    return inspection_id, recommendation_created_by


def _is_team_lead_of_user(conn: Any, operator: dict[str, Any], user_id: int | None) -> bool:
    if not _is_team_lead(operator) or user_id is None:
        return False

    author_row = conn.execute(
        "SELECT zespol_id FROM users WHERE id = ? LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if author_row is None or author_row["zespol_id"] is None:
        return False
    return int(author_row["zespol_id"]) == int(operator["zespol_id"])


def _is_team_lead_of_inspection_scope(conn: Any, operator: dict[str, Any], inspection_id: int | None) -> bool:
    if not _is_team_lead(operator) or inspection_id is None:
        return False

    leader_row = conn.execute(
        """
        SELECT 1
        FROM inspections i
        JOIN users u ON u.id = i.osoba_kierujaca_user_id
        WHERE i.id = ?
                    AND (
                                u.zespol_id = ?
                                OR (
                                        lower(COALESCE(u.account_type, '')) = 'technical'
                                        AND u.created_by_user_id = ?
                                )
                            )
        LIMIT 1
        """,
                (inspection_id, operator["zespol_id"], int(operator["id"])),
    ).fetchone()
    if leader_row is not None:
        return True

    member_row = conn.execute(
        """
        SELECT 1
        FROM inspection_members im
        JOIN users u ON u.id = im.user_id
        WHERE im.inspection_id = ?
                    AND (
                                u.zespol_id = ?
                                OR (
                                        lower(COALESCE(u.account_type, '')) = 'technical'
                                        AND u.created_by_user_id = ?
                                )
                            )
        LIMIT 1
        """,
                (inspection_id, operator["zespol_id"], int(operator["id"])),
    ).fetchone()
    return member_row is not None


def _can_create_decision_by_recommendation_scope(
    conn: Any,
    inspection_id: int | None,
    recommendation_created_by_user_id: int | None,
    operator: dict[str, Any],
) -> bool:
    if _is_director(operator):
        return True

    # Dla zaleceń przypiętych do inspekcji decyzję może utworzyć tylko kierownik
    # zespołu uczestniczącego w inspekcji (oraz dyrektor).
    if inspection_id is not None:
        return _is_team_lead_of_inspection_scope(conn, operator, inspection_id)

    # Dla zaleceń bez inspekcji: autor zalecenia, jego kierownik albo dyrektor.
    if recommendation_created_by_user_id is None:
        return False
    if int(recommendation_created_by_user_id) == int(operator["id"]):
        return True
    return _is_team_lead_of_user(conn, operator, recommendation_created_by_user_id)


def _can_create_decision_by_recommendation_code(conn: Any, operator: dict[str, Any], recommendation_kod: str | None) -> bool:
    scope = _recommendation_scope_by_code(conn, recommendation_kod)
    if scope is None:
        return False
    inspection_id, recommendation_created_by = scope
    return _can_create_decision_by_recommendation_scope(conn, inspection_id, recommendation_created_by, operator)


def _can_manage_decision_by_recommendation_scope(
    conn: Any,
    inspection_id: int | None,
    recommendation_created_by_user_id: int | None,
    operator: dict[str, Any],
) -> bool:
    if _is_director(operator):
        return True

    if inspection_id is not None:
        return _is_team_lead_of_inspection_scope(conn, operator, inspection_id)

    return _is_team_lead_of_user(conn, operator, recommendation_created_by_user_id)


def _can_manage_decision_by_recommendation_code(conn: Any, operator: dict[str, Any], recommendation_kod: str | None) -> bool:
    scope = _recommendation_scope_by_code(conn, recommendation_kod)
    if scope is None:
        return False
    inspection_id, recommendation_created_by = scope
    return _can_manage_decision_by_recommendation_scope(conn, inspection_id, recommendation_created_by, operator)


def _is_decision_person_i(conn: Any, decision_id: int, operator: dict[str, Any]) -> bool:
    code = f"OSOBA_{int(operator['id'])}"
    row = conn.execute(
        """
        SELECT 1
        FROM obligating_decisions_persons_i rel
        JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
        WHERE rel.obligating_decision_id = ?
          AND lower(sp.kod_typu) = 'osoby'
          AND (
                lower(sp.kod_pozycji) = lower(?)
                OR lower(COALESCE(sp.skrot_pozycji, '')) = lower(?)
              )
        LIMIT 1
        """,
        (int(decision_id), code, str(operator.get("login") or "")),
    ).fetchone()
    return row is not None


def _can_edit_decision(
    conn: Any,
    operator: dict[str, Any],
    decision_id: int,
    recommendation_kod: str | None,
) -> bool:
    if _can_edit_instance_i(conn, operator, decision_id):
        return True
    return _can_edit_instance_ii(conn, operator, decision_id)


def _decision_permissions(
    conn: Any,
    operator: dict[str, Any],
    decision_id: int,
    recommendation_kod: str | None,
) -> dict[str, bool]:
    can_edit_i = _can_edit_instance_i(conn, operator, decision_id)
    can_edit_ii = _can_edit_instance_ii(conn, operator, decision_id)
    can_assign_i = _can_assign_i_instance_people_for_decision(
        conn,
        operator,
        decision_id,
        recommendation_kod,
    )
    can_assign_ii = _can_assign_ii_instance_people(conn, operator, decision_id)
    can_edit_comment = can_edit_i or can_edit_ii
    return {
        "canEdit": bool(can_edit_i or can_edit_ii or can_assign_i or can_assign_ii),
        "canEditIInstance": bool(can_edit_i),
        "canAssignIInstancePeople": bool(can_assign_i),
        "canEditIIInstance": bool(can_edit_ii),
        "canAssignIIInstancePeople": bool(can_assign_ii),
        "canEditComment": bool(can_edit_comment),
    }


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


def _validate_slownik_item_id(conn: Any, kod_typu: str, slownik_id: int, field_name: str) -> int:
    row = conn.execute(
        """
                SELECT id, kod_pozycji, skrot_pozycji
        FROM slownik_pozycje
        WHERE id = ?
          AND lower(kod_typu) = lower(?)
        LIMIT 1
        """,
        (slownik_id, kod_typu),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=400, detail=f"{field_name} zawiera niepoprawne id")

    if str(kod_typu).strip().lower() == "osoby":
        linked_user = conn.execute(
            """
            SELECT id, rola_id, aktywny, list_visibility
            FROM users
            WHERE lower('OSOBA_' || CAST(id AS TEXT)) = lower(?)
               OR lower(login) = lower(COALESCE(?, ''))
            LIMIT 1
            """,
            (str(row["kod_pozycji"]), str(row["skrot_pozycji"] or "")),
        ).fetchone()
        if linked_user is not None and _normalize_list_visibility(linked_user["list_visibility"]) == "hidden":
            _raise_domain_error(
                422,
                "USER_HIDDEN_NOT_ALLOWED",
                f"{field_name} nie moze zawierac uzytkownikow ukrytych",
                field=field_name,
                userId=int(linked_user["id"]),
            )

    return int(row["id"])


def _resolve_single_slownik_id(
    conn: Any,
    kod_typu: str,
    raw_id: int | None,
    raw_name: str | None,
    id_field_name: str,
) -> int | None:
    if raw_id is not None:
        return _validate_slownik_item_id(conn, kod_typu, int(raw_id), id_field_name)
    return _resolve_or_create_slownik_item_id(conn, kod_typu, raw_name)


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


def _validate_recommendation_code(conn: Any, recommendation_kod_zalecenia: str | None) -> str | None:
    value = _norm(recommendation_kod_zalecenia)
    if value is None:
        return None

    row = conn.execute(
        "SELECT kod_zalecenia FROM recommendations WHERE kod_zalecenia = ? LIMIT 1",
        (value,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=400, detail="recommendationKodZalecenia nie istnieje")
    return str(row["kod_zalecenia"])


def _decision_year(data_decyzji_i_instancji: str | None, data_wszczecia_postepowania_i_instancji: str | None) -> str:
    for candidate in (data_decyzji_i_instancji, data_wszczecia_postepowania_i_instancji):
        if not candidate:
            continue
        raw = str(candidate).strip()
        if len(raw) >= 4 and raw[:4].isdigit():
            return raw[:4]
    return str(date.today().year)


def _next_decision_code(conn: Any, year: str) -> str:
    prefix = "DZ"
    pattern = f"{prefix}/{year}/%"
    rows = conn.execute(
        "SELECT kod_decyzji FROM obligating_decisions WHERE kod_decyzji LIKE ?",
        (pattern,),
    ).fetchall()

    max_seq = 0
    for row in rows:
        raw = str(row["kod_decyzji"] or "").strip()
        parts = raw.split("/")
        if len(parts) != 3 or parts[0] != prefix or parts[1] != year:
            continue
        try:
            seq = int(parts[2])
        except ValueError:
            continue
        max_seq = max(max_seq, seq)

    return f"{prefix}/{year}/{max_seq + 1}"


def _sync_persons(
    conn: Any,
    obligating_decision_id: int,
    instance: str,
    slownik_ids: list[int],
    operator_user_id: int,
) -> None:
    table_by_instance = {
        "I": "obligating_decisions_persons_i",
        "II": "obligating_decisions_persons_ii",
    }
    relation_table = table_by_instance[instance]

    existing_rows = conn.execute(
        f"""
        SELECT id, slownik_pozycja_id
        FROM {relation_table}
        WHERE obligating_decision_id = ?
        """,
        (obligating_decision_id,),
    ).fetchall()

    existing_by_value = {int(r["slownik_pozycja_id"]): int(r["id"]) for r in existing_rows}
    target_set = set(slownik_ids)
    existing_set = set(existing_by_value.keys())

    to_delete = sorted(existing_set - target_set)
    to_insert = sorted(target_set - existing_set)

    for slownik_id in to_delete:
        conn.execute(f"DELETE FROM {relation_table} WHERE id = ?", (existing_by_value[slownik_id],))

    for slownik_id in to_insert:
        conn.execute(
            f"""
            INSERT INTO {relation_table} (
                obligating_decision_id,
                slownik_pozycja_id,
                created_by_user_id,
                updated_by_user_id
            ) VALUES (?, ?, ?, ?)
            """,
            (obligating_decision_id, slownik_id, operator_user_id, operator_user_id),
        )


def _base_select_sql() -> str:
    return """
        SELECT
            d.id,
            d.kod_decyzji,
            d.recommendation_kod_zalecenia,
            d.nazwa_podmiotu_id,
            np.nazwa_pozycji AS nazwa_podmiotu,
            np.skrot_pozycji AS nazwa_podmiotu_skrot,
            d.liczba_zalecen,
            d.data_wszczecia_postepowania_i_instancji,
            d.data_decyzji_i_instancji,
            d.data_doreczenia_decyzji_i_instancji,
            d.rozstrzygniecie_i_id,
            ri.nazwa_pozycji AS rozstrzygniecie_i,
            ri.skrot_pozycji AS rozstrzygniecie_i_skrot,
            d.data_wniosku_ponowne_rozpatrzenie,
            d.data_wplywu_wniosku_ponowne_rozpatrzenie,
            d.data_decyzji_ii_instancji,
            d.data_doreczenia_decyzji_ii_instancji,
            d.rozstrzygniecie_ii_id,
            rii.nazwa_pozycji AS rozstrzygniecie_ii,
            rii.skrot_pozycji AS rozstrzygniecie_ii_skrot,
            d.komentarz,
            d.created_by_user_id,
            d.utworzono_o,
            d.zaktualizowano_o,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT CAST(rel.slownik_pozycja_id AS TEXT) AS v
                    FROM obligating_decisions_persons_i rel
                    WHERE rel.obligating_decision_id = d.id
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS osoby_i_ids_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.nazwa_pozycji AS v
                    FROM obligating_decisions_persons_i rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.obligating_decision_id = d.id
                    ORDER BY v ASC
                ) x
            ) AS osoby_i_list_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT CAST(rel.slownik_pozycja_id AS TEXT) AS v
                    FROM obligating_decisions_persons_ii rel
                    WHERE rel.obligating_decision_id = d.id
                    ORDER BY rel.slownik_pozycja_id ASC
                ) x
            ) AS osoby_ii_ids_csv,
            (
                SELECT group_concat(x.v, ';')
                FROM (
                    SELECT sp.nazwa_pozycji AS v
                    FROM obligating_decisions_persons_ii rel
                    JOIN slownik_pozycje sp ON sp.id = rel.slownik_pozycja_id
                    WHERE rel.obligating_decision_id = d.id
                    ORDER BY v ASC
                ) x
            ) AS osoby_ii_list_csv
        FROM obligating_decisions d
        LEFT JOIN slownik_pozycje np ON np.id = d.nazwa_podmiotu_id
        LEFT JOIN slownik_pozycje ri ON ri.id = d.rozstrzygniecie_i_id
        LEFT JOIN slownik_pozycje rii ON rii.id = d.rozstrzygniecie_ii_id
    """


def _row_to_payload(row: dict[str, Any], permissions: dict[str, bool]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "kodDecyzji": row.get("kod_decyzji"),
        "canEdit": bool(permissions.get("canEdit", False)),
        "canEditIInstance": bool(permissions.get("canEditIInstance", False)),
        "canAssignIInstancePeople": bool(permissions.get("canAssignIInstancePeople", False)),
        "canEditIIInstance": bool(permissions.get("canEditIIInstance", False)),
        "canAssignIIInstancePeople": bool(permissions.get("canAssignIIInstancePeople", False)),
        "canEditComment": bool(permissions.get("canEditComment", False)),
        "recommendationKodZalecenia": row.get("recommendation_kod_zalecenia"),
        "nazwaPodmiotuId": int(row["nazwa_podmiotu_id"]) if row.get("nazwa_podmiotu_id") is not None else None,
        "nazwaPodmiotu": row.get("nazwa_podmiotu"),
        "nazwaPodmiotuSkrocona": row.get("nazwa_podmiotu_skrot"),
        "nazwaPodmiotuSkrot": row.get("nazwa_podmiotu_skrot"),
        "liczbaZalecen": int(row["liczba_zalecen"]) if row.get("liczba_zalecen") is not None else None,
        "dataWszczeciaPostepowaniaIInstancji": row.get("data_wszczecia_postepowania_i_instancji"),
        "osobyProwadzaceIInstancjeIds": _parse_int_csv(row.get("osoby_i_ids_csv")),
        "osobyProwadzaceIInstancjeList": _parse_csv(row.get("osoby_i_list_csv")),
        "dataDecyzjiIInstancji": row.get("data_decyzji_i_instancji"),
        "dataDoreczeniaDecyzjiIInstancji": row.get("data_doreczenia_decyzji_i_instancji"),
        "rozstrzygniecieIId": int(row["rozstrzygniecie_i_id"]) if row.get("rozstrzygniecie_i_id") is not None else None,
        "rozstrzygniecieI": row.get("rozstrzygniecie_i"),
        "rozstrzygniecieISkrocona": row.get("rozstrzygniecie_i_skrot"),
        "rozstrzygniecieISkrot": row.get("rozstrzygniecie_i_skrot"),
        "dataWnioskuPonowneRozpatrzenie": row.get("data_wniosku_ponowne_rozpatrzenie"),
        "dataWplywuWnioskuPonowneRozpatrzenie": row.get("data_wplywu_wniosku_ponowne_rozpatrzenie"),
        "osobyProwadzaceIIInstancjeIds": _parse_int_csv(row.get("osoby_ii_ids_csv")),
        "osobyProwadzaceIIInstancjeList": _parse_csv(row.get("osoby_ii_list_csv")),
        "dataDecyzjiIIInstancji": row.get("data_decyzji_ii_instancji"),
        "dataDoreczeniaDecyzjiIIInstancji": row.get("data_doreczenia_decyzji_ii_instancji"),
        "rozstrzygniecieIIId": int(row["rozstrzygniecie_ii_id"]) if row.get("rozstrzygniecie_ii_id") is not None else None,
        "rozstrzygniecieII": row.get("rozstrzygniecie_ii"),
        "rozstrzygniecieIISkrocona": row.get("rozstrzygniecie_ii_skrot"),
        "rozstrzygniecieIISkrot": row.get("rozstrzygniecie_ii_skrot"),
        "komentarz": row.get("komentarz"),
        "utworzonoO": row.get("utworzono_o"),
        "zaktualizowanoO": row.get("zaktualizowano_o"),
    }


@router.get("/api/decyzje-zobowiazujace", response_model=ObligatingDecisionListResponse)
@router.get("/api/obligating-decisions", response_model=ObligatingDecisionListResponse)
def list_obligating_decisions(
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_OBLIGATING_DECISIONS_READ)
        rows = conn.execute(_base_select_sql() + " ORDER BY d.id DESC").fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            permissions = _decision_permissions(
                conn,
                operator,
                int(row_dict["id"]),
                row_dict.get("recommendation_kod_zalecenia"),
            )
            items.append(
                _row_to_payload(
                    row_dict,
                    permissions=permissions,
                )
            )

    return {"items": items, "total": len(items)}


@router.get(
    "/api/obligating-decisions/available-recommendations",
    response_model=list[DecisionRecommendationOption],
)
def list_available_recommendations_for_decisions(
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_OBLIGATING_DECISIONS_READ)
        rows = conn.execute(
            """
            SELECT
                r.id,
                r.kod_zalecenia,
                r.pozycja,
                r.inspection_id,
                r.created_by_user_id,
                i.kod_inspekcji,
                COALESCE(rsp.nazwa_pozycji, isp.nazwa_pozycji, 'brak') AS nazwa_podmiotu,
                COALESCE(NULLIF(trim(rsp.skrot_pozycji), ''), NULLIF(trim(isp.skrot_pozycji), '')) AS nazwa_podmiotu_skrot
            FROM recommendations r
            LEFT JOIN inspections i ON i.id = r.inspection_id
            LEFT JOIN slownik_pozycje rsp ON rsp.id = r.nazwa_podmiotu_id
            LEFT JOIN slownik_pozycje isp ON isp.id = i.nazwa_podmiotu_id
            WHERE r.kod_zalecenia IS NOT NULL AND trim(r.kod_zalecenia) <> ''
            ORDER BY r.id DESC
            """
        ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            inspection_id = int(row["inspection_id"]) if row["inspection_id"] is not None else None
            recommendation_created_by = (
                int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
            )
            can_create = _can_create_decision_by_recommendation_scope(
                conn,
                inspection_id,
                recommendation_created_by,
                operator,
            )
            if not can_create:
                continue

            blocked, _, _ = _recommendation_status_is_blocked(
                conn,
                row["kod_zalecenia"],
                operator,
            )
            if blocked:
                continue

            items.append(
                {
                    "id": int(row["id"]),
                    "kodZalecenia": str(row["kod_zalecenia"]),
                    "pozycja": int(row["pozycja"]),
                    "inspectionId": inspection_id,
                    "inspectionKod": row["kod_inspekcji"],
                    "nazwaPodmiotu": row["nazwa_podmiotu"],
                    "nazwaPodmiotuSkrocona": row["nazwa_podmiotu_skrot"] or row["nazwa_podmiotu"],
                    "nazwaPodmiotuSkrot": row["nazwa_podmiotu_skrot"] or row["nazwa_podmiotu"],
                    "canCreateDecisionForRecommendation": bool(can_create),
                    "createDeniedReasonCode": None,
                }
            )

    return items


@router.get(
    "/api/decyzje-zobowiazujace/dostepne-osoby-i-instancja",
    response_model=DecisionPeopleOptionsResponse,
)
@router.get(
    "/api/obligating-decisions/available-first-instance-people",
    response_model=DecisionPeopleOptionsResponse,
)
def list_available_first_instance_people_for_decisions(
    recommendationKodZalecenia: str | None = Query(default=None),
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, list[dict[str, Any]]]:
    """Return person options for I instance.

    Response contract:
    - `displayName` is the preferred UI label.
    - `label` is a backward-compatible alias.
    - `aktywny` is not a filter; hidden users are excluded by `listVisibility=hidden`.
    """
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_OBLIGATING_DECISIONS_READ)
        force_full_scope = _is_author_of_unlinked_recommendation(conn, operator, recommendationKodZalecenia)
        if force_full_scope:
            items = _build_available_decision_people(
                conn,
                operator,
                endpoint_name="first_instance_author_unlinked_recommendation",
                force_full_scope=True,
            )
        else:
            items = _available_first_instance_people(conn, operator)
        return {"items": items}


@router.get(
    "/api/decyzje-zobowiazujace/dostepne-osoby-ii-instancja",
    response_model=DecisionPeopleOptionsResponse,
)
@router.get(
    "/api/obligating-decisions/available-second-instance-people",
    response_model=DecisionPeopleOptionsResponse,
)
def list_available_second_instance_people_for_decisions(
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, list[dict[str, Any]]]:
    """Return person options for II instance.

    Uses the same filtering and RBAC logic as I instance.
    """
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_OBLIGATING_DECISIONS_READ)
        return {"items": _available_second_instance_people(conn, operator)}


@router.get("/api/decyzje-zobowiazujace/{decision_id}", response_model=ObligatingDecisionRead)
@router.get("/api/obligating-decisions/{decision_id}", response_model=ObligatingDecisionRead)
def get_obligating_decision(
    decision_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_permission(conn, operator, PERMISSION_OBLIGATING_DECISIONS_READ)
        row = conn.execute(_base_select_sql() + " WHERE d.id = ? LIMIT 1", (decision_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Obligating decision not found")

        row_dict = dict(row)
        permissions = _decision_permissions(
            conn,
            operator,
            int(row_dict["id"]),
            row_dict.get("recommendation_kod_zalecenia"),
        )

    return _row_to_payload(row_dict, permissions=permissions)


@router.post("/api/decyzje-zobowiazujace", response_model=ObligatingDecisionRead, status_code=201)
@router.post("/api/obligating-decisions", response_model=ObligatingDecisionRead, status_code=201)
def create_obligating_decision(
    payload: ObligatingDecisionCreate,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    """Create obligating decision.

    Domain error codes (detail.code):
    - `VALIDATION_RECOMMENDATION_REQUIRED` (400)
    - `PERMISSION_DENIED_CREATE_FOR_RECOMMENDATION` (403)
    - `PERMISSION_DENIED_II_INSTANCE` (403)
    - `LEADER_OUT_OF_SCOPE` (403)
    - `USER_HIDDEN_NOT_ALLOWED` (422)
    """
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)

        payload_fields = payload.model_dump(exclude_unset=True)
        _ensure_second_instance_write_allowed(operator, payload_fields)

        touches_ii_on_create = any(field in payload_fields for field in _SECOND_INSTANCE_FIELDS)
        if touches_ii_on_create and not _norm(payload.dataWplywuWnioskuPonowneRozpatrzenie):
            _raise_domain_error(
                403,
                "PERMISSION_DENIED_II_INSTANCE",
                "II instancja jest dostepna dopiero po uzupelnieniu daty wplywu wniosku",
            )

        recommendation_kod = _validate_recommendation_code(conn, payload.recommendationKodZalecenia)
        if recommendation_kod is None:
            _raise_domain_error(
                400,
                "VALIDATION_RECOMMENDATION_REQUIRED",
                "recommendationKodZalecenia jest wymagane",
            )
        _ensure_recommendation_status_link_allowed(conn, operator, recommendation_kod)
        if not _can_create_decision_by_recommendation_code(conn, operator, recommendation_kod):
            _raise_domain_error(
                403,
                "PERMISSION_DENIED_CREATE_FOR_RECOMMENDATION",
                "Brak uprawnien do powiazanego zalecenia",
            )
        recommendation_scope = _recommendation_scope_by_code(conn, recommendation_kod)
        has_inspection_link = recommendation_scope is not None and recommendation_scope[0] is not None
        nazwa_podmiotu_id = _resolve_single_slownik_id(
            conn,
            "nazwy_podmiotow",
            payload.nazwaPodmiotuId,
            payload.nazwaPodmiotu,
            "nazwaPodmiotuId",
        )
        osoby_i_ids = _resolve_multi_slownik_ids(
            conn,
            "osoby",
            payload.osobyProwadzaceIInstancjeIds,
            payload.osobyProwadzaceIInstancjeList,
            "osobyProwadzaceIInstancjeIds",
            "osobyProwadzaceIInstancjeList",
        )
        _ensure_first_instance_person_scope_on_create_for_recommendation(
            conn,
            operator,
            recommendation_kod,
            osoby_i_ids,
        )
        osoby_ii_ids = _resolve_multi_slownik_ids(
            conn,
            "osoby",
            payload.osobyProwadzaceIIInstancjeIds,
            payload.osobyProwadzaceIIInstancjeList,
            "osobyProwadzaceIIInstancjeIds",
            "osobyProwadzaceIIInstancjeList",
        )
        _ensure_first_instance_person_scope_on_create(conn, operator, osoby_ii_ids)
        if osoby_ii_ids:
            can_assign_ii_on_create = _is_team_lead_of_slownik_users(conn, operator, osoby_i_ids)
            if _is_team_lead(operator):
                can_assign_ii_on_create = True
            if has_inspection_link and _is_director(operator):
                can_assign_ii_on_create = True

            if not can_assign_ii_on_create:
                message = (
                    "Osoby II instancji moze dodawac tylko kierownik osob z I instancji"
                    if not has_inspection_link
                    else "Osoby II instancji moze dodawac dyrektor lub kierownik osob z I instancji"
                )
                _raise_domain_error(
                    403,
                    "PERMISSION_DENIED_II_INSTANCE",
                    message,
                )
        rozstrzygniecie_i_id = _resolve_single_slownik_id(
            conn,
            "rozstrzygniecie_decyzji_i",
            payload.rozstrzygniecieIId,
            payload.rozstrzygniecieI,
            "rozstrzygniecieIId",
        )
        rozstrzygniecie_ii_id = _resolve_single_slownik_id(
            conn,
            "rozstrzygniecie_decyzji_ii",
            payload.rozstrzygniecieIIId,
            payload.rozstrzygniecieII,
            "rozstrzygniecieIIId",
        )

        data_wszczecia = _validate_optional_iso_date(
            payload.dataWszczeciaPostepowaniaIInstancji,
            "dataWszczeciaPostepowaniaIInstancji",
        )
        data_decyzji_i = _validate_optional_iso_date(payload.dataDecyzjiIInstancji, "dataDecyzjiIInstancji")
        data_doreczenia_i = _validate_optional_iso_date(
            payload.dataDoreczeniaDecyzjiIInstancji,
            "dataDoreczeniaDecyzjiIInstancji",
        )
        data_wniosku = _validate_optional_iso_date(
            payload.dataWnioskuPonowneRozpatrzenie,
            "dataWnioskuPonowneRozpatrzenie",
        )
        data_wplywu = _validate_optional_iso_date(
            payload.dataWplywuWnioskuPonowneRozpatrzenie,
            "dataWplywuWnioskuPonowneRozpatrzenie",
        )
        data_decyzji_ii = _validate_optional_iso_date(payload.dataDecyzjiIIInstancji, "dataDecyzjiIIInstancji")
        data_doreczenia_ii = _validate_optional_iso_date(
            payload.dataDoreczeniaDecyzjiIIInstancji,
            "dataDoreczeniaDecyzjiIIInstancji",
        )

        kod_decyzji = _next_decision_code(conn, _decision_year(data_decyzji_i, data_wszczecia))

        cursor = conn.execute(
            """
            INSERT INTO obligating_decisions (
                kod_decyzji,
                recommendation_kod_zalecenia,
                nazwa_podmiotu_id,
                liczba_zalecen,
                data_wszczecia_postepowania_i_instancji,
                data_decyzji_i_instancji,
                data_doreczenia_decyzji_i_instancji,
                rozstrzygniecie_i_id,
                data_wniosku_ponowne_rozpatrzenie,
                data_wplywu_wniosku_ponowne_rozpatrzenie,
                data_decyzji_ii_instancji,
                data_doreczenia_decyzji_ii_instancji,
                rozstrzygniecie_ii_id,
                komentarz,
                created_by_user_id,
                updated_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kod_decyzji,
                recommendation_kod,
                nazwa_podmiotu_id,
                payload.liczbaZalecen,
                data_wszczecia,
                data_decyzji_i,
                data_doreczenia_i,
                rozstrzygniecie_i_id,
                data_wniosku,
                data_wplywu,
                data_decyzji_ii,
                data_doreczenia_ii,
                rozstrzygniecie_ii_id,
                payload.komentarz,
                operator["id"],
                operator["id"],
            ),
        )
        decision_id = int(cursor.lastrowid)

        _sync_persons(conn, decision_id, "I", osoby_i_ids, operator["id"])
        _sync_persons(conn, decision_id, "II", osoby_ii_ids, operator["id"])
        kod_row = conn.execute(
            "SELECT kod_decyzji FROM obligating_decisions WHERE id = ? LIMIT 1",
            (decision_id,),
        ).fetchone()
        rekord_kod_cr = str(kod_row["kod_decyzji"]) if kod_row and kod_row["kod_decyzji"] else str(decision_id)
        row = conn.execute(_base_select_sql() + " WHERE d.id = ? LIMIT 1", (decision_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="Failed to fetch created obligating decision")

        created_permissions = _decision_permissions(conn, operator, decision_id, recommendation_kod)
        created_payload = _row_to_payload(dict(row), permissions=created_permissions)
        changes = build_create_changes(
            [
                ("Kod decyzji", created_payload.get("kodDecyzji")),
                ("Powiązane zalecenie", created_payload.get("recommendationKodZalecenia")),
                ("Nazwa podmiotu", created_payload.get("nazwaPodmiotu")),
                ("Liczba zaleceń", created_payload.get("liczbaZalecen")),
                ("Data wszczęcia postęp. I inst.", created_payload.get("dataWszczeciaPostepowaniaIInstancji")),
                ("Osoby I instancji", created_payload.get("osobyProwadzaceIInstancjeList")),
                ("Data decyzji I inst.", created_payload.get("dataDecyzjiIInstancji")),
                ("Data doręczenia decyzji I inst.", created_payload.get("dataDoreczeniaDecyzjiIInstancji")),
                ("Rozstrzygnięcie I instancji", created_payload.get("rozstrzygniecieI")),
                ("Data wniosku ponowne rozpatrzenie", created_payload.get("dataWnioskuPonowneRozpatrzenie")),
                ("Data wpływu wniosku", created_payload.get("dataWplywuWnioskuPonowneRozpatrzenie")),
                ("Osoby II instancji", created_payload.get("osobyProwadzaceIIInstancjeList")),
                ("Data decyzji II inst.", created_payload.get("dataDecyzjiIIInstancji")),
                ("Data doręczenia decyzji II inst.", created_payload.get("dataDoreczeniaDecyzjiIIInstancji")),
                ("Rozstrzygnięcie II instancji", created_payload.get("rozstrzygniecieII")),
                ("Komentarz", created_payload.get("komentarz")),
            ]
        )

        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_CREATE,
                        REJESTR_DECYZJE, rekord_kod_cr, changes)
        conn.commit()

    if row is None:
        raise HTTPException(status_code=500, detail="Failed to fetch created obligating decision")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        row_dict = dict(row)
        permissions = _decision_permissions(conn, operator, int(row_dict["id"]), row_dict.get("recommendation_kod_zalecenia"))
    return _row_to_payload(row_dict, permissions=permissions)


@router.put("/api/decyzje-zobowiazujace/{decision_id}", response_model=ObligatingDecisionRead)
@router.put("/api/obligating-decisions/{decision_id}", response_model=ObligatingDecisionRead)
def update_obligating_decision(
    decision_id: int,
    payload: ObligatingDecisionUpdate,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, Any]:
    """Update obligating decision.

    Domain error codes (detail.code):
    - `PERMISSION_DENIED_I_INSTANCE` (403)
    - `PERMISSION_DENIED_II_INSTANCE` (403)
    - `PERMISSION_DENIED_CREATE_FOR_RECOMMENDATION` (403)
    - `LEADER_OUT_OF_SCOPE` (403)
    - `USER_HIDDEN_NOT_ALLOWED` (422)
    """
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
            """SELECT id, created_by_user_id, kod_decyzji, recommendation_kod_zalecenia,
                      nazwa_podmiotu_id, liczba_zalecen, rozstrzygniecie_i_id, rozstrzygniecie_ii_id,
                      data_wszczecia_postepowania_i_instancji, data_decyzji_i_instancji,
                      data_doreczenia_decyzji_i_instancji, data_wniosku_ponowne_rozpatrzenie,
                      data_wplywu_wniosku_ponowne_rozpatrzenie, data_decyzji_ii_instancji,
                      data_doreczenia_decyzji_ii_instancji, komentarz, zaktualizowano_o
               FROM obligating_decisions WHERE id = ? LIMIT 1""",
            (decision_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail="Obligating decision not found")

        current_recommendation_kod = current["recommendation_kod_zalecenia"]
        _ensure_update_instance_permissions(conn, operator, decision_id, current_recommendation_kod, fields)

        assert_lock_for_save(conn, "obligating-decisions", decision_id, operator, lock_token)
        assert_expected_updated_at(expected_updated_at, str(current["zaktualizowano_o"] or ""))

        # Pobierz osoby przed zmianą (dla audit log)
        _pi_rows = conn.execute(
            "SELECT sp.nazwa_pozycji FROM obligating_decisions_persons_i pi JOIN slownik_pozycje sp ON sp.id = pi.slownik_pozycja_id WHERE pi.obligating_decision_id = ? ORDER BY sp.nazwa_pozycji",
            (decision_id,),
        ).fetchall()
        _pii_rows = conn.execute(
            "SELECT sp.nazwa_pozycji FROM obligating_decisions_persons_ii pii JOIN slownik_pozycje sp ON sp.id = pii.slownik_pozycja_id WHERE pii.obligating_decision_id = ? ORDER BY sp.nazwa_pozycji",
            (decision_id,),
        ).fetchall()
        persons_i_before = ", ".join(r["nazwa_pozycji"] for r in _pi_rows) or None
        persons_ii_before = ", ".join(r["nazwa_pozycji"] for r in _pii_rows) or None

        set_parts: list[str] = []
        values: list[Any] = []
        row_touched = False
        first_instance_person_ids: list[int] | None = None

        if "recommendationKodZalecenia" in fields:
            target_recommendation_kod = _validate_recommendation_code(conn, fields.get("recommendationKodZalecenia"))
            if target_recommendation_kod is not None:
                _ensure_recommendation_status_link_allowed(conn, operator, target_recommendation_kod)
            if target_recommendation_kod is not None and not _can_create_decision_by_recommendation_code(
                conn,
                operator,
                target_recommendation_kod,
            ):
                _raise_domain_error(
                    403,
                    "PERMISSION_DENIED_CREATE_FOR_RECOMMENDATION",
                    "Brak uprawnien do powiazanego zalecenia",
                )
            set_parts.append("recommendation_kod_zalecenia = ?")
            values.append(target_recommendation_kod)

        if "nazwaPodmiotuId" in fields or "nazwaPodmiotu" in fields:
            set_parts.append("nazwa_podmiotu_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "nazwy_podmiotow",
                    fields.get("nazwaPodmiotuId"),
                    fields.get("nazwaPodmiotu"),
                    "nazwaPodmiotuId",
                )
            )

        if "liczbaZalecen" in fields:
            set_parts.append("liczba_zalecen = ?")
            values.append(fields.get("liczbaZalecen"))

        if "dataWszczeciaPostepowaniaIInstancji" in fields:
            set_parts.append("data_wszczecia_postepowania_i_instancji = ?")
            values.append(
                _validate_optional_iso_date(
                    fields.get("dataWszczeciaPostepowaniaIInstancji"),
                    "dataWszczeciaPostepowaniaIInstancji",
                )
            )
        if "dataDecyzjiIInstancji" in fields:
            set_parts.append("data_decyzji_i_instancji = ?")
            values.append(_validate_optional_iso_date(fields.get("dataDecyzjiIInstancji"), "dataDecyzjiIInstancji"))
        if "dataDoreczeniaDecyzjiIInstancji" in fields:
            set_parts.append("data_doreczenia_decyzji_i_instancji = ?")
            values.append(
                _validate_optional_iso_date(
                    fields.get("dataDoreczeniaDecyzjiIInstancji"),
                    "dataDoreczeniaDecyzjiIInstancji",
                )
            )
        if "rozstrzygniecieIId" in fields or "rozstrzygniecieI" in fields:
            set_parts.append("rozstrzygniecie_i_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "rozstrzygniecie_decyzji_i",
                    fields.get("rozstrzygniecieIId"),
                    fields.get("rozstrzygniecieI"),
                    "rozstrzygniecieIId",
                )
            )

        if "dataWnioskuPonowneRozpatrzenie" in fields:
            set_parts.append("data_wniosku_ponowne_rozpatrzenie = ?")
            values.append(
                _validate_optional_iso_date(
                    fields.get("dataWnioskuPonowneRozpatrzenie"),
                    "dataWnioskuPonowneRozpatrzenie",
                )
            )
        if "dataWplywuWnioskuPonowneRozpatrzenie" in fields:
            set_parts.append("data_wplywu_wniosku_ponowne_rozpatrzenie = ?")
            values.append(
                _validate_optional_iso_date(
                    fields.get("dataWplywuWnioskuPonowneRozpatrzenie"),
                    "dataWplywuWnioskuPonowneRozpatrzenie",
                )
            )

        if "dataDecyzjiIIInstancji" in fields:
            set_parts.append("data_decyzji_ii_instancji = ?")
            values.append(_validate_optional_iso_date(fields.get("dataDecyzjiIIInstancji"), "dataDecyzjiIIInstancji"))
        if "dataDoreczeniaDecyzjiIIInstancji" in fields:
            set_parts.append("data_doreczenia_decyzji_ii_instancji = ?")
            values.append(
                _validate_optional_iso_date(
                    fields.get("dataDoreczeniaDecyzjiIIInstancji"),
                    "dataDoreczeniaDecyzjiIIInstancji",
                )
            )
        if "rozstrzygniecieIIId" in fields or "rozstrzygniecieII" in fields:
            set_parts.append("rozstrzygniecie_ii_id = ?")
            values.append(
                _resolve_single_slownik_id(
                    conn,
                    "rozstrzygniecie_decyzji_ii",
                    fields.get("rozstrzygniecieIIId"),
                    fields.get("rozstrzygniecieII"),
                    "rozstrzygniecieIIId",
                )
            )

        if "komentarz" in fields:
            set_parts.append("komentarz = ?")
            values.append(fields.get("komentarz"))

        if set_parts:
            set_parts.append("updated_by_user_id = ?")
            values.append(operator["id"])
            set_parts.append("zaktualizowano_o = ?")
            values.append(now_rfc3339_utc_ms())
            values.append(decision_id)
            conn.execute(
                f"UPDATE obligating_decisions SET {', '.join(set_parts)} WHERE id = ?",
                tuple(values),
            )
            row_touched = True

        if "osobyProwadzaceIInstancjeIds" in fields or "osobyProwadzaceIInstancjeList" in fields:
            first_instance_person_ids = _resolve_multi_slownik_ids(
                conn,
                "osoby",
                fields.get("osobyProwadzaceIInstancjeIds"),
                fields.get("osobyProwadzaceIInstancjeList"),
                "osobyProwadzaceIInstancjeIds",
                "osobyProwadzaceIInstancjeList",
            )
            _ensure_first_instance_person_scope(conn, operator, first_instance_person_ids)
            _sync_persons(
                conn,
                decision_id,
                "I",
                first_instance_person_ids,
                operator["id"],
            )

        if "osobyProwadzaceIIInstancjeIds" in fields or "osobyProwadzaceIIInstancjeList" in fields:
            second_instance_person_ids = _resolve_multi_slownik_ids(
                conn,
                "osoby",
                fields.get("osobyProwadzaceIIInstancjeIds"),
                fields.get("osobyProwadzaceIIInstancjeList"),
                "osobyProwadzaceIIInstancjeIds",
                "osobyProwadzaceIIInstancjeList",
            )
            _sync_persons(
                conn,
                decision_id,
                "II",
                second_instance_person_ids,
                operator["id"],
            )

        if not row_touched and (
            "osobyProwadzaceIInstancjeIds" in fields
            or "osobyProwadzaceIInstancjeList" in fields
            or "osobyProwadzaceIIInstancjeIds" in fields
            or "osobyProwadzaceIIInstancjeList" in fields
        ):
            conn.execute(
                "UPDATE obligating_decisions SET updated_by_user_id = ?, zaktualizowano_o = ? WHERE id = ?",
                (operator["id"], now_rfc3339_utc_ms(), decision_id),
            )

        # Pobierz osoby po zmianie (dla audit log)
        _pi_after = conn.execute(
            "SELECT sp.nazwa_pozycji FROM obligating_decisions_persons_i pi JOIN slownik_pozycje sp ON sp.id = pi.slownik_pozycja_id WHERE pi.obligating_decision_id = ? ORDER BY sp.nazwa_pozycji",
            (decision_id,),
        ).fetchall()
        _pii_after = conn.execute(
            "SELECT sp.nazwa_pozycji FROM obligating_decisions_persons_ii pii JOIN slownik_pozycje sp ON sp.id = pii.slownik_pozycja_id WHERE pii.obligating_decision_id = ? ORDER BY sp.nazwa_pozycji",
            (decision_id,),
        ).fetchall()
        persons_i_after = ", ".join(r["nazwa_pozycji"] for r in _pi_after) or None
        persons_ii_after = ", ".join(r["nazwa_pozycji"] for r in _pii_after) or None

        # --- Audit log ---
        current_dict_d = dict(current)
        rekord_kod_d = str(current_dict_d.get("kod_decyzji") or decision_id)
        changes_d = build_decision_changes(
            conn, current_dict_d, fields,
            persons_i_before, persons_ii_before,
            persons_i_after, persons_ii_after,
        )
        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_UPDATE,
                        REJESTR_DECYZJE, rekord_kod_d, changes_d)
        # --- koniec audit log ---

        conn.commit()

        row = conn.execute(_base_select_sql() + " WHERE d.id = ? LIMIT 1", (decision_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Failed to fetch updated obligating decision")

    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        row_dict = dict(row)
        permissions = _decision_permissions(conn, operator, int(row_dict["id"]), row_dict.get("recommendation_kod_zalecenia"))
    return _row_to_payload(row_dict, permissions=permissions)


@router.delete("/api/decyzje-zobowiazujace/{decision_id}")
@router.delete("/api/obligating-decisions/{decision_id}")
def delete_obligating_decision(
    decision_id: int,
    x_operator_login: str | None = Header(default=None, alias="X-Operator-Login"),
) -> dict[str, bool]:
    with get_connection() as conn:
        operator = _resolve_operator(conn, x_operator_login)
        require_write_access(conn, operator)
        _ensure_director(operator)
        current = conn.execute(
            "SELECT id, created_by_user_id, kod_decyzji FROM obligating_decisions WHERE id = ? LIMIT 1",
            (decision_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail="Obligating decision not found")

        conn.execute("DELETE FROM obligating_decisions WHERE id = ?", (decision_id,))
        write_audit_log(conn, new_session_id(), operator["login"], AKCJA_DELETE,
                        REJESTR_DECYZJE, str(current["kod_decyzji"] or decision_id), [])
        conn.commit()

    return {"ok": True}
