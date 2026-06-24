from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Response, status
from pydantic import BaseModel

from app.database import get_connection
from app.permissions import PERMISSION_MANAGEMENT_DICTIONARIES_WRITE, require_permission

router = APIRouter()

_ALLOWED_INSPECTION_TYPE_NAMES = {
    "kontrola",
    "wizyta nadzorcza",
}

_DECISION_RESOLUTION_KOD_TYPY = {
    "rozstrzygniecie_decyzji_i",
    "rozstrzygniecie_decyzji_ii",
}

_STATUS_STYLE_ALLOWED_KOLORY = {
    "emerald",
    "green",
    "teal",
    "lime",
    "sky",
    "cyan",
    "blue",
    "indigo",
    "rose",
    "red",
    "pink",
    "fuchsia",
    "yellow",
    "amber",
    "orange",
}

_STATUS_STYLE_ALLOWED_ODCIEN = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950}
_STATUS_STYLE_ALLOWED_KOD_TYPY = {
    "statusy_inspekcji",
    "statusy_zalecen",
}


class SlownikTypRead(BaseModel):
    id: int
    kodTypu: str
    nazwaTypu: str
    kategoria: int
    kategoriaNazwa: str
    opis: str | None = None
    aktywny: bool


class SlownikPozycjaRead(BaseModel):
    id: int
    typId: int
    kodTypu: str
    kodPozycji: str
    nazwaPozycji: str
    nazwaUzytkowa: str | None = None
    skrotPozycji: str | None = None
    nazwaPozycjiSkrocona: str | None = None
    nazwaPozycjiSkrot: str | None = None
    kolor: str | None = None
    odcien: int | None = None
    intensywnosc: int | None = None
    kolejnosc: int
    aktywny: bool


class SlownikPozycjaCreate(BaseModel):
    kodTypu: str
    kodPozycji: str | None = None
    nazwaPozycji: str
    nazwaUzytkowa: str | None = None
    skrotPozycji: str | None = None
    kolejnosc: int | None = None
    aktywny: bool = True


class SlownikPozycjaUpdate(BaseModel):
    kodTypu: str
    kodPozycji: str | None = None
    nazwaPozycji: str
    nazwaUzytkowa: str | None = None
    skrotPozycji: str | None = None
    aktywny: bool


class SlownikPozycjaUpdateById(BaseModel):
    nazwaPozycji: str
    nazwaUzytkowa: str | None = None
    skrotPozycji: str | None = None
    aktywny: bool


class StatusInspekcjiStylUpsert(BaseModel):
    slownikPozycjaId: int
    kolor: str | None = None
    odcien: int | None = None
    intensywnosc: int | None = None


class StatusInspekcjiStylRead(BaseModel):
    id: int
    slownikPozycjaId: int
    kolor: str
    odcien: int
    intensywnosc: int


def _get_operator(conn: Any, operator_login: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, login, rola_id, aktywny
        FROM users
        WHERE lower(login) = lower(?)
        LIMIT 1
        """,
        (operator_login.strip(),),
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail="Operator nie istnieje")

    data = dict(row)
    if int(data["aktywny"]) != 1:
        raise HTTPException(status_code=403, detail="Operator jest nieaktywny")

    if int(data["rola_id"]) not in (2, 3):
        raise HTTPException(status_code=403, detail="Brak uprawnien do zarzadzania slownikami")

    require_permission(conn, data, PERMISSION_MANAGEMENT_DICTIONARIES_WRITE)

    return data


def _parse_env_csv(name: str) -> list[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for item in raw.split(","):
        cleaned = item.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def _manager_blocked_types() -> set[str]:
    return {str(item).strip().lower() for item in _parse_env_csv("MANAGER_SLOWNIKI_BLOCKED_TYPES")}


def _manager_allowed_types() -> set[str]:
    return {str(item).strip().lower() for item in _parse_env_csv("MANAGER_SLOWNIKI_ALLOWED_TYPES")}


def _manager_blocked_items() -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for item in _parse_env_csv("MANAGER_SLOWNIKI_BLOCKED_ITEMS"):
        # Format: kod_typu:kod_pozycji
        parts = item.split(":", 1)
        if len(parts) != 2:
            continue
        kod_typu = parts[0].strip().lower()
        kod_pozycji = parts[1].strip().upper()
        if not kod_typu or not kod_pozycji:
            continue
        result.add((kod_typu, kod_pozycji))
    return result


def _ensure_manager_can_modify_entry(operator: dict[str, Any], kod_typu: str, kod_pozycji: str | None = None) -> None:
    # Restriction applies only to team leads (role_id=2).
    if int(operator.get("rola_id") or 0) != 2:
        return

    normalized_type = str(kod_typu or "").strip().lower()
    normalized_code = str(kod_pozycji or "").strip().upper()

    allowed_types = _manager_allowed_types()
    if allowed_types and normalized_type not in allowed_types:
        raise HTTPException(status_code=403, detail="Ta pozycja slownika jest zablokowana do edycji dla kierownika")

    if normalized_type in _manager_blocked_types():
        raise HTTPException(status_code=403, detail="Ta pozycja slownika jest zablokowana do edycji dla kierownika")

    if normalized_code and (normalized_type, normalized_code) in _manager_blocked_items():
        raise HTTPException(status_code=403, detail="Ta pozycja slownika jest zablokowana do edycji dla kierownika")


def _slownik_aux_column(conn: Any) -> str:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(slownik_pozycje)").fetchall()}
    if "nazwa_uzytkowa" in columns:
        return "nazwa_uzytkowa"
    if "pomocnicza" in columns:
        return "pomocnicza"
    # Fallback to legacy contract name.
    return "nazwa_uzytkowa"


def _map_slownik_pozycja_row(row: dict[str, Any]) -> dict[str, Any]:
    skrot = row["skrot_pozycji"]
    if isinstance(skrot, str):
        skrot = skrot.strip() or None
    if skrot is None:
        raw_code = row.get("kod_pozycji")
        if isinstance(raw_code, str):
            skrot = raw_code.strip() or None

    nazwa_uzytkowa = row.get("nazwa_uzytkowa")
    if nazwa_uzytkowa is None:
        nazwa_uzytkowa = row.get("pomocnicza")
    if isinstance(nazwa_uzytkowa, str):
        nazwa_uzytkowa = nazwa_uzytkowa.strip() or None

    kod_typu = str(row["kod_typu"] or "").strip().lower()
    aktywny = bool(row["aktywny"])
    if skrot is None and aktywny and kod_typu in _DECISION_RESOLUTION_KOD_TYPY:
        skrot = str(row["nazwa_pozycji"] or "").strip() or None
    if kod_typu != "zakresy_inspekcji":
        nazwa_uzytkowa = None

    kolor = row.get("kolor")
    if isinstance(kolor, str):
        kolor = kolor.strip().lower() or None
    odcien = row.get("odcien")
    intensywnosc = row.get("intensywnosc")
    if kod_typu not in _STATUS_STYLE_ALLOWED_KOD_TYPY:
        kolor = None
        odcien = None
        intensywnosc = None
    else:
        odcien = int(odcien) if odcien is not None else None
        intensywnosc = int(intensywnosc) if intensywnosc is not None else None

    return {
        "id": row["id"],
        "typId": row["typ_id"],
        "kodTypu": row["kod_typu"],
        "kodPozycji": row["kod_pozycji"],
        "nazwaPozycji": row["nazwa_pozycji"],
        "nazwaUzytkowa": nazwa_uzytkowa,
        "skrotPozycji": skrot,
        "nazwaPozycjiSkrocona": skrot,
        "nazwaPozycjiSkrot": skrot,
        "kolor": kolor,
        "odcien": odcien,
        "intensywnosc": intensywnosc,
        "kolejnosc": row["kolejnosc"],
        "aktywny": bool(row["aktywny"]),
    }


def _validate_status_style_payload(kolor: str, odcien: int, intensywnosc: int) -> tuple[str, int, int]:
    normalized_kolor = str(kolor or "").strip().lower()
    if normalized_kolor not in _STATUS_STYLE_ALLOWED_KOLORY:
        raise HTTPException(status_code=400, detail="Nieprawidlowy kolor")
    if int(odcien) not in _STATUS_STYLE_ALLOWED_ODCIEN:
        raise HTTPException(status_code=400, detail="Nieprawidlowy odcien")
    if int(intensywnosc) < 0 or int(intensywnosc) > 100:
        raise HTTPException(status_code=400, detail="Nieprawidlowa intensywnosc")
    return normalized_kolor, int(odcien), int(intensywnosc)


def _resolve_status_style_target(conn: Any, slownik_pozycja_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT kod_typu
        FROM slownik_pozycje
        WHERE id = ?
        LIMIT 1
        """,
        (int(slownik_pozycja_id),),
    ).fetchone()
    if row is None:
        return None
    return str(row["kod_typu"] or "").strip().lower()


def _upsert_status_style(
    payload: StatusInspekcjiStylUpsert,
    x_operator_login: str,
    expected_kod_typu: str,
) -> dict[str, Any] | Response:
    clear_requested = payload.kolor is None and payload.odcien is None and payload.intensywnosc is None
    if not clear_requested and (payload.kolor is None or payload.odcien is None or payload.intensywnosc is None):
        raise HTTPException(status_code=400, detail="Dla czyszczenia ustaw kolor, odcien i intensywnosc na null")

    kolor: str
    odcien: int
    intensywnosc: int
    if not clear_requested:
        kolor, odcien, intensywnosc = _validate_status_style_payload(
            payload.kolor,
            payload.odcien,
            payload.intensywnosc,
        )

    with get_connection() as conn:
        operator = _get_operator(conn, x_operator_login)
        actual_kod_typu = _resolve_status_style_target(conn, int(payload.slownikPozycjaId))
        if actual_kod_typu is None:
            raise HTTPException(status_code=404, detail="Pozycja slownika nie istnieje")
        if actual_kod_typu != expected_kod_typu:
            raise HTTPException(status_code=400, detail=f"Pozycja nie nalezy do {expected_kod_typu}")

        if clear_requested:
            conn.execute(
                """
                DELETE FROM slownik_status_inspekcji_styl
                WHERE slownik_pozycja_id = ?
                """,
                (int(payload.slownikPozycjaId),),
            )
            conn.commit()
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        conn.execute(
            """
            INSERT INTO slownik_status_inspekcji_styl
                (slownik_pozycja_id, kolor, odcien, intensywnosc, utworzono_przez, zaktualizowano_przez)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(slownik_pozycja_id)
            DO UPDATE SET
                kolor = excluded.kolor,
                odcien = excluded.odcien,
                intensywnosc = excluded.intensywnosc,
                zaktualizowano_o = CURRENT_TIMESTAMP,
                zaktualizowano_przez = excluded.zaktualizowano_przez
            """,
            (
                int(payload.slownikPozycjaId),
                kolor,
                odcien,
                intensywnosc,
                str(operator["login"]),
                str(operator["login"]),
            ),
        )
        conn.commit()

        style_row = conn.execute(
            """
            SELECT id, slownik_pozycja_id, kolor, odcien, intensywnosc
            FROM slownik_status_inspekcji_styl
            WHERE slownik_pozycja_id = ?
            LIMIT 1
            """,
            (int(payload.slownikPozycjaId),),
        ).fetchone()

    if style_row is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie zapisac stylu statusu")

    return {
        "id": int(style_row["id"]),
        "slownikPozycjaId": int(style_row["slownik_pozycja_id"]),
        "kolor": str(style_row["kolor"]),
        "odcien": int(style_row["odcien"]),
        "intensywnosc": int(style_row["intensywnosc"]),
    }


def _delete_status_style(
    slownik_pozycja_id: int,
    x_operator_login: str,
    expected_kod_typu: str,
    *,
    idempotent_not_found: bool = False,
) -> Response:
    with get_connection() as conn:
        _get_operator(conn, x_operator_login)

        actual_kod_typu = _resolve_status_style_target(conn, int(slownik_pozycja_id))
        if actual_kod_typu is None:
            if idempotent_not_found:
                return Response(status_code=status.HTTP_204_NO_CONTENT)
            raise HTTPException(status_code=404, detail="Pozycja slownika nie istnieje")
        if actual_kod_typu != expected_kod_typu:
            raise HTTPException(status_code=400, detail=f"Pozycja nie nalezy do {expected_kod_typu}")

        conn.execute(
            """
            DELETE FROM slownik_status_inspekcji_styl
            WHERE slownik_pozycja_id = ?
            """,
            (int(slownik_pozycja_id),),
        )
        conn.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _validate_inspection_type_name(raw_name: str) -> None:
    normalized = " ".join(raw_name.casefold().split())
    if normalized not in _ALLOWED_INSPECTION_TYPE_NAMES:
        raise HTTPException(status_code=400, detail="Dozwolone typy inspekcji: Kontrola, Wizyta nadzorcza")


def _kategoria_nazwa(kategoria: int) -> str:
    if kategoria == 3:
        return "Ogolne"
    if kategoria == 1:
        return "Zalecenia"
    if kategoria == 2:
        return "Wnioski sankcyjne"
    return "Inspekcje"


def _list_items_by_kod_typu(conn: Any, kod_typu: str) -> list[dict[str, Any]]:
    aux_col = _slownik_aux_column(conn)
    rows = conn.execute(
        f"""
        SELECT
            p.id,
            t.id AS typ_id,
            t.kod_typu,
            p.kod_pozycji,
            p.nazwa_pozycji,
            p.{aux_col} AS nazwa_uzytkowa,
            p.skrot_pozycji,
            st.kolor,
            st.odcien,
            st.intensywnosc,
            p.kolejnosc,
            p.aktywny
        FROM slownik_pozycje p
        JOIN slownik_typy t ON t.kod_typu = p.kod_typu
        LEFT JOIN slownik_status_inspekcji_styl st ON st.slownik_pozycja_id = p.id
        WHERE lower(p.kod_typu) = lower(?)
          AND (
              lower(?) <> 'osoby'
                            OR EXISTS (
                  SELECT 1
                  FROM users u
                                    WHERE lower(p.kod_pozycji) = lower('OSOBA_' || CAST(u.id AS TEXT))
                                        AND u.aktywny = 1
                                        AND u.rola_id <> 4
                                        AND lower(COALESCE(u.list_visibility, 'visible')) <> 'hidden'
              )
          )
                    AND (
                            lower(?) <> 'typy_inspekcji'
                            OR lower(trim(p.nazwa_pozycji)) IN ('kontrola', 'wizyta nadzorcza')
                    )
        ORDER BY COALESCE(p.kolejnosc, 2147483647) ASC, p.id ASC
        """,
                (kod_typu, kod_typu, kod_typu),
    ).fetchall()

    return [_map_slownik_pozycja_row(dict(row)) for row in rows]


def _normalize_auto_code(raw_name: str, max_len: int = 12) -> str:
    normalized = unicodedata.normalize("NFKD", raw_name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_").upper()
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = "POZYCJA"
    return slug[:max_len].rstrip("_") or "POZYCJA"


def _resolve_next_prefixed_code(conn: Any, kod_typu: str, code_prefix: str) -> str:
    prefix = str(code_prefix or "").strip().upper()
    if not prefix:
        raise HTTPException(status_code=400, detail="Brak prefix_kodu dla typu slownika")

    rows = conn.execute(
        """
        SELECT kod_pozycji
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?)
          AND lower(kod_pozycji) LIKE lower(?)
        """,
        (kod_typu, f"{prefix}_%"),
    ).fetchall()

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$", re.IGNORECASE)
    max_suffix = 0
    for row in rows:
        current_code = str(row["kod_pozycji"] or "").strip()
        match = pattern.fullmatch(current_code)
        if match is None:
            continue
        try:
            suffix = int(match.group(1))
        except ValueError:
            continue
        if suffix > max_suffix:
            max_suffix = suffix

    candidate_suffix = max_suffix + 1
    while True:
        candidate = f"{prefix}_{candidate_suffix}"
        exists = conn.execute(
            """
            SELECT id FROM slownik_pozycje
            WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?)
            LIMIT 1
            """,
            (kod_typu, candidate),
        ).fetchone()
        if exists is None:
            return candidate
        candidate_suffix += 1


def _resolve_unique_code(
    conn: Any,
    kod_typu: str,
    nazwa_pozycji: str,
    explicit_code: str | None,
    code_prefix: str | None = None,
    exclude_id: int | None = None,
) -> str:
    if explicit_code is not None and explicit_code.strip():
        candidate = explicit_code.strip().upper()
        if not re.fullmatch(r"[A-Z0-9_]+", candidate):
            raise HTTPException(status_code=400, detail="kodPozycji musi byc uppercase i bez spacji")

        if exclude_id is None:
            exists = conn.execute(
                """
                SELECT id FROM slownik_pozycje
                WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?)
                LIMIT 1
                """,
                (kod_typu, candidate),
            ).fetchone()
        else:
            exists = conn.execute(
                """
                SELECT id FROM slownik_pozycje
                WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?) AND id <> ?
                LIMIT 1
                """,
                (kod_typu, candidate, exclude_id),
            ).fetchone()

        if exists is not None:
            raise HTTPException(status_code=409, detail="kodPozycji juz istnieje w tym slowniku")
        return candidate

    if code_prefix is not None and code_prefix.strip():
        if exclude_id is not None:
            raise HTTPException(status_code=400, detail="Automatyczny kod z prefixem nie obsluguje exclude_id")
        return _resolve_next_prefixed_code(conn, kod_typu, code_prefix)

    base_code = _normalize_auto_code(nazwa_pozycji)
    candidate = base_code
    suffix = 2
    while True:
        params: tuple[Any, ...]
        sql: str
        if exclude_id is None:
            sql = (
                "SELECT id FROM slownik_pozycje "
                "WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?) LIMIT 1"
            )
            params = (kod_typu, candidate)
        else:
            sql = (
                "SELECT id FROM slownik_pozycje "
                "WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?) AND id <> ? LIMIT 1"
            )
            params = (kod_typu, candidate, exclude_id)

        exists = conn.execute(sql, params).fetchone()
        if exists is None:
            return candidate
        candidate = f"{base_code}_{suffix}"
        suffix += 1


@router.get("/api/slowniki/typy", response_model=list[SlownikTypRead])
def list_slownik_types() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, kod_typu, nazwa_typu, kategoria, opis, aktywny
            FROM slownik_typy
            ORDER BY kategoria ASC, id ASC
            """
        ).fetchall()

    return [
        {
            "id": row["id"],
            "kodTypu": row["kod_typu"],
            "nazwaTypu": row["nazwa_typu"],
            "kategoria": int(row["kategoria"]),
            "kategoriaNazwa": _kategoria_nazwa(int(row["kategoria"])),
            "opis": row["opis"],
            "aktywny": bool(row["aktywny"]),
        }
        for row in rows
    ]


@router.get("/api/slowniki/pozycje", response_model=list[SlownikPozycjaRead])
def list_slownik_items(
    kod_typu: str = Query(..., alias="kodTypu"),
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        return _list_items_by_kod_typu(conn, kod_typu)


@router.get(
    "/api/slowniki/wnioski-sankcyjne/rozstrzygniecie-i",
    response_model=list[SlownikPozycjaRead],
)
def list_sanction_decisions_i() -> list[dict[str, Any]]:
    with get_connection() as conn:
        return _list_items_by_kod_typu(conn, "rozstrzygniecie_wniosku_sankcyjnego_I")


@router.post("/api/slowniki/pozycje", response_model=SlownikPozycjaRead)
def create_slownik_item(
    payload: SlownikPozycjaCreate,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    kod_typu = payload.kodTypu.strip().lower()
    nazwa_pozycji = payload.nazwaPozycji.strip()

    if not kod_typu:
        raise HTTPException(status_code=400, detail="kodTypu jest wymagane")
    if not nazwa_pozycji:
        raise HTTPException(status_code=400, detail="nazwaPozycji jest wymagane")

    nazwa_uzytkowa = payload.nazwaUzytkowa.strip() if payload.nazwaUzytkowa else None
    if nazwa_uzytkowa == "":
        nazwa_uzytkowa = None
    if kod_typu != "zakresy_inspekcji":
        nazwa_uzytkowa = None

    if kod_typu == "typy_inspekcji":
        _validate_inspection_type_name(nazwa_pozycji)

    with get_connection() as conn:
        operator = _get_operator(conn, x_operator_login)
        aux_col = _slownik_aux_column(conn)

        typ = conn.execute(
            "SELECT id, kod_typu, prefix_kodu FROM slownik_typy WHERE lower(kod_typu) = lower(?) LIMIT 1",
            (kod_typu,),
        ).fetchone()
        if typ is None:
            raise HTTPException(status_code=400, detail="Nie znaleziono kodTypu")

        resolved_kod_typu = str(typ["kod_typu"])
        default_kod_pozycji = payload.kodPozycji
        resolved_kod_pozycji = _resolve_unique_code(
            conn,
            resolved_kod_typu,
            nazwa_pozycji,
            default_kod_pozycji,
            str(typ["prefix_kodu"] or "").strip() or None,
        )

        _ensure_manager_can_modify_entry(operator, resolved_kod_typu, resolved_kod_pozycji)

        resolved_kolejnosc = payload.kolejnosc
        if resolved_kolejnosc is None:
            max_row = conn.execute(
                "SELECT COALESCE(MAX(kolejnosc), 0) AS max_kolejnosc FROM slownik_pozycje WHERE lower(kod_typu) = lower(?)",
                (resolved_kod_typu,),
            ).fetchone()
            resolved_kolejnosc = int(max_row["max_kolejnosc"]) + 1

        cursor = conn.execute(
            f"""
            INSERT INTO slownik_pozycje
            (kod_typu, kod_pozycji, nazwa_pozycji, {aux_col}, skrot_pozycji, kolejnosc, aktywny)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_kod_typu,
                resolved_kod_pozycji,
                nazwa_pozycji,
                nazwa_uzytkowa,
                payload.skrotPozycji.strip() if payload.skrotPozycji else None,
                resolved_kolejnosc,
                1 if payload.aktywny else 0,
            ),
        )
        conn.commit()

        row = conn.execute(
            f"""
            SELECT
                p.id,
                t.id AS typ_id,
                t.kod_typu,
                p.kod_pozycji,
                p.nazwa_pozycji,
                p.{aux_col} AS nazwa_uzytkowa,
                p.skrot_pozycji,
                p.kolejnosc,
                p.aktywny
            FROM slownik_pozycje p
            JOIN slownik_typy t ON t.kod_typu = p.kod_typu
            WHERE p.id = ?
            LIMIT 1
            """,
            (cursor.lastrowid,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac zapisanej pozycji")

    return _map_slownik_pozycja_row(dict(row))


@router.put("/api/slowniki/statusy-inspekcji/styl", response_model=StatusInspekcjiStylRead)
def upsert_status_inspekcji_style(
    payload: StatusInspekcjiStylUpsert,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any] | Response:
    return _upsert_status_style(payload, x_operator_login, "statusy_inspekcji")


@router.delete("/api/slowniki/statusy-inspekcji/styl/{slownik_pozycja_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_status_inspekcji_style(
    slownik_pozycja_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> Response:
    return _delete_status_style(slownik_pozycja_id, x_operator_login, "statusy_inspekcji")


@router.put("/api/slowniki/statusy-zalecen/styl", response_model=StatusInspekcjiStylRead)
def upsert_status_zalecen_style(
    payload: StatusInspekcjiStylUpsert,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any] | Response:
    return _upsert_status_style(payload, x_operator_login, "statusy_zalecen")


@router.delete("/api/slowniki/statusy-zalecen/styl/{slownik_pozycja_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_status_zalecen_style(
    slownik_pozycja_id: int,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> Response:
    return _delete_status_style(
        slownik_pozycja_id,
        x_operator_login,
        "statusy_zalecen",
        idempotent_not_found=True,
    )


@router.put("/api/slowniki/pozycje", response_model=SlownikPozycjaRead)
def update_slownik_item(
    payload: SlownikPozycjaUpdate,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    kod_typu = payload.kodTypu.strip().lower()
    nazwa_pozycji = payload.nazwaPozycji.strip()

    if not kod_typu:
        raise HTTPException(status_code=400, detail="kodTypu jest wymagane")
    if not nazwa_pozycji:
        raise HTTPException(status_code=400, detail="nazwaPozycji jest wymagane")

    nazwa_uzytkowa = payload.nazwaUzytkowa.strip() if payload.nazwaUzytkowa else None
    if nazwa_uzytkowa == "":
        nazwa_uzytkowa = None
    if kod_typu != "zakresy_inspekcji":
        nazwa_uzytkowa = None

    if kod_typu == "typy_inspekcji":
        _validate_inspection_type_name(nazwa_pozycji)

    with get_connection() as conn:
        operator = _get_operator(conn, x_operator_login)
        aux_col = _slownik_aux_column(conn)

        typ = conn.execute(
            "SELECT id, kod_typu FROM slownik_typy WHERE lower(kod_typu) = lower(?) LIMIT 1",
            (kod_typu,),
        ).fetchone()
        if typ is None:
            raise HTTPException(status_code=400, detail="Nie znaleziono kodTypu")

        resolved_kod_typu = str(typ["kod_typu"])

        if payload.kodPozycji is None:
            raise HTTPException(status_code=400, detail="Dla tego endpointu kodPozycji jest wymagane (lub uzyj PUT /api/slowniki/pozycje/{id})")

        kod_pozycji = payload.kodPozycji.strip().upper()
        if not kod_pozycji:
            raise HTTPException(status_code=400, detail="kodPozycji jest wymagane")
        if not re.fullmatch(r"[A-Z0-9_]+", kod_pozycji):
            raise HTTPException(status_code=400, detail="kodPozycji musi byc uppercase i bez spacji")

        _ensure_manager_can_modify_entry(operator, resolved_kod_typu, kod_pozycji)

        current = conn.execute(
            """
            SELECT id
            FROM slownik_pozycje
            WHERE lower(kod_typu) = lower(?) AND lower(kod_pozycji) = lower(?)
            LIMIT 1
            """,
            (resolved_kod_typu, kod_pozycji),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail="Pozycja slownika nie istnieje")

        conn.execute(
            f"""
            UPDATE slownik_pozycje
            SET nazwa_pozycji = ?,
                {aux_col} = ?,
                skrot_pozycji = ?,
                aktywny = ?,
                zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                nazwa_pozycji,
                nazwa_uzytkowa,
                payload.skrotPozycji.strip() if payload.skrotPozycji else None,
                1 if payload.aktywny else 0,
                int(current["id"]),
            ),
        )
        conn.commit()

        row = conn.execute(
            f"""
            SELECT
                p.id,
                t.id AS typ_id,
                t.kod_typu,
                p.kod_pozycji,
                p.nazwa_pozycji,
                p.{aux_col} AS nazwa_uzytkowa,
                p.skrot_pozycji,
                p.kolejnosc,
                p.aktywny
            FROM slownik_pozycje p
            JOIN slownik_typy t ON t.kod_typu = p.kod_typu
            WHERE p.id = ?
            LIMIT 1
            """,
            (int(current["id"]),),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac zaktualizowanej pozycji")

    return _map_slownik_pozycja_row(dict(row))


@router.put("/api/slowniki/pozycje/{pozycja_id}", response_model=SlownikPozycjaRead)
def update_slownik_item_by_id(
    pozycja_id: int,
    payload: SlownikPozycjaUpdateById,
    x_operator_login: str = Header(..., alias="X-Operator-Login"),
) -> dict[str, Any]:
    nazwa_pozycji = payload.nazwaPozycji.strip()
    if not nazwa_pozycji:
        raise HTTPException(status_code=400, detail="nazwaPozycji jest wymagane")

    nazwa_uzytkowa = payload.nazwaUzytkowa.strip() if payload.nazwaUzytkowa else None
    if nazwa_uzytkowa == "":
        nazwa_uzytkowa = None

    with get_connection() as conn:
        operator = _get_operator(conn, x_operator_login)
        aux_col = _slownik_aux_column(conn)

        current = conn.execute(
            """
            SELECT p.id, p.kod_typu, p.kod_pozycji
            FROM slownik_pozycje p
            WHERE p.id = ?
            LIMIT 1
            """,
            (pozycja_id,),
        ).fetchone()
        if current is None:
            raise HTTPException(status_code=404, detail="Pozycja slownika nie istnieje")

        _ensure_manager_can_modify_entry(
            operator,
            str(current["kod_typu"]),
            str(current["kod_pozycji"]),
        )

        if str(current["kod_typu"]).strip().lower() == "typy_inspekcji":
            _validate_inspection_type_name(nazwa_pozycji)
        if str(current["kod_typu"]).strip().lower() != "zakresy_inspekcji":
            nazwa_uzytkowa = None

        conn.execute(
            f"""
            UPDATE slownik_pozycje
            SET nazwa_pozycji = ?,
                {aux_col} = ?,
                skrot_pozycji = ?,
                aktywny = ?,
                zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                nazwa_pozycji,
                nazwa_uzytkowa,
                payload.skrotPozycji.strip() if payload.skrotPozycji else None,
                1 if payload.aktywny else 0,
                pozycja_id,
            ),
        )
        conn.commit()

        row = conn.execute(
            f"""
            SELECT
                p.id,
                t.id AS typ_id,
                t.kod_typu,
                p.kod_pozycji,
                p.nazwa_pozycji,
                p.{aux_col} AS nazwa_uzytkowa,
                p.skrot_pozycji,
                p.kolejnosc,
                p.aktywny
            FROM slownik_pozycje p
            JOIN slownik_typy t ON t.kod_typu = p.kod_typu
            WHERE p.id = ?
            LIMIT 1
            """,
            (pozycja_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail="Nie udalo sie pobrac zaktualizowanej pozycji")

    return _map_slownik_pozycja_row(dict(row))
