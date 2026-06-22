from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.database.connection import get_connection


@dataclass(frozen=True)
class DictionaryPools:
    nazwy_podmiotow: list[int]
    typy_inspekcji: list[int]
    zakresy_inspekcji: list[int]
    rynki: list[int]
    rodzaje_podmiotu: list[int]
    statusy_inspekcji: list[int]
    statusy_zalecen: list[int]
    rozstrzygniecie_i: list[int]
    rozstrzygniecie_ii: list[int]
    department: list[int]
    info_wszczecie: list[int]
    rozstrzygniecie_wniosku: list[int]
    nazwy_podmiotow_sankcje: list[int]
    sankcja: list[int]
    podstawa_prawna_sankcji: list[int]
    naruszenia_skutkujace_sankcja: list[int]
    osoby: list[int]


def _active_user_ids(conn: Any) -> list[int]:
    rows = conn.execute(
        """
        SELECT id
        FROM users
        WHERE aktywny = 1
        ORDER BY id ASC
        """
    ).fetchall()
    return [int(r["id"]) for r in rows]


def _dictionary_ids(conn: Any, kod_typu: str) -> list[int]:
    rows = conn.execute(
        """
        SELECT id
        FROM slownik_pozycje
        WHERE lower(kod_typu) = lower(?)
          AND aktywny = 1
        ORDER BY COALESCE(kolejnosc, 999999), id
        """,
        (kod_typu,),
    ).fetchall()
    return [int(r["id"]) for r in rows]


def _inspection_status_ids_without_closed(conn: Any) -> list[int]:
    rows = conn.execute(
        """
        SELECT id, COALESCE(kod_pozycji, '') AS kod_pozycji
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'statusy_inspekcji'
          AND aktywny = 1
        ORDER BY COALESCE(kolejnosc, 999999), id
        """
    ).fetchall()

    blocked = {"CLOSED_WITH_RECOMMENDATIONS", "CLOSED_WITHOUT_RECOMMENDATIONS"}
    filtered = [int(r["id"]) for r in rows if str(r["kod_pozycji"] or "").upper() not in blocked]
    if filtered:
        return filtered
    return [int(r["id"]) for r in rows]


def _load_dictionaries(conn: Any) -> DictionaryPools:
    return DictionaryPools(
        nazwy_podmiotow=_dictionary_ids(conn, "nazwy_podmiotow"),
        typy_inspekcji=_dictionary_ids(conn, "typy_inspekcji"),
        zakresy_inspekcji=_dictionary_ids(conn, "zakresy_inspekcji"),
        rynki=_dictionary_ids(conn, "rynki"),
        rodzaje_podmiotu=_dictionary_ids(conn, "rodzaje_podmiotu"),
        statusy_inspekcji=_inspection_status_ids_without_closed(conn),
        statusy_zalecen=_dictionary_ids(conn, "statusy_zalecen"),
        rozstrzygniecie_i=_dictionary_ids(conn, "rozstrzygniecie_decyzji_i"),
        rozstrzygniecie_ii=_dictionary_ids(conn, "rozstrzygniecie_decyzji_ii"),
        department=_dictionary_ids(conn, "department"),
        info_wszczecie=_dictionary_ids(conn, "informacja_o_wszczeciu_postepowania_sankcyjnego"),
        rozstrzygniecie_wniosku=_dictionary_ids(conn, "rozstrzygniecie_wniosku_sankcyjnego_I"),
        nazwy_podmiotow_sankcje=_dictionary_ids(conn, "nazwy_podmiotow_sankcje"),
        sankcja=_dictionary_ids(conn, "sankcja"),
        podstawa_prawna_sankcji=_dictionary_ids(conn, "podstawa_prawna_sankcji"),
        naruszenia_skutkujace_sankcja=_dictionary_ids(conn, "naruszenia_skutkujace_sankcja"),
        osoby=_dictionary_ids(conn, "osoby"),
    )


def _pick(pool: list[int], rng: random.Random) -> int | None:
    if not pool:
        return None
    return int(rng.choice(pool))


def _pick_cycle(pool: list[int], idx: int) -> int | None:
    if not pool:
        return None
    return int(pool[idx % len(pool)])


def _next_lp(conn: Any, table: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX(lp), 0) AS max_lp FROM {table}").fetchone()
    return int(row["max_lp"]) + 1


def _next_code_with_prefix_year(conn: Any, table: str, col: str, prefix: str, year: int) -> str:
    pattern = f"{prefix}/{year}/%"
    rows = conn.execute(f"SELECT {col} AS code FROM {table} WHERE {col} LIKE ?", (pattern,)).fetchall()
    max_seq = 0
    for row in rows:
        raw = str(row["code"] or "").strip()
        parts = raw.split("/")
        if len(parts) != 3:
            continue
        if parts[0] != prefix or parts[1] != str(year):
            continue
        try:
            seq = int(parts[2])
        except ValueError:
            continue
        max_seq = max(max_seq, seq)
    return f"{prefix}/{year}/{max_seq + 1}"


def _inspection_prefix_from_typ_name(typ_name: str | None) -> str | None:
    normalized = " ".join(str(typ_name or "").casefold().split())
    if normalized == "kontrola":
        return "K"
    if normalized == "wizyta nadzorcza":
        return "WN"
    return None


def _inspection_typ_name(conn: Any, typ_inspekcji_id: int | None) -> str | None:
    if typ_inspekcji_id is None:
        return None
    row = conn.execute(
        "SELECT nazwa_pozycji FROM slownik_pozycje WHERE id = ? LIMIT 1",
        (int(typ_inspekcji_id),),
    ).fetchone()
    if row is None:
        return None
    return str(row["nazwa_pozycji"] or "").strip() or None


def _insert_inspection_scope(conn: Any, inspection_id: int, scope_id: int) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO inspection_scopes (inspection_id, scope_id)
        VALUES (?, ?)
        """,
        (inspection_id, scope_id),
    )


def _insert_inspection_member(conn: Any, inspection_id: int, user_id: int) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO inspection_members (inspection_id, user_id)
        VALUES (?, ?)
        """,
        (inspection_id, user_id),
    )


def _insert_decision_person(conn: Any, decision_id: int, slownik_pozycja_id: int, user_id: int, instance: str) -> None:
    if instance == "I":
        table = "obligating_decisions_persons_i"
    else:
        table = "obligating_decisions_persons_ii"

    conn.execute(
        f"""
        INSERT OR IGNORE INTO {table}
        (obligating_decision_id, slownik_pozycja_id, created_by_user_id, updated_by_user_id)
        VALUES (?, ?, ?, ?)
        """,
        (decision_id, slownik_pozycja_id, user_id, user_id),
    )


def _insert_risk_multi(
    conn: Any,
    table: str,
    risk_exposure_id: int,
    slownik_pozycja_id: int,
    user_id: int,
) -> None:
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {table}
        (risk_exposure_id, slownik_pozycja_id, created_by_user_id, updated_by_user_id)
        VALUES (?, ?, ?, ?)
        """,
        (risk_exposure_id, slownik_pozycja_id, user_id, user_id),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dodaje lancuchy biznesowe: inspekcje -> zalecenia -> decyzje zobowiazujace + wnioski sankcyjne. "
            "Wartosci slownikowe sa pobierane tylko z istniejacych slownikow."
        )
    )
    parser.add_argument("--count", type=int, default=100, help="Liczba rekordow dla kazdego rejestru (domyslnie 100).")
    parser.add_argument("--seed", type=int, default=20260622, help="Seed dla losowania (powtarzalnosc).")
    parser.add_argument("--start-year", type=int, default=date.today().year - 1, help="Rok startowy dat inspekcji.")
    parser.add_argument("--dry-run", action="store_true", help="Nie zapisuje zmian, tylko liczy i waliduje.")
    args = parser.parse_args()

    if args.count < 1:
        raise RuntimeError("--count musi byc >= 1")

    rng = random.Random(args.seed)

    created_inspection_ids: list[int] = []
    created_recommendation_ids: list[int] = []
    created_decision_ids: list[int] = []
    created_risk_ids: list[int] = []

    with get_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")

        users = _active_user_ids(conn)
        if not users:
            raise RuntimeError("Brak aktywnych uzytkownikow. Skrypt nie moze przypisac created_by_user_id.")

        dictionaries = _load_dictionaries(conn)

        next_inspection_lp = _next_lp(conn, "inspections")
        next_risk_lp = _next_lp(conn, "risk_exposure_requests")

        for i in range(args.count):
            idx = i + 1
            owner_user_id = int(users[i % len(users)])
            leader_user_id = int(users[(i + 1) % len(users)])

            start_day = date(args.start_year, 1, 1) + timedelta(days=i * 3)
            end_day = start_day + timedelta(days=2)
            protocol_day = end_day + timedelta(days=14)
            recommendation_day = protocol_day + timedelta(days=7)
            sanction_request_day = protocol_day + timedelta(days=3)

            nazwa_podmiotu_id = _pick_cycle(dictionaries.nazwy_podmiotow, i)
            typ_inspekcji_id = _pick(dictionaries.typy_inspekcji, rng)
            zakres_inspekcji_id = _pick(dictionaries.zakresy_inspekcji, rng)
            rynek_id = _pick(dictionaries.rynki, rng)
            rodzaj_podmiotu_id = _pick(dictionaries.rodzaje_podmiotu, rng)
            status_inspekcji_id = _pick(dictionaries.statusy_inspekcji, rng)

            typ_name = _inspection_typ_name(conn, typ_inspekcji_id)
            inspection_prefix = _inspection_prefix_from_typ_name(typ_name)
            inspection_code = (
                _next_code_with_prefix_year(
                    conn,
                    table="inspections",
                    col="kod_inspekcji",
                    prefix=inspection_prefix,
                    year=start_day.year,
                )
                if inspection_prefix
                else None
            )

            inspection_cursor = conn.execute(
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
                    komentarz,
                    status_inspekcji_id,
                    data_protokolu_sprawozdania
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_inspection_lp,
                    inspection_code,
                    owner_user_id,
                    nazwa_podmiotu_id,
                    typ_inspekcji_id,
                    zakres_inspekcji_id,
                    start_day.isoformat(),
                    end_day.isoformat(),
                    leader_user_id,
                    rynek_id,
                    rodzaj_podmiotu_id,
                    f"Seed biznesowy #{idx}",
                    status_inspekcji_id,
                    protocol_day.isoformat(),
                ),
            )
            inspection_id = int(inspection_cursor.lastrowid)
            created_inspection_ids.append(inspection_id)
            next_inspection_lp += 1

            if zakres_inspekcji_id is not None:
                _insert_inspection_scope(conn, inspection_id, zakres_inspekcji_id)
            _insert_inspection_member(conn, inspection_id, leader_user_id)

            recommendation_code = _next_code_with_prefix_year(
                conn,
                table="recommendations",
                col="kod_zalecenia",
                prefix="Z",
                year=recommendation_day.year,
            )
            recommendation_status_id = _pick(dictionaries.statusy_zalecen, rng)

            recommendation_cursor = conn.execute(
                """
                INSERT INTO recommendations (
                    inspection_id,
                    pozycja,
                    kod_zalecenia,
                    nazwa_podmiotu_id,
                    data_zalecen,
                    status_zalecenia_id,
                    komentarz,
                    created_by_user_id,
                    updated_by_user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inspection_id,
                    1,
                    recommendation_code,
                    nazwa_podmiotu_id,
                    recommendation_day.isoformat(),
                    recommendation_status_id,
                    f"Seed zalecenie #{idx}",
                    owner_user_id,
                    owner_user_id,
                ),
            )
            recommendation_id = int(recommendation_cursor.lastrowid)
            created_recommendation_ids.append(recommendation_id)

            decision_code = _next_code_with_prefix_year(
                conn,
                table="obligating_decisions",
                col="kod_decyzji",
                prefix="DZ",
                year=recommendation_day.year,
            )
            decision_roz_i = _pick(dictionaries.rozstrzygniecie_i, rng)
            decision_roz_ii = _pick(dictionaries.rozstrzygniecie_ii, rng)

            decision_cursor = conn.execute(
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
                    decision_code,
                    recommendation_code,
                    nazwa_podmiotu_id,
                    1,
                    (recommendation_day + timedelta(days=2)).isoformat(),
                    (recommendation_day + timedelta(days=10)).isoformat(),
                    (recommendation_day + timedelta(days=14)).isoformat(),
                    decision_roz_i,
                    (recommendation_day + timedelta(days=20)).isoformat(),
                    (recommendation_day + timedelta(days=21)).isoformat(),
                    (recommendation_day + timedelta(days=30)).isoformat(),
                    (recommendation_day + timedelta(days=35)).isoformat(),
                    decision_roz_ii,
                    f"Seed decyzja #{idx}",
                    owner_user_id,
                    owner_user_id,
                ),
            )
            decision_id = int(decision_cursor.lastrowid)
            created_decision_ids.append(decision_id)

            decision_person_i_id = _pick(dictionaries.osoby, rng)
            decision_person_ii_id = _pick(dictionaries.osoby, rng)
            if decision_person_i_id is not None:
                _insert_decision_person(conn, decision_id, decision_person_i_id, owner_user_id, instance="I")
            if decision_person_ii_id is not None:
                _insert_decision_person(conn, decision_id, decision_person_ii_id, owner_user_id, instance="II")

            sanction_code = _next_code_with_prefix_year(
                conn,
                table="risk_exposure_requests",
                col="kod_sankcji",
                prefix="WS",
                year=sanction_request_day.year,
            )
            wniosek_do_id = _pick(dictionaries.department, rng)
            wszczecie_id = _pick(dictionaries.info_wszczecie, rng)
            rozstrzygniecie_wniosku_id = _pick(dictionaries.rozstrzygniecie_wniosku, rng)

            risk_cursor = conn.execute(
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
                    next_risk_lp,
                    sanction_code,
                    inspection_id,
                    nazwa_podmiotu_id,
                    sanction_request_day.isoformat(),
                    wniosek_do_id,
                    wszczecie_id,
                    rozstrzygniecie_wniosku_id,
                    f"Seed wniosek sankcyjny #{idx}",
                    owner_user_id,
                    owner_user_id,
                ),
            )
            risk_id = int(risk_cursor.lastrowid)
            created_risk_ids.append(risk_id)
            next_risk_lp += 1

            risk_subject_id = _pick(dictionaries.nazwy_podmiotow_sankcje, rng)
            risk_sankcja_id = _pick(dictionaries.sankcja, rng)
            risk_legal_basis_id = _pick(dictionaries.podstawa_prawna_sankcji, rng)
            risk_violation_id = _pick(dictionaries.naruszenia_skutkujace_sankcja, rng)

            if risk_subject_id is not None:
                _insert_risk_multi(conn, "risk_exposure_sanction_subjects", risk_id, risk_subject_id, owner_user_id)
            if risk_sankcja_id is not None:
                _insert_risk_multi(conn, "risk_exposure_sanctions", risk_id, risk_sankcja_id, owner_user_id)
            if risk_legal_basis_id is not None:
                _insert_risk_multi(conn, "risk_exposure_legal_bases", risk_id, risk_legal_basis_id, owner_user_id)
            if risk_violation_id is not None:
                _insert_risk_multi(conn, "risk_exposure_violations", risk_id, risk_violation_id, owner_user_id)

        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()

    result = {
        "dryRun": bool(args.dry_run),
        "countRequested": int(args.count),
        "inserted": {
            "inspections": len(created_inspection_ids),
            "recommendations": len(created_recommendation_ids),
            "obligatingDecisions": len(created_decision_ids),
            "sanctionRequests": len(created_risk_ids),
        },
        "sampleIds": {
            "inspection": created_inspection_ids[:5],
            "recommendation": created_recommendation_ids[:5],
            "obligatingDecision": created_decision_ids[:5],
            "sanctionRequest": created_risk_ids[:5],
        },
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
