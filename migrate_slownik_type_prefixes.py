from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any


def _default_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "Baza" / "rejestr.db"


# Explicit mapping for known dictionary types used by UI/business.
PREFIX_BY_KOD_TYPU: dict[str, str] = {
    # Kategoria: Ogolne
    "department_ogolne": "o_d",
    # Kategoria: Inspekcje
    "nazwy_podmiotow": "i_np",
    "typy_inspekcji": "i_ti",
    "zakresy_inspekcji": "i_zi",
    "rynki": "i_r",
    "rodzaje_podmiotu": "i_rp",
    "statusy_inspekcji": "i_si",
    "osoby": "i_o",
    "zespoly": "i_z",
    # Kategoria: Zalecenia
    "statusy_zalecen": "z_sz",
    # Kategoria: Wnioski sankcyjne
    "department": "w_d",
    "sankcja": "w_s",
    "podstawa_prawna_sankcji": "w_pps",
    "naruszenia_skutkujace_sankcja": "w_nss",
    "informacja_o_wszczeciu_postepowania_sankcyjnego": "w_iowps",
    "rozstrzygniecie_wniosku_sankcyjnego_i": "w_rws_i",
    "nazwy_podmiotow_sankcje": "w_nps",
    # Kategoria: Decyzje zobowiazujace (oddzielny tab w UI)
    "rozstrzygniecie_decyzji_i": "d_rd_i",
    "rozstrzygniecie_decyzji_ii": "d_rd_ii",
}


CATEGORY_PREFIX: dict[int, str] = {
    0: "i",  # Inspekcje
    1: "z",  # Zalecenia
    2: "w",  # Wnioski sankcyjne
    3: "o",  # Ogolne
}


def _normalize_code(value: str) -> str:
    return "_".join(part for part in value.strip().lower().split("_") if part)


def _fallback_prefix(kod_typu: str, kategoria: int) -> str:
    base = CATEGORY_PREFIX.get(int(kategoria), "x")
    parts = [part for part in _normalize_code(kod_typu).split("_") if part]
    initials = "".join(part[0] for part in parts[:6])
    if not initials:
        initials = "t"
    return f"{base}_{initials}"


def _ensure_prefix_column(conn: sqlite3.Connection, apply_changes: bool) -> bool:
    cols = conn.execute("PRAGMA table_info(slownik_typy)").fetchall()
    has_column = any(str(col[1]).strip().lower() == "prefix_kodu" for col in cols)
    if has_column:
        return False

    if apply_changes:
        conn.execute("ALTER TABLE slownik_typy ADD COLUMN prefix_kodu TEXT")
    return True


def _ensure_legacy_code_column(conn: sqlite3.Connection, apply_changes: bool) -> bool:
    cols = conn.execute("PRAGMA table_info(slownik_pozycje)").fetchall()
    has_column = any(str(col[1]).strip().lower() == "kod_pozycji_legacy" for col in cols)
    if has_column:
        return False

    if apply_changes:
        conn.execute("ALTER TABLE slownik_pozycje ADD COLUMN kod_pozycji_legacy TEXT")
    return True


def _fetch_types(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cols = conn.execute("PRAGMA table_info(slownik_typy)").fetchall()
    has_prefix = any(str(col[1]).strip().lower() == "prefix_kodu" for col in cols)

    if has_prefix:
        rows = conn.execute(
            """
            SELECT id, kod_typu, nazwa_typu, kategoria, prefix_kodu
            FROM slownik_typy
            ORDER BY kategoria ASC, id ASC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, kod_typu, nazwa_typu, kategoria, NULL AS prefix_kodu
            FROM slownik_typy
            ORDER BY kategoria ASC, id ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def _detect_collisions(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    collisions: dict[str, list[str]] = {}
    for row in rows:
        pref = str(row["target_prefix"])
        collisions.setdefault(pref, []).append(str(row["kod_typu"]))
    return {k: v for k, v in collisions.items() if len(v) > 1}


def _print_type_summary(rows: list[dict[str, Any]]) -> None:
    print("\n[PLAN] slownik_typy -> prefix_kodu")
    print("id | kategoria | kod_typu | obecny_prefix | target_prefix")
    for row in rows:
        current = row.get("prefix_kodu")
        current_norm = str(current).strip() if current is not None else ""
        print(
            f"{int(row['id'])} | {int(row['kategoria'])} | {row['kod_typu']} | "
            f"{(current_norm or '-')} | {row['target_prefix']}"
        )



def _print_positions_summary(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT kod_typu, COUNT(*) AS cnt, MIN(id) AS min_id, MAX(id) AS max_id
        FROM slownik_pozycje
        GROUP BY kod_typu
        ORDER BY lower(kod_typu) ASC
        """
    ).fetchall()

    print("\n[INFO] Rozklad danych w slownik_pozycje (wymieszane ID to normalne)")
    print("kod_typu | count | min_id | max_id")
    for row in rows:
        print(f"{row['kod_typu']} | {int(row['cnt'])} | {int(row['min_id'])} | {int(row['max_id'])}")


def _build_position_renumber_plan(
    conn: sqlite3.Connection,
    type_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for type_row in type_rows:
        kod_typu = str(type_row["kod_typu"])
        target_prefix = str(type_row["target_prefix"] or "").strip().upper()
        if not target_prefix:
            continue

        pos_rows = conn.execute(
            """
            SELECT id, kod_pozycji, kolejnosc
            FROM slownik_pozycje
            WHERE lower(kod_typu) = lower(?)
            ORDER BY COALESCE(kolejnosc, 2147483647) ASC, id ASC
            """,
            (kod_typu,),
        ).fetchall()

        for idx, pos in enumerate(pos_rows, start=1):
            current_code = str(pos["kod_pozycji"] or "").strip()
            target_code = f"{target_prefix}_{idx}"
            if current_code.lower() == target_code.lower():
                continue
            plan.append(
                {
                    "id": int(pos["id"]),
                    "kod_typu": kod_typu,
                    "old_code": current_code,
                    "new_code": target_code,
                }
            )
    return plan


def _print_renumber_plan(plan: list[dict[str, Any]], limit: int = 60) -> None:
    print("\n[PLAN] Renumeracja slownik_pozycje do formatu prefix_n")
    print(f"[PLAN] Liczba zmian: {len(plan)}")
    if not plan:
        return

    print("id | kod_typu | old_code -> new_code")
    for row in plan[:limit]:
        print(f"{row['id']} | {row['kod_typu']} | {row['old_code']} -> {row['new_code']}")
    if len(plan) > limit:
        print(f"... i jeszcze {len(plan) - limit} zmian")


def _apply_renumber_plan(conn: sqlite3.Connection, plan: list[dict[str, Any]]) -> int:
    if not plan:
        return 0

    # Dwuetapowa aktualizacja minimalizuje ryzyko kolizji UNIQUE(kod_typu, kod_pozycji).
    for row in plan:
        temp_code = f"__TMP_MIG__{int(row['id'])}"
        conn.execute(
            """
            UPDATE slownik_pozycje
            SET kod_pozycji_legacy = CASE
                    WHEN kod_pozycji_legacy IS NULL OR trim(kod_pozycji_legacy) = '' THEN kod_pozycji
                    ELSE kod_pozycji_legacy
                END,
                kod_pozycji = ?,
                zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (temp_code, int(row["id"])),
        )

    for row in plan:
        conn.execute(
            """
            UPDATE slownik_pozycje
            SET kod_pozycji = ?, zaktualizowano_o = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (str(row["new_code"]).upper(), int(row["id"])),
        )

    return len(plan)



def run(db_path: Path, apply_changes: bool, force_overwrite: bool, renumber_positions: bool) -> int:
    if not db_path.exists():
        print(f"[ERROR] Nie znaleziono bazy: {db_path}")
        return 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        added_column = _ensure_prefix_column(conn, apply_changes=apply_changes)
        added_legacy_column = False
        if renumber_positions:
            added_legacy_column = _ensure_legacy_code_column(conn, apply_changes=apply_changes)

        rows = _fetch_types(conn)
        for row in rows:
            kod_typu = _normalize_code(str(row["kod_typu"]))
            explicit = PREFIX_BY_KOD_TYPU.get(kod_typu)
            target = explicit if explicit else _fallback_prefix(kod_typu, int(row["kategoria"]))
            target = str(target).strip().upper()
            row["target_prefix"] = target

        _print_type_summary(rows)
        _print_positions_summary(conn)

        collisions = _detect_collisions(rows)
        if collisions:
            print("\n[WARN] Wykryto kolizje prefixow:")
            for pref, types_ in collisions.items():
                print(f"  {pref}: {', '.join(types_)}")
            print("[WARN] Rozwaz poprawienie mapowania PREFIX_BY_KOD_TYPU przed apply.")

        renumber_plan: list[dict[str, Any]] = []
        if renumber_positions:
            renumber_plan = _build_position_renumber_plan(conn, rows)
            _print_renumber_plan(renumber_plan)

        if not apply_changes:
            if added_column:
                print("\n[DRY-RUN] Zostalaby dodana kolumna slownik_typy.prefix_kodu")
            if added_legacy_column:
                print("[DRY-RUN] Zostalaby dodana kolumna slownik_pozycje.kod_pozycji_legacy")
            if renumber_positions:
                print("[DRY-RUN] Zostalaby wykonana renumeracja slownik_pozycje (prefix_n)")
            print("[DRY-RUN] Brak zmian w bazie. Uruchom z --apply, aby zapisac.")
            return 0

        updates = 0
        for row in rows:
            current = row.get("prefix_kodu")
            current_norm = str(current).strip() if current is not None else ""
            target = str(row["target_prefix"]).strip().upper()

            if current_norm == target:
                continue

            if not force_overwrite and current_norm:
                continue

            conn.execute(
                "UPDATE slownik_typy SET prefix_kodu = ?, zaktualizowano_o = CURRENT_TIMESTAMP WHERE id = ?",
                (target, int(row["id"])),
            )
            updates += 1

        renumber_updates = 0
        if renumber_positions:
            renumber_updates = _apply_renumber_plan(conn, renumber_plan)

        conn.commit()

        print("\n[DONE] Migracja zakonczona.")
        print(f"[DONE] Zaktualizowano rekordow slownik_typy: {updates}")
        if renumber_positions:
            print(f"[DONE] Zmieniono kod_pozycji w slownik_pozycje: {renumber_updates}")
        if added_column:
            print("[DONE] Dodano kolumne: slownik_typy.prefix_kodu")
        if added_legacy_column:
            print("[DONE] Dodano kolumne: slownik_pozycje.kod_pozycji_legacy")
        return 0
    finally:
        conn.close()



def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Jednorazowa migracja: dodaje slownik_typy.prefix_kodu i uzupelnia prefiksy "
            "dla typow slownikow na podstawie mapowania/fallbacku."
        )
    )
    parser.add_argument(
        "--db",
        default=str(_default_db_path()),
        help="Sciezka do pliku SQLite (domyslnie: backend/Baza/rejestr.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Zapisz zmiany do bazy (bez tego dziala tylko dry-run)",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Nadpisz rowniez rekordy, ktore maja juz ustawiony prefix_kodu",
    )
    parser.add_argument(
        "--renumber-positions",
        action="store_true",
        help=(
            "Renumeruj istniejace slownik_pozycje do formatu prefix_n "
            "(np. o_d_1, o_d_2) wedlug kolejnosc, a potem id"
        ),
    )
    args = parser.parse_args()

    return run(
        Path(args.db),
        apply_changes=bool(args.apply),
        force_overwrite=bool(args.force_overwrite),
        renumber_positions=bool(args.renumber_positions),
    )


if __name__ == "__main__":
    raise SystemExit(main())
