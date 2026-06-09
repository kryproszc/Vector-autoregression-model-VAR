from __future__ import annotations

from pathlib import Path

from migrate_slownik_type_prefixes import run


def main() -> int:
    db_path = Path(__file__).resolve().parents[1] / "Baza" / "rejestr.db"

    print("[ONE-CLICK] Start migracji prefixow i renumeracji pozycji")
    print(f"[ONE-CLICK] Baza: {db_path}")

    exit_code = run(
        db_path=db_path,
        apply_changes=True,
        force_overwrite=True,
        renumber_positions=True,
    )

    if exit_code == 0:
        print("[ONE-CLICK] OK")
    else:
        print(f"[ONE-CLICK] ERROR (kod: {exit_code})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
