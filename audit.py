from __future__ import annotations

import uuid
from typing import Any

from app.record_locks import now_rfc3339_utc_ms

# Nazwy rejestrów
REJESTR_INSPEKCJE = "inspekcje"
REJESTR_ZALECENIA = "zalecenia"
REJESTR_DECYZJE = "decyzje"
REJESTR_WNIOSKI = "wnioski_sankcyjne"

# Akcje
AKCJA_CREATE = "CREATE"
AKCJA_UPDATE = "UPDATE"
AKCJA_DELETE = "DELETE"


def new_session_id() -> str:
    return uuid.uuid4().hex


def _slownik_name(conn: Any, item_id: int | None) -> str | None:
    if item_id is None:
        return None
    row = conn.execute(
        "SELECT nazwa_pozycji FROM slownik_pozycje WHERE id = ? LIMIT 1",
        (int(item_id),),
    ).fetchone()
    return str(row["nazwa_pozycji"]) if row is not None else None


def _user_display(conn: Any, user_id: int | None) -> str | None:
    if user_id is None:
        return None
    row = conn.execute(
        "SELECT imie, nazwisko, login FROM users WHERE id = ? LIMIT 1",
        (int(user_id),),
    ).fetchone()
    if row is None:
        return None
    full = f"{(row['imie'] or '').strip()} {(row['nazwisko'] or '').strip()}".strip()
    return full or str(row["login"])


def write_audit_log(
    conn: Any,
    session_id: str,
    uzytkownik: str,
    akcja: str,
    rejestr: str,
    rekord_kod: str,
    changes: list[dict[str, Any]],
) -> None:
    """
    Wstawia wiersze do audit_log w ramach bieżącej transakcji.

    changes – lista słowników z kluczami:
        pole  (str)  – nazwa pola po polsku, np. "Status"
        przed (str | None) – wartość przed zmianą
        po    (str | None) – wartość po zmianie

    Dla CREATE/DELETE wywołaj z pustą listą changes=[].
    """
    data_godz = now_rfc3339_utc_ms()

    if not changes:
        conn.execute(
            """
            INSERT INTO audit_log (session_id, uzytkownik, akcja, data_godz, rejestr, rekord_kod, pole, przed, po)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (session_id, uzytkownik, akcja, data_godz, rejestr, rekord_kod),
        )
        return

    for change in changes:
        pole = change.get("pole")
        przed = _normalize_audit_value(change.get("przed"))
        po = _normalize_audit_value(change.get("po"))
        # Pomijaj pola które się nie zmieniły
        if str(przed or "") == str(po or ""):
            continue
        conn.execute(
            """
            INSERT INTO audit_log (session_id, uzytkownik, akcja, data_godz, rejestr, rekord_kod, pole, przed, po)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, uzytkownik, akcja, data_godz, rejestr, rekord_kod, pole, przed, po),
        )


def _normalize_audit_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.casefold() == "brak":
            return None
        return cleaned

    if isinstance(value, (list, tuple, set)):
        parts = [_normalize_audit_value(item) for item in value]
        compact = [part for part in parts if part is not None]
        if not compact:
            return None
        return ", ".join(compact)

    if isinstance(value, bool):
        return "Tak" if value else "Nie"

    return str(value)


def build_create_changes(items: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for pole, value in items:
        normalized = _normalize_audit_value(value)
        if normalized is None:
            continue
        changes.append({"pole": pole, "przed": None, "po": normalized})
    return changes


# ---------------------------------------------------------------------------
# Helpery do budowania listy zmian dla każdego modułu
# ---------------------------------------------------------------------------

def build_inspection_changes(
    conn: Any,
    before: dict[str, Any],
    fields: dict[str, Any],
    leader_display_after: str | None,
    members_after: str | None,
    scopes_after: str | None,
    members_before: str | None,
    scopes_before: str | None,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    def _add(pole: str, przed: str | None, po: str | None) -> None:
        changes.append({"pole": pole, "przed": przed, "po": po})

    if "nazwaPodmiotu" in fields:
        _add("Nazwa podmiotu",
             _slownik_name(conn, before.get("nazwa_podmiotu_id")),
             _slownik_name(conn, _latest_slownik_id(conn, "nazwy_podmiotow", fields["nazwaPodmiotu"])))

    if "typInspekcji" in fields:
        _add("Typ inspekcji",
             _slownik_name(conn, before.get("typ_inspekcji_id")),
             fields.get("typInspekcji"))

    if "status" in fields:
        _add("Status",
             _slownik_name(conn, before.get("status_inspekcji_id")),
             fields.get("status"))

    if "poczatekInspekcji" in fields:
        _add("Początek inspekcji", before.get("poczatek_inspekcji"), fields["poczatekInspekcji"])

    if "koniecInspekcji" in fields:
        _add("Koniec inspekcji", before.get("koniec_inspekcji"), fields["koniecInspekcji"])

    if "osobaKierujacaUserId" in fields:
        _add("Osoba kierująca",
             _user_display(conn, before.get("osoba_kierujaca_user_id")),
             leader_display_after)

    if members_after is not None:
        _add("Skład zespołu", members_before, members_after)

    if scopes_after is not None:
        _add("Zakres inspekcji", scopes_before, scopes_after)

    if "rynek" in fields:
        _add("Rynek",
             _slownik_name(conn, before.get("rynek_id")),
             fields.get("rynek"))

    if "rodzajPodmiotu" in fields:
        _add("Rodzaj podmiotu",
             _slownik_name(conn, before.get("rodzaj_podmiotu_id")),
             fields.get("rodzajPodmiotu"))

    if "aspektKonsumencki" in fields:
        _add("Aspekt konsumencki", before.get("aspekt_konsumencki"), fields["aspektKonsumencki"])

    if "komentarz" in fields:
        _add("Komentarz", before.get("komentarz"), fields["komentarz"])

    for date_field, label in [
        ("dataProtokolu", "Data protokołu"),
        ("dataDoreczeniaProtokolu", "Data doręczenia protokołu"),
        ("dataAkceptacjiSprawozdania", "Data akceptacji sprawozdania"),
        ("dataDoreczeniaPisma", "Data doręczenia pisma"),
        ("dataWyslaniaPismaZZastrzezeniami", "Data wysłania pisma z zastrzeżeniami"),
        ("dataPismaZastrzezenia", "Data pisma zastrzeżenia"),
        ("dataWplywuPisma", "Data wpływu pisma"),
        ("dataWyslaniaPismaZOdpowiedzia", "Data wysłania pisma z odpowiedzią"),
        ("dataPismaZOdpowiedzia", "Data pisma z odpowiedzią"),
    ]:
        if date_field in fields:
            db_col = {
                "dataProtokolu": "data_protokolu_sprawozdania",
                "dataDoreczeniaProtokolu": "data_doreczenia_protokolu",
                "dataAkceptacjiSprawozdania": "data_akceptacji_sprawozdania",
                "dataDoreczeniaPisma": "data_doreczenia_pisma",
                "dataWyslaniaPismaZZastrzezeniami": "data_wyslania_pisma_z_zastrzezeniami",
                "dataPismaZastrzezenia": "data_pisma_zastrzezenia",
                "dataWplywuPisma": "data_wplywu_pisma",
                "dataWyslaniaPismaZOdpowiedzia": "data_wyslania_pisma_z_odpowiedzia",
                "dataPismaZOdpowiedzia": "data_pisma_z_odpowiedzia",
            }[date_field]
            _add(label, before.get(db_col), fields[date_field])

    if "dataAkceptacjiNotyList" in fields:
        _add("Daty akceptacji noty",
             before.get("_dates_akceptacji"),
             ", ".join(fields["dataAkceptacjiNotyList"] or []) or None)

    if "dataZalecenList" in fields:
        _add("Daty zaleceń",
             before.get("_dates_zalecen"),
             ", ".join(fields["dataZalecenList"] or []) or None)

    return changes


def build_recommendation_changes(
    conn: Any,
    before: dict[str, Any],
    fields: dict[str, Any],
    dates_zalecen_before: str | None,
    dates_akceptacji_before: str | None,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    def _add(pole: str, przed: str | None, po: str | None) -> None:
        changes.append({"pole": pole, "przed": przed, "po": po})

    if "nazwaPodmiotu" in fields:
        _add("Nazwa podmiotu",
             _slownik_name(conn, before.get("nazwa_podmiotu_id")),
             fields.get("nazwaPodmiotu"))
    elif "nazwaPodmiotuId" in fields:
        _add(
            "Nazwa podmiotu",
            _slownik_name(conn, before.get("nazwa_podmiotu_id")),
            _slownik_name(conn, fields.get("nazwaPodmiotuId")),
        )

    if "status" in fields:
        _add("Status",
             _slownik_name(conn, before.get("status_zalecenia_id")),
             fields.get("status"))
    elif "statusId" in fields:
        _add(
            "Status",
            _slownik_name(conn, before.get("status_zalecenia_id")),
            _slownik_name(conn, fields.get("statusId")),
        )

    if "dataZalecen" in fields:
        after_data = fields.get("dataZalecen")
        _add(
            "Data zaleceń",
            before.get("data_zalecen"),
            after_data,
        )

    if "komentarz" in fields:
        _add("Komentarz", before.get("komentarz"), fields["komentarz"])

    if "pozycja" in fields:
        _add("Pozycja", str(before.get("pozycja") or ""), str(fields["pozycja"]))

    if "terminyWykonaniaZalecenList" in fields:
        after_terminy = fields.get("terminyWykonaniaZalecenList") or []
        _add(
            "Terminy wykonania zaleceń",
            dates_zalecen_before,
            ", ".join(after_terminy) or None,
        )

    if "dataAkceptacjiNotyWeryfikacjiList" in fields:
        _add("Daty akceptacji noty weryfikacji",
             dates_akceptacji_before,
             ", ".join(fields["dataAkceptacjiNotyWeryfikacjiList"] or []) or None)

    if "inspectionTeamIds" in fields:
        _add(
            "Zespoły inspekcji",
            before.get("_inspection_team_ids"),
            ", ".join(str(x) for x in (fields["inspectionTeamIds"] or [])) or None,
        )

    return changes


def build_decision_changes(
    conn: Any,
    before: dict[str, Any],
    fields: dict[str, Any],
    persons_i_before: str | None,
    persons_ii_before: str | None,
    persons_i_after: str | None,
    persons_ii_after: str | None,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    def _add(pole: str, przed: str | None, po: str | None) -> None:
        changes.append({"pole": pole, "przed": przed, "po": po})

    if "recommendationKodZalecenia" in fields:
        _add("Powiązane zalecenie", before.get("recommendation_kod_zalecenia"), fields.get("recommendationKodZalecenia"))

    if "nazwaPodmiotu" in fields or "nazwaPodmiotuId" in fields:
        _add("Nazwa podmiotu",
             _slownik_name(conn, before.get("nazwa_podmiotu_id")),
             fields.get("nazwaPodmiotu"))

    if "liczbaZalecen" in fields:
        _add("Liczba zaleceń", str(before.get("liczba_zalecen") or ""), str(fields.get("liczbaZalecen") or ""))

    if "rozstrzygniecieI" in fields or "rozstrzygniecieIId" in fields:
        _add("Rozstrzygnięcie I instancji",
             _slownik_name(conn, before.get("rozstrzygniecie_i_id")),
             fields.get("rozstrzygniecieI"))

    if "rozstrzygniecieII" in fields or "rozstrzygniecieIIId" in fields:
        _add("Rozstrzygnięcie II instancji",
             _slownik_name(conn, before.get("rozstrzygniecie_ii_id")),
             fields.get("rozstrzygniecieII"))

    if persons_i_after is not None:
        _add("Osoby I instancji", persons_i_before, persons_i_after)

    if persons_ii_after is not None:
        _add("Osoby II instancji", persons_ii_before, persons_ii_after)

    for date_field, label, db_col in [
        ("dataWszczeciaPostepowaniaIInstancji", "Data wszczęcia postęp. I inst.", "data_wszczecia_postepowania_i_instancji"),
        ("dataDecyzjiIInstancji", "Data decyzji I inst.", "data_decyzji_i_instancji"),
        ("dataDoreczeniaDecyzjiIInstancji", "Data doręczenia decyzji I inst.", "data_doreczenia_decyzji_i_instancji"),
        ("dataWnioskuPonowneRozpatrzenie", "Data wniosku ponowne rozpatrzenie", "data_wniosku_ponowne_rozpatrzenie"),
        ("dataWplywuWnioskuPonowneRozpatrzenie", "Data wpływu wniosku", "data_wplywu_wniosku_ponowne_rozpatrzenie"),
        ("dataDecyzjiIIInstancji", "Data decyzji II inst.", "data_decyzji_ii_instancji"),
        ("dataDoreczeniaDecyzjiIIInstancji", "Data doręczenia decyzji II inst.", "data_doreczenia_decyzji_ii_instancji"),
    ]:
        if date_field in fields:
            _add(label, before.get(db_col), fields.get(date_field))

    if "komentarz" in fields:
        _add("Komentarz", before.get("komentarz"), fields.get("komentarz"))

    return changes


def build_risk_exposure_changes(
    conn: Any,
    before: dict[str, Any],
    fields: dict[str, Any],
    subjects_before: str | None,
    subjects_after: str | None,
    sanctions_before: str | None,
    sanctions_after: str | None,
    legal_bases_before: str | None,
    legal_bases_after: str | None,
    violations_before: str | None,
    violations_after: str | None,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    def _add(pole: str, przed: str | None, po: str | None) -> None:
        changes.append({"pole": pole, "przed": przed, "po": po})

    if "dataWniosku" in fields:
        _add("Data wniosku", before.get("data_wniosku"), fields.get("dataWniosku"))

    if "wniosekDo" in fields or "wniosekDoId" in fields:
        _add("Wniosek do",
             _slownik_name(conn, before.get("wniosek_do_id")),
             fields.get("wniosekDo") or _slownik_name(conn, fields.get("wniosekDoId")))

    if "czyMamyInformacjeOWszczeciuPostepowania" in fields or "czyMamyInformacjeOWszczeciuPostepowaniaId" in fields:
        _add("Informacja o wszczęciu postępowania",
             _slownik_name(conn, before.get("czy_mamy_informacje_o_wszczeciu_postepowania_id")),
             fields.get("czyMamyInformacjeOWszczeciuPostepowania") or _slownik_name(conn, fields.get("czyMamyInformacjeOWszczeciuPostepowaniaId")))

    if "rozstrzygniecie" in fields or "rozstrzygniecieId" in fields:
        _add("Rozstrzygnięcie",
             _slownik_name(conn, before.get("rozstrzygniecie_id")),
             fields.get("rozstrzygniecie") or _slownik_name(conn, fields.get("rozstrzygniecieId")))

    if subjects_after is not None:
        _add("Podmioty objęte sankcją", subjects_before, subjects_after)

    if sanctions_after is not None:
        _add("Sankcje", sanctions_before, sanctions_after)

    if legal_bases_after is not None:
        _add("Podstawy prawne", legal_bases_before, legal_bases_after)

    if violations_after is not None:
        _add("Naruszenia", violations_before, violations_after)

    if "komentarz" in fields:
        _add("Komentarz", before.get("komentarz"), fields.get("komentarz"))

    return changes


# ---------------------------------------------------------------------------
# Helper wewnętrzny – pobiera id ostatnio wstawionej pozycji słownikowej
# ---------------------------------------------------------------------------

def _latest_slownik_id(conn: Any, kod_typu: str, nazwa: str | None) -> int | None:
    if not nazwa:
        return None
    row = conn.execute(
        "SELECT id FROM slownik_pozycje WHERE lower(kod_typu)=lower(?) AND lower(nazwa_pozycji)=lower(?) LIMIT 1",
        (kod_typu, str(nazwa).strip()),
    ).fetchone()
    return int(row["id"]) if row else None
