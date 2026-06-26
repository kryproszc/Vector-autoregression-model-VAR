from __future__ import annotations

from datetime import date, datetime
import json
import os
import re
import sqlite3
import unicodedata

from app.auth_security import hash_password
from app.database.connection import get_connection


CREATE_INSPECTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lp INTEGER NOT NULL UNIQUE,
    kod_inspekcji TEXT UNIQUE,
    created_by_user_id INTEGER NOT NULL,
    nazwa_podmiotu_id INTEGER,
    typ_inspekcji_id INTEGER,
    zakres_inspekcji_id INTEGER,
    poczatek_inspekcji TEXT NOT NULL,
    koniec_inspekcji TEXT NOT NULL,
    osoba_kierujaca_user_id INTEGER,
    rynek_id INTEGER,
    rodzaj_podmiotu_id INTEGER,
    aspekt_konsumencki TEXT,
    komentarz TEXT,
    szczegoly_dotyczace_zakresu TEXT,
    status_inspekcji_id INTEGER,
    data_protokolu_sprawozdania TEXT,
    data_doreczenia_protokolu TEXT,
    data_akceptacji_sprawozdania TEXT,
    data_doreczenia_pisma TEXT,
    brak_data_doreczenia_pisma INTEGER NOT NULL DEFAULT 0,
    data_wyslania_pisma_z_zastrzezeniami TEXT,
    brak_data_wyslania_pisma_z_zastrzezeniami INTEGER NOT NULL DEFAULT 0,
    data_pisma_zastrzezenia TEXT,
    brak_data_pisma_zastrzezenia INTEGER NOT NULL DEFAULT 0,
    data_wplywu_pisma TEXT,
    brak_data_wplywu_pisma INTEGER NOT NULL DEFAULT 0,
    data_wyslania_pisma_z_odpowiedzia TEXT,
    brak_data_wyslania_pisma_z_odpowiedzia INTEGER NOT NULL DEFAULT 0,
    data_pisma_z_odpowiedzia TEXT,
    brak_data_pisma_z_odpowiedzia INTEGER NOT NULL DEFAULT 0,
    brak_dat_akceptacji_noty INTEGER NOT NULL DEFAULT 0,
    data_akceptacji_noty TEXT,
    data_zalecen TEXT,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (nazwa_podmiotu_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (typ_inspekcji_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (zakres_inspekcji_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (osoba_kierujaca_user_id) REFERENCES users(id),
    FOREIGN KEY (rynek_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (rodzaj_podmiotu_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (status_inspekcji_id) REFERENCES slownik_pozycje(id)
)
"""

CREATE_INSPECTION_MEMBERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inspection_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE (inspection_id, user_id)
)
"""

CREATE_INSPECTION_SCOPES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inspection_scopes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    scope_id INTEGER NOT NULL,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id) REFERENCES slownik_pozycje(id),
    UNIQUE (inspection_id, scope_id)
)
"""

CREATE_INSPECTION_TEAMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inspection_teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id),
    UNIQUE (inspection_id, team_id)
)
"""

CREATE_INDEX_INSPECTION_SCOPES_INSPECTION_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_scopes_inspection_id
ON inspection_scopes(inspection_id)
"""

CREATE_INDEX_INSPECTION_SCOPES_SCOPE_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_scopes_scope_id
ON inspection_scopes(scope_id)
"""

CREATE_INDEX_INSPECTION_TEAMS_INSPECTION_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_teams_inspection_id
ON inspection_teams(inspection_id)
"""

CREATE_INDEX_INSPECTION_TEAMS_TEAM_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_teams_team_id
ON inspection_teams(team_id)
"""

CREATE_INDEX_INSPECTION_MEMBERS_INSPECTION_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_members_inspection_id
ON inspection_members(inspection_id)
"""

CREATE_INDEX_INSPECTION_MEMBERS_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_members_user_id
ON inspection_members(user_id)
"""

CREATE_INDEX_INSPECTIONS_LP_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_inspections_lp
ON inspections(lp)
"""

CREATE_INDEX_INSPECTIONS_KOD_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_inspections_kod_inspekcji
ON inspections(kod_inspekcji)
"""

CREATE_INSPECTION_MULTI_DATES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inspection_multi_dates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    date_type TEXT NOT NULL CHECK (date_type IN ('ZALECENIE', 'AKCEPTACJA_NOTY')),
    date_value TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (inspection_id, date_type, date_value)
)
"""

CREATE_INDEX_MULTI_DATES_INSPECTION_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_multi_dates_inspection
ON inspection_multi_dates(inspection_id)
"""

CREATE_INDEX_MULTI_DATES_TYPE_VALUE_SQL = """
CREATE INDEX IF NOT EXISTS idx_inspection_multi_dates_type_value
ON inspection_multi_dates(date_type, date_value)
"""

CREATE_RECOMMENDATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER,
    pozycja INTEGER NOT NULL,
    kod_zalecenia TEXT UNIQUE,
    nazwa_podmiotu_id INTEGER,
    data_zalecen TEXT,
    brak_terminow_wykonania_zalecen INTEGER NOT NULL DEFAULT 0 CHECK (brak_terminow_wykonania_zalecen IN (0, 1)),
    brak_dat_akceptacji_noty_weryfikacji INTEGER NOT NULL DEFAULT 0 CHECK (brak_dat_akceptacji_noty_weryfikacji IN (0, 1)),
    status_zalecenia_id INTEGER,
    komentarz TEXT,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE,
    FOREIGN KEY (nazwa_podmiotu_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (status_zalecenia_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id)
)
"""

CREATE_INDEX_RECOMMENDATIONS_INSPECTION_SQL = """
CREATE INDEX IF NOT EXISTS idx_recommendations_inspection
ON recommendations(inspection_id)
"""

CREATE_INDEX_RECOMMENDATIONS_STATUS_SQL = """
CREATE INDEX IF NOT EXISTS idx_recommendations_status
ON recommendations(status_zalecenia_id)
"""

CREATE_INDEX_RECOMMENDATIONS_KOD_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_recommendations_kod
ON recommendations(kod_zalecenia)
"""

CREATE_RECOMMENDATION_MULTI_DATES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recommendation_multi_dates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER NOT NULL,
    date_type TEXT NOT NULL CHECK (date_type IN ('TERMIN_WYKONANIA_ZALECEN', 'AKCEPTACJA_NOTY_WERYFIKACJI')),
    date_value TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (recommendation_id, date_type, date_value)
)
"""

CREATE_RECOMMENDATION_TEAMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recommendation_teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES teams(id),
    UNIQUE (recommendation_id, team_id)
)
"""

CREATE_INDEX_RECOMMENDATION_MULTI_DATES_REC_SQL = """
CREATE INDEX IF NOT EXISTS idx_recommendation_multi_dates_rec
ON recommendation_multi_dates(recommendation_id)
"""

CREATE_INDEX_RECOMMENDATION_MULTI_DATES_TYPE_VALUE_SQL = """
CREATE INDEX IF NOT EXISTS idx_recommendation_multi_dates_type_value
ON recommendation_multi_dates(date_type, date_value)
"""

CREATE_INDEX_RECOMMENDATION_TEAMS_REC_SQL = """
CREATE INDEX IF NOT EXISTS idx_recommendation_teams_rec
ON recommendation_teams(recommendation_id)
"""

CREATE_INDEX_RECOMMENDATION_TEAMS_TEAM_SQL = """
CREATE INDEX IF NOT EXISTS idx_recommendation_teams_team
ON recommendation_teams(team_id)
"""

CREATE_OBLIGATING_DECISIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS obligating_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kod_decyzji TEXT,
    recommendation_kod_zalecenia TEXT,
    nazwa_podmiotu_id INTEGER,
    liczba_zalecen INTEGER,
    data_wszczecia_postepowania_i_instancji TEXT,
    data_decyzji_i_instancji TEXT,
    data_doreczenia_decyzji_i_instancji TEXT,
    rozstrzygniecie_i_id INTEGER,
    data_wniosku_ponowne_rozpatrzenie TEXT,
    data_wplywu_wniosku_ponowne_rozpatrzenie TEXT,
    data_decyzji_ii_instancji TEXT,
    data_doreczenia_decyzji_ii_instancji TEXT,
    rozstrzygniecie_ii_id INTEGER,
    komentarz TEXT,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (recommendation_kod_zalecenia) REFERENCES recommendations(kod_zalecenia) ON DELETE CASCADE,
    FOREIGN KEY (nazwa_podmiotu_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (rozstrzygniecie_i_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (rozstrzygniecie_ii_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id)
)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_KOD_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_obligating_decisions_kod_decyzji
ON obligating_decisions(kod_decyzji)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_RECOMMENDATION_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_recommendation_kod
ON obligating_decisions(recommendation_kod_zalecenia)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_PODMIOT_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_nazwa_podmiotu
ON obligating_decisions(nazwa_podmiotu_id)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_ROZSTRZYG_I_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_rozstrzygniecie_i
ON obligating_decisions(rozstrzygniecie_i_id)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_ROZSTRZYG_II_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_rozstrzygniecie_ii
ON obligating_decisions(rozstrzygniecie_ii_id)
"""

CREATE_OBLIGATING_DECISIONS_PERSONS_I_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS obligating_decisions_persons_i (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    obligating_decision_id INTEGER NOT NULL,
    slownik_pozycja_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (obligating_decision_id) REFERENCES obligating_decisions(id) ON DELETE CASCADE,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (obligating_decision_id, slownik_pozycja_id)
)
"""

CREATE_OBLIGATING_DECISIONS_PERSONS_II_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS obligating_decisions_persons_ii (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    obligating_decision_id INTEGER NOT NULL,
    slownik_pozycja_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (obligating_decision_id) REFERENCES obligating_decisions(id) ON DELETE CASCADE,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (obligating_decision_id, slownik_pozycja_id)
)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_I_DECISION_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_persons_i_decision
ON obligating_decisions_persons_i(obligating_decision_id)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_I_SLOWNIK_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_persons_i_slownik
ON obligating_decisions_persons_i(slownik_pozycja_id)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_II_DECISION_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_persons_ii_decision
ON obligating_decisions_persons_ii(obligating_decision_id)
"""

CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_II_SLOWNIK_SQL = """
CREATE INDEX IF NOT EXISTS idx_obligating_decisions_persons_ii_slownik
ON obligating_decisions_persons_ii(slownik_pozycja_id)
"""

CREATE_RISK_EXPOSURE_REQUESTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_exposure_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lp INTEGER NOT NULL UNIQUE,
    kod_sankcji TEXT UNIQUE,
    inspection_id INTEGER,
    nazwa_podmiotu_objetego_inspekcja_id INTEGER,
    data_wniosku TEXT,
    wniosek_do_id INTEGER,
    czy_mamy_informacje_o_wszczeciu_postepowania_id INTEGER,
    rozstrzygniecie_id INTEGER,
    komentarz TEXT,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    utworzono_przez_user_id INTEGER NOT NULL,
    zaktualizowano_przez_user_id INTEGER NOT NULL,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE SET NULL,
    FOREIGN KEY (nazwa_podmiotu_objetego_inspekcja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (wniosek_do_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (czy_mamy_informacje_o_wszczeciu_postepowania_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (rozstrzygniecie_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (utworzono_przez_user_id) REFERENCES users(id),
    FOREIGN KEY (zaktualizowano_przez_user_id) REFERENCES users(id)
)
"""

CREATE_INDEX_RISK_EXPOSURE_INSPECTION_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_inspection
ON risk_exposure_requests(inspection_id)
"""

CREATE_INDEX_RISK_EXPOSURE_KOD_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_risk_exposure_kod_sankcji
ON risk_exposure_requests(kod_sankcji)
"""

CREATE_RISK_EXPOSURE_SANCTION_SUBJECTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_exposure_sanction_subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    risk_exposure_id INTEGER NOT NULL,
    slownik_pozycja_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (risk_exposure_id) REFERENCES risk_exposure_requests(id) ON DELETE CASCADE,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (risk_exposure_id, slownik_pozycja_id)
)
"""

CREATE_RISK_EXPOSURE_SANCTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_exposure_sanctions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    risk_exposure_id INTEGER NOT NULL,
    slownik_pozycja_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (risk_exposure_id) REFERENCES risk_exposure_requests(id) ON DELETE CASCADE,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (risk_exposure_id, slownik_pozycja_id)
)
"""

CREATE_RISK_EXPOSURE_LEGAL_BASES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_exposure_legal_bases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    risk_exposure_id INTEGER NOT NULL,
    slownik_pozycja_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (risk_exposure_id) REFERENCES risk_exposure_requests(id) ON DELETE CASCADE,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (risk_exposure_id, slownik_pozycja_id)
)
"""

CREATE_RISK_EXPOSURE_VIOLATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS risk_exposure_violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    risk_exposure_id INTEGER NOT NULL,
    slownik_pozycja_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by_user_id INTEGER NOT NULL,
    updated_by_user_id INTEGER NOT NULL,
    FOREIGN KEY (risk_exposure_id) REFERENCES risk_exposure_requests(id) ON DELETE CASCADE,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
    UNIQUE (risk_exposure_id, slownik_pozycja_id)
)
"""

CREATE_INDEX_RISK_EXPOSURE_SANCTION_SUBJECTS_REQ_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_sanction_subjects_req
ON risk_exposure_sanction_subjects(risk_exposure_id)
"""

CREATE_INDEX_RISK_EXPOSURE_SANCTION_SUBJECTS_SLOWNIK_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_sanction_subjects_slownik
ON risk_exposure_sanction_subjects(slownik_pozycja_id)
"""

CREATE_INDEX_RISK_EXPOSURE_SANCTIONS_REQ_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_sanctions_req
ON risk_exposure_sanctions(risk_exposure_id)
"""

CREATE_INDEX_RISK_EXPOSURE_SANCTIONS_SLOWNIK_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_sanctions_slownik
ON risk_exposure_sanctions(slownik_pozycja_id)
"""

CREATE_INDEX_RISK_EXPOSURE_LEGAL_BASES_REQ_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_legal_bases_req
ON risk_exposure_legal_bases(risk_exposure_id)
"""

CREATE_INDEX_RISK_EXPOSURE_LEGAL_BASES_SLOWNIK_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_legal_bases_slownik
ON risk_exposure_legal_bases(slownik_pozycja_id)
"""

CREATE_INDEX_RISK_EXPOSURE_VIOLATIONS_REQ_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_violations_req
ON risk_exposure_violations(risk_exposure_id)
"""

CREATE_INDEX_RISK_EXPOSURE_VIOLATIONS_SLOWNIK_SQL = """
CREATE INDEX IF NOT EXISTS idx_risk_exposure_violations_slownik
ON risk_exposure_violations(slownik_pozycja_id)
"""

CREATE_TEAMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kod TEXT NOT NULL UNIQUE,
    nazwa TEXT NOT NULL,
    slownik_pozycja_id INTEGER,
    kierownik_user_id INTEGER,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id),
    FOREIGN KEY (kierownik_user_id) REFERENCES users(id)
)
"""

CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    login TEXT NOT NULL UNIQUE,
    imie TEXT NOT NULL,
    nazwisko TEXT NOT NULL,
    email TEXT,
    password_hash TEXT,
    rola_id INTEGER NOT NULL,
    zespol_id INTEGER,
    created_by_user_id INTEGER,
    aktywny INTEGER NOT NULL DEFAULT 1,
    account_type TEXT NOT NULL DEFAULT 'diu' CHECK (account_type IN ('diu', 'observer', 'technical')),
    department_code TEXT,
    list_visibility TEXT NOT NULL DEFAULT 'visible' CHECK (list_visibility IN ('visible', 'hidden')),
    profile_changed_at TEXT,
    profile_changed_by_user_id INTEGER,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (zespol_id) REFERENCES teams(id),
    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
    FOREIGN KEY (profile_changed_by_user_id) REFERENCES users(id)
)
"""

CREATE_USER_PROFILE_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_profile_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    login TEXT,
    imie TEXT,
    nazwisko TEXT,
    email TEXT,
    rola_id INTEGER,
    aktywny INTEGER,
    account_type TEXT NOT NULL CHECK (account_type IN ('diu', 'observer', 'technical')),
    zespol_id INTEGER,
    department_code TEXT,
    department_label TEXT,
    list_visibility TEXT NOT NULL CHECK (list_visibility IN ('visible', 'hidden')),
    permissions_codes TEXT,
    changed_by_user_id INTEGER,
    changed_by_login TEXT,
    changed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (changed_by_user_id) REFERENCES users(id)
)
"""

CREATE_INDEX_USER_PROFILE_HISTORY_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_user_profile_history_user_id
ON user_profile_history(user_id)
"""

CREATE_INDEX_USER_PROFILE_HISTORY_ACTIVE_SQL = """
CREATE INDEX IF NOT EXISTS idx_user_profile_history_user_active
ON user_profile_history(user_id, valid_to)
"""

CREATE_AUTH_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""

CREATE_INDEX_AUTH_SESSIONS_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
ON auth_sessions(user_id)
"""

CREATE_USER_INVITES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_invites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    created_by_user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by_user_id) REFERENCES users(id)
)
"""

CREATE_INDEX_USER_INVITES_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_user_invites_user_id
ON user_invites(user_id)
"""

CREATE_USER_PASSWORD_RESETS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_password_resets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""

CREATE_INDEX_USER_PASSWORD_RESETS_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_user_password_resets_user_id
ON user_password_resets(user_id)
"""

CREATE_AUTH_LOGIN_ATTEMPTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth_login_attempts (
    login_key TEXT PRIMARY KEY,
    failed_count INTEGER NOT NULL DEFAULT 0,
    first_failed_at TEXT,
    locked_until TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_AUTH_PASSWORD_RESET_THROTTLE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth_password_reset_throttle (
    request_key TEXT PRIMARY KEY,
    last_requested_at TEXT NOT NULL
)
"""

CREATE_NOTIFICATION_SCHEDULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS notification_schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    module_type TEXT NOT NULL CHECK (module_type IN ('inspections', 'recommendations', 'risk_exposure')),
    date_field_a TEXT NOT NULL,
    date_field_b TEXT NOT NULL,
    days_difference INTEGER NOT NULL DEFAULT 0,
    subject_template TEXT NOT NULL DEFAULT '',
    body_template TEXT NOT NULL DEFAULT '',
    send_hour INTEGER NOT NULL CHECK (send_hour >= 0 AND send_hour <= 23),
    send_minute INTEGER NOT NULL DEFAULT 0 CHECK (send_minute >= 0 AND send_minute <= 59),
    target_inspection_leader INTEGER NOT NULL DEFAULT 1,
    target_inspection_team INTEGER NOT NULL DEFAULT 0,
    fallback_recipient TEXT NOT NULL DEFAULT 'author' CHECK (fallback_recipient IN ('author')),
    enabled INTEGER NOT NULL DEFAULT 1,
    created_by_user_id INTEGER NOT NULL,
    last_run_date TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by_user_id) REFERENCES users(id)
)
"""

CREATE_NOTIFICATION_SCHEDULE_RULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS notification_schedule_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id INTEGER NOT NULL,
    days_difference INTEGER NOT NULL,
    subject_template TEXT NOT NULL,
    body_template TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (schedule_id, days_difference),
    FOREIGN KEY (schedule_id) REFERENCES notification_schedules(id) ON DELETE CASCADE
)
"""

CREATE_NOTIFICATION_SCHEDULE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS notification_schedule_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schedule_id INTEGER NOT NULL,
    trigger_type TEXT NOT NULL CHECK (trigger_type IN ('auto', 'manual')),
    status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'partial', 'failed')),
    matched_count INTEGER NOT NULL DEFAULT 0,
    sent_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (schedule_id) REFERENCES notification_schedules(id) ON DELETE CASCADE
)
"""

CREATE_NOTIFICATION_SCHEDULE_DISPATCHES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS notification_schedule_dispatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    schedule_id INTEGER NOT NULL,
    rule_id INTEGER NOT NULL,
    module_type TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    recipient_email TEXT NOT NULL,
    recipient_type TEXT,
    status TEXT NOT NULL CHECK (status IN ('sent', 'failed')),
    error_message TEXT,
    rendered_subject TEXT,
    rendered_body TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (schedule_id) REFERENCES notification_schedules(id) ON DELETE CASCADE,
    FOREIGN KEY (rule_id) REFERENCES notification_schedule_rules(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES notification_schedule_runs(id) ON DELETE SET NULL
)
"""

CREATE_INDEX_NOTIFICATION_SCHEDULE_RULES_SCHEDULE_SQL = """
CREATE INDEX IF NOT EXISTS idx_notification_schedule_rules_schedule_id
ON notification_schedule_rules(schedule_id)
"""

CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_SCHEDULE_SQL = """
CREATE INDEX IF NOT EXISTS idx_notification_schedule_dispatches_schedule_id
ON notification_schedule_dispatches(schedule_id)
"""

CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_CREATED_SQL = """
CREATE INDEX IF NOT EXISTS idx_notification_schedule_dispatches_created_at
ON notification_schedule_dispatches(created_at)
"""

CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_RUN_SQL = """
CREATE INDEX IF NOT EXISTS idx_notification_schedule_dispatches_run_id
ON notification_schedule_dispatches(run_id)
"""

CREATE_INDEX_NOTIFICATION_SCHEDULE_RUNS_SCHEDULE_SQL = """
CREATE INDEX IF NOT EXISTS idx_notification_schedule_runs_schedule_id
ON notification_schedule_runs(schedule_id)
"""

CREATE_INDEX_NOTIFICATION_SCHEDULE_RUNS_CREATED_SQL = """
CREATE INDEX IF NOT EXISTS idx_notification_schedule_runs_created_at
ON notification_schedule_runs(created_at)
"""

CREATE_AUTH_PERMISSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth_permissions (
    code TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    permission_group TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_USER_PERMISSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_permissions (
    user_id INTEGER NOT NULL,
    permission_code TEXT NOT NULL,
    granted_by_user_id INTEGER NOT NULL,
    granted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, permission_code),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (permission_code) REFERENCES auth_permissions(code) ON DELETE CASCADE,
    FOREIGN KEY (granted_by_user_id) REFERENCES users(id)
)
"""

CREATE_INDEX_USER_PERMISSIONS_USER_SQL = """
CREATE INDEX IF NOT EXISTS idx_user_permissions_user_id
ON user_permissions(user_id)
"""

CREATE_RECORD_EDIT_LOCKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS record_edit_locks (
    module_name TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    lock_token TEXT NOT NULL,
    owner_user_id INTEGER NOT NULL,
    owner_login TEXT NOT NULL,
    owner_display_name TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY (module_name, record_id),
    FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""

CREATE_INDEX_RECORD_EDIT_LOCKS_OWNER_SQL = """
CREATE INDEX IF NOT EXISTS idx_record_edit_locks_owner
ON record_edit_locks(owner_user_id)
"""

CREATE_INDEX_RECORD_EDIT_LOCKS_EXPIRES_SQL = """
CREATE INDEX IF NOT EXISTS idx_record_edit_locks_expires
ON record_edit_locks(expires_at)
"""

SEED_AUTH_PERMISSIONS_SQL = """
INSERT OR IGNORE INTO auth_permissions (code, label, permission_group, active) VALUES
    ('registry.inspections.read', 'Inspekcje', 'rejestry', 1),
    ('registry.recommendations.read', 'Zalecenia', 'rejestry', 1),
    ('registry.obligating_decisions.read', 'Decyzje zobowiazujace', 'rejestry', 1),
    ('registry.risk_exposure.read', 'Wnioski sankcyjne', 'rejestry', 1),
    ('reports.executed_inspections.read', 'Wykonane inspekcje', 'raporty', 1),
    ('reports.protocol_time.read', 'Czas protokolu', 'raporty', 1),
    ('reports.report_time.read', 'Czas sprawozdania', 'raporty', 1),
    ('management.dictionaries.read', 'Slowniki - odczyt', 'zarzadzanie', 1),
    ('management.dictionaries.write', 'Slowniki - edycja', 'zarzadzanie', 1)
"""

CREATE_SLOWNIK_TYPY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slownik_typy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kod_typu TEXT NOT NULL UNIQUE,
    nazwa_typu TEXT NOT NULL,
    kategoria INTEGER NOT NULL DEFAULT 0,
    opis TEXT,
    aktywny INTEGER NOT NULL DEFAULT 1,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_SLOWNIK_POZYCJE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slownik_pozycje (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kod_typu TEXT NOT NULL,
    kod_pozycji TEXT NOT NULL,
    nazwa_pozycji TEXT NOT NULL,
    nazwa_uzytkowa TEXT,
    skrot_pozycji TEXT,
    kolejnosc INTEGER,
    aktywny INTEGER NOT NULL DEFAULT 1,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (kod_typu) REFERENCES slownik_typy(kod_typu),
    UNIQUE (kod_typu, kod_pozycji),
    CHECK (kod_pozycji = UPPER(kod_pozycji) AND INSTR(kod_pozycji, ' ') = 0)
)
"""

CREATE_INDEX_SLOWNIK_POZYCJE_KOD_TYPU_SQL = """
CREATE INDEX IF NOT EXISTS idx_slownik_pozycje_kod_typu
ON slownik_pozycje(kod_typu)
"""

CREATE_SLOWNIK_STATUS_INSPEKCJI_STYL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS slownik_status_inspekcji_styl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slownik_pozycja_id INTEGER NOT NULL UNIQUE,
    kolor TEXT NOT NULL,
    odcien INTEGER NOT NULL,
    intensywnosc INTEGER NOT NULL,
    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    utworzono_przez TEXT,
    zaktualizowano_przez TEXT,
    FOREIGN KEY (slownik_pozycja_id) REFERENCES slownik_pozycje(id) ON DELETE CASCADE,
    CHECK (odcien IN (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)),
    CHECK (intensywnosc >= 0 AND intensywnosc <= 100),
    CHECK (lower(kolor) IN (
        'emerald', 'green', 'teal', 'lime',
        'sky', 'cyan', 'blue', 'indigo',
        'rose', 'red', 'pink', 'fuchsia',
        'yellow', 'amber', 'orange'
    ))
)
"""

CREATE_INDEX_SLOWNIK_STATUS_INSPEKCJI_STYL_POZYCJA_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_slownik_status_inspekcji_styl_pozycja
ON slownik_status_inspekcji_styl(slownik_pozycja_id)
"""

SEED_SLOWNIK_STATUS_INSPEKCJI_STYL_SQL = """
INSERT INTO slownik_status_inspekcji_styl
    (slownik_pozycja_id, kolor, odcien, intensywnosc, utworzono_przez, zaktualizowano_przez)
SELECT
    p.id,
    CASE upper(p.kod_pozycji)
        WHEN 'I_SI_2' THEN 'green'
        WHEN 'I_SI_3' THEN 'cyan'
        WHEN 'I_SI_4' THEN 'rose'
        WHEN 'I_SI_5' THEN 'yellow'
        WHEN 'I_SI_6' THEN 'yellow'
        WHEN 'I_SI_7' THEN 'yellow'
        WHEN 'I_SI_8' THEN 'yellow'
        WHEN 'I_SI_9' THEN 'yellow'
        ELSE NULL
    END AS kolor,
    CASE upper(p.kod_pozycji)
        WHEN 'I_SI_2' THEN 300
        WHEN 'I_SI_3' THEN 400
        WHEN 'I_SI_4' THEN 200
        WHEN 'I_SI_5' THEN 200
        WHEN 'I_SI_6' THEN 200
        WHEN 'I_SI_7' THEN 200
        WHEN 'I_SI_8' THEN 200
        WHEN 'I_SI_9' THEN 200
        ELSE NULL
    END AS odcien,
    75,
    'system-seed',
    'system-seed'
FROM slownik_pozycje p
WHERE lower(p.kod_typu) = 'statusy_inspekcji'
  AND upper(p.kod_pozycji) IN ('I_SI_2', 'I_SI_3', 'I_SI_4', 'I_SI_5', 'I_SI_6', 'I_SI_7', 'I_SI_8', 'I_SI_9')
ON CONFLICT(slownik_pozycja_id)
DO UPDATE SET
    kolor = excluded.kolor,
    odcien = excluded.odcien,
    intensywnosc = excluded.intensywnosc,
    zaktualizowano_o = CURRENT_TIMESTAMP,
    zaktualizowano_przez = excluded.zaktualizowano_przez
"""

SEED_TEAMS_SQL = """
INSERT OR IGNORE INTO teams (kod, nazwa) VALUES
    ('A', 'Zespol A'),
    ('B', 'Zespol B'),
    ('C', 'Zespol C'),
    ('D', 'Zespol D')
"""

SEED_SLOWNIK_TYPY_SQL = """
INSERT OR IGNORE INTO slownik_typy (kod_typu, nazwa_typu, kategoria, opis, aktywny) VALUES
    ('zespoly', 'Zespoly', 0, 'Slownik nazw zespolow', 1),
    ('nazwy_podmiotow', 'Nazwy podmiotow', 0, 'Lista nazw podmiotow', 1),
    ('typy_inspekcji', 'Typy inspekcji', 0, 'Lista typow inspekcji', 1),
    ('zakresy_inspekcji', 'Zakresy inspekcji', 0, 'Lista zakresow inspekcji', 1),
    ('osoby', 'Osoby', 0, 'Lista osob do wyboru', 1),
    ('rynki', 'Rynki', 0, 'Lista rynkow', 1),
    ('rodzaje_podmiotu', 'Rodzaje podmiotu', 0, 'Lista rodzajow podmiotu', 1),
    ('statusy_inspekcji', 'Statusy inspekcji', 0, 'Lista statusow inspekcji', 1),
    ('statusy_zalecen', 'Statusy zalecen', 1, 'Lista statusow zalecen', 1),
    ('department', 'Department', 2, 'Slownik department', 1),
    ('department_ogolne', 'Departament', 3, 'Slownik departamentu w kategorii ogolnej', 1),
    ('sankcja', 'Sankcja', 2, 'Slownik sankcji', 1),
    ('podstawa_prawna_sankcji', 'Podstawa prawna sankcji', 2, 'Slownik podstaw prawnych sankcji', 1),
    ('naruszenia_skutkujace_sankcja', 'Naruszenia skutkujace sankcja', 2, 'Slownik naruszen skutkujacych sankcja', 1),
    ('informacja_o_wszczeciu_postepowania_sankcyjnego', 'Informacja o wszczeciu postepowania sankcyjnego', 2, 'Slownik informacji o wszczeciu postepowania sankcyjnego', 1),
    ('rozstrzygniecie_decyzji_i', 'Rozstrzygniecie decyzji I', 2, 'Slownik rozstrzygniec decyzji I', 1),
    ('rozstrzygniecie_decyzji_ii', 'Rozstrzygniecie decyzji II', 2, 'Slownik rozstrzygniec decyzji II', 1),
    (
        'rozstrzygniecie_wniosku_sankcyjnego_I',
        'Rozstrzygniecie wniosku sankcyjnego I',
        2,
        'Slownik rozstrzygniec wniosku sankcyjnego I',
        1
    ),
    ('nazwy_podmiotow_sankcje', 'Nazwy podmiotow_sankcje', 2, 'Slownik nazw podmiotow dla sankcji', 1)
"""

SEED_SLOWNIK_ZESPOLY_SQL = """
INSERT OR IGNORE INTO slownik_pozycje
(kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
VALUES
    ('zespoly', 'A', 'Zespol A', 'A', 1, 1),
    ('zespoly', 'B', 'Zespol B', 'B', 2, 1),
    ('zespoly', 'C', 'Zespol C', 'C', 3, 1),
    ('zespoly', 'D', 'Zespol D', 'D', 4, 1)
"""

SEED_SLOWNIK_DEPARTMENT_OGOLNE_SQL = """
INSERT OR IGNORE INTO slownik_pozycje
(kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
VALUES
    ('department_ogolne', 'DIU', 'DIU', 'DIU', 1, 1)
"""

SEED_SLOWNIK_TYPY_INSPEKCJI_SQL = """
INSERT OR IGNORE INTO slownik_pozycje
(kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
VALUES
    ('typy_inspekcji', 'KONTROLA', 'Kontrola', 'K', 1, 1),
    ('typy_inspekcji', 'WIZYTA_NADZORCZA', 'Wizyta nadzorcza', 'W', 2, 1)
"""

SEED_SLOWNIK_ZAKRESY_INSPEKCJI_SQL = """
INSERT OR IGNORE INTO slownik_pozycje
(kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
VALUES
    ('zakresy_inspekcji', 'RTU', 'RTU', 'RTU', 1, 1),
    ('zakresy_inspekcji', 'SCR', 'SCR', 'SCR', 2, 1),
    ('zakresy_inspekcji', 'BEZPIECZENSTWO_SYSTEMOW', 'Bezpieczenstwo systemow', 'BDS', 3, 1)
"""

SEED_SLOWNIK_DECYZJE_ZOBOWIAZUJACE_SQL = """
INSERT OR IGNORE INTO slownik_pozycje
(kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
VALUES
    ('rozstrzygniecie_decyzji_i', 'UTRZYMANO', 'Utrzymano', 'UTRZYMANO', 1, 1),
    ('rozstrzygniecie_decyzji_i', 'ZMIENIONO', 'Zmieniono', 'ZMIENIONO', 2, 1),
    ('rozstrzygniecie_decyzji_i', 'UCHYLONO', 'Uchylono', 'UCHYLONO', 3, 1),
    ('rozstrzygniecie_decyzji_ii', 'UTRZYMANO', 'Utrzymano', 'UTRZYMANO', 1, 1),
    ('rozstrzygniecie_decyzji_ii', 'ZMIENIONO', 'Zmieniono', 'ZMIENIONO', 2, 1),
    ('rozstrzygniecie_decyzji_ii', 'UCHYLONO', 'Uchylono', 'UCHYLONO', 3, 1)
"""

SYNC_SLOWNIK_OSOBY_FROM_USERS_SQL = """
INSERT OR IGNORE INTO slownik_pozycje
(kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
SELECT
    'osoby' AS kod_typu,
    'OSOBA_' || CAST(u.id AS TEXT) AS kod_pozycji,
    CASE
        WHEN trim(COALESCE(u.imie, '') || ' ' || COALESCE(u.nazwisko, '')) = ''
            THEN u.login
        ELSE trim(COALESCE(u.imie, '') || ' ' || COALESCE(u.nazwisko, ''))
    END AS nazwa_pozycji,
    u.login AS skrot_pozycji,
    u.id AS kolejnosc,
    1 AS aktywny
FROM users u
WHERE u.aktywny = 1
    AND u.rola_id <> 4
"""

SEED_SLOWNIK_ROZSTRZYGNIECIA_WNIOSKU_SQL = """
INSERT OR IGNORE INTO slownik_pozycje
(kod_typu, kod_pozycji, nazwa_pozycji, skrot_pozycji, kolejnosc, aktywny)
VALUES
    (
        'rozstrzygniecie_wniosku_sankcyjnego_I',
        'I',
        'rozstrzygniecie_wniosku_sankcyjnego_I',
        'I',
        1,
        1
    )
"""

MIGRATE_SLOWNIK_ROZSTRZYGNIECIA_I_SQL = """
UPDATE slownik_pozycje
SET
    kod_typu = 'rozstrzygniecie_wniosku_sankcyjnego_I',
        kod_pozycji = 'I',
    nazwa_pozycji = 'rozstrzygniecie_wniosku_sankcyjnego_I',
        skrot_pozycji = 'I',
        kolejnosc = 1,
        aktywny = 1,
    zaktualizowano_o = CURRENT_TIMESTAMP
WHERE lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego'
    AND lower(kod_pozycji) IN ('i', 'rozstrzygniecie_i', 'rozstrzygniecie_wniosku_sankcyjnego_i')
"""

MIGRATE_SLOWNIK_ROZSTRZYGNIECIA_I_FROM_I_SQL = """
UPDATE slownik_pozycje
SET
        kod_typu = 'rozstrzygniecie_wniosku_sankcyjnego_I',
        nazwa_pozycji = 'rozstrzygniecie_wniosku_sankcyjnego_I',
        skrot_pozycji = 'I',
        kolejnosc = 1,
        aktywny = 1,
        zaktualizowano_o = CURRENT_TIMESTAMP
WHERE lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego_i'
    AND lower(kod_pozycji) = 'i'
"""

NULLIFY_RISK_EXPOSURE_ROZSTRZYGNIECIA_II_SQL = """
UPDATE risk_exposure_requests
SET rozstrzygniecie_id = NULL,
    zaktualizowano_o = CURRENT_TIMESTAMP
WHERE rozstrzygniecie_id IN (
    SELECT id
    FROM slownik_pozycje
    WHERE lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego_ii'
)
"""

DELETE_SLOWNIK_ROZSTRZYGNIECIA_II_SQL = """
DELETE FROM slownik_pozycje
WHERE lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego_ii'
   OR (
       lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego'
       AND lower(kod_pozycji) IN ('ii', 'rozstrzygniecie_ii', 'rozstrzygniecie_wniosku_sankcyjnego_ii')
   )
"""

DELETE_SLOWNIK_ROZSTRZYGNIECIA_II_TYP_SQL = """
DELETE FROM slownik_typy
WHERE lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego_ii'
"""

DELETE_LEGACY_SLOWNIK_ROZSTRZYGNIECIA_TYP_SQL = """
DELETE FROM slownik_typy
WHERE lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego'
    AND NOT EXISTS (
                SELECT 1
                FROM slownik_pozycje
                WHERE lower(kod_typu) = 'rozstrzygniecie_wniosku_sankcyjnego'
    )
"""

SEED_USERS_SQL = """
INSERT OR IGNORE INTO users (login, imie, nazwisko, email, rola_id, zespol_id, aktywny) VALUES
    ('admin', 'Admin', 'Systemu', 'admin@rejestr.local', 3, NULL, 1)
"""

UPDATE_TEAM_LEADS_SQL = """
UPDATE teams
SET kierownik_user_id = (
    SELECT u.id FROM users u
    WHERE u.rola_id = 2 AND u.zespol_id = teams.id
    ORDER BY u.id ASC
    LIMIT 1
),
zaktualizowano_o = CURRENT_TIMESTAMP
"""

UPDATE_TEAMS_SLOWNIK_LINK_SQL = """
UPDATE teams
SET slownik_pozycja_id = (
        SELECT sp.id
        FROM slownik_pozycje sp
        WHERE sp.kod_typu = 'zespoly'
            AND (sp.kod_pozycji = teams.kod OR sp.skrot_pozycji = teams.kod)
        LIMIT 1
)
WHERE slownik_pozycja_id IS NULL
"""


def _migrate_inspection_team_ids_column_to_relation(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()}
    if "inspection_team_ids" not in columns:
        return

    inspection_rows = conn.execute(
        "SELECT id, inspection_team_ids FROM inspections"
    ).fetchall()
    for row in inspection_rows:
        inspection_id = int(row["id"])
        existing_rel = conn.execute(
            "SELECT 1 FROM inspection_teams WHERE inspection_id = ? LIMIT 1",
            (inspection_id,),
        ).fetchone()
        if existing_rel is not None:
            continue

        raw_value = row["inspection_team_ids"]
        if raw_value is None:
            continue
        cleaned = str(raw_value).strip()
        if not cleaned:
            continue
        try:
            loaded = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if not isinstance(loaded, list):
            continue

        normalized_ids: list[int] = []
        for item in loaded:
            try:
                normalized_ids.append(int(item))
            except (TypeError, ValueError):
                continue

        for team_id in sorted(set(normalized_ids)):
            team_exists = conn.execute(
                "SELECT 1 FROM teams WHERE id = ? LIMIT 1",
                (team_id,),
            ).fetchone()
            if team_exists is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO inspection_teams (inspection_id, team_id) VALUES (?, ?)",
                (inspection_id, team_id),
            )


def _ensure_inspections_schema(conn: sqlite3.Connection) -> None:
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='inspections'"
    ).fetchone()

    if has_table is not None:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
        }

        if "created_by_user_id" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN created_by_user_id INTEGER")
            conn.execute(
                """
                UPDATE inspections
                SET created_by_user_id = COALESCE(
                    created_by_user_id,
                    osoba_kierujaca_user_id,
                    (SELECT id FROM users ORDER BY id ASC LIMIT 1)
                )
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "lp" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN lp INTEGER")
            conn.execute(
                """
                UPDATE inspections
                SET lp = id
                WHERE lp IS NULL
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "kod_inspekcji" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN kod_inspekcji TEXT")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "aspekt_konsumencki" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN aspekt_konsumencki TEXT")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "szczegoly_dotyczace_zakresu" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN szczegoly_dotyczace_zakresu TEXT")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "data_wyslania_pisma_z_zastrzezeniami" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN data_wyslania_pisma_z_zastrzezeniami TEXT")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "data_wyslania_pisma_z_odpowiedzia" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN data_wyslania_pisma_z_odpowiedzia TEXT")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "brak_data_doreczenia_pisma" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN brak_data_doreczenia_pisma INTEGER NOT NULL DEFAULT 0")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "brak_data_wyslania_pisma_z_zastrzezeniami" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN brak_data_wyslania_pisma_z_zastrzezeniami INTEGER NOT NULL DEFAULT 0")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "brak_data_pisma_zastrzezenia" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN brak_data_pisma_zastrzezenia INTEGER NOT NULL DEFAULT 0")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "brak_data_wplywu_pisma" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN brak_data_wplywu_pisma INTEGER NOT NULL DEFAULT 0")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "brak_data_wyslania_pisma_z_odpowiedzia" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN brak_data_wyslania_pisma_z_odpowiedzia INTEGER NOT NULL DEFAULT 0")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        conn.execute(
            """
            UPDATE inspections
            SET
                brak_data_wyslania_pisma_z_odpowiedzia = 1,
                data_wyslania_pisma_z_odpowiedzia = NULL
            WHERE lower(trim(COALESCE(data_wyslania_pisma_z_odpowiedzia, ''))) = 'brak pisma'
            """
        )

        if "brak_data_pisma_z_odpowiedzia" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN brak_data_pisma_z_odpowiedzia INTEGER NOT NULL DEFAULT 0")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "brak_dat_akceptacji_noty" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN brak_dat_akceptacji_noty INTEGER NOT NULL DEFAULT 0")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "osoba_kierujaca_tekst" in columns:
            conn.execute("ALTER TABLE inspections DROP COLUMN osoba_kierujaca_tekst")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        if "sklad_zespolu_tekst" in columns:
            conn.execute("ALTER TABLE inspections DROP COLUMN sklad_zespolu_tekst")
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()
            }

        required = {
            "lp",
            "created_by_user_id",
            "nazwa_podmiotu_id",
            "typ_inspekcji_id",
            "zakres_inspekcji_id",
            "osoba_kierujaca_user_id",
            "rynek_id",
            "rodzaj_podmiotu_id",
            "status_inspekcji_id",
        }

        if not required.issubset(columns):
            conn.execute("DROP TABLE IF EXISTS inspection_members")
            conn.execute("DROP TABLE IF EXISTS inspection_teams")
            conn.execute("DROP TABLE inspections")

    conn.execute(CREATE_INSPECTIONS_TABLE_SQL)
    conn.execute(CREATE_INSPECTION_MEMBERS_TABLE_SQL)
    conn.execute(CREATE_INSPECTION_SCOPES_TABLE_SQL)
    conn.execute(CREATE_INSPECTION_TEAMS_TABLE_SQL)
    conn.execute(CREATE_INDEX_INSPECTION_MEMBERS_INSPECTION_SQL)
    conn.execute(CREATE_INDEX_INSPECTION_MEMBERS_USER_SQL)
    conn.execute(CREATE_INDEX_INSPECTION_SCOPES_INSPECTION_SQL)
    conn.execute(CREATE_INDEX_INSPECTION_SCOPES_SCOPE_SQL)
    conn.execute(CREATE_INDEX_INSPECTION_TEAMS_INSPECTION_SQL)
    conn.execute(CREATE_INDEX_INSPECTION_TEAMS_TEAM_SQL)
    conn.execute(CREATE_INDEX_INSPECTIONS_LP_SQL)
    conn.execute(CREATE_INDEX_INSPECTIONS_KOD_SQL)

    # Migrate legacy JSON column values to relation table and then remove the legacy column.
    _migrate_inspection_team_ids_column_to_relation(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()}
    if "inspection_team_ids" in columns:
        conn.execute("ALTER TABLE inspections DROP COLUMN inspection_team_ids")

    conn.execute(CREATE_INSPECTION_MULTI_DATES_TABLE_SQL)
    conn.execute(CREATE_INDEX_MULTI_DATES_INSPECTION_SQL)
    conn.execute(CREATE_INDEX_MULTI_DATES_TYPE_VALUE_SQL)


def _ensure_teams_schema(conn: sqlite3.Connection) -> None:
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='teams'"
    ).fetchone()

    if has_table is not None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(teams)").fetchall()}
        required = {"kod", "nazwa", "kierownik_user_id"}
        if not required.issubset(columns):
            conn.execute("DROP TABLE teams")
        elif "slownik_pozycja_id" not in columns:
            conn.execute("ALTER TABLE teams ADD COLUMN slownik_pozycja_id INTEGER")

    conn.execute(CREATE_TEAMS_TABLE_SQL)


def _ensure_users_schema(conn: sqlite3.Connection) -> None:
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()

    if has_table is not None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        required = {"imie", "nazwisko", "rola_id", "zespol_id"}
        if not required.issubset(columns):
            conn.execute("DROP TABLE users")

    conn.execute(CREATE_USERS_TABLE_SQL)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "password_hash" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "account_type" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN account_type TEXT NOT NULL DEFAULT 'diu'")
    if "department_code" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN department_code TEXT")
    if "created_by_user_id" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN created_by_user_id INTEGER")
        conn.execute(
            """
            UPDATE users
            SET created_by_user_id = COALESCE(
                profile_changed_by_user_id,
                (SELECT id FROM users WHERE rola_id = 3 ORDER BY id ASC LIMIT 1),
                id
            )
            WHERE created_by_user_id IS NULL
            """
        )
    if "list_visibility" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN list_visibility TEXT NOT NULL DEFAULT 'visible'")
    if "profile_changed_at" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN profile_changed_at TEXT")
    if "profile_changed_by_user_id" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN profile_changed_by_user_id INTEGER")


def _ensure_user_profile_history_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_USER_PROFILE_HISTORY_TABLE_SQL)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(user_profile_history)").fetchall()}
    if "zespol_id" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN zespol_id INTEGER")
    if "login" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN login TEXT")
    if "imie" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN imie TEXT")
    if "nazwisko" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN nazwisko TEXT")
    if "email" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN email TEXT")
    if "rola_id" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN rola_id INTEGER")
    if "aktywny" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN aktywny INTEGER")
    if "permissions_codes" not in columns:
        conn.execute("ALTER TABLE user_profile_history ADD COLUMN permissions_codes TEXT")
    conn.execute(CREATE_INDEX_USER_PROFILE_HISTORY_USER_SQL)
    conn.execute(CREATE_INDEX_USER_PROFILE_HISTORY_ACTIVE_SQL)


def _ensure_auth_sessions_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_AUTH_SESSIONS_TABLE_SQL)
    conn.execute(CREATE_INDEX_AUTH_SESSIONS_USER_SQL)


def _ensure_user_invites_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_USER_INVITES_TABLE_SQL)
    conn.execute(CREATE_INDEX_USER_INVITES_USER_SQL)


def _ensure_user_password_resets_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_USER_PASSWORD_RESETS_TABLE_SQL)
    conn.execute(CREATE_INDEX_USER_PASSWORD_RESETS_USER_SQL)


def _ensure_auth_login_attempts_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_AUTH_LOGIN_ATTEMPTS_TABLE_SQL)


def _ensure_auth_password_reset_throttle_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_AUTH_PASSWORD_RESET_THROTTLE_TABLE_SQL)


def _ensure_notification_schedules_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_NOTIFICATION_SCHEDULES_TABLE_SQL)
    conn.execute(CREATE_NOTIFICATION_SCHEDULE_RULES_TABLE_SQL)
    conn.execute(CREATE_NOTIFICATION_SCHEDULE_RUNS_TABLE_SQL)
    conn.execute(CREATE_NOTIFICATION_SCHEDULE_DISPATCHES_TABLE_SQL)
    conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_RULES_SCHEDULE_SQL)
    conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_RUNS_SCHEDULE_SQL)
    conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_RUNS_CREATED_SQL)
    conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_SCHEDULE_SQL)
    conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_CREATED_SQL)

    dispatch_table_sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='notification_schedule_dispatches'"
    ).fetchone()
    dispatch_table_sql = str(dispatch_table_sql_row[0] or "") if dispatch_table_sql_row is not None else ""
    if "UNIQUE (schedule_id, rule_id, module_type, record_id, recipient_email)" in dispatch_table_sql:
        conn.execute("ALTER TABLE notification_schedule_dispatches RENAME TO notification_schedule_dispatches_old")
        conn.execute(CREATE_NOTIFICATION_SCHEDULE_DISPATCHES_TABLE_SQL)
        conn.execute(
            """
            INSERT INTO notification_schedule_dispatches (
                id, schedule_id, rule_id, module_type, record_id,
                recipient_email, recipient_type, status, error_message,
                rendered_subject, rendered_body, created_at
            )
            SELECT
                id, schedule_id, rule_id, module_type, record_id,
                recipient_email, recipient_type, status, error_message,
                rendered_subject, rendered_body, created_at
            FROM notification_schedule_dispatches_old
            """
        )
        conn.execute("DROP TABLE notification_schedule_dispatches_old")
        conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_SCHEDULE_SQL)
        conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_CREATED_SQL)
        conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_RUN_SQL)

    dispatch_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(notification_schedule_dispatches)").fetchall()
    }
    if "run_id" not in dispatch_columns:
        conn.execute("ALTER TABLE notification_schedule_dispatches ADD COLUMN run_id INTEGER")
    if "rendered_subject" not in dispatch_columns:
        conn.execute("ALTER TABLE notification_schedule_dispatches ADD COLUMN rendered_subject TEXT")
    if "rendered_body" not in dispatch_columns:
        conn.execute("ALTER TABLE notification_schedule_dispatches ADD COLUMN rendered_body TEXT")
    if "recipient_type" not in dispatch_columns:
        conn.execute("ALTER TABLE notification_schedule_dispatches ADD COLUMN recipient_type TEXT")
    conn.execute(CREATE_INDEX_NOTIFICATION_SCHEDULE_DISPATCHES_RUN_SQL)

    schedule_columns = {row[1] for row in conn.execute("PRAGMA table_info(notification_schedules)").fetchall()}
    if "days_difference" not in schedule_columns:
        conn.execute("ALTER TABLE notification_schedules ADD COLUMN days_difference INTEGER NOT NULL DEFAULT 0")
    if "subject_template" not in schedule_columns:
        conn.execute("ALTER TABLE notification_schedules ADD COLUMN subject_template TEXT NOT NULL DEFAULT ''")
    if "body_template" not in schedule_columns:
        conn.execute("ALTER TABLE notification_schedules ADD COLUMN body_template TEXT NOT NULL DEFAULT ''")
    if "send_minute" not in schedule_columns:
        conn.execute("ALTER TABLE notification_schedules ADD COLUMN send_minute INTEGER NOT NULL DEFAULT 0")

    # Backfill one-rule values from the earliest active rule for existing schedules.
    conn.execute(
        """
        UPDATE notification_schedules
        SET days_difference = COALESCE((
                SELECT r.days_difference
                FROM notification_schedule_rules r
                WHERE r.schedule_id = notification_schedules.id AND r.enabled = 1
                ORDER BY r.id ASC
                LIMIT 1
            ), days_difference),
            subject_template = CASE
                WHEN subject_template = '' THEN COALESCE((
                    SELECT r.subject_template
                    FROM notification_schedule_rules r
                    WHERE r.schedule_id = notification_schedules.id AND r.enabled = 1
                    ORDER BY r.id ASC
                    LIMIT 1
                ), subject_template)
                ELSE subject_template
            END,
            body_template = CASE
                WHEN body_template = '' THEN COALESCE((
                    SELECT r.body_template
                    FROM notification_schedule_rules r
                    WHERE r.schedule_id = notification_schedules.id AND r.enabled = 1
                    ORDER BY r.id ASC
                    LIMIT 1
                ), body_template)
                ELSE body_template
            END
        """
    )

    # Keep one technical rule row per schedule for dispatch FK integrity.
    conn.execute(
        """
        INSERT INTO notification_schedule_rules (schedule_id, days_difference, subject_template, body_template, enabled)
        SELECT s.id, s.days_difference, s.subject_template, s.body_template, 1
        FROM notification_schedules s
        LEFT JOIN (
            SELECT schedule_id, MIN(id) AS min_rule_id
            FROM notification_schedule_rules
            GROUP BY schedule_id
        ) r ON r.schedule_id = s.id
        WHERE r.min_rule_id IS NULL
        """
    )


def _ensure_permissions_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_AUTH_PERMISSIONS_TABLE_SQL)
    conn.execute(SEED_AUTH_PERMISSIONS_SQL)
    conn.execute(CREATE_USER_PERMISSIONS_TABLE_SQL)
    conn.execute(CREATE_INDEX_USER_PERMISSIONS_USER_SQL)


def _ensure_record_locks_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_RECORD_EDIT_LOCKS_TABLE_SQL)
    conn.execute(CREATE_INDEX_RECORD_EDIT_LOCKS_OWNER_SQL)
    conn.execute(CREATE_INDEX_RECORD_EDIT_LOCKS_EXPIRES_SQL)


def _normalize_updated_timestamps(conn: sqlite3.Connection) -> None:
    for table_name in (
        "inspections",
        "recommendations",
        "obligating_decisions",
        "risk_exposure_requests",
    ):
        conn.execute(
            f"""
            UPDATE {table_name}
            SET zaktualizowano_o = substr(zaktualizowano_o, 1, 10) || 'T' || substr(zaktualizowano_o, 12, 8) || '.000Z'
            WHERE zaktualizowano_o LIKE '____-__-__ __:__:__'
            """
        )


def _env_truthy(name: str, default: str = "0") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _ensure_slowniki_schema(conn: sqlite3.Connection, *, seed_slownik_pozycje: bool) -> None:
    has_typy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='slownik_typy'"
    ).fetchone()
    if has_typy is not None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(slownik_typy)").fetchall()}
        required = {"kod_typu", "nazwa_typu", "aktywny"}
        if not required.issubset(columns):
            conn.execute("DROP TABLE slownik_pozycje")
            conn.execute("DROP TABLE slownik_typy")

    conn.execute(CREATE_SLOWNIK_TYPY_TABLE_SQL)
    conn.execute(SEED_SLOWNIK_TYPY_SQL)

    typy_columns = {row[1] for row in conn.execute("PRAGMA table_info(slownik_typy)").fetchall()}
    if "kategoria" not in typy_columns:
        conn.execute("ALTER TABLE slownik_typy ADD COLUMN kategoria INTEGER NOT NULL DEFAULT 0")

    conn.execute(
        """
        UPDATE slownik_typy
        SET kategoria = CASE
            WHEN lower(trim(CAST(kategoria AS TEXT))) IN ('1', 'zalecenia') THEN 1
            WHEN lower(trim(CAST(kategoria AS TEXT))) IN ('2', 'wnioski sankcyjne') THEN 2
            WHEN lower(trim(CAST(kategoria AS TEXT))) IN ('3', 'ogolne') THEN 3
            WHEN lower(kod_typu) = 'statusy_zalecen' THEN 1
            WHEN lower(kod_typu) IN ('department_ogolne') THEN 3
            WHEN lower(kod_typu) IN (
                'department',
                'sankcja',
                'podstawa_prawna_sankcji',
                'naruszenia_skutkujace_sankcja',
                'informacja_o_wszczeciu_postepowania_sankcyjnego',
                'rozstrzygniecie_decyzji_i',
                'rozstrzygniecie_decyzji_ii',
                'rozstrzygniecie_wniosku_sankcyjnego_i',
                'rozstrzygniecie_wniosku_sankcyjnego',
                'nazwy_podmiotow_sankcje'
            ) THEN 2
            ELSE 0
        END
        """
    )

    has_pozycje = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='slownik_pozycje'"
    ).fetchone()

    if has_pozycje is not None:
        pozycje_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(slownik_pozycje)").fetchall()
        }
        required_pozycje = {"kod_typu", "kod_pozycji", "kolejnosc", "aktywny"}
        if not required_pozycje.issubset(pozycje_columns):
            conn.execute("DROP TABLE slownik_pozycje")

    conn.execute(CREATE_SLOWNIK_POZYCJE_TABLE_SQL)
    conn.execute(CREATE_INDEX_SLOWNIK_POZYCJE_KOD_TYPU_SQL)

    pozycje_columns_after = {
        row[1] for row in conn.execute("PRAGMA table_info(slownik_pozycje)").fetchall()
    }
    if "nazwa_uzytkowa" not in pozycje_columns_after:
        if "pomocnicza" in pozycje_columns_after:
            try:
                conn.execute("ALTER TABLE slownik_pozycje RENAME COLUMN pomocnicza TO nazwa_uzytkowa")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE slownik_pozycje ADD COLUMN nazwa_uzytkowa TEXT")
                conn.execute("UPDATE slownik_pozycje SET nazwa_uzytkowa = pomocnicza WHERE nazwa_uzytkowa IS NULL")
        else:
            conn.execute("ALTER TABLE slownik_pozycje ADD COLUMN nazwa_uzytkowa TEXT")

    if seed_slownik_pozycje:
        conn.execute(SEED_SLOWNIK_ZESPOLY_SQL)
        conn.execute(SEED_SLOWNIK_DEPARTMENT_OGOLNE_SQL)
        conn.execute(SEED_SLOWNIK_TYPY_INSPEKCJI_SQL)
        conn.execute(SEED_SLOWNIK_ZAKRESY_INSPEKCJI_SQL)
        conn.execute(SEED_SLOWNIK_DECYZJE_ZOBOWIAZUJACE_SQL)
        conn.execute(SEED_SLOWNIK_ROZSTRZYGNIECIA_WNIOSKU_SQL)

    conn.execute(CREATE_SLOWNIK_STATUS_INSPEKCJI_STYL_TABLE_SQL)
    conn.execute(CREATE_INDEX_SLOWNIK_STATUS_INSPEKCJI_STYL_POZYCJA_SQL)
    conn.execute(SEED_SLOWNIK_STATUS_INSPEKCJI_STYL_SQL)




def _ensure_recommendations_schema(conn: sqlite3.Connection) -> None:
    def _rebuild_recommendations_table(select_sql: str) -> None:
        # Rebuild recommendations structure without triggering FK cascades
        # on dependent tables (recommendation_multi_dates, obligating_decisions).
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations__tmp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id INTEGER,
                    pozycja INTEGER NOT NULL,
                    kod_zalecenia TEXT UNIQUE,
                    nazwa_podmiotu_id INTEGER,
                    data_zalecen TEXT,
                    brak_terminow_wykonania_zalecen INTEGER NOT NULL DEFAULT 0 CHECK (brak_terminow_wykonania_zalecen IN (0, 1)),
                    brak_dat_akceptacji_noty_weryfikacji INTEGER NOT NULL DEFAULT 0 CHECK (brak_dat_akceptacji_noty_weryfikacji IN (0, 1)),
                    status_zalecenia_id INTEGER,
                    komentarz TEXT,
                    created_by_user_id INTEGER NOT NULL,
                    updated_by_user_id INTEGER NOT NULL,
                    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (inspection_id) REFERENCES inspections(id) ON DELETE CASCADE,
                    FOREIGN KEY (nazwa_podmiotu_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (status_zalecenia_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
                    FOREIGN KEY (updated_by_user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                f"""
                INSERT INTO recommendations__tmp (
                    id,
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
                    updated_by_user_id,
                    utworzono_o,
                    zaktualizowano_o
                )
                {select_sql}
                """
            )
            conn.execute("DROP TABLE recommendations")
            conn.execute("ALTER TABLE recommendations__tmp RENAME TO recommendations")
            conn.commit()
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='recommendations'"
    ).fetchone()

    if has_table is not None:
        table_info = conn.execute("PRAGMA table_info(recommendations)").fetchall()
        columns = {row[1] for row in table_info}
        required = {
            "inspection_id",
            "pozycja",
            "nazwa_podmiotu_id",
            "status_zalecenia_id",
            "created_by_user_id",
            "updated_by_user_id",
        }
        if not required.issubset(columns):
            conn.execute("DROP TABLE IF EXISTS recommendation_multi_dates")
            conn.execute("DROP TABLE IF EXISTS recommendation_teams")
            conn.execute("DROP TABLE recommendations")
        else:
            if "kod_zalecenia" not in columns:
                conn.execute("ALTER TABLE recommendations ADD COLUMN kod_zalecenia TEXT")

            columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
            if "data_zalecen" not in columns:
                conn.execute("ALTER TABLE recommendations ADD COLUMN data_zalecen TEXT")

            columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
            if "brak_terminow_wykonania_zalecen" not in columns:
                conn.execute(
                    "ALTER TABLE recommendations ADD COLUMN brak_terminow_wykonania_zalecen INTEGER NOT NULL DEFAULT 0 CHECK (brak_terminow_wykonania_zalecen IN (0, 1))"
                )

            columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
            if "brak_dat_akceptacji_noty_weryfikacji" not in columns:
                conn.execute(
                    "ALTER TABLE recommendations ADD COLUMN brak_dat_akceptacji_noty_weryfikacji INTEGER NOT NULL DEFAULT 0 CHECK (brak_dat_akceptacji_noty_weryfikacji IN (0, 1))"
                )

            columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
            if "termin_wykonania_zalecen" in columns:
                # Historical semantic alignment for old schemas before dropping legacy column.
                conn.execute(
                    """
                    UPDATE recommendations
                    SET data_zalecen = termin_wykonania_zalecen
                    WHERE (data_zalecen IS NULL OR trim(data_zalecen) = '')
                      AND termin_wykonania_zalecen IS NOT NULL
                      AND trim(termin_wykonania_zalecen) <> ''
                    """
                )

            inspection_col = next((row for row in table_info if row[1] == "inspection_id"), None)
            if inspection_col is not None and int(inspection_col[3]) == 1:
                _rebuild_recommendations_table(
                    """
                    SELECT
                        id,
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
                        updated_by_user_id,
                        utworzono_o,
                        zaktualizowano_o
                    FROM recommendations
                    """
                )

            columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)").fetchall()}
            if "termin_wykonania_zalecen" in columns:
                _rebuild_recommendations_table(
                    """
                    SELECT
                        id,
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
                        updated_by_user_id,
                        utworzono_o,
                        zaktualizowano_o
                    FROM recommendations
                    """
                )

            # Usuń historyczne ograniczenie unikalności (inspection_id, pozycja),
            # aby dopuszczać wiele zaleceń z tą samą pozycją w jednej inspekcji.
            index_rows = conn.execute("PRAGMA index_list(recommendations)").fetchall()
            has_unique_inspection_pozycja = False
            for index_row in index_rows:
                if int(index_row[2]) != 1:
                    continue
                index_name = str(index_row[1])
                cols = [str(r[2]) for r in conn.execute(f"PRAGMA index_info({index_name})").fetchall()]
                if cols == ["inspection_id", "pozycja"]:
                    has_unique_inspection_pozycja = True
                    break

            if has_unique_inspection_pozycja:
                _rebuild_recommendations_table(
                    """
                    SELECT
                        id,
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
                        updated_by_user_id,
                        utworzono_o,
                        zaktualizowano_o
                    FROM recommendations
                    """
                )

    conn.execute(CREATE_RECOMMENDATIONS_TABLE_SQL)
    conn.execute(CREATE_INDEX_RECOMMENDATIONS_INSPECTION_SQL)
    conn.execute(CREATE_INDEX_RECOMMENDATIONS_STATUS_SQL)
    conn.execute(CREATE_INDEX_RECOMMENDATIONS_KOD_SQL)

    existing_multi_dates = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='recommendation_multi_dates'"
    ).fetchone()
    if existing_multi_dates is not None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_multi_dates__tmp (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id INTEGER NOT NULL,
                date_type TEXT NOT NULL CHECK (date_type IN ('TERMIN_WYKONANIA_ZALECEN', 'AKCEPTACJA_NOTY_WERYFIKACJI')),
                date_value TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_by_user_id INTEGER NOT NULL,
                updated_by_user_id INTEGER NOT NULL,
                FOREIGN KEY (recommendation_id) REFERENCES recommendations(id) ON DELETE CASCADE,
                FOREIGN KEY (created_by_user_id) REFERENCES users(id),
                FOREIGN KEY (updated_by_user_id) REFERENCES users(id),
                UNIQUE (recommendation_id, date_type, date_value)
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO recommendation_multi_dates__tmp (
                id,
                recommendation_id,
                date_type,
                date_value,
                created_at,
                updated_at,
                created_by_user_id,
                updated_by_user_id
            )
            SELECT
                id,
                recommendation_id,
                CASE
                    WHEN date_type = 'ZALECENIE' THEN 'TERMIN_WYKONANIA_ZALECEN'
                    ELSE date_type
                END,
                date_value,
                created_at,
                updated_at,
                created_by_user_id,
                updated_by_user_id
            FROM recommendation_multi_dates
            """
        )
        conn.execute("DROP TABLE recommendation_multi_dates")
        conn.execute("ALTER TABLE recommendation_multi_dates__tmp RENAME TO recommendation_multi_dates")

    conn.execute(CREATE_RECOMMENDATION_MULTI_DATES_TABLE_SQL)
    conn.execute(CREATE_INDEX_RECOMMENDATION_MULTI_DATES_REC_SQL)
    conn.execute(CREATE_INDEX_RECOMMENDATION_MULTI_DATES_TYPE_VALUE_SQL)
    conn.execute(CREATE_RECOMMENDATION_TEAMS_TABLE_SQL)
    conn.execute(CREATE_INDEX_RECOMMENDATION_TEAMS_REC_SQL)
    conn.execute(CREATE_INDEX_RECOMMENDATION_TEAMS_TEAM_SQL)

    code_rows = conn.execute(
        "SELECT id, kod_zalecenia, data_zalecen, inspection_id, utworzono_o FROM recommendations"
    ).fetchall()
    seq_by_year: dict[str, int] = {}

    for row in code_rows:
        raw_code = str(row["kod_zalecenia"] or "").strip()
        parts = raw_code.split("/")
        if len(parts) != 3 or parts[0] != "Z":
            continue
        year = parts[1]
        try:
            seq = int(parts[2])
        except ValueError:
            continue
        seq_by_year[year] = max(seq_by_year.get(year, 0), seq)

    missing_rows = conn.execute(
        """
        SELECT
            r.id,
            r.data_zalecen,
            r.utworzono_o,
            i.poczatek_inspekcji
        FROM recommendations r
        LEFT JOIN inspections i ON i.id = r.inspection_id
        WHERE r.kod_zalecenia IS NULL OR trim(r.kod_zalecenia) = ''
        ORDER BY r.id ASC
        """
    ).fetchall()

    for row in missing_rows:
        year_source = (
            str(row["data_zalecen"] or "").strip()
            or str(row["poczatek_inspekcji"] or "").strip()
            or str(row["utworzono_o"] or "").strip()
        )
        year = year_source[:4] if len(year_source) >= 4 and year_source[:4].isdigit() else str(date.today().year)
        next_seq = seq_by_year.get(year, 0) + 1
        seq_by_year[year] = next_seq
        conn.execute(
            "UPDATE recommendations SET kod_zalecenia = ? WHERE id = ?",
            (f"Z/{year}/{next_seq}", int(row["id"])),
        )

    # Keep inspections.data_zalecen consistent with recommendation single-date semantics.
    conn.execute(
        """
        UPDATE inspections
        SET data_zalecen = (
            SELECT MAX(r.data_zalecen)
            FROM recommendations r
            WHERE r.inspection_id = inspections.id
              AND r.data_zalecen IS NOT NULL
              AND trim(r.data_zalecen) <> ''
        )
        """
    )


def _ensure_obligating_decisions_schema(conn: sqlite3.Connection) -> None:
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='obligating_decisions'"
    ).fetchone()

    if has_table is not None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(obligating_decisions)").fetchall()}
        required = {
            "kod_decyzji",
            "recommendation_kod_zalecenia",
            "nazwa_podmiotu_id",
            "liczba_zalecen",
            "rozstrzygniecie_i_id",
            "rozstrzygniecie_ii_id",
            "created_by_user_id",
            "updated_by_user_id",
        }
        legacy_required = {
            "pozycja",
            "recommendation_kod_zalecenia",
            "nazwa_podmiotu_id",
            "liczba_zalecen",
            "rozstrzygniecie_i_id",
            "rozstrzygniecie_ii_id",
            "created_by_user_id",
            "updated_by_user_id",
        }
        rebuild_for_fk = False
        if required.issubset(columns):
            fk_rows = conn.execute("PRAGMA foreign_key_list(obligating_decisions)").fetchall()
            for fk_row in fk_rows:
                if (
                    str(fk_row["from"]) == "recommendation_kod_zalecenia"
                    and str(fk_row["on_delete"] or "").upper() != "CASCADE"
                ):
                    rebuild_for_fk = True
                    break

        if required.issubset(columns) and not rebuild_for_fk:
            pass
        elif required.issubset(columns) and rebuild_for_fk:
            conn.execute(
                """
                CREATE TABLE obligating_decisions__tmp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kod_decyzji TEXT,
                    recommendation_kod_zalecenia TEXT,
                    nazwa_podmiotu_id INTEGER,
                    liczba_zalecen INTEGER,
                    data_wszczecia_postepowania_i_instancji TEXT,
                    data_decyzji_i_instancji TEXT,
                    data_doreczenia_decyzji_i_instancji TEXT,
                    rozstrzygniecie_i_id INTEGER,
                    data_wniosku_ponowne_rozpatrzenie TEXT,
                    data_wplywu_wniosku_ponowne_rozpatrzenie TEXT,
                    data_decyzji_ii_instancji TEXT,
                    data_doreczenia_decyzji_ii_instancji TEXT,
                    rozstrzygniecie_ii_id INTEGER,
                    komentarz TEXT,
                    created_by_user_id INTEGER NOT NULL,
                    updated_by_user_id INTEGER NOT NULL,
                    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recommendation_kod_zalecenia) REFERENCES recommendations(kod_zalecenia) ON DELETE CASCADE,
                    FOREIGN KEY (nazwa_podmiotu_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (rozstrzygniecie_i_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (rozstrzygniecie_ii_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
                    FOREIGN KEY (updated_by_user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO obligating_decisions__tmp (
                    id,
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
                    updated_by_user_id,
                    utworzono_o,
                    zaktualizowano_o
                )
                SELECT
                    id,
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
                    updated_by_user_id,
                    utworzono_o,
                    zaktualizowano_o
                FROM obligating_decisions
                """
            )
            conn.execute("DROP TABLE obligating_decisions")
            conn.execute("ALTER TABLE obligating_decisions__tmp RENAME TO obligating_decisions")
        elif legacy_required.issubset(columns):
            conn.execute(
                """
                CREATE TABLE obligating_decisions__tmp (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kod_decyzji TEXT,
                    recommendation_kod_zalecenia TEXT,
                    nazwa_podmiotu_id INTEGER,
                    liczba_zalecen INTEGER,
                    data_wszczecia_postepowania_i_instancji TEXT,
                    data_decyzji_i_instancji TEXT,
                    data_doreczenia_decyzji_i_instancji TEXT,
                    rozstrzygniecie_i_id INTEGER,
                    data_wniosku_ponowne_rozpatrzenie TEXT,
                    data_wplywu_wniosku_ponowne_rozpatrzenie TEXT,
                    data_decyzji_ii_instancji TEXT,
                    data_doreczenia_decyzji_ii_instancji TEXT,
                    rozstrzygniecie_ii_id INTEGER,
                    komentarz TEXT,
                    created_by_user_id INTEGER NOT NULL,
                    updated_by_user_id INTEGER NOT NULL,
                    utworzono_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    zaktualizowano_o TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recommendation_kod_zalecenia) REFERENCES recommendations(kod_zalecenia) ON DELETE CASCADE,
                    FOREIGN KEY (nazwa_podmiotu_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (rozstrzygniecie_i_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (rozstrzygniecie_ii_id) REFERENCES slownik_pozycje(id),
                    FOREIGN KEY (created_by_user_id) REFERENCES users(id),
                    FOREIGN KEY (updated_by_user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO obligating_decisions__tmp (
                    id,
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
                    updated_by_user_id,
                    utworzono_o,
                    zaktualizowano_o
                )
                SELECT
                    id,
                    NULL,
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
                    updated_by_user_id,
                    utworzono_o,
                    zaktualizowano_o
                FROM obligating_decisions
                """
            )
            conn.execute("DROP TABLE obligating_decisions")
            conn.execute("ALTER TABLE obligating_decisions__tmp RENAME TO obligating_decisions")
        else:
            conn.execute("DROP TABLE obligating_decisions")

    conn.execute(CREATE_OBLIGATING_DECISIONS_TABLE_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_KOD_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_RECOMMENDATION_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_PODMIOT_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_ROZSTRZYG_I_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_ROZSTRZYG_II_SQL)
    conn.execute(CREATE_OBLIGATING_DECISIONS_PERSONS_I_TABLE_SQL)
    conn.execute(CREATE_OBLIGATING_DECISIONS_PERSONS_II_TABLE_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_I_DECISION_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_I_SLOWNIK_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_II_DECISION_SQL)
    conn.execute(CREATE_INDEX_OBLIGATING_DECISIONS_PERSONS_II_SLOWNIK_SQL)

    code_rows = conn.execute(
        """
        SELECT id, kod_decyzji, data_decyzji_i_instancji, data_wszczecia_postepowania_i_instancji, utworzono_o
        FROM obligating_decisions
        ORDER BY id ASC
        """
    ).fetchall()

    seq_by_year: dict[str, int] = {}
    for row in code_rows:
        raw_code = str(row["kod_decyzji"] or "").strip()
        parts = raw_code.split("/")
        if len(parts) != 3 or parts[0] != "DZ":
            continue
        year = parts[1]
        try:
            seq = int(parts[2])
        except ValueError:
            continue
        seq_by_year[year] = max(seq_by_year.get(year, 0), seq)

    for row in code_rows:
        if str(row["kod_decyzji"] or "").strip():
            continue
        year_source = (
            str(row["data_decyzji_i_instancji"] or "").strip()
            or str(row["data_wszczecia_postepowania_i_instancji"] or "").strip()
            or str(row["utworzono_o"] or "").strip()
        )
        year = year_source[:4] if len(year_source) >= 4 and year_source[:4].isdigit() else str(date.today().year)
        next_seq = seq_by_year.get(year, 0) + 1
        seq_by_year[year] = next_seq
        conn.execute(
            "UPDATE obligating_decisions SET kod_decyzji = ? WHERE id = ?",
            (f"DZ/{year}/{next_seq}", int(row["id"])),
        )


def _slug_code(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    code = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_").upper()
    return code or "POZYCJA"


def _resolve_or_create_slownik_id(
    conn: sqlite3.Connection,
    kod_typu: str,
    raw_value: str | None,
) -> int | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
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


def _ensure_risk_exposure_schema(conn: sqlite3.Connection) -> None:
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_exposure_requests'"
    ).fetchone()

    if has_table is not None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_requests)").fetchall()}
        if "kod_sankcji" not in columns:
            conn.execute("ALTER TABLE risk_exposure_requests ADD COLUMN kod_sankcji TEXT")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_requests)").fetchall()}

        if "nazwa_podmiotu_objetego_inspekcja_id" not in columns:
            conn.execute("ALTER TABLE risk_exposure_requests ADD COLUMN nazwa_podmiotu_objetego_inspekcja_id INTEGER")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_requests)").fetchall()}
        if "wniosek_do_id" not in columns:
            conn.execute("ALTER TABLE risk_exposure_requests ADD COLUMN wniosek_do_id INTEGER")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_requests)").fetchall()}
        if "czy_mamy_informacje_o_wszczeciu_postepowania_id" not in columns:
            conn.execute("ALTER TABLE risk_exposure_requests ADD COLUMN czy_mamy_informacje_o_wszczeciu_postepowania_id INTEGER")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_requests)").fetchall()}
        if "rozstrzygniecie_id" not in columns:
            conn.execute("ALTER TABLE risk_exposure_requests ADD COLUMN rozstrzygniecie_id INTEGER")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_requests)").fetchall()}

        if "nazwa_podmiotu_objetego_inspekcja" in columns:
            rows = conn.execute(
                """
                SELECT id, inspection_id, nazwa_podmiotu_objetego_inspekcja
                FROM risk_exposure_requests
                WHERE nazwa_podmiotu_objetego_inspekcja_id IS NULL
                """
            ).fetchall()
            for row in rows:
                inspection_id = row["inspection_id"]
                resolved_id: int | None = None
                if inspection_id is not None:
                    inspection_row = conn.execute(
                        "SELECT nazwa_podmiotu_id FROM inspections WHERE id = ? LIMIT 1",
                        (inspection_id,),
                    ).fetchone()
                    if inspection_row is not None and inspection_row["nazwa_podmiotu_id"] is not None:
                        resolved_id = int(inspection_row["nazwa_podmiotu_id"])
                if resolved_id is None:
                    resolved_id = _resolve_or_create_slownik_id(
                        conn,
                        "nazwy_podmiotow",
                        row["nazwa_podmiotu_objetego_inspekcja"],
                    )
                if resolved_id is not None:
                    conn.execute(
                        """
                        UPDATE risk_exposure_requests
                        SET nazwa_podmiotu_objetego_inspekcja_id = ?
                        WHERE id = ?
                        """,
                        (resolved_id, int(row["id"])),
                    )

        if "wniosek_do" in columns:
            rows = conn.execute(
                """
                SELECT id, wniosek_do
                FROM risk_exposure_requests
                WHERE wniosek_do_id IS NULL
                """
            ).fetchall()
            for row in rows:
                resolved_id = _resolve_or_create_slownik_id(conn, "department", row["wniosek_do"])
                if resolved_id is not None:
                    conn.execute(
                        "UPDATE risk_exposure_requests SET wniosek_do_id = ? WHERE id = ?",
                        (resolved_id, int(row["id"])),
                    )

        if "czy_mamy_informacje_o_wszczeciu_postepowania" in columns:
            rows = conn.execute(
                """
                SELECT id, czy_mamy_informacje_o_wszczeciu_postepowania
                FROM risk_exposure_requests
                WHERE czy_mamy_informacje_o_wszczeciu_postepowania_id IS NULL
                """
            ).fetchall()
            for row in rows:
                resolved_id = _resolve_or_create_slownik_id(
                    conn,
                    "informacja_o_wszczeciu_postepowania_sankcyjnego",
                    row["czy_mamy_informacje_o_wszczeciu_postepowania"],
                )
                if resolved_id is not None:
                    conn.execute(
                        """
                        UPDATE risk_exposure_requests
                        SET czy_mamy_informacje_o_wszczeciu_postepowania_id = ?
                        WHERE id = ?
                        """,
                        (resolved_id, int(row["id"])),
                    )

        if "rozstrzygniecie" in columns:
            rows = conn.execute(
                """
                SELECT id, rozstrzygniecie
                FROM risk_exposure_requests
                WHERE rozstrzygniecie_id IS NULL
                """
            ).fetchall()
            for row in rows:
                resolved_id = _resolve_or_create_slownik_id(
                    conn,
                    "rozstrzygniecie_wniosku_sankcyjnego_I",
                    row["rozstrzygniecie"],
                )
                if resolved_id is not None:
                    conn.execute(
                        "UPDATE risk_exposure_requests SET rozstrzygniecie_id = ? WHERE id = ?",
                        (resolved_id, int(row["id"])),
                    )

        required = {
            "lp",
            "inspection_id",
            "nazwa_podmiotu_objetego_inspekcja_id",
            "wniosek_do_id",
            "czy_mamy_informacje_o_wszczeciu_postepowania_id",
            "rozstrzygniecie_id",
            "utworzono_przez_user_id",
            "zaktualizowano_przez_user_id",
        }
        if not required.issubset(columns):
            conn.execute("DROP TABLE IF EXISTS risk_exposure_sanction_subjects")
            conn.execute("DROP TABLE IF EXISTS risk_exposure_sanctions")
            conn.execute("DROP TABLE IF EXISTS risk_exposure_legal_bases")
            conn.execute("DROP TABLE IF EXISTS risk_exposure_violations")
            conn.execute("DROP TABLE IF EXISTS risk_exposure_multi_values")
            conn.execute("DROP TABLE risk_exposure_requests")

    conn.execute(CREATE_RISK_EXPOSURE_REQUESTS_TABLE_SQL)

    request_columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_requests)").fetchall()}
    if "nazwa_podmiotu_objetego_inspekcja" in request_columns:
        conn.execute("ALTER TABLE risk_exposure_requests DROP COLUMN nazwa_podmiotu_objetego_inspekcja")
    if "wniosek_do" in request_columns:
        conn.execute("ALTER TABLE risk_exposure_requests DROP COLUMN wniosek_do")
    if "czy_mamy_informacje_o_wszczeciu_postepowania" in request_columns:
        conn.execute("ALTER TABLE risk_exposure_requests DROP COLUMN czy_mamy_informacje_o_wszczeciu_postepowania")
    if "rozstrzygniecie" in request_columns:
        conn.execute("ALTER TABLE risk_exposure_requests DROP COLUMN rozstrzygniecie")

    conn.execute(CREATE_INDEX_RISK_EXPOSURE_INSPECTION_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_KOD_SQL)
    conn.execute(CREATE_RISK_EXPOSURE_SANCTION_SUBJECTS_TABLE_SQL)
    conn.execute(CREATE_RISK_EXPOSURE_SANCTIONS_TABLE_SQL)
    conn.execute(CREATE_RISK_EXPOSURE_LEGAL_BASES_TABLE_SQL)
    conn.execute(CREATE_RISK_EXPOSURE_VIOLATIONS_TABLE_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_SANCTION_SUBJECTS_REQ_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_SANCTION_SUBJECTS_SLOWNIK_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_SANCTIONS_REQ_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_SANCTIONS_SLOWNIK_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_LEGAL_BASES_REQ_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_LEGAL_BASES_SLOWNIK_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_VIOLATIONS_REQ_SQL)
    conn.execute(CREATE_INDEX_RISK_EXPOSURE_VIOLATIONS_SLOWNIK_SQL)

    has_legacy_multi = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_exposure_multi_values'"
    ).fetchone()
    if has_legacy_multi is not None:
        legacy_columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_multi_values)").fetchall()}
        if "slownik_pozycja_id" not in legacy_columns:
            conn.execute("ALTER TABLE risk_exposure_multi_values ADD COLUMN slownik_pozycja_id INTEGER")
            legacy_columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_exposure_multi_values)").fetchall()}

        value_type_map = {
            "NAZWA_PODMIOTU_OBJETEGO_SANKCJA": ("nazwy_podmiotow_sankcje", "risk_exposure_sanction_subjects"),
            "SANKCJA": ("sankcja", "risk_exposure_sanctions"),
            "PODSTAWA_PRAWNA_SANKCJI": ("podstawa_prawna_sankcji", "risk_exposure_legal_bases"),
            "NARUSZENIA_SKUTKUJACE_SANKCJA": ("naruszenia_skutkujace_sankcja", "risk_exposure_violations"),
        }

        rows = conn.execute(
            """
            SELECT id, value_type, value_text
            FROM risk_exposure_multi_values
            WHERE slownik_pozycja_id IS NULL
            """
        ).fetchall()
        for row in rows:
            mapping = value_type_map.get(str(row["value_type"]))
            if mapping is None:
                continue
            kod_typu, _ = mapping
            resolved_id = _resolve_or_create_slownik_id(conn, kod_typu, row["value_text"])
            if resolved_id is None:
                continue
            conn.execute(
                "UPDATE risk_exposure_multi_values SET slownik_pozycja_id = ? WHERE id = ?",
                (resolved_id, int(row["id"])),
            )

        for value_type, (_, target_table) in value_type_map.items():
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {target_table} (
                    risk_exposure_id,
                    slownik_pozycja_id,
                    created_at,
                    updated_at,
                    created_by_user_id,
                    updated_by_user_id
                )
                SELECT
                    mv.risk_exposure_id,
                    mv.slownik_pozycja_id,
                    mv.created_at,
                    mv.updated_at,
                    mv.created_by_user_id,
                    mv.updated_by_user_id
                FROM risk_exposure_multi_values mv
                WHERE mv.value_type = ?
                  AND mv.slownik_pozycja_id IS NOT NULL
                """,
                (value_type,),
            )

        conn.execute("DROP TABLE risk_exposure_multi_values")


CREATE_AUDIT_LOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    uzytkownik  TEXT NOT NULL,
    akcja       TEXT NOT NULL CHECK (akcja IN ('CREATE', 'UPDATE', 'DELETE')),
    data_godz   TEXT NOT NULL,
    rejestr     TEXT NOT NULL,
    rekord_kod  TEXT NOT NULL,
    pole        TEXT,
    przed       TEXT,
    po          TEXT
)
"""

CREATE_INDEX_AUDIT_LOG_REJESTR_REKORD_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_log_rejestr_rekord
ON audit_log(rejestr, rekord_kod)
"""

CREATE_INDEX_AUDIT_LOG_UZYTKOWNIK_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_log_uzytkownik
ON audit_log(uzytkownik)
"""

CREATE_INDEX_AUDIT_LOG_AKCJA_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_log_akcja
ON audit_log(akcja)
"""

CREATE_INDEX_AUDIT_LOG_REKORD_KOD_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_log_rekord_kod
ON audit_log(rekord_kod)
"""

CREATE_INDEX_AUDIT_LOG_DATA_GODZ_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_log_data_godz
ON audit_log(data_godz)
"""

CREATE_INDEX_AUDIT_LOG_SESSION_SQL = """
CREATE INDEX IF NOT EXISTS idx_audit_log_session_id
ON audit_log(session_id)
"""


def _ensure_audit_log_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_AUDIT_LOG_TABLE_SQL)
    conn.execute(CREATE_INDEX_AUDIT_LOG_REJESTR_REKORD_SQL)
    conn.execute(CREATE_INDEX_AUDIT_LOG_UZYTKOWNIK_SQL)
    conn.execute(CREATE_INDEX_AUDIT_LOG_AKCJA_SQL)
    conn.execute(CREATE_INDEX_AUDIT_LOG_REKORD_KOD_SQL)
    conn.execute(CREATE_INDEX_AUDIT_LOG_DATA_GODZ_SQL)
    conn.execute(CREATE_INDEX_AUDIT_LOG_SESSION_SQL)


def _bootstrap_user_profiles(conn: sqlite3.Connection) -> None:
    diu_row = conn.execute(
        """
        SELECT kod_pozycji, nazwa_pozycji
        FROM slownik_pozycje
        WHERE lower(kod_typu) = 'department_ogolne'
          AND upper(kod_pozycji) = 'DIU'
        LIMIT 1
        """
    ).fetchone()
    diu_code = str(diu_row["kod_pozycji"]) if diu_row is not None else "DIU"
    diu_label = str(diu_row["nazwa_pozycji"]) if diu_row is not None else "DIU"

    conn.execute(
        """
        UPDATE users
        SET account_type = CASE
                WHEN account_type IS NULL OR trim(account_type) = '' THEN
                    CASE WHEN rola_id = 4 THEN 'observer' ELSE 'diu' END
                WHEN lower(account_type) IN ('diu', 'observer', 'technical') THEN lower(account_type)
                ELSE CASE WHEN rola_id = 4 THEN 'observer' ELSE 'diu' END
            END,
            list_visibility = CASE
                WHEN list_visibility IS NULL OR trim(list_visibility) = '' THEN 'visible'
                WHEN lower(list_visibility) IN ('visible', 'hidden') THEN lower(list_visibility)
                ELSE 'visible'
            END,
            department_code = CASE
                WHEN lower(COALESCE(account_type, CASE WHEN rola_id = 4 THEN 'observer' ELSE 'diu' END)) = 'diu'
                    THEN ?
                ELSE department_code
            END,
            zespol_id = CASE
                WHEN lower(COALESCE(account_type, CASE WHEN rola_id = 4 THEN 'observer' ELSE 'diu' END)) IN ('observer', 'technical')
                    THEN NULL
                ELSE zespol_id
            END
        """,
        (diu_code,),
    )

    users = conn.execute(
        """
        SELECT
            id,
            login,
            imie,
            nazwisko,
            email,
            rola_id,
            aktywny,
            account_type,
            zespol_id,
            department_code,
            list_visibility,
            COALESCE(utworzono_o, zaktualizowano_o, CURRENT_TIMESTAMP) AS base_changed_at
        FROM users
        ORDER BY id ASC
        """
    ).fetchall()

    for row in users:
        user_id = int(row["id"])
        has_history = conn.execute(
            "SELECT 1 FROM user_profile_history WHERE user_id = ? LIMIT 1",
            (user_id,),
        ).fetchone()
        if has_history is None:
            account_type = str(row["account_type"] or "diu").strip().lower() or "diu"
            department_code = str(row["department_code"] or "").strip() or None
            zespol_id = row["zespol_id"] if account_type == "diu" else None
            if account_type == "diu":
                department_code = diu_code
            department_label = None
            if department_code is not None:
                dep_row = conn.execute(
                    """
                    SELECT nazwa_pozycji
                    FROM slownik_pozycje
                    WHERE lower(kod_typu) = 'department_ogolne'
                      AND lower(kod_pozycji) = lower(?)
                    LIMIT 1
                    """,
                    (department_code,),
                ).fetchone()
                if dep_row is not None:
                    department_label = str(dep_row["nazwa_pozycji"])
                elif account_type == "diu":
                    department_label = diu_label

            changed_at = str(row["base_changed_at"] or "").strip() or datetime.now().isoformat(timespec="seconds")
            list_visibility = str(row["list_visibility"] or "visible").strip().lower() or "visible"
            permission_rows = conn.execute(
                """
                SELECT permission_code
                FROM user_permissions
                WHERE user_id = ?
                ORDER BY permission_code ASC
                """,
                (user_id,),
            ).fetchall()
            permissions_codes = ";".join(str(item["permission_code"]) for item in permission_rows) or None
            conn.execute(
                """
                INSERT INTO user_profile_history (
                    user_id,
                    valid_from,
                    valid_to,
                    login,
                    imie,
                    nazwisko,
                    email,
                    rola_id,
                    aktywny,
                    account_type,
                    zespol_id,
                    department_code,
                    department_label,
                    list_visibility,
                    permissions_codes,
                    changed_by_user_id,
                    changed_by_login,
                    changed_at
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    changed_at,
                    str(row["login"] or "").strip() or None,
                    str(row["imie"] or "").strip() or None,
                    str(row["nazwisko"] or "").strip() or None,
                    str(row["email"] or "").strip() or None,
                    int(row["rola_id"]) if row["rola_id"] is not None else None,
                    int(row["aktywny"]) if row["aktywny"] is not None else None,
                    account_type,
                    zespol_id,
                    department_code,
                    department_label,
                    list_visibility,
                    permissions_codes,
                    user_id,
                    str(row["login"]),
                    changed_at,
                ),
            )

    conn.execute(
        """
        UPDATE users
        SET profile_changed_at = COALESCE(
                profile_changed_at,
                (
                    SELECT h.changed_at
                    FROM user_profile_history h
                    WHERE h.user_id = users.id
                    ORDER BY h.valid_from DESC, h.id DESC
                    LIMIT 1
                )
            ),
            profile_changed_by_user_id = COALESCE(
                profile_changed_by_user_id,
                (
                    SELECT h.changed_by_user_id
                    FROM user_profile_history h
                    WHERE h.user_id = users.id
                    ORDER BY h.valid_from DESC, h.id DESC
                    LIMIT 1
                )
            )
        """
    )


def init_db() -> None:
    seed_slownik_pozycje = not _env_truthy("DISABLE_SLOWNIK_POZYCJE_SEED", "0")
    seed_teams = not _env_truthy("DISABLE_TEAMS_SEED", "0")

    with get_connection() as conn:
        _ensure_slowniki_schema(conn, seed_slownik_pozycje=seed_slownik_pozycje)
        _ensure_teams_schema(conn)
        _ensure_users_schema(conn)
        if seed_teams:
            conn.execute(SEED_TEAMS_SQL)
        _ensure_user_profile_history_schema(conn)
        _ensure_auth_sessions_schema(conn)
        _ensure_user_invites_schema(conn)
        _ensure_user_password_resets_schema(conn)
        _ensure_auth_login_attempts_schema(conn)
        _ensure_auth_password_reset_throttle_schema(conn)
        _ensure_notification_schedules_schema(conn)
        _ensure_record_locks_schema(conn)
        _ensure_permissions_schema(conn)
        _ensure_inspections_schema(conn)
        _ensure_recommendations_schema(conn)
        _ensure_obligating_decisions_schema(conn)
        _ensure_risk_exposure_schema(conn)
        _ensure_audit_log_schema(conn)
        _normalize_updated_timestamps(conn)

        admin_default_password_hash = hash_password("admin1234!")
        conn.execute(
            """
            INSERT OR IGNORE INTO users (
                login, imie, nazwisko, email, password_hash, rola_id, zespol_id, aktywny
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "admin",
                "Admin",
                "Systemu",
                "admin@rejestr.local",
                admin_default_password_hash,
                3,
                None,
                1,
            ),
        )
        conn.commit()
