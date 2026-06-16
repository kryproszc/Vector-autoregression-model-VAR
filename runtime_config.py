from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = BASE_DIR / "config" / "app.env"
DEFAULT_BACKUP_ENV_FILE = BASE_DIR / "config" / "db_backup.env"


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def load_env_file(env_file: Path | None = None) -> None:
    paths = [env_file] if env_file is not None else [DEFAULT_ENV_FILE, DEFAULT_BACKUP_ENV_FILE]

    for path in paths:
        if path is None or not path.exists() or not path.is_file():
            continue

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            value = _strip_wrapping_quotes(value.strip())
            # Keep explicit process env variables as higher priority.
            os.environ.setdefault(key, value)
