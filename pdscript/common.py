from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def setup_script_logging(logger: logging.Logger, log_file: str = "") -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        lp = Path(log_file).expanduser().resolve()
        lp.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(lp, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def slugify(value: str, max_len: int = 80) -> str:
    import re

    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        value = "untitled"
    return value[:max_len].strip("-")


def build_base_name(guid: str, title: str, pub_date_iso: str) -> str:
    date_part = (pub_date_iso or "").strip()[:10] or "unknown-date"
    title_part = slugify(title, max_len=72)
    guid_part = (guid or "noguid").replace("-", "")[:8]
    return f"{date_part}__{title_part}__{guid_part}"


def read_manifest_rows(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def write_manifest_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not fieldnames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for col in fieldnames:
                out.setdefault(col, "")
            writer.writerow(out)
    tmp.replace(path)
