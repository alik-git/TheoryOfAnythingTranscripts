#!/usr/bin/env python3
import argparse
import csv
import logging
import sys
from pathlib import Path
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "transcription" / "manifests" / "pipeline_manifest.csv"
DEFAULT_EPISODES_CSV = REPO_ROOT / "episodes_source.csv"
LOGGER = logging.getLogger("build_manifest")


def setup_logging(log_file: str = "") -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    LOGGER.addHandler(sh)

    if log_file:
        lp = Path(log_file).expanduser().resolve()
        lp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(lp, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        LOGGER.addHandler(fh)


def load_existing_manifest(path: Path) -> tuple[list[str], list[dict], dict[str, dict]]:
    if not path.exists():
        return [], [], {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    by_guid = {}
    for r in rows:
        g = (r.get("guid") or "").strip()
        if g:
            by_guid[g] = r
    return fieldnames, rows, by_guid


def ordered_fieldnames(existing: list[str]) -> list[str]:
    preferred = [
        "guid",
        "rss_guid",
        "title",
        "pub_date_iso",
        "audio_url",
        "episode_url",
        "source_transcript_url",
        "status",
        "audio_path",
        "transcript_txt",
        "transcript_json",
        "error",
        "updated_at",
    ]
    out = []
    for f in preferred:
        if f not in out:
            out.append(f)
    for f in existing:
        if f not in out:
            out.append(f)
    return out


def resolve_episodes_csv(explicit_path: str) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    if DEFAULT_EPISODES_CSV.exists():
        return DEFAULT_EPISODES_CSV
    candidates = sorted(REPO_ROOT.glob("*episodes*.csv"))
    if candidates:
        return candidates[0]
    return DEFAULT_EPISODES_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-csv",
        default="",
        help="Path to source episodes CSV (defaults to repo ./episodes_source.csv, then legacy fallback).",
    )
    parser.add_argument(
        "--manifest-csv",
        default=str(MANIFEST_CSV),
        help="Path to output manifest CSV.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path (in addition to stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    episodes_csv = resolve_episodes_csv(args.episodes_csv)
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()

    if not episodes_csv.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {episodes_csv}. "
            "Provide --episodes-csv or create ./episodes_source.csv."
        )

    with episodes_csv.open(newline="", encoding="utf-8") as f:
        episodes = list(csv.DictReader(f))

    existing_fields, existing_rows, existing_by_guid = load_existing_manifest(manifest_csv)
    existing_order = [r.get("guid", "") for r in existing_rows if (r.get("guid") or "").strip()]

    rows = []
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for ep in episodes:
        guid = (ep.get("guid") or "").strip()
        title = (ep.get("title") or "").strip()
        pub_date_iso = (ep.get("pub_date_iso") or "").strip()
        audio_url = (ep.get("audio_url") or "").strip()
        episode_url = (ep.get("link") or "").strip()
        source_transcript_url = (ep.get("transcript_url") or "").strip()
        prev = dict(existing_by_guid.get(guid, {}))
        is_new = not bool(prev)

        prev["guid"] = guid
        prev["rss_guid"] = guid
        prev["title"] = title
        prev["pub_date_iso"] = pub_date_iso
        prev["audio_url"] = audio_url
        prev["episode_url"] = episode_url
        prev["source_transcript_url"] = source_transcript_url
        if is_new:
            prev.setdefault("status", "pending")
            prev.setdefault("audio_path", "")
            prev.setdefault("transcript_txt", "")
            prev.setdefault("transcript_json", "")
            prev.setdefault("error", "")
            prev.setdefault("updated_at", now)
        rows.append(prev)

    # Keep orphaned rows (if any) that are not currently in episodes CSV.
    episode_guids = {r["guid"] for r in rows}
    for g in existing_order:
        if g and g not in episode_guids:
            rows.append(existing_by_guid[g])

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(existing_fields)
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            for fn in fieldnames:
                out.setdefault(fn, "")
            writer.writerow(out)

    LOGGER.info("Source CSV: %s", episodes_csv)
    LOGGER.info("Wrote %s rows to %s", len(rows), manifest_csv)


if __name__ == "__main__":
    main()
