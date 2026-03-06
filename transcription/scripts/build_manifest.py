#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pdscript.common import setup_script_logging  # noqa: E402
from pdscript.config import DEFAULT_CONFIG_PATH, choose_value, get_cfg, load_config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "transcription" / "manifests" / "pipeline_manifest.csv"
DEFAULT_EPISODES_CSV = REPO_ROOT / "episodes_source.csv"
LOGGER = logging.getLogger("build_manifest")


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
        "spotify_url",
        "apple_url",
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


def _exists(path_value: str) -> bool:
    text = (path_value or "").strip()
    if not text:
        return False
    return Path(text).exists()


def reconcile_row_statuses(row: dict) -> dict:
    """Normalize stale 'done' states when expected output files are missing."""
    out = dict(row)

    if (out.get("status") or "").strip() == "done":
        has_audio = _exists(out.get("audio_path", ""))
        has_txt = _exists(out.get("transcript_txt", ""))
        has_json = _exists(out.get("transcript_json", ""))
        if not (has_audio and has_txt and has_json):
            out["status"] = "pending"
            out["error"] = ""

    if (out.get("speaker_status") or "").strip() == "done":
        has_speaker = all(
            _exists(out.get(col, ""))
            for col in ["speaker_md", "speaker_diar_json", "speaker_word_csv", "speaker_segment_csv"]
        )
        if not has_speaker:
            out["speaker_status"] = "pending"
            out["speaker_error"] = ""

    if (out.get("clean_python_status") or "").strip() == "done":
        has_clean_py = _exists(out.get("clean_python_md", "")) and _exists(out.get("clean_python_json", ""))
        if not has_clean_py:
            out["clean_python_status"] = "pending"
            out["clean_python_error"] = ""

    if (out.get("clean_llm_status") or "").strip() == "done":
        has_clean_llm = _exists(out.get("clean_llm_json", ""))
        if not has_clean_llm:
            out["clean_llm_status"] = "pending"
            out["clean_llm_error"] = ""

    if (out.get("web_status") or "").strip() == "done":
        has_web = _exists(out.get("web_md", ""))
        if not has_web:
            out["web_status"] = "pending"
            out["web_error"] = ""

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
    parser.add_argument(
        "--rss-feed-url",
        default="",
        help="Optional podcast RSS feed URL to enrich per-episode links.",
    )
    parser.add_argument(
        "--apple-show-id",
        default="",
        help="Optional Apple Podcasts show ID for per-episode Apple links.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def fetch_rss_episode_links(rss_feed_url: str) -> dict[str, str]:
    if not rss_feed_url:
        return {}
    with urllib.request.urlopen(rss_feed_url, timeout=30) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    by_guid: dict[str, str] = {}
    for item in root.findall("./channel/item"):
        guid = (item.findtext("guid") or "").strip()
        link = (item.findtext("link") or "").strip()
        if guid and link:
            by_guid[guid] = link
    return by_guid


def fetch_apple_episode_links(apple_show_id: str) -> dict[str, str]:
    if not apple_show_id:
        return {}
    url = f"https://itunes.apple.com/lookup?id={apple_show_id}&entity=podcastEpisode&limit=200"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    by_guid: dict[str, str] = {}
    for row in payload.get("results", []):
        if row.get("wrapperType") != "podcastEpisode":
            continue
        guid = (row.get("episodeGuid") or "").strip()
        track_url = (row.get("trackViewUrl") or "").strip()
        if guid and track_url:
            by_guid[guid] = track_url
    return by_guid


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_path, required=True)
    args.rss_feed_url = choose_value(args.rss_feed_url, get_cfg(cfg, "podcast.rss_feed_url", ""))
    args.apple_show_id = choose_value(args.apple_show_id, get_cfg(cfg, "podcast.apple_show_id", ""))
    if not args.rss_feed_url:
        raise ValueError("Missing required podcast.rss_feed_url (set in config or --rss-feed-url).")

    setup_script_logging(LOGGER, args.log_file)
    episodes_csv = resolve_episodes_csv(args.episodes_csv)
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()

    if not episodes_csv.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {episodes_csv}. "
            "Provide --episodes-csv or create ./episodes_source.csv."
        )

    with episodes_csv.open(newline="", encoding="utf-8") as f:
        episodes = list(csv.DictReader(f))

    rss_links: dict[str, str] = {}
    apple_links: dict[str, str] = {}
    if args.rss_feed_url:
        try:
            rss_links = fetch_rss_episode_links(args.rss_feed_url)
            LOGGER.info("RSS link enrichment loaded: %s episodes", len(rss_links))
        except Exception as exc:
            LOGGER.warning("RSS link enrichment failed: %s: %s", type(exc).__name__, exc)
    if args.apple_show_id:
        try:
            apple_links = fetch_apple_episode_links(args.apple_show_id)
            LOGGER.info("Apple link enrichment loaded: %s episodes", len(apple_links))
        except Exception as exc:
            LOGGER.warning("Apple link enrichment failed: %s: %s", type(exc).__name__, exc)

    existing_fields, existing_rows, existing_by_guid = load_existing_manifest(manifest_csv)
    existing_order = [r.get("guid", "") for r in existing_rows if (r.get("guid") or "").strip()]

    rows = []
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for ep in episodes:
        guid = (ep.get("guid") or "").strip()
        title = (ep.get("title") or "").strip()
        pub_date_iso = (ep.get("pub_date_iso") or "").strip()
        audio_url = (ep.get("audio_url") or "").strip()
        episode_url = (ep.get("link") or "").strip() or (rss_links.get(guid) or "").strip()
        source_transcript_url = (ep.get("transcript_url") or "").strip()
        spotify_url = rss_links.get(guid, "")
        apple_url = apple_links.get(guid, "")
        prev = dict(existing_by_guid.get(guid, {}))
        is_new = not bool(prev)

        prev["guid"] = guid
        prev["rss_guid"] = guid
        prev["title"] = title
        prev["pub_date_iso"] = pub_date_iso
        prev["audio_url"] = audio_url
        prev["episode_url"] = episode_url
        prev["spotify_url"] = spotify_url or (prev.get("spotify_url") or "")
        prev["apple_url"] = apple_url or (prev.get("apple_url") or "")
        prev["source_transcript_url"] = source_transcript_url
        if is_new:
            prev.setdefault("status", "pending")
            prev.setdefault("audio_path", "")
            prev.setdefault("transcript_txt", "")
            prev.setdefault("transcript_json", "")
            prev.setdefault("error", "")
            prev.setdefault("updated_at", now)
        rows.append(reconcile_row_statuses(prev))

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
