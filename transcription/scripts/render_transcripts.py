#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from clean_dialogue_batch import (  # noqa: E402
    ARTIFACTS_ROOT,
    MANIFEST_PATH,
    PODCAST_APPLE_SHOW_URL,
    PODCAST_SPOTIFY_SHOW_URL,
    infer_episode_links,
    render_named_turns_md,
    slug_base_to_title,
    write_site_episode_page,
)

CLEAN_JSON_DIR = ARTIFACTS_ROOT / "04_clean_llm" / "json"
WEB_MD_DIR = ARTIFACTS_ROOT / "05_webformat" / "md"
EPISODES_DIR = Path(__file__).resolve().parents[2] / "episodes"
LOGGER = logging.getLogger("render_transcripts")


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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


def base_from_clean_json(path: Path) -> str:
    n = path.name
    if n.endswith(".clean.json"):
        return n[: -len(".clean.json")]
    return path.stem


def _episode_num_from_text(text: str) -> str:
    import re

    m = re.search(r"\bepisode[-\s_:]*(\d+)\b", text or "", flags=re.IGNORECASE)
    if not m:
        return ""
    return str(int(m.group(1)))


def load_manifest(path: Path) -> tuple[list[str], list[dict], dict[str, dict], dict[str, dict]]:
    if not path.exists():
        return [], [], {}, {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    by_base: dict[str, dict] = {}
    by_episode_num: dict[str, dict] = {}
    for r in rows:
        seg = (r.get("speaker_segment_csv") or "").strip()
        if seg:
            name = Path(seg).name
            if name.endswith(".segments.csv"):
                by_base[name[: -len(".segments.csv")]] = r
        ep_num = _episode_num_from_text((r.get("title") or ""))
        if ep_num and ep_num not in by_episode_num:
            by_episode_num[ep_num] = r
    return fieldnames, rows, by_base, by_episode_num


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not fieldnames:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for c in fieldnames:
                out.setdefault(c, "")
            w.writerow(out)
    tmp.replace(path)


def ensure_clean_llm_columns(fieldnames: list[str]) -> list[str]:
    out = list(fieldnames)
    for c in ["clean_llm_md", "clean_llm_updated_at"]:
        if c not in out:
            out.append(c)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    p.add_argument("--clean-json-dir", default=str(CLEAN_JSON_DIR))
    p.add_argument("--web-md-dir", default=str(WEB_MD_DIR))
    p.add_argument("--episodes-dir", default=str(EPISODES_DIR))
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--log-file", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    manifest_path = Path(args.manifest_path)
    clean_json_dir = Path(args.clean_json_dir)
    web_md_dir = Path(args.web_md_dir)
    episodes_dir = Path(args.episodes_dir)

    web_md_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows, by_base, by_episode_num = load_manifest(manifest_path)
    fieldnames = ensure_clean_llm_columns(fieldnames)

    files = sorted(clean_json_dir.glob("*.clean.json"))
    if not files:
        raise SystemExit(f"No clean JSON files found in {clean_json_dir}")

    processed = 0
    for i, jf in enumerate(files, start=1):
        if args.max_episodes > 0 and processed >= args.max_episodes:
            break
        base = base_from_clean_json(jf)
        md_out = web_md_dir / f"{base}.clean.md"
        if md_out.exists() and not args.redo:
            continue
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
            turns = payload.get("turns", [])
            row = by_base.get(base)
            if row is None:
                ep_num = _episode_num_from_text(base.replace("__", " "))
                if ep_num:
                    row = by_episode_num.get(ep_num)
            title = (row or {}).get("title") or slug_base_to_title(base)
            spotify, apple = infer_episode_links(title, row)
            if not spotify:
                spotify = PODCAST_SPOTIFY_SHOW_URL
            if not apple:
                apple = PODCAST_APPLE_SHOW_URL

            md = render_named_turns_md(title=title, turns=turns, spotify_url=spotify, apple_url=apple)
            md_out.write_text(md, encoding="utf-8")
            write_site_episode_page(base, title, md)
            LOGGER.info("[%s/%s] rendered %s", i, len(files), base)

            if row is not None:
                row["clean_llm_md"] = str(md_out)
                row["clean_llm_updated_at"] = now_utc()
                if fieldnames and rows:
                    write_manifest(manifest_path, fieldnames, rows)
        except Exception as exc:
            LOGGER.error("[%s/%s] %s ERROR: %s: %s", i, len(files), base, type(exc).__name__, exc)
        processed += 1

    LOGGER.info("Finished render. Episodes attempted: %s", processed)


if __name__ == "__main__":
    main()
