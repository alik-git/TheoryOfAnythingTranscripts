#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import re
import shlex
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTION_ROOT = REPO_ROOT / "transcription"
SCRIPTS_DIR = TRANSCRIPTION_ROOT / "scripts"

BUILD_MANIFEST_SCRIPT = SCRIPTS_DIR / "build_manifest.py"
TRANSCRIBE_SCRIPT = SCRIPTS_DIR / "transcribe_batch.py"
SPEAKER_SCRIPT = SCRIPTS_DIR / "speaker_batch.py"
CLEAN_SCRIPT = SCRIPTS_DIR / "clean_dialogue_batch.py"
RENDER_SCRIPT = SCRIPTS_DIR / "render_transcripts.py"

DEFAULT_MANIFEST = TRANSCRIPTION_ROOT / "manifests" / "pipeline_manifest.csv"
ARTIFACTS_ROOT = TRANSCRIPTION_ROOT / "artifacts"
DEFAULT_AUDIO_DIR = ARTIFACTS_ROOT / "01_whisper_transcript" / "audio"
DEFAULT_TRANSCRIPTS_DIR = ARTIFACTS_ROOT / "01_whisper_transcript" / "transcripts"
DEFAULT_DIARIZATION_ROOT = ARTIFACTS_ROOT / "02_diarization"
DEFAULT_SEGMENTS_DIR = DEFAULT_DIARIZATION_ROOT / "debug"
DEFAULT_CLEAN_PY_ROOT = ARTIFACTS_ROOT / "03_clean_python"
DEFAULT_CLEAN_LLM_ROOT = ARTIFACTS_ROOT / "04_clean_llm"
DEFAULT_CLEAN_JSON_DIR = DEFAULT_CLEAN_LLM_ROOT / "json"
DEFAULT_WEBFORMAT_ROOT = ARTIFACTS_ROOT / "05_webformat"
DEFAULT_WEB_MD_DIR = DEFAULT_WEBFORMAT_ROOT / "md"
DEFAULT_EPISODES_DIR = REPO_ROOT / "episodes"
DEFAULT_LOGS_DIR = TRANSCRIPTION_ROOT / "logs"
DEFAULT_RSS_FEED_URL = "https://anchor.fm/s/14b6fc10/podcast/rss"
DEFAULT_APPLE_SHOW_ID = "1503194218"
LOGGER = logging.getLogger("pdscript.cli")

STAGE_DIRS = {
    "01": "01_whisper_transcript",
    "02": "02_diarization",
    "03": "03_clean_python",
    "04": "04_clean_llm",
    "05": "05_webformat",
}


def _ensure_python_311_plus() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f"pdscript requires Python 3.11+ (detected {sys.version_info.major}.{sys.version_info.minor})."
        )


def _rotate_existing_logs(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    existing_logs = sorted(
        p for p in logs_dir.iterdir() if p.is_file() and p.suffix in {".log", ".pid"}
    )
    if not existing_logs:
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")
    old_dir = logs_dir / "old" / f"run_{stamp}"
    old_dir.mkdir(parents=True, exist_ok=True)
    for artifact_path in existing_logs:
        shutil.move(str(artifact_path), str(old_dir / artifact_path.name))


def setup_logging(log_file: str = "", write_file: bool = True) -> str:
    resolved = ""
    if write_file:
        _rotate_existing_logs(DEFAULT_LOGS_DIR)
        if not log_file:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")
            log_file = str(DEFAULT_LOGS_DIR / f"pipeline_{stamp}.log")
        lp = Path(log_file).expanduser().resolve()
        lp.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(lp)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    LOGGER.addHandler(sh)

    if write_file:
        fh = logging.FileHandler(resolved, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        LOGGER.addHandler(fh)
    return resolved


def run_cmd(cmd: list[str]) -> None:
    LOGGER.info("[run] %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("[run-failed] exit_code=%s cmd=%s", exc.returncode, " ".join(cmd))
        raise


def add_bool_arg(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def add_value_arg(cmd: list[str], flag: str, value: str) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    cmd.extend([flag, text])


def run_manifest(
    episodes_csv: str,
    manifest: str,
    log_file: str,
    rss_feed_url: str = "",
    apple_show_id: str = "",
) -> None:
    cmd = [
        sys.executable,
        str(BUILD_MANIFEST_SCRIPT),
        "--manifest-csv",
        manifest,
        "--log-file",
        log_file,
    ]
    if episodes_csv:
        cmd.extend(["--episodes-csv", episodes_csv])
    if rss_feed_url:
        cmd.extend(["--rss-feed-url", rss_feed_url])
    if apple_show_id:
        cmd.extend(["--apple-show-id", apple_show_id])
    run_cmd(cmd)


def run_transcribe(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(TRANSCRIBE_SCRIPT),
        "--manifest",
        str(args.manifest),
        "--audio-dir",
        str(args.audio_dir),
        "--output-dir",
        str(args.output_dir),
        "--model-size",
        str(args.model_size),
        "--device",
        str(args.device),
        "--compute-type",
        str(args.compute_type),
        "--gpu-failure-policy",
        str(args.gpu_failure_policy),
        "--max-episodes",
        str(args.max_episodes),
        "--download-retries",
        str(args.download_retries),
        "--retry-delay-sec",
        str(args.retry_delay_sec),
        "--episode-progress-step",
        str(args.episode_progress_step),
        "--log-file",
        str(args.log_file),
    ]
    add_bool_arg(cmd, "--redo", args.redo)
    run_cmd(cmd)


def run_speaker(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(SPEAKER_SCRIPT),
        "--manifest",
        str(args.manifest),
        "--audio-dir",
        str(args.audio_dir),
        "--transcripts-dir",
        str(args.output_dir),
        "--out-root",
        str(args.out_root),
        "--model-size",
        str(args.model_size),
        "--device",
        str(args.device),
        "--compute-type",
        str(args.compute_type),
        "--gpu-failure-policy",
        str(args.gpu_failure_policy),
        "--max-episodes",
        str(args.max_episodes),
        "--episode-progress-step",
        str(args.episode_progress_step),
        "--diarization-progress-step",
        str(args.diarization_progress_step),
        "--partial-every-segments",
        str(args.partial_every_segments),
        "--min-speakers",
        str(args.min_speakers),
        "--max-speakers",
        str(args.max_speakers),
        "--telemetry-interval-sec",
        str(args.telemetry_interval_sec),
        "--gpu-init-retries",
        str(args.gpu_init_retries),
        "--gpu-init-retry-delay-sec",
        str(args.gpu_init_retry_delay_sec),
        "--log-file",
        str(args.log_file),
    ]
    add_bool_arg(cmd, "--redo", args.redo)
    run_cmd(cmd)


def run_clean(args: argparse.Namespace, mode: str) -> None:
    cmd = [
        sys.executable,
        str(CLEAN_SCRIPT),
        "--mode",
        mode,
        "--segments-dir",
        str(args.segments_dir),
        "--max-episodes",
        str(args.max_episodes),
        "--max-gap-sec",
        str(args.max_gap_sec),
        "--manifest-path",
        str(args.manifest),
        "--log-dir",
        str(args.logs_dir),
    ]
    add_value_arg(cmd, "--log-file", getattr(args, "log_file", ""))
    add_bool_arg(cmd, "--redo", args.redo)

    if mode in {"llm", "both"}:
        cmd.extend(
            [
                "--llm-model",
                str(args.llm_model),
                "--llm-temperature",
                str(args.llm_temperature),
                "--llm-max-chars-per-chunk",
                str(args.llm_max_chars_per_chunk),
                "--llm-max-words-per-chunk",
                str(args.llm_max_words_per_chunk),
                "--llm-overlap-words",
                str(args.llm_overlap_words),
                "--llm-chunk-sentence-overrun-words",
                str(args.llm_chunk_sentence_overrun_words),
                "--llm-request-timeout-sec",
                str(args.llm_request_timeout_sec),
                "--llm-max-retries",
                str(args.llm_max_retries),
                "--llm-retry-backoff-sec",
                str(args.llm_retry_backoff_sec),
            ]
        )
    run_cmd(cmd)


def run_status(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    logs_dir = Path(args.logs_dir)
    LOGGER.info("Manifest: %s", manifest)
    if not manifest.exists():
        LOGGER.info("Status: manifest missing")
    else:
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        done = sum(1 for r in rows if (r.get("status") or "").strip() == "done")
        err = sum(1 for r in rows if (r.get("status") or "").strip() == "error")
        pending = len(rows) - done - err
        LOGGER.info("Status: total=%s done=%s error=%s pending=%s", len(rows), done, err, pending)

    latest = None
    if logs_dir.exists():
        def safe_mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except FileNotFoundError:
                return -1.0

        candidates = sorted(logs_dir.glob("*.log"), key=safe_mtime, reverse=True)
        candidates = [c for c in candidates if safe_mtime(c) >= 0]
        if candidates:
            latest = candidates[0]
    if latest is None:
        LOGGER.info("Logs: no log files found")
        return

    LOGGER.info("Latest log: %s", latest)
    try:
        tail_lines = max(1, int(args.tail_lines))
        content = latest.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in content[-tail_lines:]:
            LOGGER.info("%s", line)
    except Exception as exc:
        LOGGER.error("Could not read log tail: %s: %s", type(exc).__name__, exc)


def run_render(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--manifest-path",
        str(args.manifest),
        "--clean-json-dir",
        str(args.clean_json_dir),
        "--web-md-dir",
        str(args.web_md_dir),
        "--episodes-dir",
        str(args.episodes_dir),
        "--max-episodes",
        str(args.max_episodes),
        "--log-file",
        str(args.log_file),
    ]
    add_bool_arg(cmd, "--redo", args.redo)
    run_cmd(cmd)


def _parse_archive_stages(raw: str) -> list[str]:
    value = (raw or "all").strip().lower()
    if value == "all":
        return [STAGE_DIRS[k] for k in sorted(STAGE_DIRS.keys())]
    alias = {
        "whisper": "01",
        "transcribe": "01",
        "diarization": "02",
        "speaker": "02",
        "clean_python": "03",
        "clean-llm": "04",
        "clean_llm": "04",
        "llm": "04",
        "web": "05",
        "webformat": "05",
        "render": "05",
    }
    out: list[str] = []
    seen: set[str] = set()
    for token in [t.strip() for t in value.split(",") if t.strip()]:
        key = alias.get(token, token)
        if key in STAGE_DIRS:
            stage_dir = STAGE_DIRS[key]
        elif token in STAGE_DIRS.values():
            stage_dir = token
        else:
            raise ValueError(f"Unknown archive stage token: {token}")
        if stage_dir not in seen:
            out.append(stage_dir)
            seen.add(stage_dir)
    return out


def run_archive(args: argparse.Namespace) -> None:
    stages = _parse_archive_stages(args.archive_stages)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    tag = (args.archive_tag or "").strip()
    tag_part = f"_{tag}" if tag else ""
    archive_root = ARTIFACTS_ROOT / "old" / f"archive_{stamp}{tag_part}"
    archive_root.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    for stage_dir in stages:
        src = ARTIFACTS_ROOT / stage_dir
        if src.exists():
            dst = archive_root / stage_dir
            shutil.move(str(src), str(dst))
            moved.append(stage_dir)

        # recreate expected directory skeleton for each stage
        if stage_dir == "01_whisper_transcript":
            (src / "audio").mkdir(parents=True, exist_ok=True)
            (src / "transcripts").mkdir(parents=True, exist_ok=True)
        elif stage_dir == "02_diarization":
            (src / "md").mkdir(parents=True, exist_ok=True)
            (src / "diarization").mkdir(parents=True, exist_ok=True)
            (src / "debug").mkdir(parents=True, exist_ok=True)
        elif stage_dir == "03_clean_python":
            (src / "md").mkdir(parents=True, exist_ok=True)
            (src / "json").mkdir(parents=True, exist_ok=True)
        elif stage_dir == "04_clean_llm":
            (src / "md").mkdir(parents=True, exist_ok=True)
            (src / "json").mkdir(parents=True, exist_ok=True)
            (src / "raw").mkdir(parents=True, exist_ok=True)
            (src / "meta").mkdir(parents=True, exist_ok=True)
        elif stage_dir == "05_webformat":
            (src / "md").mkdir(parents=True, exist_ok=True)

    LOGGER.info("Archive complete. root=%s moved=%s", archive_root, ",".join(moved) if moved else "none")


def _extract_episode_number(row: dict) -> int | None:
    raw = (row.get("episode_number") or "").strip()
    if raw.isdigit():
        return int(raw)
    title = (row.get("title") or "").strip()
    m = re.search(r"\bepisode\s+(\d+)\b", title, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_episode_numbers(raw_values: list[str] | None) -> list[int]:
    if not raw_values:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        for token in (raw or "").split(","):
            t = token.strip()
            if not t:
                continue
            if not t.isdigit():
                raise ValueError(f"Invalid episode number token: '{t}'")
            n = int(t)
            if n <= 0:
                raise ValueError(f"Episode number must be positive: {n}")
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _scope_manifest_to_episodes(manifest_path: Path, episode_numbers: list[int], *, reason: str) -> Path:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not episode_numbers:
        raise ValueError("No episode numbers provided for manifest scoping.")

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    rows_by_ep: dict[int, list[dict]] = {}
    for r in rows:
        ep = _extract_episode_number(r)
        if ep is None:
            continue
        rows_by_ep.setdefault(ep, []).append(r)

    missing = [n for n in episode_numbers if n not in rows_by_ep]
    if missing:
        raise ValueError(f"Episode number(s) not found in manifest: {missing}")

    ambiguous = [n for n in episode_numbers if len(rows_by_ep.get(n, [])) > 1]
    if ambiguous:
        raise ValueError(f"Episode number(s) ambiguous in manifest: {ambiguous}")

    selected_by_ep = {n: rows_by_ep[n][0] for n in episode_numbers}
    selected_rows = [r for r in rows if _extract_episode_number(r) in selected_by_ep]

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    tmp_dir = TRANSCRIPTION_ROOT / "manifests" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    eps_slug = "-".join(str(n) for n in episode_numbers[:8])
    if len(episode_numbers) > 8:
        eps_slug += "-more"
    scoped_path = tmp_dir / f"scoped_ep{eps_slug}_{stamp}.csv"
    with scoped_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    LOGGER.info(
        "Scoped manifest for %s: episodes=%s rows=%s path=%s",
        reason,
        episode_numbers,
        len(selected_rows),
        scoped_path,
    )
    return scoped_path


def run_all(args: argparse.Namespace) -> None:
    LOGGER.info("[pipeline] Stage 1/6: manifest")
    run_manifest(
        args.episodes_csv,
        str(args.manifest),
        str(args.log_file),
        args.rss_feed_url,
        args.apple_show_id,
    )
    if args.episode_number:
        episode_numbers = _parse_episode_numbers(args.episode_number)
        args.manifest = str(
            _scope_manifest_to_episodes(
                Path(args.manifest),
                episode_numbers,
                reason="all",
            )
        )
    LOGGER.info("[pipeline] Stage 2/6: transcribe")
    run_transcribe(args)
    LOGGER.info("[pipeline] Stage 3/6: speaker")
    run_speaker(args)
    LOGGER.info("[pipeline] Stage 4/6: clean-python")
    run_clean(args, "python")
    LOGGER.info("[pipeline] Stage 5/6: clean-llm")
    run_clean(args, "llm")
    LOGGER.info("[pipeline] Stage 6/6: render")
    run_render(args)
    LOGGER.info("[pipeline] done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Podcast transcription pipeline entrypoint (manifest/transcribe/speaker/clean/status)."
    )
    parser.add_argument("--all", action="store_true", help="Run all stages sequentially.")
    parser.add_argument("--episodes-csv", default="", help="Input episodes source CSV path.")
    parser.add_argument(
        "--episode-number",
        action="append",
        default=[],
        help="Run specific episode number(s). Repeat flag and/or use comma list, e.g. --episode-number 131 --episode-number 132,133",
    )
    parser.add_argument("--rss-feed-url", default=DEFAULT_RSS_FEED_URL)
    parser.add_argument("--apple-show-id", default=DEFAULT_APPLE_SHOW_ID)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Pipeline manifest path.")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_TRANSCRIPTS_DIR))
    parser.add_argument("--out-root", default=str(DEFAULT_DIARIZATION_ROOT))
    parser.add_argument("--segments-dir", default=str(DEFAULT_SEGMENTS_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--clean-json-dir", default=str(DEFAULT_CLEAN_JSON_DIR))
    parser.add_argument("--web-md-dir", default=str(DEFAULT_WEB_MD_DIR))
    parser.add_argument("--episodes-dir", default=str(DEFAULT_EPISODES_DIR))
    parser.add_argument("--log-file", default="", help="Pipeline log file path. Defaults to transcription/logs/pipeline_<timestamp>.log")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument(
        "--gpu-failure-policy",
        default="fallback",
        choices=["fallback", "error"],
        help="When CUDA init fails: fallback to CPU or fail fast.",
    )
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=3.0)
    parser.add_argument("--episode-progress-step", type=int, default=5)
    parser.add_argument("--diarization-progress-step", type=int, default=5)
    parser.add_argument("--partial-every-segments", type=int, default=8)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=15)
    parser.add_argument("--telemetry-interval-sec", type=int, default=30)
    parser.add_argument("--gpu-init-retries", type=int, default=3)
    parser.add_argument("--gpu-init-retry-delay-sec", type=float, default=5.0)
    parser.add_argument("--max-gap-sec", type=float, default=1.2)
    parser.add_argument("--llm-model", default="gpt-5-nano")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-chars-per-chunk", type=int, default=12000)
    parser.add_argument("--llm-max-words-per-chunk", type=int, default=500)
    parser.add_argument("--llm-overlap-words", type=int, default=100)
    parser.add_argument("--llm-chunk-sentence-overrun-words", type=int, default=120)
    parser.add_argument("--llm-request-timeout-sec", type=float, default=120.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff-sec", type=float, default=4.0)
    parser.add_argument("--tail-lines", type=int, default=30, help="For status command.")
    parser.add_argument("--archive-stages", default="all", help="Comma list like '03,04,05' or 'all'.")
    parser.add_argument("--archive-tag", default="", help="Optional suffix for archive folder name.")
    parser.add_argument("--redo", action="store_true")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("manifest", help="Build/refresh pipeline manifest from episodes CSV.")
    sub.add_parser("transcribe", help="Run audio download + ASR transcription stage.")
    sub.add_parser("speaker", help="Run speaker diarization + alignment stage.")
    sub.add_parser("clean-python", help="Run deterministic cleanup stage.")
    sub.add_parser("clean-llm", help="Run LLM cleanup stage.")
    sub.add_parser("clean-both", help="Run both cleanup stages in one call.")
    sub.add_parser("render", help="Render website markdown/pages from clean JSON (no LLM calls).")
    sub.add_parser("archive", help="Move selected stage output dirs under transcription/artifacts/old and recreate empty stage dirs.")
    sub.add_parser("status", help="Show manifest summary and latest log tail.")
    sub.add_parser("all", help="Run all stages sequentially.")
    return parser


def main() -> None:
    _ensure_python_311_plus()
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command
    if args.all:
        cmd = "all"

    if cmd == "status" and not args.log_file:
        setup_logging(write_file=False)
    else:
        args.log_file = setup_logging(args.log_file, write_file=True)
        LOGGER.info("pipeline_log=%s", args.log_file)
    LOGGER.info("invocation=%s", shlex.join([sys.executable, *sys.argv]))

    if cmd == "manifest":
        run_manifest(
            args.episodes_csv,
            str(args.manifest),
            str(args.log_file),
            args.rss_feed_url,
            args.apple_show_id,
        )
        return
    if cmd in {"transcribe", "speaker", "clean-python", "clean-llm", "clean-both", "render"} and args.episode_number:
        episode_numbers = _parse_episode_numbers(args.episode_number)
        args.manifest = str(
            _scope_manifest_to_episodes(
                Path(args.manifest),
                episode_numbers,
                reason=cmd,
            )
        )
    if cmd == "transcribe":
        run_transcribe(args)
        return
    if cmd == "speaker":
        run_speaker(args)
        return
    if cmd == "clean-python":
        run_clean(args, "python")
        return
    if cmd == "clean-llm":
        run_clean(args, "llm")
        return
    if cmd == "clean-both":
        run_clean(args, "both")
        return
    if cmd == "status":
        run_status(args)
        return
    if cmd == "archive":
        run_archive(args)
        return
    if cmd == "render":
        run_render(args)
        return
    if cmd == "all":
        run_all(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
