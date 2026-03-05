#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTION_ROOT = REPO_ROOT / "transcription"
SCRIPTS_DIR = TRANSCRIPTION_ROOT / "scripts"

BUILD_MANIFEST_SCRIPT = SCRIPTS_DIR / "build_manifest.py"
TRANSCRIBE_SCRIPT = SCRIPTS_DIR / "transcribe_batch.py"
SPEAKER_SCRIPT = SCRIPTS_DIR / "speaker_batch.py"
CLEAN_SCRIPT = SCRIPTS_DIR / "clean_dialogue_batch.py"

DEFAULT_MANIFEST = TRANSCRIPTION_ROOT / "manifests" / "pipeline_manifest.csv"
ARTIFACTS_ROOT = TRANSCRIPTION_ROOT / "artifacts"
DEFAULT_AUDIO_DIR = ARTIFACTS_ROOT / "01_whisper_transcript" / "audio"
DEFAULT_TRANSCRIPTS_DIR = ARTIFACTS_ROOT / "01_whisper_transcript" / "transcripts"
DEFAULT_DIARIZATION_ROOT = ARTIFACTS_ROOT / "02_diarization"
DEFAULT_SEGMENTS_DIR = DEFAULT_DIARIZATION_ROOT / "debug"
DEFAULT_CLEAN_PY_ROOT = ARTIFACTS_ROOT / "03_clean_python"
DEFAULT_CLEAN_LLM_ROOT = ARTIFACTS_ROOT / "04_clean_llm"
DEFAULT_LOGS_DIR = TRANSCRIPTION_ROOT / "logs"
DEFAULT_RSS_FEED_URL = "https://anchor.fm/s/14b6fc10/podcast/rss"
DEFAULT_APPLE_SHOW_ID = "1503194218"
LOGGER = logging.getLogger("pdscript.cli")


def setup_logging(log_file: str = "", write_file: bool = True) -> str:
    resolved = ""
    if write_file:
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
    subprocess.run(cmd, check=True, env=env)


def add_bool_arg(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


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
        "--out-root",
        str(args.out_root),
        "--model-size",
        str(args.model_size),
        "--device",
        str(args.device),
        "--compute-type",
        str(args.compute_type),
        "--max-episodes",
        str(args.max_episodes),
        "--episode-progress-step",
        str(args.episode_progress_step),
        "--partial-every-segments",
        str(args.partial_every_segments),
        "--min-speakers",
        str(args.min_speakers),
        "--max-speakers",
        str(args.max_speakers),
        "--telemetry-interval-sec",
        str(args.telemetry_interval_sec),
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


def run_all(args: argparse.Namespace) -> None:
    LOGGER.info("[pipeline] Stage 1/5: manifest")
    run_manifest(
        args.episodes_csv,
        str(args.manifest),
        str(args.log_file),
        args.rss_feed_url,
        args.apple_show_id,
    )
    LOGGER.info("[pipeline] Stage 2/5: transcribe")
    run_transcribe(args)
    LOGGER.info("[pipeline] Stage 3/5: speaker")
    run_speaker(args)
    LOGGER.info("[pipeline] Stage 4/5: clean-python")
    run_clean(args, "python")
    LOGGER.info("[pipeline] Stage 5/5: clean-llm")
    run_clean(args, "llm")
    LOGGER.info("[pipeline] done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Podcast transcription pipeline entrypoint (manifest/transcribe/speaker/clean/status)."
    )
    parser.add_argument("--all", action="store_true", help="Run all stages sequentially.")
    parser.add_argument("--episodes-csv", default="", help="Input episodes source CSV path.")
    parser.add_argument("--rss-feed-url", default=DEFAULT_RSS_FEED_URL)
    parser.add_argument("--apple-show-id", default=DEFAULT_APPLE_SHOW_ID)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Pipeline manifest path.")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_TRANSCRIPTS_DIR))
    parser.add_argument("--out-root", default=str(DEFAULT_DIARIZATION_ROOT))
    parser.add_argument("--segments-dir", default=str(DEFAULT_SEGMENTS_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--log-file", default="", help="Pipeline log file path. Defaults to transcription/logs/pipeline_<timestamp>.log")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=3.0)
    parser.add_argument("--episode-progress-step", type=int, default=5)
    parser.add_argument("--partial-every-segments", type=int, default=8)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=15)
    parser.add_argument("--telemetry-interval-sec", type=int, default=30)
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
    parser.add_argument("--redo", action="store_true")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("manifest", help="Build/refresh pipeline manifest from episodes CSV.")
    sub.add_parser("transcribe", help="Run audio download + ASR transcription stage.")
    sub.add_parser("speaker", help="Run speaker diarization + alignment stage.")
    sub.add_parser("clean-python", help="Run deterministic cleanup stage.")
    sub.add_parser("clean-llm", help="Run LLM cleanup stage.")
    sub.add_parser("clean-both", help="Run both cleanup stages in one call.")
    sub.add_parser("status", help="Show manifest summary and latest log tail.")
    sub.add_parser("all", help="Run all stages sequentially.")
    return parser


def main() -> None:
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

    if cmd == "manifest":
        run_manifest(
            args.episodes_csv,
            str(args.manifest),
            str(args.log_file),
            args.rss_feed_url,
            args.apple_show_id,
        )
        return
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
    if cmd == "all":
        run_all(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
