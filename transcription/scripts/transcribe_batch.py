#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import re
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTION_ROOT = REPO_ROOT / "transcription"
LOGGER = logging.getLogger("transcribe_batch")


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


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slugify(value: str, max_len: int = 80) -> str:
    value = value.lower().strip()
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


def migrate_legacy_paths(
    guid: str,
    audio_dir: Path,
    output_dir: Path,
    base_name: str,
) -> None:
    old_base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in guid)
    if not old_base or old_base == base_name:
        return

    migrations = [
        (audio_dir / f"{old_base}.mp3", audio_dir / f"{base_name}.mp3"),
        (output_dir / f"{old_base}.txt", output_dir / f"{base_name}.txt"),
        (output_dir / f"{old_base}.json", output_dir / f"{base_name}.json"),
    ]
    for old_path, new_path in migrations:
        if old_path.exists() and not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)


def download_audio(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target_path)


def download_audio_with_retries(
    url: str,
    target_path: Path,
    retries: int,
    retry_delay_sec: float,
) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            download_audio(url, target_path)
            return
        except Exception as exc:
            last_exc = exc
            LOGGER.warning(
                f"Download attempt {attempt}/{retries} failed for {target_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < retries:
                sleep_for = retry_delay_sec * attempt
                LOGGER.info("Retrying in %.1fs...", sleep_for)
                time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def write_outputs(
    base_name: str,
    output_dir: Path,
    transcript: dict,
) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{base_name}.txt"
    json_path = output_dir / f"{base_name}.json"

    with txt_path.open("w", encoding="utf-8") as f:
        for seg in transcript["segments"]:
            f.write(seg["text"].strip() + "\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    return str(txt_path), str(json_path)


def write_partial_outputs(
    base_name: str,
    output_dir: Path,
    transcript: dict,
) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{base_name}.partial.txt"
    json_path = output_dir / f"{base_name}.partial.json"

    with txt_path.open("w", encoding="utf-8") as f:
        for seg in transcript["segments"]:
            f.write(seg["text"].strip() + "\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    return str(txt_path), str(json_path)


def load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_file(model: WhisperModel, audio_path: Path) -> dict:
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
    )

    data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": [],
    }
    for seg in segments:
        words = []
        for w in (seg.words or []):
            if w.start is None or w.end is None:
                continue
            words.append(
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": (w.word or "").strip(),
                }
            )
        data["segments"].append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            }
        )
    return data


def transcribe_file_with_progress(
    model: WhisperModel,
    audio_path: Path,
    episode_label: str,
    progress_step_pct: int,
    base_name: str,
    output_dir: Path,
    partial_every_segments: int = 8,
) -> dict:
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=5,
        word_timestamps=True,
    )

    total_duration = float(info.duration or 0.0)
    next_mark = max(1, progress_step_pct)
    data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": [],
    }

    if total_duration <= 0:
        LOGGER.info("[episode-progress] %s: duration unknown", episode_label)

    for seg in segments:
        words = []
        for w in (seg.words or []):
            if w.start is None or w.end is None:
                continue
            words.append(
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": (w.word or "").strip(),
                }
            )
        data["segments"].append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
            }
        )

        if len(data["segments"]) % max(1, partial_every_segments) == 0:
            write_partial_outputs(base_name, output_dir, data)

        if total_duration > 0:
            pct = int(min(100, max(0, (seg.end / total_duration) * 100)))
            while pct >= next_mark and next_mark <= 100:
                LOGGER.info("[episode-progress] %s: %s%%", episode_label, next_mark)
                next_mark += progress_step_pct

    if next_mark <= 100:
        LOGGER.info("[episode-progress] %s: 100%%", episode_label)
    write_partial_outputs(base_name, output_dir, data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(TRANSCRIPTION_ROOT / "manifests" / "pipeline_manifest.csv"),
    )
    parser.add_argument(
        "--audio-dir",
        default=str(TRANSCRIPTION_ROOT / "artifacts" / "01_whisper_transcript" / "audio"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(TRANSCRIPTION_ROOT / "artifacts" / "01_whisper_transcript" / "transcripts"),
    )
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument(
        "--gpu-failure-policy",
        choices=["fallback", "error"],
        default="fallback",
        help="When --device=cuda and CUDA init fails: fallback to CPU or fail fast.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-delay-sec", type=float, default=3.0)
    parser.add_argument("--episode-progress-step", type=int, default=5)
    parser.add_argument("--log-file", default="", help="Optional log file path.")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    setup_logging(args.log_file)

    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    rows = load_manifest(manifest_path)
    total_rows = len(rows)
    pending_rows = sum(1 for r in rows if (r.get("status") or "").strip() != "done")
    LOGGER.info("Manifest loaded: total=%s, to_process=%s, redo=%s", total_rows, pending_rows, args.redo)

    try:
        model = load_model(args.model_size, args.device, args.compute_type)
    except Exception as exc:
        if args.device == "cuda":
            if args.gpu_failure_policy == "error":
                raise RuntimeError(
                    f"CUDA model load failed and gpu-failure-policy=error: {type(exc).__name__}: {exc}"
                ) from exc
            LOGGER.warning(
                "CUDA model load failed (%s: %s); falling back to CPU int8.",
                type(exc).__name__,
                exc,
            )
            model = load_model(args.model_size, "cpu", "int8")
        else:
            raise

    processed = 0
    for idx, row in enumerate(rows, start=1):
        status = (row.get("status") or "").strip()
        if not args.redo and status == "done":
            continue

        guid = (row.get("guid") or "").strip() or f"row_{processed}"
        title = (row.get("title") or "").strip()
        pub_date_iso = (row.get("pub_date_iso") or "").strip()
        audio_url = (row.get("audio_url") or "").strip()
        base_name = build_base_name(guid, title, pub_date_iso)
        audio_path = audio_dir / f"{base_name}.mp3"
        migrate_legacy_paths(guid, audio_dir, output_dir, base_name)

        try:
            if not audio_url:
                raise ValueError("Missing audio_url in manifest row.")

            LOGGER.info("[%s/%s] Start: %s", idx, total_rows, title)
            if not audio_path.exists():
                LOGGER.info("Downloading: %s", title)
                download_audio_with_retries(
                    audio_url,
                    audio_path,
                    retries=max(1, args.download_retries),
                    retry_delay_sec=max(0.5, args.retry_delay_sec),
                )

            LOGGER.info("Transcribing: %s", title)
            transcript = transcribe_file_with_progress(
                model,
                audio_path,
                episode_label=title,
                progress_step_pct=max(1, min(25, args.episode_progress_step)),
                base_name=base_name,
                output_dir=output_dir,
            )
            txt_path, json_path = write_outputs(base_name, output_dir, transcript)
            partial_txt = output_dir / f"{base_name}.partial.txt"
            partial_json = output_dir / f"{base_name}.partial.json"
            if partial_txt.exists():
                partial_txt.unlink()
            if partial_json.exists():
                partial_json.unlink()

            row["status"] = "done"
            row["audio_path"] = str(audio_path)
            row["transcript_txt"] = txt_path
            row["transcript_json"] = json_path
            row["error"] = ""
            row["updated_at"] = now_utc()
            LOGGER.info("[%s/%s] Done: %s", idx, total_rows, title)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["updated_at"] = now_utc()
            LOGGER.error("Error on %s: %s", title, row["error"])
            LOGGER.error(traceback.format_exc())
        finally:
            save_manifest(manifest_path, rows)
            done_count = sum(1 for r in rows if (r.get("status") or "").strip() == "done")
            err_count = sum(1 for r in rows if (r.get("status") or "").strip() == "error")
            pend_count = sum(
                1
                for r in rows
                if (r.get("status") or "").strip() not in {"done", "error"}
            )
            LOGGER.info("[progress] done=%s error=%s pending=%s", done_count, err_count, pend_count)

        processed += 1
        if args.max_episodes > 0 and processed >= args.max_episodes:
            break

    LOGGER.info("Finished. Episodes attempted: %s", processed)


if __name__ == "__main__":
    main()
