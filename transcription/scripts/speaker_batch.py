#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from huggingface_hub import get_token
from pyannote.audio import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTION_ROOT = REPO_ROOT / "transcription"
LOGGER = logging.getLogger("speaker_batch")
SPEAKER_COLORS = {
    "SPEAKER_00": "#1f77b4",
    "SPEAKER_01": "#d62728",
    "SPEAKER_02": "#2ca02c",
    "SPEAKER_03": "#ff7f0e",
    "SPEAKER_04": "#9467bd",
    "UNKNOWN": "#7f7f7f",
}


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_self_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return round(kb / 1024.0, 1)
    except Exception:
        return None
    return None


def gpu_telemetry() -> dict:
    out = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_allocated_mb": None,
        "torch_reserved_mb": None,
        "nvidia_util_pct": None,
        "nvidia_mem_used_mb": None,
    }
    if torch.cuda.is_available():
        try:
            out["torch_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
            out["torch_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 1)
        except Exception:
            pass
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            first = res.stdout.strip().splitlines()[0]
            util, mem = [x.strip() for x in first.split(",")[:2]]
            out["nvidia_util_pct"] = int(util)
            out["nvidia_mem_used_mb"] = int(mem)
    except Exception:
        pass
    return out


def start_episode_telemetry_monitor(
    *,
    episode_title: str,
    partial_path: Path,
    stage_ref: dict,
    interval_sec: int,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    started = time.time()

    def _loop() -> None:
        while not stop_event.wait(max(3, interval_sec)):
            elapsed = int(time.time() - started)
            rss_mb = read_self_rss_mb()
            gpu = gpu_telemetry()
            payload = {
                "episode_title": episode_title,
                "status": "running",
                "stage": stage_ref.get("stage", "unknown"),
                "elapsed_sec": elapsed,
                "updated_at": now_utc(),
                "resources": {
                    "rss_mb": rss_mb,
                    **gpu,
                },
                "segments_available": 0,
                "note": "Diarization is blocking; segments appear after diarization completes.",
            }
            write_json(partial_path, payload)
            LOGGER.info(
                "[resource] "
                f"{episode_title} stage={payload['stage']} elapsed={elapsed}s "
                f"rss_mb={rss_mb} gpu_util={gpu.get('nvidia_util_pct')} "
                f"gpu_mem_mb={gpu.get('nvidia_mem_used_mb')}"
            )

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return stop_event, thread


def load_manifest(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def save_manifest(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_columns(rows: list[dict], fieldnames: list[str], columns: list[str]) -> list[str]:
    out = list(fieldnames)
    for c in columns:
        if c not in out:
            out.append(c)
        for r in rows:
            r.setdefault(c, "")
    return out


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


def ts(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(start: float, end: float, diar_segments: list[dict]) -> tuple[str, float]:
    dur = max(0.001, end - start)
    by_speaker = {}
    for d in diar_segments:
        ov = overlap(start, end, float(d["start"]), float(d["end"]))
        if ov <= 0:
            continue
        spk = d["speaker"]
        by_speaker[spk] = by_speaker.get(spk, 0.0) + ov

    if not by_speaker:
        return "UNKNOWN", 0.0

    ranked = sorted(by_speaker.items(), key=lambda x: x[1], reverse=True)
    best_speaker, best_ov = ranked[0]
    return best_speaker, min(1.0, best_ov / dur)


def should_break(prev_word: dict, cur_word: dict) -> bool:
    if prev_word["speaker"] != cur_word["speaker"]:
        return True
    if (cur_word["start"] - prev_word["end"]) > 0.9:
        return True
    if prev_word["text"].strip().endswith((".", "?", "!")):
        return True
    return False


def words_to_utterances(words: list[dict]) -> list[dict]:
    utterances = []
    cur = None
    for w in words:
        if cur is None:
            cur = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "conf_sum": w["speaker_conf"],
                "conf_n": 1,
                "tokens": [w["text"]],
            }
            continue

        prev_word = {
            "start": cur["end"],
            "end": cur["end"],
            "speaker": cur["speaker"],
            "text": cur["tokens"][-1],
        }
        cur_word = {
            "start": w["start"],
            "end": w["end"],
            "speaker": w["speaker"],
            "text": w["text"],
        }

        if should_break(prev_word, cur_word):
            utterances.append(cur)
            cur = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "conf_sum": w["speaker_conf"],
                "conf_n": 1,
                "tokens": [w["text"]],
            }
        else:
            cur["end"] = w["end"]
            cur["conf_sum"] += w["speaker_conf"]
            cur["conf_n"] += 1
            cur["tokens"].append(w["text"])

    if cur is not None:
        utterances.append(cur)

    cleaned = []
    for u in utterances:
        text = " ".join(t for t in u["tokens"] if t).strip()
        text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        conf = u["conf_sum"] / max(1, u["conf_n"])
        cleaned.append(
            {
                "start": round(u["start"], 3),
                "end": round(u["end"], 3),
                "speaker": u["speaker"],
                "speaker_conf": round(conf, 3),
                "text": text,
            }
        )
    return cleaned


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["start", "end", "speaker", "speaker_conf", "text"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, title: str, utterances: list[dict], partial: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    speakers = sorted({r["speaker"] for r in utterances})

    md = [
        f"# {title} - Speaker Transcript",
        "",
        "- Pipeline: Option A (word-level timestamps + pyannote diarization overlap)",
        "- Labels are speaker IDs only (no name mapping yet)",
        "",
        "## Speaker Colors",
        "",
    ]

    for i, spk in enumerate(speakers):
        color = SPEAKER_COLORS.get(spk, SPEAKER_COLORS.get(f"SPEAKER_0{i}", "#8c564b"))
        md.append(f'- <span style="color:{color}; font-weight:700;">{spk}</span>')

    md.extend(["", "## Transcript", ""])

    for r in utterances:
        color = SPEAKER_COLORS.get(r["speaker"], "#8c564b")
        mark = " (!)" if r["speaker_conf"] < 0.5 else ""
        text_safe = r["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        md.append(
            f'- [{ts(r["start"])}] '
            f'<span style="color:{color}; font-weight:700;">{r["speaker"]}{mark}</span>: '
            f'<span style="color:{color};">{text_safe}</span>'
        )

    if partial:
        md.extend(["", "---", "_Partial output in progress..._"])

    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def load_waveform(audio_path: Path) -> tuple[torch.Tensor, int]:
    try:
        waveform_np, sample_rate = sf.read(str(audio_path), always_2d=True)
        waveform_np = np.asarray(waveform_np, dtype=np.float32).T
        waveform = torch.from_numpy(waveform_np)
        return waveform, int(sample_rate)
    except Exception:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return waveform.float(), int(sample_rate)


def diarize_audio(
    pipeline: Pipeline,
    audio_path: Path,
    min_speakers: int,
    max_speakers: int,
) -> list[dict]:
    waveform, sample_rate = load_waveform(audio_path)
    diarization_output = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    if hasattr(diarization_output, "write_rttm"):
        diarization = diarization_output
    elif hasattr(diarization_output, "speaker_diarization"):
        diarization = diarization_output.speaker_diarization
    elif hasattr(diarization_output, "to_annotation"):
        diarization = diarization_output.to_annotation()
    else:
        raise TypeError(f"Unsupported diarization output type: {type(diarization_output)}")

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": speaker,
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "duration": round(float(turn.duration), 3),
            }
        )

    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(TRANSCRIPTION_ROOT / "manifests" / "pipeline_manifest.csv"))
    parser.add_argument(
        "--audio-dir",
        default=str(TRANSCRIPTION_ROOT / "artifacts" / "01_whisper_transcript" / "audio"),
    )
    parser.add_argument(
        "--out-root",
        default=str(TRANSCRIPTION_ROOT / "artifacts" / "02_diarization"),
    )
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-progress-step", type=int, default=5)
    parser.add_argument("--partial-every-segments", type=int, default=8)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=15)
    parser.add_argument("--telemetry-interval-sec", type=int, default=30)
    parser.add_argument("--log-file", default="", help="Optional log file path.")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    setup_logging(args.log_file)

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None  # type: ignore[attr-defined]

    manifest_path = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    out_root = Path(args.out_root)
    md_dir = out_root / "md"
    diar_dir = out_root / "diarization"
    dbg_dir = out_root / "debug"

    rows, fieldnames = load_manifest(manifest_path)
    fieldnames = ensure_columns(
        rows,
        fieldnames,
        [
            "speaker_status",
            "speaker_md",
            "speaker_diar_json",
            "speaker_word_csv",
            "speaker_segment_csv",
            "speaker_error",
            "speaker_updated_at",
        ],
    )

    eligible = [r for r in rows if (r.get("status") or "").strip() == "done"]
    pending = sum(1 for r in eligible if (r.get("speaker_status") or "").strip() != "done")
    LOGGER.info("Manifest loaded: total=%s eligible=%s to_process=%s redo=%s", len(rows), len(eligible), pending, args.redo)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or get_token()
    LOGGER.info("hf_token_set=%s", bool(hf_token))

    whisper = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
    diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)
    if torch.cuda.is_available() and args.device == "cuda":
        diar_pipeline.to(torch.device("cuda"))
        LOGGER.info("diar_pipeline_device=cuda")
    else:
        LOGGER.info("diar_pipeline_device=cpu")

    attempted = 0
    total_eligible = len(eligible)
    for idx, row in enumerate(rows, start=1):
        if (row.get("status") or "").strip() != "done":
            continue

        speaker_status = (row.get("speaker_status") or "").strip()
        guid = (row.get("guid") or "").strip() or f"row_{idx}"
        title = (row.get("title") or "").strip() or f"Episode {idx}"
        pub_date_iso = (row.get("pub_date_iso") or "").strip()
        base_name = build_base_name(guid, title, pub_date_iso)

        audio_path_str = (row.get("audio_path") or "").strip()
        legacy_guid_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in guid)
        audio_candidates = []
        if audio_path_str:
            audio_candidates.append(Path(audio_path_str))
        audio_candidates.append(audio_dir / f"{base_name}.mp3")
        audio_candidates.append(audio_dir / f"{legacy_guid_name}.mp3")
        audio_path = next((p for p in audio_candidates if p.exists()), audio_candidates[0])

        md_path = md_dir / f"{base_name}.md"
        partial_md_path = md_dir / f"{base_name}.partial.md"
        diar_json_path = diar_dir / f"{base_name}.diarization.json"
        diar_partial_path = diar_dir / f"{base_name}.diarization.partial.json"
        diar_rttm_path = diar_dir / f"{base_name}.rttm"
        word_csv_path = dbg_dir / f"{base_name}.words.csv"
        seg_csv_path = dbg_dir / f"{base_name}.segments.csv"

        if not args.redo and speaker_status == "done" and md_path.exists():
            continue

        monitor_stop = None
        monitor_thread = None
        stage_ref = {"stage": "idle"}
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio file: {audio_path}")

            LOGGER.info("[%s] Start: %s", attempted + 1, title)
            stage_ref["stage"] = "diarization"
            monitor_stop, monitor_thread = start_episode_telemetry_monitor(
                episode_title=title,
                partial_path=diar_partial_path,
                stage_ref=stage_ref,
                interval_sec=max(5, args.telemetry_interval_sec),
            )
            write_json(
                diar_partial_path,
                {
                    "episode_title": title,
                    "status": "running",
                    "stage": "diarization",
                    "updated_at": now_utc(),
                    "segments_available": 0,
                    "note": "Diarization started; segments become available after diarization completes.",
                },
            )
            LOGGER.info("[stage] diarization: %s", title)
            diar_segments = diarize_audio(
                diar_pipeline,
                audio_path,
                min_speakers=max(1, args.min_speakers),
                max_speakers=max(1, args.max_speakers),
            )

            write_json(diar_json_path, {"segments": diar_segments})
            write_json(
                diar_partial_path,
                {
                    "episode_title": title,
                    "status": "running",
                    "stage": "diarization_completed",
                    "updated_at": now_utc(),
                    "segments_available": len(diar_segments),
                    "segments_preview_count": min(10, len(diar_segments)),
                    "segments_preview": diar_segments[:10],
                },
            )

            with diar_rttm_path.open("w", encoding="utf-8") as f:
                for seg in diar_segments:
                    dur = max(0.0, float(seg["end"]) - float(seg["start"]))
                    f.write(
                        "SPEAKER file 1 "
                        f"{float(seg['start']):.3f} {dur:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n"
                    )

            stage_ref["stage"] = "asr+alignment"
            LOGGER.info("[stage] asr+alignment: %s", title)
            segments_iter, info = whisper.transcribe(
                str(audio_path),
                word_timestamps=True,
                vad_filter=True,
                beam_size=5,
            )

            total_duration = float(info.duration or 0.0)
            next_mark = max(1, min(25, args.episode_progress_step))
            words: list[dict] = []
            seg_counter = 0

            for seg in segments_iter:
                seg_counter += 1
                for w in seg.words or []:
                    if w.start is None or w.end is None:
                        continue
                    speaker, conf = assign_speaker(float(w.start), float(w.end), diar_segments)
                    words.append(
                        {
                            "start": float(w.start),
                            "end": float(w.end),
                            "speaker": speaker,
                            "speaker_conf": round(conf, 3),
                            "text": (w.word or "").strip(),
                        }
                    )

                if seg_counter % max(1, args.partial_every_segments) == 0 and words:
                    partial_utt = words_to_utterances(words)
                    write_markdown(partial_md_path, title, partial_utt, partial=True)

                if total_duration > 0:
                    pct = int(min(100, max(0, (float(seg.end) / total_duration) * 100)))
                    while pct >= next_mark and next_mark <= 100:
                        LOGGER.info("[episode-progress] %s: %s%%", title, next_mark)
                        next_mark += max(1, min(25, args.episode_progress_step))

            if next_mark <= 100:
                LOGGER.info("[episode-progress] %s: 100%%", title)

            if not words:
                raise RuntimeError("No words produced by ASR.")

            utterances = words_to_utterances(words)
            write_csv(word_csv_path, words)
            write_csv(seg_csv_path, utterances)
            write_markdown(md_path, title, utterances)
            write_json(
                diar_partial_path,
                {
                    "episode_title": title,
                    "status": "done",
                    "stage": "completed",
                    "updated_at": now_utc(),
                    "segments_available": len(diar_segments),
                    "speaker_words": len(words),
                    "speaker_utterances": len(utterances),
                },
            )
            if partial_md_path.exists():
                partial_md_path.unlink()
            if diar_partial_path.exists():
                diar_partial_path.unlink()

            row["speaker_status"] = "done"
            row["speaker_md"] = str(md_path)
            row["speaker_diar_json"] = str(diar_json_path)
            row["speaker_word_csv"] = str(word_csv_path)
            row["speaker_segment_csv"] = str(seg_csv_path)
            row["speaker_error"] = ""
            row["speaker_updated_at"] = now_utc()
            LOGGER.info("[done] %s words=%s utterances=%s", title, len(words), len(utterances))
        except Exception as exc:
            row["speaker_status"] = "error"
            row["speaker_error"] = f"{type(exc).__name__}: {exc}"
            row["speaker_updated_at"] = now_utc()
            LOGGER.error("[error] %s: %s", title, row["speaker_error"])
            LOGGER.error(traceback.format_exc())
            try:
                write_json(
                    diar_partial_path,
                    {
                        "episode_title": title,
                        "status": "error",
                        "stage": "error",
                        "updated_at": now_utc(),
                        "error": row["speaker_error"],
                    },
                )
            except Exception:
                pass
        finally:
            if monitor_stop is not None:
                monitor_stop.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=1.0)
            save_manifest(manifest_path, rows, fieldnames)
            done_count = sum(1 for r in rows if (r.get("speaker_status") or "").strip() == "done")
            err_count = sum(1 for r in rows if (r.get("speaker_status") or "").strip() == "error")
            elig_pending = sum(
                1
                for r in rows
                if (r.get("status") or "").strip() == "done"
                and (r.get("speaker_status") or "").strip() not in {"done", "error"}
            )
            LOGGER.info("[progress] speaker_done=%s speaker_error=%s speaker_pending=%s", done_count, err_count, elig_pending)

        attempted += 1
        if args.max_episodes > 0 and attempted >= args.max_episodes:
            break

    LOGGER.info("Finished speaker batch. Episodes attempted: %s / eligible=%s", attempted, total_eligible)


if __name__ == "__main__":
    main()
