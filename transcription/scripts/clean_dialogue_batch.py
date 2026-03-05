#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTION_ROOT = REPO_ROOT / "transcription"
ARTIFACTS_ROOT = TRANSCRIPTION_ROOT / "artifacts"
DIARIZATION_ROOT = ARTIFACTS_ROOT / "02_diarization"
DEBUG_DIR = DIARIZATION_ROOT / "debug"
PYTHON_ROOT = ARTIFACTS_ROOT / "03_clean_python"
PYTHON_OUT_DIR = PYTHON_ROOT / "md"
PYTHON_JSON_DIR = PYTHON_ROOT / "json"
LLM_ROOT = ARTIFACTS_ROOT / "04_clean_llm"
META_DIR = LLM_ROOT / "meta"
RAW_DIR = LLM_ROOT / "raw"
CLEAN_JSON_DIR = LLM_ROOT / "json"
WEB_ROOT = ARTIFACTS_ROOT / "05_webformat"
WEB_MD_DIR = WEB_ROOT / "md"
ARCHIVE_DIR = ARTIFACTS_ROOT / "old" / "clean_dialogue_archive"
LOG_DIR = TRANSCRIPTION_ROOT / "logs"
MANIFEST_PATH = TRANSCRIPTION_ROOT / "manifests" / "pipeline_manifest.csv"
SITE_EPISODES_DIR = REPO_ROOT / "episodes"

PODCAST_SPOTIFY_SHOW_URL = "https://open.spotify.com/show/0bmymUJs50SVO1WkqfrccB"
PODCAST_APPLE_SHOW_URL = "https://podcasts.apple.com/us/podcast/the-theory-of-anything/id1503194218"
PODCAST_TRANSCRIPTOR_URL = "https://github.com/alik-git/PodcastTranscriptor"

MANIFEST_CLEAN_COLUMNS = [
    "clean_python_status",
    "clean_python_md",
    "clean_python_json",
    "clean_python_error",
    "clean_python_updated_at",
    "clean_llm_status",
    "clean_llm_md",
    "clean_llm_json",
    "clean_llm_raw_json",
    "clean_llm_meta",
    "clean_llm_model",
    "clean_llm_error",
    "clean_llm_updated_at",
]

COLOR_NAMES = [
    "Blue",
    "Red",
    "Green",
    "Orange",
    "Purple",
    "Teal",
    "Brown",
    "Pink",
    "Gray",
    "Cyan",
    "Gold",
    "Indigo",
    "Magenta",
    "Olive",
    "Navy",
]

SPEAKER_COLOR_HEX = {
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "orange": "#ea580c",
    "purple": "#9333ea",
    "teal": "#0f766e",
    "brown": "#92400e",
    "pink": "#db2777",
    "gray": "#4b5563",
    "cyan": "#0891b2",
    "gold": "#ca8a04",
    "indigo": "#4f46e5",
    "magenta": "#c026d3",
    "olive": "#65a30d",
    "navy": "#1e3a8a",
    "unknown": "#6b7280",
}

CONNECTOR_STARTS = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "that",
    "which",
    "who",
    "what",
    "when",
    "where",
    "if",
    "then",
    "also",
    "well",
    "you",
    "i",
    "we",
    "they",
    "he",
    "she",
}

SYSTEM_PROMPT_LABEL_CORRECTION = """
You are a speaker-label correction assistant for AI generated transcripts.
Do NOT rewrite transcript text. Do NOT remove/add/reorder spoken words.
Your job is to identify mislabeled speaker tags in CORE_TO_REVIEW.
The model used to generate the transcripts often makes mistakes, you have to correct them.
Most often, a word or few words (especially those at the beginning or the end of a sentence) are mislabeled as the wrong speaker (especially as "Unknown").
You must use your common sense understanding of the surrounding context to correct the mistake. Most often you need to merge the mislabeled word or words by assigning them to the speaker of the sentence before or after the mislabeled word or words, based on context.
Here are some examples.
1. In a sentence that is clearly by one speaker, a single word or a few words are labeled another speaker, or an Unknown speaker.
Example A input:
1) Blue: Hello, welcome
2) Red: to
3) Blue: the podcast
Example A output:
{"corrections": [{"line": 2, "speaker": "Blue"}]}
Example B input:
1) Blue: This week on the podcast,
2) Unknown: Bruce makes
3) Blue: his most epic statement...
Example B output:
{"corrections": [{"line": 2, "speaker": "Blue"}]}
2. When multiple people are speaking, some words get mislabeled as the wrong speaker, which is clear from the context.
Example C input:
1) Unknown: And
2) Red: so it sounds like they're just
3) Unknown: But
4) Red: I'm pretty sure they said they were quoting that source.
5) Unknown: So
6) Red: that's why he did.
7) Blue: Yeah.
8) Red: So
9) Blue: that's why Dan said that.
Example C output:
{"corrections": [{"line": 1, "speaker": "Red"}, {"line": 3, "speaker": "Red"}, {"line": 5, "speaker": "Red"}, {"line": 8, "speaker": "Blue"}]}
Words like "Yeah", "Sure", "Okay", "Maybe", "Right", "Makes Sense", "Gotcha", "Interesting", "Hmm", can be spoken as a full sentence so likely should not be merged even if surrounded by different speakers.
Words like "is", "so", "I", "the", "as", "was", "many", "to", are likely not spoken as a full sentence and should be merged if surrounded by different speakers.
Use surrounding context to infer whether each line's speaker label is correct.
Unknown labels with only 1-3 words should usually be reassigned based on context.
Non-Unknown labels can also be wrong; correct them when context clearly indicates a different speaker.
Use only speakers from Allowed speakers.
Return JSON only in this format: {"corrections": [{"line": 12, "speaker": "Red"}]}
Only include lines that should change. If none: {"corrections": []}
""".strip()

_LOG_FP = None


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log(message: str) -> None:
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    if _LOG_FP is not None:
        _LOG_FP.write(line + "\n")
        _LOG_FP.flush()


def local_run_stamp() -> str:
    # Human-readable and lexicographically sortable (local timezone).
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")


def slug_base_to_title(base_name: str) -> str:
    parts = base_name.split("__")
    if len(parts) >= 2:
        return parts[1].replace("-", " ").strip().title()
    return base_name


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    text = text.replace(" ;", ";").replace(" :", ":")
    return text


def sec_to_hms(sec: float) -> str:
    s = max(0, int(round(float(sec))))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def parse_hms_to_sec(value) -> int:
    if isinstance(value, (int, float)):
        return max(0, int(round(float(value))))
    t = str(value or "").strip()
    if not t:
        raise ValueError("empty timestamp")
    if re.fullmatch(r"\d+", t):
        return max(0, int(t))
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", t)
    if not m:
        raise ValueError(f"invalid timestamp: {value}")
    hh, mm, ss = map(int, m.groups())
    return max(0, hh * 3600 + mm * 60 + ss)


def is_strong_sentence_end(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] in ".?!"


def starts_with_connector(text: str) -> bool:
    m = re.match(r"[A-Za-z']+", text.strip().lower())
    if not m:
        return False
    return m.group(0) in CONNECTOR_STARTS


def map_speaker_names(rows: list[dict]) -> dict[str, str]:
    order = []
    for r in rows:
        spk = r["speaker"]
        if spk not in order and spk != "UNKNOWN":
            order.append(spk)

    mapping = {"UNKNOWN": "Unknown"}
    for i, spk in enumerate(order):
        if i < len(COLOR_NAMES):
            mapping[spk] = COLOR_NAMES[i]
        else:
            mapping[spk] = f"Color{i + 1}"
    return mapping


def should_merge(prev: dict, cur: dict, max_gap_sec: float) -> bool:
    if prev["speaker"] != cur["speaker"]:
        return False
    gap = max(0.0, cur["start"] - prev["end"])
    if gap > max_gap_sec:
        return False

    prev_text = prev["text"]
    cur_text = cur["text"]
    if not is_strong_sentence_end(prev_text):
        return True
    if starts_with_connector(cur_text):
        return True
    if len(cur_text.split()) <= 4:
        return True
    return False


def merge_rows(rows: list[dict], max_gap_sec: float) -> list[dict]:
    merged: list[dict] = []
    for r in rows:
        if not merged:
            merged.append(dict(r))
            continue
        prev = merged[-1]
        if should_merge(prev, r, max_gap_sec=max_gap_sec):
            prev["text"] = normalize_text(f"{prev['text']} {r['text']}")
            prev["end"] = max(prev["end"], r["end"])
            prev["speaker_conf"] = round((prev["speaker_conf"] + r["speaker_conf"]) / 2.0, 3)
        else:
            merged.append(dict(r))
    return merged


def read_segments_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = normalize_text(r.get("text", ""))
            if not text:
                continue
            rows.append(
                {
                    "start": float(r.get("start", 0.0) or 0.0),
                    "end": float(r.get("end", 0.0) or 0.0),
                    "speaker": (r.get("speaker") or "UNKNOWN").strip() or "UNKNOWN",
                    "speaker_conf": float(r.get("speaker_conf", 0.0) or 0.0),
                    "text": text,
                }
            )
    rows.sort(key=lambda x: (x["start"], x["end"]))
    return rows


def render_clean_md(title: str, turns: list[dict], speaker_map: dict[str, str], source_name: str) -> str:
    out = [
        f"# {title} - Clean Transcript",
        "",
        f"- Source: `{source_name}`",
        "- Speakers are anonymized as color names.",
        "- Goal: preserve wording while improving readability.",
        "",
        "## Transcript",
        "",
    ]
    for t in turns:
        speaker_name = speaker_map.get(t["speaker"], t["speaker"])
        text = t["text"].strip()
        start_sec = int(round(float(t.get("start", 0.0) or 0.0)))
        if not text:
            continue
        out.append(f"[{sec_to_hms(start_sec)}] {speaker_name}: {text}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_named_turns_md(
    title: str,
    turns: list[dict],
    spotify_url: str,
    apple_url: str,
) -> str:
    spotify_link = spotify_url or PODCAST_SPOTIFY_SHOW_URL
    apple_link = apple_url or PODCAST_APPLE_SHOW_URL
    out = [
        f"# {title}",
        "",
        f"- Links to this episode: [Spotify]({spotify_link}) / [Apple Podcasts]({apple_link})",
        f"- This transcript was generated with AI using [PodcastTranscriptor]({PODCAST_TRANSCRIPTOR_URL}).",
        "- **It may contain mistakes.** Please check against the actual podcast.",
        "- Speakers are denoted as color names.",
    ]
    out.extend(
        [
            "",
            "## Transcript",
            "",
        ]
    )
    for t in turns:
        speaker_name = t["speaker_name"]
        text = normalize_text(t["text"])
        start_sec = int(round(float(t.get("timestamp_sec", 0) or 0)))
        if not text:
            continue
        color = SPEAKER_COLOR_HEX.get(str(speaker_name).strip().lower(), "#374151")
        ts = f"<em>[{sec_to_hms(start_sec)}]</em>"
        spk = f"<strong><span style=\"color:{color}\">{speaker_name}:</span></strong>"
        out.append(f"{ts}&nbsp;&nbsp;{spk} {text}")
        out.append("")
    out.extend(
        [
            "---",
            "",
            f"*Links to this episode:* [Spotify]({spotify_link}) / [Apple Podcasts]({apple_link})",
            "",
            f"*Generated with AI using [PodcastTranscriptor]({PODCAST_TRANSCRIPTOR_URL}). May contain mistakes; please verify against the actual podcast.*",
            "",
        ]
    )
    return "\n".join(out).rstrip() + "\n"


def infer_episode_links(title: str, manifest_row: dict | None) -> tuple[str, str]:
    row = manifest_row or {}
    spotify = (row.get("spotify_url") or "").strip()
    apple = (row.get("apple_url") or "").strip()
    if not spotify:
        spotify = PODCAST_SPOTIFY_SHOW_URL
    if not apple:
        apple = PODCAST_APPLE_SHOW_URL
    return spotify, apple


def write_site_episode_page(base: str, title: str, body_md: str) -> None:
    SITE_EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    parts = base.split("__")
    date_part = parts[0] if parts else "unknown-date"
    slug_part = parts[1] if len(parts) >= 2 else base
    out_name = f"{date_part}-{slug_part}.md"
    out_path = SITE_EPISODES_DIR / out_name

    m = re.search(r"Episode\s+(\d+)", title, flags=re.IGNORECASE)
    nav_order = int(m.group(1)) if m else 9999
    safe_title = title.replace('"', '\\"')
    permalink_line = f"permalink: /episodes/{nav_order}/\n" if m else ""

    front_matter = (
        "---\n"
        "layout: default\n"
        f"title: \"{safe_title}\"\n"
        "parent: Episodes\n"
        f"nav_order: {nav_order}\n"
        f"{permalink_line}"
        "---\n\n"
    )
    out_path.write_text(front_matter + body_md, encoding="utf-8")


def chunk_turns_for_llm(turns: list[dict], max_chars: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_chars = 0
    for t in turns:
        line = f"{t['speaker_name']}: {t['text']}\n"
        if cur and cur_chars + len(line) > max_chars:
            chunks.append(cur)
            cur = []
            cur_chars = 0
        cur.append(t)
        cur_chars += len(line)
    if cur:
        chunks.append(cur)
    return chunks


def merge_same_speaker_turns_under_cap(turns: list[dict], max_words: int = 300) -> tuple[list[dict], dict]:
    cap = max(20, int(max_words))
    out: list[dict] = []
    merges_applied = 0
    max_words_before = max([count_words(t.get("text", "")) for t in turns] or [0])

    for t in turns:
        speaker = t.get("speaker_name", "Unknown")
        text = normalize_text(str(t.get("text", "")))
        if not text:
            continue
        ts = int(round(float(t.get("timestamp_sec", 0) or 0)))
        if not out:
            out.append({"speaker_name": speaker, "timestamp_sec": ts, "text": text})
            continue

        prev = out[-1]
        if prev["speaker_name"] != speaker:
            out.append({"speaker_name": speaker, "timestamp_sec": ts, "text": text})
            continue

        prev_words = count_words(prev["text"])
        cur_words = count_words(text)
        if prev_words + cur_words <= cap:
            prev["text"] = normalize_text(f"{prev['text']} {text}")
            merges_applied += 1
        else:
            out.append({"speaker_name": speaker, "timestamp_sec": ts, "text": text})

    max_words_after = max([count_words(t.get("text", "")) for t in out] or [0])
    stats = {
        "word_cap": cap,
        "turns_before_merge_cap": len(turns),
        "turns_after_merge_cap": len(out),
        "same_speaker_merges_applied": merges_applied,
        "max_turn_words_before_merge_cap": max_words_before,
        "max_turn_words_after_merge_cap": max_words_after,
    }
    return out, stats


@dataclass
class LLMConfig:
    model: str
    temperature: float
    max_chars_per_chunk: int
    max_words_per_chunk: int
    overlap_words: int
    chunk_sentence_overrun_words: int
    request_timeout_sec: float
    max_retries: int
    retry_backoff_sec: float


def words_of(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def count_words(text: str) -> int:
    return len(words_of(text))


def last_n_words(text: str, n: int) -> str:
    toks = re.findall(r"\S+", (text or "").strip())
    if not toks:
        return ""
    return " ".join(toks[-max(1, n):])


def build_core_chunks_by_words(
    turns: list[dict],
    max_words_per_chunk: int,
    sentence_overrun_words: int,
) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    n = len(turns)
    i = 0
    while i < n:
        w = 0
        j = i
        while j < n:
            t_words = max(1, count_words(turns[j]["text"]))
            if j > i and w + t_words > max_words_per_chunk:
                break
            w += t_words
            j += 1
        # Soft extension: avoid awkward cutpoints by extending to a sentence end,
        # with a bounded overrun to keep chunk sizes predictable.
        over = 0
        while j < n and over < max(0, sentence_overrun_words):
            last_text = turns[j - 1]["text"] if j - 1 >= i else ""
            if is_strong_sentence_end(last_text):
                break
            t_words = max(1, count_words(turns[j]["text"]))
            over += t_words
            j += 1
        if j == i:
            j = min(i + 1, n)
        chunks.append((i, j))
        i = j
    return chunks


def context_start_by_words(turns: list[dict], core_start: int, overlap_words: int) -> int:
    if overlap_words <= 0:
        return core_start
    i = core_start
    w = 0
    while i > 0 and w < overlap_words:
        i -= 1
        w += max(1, count_words(turns[i]["text"]))
    return i


def context_end_by_words(turns: list[dict], core_end: int, overlap_words: int) -> int:
    if overlap_words <= 0:
        return core_end
    j = core_end
    w = 0
    n = len(turns)
    while j < n and w < overlap_words:
        w += max(1, count_words(turns[j]["text"]))
        j += 1
    return j


def speaker_stats(turns: list[dict]) -> dict:
    per_speaker = {}
    all_words = []
    for t in turns:
        spk = t["speaker_name"]
        w = words_of(t["text"])
        all_words.extend(w)
        if spk not in per_speaker:
            per_speaker[spk] = {"turns": 0, "words": 0}
        per_speaker[spk]["turns"] += 1
        per_speaker[spk]["words"] += len(w)
    return {
        "turns_total": len(turns),
        "words_total": len(all_words),
        "per_speaker": per_speaker,
        "all_words": all_words,
    }


def compare_turn_sets(before_turns: list[dict], after_turns: list[dict]) -> dict:
    b = speaker_stats(before_turns)
    a = speaker_stats(after_turns)
    b_words = b["all_words"]
    a_words = a["all_words"]
    seq_ratio = SequenceMatcher(a=" ".join(b_words), b=" ".join(a_words)).ratio() if b_words or a_words else 1.0
    b_count = Counter(b_words)
    a_count = Counter(a_words)
    removed = sum((b_count - a_count).values())
    added = sum((a_count - b_count).values())

    spk_delta = {}
    speakers = sorted(set(b["per_speaker"]) | set(a["per_speaker"]))
    for s in speakers:
        b_s = b["per_speaker"].get(s, {"turns": 0, "words": 0})
        a_s = a["per_speaker"].get(s, {"turns": 0, "words": 0})
        spk_delta[s] = {
            "turns_before": b_s["turns"],
            "turns_after": a_s["turns"],
            "words_before": b_s["words"],
            "words_after": a_s["words"],
            "turns_delta": a_s["turns"] - b_s["turns"],
            "words_delta": a_s["words"] - b_s["words"],
        }

    words_before = max(1, b["words_total"])
    return {
        "turns_before": b["turns_total"],
        "turns_after": a["turns_total"],
        "words_before": b["words_total"],
        "words_after": a["words_total"],
        "word_delta_pct": round((a["words_total"] - b["words_total"]) / words_before, 4),
        "sequence_similarity": round(seq_ratio, 4),
        "removed_word_count": int(removed),
        "added_word_count": int(added),
        "per_speaker_delta": spk_delta,
    }


def quality_flags(stats: dict) -> list[str]:
    flags = []
    if abs(stats["word_delta_pct"]) > 0.12:
        flags.append("large_word_count_shift")
    if stats["sequence_similarity"] < 0.88:
        flags.append("low_sequence_similarity")
    if stats["removed_word_count"] > max(80, int(0.08 * max(1, stats["words_before"]))):
        flags.append("high_removed_word_count")
    return flags


def parse_label_corrections(raw_text: str, allowed_speakers: set[str], core_turn_count: int) -> tuple[list[dict], bool, str]:
    try:
        obj = json.loads(extract_json_object(raw_text))
        corr = obj.get("corrections", [])
        if not isinstance(corr, list):
            raise ValueError("corrections is not a list")
        out = []
        for c in corr:
            if not isinstance(c, dict):
                continue
            line_val = c.get("line", c.get("core_index", None))
            speaker = str(c.get("speaker", "")).strip()
            try:
                line = int(line_val)
            except Exception:
                continue
            if line < 1 or line > core_turn_count:
                continue
            if speaker not in allowed_speakers:
                speaker = "Unknown"
            out.append({"line": line, "speaker": speaker})
        return out, True, ""
    except Exception as exc:
        return [], False, f"{type(exc).__name__}: {exc}"


def apply_label_corrections(core_chunk: list[dict], corrections: list[dict]) -> tuple[list[dict], dict]:
    out = [
        {
            "speaker_name": t["speaker_name"],
            "timestamp_sec": int(round(float(t.get("timestamp_sec", 0) or 0))),
            "text": t["text"],
        }
        for t in core_chunk
    ]
    requested = len(corrections)
    applied = 0
    for c in corrections:
        idx = int(c["line"]) - 1
        if idx < 0 or idx >= len(out):
            continue
        new_speaker = c["speaker"]
        if out[idx]["speaker_name"] != new_speaker:
            out[idx]["speaker_name"] = new_speaker
            applied += 1
    return out, {"requested": requested, "applied": applied}


def summarize_label_corrections(chunk_records: list[dict]) -> dict:
    total = len(chunk_records)
    parse_fail = 0
    requested = 0
    applied = 0
    for r in chunk_records:
        if not r.get("parse_ok", True):
            parse_fail += 1
        cs = r.get("correction_stats") or {}
        requested += int(cs.get("requested", 0) or 0)
        applied += int(cs.get("applied", 0) or 0)
    return {
        "chunks_completed": total,
        "parse_failures": parse_fail,
        "corrections_requested": requested,
        "corrections_applied": applied,
    }


def label_correction_notes(summary: dict) -> list[str]:
    return [
        "Label correction summary: "
        f"parse_failures={summary.get('parse_failures', 0)}, "
        f"corrections_requested={summary.get('corrections_requested', 0)}, "
        f"corrections_applied={summary.get('corrections_applied', 0)}"
    ]


def extract_json_object(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM output did not contain a JSON object.")
    return t[start : end + 1]


def parse_llm_turns_json(raw_text: str, allowed_speakers: set[str], fallback_chunk: list[dict]) -> list[dict]:
    try:
        obj = json.loads(extract_json_object(raw_text))
        turns = obj.get("turns", [])
        if not isinstance(turns, list):
            raise ValueError("turns is not a list")
        parsed = []
        for t in turns:
            speaker = str(t.get("speaker", "")).strip()
            text = normalize_text(str(t.get("text", "")))
            ts_val = t.get("timestamp", t.get("timestamp_sec", ""))
            if not text:
                continue
            if speaker not in allowed_speakers:
                speaker = "Unknown"
            try:
                ts_sec = parse_hms_to_sec(ts_val)
            except Exception:
                ts_sec = int(round(float(fallback_chunk[0].get("timestamp_sec", 0) or 0)))
            parsed.append({"speaker_name": speaker, "timestamp_sec": ts_sec, "text": text})
        if not parsed:
            raise ValueError("No parsed turns from LLM output.")
        return parsed
    except Exception:
        return [
            {
                "speaker_name": t["speaker_name"],
                "timestamp_sec": int(round(float(t.get("timestamp_sec", 0) or 0))),
                "text": t["text"],
            }
            for t in fallback_chunk
        ]


def llm_cleanup_turns(
    turns: list[dict],
    cfg: LLMConfig,
    on_chunk_done: Callable[[list[dict], int, int, int, int, dict], None] | None = None,
) -> tuple[list[dict], int, int, list[dict]]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai package not installed in current environment.") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(timeout=max(5.0, cfg.request_timeout_sec))
    cleaned_turns: list[dict] = []
    raw_chunks: list[dict] = []
    in_tokens = 0
    out_tokens = 0
    allowed = sorted({t["speaker_name"] for t in turns})
    core_chunks = build_core_chunks_by_words(
        turns,
        max_words_per_chunk=max(80, cfg.max_words_per_chunk),
        sentence_overrun_words=max(0, cfg.chunk_sentence_overrun_words),
    )

    system_prompt = SYSTEM_PROMPT_LABEL_CORRECTION

    for idx, (core_start, core_end) in enumerate(core_chunks, start=1):
        cstart = context_start_by_words(turns, core_start, overlap_words=max(0, cfg.overlap_words))
        cend = context_end_by_words(turns, core_end, overlap_words=max(0, cfg.overlap_words))
        context_before = turns[cstart:core_start]
        core_chunk = turns[core_start:core_end]
        context_after = turns[core_end:cend]
        prev_clean_tail = "(none)"
        if cleaned_turns:
            prev_clean_tail = last_n_words(cleaned_turns[-1]["text"], 25) or "(none)"
        before_text = "\n".join(f"{t['speaker_name']}: {t['text']}" for t in context_before) or "(none)"
        core_text = "\n".join(
            f"{i + 1}) {t['speaker_name']}: {t['text']}"
            for i, t in enumerate(core_chunk)
        )
        after_text = "\n".join(f"{t['speaker_name']}: {t['text']}" for t in context_after) or "(none)"
        user_prompt = (
            f"Chunk {idx}/{len(core_chunks)}\n"
            f"Allowed speakers: {', '.join(allowed)}\n"
            "You will receive context before and after, plus a numbered CORE segment to review.\n"
            "Use context for disambiguation, but output JSON corrections for CORE only.\n"
            "PREVIOUS_CLEAN_TAIL is from the already-cleaned prior chunk and is provided "
            "only to improve continuation consistency.\n"
            "Do NOT rewrite text; only propose speaker label corrections by line number.\n\n"
            f"PREVIOUS_CLEAN_TAIL (read-only):\n{prev_clean_tail}\n\n"
            f"CONTEXT_BEFORE (read-only):\n{before_text}\n\n"
            f"CORE_TO_REVIEW (output corrections for this section only):\n{core_text}\n\n"
            f"CONTEXT_AFTER (read-only):\n{after_text}\n"
        )
        request_input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def call_llm(req_input: list[dict]):
            local_resp = None
            for attempt in range(1, max(1, cfg.max_retries) + 1):
                try:
                    local_resp = client.responses.create(
                        model=cfg.model,
                        temperature=cfg.temperature,
                        input=req_input,
                    )
                    break
                except Exception as exc:
                    msg = str(exc)
                    if "Unsupported parameter: 'temperature'" in msg:
                        local_resp = client.responses.create(
                            model=cfg.model,
                            input=req_input,
                        )
                        break
                    if attempt >= max(1, cfg.max_retries):
                        raise
                    sleep_s = max(0.5, cfg.retry_backoff_sec * attempt)
                    log(
                        f"[llm] chunk {idx}/{len(core_chunks)} retry {attempt}/{cfg.max_retries} "
                        f"after error={type(exc).__name__}: {exc} (sleep {sleep_s:.1f}s)"
                    )
                    time.sleep(sleep_s)
            if local_resp is None:
                raise RuntimeError("LLM call failed without a response.")
            return local_resp

        log(
            f"[llm] chunk {idx}/{len(core_chunks)} start "
            f"(core_turns={len(core_chunk)} context_before={len(context_before)} context_after={len(context_after)})"
        )
        resp = call_llm(request_input)
        text = (resp.output_text or "").strip()
        corrections, parse_ok, parse_error = parse_label_corrections(
            text,
            set(allowed),
            core_turn_count=len(core_chunk),
        )
        parsed, correction_stats = apply_label_corrections(core_chunk, corrections)
        if not parse_ok:
            log(f"[warn] chunk {idx}/{len(core_chunks)} correction_parse_failed error={parse_error}")
        if correction_stats["applied"] > 0:
            log(
                f"[llm] chunk {idx}/{len(core_chunks)} label_corrections_applied "
                f"requested={correction_stats['requested']} applied={correction_stats['applied']}"
            )

        cleaned_turns.extend(parsed)

        usage = getattr(resp, "usage", None)
        if usage is not None:
            in_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            out_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        log(
            f"[llm] chunk {idx}/{len(core_chunks)} done "
            f"(parsed_turns={len(parsed)} total_in_tokens={in_tokens} total_out_tokens={out_tokens})"
        )
        chunk_record = {
            "chunk_index": idx,
            "chunk_total": len(core_chunks),
            "core_turn_count": len(core_chunk),
            "context_before_turn_count": len(context_before),
            "context_after_turn_count": len(context_after),
            "response_text": text,
            "parsed_turns": parsed,
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "corrections": corrections,
            "correction_stats": correction_stats,
            "total_input_tokens_after_chunk": in_tokens,
            "total_output_tokens_after_chunk": out_tokens,
        }
        raw_chunks.append(chunk_record)
        if on_chunk_done is not None:
            on_chunk_done(cleaned_turns, idx, len(core_chunks), in_tokens, out_tokens, chunk_record)

    return cleaned_turns, in_tokens, out_tokens, raw_chunks


def ensure_dirs() -> None:
    PYTHON_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PYTHON_JSON_DIR.mkdir(parents=True, exist_ok=True)
    WEB_MD_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_JSON_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def archive_existing(path: Path, category: str) -> None:
    if not path.exists():
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")
    target_dir = ARCHIVE_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{path.name}.{stamp}.bak"
    shutil.move(str(path), str(target))
    log(f"[archive] moved {path} -> {target}")


def _ensure_manifest_columns(fieldnames: list[str]) -> list[str]:
    out = list(fieldnames)
    for c in MANIFEST_CLEAN_COLUMNS:
        if c not in out:
            out.append(c)
    return out


def _episode_num_from_text(text: str) -> str:
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
        fieldnames = _ensure_manifest_columns(list(reader.fieldnames or []))
    by_seg_csv_name: dict[str, dict] = {}
    by_episode_num: dict[str, dict] = {}
    for r in rows:
        seg_path = (r.get("speaker_segment_csv") or "").strip()
        if seg_path:
            by_seg_csv_name[Path(seg_path).name] = r
        ep_num = _episode_num_from_text((r.get("title") or ""))
        if ep_num and ep_num not in by_episode_num:
            by_episode_num[ep_num] = r
    return fieldnames, rows, by_seg_csv_name, by_episode_num


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not fieldnames:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            for c in fieldnames:
                out.setdefault(c, "")
            writer.writerow(out)
    tmp.replace(path)


def _state_for_manifest(state: str) -> str:
    if state in {"written", "done"}:
        return "done"
    if state in {"skipped", "not_requested", "error"}:
        return state
    return state or ""


def update_manifest_row(
    row: dict,
    *,
    mode: str,
    python_state: str,
    llm_state: str,
    python_out: Path,
    python_json_out: Path,
    llm_out: Path,
    llm_json_out: Path,
    llm_raw_out: Path,
    meta_out: Path,
    llm_model: str,
    error_text: str = "",
) -> None:
    ts = now_utc()
    if mode in {"python", "both"}:
        row["clean_python_status"] = _state_for_manifest(python_state)
        row["clean_python_updated_at"] = ts
        row["clean_python_md"] = str(python_out) if python_out.exists() else row.get("clean_python_md", "")
        row["clean_python_json"] = str(python_json_out) if python_json_out.exists() else row.get("clean_python_json", "")
        row["clean_python_error"] = error_text if _state_for_manifest(python_state) == "error" else ""
    if mode in {"llm", "both"}:
        row["clean_llm_status"] = _state_for_manifest(llm_state)
        row["clean_llm_updated_at"] = ts
        row["clean_llm_model"] = llm_model
        row["clean_llm_md"] = str(llm_out) if llm_out.exists() else row.get("clean_llm_md", "")
        row["clean_llm_json"] = str(llm_json_out) if llm_json_out.exists() else row.get("clean_llm_json", "")
        row["clean_llm_raw_json"] = str(llm_raw_out) if llm_raw_out.exists() else row.get("clean_llm_raw_json", "")
        row["clean_llm_meta"] = str(meta_out) if meta_out.exists() else row.get("clean_llm_meta", "")
        row["clean_llm_error"] = error_text if _state_for_manifest(llm_state) == "error" else ""


def base_from_segments_path(path: Path) -> str:
    name = path.name
    if name.endswith(".segments.csv"):
        return name[: -len(".segments.csv")]
    return path.stem


def process_episode(
    seg_csv: Path,
    mode: str,
    max_gap_sec: float,
    llm_cfg: LLMConfig,
    redo: bool,
    manifest_row: dict | None = None,
) -> tuple[str, str]:
    base = base_from_segments_path(seg_csv)
    title = slug_base_to_title(base)
    python_out = PYTHON_OUT_DIR / f"{base}.clean.md"
    python_json_out = PYTHON_JSON_DIR / f"{base}.clean.json"
    web_md_out = WEB_MD_DIR / f"{base}.clean.md"
    web_partial_out = WEB_MD_DIR / f"{base}.clean.partial.md"
    llm_raw_out = RAW_DIR / f"{base}.llm_raw.json"
    llm_raw_partial_out = RAW_DIR / f"{base}.llm_raw.partial.json"
    llm_json_out = CLEAN_JSON_DIR / f"{base}.clean.json"
    llm_json_partial_out = CLEAN_JSON_DIR / f"{base}.clean.partial.json"
    meta_out = META_DIR / f"{base}.clean_meta.json"

    if mode in {"python", "both"} and python_out.exists() and python_json_out.exists() and not redo:
        python_state = "skipped"
    else:
        if mode in {"python", "both"} and python_out.exists():
            archive_existing(python_out, "clean_python")
        if mode in {"python", "both"} and python_json_out.exists():
            archive_existing(python_json_out, "clean_python_json")
        rows = read_segments_csv(seg_csv)
        speaker_map = map_speaker_names(rows)
        python_md = render_clean_md(title, rows, speaker_map, seg_csv.name)
        python_out.write_text(python_md, encoding="utf-8")
        python_turns = [
            {
                "speaker_name": speaker_map.get(r["speaker"], r["speaker"]),
                "timestamp_sec": int(round(float(r["start"]))),
                "text": r["text"],
                "start": r["start"],
                "end": r["end"],
                "speaker_conf": r.get("speaker_conf", 0.0),
            }
            for r in rows
        ]
        python_json_out.write_text(
            json.dumps(
                {
                    "base": base,
                    "source_segments_csv": str(seg_csv),
                    "updated_at": now_utc(),
                    "status": "complete",
                    "turns": python_turns,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        python_state = "written"

    if mode in {"llm", "both"}:
        if web_md_out.exists() and llm_json_out.exists() and not redo:
            llm_state = "skipped"
        else:
            archive_existing(web_md_out, "webformat_md")
            archive_existing(llm_raw_out, "clean_llm_raw")
            archive_existing(llm_json_out, "clean_llm_json")
            archive_existing(meta_out, "clean_meta")
            if web_partial_out.exists():
                web_partial_out.unlink()
            if llm_raw_partial_out.exists():
                llm_raw_partial_out.unlink()
            if llm_json_partial_out.exists():
                llm_json_partial_out.unlink()
            rows = read_segments_csv(seg_csv)
            speaker_map = map_speaker_names(rows)
            llm_turns = [
                {
                    "speaker_name": speaker_map.get(t["speaker"], t["speaker"]),
                    "timestamp_sec": int(round(float(t["start"]))),
                    "text": t["text"],
                    "start": t["start"],
                    "end": t["end"],
                }
                for t in rows
            ]
            spotify_url, apple_url = infer_episode_links(title, manifest_row)
            raw_records_partial: list[dict] = []

            def write_partial(
                partial_turns: list[dict],
                chunk_idx: int,
                chunk_total: int,
                in_tok_partial: int,
                out_tok_partial: int,
                chunk_record: dict,
            ) -> None:
                raw_records_partial.append(chunk_record)
                corr_summary = summarize_label_corrections(raw_records_partial)
                partial_rebalanced, _partial_merge_stats = merge_same_speaker_turns_under_cap(
                    partial_turns,
                    max_words=300,
                )
                partial_text = render_named_turns_md(
                    title,
                    partial_rebalanced,
                    spotify_url=spotify_url,
                    apple_url=apple_url,
                )
                web_partial_out.write_text(partial_text, encoding="utf-8")
                llm_json_partial_out.write_text(
                    json.dumps(
                        {
                            "base": base,
                            "updated_at": now_utc(),
                            "llm_model": llm_cfg.model,
                            "status": f"partial {chunk_idx}/{chunk_total}",
                            "correction_summary": corr_summary,
                            "turns": partial_rebalanced,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                llm_raw_partial_out.write_text(
                    json.dumps(
                        {
                            "base": base,
                            "updated_at": now_utc(),
                            "llm_model": llm_cfg.model,
                            "status": f"partial {chunk_idx}/{chunk_total}",
                            "correction_summary": corr_summary,
                            "chunks": raw_records_partial,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                log(
                    f"[llm] {base} partial_written chunk={chunk_idx}/{chunk_total} "
                    f"in_tokens={in_tok_partial} out_tokens={out_tok_partial}"
                )

            cleaned_turns, in_tok, out_tok, raw_chunks = llm_cleanup_turns(llm_turns, cfg=llm_cfg, on_chunk_done=write_partial)
            cleaned_turns, merge_cap_stats = merge_same_speaker_turns_under_cap(cleaned_turns, max_words=300)
            final_corr_summary = summarize_label_corrections(raw_chunks)
            llm_text = render_named_turns_md(
                title,
                cleaned_turns,
                spotify_url=spotify_url,
                apple_url=apple_url,
            )
            web_md_out.write_text(llm_text, encoding="utf-8")
            write_site_episode_page(base, title, llm_text)
            if web_partial_out.exists():
                web_partial_out.unlink()
            llm_json_out.write_text(
                json.dumps(
                    {
                        "base": base,
                        "source_segments_csv": str(seg_csv),
                        "updated_at": now_utc(),
                        "llm_model": llm_cfg.model,
                        "status": "complete",
                        "correction_summary": final_corr_summary,
                        "turns": cleaned_turns,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if llm_json_partial_out.exists():
                llm_json_partial_out.unlink()
            llm_raw_out.write_text(
                json.dumps(
                    {
                        "base": base,
                        "source_segments_csv": str(seg_csv),
                        "updated_at": now_utc(),
                        "llm_model": llm_cfg.model,
                        "status": "complete",
                        "chunks": raw_chunks,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if llm_raw_partial_out.exists():
                llm_raw_partial_out.unlink()
            stats = compare_turn_sets(llm_turns, cleaned_turns)
            flags = quality_flags(stats)
            meta_out.write_text(
                json.dumps(
                    {
                        "base": base,
                        "source_segments_csv": str(seg_csv),
                        "updated_at": now_utc(),
                        "llm_model": llm_cfg.model,
                        "llm_input_tokens": in_tok,
                        "llm_output_tokens": out_tok,
                        "correction_summary": final_corr_summary,
                        "merge_cap_stats": merge_cap_stats,
                        "comparison_stats": stats,
                        "quality_flags": flags,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if flags:
                log(f"[warn] {base} quality_flags={','.join(flags)}")
            llm_state = "written"
    else:
        llm_state = "not_requested"

    return python_state, llm_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", default=str(DEBUG_DIR))
    parser.add_argument("--mode", choices=["python", "llm", "both"], default="python")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-gap-sec", type=float, default=1.2)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--llm-model", default="gpt-5-nano")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-chars-per-chunk", type=int, default=12000)
    parser.add_argument("--llm-max-words-per-chunk", type=int, default=500)
    parser.add_argument("--llm-overlap-words", type=int, default=100)
    parser.add_argument("--llm-chunk-sentence-overrun-words", type=int, default=120)
    parser.add_argument("--llm-request-timeout-sec", type=float, default=120.0)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff-sec", type=float, default=4.0)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    ensure_dirs()
    run_stamp = local_run_stamp()
    mode_slug = args.mode
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else log_dir / f"clean_dialogue_{mode_slug}_{run_stamp}.log"

    global _LOG_FP
    _LOG_FP = log_file.open("a", encoding="utf-8")
    log(f"Run started. log_file={log_file}")
    log(f"Args: {vars(args)}")

    seg_dir = Path(args.segments_dir)
    seg_files = sorted(seg_dir.glob("*.segments.csv"))
    if not seg_files:
        raise SystemExit(f"No segment CSV files found in {seg_dir}")
    log(f"Discovered segment files: {len(seg_files)} in {seg_dir}")

    manifest_path = Path(args.manifest_path)
    manifest_fieldnames, manifest_rows, manifest_by_seg, manifest_by_episode = load_manifest(manifest_path)
    if manifest_rows:
        log(f"Manifest loaded: rows={len(manifest_rows)} path={manifest_path}")
    else:
        log(f"Manifest not loaded or empty: {manifest_path}")

    llm_cfg = LLMConfig(
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_chars_per_chunk=max(2000, args.llm_max_chars_per_chunk),
        max_words_per_chunk=max(80, args.llm_max_words_per_chunk),
        overlap_words=max(0, args.llm_overlap_words),
        chunk_sentence_overrun_words=max(0, args.llm_chunk_sentence_overrun_words),
        request_timeout_sec=max(5.0, float(args.llm_request_timeout_sec)),
        max_retries=max(1, int(args.llm_max_retries)),
        retry_backoff_sec=max(0.5, float(args.llm_retry_backoff_sec)),
    )

    processed = 0
    for i, seg_csv in enumerate(seg_files, start=1):
        if args.max_episodes > 0 and processed >= args.max_episodes:
            break
        base = base_from_segments_path(seg_csv)
        python_out = PYTHON_OUT_DIR / f"{base}.clean.md"
        python_json_out = PYTHON_JSON_DIR / f"{base}.clean.json"
        llm_json_out = CLEAN_JSON_DIR / f"{base}.clean.json"
        llm_raw_out = RAW_DIR / f"{base}.llm_raw.json"
        meta_out = META_DIR / f"{base}.clean_meta.json"
        web_md_out = WEB_MD_DIR / f"{base}.clean.md"
        manifest_row = manifest_by_seg.get(seg_csv.name)
        if manifest_row is None:
            ep_num = _episode_num_from_text(base.replace("__", " "))
            if ep_num:
                manifest_row = manifest_by_episode.get(ep_num)
        log(f"[{i}/{len(seg_files)}] start base={base}")
        try:
            py_state, llm_state = process_episode(
                seg_csv=seg_csv,
                mode=args.mode,
                max_gap_sec=max(0.1, args.max_gap_sec),
                llm_cfg=llm_cfg,
                redo=args.redo,
                manifest_row=manifest_row,
            )
            log(f"[{i}/{len(seg_files)}] {base} python={py_state} llm={llm_state}")
            if manifest_row is not None:
                update_manifest_row(
                    manifest_row,
                    mode=args.mode,
                    python_state=py_state,
                    llm_state=llm_state,
                    python_out=python_out,
                    python_json_out=python_json_out,
                    llm_out=web_md_out,
                    llm_json_out=llm_json_out,
                    llm_raw_out=llm_raw_out,
                    meta_out=meta_out,
                    llm_model=llm_cfg.model,
                    error_text="",
                )
                if manifest_fieldnames and manifest_rows:
                    write_manifest(manifest_path, manifest_fieldnames, manifest_rows)
            else:
                log(f"[warn] no manifest row matched seg_csv={seg_csv.name}")
        except Exception as exc:
            log(f"[{i}/{len(seg_files)}] {base} ERROR: {type(exc).__name__}: {exc}")
            if manifest_row is not None:
                py_state_err = "not_requested"
                llm_state_err = "not_requested"
                if args.mode in {"python", "both"}:
                    py_state_err = "error"
                if args.mode in {"llm", "both"}:
                    llm_state_err = "error"
                update_manifest_row(
                    manifest_row,
                    mode=args.mode,
                    python_state=py_state_err,
                    llm_state=llm_state_err,
                    python_out=python_out,
                    python_json_out=python_json_out,
                    llm_out=web_md_out,
                    llm_json_out=llm_json_out,
                    llm_raw_out=llm_raw_out,
                    meta_out=meta_out,
                    llm_model=llm_cfg.model,
                    error_text=f"{type(exc).__name__}: {exc}",
                )
                if manifest_fieldnames and manifest_rows:
                    write_manifest(manifest_path, manifest_fieldnames, manifest_rows)
        processed += 1

    log(f"Finished clean batch. Episodes attempted: {processed}")
    if _LOG_FP is not None:
        _LOG_FP.close()


if __name__ == "__main__":
    main()
