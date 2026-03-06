from __future__ import annotations

import re
from pathlib import Path


SITE_CONTEXT = {
    "spotify_show_url": "",
    "apple_show_url": "",
    "generator_name": "PodcastTranscriptor",
    "generator_repo_url": "https://github.com/alik-git/PodcastTranscriptor",
    "warning_text": "It may contain mistakes.",
    "speakers_note": "Speakers are denoted as color names.",
}


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


def sec_to_hms(sec: float) -> str:
    s = max(0, int(round(float(sec))))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    text = text.replace(" ;", ";").replace(" :", ":")
    return text


def slug_base_to_title(base_name: str) -> str:
    parts = base_name.split("__")
    if len(parts) >= 2:
        return parts[1].replace("-", " ").strip().title()
    return base_name


def configure_site_context(cfg: dict, choose_value, get_cfg) -> None:
    SITE_CONTEXT["spotify_show_url"] = choose_value(
        get_cfg(cfg, "podcast.spotify_show_url", ""),
        SITE_CONTEXT["spotify_show_url"],
    )
    SITE_CONTEXT["apple_show_url"] = choose_value(
        get_cfg(cfg, "podcast.apple_show_url", ""),
        SITE_CONTEXT["apple_show_url"],
    )
    SITE_CONTEXT["generator_name"] = choose_value(
        get_cfg(cfg, "site.generator_name", ""),
        SITE_CONTEXT["generator_name"],
    )
    SITE_CONTEXT["generator_repo_url"] = choose_value(
        get_cfg(cfg, "site.generator_repo_url", ""),
        SITE_CONTEXT["generator_repo_url"],
    )
    SITE_CONTEXT["warning_text"] = choose_value(
        get_cfg(cfg, "site.warning_text", ""),
        SITE_CONTEXT["warning_text"],
    )
    SITE_CONTEXT["speakers_note"] = choose_value(
        get_cfg(cfg, "site.speakers_note", ""),
        SITE_CONTEXT["speakers_note"],
    )


def infer_episode_links(manifest_row: dict | None) -> tuple[str, str]:
    row = manifest_row or {}
    spotify = (row.get("spotify_url") or "").strip() or SITE_CONTEXT["spotify_show_url"]
    apple = (row.get("apple_url") or "").strip() or SITE_CONTEXT["apple_show_url"]
    return spotify, apple


def render_named_turns_md(
    title: str,
    turns: list[dict],
    spotify_url: str,
    apple_url: str,
) -> str:
    spotify_link = spotify_url or SITE_CONTEXT["spotify_show_url"]
    apple_link = apple_url or SITE_CONTEXT["apple_show_url"]
    gen_name = SITE_CONTEXT["generator_name"]
    gen_repo_url = SITE_CONTEXT["generator_repo_url"]
    warning_text = SITE_CONTEXT["warning_text"]
    speakers_note = SITE_CONTEXT["speakers_note"]
    warning_sentence = warning_text.strip().rstrip(".")
    out = [
        f"# {title}",
        "",
        f"- This transcript was generated with AI using [{gen_name}]({gen_repo_url}).",
        f"- **{warning_text}** Please check against the actual podcast.",
        f"- {speakers_note}",
    ]
    if spotify_link or apple_link:
        out.insert(2, f"- Links to this episode: [Spotify]({spotify_link}) / [Apple Podcasts]({apple_link})")
    out.extend(["", "## Transcript", ""])

    for t in turns:
        speaker_name = t["speaker_name"]
        text = normalize_text(t["text"])
        start_sec = int(round(float(t.get("timestamp_sec", 0) or 0)))
        if not text:
            continue
        color = SPEAKER_COLOR_HEX.get(str(speaker_name).strip().lower(), "#374151")
        ts = f"<em><strong>[{sec_to_hms(start_sec)}]</strong></em>"
        spk = f"<strong><span style=\"color:{color}\">{speaker_name}:</span></strong>"
        out.append(f"{ts}&nbsp;&nbsp;{spk} {text}")
        out.append("")

    out.extend(
        [
            "---",
            "",
            f"*Generated with AI using [{gen_name}]({gen_repo_url}). {warning_sentence}; please verify against the actual podcast.*",
            "",
        ]
    )
    if spotify_link or apple_link:
        out.insert(-2, f"*Links to this episode:* [Spotify]({spotify_link}) / [Apple Podcasts]({apple_link})")
        out.insert(-2, "")
    return "\n".join(out).rstrip() + "\n"


def write_site_episode_page(base: str, title: str, body_md: str, episodes_dir: Path) -> None:
    episodes_dir.mkdir(parents=True, exist_ok=True)
    parts = base.split("__")
    date_part = parts[0] if parts else "unknown-date"
    slug_part = parts[1] if len(parts) >= 2 else base
    out_name = f"{date_part}-{slug_part}.md"
    out_path = episodes_dir / out_name

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
