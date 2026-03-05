# PodcastTranscriptor

End-to-end pipeline for turning podcast episodes into readable, speaker-attributed transcripts.

This README is the canonical documentation for the repo.

## What This Repo Does
- Ingests episode metadata from RSS-exported CSV.
- Builds/updates a global episode manifest.
- Downloads audio and runs ASR transcription.
- Runs speaker diarization and word-to-speaker alignment.
- Produces deterministic cleanup output (`clean_python`).
- Produces LLM-assisted label-correction output (`clean_llm`) and final readable transcript Markdown.

## Quick Start

### 1) Prerequisites
- Linux/macOS shell
- Conda
- `ffmpeg` available on system
- Optional but recommended: NVIDIA GPU + CUDA
- Hugging Face account/token (for diarization model)
- OpenAI API key (for LLM cleanup pass)

### 2) Create and Prepare the Conda Env (`pds_env`)
```bash
# cd to repo root
conda create -n pds_env python=3.11 -y
conda run -n pds_env python -m pip install -U pip
conda run -n pds_env python -m pip install -r transcription/requirements.txt
# Optional packaging install (if you want a shell command entrypoint):
# conda run -n pds_env python -m pip install -e .

# Additional runtime deps used by diarization + LLM cleanup scripts
conda run -n pds_env python -m pip install \
  openai \
  huggingface_hub \
  pyannote.audio \
  torch \
  torchaudio \
  soundfile \
  numpy
```

### 3) Auth Setup
```bash
# cd to repo root
# Hugging Face (for pyannote model access)
conda run -n pds_env hf auth login

# OpenAI (for clean_llm pass)
export OPENAI_API_KEY='your_key_here'
```

If you want the OpenAI key persistent for your shell sessions, add this to your shell rc (for example `~/.bashrc`):
```bash
# cd to repo root
export OPENAI_API_KEY='your_key_here'
```

## Models Used (Downloaded on First Use)

### ASR
- `faster-whisper` model: typically `small` (configurable).
- Download happens automatically on first run and is cached under repo-local model/cache dirs.

### Diarization
- Hugging Face model: `pyannote/speaker-diarization-community-1`
- Requires HF token + accepted access terms on Hugging Face.

### LLM Cleanup
- OpenAI model default in script: `gpt-5-nano`
- Used for speaker-label correction (not full transcript rewriting).

## Repo Structure
```text
PodcastTranscriptor/
  episodes_source.csv                                     # recommended episode metadata source
  dev_log.md                                       # concise running engineering log
  AGENTS.md                                        # repo-level operating preferences
  README.md                                        # canonical docs (this file)
  pyproject.toml                                  # package metadata

  pdscript/
    __init__.py
    cli.py                                         # package CLI entrypoint (`python -m pdscript.cli`)

  transcription/
    manifests/
      pipeline_manifest.csv                        # global pipeline state by episode

    scripts/
      build_manifest.py
      transcribe_batch.py
      speaker_batch.py
      clean_dialogue_batch.py

    artifacts/
      01_whisper_transcript/
        audio/                                     # downloaded audio
        transcripts/                               # whisper txt/json (+ partial files while running)
      02_diarization/
        md/                                        # speaker-attributed markdown
        diarization/                               # diarization json + rttm
        debug/                                     # words.csv + segments.csv
      03_clean_python/
        md/                                        # deterministic cleaned markdown
        json/                                      # deterministic cleaned json
      04_clean_llm/
        md/                                        # LLM-cleaned markdown
        json/                                      # LLM-cleaned canonical json
        raw/                                       # raw LLM outputs (json)
        meta/                                      # per-episode clean/validation stats
      old/                                         # archived/legacy scratch outputs

    logs/                                          # run logs (generated)
    tmp/                                           # temp/cache (generated)
    models/                                        # local model cache (generated)
```

## End-to-End Pipeline (RSS to Final Transcript)

### Stage 0: Source Metadata
Input CSV:
- `episodes_source.csv`
- You can also pass any CSV path via `--episodes-csv`.

Key fields used downstream:
- `guid`, `title`, `pub_date_iso`, `link`, `audio_url`

### Preferred Entrypoint (Python-Only)
Run all stages sequentially:
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli --all
```

Run one stage at a time:
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli manifest
conda run -n pds_env python -m pdscript.cli transcribe
conda run -n pds_env python -m pdscript.cli speaker
conda run -n pds_env python -m pdscript.cli clean-python
conda run -n pds_env python -m pdscript.cli clean-llm
conda run -n pds_env python -m pdscript.cli status
```

### Stage 1: Build/Refresh Manifest
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli manifest
```
Or explicitly choose a source CSV:
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli manifest \
  --episodes-csv /path/to/your_episodes_source.csv
```
Output:
- `transcription/manifests/pipeline_manifest.csv`

Purpose:
- Single state table that tracks each episode across transcription, diarization, and cleanup.

### Stage 2: Transcription (Audio + ASR)
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli transcribe \
  --model-size small \
  --device cuda \
  --compute-type int8_float16 \
  --download-retries 3 \
  --retry-delay-sec 3 \
  --episode-progress-step 5
```
Outputs:
- `transcription/artifacts/01_whisper_transcript/audio/<base>.mp3`
- `transcription/artifacts/01_whisper_transcript/transcripts/<base>.txt`
- `transcription/artifacts/01_whisper_transcript/transcripts/<base>.json`
- Live partials during run: `*.partial.txt`, `*.partial.json`

Behavior:
- Episode-by-episode processing, resumable on rerun.
- Errors are recorded per episode without stopping the entire batch.

### Stage 3: Speaker Diarization + Alignment
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli speaker \
  --manifest transcription/manifests/pipeline_manifest.csv \
  --model-size small \
  --device cuda \
  --compute-type int8_float16 \
  --min-speakers 1 \
  --max-speakers 15 \
  --telemetry-interval-sec 30
```
Outputs:
- `transcription/artifacts/02_diarization/md/<base>.md`
- `transcription/artifacts/02_diarization/diarization/<base>.diarization.json`
- `transcription/artifacts/02_diarization/debug/<base>.words.csv`
- `transcription/artifacts/02_diarization/debug/<base>.segments.csv`

### Stage 4: Deterministic Cleanup (`clean_python`)
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli clean-python \
  --segments-dir transcription/artifacts/02_diarization/debug
```
Outputs:
- `transcription/artifacts/03_clean_python/md/<base>.clean.md`
- `transcription/artifacts/03_clean_python/json/<base>.clean.json`

### Stage 5: LLM Label-Correction Cleanup (`clean_llm`)
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli clean-llm \
  --segments-dir transcription/artifacts/02_diarization/debug \
  --llm-model gpt-5-nano \
  --llm-max-words-per-chunk 700 \
  --llm-overlap-words 100 \
  --llm-request-timeout-sec 180 \
  --llm-max-retries 4 \
  --llm-retry-backoff-sec 6
```
Outputs:
- `transcription/artifacts/04_clean_llm/md/<base>.clean.md`
- `transcription/artifacts/04_clean_llm/json/<base>.clean.json`
- `transcription/artifacts/04_clean_llm/raw/<base>.llm_raw.json`
- `transcription/artifacts/04_clean_llm/meta/<base>.clean_meta.json`
- Live partials while running:
  - `*.clean.partial.md`
  - `*.clean.partial.json`
  - `*.llm_raw.partial.json`

Current LLM behavior:
- Works on chunked turns with context windows.
- Returns speaker-label corrections by line index.
- Pipeline applies label changes deterministically to original text/timestamps.
- Includes retry/backoff handling and chunk-level logging.

## Monitoring

### Quick status snapshot
```bash
# cd to repo root
conda run -n pds_env python -m pdscript.cli status
```

### Follow latest pipeline log (all stages)
```bash
# cd to repo root
LATEST_LOG=$(ls -1t transcription/logs/*.log | head -n 1)
tail -f "$LATEST_LOG"
```

### Watch live partial outputs while a run is active
```bash
# cd to repo root
find transcription/artifacts -type f \( -name '*.partial.txt' -o -name '*.partial.json' -o -name '*.partial.md' \) | sort
```

### Log retention
- Keep only the latest active `.log` in `transcription/logs/`.
- Move older logs to `transcription/logs/old/`.

## Rerun/Resume Rules
- Default behavior is resumable and skips completed outputs.
- Use `--redo` only when intentionally reprocessing.
- Batch jobs continue past per-episode errors and log them.

## Website Hosting Direction
- Target: GitHub Pages + Jekyll (`just-the-docs` theme) using generated Markdown transcripts.
- For public repos, GitHub Pages hosting is free (subject to GitHub Pages usage limits).

### GitHub Pages Setup (Just the Docs)
This repo now includes:
- `_config.yml` (site config)
- `Gemfile` (Jekyll + just-the-docs gems)
- `.github/workflows/pages.yml` (build + deploy workflow)
- `index.md` and `episodes/index.md` (site entry pages)

To enable publishing:
1. Push these files to `main`.
2. In GitHub, open `Settings -> Pages`.
3. Set `Source` to `GitHub Actions`.
4. Wait for the `Deploy Jekyll site to Pages` workflow to finish.

Published URL pattern:
- `https://alik-git.github.io/TheoryOfAnythingTranscripts/`

### Local Preview (Optional)
```bash
# cd to repo root
bundle install
bundle exec jekyll serve
```

## Notes
- Keep `dev_log.md` concise with timestamped `##` headings.
- Keep legacy artifacts under `transcription/artifacts/old/`.
