# Draftenheimer

## Copy/Paste Quickstart

```bash
# 0) Start local SLM runtime (Ollama)
./slm_start.sh

# 1) Pull local model
python3 qa_models.py pull qwen2.5:14b

# 2) Run QA scan + auto-refresh learned profile from reports/
./draftenheimer /path/to/report_v0.1.docx \
  --llm \
  --provider ollama \
  --model qwen2.5:14b \
  --auto-learn \
  --json-out /tmp/report.qa.json

# 3) Stop local SLM runtime when done
./slm_stop.sh
```

Local QA tooling for penetration test reports (`.docx`) with optional LLM-assisted narrative review.

## Why this setup

- Keeps confidential report processing local when using Ollama.
- Learns reusable QA patterns from historical `v0.1 -> v1.0` report revisions.
- Lets you switch models later without losing learned behavior.

## Command name

- Local repo usage: `./draftenheimer ...`
- Optional global command:

```bash
ln -sf "$PWD/draftenheimer" /usr/local/bin/draftenheimer
```

Then you can run: `draftenheimer ...` from anywhere.

## SLM lifecycle

- Start runtime: `./slm_start.sh`
- Stop runtime (server + unloaded models): `./slm_stop.sh`
- Full stop including desktop app: `./slm_stop.sh --full`

## Continuous learning behavior

- `qa_scan.py` always reads `draftenheimer_profile.json` for QA rules/patterns.
- Use `--auto-learn` to refresh that profile from local report pairs (`reports/`) before each scan.
- Optional deeper learning: add `--auto-learn-ai` to include model-based pair comparison (slower).

Example (deep auto-learning):

```bash
./draftenheimer /path/to/report_v0.1.docx \
  --llm \
  --provider ollama \
  --model qwen2.5:14b \
  --auto-learn \
  --auto-learn-ai \
  --json-out /tmp/report.qa.json
```

## Quick start (local model)

1. Start Ollama locally.
2. Pull a model (recommended):

```bash
python3 qa_models.py pull qwen2.5:14b
```

3. Run report QA:

```bash
./draftenheimer /path/to/report_v0.1.docx \
  --llm \
  --provider ollama \
  --model qwen2.5:14b \
  --auto-learn \
  --json-out /tmp/report.qa.json
```

4. Optional: write comments back into an annotated DOCX:

```bash
./draftenheimer /path/to/report_v0.1.docx \
  --llm \
  --provider ollama \
  --model qwen2.5:14b \
  --auto-learn \
  --json-out /tmp/report.qa.json \
  --annotate \
  --annotate-out /tmp/report.annotated.docx
```

## Manual profile rebuild (optional)

Rebuild rule-based learned patterns from local report pairs in `reports/`:

```bash
python3 build_learned_profile.py
```

Run AI-assisted pair comparison (local Ollama) to learn additional patterns:

```bash
python3 build_learned_profile.py \
  --ai-compare \
  --ai-provider ollama \
  --ai-model qwen2.5:14b \
  --ollama-url http://localhost:11434
```

This updates `draftenheimer_profile.json`, which `qa_scan.py` uses automatically.

## Model changes later

You do not lose learned behavior when switching models.

- Learned behavior is stored in `draftenheimer_profile.json`, not in model weights.
- To switch model, change only `--model` in `qa_scan.py`.
- To refresh model-derived patterns for the new model, rerun the AI-assisted profile build with the new `--ai-model`.

## Keeping reports out of GitHub

- Store confidential report files under `reports/` (or other ignored directories).
- Do not commit raw `*.docx`, `*.pdf`, `*.pptx`, `*.xlsx`.
- Generated report artifacts like `*.qa.json` and `*.annotated.docx` are also ignored.

## Pairing convention

- `*v0.1*.docx` is treated as draft.
- `*v1.0*.docx` is treated as final.
- Matching base names are paired automatically.

## Config

- Keep local private overrides in `draftenheimer_ignore.json` (gitignored).
- Keep acceptance/rejection feedback in `draftenheimer_feedback.json` (gitignored).
- Start from `draftenheimer_ignore.example.json` and `draftenheimer_feedback.example.json`.

### Ignore boilerplate comments

Use `draftenheimer_ignore.json` to suppress known-safe annotations.

```json
{
  "ignore_rewrite_from_phrases": [
    "While this type of assessment is intended to mimic a real-world attack scenario"
  ],
  "ignore_diagnostic_codes": [
    "DOC-STYLE-REWRITE-001"
  ],
  "ignore_message_contains": [
    "Preferred rewrite: \"A Medical Device Security Testing is comprised"
  ]
}
```

### Teach accepted vs rejected revisions

Use `draftenheimer_feedback.json`:

```json
{
  "accepted_rewrites": [
    {"from": "old sentence", "to": "new sentence"}
  ],
  "rejected_rewrites": [
    {"from": "old boilerplate", "to": "new boilerplate"}
  ],
  "accepted_diagnostics": [
    {"code": "LLM-NARR-001", "message_contains": "Clarify that the objective"}
  ],
  "rejected_diagnostics": [
    {"code": "LLM-NARR-001", "message_contains": "Avoid overused phrasing"}
  ]
}
```

Behavior:
- `accepted_rewrites`: promoted as preferred rewrite rules in future runs.
- `rejected_rewrites`: suppressed so those exact rewrite suggestions stop appearing.
- `accepted_diagnostics`: keeps matching diagnostics even if broad ignore rules exist.
- `rejected_diagnostics`: suppresses matching diagnostics in future runs.

Optional explicit paths per run:

```bash
./draftenheimer /path/to/report.docx \
  --llm --provider ollama --model qwen2.5:14b \
  --ignore-config /path/to/draftenheimer_ignore.json \
  --feedback-config /path/to/draftenheimer_feedback.json
```

### Import decisions from reviewed annotated DOCX

You can review comments in Word and mark decisions directly:
- Put `[ACCEPT]` or `[REJECT]` (also supports `REJECTED`, `ACCEPT`, `REJECT`) in the original Draftenheimer comment text, or
- Reply to the comment with those markers.
- Leave comment unchanged to skip it.

Imports now include both:
- rewrite decisions (`DOC-STYLE-REWRITE-001` style comments)
- non-rewrite diagnostic decisions (for example `LLM-NARR-001`)

Then import decisions into `draftenheimer_feedback.json`:

```bash
draftenheimer --import-feedback-docx /path/to/reviewed.annotated.docx
```

Preview only (no write):

```bash
draftenheimer --import-feedback-docx /path/to/reviewed.annotated.docx --feedback-dry-run
```
