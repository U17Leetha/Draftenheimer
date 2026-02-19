# ReportsQA

Local QA tooling for penetration test reports (`.docx`) with optional LLM-assisted narrative review.

## Why this setup

- Keeps confidential report processing local when using Ollama.
- Learns reusable QA patterns from historical `v0.1 -> v1.0` report revisions.
- Lets you switch models later without losing learned behavior.

## Quick start (local model)

1. Start Ollama locally.
2. Pull a model (recommended):

```bash
python3 qa_models.py pull qwen2.5:14b
```

3. Run report QA:

```bash
python3 qa_scan.py /path/to/report_v0.1.docx \
  --llm \
  --provider ollama \
  --model qwen2.5:14b \
  --json-out /tmp/report.qa.json
```

4. Optional: write comments back into an annotated DOCX:

```bash
python3 qa_scan.py /path/to/report_v0.1.docx \
  --llm \
  --provider ollama \
  --model qwen2.5:14b \
  --json-out /tmp/report.qa.json \
  --annotate \
  --annotate-out /tmp/report.annotated.docx
```

## Rebuild learned profile from training pairs

Rebuild rule-based learned patterns from `training/qa_pair_training/reports/`:

```bash
python3 training/qa_pair_training/build_learned_profile.py
```

Run AI-assisted pair comparison (local Ollama) to learn additional patterns:

```bash
python3 training/qa_pair_training/build_learned_profile.py \
  --ai-compare \
  --ai-provider ollama \
  --ai-model qwen2.5:14b \
  --ollama-url http://localhost:11434
```

This updates `reportsqa_profile.json`, which `qa_scan.py` uses automatically.

## Model changes later

You do not lose learned behavior when switching models.

- Learned behavior is stored in `reportsqa_profile.json`, not in model weights.
- To switch model, change only `--model` in `qa_scan.py`.
- To refresh model-derived patterns for the new model, rerun the AI-assisted profile build with the new `--ai-model`.

## More details

- Full training-pair workflow: `training/qa_pair_training/README.md`
- Ignore/suppress specific boilerplate suggestions: `reportsqa_ignore.json`
