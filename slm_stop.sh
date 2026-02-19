#!/usr/bin/env bash
set -euo pipefail

FULL_MODE="false"
if [[ "${1:-}" == "--full" ]]; then
  FULL_MODE="true"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama CLI not found. Nothing to stop."
  exit 0
fi

# Unload any active models from memory.
if ollama ps >/tmp/ollama_ps.out 2>/dev/null; then
  awk 'NR>1 && $1 != "" {print $1}' /tmp/ollama_ps.out | while read -r model; do
    ollama stop "$model" >/dev/null 2>&1 || true
  done
fi
rm -f /tmp/ollama_ps.out

# Stop background server process.
PIDS="$(pgrep -f '[o]llama serve' || true)"
if [[ -n "${PIDS}" ]]; then
  echo "${PIDS}" | xargs kill >/dev/null 2>&1 || true
  sleep 1
fi

if [[ "${FULL_MODE}" == "true" ]]; then
  # Optional: quit the desktop Ollama app so it does not restart the server.
  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1 || true
  fi
fi

if ollama list >/dev/null 2>&1; then
  echo "Ollama still reachable. It may be managed by the desktop app; run ./slm_stop.sh --full to fully quit app + server."
else
  echo "Ollama server stopped."
fi
