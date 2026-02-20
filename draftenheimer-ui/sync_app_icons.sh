#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYMBOL_PATH="$ROOT_DIR/src/assets/draftenheimer-symbol.png"

if [[ ! -f "$SYMBOL_PATH" ]]; then
  echo "Missing symbol file: $SYMBOL_PATH" >&2
  echo "Add your symbol logo there, then re-run this script." >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "Generating Tauri app icons from: $SYMBOL_PATH"
npm run tauri icon "$SYMBOL_PATH"
echo "Done. Icons updated under src-tauri/icons/."
