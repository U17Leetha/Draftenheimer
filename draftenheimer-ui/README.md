# Draftenheimer UI

Tauri desktop wrapper for the Draftenheimer CLI.

## Features

- Main screen focused on scan + feedback actions.
- Native Browse pickers for report/output/config paths.
- Settings drawer for AI provider/model config, runtime controls, and theme.
- Local Ollama controls: start/stop/full-stop, refresh models, pull model.
- Live run status (keeps `Model Ready`) and full stdout/stderr output.
- Theme mode: Light, Dark, or System.

## Run

```bash
npm install
npm run dev
```

## Notes

- The app auto-detects the Draftenheimer root (expects `qa_scan.py` and `draftenheimer`).
- All execution remains local through your existing CLI scripts.

## Logo

Place your branding assets at:

```
src/assets/draftenheimer-symbol.png
src/assets/draftenheimer-wordmark.png
```

Optional dark-mode wordmark override:

```
src/assets/draftenheimer-wordmark-dark.png
```

If the dark file exists, it is used in dark theme. If not, the default wordmark is used.

Optional dark-mode symbol override:

```
src/assets/draftenheimer-symbol-dark.png
```

If present, dark theme uses this symbol; otherwise it uses `draftenheimer-symbol.png`.


## App Icons

To regenerate Tauri app icons from the symbol logo:

```bash
./sync_app_icons.sh
```

This uses `src/assets/draftenheimer-symbol.png` as the icon source.
