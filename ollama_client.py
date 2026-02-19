#!/usr/bin/env python3
import json
import urllib.request
import urllib.error


def _url(base_url, path):
    return f"{base_url.rstrip('/')}{path}"


def list_models(base_url="http://localhost:11434"):
    req = urllib.request.Request(_url(base_url, "/api/tags"), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Failed to list models: Ollama is not reachable. "
            "Start it with `ollama serve` or set --ollama-url to the correct host."
        ) from e


def pull_model(base_url, name, verbose=True):
    payload = json.dumps({"name": name}).encode("utf-8")
    req = urllib.request.Request(
        _url(base_url, "/api/pull"),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if verbose:
                    status = msg.get("status")
                    completed = msg.get("completed")
                    total = msg.get("total")
                    if completed is not None and total:
                        pct = int((completed / total) * 100)
                        print(f"{status} {pct}%")
                    elif status:
                        print(status)
            return True
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to pull model: {e}")


def chat(base_url, model, messages, options=None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url(base_url, "/api/chat"),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body.get("message", {})
            return msg.get("content", "")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama chat failed: {e}")
