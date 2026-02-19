#!/usr/bin/env python3
import argparse
import subprocess
import json as _json
from ollama_client import list_models, pull_model


def _run_ollama_cli(args):
    try:
        res = subprocess.run(
            ["ollama"] + args,
            check=True,
            text=True,
            capture_output=True,
        )
        return res.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("Ollama CLI not found. Install Ollama or set --ollama-url for API access.")
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() or e.stdout.strip() or str(e)
        raise RuntimeError(f"Ollama CLI failed: {msg}")


def _run_aws_cli(args):
    try:
        res = subprocess.run(
            ["aws"] + args,
            check=True,
            text=True,
            capture_output=True,
        )
        return res.stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("AWS CLI not found. Install AWS CLI and configure credentials.")
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() or e.stdout.strip() or str(e)
        raise RuntimeError(f"AWS CLI failed: {msg}")


def _bedrock_list(region, profile=None):
    args = ["bedrock", "list-foundation-models", "--region", region, "--output", "json"]
    if profile:
        args.extend(["--profile", profile])
    out = _run_aws_cli(args)
    data = _json.loads(out)
    models = []
    for m in data.get("modelSummaries", []):
        mid = m.get("modelId")
        name = m.get("modelName") or mid
        if mid:
            models.append((mid, name))
    return models


def main():
    ap = argparse.ArgumentParser(description='Manage models for Draftenheimer')
    ap.add_argument('--ollama-url', default='http://localhost:11434')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list', help='List available local models (Ollama)')

    pull = sub.add_parser('pull', help='Pull a model (Ollama)')
    pull.add_argument('name')

    bed = sub.add_parser('bedrock', help='List AWS Bedrock models')
    bed.add_argument('action', choices=['list'])
    bed.add_argument('--region', default='us-east-1')
    bed.add_argument('--profile', default='sci_bedrock')

    args = ap.parse_args()

    if args.cmd == 'list':
        try:
            models = list_models(args.ollama_url)
        except RuntimeError as e:
            # Fallback to CLI list (UARTillery-style)
            output = _run_ollama_cli(["list"])
            print(output)
            raise SystemExit(0)
        for m in models:
            print(m)
    elif args.cmd == 'pull':
        try:
            pull_model(args.ollama_url, args.name, verbose=True)
        except RuntimeError as e:
            # Fallback to CLI pull (UARTillery-style)
            _run_ollama_cli(["pull", args.name])
            raise SystemExit(0)
    elif args.cmd == 'bedrock':
        if args.action == 'list':
            try:
                models = _bedrock_list(args.region, args.profile)
            except RuntimeError as e:
                print(str(e))
                raise SystemExit(2)
            for mid, name in models:
                print(f"{mid}\t{name}")


if __name__ == '__main__':
    main()
