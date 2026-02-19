#!/usr/bin/env python3
"""Build Draftenheimer learned QA profile from v0.1/v1.0 report pairs."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from qa_scan import parse_docx

OLD_VER_RE = re.compile(r"(?i)(v0\.1|0\.1v)")
NEW_VER_RE = re.compile(r"(?i)(v1\.0|1\.0v)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z']+")


def canonical_report_key(file_path: Path) -> str | None:
    stem = file_path.stem
    if not OLD_VER_RE.search(stem) and not NEW_VER_RE.search(stem):
        return None
    stem = OLD_VER_RE.sub("", stem)
    stem = NEW_VER_RE.sub("", stem)
    stem = re.sub(r"[\W_]+", "", stem).lower()
    return stem or None


def discover_pairs(reports_dir: Path) -> list[tuple[Path, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    if not reports_dir.exists():
        return []
    for file_path in sorted(reports_dir.iterdir()):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        key = canonical_report_key(file_path)
        if not key:
            continue
        bucket = pairs.setdefault(key, {})
        stem = file_path.stem
        if OLD_VER_RE.search(stem):
            bucket["old"] = file_path
        if NEW_VER_RE.search(stem):
            bucket["new"] = file_path

    out: list[tuple[Path, Path]] = []
    for versions in pairs.values():
        if "old" in versions and "new" in versions:
            out.append((versions["old"], versions["new"]))
    return sorted(out, key=lambda p: p[0].name.lower())


def paragraphs_from_docx(path: Path) -> list[str]:
    blocks = parse_docx(str(path))
    return [b["text"].strip() for b in blocks if b["type"] == "p" and b["text"].strip()]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]


def ngrams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def collect_case_normalization(old_paras: list[str], new_paras: list[str], case_votes: dict[str, Counter]) -> None:
    old_set = set(old_paras)
    new_set = set(new_paras)
    only_old = [p for p in old_paras if p not in new_set]
    only_new = [p for p in new_paras if p not in old_set]
    new_by_lower: dict[str, list[str]] = defaultdict(list)
    for p in only_new:
        new_by_lower[p.lower()].append(p)
    for old_p in only_old:
        lower = old_p.lower()
        if lower in new_by_lower:
            for preferred in new_by_lower[lower]:
                if preferred != old_p:
                    case_votes[lower][preferred] += 1


def collect_rewrites(old_text: str, new_text: str, rewrite_votes: Counter, rewrite_examples: dict[str, tuple[str, str]]) -> None:
    old_sentences = split_sentences(old_text)
    new_sentences = split_sentences(new_text)
    matcher = difflib.SequenceMatcher(a=old_sentences, b=new_sentences)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        old_chunk = old_sentences[i1:i2]
        new_chunk = new_sentences[j1:j2]
        limit = min(len(old_chunk), len(new_chunk))
        for idx in range(limit):
            old_s = old_chunk[idx].strip()
            new_s = new_chunk[idx].strip()
            if not old_s or not new_s or old_s == new_s:
                continue
            old_n = norm_text(old_s)
            new_n = norm_text(new_s)
            if old_n == new_n:
                continue
            key = f"{old_n}|||{new_n}"
            rewrite_votes[key] += 1
            rewrite_examples.setdefault(key, (old_s, new_s))


def _run_aws_cli(args):
    try:
        res = subprocess.run(["aws"] + args, check=True, text=True, capture_output=True)
        return res.stdout.strip()
    except FileNotFoundError as e:
        raise RuntimeError("AWS CLI not found.") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() or e.stdout.strip() or str(e)
        raise RuntimeError(f"AWS CLI failed: {msg}") from e


def _extract_json_object_from_text(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL):
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _invoke_bedrock(prompt: str, model: str, region: str, profile: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "temperature": 0.2,
    }
    body_path = None
    out_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            tf.write(json.dumps(body))
            tf.flush()
            body_path = tf.name
        args = [
            "bedrock-runtime",
            "invoke-model",
            "--region",
            region,
            "--model-id",
            model,
            "--content-type",
            "application/json",
            "--accept",
            "application/json",
            "--body",
            f"file://{body_path}",
        ]
        out_file = tempfile.NamedTemporaryFile(delete=False).name
        args.append(out_file)
        if profile:
            args.extend(["--profile", profile])
        _run_aws_cli(args)
        if not out_file or not os.path.exists(out_file):
            return ""
        with open(out_file, "r", encoding="utf-8") as rf:
            raw_body = rf.read().strip()
        try:
            content_json = json.loads(raw_body)
        except Exception:
            return ""
        if isinstance(content_json, dict) and "content" in content_json:
            pieces = content_json.get("content") or []
            if isinstance(pieces, list) and pieces:
                return pieces[0].get("text", "") or ""
        return ""
    finally:
        if body_path and os.path.exists(body_path):
            os.unlink(body_path)
        if out_file and os.path.exists(out_file):
            os.unlink(out_file)


def _invoke_ollama(prompt: str, model: str, ollama_url: str) -> str:
    from ollama_client import chat

    return chat(
        ollama_url,
        model,
        [{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )


def _pair_change_samples(old_text: str, new_text: str, max_samples: int = 32) -> dict:
    old_sentences = split_sentences(old_text)
    new_sentences = split_sentences(new_text)
    matcher = difflib.SequenceMatcher(a=old_sentences, b=new_sentences)
    replaced = []
    removed = []
    added = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            old_chunk = old_sentences[i1:i2]
            new_chunk = new_sentences[j1:j2]
            limit = min(len(old_chunk), len(new_chunk))
            for idx in range(limit):
                o = old_chunk[idx].strip()
                n = new_chunk[idx].strip()
                if o and n and o != n:
                    replaced.append({"from": o, "to": n})
        elif tag == "delete":
            removed.extend([o.strip() for o in old_sentences[i1:i2] if o.strip()])
        elif tag == "insert":
            added.extend([n.strip() for n in new_sentences[j1:j2] if n.strip()])
    return {
        "replaced_pairs": replaced[:max_samples],
        "removed_sentences": removed[:max_samples],
        "added_sentences": added[:max_samples],
    }


def _ai_pair_patterns(old_text: str, new_text: str, provider: str, model: str, bedrock_region: str, bedrock_profile: str, ollama_url: str) -> dict:
    sample = _pair_change_samples(old_text, new_text)
    prompt = (
        "You are extracting QA correction patterns from report revisions.\n"
        "Return JSON object only with keys:\n"
        "- deprecated_phrases: array of strings\n"
        "- preferred_rewrites: array of objects {from,to,reason}\n"
        "- style_rules: array of strings\n"
        "Keep only high-signal reusable patterns. Ignore one-off names/IDs/URLs/IPs.\n"
        f"Diff samples:\n{json.dumps(sample, ensure_ascii=False)}"
    )
    if provider == "bedrock":
        raw = _invoke_bedrock(prompt, model, bedrock_region, bedrock_profile)
    else:
        raw = _invoke_ollama(prompt, model, ollama_url)
    parsed = _extract_json_object_from_text(raw)
    return parsed if isinstance(parsed, dict) else {}


def build_profile(
    pairs: list[tuple[Path, Path]],
    min_drop: int,
    min_len: int,
    ngram_min: int,
    ngram_max: int,
    min_rewrite_count: int,
    ai_compare: bool,
    ai_provider: str,
    ai_model: str | None,
    bedrock_region: str,
    bedrock_profile: str,
    ollama_url: str,
    min_ai_votes: int,
) -> dict:
    sentence_old = Counter()
    sentence_new = Counter()
    ngram_old = Counter()
    ngram_new = Counter()
    case_votes: dict[str, Counter] = defaultdict(Counter)
    rewrite_votes = Counter()
    rewrite_examples: dict[str, tuple[str, str]] = {}
    ai_phrase_votes = Counter()
    ai_rewrite_votes = Counter()
    ai_rewrite_examples: dict[str, tuple[str, str, str]] = {}
    ai_style_votes = Counter()
    sources: list[dict[str, str]] = []

    for old_path, new_path in pairs:
        old_paras = paragraphs_from_docx(old_path)
        new_paras = paragraphs_from_docx(new_path)
        old_text = "\n".join(old_paras)
        new_text = "\n".join(new_paras)

        for s in split_sentences(old_text):
            sentence_old[norm_text(s)] += 1
        for s in split_sentences(new_text):
            sentence_new[norm_text(s)] += 1

        w_old = WORD_RE.findall(old_text.lower())
        w_new = WORD_RE.findall(new_text.lower())
        for n in range(ngram_min, ngram_max + 1):
            ngram_old.update(ngrams(w_old, n))
            ngram_new.update(ngrams(w_new, n))

        collect_case_normalization(old_paras, new_paras, case_votes)
        collect_rewrites(old_text, new_text, rewrite_votes, rewrite_examples)

        if ai_compare and ai_model:
            ai_result = _ai_pair_patterns(old_text, new_text, ai_provider, ai_model, bedrock_region, bedrock_profile, ollama_url)
            for p in ai_result.get("deprecated_phrases", []):
                if isinstance(p, str) and p.strip():
                    ai_phrase_votes[p.strip().lower()] += 1
            for r in ai_result.get("preferred_rewrites", []):
                if not isinstance(r, dict):
                    continue
                old_t = (r.get("from") or "").strip()
                new_t = (r.get("to") or "").strip()
                reason = (r.get("reason") or "").strip()
                if old_t and new_t and old_t != new_t:
                    key = f"{norm_text(old_t)}|||{norm_text(new_t)}"
                    ai_rewrite_votes[key] += 1
                    ai_rewrite_examples.setdefault(key, (old_t, new_t, reason))
            for rule in ai_result.get("style_rules", []):
                if isinstance(rule, str) and rule.strip():
                    ai_style_votes[rule.strip()] += 1

        sources.append({"old": str(old_path), "new": str(new_path)})

    deprecated_phrases = []
    for phrase, old_count in sentence_old.items():
        new_count = sentence_new.get(phrase, 0)
        if old_count - new_count >= min_drop and len(phrase) >= min_len:
            deprecated_phrases.append({"phrase": phrase, "old": old_count, "new": new_count})
    deprecated_phrases.sort(key=lambda x: (x["old"] - x["new"], x["old"]), reverse=True)

    deprecated_ngrams = []
    for phrase, old_count in ngram_old.items():
        new_count = ngram_new.get(phrase, 0)
        if old_count - new_count >= min_drop:
            deprecated_ngrams.append({"phrase": phrase, "old": old_count, "new": new_count})
    deprecated_ngrams.sort(key=lambda x: (x["old"] - x["new"], x["old"]), reverse=True)

    case_normalization = {}
    for lower, votes in case_votes.items():
        preferred, count = votes.most_common(1)[0]
        if count >= 1:
            case_normalization[lower] = preferred

    preferred_rewrites = []
    for key, count in rewrite_votes.items():
        if count < min_rewrite_count:
            continue
        old_ex, new_ex = rewrite_examples[key]
        preferred_rewrites.append({"from": old_ex, "to": new_ex, "count": count})
    for key, count in ai_rewrite_votes.items():
        if count < min_ai_votes:
            continue
        old_ex, new_ex, reason = ai_rewrite_examples[key]
        preferred_rewrites.append({"from": old_ex, "to": new_ex, "count": count, "source": "ai_compare", "reason": reason})
    preferred_rewrites.sort(key=lambda x: x["count"], reverse=True)

    ai_deprecated_phrases = [{"phrase": phrase, "count": count} for phrase, count in ai_phrase_votes.items() if count >= min_ai_votes]
    ai_deprecated_phrases.sort(key=lambda x: x["count"], reverse=True)

    ai_style_rules = [{"rule": rule, "count": count} for rule, count in ai_style_votes.items() if count >= min_ai_votes]
    ai_style_rules.sort(key=lambda x: x["count"], reverse=True)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pairs_count": len(sources),
        "source_pairs": sources,
        "deprecated_phrases": deprecated_phrases,
        "deprecated_ngrams": deprecated_ngrams,
        "case_normalization": case_normalization,
        "preferred_rewrites": preferred_rewrites,
        "ai_compare_enabled": ai_compare,
        "ai_compare_provider": ai_provider if ai_compare else None,
        "ai_compare_model": ai_model if ai_compare else None,
        "ai_deprecated_phrases": ai_deprecated_phrases,
        "ai_style_rules": ai_style_rules,
    }


def default_reports_dir() -> str:
    candidates = [
        Path("training/qa_pair_training/reports"),
        Path("reports"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)
    return "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Draftenheimer learned QA profile from report pairs.")
    parser.add_argument("--reports-dir", default=default_reports_dir(), help="Directory containing v0.1/v1.0 report pairs.")
    parser.add_argument("--out", default="draftenheimer_profile.json", help="Output profile JSON path.")
    parser.add_argument("--min-drop", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=20)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--min-rewrite-count", type=int, default=2)
    parser.add_argument("--ai-compare", action="store_true")
    parser.add_argument("--ai-provider", choices=["bedrock", "ollama"], default="ollama")
    parser.add_argument("--ai-model", default=None)
    parser.add_argument("--bedrock-region", default="us-east-1")
    parser.add_argument("--bedrock-profile", default="sci_bedrock")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--min-ai-votes", type=int, default=2)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    pairs = discover_pairs(reports_dir)
    if not pairs:
        raise SystemExit(f"No v0.1/v1.0 pairs found in: {reports_dir}")
    if args.ai_compare and not args.ai_model:
        raise SystemExit("--ai-compare requires --ai-model")

    profile = build_profile(
        pairs=pairs,
        min_drop=args.min_drop,
        min_len=args.min_len,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_rewrite_count=args.min_rewrite_count,
        ai_compare=args.ai_compare,
        ai_provider=args.ai_provider,
        ai_model=args.ai_model,
        bedrock_region=args.bedrock_region,
        bedrock_profile=args.bedrock_profile,
        ollama_url=args.ollama_url,
        min_ai_votes=args.min_ai_votes,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pairs used: {len(pairs)}")
    print(f"Profile written: {out_path}")
    print(f"Deprecated ngrams: {len(profile['deprecated_ngrams'])}")
    print(f"Preferred rewrites: {len(profile['preferred_rewrites'])}")
    if args.ai_compare:
        print(f"AI deprecated phrases: {len(profile.get('ai_deprecated_phrases', []))}")
        print(f"AI style rules: {len(profile.get('ai_style_rules', []))}")


if __name__ == "__main__":
    main()
