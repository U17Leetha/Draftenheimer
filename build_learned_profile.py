#!/usr/bin/env python3
"""Build Draftenheimer learned QA profile from versioned report pairs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from qa_scan import NS, extract_format_snapshot, parse_docx

VERSION_TOKEN_RE = re.compile(r"(?i)(?:^|[^a-z0-9])v?(\d+)\.(\d+)(?:v)?(?:$|[^a-z0-9])")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z']+")
CACHE_SCHEMA_VERSION = 2
AI_PROMPT_VERSION = 1


def parse_version(stem: str) -> tuple[int, int] | None:
    matches = list(VERSION_TOKEN_RE.finditer(stem))
    if not matches:
        return None
    m = matches[-1]
    return int(m.group(1)), int(m.group(2))


def strip_version_tokens(stem: str) -> str:
    return VERSION_TOKEN_RE.sub(" ", stem)


def canonical_report_key(file_path: Path) -> str | None:
    stem = file_path.stem
    if parse_version(stem) is None:
        return None
    stem = strip_version_tokens(stem)
    stem = re.sub(r"[\W_]+", "", stem).lower()
    return stem or None


def discover_pairs(reports_dir: Path, pair_mode: str = "consecutive") -> list[tuple[Path, Path]]:
    versioned: dict[str, list[tuple[tuple[int, int], Path]]] = defaultdict(list)
    if not reports_dir.exists():
        return []
    for file_path in sorted(reports_dir.iterdir()):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        key = canonical_report_key(file_path)
        if not key:
            continue
        version = parse_version(file_path.stem)
        if version is None:
            continue
        versioned[key].append((version, file_path))

    out: list[tuple[Path, Path]] = []
    for items in versioned.values():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda x: (x[0][0], x[0][1], x[1].name.lower()))
        if pair_mode == "latest":
            out.append((items[0][1], items[-1][1]))
            continue
        for i in range(len(items) - 1):
            out.append((items[i][1], items[i + 1][1]))
    return sorted(out, key=lambda p: p[0].name.lower())


def _revision_text(node, deleted: bool) -> str:
    pieces = []
    if deleted:
        for t in node.findall('.//w:delText', NS):
            if t.text:
                pieces.append(t.text)
    else:
        for t in node.findall('.//w:t', NS):
            if t.text:
                pieces.append(t.text)
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def extract_track_changes(docx_path: Path) -> dict:
    out = {"inserted": [], "deleted": [], "replacements": []}
    try:
        with zipfile.ZipFile(docx_path) as zf:
            with zf.open('word/document.xml') as f:
                root = ET.parse(f).getroot()
    except Exception:
        return out

    body = root.find('w:body', NS)
    if body is None:
        return out

    for p in body.findall('w:p', NS):
        pending_deleted: list[str] = []
        for child in p.iter():
            tag = child.tag.split('}')[-1]
            if tag == 'del':
                txt = _revision_text(child, deleted=True)
                if txt:
                    out['deleted'].append(txt)
                    pending_deleted.append(txt)
            elif tag == 'ins':
                txt = _revision_text(child, deleted=False)
                if txt:
                    out['inserted'].append(txt)
                    if pending_deleted:
                        out['replacements'].append({'from': pending_deleted.pop(0), 'to': txt})
    return out


def paragraphs_from_docx(path: Path) -> list[str]:
    blocks = parse_docx(str(path))
    return [b["text"].strip() for b in blocks if b["type"] == "p" and b["text"].strip()]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]


def ngrams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _rewrite_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", str(text or '').lower())
    stop = {
        'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'for', 'with',
        'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'that', 'this',
    }
    return [w for w in words if w and w not in stop]


def _rewrite_is_high_signal(old_text: str, new_text: str) -> bool:
    if '\n' in str(old_text) or '\n' in str(new_text):
        return False
    old_toks = _rewrite_tokens(old_text)
    new_toks = _rewrite_tokens(new_text)
    if len(old_toks) < 2 or len(new_toks) < 2:
        return False
    overlap = set(old_toks).intersection(set(new_toks))
    if len(old_toks) >= 6 and len(new_toks) >= 6 and len(overlap) < 2:
        return False
    if len(overlap) == 0:
        return False
    return True


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
            if not _rewrite_is_high_signal(old_s, new_s):
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


def _summarize_format_rules(format_snapshots: list[dict]) -> dict:
    para_styles = Counter()
    para_align = Counter()
    tbl_cols = Counter()
    image_widths = []
    image_aspects = []

    for snap in format_snapshots:
        para_styles.update(snap.get('paragraph_style_counts', {}))
        para_align.update(snap.get('paragraph_alignment_counts', {}))
        tbl_cols.update(snap.get('table_column_counts', {}))
        image_widths.extend([int(x) for x in snap.get('image_width_emu', []) if isinstance(x, (int, float)) and int(x) > 0])
        image_aspects.extend([float(x) for x in snap.get('image_aspect_ratios', []) if isinstance(x, (int, float)) and float(x) > 0])

    total_para = sum(para_styles.values())
    total_tbl = sum(tbl_cols.values())

    preferred_styles = []
    if total_para > 0:
        for style_id, count in para_styles.most_common():
            pct = count / total_para
            if count >= 2 and pct >= 0.03:
                preferred_styles.append({'style_id': style_id, 'count': count, 'pct': round(pct, 4)})

    preferred_alignments = []
    if total_para > 0:
        for align, count in para_align.most_common():
            pct = count / total_para
            if count >= 2 and pct >= 0.03:
                preferred_alignments.append({'alignment': align, 'count': count, 'pct': round(pct, 4)})

    preferred_tbl_cols = []
    if total_tbl > 0:
        for cols, count in sorted(tbl_cols.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_tbl
            if count >= 2 and pct >= 0.2:
                preferred_tbl_cols.append({'columns': int(cols), 'count': count, 'pct': round(pct, 4)})

    image_width_rule = None
    if len(image_widths) >= 3:
        ws = sorted(image_widths)
        image_width_rule = {
            'min': ws[int(len(ws) * 0.1)],
            'max': ws[int(len(ws) * 0.9)],
            'median': ws[len(ws) // 2],
            'samples': len(ws),
        }

    image_aspect_rule = None
    if len(image_aspects) >= 3:
        ars = sorted(image_aspects)
        image_aspect_rule = {
            'min': round(ars[int(len(ars) * 0.1)], 4),
            'max': round(ars[int(len(ars) * 0.9)], 4),
            'median': round(ars[len(ars) // 2], 4),
            'samples': len(ars),
        }

    return {
        'preferred_paragraph_styles': preferred_styles,
        'preferred_paragraph_alignments': preferred_alignments,
        'preferred_table_column_counts': preferred_tbl_cols,
        'image_width_emu': image_width_rule,
        'image_aspect_ratio': image_aspect_rule,
        'training_docs_count': len(format_snapshots),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_sig(path: Path) -> dict:
    st = path.stat()
    return {
        'size': int(st.st_size),
        'mtime_ns': int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1_000_000_000))),
    }


def _sig_equal(a: dict | None, b: dict | None) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    return int(a.get('size', -1)) == int(b.get('size', -2)) and int(a.get('mtime_ns', -1)) == int(b.get('mtime_ns', -2))


def _pair_id(reports_dir: Path, old_path: Path, new_path: Path) -> str:
    try:
        old_rel = old_path.resolve().relative_to(reports_dir.resolve())
    except Exception:
        old_rel = old_path.name
    try:
        new_rel = new_path.resolve().relative_to(reports_dir.resolve())
    except Exception:
        new_rel = new_path.name
    return f"{old_rel}::{new_rel}"


def _analysis_fingerprint(ngram_min: int, ngram_max: int, include_track_changes: bool, track_weight: int, ai_compare: bool, ai_provider: str, ai_model: str | None, bedrock_region: str, bedrock_profile: str, ollama_url: str) -> str:
    payload = {
        'schema': CACHE_SCHEMA_VERSION,
        'ai_prompt_version': AI_PROMPT_VERSION,
        'ngram_min': ngram_min,
        'ngram_max': ngram_max,
        'include_track_changes': include_track_changes,
        'track_weight': track_weight,
        'ai_compare': ai_compare,
        'ai_provider': ai_provider if ai_compare else None,
        'ai_model': ai_model if ai_compare else None,
        'bedrock_region': bedrock_region if ai_compare and ai_provider == 'bedrock' else None,
        'bedrock_profile': bedrock_profile if ai_compare and ai_provider == 'bedrock' else None,
        'ollama_url': ollama_url if ai_compare and ai_provider == 'ollama' else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {'version': CACHE_SCHEMA_VERSION, 'pair_cache': {}}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return {'version': CACHE_SCHEMA_VERSION, 'pair_cache': {}}
        if not isinstance(data.get('pair_cache'), dict):
            data['pair_cache'] = {}
        return data
    except Exception:
        return {'version': CACHE_SCHEMA_VERSION, 'pair_cache': {}}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state['version'] = CACHE_SCHEMA_VERSION
    state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')


def _compute_pair_contribution(old_path: Path, new_path: Path, ngram_min: int, ngram_max: int, ai_compare: bool, ai_provider: str, ai_model: str | None, bedrock_region: str, bedrock_profile: str, ollama_url: str, include_track_changes: bool, track_weight: int) -> dict:
    old_paras = paragraphs_from_docx(old_path)
    new_paras = paragraphs_from_docx(new_path)
    old_text = "\n".join(old_paras)
    new_text = "\n".join(new_paras)

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
    track_change_totals = {'inserted': 0, 'deleted': 0, 'replacements': 0}

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

    if include_track_changes:
        for tracked in (extract_track_changes(old_path), extract_track_changes(new_path)):
            inserted = tracked.get('inserted', [])
            deleted = tracked.get('deleted', [])
            replacements = tracked.get('replacements', [])
            track_change_totals['inserted'] += len(inserted)
            track_change_totals['deleted'] += len(deleted)
            track_change_totals['replacements'] += len(replacements)

            for s in inserted:
                for sentence in split_sentences(s):
                    sentence_new[norm_text(sentence)] += max(1, track_weight)
                words = WORD_RE.findall(s.lower())
                for n in range(ngram_min, ngram_max + 1):
                    ngram_new.update(ngrams(words, n))

            for s in deleted:
                for sentence in split_sentences(s):
                    sentence_old[norm_text(sentence)] += max(1, track_weight)
                words = WORD_RE.findall(s.lower())
                for n in range(ngram_min, ngram_max + 1):
                    ngram_old.update(ngrams(words, n))

            for r in replacements:
                old_s = (r.get('from') or '').strip()
                new_s = (r.get('to') or '').strip()
                if not old_s or not new_s or old_s == new_s:
                    continue
                key = f"{norm_text(old_s)}|||{norm_text(new_s)}"
                rewrite_votes[key] += max(1, track_weight)
                rewrite_examples.setdefault(key, (old_s, new_s))

    ai_compare_error = None
    if ai_compare and ai_model:
        try:
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
                if not _rewrite_is_high_signal(old_t, new_t):
                    continue
                if old_t and new_t and old_t != new_t:
                    key = f"{norm_text(old_t)}|||{norm_text(new_t)}"
                    ai_rewrite_votes[key] += 1
                    ai_rewrite_examples.setdefault(key, (old_t, new_t, reason))
            for rule in ai_result.get("style_rules", []):
                if isinstance(rule, str) and rule.strip():
                    ai_style_votes[rule.strip()] += 1
        except Exception as e:
            ai_compare_error = str(e)

    return {
        'source': {'old': str(old_path), 'new': str(new_path)},
        'sentence_old': dict(sentence_old),
        'sentence_new': dict(sentence_new),
        'ngram_old': dict(ngram_old),
        'ngram_new': dict(ngram_new),
        'case_votes': {k: dict(v) for k, v in case_votes.items()},
        'rewrite_votes': dict(rewrite_votes),
        'rewrite_examples': {k: [v[0], v[1]] for k, v in rewrite_examples.items()},
        'ai_phrase_votes': dict(ai_phrase_votes),
        'ai_rewrite_votes': dict(ai_rewrite_votes),
        'ai_rewrite_examples': {k: [v[0], v[1], v[2]] for k, v in ai_rewrite_examples.items()},
        'ai_style_votes': dict(ai_style_votes),
        'ai_compare_error': ai_compare_error,
        'track_change_totals': track_change_totals,
        'format_snapshot': extract_format_snapshot(new_path),
    }


def _apply_pair_contribution(contrib: dict, sentence_old: Counter, sentence_new: Counter, ngram_old: Counter, ngram_new: Counter, case_votes: dict[str, Counter], rewrite_votes: Counter, rewrite_examples: dict[str, tuple[str, str]], ai_phrase_votes: Counter, ai_rewrite_votes: Counter, ai_rewrite_examples: dict[str, tuple[str, str, str]], ai_style_votes: Counter, format_snapshots: list[dict], track_change_totals: dict) -> None:
    sentence_old.update(contrib.get('sentence_old', {}))
    sentence_new.update(contrib.get('sentence_new', {}))
    ngram_old.update(contrib.get('ngram_old', {}))
    ngram_new.update(contrib.get('ngram_new', {}))

    for lower, votes in contrib.get('case_votes', {}).items():
        case_votes[lower].update(votes)

    rewrite_votes.update(contrib.get('rewrite_votes', {}))
    for key, ex in contrib.get('rewrite_examples', {}).items():
        if isinstance(ex, list) and len(ex) >= 2:
            rewrite_examples.setdefault(key, (ex[0], ex[1]))

    ai_phrase_votes.update(contrib.get('ai_phrase_votes', {}))
    ai_rewrite_votes.update(contrib.get('ai_rewrite_votes', {}))
    for key, ex in contrib.get('ai_rewrite_examples', {}).items():
        if isinstance(ex, list) and len(ex) >= 3:
            ai_rewrite_examples.setdefault(key, (ex[0], ex[1], ex[2]))
    ai_style_votes.update(contrib.get('ai_style_votes', {}))

    fmt = contrib.get('format_snapshot')
    if isinstance(fmt, dict):
        format_snapshots.append(fmt)

    tt = contrib.get('track_change_totals', {})
    for k in ('inserted', 'deleted', 'replacements'):
        track_change_totals[k] += int(tt.get(k, 0) or 0)


def build_profile(
    reports_dir: Path,
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
    include_track_changes: bool,
    track_weight: int,
    state_file: Path,
    rebuild_mode: str,
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
    track_change_totals = {"inserted": 0, "deleted": 0, "replacements": 0}
    format_snapshots: list[dict] = []

    state = _load_state(state_file)
    cache: dict = state.get('pair_cache', {})
    active_keys = set()
    reused_pairs = 0
    recomputed_pairs = 0
    ai_compare_errors = 0

    fp = _analysis_fingerprint(
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        include_track_changes=include_track_changes,
        track_weight=track_weight,
        ai_compare=ai_compare,
        ai_provider=ai_provider,
        ai_model=ai_model,
        bedrock_region=bedrock_region,
        bedrock_profile=bedrock_profile,
        ollama_url=ollama_url,
    )

    for old_path, new_path in pairs:
        pid = _pair_id(reports_dir, old_path, new_path)
        active_keys.add(pid)
        old_sig = _file_sig(old_path)
        new_sig = _file_sig(new_path)

        cached = cache.get(pid)
        contrib = None
        if rebuild_mode != 'full' and isinstance(cached, dict):
            if (
                cached.get('analysis_fingerprint') == fp
                and _sig_equal(cached.get('old_sig'), old_sig)
                and _sig_equal(cached.get('new_sig'), new_sig)
                and isinstance(cached.get('contribution'), dict)
            ):
                contrib = cached['contribution']
                reused_pairs += 1

        old_hash = None
        new_hash = None
        if contrib is None and rebuild_mode != 'full' and isinstance(cached, dict):
            # Fallback safety path: if metadata changed, validate via content hashes.
            old_hash = _sha256_file(old_path)
            new_hash = _sha256_file(new_path)
            if (
                cached.get('analysis_fingerprint') == fp
                and cached.get('old_hash') == old_hash
                and cached.get('new_hash') == new_hash
                and isinstance(cached.get('contribution'), dict)
            ):
                contrib = cached['contribution']
                reused_pairs += 1

        if contrib is None:
            if old_hash is None:
                old_hash = _sha256_file(old_path)
            if new_hash is None:
                new_hash = _sha256_file(new_path)
            contrib = _compute_pair_contribution(
                old_path=old_path,
                new_path=new_path,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
                ai_compare=ai_compare,
                ai_provider=ai_provider,
                ai_model=ai_model,
                bedrock_region=bedrock_region,
                bedrock_profile=bedrock_profile,
                ollama_url=ollama_url,
                include_track_changes=include_track_changes,
                track_weight=track_weight,
            )
            cache[pid] = {
                'old': str(old_path),
                'new': str(new_path),
                'old_sig': old_sig,
                'new_sig': new_sig,
                'old_hash': old_hash,
                'new_hash': new_hash,
                'analysis_fingerprint': fp,
                'contribution': contrib,
            }
            recomputed_pairs += 1

        sources.append(contrib.get('source', {'old': str(old_path), 'new': str(new_path)}))
        if contrib.get('ai_compare_error'):
            ai_compare_errors += 1
        _apply_pair_contribution(
            contrib,
            sentence_old,
            sentence_new,
            ngram_old,
            ngram_new,
            case_votes,
            rewrite_votes,
            rewrite_examples,
            ai_phrase_votes,
            ai_rewrite_votes,
            ai_rewrite_examples,
            ai_style_votes,
            format_snapshots,
            track_change_totals,
        )

    # prune stale cache entries not in this dataset
    for key in list(cache.keys()):
        if key not in active_keys:
            cache.pop(key, None)

    state['pair_cache'] = cache
    _save_state(state_file, state)

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

    format_rules = _summarize_format_rules(format_snapshots)

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
        "format_rules": format_rules,
        "track_changes_enabled": include_track_changes,
        "track_changes_weight": track_weight,
        "track_changes_totals": track_change_totals,
        "learning_cache": {
            "rebuild_mode": rebuild_mode,
            "state_file": str(state_file),
            "pairs_reused": reused_pairs,
            "pairs_recomputed": recomputed_pairs,
            "ai_compare_errors": ai_compare_errors,
        },
    }


def default_reports_dir() -> str:
    base = Path(__file__).resolve().parent
    candidates = [
        base / "reports",
        Path("reports"),
        Path("training/qa_pair_training/reports"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)
    return str((Path(__file__).resolve().parent / "reports"))


def default_state_file() -> str:
    return str(Path(__file__).resolve().parent / "draftenheimer_learning_state.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Draftenheimer learned QA profile from versioned report pairs.")
    parser.add_argument("--reports-dir", default=default_reports_dir(), help="Directory containing versioned report files (v0.1, v0.2, v0.3, ...).")
    parser.add_argument("--pair-mode", choices=["consecutive", "latest"], default="consecutive", help="consecutive: v0.1->v0.2->v0.3 (default). latest: first->latest only.")
    parser.add_argument("--out", default="draftenheimer_profile.json", help="Output profile JSON path.")
    parser.add_argument("--state-file", default=default_state_file(), help="Path to incremental learning cache state JSON.")
    parser.add_argument("--rebuild-mode", choices=["incremental", "full"], default="incremental", help="incremental: reuse unchanged pair analysis from cache. full: recompute all pairs.")
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
    parser.add_argument("--include-track-changes", action="store_true", default=True, help="Use Word tracked insert/delete changes as extra learning signals.")
    parser.add_argument("--no-track-changes", action="store_false", dest="include_track_changes", help="Disable tracked-change learning.")
    parser.add_argument("--track-weight", type=int, default=2, help="Weight applied to tracked-change signals (default: 2).")
    parser.add_argument("--allow-empty", action="store_true", help="Exit successfully when no versioned pairs are found.")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    pairs = discover_pairs(reports_dir, pair_mode=args.pair_mode)
    if not pairs:
        if args.allow_empty:
            print(f"No versioned report pairs found in: {reports_dir}. Skipping profile rebuild.")
            raise SystemExit(0)
        raise SystemExit(f"No versioned report pairs found in: {reports_dir}")

    if args.ai_compare and not args.ai_model:
        raise SystemExit("--ai-compare requires --ai-model")

    profile = build_profile(
        reports_dir=reports_dir,
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
        include_track_changes=args.include_track_changes,
        track_weight=args.track_weight,
        state_file=Path(args.state_file),
        rebuild_mode=args.rebuild_mode,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pairs used: {len(pairs)}")
    print(f"Profile written: {out_path}")
    print(f"Deprecated ngrams: {len(profile['deprecated_ngrams'])}")
    print(f"Preferred rewrites: {len(profile['preferred_rewrites'])}")
    fmt = profile.get('format_rules', {})
    if isinstance(fmt, dict):
        print(
            "Format rules: "
            f"styles={len(fmt.get('preferred_paragraph_styles', []))}, "
            f"table_patterns={len(fmt.get('preferred_table_column_counts', []))}, "
            f"image_width_rule={'yes' if fmt.get('image_width_emu') else 'no'}"
        )
    lc = profile.get('learning_cache', {})
    print(
        "Learning cache: "
        f"mode={lc.get('rebuild_mode')}, reused={lc.get('pairs_reused', 0)}, recomputed={lc.get('pairs_recomputed', 0)}, ai_compare_errors={lc.get('ai_compare_errors', 0)}"
    )
    if args.include_track_changes:
        t = profile.get('track_changes_totals', {})
        print(f"Track changes (ins/del/repl): {t.get('inserted', 0)}/{t.get('deleted', 0)}/{t.get('replacements', 0)}")
    if args.ai_compare:
        print(f"AI deprecated phrases: {len(profile.get('ai_deprecated_phrases', []))}")
        print(f"AI style rules: {len(profile.get('ai_style_rules', []))}")


if __name__ == "__main__":
    main()
