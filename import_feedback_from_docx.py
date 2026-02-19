#!/usr/bin/env python3
import argparse
import json
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
}


def _text_from_comment(comment_el):
    texts = []
    for t in comment_el.findall('.//w:t', NS):
        if t.text:
            texts.append(t.text)
    return ''.join(texts).strip()


def _decision_from_text(text):
    if not text:
        return None
    t = text.strip()
    if re.search(r'\[(?:\s*)reject(?:ed)?(?:\s*)\]', t, flags=re.IGNORECASE):
        return 'reject'
    if re.search(r'\[(?:\s*)accept(?:ed)?(?:\s*)\]', t, flags=re.IGNORECASE):
        return 'accept'
    if re.search(r'\breject(?:ed)?\s*[:\-]', t, flags=re.IGNORECASE):
        return 'reject'
    if re.search(r'\baccept(?:ed)?\s*[:\-]', t, flags=re.IGNORECASE):
        return 'accept'
    if re.fullmatch(r'\s*reject(?:ed)?\s*', t, flags=re.IGNORECASE):
        return 'reject'
    if re.fullmatch(r'\s*accept(?:ed)?\s*', t, flags=re.IGNORECASE):
        return 'accept'
    return None


def _extract_rewrite_pair(text):
    if not text:
        return None
    normalized = (
        text.replace('“', '"')
            .replace('”', '"')
            .replace('‘', "'")
            .replace('’', "'")
            .replace('→', '->')
            .replace('⟶', '->')
            .replace('⟹', '->')
    )
    m = re.search(r'Preferred rewrite:\s*"([^"]+)"\s*(?:->|=>)\s*"([^"]+)"', normalized, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'"([^"]+)"\s*(?:->|=>)\s*"([^"]+)"', normalized, flags=re.IGNORECASE)
        if not m:
            return None
    old_t = m.group(1).strip()
    new_t = m.group(2).strip()
    if not old_t or not new_t:
        return None
    return {'from': old_t, 'to': new_t}


def _extract_diag_rule(text):
    if not text:
        return None
    m = re.match(r'^\[[^\]]+\]\[([^\]]+)\]\[[^\]]+\]\s*(.*)$', text.strip(), flags=re.DOTALL)
    if not m:
        return None
    code = m.group(1).strip()
    msg = re.sub(r'\s+', ' ', (m.group(2) or '').strip())
    if not code and not msg:
        return None
    if len(msg) > 220:
        msg = msg[:220].rstrip()
    return {'code': code, 'message_contains': msg}


def _parent_comment_id(comment_el):
    for k, v in comment_el.attrib.items():
        if k.endswith('}parentId') or k == 'parentId':
            return str(v)
    return None


def _load_feedback(path):
    default = {
        'accepted_rewrites': [],
        'rejected_rewrites': [],
        'accepted_diagnostics': [],
        'rejected_diagnostics': [],
    }
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        out = dict(default)

        for key in ('accepted_rewrites', 'rejected_rewrites'):
            vals = data.get(key, [])
            cleaned = []
            if isinstance(vals, list):
                for v in vals:
                    if not isinstance(v, dict):
                        continue
                    old_t = str(v.get('from', '')).strip()
                    new_t = str(v.get('to', '')).strip()
                    if old_t and new_t:
                        cleaned.append({'from': old_t, 'to': new_t})
            out[key] = cleaned

        for key in ('accepted_diagnostics', 'rejected_diagnostics'):
            vals = data.get(key, [])
            cleaned = []
            if isinstance(vals, list):
                for v in vals:
                    if not isinstance(v, dict):
                        continue
                    code = str(v.get('code', '')).strip()
                    msg = str(v.get('message_contains', '')).strip()
                    if code or msg:
                        cleaned.append({'code': code, 'message_contains': msg})
            out[key] = cleaned

        return out
    except Exception:
        return default


def _norm_pair_key(pair):
    old_t = re.sub(r'\s+', ' ', pair['from'].strip().lower())
    new_t = re.sub(r'\s+', ' ', pair['to'].strip().lower())
    return (old_t, new_t)


def _norm_diag_key(rule):
    code = re.sub(r'\s+', ' ', rule.get('code', '').strip().upper())
    msg = re.sub(r'\s+', ' ', rule.get('message_contains', '').strip().lower())
    return (code, msg)


def _extract_items_from_base_comment(text):
    items = []
    pair = _extract_rewrite_pair(text)
    if pair:
        items.append(('rewrite', pair))
    diag = _extract_diag_rule(text)
    if diag:
        items.append(('diagnostic', diag))
    return items


def _find_decision_for_base(ordered_comments, children_by_parent, idx, cid, base_author, author):
    c = ordered_comments[idx][1]
    base_text = _text_from_comment(c)

    decision = _decision_from_text(base_text)
    if decision is not None:
        return decision

    for reply in children_by_parent.get(cid, []):
        reply_text = _text_from_comment(reply)
        decision = _decision_from_text(reply_text)
        if decision is not None:
            return decision

    # Fallback: nearby top-level decision comment.
    try:
        base_id_int = int(cid)
    except Exception:
        base_id_int = None

    for j in range(idx + 1, min(idx + 7, len(ordered_comments))):
        next_cid, next_c = ordered_comments[j]
        if _parent_comment_id(next_c) is not None:
            continue
        next_author = (next_c.get(f"{{{NS['w']}}}author") or '').strip()
        if author and next_author == author:
            continue
        next_text = _text_from_comment(next_c)
        d = _decision_from_text(next_text)
        if d is None:
            continue
        if base_id_int is not None:
            try:
                next_id_int = int(next_cid)
                if next_id_int - base_id_int > 8:
                    continue
            except Exception:
                pass
        return d

    return None


def import_feedback_from_docx(docx_path, feedback_path, author='Draftenheimer', dry_run=False):
    with tempfile.TemporaryDirectory(prefix='draftenheimer_feedback_') as td:
        with zipfile.ZipFile(docx_path, 'r') as zf:
            zf.extractall(td)

        comments_xml = os.path.join(td, 'word', 'comments.xml')
        if not os.path.exists(comments_xml):
            return {
                'accepted_added': 0,
                'rejected_added': 0,
                'accepted_diagnostics_added': 0,
                'rejected_diagnostics_added': 0,
                'reviewed_rewrites_found': 0,
                'reviewed_diagnostics_found': 0,
                'message': 'No comments.xml found in document.',
            }

        tree = ET.parse(comments_xml)
        root = tree.getroot()

        children_by_parent = defaultdict(list)
        ordered_comments = []
        for c in root.findall('w:comment', NS):
            cid = c.get(f"{{{NS['w']}}}id")
            if cid is None:
                continue
            cid_s = str(cid)
            ordered_comments.append((cid_s, c))
            pid = _parent_comment_id(c)
            if pid is not None:
                children_by_parent[str(pid)].append(c)

        reviewed_rewrites = []
        reviewed_diags = []

        for idx, (cid, c) in enumerate(ordered_comments):
            c_author = (c.get(f"{{{NS['w']}}}author") or '').strip()
            if author and c_author != author:
                continue

            base_text = _text_from_comment(c)
            items = _extract_items_from_base_comment(base_text)
            if not items:
                continue

            decision = _find_decision_for_base(ordered_comments, children_by_parent, idx, cid, c_author, author)
            if decision is None:
                continue

            for item_type, item in items:
                if item_type == 'rewrite':
                    reviewed_rewrites.append((item, decision))
                elif item_type == 'diagnostic':
                    reviewed_diags.append((item, decision))

        feedback = _load_feedback(feedback_path)
        accepted = feedback['accepted_rewrites']
        rejected = feedback['rejected_rewrites']
        accepted_diag = feedback['accepted_diagnostics']
        rejected_diag = feedback['rejected_diagnostics']

        accepted_keys = {_norm_pair_key(x) for x in accepted}
        rejected_keys = {_norm_pair_key(x) for x in rejected}
        accepted_diag_keys = {_norm_diag_key(x) for x in accepted_diag}
        rejected_diag_keys = {_norm_diag_key(x) for x in rejected_diag}

        accepted_added = 0
        rejected_added = 0
        for pair, decision in reviewed_rewrites:
            key = _norm_pair_key(pair)
            if decision == 'accept':
                if key not in accepted_keys:
                    accepted.append(pair)
                    accepted_keys.add(key)
                    accepted_added += 1
            elif decision == 'reject':
                if key not in rejected_keys:
                    rejected.append(pair)
                    rejected_keys.add(key)
                    rejected_added += 1

        accepted_diag_added = 0
        rejected_diag_added = 0
        for rule, decision in reviewed_diags:
            key = _norm_diag_key(rule)
            if decision == 'accept':
                if key not in accepted_diag_keys:
                    accepted_diag.append(rule)
                    accepted_diag_keys.add(key)
                    accepted_diag_added += 1
            elif decision == 'reject':
                if key not in rejected_diag_keys:
                    rejected_diag.append(rule)
                    rejected_diag_keys.add(key)
                    rejected_diag_added += 1

        if not dry_run:
            out = {
                'accepted_rewrites': accepted,
                'rejected_rewrites': rejected,
                'accepted_diagnostics': accepted_diag,
                'rejected_diagnostics': rejected_diag,
            }
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

        return {
            'accepted_added': accepted_added,
            'rejected_added': rejected_added,
            'accepted_diagnostics_added': accepted_diag_added,
            'rejected_diagnostics_added': rejected_diag_added,
            'reviewed_rewrites_found': len(reviewed_rewrites),
            'reviewed_diagnostics_found': len(reviewed_diags),
            'feedback_path': feedback_path,
            'dry_run': bool(dry_run),
        }


def main():
    ap = argparse.ArgumentParser(description='Import accepted/rejected rewrite and diagnostic decisions from an annotated DOCX into draftenheimer_feedback.json')
    ap.add_argument('annotated_docx', help='Path to reviewed annotated DOCX')
    ap.add_argument('--feedback', default=os.path.join(os.path.dirname(__file__), 'draftenheimer_feedback.json'), help='Path to feedback JSON output')
    ap.add_argument('--author', default='Draftenheimer', help='Only consider base comments from this author')
    ap.add_argument('--dry-run', action='store_true', help='Parse and report only; do not write feedback file')
    args = ap.parse_args()

    result = import_feedback_from_docx(args.annotated_docx, args.feedback, author=args.author, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
