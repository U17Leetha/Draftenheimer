#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import subprocess
import tempfile
import base64
import os as _os
from pathlib import Path

NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
}


def _text_from_p(p):
    texts = []
    for t in p.findall('.//w:t', NS):
        if t.text:
            texts.append(t.text)
    return ''.join(texts).strip()


def _style_id_from_p(p):
    ppr = p.find('w:pPr', NS)
    if ppr is None:
        return None
    pstyle = ppr.find('w:pStyle', NS)
    if pstyle is None:
        return None
    return pstyle.get(f'{{{NS["w"]}}}val')


def _load_styles(zf):
    styles = {}
    try:
        with zf.open('word/styles.xml') as f:
            tree = ET.parse(f)
        root = tree.getroot()
        for st in root.findall('w:style', NS):
            style_id = st.get(f'{{{NS["w"]}}}styleId')
            name_el = st.find('w:name', NS)
            if style_id and name_el is not None:
                name = name_el.get(f'{{{NS["w"]}}}val')
                if name:
                    styles[style_id] = name
    except KeyError:
        pass
    return styles


def _heading_level(style_name):
    if not style_name:
        return None
    s = style_name.lower().replace(' ', '')
    if s.startswith('heading'):
        m = re.match(r'heading(\d+)', s)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def _iter_blocks(root):
    body = root.find('w:body', NS)
    for child in body:
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            yield ('p', child)
        elif tag == 'tbl':
            yield ('tbl', child)


def _table_rows(tbl):
    rows = []
    for tr in tbl.findall('.//w:tr', NS):
        cells = []
        for tc in tr.findall('.//w:tc', NS):
            cell_txt = ' '.join(_text_from_p(p) for p in tc.findall('.//w:p', NS)).strip()
            cells.append(cell_txt)
        if any(cells):
            rows.append(cells)
    return rows


def parse_docx(docx_path):
    with zipfile.ZipFile(docx_path) as zf:
        styles = _load_styles(zf)
        with zf.open('word/document.xml') as f:
            tree = ET.parse(f)
    root = tree.getroot()
    blocks = []
    for kind, node in _iter_blocks(root):
        if kind == 'p':
            text = _text_from_p(node)
            if text == '':
                continue
            style_id = _style_id_from_p(node)
            style_name = styles.get(style_id)
            level = _heading_level(style_name)
            blocks.append({
                'type': 'p',
                'text': text,
                'style_id': style_id,
                'style_name': style_name,
                'heading_level': level,
            })
        else:
            rows = _table_rows(node)
            if rows:
                blocks.append({
                    'type': 'tbl',
                    'rows': rows,
                })
    return blocks


def build_sections(paragraphs):
    sections = []
    stack = []

    def add_section(sec):
        nonlocal sections, stack
        while stack and stack[-1]['level'] >= sec['level']:
            stack.pop()
        if stack:
            stack[-1]['children'].append(sec)
        else:
            sections.append(sec)
        stack.append(sec)

    current = None
    for p in paragraphs:
        if p['heading_level']:
            sec = {
                'title': p['text'],
                'level': p['heading_level'],
                'content': [],
                'children': [],
            }
            add_section(sec)
            current = sec
        else:
            if current is None:
                # Pre-heading content
                sec = {
                    'title': 'Preamble',
                    'level': 0,
                    'content': [],
                    'children': [],
                }
                add_section(sec)
                current = sec
            current['content'].append(p['text'])
    return sections


def iter_sections(sections):
    for s in sections:
        yield s
        for c in iter_sections(s['children']):
            yield c


def section_text_by_title(sections, keyword):
    texts = []
    for s in iter_sections(sections):
        if keyword.lower() in s['title'].lower():
            texts.append(collect_content_text(s))
    return '\n'.join(t for t in texts if t)


def _table_has_threat_level(rows):
    for r in rows[:4]:
        for c in r:
            if 'Threat Level' in c:
                return True
    return False


def _table_value(rows, label):
    # Find value in row like ["Threat Level", "High", "CVSS", "7.1 ..."]
    for r in rows:
        for i, c in enumerate(r):
            if c.strip().lower() == label.lower():
                if i + 1 < len(r):
                    return r[i + 1].strip()
    return None


def find_findings(blocks):
    findings = []
    prev_p = None
    for i, b in enumerate(blocks):
        if b['type'] == 'p':
            if b['text'].strip():
                prev_p = b
            continue
        if b['type'] == 'tbl' and _table_has_threat_level(b['rows']):
            title = prev_p['text'].strip() if prev_p else 'Untitled Finding'
            threat = _table_value(b['rows'], 'Threat Level')
            cvss = _table_value(b['rows'], 'CVSS')
            finding = {
                'title': title,
                'threat_level': threat,
                'cvss_text': cvss,
                'table_rows': b['rows'],
                'content_paragraphs': [],
            }
            # Collect following paragraphs until next threat-level table
            j = i + 1
            while j < len(blocks):
                nb = blocks[j]
                if nb['type'] == 'tbl' and _table_has_threat_level(nb['rows']):
                    break
                if nb['type'] == 'p':
                    finding['content_paragraphs'].append(nb['text'])
                j += 1
            findings.append(finding)
    return findings


def collect_content_text(section):
    texts = []
    texts.extend(section.get('content', []))
    for c in section.get('children', []):
        texts.append(c['title'])
        texts.extend(c.get('content', []))
        for cc in c.get('children', []):
            texts.append(cc['title'])
            texts.extend(cc.get('content', []))
    return '\n'.join(texts)


def detect_fields_from_paras(paragraphs):
    fields = defaultdict(list)
    label_map = {
        'narrative': ['narrative', 'vulnerability', 'description'],
        'business_impact': ['business impact', 'business imapct', 'impact'],
        'recommendations': ['recommendations', 'recommendation', 'remediation'],
        'evidence': ['evidence', 'proof', 'observation', 'poc', 'proof of concept'],
        'reproduction': ['steps to reproduce', 'reproduction steps', 'reproduction', 'reproduce'],
    }
    current = None
    for p in paragraphs:
        line = p.strip()
        if not line:
            continue
        m = re.match(r'^([A-Za-z\s]+):\s*$', line)
        if m:
            label = m.group(1).strip().lower()
            found = None
            for k, names in label_map.items():
                if label in names:
                    found = k
                    break
            current = found
            if current and current not in fields:
                fields[current] = []
            continue
        if current:
            fields[current].append(line)
    return fields


def detect_implicit_evidence(paragraphs):
    # Treat figure/screenshot references or PoC mentions as evidence
    markers = [
        r'^figure\b',
        r'^fig\.\b',
        r'\bscreenshot\b',
        r'\bproof of concept\b',
        r'\bpoc\b',
        r'\bpacket capture\b',
        r'\bburp\b',
    ]
    for p in paragraphs:
        line = p.strip().lower()
        if not line:
            continue
        for m in markers:
            if re.search(m, line):
                return True
    return False


def detect_threat_level(text):
    # Prefer explicit "Threat Level" label when present
    m = re.search(r'\bthreat\s*level\s*[:=]?\s*(critical|high|medium|low|informational)\b', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # Fallback to severity keywords if no explicit label exists
    m = re.search(r'\b(critical|high|medium|low|informational)\b', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return None


def detect_severity(text):
    m = re.search(r'\b(critical|high|medium|low)\b', text.lower())
    return m.group(1).capitalize() if m else None


def detect_cvss(text):
    if not text:
        return None
    lower = text.lower()
    # Prefer patterns like "7.1 CVSS:4.0/AV:N/..."
    m = re.search(r'(\d(?:\.\d)?)\s*cvss\s*:', lower)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Handle "CVSS 3.1: 7.5" or "CVSS: 7.5"
    m = re.search(r'cvss\s*(v\d(?:\.\d)?)?\s*[:=]\s*(\d(?:\.\d)?)', lower)
    if m:
        try:
            return float(m.group(2))
        except ValueError:
            pass
    # Fallback: first plausible score in 0.0-10.0
    for m in re.finditer(r'\b(\d(?:\.\d)?)\b', lower):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if 0.0 <= val <= 10.0 and val != 4.0:  # avoid CVSS 4.0 version token
            return val
    return None


def cvss_to_severity(score):
    if score is None:
        return None
    if score >= 9.0:
        return 'Critical'
    if score >= 7.0:
        return 'High'
    if score >= 4.0:
        return 'Medium'
    if score > 0:
        return 'Low'
    return 'None'


def split_sentences(text):
    # rough sentence split
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_sentence(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _has_url_or_ip(text):
    # Skip repeated-phrase checks on link/IP-heavy lines to reduce noise.
    url_pat = re.compile(r'(https?://|www\.)\S+', flags=re.IGNORECASE)
    ipv4_pat = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    # Simple IPv6 detection (compressed/full forms).
    ipv6_pat = re.compile(r'\b(?:[0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:]{1,4}\b')
    return bool(url_pat.search(text) or ipv4_pat.search(text) or ipv6_pat.search(text))


def normalize_for_substring(s):
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _load_ignore_config(path):
    default = {
        'ignore_deprecated_phrases': [],
        'ignore_rewrite_from_phrases': [],
        'ignore_diagnostic_codes': [],
        'ignore_message_contains': [],
    }
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        out = dict(default)
        for key in default:
            vals = data.get(key, [])
            if isinstance(vals, list):
                out[key] = [str(v) for v in vals if str(v).strip()]
        return out
    except Exception:
        return default


def _matches_any_phrase(text, phrases):
    if not text or not phrases:
        return False
    t = normalize_for_substring(text)
    for p in phrases:
        np = normalize_for_substring(p)
        if np and (np in t or t in np):
            return True
    return False




def _load_feedback_config(path):
    default = {
        'accepted_rewrites': [],
        'rejected_rewrites': [],
        'accepted_diagnostics': [],
        'rejected_diagnostics': [],
    }
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        out = dict(default)

        # Rewrite feedback
        for key in ('accepted_rewrites', 'rejected_rewrites'):
            vals = data.get(key, [])
            if isinstance(vals, list):
                cleaned = []
                for v in vals:
                    if not isinstance(v, dict):
                        continue
                    old_t = str(v.get('from', '')).strip()
                    new_t = str(v.get('to', '')).strip()
                    if old_t and new_t:
                        cleaned.append({'from': old_t, 'to': new_t})
                out[key] = cleaned

        # Diagnostic feedback
        for key in ('accepted_diagnostics', 'rejected_diagnostics'):
            vals = data.get(key, [])
            if isinstance(vals, list):
                cleaned = []
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


def _rewrite_key(old_text, new_text):
    return (
        normalize_for_substring(str(old_text or '')),
        normalize_for_substring(str(new_text or '')),
    )


def _merge_rewrite_rules(profile_rules, accepted_rules):
    merged = []
    seen = set()
    for src in (accepted_rules or []):
        key = _rewrite_key(src.get('from', ''), src.get('to', ''))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        merged.append({'from': src.get('from', ''), 'to': src.get('to', ''), 'count': 9999, 'source': 'accepted_feedback'})
    for src in (profile_rules or []):
        key = _rewrite_key(src.get('from', ''), src.get('to', ''))
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        merged.append(src)
    return merged


def _diag_is_ignored(diag, ignore_codes, ignore_msg_parts):
    code = str(diag.get('code', '')).strip()
    msg = normalize_for_substring(str(diag.get('message', '')))
    if code and code in (ignore_codes or []):
        return True
    for part in (ignore_msg_parts or []):
        np = normalize_for_substring(part)
        if np and np in msg:
            return True
    return False



def _diag_matches_feedback_rule(diag, rule):
    if not isinstance(rule, dict):
        return False
    rule_code = str(rule.get('code', '')).strip()
    rule_msg = normalize_for_substring(str(rule.get('message_contains', '')))
    if not rule_code and not rule_msg:
        return False

    diag_code = str(diag.get('code', '')).strip()
    diag_msg = normalize_for_substring(str(diag.get('message', '')))

    if rule_code and diag_code != rule_code:
        return False
    if rule_msg and rule_msg not in diag_msg:
        return False
    return True


def _diag_matches_any_feedback_rule(diag, rules):
    for rule in (rules or []):
        if _diag_matches_feedback_rule(diag, rule):
            return True
    return False


def _extract_quoted_phrases(text):
    if not text:
        return []
    normalized = (
        text.replace('“', '"')
            .replace('”', '"')
            .replace('‘', "'")
            .replace('’', "'")
    )
    phrases = []
    for m in re.finditer(r'"([^"]{3,120})"', normalized):
        phrases.append(m.group(1).strip())
    for m in re.finditer(r"'([^']{3,120})'", normalized):
        phrases.append(m.group(1).strip())
    out = []
    seen = set()
    for p in phrases:
        key = normalize_for_substring(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _count_phrase_occurrences(text, phrase):
    if not text or not phrase:
        return 0
    try:
        return len(re.findall(re.escape(phrase), text, flags=re.IGNORECASE))
    except re.error:
        return text.lower().count(phrase.lower())


def _augment_llm_message_with_counts(message, text):
    if not message or not text:
        return message, 0
    msg_l = message.lower()
    if (
        'overused phrase' not in msg_l
        and 'overused phrases' not in msg_l
        and 'avoid using' not in msg_l
        and 'avoid' not in msg_l
        and 'phrasing' not in msg_l
    ):
        return message, 0

    counts = []
    for phrase in _extract_quoted_phrases(message):
        c = _count_phrase_occurrences(text, phrase)
        if c > 0:
            counts.append((phrase, c))

    if not counts:
        return message, 0

    summary = ', '.join(f'"{p}"={c}x' for p, c in counts[:4])
    max_count = max(c for _, c in counts)
    if 'counts:' in msg_l:
        return message, max_count
    return f"{message} Counts in narrative: {summary}.", max_count


def _elevate_severity_for_repeat_count(severity, max_count):
    s = str(severity or 'suggestion').lower()
    if s == 'error':
        return 'error'
    if max_count >= 6:
        return 'error'
    if max_count >= 3 and s == 'suggestion':
        return 'warning'
    return s


def _message_is_phrase_repetition_warning(message):
    if not message:
        return False
    msg_l = message.lower()
    if not _extract_quoted_phrases(message):
        return False
    if 'overused phrase' in msg_l or 'overused phrases' in msg_l:
        return True
    if 'avoid using' in msg_l and 'phrasing' in msg_l:
        return True
    return False


def _tokens_for_overlap(text):
    words = re.findall(r'[a-z0-9]+', str(text or '').lower())
    stop = {
        'the', 'and', 'for', 'with', 'from', 'that', 'this', 'was', 'were',
        'are', 'is', 'to', 'of', 'in', 'on', 'by', 'or', 'a', 'an', 'as',
        'avoid', 'using', 'phrase', 'phrasing', 'consider', 'rephrasing',
        'obvious', 'redundant', 'information', 'stating', 'remove', 'review',
    }
    return {w for w in words if len(w) > 2 and w not in stop}


def _best_supporting_sentence(text, message=None, location_hint=None):
    sentences = split_sentences(text or '')
    if not sentences:
        return ''

    quoted = _extract_quoted_phrases(message or '')
    for q in quoted:
        for s in sentences:
            if re.search(re.escape(q), s, flags=re.IGNORECASE):
                return s

    target_tokens = set()
    target_tokens.update(_tokens_for_overlap(location_hint or ''))
    target_tokens.update(_tokens_for_overlap(message or ''))

    best = ''
    best_score = -1
    for s in sentences:
        st = _tokens_for_overlap(s)
        if not st:
            continue
        score = len(target_tokens.intersection(st)) if target_tokens else 0
        if score > best_score:
            best_score = score
            best = s

    return best or sentences[0]


def _augment_vague_llm_message(message, text, location_hint=None):
    if not message or not text:
        return message
    msg_l = message.lower()
    if not any(k in msg_l for k in ('obvious', 'redundant', 'stating the obvious')):
        return message
    if 'sentence:' in msg_l or 'excerpt:' in msg_l:
        return message

    sentence = _best_supporting_sentence(text, message=message, location_hint=location_hint)
    if not sentence:
        return message
    snippet = sentence.strip().replace('\n', ' ')
    if len(snippet) > 220:
        snippet = snippet[:217] + '...'
    return f"{message} Example sentence: \"{snippet}\""


def _llm_issue_supported_by_text(message, text):
    if not message or not text:
        return True

    msg = (
        message.replace('“', '"')
               .replace('”', '"')
               .replace('‘', "'")
               .replace('’', "'")
    )

    patterns = [
        r'instead of\s+[\'"]([^\'"]+)[\'"]',
        r'replace\s+[\'"]([^\'"]+)[\'"]\s+with',
        r'rather than\s+[\'"]([^\'"]+)[\'"]',
    ]

    old_phrases = []
    for pat in patterns:
        for m in re.finditer(pat, msg, flags=re.IGNORECASE):
            phrase = (m.group(1) or '').strip()
            if phrase:
                old_phrases.append(phrase)

    if not old_phrases:
        return True

    for phrase in old_phrases:
        if _count_phrase_occurrences(text, phrase) > 0:
            return True
    return False


def _extract_json_array_from_text(text):
    if not text:
        return None
    s = text.strip()
    # 1) Direct JSON array.
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # 2) Markdown code fences.
    for m in re.finditer(r'```(?:json)?\s*(.*?)```', s, flags=re.IGNORECASE | re.DOTALL):
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return obj
        except Exception:
            continue

    # 3) First balanced JSON array in text.
    start = s.find('[')
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, list):
                            return obj
                    except Exception:
                        break
                    break
        start = s.find('[', start + 1)
    return None


def detect_repetition(paragraphs, min_len=60, min_count=3):
    sentences = []
    for p in paragraphs:
        for s in split_sentences(p['text']):
            if _has_url_or_ip(s):
                continue
            if len(s) >= min_len:
                sentences.append(s)
    norm = [normalize_sentence(s) for s in sentences]
    counts = Counter(norm)
    repeated = []
    for n, c in counts.items():
        if c >= min_count:
            # find example original sentence
            example = next((s for s in sentences if normalize_sentence(s) == n), n)
            repeated.append((example, c))
    return repeated


def detect_sensitive(text):
    patterns = {
        'AWS Access Key': r'AKIA[0-9A-Z]{16}',
        'Private Key': r'-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----',
        'JWT': r'eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}',
        'Bearer Token': r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*',
        'Password Assignment': r'\bpassword\b\s*[:=]\s*[^\s]{6,}',
        'API Key Assignment': r'\b(api[_-]?key|token|secret)\b\s*[:=]\s*[^\s]{8,}',
    }
    hits = []
    for name, pat in patterns.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            snippet = text[max(0, m.start()-20):m.end()+20]
            hits.append((name, snippet.strip()))
    return hits


def detect_hygiene(text):
    issues = []
    if re.search(r'\b(\w+)\s+\1\b', text, flags=re.IGNORECASE):
        issues.append('Repeated word')
    if '  ' in text:
        issues.append('Double spaces')
    # Flag mid-sentence capitalization for specific words
    target_words = [
        'critical', 'high', 'medium', 'low', 'informational',
        'confidentiality', 'integrity', 'availability'
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # find the first word to avoid start-of-sentence capitalization
        first = re.match(r'^[^A-Za-z]*([A-Za-z]+)', s)
        first_word = first.group(1) if first else ''
        for w in target_words:
            pat = rf'\\b{w.capitalize()}\\b'
            if re.search(pat, s):
                if first_word.lower() == w and first_word == w.capitalize():
                    # capitalization at sentence start is fine
                    continue
                issues.append(f'Mid-sentence capitalization: "{w.capitalize()}"')
    return issues


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


def _annotated_filename_for(src_path):
    src = Path(src_path)
    stem = src.stem
    if stem.lower().endswith('.annotated'):
        stem = stem[:-10]
    return f"{stem}.annotated{src.suffix}"


def test_bedrock_connection(model, bedrock_region="us-east-1", bedrock_profile="sci_bedrock"):
    result = {
        'ok': False,
        'provider': 'bedrock',
        'region': bedrock_region,
        'profile': bedrock_profile,
        'model': model,
    }

    if not model:
        result['error'] = 'Model is required for Bedrock connectivity test.'
        return result

    try:
        sts_args = ['sts', 'get-caller-identity']
        if bedrock_profile:
            sts_args.extend(['--profile', bedrock_profile])
        sts_raw = _run_aws_cli(sts_args)
        try:
            sts_info = json.loads(sts_raw) if sts_raw else {}
        except Exception:
            sts_info = {}
        result['aws_identity'] = {
            'account': sts_info.get('Account'),
            'arn': sts_info.get('Arn'),
            'user_id': sts_info.get('UserId'),
        }
    except Exception as e:
        result['error'] = f'AWS identity check failed: {e}'
        return result

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": "Reply with exactly: BEDROCK_OK"}
        ],
        "max_tokens": 32,
        "temperature": 0,
    }
    body_path = None
    out_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            tf.write(json.dumps(body))
            tf.flush()
            body_path = tf.name
        args = [
            "bedrock-runtime", "invoke-model",
            "--region", bedrock_region,
            "--model-id", model,
            "--content-type", "application/json",
            "--accept", "application/json",
            "--body", f"file://{body_path}",
        ]
        out_file = tempfile.NamedTemporaryFile(delete=False).name
        args.append(out_file)
        if bedrock_profile:
            args.extend(["--profile", bedrock_profile])
        _run_aws_cli(args)

        if out_file and os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as rf:
                raw_body = rf.read().strip()
        else:
            raw_body = ""

        content_json = {}
        if raw_body:
            try:
                content_json = json.loads(raw_body)
            except Exception:
                try:
                    decoded = base64.b64decode(raw_body).decode("utf-8")
                    content_json = json.loads(decoded)
                except Exception:
                    content_json = {}

        text = ""
        if isinstance(content_json, dict) and "content" in content_json:
            pieces = content_json.get("content") or []
            if isinstance(pieces, list) and pieces:
                text = pieces[0].get("text", "")

        result['response_preview'] = text[:200] if text else ''
        result['ok'] = True
        return result
    except Exception as e:
        result['error'] = f'Bedrock invoke-model failed: {e}'
        return result
    finally:
        if body_path and os.path.exists(body_path):
            try:
                os.unlink(body_path)
            except Exception:
                pass
        if out_file and os.path.exists(out_file):
            try:
                os.unlink(out_file)
            except Exception:
                pass


def scan(docx_path, llm=False, provider="ollama", ollama_url=None, model=None, llm_pull=False, bedrock_region="us-east-1", bedrock_profile="sci_bedrock", ignore_config_path=None, feedback_config_path=None):
    blocks = parse_docx(docx_path)
    paragraphs = [b for b in blocks if b['type'] == 'p']
    table_texts = []
    for b in blocks:
        if b['type'] != 'tbl':
            continue
        for row in b['rows']:
            for cell in row:
                if cell:
                    table_texts.append({'text': cell})
    sections = build_sections(paragraphs)
    findings = find_findings(blocks)

    diagnostics = []
    if not ignore_config_path:
        ignore_config_path = os.path.join(os.path.dirname(__file__), 'draftenheimer_ignore.json')
    ignore_cfg = _load_ignore_config(ignore_config_path)
    ignore_deprecated = ignore_cfg.get('ignore_deprecated_phrases', [])
    ignore_rewrites = ignore_cfg.get('ignore_rewrite_from_phrases', [])
    ignore_codes = ignore_cfg.get('ignore_diagnostic_codes', [])
    ignore_message_contains = ignore_cfg.get('ignore_message_contains', [])

    if not feedback_config_path:
        feedback_config_path = os.path.join(os.path.dirname(__file__), 'draftenheimer_feedback.json')
    feedback_cfg = _load_feedback_config(feedback_config_path)
    accepted_rewrites = feedback_cfg.get('accepted_rewrites', [])
    rejected_rewrites = feedback_cfg.get('rejected_rewrites', [])
    accepted_diag_rules = feedback_cfg.get('accepted_diagnostics', [])
    rejected_diag_rules = feedback_cfg.get('rejected_diagnostics', [])
    rejected_rewrite_keys = {
        _rewrite_key(x.get('from', ''), x.get('to', ''))
        for x in rejected_rewrites
    }

    # Repetition
    repeated = detect_repetition(paragraphs + table_texts)
    for sentence, count in repeated:
        diagnostics.append({
            'severity': 'warning',
            'code': 'DOC-STYLE-REDUND-001',
            'message': f'Repeated phrase appears {count} times: "{sentence[:120]}"',
            'location': {'section': 'Document'},
        })

    # Sensitive data
    full_text = '\n'.join(p['text'] for p in paragraphs + table_texts)
    for name, snippet in detect_sensitive(full_text):
        diagnostics.append({
            'severity': 'warning',
            'code': 'DOC-SENS-001',
            'message': f'Possible sensitive data detected: {name}',
            'location': {'section': 'Document'},
            'evidence': snippet,
        })

    # Hygiene (basic)
    for p in paragraphs:
        issues = detect_hygiene(p['text'])
        for i in issues:
            diagnostics.append({
                'severity': 'suggestion',
                'code': 'DOC-STYLE-HYGIENE-001',
                'message': i,
                'location': {'section': 'Document', 'text': p['text'][:120]},
            })

    # Profile-based checks (optional)
    profile_path = os.path.join(os.path.dirname(__file__), 'draftenheimer_profile.json')
    profile = None
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r', encoding='utf-8') as pf:
                profile = json.load(pf)
        except Exception:
            profile = None
        if profile:
            full_text_lower = full_text.lower()
            # Deprecated phrase n-grams
            for item in profile.get('deprecated_ngrams', []):
                phrase = item.get('phrase')
                if not phrase:
                    continue
                if _matches_any_phrase(phrase, ignore_deprecated):
                    continue
                count = full_text_lower.count(phrase)
                if count > 4:
                    diagnostics.append({
                        'severity': 'suggestion',
                        'code': 'DOC-STYLE-DEPRECATED-001',
                        'message': f'Deprecated phrase found: \"{phrase}\" (count {count})',
                        'location': {'section': 'Document'},
                    })
            # AI-learned deprecated phrase checks
            for item in profile.get('ai_deprecated_phrases', []):
                phrase = item.get('phrase')
                if not phrase:
                    continue
                if _matches_any_phrase(phrase, ignore_deprecated):
                    continue
                count = full_text_lower.count(str(phrase).lower())
                if count > 4:
                    diagnostics.append({
                        'severity': 'suggestion',
                        'code': 'DOC-STYLE-DEPRECATED-AI-001',
                        'message': f'AI-learned deprecated phrase found: \"{phrase}\" (count {count})',
                        'location': {'section': 'Document'},
                    })
            # Case normalization
            case_map = profile.get('case_normalization', {})
            for lower, preferred in case_map.items():
                if lower in full_text_lower and preferred not in full_text:
                    diagnostics.append({
                        'severity': 'suggestion',
                        'code': 'DOC-STYLE-CASE-001',
                        'message': f'Preferred casing: \"{preferred}\"',
                        'location': {'section': 'Document'},
                    })
            # Preferred sentence rewrites learned from prior report corrections
            rewrite_rules = _merge_rewrite_rules(profile.get('preferred_rewrites', []), accepted_rewrites)
            normalized_full = normalize_for_substring(full_text)
            for rule in rewrite_rules:
                old_text = rule.get('from', '')
                new_text = rule.get('to', '')
                if not old_text or not new_text:
                    continue
                if _matches_any_phrase(old_text, ignore_rewrites):
                    continue
                if _rewrite_key(old_text, new_text) in rejected_rewrite_keys:
                    continue
                old_norm = normalize_for_substring(old_text)
                new_norm = normalize_for_substring(new_text)
                if old_norm and old_norm in normalized_full and new_norm not in normalized_full:
                    diagnostics.append({
                        'severity': 'suggestion',
                        'code': 'DOC-STYLE-REWRITE-001',
                        'message': f'Preferred rewrite: \"{old_text}\" -> \"{new_text}\"',
                        'location': {'section': 'Document'},
                    })

    # Optional LLM QA (Narrative/professionalism)
    if llm and model:
        try:
            narrative = section_text_by_title(sections, 'narrative')
            if not narrative:
                chunks = []
                for f in findings:
                    fields = detect_fields_from_paras(f['content_paragraphs'])
                    if fields.get('narrative'):
                        chunks.append(' '.join(fields['narrative']))
                narrative = '\n'.join(chunks)
            if not narrative:
                # Fallback to whole document excerpt when narrative sections are not clearly detected.
                narrative = full_text

            if narrative:
                narrative = narrative[:8000]
                profile_guidance = []
                if profile:
                    deprecated = profile.get('deprecated_ngrams', [])[:12]
                    ai_deprecated = profile.get('ai_deprecated_phrases', [])[:12]
                    ai_style_rules = profile.get('ai_style_rules', [])[:10]
                    rewrites = []
                    for r in _merge_rewrite_rules(profile.get('preferred_rewrites', []), accepted_rewrites):
                        old_t = r.get('from', '')
                        if _matches_any_phrase(old_t, ignore_rewrites):
                            continue
                        if _rewrite_key(r.get('from', ''), r.get('to', '')) in rejected_rewrite_keys:
                            continue
                        rewrites.append(r)
                        if len(rewrites) >= 12:
                            break
                    if deprecated:
                        phrases = ', '.join(
                            x.get('phrase', '')
                            for x in deprecated
                            if x.get('phrase') and not _matches_any_phrase(x.get('phrase', ''), ignore_deprecated)
                        )
                        if phrases:
                            profile_guidance.append(f"Avoid overused/deprecated phrasing like: {phrases}.")
                    if ai_deprecated:
                        phrases = ', '.join(
                            x.get('phrase', '')
                            for x in ai_deprecated
                            if x.get('phrase') and not _matches_any_phrase(x.get('phrase', ''), ignore_deprecated)
                        )
                        if phrases:
                            profile_guidance.append(f"AI-learned deprecated phrasing to avoid: {phrases}.")
                    if rewrites:
                        lines = []
                        for r in rewrites:
                            old_t = r.get('from')
                            new_t = r.get('to')
                            if old_t and new_t:
                                lines.append(f"\"{old_t}\" -> \"{new_t}\"")
                        if lines:
                            profile_guidance.append("Preferred rewrite patterns:\n- " + "\n- ".join(lines[:8]))
                    if ai_style_rules:
                        lines = [x.get('rule', '') for x in ai_style_rules if x.get('rule')]
                        if lines:
                            profile_guidance.append("AI-learned style rules:\n- " + "\n- ".join(lines))
                guidance_text = "\n".join(profile_guidance) if profile_guidance else ""
                prompt = (
                    "You are a QA reviewer for penetration testing reports. "
                    "Review the narrative text for clarity, flow, professionalism, and repetition. "
                    "Use correction patterns learned from prior versioned report revisions when provided. "
                    "Do NOT challenge technical conclusions. "
                    "Return JSON array of issues with fields: severity (error|warning|suggestion), "
                    "message (actionable), and location_hint (short). "
                    "For any rewrite/substitution suggestion (for example 'use X instead of Y'), only suggest Y if it appears verbatim in the provided text. "
                    "If no issues, return empty array. "
                    "Return JSON only."
                )
                llm_user_content = (
                    f"Learned guidance:\n{guidance_text}\n\n"
                    f"Text to review:\n{narrative}"
                )
                if provider == "ollama":
                    from ollama_client import list_models, pull_model, chat
                    if not ollama_url:
                        ollama_url = "http://localhost:11434"
                    models = list_models(ollama_url)
                    if model not in models and llm_pull:
                        pull_model(ollama_url, model, verbose=False)
                        models = list_models(ollama_url)
                    if model not in models and not llm_pull:
                        diagnostics.append({
                            'severity': 'warning',
                            'code': 'LLM-SETUP-001',
                            'message': f'Ollama model \"{model}\" not found. Run qa_models.py pull {model}.',
                            'location': {'section': 'Document'},
                        })
                    elif model in models:
                        content = chat(ollama_url, model, [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": llm_user_content},
                        ], options={"temperature": 0.2})
                    else:
                        content = "[]"
                elif provider == "bedrock":
                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [
                            {"role": "user", "content": f"{prompt}\n\n{llm_user_content}"}
                        ],
                        "max_tokens": 800,
                        "temperature": 0.2,
                    }
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                        tf.write(json.dumps(body))
                        tf.flush()
                        body_path = tf.name
                    args = [
                        "bedrock-runtime", "invoke-model",
                        "--region", bedrock_region,
                        "--model-id", model,
                        "--content-type", "application/json",
                        "--accept", "application/json",
                        "--body", f"file://{body_path}",
                    ]
                    out_file = tempfile.NamedTemporaryFile(delete=False).name
                    args.append(out_file)
                    if bedrock_profile:
                        args.extend(["--profile", bedrock_profile])
                    out = _run_aws_cli(args)
                    try:
                        if os.path.exists(out_file):
                            with open(out_file, "r", encoding="utf-8") as rf:
                                raw_body = rf.read().strip()
                        else:
                            raw_body = out
                        try:
                            content_json = json.loads(raw_body)
                        except Exception:
                            try:
                                decoded = base64.b64decode(raw_body).decode("utf-8")
                                content_json = json.loads(decoded)
                            except Exception:
                                content_json = {}
                        # Anthropic response content
                        if isinstance(content_json, dict) and "content" in content_json:
                            pieces = content_json["content"]
                            if isinstance(pieces, list) and pieces:
                                content = pieces[0].get("text", "")
                            else:
                                content = ""
                        else:
                            content = ""
                    except Exception:
                        content = ""
                else:
                    content = "[]"

                issues = _extract_json_array_from_text(content)
                if isinstance(issues, list):
                    for it in issues:
                        if not isinstance(it, dict):
                            continue
                        base_severity = it.get('severity', 'suggestion')
                        base_message = it.get('message', 'Narrative issue detected.')
                        if not _llm_issue_supported_by_text(base_message, narrative):
                            continue
                        location_hint = it.get('location_hint', 'Narrative')
                        counted_message, max_count = _augment_llm_message_with_counts(base_message, narrative)
                        if _message_is_phrase_repetition_warning(base_message) and max_count <= 3:
                            # Only flag phrase-style warnings when the phrase is truly repeated.
                            continue
                        final_message = _augment_vague_llm_message(counted_message, narrative, location_hint)
                        final_severity = _elevate_severity_for_repeat_count(base_severity, max_count)
                        diagnostics.append({
                            'severity': final_severity,
                            'code': 'LLM-NARR-001',
                            'message': final_message,
                            'location': {'section': location_hint},
                            'source': 'ai_review',
                        })
                else:
                    diagnostics.append({
                        'severity': 'warning',
                        'code': 'LLM-PARSE-001',
                        'message': 'LLM returned non-JSON output for narrative review.',
                        'location': {'section': 'Narrative'},
                    })
        except Exception as e:
            diagnostics.append({
                'severity': 'warning',
                'code': 'LLM-ERR-001',
                'message': f'LLM narrative QA failed: {e}',
                'location': {'section': 'Document'},
            })

    # Findings checks
    for f in findings:
        f_text = '\n'.join(f['content_paragraphs'])
        fields = detect_fields_from_paras(f['content_paragraphs'])
        threat = detect_threat_level(f.get('threat_level') or (f['title'] + ' ' + f_text))
        if not fields.get('evidence') and detect_implicit_evidence(f['content_paragraphs']):
            fields['evidence'].append('implicit')

        # Narrative quality
        if not fields.get('narrative'):
            diagnostics.append({
                'severity': 'warning',
                'code': 'FND-STRUCT-005',
                'message': 'Missing narrative section.',
                'location': {'finding': f['title']},
            })
        else:
            narrative_text = ' '.join(fields.get('narrative', []))
            if len(narrative_text) < 200:
                diagnostics.append({
                    'severity': 'suggestion',
                    'code': 'FND-QUAL-003',
                    'message': 'Narrative appears too short for a clear story of the issue.',
                    'location': {'finding': f['title']},
                })
            # repeated sentences within narrative
            reps = detect_repetition([{'text': narrative_text}])
            for sentence, count in reps:
                diagnostics.append({
                    'severity': 'suggestion',
                    'code': 'FND-QUAL-004',
                    'message': f'Narrative repeats a sentence {count} times: \"{sentence[:120]}\"',
                    'location': {'finding': f['title']},
                })

        # If Threat Level is Informational, allow missing sections
        if threat != 'Informational':
            for key, code in [
                ('business_impact', 'FND-STRUCT-001'),
                ('recommendations', 'FND-STRUCT-002'),
                ('evidence', 'FND-STRUCT-003'),
                ('reproduction', 'FND-STRUCT-004'),
            ]:
                if not fields.get(key):
                    diagnostics.append({
                        'severity': 'error' if key in ('business_impact', 'recommendations') else 'warning',
                        'code': code,
                        'message': f'Missing {key.replace("_", " ")} section.',
                        'location': {'finding': f['title']},
                    })

        # Weak evidence / reproduction
        for key, code in [
            ('evidence', 'FND-QUAL-001'),
            ('reproduction', 'FND-QUAL-002'),
        ]:
            if fields.get(key):
                content = f_text
                if len(content) < 200:
                    diagnostics.append({
                        'severity': 'warning',
                        'code': code,
                        'message': f'{key.replace("_", " ").title()} content appears too short.',
                        'location': {'finding': f['title']},
                    })

        # CVSS vs severity
        sev = threat or detect_severity(f['title'] + ' ' + f_text)
        cvss = detect_cvss(f.get('cvss_text') or f_text)
        if sev and cvss is not None and sev != 'Informational':
            expected = cvss_to_severity(cvss)
            if expected and expected != sev:
                diagnostics.append({
                    'severity': 'warning',
                    'code': 'FND-CONSIST-001',
                    'message': f'Severity "{sev}" does not match CVSS {cvss:.1f} (expected {expected}).',
                    'location': {'finding': f['title']},
                })

    filtered_diags = []
    for d in diagnostics:
        # Explicit accepted feedback has highest precedence.
        if _diag_matches_any_feedback_rule(d, accepted_diag_rules):
            filtered_diags.append(d)
            continue
        # Explicit rejected feedback suppresses recurring unwanted suggestions.
        if _diag_matches_any_feedback_rule(d, rejected_diag_rules):
            continue
        # Then apply static ignore config.
        if _diag_is_ignored(d, ignore_codes, ignore_message_contains):
            continue
        filtered_diags.append(d)
    diagnostics = filtered_diags

    return {
        'document': os.path.basename(docx_path),
        'findings_count': len(findings),
        'diagnostics': diagnostics,
    }


def main():
    ap = argparse.ArgumentParser(prog='draftenheimer', description='Pentest report QA scan')
    ap.add_argument('docx_path', nargs='?')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--llm', action='store_true', help='Enable Ollama-based narrative QA')
    ap.add_argument('--ollama-url', default='http://localhost:11434')
    ap.add_argument('--model', default=None)
    ap.add_argument('--llm-pull', action='store_true', help='Pull model if missing')
    ap.add_argument('--provider', choices=['ollama', 'bedrock'], default='ollama')
    ap.add_argument('--bedrock-region', default='us-east-1')
    ap.add_argument('--bedrock-profile', default='sci_bedrock')
    ap.add_argument('--annotate', action='store_true', help='Write comments into a copy of the report')
    ap.add_argument('--annotate-out', default=None, help='Output path for annotated DOCX')
    ap.add_argument(
        '--ignore-config',
        default=None,
        help='Path to JSON config for skipping boilerplate phrase rewrites/deprecations/codes/messages.',
    )
    ap.add_argument(
        '--feedback-config',
        default=None,
        help='Path to JSON file containing accepted/rejected rewrite feedback.',
    )
    ap.add_argument(
        '--import-feedback-docx',
        default=None,
        help='Import [ACCEPT]/[REJECT] decisions from a reviewed annotated DOCX into feedback config.',
    )
    ap.add_argument(
        '--feedback-dry-run',
        action='store_true',
        help='With --import-feedback-docx, parse and report only; do not write feedback file.',
    )
    ap.add_argument(
        '--feedback-author',
        default='Draftenheimer',
        help='With --import-feedback-docx, only use base comments from this author.',
    )
    ap.add_argument(
        '--rebuild-learned-profile',
        action='store_true',
        help='Rebuild draftenheimer_profile.json from reports/ before scanning.',
    )
    ap.add_argument(
        '--auto-learn',
        action='store_true',
        help='Auto-refresh learned profile from reports/ before scanning (non-fatal when no pairs). ',
    )
    ap.add_argument(
        '--auto-learn-ai',
        action='store_true',
        help='When used with --auto-learn, include AI pair-comparison learning (slower). ',
    )
    ap.add_argument(
        '--reports-dir',
        default=str(Path(__file__).resolve().parent / 'reports'),
        help='Directory containing versioned report files for learned profile rebuild.',
    )
    ap.add_argument(
        '--learn-pair-mode',
        choices=['consecutive', 'latest'],
        default='consecutive',
        help='Auto-learn pairing mode for versioned reports.',
    )
    ap.add_argument(
        '--learn-track-weight',
        type=int,
        default=2,
        help='Weight for Track Changes learning signals when rebuilding profile.',
    )
    ap.add_argument(
        '--learn-no-track-changes',
        action='store_true',
        help='Disable Track Changes as an auto-learn signal when rebuilding profile.',
    )
    ap.add_argument(
        '--test-bedrock',
        action='store_true',
        help='Test Bedrock access using AWS profile and model, then exit.',
    )
    args = ap.parse_args()

    if args.test_bedrock:
        test_result = test_bedrock_connection(
            model=args.model,
            bedrock_region=args.bedrock_region,
            bedrock_profile=args.bedrock_profile,
        )
        print(json.dumps(test_result, indent=2))
        if args.json_out:
            with open(args.json_out, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, indent=2)
        if not test_result.get('ok'):
            raise SystemExit(1)
        raise SystemExit(0)

    if args.import_feedback_docx:
        feedback_script = Path(__file__).resolve().parent / 'import_feedback_from_docx.py'
        feedback_path = args.feedback_config or str(Path(__file__).resolve().parent / 'draftenheimer_feedback.json')
        feedback_cmd = [
            sys.executable,
            str(feedback_script),
            args.import_feedback_docx,
            '--feedback',
            feedback_path,
            '--author',
            args.feedback_author,
        ]
        if args.feedback_dry_run:
            feedback_cmd.append('--dry-run')
        try:
            subprocess.run(feedback_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(json.dumps({
                'error': f'Failed to import feedback from annotated DOCX: {e}'
            }, indent=2))
            raise SystemExit(1)
        if not args.docx_path:
            raise SystemExit(0)

    if not args.docx_path:
        print(json.dumps({
            'error': 'docx_path is required unless --test-bedrock or --import-feedback-docx is used.'
        }, indent=2))
        raise SystemExit(1)

    rebuild_requested = args.rebuild_learned_profile or args.auto_learn
    if rebuild_requested:
        script_path = Path(__file__).resolve().parent / 'build_learned_profile.py'
        rebuild_cmd = [
            sys.executable,
            str(script_path),
            '--reports-dir',
            args.reports_dir,
            '--pair-mode',
            args.learn_pair_mode,
            '--track-weight',
            str(max(1, int(args.learn_track_weight))),
        ]

        if args.learn_no_track_changes:
            rebuild_cmd.append('--no-track-changes')

        if args.auto_learn:
            rebuild_cmd.append('--allow-empty')

        if args.auto_learn and args.auto_learn_ai and not args.model:
            print(json.dumps({"error": "--auto-learn-ai requires --model so model-specific AI learning can run."}, indent=2))
            raise SystemExit(1)

        if args.auto_learn and args.auto_learn_ai:
            rebuild_cmd.extend(['--ai-compare', '--ai-provider', args.provider])
            if args.model:
                rebuild_cmd.extend(['--ai-model', args.model])
            if args.provider == 'bedrock':
                rebuild_cmd.extend(['--bedrock-region', args.bedrock_region, '--bedrock-profile', args.bedrock_profile])
            else:
                rebuild_cmd.extend(['--ollama-url', args.ollama_url])

        try:
            subprocess.run(rebuild_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(json.dumps({
                'error': f'Failed to rebuild learned profile: {e}'
            }, indent=2))
            raise SystemExit(1)

    result = scan(
        args.docx_path,
        llm=args.llm,
        provider=args.provider,
        ollama_url=args.ollama_url,
        model=args.model,
        llm_pull=args.llm_pull,
        bedrock_region=args.bedrock_region,
        bedrock_profile=args.bedrock_profile,
        ignore_config_path=args.ignore_config,
        feedback_config_path=args.feedback_config,
    )
    print(json.dumps(result, indent=2))

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

    if args.annotate:
        from qa_annotate import annotate
        qa_json_path = args.json_out
        if not qa_json_path:
            base, ext = _os.path.splitext(args.docx_path)
            qa_json_path = f"{base}.qa.json"
            with open(qa_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        out_path = args.annotate_out
        if not out_path:
            src = Path(args.docx_path)
            out_path = str(src.parent / _annotated_filename_for(src))
        else:
            out_candidate = Path(out_path)
            if out_candidate.exists() and out_candidate.is_dir():
                src = Path(args.docx_path)
                out_path = str(out_candidate / _annotated_filename_for(src))
        annotate(args.docx_path, qa_json_path, out_path)
        print(out_path)


if __name__ == '__main__':
    main()
