#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from qa_scan import parse_docx


def sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def norm(s):
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def _ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def build_profile(old_docx, new_docx, min_drop=5, min_len=15, ngram_min=3, ngram_max=5):
    b1 = parse_docx(old_docx)
    b2 = parse_docx(new_docx)
    p1 = [b['text'] for b in b1 if b['type'] == 'p']
    p2 = [b['text'] for b in b2 if b['type'] == 'p']
    text1 = '\n'.join(p1)
    text2 = '\n'.join(p2)

    # Phrase counts (simple: sentences)
    s1 = Counter(norm(s) for s in sentences(text1))
    s2 = Counter(norm(s) for s in sentences(text2))

    deprecated = []
    for s, c1 in s1.items():
        c2 = s2.get(s, 0)
        if c1 - c2 >= min_drop and len(s) >= min_len:
            deprecated.append({'phrase': s, 'old': c1, 'new': c2})
    deprecated.sort(key=lambda x: (x['old'] - x['new']), reverse=True)

    # Case-only changes (paragraphs)
    set1 = set(p1)
    set2 = set(p2)
    only1 = [p for p in p1 if p not in set2]
    only2 = [p for p in p2 if p not in set1]
    map2 = {p.lower(): p for p in only2}
    case_map = {}
    for p in only1:
        pl = p.lower()
        if pl in map2:
            case_map[pl] = map2[pl]

    # N-gram phrase drops (to catch repeated patterns like "the objective was")
    words1 = re.findall(r"[A-Za-z']+", text1.lower())
    words2 = re.findall(r"[A-Za-z']+", text2.lower())
    ngrams1 = Counter()
    ngrams2 = Counter()
    for n in range(ngram_min, ngram_max + 1):
        ngrams1.update(_ngrams(words1, n))
        ngrams2.update(_ngrams(words2, n))
    deprecated_ngrams = []
    for g, c1 in ngrams1.items():
        c2 = ngrams2.get(g, 0)
        if c1 - c2 >= min_drop:
            deprecated_ngrams.append({'phrase': g, 'old': c1, 'new': c2})
    deprecated_ngrams.sort(key=lambda x: (x['old'] - x['new']), reverse=True)

    return {
        'deprecated_phrases': deprecated,
        'deprecated_ngrams': deprecated_ngrams,
        'case_normalization': case_map,
        'source_docs': {
            'old': old_docx,
            'new': new_docx,
        },
    }


def main():
    ap = argparse.ArgumentParser(description='Build change profile from two report versions')
    ap.add_argument('old_docx')
    ap.add_argument('new_docx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-drop', type=int, default=5)
    ap.add_argument('--min-len', type=int, default=15)
    ap.add_argument('--ngram-min', type=int, default=3)
    ap.add_argument('--ngram-max', type=int, default=5)
    args = ap.parse_args()

    profile = build_profile(args.old_docx, args.new_docx, args.min_drop, args.min_len, args.ngram_min, args.ngram_max)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)


if __name__ == '__main__':
    main()
