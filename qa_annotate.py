#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rels': 'http://schemas.openxmlformats.org/package/2006/relationships',
    'ct': 'http://schemas.openxmlformats.org/package/2006/content-types',
}

ET.register_namespace('w', NS['w'])
ET.register_namespace('r', NS['r'])


def _text_from_p(p):
    texts = []
    for t in p.findall('.//w:t', NS):
        if t.text:
            texts.append(t.text)
    return ''.join(texts).strip()


def _iter_paragraphs(root):
    body = root.find('w:body', NS)
    for p in body.findall('w:p', NS):
        yield p


def _load_comments(path):
    if not os.path.exists(path):
        root = ET.Element(f"{{{NS['w']}}}comments")
        return ET.ElementTree(root), 0
    tree = ET.parse(path)
    root = tree.getroot()
    max_id = 0
    for c in root.findall('w:comment', NS):
        cid = c.get(f"{{{NS['w']}}}id")
        if cid is not None:
            try:
                max_id = max(max_id, int(cid))
            except ValueError:
                pass
    return tree, max_id


def _recompute_max_comment_id(root_comments):
    max_id = 0
    for c in root_comments.findall('w:comment', NS):
        cid = c.get(f"{{{NS['w']}}}id")
        if cid is None:
            continue
        try:
            max_id = max(max_id, int(cid))
        except ValueError:
            continue
    return max_id


def _strip_existing_author_comments(doc_root, comments_root, author):
    # Remove existing comments written by this tool author so reruns don't stack old anchors.
    remove_ids = set()
    for c in list(comments_root.findall('w:comment', NS)):
        if c.get(f"{{{NS['w']}}}author") == author:
            cid = c.get(f"{{{NS['w']}}}id")
            if cid is not None:
                remove_ids.add(cid)
            comments_root.remove(c)

    if not remove_ids:
        return

    body = doc_root.find('w:body', NS)
    if body is None:
        return

    comment_tags = {
        f"{{{NS['w']}}}commentRangeStart",
        f"{{{NS['w']}}}commentRangeEnd",
        f"{{{NS['w']}}}commentReference",
    }
    for parent in body.iter():
        for child in list(parent):
            if child.tag not in comment_tags:
                continue
            cid = child.get(f"{{{NS['w']}}}id")
            if cid in remove_ids:
                parent.remove(child)


def _ensure_comments_rel(rels_path):
    if not os.path.exists(rels_path):
        root = ET.Element(f"{{{NS['rels']}}}Relationships")
        tree = ET.ElementTree(root)
    else:
        tree = ET.parse(rels_path)
        root = tree.getroot()

    # Check if comments relationship exists
    for rel in root.findall('rels:Relationship', NS):
        if rel.get('Type') == 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments':
            return tree

    # Determine next rId
    max_id = 0
    for rel in root.findall('rels:Relationship', NS):
        rid = rel.get('Id')
        if rid and rid.startswith('rId'):
            try:
                max_id = max(max_id, int(rid[3:]))
            except ValueError:
                pass
    new_id = f"rId{max_id + 1}"
    new_rel = ET.SubElement(root, f"{{{NS['rels']}}}Relationship")
    new_rel.set('Id', new_id)
    new_rel.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments')
    new_rel.set('Target', 'comments.xml')
    return tree


def _ensure_content_types(ct_path):
    tree = ET.parse(ct_path)
    root = tree.getroot()
    # Check existing override
    for ov in root.findall('ct:Override', NS):
        if ov.get('PartName') == '/word/comments.xml':
            return tree
    ov = ET.SubElement(root, f"{{{NS['ct']}}}Override")
    ov.set('PartName', '/word/comments.xml')
    ov.set('ContentType', 'application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml')
    return tree


def _add_comment(root_comments, comment_id, author, text):
    c = ET.SubElement(root_comments, f"{{{NS['w']}}}comment")
    c.set(f"{{{NS['w']}}}id", str(comment_id))
    c.set(f"{{{NS['w']}}}author", author)
    c.set(f"{{{NS['w']}}}date", datetime.now(timezone.utc).isoformat())
    p = ET.SubElement(c, f"{{{NS['w']}}}p")
    r = ET.SubElement(p, f"{{{NS['w']}}}r")
    t = ET.SubElement(r, f"{{{NS['w']}}}t")
    t.text = text


def _attach_comment_to_paragraph(p, comment_id):
    # Insert range start at beginning
    crs = ET.Element(f"{{{NS['w']}}}commentRangeStart")
    crs.set(f"{{{NS['w']}}}id", str(comment_id))
    # Insert range end at end
    cre = ET.Element(f"{{{NS['w']}}}commentRangeEnd")
    cre.set(f"{{{NS['w']}}}id", str(comment_id))
    # comment reference run
    r = ET.Element(f"{{{NS['w']}}}r")
    cr = ET.SubElement(r, f"{{{NS['w']}}}commentReference")
    cr.set(f"{{{NS['w']}}}id", str(comment_id))

    # Place at start and end
    children = list(p)
    if children:
        p.insert(0, crs)
        p.append(cre)
        p.append(r)
    else:
        p.append(crs)
        p.append(cre)
        p.append(r)


def _find_paragraph_by_text(paragraphs, text):
    target = text.strip()
    if not target:
        return None
    for p in paragraphs:
        ptxt = _text_from_p(p)
        if ptxt == target:
            return p
    target_lower = target.lower()
    for p in paragraphs:
        ptxt = _text_from_p(p)
        if ptxt.lower() == target_lower:
            return p
    # fallback: contains
    for p in paragraphs:
        if target in _text_from_p(p):
            return p
    for p in paragraphs:
        if target_lower in _text_from_p(p).lower():
            return p
    return None


def _find_paragraph_by_snippet(paragraphs, snippet):
    if not snippet:
        return None
    snip = snippet.strip()
    for p in paragraphs:
        if snip in _text_from_p(p):
            return p
    snip_lower = snip.lower()
    for p in paragraphs:
        if snip_lower in _text_from_p(p).lower():
            return p
    return None


def _extract_quoted_strings(text):
    if not text:
        return []
    quoted = re.findall(r'"([^"]+)"', text)
    # Prefer longer matches first for better anchoring specificity.
    quoted = sorted((q.strip() for q in quoted if q.strip()), key=len, reverse=True)
    return quoted


def _find_target_paragraph(paragraphs, diagnostic):
    loc = diagnostic.get('location', {}) or {}
    message = diagnostic.get('message', '') or ''

    # 1) Most specific location anchors first.
    if 'finding' in loc:
        p = _find_paragraph_by_text(paragraphs, loc['finding'])
        if p is not None:
            return p
    if 'text' in loc:
        p = _find_paragraph_by_snippet(paragraphs, loc['text'])
        if p is not None:
            return p
    if 'section' in loc and loc['section'] and loc['section'] != 'Document':
        p = _find_paragraph_by_text(paragraphs, loc['section'])
        if p is not None:
            return p

    # 2) Use quoted snippets from message text when present.
    for snippet in _extract_quoted_strings(message):
        p = _find_paragraph_by_snippet(paragraphs, snippet)
        if p is not None:
            return p

    # 3) Last-resort fallback: end of document (not top).
    return paragraphs[-1] if paragraphs else None


def _diagnostic_source_label(diagnostic):
    source = (diagnostic.get('source') or '').strip().lower()
    if source in ('ai', 'ai_review', 'llm'):
        return 'AI-REVIEW'
    code = (diagnostic.get('code') or '').upper()
    if code.startswith('LLM-'):
        return 'AI-REVIEW'
    return 'RULE-PATTERN'


def annotate(docx_path, qa_json_path, out_path, author='ReportsQA'):
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tmpdir = tempfile.mkdtemp(prefix='reportsqa_')
    try:
        with zipfile.ZipFile(docx_path, 'r') as zf:
            zf.extractall(tmpdir)

        doc_xml = os.path.join(tmpdir, 'word', 'document.xml')
        rels_xml = os.path.join(tmpdir, 'word', '_rels', 'document.xml.rels')
        comments_xml = os.path.join(tmpdir, 'word', 'comments.xml')
        ct_xml = os.path.join(tmpdir, '[Content_Types].xml')

        doc_tree = ET.parse(doc_xml)
        doc_root = doc_tree.getroot()
        paragraphs = list(_iter_paragraphs(doc_root))

        comments_tree, max_id = _load_comments(comments_xml)
        comments_root = comments_tree.getroot()
        _strip_existing_author_comments(doc_root, comments_root, author)
        max_id = _recompute_max_comment_id(comments_root)

        rels_tree = _ensure_comments_rel(rels_xml)
        ct_tree = _ensure_content_types(ct_xml)

        comment_id = max_id + 1
        for d in data.get('diagnostics', []):
            target_p = _find_target_paragraph(paragraphs, d)
            if target_p is None:
                continue

            source_label = _diagnostic_source_label(d)
            message = f"[{source_label}][{d.get('code')}][{d.get('severity')}] {d.get('message')}"
            _add_comment(comments_root, comment_id, author, message)
            _attach_comment_to_paragraph(target_p, comment_id)
            comment_id += 1

        doc_tree.write(doc_xml, encoding='utf-8', xml_declaration=True)
        comments_tree.write(comments_xml, encoding='utf-8', xml_declaration=True)
        rels_tree.write(rels_xml, encoding='utf-8', xml_declaration=True)
        ct_tree.write(ct_xml, encoding='utf-8', xml_declaration=True)

        # Repack
        with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmpdir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, tmpdir)
                    zf.write(full, rel)
    finally:
        shutil.rmtree(tmpdir)


def main():
    ap = argparse.ArgumentParser(description='Annotate report with QA comments')
    ap.add_argument('docx_path')
    ap.add_argument('qa_json_path')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.out is None:
        base, ext = os.path.splitext(args.docx_path)
        args.out = f"{base}.annotated{ext}"

    annotate(args.docx_path, args.qa_json_path, args.out)
    print(args.out)


if __name__ == '__main__':
    main()
