"""Turn Nghị quyết of the Hội đồng Thẩm phán TANDTC (and their consolidated
texts, VBHN-TANDTC) into guidance_links records, like guidance_congvan.py
does for official letters.

A resolution is split into units: each numbered khoản of an Điều ("1. ..."),
or the whole Điều when it has none, prefixed with the Điều heading. Điều
"Phạm vi điều chỉnh", "Đối tượng áp dụng", "Hiệu lực thi hành" and repealed
ones are dropped (forms appended after "Hiệu lực thi hành" go with it).

Links: every "Điều N" in a unit is given the code named next after it
("khoản 1 Điều 141, khoản 1 Điều 142 ... của Bộ luật Hình sự" -> BLHS 141,
142), skipping "Điều N của Nghị quyết này" and "Điều 4 Luật số 86/2025".
A unit that cites nothing gets the articles named in the resolution's title
("Hướng dẫn áp dụng Điều 65 của Bộ luật Hình sự về án treo").
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .parse_blhs import load_paragraphs_from_docx

DIEU_HEAD_RE = re.compile(r"^Điều\s+(\d+[a-zđ]?)\.\s*(.*)$")
KHOAN_RE = re.compile(r"^(\d+)\.\s+\S")
SKIP_TITLES = ("phạm vi điều chỉnh", "đối tượng áp dụng", "hiệu lực thi hành", "(được bãi bỏ)", "được bãi bỏ")
REF_RE = re.compile(r"[Đđ]iều\s+(\d+[a-zđ]?)\b")
# What an "Điều N" belongs to: the first of these after it.
OWNER_RE = re.compile(
    r"(Bộ luật Tố tụng hình sự|BLTTHS|Bộ luật Hình sự|BLHS|Nghị quyết|Luật\s+(?:số|sửa đổi|Tổ chức|Thi hành|Xử lý|Tư pháp|Phòng)|Nghị định|Thông tư)",
    re.IGNORECASE,
)
OWNER_CODE = {"bộ luật tố tụng hình sự": "BLTTHS", "bltths": "BLTTHS", "bộ luật hình sự": "BLHS", "blhs": "BLHS"}
# Resolutions whose units name no BLHS article (03/2025 cites only Điều 4 of
# Luật 86/2025): attach them to the article they are about.
DEFAULT_LAWS = {"03/2025/NQ-HĐTP": ["BLHS:40"]}
SO_HIEU_FILE_RE = [
    (re.compile(r"(\d+)_(\d{4})_NQ-HDTP"), "{0}/{1}/NQ-HĐTP"),
    (re.compile(r"(\d+)_VBHN-TANDTC_(\d{4})"), "{0}/VBHN-TANDTC"),
]


def article_refs(text: str, window: int = 400) -> List[str]:
    """"CODE:label" for each "Điều N" in text that belongs to a code. The
    resolution's own "Điều 2. Về các tình tiết ... của Bộ luật Hình sự"
    heading number is not a reference."""
    text = re.sub(r"^Điều\s+\d+[a-zđ]?\.\s*", "", text, flags=re.MULTILINE)
    refs: List[str] = []
    for m in REF_RE.finditer(text):
        owner = OWNER_RE.search(text, m.end(), m.end() + window)
        if not owner:
            continue
        code = OWNER_CODE.get(owner.group(1).lower())
        if not code:
            continue
        ref = f"{code}:{m.group(1).lower()}"
        if ref not in refs:
            refs.append(ref)
    return refs


def split_units(paragraphs: List[str]) -> Tuple[str, List[Dict]]:
    """(title, units). Title = paragraphs before the first Điều, joined."""
    paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
    first = next((i for i, p in enumerate(paragraphs) if DIEU_HEAD_RE.match(p)), len(paragraphs))
    title = " ".join(p for p in paragraphs[:first] if not p.startswith("Căn cứ"))
    units: List[Dict] = []
    dieu: Optional[Dict] = None
    for p in paragraphs[first:]:
        m = DIEU_HEAD_RE.match(p)
        if m:
            dieu = {"dieu": m.group(1), "heading": p, "skip": m.group(2).strip().lower().startswith(SKIP_TITLES) or "bãi bỏ" in m.group(2).lower(), "khoan": [], "body": []}
            units.append(dieu)
            continue
        k = KHOAN_RE.match(p)
        if k and int(k.group(1)) == len(dieu["khoan"]) + 1:
            dieu["khoan"].append({"so": k.group(1), "text": [p]})
        elif dieu["khoan"]:
            dieu["khoan"][-1]["text"].append(p)
        else:
            dieu["body"].append(p)
    out = []
    for d in units:
        if d["skip"]:
            continue
        if d["khoan"]:
            for k in d["khoan"]:
                body = "\n".join(([" ".join(d["body"])] if d["body"] else []) + k["text"])
                out.append({"dieu": d["dieu"], "khoan": k["so"], "text": f"{d['heading']}\n{body}"})
        else:
            out.append({"dieu": d["dieu"], "khoan": "", "text": "\n".join([d["heading"]] + d["body"])})
    return title, out


def so_hieu_of(path: str, paragraphs: List[str], cells: List[str]) -> str:
    m = re.search(r"Số:\s*(\d+/(?:\d{4}/NQ-HĐTP|VBHN-TANDTC))", "\n".join(cells + paragraphs[:15]))
    if m:
        return m.group(1)
    for rx, fmt in SO_HIEU_FILE_RE:
        f = rx.search(path)
        if f:
            return fmt.format(*f.groups())
    return ""


def links_from_resolution(path: str) -> List[Dict]:
    import docx

    from .parse_blhs import _docx_source

    document = docx.Document(_docx_source(path))
    cells = [c.text.strip() for t in document.tables for r in t.rows for c in r.cells if c.text.strip()]
    paragraphs = [p for p in load_paragraphs_from_docx(path) if p]
    so_hieu = so_hieu_of(path, paragraphs, cells)
    title, units = split_units(paragraphs)
    title_refs = article_refs(title) or DEFAULT_LAWS.get(so_hieu, [])
    links = []
    for u in units:
        refs = article_refs(u["text"]) or title_refs
        if not refs:
            continue
        where = f"Điều {u['dieu']}" + (f" khoản {u['khoan']}" if u["khoan"] else "")
        links.append({"explain": u["text"], "from": f"Nghị quyết {so_hieu} của Hội đồng Thẩm phán TANDTC, {where}"
                      if "NQ-HĐTP" in so_hieu else f"Văn bản hợp nhất {so_hieu} (Nghị quyết HĐTP), {where}", "laws": refs})
    return links
