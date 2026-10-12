"""First-instance criminal judgments from congbobanan.toaan.gov.vn (the
courts' official publication site), to give every crime the paper's
quota: <=20 corpus + 5 test cases per crime (HANDOFF 0, 2026-10-11).
ViCSR is ~95% Điều 249 and anle.toaan.gov.vn has 1-20 cases for 55 crimes.

The site is ASP.NET WebForms: every search and page change is a postback
carrying the previous page's __VIEWSTATE, inside one session (the cookies
pin the server; a view state from another session fails its MAC check).
Search flow, checked on the live site 2026-10-11:

    GET  /                          (session cookies)
    GET  /0t15at1cvn1/Tra-cu-ban-an
    POST Drop_CASES_STYLES_SEARCH_top=50 (Hình sự)  -> fills the crime list
         ("Ra_Case_shows_search_top", 601 entries, BLHS 1999 and 2015)
    POST + Ra_Case_shows_search_top=<crime> + cmd_search_banner -> 20 results/page
    POST + cmdnext                  -> next page

The court-level filter has no options, so first-instance ("Sơ thẩm") is
filtered from the result list. A judgment's PDF is /3ta<id>t1cvn/.

The server sends no intermediate certificate; the GlobalSign one is in
certs/ and appended to certifi's bundle, so TLS is still verified.

Records: same schema as ViCSR / anle (build_cases.py), labels from the
conviction sentence of the decision part ("tuyên bố bị cáo ... phạm tội
<tên>"), facts from "nội dung vụ án" cut before the indictment, masked.
"""
from __future__ import annotations

import collections
import html as htmllib
import json
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from typing import Dict, List, Optional, Sequence

from .anle import mask_short_names, mask_subsequence_names
from .build_cases import extract_dieu_khoan, extract_facts
from .vicsr import (
    CRIME_NAME_ALIASES,
    _norm_name,
    extract_convicted_crimes,
    is_appellate,
    split_sections,
)

BASE = "https://congbobanan.toaan.gov.vn"
SEARCH_PATH = "/0t15at1cvn1/Tra-cu-ban-an"
FIELD = "ctl00$Content_home_Public$ctl00$"
CRIMINAL = "50"  # Drop_CASES_STYLES_SEARCH value of "Hình sự"
JUDGMENT = "0"  # Drop_STATUS_JUDGMENT_SEARCH value of "Bản án" (not "Quyết định")
CERT = os.path.join(os.path.dirname(__file__), "certs", "globalsign_rsa_ov_ssl_ca_2018.pem")

# "108.Tội phản bội Tổ quốc (Luật sửa đổi, bổ sung một số điều của BLHS năm 2015)";
# 256a is listed under "... BLHS năm 2025". BLHS 1999 entries are skipped.
CRIME_OPTION_RE = re.compile(r"^(\d+[a-z]?)\.\s*Tội .*BLHS năm 20(?:15|25)\)\s*$")
LISTING_ARTICLE_RE = re.compile(r"\(điều (\d+[a-z]?)\b", re.IGNORECASE)
# Fewer words: a scanned PDF without a text layer (no OCR here).
MIN_TEXT_WORDS = 200


# --------------------------------------------------------------------------
# HTML parsing (no network)
# --------------------------------------------------------------------------

def _text(fragment: str) -> str:
    s = htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment))
    return re.sub(r"\s+", " ", s).strip()


def parse_form(page: str) -> Dict[str, str]:
    """Every text/hidden input and select of the page with its current value,
    as the browser would post it (buttons, radios and checkboxes left out)."""
    fields: Dict[str, str] = {}
    for m in re.finditer(r"<input([^>]*)>", page):
        attrs = m.group(1)
        name = re.search(r'name="([^"]+)"', attrs)
        kind = re.search(r'type="([^"]+)"', attrs)
        if not name or (kind and kind.group(1) in ("submit", "checkbox", "radio", "image", "button")):
            continue
        value = re.search(r'value="([^"]*)"', attrs)
        fields[name.group(1)] = htmllib.unescape(value.group(1)) if value else ""
    for m in re.finditer(r'<select[^>]*name="([^"]+)"[^>]*>(.*?)</select>', page, re.S):
        options = m.group(2)
        chosen = re.search(r'<option[^>]*selected="selected"[^>]*value="([^"]*)"', options) or re.search(
            r'<option[^>]*value="([^"]*)"', options
        )
        fields[m.group(1)] = chosen.group(1) if chosen else ""
    return fields


def select_options(page: str, name: str) -> List[tuple]:
    m = re.search(r'<select[^>]*name="' + re.escape(name) + r'"[^>]*>(.*?)</select>', page, re.S)
    if not m:
        return []
    return [(v, htmllib.unescape(t).strip()) for v, t in re.findall(r'<option[^>]*value="([^"]*)"[^>]*>([^<]*)', m.group(1))]


def parse_crime_options(page: str) -> Dict[str, str]:
    """BLHS 2015/2025 article ("173", "256a") -> option value of the crime list."""
    out: Dict[str, str] = {}
    for value, label in select_options(page, FIELD + "Ra_Case_shows_search_top"):
        m = CRIME_OPTION_RE.match(label)
        if m and value:
            out.setdefault(m.group(1), value)
    return out


def parse_count(page: str) -> Optional[int]:
    m = re.search(r'id="[^"]*lbl_count_record_top"[^>]*>\s*([\d.,]*)\s*<', page)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    return int(digits) if digits else 0


def _labelled(item: str, label: str) -> str:
    m = re.search(r"<label>\s*" + re.escape(label) + r"\s*</label>\s*<span[^>]*>(.*?)</span>", item, re.S)
    return _text(m.group(1)) if m else ""


def parse_listing(page: str) -> List[Dict]:
    """The result list: one dict per judgment."""
    out = []
    for item in re.split(r'<div class="list-group-item">', page)[1:]:
        m = re.search(r'href="/2ta(\d+)t1cvn/chi-tiet-ban-an"', item)
        if not m:
            continue
        heading = re.search(r'<h4 class="list-group-item-heading">(.*?)</h4>', item, re.S)
        heading = _text(heading.group(1)) if heading else ""
        title = re.search(r'<p><span style="line-height:18px">(.*?)</span></p>', item, re.S)
        title = _text(title.group(1)) if title else ""
        date = re.search(r"ngày (\d{1,2})/(\d{1,2})/(\d{4})", heading)
        out.append({
            "id": m.group(1),
            "ban_an": heading,
            "nam": int(date.group(3)) if date else None,
            "tieu_de": title,
            "dieu_trang": [a.lower() for a in LISTING_ARTICLE_RE.findall(title)],
            "cap_xet_xu": _labelled(item, "Cấp xét xử:"),
            "loai_an": _labelled(item, "Loại án:"),
        })
    return out


# --------------------------------------------------------------------------
# Network client
# --------------------------------------------------------------------------

def ca_bundle(path: Optional[str] = None) -> str:
    """certifi's bundle plus the GlobalSign intermediate the server omits."""
    import certifi

    path = path or os.path.join(tempfile.gettempdir(), "congbobanan_ca_bundle.pem")
    with open(certifi.where(), encoding="ascii") as f, open(CERT, encoding="ascii") as g:
        content = f.read() + "\n" + g.read()
    with open(path, "w", encoding="ascii") as out:
        out.write(content)
    return path


class Client:
    """One browsing session. Waits ``delay`` seconds before each request and
    retries timeouts / 5xx with a growing pause."""

    def __init__(self, delay: float = 2.0, timeout: float = 60.0, retries: int = 4, log=print):
        import requests

        self.session = requests.Session()
        self.session.verify = ca_bundle()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (research crawler; UIT thesis, rate-limited)"
        self.delay, self.timeout, self.retries, self.log = delay, timeout, retries, log
        self.page = ""  # last search page, whose form the next postback carries
        self.crimes: Dict[str, str] = {}

    def _request(self, method: str, path: str, retries: Optional[int] = None, **kw):
        import requests

        retries = self.retries if retries is None else retries
        for attempt in range(retries + 1):
            time.sleep(self.delay)
            try:
                r = self.session.request(method, BASE + path, timeout=self.timeout, **kw)
                if 400 <= r.status_code < 500:
                    raise RuntimeError(f"{method} {path}: HTTP {r.status_code}")
                if r.status_code < 400:
                    return r
                err = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                err = type(e).__name__
            if attempt == retries:
                raise RuntimeError(f"{method} {path}: {err}")
            pause = 15 * (attempt + 1)
            self.log(f"  {method} {path}: {err}, thử lại sau {pause}s")
            time.sleep(pause)

    def _postback(self, target: str = "", **values) -> str:
        fields = parse_form(self.page)
        fields["__EVENTTARGET"], fields["__EVENTARGUMENT"] = target, ""
        for key, value in values.items():
            fields[FIELD + key] = value
        self.page = self._request("POST", SEARCH_PATH, data=fields).text
        return self.page

    def open(self) -> Dict[str, str]:
        """New session on the search page with the criminal crime list loaded."""
        import requests

        self.session.cookies = requests.cookies.RequestsCookieJar()
        self._request("GET", "/")
        self.page = self._request("GET", SEARCH_PATH).text
        self._postback(FIELD + "Drop_CASES_STYLES_SEARCH_top", Drop_CASES_STYLES_SEARCH_top=CRIMINAL)
        self.crimes = parse_crime_options(self.page)
        if not self.crimes:
            raise RuntimeError("không đọc được danh sách tội danh của trang tìm kiếm")
        return self.crimes

    def search(self, article: str) -> str:
        """First result page of criminal judgments for one BLHS article."""
        if not self.crimes:
            self.open()
        return self._postback(
            "",
            Drop_CASES_STYLES_SEARCH_top=CRIMINAL,
            Drop_STATUS_JUDGMENT_SEARCH_top=JUDGMENT,
            Ra_Case_shows_search_top=self.crimes[article],
            cmd_search_banner="Tìm kiếm",
        )

    def next_page(self) -> str:
        return self._postback("", cmdnext=">>")

    def pdf(self, judgment_id: str) -> bytes:
        """The judgment PDF: /3ta<id>t1cvn/, or else the /5ta<id>t1cvn/<file>.pdf
        link of its detail page (some ids answer 500 on the first)."""
        try:
            r = self._request("GET", f"/3ta{judgment_id}t1cvn/", retries=1)
        except RuntimeError:
            detail = self._request("GET", f"/2ta{judgment_id}t1cvn/chi-tiet-ban-an", retries=0).text
            link = pdf_link(detail, judgment_id)
            if link is None:
                raise
            r = self._request("GET", link, retries=0)
        if not r.content.startswith(b"%PDF"):
            raise RuntimeError(f"bản án {judgment_id}: không phải PDF ({r.headers.get('content-type')})")
        return r.content


def pdf_link(detail_page: str, judgment_id: str) -> Optional[str]:
    """URL-quoted path of the "Tải bản án" link on a detail page."""
    from urllib.parse import quote

    m = re.search(r'href="(/5ta' + re.escape(judgment_id) + r't1cvn/[^"]+?\.pdf)"', detail_page, re.IGNORECASE)
    return quote(htmllib.unescape(m.group(1))) if m else None


def pdf_to_text(data: bytes) -> str:
    """Text of a PDF with poppler's pdftotext (reading order, no layout)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", f.name, "-"], capture_output=True, check=True)
    return out.stdout.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Judgment text -> case record (no network)
# --------------------------------------------------------------------------

def normalize_judgment(text: str) -> str:
    """PDF text in ViCSR form: NFC, Eth/Oi look-alikes fixed, page numbers
    (lines with only a number) dropped, lower case, no punctuation, single
    spaces."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", " ", text, flags=re.M)
    text = text.replace("Ð", "Đ").replace("ð", "đ").replace("ƣ", "ư").replace("Ƣ", "Ư")
    return _norm_name(text)


def crime_index_from_law_json(laws: Sequence[Dict]) -> Dict[str, str]:
    """Normalised crime name (no "tội ") -> article ("173", "256a"), with
    CRIME_NAME_ALIASES, from law_to_crime_vn.json (BLHS 2025 names; the
    judgments are 2025-2026, after Luật 86/2025)."""
    index: Dict[str, str] = {}
    for law in laws:
        for crime in law["items"][0].get("crime") or []:
            index[_norm_name(re.sub(r"^Tội\s+", "", crime))] = f"{law['id']}{law.get('suffix') or ''}"
    for alias, article in CRIME_NAME_ALIASES.items():
        index.setdefault(_norm_name(alias), str(article))
    return index


CITED_RE = re.compile(r"\bđiều (\d+[a-z]?)\b")
# The code an article list belongs to is named after it ("điều 51 điều 38
# của bộ luật hình sự điều 106 điều 331 bộ luật tố tụng hình sự"); BLTTHS
# 331/333 (appeal rights) fall in the BLHS crime range, so they must be told apart.
CODE_RE = re.compile(r"bộ luật tố tụng hình sự|blttths|bộ luật hình sự|\bblhs\b|bộ luật dân sự|\bblds\b|luật thi hành|nghị quyết|thông tư|nghị định")


def cited_blhs_articles(decision: str) -> List[str]:
    """Articles the (normalised) decision cites as BLHS ones, in order."""
    codes = [(m.start(), m.group(0)) for m in CODE_RE.finditer(decision)]
    out: List[str] = []
    for m in CITED_RE.finditer(decision):
        code = next((c for pos, c in codes if pos > m.start()), None)
        if code in ("bộ luật hình sự", "blhs") and m.group(1) not in out:
            out.append(m.group(1))
    return out


def confirmed_cited_crimes(decision: str, name_index: Dict[str, str], site_articles: Sequence[str]) -> List[str]:
    """Fallback when the decision has no "phạm tội <full name>" sentence
    ("xử phạt bị cáo ... về tội cố ý gây thương tích", "phạm tội hủy hoại
    tài sản"): crime articles cited as BLHS in the decision, kept when the
    site lists the article or "tội" + the first two words of its name occur."""
    head_words: Dict[str, List[str]] = {}
    for name, article in name_index.items():
        head_words.setdefault(article, []).append(" ".join(name.split()[:2]))
    return [
        a for a in cited_blhs_articles(decision)
        if a in head_words and (a in site_articles or any(f"tội {h}" in decision for h in head_words[a]))
    ]


def _article(a: str):
    return int(a) if a.isdigit() else a


def record_from_text(
    judgment_id: str, text: str, meta: Dict, name_index: Dict[str, str], title: Dict[str, str]
) -> tuple:
    """(record, None) or (None, reason it was dropped)."""
    detail = normalize_judgment(text)
    if len(detail.split()) < MIN_TEXT_WORDS:
        return None, "PDF không có chữ (ảnh quét)"
    if is_appellate(detail):
        return None, "án phúc thẩm"
    sections = split_sections(detail)
    decision = sections["quyet_dinh"] or ""
    articles = [str(a) for a in extract_convicted_crimes(decision, name_index)]
    method = "ten_toi"
    if not articles:
        articles = confirmed_cited_crimes(decision, name_index, meta.get("dieu_trang") or [])
        method = "dieu_luat_xac_nhan"
    if not articles:
        return None, "không gắn được nhãn"
    crime_names = sorted(name_index, key=len, reverse=True)
    facts = extract_facts(sections, None, crime_names)
    if facts is None:
        return None, "diễn biến quá ngắn"
    rid = f"cbba-{judgment_id}"
    return {
        "id": rid,
        "nguon": f"congbobanan.toaan.gov.vn#{judgment_id} ({meta.get('ban_an', '')})",
        "nam": meta.get("nam"),
        "bi_cao": "ẩn danh",
        "dien_bien": mask_subsequence_names(mask_short_names(facts[0], crime_names), crime_names),
        "dien_bien_nguon": facts[1],
        "toi_danh": [title[a] for a in articles],
        "dieu": [_article(a) for a in articles],
        "dieu_khoan": extract_dieu_khoan(sections["quyet_dinh"] or "", articles),
        "hinh_phat": None,
        "cach_gan_nhan": method,
        "dieu_trang": meta.get("dieu_trang") or [],
        "cap_xet_xu": meta.get("cap_xet_xu"),
    }, None


def records_from_dir(directory: str, laws: Sequence[Dict]) -> tuple:
    """Records keyed by "cbba-<id>" from a crawl_congbobanan.py directory
    (listings.jsonl + txt/), and a Counter of drop reasons."""
    name_index = crime_index_from_law_json(laws)
    title = {f"{l['id']}{l.get('suffix') or ''}": l["items"][0]["crime"][0] for l in laws if l["items"][0].get("crime")}
    meta: Dict[str, Dict] = {}
    with open(os.path.join(directory, "listings.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta.setdefault(row["id"], row)
    out: Dict[str, Dict] = {}
    dropped = collections.Counter()
    txt_dir = os.path.join(directory, "txt")
    for name in sorted(os.listdir(txt_dir)):
        if not name.endswith(".txt"):
            continue
        judgment_id = name[: -len(".txt")]
        with open(os.path.join(txt_dir, name), encoding="utf-8") as f:
            record, reason = record_from_text(judgment_id, f.read(), meta.get(judgment_id, {}), name_index, title)
        if record is None:
            dropped[reason] += 1
        else:
            out[record["id"]] = record
    return out, dropped
