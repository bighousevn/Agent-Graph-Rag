"""Criminal judgments from anle.toaan.gov.vn (HuggingFace dataset
tmquan/anle-toaan-gov-vn, CC-BY-4.0), as case records in the same schema
as the ViCSR ones (build_cases.py), to cover crimes ViCSR lacks: ViCSR is
~95% Điều 249, while these 459 criminal judgments span 90 crimes.

    data/raw/anle/documents.parquet   (huggingface.co/datasets/tmquan/anle-toaan-gov-vn)

Labels: the BLHS crime articles the judgment cites in its decision part
(after "Vì các lẽ trên" / "QUYẾT ĐỊNH:"), from the dataset's own
citations_law spans. Citations before it also cover prior convictions and
charges the court rejected.

Facts: the "nội dung vụ án" section up to the court's reasoning, cut
before the first-instance judgment summary, appeals and indictment (they
name the crime and articles), normalised like ViCSR (lower case, no
punctuation) and masked with mask_leaks ("tội ××", "điều ××").

Most judgments are appellate (307/459) or cassation (111): one judgment is
one record, not split per defendant, as for ViCSR.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

from .build_cases import MIN_FACT_WORDS, PROSECUTION_RES, _norm_name, mask_leaks

# Headings: the PDF text sometimes spaces letters out ("N ỘI DUNG V Ụ ÁN").
def _spaced(word: str) -> str:
    return r"\s*".join(re.escape(ch) for ch in word if not ch.isspace())


FACTS_START_RE = re.compile(_spaced("nội dung vụ án"), re.IGNORECASE)
FACTS_END_RE = re.compile(
    "|".join(_spaced(w) for w in ("nhận định của tòa án", "nhận định của hội đồng", "xét thấy")), re.IGNORECASE
)
DECISION_RE = re.compile("|".join([_spaced("vì các lẽ trên"), _spaced("quyết định") + r"\s*:"]), re.IGNORECASE)

# Where the facts give way to the indictment, the first-instance judgment
# and appeals. Matched on the text with every space removed, because the
# PDF extraction splits words ("cáo tr ạng s ố", "hình s ự sơ thẩm").
CUT_MARKERS_NOSPACE = re.compile(
    "|".join([
        r"(?:tại|theo)?bảncáotrạng", r"cáotrạngsố", r"đãtruytố", r"truytố(?:\S{0,40}?)vềtội",
        r"(?:tại|theo)?bảnán(?:hìnhsự)?(?:hìnhsự)?sơthẩm", r"bảnánsố\d", r"đãxétxử", r"đưavụánraxétxử",
        r"tuyênbố(?:các)?bịcáo", r"cóđơnkhángcáo", r"khángcáo", r"khángnghị",
        r"đạidiệnviệnkiểmsát", r"kiểmsátviên", r"đềnghịhộiđồngxétxử",
        # sentencing text; OCR sometimes drops letters ("tuyên ố các ị cáo")
        r"tuyênb?ố(?:các)?b?ịcáo", r"căncứ(?:vào)?(?:điểm|khoản|điều)", r"ápdụng(?:điểm|khoản|điều)", r"xửphạt(?:bịcáo)?",
    ])
)


def decision_start(markdown: str) -> Optional[int]:
    matches = list(DECISION_RE.finditer(markdown or ""))
    return matches[-1].start() if matches else None


def decision_articles(doc: Dict, crime_articles: Iterable[str]) -> List[str]:
    """BLHS crime articles cited in the decision part, in citation order."""
    crime_articles = set(crime_articles)
    start = decision_start(doc.get("markdown") or "")
    if start is None:
        return []
    out: List[str] = []
    for c in doc.get("citations_law") or []:
        name = (c.get("law_name") or "").lower()
        art = str(c.get("article") or "")
        if "hình sự" not in name or "tố tụng" in name or "thi hành" in name or art not in crime_articles:
            continue
        if any(int(a) >= start for a, _ in (c.get("span") or [])) and art not in out:
            out.append(art)
    return out


def decision_clauses(doc: Dict, articles: Sequence[str]) -> List[str]:
    """"123.1.a" style labels from the decision's citations."""
    start = decision_start(doc.get("markdown") or "") or 0
    out: List[str] = []
    for c in doc.get("citations_law") or []:
        art = str(c.get("article") or "")
        if art not in articles or not any(int(a) >= start for a, _ in (c.get("span") or [])):
            continue
        label = ".".join(x for x in (art, str(c.get("clause") or ""), str(c.get("point") or "")) if x and x != "None")
        if label not in out:
            out.append(label)
    return out


def raw_facts(markdown: str) -> Optional[str]:
    m = FACTS_START_RE.search(markdown or "")
    if not m:
        return None
    end = FACTS_END_RE.search(markdown, m.end())
    return markdown[m.end() : end.start() if end else None]


def _nospace_index(text: str):
    """(text without spaces, position in text of each kept character)."""
    chars, pos = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            pos.append(i)
    return "".join(chars), pos


def cut_facts(text: str, min_words: int = MIN_FACT_WORDS) -> str:
    """Cut at the earliest indictment / first-instance / appeal marker after
    the first ``min_words`` words (an opening "theo bản án sơ thẩm và các
    tài liệu ..." is not a cut)."""
    squeezed, pos = _nospace_index(text)
    words_before = lambda i: len(text[:i].split())
    starts = [pos[m.start()] for m in CUT_MARKERS_NOSPACE.finditer(squeezed) if words_before(pos[m.start()]) >= min_words]
    starts += [m.start() for pattern in PROSECUTION_RES for m in pattern.finditer(text) if words_before(m.start()) >= min_words]
    return (text[: min(starts)] if starts else text).strip()


def mask_short_names(text: str, crime_names: Sequence[str]) -> str:
    """Mask "tội <short name>" that mask_leaks misses: courts write "tội cố
    ý gây thương tích" for "cố ý gây thương tích hoặc gây tổn hại cho sức
    khỏe của người khác". A name is split at " hoặc "; "tội" followed by
    the first 3 words of a part is masked through the longest common run
    of words with that part."""
    parts = {}
    for name in crime_names:
        for part in name.split(" hoặc "):
            w = part.split()
            if len(w) >= 3:
                parts.setdefault(" ".join(w[:3]), []).append(w)
    out, last = [], 0
    for m in re.finditer(r"\btội ", text):
        if m.start() < last:
            continue
        tail = text[m.end():].split(" ", 12)
        key = " ".join(tail[:3])
        if key not in parts:
            continue
        n = max(next((i for i, (a, b) in enumerate(zip(tail, w)) if a != b), min(len(tail), len(w))) for w in parts[key])
        end = m.end() + len(" ".join(tail[:n]))
        out.append(text[last : m.end()] + "××")
        last = end
    out.append(text[last:])
    return "".join(out)


def facts_of(doc: Dict, crime_names: Sequence[str]) -> Optional[str]:
    raw = raw_facts(doc.get("markdown") or "")
    if raw is None:
        return None
    facts = cut_facts(_norm_name(raw))
    if len(facts.split()) < MIN_FACT_WORDS:
        return None
    return mask_short_names(mask_leaks(facts, crime_names), crime_names)


def crime_names_from_law_json(laws: Sequence[Dict]) -> List[str]:
    names = set()
    for law in laws:
        for crime in law["items"][0].get("crime") or []:
            names.add(_norm_name(re.sub(r"^Tội\s+", "", crime)))
    return sorted(names, key=len, reverse=True)


def records_from_documents(docs: Sequence[Dict], laws: Sequence[Dict]) -> Dict[str, Dict]:
    """Case records keyed by "anle-<doc_name>", ViCSR schema."""
    crime_articles = {f"{l['id']}{l.get('suffix') or ''}" for l in laws if l["items"][0].get("crime")}
    title = {f"{l['id']}{l.get('suffix') or ''}": l["items"][0]["crime"][0] for l in laws if l["items"][0].get("crime")}
    names = crime_names_from_law_json(laws)
    out: Dict[str, Dict] = {}
    for doc in docs:
        if doc.get("category") != "Criminal":
            continue
        articles = decision_articles(doc, crime_articles)
        if not articles:
            continue
        facts = facts_of(doc, names)
        if facts is None:
            continue
        rid = f"anle-{doc['doc_name']}"
        out[rid] = {
            "id": rid,
            "nguon": f"anle.toaan.gov.vn#{doc['doc_name']} ({doc.get('official_document_id')})",
            "nam": int(float(doc["year"])) if doc.get("year") else None,
            "bi_cao": "ẩn danh",
            "dien_bien": facts,
            "dien_bien_nguon": "noi_dung",
            "toi_danh": [title[a] for a in articles],
            "dieu": [int(a) if a.isdigit() else a for a in articles],
            "dieu_khoan": decision_clauses(doc, articles),
            "hinh_phat": None,
            "cach_gan_nhan": "trich_dan_phan_quyet_dinh",
            "cap_xet_xu": doc.get("instance_level"),
        }
    return out
