"""Answer a file of questions that has no reference answers (no scoring):
read a CSV/XLSX, clean it, keep what the system answered, write it to Excel
for a person to judge later. Used by scripts/clean_questions.py and
scripts/answer_batch.py.
"""
from __future__ import annotations

import csv
import re
from typing import Dict, List, Optional, Tuple

# Vietnamese phone numbers: 0xxxxxxxxx (10 digits, spaces/dots allowed between groups) or +84...
PHONE_RE = re.compile(r"(?<![\d])(?:\+84|0)(?:[\s.-]?\d){9}(?![\d])")
ALIASES = {"blhs": "BLHS", "bộ luật hình sự": "BLHS", "bltths": "BLTTHS", "bộ luật tố tụng hình sự": "BLTTHS"}


def read_table(path: str) -> List[Dict]:
    """Rows of a .csv (UTF-8, BOM or not) or the first sheet of an .xlsx."""
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    import openpyxl

    rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    return [dict(zip(header, r)) for r in rows[1:] if any(v not in (None, "") for v in r)]


def mask_phones(text: str) -> Tuple[str, int]:
    return PHONE_RE.subn("[SĐT]", text)


def clean_questions(rows: List[Dict], id_col: str, text_col: str,
                    title_col: Optional[str] = None) -> Tuple[List[Dict], Dict[str, int]]:
    """Keep id + question (+ title) only, phone numbers masked; drop empty
    questions, repeated ids and repeated texts (first one kept)."""
    out: List[Dict] = []
    seen_ids, seen_texts = set(), set()
    report = {"vao": len(rows), "trong": 0, "trung_id": 0, "trung_noi_dung": 0, "che_sdt": 0}
    for r in rows:
        if id_col not in r or text_col not in r:
            raise KeyError(f"Thiếu cột {id_col!r} hoặc {text_col!r}; có các cột: {list(r)}")
        qid = str(r[id_col] or "").strip()
        text = " ".join(str(r[text_col] or "").split())
        if not qid or not text:
            report["trong"] += 1
            continue
        if qid in seen_ids:
            report["trung_id"] += 1
            continue
        if text in seen_texts:
            report["trung_noi_dung"] += 1
            continue
        seen_ids.add(qid)
        seen_texts.add(text)
        text, n = mask_phones(text)
        report["che_sdt"] += n
        row = {"id": qid, "cau_hoi": text}
        if title_col:
            row["tieu_de"] = " ".join(str(r.get(title_col) or "").split())
        out.append(row)
    report["ra"] = len(out)
    return out, report


def write_csv(path: str, rows: List[Dict]) -> None:
    cols = list(rows[0]) if rows else ["id", "cau_hoi"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def question_of(row: Dict, text_col: str, title_col: Optional[str] = None) -> str:
    """What the system reads: title (if any) + question, like question_text."""
    title = str(row.get(title_col) or "").strip() if title_col else ""
    return f"{title}\n{str(row.get(text_col) or '').strip()}".strip()


def _cite(c: Dict) -> str:
    code = ALIASES.get(str(c.get("luat") or "").strip().lower(), str(c.get("luat") or "").strip())
    s = f"{code} Điều {c.get('dieu', '')}"
    if c.get("khoan"):
        s += f" khoản {c['khoan']}"
    if c.get("diem"):
        s += f" điểm {c['diem']}"
    return s


EXCEL_COLUMNS = [
    ("Mã câu", 18), ("Câu hỏi", 70), ("Câu trả lời của hệ thống", 70), ("Căn cứ hệ thống trích", 36),
    ("Tội danh", 28), ("Điều thẩm định thoả mãn", 30), ("Ứng viên", 40), ("Câu hỏi diễn đạt lại", 50),
    ("Lỗi định dạng", 10), ("Kiểm tra khoản/điểm (tự động)", 40), ("Đánh giá (đúng / một phần / sai)", 16), ("Điều luật đúng (người đánh giá điền)", 30),
    ("Ghi chú", 30),
]


def excel_row(r: Dict) -> List:
    answer = r.get("tra_loi") or {}
    accepted = [k for k, v in (r.get("judge") or {}).items() if v.get("ap_dung")]
    return [
        r["id"], r["cau_hoi"], answer.get("cau_tra_loi", ""),
        "; ".join(_cite(c) for c in answer.get("dieu_luat", []) if isinstance(c, dict)),
        "; ".join(c if isinstance(c, str) else _cite(c) if isinstance(c, dict) else str(c)
                  for c in answer.get("toi_danh", [])), "; ".join(accepted),
        "; ".join((r.get("truy_xuat") or {}).get("ung_vien", [])), r.get("dien_dat_lai", ""),
        "có" if r.get("tra_loi_loi") else "", citation_check_text(r.get("kiem_tra_trich_dan"), r.get("nhac_ma_khong_trich")), "", "", "",
    ]


def citation_check_text(rows, not_cited=None) -> str:
    """One line per citation the answer gave: where its quote sits in the
    article and whether that is the cited khoản/điểm."""
    out = []
    for c in rows or []:
        if not c["trong_tai_lieu"]:
            note = "điều không có trong tài liệu đưa vào"
        elif c["trich_dan_co_that"] is None:
            note = "không có trích dẫn"
        elif not c["trich_dan_co_that"]:
            note = "trích dẫn không có trong điều luật"
        else:
            note = f"đúng vị trí ({c['vi_tri'] or 'phần đầu điều'})" if c["khop_vi_tri"] else \
                f"LỆCH: đoạn trích nằm ở {c['vi_tri'] or 'phần đầu điều'}"
        out.append(f"{c['cite']}: {note}")
    if not_cited:
        out.append(f"Nêu trong câu trả lời nhưng không ghi căn cứ: {', '.join(not_cited)}")
    return "\n".join(out)


def write_excel(path: str, results: List[Dict]) -> None:
    """One row per question; the last three columns are left for the person judging."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Câu trả lời"
    for j, (name, width) in enumerate(EXCEL_COLUMNS, 1):
        c = ws.cell(row=1, column=j, value=name)
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78" if j <= 10 else "C55A11")
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = width
    for i, r in enumerate(results, 2):
        for j, v in enumerate(excel_row(r), 1):
            c = ws.cell(row=i, column=j, value=v)
            c.font = Font(name="Arial")
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(EXCEL_COLUMNS))}{len(results) + 1}"
    wb.save(path)
