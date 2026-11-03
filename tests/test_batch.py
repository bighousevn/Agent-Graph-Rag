"""Tests for vn_legal_graph.qa.batch and scripts/answer_batch.py (no LLM)."""
import json
import subprocess
import sys

from openpyxl import Workbook, load_workbook

from vn_legal_graph.qa.batch import clean_questions, excel_row, mask_phones, question_of, read_table, write_csv


def test_mask_phones():
    assert mask_phones("gọi 0912345678 hoặc 0912 345 678 hay +84912345678") == (
        "gọi [SĐT] hoặc [SĐT] hay [SĐT]", 3)
    assert mask_phones("số tiền 40000000 đồng, năm 2026")[1] == 0  # 8 digits, not a phone


def test_clean_questions_keeps_id_and_text_only():
    rows = [
        {"post_id": 1, "author": "A", "url": "u1", "text": "Bị lừa  0912345678\n chuyển tiền?"},
        {"post_id": 2, "author": "B", "url": "u2", "text": "Bị lừa 0912345678 chuyển tiền?"},  # same text after spacing
        {"post_id": 1, "author": "C", "url": "u3", "text": "khác"},  # repeated id
        {"post_id": 4, "author": "D", "url": "u4", "text": "  "},
        {"post_id": 5, "author": "E", "url": "u5", "text": "Trộm xe 30 triệu?"},
    ]
    out, report = clean_questions(rows, "post_id", "text")
    assert out == [{"id": "1", "cau_hoi": "Bị lừa [SĐT] chuyển tiền?"}, {"id": "5", "cau_hoi": "Trộm xe 30 triệu?"}]
    assert report == {"vao": 5, "trong": 1, "trung_id": 1, "trung_noi_dung": 1, "che_sdt": 1, "ra": 2}


def test_read_table_csv_and_xlsx(tmp_path):
    write_csv(str(tmp_path / "q.csv"), [{"id": "7", "cau_hoi": "Đánh bạc 40 triệu?"}])
    assert read_table(str(tmp_path / "q.csv")) == [{"id": "7", "cau_hoi": "Đánh bạc 40 triệu?"}]
    wb = Workbook()
    wb.active.append(["id", "cau_hoi"])
    wb.active.append([8, "Cố ý gây thương tích 11%?"])
    wb.save(tmp_path / "q.xlsx")
    assert read_table(str(tmp_path / "q.xlsx")) == [{"id": 8, "cau_hoi": "Cố ý gây thương tích 11%?"}]


def test_question_of_puts_the_title_first():
    assert question_of({"cau_hoi": "nội dung"}, "cau_hoi") == "nội dung"
    assert question_of({"t": "Tiêu đề", "cau_hoi": "nội dung"}, "cau_hoi", "t") == "Tiêu đề\nnội dung"


def _result(qid):
    return {"id": qid, "cau_hoi": f"câu {qid}", "dien_dat_lai": "",
            "tra_loi": {"cau_tra_loi": "Theo Điều 321.", "toi_danh": ["Tội đánh bạc"],
                        "dieu_luat": [{"luat": "Bộ luật Hình sự", "dieu": "321", "khoan": "1", "diem": ""}]},
            "judge": {"BLHS Điều 321": {"ap_dung": True}, "BLHS Điều 322": {"ap_dung": False}},
            "truy_xuat": {"ung_vien": ["BLHS Điều 321", "BLHS Điều 322"]}, "tra_loi_loi": False}


def test_excel_row_and_merge_writes_json_and_excel(tmp_path):
    row = excel_row(_result("1"))
    assert row[:6] == ["1", "câu 1", "Theo Điều 321.", "BLHS Điều 321 khoản 1", "Tội đánh bạc", "BLHS Điều 321"]
    assert row[-3:] == ["", "", ""]  # left for the person judging
    assert row[-4] == ""  # no citation check in an old result
    for k in (1, 2):
        (tmp_path / f"p{k}.json").write_text(json.dumps([_result(str(k))], ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "all.json"
    subprocess.run([sys.executable, "scripts/answer_batch.py", "--merge", str(tmp_path / "p1.json"),
                    str(tmp_path / "p2.json"), "--merge-out", str(out)], check=True, capture_output=True)
    assert [r["id"] for r in json.loads(out.read_text(encoding="utf-8"))] == ["1", "2"]
    ws = load_workbook(tmp_path / "all.xlsx").active
    assert ws["A1"].value == "Mã câu" and ws["A3"].value == "2" and ws.max_row == 3


def test_dry_run_reads_the_file_and_shards(tmp_path):
    write_csv(str(tmp_path / "q.csv"), [{"id": str(i), "cau_hoi": f"câu hỏi {i}"} for i in range(5)])
    res = subprocess.run([sys.executable, "scripts/answer_batch.py", "--input", str(tmp_path / "q.csv"),
                          "--shard", "2/2", "--dry-run"], check=True, capture_output=True, text=True)
    assert "2 câu" in res.stdout and "  1: câu hỏi 1" in res.stdout and "  3: câu hỏi 3" in res.stdout


def test_citation_check_text():
    from vn_legal_graph.qa.batch import citation_check_text

    rows = [{"cite": "BLHS Điều 134 khoản 6 điểm a", "trong_tai_lieu": True, "trich_dan_co_that": True,
             "vi_tri": "khoản 6", "khop_vi_tri": False},
            {"cite": "BLHS Điều 134 khoản 1 điểm b", "trong_tai_lieu": True, "trich_dan_co_that": True,
             "vi_tri": "khoản 1 điểm b", "khop_vi_tri": True},
            {"cite": "BLHS Điều 173", "trong_tai_lieu": False, "trich_dan_co_that": None, "vi_tri": None, "khop_vi_tri": None}]
    assert citation_check_text(rows).splitlines() == [
        "BLHS Điều 134 khoản 6 điểm a: LỆCH: đoạn trích nằm ở khoản 6",
        "BLHS Điều 134 khoản 1 điểm b: đúng vị trí (khoản 1 điểm b)",
        "BLHS Điều 173: điều không có trong tài liệu đưa vào"]
    assert citation_check_text([], ["Điều 174"]) == "Nêu trong câu trả lời nhưng không ghi căn cứ: Điều 174"
