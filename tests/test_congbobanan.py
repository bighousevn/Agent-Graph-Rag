"""Tests for vn_legal_graph.cases.congbobanan and scripts/crawl_congbobanan.py
(no network: the HTML below copies the live site's markup, 2026-10-11)."""
import argparse
import importlib.util
import json
import os

from vn_legal_graph.cases.congbobanan import (
    FIELD,
    crime_index_from_law_json,
    normalize_judgment,
    parse_count,
    parse_crime_options,
    parse_form,
    parse_listing,
    cited_blhs_articles,
    confirmed_cited_crimes,
    pdf_link,
    record_from_text,
    records_from_dir,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def item(judgment_id, title, level="Sơ thẩm", kind="Hình sự"):
    return f"""
<div class="list-group-item"> <a class="echo_id_pub" target="_blank" href="/2ta{judgment_id}t1cvn/chi-tiet-ban-an">
<h4 class="list-group-item-heading"> <label>Bản án: </label><span>số 498 ngày 14/08/2026 của Tòa án nhân dân khu vực 1 - Đồng Nai,  tỉnh Đồng Nai <time>(30.09.2026)</time></span> </h4> </a>
<div class="row"> <div class="col-md-12"> <p><span style="line-height:18px">{title}</span></p> </div> </div>
<div class="row"> <div class="col-md-6"> <label>Cấp xét xử: </label><span>{level}</span> </div>
<div class="col-md-6"> <label>Áp dụng án lệ: </label><span> Không</span> </div> </div>
<div class="row"> <div class="col-md-6"> <label>Loại án:</label><span>{kind}</span> </div> </div>
<p class="Description_pub"> <label>Thông tin về vụ án:</label> <span>Bản án số 498</span> </p>
</div>"""


def page(items, count=60):
    return f"""<form><input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="abc&amp;d" />
<input name="{FIELD}txtKeyword_top" type="text" id="k" value="" />
<input type="submit" name="{FIELD}cmd_search_banner" value="Tìm kiếm" />
<select name="{FIELD}Ra_Case_shows_search_top" id="x">
<option value="">-----chọn-----</option>
<option value="475">175.Tội vi phạm các quy định về khai thác và bảo vệ rừng (Bộ luật hình sự năm 1999)</option>
<option selected="selected" value="1007">175.Tội lạm dụng tín nhiệm chiếm đoạt tài sản (Luật sửa đổi, bổ sung một số điều của BLHS năm 2015)</option>
<option value="1321">256a.Tội sử dụng trái phép chất ma túy (Luật sửa đổi, bổ sung một số điều của BLHS năm 2025)</option>
</select>
<h3>Kết quả tìm kiếm: <span id="ctl00_Content_home_Public_ctl00_lbl_count_record_top" class="PageNumber">{count}</span></h3>
<div id="List_group_pub" class="list-group">{"".join(items)}</div></form>"""


LAWS = [
    {"id": 173, "suffix": "", "items": [{"crime": ["Tội trộm cắp tài sản"]}]},
    {"id": 134, "suffix": "", "items": [{"crime": ["Tội cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác"]}]},
    {"id": 256, "suffix": "a", "items": [{"crime": ["Tội sử dụng trái phép chất ma túy"]}]},
    {"id": 51, "suffix": "", "items": [{"crime": []}]},
]
TITLE = {"173": "Tội trộm cắp tài sản", "134": LAWS[1]["items"][0]["crime"][0], "256a": "Tội sử dụng trái phép chất ma túy"}

FACTS = "Vào khoảng 14 giờ 00 phút, ngày 11-5-2026, Nguyễn Tấn H lén lút vào nhà bà M lấy trộm 01 điện thoại di động. " * 5
JUDGMENT = (
    "TÒA ÁN NHÂN DÂN KHU VỰC 11 - TÂY NINH\nBản án số: 204/2026/HS-ST\n"
    "NỘI DUNG VỤ ÁN:\n" + FACTS + "\n2\n"
    "Tại Cáo trạng số 12/CT-VKS, Viện kiểm sát truy tố bị cáo về tội “Trộm cắp tài sản” theo khoản 1 Điều 173.\n"
    "NHẬN ĐỊNH CỦA TÒA ÁN:\nHành vi của bị cáo phạm tội “Trộm cắp tài sản”.\n"
    "Vì các lẽ trên,\nQUYẾT ĐỊNH:\n1. Căn cứ khoản 1 Điều 173; điểm i, s khoản 1 Điều 51 của Bộ luật Hình sự.\n"
    "Tuyên bố bị cáo Nguyễn Tấn H phạm tội “Trộm cắp tài sản”.\n"
)


def test_parse_form_keeps_inputs_and_selected_options():
    f = parse_form(page([]))
    assert f["__VIEWSTATE"] == "abc&d"
    assert f[FIELD + "txtKeyword_top"] == ""
    assert f[FIELD + "Ra_Case_shows_search_top"] == "1007"
    assert FIELD + "cmd_search_banner" not in f  # buttons are posted only when clicked


def test_crime_options_skip_blhs_1999():
    assert parse_crime_options(page([])) == {"175": "1007", "256a": "1321"}


def test_listing_and_count():
    p = page([item("2266556", "Nguyễn Văn P - vô ý gây thiệt hại (điều 180 Luật sửa đổi, bổ sung một số điều của BLHS năm 2015)"),
              item("2280339", "Nguyễn Tấn B tội &quot;Trộm cắp tài sản&quot; (điều 173 Luật ...)", level="Phúc thẩm")])
    rows = parse_listing(p)
    assert parse_count(p) == 60
    assert [r["id"] for r in rows] == ["2266556", "2280339"]
    assert rows[0]["dieu_trang"] == ["180"] and rows[0]["nam"] == 2026
    assert rows[0]["cap_xet_xu"] == "Sơ thẩm" and rows[1]["cap_xet_xu"] == "Phúc thẩm" and rows[0]["loai_an"] == "Hình sự"
    assert '"Trộm cắp tài sản"' in rows[1]["tieu_de"]


def test_pdf_link_is_quoted():
    detail = '<a href="/5ta2280613t1cvn/170. QĐ Quàng Văn Dũng 11-10-2026.pdf">170.pdf</a>'
    assert pdf_link(detail, "2280613") == "/5ta2280613t1cvn/170.%20Q%C4%90%20Qu%C3%A0ng%20V%C4%83n%20D%C5%A9ng%2011-10-2026.pdf"
    assert pdf_link(detail, "999") is None


def test_normalize_drops_page_numbers_and_decomposed_marks():
    assert normalize_judgment("Lê Văn A\n 2 \nđie\u0302\u0300u") == "lê văn a đi\u1ec1u"  # NFC: e + mũ + huyền


def test_record_from_text_labels_facts_and_masks():
    idx = crime_index_from_law_json(LAWS)
    rec, why = record_from_text("2280339", JUDGMENT, {"ban_an": "số 204", "nam": 2026, "dieu_trang": ["173"]}, idx, TITLE)
    assert why is None
    assert rec["id"] == "cbba-2280339" and rec["dieu"] == [173] and rec["toi_danh"] == ["Tội trộm cắp tài sản"]
    assert rec["dieu_khoan"] == ["173.1"] and rec["cach_gan_nhan"] == "ten_toi"
    facts = rec["dien_bien"]
    assert "lấy trộm" in facts and "cáo trạng" not in facts and "điều 173" not in facts and " 2 tại" not in facts


def test_short_crime_name_uses_blhs_article_cited_in_decision():
    text = JUDGMENT.replace("“Trộm cắp tài sản”", "“Cố ý gây thương tích”").replace("Điều 173", "Điều 134")
    idx = crime_index_from_law_json(LAWS)
    rec, _ = record_from_text("1", text, {"dieu_trang": ["173"]}, idx, TITLE)  # site label wrong
    assert rec["dieu"] == [134] and rec["cach_gan_nhan"] == "dieu_luat_xac_nhan"


def test_cited_articles_keep_only_blhs_ones():
    decision = normalize_judgment(
        "QUYẾT ĐỊNH: Căn cứ khoản 1 Điều 173; Điều 51, Điều 38 của Bộ luật Hình sự; Điều 136, Điều 331, "
        "Điều 333 Bộ luật Tố tụng hình sự; Nghị quyết 326. Xử phạt bị cáo về tội trộm cắp tài sản."
    )
    assert cited_blhs_articles(decision) == ["173", "51", "38"]
    idx = crime_index_from_law_json(LAWS)
    assert confirmed_cited_crimes(decision, idx, []) == ["173"]
    # 134 cited but neither named nor listed by the site: not a label
    other = decision.replace("điều 51", "điều 134")
    assert confirmed_cited_crimes(other, idx, []) == ["173"]
    assert confirmed_cited_crimes(other, idx, ["134"]) == ["173", "134"]


def test_scanned_pdf_dropped():
    rec, why = record_from_text("1", "\n\n2\n", {}, crime_index_from_law_json(LAWS), TITLE)
    assert rec is None and why == "PDF không có chữ (ảnh quét)"


def test_appellate_judgment_dropped():
    rec, why = record_from_text("1", "BẢN ÁN HÌNH SỰ PHÚC THẨM\n" + JUDGMENT, {}, crime_index_from_law_json(LAWS), TITLE)
    assert rec is None and why == "án phúc thẩm"


def test_records_from_dir(tmp_path):
    (tmp_path / "txt").mkdir()
    (tmp_path / "txt" / "2280339.txt").write_text(JUDGMENT, encoding="utf-8")
    (tmp_path / "txt" / "5.txt").write_text("QUYẾT ĐỊNH: không có gì. " * 60, encoding="utf-8")
    (tmp_path / "listings.jsonl").write_text(
        json.dumps({"id": "2280339", "ban_an": "số 204", "nam": 2026, "dieu_trang": ["173"], "dieu_tim": "173"}) + "\n", encoding="utf-8")
    records, dropped = records_from_dir(str(tmp_path), LAWS)
    assert list(records) == ["cbba-2280339"] and records["cbba-2280339"]["nam"] == 2026
    assert dropped == {"không gắn được nhãn": 1}


# --- scripts/crawl_congbobanan.py with a fake client ----------------------

def _crawl_module():
    spec = importlib.util.spec_from_file_location("crawl_congbobanan", os.path.join(ROOT, "scripts", "crawl_congbobanan.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeClient:
    def __init__(self, pages, bad=(), scanned=()):
        self.pages, self.bad, self.scanned, self.i, self.downloads = pages, set(bad), set(scanned), 0, []

    def search(self, article):
        self.i = 0
        return self.pages[0]

    def next_page(self):
        self.i += 1
        return self.pages[self.i]

    def pdf(self, judgment_id):
        self.downloads.append(judgment_id)
        if judgment_id in self.bad:
            raise RuntimeError("HTTP 500")
        return b"%PDF-scan" if judgment_id in self.scanned else b"%PDF"


def test_crawl_pages_until_quota_of_judgments_with_text(tmp_path, monkeypatch):
    mod = _crawl_module()
    monkeypatch.setattr(mod, "pdf_to_text", lambda data: "chữ " * (300 if data == b"%PDF" else 3))
    p1 = page([item(str(i), "x (điều 173 ...)") for i in range(1, 4)] + [item("9", "x", level="Phúc thẩm")], count=200)
    p2 = page([item(str(i), "x (điều 173 ...)") for i in range(4, 8)], count=200)
    client = FakeClient([p1, p2, page([], 200)], bad={"2"}, scanned={"3"})
    args = argparse.Namespace(count_only=False, per_crime=3, max_pages=10)
    (tmp_path / "txt").mkdir()
    listings = tmp_path / "listings.jsonl"
    state = mod.crawl_crime(client, "173", args, str(tmp_path / "txt"), str(listings), set())
    assert state == {"count": 200, "so_tham": 3, "anh_quet": 1, "loi_tai": 1, "pages": 2, "done": True}
    assert client.downloads == ["1", "2", "3", "4", "5"]  # "9" is appellate; stops once 1, 4, 5 have text
    assert sorted(os.listdir(tmp_path / "txt")) == ["1.txt", "3.txt", "4.txt", "5.txt"]  # scanned kept, not re-fetched
    rows = [json.loads(l) for l in listings.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and all(r["dieu_tim"] == "173" for r in rows)


def test_crawl_counts_saved_texts_without_downloading(tmp_path):
    mod = _crawl_module()
    (tmp_path / "1.txt").write_text("chữ " * 300, encoding="utf-8")
    client = FakeClient([page([item("1", "x"), item("2", "x")], count=2)])
    args = argparse.Namespace(count_only=False, per_crime=1, max_pages=10)
    state = mod.crawl_crime(client, "173", args, str(tmp_path), str(tmp_path / "l.jsonl"), set())
    assert state["so_tham"] == 1 and client.downloads == []


def test_crawl_count_only_reads_one_page(tmp_path):
    mod = _crawl_module()
    client = FakeClient([page([item("1", "x")], count=7)])
    args = argparse.Namespace(count_only=True, per_crime=3, max_pages=10)
    state = mod.crawl_crime(client, "173", args, str(tmp_path), str(tmp_path / "l.jsonl"), set())
    assert state["count"] == 7 and state["pages"] == 1 and state["done"] and client.downloads == []
