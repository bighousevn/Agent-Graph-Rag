"""Tests for vn_legal_graph.law.guidance_congvan and code-aware linking."""
import json

from vn_legal_graph.law.guidance_congvan import law_refs, letter_metadata, split_items
from vn_legal_graph.law.link_guidance import attach_guidance_links

LETTER = [
    "Qua công tác tổng kết thực tiễn xét xử, ...",
    "I. HÌNH SỰ, TỐ TỤNG HÌNH SỰ",
    "1. Điều 63 của Bộ luật Hình sự năm 2015 quy định điều kiện giảm mức hình phạt...",
    "Trả lời: ...",
    "2. Nguyễn Văn A làm giả căn cước để lừa đảo; hỏi về Điều 174 và Điều 341 của Bộ luật Hình sự?",
    "Điều 341 quy định:",
    "1. Người nào làm giả con dấu ...",       # quoted law text, not a new item
    "2. Phạm tội thuộc một trong ...",
    "3. Bị hại rút yêu cầu theo khoản 3 Điều 155 Bộ luật Tố tụng hình sự thì ...",
    "II. DÂN SỰ, TỐ TỤNG DÂN SỰ",
    "1. Đương sự ủy quyền ... Điều 388 của Bộ luật Tố tụng dân sự",
]


def test_split_items_keeps_quoted_numbers_inside_item_and_drops_civil_part():
    items = split_items(LETTER)
    assert [(i["phan"], i["muc"]) for i in items] == [("I", "1"), ("I", "2"), ("I", "3")]
    assert "1. Người nào làm giả con dấu" in items[1]["text"]
    assert "Tố tụng dân sự" not in "".join(i["text"] for i in items)


def test_law_refs_codes_and_crime_names():
    titles = {"tội lừa đảo chiếm đoạt tài sản": 174}
    assert law_refs("theo khoản 3 Điều 155 Bộ luật Tố tụng hình sự", titles) == ["BLTTHS:155"]
    assert law_refs("Điều 256a của Bộ luật Hình sự", titles) == ["BLHS:256a"]
    assert law_refs("bị truy cứu về Tội lừa đảo chiếm đoạt tài sản", titles) == ["BLHS:174"]
    assert law_refs("Điều 388 của Bộ luật Tố tụng dân sự", titles) == []


def test_letter_metadata():
    meta = letter_metadata(["..."], ["Số: 163/TANDTC-PC", "Hà Nội, ngày 10 tháng 9 năm 2024"])
    assert meta == {"so_hieu": "163/TANDTC-PC", "ngay": "10/09/2024"}


def test_attach_is_code_aware():
    links = [{"explain": "x", "from": "CV 1", "laws": ["BLTTHS:155"]},
             {"explain": "y", "from": "CV 2", "laws": ["BLHS:155", 134]}]
    blhs_155 = attach_guidance_links({"id": 155, "suffix": ""}, links, "BLHS")
    bltths_155 = attach_guidance_links({"id": 155, "suffix": ""}, links, "BLTTHS")
    assert [r["id"] for r in blhs_155] == ["CV 2"]
    assert [r["id"] for r in bltths_155] == ["CV 1"]
    assert [r["id"] for r in attach_guidance_links({"id": 134, "suffix": ""}, links)] == ["CV 2"]


def test_crime_title_index_keeps_suffix(tmp_path):
    from vn_legal_graph.law.guidance_congvan import crime_title_index

    path = tmp_path / "law.json"
    path.write_text(json.dumps([
        {"id": 256, "suffix": "", "items": [{"crime": ["Tội chứa chấp việc sử dụng trái phép chất ma túy"]}]},
        {"id": 256, "suffix": "a", "items": [{"crime": ["Tội sử dụng trái phép chất ma túy"]}]},
    ], ensure_ascii=False), encoding="utf-8")
    titles = crime_title_index(str(path))
    assert titles["tội sử dụng trái phép chất ma túy"] == "256a"
    assert law_refs("Về Tội sử dụng trái phép chất ma túy theo Điều 256a Bộ luật Hình sự", titles) == ["BLHS:256a"]
