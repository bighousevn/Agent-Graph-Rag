"""Tests for vn_legal_graph.law.parse_blhs.

Uses a small fixture (tests/fixtures/blhs_sample_paragraphs.json) that
mimics the real docx paragraph structure of the Bộ luật Hình sự, instead
of depending on the full ~5000-paragraph source document. This keeps the
test fast and independent of python-docx / the actual law file being
present in the repo.
"""
import json
import os

import pytest

from vn_legal_graph.law.parse_blhs import normalize_text, parse_paragraphs

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "blhs_sample_paragraphs.json"
)


@pytest.fixture()
def articles():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        raw_paragraphs = json.load(f)
    normalized = [normalize_text(p) for p in raw_paragraphs]
    return parse_paragraphs(normalized)


def by_id(articles, article_id):
    matches = [a for a in articles if a.id == article_id]
    assert len(matches) == 1, f"expected exactly one Điều {article_id}"
    return matches[0]


def test_normalize_fixes_eth_glyph():
    assert normalize_text("Ðiều 317. Tội vi phạm") == "Điều 317. Tội vi phạm"
    assert normalize_text("Ðã có") == "Đã có"


def test_article_count(articles):
    ids = sorted(a.id for a in articles)
    assert ids == [1, 2, 122, 168, 169, 170, 247, 426]


def test_part_and_chapter_assignment(articles):
    dieu_168 = by_id(articles, 168)
    assert dieu_168.phan == "Phần thứ hai"
    assert dieu_168.phan_title == "CÁC TỘI PHẠM"
    assert dieu_168.chuong == "XVI"
    assert dieu_168.chuong_title == "CÁC TỘI XÂM PHẠM SỞ HỮU"
    assert dieu_168.muc is None

    dieu_1 = by_id(articles, 1)
    assert dieu_1.phan == "Phần thứ nhất"
    assert dieu_1.chuong == "I"
    assert dieu_1.chuong_title == "ĐIỀU KHOẢN CƠ BẢN"


def test_title_extraction(articles):
    assert by_id(articles, 168).title == "Tội cướp tài sản"
    assert by_id(articles, 247).title == (
        "Tội trồng cây thuốc phiện, cây côca, cây cần sa hoặc các loại cây "
        "khác có chứa chất ma túy"
    )


def test_is_crime_article(articles):
    assert by_id(articles, 168).is_crime_article is True
    assert by_id(articles, 1).is_crime_article is False
    # Điều 122 sits inside Phần thứ hai but is a general provision, not a
    # "Tội ..." article — it must NOT be treated as a crime article.
    assert by_id(articles, 122).is_crime_article is False


def test_khoan_and_diem_parsing(articles):
    dieu_168 = by_id(articles, 168)
    assert [k.so for k in dieu_168.khoan] == [1, 2]
    khoan_2 = dieu_168.khoan[1]
    assert [d.ky_hieu for d in khoan_2.diem] == ["a", "b", "đ"]
    assert khoan_2.diem[2].text == (
        "Chiếm đoạt tài sản trị giá từ 50.000.000 đồng đến dưới 200.000.000 đồng;"
    )


def test_preamble_only_article_has_no_khoan(articles):
    # Điều 1 in the fixture has plain descriptive text with no "N." marker.
    dieu_1 = by_id(articles, 1)
    assert dieu_1.khoan == []
    assert "bảo vệ chủ quyền quốc gia" in dieu_1.preamble


def test_eth_glyph_article_is_still_parsed(articles):
    # Điều 170 is written as "Ðiều 170" (wrong glyph) in the fixture to
    # simulate the real document's quirk; it must still be found.
    dieu_170 = by_id(articles, 170)
    assert dieu_170.title == "Tội cưỡng đoạt tài sản"
    assert dieu_170.chuong == "XVI"


def test_final_part_parsed(articles):
    dieu_426 = by_id(articles, 426)
    assert dieu_426.phan == "Phần thứ ba"
    assert dieu_426.phan_title == "ĐIỀU KHOẢN THI HÀNH"


def test_entry_label_and_full_text(articles):
    dieu_168 = by_id(articles, 168)
    assert dieu_168.entry_label == "168"
    text = dieu_168.full_text()
    assert text.startswith("Điều 168. Tội cướp tài sản")
    assert "a) Có tổ chức;" in text


def test_normalize_fixes_oi_glyph():
    assert normalize_text("đƣợc Ƣu tiên") == "được Ưu tiên"
