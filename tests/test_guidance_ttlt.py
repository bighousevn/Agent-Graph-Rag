"""Tests for vn_legal_graph.law.guidance_ttlt on a shortened copy of the
TTLT 17/2007 section structure."""
import pytest

from vn_legal_graph.law.guidance_ttlt import build_links, extract_point_37, extract_section

PARAS = [
    "II. VỀ CÁC TỘI PHẠM CỤ THỂ",
    "3. Tội tàng trữ, vận chuyển, mua bán trái phép hoặc chiếm đoạt chất ma túy (Điều 194)",
    "3.1. “Tàng trữ trái phép chất ma túy” là cất giữ ... mà không nhằm mục đích mua bán, vận chuyển hay sản xuất trái phép chất ma túy.",
    "3.2. “Vận chuyển trái phép chất ma túy” là hành vi chuyển dịch ...",
    "Người giữ hộ ... biết rõ mục đích mua bán ... với vai trò đồng phạm.",
    "3.3. “Mua bán trái phép chất ma túy” là một trong các hành vi sau đây:",
    "a) Bán trái phép chất ma túy cho người khác ...;",
    "b) Mua chất ma túy nhằm bán trái phép cho người khác;",
    "3.4. “Chiếm đoạt chất ma túy” là ...",
    "3.7. Khi truy cứu trách nhiệm hình sự ... cần phân biệt:",
    "c) Người nào biết người khác đi mua chất ma túy để sử dụng ...",
    "d) Người nào biết người khác mua chất ma túy để sử dụng ... dùng phương tiện để chở ...",
    "đ) (điểm đã bị TTLT 08/2015 bãi bỏ)",
    "4. Tội tàng trữ, vận chuyển, mua bán hoặc chiếm đoạt tiền chất ... (Điều 195).",
    "4.1. “Tàng trữ tiền chất ...”",
]


def test_section_with_following_paragraphs():
    s = extract_section(PARAS, "3.3")
    assert s.startswith("3.3. “Mua bán trái phép chất ma túy”")
    assert "b) Mua chất ma túy nhằm bán trái phép" in s
    assert "3.4." not in s
    assert "Người giữ hộ" in extract_section(PARAS, "3.2")


def test_stays_inside_part_ii_section_3():
    with pytest.raises(ValueError):
        extract_section(PARAS, "4.1")


def test_point_37_keeps_heading_and_excludes_repealed_point():
    c = extract_point_37(PARAS, "c")
    assert c.startswith("3.7.") and "mua chất ma túy để sử dụng" in c
    links = build_links(PARAS)
    assert not any("đã bị TTLT 08/2015 bãi bỏ" in link["explain"] for link in links)


def test_links_label_status_and_map_articles():
    links = build_links(PARAS)
    by_section = {link["from"].split("mục ")[1].split(" (")[0]: link for link in links}
    assert by_section["3.1"]["laws"] == [249]
    assert 251 in by_section["3.3"]["laws"]
    assert all("chỉ dùng tham khảo" in link["from"] for link in links)
    assert all("BLHS 1999" in link["from"] for link in links)
