"""Tests for vn_legal_graph.qa.citations (no LLM)."""
from vn_legal_graph.qa.citations import check_citations, locate_quote

D134 = """Điều 134. Tội cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác
1. Người nào cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác mà tỷ lệ tổn thương cơ thể từ 11% đến 30% hoặc dưới 11% nhưng thuộc một trong các trường hợp sau đây, thì bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm:
a) Dùng vũ khí, vật liệu nổ, hung khí nguy hiểm hoặc thủ đoạn có khả năng gây nguy hại cho nhiều người;
b) Dùng a-xít nguy hiểm hoặc hóa chất nguy hiểm;
2. Phạm tội gây thương tích cho người khác mà tỷ lệ tổn thương cơ thể từ 31% đến 60%, thì bị phạt tù từ 02 năm đến 06 năm:
a) Đối với 02 người trở lên;
6. Người nào chuẩn bị vũ khí, vật liệu nổ, hung khí nguy hiểm, a-xít nguy hiểm, hóa chất nguy hiểm hoặc thành lập hoặc tham gia nhóm tội phạm nhằm gây thương tích, thì bị phạt cải tạo không giam giữ đến 02 năm hoặc phạt tù từ 03 tháng đến 02 năm."""


def test_locate_quote_finds_khoan_and_diem():
    assert locate_quote(D134, "Dùng a-xít nguy hiểm hoặc hóa chất nguy hiểm") == ("1", "b")
    assert locate_quote(D134, "“tỷ lệ tổn thương cơ thể từ 31% đến 60%”") == ("2", "")
    # a point letter from khoản 1 does not leak into khoản 6
    assert locate_quote(D134, "Người nào chuẩn bị vũ khí, vật liệu nổ") == ("6", "")
    # ellipsis: the longest piece is searched; spacing and case do not matter
    assert locate_quote(D134, "dùng VŨ KHÍ,   vật liệu nổ, hung khí nguy hiểm ... nhiều người") == ("1", "a")
    assert locate_quote(D134, "Người nào giết người") is None
    assert locate_quote(D134, "ngắn") is None  # too short to tell


def test_check_citations_flags_wrong_place_missing_article_and_no_quote():
    laws = {"BLHS Điều 134": D134}
    rows = check_citations([
        {"luat": "Bộ luật Hình sự", "dieu": "134", "khoan": "1", "diem": "b", "trich_dan": "Dùng a-xít nguy hiểm"},
        {"luat": "BLHS", "dieu": "134", "khoan": "6", "diem": "a",
         "trich_dan": "chuẩn bị vũ khí, vật liệu nổ, hung khí nguy hiểm"},
        {"luat": "BLHS", "dieu": "134", "khoan": "2", "trich_dan": "Người nào giết người"},
        {"luat": "BLHS", "dieu": "173", "khoan": "1"},
        {"luat": "BLHS", "dieu": "134", "khoan": "1"},
        "rác",
    ], laws)
    assert [r["cite"] for r in rows] == ["BLHS Điều 134 khoản 1 điểm b", "BLHS Điều 134 khoản 6 điểm a",
                                         "BLHS Điều 134 khoản 2", "BLHS Điều 173 khoản 1", "BLHS Điều 134 khoản 1"]
    assert (rows[0]["trich_dan_co_that"], rows[0]["vi_tri"], rows[0]["khop_vi_tri"]) == (True, "khoản 1 điểm b", True)
    assert (rows[1]["vi_tri"], rows[1]["khop_vi_tri"]) == ("khoản 6", False)  # khoản 6 has no point a
    assert rows[2]["trich_dan_co_that"] is False and rows[2]["khop_vi_tri"] is None
    assert rows[3]["trong_tai_lieu"] is False and rows[3]["trich_dan_co_that"] is None
    assert rows[4]["trong_tai_lieu"] and rows[4]["trich_dan_co_that"] is None


def test_repeated_phrase_counts_if_any_occurrence_fits():
    from vn_legal_graph.qa.citations import quote_places

    text = "Điều 9.\n1. Chung:\na) hung khí nguy hiểm khác nhau;\n2. Riêng:\nb) hung khí nguy hiểm khác nhau;"
    assert quote_places(text, "hung khí nguy hiểm khác") == [("1", "a"), ("2", "b")]
    row = check_citations([{"luat": "BLHS", "dieu": "9", "khoan": "2", "diem": "b",
                            "trich_dan": "hung khí nguy hiểm khác"}], {"BLHS Điều 9": text})[0]
    assert (row["vi_tri"], row["khop_vi_tri"]) == ("khoản 2 điểm b", True)


def test_mentioned_not_cited():
    from vn_legal_graph.qa.citations import mentioned_not_cited

    answer = {"cau_tra_loi": "Không phạm tội vì chưa đủ yếu tố của Điều 174; xem thêm Điều 51 và Điều 174, Điều 2a.",
              "dieu_luat": [{"luat": "BLHS", "dieu": "51"}]}
    assert mentioned_not_cited(answer) == ["Điều 174", "Điều 2a"]
    assert mentioned_not_cited({"cau_tra_loi": "Không phạm tội.", "dieu_luat": []}) == []
