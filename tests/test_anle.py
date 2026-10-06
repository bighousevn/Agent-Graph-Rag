"""Tests for vn_legal_graph.cases.anle (criminal judgments from anle.toaan.gov.vn)."""
from vn_legal_graph.cases.anle import cut_facts, decision_articles, facts_of, mask_short_names

INTRO = " ".join(["khoảng 22 giờ ngày 5 3 2021 nguyễn văn a dùng dao đâm anh b nhiều nhát"] * 4)
MD = (
    "TÒA ÁN NHÂN DÂN TỈNH X\nN ỘI DUNG V Ụ ÁN:\n"
    "Khoảng 22 giờ ngày 5/3/2021, Nguyễn Văn A dùng dao đâm anh B nhiều nhát. " * 5
    + "Tại b ản cáo tr ạng s ố 12/CT-VKS, Viện kiểm sát truy tố bị cáo về tội Giết người theo Điều 123.\n"
    "NHẬN ĐỊNH CỦA TÒA ÁN: ...\nVì các lẽ trên,\nQUYẾT ĐỊNH:\nTuyên bố bị cáo phạm tội Giết người. Áp dụng điểm n khoản 1 Điều 123."
)


def test_decision_articles_only_from_decision_part():
    doc = {"markdown": MD, "citations_law": [
        {"law_name": "Hình sự", "article": "123", "span": [[MD.index("Điều 123."), MD.index("Điều 123.") + 9], [len(MD) - 10, len(MD) - 1]]},
        {"law_name": "Hình sự", "article": "173", "span": [[10, 20]]},  # e.g. a prior conviction, before the decision
        {"law_name": "Tố tụng hình sự", "article": "123", "span": [[len(MD) - 5, len(MD) - 1]]},
    ]}
    assert decision_articles(doc, {"123", "173"}) == ["123"]


def test_facts_cut_at_spaced_out_indictment_and_masked():
    facts = facts_of({"markdown": MD}, ["giết người"])
    assert facts.startswith("khoảng 22 giờ") and "cáo tr" not in facts and "giết người" not in facts
    assert "điều" not in facts


def test_cut_ignores_markers_in_the_opening_words():
    text = "theo bản án sơ thẩm và các tài liệu có trong hồ sơ " + INTRO + " tại bản án hình s ự sơ thẩm số 3"
    out = cut_facts(text)
    assert out.startswith("theo bản án sơ thẩm") and out.endswith("nhiều nhát")


def test_mask_short_crime_names():
    names = ["cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác", "hủy hoại hoặc cố ý làm hư hỏng tài sản"]
    text = "yêu cầu khởi tố về tội cố ý gây thương tích và tội cố ý làm hư hỏng tài sản của anh b"
    assert mask_short_names(text, names) == "yêu cầu khởi tố về tội ×× và tội ×× của anh b"
