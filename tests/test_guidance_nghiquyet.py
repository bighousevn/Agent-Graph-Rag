"""Tests for vn_legal_graph.law.guidance_nghiquyet (Nghị quyết HĐTP -> guidance links)."""
from vn_legal_graph.law.guidance_nghiquyet import article_refs, so_hieu_of, split_units

RESOLUTION = [
    "NGHỊ QUYẾT",
    "Hướng dẫn áp dụng Điều 65 của Bộ luật Hình sự về án treo",
    "Căn cứ Luật Tổ chức Tòa án nhân dân số 34/2024/QH15;",
    "Điều 1. Phạm vi điều chỉnh",
    "Nghị quyết này hướng dẫn áp dụng Điều 65 của Bộ luật Hình sự.",
    "Điều 2. Về các tình tiết giảm nhẹ quy định tại khoản 1 Điều 51 của Bộ luật Hình sự",
    "1. “Lập công chuộc tội” quy định tại điểm u khoản 1 Điều 51 của Bộ luật Hình sự là ...",
    "Ví dụ: Nguyễn Văn A ...",
    "2. Giao cấu quy định tại khoản 1 Điều 141, khoản 1 Điều 142 và khoản 1 Điều 145 của Bộ luật Hình sự là ...",
    "Điều 3. Thời hạn",
    "Thời gian thử thách theo Điều 5 của Nghị quyết này và khoản 3 Điều 4 của Luật sửa đổi, bổ sung một số điều của Bộ luật Hình sự số 86/2025/QH15.",
    "Điều 4. (được bãi bỏ)",
    "Điều 5. Hiệu lực thi hành",
    "Nghị quyết này có hiệu lực ...",
    "(15) biểu mẫu kèm theo Điều 361 của Bộ luật Tố tụng hình sự",
]


def test_split_units_by_khoan_and_skip_scope_and_effect():
    title, units = split_units(RESOLUTION)
    assert "Điều 65" in title and "Căn cứ" not in title
    assert [(u["dieu"], u["khoan"]) for u in units] == [("2", "1"), ("2", "2"), ("3", "")]
    assert units[0]["text"].startswith("Điều 2. Về các tình tiết") and "Ví dụ" in units[0]["text"]


def test_article_refs_list_form_and_non_code_owners():
    _, units = split_units(RESOLUTION)
    assert article_refs(units[0]["text"]) == ["BLHS:51"]  # heading "Điều 2." is the resolution's own
    assert article_refs(units[1]["text"]) == ["BLHS:51", "BLHS:141", "BLHS:142", "BLHS:145"]
    assert article_refs(units[2]["text"]) == []  # "của Nghị quyết này", "của Luật sửa đổi ..."
    assert article_refs("khoản 3 Điều 155 của Bộ luật Tố tụng hình sự") == ["BLTTHS:155"]


def test_so_hieu_from_text_or_file_name():
    assert so_hieu_of("x.docx", ["Số: 04/2025/NQ-HĐTP"], []) == "04/2025/NQ-HĐTP"
    assert so_hieu_of("data/raw/guidance/06_2019_NQ-HDTP.docx", ["NGHỊ QUYẾT"], []) == "06/2019/NQ-HĐTP"
    assert so_hieu_of("data/raw/guidance/02_VBHN-TANDTC_2022.docx", [], []) == "02/VBHN-TANDTC"
