"""Tests for vn_legal_graph.cases.vicsr.

Fixtures mimic the real ViCSR files: ``<id> [---] <text> [-/-]`` records,
lowercase text, the Eth glyph "ð" in place of "đ", and a ground-truth label
using BLHS 1999 numbering (194) for a drug case.
"""
import pytest

from vn_legal_graph.cases.vicsr import (
    build_crime_name_index,
    extract_convicted_crimes,
    judgment_year,
    parse_law_entry,
    parse_records,
    split_sections,
)

LAW_SHORTEN = (
    "51 [---] điều 51 bộ luật hình sự các tình tiết giảm nhẹ trách nhiệm hình sự [-/-] "
    "173 [---] điều 173 bộ luật hình sự tội trộm cắp tài sản [-/-] "
    "249 [---] điều 249 bộ luật hình sự tội tàng trữ trái phép chất ma túy [-/-] "
    "250 [---] điều 250 bộ luật hình sự tội vận chuyển trái phép chất ma túy [-/-] "
    "251 [---] điều 251 bộ luật hình sự tội mua bán trái phép chất ma túy [-/-] "
    "253 [---] điều 253 bộ luật hình sự tội tàng trữ vận chuyển mua bán hoặc chiếm đoạt "
    "tiền chất dùng vào việc sản xuất trái phép chất ma túy [-/-] "
    "532 [---] điều 106 bộ luật tố tụng hình sự xử lý vật chứng [-/-] "
    "839 [---] ðiều 413 bộ luật tố tụng hình sự phạm vi áp dụng [-/-] "
    "1005 [---] điều 69 luật thi hành án hình sự xử lý người được tha tù trước thời hạn [-/-]"
)

# BLHS 1999-era drug case: ground truth says 194, the verdict names the
# 2015 crime "tàng trữ trái phép chất ma túy" (Điều 249).
DETAIL_DRUG = (
    "tòa án nhân dân huyện n bản án số 37 2018 hs st ngày 16 4 2018 "
    "theo quyết định đưa vụ án ra xét xử số 40 2018 "
    "nội dung vụ án hồi 21 giờ ngày 16 4 2018 tổ công tác bắt quả tang phùng thanh h "
    "về hành vi tàng trữ trái phép chất ma túy thu giữ 01 gói heroine "
    "nhận định của hội đồng xét xử hành vi của bị cáo đã đủ yếu tố cấu thành "
    "quyết định tuyên bố bị cáo phùng thanh h phạm tội tàng trữ trái phép chất ma túy "
    "về hình phạt căn cứ khoản 1 điều 194 điều 46 của bộ luật hình sự"
)

DETAIL_TWO_CRIMES = (
    "bản án số 12 2019 hs st nội dung vụ án bị cáo mua heroine rồi bán lại "
    "nhận định của tòa án bị cáo phạm tội mua bán trái phép chất ma túy "
    "quyết định tuyên bố bị cáo nguyễn văn t phạm tội mua bán trái phép chất ma túy "
    "và tội trộm cắp tài sản căn cứ điều 251 điều 173"
)

DETAIL_PRECURSOR = (
    "nội dung vụ án bị cáo cất giữ tiền chất nhận định của tòa án đủ căn cứ "
    "quyết định tuyên bố bị cáo phạm tội tàng trữ vận chuyển mua bán hoặc chiếm đoạt "
    "tiền chất dùng vào việc sản xuất trái phép chất ma túy"
)


@pytest.fixture()
def laws():
    return {i: parse_law_entry(i, t) for i, t in parse_records(LAW_SHORTEN).items()}


@pytest.fixture()
def name_index(laws):
    return build_crime_name_index(laws)


def test_parse_records_splits_and_normalizes_eth():
    recs = parse_records(LAW_SHORTEN)
    assert sorted(recs) == [51, 173, 249, 250, 251, 253, 532, 839, 1005]
    assert recs[839].startswith("điều 413")  # "ð" fixed to "đ"


def test_parse_records_rejects_malformed():
    with pytest.raises(ValueError):
        parse_records("1 không có dấu phân cách [-/-]")


def test_law_entry_code_and_article(laws):
    assert (laws[249].bo_luat, laws[249].dieu) == ("blhs", "249")
    assert laws[249].title == "tội tàng trữ trái phép chất ma túy"
    assert (laws[532].bo_luat, laws[532].dieu) == ("bltths", "106")
    assert (laws[839].bo_luat, laws[839].dieu) == ("bltths", "413")
    assert (laws[1005].bo_luat, laws[1005].dieu) == ("lthahs", "69")


def test_is_crime_only_for_blhs_toi(laws):
    assert laws[173].is_crime
    assert not laws[51].is_crime  # BLHS, but not a crime article
    assert not laws[532].is_crime


def test_split_sections_skips_early_quyet_dinh():
    s = split_sections(DETAIL_DRUG)
    assert s["noi_dung"].startswith("nội dung vụ án")
    assert "quyết định đưa vụ án" not in s["noi_dung"]
    assert s["nhan_dinh"].startswith("nhận định của hội đồng xét xử")
    assert s["quyet_dinh"].startswith("quyết định tuyên bố")
    assert "nhận định" not in s["noi_dung"]


def test_split_sections_missing_markers():
    s = split_sections("văn bản không có cấu trúc")
    assert s == {"noi_dung": None, "nhan_dinh": None, "quyet_dinh": None}


def test_old_code_label_resolved_by_crime_name(name_index):
    qd = split_sections(DETAIL_DRUG)["quyet_dinh"]
    assert extract_convicted_crimes(qd, name_index) == [249]


def test_multiple_crimes_in_verdict(name_index):
    qd = split_sections(DETAIL_TWO_CRIMES)["quyet_dinh"]
    assert extract_convicted_crimes(qd, name_index) == [251, 173]


def test_several_defendants_in_one_sentence(name_index):
    qd = (
        "vì các lẽ trên quyết định tuyên bố bị cáo hoàng văn t phạm tội tàng trữ trái phép "
        "chất ma túy bị cáo hà tiến y phạm tội mua bán trái phép chất ma túy xử phạt bị cáo t "
        "bị cáo y có 01 tiền án về tội trộm cắp tài sản"
    )
    assert extract_convicted_crimes(qd, name_index) == [249, 251]


def test_longest_name_wins(name_index):
    qd = split_sections(DETAIL_PRECURSOR)["quyet_dinh"]
    assert extract_convicted_crimes(qd, name_index) == [253]


# Real-data failure modes found on the full ViCSR run.
DETAIL_OLD_TONE_AND_PRIOR = (
    "nội dung vụ án bị cáo cất giữ heroine nhận định của tòa án bị cáo có 01 tiền án "
    "hđxx xem xét khi quyết định hình phạt cho bị cáo "
    "vì các lẽ trên quyết định tuyên bố bị cáo lê văn c phạm tội tàng trữ trái phép chất ma tuý "
    "tổng hợp với hình phạt của bản án số 5 2017 về tội trộm cắp tài sản"
)


def test_verdict_starts_at_vi_cac_le_tren():
    s = split_sections(DETAIL_OLD_TONE_AND_PRIOR)
    assert s["quyet_dinh"].startswith("vì các lẽ trên")
    assert "khi quyết định hình phạt" in s["nhan_dinh"]


def test_old_tone_placement_and_prior_conviction_ignored(name_index):
    qd = split_sections(DETAIL_OLD_TONE_AND_PRIOR)["quyet_dinh"]
    # "ma tuý" matches "ma túy"; "về tội trộm cắp" (prior sentence) is not read.
    assert extract_convicted_crimes(qd, name_index) == [249]


def test_tone_normalization_leaves_quy_alone():
    from vn_legal_graph.cases.vicsr import _norm_name

    assert _norm_name("ma tuý") == "ma túy"
    assert _norm_name("quý hiếm") == "quý hiếm"
    assert _norm_name("hoà giải") == "hòa giải"


def test_short_name_alias_for_247():
    laws = {247: parse_law_entry(
        247, "điều 247 bộ luật hình sự tội trồng cây thuốc phiện cây côca cây cần sa "
             "hoặc các loại cây khác có chứa chất ma túy")}
    idx = build_crime_name_index(laws)
    qd = "vì các lẽ trên tuyên bố bị cáo giàng a s phạm tội trồng cây cần sa áp dụng điểm c"
    assert extract_convicted_crimes(qd, idx) == [247]


def test_fuzzy_match_absorbs_ocr_damage(name_index):
    qd = "vì các lẽ trên tuyên bố bị cáo h phạm tội tàng trư trái phép chất ma túy xử phạt"
    assert extract_convicted_crimes(qd, name_index, with_fuzzy_flag=True) == ([249], True)


def test_no_fuzzy_match_on_unrelated_text(name_index):
    qd = "vì các lẽ trên tuyên bố bị cáo h phạm tội đánh bạc xử phạt"
    assert extract_convicted_crimes(qd, name_index) == []


def test_garbled_opening_falls_back_to_conviction_sentence():
    detail = (
        "nội dung vụ án bị cáo cất giữ heroine nhận định của tòa án bị cáo phải chịu án phí "
        "v c c lï trªn quyõt þnh 1 tuyên bố bị cáo đào chiến t phạm tội tàng trữ trái phép "
        "chất ma túy 2 áp dụng điểm c khoản 1 điều 249"
    )
    s = split_sections(detail)
    assert s["quyet_dinh"].startswith("tuyên bố bị cáo đào chiến t")


LAWS_WITH_194 = (
    "194 [---] điều 194 bộ luật hình sự tội sản xuất buôn bán hàng giả là thuốc chữa bệnh "
    "thuốc phòng bệnh [-/-] "
    "249 [---] điều 249 bộ luật hình sự tội tàng trữ trái phép chất ma túy [-/-] "
    "51 [---] điều 51 bộ luật hình sự các tình tiết giảm nhẹ trách nhiệm hình sự [-/-]"
)


@pytest.fixture()
def laws_194():
    return {i: parse_law_entry(i, t) for i, t in parse_records(LAWS_WITH_194).items()}


def test_cited_article_confirmed_by_name(laws_194):
    from vn_legal_graph.cases.vicsr import extract_cited_crimes

    judgment = "nội dung vụ án bị cáo tàng trữ trái phép chất ma tuý 0 5 gam heroine"
    qd = "vì các lẽ trên quyết định căn cứ điểm c khoản 1 điều 249 điều 51 xử phạt bị cáo"
    assert extract_cited_crimes(qd, judgment, laws_194) == [249]


def test_cited_old_code_article_rejected(laws_194):
    from vn_legal_graph.cases.vicsr import extract_cited_crimes

    # BLHS 1999 Điều 194 (drugs); 2015's Điều 194 name never appears.
    judgment = "nội dung vụ án bị cáo tàng trữ trái phép heroine"
    qd = "vì các lẽ trên quyết định căn cứ khoản 1 điều 194 điều 46 xử phạt bị cáo"
    assert extract_cited_crimes(qd, judgment, laws_194) == []


def test_label_case_reports_method(laws_194):
    from vn_legal_graph.cases.vicsr import label_case

    idx = build_crime_name_index(laws_194)
    by_sentence = label_case(DETAIL_DRUG, laws_194, idx)
    assert (by_sentence["toi_danh"], by_sentence["cach_gan_nhan"]) == ([249], "ten_toi")

    by_article = label_case(
        "nội dung vụ án tàng trữ trái phép chất ma túy nhận định của tòa án đủ căn cứ "
        "vì các lẽ trên quyết định căn cứ điểm c khoản 1 điều 249 xử phạt bị cáo",
        laws_194,
        idx,
    )
    assert (by_article["toi_danh"], by_article["cach_gan_nhan"]) == ([249], "dieu_luat")

    unlabeled = label_case("văn bản không có cấu trúc", laws_194, idx)
    assert (unlabeled["toi_danh"], unlabeled["cach_gan_nhan"]) == ([], None)


def test_judgment_year():
    assert judgment_year(DETAIL_DRUG) == 2018
    assert judgment_year(DETAIL_TWO_CRIMES) == 2019
    assert judgment_year("không có năm") is None


def test_is_appellate():
    from vn_legal_graph.cases.vicsr import is_appellate

    assert is_appellate("tòa án nhân dân tp hcm bản án số 197 2019 hs pt ngày 22 04 2019")
    assert is_appellate("thành phần hội đồng xét xử phúc thẩm gồm có")
    assert not is_appellate(DETAIL_DRUG)
