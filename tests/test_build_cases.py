"""Tests for vn_legal_graph.cases.build_cases."""
from vn_legal_graph.cases.build_cases import (
    cut_before_prosecution,
    dedupe,
    extract_dieu_khoan,
    extract_facts,
    mask_leaks,
    split_by_crime,
)

FACTS = " ".join(["bị cáo đi xe máy đến nhà anh b lấy trộm chiếc điện thoại"] * 5)

NOI_DUNG = (
    "nội dung vụ án bị viện kiểm sát nhân dân huyện n truy tố về hành vi phạm tội như sau "
    + FACTS
    + " tại bản cáo trạng số 12 viện kiểm sát truy tố bị cáo về tội trộm cắp tài sản theo "
    "khoản 1 điều 173 bộ luật hình sự tại phiên tòa kiểm sát viên đề nghị tuyên bố bị cáo "
    "phạm tội trộm cắp tài sản"
)

NAMES = ["tàng trữ trái phép chất ma túy", "trộm cắp tài sản", "trồng cây thuốc phiện"]


def test_dedupe_keeps_lowest_id():
    kept, dropped = dedupe({5: "a  b", 2: "a b", 7: "c"})
    assert sorted(kept) == [2, 7]
    assert dropped == {5: 2}


def test_cut_before_prosecution_ignores_opening_marker():
    out = cut_before_prosecution(NOI_DUNG)
    assert out.endswith("lấy trộm chiếc điện thoại")
    assert "cáo trạng" not in out
    # The opening "truy tố về hành vi phạm tội như sau" is within the first
    # MIN_FACT_WORDS words, so it does not trigger a cut.
    assert out.startswith("nội dung vụ án bị viện kiểm sát")


def test_cut_strips_trailing_connector_words():
    text = FACTS + " tại bán cáo trạng số 5 truy tố bị cáo"  # "bán" is a typo for "bản"
    assert cut_before_prosecution(text) == FACTS


def test_extract_facts_skips_legacy_font_text():
    garbled = FACTS + " cña viön kióm s t nhµ n­íc ngµy héi ång xđt xö bþ c o µ µ µ"
    facts, source = extract_facts({"noi_dung": garbled}, FACTS, NAMES)
    assert source == "tom_tat"


def test_cut_at_truy_to_ve_toi_without_cao_trang():
    text = (
        FACTS + " viện kiểm sát nhân dân huyện văn quan truy tố các bị cáo đinh văn b và đinh thị t "
        "về tội trồng cây thuốc phiện theo điểm c khoản 1 điều 247"
    )
    assert cut_before_prosecution(text) == FACTS


def test_cut_falls_back_to_prosecutor_marker():
    text = FACTS + " kiểm sát viên đề nghị tuyên bố bị cáo phạm tội trộm cắp tài sản"
    assert cut_before_prosecution(text) == FACTS


def test_mask_leaks_keeps_behaviour_description():
    text = (
        "bị cáo có hành vi tàng trữ trái phép chất ma túy bị khởi tố về tội tàng trữ trái phép "
        "chất ma túy theo khoản 1 điều 249"
    )
    out = mask_leaks(text, NAMES)
    assert "có hành vi tàng trữ trái phép chất ma túy" in out
    assert "về tội ××" in out
    assert "điều ××" in out
    assert "249" not in out


def test_extract_facts_prefers_noi_dung_and_masks():
    facts, source = extract_facts({"noi_dung": NOI_DUNG}, "tóm tắt ngắn", NAMES)
    assert source == "noi_dung"
    assert "trộm cắp tài sản" not in facts
    assert "điều 173" not in facts


def test_extract_facts_falls_back_to_summary():
    facts, source = extract_facts({"noi_dung": "quá ngắn"}, FACTS, NAMES)
    assert source == "tom_tat"
    assert facts == FACTS


def test_extract_facts_none_when_too_short():
    assert extract_facts({"noi_dung": None}, "quá ngắn", NAMES) is None


def test_extract_dieu_khoan():
    qd = "căn cứ điểm c khoản 1 điều 249 điểm s khoản 1 điều 51 khoản 2 điều 173 điều 247"
    assert extract_dieu_khoan(qd, [249, 173, 247, 251]) == ["249.1.c", "173.2", "247"]


def test_split_by_crime_disjoint_and_capped():
    case_crimes = {i: [249] for i in range(100)}
    case_crimes.update({100 + i: [251] for i in range(8)})
    case_crimes[200] = [249, 251]  # counts toward the rarer crime (251)
    corpus, test = split_by_crime(case_crimes, test_per_crime=3, corpus_max_per_crime=20)
    assert not set(corpus) & set(test)
    assert sum(1 for c in test if 249 in case_crimes[c] and 251 not in case_crimes[c]) == 3
    assert sum(1 for c in corpus if case_crimes[c] == [249]) == 20
    # 251 has 9 cases (8 + the shared one): 3 test, 6 corpus.
    assert sum(1 for c in test + corpus if 251 in case_crimes[c]) == 9


def test_split_by_crime_is_deterministic():
    case_crimes = {i: [249] for i in range(50)}
    assert split_by_crime(case_crimes, 5, 10, seed=1) == split_by_crime(case_crimes, 5, 10, seed=1)


def test_cut_uses_earliest_marker_across_patterns():
    # "cáo trạng số" (first pattern) appears after "đại diện viện kiểm sát".
    text = FACTS + " tại phiên tòa đại diện viện kiểm sát giữ nguyên quan điểm theo cáo trạng số 5"
    assert cut_before_prosecution(text) == FACTS + " tại phiên tòa"


def test_mask_ocr_damaged_crime_name():
    out = mask_leaks("truy tố bị cáo về tội ta ng trư trái phép chất ma túy theo quy định", NAMES)
    assert "về tội ××" in out
    assert "trái phép chất ma túy" not in out
    # unrelated phrases after "tội" stay
    assert mask_leaks("bắt người phạm tội quả tang", NAMES) == "bắt người phạm tội quả tang"
