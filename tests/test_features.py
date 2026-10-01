"""Tests for vn_legal_graph.cases.features, with a fake LLM."""
from vn_legal_graph.cases.features import (
    annotate_cases,
    build_prompt,
    concat_feature_description,
    parse_features,
)

GOOD = (
    'Đây là kết quả:\n{"defendant_info": ["đã thành niên", "có tiền án"], '
    '"criminal_acts": ["tàng trữ", "ma túy tổng hợp"], "victim_property_details": [], '
    '"intent_remorse": ["thành khẩn khai báo"]}'
)


def test_parse_features_tolerates_surrounding_text():
    f = parse_features(GOOD)
    assert f["defendant_info"] == ["đã thành niên", "có tiền án"]
    assert f["victim_property_details"] == []


def test_parse_features_fills_missing_keys_and_drops_non_lists():
    f = parse_features('{"criminal_acts": ["trộm cắp", " "], "intent_remorse": "tự thú"}')
    assert f == {
        "defendant_info": [],
        "criminal_acts": ["trộm cắp"],
        "victim_property_details": [],
        "intent_remorse": [],
    }


def test_parse_features_failure_returns_none():
    assert parse_features("không có json") is None
    assert parse_features("{hỏng json}") is None


def test_concat_skips_empty_groups():
    text = concat_feature_description(parse_features(GOOD))
    assert text == (
        "Nhân thân bị cáo: đã thành niên, có tiền án. "
        "Hành vi phạm tội: tàng trữ, ma túy tổng hợp. "
        "Lỗi và thái độ: thành khẩn khai báo."
    )


def test_build_prompt_with_and_without_crime():
    with_crime = build_prompt("bị cáo cất giữ heroine", ["Tội tàng trữ trái phép chất ma túy"])
    assert "Tội danh: Tội tàng trữ trái phép chất ma túy" in with_crime
    without = build_prompt("bị cáo cất giữ heroine")
    assert "Tội danh" not in without
    assert without.endswith("Diễn biến vụ án: bị cáo cất giữ heroine")


def test_build_prompt_truncates_fact():
    p = build_prompt("x" * 100, max_chars=10)
    assert p.endswith("Diễn biến vụ án: " + "x" * 10)


def test_annotate_never_gives_crime_to_test_cases():
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return GOOD

    cases = [
        {"id": "a", "dien_bien": "d1", "toi_danh": ["Tội trộm cắp tài sản"], "vai_tro": "corpus"},
        {"id": "b", "dien_bien": "d2", "toi_danh": ["Tội trộm cắp tài sản"], "vai_tro": "test"},
    ]
    out = annotate_cases(cases, fake_llm, use_crime_hint=True)
    assert "Tội danh" in prompts[0]
    assert "Tội danh" not in prompts[1]
    assert out[1]["mo_ta_dac_trung"].startswith("Nhân thân bị cáo")
    assert "dac_trung" not in cases[0]  # input not mutated


def test_annotate_marks_parse_failures():
    out = annotate_cases(
        [{"id": "a", "dien_bien": "d", "toi_danh": [], "vai_tro": "corpus"}],
        lambda p: "xin lỗi, tôi không thể",
        use_crime_hint=False,
    )
    assert out[0]["dac_trung_loi"] is True
    assert out[0]["mo_ta_dac_trung"] == ""
