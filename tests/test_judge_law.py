"""Tests for vn_legal_graph.judge.judge_law with a fake LLM."""
import pytest

from vn_legal_graph.judge.judge_law import (
    batch_prompt,
    judge_law,
    parse_bool,
    parse_numbered_bools,
    render_related,
    rerank_by_judgment,
)

LAW_251 = {
    "entry": 251,
    "description": "Điều 251. Tội mua bán trái phép chất ma túy\n1. Người nào mua bán trái phép chất ma túy...",
    "judge_dep": ["Có mua bán trái phép chất ma túy không?", "Có phạm tội 02 lần trở lên không?"],
    "related_laws": [{"loai": "dan_chieu_dieu_luat", "id": "Điều 255", "text": "nội dung"}],
}
CASE = "bị cáo cất giữ 0,2 gam heroine trong túi quần để sử dụng"


def test_parse_bool():
    assert parse_bool("true") is True
    assert parse_bool(" False.") is False
    assert parse_bool("Trả lời: TRUE") is True
    assert parse_bool("không chắc") is None


def test_parse_numbered_bools():
    assert parse_numbered_bools('{"1": true, "2": False}', 2) == [True, False]
    assert parse_numbered_bools('```json\n{"2": false, "1": true}\n```', 2) == [True, False]
    # missing, extra and non-boolean entries do not throw the answer away
    assert parse_numbered_bools('{"1": true, "3": false, "9": true}', 3) == [True, None, False]
    assert parse_numbered_bools('{"1": 1}', 1) == [None]
    assert parse_numbered_bools("[true, false]", 2) is None
    assert parse_numbered_bools("không", 2) is None


def test_render_related():
    assert render_related(LAW_251["related_laws"]) == "Điều 255: nội dung"
    assert render_related([]) == ""


def test_faithful_mode_one_call_per_element():
    calls = []

    def fake(prompt, max_tokens=None):
        calls.append(prompt)
        if "Yếu tố cần xét: Có mua bán" in prompt:
            return "false"
        if "Yếu tố cần xét: Có phạm tội 02 lần" in prompt:
            return "không rõ"
        return "false"  # final decision

    out = judge_law(fake, CASE, LAW_251, mode="trung-thanh")
    assert len(calls) == 3  # 2 elements + final
    assert out["ap_dung"] is False
    assert out["sai"] == ["Có mua bán trái phép chất ma túy không?"]
    assert out["khong_ro"] == ["Có phạm tội 02 lần trở lên không?"]
    assert "Yếu tố không thỏa mãn: ['Có mua bán trái phép chất ma túy không?']" in calls[-1]


def test_batch_mode_one_call_for_all_elements():
    calls = []

    def fake(prompt, max_tokens=None):
        calls.append(prompt)
        return '{"1": true, "2": false}' if "Các yếu tố cần xét" in prompt else "true"

    out = judge_law(fake, CASE, LAW_251, mode="gop")
    assert len(calls) == 2
    assert out["ap_dung"] is True
    assert out["dung"] == ["Có mua bán trái phép chất ma túy không?"]
    assert "1. Có mua bán" in batch_prompt(CASE, LAW_251)


def test_batch_mode_partial_answer_counts_missing_as_unknown():
    out = judge_law(
        lambda p, max_tokens=None: '{"1": true}' if "Các yếu tố" in p else "true", CASE, LAW_251, mode="gop"
    )
    assert out["dung"] == ["Có mua bán trái phép chất ma túy không?"]
    assert out["khong_ro"] == ["Có phạm tội 02 lần trở lên không?"]
    assert out["loi_doc_danh_sach"] is False


def test_batch_mode_unreadable_list():
    out = judge_law(lambda p, max_tokens=None: "không biết" if "Các yếu tố" in p else "false", CASE, LAW_251, mode="gop")
    assert out["loi_doc_danh_sach"] is True
    assert out["khong_ro"] == LAW_251["judge_dep"]


def test_bad_mode():
    with pytest.raises(ValueError):
        judge_law(lambda p, max_tokens=None: "true", CASE, LAW_251, mode="khác")


def test_rerank_by_judgment():
    # retrieval said 251 first; judge rejects 251, accepts 249
    assert rerank_by_judgment([251, 249, 173], {251: False, 249: True}) == [249, 251, 173]
    assert rerank_by_judgment([251, 249], {}) == [251, 249]


def test_final_prompt_says_aggravating_circumstances_are_optional():
    from vn_legal_graph.judge.judge_law import final_prompt

    p = final_prompt(CASE, LAW_251, ["Có mua bán trái phép chất ma túy không?"], ["Có tổ chức không?"])
    assert "yếu tố cấu thành cơ bản" in p
    assert "không bắt buộc" in p
