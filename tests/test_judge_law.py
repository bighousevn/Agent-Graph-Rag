"""Tests for vn_legal_graph.judge.judge_law with a fake LLM."""
import pytest

from vn_legal_graph.judge.judge_law import (
    batch_prompt,
    judge_law,
    parse_bool,
    parse_bool_list,
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


def test_parse_bool_list():
    assert parse_bool_list("[true, False]", 2) == [True, False]
    assert parse_bool_list("```json\n[true,false]\n```", 2) == [True, False]
    assert parse_bool_list("[true]", 2) is None
    assert parse_bool_list("[1, 0]", 2) is None
    assert parse_bool_list("không", 2) is None


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
        return "[true, false]" if "Các yếu tố cần xét" in prompt else "true"

    out = judge_law(fake, CASE, LAW_251, mode="gop")
    assert len(calls) == 2
    assert out["ap_dung"] is True
    assert out["dung"] == ["Có mua bán trái phép chất ma túy không?"]
    assert "1. Có mua bán" in batch_prompt(CASE, LAW_251)


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
