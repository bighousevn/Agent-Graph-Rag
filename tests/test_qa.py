"""Tests for vn_legal_graph.qa (scoring + pipeline) with a fake LLM."""
import json

import numpy as np

from tests.test_graph import fake_embed
from vn_legal_graph.graph.build import build_base_graph
from vn_legal_graph.qa.pipeline import answer_question, parse_answer, parse_crime_list
from vn_legal_graph.qa.scoring import parse_grade, references_text, same_crime, score

GOLD_321 = {
    "dieu_luat_bat_buoc": [{"luat": "BLHS", "dieu": "321", "khoan": "2", "diem": "c"}],
    "dieu_luat_tuy_chon": [],
    "toi_danh": ["Tội đánh bạc"],
}


def test_score_exact_citation():
    pred = {"cau_tra_loi": "Bị xử lý theo điểm c khoản 2 Điều 321.", "toi_danh": ["Tội đánh bạc"],
            "dieu_luat": [{"luat": "Bộ luật Hình sự", "dieu": "321", "khoan": "2", "diem": "C"}]}
    s = score(pred, GOLD_321)
    assert (s["dieu_recall"], s["dieu_precision"], s["khoan_diem"]) == (1.0, 1.0, 1.0)
    assert (s["toi_danh_recall"], s["toi_danh_precision"]) == (1.0, 1.0)
    assert s["so_tu"] == 10


def test_score_wrong_clause_and_extra_article():
    pred = {"cau_tra_loi": "x", "toi_danh": ["Tội đánh bạc", "Tội tổ chức đánh bạc hoặc gá bạc"],
            "dieu_luat": [{"luat": "BLHS", "dieu": "321", "khoan": "1"}, {"luat": "BLHS", "dieu": "322"}]}
    s = score(pred, GOLD_321)
    assert s["dieu_recall"] == 1.0 and s["dieu_precision"] == 0.5
    assert s["khoan_diem"] == 0.0
    assert s["toi_danh_precision"] == 0.5


def test_optional_article_is_not_an_error():
    gold = {"dieu_luat_bat_buoc": [{"luat": "BLHS", "dieu": "175"}],
            "dieu_luat_tuy_chon": [{"luat": "BLHS", "dieu": "353", "khoan": "6"}], "toi_danh": []}
    pred = {"dieu_luat": [{"luat": "BLHS", "dieu": "175"}, {"luat": "BLHS", "dieu": "353"}], "toi_danh": []}
    s = score(pred, gold)
    assert (s["dieu_recall"], s["dieu_precision"], s["khoan_diem"]) == (1.0, 1.0, None)
    assert s["toi_danh_recall"] == 1.0 and s["toi_danh_precision"] == 1.0


def test_procedural_code_kept_apart_from_penal_code():
    gold = {"dieu_luat_bat_buoc": [{"luat": "BLTTHS", "dieu": "155", "khoan": "3"}], "toi_danh": []}
    pred = {"dieu_luat": [{"luat": "BLHS", "dieu": "155", "khoan": "3"}], "toi_danh": []}
    assert score(pred, gold)["dieu_recall"] == 0.0


def test_same_crime_prefix_and_no_false_match():
    assert same_crime("Tội cố ý gây thương tích", "Tội cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác")
    assert not same_crime("Tội đánh bạc", "Tội tổ chức đánh bạc hoặc gá bạc")


def test_parse_grade_and_reference_cut():
    assert parse_grade('```json {"diem": "mot_phan", "ly_do": "thiếu tội"} ```') == {"diem": "mot_phan", "ly_do": "thiếu tội"}
    assert parse_grade('{"diem": "tốt"}') is None
    assert references_text("Điều 1...\nXem thêm: abc\nTrên đây là nội dung tư vấn") == "Điều 1..."


def test_parse_helpers():
    assert parse_crime_list('Kết quả: ["Tội đánh bạc", "Tội tổ chức đánh bạc hoặc gá bạc"]') == [
        "Tội đánh bạc", "Tội tổ chức đánh bạc hoặc gá bạc"]
    assert parse_crime_list("không có") == []
    assert parse_answer('{"cau_tra_loi": "ok"}') == {"cau_tra_loi": "ok", "toi_danh": [], "dieu_luat": []}
    assert parse_answer("không phải json") is None


LAWS = [
    {"id": 321, "suffix": "", "title": "Tội đánh bạc", "items": [{"text": "Điều 321. Tội đánh bạc ... ma túy", "crime": ["Tội đánh bạc"],
     "judge_dep": ["Có đánh bạc trái phép không?"], "related_laws": []}]},
    {"id": 51, "suffix": "", "title": "Các tình tiết giảm nhẹ", "items": [{"text": "Điều 51. Các tình tiết giảm nhẹ ... trộm cắp", "crime": [],
     "judge_dep": [], "related_laws": []}]},
]


def test_answer_question_end_to_end_with_fake_llm():
    g, _ = build_base_graph(LAWS, [], fake_embed)
    calls = []

    def fake(prompt, max_tokens=None):
        calls.append(prompt[:40])
        if "Nhân thân bị cáo" in prompt and "JSON" in prompt and "Diễn biến vụ án" in prompt:
            return '{"criminal_acts": ["đánh bạc qua app"]}'
        if "tối đa ba tội danh" in prompt:
            return '["Tội đánh bạc"]'
        if "Các yếu tố cần xét" in prompt:
            return '{"1": true}'
        if "Yếu tố thỏa mãn" in prompt:
            return "true"
        if "Đánh giá tình huống" in prompt:
            return "false"
        if "luật sư tư vấn" in prompt:
            assert "[BLHS Điều 321] (kiểm tra yếu tố cấu thành, chỉ tham khảo: thỏa mãn)" in prompt
            return json.dumps({"cau_tra_loi": "Khoản 2 Điều 321.", "toi_danh": ["Tội đánh bạc"],
                               "dieu_luat": [{"luat": "BLHS", "dieu": "321", "khoan": "2", "diem": "c"}]}, ensure_ascii=False)
        raise AssertionError(prompt[:80])

    out = answer_question(g, "Đánh bạc qua app 40 triệu, ma túy?", fake, fake_embed)
    assert "BLHS Điều 321" in out["truy_xuat"]["tuyen"]["llm_doan_toi"]
    assert out["judge"]["BLHS Điều 321"]["cach"] == "judge_law"
    assert out["judge"]["BLHS Điều 321"]["ap_dung"] is True
    assert out["judge"].get("BLHS Điều 51", {"cach": "don_gian"})["cach"] == "don_gian"
    assert out["dieu_dung_de_tra_loi"][0] == "BLHS Điều 321"  # accepted first, others after
    assert out["tra_loi"]["dieu_luat"][0]["dieu"] == "321"
    assert not out["tra_loi_loi"]


def test_hard_filter_option_keeps_original_behaviour():
    g, _ = build_base_graph(LAWS, [], fake_embed)

    def fake(prompt, max_tokens=None):
        if "Nhân thân bị cáo" in prompt and "Diễn biến vụ án" in prompt:
            return '{"criminal_acts": ["đánh bạc"]}'
        if "tối đa ba tội danh" in prompt:
            return '["Tội đánh bạc"]'
        if "Các yếu tố cần xét" in prompt:
            return '{"1": true}'
        if "Yếu tố thỏa mãn" in prompt or "Đánh giá tình huống" in prompt:
            return "true" if "Điều 321" in prompt else "false"
        assert "Điều 51." not in prompt
        return '{"cau_tra_loi": "ok"}'

    out = answer_question(g, "Đánh bạc qua app?", fake, fake_embed, {"loc_theo_judge": True})
    assert out["dieu_dung_de_tra_loi"] == ["BLHS Điều 321"]


def test_guidance_route_and_pick_closest_unit():
    from vn_legal_graph.qa.pipeline import guidance_laws, pick_guidance

    g51 = {"loai": "van_ban_huong_dan", "id": "NQ 04/2025, Điều 2 khoản 8", "text": "Phạm tội nhưng chưa gây thiệt hại: trộm cắp xe máy bị bắt"}
    g51b = {"loai": "van_ban_huong_dan", "id": "NQ 04/2025, Điều 2 khoản 1", "text": "Ngăn chặn tác hại, cây cối " + "x" * 300}
    laws = [dict(LAWS[0]), {**LAWS[1], "items": [{**LAWS[1]["items"][0], "related_laws": [g51b, g51]}]}]
    g, _ = build_base_graph(laws, [], fake_embed)
    q = fake_embed("trộm cắp xe máy rồi bị bắt có phải chưa gây thiệt hại?")
    assert guidance_laws(g, q, fake_embed, top_k=1) == ["law:51"]
    # closest unit first, and the far one is dropped when it does not fit
    text = pick_guidance([g51b, g51], q, fake_embed, budget=120)
    assert text.startswith("NQ 04/2025, Điều 2 khoản 8") and "khoản 1" not in text
    # without a query vector: document order, still within budget
    assert len(pick_guidance([g51b, g51], None, None, budget=120)) <= 120


def test_parse_rank_and_apply_rank():
    from vn_legal_graph.qa.pipeline import _apply_rank, parse_rank

    assert parse_rank("rank: [3, 1,2]") == [3, 1, 2]
    assert parse_rank("không rõ") == []
    assert _apply_rank(["a", "b", "c"], [3, 9, 1], keep=2) == ["c", "a"]  # 9 ignored
    assert _apply_rank(["a", "b", "c"], [], keep=2) == ["a", "b"]  # unreadable -> cosine order


class _StubGraph:
    """Two clusters; the LLM prefers the second by cosine."""

    def search(self, q, node_type, top_k=5, among=None):
        if node_type == "Cluster":
            return [("cl:ma_tuy", 0.9), ("cl:giet_nguoi", 0.8)][:top_k]
        pool = list(among) if among is not None else ["case:1", "case:2"]
        return [(c, 0.5) for c in pool][:top_k]

    def predecessors(self, node, rel):
        return {"cl:ma_tuy": ["case:1"], "cl:giet_nguoi": ["case:9"]}[node]

    def node(self, n):
        return {"description": n}


def test_reranked_cases_follow_llm_order():
    from vn_legal_graph.qa.pipeline import DEFAULTS, reranked_cases

    def fake(prompt, max_tokens=None):
        if "nhóm tội phạm" in prompt:
            assert "1. cl:ma_tuy" in prompt and "2. cl:giet_nguoi" in prompt
            return "rank: [2, 1]"
        assert "code1: case:9" in prompt  # cases of the kept cluster first, then direct ones
        return "[2, 1]"

    out = reranked_cases(_StubGraph(), np.ones(3), "dùng dao đâm chết người", fake, {**DEFAULTS, "n_clusters": 1})
    assert out["cum"] == ["cl:giet_nguoi"]
    assert out["an_ung_vien"] == ["case:9", "case:1", "case:2"]
    assert out["an"] == ["case:1", "case:9", "case:2"]


def test_answer_without_graph_uses_no_docs_prompt():
    from vn_legal_graph.qa.pipeline import answer_without_graph

    seen = []

    def fake(prompt, max_tokens=None):
        seen.append(prompt)
        return '{"cau_tra_loi": "Có", "toi_danh": ["Tội trộm cắp tài sản"], "dieu_luat": [{"luat": "BLHS", "dieu": "173"}]}'

    out = answer_without_graph("Trộm 3 triệu có bị tù không?", fake)
    assert len(seen) == 1 and "Các điều luật và tài liệu được cung cấp" not in seen[0]
    assert "Trộm 3 triệu có bị tù không?" in seen[0]
    assert out["tra_loi"]["dieu_luat"][0]["dieu"] == "173" and not out["tra_loi_loi"]
    bad = answer_without_graph("x", lambda p, max_tokens=None: "không phải json")
    assert bad["tra_loi_loi"] and bad["tra_loi"]["cau_tra_loi"] == ""
