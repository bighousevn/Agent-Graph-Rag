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
