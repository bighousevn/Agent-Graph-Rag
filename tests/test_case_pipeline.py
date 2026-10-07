"""Tests for vn_legal_graph.judge.case_pipeline with a fake LLM."""
import json

from tests.test_graph import fake_embed
from vn_legal_graph.graph.build import build_base_graph
from vn_legal_graph.judge.case_pipeline import (
    analyze_case,
    article_numbers,
    parse_adjudication,
    score_case,
    summarize,
    union_prediction,
)

LAWS = [
    {"id": 173, "suffix": "", "title": "Tội trộm cắp tài sản", "items": [{"text": "Điều 173. Tội trộm cắp tài sản ... trộm cắp",
     "crime": ["Tội trộm cắp tài sản"], "judge_dep": ["Có lén lút chiếm đoạt tài sản không?"], "related_laws": []}]},
    {"id": 249, "suffix": "", "title": "Tội tàng trữ trái phép chất ma túy", "items": [{"text": "Điều 249. ma túy",
     "crime": ["Tội tàng trữ trái phép chất ma túy"], "judge_dep": ["Có tàng trữ ma túy không?"], "related_laws": []}]},
    {"id": 51, "suffix": "", "title": "Các tình tiết giảm nhẹ", "items": [{"text": "Điều 51. giảm nhẹ trộm cắp", "crime": [],
     "judge_dep": [], "related_laws": []}]},
]


def test_parse_and_union():
    v = parse_adjudication('xx {"toi_danh": "Tội trộm cắp tài sản", "dieu_luat": ["Điều 173 khoản 1", "điểm s khoản 1 Điều 51"], '
                           '"hinh_phat": {"tu_co_thoi_han_thang": 9}} yy')
    assert v["toi_danh"] == ["Tội trộm cắp tài sản"] and article_numbers(v["dieu_luat"]) == ["173", "51"]
    assert parse_adjudication("không phải json") is None
    u = union_prediction([{"adjudicator": v}, {"adjudicator": {"toi_danh": ["Tội trộm cắp tài sản"], "dieu_luat": ["Điều 249"],
                                                               "hinh_phat": {"tu_co_thoi_han_thang": 24}}}])
    assert u == {"du_doan_toi_danh": ["Tội trộm cắp tài sản"], "du_doan_dieu": ["173", "51", "249"], "du_doan_tu_thang": 24}


def test_score_ignores_general_part_articles():
    pred = {"du_doan_dieu": ["173", "51"], "du_doan_toi_danh": ["Tội trộm cắp tài sản"]}
    s = score_case(pred, [173], ["Tội trộm cắp tài sản"], crime_articles={"173", "249"})
    assert s["dieu_dung_het"] and s["toi_dung_het"] and s["dieu_fp"] == 0
    wrong = score_case({"du_doan_dieu": ["249"], "du_doan_toi_danh": []}, [173], ["Tội trộm cắp tài sản"], {"173", "249"})
    m = summarize([s, wrong])
    assert m["dieu_accuracy"] == 0.5 and m["toi_danh_accuracy"] == 0.5 and 0 < m["dieu_micro_f1"] < 1


def test_analyze_case_researcher_auditor_adjudicator():
    g, _ = build_base_graph(LAWS, [], fake_embed)
    seen = []

    def fake(prompt, max_tokens=None):
        if "liệt kê tên các bị cáo" in prompt:
            return '["nguyễn văn a", "trần văn b"]'
        if "sắp xếp lại thành một đoạn mô tả" in prompt:
            name = "nguyễn văn a" if "Tên bị cáo: nguyễn văn a" in prompt else "trần văn b"
            return f"{name} lén lút trộm cắp xe máy" if name == "nguyễn văn a" else f"{name} cất giấu ma túy"
        if "Nhân thân bị cáo" in prompt and "Diễn biến vụ án" in prompt:
            return '{"criminal_acts": ["trộm cắp"]}' if "trộm" in prompt else '{"criminal_acts": ["tàng trữ ma túy"]}'
        if "nhóm tội phạm" in prompt or "vụ án tương tự" in prompt:
            return "[]"
        if "tối đa ba tội danh" in prompt:
            return '["Tội trộm cắp tài sản"]' if "trộm" in prompt else '["Tội tàng trữ trái phép chất ma túy"]'
        if "Các yếu tố cần xét" in prompt:
            return '{"1": true}'
        if "Yếu tố thỏa mãn" in prompt:
            return "true"
        if "điều luật ứng viên" in prompt:
            seen.append(prompt)
            assert "Điều 51" not in prompt.split("Các điều luật ứng viên:")[1]  # only crime articles are judged
            art = "173" if "Điều 173 Bộ luật Hình sự" in prompt else "249"
            return json.dumps({"toi_danh": ["Tội trộm cắp tài sản" if art == "173" else "Tội tàng trữ trái phép chất ma túy"],
                               "dieu_luat": [f"Điều {art}"], "hinh_phat": {"tu_co_thoi_han_thang": 12}}, ensure_ascii=False)
        raise AssertionError(prompt[:80])

    out = analyze_case(g, "a trộm xe, b cất giấu ma túy", fake, fake_embed, {"rerank_clusters_from": 0})
    assert out["bi_cao"] == ["nguyễn văn a", "trần văn b"] and len(seen) == 2
    assert out["theo_bi_cao"][0]["auditor"]["chap_nhan"] == ["173"]
    assert set(out["du_doan_dieu"]) == {"173", "249"}


def test_auditor_hint_mode_keeps_rejected_articles():
    from vn_legal_graph.judge.case_pipeline import format_laws

    g, _ = build_base_graph(LAWS, [], fake_embed)
    text = format_laws(g, ["law:173", "law:249"], {"law:173": True, "law:249": False})
    assert "Điều 173 Bộ luật Hình sự (kiểm tra yếu tố cấu thành, chỉ tham khảo: thỏa mãn)" in text
    assert "Điều 249 Bộ luật Hình sự (kiểm tra yếu tố cấu thành, chỉ tham khảo: không thỏa mãn)" in text
    assert "chỉ tham khảo" not in format_laws(g, ["law:173"])


def test_without_graph_segmented_unions_defendants():
    from vn_legal_graph.judge.case_pipeline import adjudicate_without_graph

    def fake(prompt, max_tokens=None):
        if "liệt kê tên các bị cáo" in prompt:
            return '["a", "b"]'
        if "sắp xếp lại thành một đoạn mô tả" in prompt:
            return "đánh bạc" if "Tên bị cáo: a" in prompt else "tổ chức đánh bạc"
        art = "322" if "tổ chức" in prompt.split("Vụ án:")[1] else "321"
        return json.dumps({"toi_danh": [], "dieu_luat": [f"Điều {art}"]})

    out = adjudicate_without_graph(fake, "a đánh bạc, b tổ chức", segment_defendants=True)
    assert out["du_doan_dieu"] == ["321", "322"] and out["bi_cao"] == ["a", "b"]
    assert adjudicate_without_graph(fake, "x đánh bạc")["du_doan_dieu"] == ["321"]


def test_adjudicator_context_shows_cases_and_guidance():
    from vn_legal_graph.judge.case_pipeline import adjudicator_context

    laws = json.loads(json.dumps(LAWS))
    laws[0]["items"][0]["related_laws"] = [{"loai": "van_ban_huong_dan", "id": "CV 01/2017 mục 3", "text": "trộm cắp xe máy của người thân"}]
    g, _ = build_base_graph(laws, [], fake_embed)
    g.add_node("case:x", "Case", description="lén lút lấy xe máy", crime="['tội trộm cắp tài sản']", law="[173]")
    cfg = {"case_chars": 800, "guidance_chars": 2000}
    q = fake_embed("trộm cắp")
    assert adjudicator_context(g, "", ["law:173"], ["case:x"], q, fake_embed, cfg) == ""
    only_cases = adjudicator_context(g, "an", ["law:173"], ["case:x"], q, fake_embed, cfg)
    assert "Án 1: tội danh: tội trộm cắp tài sản; điều luật áp dụng: Điều 173. Tóm tắt: lén lút lấy xe máy" in only_cases
    assert "hướng dẫn" not in only_cases
    both = adjudicator_context(g, "an+huong-dan", ["law:173", "law:249"], ["case:x"], q, fake_embed, cfg)
    assert "CV 01/2017 mục 3: trộm cắp xe máy của người thân" in both
    assert adjudicator_context(g, "an+huong-dan", ["law:249"], [], q, fake_embed, cfg) == ""
