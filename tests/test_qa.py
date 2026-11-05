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
    # a citation written into toi_danh (seen on the FB run) goes to dieu_luat
    moved = parse_answer('{"cau_tra_loi": "ok", "toi_danh": ["Tội lừa đảo", {"luat": "BLHS", "dieu": "174", '
                         '"khoan": "1", "diem": ""}], "dieu_luat": []}')
    assert moved["toi_danh"] == ["Tội lừa đảo"]
    assert moved["dieu_luat"] == [{"luat": "BLHS", "dieu": "174", "khoan": "1", "diem": ""}]


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
        if "diễn đạt lại câu hỏi của người dân" in prompt:
            return "Hành vi đánh bạc trái phép bằng ứng dụng với số tiền 40 triệu đồng có cấu thành tội đánh bạc không."
        if "Nhân thân bị cáo" in prompt and "JSON" in prompt and "Diễn biến vụ án" in prompt:
            return '{"criminal_acts": ["đánh bạc qua app"]}'
        if "tối đa ba tội danh" in prompt:
            return '["Tội đánh bạc"]'
        if "KHÔNG gắn riêng với một tội" in prompt:
            return '[{"van_de": "tình tiết giảm nhẹ", "trich": "Đánh bạc qua app"}]'
        if "Các yếu tố cần xét" in prompt:
            assert "câu hỏi tư vấn của người dân" in prompt and "null" in prompt  # Q&A judge prompt
            return '{"1": true}'
        if "Yếu tố thỏa mãn" in prompt:
            return "true"
        if "Đánh giá tình huống" in prompt:
            return "false"
        if "luật sư tư vấn" in prompt:
            assert "[BLHS Điều 321]\nKiểm tra yếu tố cấu thành (chỉ tham khảo): thỏa mãn" in prompt
            assert "- Câu hỏi đã nêu, thỏa mãn: Có đánh bạc trái phép không?" in prompt
            return json.dumps({"cau_tra_loi": "Khoản 2 Điều 321.", "toi_danh": ["Tội đánh bạc"],
                               "dieu_luat": [{"luat": "BLHS", "dieu": "321", "khoan": "2", "diem": "c",
                                              "trich_dan": "Tội đánh bạc"}]}, ensure_ascii=False)
        raise AssertionError(prompt[:80])

    out = answer_question(g, "Đánh bạc qua app 40 triệu, ma túy?", fake, fake_embed)
    assert "BLHS Điều 321" in out["truy_xuat"]["tuyen"]["llm_doan_toi"]
    assert out["judge"]["BLHS Điều 321"]["cach"] == "judge_law"
    assert out["judge"]["BLHS Điều 321"]["ap_dung"] is True
    assert out["judge"].get("BLHS Điều 51", {"cach": "don_gian"})["cach"] == "don_gian"
    assert out["dieu_dung_de_tra_loi"][0] == "BLHS Điều 321"  # accepted first, others after
    assert out["tra_loi"]["dieu_luat"][0]["dieu"] == "321"
    assert not out["tra_loi_loi"]
    assert out["truy_xuat"]["tuyen"]["van_de_phap_ly"][0] == "BLHS Điều 51"
    assert out["hoi_lai_vi_ngan"]  # "Khoản 2 Điều 321." is asked again
    assert out["judge"]["BLHS Điều 321"]["dung"] == ["Có đánh bạc trái phép không?"]
    check = out["kiem_tra_trich_dan"][0]
    assert check["cite"] == "BLHS Điều 321 khoản 2 điểm c" and check["trong_tai_lieu"]
    assert check["trich_dan_co_that"] and check["khop_vi_tri"] is False  # the text has no khoản 2


def test_render_judgment_lists_elements_and_unknown_count():
    from vn_legal_graph.qa.pipeline import render_judgment

    text = render_judgment({"ap_dung": False, "dung": ["a?", "b?", "c?"], "sai": ["d?"], "so_chua_ro": 4}, max_items=2)
    assert text.splitlines() == [
        "Kiểm tra yếu tố cấu thành (chỉ tham khảo): không thỏa mãn",
        "- Câu hỏi đã nêu, thỏa mãn: a?; b?; và 1 yếu tố khác",
        "- Câu hỏi nêu ngược lại: d?",
        "- Chưa rõ (câu hỏi không nhắc tới): 4 yếu tố",
    ]
    assert render_judgment({"ap_dung": True}, 15) == "Kiểm tra yếu tố cấu thành (chỉ tham khảo): thỏa mãn"


def test_judge_candidate_null_is_not_stated_and_case_prompt_still_available():
    from vn_legal_graph.qa.pipeline import judge_candidate

    laws = [{**LAWS[0], "items": [{**LAWS[0]["items"][0], "judge_dep": ["Có đánh bạc?", "Trị giá từ 5 triệu?", "Tái phạm?"]}]}]
    g, _ = build_base_graph(laws, [], fake_embed)
    seen = []

    def fake(prompt, max_tokens=None):
        seen.append(prompt)
        return '{"1": true, "2": null, "3": false}' if "Các yếu tố cần xét" in prompt else "true"

    out = judge_candidate(g, "law:321", "Đánh bạc qua app", fake, "gop", chua_ro=True)
    assert (out["dung"], out["sai"], out["so_chua_ro"]) == (["Có đánh bạc?"], ["Tái phạm?"], 1)
    assert "Câu hỏi: Đánh bạc qua app" in seen[0]
    seen.clear()
    judge_candidate(g, "law:321", "Đánh bạc qua app", fake, "gop", chua_ro=False)
    assert "Vụ án: Đánh bạc qua app" in seen[0] and "null" not in seen[0]  # case prompt unchanged


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
    assert len(seen) == 2 and "Các điều luật và tài liệu được cung cấp" not in seen[0]
    assert "lần trả lời trước chỉ có 1 từ" in seen[1]  # "Có" alone is asked again
    assert "Trộm 3 triệu có bị tù không?" in seen[0]
    assert out["tra_loi"]["dieu_luat"][0]["dieu"] == "173" and not out["tra_loi_loi"]
    bad = answer_without_graph("x", lambda p, max_tokens=None: "không phải json")
    assert bad["tra_loi_loi"] and bad["tra_loi"]["cau_tra_loi"] == ""


def test_general_slots_are_added_after_other_candidates():
    from vn_legal_graph.qa.pipeline import DEFAULTS, retrieve_candidates

    laws = LAWS + [{"id": 65, "suffix": "", "title": "Án treo", "items": [{"text": "Điều 65. Án treo", "crime": [], "judge_dep": [], "related_laws": []}]},
                   {"id": 173, "suffix": "", "title": "Tội trộm cắp tài sản", "items": [{"text": "Điều 173. Tội trộm cắp tài sản",
                    "crime": ["Tội trộm cắp tài sản"], "judge_dep": [], "related_laws": []}]}]
    g, _ = build_base_graph(laws, [], fake_embed)

    def fake(prompt, max_tokens=None):
        if "tối đa ba tội danh" in prompt:
            return '["Tội trộm cắp tài sản", "Tội đánh bạc"]'
        if "KHÔNG gắn riêng với một tội" in prompt:
            return '[{"van_de": "án treo", "trich": "hưởng án treo"}, {"van_de": "Tội trộm cắp tài sản", "trich": "Trộm xe"}]'
        return "[]"

    q = "Trộm xe có được hưởng án treo không?"
    out = retrieve_candidates(g, q, fake_embed(q), fake, fake_embed, {**DEFAULTS, "max_candidates": 2, "laws_per_issue": 1})
    assert out["ung_vien"][:2] == ["law:173", "law:321"]  # crime articles fill max_candidates
    assert "law:65" in out["ung_vien"][2:]  # the general article still gets in
    assert all(n not in ("law:173", "law:321") for n in out["tuyen"]["van_de_phap_ly"])  # only general articles


def test_retry_keeps_the_longer_answer():
    from vn_legal_graph.qa.pipeline import answer_with_retry

    answers = iter(['{"cau_tra_loi": "Có"}', '{"cau_tra_loi": "' + " ".join(["từ"] * 50) + '"}'])
    raw, ans, retried = answer_with_retry("p", lambda p, max_tokens=None: next(answers), 40)
    assert retried and len(ans["cau_tra_loi"].split()) == 50
    _, ans, retried = answer_with_retry("p", lambda p, max_tokens=None: '{"cau_tra_loi": "' + " ".join(["a"] * 45) + '"}', 40)
    assert not retried


def test_guidance_citations_are_not_scored():
    gold = {"dieu_luat_bat_buoc": [{"luat": "BLHS", "dieu": "321"}], "toi_danh": []}
    pred = {"dieu_luat": [{"luat": "Bộ luật Hình sự", "dieu": "321"}, {"luat": "Nghị quyết 04/2025/NQ-HĐTP", "dieu": "3"},
                          {"luat": "Công văn 250/TANDTC-PC", "dieu": "1"}], "toi_danh": []}
    s = score(pred, gold)
    assert s["dieu_recall"] == 1.0 and s["dieu_precision"] == 1.0


def test_qa_mode_skips_cases_and_features_by_default():
    from vn_legal_graph.qa.pipeline import DEFAULTS, retrieve_candidates

    g, _ = build_base_graph(LAWS, [], fake_embed)
    asked = []

    def fake(prompt, max_tokens=None):
        asked.append(prompt)
        assert "Nhân thân bị cáo" not in prompt and "nhóm tội phạm" not in prompt and "vụ án tương tự" not in prompt
        if "diễn đạt lại câu hỏi của người dân" in prompt:
            return "Hành vi đánh bạc trái phép có bị phạt tù không."
        if "tối đa ba tội danh" in prompt:
            return '["Tội đánh bạc"]'
        if "KHÔNG gắn riêng với một tội" in prompt:
            assert "Tối đa năm vấn đề" in prompt and "Bộ luật Tố tụng hình sự" in prompt
            assert "Hành vi đánh bạc trái phép có bị phạt tù không." in prompt  # the rewrite is passed on
            return "[]"
        if "luật sư tư vấn" in prompt:
            return '{"cau_tra_loi": "' + " ".join(["x"] * 50) + '", "dieu_luat": []}'
        return "true"

    out = answer_question(g, "Đánh bạc có bị tù không?", fake, fake_embed)
    assert out["dien_dat_lai"] == "Hành vi đánh bạc trái phép có bị phạt tù không."
    assert out["dac_trung"] == "" and not any(k in out["truy_xuat"]["tuyen"] for k in ("truc_tiep", "qua_cum", "an_tuong_tu"))
    with_cases = retrieve_candidates(g, "q", fake_embed("q"), lambda p, max_tokens=None: "[]", fake_embed, {**DEFAULTS, "dung_an": True})
    assert "truc_tiep" in with_cases["tuyen"]


def test_crime_route_takes_two_nearest_crime_nodes():
    from vn_legal_graph.qa.pipeline import llm_crime_laws

    laws = LAWS + [{"id": 322, "suffix": "", "title": "Tội tổ chức đánh bạc hoặc gá bạc", "items": [{"text": "Điều 322.",
                    "crime": ["Tội tổ chức đánh bạc hoặc gá bạc"], "judge_dep": [], "related_laws": []}]}]
    g, _ = build_base_graph(laws, [], fake_embed)
    one, _ = llm_crime_laws(g, "x", lambda p, max_tokens=None: '["Tội đánh bạc"]', fake_embed)
    two, hits = llm_crime_laws(g, "x", lambda p, max_tokens=None: '["Tội đánh bạc"]', fake_embed, 2)
    assert len(one) == 1 and set(two) == {"law:321", "law:322"} and len(hits) == 2


def test_issue_laws_ranked_by_closeness_to_the_question():
    from vn_legal_graph.qa.pipeline import llm_issue_laws

    def embed(text):  # tuổi -> x, giảm nhẹ -> y
        return np.array([1.0 if "uổi" in text else 0.0, 1.0 if "giảm nhẹ" in text else 0.0, 0.1])

    laws = LAWS + [{"id": 12, "suffix": "", "title": "Tuổi chịu trách nhiệm hình sự", "items": [{"text": "Điều 12.", "crime": [],
                    "judge_dep": [], "related_laws": []}]}]
    g, _ = build_base_graph(laws, [], embed)
    fake = lambda p, max_tokens=None: ('[{"van_de": "tuổi chịu trách nhiệm hình sự", "trich": "q"}, '
                                       '{"van_de": "tình tiết giảm nhẹ", "trich": "q"}]')
    in_order, _ = llm_issue_laws(g, "q", fake, embed, 1)
    assert in_order == ["law:12", "law:51"]  # as the LLM listed them
    ranked, _ = llm_issue_laws(g, "q", fake, embed, 1, rank_vec=embed("có được giảm nhẹ không"))
    assert ranked == ["law:51", "law:12"]  # the title closest to the question first


def test_issues_must_quote_the_question_and_not_repeat_the_prompt():
    from vn_legal_graph.prompts.vi import LEGAL_ISSUES_PROMPT
    from vn_legal_graph.qa.pipeline import copied_from_prompt, grounded_issues, parse_issues

    q = "Bị hại đã rút đơn yêu cầu khởi tố thì có được yêu cầu lại không?"
    raw = ('[{"van_de": "rút yêu cầu khởi tố", "trich": "Bị hại đã rút đơn yêu cầu khởi tố"},'
           ' {"van_de": "tuổi chịu trách nhiệm hình sự", "trich": "người dưới 16 tuổi"},'
           ' {"van_de": "quy định chung của Bộ luật Hình sự", "trich": "có được yêu cầu lại"}]')
    kept, dropped = grounded_issues(parse_issues(raw), q, LEGAL_ISSUES_PROMPT)
    assert kept == ["rút yêu cầu khởi tố"]
    assert dropped[0]["ly_do"] == "trích dẫn không có trong câu hỏi"  # invented detail
    assert dropped[1]["ly_do"].startswith("chép chữ của prompt")  # repeats the instructions
    assert copied_from_prompt("rút yêu cầu khởi tố", LEGAL_ISSUES_PROMPT, q) == []
    assert parse_issues('["án treo"]') == [{"van_de": "án treo", "trich": ""}]


def test_law_names_are_not_counted_as_copying():
    from vn_legal_graph.prompts.vi import LEGAL_ISSUES_PROMPT
    from vn_legal_graph.qa.pipeline import copied_from_prompt

    q = "Bị hại có được xem hồ sơ vụ án không?"
    assert copied_from_prompt("quyền của bị hại trong tố tụng hình sự", LEGAL_ISSUES_PROMPT, q) == []
    assert copied_from_prompt("quy định chung của Bộ luật Hình sự", LEGAL_ISSUES_PROMPT, q) != []


def test_order_by_confidence():
    from vn_legal_graph.qa.pipeline import order_by_confidence

    laws = ["a", "b", "c", "d"]
    judgments = {"a": {"ap_dung": False}, "b": {"ap_dung": True}, "c": {"ap_dung": False}, "d": {"ap_dung": True}}
    routes = {"r1": ["a", "c", "d"], "r2": ["c"], "r3": ["c", "b"]}
    # accepted first (b, d), b found by 1 route, d by 1: kept order; then c (3 routes) before a (1 route)
    assert order_by_confidence(laws, judgments, routes) == ["b", "d", "c", "a"]


def test_answer_prompt_asks_role_and_main_question_and_no_point_lists():
    from vn_legal_graph.prompts.vi import QA_ANSWER_PROMPT

    p = QA_ANSWER_PROMPT.format(question="Tôi bị đánh, phải làm gì?", laws="[BLHS Điều 134]\n...")
    # the role and the main question come first in the JSON, before the facts
    out = p[p.index("Đầu ra là"):]
    assert out.index('"nguoi_hoi"') < out.index('"cau_hoi_chinh"') < out.index('"tinh_tiet"') < out.index('"cau_tra_loi"')
    assert "không mở đầu bằng \"Không bị xử lý hình sự\"" in p
    # an article used only for comparison is cited by khoản, without a list of points
    assert "Không liệt kê lần lượt các điểm của một khoản" in p and "không có điểm f" in p
    assert "Kết luận không được trái với căn cứ đã nêu" in p
    assert "Tôi bị đánh, phải làm gì?" in p and "{" in p and "{{" not in p


def test_collapse_point_lists_merges_point_runs_and_default_point_a():
    from vn_legal_graph.qa.pipeline import collapse_point_lists, parse_answer

    run = [{"luat": "BLHS", "dieu": "134", "khoan": "1", "diem": d, "trich_dan": f"q{d}"} for d in "abcde"]
    out, notes = collapse_point_lists(run + [{"luat": "BLHS", "dieu": "174", "khoan": "2", "diem": "c"}])
    assert out == [{"luat": "BLHS", "dieu": "134", "khoan": "1", "diem": "", "trich_dan": ""},
                   {"luat": "BLHS", "dieu": "174", "khoan": "2", "diem": "c"}]
    assert notes == ["Điều 134 khoản 1: gộp 5 điểm (a, b, c, d, e) thành cả khoản"]
    # the khoản's own quote is kept when the model also cited the khoản
    out, _ = collapse_point_lists([{"luat": "BLHS", "dieu": "134", "khoan": "1", "trich_dan": "Người nào"}] + run)
    assert out == [{"luat": "BLHS", "dieu": "134", "khoan": "1", "trich_dan": "Người nào", "diem": ""}]
    # two points of one khoản are a real choice, kept
    assert collapse_point_lists(run[:2])[0] == run[:2]
    # several mitigating circumstances (Điều 51) often apply together, kept
    d51 = [{"luat": "BLHS", "dieu": "51", "khoan": "1", "diem": d} for d in "bst"]
    assert collapse_point_lists(d51) == (d51, [])
    # point a given to every one of 3+ articles is dropped
    many_a = [{"luat": "BLHS", "dieu": n, "khoan": "1", "diem": "a"} for n in ("322", "175", "150")]
    out, notes = collapse_point_lists(many_a + [{"luat": "BLHS", "dieu": "384", "khoan": "1", "diem": ""}])
    assert [c["diem"] for c in out] == ["", "", "", ""] and notes == ["bỏ điểm a gán cho cả 3 điều"]
    # but not when another point shows the model did choose
    mixed = many_a + [{"luat": "BLHS", "dieu": "174", "khoan": "2", "diem": "c"}]
    assert collapse_point_lists(mixed) == (mixed, [])
    # parse_answer applies it and keeps the notes
    ans = parse_answer(json.dumps({"cau_tra_loi": "x", "dieu_luat": run}, ensure_ascii=False))
    assert len(ans["dieu_luat"]) == 1 and ans["gop_diem"]


def test_answer_prompt_limits_the_preparation_khoan():
    from vn_legal_graph.prompts.vi import QA_ANSWER_PROMPT

    assert "chỉ dùng khi hành vi chưa được thực hiện" in QA_ANSWER_PROMPT
    assert "tùy kết quả giám định, định giá" in QA_ANSWER_PROMPT
