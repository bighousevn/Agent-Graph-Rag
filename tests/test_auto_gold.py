"""Tests for vn_legal_graph.qa.auto_gold (reference labels from the lawyer's answer)."""
from vn_legal_graph.qa.auto_gold import build_gold, extract_citations, guidance_match, khoan_precision
from vn_legal_graph.qa.scoring import score

ANSWER = """Điều 155 Bộ luật Tố tụng hình sự 2015, được sửa đổi bởi khoản 3 Điều 1 Luật sửa đổi Bộ luật Tố tụng hình sự 2021 quy định:
Điều 155. Khởi tố vụ án hình sự theo yêu cầu của bị hại
1. Chỉ được khởi tố vụ án hình sự về tội phạm quy định tại khoản 1 các điều 134, 135, 136 của Bộ Luật hình sự khi có yêu cầu của bị hại.
a) Điểm trích dẫn Điều 141 Bộ luật Hình sự.
Căn cứ theo khoản 3 Điều 155 thì bị hại đã rút yêu cầu không có quyền yêu cầu lại.
Hành vi cấu thành Tội hiếp dâm người dưới 16 tuổi theo Điều 142 và Điều 145 của Bộ luật Hình sự; khung tại điểm c khoản 2 Điều 321 BLHS và Điều 181, Bộ luật Hình sự 2015."""


def keys(cits):
    return [(c["luat"], c["dieu"], c["khoan"], c["diem"]) for c in cits]


def test_extract_citations_from_lawyer_prose_only():
    got = keys(extract_citations(ANSWER))
    assert ("BLTTHS", "155", "", "") in got
    # bare "khoản 3 Điều 155" takes the code the answer gave article 155
    assert ("BLTTHS", "155", "3", "") in got
    assert ("BLHS", "142", "", "") in got and ("BLHS", "145", "", "") in got
    assert ("BLHS", "321", "2", "c") in got
    assert ("BLHS", "181", "", "") in got
    # amending law and quoted law text are not citations
    assert not any(d == "1" for _, d, _, _ in got)
    assert not any(d in ("134", "135", "136", "141") for _, d, _, _ in got)


def test_build_gold_groups_and_scores():
    items = [{"explain": "4. Nguyễn Văn A là nhân viên của cửa hàng điện thoại di động MT do ông B làm chủ", "from": "CV 163 mục 4"}]
    q = {"qa_number": 1, "question": "Nguyễn Văn A là nhân viên của cửa hàng điện thoại di động MT do ông B làm chủ, phạm tội gì?",
         "answer": "Theo Điều 175 Bộ luật Hình sự, A phạm tội lạm dụng tín nhiệm chiếm đoạt tài sản."}
    gold = build_gold(q, items, ["163"], {"BLHS:175": ["Tội lạm dụng tín nhiệm chiếm đoạt tài sản"]})
    assert gold["nhom"] == "co_cong_van" and gold["cong_van_trung"]["from"] == "CV 163 mục 4"
    assert keys(gold["dieu_luat_bat_buoc"]) == [("BLHS", "175", "", "")]
    pred = {"cau_tra_loi": "Phạm tội lạm dụng tín nhiệm.", "toi_danh": ["Tội lạm dụng tín nhiệm chiếm đoạt tài sản"],
            "dieu_luat": [{"luat": "BLHS", "dieu": "175", "khoan": "1", "diem": ""}]}
    s = score(pred, gold)
    assert s["dieu_recall"] == 1.0 and s["dieu_precision"] == 1.0 and s["toi_danh_precision"] == 1.0

    other = dict(q, question="Câu hỏi khác hẳn về thủ tục", answer="Điều 155 Bộ luật Tố tụng hình sự quy định.")
    assert build_gold(other, items, ["163"], {})["nhom"] == "khong_cong_van"
    assert build_gold(dict(other, qa_number=7), items, ["163"], {}, manual={"7": "CV 163 mục 4"})["nhom"] == "co_cong_van"
    assert guidance_match("", items) is None


def test_khoan_precision():
    gold = {"trich_dan_luat_su": [{"luat": "BLHS", "dieu": "321", "khoan": "2", "diem": "c"},
                                  {"luat": "BLHS", "dieu": "175", "khoan": "", "diem": ""}]}
    right = {"dieu_luat": [{"luat": "BLHS", "dieu": "321", "khoan": "2", "diem": "c"}]}
    wrong = {"dieu_luat": [{"luat": "BLHS", "dieu": "321", "khoan": "1", "diem": ""}]}
    unknown = {"dieu_luat": [{"luat": "BLHS", "dieu": "175", "khoan": "1", "diem": ""}]}
    assert khoan_precision(right, gold) == 1.0
    assert khoan_precision(wrong, gold) == 0.0
    assert khoan_precision(unknown, gold) is None
