"""Web UI: legal Q&A with the HierarGraph and Qwen3-8B on Ollama.

    .venv/bin/streamlit run app/ui.py      (or ./run_ui.sh)

Tab "Hỏi–đáp" runs the Q&A pipeline live; "Hỏi án" and "Kết quả thí
nghiệm" come next.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # graph, questions and caches are read from repo-relative paths

import streamlit as st  # noqa: E402

from app import service  # noqa: E402

st.set_page_config(page_title="Trợ lý pháp luật hình sự", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Đang nạp graph…")
def graph():
    return service.load_graph()


@st.cache_resource(show_spinner="Đang nạp model embedding…")
def embedder():
    return service.load_embedder()


@st.cache_resource
def client(base_url: str, model: str):
    return service.make_client(service.UISettings(base_url=base_url, model=model))


@st.cache_data
def samples():
    return service.sample_questions()


# ---- sidebar: settings and status ----------------------------------------
defaults = service.UISettings.from_env()
with st.sidebar:
    st.header("Cấu hình")
    model = st.text_input("Model Ollama", defaults.model)
    base_url = st.text_input("Địa chỉ Ollama", defaults.base_url)
    settings = service.UISettings(base_url=base_url, model=model)
    ready, status_msg = service.check_ollama(settings)
    (st.success if ready else st.error)(status_msg)
    if st.button("Kiểm tra lại"):
        st.rerun()
    g = graph()
    stats = g.stats()["nodes"]
    st.caption("Graph: " + " · ".join(f"{k} {v:,}".replace(",", ".") for k, v in stats.items()))
    st.caption("Hỏi–đáp dùng điều luật, tội danh và văn bản hướng dẫn; không dùng bản án.")

st.title("⚖️ Trợ lý pháp luật hình sự")
st.caption("LegalGraphRAG cho Bộ luật Hình sự Việt Nam · Qwen3-8B chạy trên máy")

tab_qa, tab_case, tab_results = st.tabs(["Hỏi–đáp", "Hỏi án", "Kết quả thí nghiệm"])


# ---- tab 1: Q&A ----------------------------------------------------------
def pick_sample():
    choice = st.session_state.get("sample")
    for s in samples():
        if s["title"] == choice:
            st.session_state["question"] = s["text"]
            st.session_state["sample_answer"] = s
            return
    st.session_state.pop("sample_answer", None)


def run_question(question: str):
    labels = dict(service.STEPS)
    with st.status("Đang xử lý…", expanded=True) as box:
        lines = {key: st.empty() for key, _ in service.STEPS}
        for key, name in service.STEPS:
            lines[key].markdown(f"○ {name}")
        lines["dien_dat_lai"].markdown(f"◐ {labels['dien_dat_lai']}…")

        def progress(step, info):
            if step == "dien_dat_lai":
                lines[step].markdown(f"✔ {labels[step]}: _{info.get('text', '')}_")
                lines["tim_dieu"].markdown(f"◐ {labels['tim_dieu']}…")
            elif step == "tim_dieu":
                found = info["ung_vien"]
                lines[step].markdown(f"✔ {labels[step]}: {len(found)} điều — " + ", ".join(found))
                lines["tham_dinh"].markdown(f"◐ {labels['tham_dinh']} 0/{len(found)}")
            elif step == "tham_dinh":
                mark = "thỏa mãn" if info["ap_dung"] else "không thỏa mãn"
                done = info["xong"] == info["tong"]
                lines[step].markdown(f"{'✔' if done else '◐'} {labels[step]} {info['xong']}/{info['tong']}"
                                     f" — {info['dieu']}: {mark}")
            elif step == "tra_loi":
                lines[step].markdown(f"◐ {labels[step]} từ {info['so_dieu']} điều…")

        try:
            result = service.ask(graph(), question, client(base_url, model).generate, embedder(), progress)
        except Exception as e:  # noqa: BLE001 - shown to the user, not swallowed
            box.update(label="Lỗi", state="error")
            st.error(f"Không trả lời được: {e}")
            return None
        lines["tra_loi"].markdown(f"✔ {labels['tra_loi']}")
        box.update(label=f"Xong trong {result['seconds']:.0f} giây", state="complete", expanded=False)
    return result


def show_result(result):
    st.subheader("📝 Trả lời")
    if result["parse_error"] or not result["answer"]:
        st.warning("Model không trả về câu trả lời đúng định dạng.")
        st.code(result["trace"].get("tra_loi_tho") or "", language=None)
    else:
        st.write(result["answer"])
    if result["citations"]:
        st.markdown("**Căn cứ:** " + "; ".join(result["citations"]))
    if result["crimes"]:
        st.markdown("**Tội danh:** " + "; ".join(result["crimes"]))
    st.caption("⚠️ Kết quả do AI tạo, chỉ để tham khảo, không thay thế tư vấn của luật sư.")

    cards = result["cards"]
    st.subheader(f"📚 Điều luật đã xem xét ({len(cards)})")
    for c in cards:
        icon = "✅" if c["ap_dung"] else "❌"
        star = " · ⭐ được trích" if c["duoc_trich"] else ""
        with st.expander(f"{icon} {c['label']} – {c['title']}   ·   tìm qua: {', '.join(c['routes']) or '—'}{star}"):
            st.caption(f"Thẩm định: {'thỏa mãn' if c['ap_dung'] else 'không thỏa mãn'}"
                       f" ({'từng yếu tố cấu thành' if c['cach_tham_dinh'] == 'judge_law' else 'một câu hỏi áp dụng'})")
            st.text(c["text"])
            if c["yeu_to"]:
                st.markdown("**Yếu tố cấu thành được hỏi:**")
                st.markdown("\n".join(f"- {q}" for q in c["yeu_to"]))
            if c["huong_dan"]:
                st.markdown(f"**Văn bản hướng dẫn ({len(c['huong_dan'])}):**")
                for h in c["huong_dan"]:
                    st.markdown(f"_{h['id']}_")
                    st.text(h["text"][:1500] + ("…" if len(h["text"]) > 1500 else ""))

    trace = result["trace"]
    with st.expander("🔍 Quá trình xử lý"):
        st.markdown(f"**Câu hỏi diễn đạt lại:** {trace.get('dien_dat_lai') or '—'}")
        guesses = trace["truy_xuat"].get("llm_toi_danh") or []
        if guesses:
            st.markdown("**Tội danh Qwen đoán → node Tội gần nhất:**")
            st.markdown("\n".join(f"- {h['ten_llm']} → {h['crime_node']} ({h['cosine']})" for h in guesses))
        issues = trace["truy_xuat"].get("llm_van_de") or []
        if issues:
            st.markdown("**Vấn đề pháp lý:**")
            st.markdown("\n".join(
                f"- {h['van_de']} → {', '.join(h['dieu'])}" if "dieu" in h else f"- ~~{h['van_de']}~~ (bỏ: {h['bo']})"
                for h in issues))
        st.markdown("**Ứng viên theo từng đường tìm:**")
        for route, found in trace["truy_xuat"]["tuyen"].items():
            st.markdown(f"- {service.ROUTE_NAMES.get(route, route)}: {', '.join(found) or '—'}")


with tab_qa:
    titles = [""] + [s["title"] for s in samples()]
    st.selectbox("Câu hỏi mẫu (từ bộ test)", titles, key="sample", on_change=pick_sample,
                 format_func=lambda t: t or "— tự nhập câu hỏi —")
    question = st.text_area("Câu hỏi của bạn", key="question", height=140,
                            placeholder="Ví dụ: Người 15 tuổi trộm xe máy trị giá 30 triệu có bị truy cứu trách nhiệm hình sự không?")
    if st.button("Gửi câu hỏi", type="primary", disabled=not ready or not question.strip()):
        st.session_state["result"] = run_question(question.strip())
    if not ready:
        st.info("Chưa gửi được câu hỏi: " + status_msg)
    if st.session_state.get("result"):
        show_result(st.session_state["result"])
        ref = st.session_state.get("sample_answer")
        if ref and ref["text"] == question:
            with st.expander("👩‍⚖️ Câu trả lời của luật sư (để so sánh)"):
                st.text(ref["answer"])
                if ref["url"]:
                    st.markdown(f"[Nguồn]({ref['url']})")

with tab_case:
    st.info("Đang làm: xem kết quả hỏi án đã chạy trên Kaggle (diễn biến, ứng viên, thẩm định, kết luận).")

with tab_results:
    st.info("Đang làm: bảng so sánh các lượt chạy với bài báo và phễu ứng viên → thẩm định → kết luận.")
