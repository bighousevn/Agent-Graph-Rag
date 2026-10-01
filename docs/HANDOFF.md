# HANDOFF — LegalGraphRAG cho Bộ luật Hình sự Việt Nam

> **Gửi phiên Claude Code tiếp theo:** đọc hết file này trước khi làm gì.
> Nó thay cho toàn bộ cuộc trò chuyện trước, diễn ra trong một container cloud và không chuyển sang máy này được.
> Đừng hỏi lại những gì đã ghi ở mục "Quyết định đã chốt".

Cập nhật lần cuối: 2026-10-01. Nhánh: `claude/legalgraphrag-framework-hv23z3`.

---

## 1. Dự án là gì

Áp dụng phương pháp graph của **LegalGraphRAG** (ACL 2026, <https://github.com/XMUDeepLIT/LegalGraphRAG>) cho **Bộ luật Hình sự Việt Nam (BLHS)**.

- **Giai đoạn hiện tại chỉ xây graph.** Chưa làm các agent Researcher / Auditor / Adjudicator.
- **Phạm vi thử nghiệm:** 2 chương, 26 điều, 26 tội.
  - **Chương XVI**: Điều 168–180, các tội xâm phạm sở hữu.
  - **Chương XX**: Điều 247–259, các tội phạm về ma túy.
- **Mở rộng ra toàn bộ BLHS** chỉ làm sau khi Phase 4 cho kết quả đo được.

## 2. Bảo mật API key (người dùng yêu cầu)

- Người dùng **không muốn Claude biết LLM API key**.
- Key nằm trong `.env`. File này đã có trong `.gitignore` và bị chặn đọc bằng `.claude/settings.json`.
- **Claude không đọc, in, hay sửa `.env`.** Claude không chạy các lệnh có gọi LLM thật, chỉ viết code cho các lệnh đó. Người dùng tự chạy trong terminal của mình, ví dụ:
  ```bash
  python scripts/build_law_layer.py --chapters XVI XX
  ```
- **Claude được tự chạy:** các lệnh không cần key, gồm `pytest`, `--dry-run`, parse dữ liệu, thống kê.
- Quy tắc deny trong `.claude/settings.json` chỉ chặn ở mức tốt nhất có thể (Bash có nhiều cách đọc file). Tôn trọng yêu cầu này cả ở những chỗ quy tắc không chặn được.

## 3. Khởi động trên máy local

```bash
git checkout claude/legalgraphrag-framework-hv23z3
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v                                        # kỳ vọng: 14 passed
python scripts/build_law_layer.py --chapters XVI XX --dry-run
#   -> "26 Điều parsed, 26 are crime ('Tội ...') articles."
```

Trước đây phiên cloud bị proxy chặn các trang: `congbobanan.toaan.gov.vn`, `anle.toaan.gov.vn`, `vbpl.vn`, `thuvienphapluat.vn`, `huggingface.co`, `doi.org`, `drive.google.com`. **Trên máy local thì các trang này nên truy cập được.** Đó là lý do chuyển phiên. Khi cào dữ liệu phải giới hạn tốc độ (≥ 1–2 giây mỗi request) để tránh bị chặn IP.

## 4. Kiến trúc graph của bài gốc (đã đọc trực tiếp từ code)

Các file gốc cần tham chiếu:
- `core/graph_construct/graph_db.py`
- `core/graph_construct/feature_graph.py`
- `core/LegalGraphRAG.py`
- `core/utils/util.py`

### 4.1. Một graph duy nhất, không phải 3 graph

- Graph là `networkx.MultiDiGraph`, lưu trong RAM, ghi ra file bằng pickle.
- Đây là **property graph**, không phải triple-store/RDF:
  - Mỗi node là một dict thuộc tính tùy ý.
  - Thuộc tính dạng list như `judge_dep` **không** tách thành các triple riêng.
- Có **4 loại node**:

| Node | Thuộc tính | Trường được embed |
|---|---|---|
| `Cases` | `description` (4 nhóm đặc trưng do LLM trích, ghép thành text), `caseId`, `crime[]`, `law[]` | `description` |
| `Laws` | `entry` (số điều), `description` (toàn văn điều), `crimes[]`, `judge_dep[]`, `related_laws[]`, `insights` (để trống, agent điền sau) | **chỉ** `description` |
| `Crimes` | `description` (chỉ tên tội) | `description` |
| `Cluster` | `summary` (LLM tóm tắt cụm), `top_crimes[]`, `top_crime_counts[]` | `summary` |

- **Các cạnh**:

| Cạnh | Chiều | Cách tạo |
|---|---|---|
| `RELATES_TO_LAW` | Case → Law | khớp số điều |
| `RELATED_CRIME` | Law → Crime | khớp tên tội: exact trước, rồi fuzzy theo chuỗi con; ghi `match_type` |
| `SIMILAR_TO` | Case → Case | kNN cosine top-3, ghi `score`. Code gốc tính O(n²) |
| `BELONGS_TO` | Case → Cluster | Louvain community detection; chọn case đại diện theo PageRank·0.7 + degree·0.3; LLM tóm tắt thành Cluster |

### 4.2. Cách truy vấn của bài gốc (`core/utils/util.py::analyze_case`)

Embedding **không** phải khâu quyết định cuối cùng. Truy vấn chạy qua 5 bước:

1. **Chuẩn hoá đầu vào giống hệt corpus.**
   - Tách vụ án theo từng bị cáo (`segment_case_text_withname`).
   - `get_features()` trích 4 nhóm đặc trưng, dùng **cùng prompt** đã dùng khi xây node Case.
   - `concat_feature_descriptions()` ghép lại thành text, rồi mới embed. **Không embed câu hỏi thô.**
2. **Hai nhánh tìm Case.**
   - `top_retrieve`: tìm top-5 Cluster theo cosine → LLM rerank cluster → tìm Case gần nhất trong cluster được chọn.
   - `direct_retrieve`: kNN trên toàn bộ Case.
   - Gộp kết quả, LLM rerank, giữ lại ≤ 3 case.
   - Đi theo cạnh `RELATES_TO_LAW` để lấy các Law liên quan.
3. **Nhánh "augment" (`retrieve_law`).**
   - LLM đọc fact thô và tự đoán tên tội (`RETRIEVE_LAW_PROMPT`).
   - Embed **tên tội** (cụm từ ngắn), so với node Crime, rồi đi ngược cạnh `RELATED_CRIME` để ra Law.
4. **`judge_law`.** LLM xác minh từng Law ứng viên có thật sự áp dụng cho vụ án không. Law nào không qua thì loại.
5. **`judge_crime_all`.** LLM ra kết luận tội danh.

**Hai bước còn thiếu trong sơ đồ Phase 4 cũ, phải thêm khi code Phase 4:**
- Bước 1: chuẩn hoá truy vấn về 4 nhóm đặc trưng trước khi embed.
- Bước 4: xác minh kiểu `judge_law`.

## 5. Trạng thái các phase

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Nền tảng: `config.py`, `llm.py` (client OpenAI-compatible, cache đĩa theo hash prompt), `embedding.py`, `prompts/vi.py` | ✅ Xong |
| 1 | Tầng Law + Crime: `law/parse_blhs.py` → `judge_dep.py` → `link_guidance.py` → `build_law_crime.py`, CLI `scripts/build_law_layer.py` | ✅ Code xong và đã kiểm chứng. ⏳ `judge_dep` **chưa chạy thật** (chờ người dùng tự chạy với key) |
| **2** | **Thu thập và làm sạch bản án** | ⏸️ **Đang ở đây.** Chưa có dữ liệu, chưa có code |
| 3 | Tầng Case: trích đặc trưng, embedding, kNN, Louvain/Cluster, ráp graph hoàn chỉnh | Chưa làm |
| 4 | Truy vấn (mục 4.2) và đo Recall@k của Điều luật và tội danh trên tập test | Chưa làm |
| 5+ | Agent Researcher / Auditor / Adjudicator; mở rộng toàn bộ BLHS | Chưa làm |

### Kết quả đã kiểm chứng ở Phase 1 (file thật `data/raw/law/100_2015_QH13_296661.docx`)

- Parse được **426/426 Điều**. Trong đó **314 điều** có tiêu đề "Tội ...".
- Phần thứ hai có 4 điều không phải tội: **122, 352, 367, 392**. `is_crime_article` loại đúng 4 điều này.
- Chương XVI có đúng 13/13 điều (168–180). Chương XX có đúng 13/13 điều (247–259).
- **Lỗi dữ liệu đã xử lý:** một số đoạn dùng ký tự `Ð/ð` (Latin Eth, U+00D0/U+00F0) thay cho `Đ/đ` (U+0110/U+0111), ví dụ "Ðiều 317". Nếu không chuẩn hoá thì điều đó **mất mà không báo lỗi**. `normalize_text()` đã xử lý. Với file luật mới, luôn đối chiếu lại số lượng điều.
- **Dẫn chiếu chéo tìm được:**
  - Điều 172, 173, 174, 175 → Điều 168.
  - Điều 256 → Điều 255.
- **Lưu ý về nguồn luật:** file này là **Luật 100/2015/QH13 bản gốc**, chưa hợp nhất các lần sửa đổi:
  - Luật 12/2017/QH14.
  - Luật 86/2025/QH15, hiệu lực từ 01/7/2025.
  Dùng tạm được để thử cấu trúc, nhưng phải thay bằng bản hợp nhất trước khi dùng nghiêm túc.
- Văn bản hướng dẫn: `data/raw/guidance/guidance_manifest.json` chỉ là **danh sách đề xuất**, mọi mục đều ở trạng thái `can_xac_minh`. Chưa có nội dung text. `guidance_links.json` chưa tồn tại; khi chưa có file này, `link_guidance` chỉ gắn dẫn chiếu chéo.

## 6. Quyết định đã chốt

- **Không copy repo gốc.** Viết code mới, chỉ mượn ý tưởng và cấu trúc.
- **Node Law ở cấp Điều.** Khoản và điểm lưu làm thuộc tính. Ở luật Việt Nam, 1 Điều = 1 Tội, nên `RELATED_CRIME` khớp chính xác, không cần fuzzy.
- **Phần thứ nhất (Điều 1–107) không thành node**, nhưng vẫn được parse và lưu. Điều 51 và 52 (tình tiết giảm nhẹ, tăng nặng) sẽ cần cho tầng agent.
- **LLM gọi qua API OpenAI-compatible** (`llm.py`). Key đọc từ `.env`, xem mục 2.
- **Embedding:** `bkai-foundation-models/vietnamese-bi-encoder` (dẫn xuất từ PhoBERT, đã fine-tune cho retrieval).
  - **Không** dùng `vinai/phobert-base` gốc: đó là masked LM, không cho sentence embedding tốt.
  - Input phải tách từ bằng `pyvi`. Giới hạn 256 token; văn bản dài thì chia chunk rồi lấy trung bình.
  - Giữ backend `bge-m3` để so sánh ở Phase 4.
- **Giữ triết lý property graph** như bài gốc, không chuyển sang triple-store.
- **Luôn viết test trước khi tin số liệu.** Lỗi Ð/đ được phát hiện nhờ đối chiếu số lượng điều.
- **Văn bản hướng dẫn giữ nhãn "cần xác minh"** cho tới khi kiểm tra hiệu lực thật trên vbpl.vn.
- Người dùng trả lời **"không có ưu tiên"** cho cách lấy bản án. Claude tự chọn hướng hợp lý, có giải thích.

## 7. Phase 2 — đặc tả

**Mục tiêu:** biến bản án thô thành dữ liệu sạch cho Phase 3. Phase 2 không đụng tới graph.

**Đầu vào:**
- Bản án hình sự sơ thẩm của 26 tội nêu ở mục 1.
- Mục tiêu 50–200 án mỗi tội, tổng khoảng 1.000–3.000 án.
- Tối thiểu chấp nhận được: 20–30 án mỗi tội.

**Các bước:**
1. **Thu thập.**
2. **PDF/DOCX → text**, rồi tách 3 phần: *NỘI DUNG VỤ ÁN*, *NHẬN ĐỊNH CỦA TÒA ÁN*, *QUYẾT ĐỊNH*.
3. **Gắn nhãn** từ phần QUYẾT ĐỊNH: tội danh, điều/khoản/điểm, hình phạt. Dùng regex trước, không được thì dùng LLM.
4. **Chống lộ đáp án.** Xoá hoặc che các cụm như "phạm tội …", "theo Điều …" trong phần diễn biến, giống cách CAIL thay bằng `××`.
5. **Ẩn danh** tên, địa chỉ, CCCD, số tài khoản.
6. **Tách theo bị cáo.** Bài gốc dùng `CASE_SEG_PROMPT`.
7. **Chia tập** corpus/test theo từng tội, không có bản án nào nằm ở cả hai tập. Tham khảo `scripts/sample_source_cases.py` của bài gốc.

**Đầu ra:**
- `data/processed/cases_vn.json` (`vai_tro: "corpus"`)
- `data/processed/cases_vn_test.json` (`vai_tro: "test"`)

Mỗi bản ghi có dạng:
```json
{
  "id": "vn-000123",
  "bi_cao": "ẩn danh",
  "dien_bien": "... đã che đáp án, đã ẩn danh ...",
  "toi_danh": ["Tội trộm cắp tài sản"],
  "dieu_khoan": ["173.2.d"],
  "hinh_phat": {"tu_thang": 30},
  "nguon": "url hoặc tên dataset",
  "vai_tro": "corpus"
}
```

**Tiêu chí xong:**
- Đủ số lượng án tối thiểu cho từng tội.
- Mọi bản ghi đã qua bước 2–6.
- Kiểm tra bằng mắt ~20 bản ghi ngẫu nhiên: không còn lộ đáp án, không còn tên thật.
- Hai tập corpus và test không trùng nhau.

## 8. Nguồn dữ liệu bản án — đã tìm hiểu

1. **ViCSR** (SIGIR 2026, DOI 10.1145/3805712.3808526). Tác giả: Minh-Hien Nguyen, Khanh Huyen Nguyen, Tan-Minh Nguyen, Hoang-Quynh Le, Thi-Hai-Yen Vuong (VNU-UET).
   - Gồm 10.000 bản án hình sự tiếng Việt và 1.122 điều luật. Nhãn lấy từ điều luật mà bản án trích dẫn.
   - Rất khớp với dự án: nhãn chính là cạnh `RELATES_TO_LAW`, và bài toán của họ chính là phép đo Recall@k ở Phase 4. Họ có fine-tune một bi-encoder tiếng Việt, có thể thay embedding đang dùng.
   - **Trạng thái:** tác giả nói "sẽ public". Phiên cloud chưa tìm thấy link tải.
   - Việc cần làm: mở trang paper trên ACM DL tìm link, hoặc email tác giả (TS. Vương Thị Hải Yến, UET).
   - Khi có dữ liệu, kiểm tra:
     - Có text phần diễn biến không.
     - Có ẩn danh chưa.
     - License.
     - 1.122 điều gồm những luật nào. Con số này lớn hơn 426 điều của BLHS, có thể gồm BLTTHS hoặc nhãn ở cấp khoản.
     - Dùng phiên bản BLHS nào.
2. **`tmquan/anle-toaan-gov-vn`** (HuggingFace, Parquet), cào từ anle.toaan.gov.vn.
   - Gồm 1.963 văn bản, trong đó 1.155 có `doc_type == "ban_an"`.
   - Các trường: `case_type` (có `hinh_su`), `pdf_url`, `detail_url`, `doc_code`; nội dung là markdown chia section/paragraph.
   - Không đủ số lượng cho 26 tội. **Dùng làm mẫu thật** để viết parser và để bổ sung corpus.
3. **congbobanan.toaan.gov.vn**: nguồn chính nếu phải tự cào. Có bộ lọc loại án / cấp xét xử / tội danh; tải được file PDF.
4. **Link Google Drive người dùng gửi:** <https://drive.google.com/drive/folders/1heahFSQ2DR5IU6OTuZPsE4z36q7TeICw>. Phiên cloud không mở được nên chưa rõ nội dung. **Việc đầu tiên trên local: mở link này xem có gì.** Có thể là dữ liệu ViCSR hoặc bản án.

## 9. Việc tiếp theo, theo thứ tự

1. Xem nội dung thư mục Google Drive ở mục 8.4. Nếu là bản án hoặc ViCSR, bỏ qua bước cào.
2. Kiểm tra ViCSR đã public chưa (mục 8.1).
3. Tải `tmquan/anle-toaan-gov-vn`, lọc `case_type == "hinh_su"` và `doc_type == "ban_an"`, rồi thống kê số án cho từng tội trong 26 tội.
4. Dựa trên mẫu thật, viết `vn_legal_graph/cases/`:
   - `parse_judgment.py`: tách 3 phần, gắn nhãn.
   - `anonymize.py`.
   - `leak_filter.py`.
   - `split.py`.

   Kèm test bằng fixture, theo cùng cách làm với `parse_blhs.py`.
5. Nếu thiếu dữ liệu, viết scraper cho congbobanan, có giới hạn tốc độ.
6. Người dùng tự chạy `judge_dep` thật cho 26 điều. Claude rà lại chất lượng các câu hỏi "Có … không?".

## 10. Tham chiếu

- Sơ đồ tiến độ và kiến trúc truy vấn: <https://claude.ai/artifact/BMLpjNNcp1J9sgwnoyUX11> (artifact riêng của người dùng). Lưu ý: sơ đồ truy vấn trong đó còn thiếu 2 bước đã nêu ở mục 4.2.
- Repo gốc: <https://github.com/XMUDeepLIT/LegalGraphRAG>. Có thể clone ra một thư mục ngoài repo này để đọc. `docs/TABLE2_REPRODUCTION.md` của họ mô tả cách dựng corpus 14.049 án từ CAIL, JuDGE và CMDL.
