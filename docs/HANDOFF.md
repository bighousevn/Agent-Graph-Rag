# HANDOFF — LegalGraphRAG cho Bộ luật Hình sự Việt Nam

> **Gửi phiên Claude Code tiếp theo:** đọc hết file này trước khi làm gì.
> Nó thay cho toàn bộ cuộc trò chuyện trước, diễn ra trong một container cloud và không chuyển sang máy này được.
> Đừng hỏi lại những gì đã ghi ở mục "Quyết định đã chốt".

Cập nhật lần cuối: 2026-10-01 (phiên local, sau khi phân tích ViCSR). Nhánh: `claude/legalgraphrag-framework-hv23z3`.

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
pytest tests/ -v                                        # kỳ vọng: 35 passed
python scripts/build_law_layer.py --chapters XVI XX --dry-run
#   -> "26 Điều parsed, 26 are crime ('Tội ...') articles."
```

Phiên cloud trước đây bị proxy chặn nhiều trang. **Trên máy local (đã kiểm tra 2026-10-01):**
- `huggingface.co`, `drive.google.com`: vào được.
- `congbobanan.toaan.gov.vn`: HTTP vào được (302), nhưng **HTTPS lỗi chứng chỉ SSL** (chuỗi chứng chỉ không đầy đủ, thường gặp ở các trang `.gov.vn`). Scraper sau này phải xử lý chuyện này.

Khi cào dữ liệu phải giới hạn tốc độ (≥ 1–2 giây mỗi request) để tránh bị chặn IP.

Máy local không có `pip` cho Python hệ thống. Luôn dùng `.venv/bin/python`.

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
| **2** | **Thu thập và làm sạch bản án** | 🔄 **Đang ở đây.** Đã có dữ liệu ViCSR (mục 8.1) và loader `vn_legal_graph/cases/vicsr.py` gắn lại nhãn tội danh. **Chờ người dùng quyết định phạm vi** (mục 9), vì ViCSR gần như không có án Chương XVI |
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

### 8.1. ViCSR — ĐÃ CÓ (thư mục Google Drive người dùng gửi)

- Nguồn: <https://drive.google.com/drive/folders/1heahFSQ2DR5IU6OTuZPsE4z36q7TeICw> (thư mục "Dataset").
- Đã tải về `data/raw/dataset_drive/`. **Thư mục này nằm trong `.gitignore`, không commit** (150 MB).
- Paper: ViCSR, SIGIR 2026, DOI 10.1145/3805712.3808526. Tác giả: Minh-Hien Nguyen, Khanh Huyen Nguyen, Tan-Minh Nguyen, Hoang-Quynh Le, Thi-Hai-Yen Vuong (VNU-UET). License chưa rõ.

**Các file:**

| File | Kích thước | Nội dung |
|---|---|---|
| `law_shorten.txt` | 143 KB | 1.122 điều: số hiệu + tên điều |
| `law_detail.txt` | 1,7 MB | 1.122 điều: toàn văn |
| `case_sumary.txt` | 44 MB | 10.001 án: đoạn tóm tắt, phần lớn là "nội dung vụ án" |
| `case_detail.txt` | 150 MB | 10.001 án: toàn văn |
| `ground_truth.json` | 863 KB | `{case_id: [law_id, ...]}`, trung bình 7,1 nhãn mỗi án |

**Định dạng và ID:**
- Bản ghi có dạng `<id> [---] <text> [-/-]`.
- Văn bản viết thường. `case_detail` đã bị bỏ dấu câu. Tên người đã được viết tắt một phần (ví dụ "vũ tuấn t").
- Law ID 1–426 là BLHS (ID trùng số điều). 427–927 là BLTTHS (501 điều). 928–1122 là Luật Thi hành án hình sự.
- Hơn 90% án xử năm 2017–2019.

**Các vấn đề dữ liệu đã phát hiện (đã đối chiếu trên dữ liệu thật):**
1. **Lỗi ký tự `ð` thay cho `đ`**, y như file BLHS (2.244 chỗ trong `case_detail`). `normalize_text()` xử lý.
2. **Nhãn số trộn số điều BLHS 1999 và BLHS 2015.**
   - 755 án gắn "194", trong đó 750 án là án ma túy. Đây là Điều 194 của BLHS 1999. Ở BLHS 2015, Điều 194 là tội hàng giả là thuốc chữa bệnh.
   - Tương tự: 138 (trộm cắp, bản 1999), 136, 135, 139, và 46 (giảm nhẹ, bản 1999; 1.074 án).
3. **Nhãn gốc bị thổi phồng.** Có án gắn cả `[248, 249, 250, 251, 252]`, tức tội ma túy gộp của bản 1999 bị tách ra thành cả 4 điều. Vì vậy nhãn gốc của 248, 250, 252 **không đáng tin**.
4. **Cách đặt dấu thanh cũ "ma tuý"** (1.634 án) so với "ma túy" trong BLHS.
5. **Lỗi font cũ TCVN3/ABC** ở khoảng 500 án ("héi ång xđt xö", "ngµy"). Đôi khi chỉ hỏng phần tiêu đề.
6. **Án trùng:** 135 nhóm văn bản giống hệt nhau, thừa 140 án. 14 nhóm trong đó có nhãn gốc khác nhau. **Phải loại trùng trước khi chia corpus/test.**
7. Nhãn gốc đôi khi sai hẳn, ví dụ gắn 251 trong khi bản án không nhắc tới "mua bán".

**Loader `vn_legal_graph/cases/vicsr.py` gắn lại nhãn theo BLHS 2015 (`label_case`):**
- Cắt phần quyết định từ "vì các lẽ trên…". Không dùng chữ "quyết định" đứng một mình, vì nó cũng xuất hiện trong "khi quyết định hình phạt".
- Đọc câu tuyên án "tuyên bố bị cáo … phạm tội <tên>", kể cả câu có nhiều bị cáo, rồi khớp **tên tội dài nhất** với tên tội của BLHS 2015. Có chuẩn hoá dấu thanh, bảng tên rút gọn (247: "trồng cây thuốc phiện / cần sa"), và khớp gần đúng cho lỗi OCR.
- Phương án dự phòng: lấy số điều trong phần quyết định, chỉ chấp nhận khi tên tội theo BLHS 2015 của điều đó có xuất hiện trong bản án. Cách này tự loại số điều của bản 1999.
- Không dùng câu "về tội" (thường là tiền án hoặc tổng hợp hình phạt).

**Kết quả trên 10.001 án** (`.venv/bin/python scripts/vicsr_stats.py`, ghi ra `data/processed/vicsr_stats.json`):
- Gắn được nhãn 9.499 án: 8.307 qua câu tuyên án, 238 qua khớp gần đúng, 954 qua số điều có xác nhận tên. Không gắn được: 502 án.
- Khớp nhãn gốc: 8.095 án. Phần lệch chủ yếu đã giải thích được bằng các vấn đề 2, 3, 7 ở trên.
- Hạn chế đã biết, ngoài phạm vi: tên tội Điều 244 (động vật nguy cấp) bị khớp nhầm thành 410, vì `law_shorten` ghi tên 410 bị cụt ("vi phạm quy định về bảo vệ"), còn tên 244 trong đó là tên sau sửa đổi 2017.

**Số án cho 26 tội mục tiêu (nhãn mới, chưa loại trùng):**

| Chương | Điều | Số án |
|---|---|---|
| XX | 249 tàng trữ ma túy | **9.054** |
| XX | 251 mua bán ma túy | 150 |
| XX | 247 trồng cây có chất ma túy | 32 |
| XX | 248 sản xuất ma túy | 9 |
| XX | 250 vận chuyển ma túy | 5 |
| XX | 256, 258 | 3, 1 |
| XX | 252–255, 257, 259 | 0 |
| XVI | 173 trộm cắp | 51 |
| XVI | 175 lạm dụng tín nhiệm | 6 |
| XVI | 168, 174 | 1 mỗi điều |
| XVI | 169–172, 176–180 | 0 |

**Kết luận:** ViCSR lệch rất mạnh về Điều 249. **Chương XVI gần như không có dữ liệu**, trừ Điều 173. Nếu không cân bằng thì Recall@k vô nghĩa: luôn đoán 249 cũng đúng khoảng 95%. Bài gốc cũng gặp chuyện này và lấy mẫu theo từng tội (corpus tối đa 20 án/tội, test 5–10 án/tội).

### 8.2. `tmquan/anle-toaan-gov-vn` (HuggingFace)

- Cào từ anle.toaan.gov.vn: 1.963 văn bản, trong đó 1.155 là bản án.
- Có các trường `case_type` (gồm `hinh_su`), `pdf_url`, `detail_url`, `doc_code`. Nội dung là markdown.
- Chưa tải. Chỉ có ích nếu cần bổ sung án cho Chương XVI.

### 8.3. congbobanan.toaan.gov.vn

- Nguồn để tự cào nếu cần thêm án cho Chương XVI. Có bộ lọc loại án / cấp xét xử / tội danh, tải được PDF.
- Lưu ý lỗi SSL đã nêu ở mục 3.

## 9. Việc tiếp theo, theo thứ tự

1. **Người dùng chọn phạm vi** vì Chương XVI thiếu dữ liệu. Các phương án:
   - (a) Giữ 2 chương, cào thêm án Chương XVI từ congbobanan.
   - (b) Thu hẹp về các tội ViCSR có đủ án (249, 251, 247, 173) để chạy hết pipeline ngay, mở rộng sau.
   - (c) Kết hợp: làm (b) trước, (a) song song.
2. Viết `vn_legal_graph/cases/build_cases.py`:
   - loại trùng;
   - lấy phần diễn biến (`noi_dung` của `case_detail`; dự phòng bằng `case_sumary`);
   - che đáp án ("phạm tội …", "điều N" trong diễn biến);
   - lấy mẫu theo tội;
   - chia corpus/test không trùng nhau;
   - ghi `cases_vn.json` và `cases_vn_test.json` theo schema ở mục 7.

   Kèm test.
3. Người dùng tự chạy `judge_dep` thật cho 26 điều. Claude rà lại chất lượng các câu hỏi "Có … không?".

## 10. Tham chiếu

- Sơ đồ tiến độ và kiến trúc truy vấn: <https://claude.ai/artifact/BMLpjNNcp1J9sgwnoyUX11> (artifact riêng của người dùng). Lưu ý: sơ đồ truy vấn trong đó còn thiếu 2 bước đã nêu ở mục 4.2.
- Repo gốc: <https://github.com/XMUDeepLIT/LegalGraphRAG>. Có thể clone ra một thư mục ngoài repo này để đọc. `docs/TABLE2_REPRODUCTION.md` của họ mô tả cách dựng corpus 14.049 án từ CAIL, JuDGE và CMDL.
