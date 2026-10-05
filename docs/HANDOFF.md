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
pytest tests/ -v                                        # kỳ vọng: 108 passed
python scripts/build_law_layer.py --chapters XVI XX --dry-run
#   -> "26 Điều parsed, 26 are crime ('Tội ...') articles."
```

Phiên cloud trước đây bị proxy chặn nhiều trang. **Trên máy local (đã kiểm tra 2026-10-01):**
- `huggingface.co`, `drive.google.com`: vào được.
- `congbobanan.toaan.gov.vn`: HTTP vào được (302), nhưng **HTTPS lỗi chứng chỉ SSL** (chuỗi chứng chỉ không đầy đủ, thường gặp ở các trang `.gov.vn`). Scraper sau này phải xử lý chuyện này.

Khi cào dữ liệu phải giới hạn tốc độ (≥ 1–2 giây mỗi request) để tránh bị chặn IP.

Máy local không có `pip` cho Python hệ thống. Luôn dùng `.venv/bin/python`.

Máy local **không có GPU NVIDIA**. `torch` trong `.venv` là bản CPU, cài bằng:
```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Không cài bản CUDA mặc định (~2,5 GB, vô ích ở máy này).

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

1. **Chuẩn hoá đầu vào về cùng khuôn với corpus.**
   - Tách vụ án theo từng bị cáo (`segment_case_text_withname`).
   - `get_features()` trích 4 nhóm đặc trưng.
   - `concat_feature_descriptions()` ghép lại thành text, rồi mới embed. **Không embed câu hỏi thô.**
   - **Đính chính (đọc lại `scripts/prepare_case_features.py`, 2026-10-01):** prompt khi dựng corpus **có kèm tội danh** ("关键词需要围绕被告罪名展开": từ khoá xoay quanh tội danh), còn prompt lúc truy vấn chỉ có diễn biến. Hai bên cùng khuôn đầu ra, nhưng prompt không giống hệt. Bản Việt làm theo đúng như vậy; án test không bao giờ được gợi ý tội danh.
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
| 2 | Thu thập và làm sạch bản án | ✅ **Xong cho phạm vi thu hẹp** (mục 7.1): 257 án corpus + 40 án test, lấy từ ViCSR. Chương XVI (trừ Điều 173) chưa có dữ liệu |
| 3 | Tầng Case: trích đặc trưng, embedding, kNN, Louvain/Cluster, ráp graph hoàn chỉnh | ✅ **Xong** (2026-10-02). `outputs/hierargraph.pkl` có đủ 4 loại node. Còn vấn đề tóm tắt cụm (mục 9.2) |
| **4** | **Truy vấn (mục 4.2) và đo Recall@k của Điều luật và tội danh trên tập test** | 🔄 **Bước 1 xong:** baseline không dùng LLM (mục 9.3). Tiếp theo là các bước LLM |
| 5+ | Agent Researcher / Auditor / Adjudicator; mở rộng toàn bộ BLHS | Chưa làm |

### Kết quả đã kiểm chứng ở Phase 1 (file thật `data/raw/law/100_2015_QH13_296661.docx`)

- Parse được **426/426 Điều**. Trong đó **314 điều** có tiêu đề "Tội ...".
- Phần thứ hai có 4 điều không phải tội: **122, 352, 367, 392**. `is_crime_article` loại đúng 4 điều này.
- Chương XVI có đúng 13/13 điều (168–180). Chương XX có đúng 13/13 điều (247–259).
- **Lỗi dữ liệu đã xử lý:** một số đoạn dùng ký tự `Ð/ð` (Latin Eth, U+00D0/U+00F0) thay cho `Đ/đ` (U+0110/U+0111), ví dụ "Ðiều 317". Nếu không chuẩn hoá thì điều đó **mất mà không báo lỗi**. `normalize_text()` đã xử lý. Với file luật mới, luôn đối chiếu lại số lượng điều.
- **Dẫn chiếu chéo tìm được:**
  - Điều 172, 173, 174, 175 → Điều 168.
  - Điều 256 → Điều 255.
- **Cập nhật 2026-10-04: nguồn luật đã chuyển sang Văn bản hợp nhất** `data/raw/law/11_VBHN-VPQH_650257.docx` (người dùng cung cấp). Chi tiết ở mục 6. Các ghi chú bên dưới là về file bản gốc 2015.
- **Lưu ý về nguồn luật (cũ):** file `100_2015_QH13_296661.docx` là **Luật 100/2015/QH13 bản gốc**, chưa hợp nhất các lần sửa đổi:
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
- **Nguồn luật = Văn bản hợp nhất số 11/VBHN-VPQH** (`data/raw/law/11_VBHN-VPQH_650257.docx`, chọn ngày 2026-10-04).
  - Gộp Luật 12/2017/QH14 và Luật 59/2024/QH15; **chưa gồm Luật 86/2025/QH15**. Khớp với luật mà các án ViCSR (2017–2022) áp dụng.
  - **20/26 điều trong phạm vi khác bản gốc 2015.** Ví dụ: 173, 174, 176 chỉ còn giống khoảng 40–50%; 249 có tiêu chí tái phạm mới và thêm XLR-11, lá khát vào danh mục.
  - Chỉ đổi tên 1 tội trong phạm vi: Điều 259.
  - Parser đọc ra 407 điều. Các điều 33, 69, 90–107 đã bị bãi bỏ; có thêm Điều 217a.
  - File có 403 chú thích sửa đổi `[n]`. Parser bỏ ký hiệu `[n]` trong câu và dừng đọc ở phần thân chú thích cuối file; trước đó cả 403 chú thích bị dán vào Điều 426.
  - `judge_dep` sinh từ bản 2015 **phải chạy lại**.
- **Dữ liệu bản án = ViCSR. Phạm vi thu hẹp còn 4 tội: 249, 251, 173, 247** (người dùng chọn ngày 2026-10-01: thu hẹp trước, mở rộng sau). Nhãn tội danh lấy lại từ câu tuyên án theo BLHS 2015, không dùng nhãn số gốc của ViCSR.

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

### 7.1. Kết quả Phase 2 (2026-10-01)

**Phạm vi đã chốt với người dùng:** "thu hẹp trước, mở rộng sau". Chỉ dùng 4 tội ViCSR có đủ án: **249, 251, 173, 247**. Các tội còn lại của Chương XVI/XX bổ sung sau, bằng cách cào congbobanan.

**Lệnh chạy** (không gọi LLM, ~30 giây):
```bash
.venv/bin/python scripts/build_cases.py
# tuỳ chọn: --articles 249 251 173 247 --test-per-crime 10 --corpus-max-per-crime 100 --seed 42
```
Code: `vn_legal_graph/cases/build_cases.py`. Đầu ra nằm trong `data/processed/`, thư mục này không commit; muốn có lại thì chạy lại lệnh trên.

**Pipeline thực tế:**
- 10.001 án.
- Loại trùng: −140.
- Loại án phúc thẩm: −363. Lý do: phần nội dung của án phúc thẩm chép lại bản án sơ thẩm.
- Gắn nhãn (`label_case`), giữ án có mọi tội nằm trong phạm vi:
  - −428 không gắn được nhãn;
  - −251 án ngoài phạm vi;
  - −18 án có thêm tội ngoài phạm vi.
- Lấy diễn biến:
  - Dùng phần `noi_dung`. Nếu phần đó lỗi font hoặc ngắn hơn 40 từ thì dùng `case_sumary`. −112 án không có diễn biến dùng được.
  - Cắt trước "cáo trạng số" / "kiểm sát viên" / "đề nghị hội đồng xét xử".
  - Che "tội <tên>" thành "tội ××" và "điều N" thành "điều ××".
- Lấy mẫu theo tội, tội hiếm làm trước: mỗi tội 10 án test, tối đa 100 án corpus.

**Kết quả:**

| Tội | Corpus | Test | Ghi chú |
|---|---|---|---|
| 249 | 159 | 18 | vượt trần 100 vì có án mang cả nhãn 249 và 251, được lấy theo 251 |
| 251 | 100 | 12 | |
| 173 | 41 | 10 | |
| 247 | 16 | 10 | |

Diễn biến dài trung vị ~600 từ, tối đa 4.602 từ. Kiểm tra tự động: không còn "tội <tên>" hay "điều <số>" nào lọt lại.

**Schema bản ghi** (mở rộng so với mục 7):
`id`, `nguon` ("ViCSR#<id>"), `nam`, `bi_cao`, `dien_bien`, `dien_bien_nguon` (`noi_dung` | `tom_tat`), `toi_danh` (tên đầy đủ), `dieu` ([249]), `dieu_khoan` (["249.1.c"]), `hinh_phat`, `cach_gan_nhan`, `nhan_goc` (nhãn số gốc của ViCSR, để đối chiếu), `vai_tro`.

**Lệch so với đặc tả mục 7, có chủ đích:**
- **Không tách theo bị cáo.** Việc này cần LLM (`CASE_SEG_PROMPT`). Mỗi bản án là một bản ghi, mang mọi tội của nó.
- **Không ẩn danh thêm.** ViCSR đã được tòa công bố với tên viết tắt ("phùng thanh h").
- **`hinh_phat` để `None`**, chưa trích xuất.
- Diễn biến là chữ thường, không dấu câu (do ViCSR). Mô tả hành vi như "có hành vi tàng trữ trái phép chất ma túy" được giữ lại, vì đó là tình tiết chứ không phải đáp án.

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
1. **Lỗi ký tự `ð` thay cho `đ`**, y như file BLHS (2.244 chỗ trong `case_detail`), và **`ƣ` (U+01A3) thay cho `ư`** ("đƣợc", 23.907 chỗ). `normalize_text()` xử lý cả hai.
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

### 9.1. Người dùng chạy (cần key), theo đúng thứ tự

```bash
# (a) Đặc trưng án: ~400k token. Thử trước, đọc kết quả, rồi chạy hết.
.venv/bin/python scripts/extract_case_features.py   # ~447k token; in báo cáo kiểm tra ở cuối

# (b) Dựng graph (không LLM; lần đầu tải model ~540 MB, ~1 phút trên CPU)
.venv/bin/python scripts/build_graph.py

# (c) Tóm tắt cụm: ~9k token
.venv/bin/python scripts/summarize_clusters.py

# (d) Dựng lại để gắn node Cluster (embedding lấy từ cache, rất nhanh)
.venv/bin/python scripts/build_graph.py

# (e) Không bắt buộc ngay: judge_dep thật cho 26 điều (ghi đè bản để trống)
.venv/bin/python scripts/build_law_layer.py --chapters XVI XX
```

Claude được tự chạy các bước không cần key: `--dry-run`, `build_graph.py`, `build_cases.py`, `pytest`.

**Model và nhà cung cấp:**
- **Thứ tự chọn key:** `LLM_API_KEY` (kèm `LLM_PROVIDER`), rồi tới `DEEPSEEK_API_KEY`, rồi tới `OPENAI_API_KEY`.
- **Từ 2026-10-05 người dùng có `DEEPSEEK_API_KEY`, nên các lần chạy LLM mới dùng DeepSeek:**
  - base URL `https://api.deepseek.com`; **model mặc định `deepseek-flash`** (DeepSeek-V4.1-Flash, người dùng chọn); đổi sang `LLM_MODEL=deepseek-v4-pro` (DeepSeek-V4-Pro-0813) nếu cần.
  - Theo tài liệu DeepSeek tra ngày 2026-10-05: không còn `deepseek-chat` hay `deepseek-reasoner`; cả hai model đều bật thinking mặc định.
  - **Thinking được tắt** (`extra_body={"thinking": {"type": "disabled"}}`) trừ khi đặt `LLM_THINKING=enabled`. Lý do: các câu trả lời đều ngắn, thinking bỏ qua `temperature`, và tài liệu không nói `max_tokens` có tính cả phần suy nghĩ hay không.
- Đặc trưng án (Phase 3) và `judge_dep` đã chạy bằng `gpt-4o-mini`. Chỉ các lần chạy judge từ đây mới dùng DeepSeek.
- Kết quả `judge_retrieval.py` giờ ghi vào `outputs/judge_law_<mode>[_khong_huong_dan]_<model>.json`. Các file cũ đã đổi tên thành `..._gpt-4o-mini.json`.

**Quyền chạy:** bộ phân loại an toàn của Claude Code (chế độ auto) **chặn Claude chạy các lệnh gọi LLM**, dù người dùng đã bảo chạy. Người dùng tự chạy, hoặc thêm quy tắc `allow` cho đúng các script đó vào `.claude/settings.json`.

**Lần chạy thử 1 (2026-10-02, `--limit 5`, 10 án, đều là Điều 247) phát hiện LLM bịa thông tin:**
- "không thành khẩn khai báo" ở 6/10 án trong khi diễn biến không hề nói tới việc khai báo. Phần diễn biến bị cắt trước đoạn tại phiên tòa, nên LLM tự đoán.
- "giá trị lớn" chép từ câu ví dụ trong prompt.
- Có chi tiết cá nhân lọt vào ("sinh năm 1968", "có vợ").

→ Đã thêm vào `GET_CASE_FEATURES_PROMPT` một mục "Nguyên tắc bắt buộc": chỉ trích những gì văn bản nêu, không nêu thì để mảng rỗng. **Đây là chỗ khác với prompt gốc.** Cần chạy thử lại (`--per-crime 2`) và đọc kết quả trước khi chạy toàn bộ.

**Sửa bộ cắt diễn biến sau lần thử 1:**
- Cắt tại marker **sớm nhất** trong mọi mẫu. Trước đó code dừng ở mẫu đầu tiên có khớp, nên án 362 giữ lại câu "đại diện viện kiểm sát … truy tố".
- Thêm mẫu "viện kiểm sát … truy tố … về tội".
- Che cả tên tội bị vỡ chữ do OCR, bằng bộ khớp gần đúng.
- Còn 4/297 bản ghi giữ câu truy tố, nhưng tên tội và số điều đã được che: 517, 653, 2625 có câu này nằm trong 40 từ đầu; 4217 có chữ bị vỡ.

**Lần chạy thử 2 (`--per-crime 2`, 32 án), sau khi thêm "Nguyên tắc bắt buộc":**
- "không thành khẩn khai báo" bịa ra: từ 6/10 xuống 0. Thái độ không có trong văn bản: còn 1/32.
- Nhưng **"giá trị lớn" xuất hiện ở 20/32 án**, phần lớn là án ma túy; 9 án trong đó văn bản không hề nêu giá trị. Thêm 3 án có chi tiết cá nhân, trong đó 1 án lộ cả họ tên người thân.
- Nguyên nhân: cả phần ví dụ lẫn phần giải thích trong prompt đều gợi ý "giá trị lớn". Bài gốc làm vậy vì "数额较大" là thuật ngữ pháp lý của Trung Quốc, còn BLHS Việt Nam định khung theo khoảng giá trị (Điều 173) hoặc khối lượng ma túy (Điều 249).
- → **Prompt v3:**
  - ghi giá trị hoặc khối lượng **theo khoảng, đúng như văn bản nêu**;
  - cấm các cụm chung chung như "giá trị lớn";
  - cấm ghi tên người;
  - bỏ "giá trị lớn" và "tự thú" khỏi phần ví dụ.
- Thêm `audit_features()` và `extract_case_features.py --audit` (không gọi LLM) để gắn cờ các đặc trưng không có căn cứ trong văn bản. Đây là bộ lọc theo từ khoá, chỉ dùng để chỉ ra án cần xem lại.

**Lần chạy thử 3 (prompt v3, 32 án):**
- Bộ kiểm tra tự động: corpus 0 cờ; test có 3 án bị cờ `chi_tiet_ca_nhan` ("sinh năm …"). Không còn "giá trị lớn".
- **Đối chiếu tay khối lượng ma túy với văn bản:** khớp ở 13/13 án kiểm tra, kể cả các con số tổng cộng (2078: 0,155 + 0,038 = 0,193 gam).
- **Giá trị tài sản:** LLM tự đặt ranh giới khoảng (6150: "3 đến dưới 15 triệu" cho giá trị thật 13 triệu); có án lẫn giá mua ma túy với giá trị tài sản (440).
- → **Prompt v4:** ghi đúng con số văn bản nêu, không tự đặt khoảng.
- → `sanitize_features()` **lọc bằng code** chi tiết cá nhân (sinh năm, quan hệ gia đình) và biển số xe sau khi LLM trả về. Cách này chắc hơn sửa prompt thêm.
- → Kết luận: đủ tốt để chạy toàn bộ (~447k token). Sau khi chạy xong, đọc báo cáo kiểm tra và lấy mẫu vài án để đọc tay.

**Chạy toàn bộ (2026-10-02, prompt v4, `gpt-4o-mini`):**
- 297/297 án đọc được JSON, 0 án thiếu hành vi phạm tội.
- Bộ kiểm tra tự động chỉ gắn cờ 1 án (4360). Đó là lỗi diễn đạt: "tài sản trị giá 0,8428 gam methamphetamine", con số có thật trong văn bản.
- Đọc tay 5 án ngẫu nhiên: đều đúng trọng tâm.
- Mô tả đặc trưng dài trung vị 47 từ (min 8, max 102).

**Graph dựng từ đặc trưng thật** (`outputs/hierargraph.pkl`, 38 giây trên CPU):
- 26 Law, 26 Crime, 257 Case; 316 cạnh `RELATES_TO_LAW`, 771 cạnh `SIMILAR_TO`.
- **Chất lượng kNN:** 92,2% cạnh `SIMILAR_TO` nối 2 án có chung tội, so với 53,1% nếu ghép ngẫu nhiên. Theo tội: 247 đạt 100%, 173 đạt 97%, 249 đạt 94%, 251 đạt 90%.
- **10 cụm Louvain** (11–42 án):
  - Cụm 9 thuần Điều 247 (16/16 án).
  - Cụm 6 phần lớn là Điều 173 (35/37), cộng các án vừa trộm cắp vừa ma túy.
  - Án 249 và 251 chia ra 7 cụm trộn lẫn nhau. Điều này dễ hiểu, vì hành vi gần nhau và có 59 án mang cả hai nhãn.

**Tóm tắt cụm (người dùng chạy, 10 cụm, ~8k token) — vấn đề cần xử lý ở Phase 4:**
- 8/10 cụm nhận tóm tắt gần như giống hệt nhau ("Nhóm hành vi phạm tội: Tội phạm liên quan đến (chất) ma túy"). Cosine giữa các node Cluster của 8 cụm ma túy là 0,92–1,0 (trung bình 0,958).
- Hai cụm còn lại có tóm tắt riêng: cụm 6 "chiếm đoạt tài sản và tàng trữ trái phép chất ma túy", cụm 9 "trồng cây chứa chất ma túy".
- Nguyên nhân: prompt gốc yêu cầu "một loại khái quát ở mức cao". Với CAIL có hàng trăm tội, cách này phân biệt được các cụm. Ở phạm vi 4 tội, trong đó 3 tội là ma túy, các bản tóm tắt bị dồn về cùng một câu.
- **Hệ quả:** nhánh `top_retrieve` (truy vấn → Cluster → Case) gần như không chọn được cụm ma túy nào cụ thể. Gánh nặng dồn hết sang nhánh `direct_retrieve`.
- **Đề xuất cho Phase 4:**
  - Giữ bản tóm tắt theo đúng bài gốc làm baseline.
  - Thêm một biến thể "nêu đặc điểm phân biệt cụm này với các cụm khác" làm ablation, rồi đo xem có cải thiện không.
  - Chi phí thêm khoảng 8k token.

### 9.2. Phase 3 — các quyết định và kết quả chạy thử

- **Code:** `vn_legal_graph/graph/graph_db.py` (class `HierarGraph`), `vn_legal_graph/graph/build.py`, CLI `scripts/build_graph.py` và `scripts/summarize_clusters.py`.
- **ID node** cố định theo loại: `law:249`, `crime:249`, `case:vicsr-123`, `cluster:3`.
- **Embedding:** một ma trận chuẩn hoá cho mỗi loại node; tìm kiếm bằng phép nhân ma trận.
- **Cache embedding** ở `.cache/emb/`, khoá theo (model, văn bản).
- **Giữ y như bài gốc:**
  - bỏ các án không có "hành vi phạm tội";
  - chọn đại diện cụm theo 0,7·PageRank + 0,3·degree (trong thực tế degree lấn át);
  - chạy Louvain và PageRank trên toàn graph (gồm cả Law và Crime);
  - degree tính trên multigraph có hướng.
- **An toàn dữ liệu:**
  - `build_base_graph` từ chối án có `vai_tro != "corpus"`.
  - Bản tóm tắt cụm chỉ được gắn khi đầu vào khớp nguyên văn, để tránh gắn nhầm bản tóm tắt cũ sau khi graph thay đổi.
- **Model embedding** (đã đọc model card): PhoBERT-base-v2, 768 chiều, Apache-2.0, có huấn luyện trên Zalo Legal 2021. Đầu vào phải tách từ trước (đã làm bằng `pyvi`).
  - Đoạn 180 từ của các điều luật dài tối đa 216 token, dưới giới hạn 256, nên không bị cắt ngầm.
  - Kiểm tra thô: câu về ma túy gần Điều 249 hơn Điều 173 (0,42 so với 0,24), câu về trộm cắp thì ngược lại.
- **Chạy thử với đặc trưng giả** (40 từ đầu của diễn biến, chỉ để kiểm tra code):
  - 309 embedding; 26 Law, 26 Crime, 257 Case;
  - 316 cạnh `RELATES_TO_LAW` (59 án có 2 tội), 771 cạnh `SIMILAR_TO`;
  - 12 cụm; sau khi gắn tóm tắt giả thì có 12 node Cluster và 257 cạnh `BELONGS_TO`.
  - **Số cụm và chất lượng cụm phải xem lại khi có đặc trưng thật.**

### 9.3. Phase 4 — bước 1: baseline không dùng LLM (2026-10-02)

**Lệnh** (không LLM, ~1 phút): `.venv/bin/python scripts/evaluate_retrieval.py` → `outputs/retrieval_eval.json`.
**Code:** `vn_legal_graph/retrieval/search.py`, `vn_legal_graph/retrieval/metrics.py`.

**Thiết lập:**
- **Truy vấn:** đặc trưng LLM của án test (không gợi ý tội danh), embed cùng model và cùng định dạng với node Case.
- **Lấy án:** top-5 án. Điều luật xếp theo thứ tự hạng của án, như bài gốc.
- **Đi qua cụm:** lấy top-2 cụm theo cosine. Bài gốc dùng LLM để xếp hạng lại cụm.
- **Dữ liệu test:** 40 án; theo điều: 173: 10, 247: 10, 249: 18, 251: 12. Có 8 án nhiều tội.

| Cách | R@1 | R@2 | R@3 | Hit@1 | R@1: 173 / 247 / 249 / 251 |
|---|---|---|---|---|---|
| Tần suất (không nhìn án) | 0,34 | 0,57 | 0,75 | 0,45 | 0 / 0 / 1,00 / 0 |
| **Tìm án trực tiếp** | **0,79** | **0,96** | 0,97 | **0,90** | 0,70 / 1,00 / 0,56 / 0,75 |
| Đi qua cụm | 0,72 | 0,96 | 1,00 | 0,82 | 0,70 / 1,00 / 0,61 / 0,42 |
| So thẳng với điều luật | 0,44 | 0,45 | 0,64 | 0,45 | 0,40 / 1,00 / 0,06 / 0,25 |
| Tìm án trực tiếp, diễn biến thô (bỏ trích đặc trưng) | 0,69 | 0,96 | 0,99 | 0,80 | 0,70 / 1,00 / 0,56 / 0,42 |

**Đọc kết quả:**
- **Mọi lỗi top-1 đều là nhầm giữa 249 (tàng trữ) và 251 (mua bán).**
  - Tìm án trực tiếp: 4 án đúng là 249 bị đoán thành 251.
  - Đi qua cụm: lỗi theo cả hai chiều (249→251 4 án, 251→249 3 án).
  - Ở án một tội, tìm án trực tiếp đúng top-1 28/32. Ở án nhiều tội, top-1 luôn thuộc tập điều đúng.
- **Bước trích đặc trưng có tác dụng:** R@1 tăng từ 0,69 lên 0,79, chủ yếu nhờ Điều 251 (0,42 lên 0,75).
- **Đi qua cụm kém hơn tìm trực tiếp**, đúng như dự đoán: 8 bản tóm tắt cụm ma túy gần như giống hệt nhau (mục 9.2).
- **So thẳng với điều luật rất kém ở 249** (0,06). Đó là lý do bài gốc đi qua án tương tự chứ không so trực tiếp với điều luật.
- **Cảnh báo:**
  - Tập test nhỏ (40 án); một án đổi kết quả là R@1 của một điều đổi 6–10 điểm.
  - Nhánh đi qua án chỉ trả về được 4 điều có án trong corpus. R@3 gần 1 phần lớn là do không gian ứng viên hẹp. **R@1 mới là chỉ số có ý nghĩa.**

### 9.4. Phase 4 — bước 2: judge_law (code xong, chờ người dùng chạy)

**`judge_dep` thật, lần chạy đầu (người dùng, 2026-10-02):**
- 23/26 điều có câu hỏi, chất lượng tốt. Ví dụ Điều 173 có đủ các mốc 2 triệu và 50 triệu cùng các trường hợp tái phạm.
- **Điều 249, 250, 252 bị cắt cụt** ở giới hạn 1.024 token (~3.180 ký tự): điều luật ma túy sinh ra 30–40 câu hỏi.
- Đã sửa (commit `19aa1e1`):
  - `judge_dep` được trả lời tới 4.096 token.
  - **`llm.py` không cache câu trả lời bị cắt** (`finish_reason == "length"`). Trước đó bản bị cắt nằm trong cache và lần chạy lại sẽ nhận lại đúng bản đó.
  - **→ Người dùng cần chạy lại** `build_law_layer.py --chapters XVI XX`. Cả 26 điều được gọi lại vì khoá cache đã đổi; tốn rất ít.

**Code:**
- `vn_legal_graph/judge/judge_law.py` và các prompt `JUDGE_*` trong `prompts/vi.py`, port từ `core/judge/judge_law.py` của bài gốc.
- CLI `scripts/judge_retrieval.py`. Đầu vào là xếp hạng của `outputs/retrieval_eval.json`.

**Cách chạy:**
- Xét top-3 điều của cách "tìm án trực tiếp".
- Xếp lại danh sách: điều được chấp nhận lên trước, điều bị bác xuống sau; thứ tự trong mỗi nhóm giữ nguyên.
- So Recall@k trước và sau. Báo thêm bảng chéo: điều đúng/sai (theo nhãn) × chấp nhận/bác.

**Hai chế độ:**
- `trung-thanh`: đúng bài gốc, 1 lần gọi cho mỗi yếu tố `judge_dep`, cộng 1 lần quyết định cuối.
- `gop`: biến thể, hỏi mọi yếu tố của một điều trong 1 lần gọi, trả về mảng JSON.

**Ước lượng chi phí** (40 án, 72 lần xét, khi chưa có `judge_dep` của 249):
- `trung-thanh`: 1.208 lần gọi, ~2,5 triệu token.
- `gop`: 116 lần gọi, ~0,26 triệu token.

**Khác bài gốc:**
- Câu trả lời yếu tố không phải true/false được giữ riêng là "không rõ". Bài gốc coi là "không true".
- Văn bản vụ án là `dien_bien`, cắt ở 6.000 ký tự.

**Lệnh cho người dùng:**
```bash
.venv/bin/python scripts/build_law_layer.py --chapters XVI XX     # sửa judge_dep 249/250/252
.venv/bin/python scripts/judge_retrieval.py --dry-run             # xem lại chi phí
.venv/bin/python scripts/judge_retrieval.py --mode gop --per-crime 2
```

**Lần chạy lại `judge_dep` (2026-10-04):** đủ 26 điều; 249: 41, 250: 45, 252: 43 câu hỏi.

**Lần thử `judge_law --mode gop --per-crime 2` (16 án, 31 lần xét) — lỗi định dạng:**
- R@1 không đổi (0,67 → 0,67), vì **22/31 câu trả lời gộp không đọc được**. `gpt-4o-mini` trả đúng dạng JSON nhưng **đếm sai số phần tử** (43/41, 35/36, 34/36), mà code đòi đúng số lượng nên bỏ cả câu trả lời.
- Hậu quả: LLM ra kết luận cuối gần như mù, và 8 điều đúng bị bác.
- Ở 9 lần đọc được: 8/9 điều đúng được chấp nhận.
- → **Sửa:** prompt gộp trả về object có khoá theo số thứ tự (`{"1": true, ...}`); parser `parse_numbered_bools` coi khoá thiếu là "không rõ" thay vì bỏ cả câu trả lời. Cần chạy thử lại.
- Lưu ý: R@1 trước khi xét của mẫu 16 án này là 0,67, thấp hơn 0,79 của cả 40 án, vì `--per-crime` lấy mẫu theo tổ hợp nhãn nên có nhiều án nhiều tội hơn.

**Lần thử 2 (định dạng có số thứ tự):**
- 0 lỗi đọc, 0/1.050 yếu tố "không rõ".
- Nhưng **R@1 giảm từ 0,67 xuống 0,60** (1 án bị hỏng: 2078), và **8/23 điều đúng bị bác**.
- Ở cả 8 lần bác sai, yếu tố cơ bản ("Có tàng trữ trái phép chất ma túy không?") được trả lời **true**, nhưng 37–39 câu về tình tiết định khung tăng nặng là false, nên bước kết luận cuối ra "không áp dụng".
- → **Sửa `JUDGE_LAW_FINAL_PROMPT`:** điều luật áp dụng khi thoả yếu tố cấu thành cơ bản (khoản 1); tình tiết định khung không bắt buộc. **Đây là chỗ khác prompt gốc.**
- Phần trả lời từng yếu tố đã có trong cache, nên chạy lại chỉ tốn 31 lần gọi kết luận.

**Lần thử 3 (sau khi sửa prompt kết luận):**
- R@1 0,67 → 0,67 (không đổi).
- Điều đúng: chấp nhận 18 / bác 5. **Điều sai: chấp nhận 6 / bác 2**, tức là judge_law gần như chấp nhận mọi thứ.
- **Không phân biệt được 249 và 251**, nơi tập trung mọi lỗi của retrieval. Hai nguyên nhân thấy được trong các câu trả lời:
  1. Án mua ma túy về để tự dùng (1385, 1463; nhãn 249) vẫn được trả lời true cho "Có mua bán trái phép chất ma túy không?". LLM coi "mua để dùng" là "mua bán".
  2. Án 783 (nhãn 251): yếu tố "Có nhằm mục đích mua bán … không?" của Điều 249 được trả lời **true**, nhưng bước kết luận vẫn chấp nhận 249. Câu hỏi đã mất đi ý phủ định của điều luật ("**mà không** nhằm mục đích mua bán …").
- Điều này gợi ý rằng thứ thiếu là **kiến thức hướng dẫn áp dụng** (phân biệt tàng trữ và mua bán theo mục đích). Ở bài gốc, phần này đến từ `related_laws` (giải thích tư pháp). Ở đây `guidance_links.json` vẫn còn trống.
- **Cần người dùng quyết định hướng đi tiếp** (xem các phương án trong cuộc trò chuyện).

**Nguồn luật mới (2026-10-04), xem mục 6:**
- Phải chạy lại `build_law_layer.py --chapters XVI XX` (LLM, người dùng chạy), vì nội dung điều luật đã đổi.
- Sau đó Claude chạy lại `build_graph.py` (embedding của node Law đổi; node Case dùng lại cache) và `evaluate_retrieval.py`.
- Cách "so thẳng với điều luật" sẽ đổi kết quả; cách "tìm án trực tiếp" thì không.

**Câu hỏi còn mở về phân biệt 249 và 251** (tra cứu ngày 2026-10-04):
- **Không có văn bản hướng dẫn nào đang có hiệu lực** cho BLHS 2015 về định nghĩa "mua bán" và "tàng trữ" chất ma túy.
  - TTLT 17/2007 (sửa bởi TTLT 08/2015) hướng dẫn BLHS 1999, đã hết hiệu lực nhưng vẫn được tham khảo trong thực tiễn.
  - Nghị quyết HĐTP về tội phạm ma túy vẫn ở dạng dự thảo; có góp ý tới tháng 11/2024.
  - Tạp chí Kiểm sát viết: "chưa có hướng dẫn cụ thể đối với các tội phạm về ma túy của BLHS năm 2015".
- **Người dùng đã chốt (2026-10-04):** dùng câu chữ Điều 249 cộng với định nghĩa "mua bán" và "tàng trữ" của TTLT 17/2007, gắn nhãn "hướng dẫn BLHS 1999, chỉ tham khảo".
  - **Đã có file**, do người dùng tải về: `data/raw/guidance/17_2007_TTLT-BCA-VKSNDTC-TANDTC-BTP_m_61683.docx` và `08_2015_TTLT-BCA-VKSNDTC-TANDTC-BTP_m_295981.docx`.
    - Lần tải đầu, người dùng lấy nhầm `01_2017_TTLT-VKSNDTC-TANDTC-BCA-BTP` (văn bản về giám định tư pháp vụ án tham nhũng, kinh tế). File này **không dùng**, vẫn nằm untracked trong `data/raw/law/`.
  - **`vn_legal_graph/law/guidance_ttlt.py`** trích **nguyên văn theo số mục** phần II mục 3 (Điều 194 của BLHS 1999) vào `data/raw/guidance/guidance_links.json`:
    - 3.1 (tàng trữ, "mà không nhằm mục đích mua bán…") → Điều 249;
    - 3.2 (vận chuyển; giữ hộ mà biết mục đích mua bán thì là đồng phạm mua bán) → Điều 250, 251;
    - 3.3 (7 hành vi "mua bán", đều "nhằm bán") → Điều 251, 249;
    - 3.4 (chiếm đoạt) → Điều 252;
    - 3.7 điểm c (mua hộ ma túy để sử dụng) → Điều 249, 251; điểm d → Điều 249, 250.
  - **TTLT 08/2015 không sửa các mục trên.** Nó chỉ sửa mục I.1.1, I.1.4 và bãi bỏ điểm đ mục 3.7; điểm đ không được dùng.
  - Trường `from` của mỗi bản ghi đều ghi rõ: "hướng dẫn BLHS 1999 Điều 194, nay tương ứng Điều 249–252 BLHS 2015; văn bản đã hết hiệu lực, chỉ dùng tham khảo".
  - **Đã gắn vào tầng Law** bằng cách chạy `link_guidance` và `build_law_crime` trực tiếp, không gọi LLM.
  - **Sửa lỗi:** phần tài liệu bổ trợ của Điều 249 mở đầu bằng toàn văn Điều 248 (điều được dẫn chiếu, ~2.000 ký tự), nên khi cắt ở 2.000 ký tự thì mất hết phần TTLT. `render_related` giờ đặt văn bản hướng dẫn lên trước và cắt ở 6.000 ký tự.
  - Thêm `judge_retrieval.py --no-guidance` (bỏ phần TTLT) để so sánh có và không có hướng dẫn.
  - **Đã làm (từ câu chữ Điều 249):** thêm quy tắc 4 vào `JUDGE_LAW_FINAL_PROMPT`. Nếu điều luật có điều kiện loại trừ và vụ án thoả đúng yếu tố bị loại trừ, thì điều luật không áp dụng.
- **Sau khi đổi sang nguồn luật mới (Claude đã chạy):**
  - `judge_dep` mới đủ 26 điều (249: 42 câu hỏi, có XLR-11).
  - Graph dựng lại; 21 embedding phải tính mới.
  - Retrieval: các cách đi qua án không đổi (tìm trực tiếp R@1 0,79). "So thẳng với điều luật": R@1 0,44 → 0,45, R@2 0,45 → 0,65.
- **Ablation đề xuất:** chạy `judge_law --mode gop --per-crime 2` (a) chỉ với quy tắc loại trừ, (b) sau khi thêm TTLT, rồi so sánh.

**Lần thử 4 (có quy tắc loại trừ; có và không có TTLT, `--per-crime 2`, 16 án):**
- Cả hai cấu hình đều cho R@1 0,67 → 0,67. Các thay đổi ở top-1 chỉ xảy ra trong án nhiều tội và vẫn nằm trong tập điều đúng.
- Quy tắc loại trừ làm judge chặt hơn: điều sai được chấp nhận giảm từ 6 xuống 1, nhưng điều đúng bị bác tăng từ 5 lên 10–11.
- **TTLT không tạo khác biệt đo được** (13 so với 12 điều đúng được chấp nhận).
- **9/11 điều đúng bị bác thuộc án nhiều tội**, chủ yếu là 249 trong án có cả 249 và 251. Nguyên nhân là **không tách theo bị cáo**: bài gốc chạy `segment_case_text_withname` (mỗi bị cáo một mô tả) trước khi judge, còn ở đây judge xét cả bản án. Bị cáo A tàng trữ, bị cáo B mua bán, nên quy tắc loại trừ bác nhầm 249.
- **Mẫu `--per-crime 2` không phù hợp để đo sửa lỗi 249/251:** trong 4 án sai top-1 trên cả 40 án (1463, 2119, 2196, 8265; đều là 249 bị đoán thành 251), mẫu chỉ có 1463.
- → Bước tiếp theo: chạy judge trên đủ 40 án, và/hoặc port bước tách theo bị cáo.

**Chạy đủ 40 án (`judge_law --mode gop`, có TTLT, `gpt-4o-mini`, 2026-10-05):**

| | trước khi xét | sau khi xét |
|---|---|---|
| R@1 | 0,792 | **0,767** |
| R@2 | 0,963 | 0,963 |
| Hit@1 | 0,900 | 0,875 |
| R@1 theo điều 173 / 247 / 249 / 251 | 0,70 / 1,00 / 0,56 / 0,75 | 0,80 / 1,00 / **0,39** / 0,83 |

- **Không sửa được án nào trong 4 án sai top-1** (1463, 2119, 2196, 8265). **Làm hỏng thêm 1 án** (3110: 249 đang đúng, sau khi xét thì 251 lên đầu).
- **Chẩn đoán: bước trả lời từng yếu tố làm đúng, bước kết luận cuối làm sai.**
  - Ở cả 4 án sai và án 3110, các yếu tố của 249 đều đúng như văn bản: "Có tàng trữ…" là true; "Heroine/Methamphetamine… từ 0,1 gam đến dưới 05 gam" là true; "Có nhằm mục đích mua bán…" là false; các tình tiết tăng nặng là false.
  - Dù vậy `JUDGE_LAW_FINAL_PROMPT` vẫn trả "false": 2 yếu tố đúng giữa 40 yếu tố sai bị đọc thành "không áp dụng", kể cả khi prompt đã dặn tình tiết định khung không bắt buộc.
  - Lỗi ở mức yếu tố còn lại: án 1463 và 2196 vẫn chấp nhận 251. LLM coi "mua về để sử dụng" là "mua bán", dù có TTLT mục 3.3.
- Tổng hợp quyết định, chia theo án một tội / án nhiều tội:
  - án một tội: điều đúng nhận 24 / bác 8, điều sai nhận 5 / bác 18;
  - án nhiều tội: điều đúng nhận 6 / bác 9, điều sai bác 2.
- **Kết luận tạm:** với `gpt-4o-mini`, bước judge_law như đã port không cải thiện truy xuất. Điểm yếu chính là lời gọi tổng hợp cuối.
- **Các hướng có thể làm** (chờ người dùng chọn):
  - (a) Thay lời gọi tổng hợp bằng luật trong code, dựa trên nhãn của từng câu hỏi `judge_dep` (cơ bản / định khung / loại trừ). Khác bài gốc.
  - (b) Dùng model mạnh hơn cho judge (đổi `LLM_MODEL`), như một ablation.
  - (c) Ghi nhận kết quả và chuyển sang các bước khác (rerank, `retrieve_law`, tách theo bị cáo).

**Chạy đủ 40 án với `deepseek-flash` (gop, có TTLT, thinking tắt, 2026-10-05) — KẾT QUẢ CHÍNH:**

| | chỉ retrieval | + judge `gpt-4o-mini` | **+ judge `deepseek-flash`** |
|---|---|---|---|
| R@1 | 0,792 | 0,767 | **0,892** |
| R@2 | 0,963 | 0,963 | 0,963 |
| Hit@1 | 0,900 | 0,875 | **1,000** (40/40) |
| R@1 điều 249 | 0,56 | 0,39 | **0,78** |
| điều sai được chấp nhận | — | 5/25 | **0/25** |
| điều đúng bị bác | — | 17/47 | **6/47** |

- **Sửa được cả 4 án sai top-1** (1463, 2119, 2196, 8265): bác 251, chấp nhận 249. Không làm hỏng án nào.
- 6 điều đúng bị bác: 5 nằm trong án nhiều tội (do chưa tách theo bị cáo), 1 trong án một tội (2078, Điều 251; top-1 vẫn đúng).
- R@1 riêng điều 173 giảm 0,70 → 0,60 chỉ vì cách đo: ở án 10971 (3 tội), 251 lên đầu thay 173. Hit@1 vẫn 40/40.
- **Kết luận:** lỗi ở bước tổng hợp cuối của `gpt-4o-mini` là lỗi **năng lực model**, không phải lỗi thiết kế. Với `deepseek-flash`, judge_law như bài gốc (cộng 2 quy tắc đã thêm và TTLT) cải thiện R@1 thêm 10 điểm.
- **Lưu ý:** chỉ có 40 án test, 18 trong số đó là Điều 249; một án đổi kết quả làm R@1 tổng đổi 2,5 điểm.
- Các phần khác của pipeline (đặc trưng, `judge_dep`) vẫn chạy bằng `gpt-4o-mini`.

### 9.5. Sau đó

1. Port bước LLM xếp hạng lại án và cụm (`RERANK_*`), cùng nhánh `retrieve_law` (LLM đoán tên tội → Crime node → Law).
2. Ablation tóm tắt cụm theo kiểu "nêu đặc điểm phân biệt" (mục 9.2).
3. **Mở rộng:** cào congbobanan cho Chương XVI và các tội ma túy hiếm. Lưu ý lỗi SSL ở mục 3.

## 10. Tham chiếu

- Sơ đồ tiến độ và kiến trúc truy vấn: <https://claude.ai/artifact/BMLpjNNcp1J9sgwnoyUX11> (artifact riêng của người dùng). Lưu ý: sơ đồ truy vấn trong đó còn thiếu 2 bước đã nêu ở mục 4.2.
- Repo gốc: <https://github.com/XMUDeepLIT/LegalGraphRAG>. Có thể clone ra một thư mục ngoài repo này để đọc. `docs/TABLE2_REPRODUCTION.md` của họ mô tả cách dựng corpus 14.049 án từ CAIL, JuDGE và CMDL.
