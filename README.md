# Agent-Graph-Rag

Áp dụng phương pháp **LegalGraphRAG** (ACL 2026, [XMUDeepLIT/LegalGraphRAG](https://github.com/XMUDeepLIT/LegalGraphRAG)) cho **Bộ luật Hình sự Việt Nam**.

Giai đoạn hiện tại **chỉ xây dựng HierarGraph (3 tầng: Crime – Law – Case)**, chưa triển khai các agent Researcher/Auditor/Adjudicator của bài báo gốc. Kế hoạch chi tiết và các quyết định thiết kế nằm trong lịch sử làm việc của phiên Claude Code; tóm tắt lại dưới đây.

## HierarGraph là gì

Đây không phải 3 graph tách biệt, mà là **một graph có 3 tầng node** (dựa trên `core/graph_construct/` của bài gốc):

| Tầng | Node | Nội dung |
|---|---|---|
| Crime (tội danh) | `Crimes` | Tên tội, lấy từ tiêu đề Điều |
| Law (điều luật) | `Laws` | Nội dung Điều, `judge_dep` (câu hỏi "Có … không?" do LLM tách từ yếu tố cấu thành), `related_laws` (dẫn chiếu + văn bản hướng dẫn) |
| Case (bản án) | `Cases` + `Cluster` | Đặc trưng vụ án do LLM trích, cụm án tương tự do LLM tóm tắt |

Cạnh: `RELATES_TO_LAW` (Case→Law), `RELATED_CRIME` (Law→Crime), `SIMILAR_TO` (Case↔Case, kNN), `BELONGS_TO` (Case→Cluster).

## Trạng thái hiện tại: Phase 0–1 (tầng Law + Crime)

Đã hoàn thành:
- Parser tách BLHS từ file `.docx` gốc thành cây Phần → Chương → Mục → Điều → Khoản → Điểm.
- Sinh `judge_dep` cho từng Điều bằng LLM.
- Liên kết dẫn chiếu giữa các Điều và (khi có) văn bản hướng dẫn thi hành.
- Lắp ráp `law_to_crime_vn.json` và `crimes_by_part_vn.json`.

Hai chương thử nghiệm: **Chương XVI** (Điều 168–180, xâm phạm sở hữu) và **Chương XX** (Điều 247–259, ma túy).

**Chưa làm:** thu thập bản án (Phase 2), tầng Case và lắp ráp graph hoàn chỉnh (Phase 3), kiểm định truy xuất (Phase 4), các agent (Phase 5+).

**Lưu ý về nguồn luật:** file `data/raw/law/100_2015_QH13_296661.docx` là **văn bản Luật 100/2015/QH13 gốc**, chưa hợp nhất các sửa đổi 2017 (Luật 12/2017/QH14) và 2025 (Luật 86/2025/QH15). Cần thay bằng bản hợp nhất mới nhất trước khi dùng cho mục đích ngoài thử nghiệm cấu trúc.

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env   # rồi điền LLM_API_KEY
```

## Chạy Phase 1 (tầng Law + Crime)

```bash
# 1. Kiểm tra cấu trúc, không gọi LLM (miễn phí)
python scripts/build_law_layer.py --chapters XVI XX --dry-run

# 2. Chạy đầy đủ (có gọi LLM để sinh judge_dep) cho 2 chương thử nghiệm
python scripts/build_law_layer.py --chapters XVI XX

# 3. Toàn bộ BLHS (bỏ --chapters)
python scripts/build_law_layer.py
```

Kết quả nằm trong `data/processed/`:
- `criminal_law_vn.json` — cây Phần/Chương/Mục/Điều/Khoản/Điểm thô.
- `criminal_law_vn_judge_dep.json` — đã thêm `judge_dep`.
- `criminal_law_vn_linked.json` — đã thêm `related_laws`.
- `law_to_crime_vn.json`, `crimes_by_part_vn.json` — schema cuối cho tầng Law + Crime, sẵn sàng cho bước xây graph (Phase 3).

Văn bản hướng dẫn thi hành (Nghị quyết, Nghị định, Án lệ) được đề xuất trong `data/raw/guidance/guidance_manifest.json` (trạng thái `can_xac_minh` — cần kiểm tra hiệu lực và bổ sung nội dung trước khi dùng). Khi đã có nội dung, tạo `data/raw/guidance/guidance_links.json` theo schema `[{"explain", "from", "laws": [số điều]}]` để `link_guidance` tự động gắn vào đúng Điều.

## Kiểm tra

```bash
pytest tests/ -v
```

## Cấu trúc thư mục

```text
vn_legal_graph/
├── config.py          # đọc .env
├── llm.py             # client OpenAI-compatible, có cache theo hash prompt
├── embedding.py        # embedding tiếng Việt (PhoBERT-family) + backend bge-m3 để so sánh
├── prompts/vi.py        # toàn bộ prompt tiếng Việt
└── law/
    ├── parse_blhs.py    # Phase 1: parser .docx -> cây Điều/Khoản/Điểm
    ├── judge_dep.py      # Phase 1: LLM tách yếu tố cấu thành
    ├── link_guidance.py  # Phase 1: dẫn chiếu + văn bản hướng dẫn
    └── build_law_crime.py# Phase 1: lắp schema law_to_crime_vn.json
scripts/build_law_layer.py  # CLI chạy toàn bộ Phase 1
tests/                      # pytest, dùng fixture riêng, không phụ thuộc file luật thật
data/raw/law/               # văn bản luật gốc (.docx)
data/raw/guidance/           # văn bản hướng dẫn thi hành (đề xuất + đã duyệt)
data/raw/cases/               # (Phase 2) bản án thu thập được
data/processed/                # JSON đã xử lý qua từng bước
```
