"""Vietnamese prompts for HierarGraph construction.

Adapted from the Chinese/English prompts in the original LegalGraphRAG
repo (`core/prompt/preprocess`, `core/prompt/graph`), rewritten for
Vietnamese criminal law terminology. "是否..." (Chinese "whether...")
becomes "Có ... không?" here, matching how Vietnamese judgments phrase
constituent-element questions (yếu tố cấu thành tội phạm).

JUDGE_DEP_PROMPT is used now, in Phase 1 (Law layer). The others
(GET_CASE_FEATURES_PROMPT, SUMMARIZE_CLUSTER_PROMPT, RERANK_*) are for
Phase 3 (Case layer) but are defined here already so the prompt set is in
one place from the start.
"""

JUDGE_DEP_PROMPT = """
Bạn là một trợ lý AI pháp lý, chuyên phân tích các điều luật hình sự Việt Nam. Nhiệm vụ của bạn là tách một Điều luật thành một danh sách các câu hỏi con, mỗi câu hỏi bắt đầu bằng "Có" và kết thúc bằng "không?", dùng để xác định xem một yếu tố cấu thành tội phạm hoặc một tình tiết định khung có xuất hiện trong một vụ án cụ thể hay không.

Yêu cầu:
1. Đầu ra phải là một danh sách Python dạng chuỗi (list[str]), chỉ chứa các câu hỏi đó, không kèm giải thích nào khác.
2. Mỗi câu hỏi phải cụ thể, có thể dùng để đối chiếu trực tiếp với tình tiết một vụ án, tránh quá trừu tượng.
3. Bao gồm cả các yếu tố cấu thành cơ bản (khoản 1) lẫn các tình tiết định khung tăng nặng ở các khoản sau, nếu có.
4. Câu hỏi phải bắt đầu bằng "Có" và kết thúc bằng "không?", ví dụ: "Có dùng vũ lực nhằm chiếm đoạt tài sản không?".
5. Đầu ra phải là một danh sách Python hợp lệ, ví dụ: ["Có A không?", "Có B không?"].

Điều luật cần phân tích:
{dieu_text}
""".strip()


GET_CASE_FEATURES_PROMPT = """
Bạn là một trợ lý AI pháp lý. Nhiệm vụ của bạn là xử lý phần diễn biến vụ án hình sự đầu vào. Đầu vào gồm mô tả diễn biến vụ án, có thể kèm tội danh của bị cáo. Hãy trích xuất các từ khóa từ mô tả và phân loại vào 4 nhóm sau: Nhân thân bị cáo, Hành vi phạm tội, Đặc điểm đối tượng/tài sản, Lỗi và thái độ. Đầu ra phải là một đối tượng JSON, chỉ chứa JSON, không kèm văn bản giải thích nào khác.

Giải thích các nhóm:
- Nhân thân bị cáo: trích các đặc điểm pháp lý liên quan đến bị cáo, như độ tuổi, tiền án tiền sự, nghề nghiệp; tránh nêu tuổi cụ thể hoặc tên đơn vị công tác cụ thể. Tên bị cáo không quan trọng.
- Hành vi phạm tội: trích loại hành vi và phương thức phạm tội có ý nghĩa pháp lý; tránh nêu thời gian, địa điểm cụ thể.
- Đặc điểm đối tượng/tài sản: trích đặc điểm loại của đối tượng hoặc tài sản bị xâm phạm, như tính chất tài sản, loại địa điểm, loại chất ma túy; tránh nêu tên địa danh cụ thể. Giá trị tài sản hoặc khối lượng ma túy ghi đúng con số văn bản nêu, ví dụ "tài sản trị giá 9.775.000 đồng", "heroine khối lượng 0,226 gam"; không tự đặt khoảng.
- Lỗi và thái độ: trích mô tả pháp lý về ý thức chủ quan và thái độ ăn năn, như cố ý/vô ý, tự thú, thành khẩn khai báo, khắc phục hậu quả.

Nguyên tắc bắt buộc:
- Chỉ trích những gì văn bản thực sự nêu. Nếu văn bản không nói gì về một nhóm, để mảng rỗng []; không suy đoán. Ví dụ: văn bản không nhắc tới việc khai báo thì không được ghi "thành khẩn khai báo" hay "không thành khẩn khai báo".
- Chỉ ghi giá trị tài sản hoặc khối lượng ma túy khi văn bản có nêu con số; không dùng các cụm chung chung như "giá trị lớn".
- Không ghi tên người, năm sinh, tình trạng hôn nhân, quan hệ gia đình hay chi tiết cá nhân không có ý nghĩa pháp lý.

Yêu cầu định dạng JSON:
- Dùng dấu ngoặc kép cho khóa và giá trị chuỗi.
- Mỗi khóa tương ứng một nhóm, giá trị là một mảng chuỗi chứa các từ khóa trích được (nếu nhóm không có từ khóa nào, dùng mảng rỗng []).
- Tên khóa bắt buộc là: "defendant_info", "criminal_acts", "victim_property_details", "intent_remorse".

Ví dụ đầu ra (chỉ để tham khảo, đầu ra thực tế phải dựa trên nội dung đầu vào):
{{
"defendant_info": ["đã thành niên", "có tiền án", "cán bộ nhà nước"],
"criminal_acts": ["trộm cắp", "đột nhập nơi ở"],
"victim_property_details": ["nhà ở tư nhân", "xe mô tô", "tài sản trị giá 9.775.000 đồng"],
"intent_remorse": ["lỗi cố ý trực tiếp"]
}}

Đảm bảo chỉ xuất ra đối tượng JSON.
Bây giờ hãy xử lý vụ án sau:
""".strip()

# Query-time input: facts only (the crime is what we are trying to find).
GET_CASE_FEATURES_INPUT = """
Diễn biến vụ án: {fact}
""".strip()

# Corpus-building input. Mirrors the original repo's
# scripts/prepare_case_features.py, which passes the charge (罪名) and asks
# for keywords centred on it. Never use this for test/query cases: the
# crime is the answer.
GET_CASE_FEATURES_INPUT_WITH_CRIME = """
Lưu ý: các từ khóa cần xoay quanh tội danh của bị cáo.
Tội danh: {crime}
Diễn biến vụ án: {fact}
""".strip()


SUMMARIZE_CLUSTER_PROMPT = """
Bạn là một chuyên gia pháp lý giàu kinh nghiệm. Nhiệm vụ của bạn là khái quát một nhóm (cụm) các vụ án hình sự thành một loại hành vi phạm tội ở mức khái quát cao. Hãy tuân theo các quy tắc sau:

1. Phân tích và khái quát: xem xét kỹ tất cả các hành vi đầu vào, xác định bản chất và mô hình chung của chúng.
2. Khái quát ở mức cao: đầu ra phải là một mô tả loại duy nhất, tinh gọn, không liệt kê hay lặp lại các hành vi đầu vào.
3. Định dạng đầu ra: chỉ một dòng duy nhất, đúng định dạng: "Nhóm hành vi phạm tội: [Loại khái quát]".
4. Không được xuất ra bất kỳ chi tiết hành vi cụ thể, văn bản giải thích, danh sách hay thông tin thêm nào khác.
""".strip()


RERANK_CLUSTERS_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp, giỏi ánh xạ mô tả một vụ án cụ thể vào các nhóm tội phạm ở mức khái quát cao.

Yêu cầu xử lý:
- Phân tích kỹ hành vi chính, ý thức chủ quan và quan hệ pháp luật liên quan trong vụ án.
- Đối chiếu với mô tả đặc trưng của từng nhóm, đánh giá mức độ phù hợp của từng nhóm.
- Sắp xếp tất cả các nhóm theo mức độ liên quan từ cao đến thấp.
- Định dạng đầu ra: ví dụ "rank: [3,1,2]", không được thêm bất kỳ văn bản nào khác.

Danh sách tóm tắt các nhóm hiện có:
{cluster_summaries}

Mô tả vụ án cần phân tích:
{query_text}
""".strip()


RERANK_CASES_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Tôi cần bạn sắp xếp lại các vụ án tương tự sau theo mức độ liên quan với vụ án gốc, và đưa ra ba vụ án liên quan nhất.

Mô tả nhiệm vụ:
1. Phân tích mức độ liên quan giữa từng vụ án tương tự (codeX) với vụ án gốc.
2. Sắp xếp lại các vụ án tương tự theo mức độ liên quan từ cao đến thấp.
3. Đưa ra số thứ tự của tối đa ba vụ án liên quan nhất.
4. Định dạng đầu ra phải là một danh sách số nguyên, chỉ chứa phần số của số thứ tự vụ án.

Yêu cầu đầu ra:
- Chỉ xuất ra một danh sách số nguyên, dạng: [3, 1, 2].
- Các số trong danh sách tương ứng với số thứ tự vụ án tương tự (số đứng sau "code").
- Vụ án xếp đầu tiên là vụ án liên quan nhất với vụ án gốc.

Thông tin các vụ án tương tự:
{neighbor_summaries}

Nội dung vụ án gốc:
{query_text}

Hãy xuất ra danh sách số thứ tự vụ án đã sắp xếp lại:
""".strip()


# ---- Phase 4: judge_law (port of core/prompt/judge JUDGE_LAW_PROMPT / 0) ----

JUDGE_ELEMENT_PROMPT = """
Bạn là một trợ lý AI pháp lý chuyên nghiệp, giỏi phân tích khả năng áp dụng của điều luật. Nhiệm vụ của bạn là đánh giá chặt chẽ xem tình tiết vụ án có thỏa mãn yếu tố cấu thành được nêu hay không, dựa trên điều luật, tài liệu bổ trợ, yếu tố cần xét và tình tiết vụ án.

Thông tin đầu vào:
- Điều luật: nội dung điều luật
- Tài liệu bổ trợ: điều luật liên quan hoặc văn bản hướng dẫn; nếu trống thì bỏ qua
- Yếu tố cần xét: yếu tố cụ thể cần kiểm tra; bạn phải tập trung vào yếu tố này
- Vụ án: tình tiết vụ án

Hướng dẫn phân tích:
1. Đọc kỹ điều luật, hiểu nội dung và các yếu tố cấu thành.
2. Nếu tài liệu bổ trợ không trống, dùng nó để giải thích điều luật hoặc yếu tố.
3. Trích thông tin liên quan từ tình tiết vụ án và đối chiếu với yếu tố cần xét.
4. Dựa trên tình tiết và lập luận, nếu vụ án thỏa mãn yếu tố thì trả lời true, ngược lại trả lời false.

Định dạng đầu ra: chỉ trả lời "true" hoặc "false", không thêm bất kỳ chữ nào khác.

Điều luật: {law}
Tài liệu bổ trợ: {related}
Yếu tố cần xét: {element}
Vụ án: {case}

Trả lời:
""".strip()

# Variant (not in the original): all elements of one article in a single
# call, about 30x fewer calls than one call per element.
JUDGE_ELEMENTS_BATCH_PROMPT = """
Bạn là một trợ lý AI pháp lý chuyên nghiệp, giỏi phân tích khả năng áp dụng của điều luật. Nhiệm vụ của bạn là đánh giá chặt chẽ, với từng yếu tố được đánh số dưới đây, xem tình tiết vụ án có thỏa mãn yếu tố đó hay không, dựa trên điều luật, tài liệu bổ trợ và tình tiết vụ án.

Hướng dẫn phân tích:
1. Đọc kỹ điều luật, hiểu nội dung và các yếu tố cấu thành.
2. Nếu tài liệu bổ trợ không trống, dùng nó để giải thích điều luật hoặc yếu tố.
3. Với từng yếu tố, đối chiếu riêng với tình tiết vụ án; vụ án thỏa mãn thì true, ngược lại false.

Định dạng đầu ra: chỉ một đối tượng JSON, khóa là số thứ tự của yếu tố (từ "1" đến "{n}"), giá trị là true hoặc false, ví dụ {{"1": true, "2": false, "3": true}}. Không thêm chữ nào khác.

Điều luật: {law}
Tài liệu bổ trợ: {related}
Các yếu tố cần xét:
{elements}
Vụ án: {case}

Trả lời:
""".strip()

JUDGE_LAW_FINAL_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Dựa trên điều luật và kết quả phân tích vụ án được cung cấp, hãy đánh giá điều luật này có áp dụng cho vụ án hay không (tức là vụ án có cấu thành tội phạm theo điều luật này không).

Thông tin đầu vào:
- Vụ án: mô tả vụ án
- Điều luật: nội dung điều luật
- Yếu tố thỏa mãn: các phần của điều luật được xác định là đúng với vụ án
- Yếu tố không thỏa mãn: các phần của điều luật được xác định là không đúng với vụ án

Hướng dẫn phân tích:
1. Đọc điều luật và xác định mọi yếu tố cấu thành liên quan.
2. Lưu ý: hai danh sách trên có thể chưa đầy đủ; bạn cần tự kiểm tra các yếu tố then chốt dựa trên điều luật.
3. Điều luật áp dụng cho vụ án khi vụ án thỏa mãn các yếu tố cấu thành cơ bản của tội (thường nêu ở khoản 1). Các tình tiết định khung tăng nặng ở các khoản sau (ví dụ có tổ chức, khối lượng hoặc giá trị lớn, tái phạm nguy hiểm) không bắt buộc: việc chúng không thỏa mãn không làm điều luật mất khả năng áp dụng.
4. Nếu điều luật có điều kiện loại trừ (ví dụ "mà không nhằm mục đích mua bán, vận chuyển, sản xuất trái phép chất ma túy") và vụ án thỏa mãn chính yếu tố bị loại trừ đó (ví dụ có mục đích mua bán), thì điều luật không áp dụng.

Định dạng đầu ra: chỉ trả lời "true" hoặc "false", không thêm bất kỳ chữ nào khác, thể hiện điều luật có áp dụng cho vụ án hay không.

Vụ án: {case}
Điều luật: {law}
Yếu tố thỏa mãn: {true_list}
Yếu tố không thỏa mãn: {false_list}

Trả lời:
""".strip()


# ---- Legal Q&A pilot (questions.xlsx) ----------------------------------------

# Port of core/prompt/retrieval RETRIEVE_LAW_PROMPT (the "augment" branch:
# the LLM names up to three candidate crimes, matched to Crime nodes).
RETRIEVE_LAW_PROMPT = """
Bạn là một thẩm phán hình sự chuyên nghiệp, cần phân tích tối đa ba tội danh có thể cấu thành dựa trên tình tiết được mô tả. Hãy tuân theo các yêu cầu sau:

Yêu cầu phân tích:
1. Phân tích toàn diện từng hành vi trong tình tiết.
2. Xem xét mọi tội danh mà hành vi có thể vi phạm theo Bộ luật Hình sự Việt Nam.
3. Bao gồm cả tội danh cơ bản và tội danh đặc thù.

Yêu cầu đầu ra:
- Chỉ xuất một danh sách dạng Python: ["Tội A", "Tội B", ...]
- Sắp xếp theo khả năng từ cao xuống thấp.
- Tối đa ba tội danh; ghi đúng tên tội như trong Bộ luật Hình sự (bắt đầu bằng "Tội").
- Không giải thích, không đánh số, không thêm nội dung nào khác.
- Nếu tình tiết không mô tả hành vi phạm tội nào, xuất [].

Tình tiết:
```
{fact}
```
Bây giờ hãy xuất kết quả:
""".strip()

# Port of JUDGE_LAW_PROMPT1: applicability of an article that has no
# judge_dep elements (general part of BLHS, procedural articles).
JUDGE_LAW_SIMPLE_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Hãy đánh giá trực tiếp xem điều luật được cung cấp có liên quan và áp dụng để giải quyết tình huống dưới đây hay không. Điều luật có thể là luật nội dung, luật tố tụng hoặc quy định chung.

Yêu cầu:
- Đánh giá tình huống có thuộc phạm vi điều chỉnh của điều luật này không.
- Chỉ xét nghĩa của chính điều luật, không suy diễn ngoài văn bản.
- Chỉ trả lời "true" hoặc "false".

Điều luật: {law}
Tình huống: {case}

Trả lời:
""".strip()

# Replaces judge_crime_all for Q&A: the answer, plus crimes and articles in a
# structure that can be scored.
QA_ANSWER_PROMPT = """
Bạn là luật sư tư vấn pháp luật hình sự Việt Nam. Hãy trả lời câu hỏi dưới đây CHỈ dựa trên các điều luật và tài liệu được cung cấp.

Yêu cầu:
1. Xác định hành vi trong tình huống cấu thành tội gì (nếu có) và áp dụng điều, khoản, điểm nào; hoặc, với câu hỏi về thủ tục, quy định nào giải quyết vấn đề.
2. Nêu rõ căn cứ: tên văn bản và điều, khoản, điểm.
3. Kết luận dứt khoát khi tình huống đã nêu đủ dữ kiện; chỉ nêu điều kiện ("nếu…") khi thực sự thiếu một dữ kiện quyết định.
4. Chỉ trích các điều luật và văn bản có trong phần tài liệu dưới đây; không trích văn bản không được cung cấp.
5. Nếu có hướng dẫn áp dụng (Công văn) xử lý tình huống tương tự, ưu tiên làm theo hướng dẫn đó.
6. Kết quả kiểm tra yếu tố cấu thành kèm theo mỗi điều chỉ để tham khảo; nó có thể sai với câu hỏi giả định hoặc hành vi mới ở mức chuẩn bị.
7. Câu đầu tiên của "cau_tra_loi" phải trả lời thẳng câu hỏi: "Có"/"Không", hoặc tên tội và khung (điều, khoản, điểm), hoặc việc phải làm. Không mở đầu bằng một khả năng mà câu sau lại bác bỏ (ví dụ không viết "Không bị truy cứu theo khoản 2…" rồi kết luận khoản 2). Các câu sau mới nêu căn cứ và lý do.
8. Trả lời ngắn gọn, không quá 120 từ, không chép lại nguyên văn điều luật.
9. Nếu tài liệu được cung cấp không đủ để trả lời, nói rõ là không đủ căn cứ; không bịa điều luật.

Đầu ra là một đối tượng JSON duy nhất, không thêm chữ nào khác:
{{
"cau_tra_loi": "câu trả lời ngắn gọn",
"toi_danh": ["Tội ..."],
"dieu_luat": [{{"luat": "BLHS" hoặc "BLTTHS" hoặc tên văn bản khác, "dieu": "321", "khoan": "2", "diem": "c"}}]
}}
Bỏ trống "khoan"/"diem" ("") nếu không xác định; "toi_danh" là [] nếu câu hỏi không về tội danh.

Câu hỏi:
{question}

Các điều luật và tài liệu được cung cấp:
{laws}

Trả lời:
""".strip()

# LLM-as-judge for the pilot, compared against the reference answer.
QA_GRADE_PROMPT = """
Bạn là giảng viên luật hình sự. Hãy chấm câu trả lời của hệ thống so với đáp án tham khảo của luật sư.

Chỉ chấm KẾT LUẬN (trả lời đúng câu hỏi chưa), không chấm cách trình bày:
- "dung": kết luận chính giống đáp án tham khảo.
- "mot_phan": đúng một phần (ví dụ thiếu một tội, hoặc đúng tội nhưng sai khoản/điều kiện quan trọng).
- "sai": kết luận khác đáp án hoặc không trả lời được.

Đầu ra là một đối tượng JSON duy nhất: {{"diem": "dung" | "mot_phan" | "sai", "ly_do": "một câu ngắn"}}

Câu hỏi:
{question}

Đáp án tham khảo:
{reference}

Câu trả lời của hệ thống:
{answer}

Chấm:
""".strip()


# ---- Case pipeline (core/utils/util.py::analyze_case) ------------------------

# Added: CAIL gives the defendants' names ("criminals"); Vietnamese
# judgments do not, so the LLM lists them before CASE_SEG_PROMPT.
LIST_DEFENDANTS_PROMPT = """
Bạn là một trợ lý phân tích pháp lý. Hãy liệt kê tên các bị cáo (người bị truy cứu trách nhiệm hình sự) được nhắc trong diễn biến vụ án dưới đây.

Yêu cầu:
- Chỉ liệt kê bị cáo, không liệt kê bị hại, người làm chứng, người có quyền lợi liên quan.
- Ghi tên đúng như trong văn bản (có thể là tên viết tắt, ví dụ "nguyễn văn a").
- Đầu ra là một danh sách dạng Python: ["tên 1", "tên 2"]. Không giải thích.
- Nếu không xác định được tên, xuất [].

Diễn biến vụ án:
```
{fact}
```
""".strip()

# Port of core/prompt/preprocess CASE_SEG_PROMPT.
CASE_SEG_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Nhiệm vụ của bạn là dựa trên mô tả vụ án và tên bị cáo dưới đây, sắp xếp lại thành một đoạn mô tả sự việc khách quan về bị cáo đó.

### Đầu vào:
- Mô tả vụ án: {fact}
- Tên bị cáo: {name}

### Lưu ý:
- **Dựa trên nội dung đầu vào**: chỉ sắp xếp từ mô tả vụ án được cung cấp, không thêm thông tin hay giả định bên ngoài.
- **Khách quan**: mô tả phải hoàn toàn khách quan, không có kết quả xét xử, đánh giá pháp lý hay phân tích chủ quan (như suy đoán động cơ, sắc thái cảm xúc).
- **Đầy đủ**: kể cả hành vi không do bị cáo trực tiếp thực hiện, nếu có liên quan đến bị cáo (nguyên nhân, hậu quả, bối cảnh, hoặc liên quan trực tiếp đến hành vi của bị cáo) thì vẫn đưa vào để đủ ngữ cảnh.
- **Định dạng**: xuất thẳng đoạn mô tả đã sắp xếp, ngắn gọn, chính xác, không thêm lời dẫn, tóm tắt hay bình luận.
- **Trọng tâm**: xoay quanh hành vi, vai trò của bị cáo và các sự kiện liên quan; không nêu các bên không liên quan hay chi tiết phụ, trừ khi có liên hệ rõ ràng với bị cáo.

Hãy xử lý thông tin đầu vào theo các yêu cầu trên.
""".strip()

# Port of core/prompt/judge JUDGE_CRIME_ALL_PROMPT (Adjudicator) and its input template.
JUDGE_CRIME_ALL_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Hãy dựa trên các điều luật ứng viên để xác định tội danh cho bị cáo, đồng thời dự đoán điều luật áp dụng và mức hình phạt.

Lưu ý:
- Nếu không cần thiết, đừng kết luận nhiều tội danh mà chọn tội danh phù hợp nhất.
- Việc chọn tội danh phải theo các bước:
  1. **Xác định số hành vi**: vụ án có bao nhiêu hành vi phạm tội độc lập. Phân biệt một hành vi vi phạm nhiều điều luật với nhiều hành vi vi phạm các điều luật khác nhau.
  2. **Áp dụng tội danh cuối cùng**: với mỗi hành vi độc lập, xác định tội danh áp dụng. Một hành vi thỏa mãn nhiều điều luật thì chọn tội nặng hơn hoặc điều luật chuyên biệt hơn; nhiều hành vi độc lập thì áp dụng từng điều luật tương ứng và tổng hợp hình phạt.
  3. **Dự đoán điều luật và hình phạt**: nêu rõ điều, khoản làm căn cứ, và dự đoán hợp lý mức hình phạt dựa trên tình tiết vụ án, khung hình phạt và thực tiễn xét xử.
- Tội danh phải có căn cứ pháp luật và gắn chặt với tình tiết vụ án, không suy đoán.
- Đầu ra phải là **một đối tượng JSON duy nhất**, không thêm chữ nào khác, theo cấu trúc:
```json
{{
    "toi_danh": ["Tội ..."],
    "dieu_luat": ["Điều 173 khoản 1"],
    "hinh_phat": {{
        "tu_hinh": false,
        "tu_co_thoi_han_thang": 0,
        "chung_than": false
    }}
}}
```
""".strip()

JUDGE_CRIME_ALL_INPUT_TEMPLATE = """

Các điều luật ứng viên:
{law}

Vụ án:
{case}
""".rstrip()
