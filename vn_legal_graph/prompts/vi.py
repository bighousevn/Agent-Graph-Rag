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

Định dạng đầu ra: chỉ trả lời "true" hoặc "false", không thêm bất kỳ chữ nào khác, thể hiện điều luật có áp dụng cho vụ án hay không.

Vụ án: {case}
Điều luật: {law}
Yếu tố thỏa mãn: {true_list}
Yếu tố không thỏa mãn: {false_list}

Trả lời:
""".strip()
