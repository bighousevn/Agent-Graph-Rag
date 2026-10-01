# CLAUDE.md

- Trước khi làm bất cứ việc gì, đọc `docs/HANDOFF.md`. File đó có toàn bộ bối cảnh, các quyết định đã chốt và việc tiếp theo của dự án.
- Không đọc, in hay sửa `.env`, và không chạy lệnh nào gọi LLM API thật. Người dùng tự chạy những lệnh đó. Được tự chạy `pytest` và các lệnh `--dry-run`.
- Trả lời người dùng bằng tiếng Việt.
- Code mới phải đi kèm test (`pytest tests/`). Đối chiếu số liệu với dữ liệu thật trước khi kết luận.
