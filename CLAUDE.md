# CLAUDE.md

- Trước khi làm bất cứ việc gì, đọc `docs/HANDOFF.md`. File đó có toàn bộ bối cảnh, các quyết định đã chốt và việc tiếp theo của dự án.
- Không đọc, in hay sửa `.env`. Từ 2026-10-05 người dùng cho phép Claude chạy các script gọi LLM được liệt kê trong `permissions.allow` của `.claude/settings.json`. Luôn chạy `--dry-run` trước để báo số lời gọi và token. Không chạy lệnh nào in ra key.
- Trả lời người dùng bằng tiếng Việt.
- Code mới phải đi kèm test (`pytest tests/`). Đối chiếu số liệu với dữ liệu thật trước khi kết luận.
